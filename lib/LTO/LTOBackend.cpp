//===-LTOBackend.cpp - LLVM Link Time Optimizer Backend -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the "backend" phase of LTO, i.e. it performs
// optimization and code generation on a loaded module. It is generally used
// internally by the LTO class but can also be used independently, for example
// to implement a standalone ThinLTO backend.
//
//===----------------------------------------------------------------------===//

#include "llvm/LTO/LTOBackend.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/FunctionImportUtils.h"
#include "llvm/Transforms/Utils/SplitModule.h"

using namespace llvm;
using namespace lto;

Error Config::addSaveTemps(std::string OutputFileName,
                           bool UseInputModulePath) {
  ShouldDiscardValueNames = false;

  std::error_code EC;
  ResolutionFile = llvm::make_unique<raw_fd_ostream>(
      OutputFileName + ".resolution.txt", EC, sys::fs::OpenFlags::F_Text);
  if (EC)
    return errorCodeToError(EC);

  auto setHook = [&](std::string PathSuffix, ModuleHookFn &Hook) {
    // Keep track of the hook provided by the linker, which also needs to run.
    ModuleHookFn LinkerHook = Hook;
    Hook = [=](unsigned Task, Module &M) {
      // If the linker's hook returned false, we need to pass that result
      // through.
      if (LinkerHook && !LinkerHook(Task, M))
        return false;

      std::string PathPrefix;
      // If this is the combined module (not a ThinLTO backend compile) or the
      // user hasn't requested using the input module's path, emit to a file
      // named from the provided OutputFileName with the Task ID appended.
      if (M.getModuleIdentifier() == "ld-temp.o" || !UseInputModulePath) {
        PathPrefix = OutputFileName;
        if (Task != 0)
          PathPrefix += "." + utostr(Task);
      } else
        PathPrefix = M.getModuleIdentifier();
      std::string Path = PathPrefix + "." + PathSuffix + ".bc";
      std::error_code EC;
      raw_fd_ostream OS(Path, EC, sys::fs::OpenFlags::F_None);
      if (EC) {
        // Because -save-temps is a debugging feature, we report the error
        // directly and exit.
        llvm::errs() << "failed to open " << Path << ": " << EC.message()
                     << '\n';
        exit(1);
      }
      WriteBitcodeToFile(&M, OS, /*ShouldPreserveUseListOrder=*/false);
      return true;
    };
  };

  setHook("0.preopt", PreOptModuleHook);
  setHook("1.promote", PostPromoteModuleHook);
  setHook("2.internalize", PostInternalizeModuleHook);
  setHook("3.import", PostImportModuleHook);
  setHook("4.opt", PostOptModuleHook);
  setHook("5.precodegen", PreCodeGenModuleHook);

  CombinedIndexHook = [=](const ModuleSummaryIndex &Index) {
    std::string Path = OutputFileName + ".index.bc";
    std::error_code EC;
    raw_fd_ostream OS(Path, EC, sys::fs::OpenFlags::F_None);
    if (EC) {
      // Because -save-temps is a debugging feature, we report the error
      // directly and exit.
      llvm::errs() << "failed to open " << Path << ": " << EC.message() << '\n';
      exit(1);
    }
    WriteIndexToFile(Index, OS);
    return true;
  };

  return Error();
}

namespace {

std::unique_ptr<TargetMachine>
createTargetMachine(Config &C, StringRef TheTriple, const Target *TheTarget) {
  SubtargetFeatures Features;
  Features.getDefaultSubtargetFeatures(Triple(TheTriple));
  for (const std::string &A : C.MAttrs)
    Features.AddFeature(A);

  return std::unique_ptr<TargetMachine>(TheTarget->createTargetMachine(
      TheTriple, C.CPU, Features.getString(), C.Options, C.RelocModel,
      C.CodeModel, C.CGOptLevel));
}

bool opt(Config &C, TargetMachine *TM, unsigned Task, Module &M,
         bool IsThinLto) {
  M.setDataLayout(TM->createDataLayout());

  legacy::PassManager passes;
  passes.add(createTargetTransformInfoWrapperPass(TM->getTargetIRAnalysis()));

  PassManagerBuilder PMB;
  PMB.LibraryInfo = new TargetLibraryInfoImpl(Triple(TM->getTargetTriple()));
  PMB.Inliner = createFunctionInliningPass();
  // Unconditionally verify input since it is not verified before this
  // point and has unknown origin.
  PMB.VerifyInput = true;
  PMB.VerifyOutput = !C.DisableVerify;
  PMB.LoopVectorize = true;
  PMB.SLPVectorize = true;
  PMB.OptLevel = C.OptLevel;
  if (IsThinLto)
    PMB.populateThinLTOPassManager(passes);
  else
    PMB.populateLTOPassManager(passes);
  passes.run(M);

  if (C.PostOptModuleHook && !C.PostOptModuleHook(Task, M))
    return false;

  return true;
}

void codegen(Config &C, TargetMachine *TM, AddStreamFn AddStream, unsigned Task,
             Module &M) {
  if (C.PreCodeGenModuleHook && !C.PreCodeGenModuleHook(Task, M))
    return;

  std::unique_ptr<raw_pwrite_stream> OS = AddStream(Task);
  legacy::PassManager CodeGenPasses;
  if (TM->addPassesToEmitFile(CodeGenPasses, *OS,
                              TargetMachine::CGFT_ObjectFile))
    report_fatal_error("Failed to setup codegen");
  CodeGenPasses.run(M);
}

void splitCodeGen(Config &C, TargetMachine *TM, AddStreamFn AddStream,
                  unsigned ParallelCodeGenParallelismLevel,
                  std::unique_ptr<Module> M) {
  ThreadPool CodegenThreadPool(ParallelCodeGenParallelismLevel);
  unsigned ThreadCount = 0;
  const Target *T = &TM->getTarget();

  SplitModule(
      std::move(M), ParallelCodeGenParallelismLevel,
      [&](std::unique_ptr<Module> MPart) {
        // We want to clone the module in a new context to multi-thread the
        // codegen. We do it by serializing partition modules to bitcode
        // (while still on the main thread, in order to avoid data races) and
        // spinning up new threads which deserialize the partitions into
        // separate contexts.
        // FIXME: Provide a more direct way to do this in LLVM.
        SmallString<0> BC;
        raw_svector_ostream BCOS(BC);
        WriteBitcodeToFile(MPart.get(), BCOS);

        // Enqueue the task
        CodegenThreadPool.async(
            [&](const SmallString<0> &BC, unsigned ThreadId) {
              LTOLLVMContext Ctx(C);
              ErrorOr<std::unique_ptr<Module>> MOrErr = parseBitcodeFile(
                  MemoryBufferRef(StringRef(BC.data(), BC.size()), "ld-temp.o"),
                  Ctx);
              if (!MOrErr)
                report_fatal_error("Failed to read bitcode");
              std::unique_ptr<Module> MPartInCtx = std::move(MOrErr.get());

              std::unique_ptr<TargetMachine> TM =
                  createTargetMachine(C, MPartInCtx->getTargetTriple(), T);
              codegen(C, TM.get(), AddStream, ThreadId, *MPartInCtx);
            },
            // Pass BC using std::move to ensure that it get moved rather than
            // copied into the thread's context.
            std::move(BC), ThreadCount++);
      },
      false);
}

Expected<const Target *> initAndLookupTarget(Config &C, Module &M) {
  if (!C.OverrideTriple.empty())
    M.setTargetTriple(C.OverrideTriple);
  else if (M.getTargetTriple().empty())
    M.setTargetTriple(C.DefaultTriple);

  std::string Msg;
  const Target *T = TargetRegistry::lookupTarget(M.getTargetTriple(), Msg);
  if (!T)
    return make_error<StringError>(Msg, inconvertibleErrorCode());
  return T;
}

}

Error lto::backend(Config &C, AddStreamFn AddStream,
                   unsigned ParallelCodeGenParallelismLevel,
                   std::unique_ptr<Module> M) {
  Expected<const Target *> TOrErr = initAndLookupTarget(C, *M);
  if (!TOrErr)
    return TOrErr.takeError();

  std::unique_ptr<TargetMachine> TM =
      createTargetMachine(C, M->getTargetTriple(), *TOrErr);

  if (!opt(C, TM.get(), 0, *M, /*IsThinLto=*/false))
    return Error();

  if (ParallelCodeGenParallelismLevel == 1)
    codegen(C, TM.get(), AddStream, 0, *M);
  else
    splitCodeGen(C, TM.get(), AddStream, ParallelCodeGenParallelismLevel,
                 std::move(M));
  return Error();
}

Error lto::thinBackend(Config &C, unsigned Task, AddStreamFn AddStream,
                       Module &M, ModuleSummaryIndex &CombinedIndex,
                       const FunctionImporter::ImportMapTy &ImportList,
                       const GVSummaryMapTy &DefinedGlobals,
                       MapVector<StringRef, MemoryBufferRef> &ModuleMap) {
  Expected<const Target *> TOrErr = initAndLookupTarget(C, M);
  if (!TOrErr)
    return TOrErr.takeError();

  std::unique_ptr<TargetMachine> TM =
      createTargetMachine(C, M.getTargetTriple(), *TOrErr);

  if (C.PreOptModuleHook && !C.PreOptModuleHook(Task, M))
    return Error();

  thinLTOResolveWeakForLinkerModule(M, DefinedGlobals);

  renameModuleForThinLTO(M, CombinedIndex);

  if (C.PostPromoteModuleHook && !C.PostPromoteModuleHook(Task, M))
    return Error();

  if (!DefinedGlobals.empty())
    thinLTOInternalizeModule(M, DefinedGlobals);

  if (C.PostInternalizeModuleHook && !C.PostInternalizeModuleHook(Task, M))
    return Error();

  auto ModuleLoader = [&](StringRef Identifier) {
    return std::move(getLazyBitcodeModule(MemoryBuffer::getMemBuffer(
                                              ModuleMap[Identifier], false),
                                          M.getContext(),
                                          /*ShouldLazyLoadMetadata=*/true)
                         .get());
  };

  FunctionImporter Importer(CombinedIndex, ModuleLoader);
  Importer.importFunctions(M, ImportList);

  if (C.PostImportModuleHook && !C.PostImportModuleHook(Task, M))
    return Error();

  if (!opt(C, TM.get(), Task, M, /*IsThinLto=*/true))
    return Error();

  codegen(C, TM.get(), AddStream, Task, M);
  return Error();
}
