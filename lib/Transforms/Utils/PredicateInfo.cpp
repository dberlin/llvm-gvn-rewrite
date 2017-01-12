//===-- PredicateInfo.cpp - PredicateInfo Builder--------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------===//
//
// This file implements the PredicateInfo class.
//
//===----------------------------------------------------------------===//
#include "llvm/Transforms/Utils/PredicateInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/IteratedDominanceFrontier.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/PHITransAddr.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Transforms/Scalar.h"
#include <algorithm>
#define DEBUG_TYPE "predicateinfo"
using namespace llvm;
INITIALIZE_PASS_BEGIN(PredicateInfoWrapperPass, "predicateinfo",
                      "PredicateInfo", false, true)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(PredicateInfoWrapperPass, "predicateinfo", "PredicateInfo",
                    false, true)

INITIALIZE_PASS_BEGIN(PredicateInfoPrinterLegacyPass, "print-predicateinfo",
                      "PredicateInfo Printer", false, false)
INITIALIZE_PASS_DEPENDENCY(PredicateInfoWrapperPass)
INITIALIZE_PASS_END(PredicateInfoPrinterLegacyPass, "print-predicateinfo",
                    "PredicateInfo Printer", false, false)
static cl::opt<bool> VerifyPredicateInfo(
    "verify-predicateinfo", cl::init(false), cl::Hidden,
    cl::desc("Verify PredicateInfo in legacy printer pass."));
namespace llvm {
/// \brief An assembly annotator class to print PredicateInfo information in
/// comments.
class PredicateInfoAnnotatedWriter : public AssemblyAnnotationWriter {
  friend class PredicateInfo;
  const PredicateInfo *PredInfo;

public:
  PredicateInfoAnnotatedWriter(const PredicateInfo *M) : PredInfo(M) {}

  virtual void emitBasicBlockStartAnnot(const BasicBlock *BB,
                                        formatted_raw_ostream &OS) {}

  virtual void emitInstructionAnnot(const Instruction *I,
                                    formatted_raw_ostream &OS) {}
};
}

namespace llvm {
PredicateInfo::PredicateInfo(Function &F, DominatorTree *DT) : F(F), DT(DT) {}
PredicateInfo::~PredicateInfo() {}
void PredicateInfo::verifyPredicateInfo() const {}
void PredicateInfo::print(raw_ostream &OS) const {
  PredicateInfoAnnotatedWriter Writer(this);
  F.print(OS, &Writer);
}

void PredicateInfo::dump() const {
  PredicateInfoAnnotatedWriter Writer(this);
  F.print(dbgs(), &Writer);
}

char PredicateInfoPrinterLegacyPass::ID = 0;

PredicateInfoPrinterLegacyPass::PredicateInfoPrinterLegacyPass()
    : FunctionPass(ID) {
  initializePredicateInfoPrinterLegacyPassPass(
      *PassRegistry::getPassRegistry());
}

void PredicateInfoPrinterLegacyPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<PredicateInfoWrapperPass>();
  AU.addPreserved<PredicateInfoWrapperPass>();
}

bool PredicateInfoPrinterLegacyPass::runOnFunction(Function &F) {
  auto &PredInfo = getAnalysis<PredicateInfoWrapperPass>().getPredInfo();
  PredInfo.print(dbgs());
  if (VerifyPredicateInfo)
    PredInfo.verifyPredicateInfo();
  return false;
}

AnalysisKey PredicateInfoAnalysis::Key;

PredicateInfoAnalysis::Result
PredicateInfoAnalysis::run(Function &F, FunctionAnalysisManager &AM) {
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  return PredicateInfoAnalysis::Result(make_unique<PredicateInfo>(F, &DT));
}

PreservedAnalyses PredicateInfoPrinterPass::run(Function &F,
                                                FunctionAnalysisManager &AM) {
  OS << "PredicateInfo for function: " << F.getName() << "\n";
  AM.getResult<PredicateInfoAnalysis>(F).getPredInfo().print(OS);

  return PreservedAnalyses::all();
}

PreservedAnalyses PredicateInfoVerifierPass::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  AM.getResult<PredicateInfoAnalysis>(F).getPredInfo().verifyPredicateInfo();

  return PreservedAnalyses::all();
}

char PredicateInfoWrapperPass::ID = 0;

PredicateInfoWrapperPass::PredicateInfoWrapperPass() : FunctionPass(ID) {
  initializePredicateInfoWrapperPassPass(*PassRegistry::getPassRegistry());
}

void PredicateInfoWrapperPass::releaseMemory() { PredInfo.reset(); }

void PredicateInfoWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<DominatorTreeWrapperPass>();
}

bool PredicateInfoWrapperPass::runOnFunction(Function &F) {
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  PredInfo.reset(new PredicateInfo(F, &DT));
  return false;
}

void PredicateInfoWrapperPass::verifyAnalysis() const {
  PredInfo->verifyPredicateInfo();
}

void PredicateInfoWrapperPass::print(raw_ostream &OS, const Module *M) const {
  PredInfo->print(OS);
}
}
