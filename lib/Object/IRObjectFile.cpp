//===- IRObjectFile.cpp - IR object file implementation ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Part of the IRObjectFile class implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/IRObjectFile.h"
#include "RecordStreamer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/GVMaterializer.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;
using namespace object;

IRObjectFile::IRObjectFile(MemoryBufferRef Object, std::unique_ptr<Module> Mod)
    : SymbolicFile(Binary::ID_IR, Object), M(std::move(Mod)) {
  SymTab.addModule(M.get());
}

IRObjectFile::~IRObjectFile() {}

static ModuleSymbolTable::Symbol getSym(DataRefImpl &Symb) {
  return *reinterpret_cast<ModuleSymbolTable::Symbol *>(Symb.p);
}

void IRObjectFile::moveSymbolNext(DataRefImpl &Symb) const {
  Symb.p += sizeof(ModuleSymbolTable::Symbol);
}

std::error_code IRObjectFile::printSymbolName(raw_ostream &OS,
                                              DataRefImpl Symb) const {
  SymTab.printSymbolName(OS, getSym(Symb));
  return std::error_code();
}

uint32_t IRObjectFile::getSymbolFlags(DataRefImpl Symb) const {
  return SymTab.getSymbolFlags(getSym(Symb));
}

GlobalValue *IRObjectFile::getSymbolGV(DataRefImpl Symb) {
  return getSym(Symb).dyn_cast<GlobalValue *>();
}

std::unique_ptr<Module> IRObjectFile::takeModule() { return std::move(M); }

basic_symbol_iterator IRObjectFile::symbol_begin() const {
  DataRefImpl Ret;
  Ret.p = reinterpret_cast<uintptr_t>(SymTab.symbols().data());
  return basic_symbol_iterator(BasicSymbolRef(Ret, this));
}

basic_symbol_iterator IRObjectFile::symbol_end() const {
  DataRefImpl Ret;
  Ret.p = reinterpret_cast<uintptr_t>(SymTab.symbols().data() +
                                      SymTab.symbols().size());
  return basic_symbol_iterator(BasicSymbolRef(Ret, this));
}

StringRef IRObjectFile::getTargetTriple() const { return M->getTargetTriple(); }

ErrorOr<MemoryBufferRef> IRObjectFile::findBitcodeInObject(const ObjectFile &Obj) {
  for (const SectionRef &Sec : Obj.sections()) {
    if (Sec.isBitcode()) {
      StringRef SecContents;
      if (std::error_code EC = Sec.getContents(SecContents))
        return EC;
      return MemoryBufferRef(SecContents, Obj.getFileName());
    }
  }

  return object_error::bitcode_section_not_found;
}

ErrorOr<MemoryBufferRef> IRObjectFile::findBitcodeInMemBuffer(MemoryBufferRef Object) {
  sys::fs::file_magic Type = sys::fs::identify_magic(Object.getBuffer());
  switch (Type) {
  case sys::fs::file_magic::bitcode:
    return Object;
  case sys::fs::file_magic::elf_relocatable:
  case sys::fs::file_magic::macho_object:
  case sys::fs::file_magic::coff_object: {
    Expected<std::unique_ptr<ObjectFile>> ObjFile =
        ObjectFile::createObjectFile(Object, Type);
    if (!ObjFile)
      return errorToErrorCode(ObjFile.takeError());
    return findBitcodeInObject(*ObjFile->get());
  }
  default:
    return object_error::invalid_file_type;
  }
}

Expected<std::unique_ptr<IRObjectFile>>
llvm::object::IRObjectFile::create(MemoryBufferRef Object,
                                   LLVMContext &Context) {
  ErrorOr<MemoryBufferRef> BCOrErr = findBitcodeInMemBuffer(Object);
  if (!BCOrErr)
    return errorCodeToError(BCOrErr.getError());

  Expected<std::unique_ptr<Module>> MOrErr =
      getLazyBitcodeModule(*BCOrErr, Context,
                           /*ShouldLazyLoadMetadata*/ true);
  if (!MOrErr)
    return MOrErr.takeError();

  std::unique_ptr<Module> &M = MOrErr.get();
  return llvm::make_unique<IRObjectFile>(BCOrErr.get(), std::move(M));
}
