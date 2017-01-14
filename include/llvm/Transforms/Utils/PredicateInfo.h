//===- PredicateInfo.h - Build PredicateInfo ----------------------*-C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// \file
// \brief This file exposes an interface to building/using memory SSA to
// walk memory instructions using a use/def graph.
//
// PredicateInfo class builds an SSA form that links together memory access
// instructions such as loads, stores, atomics, and calls. Additionally, it does
// a trivial form of "heap versioning" Every time the memory state changes in
// the program, we generate a new heap version. It generates MemoryDef/Uses/Phis
// that are overlayed on top of the existing instructions.
//
// As a trivial example,
// define i32 @main() #0 {
// entry:
//   %call = call noalias i8* @_Znwm(i64 4) #2
//   %0 = bitcast i8* %call to i32*
//   %call1 = call noalias i8* @_Znwm(i64 4) #2
//   %1 = bitcast i8* %call1 to i32*
//   store i32 5, i32* %0, align 4
//   store i32 7, i32* %1, align 4
//   %2 = load i32* %0, align 4
//   %3 = load i32* %1, align 4
//   %add = add nsw i32 %2, %3
//   ret i32 %add
// }
//
// Will become
// define i32 @main() #0 {
// entry:
//   ; 1 = MemoryDef(0)
//   %call = call noalias i8* @_Znwm(i64 4) #3
//   %2 = bitcast i8* %call to i32*
//   ; 2 = MemoryDef(1)
//   %call1 = call noalias i8* @_Znwm(i64 4) #3
//   %4 = bitcast i8* %call1 to i32*
//   ; 3 = MemoryDef(2)
//   store i32 5, i32* %2, align 4
//   ; 4 = MemoryDef(3)
//   store i32 7, i32* %4, align 4
//   ; MemoryUse(3)
//   %7 = load i32* %2, align 4
//   ; MemoryUse(4)
//   %8 = load i32* %4, align 4
//   %add = add nsw i32 %7, %8
//   ret i32 %add
// }
//
// Given this form, all the stores that could ever effect the load at %8 can be
// gotten by using the MemoryUse associated with it, and walking from use to def
// until you hit the top of the function.
//
// Each def also has a list of users associated with it, so you can walk from
// both def to users, and users to defs. Note that we disambiguate MemoryUses,
// but not the RHS of MemoryDefs. You can see this above at %7, which would
// otherwise be a MemoryUse(4). Being disambiguated means that for a given
// store, all the MemoryUses on its use lists are may-aliases of that store (but
// the MemoryDefs on its use list may not be).
//
// MemoryDefs are not disambiguated because it would require multiple reaching
// definitions, which would require multiple phis, and multiple memoryaccesses
// per instruction.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_PREDICATEINFO_H
#define LLVM_TRANSFORMS_UTILS_PREDICATEINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/PHITransAddr.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/OperandTraits.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/PassAnalysisSupport.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <memory>
#include <utility>

namespace llvm {

class DominatorTree;
class Function;
class Instruction;
class MemoryAccess;
class LLVMContext;
class raw_ostream;

/// \brief Encapsulates PredicateInfo, including all data associated with memory
/// accesses.
class PredicateInfo {
public:
  PredicateInfo(Function &, DominatorTree *);
  ~PredicateInfo();

  void dump() const;
  void print(raw_ostream &) const;

  /// \brief Verify that PredicateInfo is self consistent (IE definitions
  /// dominate
  /// all uses, uses appear in the right places).  This is used by unit tests.
  void verifyPredicateInfo() const;

  CmpInst *getPredicateFor(const Value *V) const {
    return PredicateMap.lookup(V);
  }

protected:
  // Used by PredicateInfo annotater, dumpers, and wrapper pass
  friend class PredicateInfoAnnotatedWriter;
  friend class PredicateInfoPrinterLegacyPass;

private:
  struct ValueDFS;
  struct SplitInfo;
  class ValueDFSStack;
  // Used to store information about each value we might rename.
  struct ValueInfo {

    // Places we may want to split, we later use liveness to determine whether
    // we actually split there or not.
    SmallPtrSet<BasicBlock *, 8> PossibleSplitBlocks;
    // Information about each possible split, indexed by the basic block of the
    // possible copy.
    DenseMap<BasicBlock *, SplitInfo> SplitInfos;
  };

  void buildPredicateInfo();
  void renameUses(SmallPtrSetImpl<Value *> &);
  void convertUsesToDFSOrdered(Value *, SmallVectorImpl<ValueDFS> &);
  ValueInfo &getOrCreateValueInfo(Value *);
  const ValueInfo &getValueInfo(Value *) const;
  Function &F;
  DominatorTree *DT;
  DenseMap<const Value *, CmpInst *> PredicateMap;
  DenseMap<std::pair<Value *, BasicBlock *>, Value *> OriginalToNewMap;
  SmallVector<ValueInfo, 32> ValueInfos;
  DenseMap<Value *, unsigned int> ValueInfoNums;
};

// This pass does eager building and then printing of PredicateInfo. It is used
// by
// the tests to be able to build, dump, and verify PredicateInfo.
class PredicateInfoPrinterLegacyPass : public FunctionPass {
public:
  PredicateInfoPrinterLegacyPass();

  static char ID;
  bool runOnFunction(Function &) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

/// An analysis that produces \c PredicateInfo for a function.
///
class PredicateInfoAnalysis : public AnalysisInfoMixin<PredicateInfoAnalysis> {
  friend AnalysisInfoMixin<PredicateInfoAnalysis>;
  static AnalysisKey Key;

public:
  // Wrap PredicateInfo result to ensure address stability of internal
  // PredicateInfo
  // pointers after construction.  Use a wrapper class instead of plain
  // unique_ptr<PredicateInfo> to avoid build breakage on MSVC.
  struct Result {
    Result(std::unique_ptr<PredicateInfo> &&PredInfo)
        : PredInfo(std::move(PredInfo)) {}
    PredicateInfo &getPredInfo() { return *PredInfo.get(); }

    std::unique_ptr<PredicateInfo> PredInfo;
  };

  Result run(Function &F, FunctionAnalysisManager &AM);
};

/// \brief Printer pass for \c PredicateInfo.
class PredicateInfoPrinterPass
    : public PassInfoMixin<PredicateInfoPrinterPass> {
  raw_ostream &OS;

public:
  explicit PredicateInfoPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

/// \brief Verifier pass for \c PredicateInfo.
struct PredicateInfoVerifierPass : PassInfoMixin<PredicateInfoVerifierPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

/// \brief Legacy analysis pass which computes \c PredicateInfo.
class PredicateInfoWrapperPass : public FunctionPass {
public:
  PredicateInfoWrapperPass();

  static char ID;
  bool runOnFunction(Function &) override;
  void releaseMemory() override;
  PredicateInfo &getPredInfo() { return *PredInfo; }
  const PredicateInfo &getPredInfo() const { return *PredInfo; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  void verifyAnalysis() const override;
  void print(raw_ostream &OS, const Module *M = nullptr) const override;

private:
  std::unique_ptr<PredicateInfo> PredInfo;
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_PREDICATEINFO_H
