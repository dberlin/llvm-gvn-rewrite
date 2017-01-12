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
// \brief
//
// This file implements the PredicateInfo analysis, which creates an Extended
// SSA form for operations used in branch comparisons and llvm.assume
// comparisons.  Copies of these operations are inserted into the true/false
// edge (and after assumes), and information attached to the copies.  All uses
// of the original operation in blocks dominated by the true/false edge (and
// assume), are replaced with uses of the copies.  This enables passes to easily
// and sparsely propagate condition based info into the operations that may be
// affected.
//
// Example:
// %cmp = icmp eq i32 %x, 50
// br i1 %cmp, label %true, label %false
// true:
// ret i32 %x
// false:
// ret i32 1
//
// will become
//
// %cmp = icmp eq i32, %x, 50
// br i1 %cmp, label %true, label %false
// true:
// %x.0 = call @llvm.predicateinfo.i32(i32 %x)
// ret i32 %x
// false:
// ret i32 1
//
// Using getPredicateInfoFor on x.0 will give you the comparison it is
// dominated by (the icmp), and that you are located in the true edge of that
// comparison, which tells you x.0 is 50.
//
// In order to reduce the number of copies inserted, predicateinfo is only
// inserted where it would actually be live.  This means if there are no uses of
// an operation dominated by the branch edges, or by an assume, the associated
// predicate info is never inserted.
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_PREDICATEINFO_H
#define LLVM_TRANSFORMS_UTILS_PREDICATEINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
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
class OrderedBasicBlock;

enum PredicateType { PT_Branch, PT_Assume };

// Base class for all predicate information we provide.
// All of our predicate information has at least a comparison.
class PredicateBase : public ilist_node<PredicateBase> {
public:
  PredicateType Type;
  CmpInst *Comparison;
  PredicateBase(const PredicateBase &) = delete;
  PredicateBase &operator=(const PredicateBase &) = delete;
  PredicateBase() = delete;
  static inline bool classof(const PredicateBase *) { return true; }

protected:
  PredicateBase(PredicateType PT, CmpInst *Comparison)
      : Type(PT), Comparison(Comparison) {}
};

// Provides predicate information for assumes.  Since assumes are always true,
// we simply provide the assume instruction, so you can tell your relative
// position to it.
class PredicateAssume : public PredicateBase {
public:
  IntrinsicInst *AssumeInst;
  PredicateAssume(IntrinsicInst *AssumeInst, CmpInst *Comparison)
      : PredicateBase(PT_Assume, Comparison), AssumeInst(AssumeInst) {}
  PredicateAssume() = delete;
  static inline bool classof(const PredicateAssume *) { return true; }
  static inline bool classof(const PredicateBase *PB) {
    return PB->Type == PT_Assume;
  }
};

// Provides predicate information for branches.
class PredicateBranch : public PredicateBase {
public:
  // This is the block that is conditional upon the comparison.
  BasicBlock *BranchBB;
  // This is one of the true/false successors of BranchBB.
  BasicBlock *SplitBB;
  // If true, SplitBB is the true successor, otherwise it's the false successor.
  bool TrueEdge;
  PredicateBranch(BasicBlock *BranchBB, BasicBlock *SplitBB,
                  CmpInst *Comparison, bool TakenEdge)
      : PredicateBase(PT_Branch, Comparison), BranchBB(BranchBB),
        SplitBB(SplitBB), TrueEdge(TakenEdge) {}
  PredicateBranch() = delete;

  static inline bool classof(const PredicateBranch *) { return true; }
  static inline bool classof(const PredicateBase *PB) {
    return PB->Type == PT_Branch;
  }
};

// This name is used in a few places, so kick it into their own namespace
namespace PredicateInfoClasses {
struct ValueDFS;
}

/// \brief Encapsulates PredicateInfo, including all data associated with memory
/// accesses.
class PredicateInfo {
public:
private:
  // Used to store information about each value we might rename.
  struct ValueInfo {
    // Information about each possible copy. During processing, this is each
    // inserted info. After processing, we move the uninserted ones to the
    // uninserted vector.
    SmallVector<PredicateBase *, 4> Infos;
    SmallVector<PredicateBase *, 4> UninsertedInfos;
  };
  // This owns the all the predicate infos in the function, placed or not.
  iplist<PredicateBase> AllInfos;

public:
  PredicateInfo(Function &, DominatorTree &, AssumptionCache &);
  ~PredicateInfo();

  void verifyPredicateInfo() const;

  void dump() const;
  void print(raw_ostream &) const;

  const PredicateBase *getPredicateInfoFor(const Value *V) const {
    return PredicateMap.lookup(V);
  }

protected:
  // Used by PredicateInfo annotater, dumpers, and wrapper pass.
  friend class PredicateInfoAnnotatedWriter;
  friend class PredicateInfoPrinterLegacyPass;

private:
  void buildPredicateInfo();
  void processAssume(IntrinsicInst *, BasicBlock *, SmallPtrSetImpl<Value *> &);
  void processBranch(BranchInst *, BasicBlock *, SmallPtrSetImpl<Value *> &);
  void renameUses(SmallPtrSetImpl<Value *> &);
  using ValueDFS = PredicateInfoClasses::ValueDFS;
  typedef SmallVectorImpl<ValueDFS> ValueDFSStack;
  void convertUsesToDFSOrdered(Value *, SmallVectorImpl<ValueDFS> &);
  Value *materializeStack(unsigned int &, ValueDFSStack &, const ValueDFS &,
                          Value *);
  bool stackIsInScope(const ValueDFSStack &, int DFSIn, int DFSOut) const;
  void popStackUntilDFSScope(ValueDFSStack &, int DFSIn, int DFSOut);
  ValueInfo &getOrCreateValueInfo(Value *);
  const ValueInfo &getValueInfo(Value *) const;
  Function &F;
  DominatorTree &DT;
  AssumptionCache &AC;
  // This maps from copy operands to Predicate Info. Note that it does not own
  // the Predicate Info, they belong to the ValueInfo structs in the ValueInfos
  // vector.
  DenseMap<const Value *, const PredicateBase *> PredicateMap;
  // This stores info about each operand or comparison result we make copies
  // of.  The real ValueInfos start at index 1, index 0 is unused so that we can
  // more easily detect invalid indexing.
  SmallVector<ValueInfo, 32> ValueInfos;
  // This gives the index into the ValueInfos array for a given Value.  Because
  // 0 is not a valid Value Info index, you can use DenseMap::lookup and tell
  // whether it returned a valid result.
  DenseMap<Value *, unsigned int> ValueInfoNums;
  // OrderedBasicBlocks used during sorting uses
  DenseMap<const BasicBlock *, OrderedBasicBlock *> OBBMap;
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
