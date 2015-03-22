//===- ControlEquivalence.h - Compute Control Equivalence-------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Determines control dependence equivalence classes for basic blocks. Any two
// blocks having the same set of control dependences land in one class.
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CONTROLEQUIVALENCE_H
#define LLVM_ANALYSIS_CONTROLEQUIVALENCE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <list>
#include <stack>

namespace llvm {
class BasicBlock;

// Note that this implementation actually uses cycle equivalence to establish
// class numbers. Any two nodes are cycle equivalent if they occur in the same
// set of cycles. It can be shown that control dependence equivalence reduces
// to undirected cycle equivalence for strongly connected control flow graphs.
//
// The algorithm is based on the paper, "The program structure tree: computing
// control regions in linear time" by Johnson, Pearson & Pingali (PLDI94) which
// also contains proofs for the aforementioned equivalence.

class ControlEquivalence : public FunctionPass {
public:
  ControlEquivalence() : FunctionPass(ID), Computed(false) {
    initializeControlEquivalencePass(*PassRegistry::getPassRegistry());
  }
  static char ID;

  // run - Calculate control Equivalence for this function
  bool runOnFunction(Function &F) override;

  // getAnalysisUsage - Implement the Pass API
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
  // releaseMemory - Reset state back to before function was analyzed
  void releaseMemory() override;
  // print - Show contents in human readable format...
  void print(raw_ostream &O, const Module * = nullptr) const override;

  unsigned getClassNumber(BasicBlock *BB) {
    assert(Computed && "Trying to get equivalence classes before computation");
    return BlockData[BB].ClassNumber;
  }

private:
  typedef std::pair<const BasicBlock *, const BasicBlock *> BasicBlockEdgeType;
  typedef std::list<BasicBlockEdgeType> EdgeList;
  struct Bracket {
    const BasicBlock *From;
    const BasicBlock *To;
    unsigned RecentClass;
    unsigned RecentSize;
  };

  typedef std::list<Bracket> BracketList;
  typedef SmallVector<const BasicBlock *, 4> FakeEdgeListType;

  class const_combined_iterator
      : public std::iterator<std::forward_iterator_tag, const BasicBlock,
                             ptrdiff_t, const BasicBlock *,
                             const BasicBlock *> {
    typedef std::iterator<std::forward_iterator_tag, const BasicBlock,
                          ptrdiff_t, const BasicBlock *,
                          const BasicBlock *> super;

  public:
    typedef typename super::pointer pointer;
    typedef typename super::reference reference;

    inline const_combined_iterator(const BasicBlock *BB,
                                   const BasicBlock *FPred,
                                   const BasicBlock *FSucc, bool End)
        : PredCurr(End ? pred_end(BB) : pred_begin(BB)), PredEnd(pred_end(BB)),
          SuccCurr(End ? succ_end(BB) : succ_begin(BB)), SuccEnd(succ_end(BB)),
          FakePred(FPred), FakeSucc(FSucc) {}

    inline bool operator==(const const_combined_iterator &x) const {
      return PredCurr == x.PredCurr && PredEnd == x.PredEnd &&
             SuccCurr == x.SuccCurr && SuccEnd == x.SuccEnd &&
             FakePred == x.FakePred && FakeSucc == x.FakeSucc;
    }
    inline bool operator!=(const const_combined_iterator &x) const {
      return !operator==(x);
    }
    inline reference operator*() const {
      if (PredCurr != PredEnd)
        return *PredCurr;
      if (SuccCurr != SuccEnd)
        return *SuccCurr;
      if (FakePred != nullptr)
        return FakePred;
      if (FakeSucc != nullptr)
        return FakeSucc;
      llvm_unreachable("trying to dereference past end of iterator");
    }
    inline pointer operator->() const { return operator*(); }
    inline const_combined_iterator &operator++() {
      if (PredCurr != PredEnd) {
        ++PredCurr;
        return *this;
      }

      if (SuccCurr != SuccEnd) {
        ++SuccCurr;
        return *this;
      }
      if (FakePred != nullptr) {
        FakePred = nullptr;
        return *this;
      }
      if (FakeSucc != nullptr) {
        FakeSucc = nullptr;
        return *this;
      }
      llvm_unreachable("Went off the end of our iterator");
    }

  private:
    const_pred_iterator PredCurr;
    const_pred_iterator PredEnd;
    succ_const_iterator SuccCurr;
    succ_const_iterator SuccEnd;
    const BasicBlock *FakePred;
    const BasicBlock *FakeSucc;
  };
  const_combined_iterator
  all_edges_begin(const BasicBlock *BB, const BasicBlock *FakePred = nullptr,
                  const BasicBlock *FakeSucc = nullptr) {
    return const_combined_iterator(BB, FakePred, FakeSucc, false);
  }
  const_combined_iterator all_edges_end(const BasicBlock *BB) {
    return const_combined_iterator(BB, nullptr, nullptr, true);
  }

  struct BlockCEData {
    // Equivalence class number assigned to Block.
    unsigned ClassNumber;
    // Pre-order DFS number assigned to Block.
    unsigned DFSNumber;
    // Indicates Block has already been visited.
    bool Visited;
    // Indicates Block is on DFS stack during walk.
    bool OnStack;
    // Indicates Block participates in DFS walk.
    bool Participates;
    // List of brackets per Block.
    BracketList BList;
    const BasicBlock *Hi;
    const BasicBlock *Parent;
    std::list<const BasicBlock *> Children;
    std::list<const BasicBlock *> Backedges;
    std::list<const BasicBlock *> Capping;
    const BasicBlock *FakeSucc;
    const BasicBlock *FakePred;
    BlockCEData()
        : ClassNumber(0), DFSNumber(0), Visited(false), OnStack(false),
          Participates(true), Hi(nullptr), Parent(nullptr), FakeSucc(nullptr),
          FakePred(nullptr) {}
    ~BlockCEData() {}
  };

  void runDFS(const BasicBlock *, SmallPtrSetImpl<const BasicBlock *> &,
              SmallVectorImpl<const BasicBlock *> &);
  void cycleEquiv(const BasicBlock *);
  void debugBracketList(const BracketList &BList);
  unsigned DFSNumber;
  unsigned ClassNumber;
  SmallDenseMap<const BasicBlock *, BlockCEData, 8> BlockData;
  bool Computed;
  const BasicBlock *FakeStart;
  const BasicBlock *FakeEnd;
};
}

#endif
