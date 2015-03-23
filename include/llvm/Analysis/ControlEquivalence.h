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
// also contains proofs for the aforementioned equivalence. References to line
// numbers in the algorithm from figure 4 have been added [line:x].
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
  typedef enum { PredDirection, SuccDirection } DFSDirection;
  struct Bracket;
  typedef std::list<Bracket> BracketList;
  struct Bracket {
    // Direction in which this bracket was added.
    DFSDirection Direction;
    // Cached class when bracket was topmost.
    unsigned RecentClass;
    // Cached set-size when bracket was topmost.
    unsigned RecentSize;
    // Block that this bracket originates from.
    const BasicBlock *From;
    // Block that this bracket points to.
    const BasicBlock *To;
  };
  typedef SmallVector<const BasicBlock *, 4> FakeEdgeListType;

  // We use combined iterators to allow fake and real edges to be next to each
  // other. We do not expect these will work for anything but our use.
  template <class RealType, class FakeType>
  class const_combined_iterator
      : public std::iterator<std::forward_iterator_tag, const BasicBlock,
                             ptrdiff_t, const BasicBlock *,
                             const BasicBlock *> {
    typedef std::iterator<std::forward_iterator_tag, const BasicBlock *,
                          ptrdiff_t, const BasicBlock *,
                          const BasicBlock *> super;

  public:
    typedef typename super::pointer pointer;
    typedef typename super::reference reference;

    // We need to know when we hit the end, so we have to have the end
    // iterator too
    inline const_combined_iterator(RealType Begin, RealType End,
                                   FakeType FBegin, FakeType FEnd)
        : Real(Begin), RealEnd(End), Fake(FBegin), FakeEnd(FEnd) {}
    inline bool operator==(const_combined_iterator &x) const {
      return Real == x.Real && Fake == x.Fake;
    }
    inline bool operator!=(const_combined_iterator &x) const {
      return !operator==(x);
    }
    inline reference operator*() const {
      if (Real != RealEnd)
        return *Real;
      if (Fake != FakeEnd)
        return *Fake;
      llvm_unreachable("Tried to access past the end of our iterator");
    }
    inline pointer *operator->() const { return &operator*(); }

    inline const_combined_iterator &operator++() {
      if (Real != RealEnd) {
        ++Real;
        return *this;
      }
      if (Fake != FakeEnd) {
        ++Fake;
        return *this;
      }
      llvm_unreachable("Fell off the end of the iterator");
    }

  private:
    RealType Real;
    RealType RealEnd;
    FakeType Fake;
    FakeType FakeEnd;
  };

  typedef const_combined_iterator<const_pred_iterator,
                                  FakeEdgeListType::const_iterator>
      const_combined_pred_iterator;
  typedef const_combined_iterator<succ_const_iterator,
                                  FakeEdgeListType::const_iterator>
      const_combined_succ_iterator;
  inline const_combined_pred_iterator
  combined_pred_begin(const BasicBlock *B, const FakeEdgeListType &EL) {
    return const_combined_pred_iterator(pred_begin(B), pred_end(B), EL.begin(),
                                        EL.end());
  }

  inline const_combined_pred_iterator
  combined_pred_end(const BasicBlock *B, const FakeEdgeListType &EL) {
    return const_combined_pred_iterator(pred_end(B), pred_end(B), EL.end(),
                                        EL.end());
  }
  inline const_combined_succ_iterator
  combined_succ_begin(const BasicBlock *B, const FakeEdgeListType &EL) {
    return const_combined_succ_iterator(succ_begin(B), succ_end(B), EL.begin(),
                                        EL.end());
  }

  inline const_combined_succ_iterator
  combined_succ_end(const BasicBlock *B, const FakeEdgeListType &EL) {
    return const_combined_succ_iterator(succ_end(B), succ_end(B), EL.end(),
                                        EL.end());
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
    // // List of brackets per Block.
    // BracketList BList;
    // List of fake successor edges, if any
    FakeEdgeListType FakeSuccEdges;
    // List of fake predecessor edges, if any
    FakeEdgeListType FakePredEdges;
    // List of bracket iterators that point to us
    std::list<std::pair<BracketList &, BracketList::iterator>> BracketIterators;
    // ID of bracket list
    unsigned BracketListID;
    BlockCEData()
        : ClassNumber(0), DFSNumber(0), Visited(false), OnStack(false),
          Participates(true), BracketListID(0) {}
    ~BlockCEData() {}
  };
  struct DFSStackEntry {
    // Direction currently used in DFS walk.
    DFSDirection Direction;
    // Iterator used for "pred" direction.
    const_combined_pred_iterator Pred;
    // Iterator end for "pred" direction
    const_combined_pred_iterator PredEnd;
    // Iterator used for "succ" direction.
    const_combined_succ_iterator Succ;
    // Iterator end for "succ" direction.
    const_combined_succ_iterator SuccEnd;
    // Parent Block of entry during DFS walk.
    const BasicBlock *ParentBlock;
    // Basic Block that this stack entry belongs to
    const BasicBlock *Block;
  };
  // The stack is used during the undirected DFS walk.
  typedef std::stack<DFSStackEntry> DFSStack;
  void runUndirectedDFS(const BasicBlock *);

  // Called at pre-visit during DFS walk.
  void visitPre(const BasicBlock *);

  // Called at mid-visit during DFS walk.
  void visitMid(const BasicBlock *, DFSDirection);

  // Called at post-visit during DFS walk.
  void visitPost(const BasicBlock *, const BasicBlock *, DFSDirection);
  void visitBackedge(const BasicBlock *, const BasicBlock *, DFSDirection);

  void pushDFS(DFSStack &, const BasicBlock *, const BasicBlock *,
               DFSDirection);
  void popDFS(DFSStack &, const BasicBlock *);
  void debugBracketList(const BracketList &);

  unsigned DFSNumber;
  unsigned ClassNumber;
  SmallDenseMap<const BasicBlock *, BlockCEData, 8> BlockData;
  bool Computed;
  const BasicBlock *FakeStart;
  const BasicBlock *FakeEnd;
  std::vector<BracketList> BracketLists;
  std::vector<unsigned> BListForwarding;
};
}

#endif
