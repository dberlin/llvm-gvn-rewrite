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
  ControlEquivalence() : FunctionPass(ID) {
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
    // Iterator that points to us in the current bracket list, to make delete
    // O(1).
    // std::list iterators only invalidate the iterator you erase.
    BracketList::iterator BListPointer;

    BlockCEData()
        : ClassNumber(0), DFSNumber(0), Visited(false), OnStack(false),
          Participates(false) {}
  };

  struct DFSStackEntry {
    // Direction currently used in DFS walk.
    DFSDirection Direction;
    // Iterator used for "pred" direction.
    const_pred_iterator Pred;
    // Iterator used for "succ" direction.
    succ_const_iterator Succ;
    // Parent Block of entry during DFS walk.
    const BasicBlock *ParentBlock;
    // Basic Block that this stack entry belongs to
    const BasicBlock *Block;
  };
  // The stack is used during the undirected DFS walk.
  typedef std::stack<DFSStackEntry> DFSStack;
  typedef SmallVectorImpl<const BasicBlock *> BlockVector;
  void collectExits(Function &, BlockVector &);
  void runUndirectedDFS(const BlockVector &);
  void determineParticipation(const BlockVector &);

  // Called at pre-visit during DFS walk.
  void visitPre(const BasicBlock *);

  // Called at mid-visit during DFS walk.
  void visitMid(const BasicBlock *, DFSDirection, const BlockVector &);

  // Called at post-visit during DFS walk.
  void visitPost(const BasicBlock *, const BasicBlock *, DFSDirection);
  void visitBackedge(const BasicBlock *, const BasicBlock *, DFSDirection);

  void pushDFS(DFSStack &, const BasicBlock *, const BasicBlock *,
               DFSDirection);
  void popDFS(DFSStack &, const BasicBlock *);
  unsigned DFSNumber;
  unsigned ClassNumber;
  DenseMap<const BasicBlock *, BlockCEData> BlockData;
};
}

#endif
