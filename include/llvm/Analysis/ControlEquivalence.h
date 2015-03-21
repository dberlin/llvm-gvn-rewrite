//===- ControlEquivalence.h - Compute Control Equivalence-------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
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
  typedef enum { InputDirection, UseDirection } DFSDirection;

  struct Bracket {
    // Direction in which this bracket was added.
    DFSDirection Direction;
    // Cached class when bracket was topmost.
    unsigned RecentClass;
    // Cached set-size when bracket was topmost.
    unsigned RecentSize;
    // Block that this bracket originates from.
    BasicBlock *From;
    // Block that this bracket points to.
    BasicBlock *To;
  };
  typedef std::list<Bracket> BracketList;

  struct ControlBlockData {
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
    BracketList Blist;
    ControlBlockData()
        : ClassNumber(0), DFSNumber(0), Visited(false), OnStack(false),
          Participates(false) {}
  };

  struct DFSStackEntry {
    // Direction currently used in DFS walk.
    DFSDirection Direction;
    // Iterator used for "input" direction.
    pred_iterator Input;
    // Iterator used for "use" direction.
    succ_iterator Use;
    // Parent Block of entry during DFS walk.
    BasicBlock *ParentBlock;
    // Basic Block that this stack entry belongs to
    BasicBlock *Block;
  };

  // The stack is used during the undirected DFS walk.
  typedef std::stack<DFSStackEntry> DFSStack;
  unsigned DFSNumber;
  unsigned ClassNumber;
  DenseMap<BasicBlock *, ControlBlockData> BlockData;
};
}

#endif
