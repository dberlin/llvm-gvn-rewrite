//===- ControlEquivalence.cpp - Compute Control Equivalence ---------------===//
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
#include "llvm/Analysis/ControlEquivalence.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <queue>
#include <utility>
#include <algorithm>
using namespace llvm;
char ControlEquivalence::ID = 0;
#define DEBUG_TYPE "controlequiv"
INITIALIZE_PASS(ControlEquivalence, "controlequiv",
                "Control Equivalence Construction", true, true);

// Run the main algorithm starting from the exit blocks. We proceed to perform
// an undirected depth-first backwards traversal that determines class numbers
// for all participating blocks. Takes O(E) time and O(N) space.

bool ControlEquivalence::runOnFunction(Function &F) {
  Computed = false;
  DFSNumber = 0;
  ClassNumber = 1;

  // The algorithm requires we transform the CFG into a strongly connected
  // component. We make a fake end, connect exit blocks to it, and then connect
  // the fake end and the real start (since we only have one of those).
  FakeEnd = BasicBlock::Create(F.getContext(), "FakeEnd");
  BlockData[FakeEnd].FakeSuccEdges.push_back(&F.getEntryBlock());
  BlockData[&F.getEntryBlock()].FakePredEdges.push_back(FakeEnd);
  //  BlockData.resize(F.size());
  for (auto &B : F) {
    BlockCEData &Info = BlockData[&B];
    /*    // If this is an unreachable block, we don't care about it
    if (pred_empty(&B)) {
      Info.Participates = false;
      }*/
    // If there are no successors, we need to connect it to the exit block
    if (succ_empty(&B)) {
      // Connect leaves to fake end
      Info.FakeSuccEdges.push_back(FakeEnd);
      BlockData[FakeEnd].FakePredEdges.push_back(&B);
    }
  }
  runUndirectedDFS(&F.getEntryBlock());
#ifndef NDEBUG
  for (auto &B : F) {
    dbgs() << "Class number for block ";
    B.printAsOperand(dbgs());
    dbgs() << " is " << BlockData[&B].ClassNumber << "\n";
  }
#endif
  Computed = true;
  return false;
}

void ControlEquivalence::releaseMemory() {
  BlockData.clear();
  delete FakeEnd;
}

// print - Show contents in human readable format...
void ControlEquivalence::print(raw_ostream &O, const Module *M) const {}

// Performs and undirected DFS walk of the CFG. Conceptually all nodes are
// expanded, splitting "predecessors" and "successors" out into separate nodes.
// During the
// traversal, edges towards the representative type are preferred.
//
//   \ /        - Pre-visit: When N1 is visited in direction D the preferred
//    x   N1      edge towards N is taken next, calling VisitPre(N).
//    |         - Mid-visit: After all edges out of N2 in direction D have
//    |   N       been visited, we switch the direction and start considering
//    |           edges out of N1 now, and we call VisitMid(N).
//    x   N2    - Post-visit: After all edges out of N1 in direction opposite
//   / \          to D have been visited, we pop N and call VisitPost(N).
//
// This will yield a true spanning tree (without cross or forward edges) and
// also discover proper back edges in both directions.
void ControlEquivalence::runUndirectedDFS(const BasicBlock *ExitBlock) {
  DFSStack Stack;
  // Start out walking backwards
  pushDFS(Stack, ExitBlock, nullptr, PredDirection);

  while (!Stack.empty()) {
    DFSStackEntry &Entry = Stack.top();
    const BasicBlock *B = Entry.Block;

    if (Entry.Direction == PredDirection) {

      // First visit in pred direction, then swap directions, then visit in succ
      // direction
      // Pred direction
      if (Entry.Pred != Entry.PredEnd) {
        const BasicBlock *Pred = *Entry.Pred;
        ++(Entry.Pred);
        BlockCEData &PredData = BlockData.find(Pred)->second;
        if (!PredData.Participates)
          continue;
        if (PredData.Visited)
          continue;
        // If it's on the stack, we've found a backedge, otherwise, push it
        // and preorder-visit
        if (PredData.OnStack) {
          if (Pred != Entry.ParentBlock)
            visitBackedge(B, Pred, PredDirection);
        } else {
          pushDFS(Stack, Pred, B, PredDirection);
          visitPre(Pred);
        }
        continue;
      }
      // Swap directions to successors
      if (Entry.Succ != Entry.SuccEnd) {
        Entry.Direction = SuccDirection;
        visitMid(B, PredDirection);
        continue;
      }
    }

    if (Entry.Direction == SuccDirection) {
      if (Entry.Succ != Entry.SuccEnd) {
        const BasicBlock *Succ = *Entry.Succ;
        ++(Entry.Succ);
        BlockCEData &SuccData = BlockData.find(Succ)->second;
        if (!SuccData.Participates)
          continue;
        if (SuccData.Visited)
          continue;
        // If it's on the stack, we've found a backedge, otherwise, push it
        // and preorder-visit
        if (SuccData.OnStack) {
          if (Succ != Entry.ParentBlock)
            visitBackedge(B, Succ, SuccDirection);
        } else {
          pushDFS(Stack, Succ, B, SuccDirection);
          visitPre(Succ);
        }
        continue;
      }
      // Swap directions to predecessors
      if (Entry.Pred != Entry.PredEnd) {
        Entry.Direction = PredDirection;
        visitMid(B, SuccDirection);
        continue;
      }
    }
    // Pop block from stack when done with all preds and succs.
    assert(Entry.Pred == Entry.PredEnd &&
           "Did not finish predecessors before popping");
    assert(Entry.Succ == Entry.SuccEnd &&
           "Did not finish successors before popping");
    popDFS(Stack, B);
    visitPost(B, Entry.ParentBlock, Entry.Direction);
  }
}

void ControlEquivalence::pushDFS(DFSStack &Stack, const BasicBlock *B,
                                 const BasicBlock *From,
                                 DFSDirection Direction) {
  auto BlockResult = BlockData.find(B);
  assert(BlockResult != BlockData.end() && "Should have found block data");

  auto &BlockResultData = BlockResult->second;
  assert(!BlockResultData.Visited && "Should not have been visited yet");
  assert(BlockResultData.Participates &&
         "Somehow visited a node that doesn't participate");

  BlockResultData.OnStack = true;
  Stack.push(DFSStackEntry{
      Direction, combined_pred_begin(B, BlockResultData.FakePredEdges),
      combined_pred_end(B, BlockResultData.FakePredEdges),
      combined_succ_begin(B, BlockResultData.FakeSuccEdges),
      combined_succ_end(B, BlockResultData.FakeSuccEdges), From, B});
}
void ControlEquivalence::popDFS(DFSStack &Stack, const BasicBlock *B) {
  assert(Stack.top().Block == B && "Stack top is wrong");
  auto BlockResult = BlockData.find(B);

  assert(BlockResult != BlockData.end() && "Should have found block data");
  auto &BlockResultData = BlockResult->second;
  BlockResultData.OnStack = false;
  BlockResultData.Visited = true;
  Stack.pop();
}

void ControlEquivalence::visitPre(const BasicBlock *B) {
  DEBUG(dbgs() << "pre-order visit of block ");
  DEBUG(B->printAsOperand(dbgs()));
  DEBUG(dbgs() << "\n");

  BlockData[B].DFSNumber = ++DFSNumber;
  DEBUG(dbgs() << "Assigned DFS Number " << BlockData[B].DFSNumber << "\n");
}

void ControlEquivalence::visitMid(const BasicBlock *B, DFSDirection Direction) {
  DEBUG(dbgs() << "mid-order visit of block ");
  DEBUG(B->printAsOperand(dbgs()));
  DEBUG(dbgs() << "\n");
  BlockCEData &Info = BlockData[B];
  BracketList &BList = Info.BList;
  // Remove brackets pointing to this node [line:19].
  for (auto BLI = BList.begin(), BLE = BList.end(); BLI != BLE;) {
    if (BLI->To == B && BLI->Direction != Direction) {
      BLI = BList.erase(BLI);
    } else {
      ++BLI;
    }
  }

  // Potentially introduce artificial dependency from start to ends.
  if (BList.empty()) {
    assert(Direction == PredDirection && "Should not have to do this unless we "
                                         "are going in the backwards "
                                         "direction");
    visitBackedge(B, FakeEnd, PredDirection);
  }

  // Potentially start a new equivalence class [line:37]
  Bracket &Recent = BList.back();
  if (Recent.RecentSize != BList.size()) {
    Recent.RecentSize = BList.size();
    Recent.RecentClass = ++ClassNumber;
  }
  Info.ClassNumber = Recent.RecentClass;
  DEBUG(dbgs() << "Assigned class number is " << Info.ClassNumber << "\n");
}
void ControlEquivalence::visitPost(const BasicBlock *B,
                                   const BasicBlock *ParentBlock,
                                   DFSDirection Direction) {
  DEBUG(dbgs() << "post-order visit of block ");
  DEBUG(B->printAsOperand(dbgs()));
  DEBUG(dbgs() << "\n");
  BlockCEData &Info = BlockData[B];
  BracketList &BList = Info.BList;
  // Remove brackets pointing to this node [line:19].

  for (auto BLI = BList.begin(), BLE = BList.end(); BLI != BLE;) {
    if (BLI->To == B && BLI->Direction != Direction) {
      BLI = BList.erase(BLI);
    } else {
      ++BLI;
    }
  }

  // Propagate bracket list up the DFS tree [line:13].
  if (ParentBlock != nullptr) {
    BracketList &ParentBList = BlockData[ParentBlock].BList;
    ParentBList.splice(ParentBList.end(), BList);
  }
}

void ControlEquivalence::visitBackedge(const BasicBlock *From,
                                       const BasicBlock *To,
                                       DFSDirection Direction) {
  DEBUG(dbgs() << "visit backedge from block ");
  DEBUG(From->printAsOperand(dbgs()));
  DEBUG(dbgs() << " to block ");
  DEBUG(To->printAsOperand(dbgs()));
  DEBUG(dbgs() << "\n");

  // Push backedge onto the bracket list [line:25].
  BlockCEData &Info = BlockData[From];
  BracketList &BList = Info.BList;
  BList.emplace_back(Bracket{Direction, 0, 0, From, To});
}
