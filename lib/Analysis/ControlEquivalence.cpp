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
  BlockData[FakeEnd].FakeSucc = &F.getEntryBlock();
  BlockData[&F.getEntryBlock()].FakePred = FakeEnd;
  //  BlockData.resize(F.size());
  for (auto &B : F) {
    BlockCEData &Info = BlockData[&B];
    // If this is an unreachable block, we don't care about it
    if (pred_empty(&B) && (&B != &F.getEntryBlock())) {
      Info.Participates = false;
    }
    // If there are no successors, we need to connect it to the exit block
    if (succ_empty(&B)) {
      Info.FakeSucc = FakeEnd;
    }
  }
  SmallPtrSet<const BasicBlock *, 8> Visited;
  SmallVector<const BasicBlock *, 32> AllNodes;

  // First do an undirected DFS, and get all the nodes in DFS preorder
  runDFS(FakeEnd, Visited, AllNodes);
  // Now go through all the nodes in DFS post order, and compute cycle
  // equivalence
  for (auto POI = AllNodes.rbegin(), POE = AllNodes.rend(); POI != POE; ++POI) {
    cycleEquiv(*POI);
  }

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

// Construct an undirected Depth First Spanning Tree, produce the list of nodes,
// in preorder, in AllNodes.
void ControlEquivalence::runDFS(const BasicBlock *BB,
                                SmallPtrSetImpl<const BasicBlock *> &Visited,
                                SmallVectorImpl<const BasicBlock *> &AllNodes) {
  assert(!Visited.count(BB) && "We already visited this node");
  BlockCEData &Info = BlockData[BB];
  Info.DFSNumber = ++DFSNumber;
  Visited.insert(BB);
  AllNodes.emplace_back(BB);

  // We get preds, then succs, then fake preds, then fake succs
  for (auto CCI = all_edges_begin(BB, Info.FakePred, Info.FakeSucc),
            CCE = all_edges_end(BB);
       CCI != CCE; ++CCI) {
    // If we haven't visited, mark this as a child, and go on
    if (!Visited.count(*CCI)) {
      Info.Children.push_back(*CCI);
      BlockData[*CCI].Parent = BB;
      runDFS(*CCI, Visited, AllNodes);
    } else if (*CCI != Info.Parent)
      // We've hit something other than our parent
      // node again so this must be a backedge
      Info.Backedges.push_back(*CCI);
  }
}

// The actual proof and understanding of this algorithm is a bit detailed, but
// basically, we are computing compact bracket set for the edges in the graph,
// based on the undirected DFS graph. Two nodes/edges in the graph are
// control/cycle equivalent iff they have the same compact bracket set
// This algorithm pretty directly corresponds to Figure 4 of the Cycle
// Equivalence paper.

void ControlEquivalence::cycleEquiv(const BasicBlock *From) {
  BlockCEData &Info = BlockData[From];
  // Compute hi(n)
  const BasicBlock *hi0 = nullptr;
  unsigned hi0DFS;
  for (auto &BB : Info.Backedges)
    if (!hi0 || BlockData[BB].DFSNumber < hi0DFS) {
      hi0DFS = BlockData[BB].DFSNumber;
      hi0 = BB;
    }
  const BasicBlock *hi1 = nullptr;
  unsigned hi1DFS;
  const BasicBlock *hi2 = nullptr;
  unsigned hi2DFS;
  for (auto &BB : Info.Children) {
    if (!hi1 || BlockData[Info.Hi].DFSNumber < hi1DFS) {
      if (!hi2 || hi1DFS < hi2DFS) {
        hi2 = hi1;
        hi2DFS = hi1DFS;
      }
      hi1 = BlockData[BB].Hi;
      hi1DFS = BlockData[hi1].DFSNumber;
    } else if (!hi2 || BlockData[Info.Hi].DFSNumber < hi2DFS) {
      hi2 = Info.Hi;
      hi2DFS = BlockData[Info.Hi].DFSNumber;
    }
  }

  // Take the min of hi0, hi1
  Info.Hi = !hi0 ? hi1 : !hi1 ? hi0 : (hi0DFS < hi1DFS) ? hi0 : hi1;

  // Compute the bracketlist
  for (auto &BB : Info.Children)
    Info.BList.splice(Info.BList.end(), BlockData[BB].BList);
  // TODO: We can remove it in constant time by tracking the container for the
  // edge
  for (auto &BB : Info.Capping)
    Info.BList.remove_if([&From, &BB](const Bracket &BR) {
      return BR.From == From && BR.To == BB;
    });
  for (auto &BB : Info.Backedges) {
    // See if it's our descendant
    auto &BBData = BlockData[BB];
    if (BBData.DFSNumber > Info.DFSNumber) {
      // TODO fix this
      Info.BList.remove_if([&From, &BB](const Bracket &BR) {
        return BR.From == From && BR.To == BB;
      });
      if (BBData.ClassNumber == 0)
        BBData.ClassNumber = ++ClassNumber;
    } else if (BBData.DFSNumber <= Info.DFSNumber) {
      auto Place = Info.BList.insert(Info.BList.end(), {From, BB, 0, 0});
    }
    if (hi2 && (!hi0 || hi0DFS > hi2DFS)) {
      // Create a capping backedge
      BlockData[hi2].Capping.push_back(From);
      auto Place = Info.BList.insert(Info.BList.end(), {From, hi2, 0, 0});
    }

    // Determine class for edge from parent(n) to n
    const BasicBlock *Parent = Info.Parent;
    if (Parent) {
      BlockCEData &ParentInfo = BlockData[Parent];
      Bracket &Back = Info.BList.back();
      if (Back.RecentSize != Info.BList.size()) {
        Back.RecentSize = Info.BList.size();
        Back.RecentClass = ++ClassNumber;
      }
      ParentInfo.ClassNumber = Back.RecentClass;

      if (Back.RecentSize == 1)
        BlockData[Back.To].ClassNumber = ParentInfo.ClassNumber;
    }
  }
}
