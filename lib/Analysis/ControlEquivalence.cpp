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
  FakeStart = BasicBlock::Create(F.getContext(), "FakeStart");
  BlockData[FakeEnd].FakeSucc = FakeStart;
  BlockData[FakeStart].FakePred = FakeEnd;
  BlockData[FakeStart].FakeSucc = &F.getEntryBlock();
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
  runDFS(FakeStart, Visited, AllNodes);
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
  delete FakeStart;
}

// print - Show contents in human readable format...
void ControlEquivalence::print(raw_ostream &O, const Module *M) const {}

// Construct an undirected Depth First Spanning Tree, produce the list of nodes,
// in preorder, in AllNodes.
void ControlEquivalence::runDFS(const BasicBlock *BB,
                                SmallPtrSetImpl<const BasicBlock *> &Visited,
                                SmallVectorImpl<const BasicBlock *> &AllNodes) {
  DEBUG(dbgs() << "Starting visit of block ");
  DEBUG(BB->printAsOperand(dbgs()));
  DEBUG(dbgs() << "\n");

  assert(!Visited.count(BB) && "We already visited this node");
  BlockCEData &Info = BlockData[BB];
  Info.DFSNumber = ++DFSNumber;
  DEBUG(dbgs() << "DFS Number set to " << Info.DFSNumber << "\n");
  Visited.insert(BB);
  AllNodes.emplace_back(BB);

  // We get preds, then succs, then fake preds, then fake succs
  for (auto CCI = all_edges_begin(BB, Info.FakePred, Info.FakeSucc),
            CCE = all_edges_end(BB);
       CCI != CCE; ++CCI) {
    // If we haven't visited, mark this as a child, and go on
    if (!Visited.count(*CCI)) {
      Info.Children.push_front(*CCI);
      BlockData[*CCI].Parent = BB;
      DEBUG(dbgs() << "Going to visit ");
      DEBUG((*CCI)->printAsOperand(dbgs()));
      DEBUG(dbgs() << " from ");
      DEBUG(BB->printAsOperand(dbgs()));
      DEBUG(dbgs() << "\n");

      runDFS(*CCI, Visited, AllNodes);
    } else if (*CCI != Info.Parent) {
      // We've hit something other than our parent
      // node again so this must be a backedge
      Info.Backedges.push_front(*CCI);
      DEBUG(dbgs() << "Adding ");
      DEBUG((*CCI)->printAsOperand(dbgs()));
      DEBUG(dbgs() << " as backedge of ");
      DEBUG(BB->printAsOperand(dbgs()));
      DEBUG(dbgs() << "\n");
    }
  }
}

void ControlEquivalence::debugBracketList(const BracketList &BList) {
  dbgs() << "{";
  for (auto &Bracket : BList) {
    dbgs() << "(";
    Bracket.From->printAsOperand(dbgs());
    dbgs() << ",";
    Bracket.To->printAsOperand(dbgs());
    dbgs() << ") ";
  }
  dbgs() << "}\n";
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
  for (auto &BB : Info.Backedges) {
    assert(BB != From);

    if (!hi0 || BlockData[BB].DFSNumber < hi0DFS) {
      hi0 = BB;
      hi0DFS = BlockData[hi0].DFSNumber;
    }
  }

  const BasicBlock *hi1 = nullptr;
  unsigned hi1DFS;
  const BasicBlock *hi2 = nullptr;
  unsigned hi2DFS;
  for (auto &BB : Info.Children) {
    assert(BB != From);
    BlockCEData &BBData = BlockData[BB];
    if (!hi1 || BlockData[BBData.Hi].DFSNumber < hi1DFS) {
      if (!hi2 || hi1DFS < hi2DFS) {
        hi2 = hi1;
        hi2DFS = hi1DFS;
      }
      hi1 = BBData.Hi;
      hi1DFS = BlockData[hi1].DFSNumber;
    } else if (!hi2 || BlockData[BBData.Hi].DFSNumber < hi2DFS) {
      hi2 = BBData.Hi;
      hi2DFS = BlockData[hi2].DFSNumber;
    }
  }

  // Take the min of hi0, hi1
  Info.Hi = !hi0 ? hi1 : !hi1 ? hi0 : (hi0DFS < hi1DFS) ? hi0 : hi1;
  // Compute the bracketlist
  for (auto &BB : Info.Children) {
    assert(BB != From);
    Info.BList.splice(Info.BList.begin(), BlockData[BB].BList);
  }
  
  // TODO: We can remove it in constant time by tracking the container for the
  // edge
  for (auto &BB : Info.Capping) {
    assert(BB != From);
    size_t beforesize = Info.BList.size();
    Info.BList.remove_if([&From, &BB](const Bracket &BR) {
      return (BR.From == From && BR.To == BB) ||
             (BR.To == From && BR.From == BB);
    });
    assert(beforesize == 0 || beforesize!= Info.BList.size());
  }

  for (auto &BB : Info.Backedges) {
    assert(BB != From);
    // See if it's our descendant
    BlockCEData &BBData = BlockData[BB];
    if (BBData.DFSNumber > Info.DFSNumber) {
      // TODO fix this
      size_t beforesize = Info.BList.size();
      Info.BList.remove_if([&From, &BB](const Bracket &BR) {
        return (BR.From == From && BR.To == BB) ||
               (BR.To == From && BR.From == BB);

      });
      DEBUG(dbgs() << "Backedge was from ");
      DEBUG(From->printAsOperand(dbgs()));
      DEBUG(dbgs() << " to  ");
      DEBUG(BB->printAsOperand(dbgs()));
      DEBUG(dbgs() << "\n");

      assert((beforesize == 0 || beforesize!= Info.BList.size()));
      if (BBData.ClassNumber == 0) {
        BBData.ClassNumber = ++ClassNumber;
        DEBUG(dbgs() << "Bracket list for ");
        DEBUG(From->printAsOperand(dbgs()));
        DEBUG(dbgs() << " is ");
        DEBUG(debugBracketList(Info.BList));
      }
    }
    
    for (auto &BB : Info.Backedges){    
      if (BBData.DFSNumber <= Info.DFSNumber) {
        auto Place = Info.BList.insert(Info.BList.begin(), {From, BB, 0, 0});
      }
    }
    
    if (hi2 && (!hi0 || hi0DFS > hi2DFS)) {
      // Create a capping backedge
      BlockData[hi2].Capping.push_front(From);
      auto Place = Info.BList.insert(Info.BList.begin(), {From, hi2, 0, 0});
    }
    DEBUG(dbgs() << "Bracket list for ");
    DEBUG(From->printAsOperand(dbgs()));
    DEBUG(dbgs() << " is ");
    DEBUG(debugBracketList(Info.BList));

    // Determine class for edge from parent(n) to n
    const BasicBlock *Parent = Info.Parent;
    if (Parent) {
      BlockCEData &ParentInfo = BlockData[Parent];
      Bracket &Top = Info.BList.front();
      if (Top.RecentSize != Info.BList.size()) {
        Top.RecentSize = Info.BList.size();
        Top.RecentClass = ++ClassNumber;
      }
      ParentInfo.ClassNumber = Top.RecentClass;

      if (Top.RecentSize == 1) {
        // assert(Top.From != From);
        assert(Top.To != From);

        BlockData[Top.To].ClassNumber = ParentInfo.ClassNumber;
      }
    }
  }
}
