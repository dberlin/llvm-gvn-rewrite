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
  // component. We make a fake start and end, connect blocks to it, then connect
  // them to each other.

  FakeEnd = BasicBlock::Create(F.getContext(), "FakeEnd");
  FakeStart = BasicBlock::Create(F.getContext(), "FakeStart");
  BlockData[FakeEnd].FakeSucc = FakeStart;
  BlockData[FakeStart].FakePreds.push_back(FakeEnd);
  BlockData[FakeStart].FakeSucc = &F.getEntryBlock();
  BlockData[&F.getEntryBlock()].FakePreds.push_back(FakeStart);
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
      BlockData[FakeEnd].FakePreds.push_back(&B);
    }
  }
  std::vector<const BasicBlock *> AllNodes;
  // First do an undirected DFS, and get all the nodes in DFS preorder
  runDFS(FakeStart, AllNodes);
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
void ControlEquivalence::runDFS(const BasicBlock *StartBlock,
                                std::vector<const BasicBlock *> &AllNodes) {
  std::stack<DFSStackEntry> DFSStack;
  BlockCEData &StartInfo = BlockData[StartBlock];
  StartInfo.DFSNumber = ++DFSNumber;
  StartInfo.OnStack = true;
  // Push the start block
  DFSStack.push(
      {StartBlock, nullptr,
       all_edges_begin(StartBlock, StartInfo.FakePreds, StartInfo.FakeSucc),
       all_edges_end(StartBlock, StartInfo.FakePreds)});
  AllNodes.push_back(StartBlock);
  while (!DFSStack.empty()) {
    DFSStackEntry &Entry = DFSStack.top();
    if (Entry.ChildIt != Entry.ChildEnd) {
      const BasicBlock *Child = *Entry.ChildIt;
      ++Entry.ChildIt;
      BlockCEData &ChildInfo = BlockData[Child];
      if (ChildInfo.Visited)
        continue;
      DEBUG(dbgs() << "Going to visit ");
      DEBUG(Child->printAsOperand(dbgs()));
      DEBUG(dbgs() << " from ");
      DEBUG(Entry.BB->printAsOperand(dbgs()));
      DEBUG(dbgs() << "\n");
      if (ChildInfo.OnStack) {
        if (Child != Entry.Parent) {
          DEBUG(dbgs() << "Adding ");
          DEBUG(Child->printAsOperand(dbgs()));
          DEBUG(dbgs() << " as backedge of ");
          DEBUG(Entry.BB->printAsOperand(dbgs()));
          DEBUG(dbgs() << "\n");
          //     DEBUG(dbgs() << " parent was ");
          //     if (Info.Parent)
          //       DEBUG(Info.Parent->printAsOperand(dbgs()));
          //     else
          //       DEBUG(dbgs() << " null");
          // This is a backedge
          //     Info.Backedges.emplace_front(BB, *CCI);
          auto &EntryInfo = BlockData[Entry.BB];
          EntryInfo.Backedges.emplace_front(Entry.BB, Child);
          ChildInfo.Backedges.emplace_front(Entry.BB, Child);
        }
      } else {
        ChildInfo.Parent = Entry.BB;
        auto &EntryInfo = BlockData[Entry.BB];

        DEBUG(dbgs() << "Parent set to ");
        if (ChildInfo.Parent)
          DEBUG(ChildInfo.Parent->printAsOperand(dbgs()));
        else
          DEBUG(dbgs() << " null");
        DEBUG(dbgs() << "\n");
        EntryInfo.Children.push_back(Child);
        DFSStack.push(
            {Child, Entry.BB,
             all_edges_begin(Child, ChildInfo.FakePreds, ChildInfo.FakeSucc),
             all_edges_end(Child, ChildInfo.FakePreds)});
        ChildInfo.OnStack = true;
        ChildInfo.DFSNumber = ++DFSNumber;
        AllNodes.push_back(Child);
        DEBUG(dbgs() << "DFS Number set to " << ChildInfo.DFSNumber << "\n");
      }
    } else {
      auto &EntryInfo = BlockData[Entry.BB];
      DFSStack.pop();
      EntryInfo.OnStack = false;
      EntryInfo.Visited = true;
    }
  }
}

static void debugEdge(std::pair<const BasicBlock *, const BasicBlock *> Edge) {
  dbgs() << "(";
  Edge.first->printAsOperand(dbgs());
  dbgs() << ",";
  Edge.second->printAsOperand(dbgs());
  dbgs() << ")";
}

void ControlEquivalence::debugNodeInfo(const BasicBlock *BB) {
  dbgs() << "Info for node ";
  BB->printAsOperand(dbgs());
  dbgs() << "\n";
  dbgs() << "Parent is ";
  if (BlockData[BB].Parent)
    BlockData[BB].Parent->printAsOperand(dbgs());
  else
    dbgs() << "null";

  dbgs() << "\nList of backedges:{";
  for (auto &BBEdge : BlockData[BB].Backedges) {
    debugEdge(BBEdge);
    dbgs() << " ";
  }
  dbgs() << "}\n";
  dbgs() << "List of children:{";
  for (auto &BB : BlockData[BB].Children) {
    BB->printAsOperand(dbgs());
    dbgs() << " ";
  }
  dbgs() << "}\n";
}

void ControlEquivalence::debugBracketList(const BracketList &BList) {
  dbgs() << "{";
  for (auto &Bracket : BList) {
    dbgs() << "(";
    Bracket.Edge.first->printAsOperand(dbgs());
    dbgs() << ",";
    Bracket.Edge.second->printAsOperand(dbgs());
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
  DEBUG(dbgs() << "Computing for ");
  DEBUG(From->printAsOperand(dbgs()));
  DEBUG(dbgs() << "\n");
  DEBUG(debugNodeInfo(From));

  BlockCEData &Info = BlockData[From];
  // Compute hi(n)

  // hi0 = min (m.dfsnum | (n, m) is a backedge)
  const BasicBlock *hi0 = nullptr;
  unsigned hi0DFS;
  for (auto &BBEdge : Info.Backedges) {
    const BasicBlock *OtherEnd = BBEdge.second == From ? BBEdge.first : BBEdge.second;

    if (!hi0 || BlockData[OtherEnd].DFSNumber < hi0DFS) {
      hi0 = OtherEnd;
      hi0DFS = BlockData[hi0].DFSNumber;
    }
  }

  // hi1 =min (c.hi | c is a child of of n)
  const BasicBlock *hi1 = nullptr;
  unsigned hi1DFS;
  for (auto &BB : Info.Children) {
    assert(BB != From);
    BlockCEData &BBData = BlockData[BB];
    assert(BBData.Hi != nullptr);
    if (!hi1 || BlockData[BBData.Hi].DFSNumber < hi1DFS) {
      hi1 = BBData.Hi;
      hi1DFS = BlockData[BBData.Hi].DFSNumber;
    }
  }
  Info.Hi = !hi0 ? hi1 : !hi1 ? hi0 : (hi0DFS < hi1DFS) ? hi0 : hi1;
  // hi2 = min (c.hi | c is a child of n other than hi1)

  const BasicBlock *hi2 = nullptr;
  unsigned hi2DFS;
  for (auto &BB : Info.Children) {
    assert(BB != From);
    BlockCEData &BBData = BlockData[BB];
    assert(BBData.Hi != nullptr);
    if (BBData.Hi == hi1)
      continue;
    if (hi2 || BlockData[BBData.Hi].DFSNumber < hi2DFS) {
      hi2 = BBData.Hi;
      hi2DFS = BlockData[BBData.Hi].DFSNumber;
    }
  }

  // for (auto &BBEdge : Info.Children) {
  //   assert(BBEdge.second != From);
  //   BlockCEData &BBData = BlockData[BBEdge.second];
  //   if (!hi1 || BlockData[BBData.Hi].DFSNumber < hi1DFS) {
  //     if (!hi2 || hi1DFS < hi2DFS) {
  //       hi2 = hi1;
  //       hi2DFS = hi1DFS;
  //     }
  //     hi1 = BBData.Hi;
  //     hi1DFS = BlockData[hi1].DFSNumber;
  //   } else if (!hi2 || BlockData[BBData.Hi].DFSNumber < hi2DFS) {
  //     hi2 = BBData.Hi;
  //     hi2DFS = BlockData[hi2].DFSNumber;
  //   }
  // }

  // Take the min of hi0, hi1
  // Compute the bracketlist
  for (auto &BB : Info.Children) {
    assert(BB != From);
    // n.blist = concat (c.blist, n.blist)
    Info.BList.splice(Info.BList.begin(), BlockData[BB].BList);
  }

  // TODO: We can remove it in constant time by tracking the container for the
  // edge
  for (auto &BB : Info.Capping) {
    assert(BB != From);
    size_t beforesize = Info.BList.size();
    Info.BList.remove_if(
        [&BB](const Bracket &BR) { return BR.Edge.second == BB; });
    assert(beforesize == 0 || beforesize != Info.BList.size());
  }

  // For each backedge b from a descendant of n to n
  for (auto &BBEdge : Info.Backedges) {
    const BasicBlock *OtherEnd = BBEdge.second == From ? BBEdge.first : BBEdge.second;
    // See if it's our descendant
    BlockCEData &BBData = BlockData[OtherEnd];
    if (BBData.DFSNumber > Info.DFSNumber) {
      // TODO fix this
      size_t beforesize = Info.BList.size();
      // delete (n.list, b)
      Info.BList.remove_if(
          [&BBEdge](const Bracket &BR) { return BR.Edge == BBEdge; });

      DEBUG(dbgs() << "Backedge was from ");
      DEBUG(From->printAsOperand(dbgs()));
      DEBUG(dbgs() << " to  ");
      DEBUG(OtherEnd->printAsOperand(dbgs()));
      DEBUG(dbgs() << "\n");

      assert((beforesize == 0 || beforesize != Info.BList.size()));
      if (BBData.ClassNumber == 0) {
        BBData.ClassNumber = ++ClassNumber;
        DEBUG(dbgs() << "Bracket list for ");
        DEBUG(From->printAsOperand(dbgs()));
        DEBUG(dbgs() << " is ");
        DEBUG(debugBracketList(Info.BList));
        DEBUG(dbgs() << "Backedge updated class number for ");
        DEBUG(OtherEnd->printAsOperand(dbgs()));
        DEBUG(dbgs() << " to " << BBData.ClassNumber << "\n");
      }
    }
  }

  for (auto &BBEdge : Info.Backedges) {
    const BasicBlock *OtherEnd = BBEdge.second == From ? BBEdge.first : BBEdge.second;
    // See if it's our ancestor
    BlockCEData &BBData = BlockData[OtherEnd];
    if (BBData.DFSNumber <= Info.DFSNumber) {
      auto Place = Info.BList.insert(Info.BList.begin(), {BBEdge, 0, 0});
    }
  }

  if (hi2 && (!hi0 || hi0DFS > hi2DFS)) {
    // Create a capping backedge
    BlockData[hi2].Capping.push_front(From);
    auto Place = Info.BList.insert(Info.BList.begin(), {{From, hi2}, 0, 0});
  }
  DEBUG(dbgs() << "Bracket list for ");
  DEBUG(From->printAsOperand(dbgs()));
  DEBUG(dbgs() << " is ");
  DEBUG(debugBracketList(Info.BList));

  // If n is not the root of the DFS tree
  if (From != FakeStart || 1) {
    // Determine class for edge from parent(n) to n
    const BasicBlock *Parent = Info.Parent;
    if (Parent) {
      Bracket &Top = Info.BList.front();
      if (Top.RecentSize != Info.BList.size()) {
        Top.RecentSize = Info.BList.size();
        Top.RecentClass = ++ClassNumber;
      }
      DEBUG(dbgs() << "Parent updated class number for ");
      DEBUG(From->printAsOperand(dbgs()));
      DEBUG(dbgs() << " to " << Top.RecentClass << "\n");
      Info.ClassNumber = Top.RecentClass;

      if (Top.RecentSize == 1) {
        // assert(Top.From != From);
        assert(Top.Edge.second != From);
        DEBUG(dbgs() << "Top Size updated class number for ");
        DEBUG(Top.Edge.second->printAsOperand(dbgs()));
        DEBUG(dbgs() << " to " << Info.ClassNumber << "\n");
        BlockData[Top.Edge.second].ClassNumber = Info.ClassNumber;
      }
    }
  }
}
