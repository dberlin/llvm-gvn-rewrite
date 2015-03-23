//===- ControlDependence.cpp - Compute Control Equivalence ---------------===//
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
#include "llvm/Analysis/ControlDependence.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <queue>
#include <utility>
#include <algorithm>
using namespace llvm;
char ControlDependence::ID = 0;
#define DEBUG_TYPE "controlequiv"
INITIALIZE_PASS(ControlDependence, "controldep",
                "Control Dependence Construction", true, true);

// Run the main algorithm starting from the exit blocks. We proceed to perform
// an undirected depth-first backwards traversal that determines class numbers
// for all participating blocks. Takes O(E) time and O(N) space.

bool ControlDependence::runOnFunction(Function &F) {
  Computed = false;
  DFSNumber = 0;
  ClassNumber = 1;
  // The algorithm requires we transform the CFG into a strongly connected
  // component. We make a fake end, connect exiting blocks to the fake end, and
  // then connect the fake end and the real start.  We then start the DFS walk
  // from the fake exit block instead of a fake start block
  FakeEnd = BasicBlock::Create(F.getContext(), "FakeEnd");
  BlockInfo[FakeEnd].FakeSuccEdges.push_back(&F.getEntryBlock());
  BlockInfo[&F.getEntryBlock()].FakePredEdges.push_back(FakeEnd);

  //  BlockInfo.resize(F.size());
  for (auto &B : F) {
    BlockData &Info = BlockInfo[&B];
    // If this is a reverse-unreachable block, we don't care about it
    if (pred_empty(&B) && &B != &F.getEntryBlock()) {
      Info.Participates = false;
    }
    // If there are no successors, we need to connect it to the exit block
    if (succ_empty(&B)) {
      // Connect leaves to fake end
      Info.FakeSuccEdges.push_back(FakeEnd);
      BlockInfo[FakeEnd].FakePredEdges.push_back(&B);
    }
  }
  runUndirectedDFS(FakeEnd);
  Computed = true;
  return false;
}

void ControlDependence::releaseMemory() {
  if (Computed) {
    BlockInfo.clear();
    delete FakeEnd;
  }
  Computed = false;
}

// print - Show contents in human readable format...
void ControlDependence::print(raw_ostream &O, const Module *M) const {}

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
void ControlDependence::runUndirectedDFS(const BasicBlock *StartBlock) {
  DFSStack Stack;
  // Start out walking backwards
  pushDFS(Stack, StartBlock, nullptr, PredDirection);

  while (!Stack.empty()) {
    DFSStackEntry &Entry = Stack.top();
    const BasicBlock *B = Entry.Block;
    DEBUG(dbgs() << "Starting from block ");
    DEBUG(B->printAsOperand(dbgs()));
    DEBUG(dbgs() << "\n");
    // First visit in pred direction, then swap directions, then visit in succ
    // direction
    if (Entry.Direction == PredDirection) {
      // Pred direction
      if (Entry.Pred != Entry.PredEnd) {
        const BasicBlock *Pred = *Entry.Pred;
        ++(Entry.Pred);
        BlockData &PredData = BlockInfo.find(Pred)->second;
        if (!PredData.Participates)
          continue;
        if (PredData.Visited)
          continue;
        // If it's on the stack, we've found a backedge, otherwise, push it
        // and preorder-visit
        if (PredData.OnStack) {
          DEBUG(dbgs() << "Maybe visit pred backedge\n");
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
        BlockData &SuccData = BlockInfo.find(Succ)->second;
        if (!SuccData.Participates)
          continue;
        if (SuccData.Visited)
          continue;
        // If it's on the stack, we've found a backedge, otherwise, push it
        // and preorder-visit
        if (SuccData.OnStack) {
          DEBUG(dbgs() << "Maybe visit succ backedge\n");
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

void ControlDependence::pushDFS(DFSStack &Stack, const BasicBlock *B,
                                const BasicBlock *From,
                                DFSDirection Direction) {
  auto BlockResult = BlockInfo.find(B);
  assert(BlockResult != BlockInfo.end() && "Should have found block data");

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
void ControlDependence::popDFS(DFSStack &Stack, const BasicBlock *B) {
  assert(Stack.top().Block == B && "Stack top is wrong");
  auto BlockResult = BlockInfo.find(B);

  assert(BlockResult != BlockInfo.end() && "Should have found block data");
  auto &BlockResultData = BlockResult->second;
  BlockResultData.OnStack = false;
  BlockResultData.Visited = true;
  Stack.pop();
}

void ControlDependence::visitPre(const BasicBlock *B) {
  DEBUG(dbgs() << "pre-order visit of block ");
  DEBUG(B->printAsOperand(dbgs()));
  DEBUG(dbgs() << "\n");

  BlockInfo[B].DFSNumber = ++DFSNumber;
  DEBUG(dbgs() << "Assigned DFS Number " << BlockInfo[B].DFSNumber << "\n");
}
void ControlDependence::debugBracketList(const BracketList &BList) {
  dbgs() << "{";
  for (auto &Bracket : BList) {
    dbgs() << "(";
    Bracket.From->printAsOperand(dbgs());
    dbgs() << ",";
    Bracket.To->printAsOperand(dbgs());
    dbgs() << ") ";
  }
  dbgs() << "}";
}

void ControlDependence::visitMid(const BasicBlock *B, DFSDirection Direction) {
  DEBUG(dbgs() << "mid-order visit of block ");
  DEBUG(B->printAsOperand(dbgs()));
  DEBUG(dbgs() << "\n");
  BlockData &Info = BlockInfo[B];
  BracketList &BList = Info.BList;
  // Remove brackets pointing to this node [line:19].
  DEBUG(dbgs() << "Removing brackets pointing to ");
  DEBUG(B->printAsOperand(dbgs()));
  DEBUG(dbgs() << "\n");

  // By the time we hit this node, we are guaranteed these iterators will point
  // into our list, because it must have been spliced into us.

  for (auto BII = Info.BracketIterators.begin(),
            BIE = Info.BracketIterators.end();
       BII != BIE;) {
    if ((*BII)->To == B && (*BII)->Direction != Direction) {
      Info.BList.erase(*BII);
      BII = Info.BracketIterators.erase(BII);
    } else
      ++BII;
  }

  // We should have at least hit the artificial edge connecting end and start as
  // a backedge, which would have started a bracket list that would have
  // propagated up to this point, so this should not be possible.
  assert(!BList.empty() && "This should have never happened at this point, we "
                           "should have hit at least one backedge");

  DEBUG(dbgs() << "Bracket list is ");
  DEBUG(debugBracketList(BList));
  DEBUG(dbgs() << "\n");

  // Potentially start a new equivalence class [line:37]
  Bracket &Recent = BList.back();
  if (Recent.RecentSize != BList.size()) {
    Recent.RecentSize = BList.size();
    Recent.RecentClass = ++ClassNumber;
  }
  Info.ClassNumber = Recent.RecentClass;

  DEBUG(dbgs() << "Assigned class number is " << Info.ClassNumber << "\n");
}
void ControlDependence::visitPost(const BasicBlock *B,
                                  const BasicBlock *ParentBlock,
                                  DFSDirection Direction) {
  DEBUG(dbgs() << "post-order visit of block ");
  DEBUG(B->printAsOperand(dbgs()));
  DEBUG(dbgs() << "\n");
  BlockData &Info = BlockInfo[B];
  BracketList &BList = Info.BList;
  // Remove brackets pointing to this node [line:19].
  DEBUG(dbgs() << "Removing brackets pointing to ");
  DEBUG(B->printAsOperand(dbgs()));
  DEBUG(dbgs() << "\n");

  // By the time we hit this node, we are guaranteed these iterators will point
  // into our list, because it must have been spliced into us.
  for (auto BII = Info.BracketIterators.begin(),
            BIE = Info.BracketIterators.end();
       BII != BIE;) {
    if ((*BII)->To == B && (*BII)->Direction != Direction) {
      Info.BList.erase(*BII);
      BII = Info.BracketIterators.erase(BII);
    } else
      ++BII;
  }

  // Propagate bracket list up the DFS tree [line:13].
  if (ParentBlock != nullptr) {
    DEBUG(dbgs() << "Splicing bracket into ");
    DEBUG(ParentBlock->printAsOperand(dbgs()));
    DEBUG(dbgs() << "\n");
    BracketList &ParentBList = BlockInfo[ParentBlock].BList;
    // TODO
    ParentBList.splice(ParentBList.end(), BList);
    DEBUG(dbgs() << "Parent bracket list is now");
    DEBUG(debugBracketList(ParentBList));
    DEBUG(dbgs() << "\n");
  }
}

void ControlDependence::visitBackedge(const BasicBlock *From,
                                      const BasicBlock *To,
                                      DFSDirection Direction) {
  DEBUG(dbgs() << "visit backedge from block ");
  DEBUG(From->printAsOperand(dbgs()));
  DEBUG(dbgs() << " to block ");
  DEBUG(To->printAsOperand(dbgs()));
  DEBUG(dbgs() << "\n");

  // Push backedge onto the bracket list [line:25].
  BlockData &Info = BlockInfo[From];
  BracketList &BList = Info.BList;
  auto Place = BList.insert(BList.end(), Bracket{Direction, 0, 0, From, To});
  BlockInfo[To].BracketIterators.push_back(Place);
}
namespace {

class ControlDependencePrinter : public FunctionPass {
  const Function *F;

public:
  ControlDependencePrinter() : FunctionPass(ID) {
    initializeControlDependencePrinterPass(*PassRegistry::getPassRegistry());
  }
  // run - Calculate control Equivalence for this function
  bool runOnFunction(Function &F) override;

  // getAnalysisUsage - Implement the Pass API
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<ControlDependence>();
    AU.setPreservesAll();
  }
  // releaseMemory - Reset state back to before function was analyzed
  void releaseMemory() override {
    F = nullptr;
    EquivClasses.clear();
  }

  // print - Show contents in human readable format...
  void print(raw_ostream &O, const Module * = nullptr) const override;

  static char ID;
  std::vector<std::vector<const BasicBlock *>> EquivClasses;
};
}
char ControlDependencePrinter::ID = 0;

INITIALIZE_PASS_BEGIN(ControlDependencePrinter, "print-controldep",
                      "Print Control Dependence of function", true, true)
INITIALIZE_PASS_DEPENDENCY(ControlDependence)
INITIALIZE_PASS_END(ControlDependencePrinter, "print-controldep",
                    "Print Control Dependence of function", true, true)

FunctionPass *llvm::createControlDependencePrinter() {
  return new ControlDependencePrinter();
}

void ControlDependencePrinter::print(raw_ostream &OS, const Module *) const {
  OS << "Equivalence classes: (";
  for (auto &V : EquivClasses) {
    OS << "{";
    for (auto &B : V) {
      if (B->hasName())
        OS << B->getName();
      else
        B->printAsOperand(OS);
      if (&B != &V.back())
        OS << ",";
    } 
    OS << "}";
    if (&V != &EquivClasses.back())
      OS << ",";
  }
  OS << ")\n";
}

bool ControlDependencePrinter::runOnFunction(Function &F) {
  this->F = &F;
  ControlDependence &CD = getAnalysis<ControlDependence>();
  // Make a nice map from class number to vectors of bb's
  std::map<unsigned, std::vector<const BasicBlock *>> ClassSets;
  for (auto &BB : F) {
    unsigned ClassNum = CD.getClassNumber(&BB);
    ClassSets[ClassNum].push_back(&BB);
  }

  // Sort each vector by bb name, putting named bbs before unnamed ones, then
  // add it to the equivalence class list
  for (auto &V : ClassSets) {
    std::stable_sort(V.second.begin(), V.second.end(),
                     [](const BasicBlock *A, const BasicBlock *B) {
                       if (A->hasName() && !B->hasName())
                         return true;
                       if (B->hasName() && !A->hasName())
                         return false;
                       if (!A->hasName() && !B->hasName())
                         return false;
                       return A->getName() < B->getName();
                     });
    EquivClasses.push_back(V.second);
  }
  // Now stable sort by size and then name
  std::stable_sort(EquivClasses.begin(), EquivClasses.end(),
                   [](const std::vector<const BasicBlock *> &A,
                      const std::vector<const BasicBlock *> &B) {
                     if (A.size() < B.size())
                       return true;
                     if (A.size() > B.size())
                       return false;
                     // The same block can't be in multiple equivalence classes,
                     // so we know
                     // that the above sorting means we can just compare the
                     // first elements
                     if (A[0]->hasName() && !B[0]->hasName())
                       return true;
                     if (B[0]->hasName() && !A[0]->hasName())
                       return false;
                     return A[0]->getName() < B[0]->getName();
                   });
  return false;
}
