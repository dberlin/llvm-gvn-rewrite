
//===- MemorySSA.cpp - Memory SSA Builder----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MemorySSA class.
//
//===----------------------------------------------------------------------===//
#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/UniqueVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Transforms/Utils/MemorySSA.h"
#include <algorithm>
#include <queue>
#define DEBUG_TYPE "memoryssa"
using namespace llvm;
STATISTIC(NumClobberCacheLookups, "Number of Memory SSA version cache lookups");
STATISTIC(NumClobberCacheHits, "Number of Memory SSA version cache hits");

INITIALIZE_PASS_BEGIN(MemorySSA, "memoryssa", "Memory SSA", false, true)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(MemorySSA, "memoryssa", "Memory SSA", false, true);

// This is a temporary (IE will be deleted once consensus is reached
// in the review) flag to determine whether we should optimize uses
// while building so they point to the nearest actual clobber
#define OPTIMIZE_USES 1

// An annotator class to print memory ssa information in comments
class MemorySSAAnnotatedWriter : public AssemblyAnnotationWriter {
  const MemorySSA *MSSA;
  UniqueVector<MemoryAccess *> SlotInfo;

public:
  MemorySSAAnnotatedWriter(const MemorySSA *M) : MSSA(M) {}

  virtual void emitBasicBlockStartAnnot(const BasicBlock *BB,
                                        formatted_raw_ostream &OS) {
    MemoryAccess *MA = MSSA->getMemoryAccess(BB);
    if (MA) {
      OS << "; ";
      MA->print(OS, SlotInfo);
      OS << "\n";
    }
  }

  virtual void emitInstructionAnnot(const Instruction *I,
                                    formatted_raw_ostream &OS) {
    MemoryAccess *MA = MSSA->getMemoryAccess(I);
    if (MA) {
      OS << "; ";
      MA->print(OS, SlotInfo);
      OS << "\n";
    }
  }
};

bool MemorySSA::isLiveOnEntryDef(const MemoryAccess *MA) const {
  return MA == LiveOnEntryDef;
}

// Get the clobbering memory access for a phi node and alias location
std::pair<MemoryAccess *, bool>
MemorySSA::getClobberingMemoryAccess(MemoryPhi *P,
                                     const AliasAnalysis::Location &Loc,
                                     SmallPtrSet<MemoryAccess *, 32> &Visited) {

  bool HitVisited = false;

  ++NumClobberCacheLookups;
  auto CCV = CachedClobberingAccess.find(std::make_pair(P, Loc));
  if (CCV != CachedClobberingAccess.end()) {
    ++NumClobberCacheHits;
    DEBUG(dbgs() << "Cached Memory SSA pointer for " << (uintptr_t)P << " is ");
    DEBUG(dbgs() << (uintptr_t)CCV->second);
    DEBUG(dbgs() << "\n");
    return std::make_pair(CCV->second, false);
  }

  // The algorithm here is fairly simple. The goal is to prove that
  // the phi node doesn't matter for this alias location, and to get
  // to whatever Access occurs before the *split* point that caused
  // the phi node.
  // There are only two cases we can walk through:
  // 1. One argument dominates the other, and the other's argument
  // defining memory access is non-aliasing with our location.
  // 2. All of the arguments are non-aliasing with our location, and
  // eventually lead back to the same defining memory access
  MemoryAccess *Result = nullptr;

  // If we already got here once, and didn't get to an answer (if we
  // did, it would have been cached below), we must be stuck in
  // mutually recursive phi nodes.  In that case, the correct answer
  // is "we can ignore the phi node if all the other arguments turn
  // out okay" (since it cycles between itself and the other
  // arguments).  We return true here, and are careful to make sure we
  // only pass through "true" when we are giving results
  // for the cycle itself.
  if (!Visited.insert(P).second)
    return std::make_pair(P, true);

  // Look through 1 argument phi nodes
  if (P->getNumIncomingValues() == 1) {
    auto SingleResult =
        getClobberingMemoryAccess(P->getIncomingValue(0), Loc, Visited);
    HitVisited = SingleResult.second;
    Result = SingleResult.first;
  } else {
    MemoryAccess *TargetResult = nullptr;

    // This is true if we hit ourselves from every argument
    bool AllVisited = true;
    for (unsigned i = 0; i < P->getNumIncomingValues(); ++i) {
      MemoryAccess *Arg = P->getIncomingValue(i);
      auto ArgResult = getClobberingMemoryAccess(Arg, Loc, Visited);
      if (!ArgResult.second) {
        AllVisited = false;
        // Fill in target result we are looking for if we haven't so far
        // Otherwise check the argument is equal to the last one
        if (!TargetResult) {
          TargetResult = ArgResult.first;
        } else if (TargetResult != ArgResult.first) {
          Result = P;
          HitVisited = false;
          break;
        }
      }
    }
    //  See if we completed either with all visited, or with success
    if (!Result && AllVisited) {
      Result = P;
      HitVisited = true;
    } else if (!Result && TargetResult) {
      Result = TargetResult;
      HitVisited = false;
    }
  }
  // Cache our result
  CachedClobberingAccess.insert(std::make_pair(std::make_pair(P, Loc), Result));

  return std::make_pair(Result, HitVisited);
}

// For a given MemoryAccess, walk backwards using Memory SSA and find
// the MemoryAccess that actually clobbers Loc.  The second part of
// the pair we return is whether we hit a cyclic phi node.
std::pair<MemoryAccess *, bool>
MemorySSA::getClobberingMemoryAccess(MemoryAccess *MA,
                                     const AliasAnalysis::Location &Loc,
                                     SmallPtrSet<MemoryAccess *, 32> &Visited) {
  MemoryAccess *CurrAccess = MA;
  while (true) {
    // If we started with a heap use, walk to the def
    if (MemoryUse *MU = dyn_cast<MemoryUse>(CurrAccess))
      CurrAccess = MU->getDefiningAccess();

    // Should be either a Memory Def or a Phi node at this point
    if (MemoryPhi *P = dyn_cast<MemoryPhi>(CurrAccess))
      return getClobberingMemoryAccess(P, Loc, Visited);
    else {
      MemoryDef *MD = dyn_cast<MemoryDef>(CurrAccess);
      assert(MD && "Use linked to something that is not a def");
      // If we hit the top, stop
      if (isLiveOnEntryDef(MD))
        return std::make_pair(CurrAccess, false);
      Instruction *DefMemoryInst = MD->getMemoryInst();
      assert(DefMemoryInst &&
             "Defining instruction not actually an instruction");
      // While we can do lookups, we can't sanely do inserts here unless we
      // were to track every thing we saw along the way, since we don't
      // know where we will stop.
      ++NumClobberCacheLookups;
      auto CCV = CachedClobberingAccess.find(std::make_pair(CurrAccess, Loc));
      if (CCV != CachedClobberingAccess.end()) {
        ++NumClobberCacheHits;
        DEBUG(dbgs() << "Cached Memory SSA pointer for " << *DefMemoryInst
                     << " is ");
        DEBUG(dbgs() << (uintptr_t)CCV->second);
        DEBUG(dbgs() << "\n");
        return std::make_pair(CCV->second, false);
      }

      if (AA->getModRefInfo(DefMemoryInst, Loc) & AliasAnalysis::Mod)
        break;
    }

    MemoryAccess *NextAccess = cast<MemoryDef>(CurrAccess)->getDefiningAccess();
    // Walk from def to def
    CurrAccess = NextAccess;
  }
  CachedClobberingAccess.insert(
      std::make_pair(std::make_pair(MA, Loc), CurrAccess));
  return std::make_pair(CurrAccess, false);
}

// For a given instruction, walk backwards using Memory SSA and find
// the memory access that actually clobbers this one, skipping non-aliasing
// ones along the way
MemoryAccess *MemorySSA::getClobberingMemoryAccess(Instruction *I) {

  MemoryAccess *StartingAccess = getMemoryAccess(I);
  assert(isa<MemoryUse>(StartingAccess) &&
         "Can only ask for clobbering accesses for a use");

  // First extract our location, then start walking until it is clobbered
  const AliasAnalysis::Location Loc = AA->getLocation(I);

  ++NumClobberCacheLookups;
  auto CCV = CachedClobberingAccess.find(std::make_pair(StartingAccess, Loc));

  if (CCV != CachedClobberingAccess.end()) {
    ++NumClobberCacheHits;
    DEBUG(dbgs() << "Cached Memory SSA Access for " << (uintptr_t)I << " is ");
    DEBUG(dbgs() << (uintptr_t)CCV->second);
    DEBUG(dbgs() << "\n");
    return CCV->second;
  }

  SmallPtrSet<MemoryAccess *, 32> Visited;
  MemoryAccess *FinalAccess =
      getClobberingMemoryAccess(StartingAccess, Loc, Visited).first;

  CachedClobberingAccess.insert(
      std::make_pair(std::make_pair(StartingAccess, Loc), FinalAccess));
  DEBUG(dbgs() << "Starting Memory SSA clobber for " << (uintptr_t)I << " is ");
  DEBUG(dbgs() << (uintptr_t)StartingAccess);
  DEBUG(dbgs() << "\n");
  DEBUG(dbgs() << "Final Memory SSA clobber for " << (uintptr_t)I << " is ");
  DEBUG(dbgs() << (uintptr_t)FinalAccess);
  DEBUG(dbgs() << "\n");

  return FinalAccess;
}

// This is the same as PromoteMemoryToRegister's Access.  The goal is
// to compute blocks in which a memory-access is Live-In.

void MemorySSA::computeLiveInBlocks(
    const AccessMap &BlockAccesses,
    const SmallPtrSetImpl<BasicBlock *> &DefBlocks,
    const SmallVector<BasicBlock *, 32> &UseBlocks,
    SmallPtrSetImpl<BasicBlock *> &LiveInBlocks) {

  // To determine liveness, we must iterate through the predecessors of blocks
  // where the def is live.  Blocks are added to the worklist if we need to
  // check their predecessors.  Start with all the using blocks.
  SmallVector<BasicBlock *, 64> LiveInBlockWorklist(UseBlocks.begin(),
                                                    UseBlocks.end());
  // If any of the using blocks is also a definition block, check to see if the
  // definition occurs before or after the use.  If it happens before the use,
  // the value isn't really live-in.
  for (unsigned i = 0, e = LiveInBlockWorklist.size(); i != e; ++i) {
    BasicBlock *BB = LiveInBlockWorklist[i];
    if (!DefBlocks.count(BB))
      continue;

    // Okay, this is a block that both uses and defines the value.  If the first
    // reference to the alloca is a def (store), then we know it isn't
    // live-in.
    auto AccessList = BlockAccesses.lookup(BB);

    if (AccessList && isa<MemoryDef>(AccessList->front())) {
      LiveInBlockWorklist[i] = LiveInBlockWorklist.back();
      LiveInBlockWorklist.pop_back();
      --i, --e;
    }
  }

  // Now that we have a set of blocks where the phi is live-in, recursively add
  // their predecessors until we find the full region the value is live.
  while (!LiveInBlockWorklist.empty()) {
    BasicBlock *BB = LiveInBlockWorklist.pop_back_val();

    // The block really is live in here, insert it into the set.  If already in
    // the set, then it has already been processed.
    if (!LiveInBlocks.insert(BB).second)
      continue;

    // Since the value is live into BB, it is either defined in a predecessor or
    // live into it to.  Add the preds to the worklist unless they are a
    // defining block.
    for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI) {
      BasicBlock *P = *PI;

      // The value is not live into a predecessor if it defines the value.
      if (DefBlocks.count(P))
        continue;

      // Otherwise it is, add to the worklist.
      LiveInBlockWorklist.push_back(P);
    }
  }
}

// We need a unique numbering for each BB.

void MemorySSA::computeBBNumbers(Function &F,
                                 DenseMap<BasicBlock *, unsigned> &BBNumbers) {
  // Assign unique ids to basic blocks
  unsigned ID = 0;
  for (auto I = F.begin(), E = F.end(); I != E; ++I)
    BBNumbers[I] = ID++;
}

// This is the same algorithm as PromoteMemoryToRegister's phi
// placement algorithm.

void MemorySSA::determineInsertionPoint(
    Function &F, AccessMap &BlockAccesses,
    const SmallPtrSetImpl<BasicBlock *> &DefBlocks,
    const SmallVector<BasicBlock *, 32> &UsingBlocks) {
  // Compute dominator levels and BB numbers
  DenseMap<DomTreeNode *, unsigned> DomLevels;
  computeDomLevels(DomLevels);

  DenseMap<BasicBlock *, unsigned> BBNumbers;
  computeBBNumbers(F, BBNumbers);

  SmallPtrSet<BasicBlock *, 32> LiveInBlocks;

  computeLiveInBlocks(BlockAccesses, DefBlocks, UsingBlocks, LiveInBlocks);
  // Use a priority queue keyed on dominator tree level so that inserted nodes
  // are handled from the bottom of the dominator tree upwards.
  typedef std::pair<DomTreeNode *, unsigned> DomTreeNodePair;
  typedef std::priority_queue<DomTreeNodePair, SmallVector<DomTreeNodePair, 32>,
                              less_second> IDFPriorityQueue;
  IDFPriorityQueue PQ;

  for (BasicBlock *BB : DefBlocks) {
    if (DomTreeNode *Node = DT->getNode(BB))
      PQ.push(std::make_pair(Node, DomLevels[Node]));
  }

  SmallVector<std::pair<unsigned, BasicBlock *>, 32> DFBlocks;
  SmallPtrSet<DomTreeNode *, 32> Visited;
  SmallVector<DomTreeNode *, 32> Worklist;
  while (!PQ.empty()) {
    DomTreeNodePair RootPair = PQ.top();
    PQ.pop();
    DomTreeNode *Root = RootPair.first;
    unsigned RootLevel = RootPair.second;

    // Walk all dominator tree children of Root, inspecting their CFG edges with
    // targets elsewhere on the dominator tree. Only targets whose level is at
    // most Root's level are added to the iterated dominance frontier of the
    // definition set.

    Worklist.clear();
    Worklist.push_back(Root);

    while (!Worklist.empty()) {
      DomTreeNode *Node = Worklist.pop_back_val();
      BasicBlock *BB = Node->getBlock();

      for (auto S : successors(BB)) {
        DomTreeNode *SuccNode = DT->getNode(S);

        // Quickly skip all CFG edges that are also dominator tree edges instead
        // of catching them below.
        if (SuccNode->getIDom() == Node)
          continue;

        unsigned SuccLevel = DomLevels[SuccNode];
        if (SuccLevel > RootLevel)
          continue;

        if (!Visited.insert(SuccNode).second)
          continue;

        BasicBlock *SuccBB = SuccNode->getBlock();
        if (!LiveInBlocks.count(SuccBB))
          continue;

        DFBlocks.push_back(std::make_pair(BBNumbers[SuccBB], SuccBB));
        if (!DefBlocks.count(SuccBB))
          PQ.push(std::make_pair(SuccNode, SuccLevel));
      }

      for (auto C : *Node)
        if (!Visited.count(C))
          Worklist.push_back(C);
    }
  }

  if (DFBlocks.size() > 1)
    std::sort(DFBlocks.begin(), DFBlocks.end());
  for (unsigned i = 0, e = DFBlocks.size(); i != e; ++i) {
    // Insert phi node
    BasicBlock *BB = DFBlocks[i].second;
    auto Accesses = BlockAccesses.lookup(BB);
    if (!Accesses) {
      Accesses = new std::list<MemoryAccess *>;
      BlockAccesses.insert(std::make_pair(BB, Accesses));
    }
    MemoryPhi *Phi = new (MemoryAccessAllocator)
        MemoryPhi(BB, std::distance(pred_begin(BB), pred_end(BB)));
    InstructionToMemoryAccess.insert(std::make_pair(BB, Phi));
    // Phi goes first
    Accesses->push_front(Phi);
  }
}

// Standard SSA renaming pass. Same algorithm as
// PromoteMemoryToRegisters

void MemorySSA::renamePass(BasicBlock *BB, BasicBlock *Pred,
                           MemoryAccess *IncomingVal, AccessMap &BlockAccesses,
                           std::vector<RenamePassData> &Worklist,
                           SmallPtrSet<BasicBlock *, 16> &Visited,
                           UseMap &Uses) {
NextIteration:
  auto Accesses = BlockAccesses.lookup(BB);

  // First rename the phi nodes
  if (Accesses && isa<MemoryPhi>(Accesses->front())) {
    MemoryPhi *Phi = cast<MemoryPhi>(Accesses->front());
    unsigned NumEdges = std::count(succ_begin(Pred), succ_end(Pred), BB);
    assert(NumEdges && "Must be at least one edge from Pred to BB!");
    for (unsigned i = 0; i != NumEdges; ++i)
      Phi->addIncoming(IncomingVal, Pred);
    addUseToMap(Uses, IncomingVal, Phi);

    IncomingVal = Phi;
  }

  // Don't revisit blocks.
  if (!Visited.insert(BB).second)
    return;

  // Skip if the list is empty, but we still have to pass thru the
  // incoming value info/etc to successors
  if (Accesses)
    for (auto L : *Accesses) {
      if (isa<MemoryPhi>(L))
        continue;

      if (MemoryUse *MU = dyn_cast<MemoryUse>(L)) {
        MU->setDefiningAccess(IncomingVal);
#if OPTIMIZE_USES
        auto RealVal = getClobberingMemoryAccess(MU->getMemoryInst());
#else
        auto RealVal = IncomingVal;
#endif
        MU->setDefiningAccess(RealVal);
        addUseToMap(Uses, RealVal, MU);
      } else if (MemoryDef *MD = dyn_cast<MemoryDef>(L)) {
        MD->setDefiningAccess(IncomingVal);
        addUseToMap(Uses, IncomingVal, MD);
        IncomingVal = MD;
      }
    }
  // 'Recurse' to our successors.
  succ_iterator I = succ_begin(BB), E = succ_end(BB);
  if (I == E)
    return;

  // Keep track of the successors so we don't visit the same successor twice
  SmallPtrSet<BasicBlock *, 8> VisitedSuccs;

  // Handle the first successor without using the worklist.
  VisitedSuccs.insert(*I);
  Pred = BB;
  BB = *I;
  ++I;

  for (; I != E; ++I)
    if (VisitedSuccs.insert(*I).second)
      Worklist.push_back(RenamePassData(*I, Pred, IncomingVal));
  goto NextIteration;
}

void MemorySSA::computeDomLevels(DenseMap<DomTreeNode *, unsigned> &DomLevels) {
  SmallVector<DomTreeNode *, 32> Worklist;

  DomTreeNode *Root = DT->getRootNode();
  DomLevels[Root] = 0;
  Worklist.push_back(Root);

  while (!Worklist.empty()) {
    DomTreeNode *Node = Worklist.pop_back_val();
    unsigned ChildLevel = DomLevels[Node] + 1;
    for (auto CI = Node->begin(), CE = Node->end(); CI != CE; ++CI) {
      DomLevels[*CI] = ChildLevel;
      Worklist.push_back(*CI);
    }
  }
}

// Handle unreachable block acccesses by deleting phi nodes in
// unreachable blocks, and marking all other unreachable
// memoryaccesses as being uses of the live on entry definition
void MemorySSA::markUnreachableAsLiveOnEntry(AccessMap &BlockAccesses,
                                             BasicBlock *BB, UseMap &Uses) {
  assert(!DT->isReachableFromEntry(BB) &&
         "Reachable block found while handling unreachable blocks");

  auto Accesses = BlockAccesses.lookup(BB);
  if (!Accesses)
    return;

  for (auto AI = Accesses->begin(), AE = Accesses->end(); AI != AE;) {
    auto Next = std::next(AI);
    // If we have a phi, just remove it. We are going to replace all
    // users with live on entry.
    if (MemoryPhi *P = dyn_cast<MemoryPhi>(*AI)) {
      delete P;
      Accesses->erase(AI);
    } else if (MemoryUse *U = dyn_cast<MemoryUse>(*AI)) {
      U->setDefiningAccess(LiveOnEntryDef);
      addUseToMap(Uses, LiveOnEntryDef, U);
    } else if (MemoryDef *D = dyn_cast<MemoryDef>(*AI)) {
      D->setDefiningAccess(LiveOnEntryDef);
      addUseToMap(Uses, LiveOnEntryDef, D);
    }
    AI = Next;
  }
}

char MemorySSA::ID = 0;

MemorySSA::MemorySSA() : FunctionPass(ID), LiveOnEntryDef(nullptr) {
  initializeMemorySSAPass(*PassRegistry::getPassRegistry());
}

MemorySSA::~MemorySSA() {}

void MemorySSA::releaseMemory() {
  CachedClobberingAccess.clear();
  InstructionToMemoryAccess.clear();
  MemoryAccessAllocator.Reset();
}

void MemorySSA::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<AliasAnalysis>();
  AU.addRequired<DominatorTreeWrapperPass>();
}

void MemorySSA::addUseToMap(UseMap &Uses, MemoryAccess *User,
                            MemoryAccess *Use) {
  std::list<MemoryAccess *> *UseList;
  UseList = Uses.lookup(User);
  if (!UseList) {
    UseList = new std::list<MemoryAccess *>;
    Uses.insert(std::make_pair(User, UseList));
  }

  UseList->push_back(Use);
}

// Build the actual use lists out of the use map
void MemorySSA::addUses(UseMap &Uses) {
  for (auto D : Uses) {
    std::list<MemoryAccess *> *UseList = D.second;
    MemoryAccess *User = D.first;
    User->UseList =
        MemoryAccessAllocator.Allocate<MemoryAccess *>(UseList->size());
    for (auto U : *UseList)
      User->addUse(U);
  }
}

bool MemorySSA::runOnFunction(Function &F) {
  this->F = &F;
  AA = &getAnalysis<AliasAnalysis>();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  buildMemorySSA(F);
  return false;
}

void MemorySSA::buildMemorySSA(Function &F) {
  // We temporarily maintain lists of memory accesses per-block,
  // trading time for memory. We could just look up the memory access
  // for every possible instruction in the stream.  Instead, we build
  // lists, and then throw it out once the use-def form is built.
  AccessMap PerBlockAccesses;
  SmallPtrSet<BasicBlock *, 32> DefiningBlocks;
  SmallVector<BasicBlock *, 32> UsingBlocks;

  for (auto FI = F.begin(), FE = F.end(); FI != FE; ++FI) {
    std::list<MemoryAccess *> *Accesses = nullptr;
    for (auto BI = FI->begin(), BE = FI->end(); BI != BE; ++BI) {
      bool use = false;
      bool def = false;
      if (isa<LoadInst>(BI)) {
        use = true;
        def = false;
      } else if (isa<StoreInst>(BI)) {
        use = false;
        def = true;
      } else {
        AliasAnalysis::ModRefResult ModRef = AA->getModRefInfo(BI);
        if (ModRef & AliasAnalysis::Mod)
          def = true;
        if (ModRef & AliasAnalysis::Ref)
          use = true;
      }

      // Defs are already uses, so use && def == def
      if (use && !def) {
        MemoryUse *MU =
            new (MemoryAccessAllocator) MemoryUse(nullptr, &*BI, FI);
        InstructionToMemoryAccess.insert(std::make_pair(&*BI, MU));
        UsingBlocks.push_back(FI);
        if (!Accesses) {
          Accesses = new std::list<MemoryAccess *>;
          PerBlockAccesses.insert(std::make_pair(FI, Accesses));
        }
        Accesses->push_back(MU);
      }
      if (def) {
        MemoryDef *MD =
            new (MemoryAccessAllocator) MemoryDef(nullptr, &*BI, FI);
        InstructionToMemoryAccess.insert(std::make_pair(&*BI, MD));
        DefiningBlocks.insert(FI);
        if (!Accesses) {
          Accesses = new std::list<MemoryAccess *>;
          PerBlockAccesses.insert(std::make_pair(FI, Accesses));
        }
        Accesses->push_back(MD);
      }
    }
  }
  // Determine where our PHI's should go
  determineInsertionPoint(F, PerBlockAccesses, DefiningBlocks, UsingBlocks);

  // We create an access to represent "live on entry", for things like
  // arguments or users of globals. We do not actually insert it in to
  // the IR.
  BasicBlock &StartingPoint = F.getEntryBlock();
  LiveOnEntryDef =
      new (MemoryAccessAllocator) MemoryDef(nullptr, nullptr, &StartingPoint);

  // Now do regular SSA renaming
  SmallPtrSet<BasicBlock *, 16> Visited;

  // Uses are allocated and built once for a memory access, then are
  // immutable. In order to count how many we need for a given memory
  // access, we first add all the uses to lists in a densemap, then
  // later we will convert it into an array and place it in the right
  // place
  UseMap Uses;

  std::vector<RenamePassData> RenamePassWorklist;
  RenamePassWorklist.push_back({F.begin(), nullptr, LiveOnEntryDef});
  do {
    RenamePassData RPD;
    RPD.swap(RenamePassWorklist.back());
    RenamePassWorklist.pop_back();
    renamePass(RPD.BB, RPD.Pred, RPD.MA, PerBlockAccesses, RenamePassWorklist,
               Visited, Uses);
  } while (!RenamePassWorklist.empty());

  // At this point, we may have unreachable blocks with unreachable accesses
  // Given any uses in unreachable blocks the live on entry definition
  if (Visited.size() != F.size()) {
    for (auto FI = F.begin(), FE = F.end(); FI != FE; ++FI) {
      if (!Visited.count(FI)) {
        markUnreachableAsLiveOnEntry(PerBlockAccesses, FI, Uses);
      }
    }
  }

  // Now convert our use lists into real uses
  addUses(Uses);
  DEBUG(dump(F));
  DEBUG(verifyDefUses(F));

  // Delete our access lists
  for (auto D : PerBlockAccesses)
    delete D.second;

  // Densemap does not like when you delete or change the value during
  // iteration.
  std::vector<std::list<MemoryAccess *> *> UseListsToDelete;
  for (auto D : Uses)
    UseListsToDelete.push_back(D.second);

  Uses.clear();
  for (unsigned i = 0, e = UseListsToDelete.size(); i != e; ++i) {
    delete UseListsToDelete[i];
    UseListsToDelete[i] = nullptr;
  }
}

void MemorySSA::print(raw_ostream &OS, const Module *M) const {
  F->print(OS, new MemorySSAAnnotatedWriter(this));
}
void MemorySSA::dump(Function &F) {
  F.print(dbgs(), new MemorySSAAnnotatedWriter(this));
}

void MemorySSA::verifyUseInDefs(MemoryAccess *Def, MemoryAccess *Use) {
  // The live on entry use may cause us to get a NULL def here
  if (Def == nullptr) {
    assert(isLiveOnEntryDef(Use) &&
           "Null def but use not point to live on entry def");
    return;
  }
  assert(std::find(Def->use_begin(), Def->use_end(), Use) != Def->use_end() &&
         "Did not find use in def's use list");
}

// Verify the immediate use information, by walking all the memory
// accesses and verifying that, for each use, it appears in the
// appropriate def's use list

void MemorySSA::verifyDefUses(Function &F) {
  for (auto FI = F.begin(), FE = F.end(); FI != FE; ++FI) {
    // Phi nodes are attached to basic blocks
    MemoryAccess *MA = getMemoryAccess(FI);
    if (MA) {
      assert(isa<MemoryPhi>(MA) &&
             "Something other than phi node on basic block");
      MemoryPhi *MP = cast<MemoryPhi>(MA);
      for (unsigned i = 0, e = MP->getNumIncomingValues(); i != e; ++i)
        verifyUseInDefs(MP->getIncomingValue(i), MP);
    }
    for (auto BI = FI->begin(), BE = FI->end(); BI != BE; ++BI) {
      MA = getMemoryAccess(BI);
      if (MA) {
        if (MemoryUse *MU = dyn_cast<MemoryUse>(MA))
          verifyUseInDefs(MU->getDefiningAccess(), MU);
        else if (MemoryDef *MD = dyn_cast<MemoryDef>(MA))
          verifyUseInDefs(MD->getDefiningAccess(), MD);
        else if (MemoryPhi *MP = dyn_cast<MemoryPhi>(MA)) {
          for (unsigned i = 0, e = MP->getNumIncomingValues(); i != e; ++i)
            verifyUseInDefs(MP->getIncomingValue(i), MP);
        }
      }
    }
  }
}

// Get a memory access for an instruction
MemoryAccess *MemorySSA::getMemoryAccess(const Value *I) const {
  return InstructionToMemoryAccess.lookup(I);
}

void MemoryDef::print(raw_ostream &OS, UniqueVector<MemoryAccess *> &SlotInfo) {
  MemoryAccess *UO = getDefiningAccess();
  OS << SlotInfo.insert(this) << " = "
     << "MemoryDef(";
  OS << SlotInfo.insert(UO) << ")";
}

void MemoryPhi::print(raw_ostream &OS, UniqueVector<MemoryAccess *> &SlotInfo) {
  OS << SlotInfo.insert(this) << " = "
     << "MemoryPhi(";
  for (unsigned int i = 0, e = getNumIncomingValues(); i != e; ++i) {
    BasicBlock *BB = getIncomingBlock(i);
    MemoryAccess *MA = getIncomingValue(i);
    OS << "{";
    if (BB->hasName())
      OS << BB->getName();
    else
      BB->printAsOperand(OS, false);
    OS << ",";
    assert((isa<MemoryDef>(MA) || isa<MemoryPhi>(MA)) &&
           "Phi node should have referred to def or another phi");
    OS << SlotInfo.insert(MA);
    OS << "}";
    if (i + 1 < e)
      OS << ",";
  }
  OS << ")";
}

void MemoryUse::print(raw_ostream &OS, UniqueVector<MemoryAccess *> &SlotInfo) {
  MemoryAccess *UO = getDefiningAccess();
  OS << "MemoryUse(";
  OS << SlotInfo.insert(UO);
  OS << ")";
}
