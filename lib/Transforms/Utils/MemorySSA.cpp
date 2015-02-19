
//===- MemorySSA.cpp - Memory SSA Builder
//----------------------------------===//
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

class MemorySSAAnnotatedWriter : public AssemblyAnnotationWriter {
  MemorySSA *MSSA;

public:
  MemorySSAAnnotatedWriter(MemorySSA *M) : MSSA(M) {}

  virtual void emitBasicBlockStartAnnot(const BasicBlock *BB,
                                        formatted_raw_ostream &OS) {
    MemoryAccess *MA = MSSA->getMemoryAccess(BB);
    if (MA) {
      OS << "; ";
      MA->print(OS);
      OS << "\n";
    }
  }

  virtual void emitInstructionAnnot(const Instruction *I,
                                    formatted_raw_ostream &OS) {
    MemoryAccess *MA = MSSA->getMemoryAccess(I);
    if (MA) {
      OS << "; ";
      MA->print(OS);
      OS << "\n";
    }
  }
};

namespace {
AliasAnalysis::Location getLocationForAA(AliasAnalysis *AA, Instruction *Inst) {
  if (auto *I = dyn_cast<LoadInst>(Inst))
    return AA->getLocation(I);
  else if (auto *I = dyn_cast<StoreInst>(Inst))
    return AA->getLocation(I);
  else if (auto *I = dyn_cast<VAArgInst>(Inst))
    return AA->getLocation(I);
  else if (auto *I = dyn_cast<AtomicCmpXchgInst>(Inst))
    return AA->getLocation(I);
  else if (auto *I = dyn_cast<AtomicRMWInst>(Inst))
    return AA->getLocation(I);
  else
    llvm_unreachable("unsupported memory instruction");
}

unsigned int getVersionNumberFromAccess(MemoryAccess *MA) {
  if (MA == nullptr)
    return 0;

  if (MemoryDef *Def = dyn_cast<MemoryDef>(MA))
    return Def->getDefVersion();
  else if (MemoryPhi *Phi = dyn_cast<MemoryPhi>(MA))
    return Phi->getDefVersion();
  else if (MemoryUse *Use = dyn_cast<MemoryUse>(MA))
    return getVersionNumberFromAccess(Use->getUseOperand());
  llvm_unreachable("Was not a definition that has a version");
}
}

bool MemorySSA::isLiveOnEntryDef(const MemoryAccess *MA) const {
  if (const MemoryDef *MD = dyn_cast<const MemoryDef>(MA))
    if (MD->getUseOperand() == nullptr)
      return true;

  return false;
}

// Get the clobbering heap version for a phi node and alias location

std::pair<MemoryAccess *, bool>
MemorySSA::getClobberingMemoryAccess(MemoryPhi *P,
                                     const AliasAnalysis::Location &Loc,
                                     SmallPtrSet<MemoryAccess *, 32> &Visited) {

  bool HitVisited = false;

  ++NumClobberCacheLookups;
  auto CCV = CachedClobberingVersion.find(std::make_pair(P, Loc));
  if (CCV != CachedClobberingVersion.end()) {
    ++NumClobberCacheHits;
    DEBUG(dbgs() << "Cached Memory SSA version for " << *P << " is ");
    DEBUG(dbgs() << getVersionNumberFromAccess(CCV->second));
    DEBUG(dbgs() << "\n");
    return std::make_pair(CCV->second, false);
  }

  // The algorithm here is fairly simple. The goal is to prove that
  // the phi node doesn't matter for this alias location, and to get
  // to whatever version occurs before the *split* point that caused
  // the phi node.
  // To do this, we
  // First, find the most dominating phi node argument, then see if we
  // end up with the same version between that and all the other phi
  // nodes.
  // This is because there are only two cases we can walk through:
  // 1. One argument dominates the other, and the other's version is a
  // "false" one.
  // 2. All the arguments have the same version number, and that
  // version is a false one for this location.
  // In both of these cases, we should arrive at the same version
  // number for all arguments. If we don't. we can't ignore this phi
  // node, because this means at least one of the arguments has a
  // memory operation that affects us.
  // In the #2 case, we will not actually find an argument that
  // dominates the others, but the algorithm will still use one of the
  // arguments's version number to compare against, coming up with
  // correct answers.
  MemoryAccess *Result;

  // If we already got here once, and didn't get to an answer (if we
  // did, it would have been cached below), we must be stuck in
  // mutually recursive phi nodes or something
  // TODO: Ensure we skip past phi nodes that we end up revisiting
  if (!Visited.insert(P).second)
    return std::make_pair(P, true);

  // Look through 1 argument phi nodes
  if (P->getNumIncomingValues() == 1) {
    auto SingleResult =
        getClobberingMemoryAccess(P->getIncomingValue(0), Loc, Visited);
    HitVisited = SingleResult.second;
    Result = SingleResult.first;
  } else {
    // Find the most dominating non-phiargument first, if there is one. It
    // will be a shorter walk
    MemoryAccess *DominatingArg = nullptr;
    for (unsigned i = 0; i < P->getNumIncomingValues(); ++i)
      if (!isa<MemoryPhi>(P->getIncomingValue(i))) {
        DominatingArg = P->getIncomingValue(i);
        break;
      }

    // Oh well, they are all defined by phi nodes, just choose one
    if (!DominatingArg)
      DominatingArg = P->getIncomingValue(0);
    if (!isa<MemoryPhi>(DominatingArg))
      for (unsigned i = 1; i < P->getNumIncomingValues(); ++i) {
        MemoryAccess *Arg = P->getIncomingValue(i);
        if (!isa<MemoryPhi>(Arg) &&
            DT->dominates(Arg->getBlock(), DominatingArg->getBlock()))
          DominatingArg = Arg;
      }

    auto TargetResult = getClobberingMemoryAccess(DominatingArg, Loc, Visited);
    MemoryAccess *TargetVersion = TargetResult.first;
    Result = TargetVersion;

    // Now reduce it against the others
    for (unsigned i = 0; i < P->getNumIncomingValues(); ++i) {
      MemoryAccess *Arg = P->getIncomingValue(i);
      if (Arg == DominatingArg)
        continue;

      auto ArgResult = getClobberingMemoryAccess(Arg, Loc, Visited);
      if (ArgResult.second)
        continue;
      MemoryAccess *HeapVersionForArg = ArgResult.first;
      // If they aren't the same, abort, we must be clobbered by one
      // or the other.
      // (Currently, we return the phi node, we could track which one
      // is clobbering)
      if (TargetVersion != HeapVersionForArg) {
        Result = P;
        HitVisited = ArgResult.second;
        break;
      }
    }
  }
  CachedClobberingVersion.insert(
      std::make_pair(std::make_pair(P, Loc), Result));
  return std::make_pair(Result, HitVisited);
}

// For a given MemoryAccess, walk backwards using Memory SSA and find
// the MemoryAccess that actually clobbers Loc.

std::pair<MemoryAccess *, bool>
MemorySSA::getClobberingMemoryAccess(MemoryAccess *HV,
                                     const AliasAnalysis::Location &Loc,
                                     SmallPtrSet<MemoryAccess *, 32> &Visited) {
  MemoryAccess *CurrVersion = HV;
  while (true) {
    MemoryAccess *UseVersion = CurrVersion;

    // If we started with a heap use, walk to the def
    if (MemoryUse *MU = dyn_cast<MemoryUse>(UseVersion))
      UseVersion = MU->getUseOperand();

    // Should be either a Memory Def or a Phi node at this point
    if (MemoryPhi *P = dyn_cast<MemoryPhi>(UseVersion))
      return getClobberingMemoryAccess(P, Loc, Visited);
    else {
      MemoryDef *MD = dyn_cast<MemoryDef>(UseVersion);
      assert(MD && "Use linked to something that is not a def");
      // If we hit the top, stop
      if (isLiveOnEntryDef(MD))
        return std::make_pair(CurrVersion, false);
      Instruction *DefMemoryInst = MD->getMemoryInst();
      assert(DefMemoryInst &&
             "Defining instruction not actually an instruction");
      // While we can do lookups, we can't sanely do inserts unless we
      // were to track every thing we saw along the way, since we don't
      // know where we will stop.
      ++NumClobberCacheLookups;
      auto CCV = CachedClobberingVersion.find(std::make_pair(UseVersion, Loc));
      if (CCV != CachedClobberingVersion.end()) {
        ++NumClobberCacheHits;
        DEBUG(dbgs() << "Cached Memory SSA version for " << *DefMemoryInst
                     << " is ");
        DEBUG(dbgs() << getVersionNumberFromAccess(CCV->second));
        DEBUG(dbgs() << "\n");
        return std::make_pair(CCV->second, false);
      }

      // If it's a call, get mod ref info, and if we have a mod,
      // we are done. Otherwise grab alias location, see if they
      // alias, and if they do, we are done.
      // Otherwise, continue
      if (isa<CallInst>(DefMemoryInst) || isa<InvokeInst>(DefMemoryInst)) { 
        if (AA->getModRefInfo(DefMemoryInst, Loc) & AliasAnalysis::Mod)
          break;
      } else if (AA->alias(getLocationForAA(AA, DefMemoryInst), Loc) !=
                 AliasAnalysis::NoAlias)
        break;
    }

    MemoryAccess *NextVersion = cast<MemoryDef>(UseVersion)->getUseOperand();
    DEBUG(dbgs() << "Walking memory SSA from version ");
    DEBUG(dbgs() << getVersionNumberFromAccess(CurrVersion));
    DEBUG(dbgs() << " to version ");
    DEBUG(dbgs() << getVersionNumberFromAccess(NextVersion));
    DEBUG(dbgs() << "\n");
    // Walk from def to def
    CurrVersion = NextVersion;
  }
  return std::make_pair(CurrVersion, false);
}

// For a given instruction, walk backwards using Memory SSA and find
// the heap version that actually clobbers this one, skipping "false"
// versions along the way
MemoryAccess *MemorySSA::getClobberingMemoryAccess(Instruction *I) {

  // First extract our location, then start walking until it is clobbered

  const AliasAnalysis::Location Loc = getLocationForAA(AA, I);
  MemoryAccess *StartingVersion = getMemoryAccess(I);
  ++NumClobberCacheLookups;
  auto CCV = CachedClobberingVersion.find(std::make_pair(StartingVersion, Loc));

  if (CCV != CachedClobberingVersion.end()) {
    ++NumClobberCacheHits;
    DEBUG(dbgs() << "Cached Memory SSA version for " << *I << " is ");
    DEBUG(dbgs() << getVersionNumberFromAccess(CCV->second));
    DEBUG(dbgs() << "\n");
    return CCV->second;
  }

  SmallPtrSet<MemoryAccess *, 32> Visited;
  MemoryAccess *FinalVersion =
      getClobberingMemoryAccess(StartingVersion, Loc, Visited).first;

  CachedClobberingVersion.insert(
      std::make_pair(std::make_pair(StartingVersion, Loc), FinalVersion));
  DEBUG(dbgs() << "Starting Memory SSA version for " << *I << " is ");
  DEBUG(dbgs() << getVersionNumberFromAccess(StartingVersion));
  DEBUG(dbgs() << "\n");
  DEBUG(dbgs() << "Final Memory SSA Version for " << *I << " is ");
  DEBUG(dbgs() << getVersionNumberFromAccess(FinalVersion));
  DEBUG(dbgs() << "\n");

  return FinalVersion;
}

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

void MemorySSA::computeBBNumbers(Function &F,
                                 DenseMap<BasicBlock *, unsigned> &BBNumbers) {
  // Assign unique ids to basic blocks
  unsigned ID = 0;
  for (auto I = F.begin(), E = F.end(); I != E; ++I)
    BBNumbers[I] = ID++;
}

void MemorySSA::determineInsertionPoint(
    Function &F, AccessMap &BlockAccesses,
    const SmallPtrSet<BasicBlock *, 32> DefBlocks,
    const SmallVector<BasicBlock *, 32> UsingBlocks) {
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

      for (auto SI = succ_begin(BB), SE = succ_end(BB); SI != SE; ++SI) {
        DomTreeNode *SuccNode = DT->getNode(*SI);

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

      for (auto CI = Node->begin(), CE = Node->end(); CI != CE; ++CI) {
        if (!Visited.count(*CI))
          Worklist.push_back(*CI);
      }
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
    MemoryPhi *Phi = new MemoryPhi(++NextHeapVersion, BB);
    InstructionToMemoryAccess.insert(std::make_pair(BB, Phi));
    // Phi goes first
    Accesses->push_front(Phi);
  }
}

void MemorySSA::renamePass(BasicBlock *BB, BasicBlock *Pred,
                           MemoryAccess *IncomingVal, AccessMap &BlockAccesses,
                           std::vector<RenamePassData> &Worklist,
                           SmallPtrSet<BasicBlock *, 16> &Visited) {
NextIteration:
  auto Accesses = BlockAccesses.lookup(BB);

  // First rename the phi nodes
  if (Accesses && isa<MemoryPhi>(Accesses->front())) {
    MemoryPhi *Phi = cast<MemoryPhi>(Accesses->front());
    unsigned NumEdges = std::count(succ_begin(Pred), succ_end(Pred), BB);
    assert(NumEdges && "Must be at least one edge from Pred to BB!");
    for (unsigned i = 0; i != NumEdges; ++i)
      Phi->addIncoming(IncomingVal, Pred);
    IncomingVal->addUse(Phi);

    IncomingVal = Phi;
  }

  // Don't revisit blocks.
  if (!Visited.insert(BB).second)
    return;

  // Skip if the list is empty, but we still have to pass thru the
  // incoming value info/etc to successors
  if (Accesses)
    for (auto LI = Accesses->begin(), LE = Accesses->end(); LI != LE; ++LI) {
      if (isa<MemoryPhi>(*LI))
        continue;

      if (MemoryUse *MU = dyn_cast<MemoryUse>(*LI)) {
        MU->setUseOperand(IncomingVal);
        if (IncomingVal)
          IncomingVal->addUse(MU);
      } else if (MemoryDef *MD = dyn_cast<MemoryDef>(*LI)) {
        MD->setUseOperand(IncomingVal);
        if (IncomingVal)
          IncomingVal->addUse(MD);
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

void MemorySSA::buildMemorySSA(Function &F) {
  // We temporarily maintain lists of memory accesses per-block,
  // trading time for memory. We could just look up the memory access
  // for every possible instruction in the stream.  Instead, we built
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
        AliasAnalysis::Location Loc;
        // FIXME: BasicAA crashes on calls if Loc.Ptr is null
        if (isa<CallInst>(BI) || isa<InvokeInst>(BI))
          Loc.Ptr = BI;
        AliasAnalysis::ModRefResult ModRef = AA->getModRefInfo(BI, Loc);
        if (ModRef & AliasAnalysis::Mod)
          def = true;
        // Defs are already uses
        if (ModRef & AliasAnalysis::Ref)
          use = true;
      }

      // Defs are already uses, so use && def is handled elsewhere
      if (use && !def) {
        MemoryUse *MU = new MemoryUse(nullptr, &*BI, FI);
        InstructionToMemoryAccess.insert(std::make_pair(&*BI, MU));
        UsingBlocks.push_back(FI);
        if (!Accesses) {
          Accesses = new std::list<MemoryAccess *>;
          PerBlockAccesses.insert(std::make_pair(FI, Accesses));
        }
        Accesses->push_back(MU);
      }
      if (def) {
        MemoryDef *MD = new MemoryDef(++NextHeapVersion, nullptr, &*BI, FI);
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

  // We create an access to represent "live on entry"
  BasicBlock &StartingPoint = F.getEntryBlock();
  MemoryAccess *DefaultAccess =
      new MemoryDef(0, nullptr, nullptr, &StartingPoint);

  // Now do regular SSA renaming
  /// The set of basic blocks the renamer has already visited.
  ///
  SmallPtrSet<BasicBlock *, 16> Visited;

  std::vector<RenamePassData> RenamePassWorklist;
  RenamePassWorklist.push_back({F.begin(), nullptr, DefaultAccess});
  do {
    RenamePassData RPD;
    RPD.swap(RenamePassWorklist.back());
    RenamePassWorklist.pop_back();
    renamePass(RPD.BB, RPD.Pred, RPD.MA, PerBlockAccesses, RenamePassWorklist,
               Visited);
  } while (!RenamePassWorklist.empty());

  DEBUG(F.print(dbgs(), new MemorySSAAnnotatedWriter(this)));
  DEBUG(verifyDefUses(F));

  for (auto DI = PerBlockAccesses.begin(), DE = PerBlockAccesses.end();
       DI != DE; ++DI) {
    delete DI->second;
  }
}

void MemorySSA::dump(Function &F, const AccessMap &AM) {
  for (auto FI = F.begin(), FE = F.end(); FI != FE; ++FI) {
    auto L = AM.lookup(FI);
    if (L) {
      for (auto LI = L->begin(), LE = L->end(); LI != LE; ++LI)
        dbgs() << *(*LI) << "\n";
    }
  }
}

static void verifyUseInDefs(MemoryAccess *Def, MemoryAccess *Use) {
  assert((isa<MemoryDef>(Def) || isa<MemoryPhi>(Def)) &&
         "Memory definition should have been a def or a phi");
  bool foundit = false;
  for (auto UI = Def->use_begin(), UE = Def->use_end(); UI != UE; ++UI)
    if (&*UI == Use) {
      foundit = true;
      break;
    }

  assert(foundit && "Did not find use in def's use list");
}

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
          verifyUseInDefs(MU->getUseOperand(), MU);
        else if (MemoryDef *MD = dyn_cast<MemoryDef>(MA))
          verifyUseInDefs(MD->getUseOperand(), MD);
        else if (MemoryPhi *MP = dyn_cast<MemoryPhi>(MA)) {
          for (unsigned i = 0, e = MP->getNumIncomingValues(); i != e; ++i)
            verifyUseInDefs(MP->getIncomingValue(i), MP);
        }
      }
    }
  }
}

MemoryAccess *MemorySSA::getMemoryAccess(const Value *I) {
  return InstructionToMemoryAccess.lookup(I);
}

void MemoryDef::print(raw_ostream &OS) {
  MemoryAccess *UO = getUseOperand();
  OS << getDefVersion() << " = "
     << "MemoryDef(";
  OS << getVersionNumberFromAccess(UO) << ")";
}

void MemoryPhi::print(raw_ostream &OS) {
  OS << getDefVersion() << " = "
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
    OS << getVersionNumberFromAccess(MA);
    OS << "}";
    if (i + 1 < e)
      OS << ",";
  }
  OS << ")";
}

void MemoryUse::print(raw_ostream &OS) {
  MemoryAccess *UO = getUseOperand();
  OS << "MemoryUse(";
  OS << getVersionNumberFromAccess(UO);
  OS << ")";
}
