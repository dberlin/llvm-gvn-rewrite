//===-- MemorySSA.cpp - Memory SSA Builder---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------===//
//
// This file implements the MemorySSA class.
//
//===----------------------------------------------------------------===//
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

INITIALIZE_PASS_WITH_OPTIONS_BEGIN(MemorySSAPrinterPass, "print-memoryssa",
                                   "Memory SSA", true, true)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(MemorySSAPrinterPass, "print-memoryssa", "Memory SSA", true,
                    true);

INITIALIZE_PASS(MemorySSALazy, "memoryssalazy", "Memory SSA", true, true);

namespace llvm {
// An annotator class to print memory ssa information in comments
class MemorySSAAnnotatedWriter : public AssemblyAnnotationWriter {
  friend class MemorySSA;
  const MemorySSA *MSSA;

public:
  MemorySSAAnnotatedWriter(const MemorySSA *M) : MSSA(M) {}

  virtual void emitBasicBlockStartAnnot(const BasicBlock *BB,
                                        formatted_raw_ostream &OS) {
    MemoryAccess *MA = MSSA->getMemoryAccess(BB);
    if (MA)
      OS << "; " << *MA << "\n";
  }

  virtual void emitInstructionAnnot(const Instruction *I,
                                    formatted_raw_ostream &OS) {
    MemoryAccess *MA = MSSA->getMemoryAccess(I);
    if (MA)
      OS << "; " << *MA << "\n";
  }
};
}

// This is the same algorithm as PromoteMemoryToRegister's phi
// placement algorithm.

void MemorySSA::determineInsertionPoint(
    AccessMap &BlockAccesses, const SmallPtrSetImpl<BasicBlock *> &DefBlocks) {
  // Compute dominator levels and BB numbers
  DenseMap<DomTreeNode *, unsigned> DomLevels;
  computeDomLevels(DomLevels);

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

  SmallVector<BasicBlock *, 32> DFBlocks;
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

        DFBlocks.push_back(SuccBB);
        if (!DefBlocks.count(SuccBB))
          PQ.push(std::make_pair(SuccNode, SuccLevel));
      }

      for (auto &C : *Node)
        if (!Visited.count(C))
          Worklist.push_back(C);
    }
  }

  for (auto &BB : DFBlocks) {
    // Insert phi node
    auto Accesses = BlockAccesses.lookup(BB);
    if (!Accesses) {
      Accesses = new iplist<MemoryAccess>;
      BlockAccesses.insert(std::make_pair(BB, Accesses));
    }
    MemoryPhi *Phi = new MemoryPhi(
        BB, std::distance(pred_begin(BB), pred_end(BB)), nextID++);
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
                           MemorySSAWalker *Walker) {
NextIteration:
  const auto Accesses = BlockAccesses.lookup(BB);

  // First rename the phi nodes
  if (Accesses && isa<MemoryPhi>(Accesses->front())) {
    MemoryPhi *Phi = cast<MemoryPhi>(&Accesses->front());
    unsigned NumEdges = std::count(succ_begin(Pred), succ_end(Pred), BB);
    assert(NumEdges && "Must be at least one edge from Pred to BB!");
    for (unsigned i = 0; i != NumEdges; ++i)
      Phi->addIncoming(IncomingVal, Pred);

    IncomingVal = Phi;
  }

  // Don't revisit blocks.
  if (!Visited.insert(BB).second)
    return;

  // Skip if the list is empty, but we still have to pass thru the
  // incoming value info/etc to successors
  if (Accesses)
    for (auto &L : *Accesses) {
      if (isa<MemoryPhi>(L))
        continue;

      if (MemoryUse *MU = dyn_cast<MemoryUse>(&L)) {
        MU->setDefiningAccess(IncomingVal);
        auto RealVal = Walker->getClobberingMemoryAccess(MU->getMemoryInst());
        MU->setDefiningAccess(RealVal);
      } else if (MemoryDef *MD = dyn_cast<MemoryDef>(&L)) {
        // We can't legally optimize defs, because we only allow single memory
        // phis/uses on operations, and if we optimize these, we can end up with
        // multiple reaching defs.  Uses do not have this problem, since they do
        // not produce a value
        MD->setDefiningAccess(IncomingVal);
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
                                             BasicBlock *BB) {
  assert(!DT->isReachableFromEntry(BB) &&
         "Reachable block found while handling unreachable blocks");

  auto Accesses = BlockAccesses.lookup(BB);
  if (!Accesses)
    return;

  for (auto AI = Accesses->begin(), AE = Accesses->end(); AI != AE;) {
    auto Next = std::next(AI);
    // If we have a phi, just remove it. We are going to replace all
    // users with live on entry.
    if (isa<MemoryPhi>(&*AI)) {
      Accesses->erase(AI);
    } else if (MemoryUse *U = dyn_cast<MemoryUse>(&*AI)) {
      U->setDefiningAccess(LiveOnEntryDef);
    } else if (MemoryDef *D = dyn_cast<MemoryDef>(&*AI)) {
      D->setDefiningAccess(LiveOnEntryDef);
    }
    AI = Next;
  }
}

MemorySSA::MemorySSA(Function &Func)
    : F(Func), LiveOnEntryDef(nullptr), nextID(0), builtAlready(false),
      Walker(nullptr) {}

MemorySSA::~MemorySSA() {
  InstructionToMemoryAccess.clear();
  // This will destroy and delete all the accesses
  PerBlockAccesses.clear();
  delete LiveOnEntryDef;
}

MemorySSAWalker *MemorySSA::buildMemorySSA(AliasAnalysis *AA,
                                           DominatorTree *DT) {
  if (builtAlready)
    return Walker;
  else
    Walker = new CachingMemorySSAWalker(this, AA);

  this->AA = AA;
  this->DT = DT;

  // We create an access to represent "live on entry", for things like
  // arguments or users of globals. We do not actually insert it in to
  // the IR.
  BasicBlock &StartingPoint = F.getEntryBlock();
  LiveOnEntryDef = new MemoryDef(nullptr, nullptr, &StartingPoint, nextID++);

  // We temporarily maintain lists of memory accesses per-block,
  // trading time for memory. We could just look up the memory access
  // for every possible instruction in the stream.  Instead, we build
  // lists, and then throw it out once the use-def form is built.
  SmallPtrSet<BasicBlock *, 32> DefiningBlocks;

  for (auto &B : F) {
    AccessListType *Accesses = nullptr;
    for (auto &I : B) {
      bool use = false;
      bool def = false;
      AliasAnalysis::ModRefResult ModRef = AA->getModRefInfo(&I);
      if (ModRef & AliasAnalysis::Mod)
        def = true;
      if (ModRef & AliasAnalysis::Ref)
        use = true;

      // Defs are already uses, so use && def == def
      if (use && !def) {
        MemoryUse *MU = new MemoryUse(nullptr, &I, &B);
        InstructionToMemoryAccess.insert(std::make_pair(&I, MU));
        if (!Accesses) {
          Accesses = new AccessListType();
          PerBlockAccesses.insert(std::make_pair(&B, Accesses));
        }
        Accesses->push_back(MU);
      }
      if (def) {
        MemoryDef *MD = new MemoryDef(nullptr, &I, &B, nextID++);
        InstructionToMemoryAccess.insert(std::make_pair(&I, MD));
        DefiningBlocks.insert(&B);
        if (!Accesses) {
          Accesses = new AccessListType();
          PerBlockAccesses.insert(std::make_pair(&B, Accesses));
        }
        Accesses->push_back(MD);
      }
    }
  }
  // Determine where our PHI's should go
  determineInsertionPoint(PerBlockAccesses, DefiningBlocks);

  // Now do regular SSA renaming
  SmallPtrSet<BasicBlock *, 16> Visited;

  std::vector<RenamePassData> RenamePassWorklist;
  RenamePassWorklist.push_back({F.begin(), nullptr, LiveOnEntryDef});
  do {
    RenamePassData RPD;
    RPD.swap(RenamePassWorklist.back());
    RenamePassWorklist.pop_back();
    renamePass(RPD.BB, RPD.Pred, RPD.MA, PerBlockAccesses, RenamePassWorklist,
               Visited, Walker);
  } while (!RenamePassWorklist.empty());

  // At this point, we may have unreachable blocks with unreachable accesses
  // Given any uses in unreachable blocks the live on entry definition
  if (Visited.size() != F.size()) {
    for (auto &B : F)
      if (!Visited.count(&B))
        markUnreachableAsLiveOnEntry(PerBlockAccesses, &B);
  }

  builtAlready = true;
  return Walker;
}

static MemoryAccess *getDefiningAccess(MemoryAccess *MA) {
  if (MemoryUse *MU = dyn_cast<MemoryUse>(MA))
    return MU->getDefiningAccess();
  else if (MemoryDef *MD = dyn_cast<MemoryDef>(MA))
    return MD->getDefiningAccess();
  else
    return nullptr;
}
static void setDefiningAccess(MemoryAccess *Of, MemoryAccess *To) {
  if (MemoryUse *MU = dyn_cast<MemoryUse>(Of))
    MU->setDefiningAccess(To);
  else if (MemoryDef *MD = dyn_cast<MemoryDef>(Of))
    MD->setDefiningAccess(To);
}

void MemorySSA::replaceMemoryAccessWithNewAccess(MemoryAccess *Replacee,
                                                 Instruction *Replacer) {

  // There is no easy way to assert that you are replacing it with a memory
  // access, and this call will assert it for us
  AliasAnalysis::ModRefResult ModRef = AA->getModRefInfo(Replacer);
  bool def = false, use = false;
  if (ModRef & AliasAnalysis::Mod)
    def = true;
  if (ModRef & AliasAnalysis::Ref)
    use = true;
  // A memory instruction that doesn't affect memory. Okay ....
  if (!def && !use)
    return;
  bool DefinedByPhi = false;
  BasicBlock *ReplacerBlock = Replacer->getParent();
  MemoryAccess *MA = nullptr;
  MemoryAccess *DefiningAccess = getDefiningAccess(Replacee);

  // Handle the case we are replacing a phi node, in which case, we don't kill
  // the phi node
  if (DefiningAccess == nullptr) {
    assert(isa<MemoryPhi>(Replacee) &&
           "Should have been a phi node if we can't get a defining access");
    DefinedByPhi = true;
    assert(DT->dominates(Replacee->getBlock(), ReplacerBlock) &&
           "Need to reuse PHI for defining access, but it will not dominate "
           "replacing instruction");
    DefiningAccess = Replacee;
  }

  if (def) {
    MemoryDef *MD =
        new MemoryDef(DefiningAccess, Replacer, ReplacerBlock, nextID++);
    MA = MD;
  } else if (use) {
    MemoryUse *MU = new MemoryUse(DefiningAccess, Replacer, ReplacerBlock);
    MA = MU;
  }

  // Insert our access after any phi nodes in the block
  AccessListType *Accesses = nullptr;
  auto Result = PerBlockAccesses.insert(std::make_pair(ReplacerBlock, nullptr));

  if (!Result.second) {
    Accesses = Result.first->second;
  } else {
    Accesses = new AccessListType();
    Result.first->second = Accesses;
  }
  auto AI = Accesses->begin();
  for (auto AE = Accesses->end(); AI != AE; ++AI) {
    if (!isa<MemoryPhi>(AI))
      break;
  }
  Accesses->insert(AI, MA);
  InstructionToMemoryAccess.insert({Replacer, MA});
  replaceMemoryAccess(Replacee, MA);
}
static Instruction *getMemoryInst(MemoryAccess *MA) {
  if (MemoryUse *MU = dyn_cast<MemoryUse>(MA))
    return MU->getMemoryInst();
  else if (MemoryDef *MD = dyn_cast<MemoryDef>(MA))
    return MD->getMemoryInst();

  return nullptr;
}

bool MemorySSA::dominatesUse(MemoryAccess *Replacer,
                             MemoryAccess *Replacee) const {
  if (isa<MemoryUse>(Replacee) || isa<MemoryDef>(Replacee))
    return DT->dominates(Replacer->getBlock(), Replacee->getBlock());
  MemoryPhi *MP = cast<MemoryPhi>(Replacee);
  for (const auto &U : MP->args())
    if (U.second == Replacee)
      if (!DT->dominates(Replacer->getBlock(), U.first))
        return false;
  return true;
}

void MemorySSA::replaceMemoryAccess(MemoryAccess *Replacee,
                                    MemoryAccess *Replacer) {
  bool replacedAllPhiEntries = true;
  // If we are replacing a phi node, we may still actually use it, since we
  // may now be defined in terms of it.
  bool usedByReplacee = getDefiningAccess(Replacer) == Replacee;
  // Just to note: We can replace the live on entry def, unlike removing it, so
  // we don't assert here, but it's almost always a bug, unless you are
  // inserting a load/store in a block that dominates the rest of the program.
  for (auto U : Replacee->uses()) {
    if (U == Replacer)
      continue;
    assert(dominatesUse(Replacer, Replacee) &&
           "Definitions will not dominate uses in replacement!");
    if (MemoryUse *MU = dyn_cast<MemoryUse>(U))
      MU->setDefiningAccess(Replacer);
    else if (MemoryDef *MD = dyn_cast<MemoryDef>(U))
      MD->setDefiningAccess(Replacer);
    else if (MemoryPhi *MP = dyn_cast<MemoryPhi>(U)) {
      for (unsigned i = 0, e = MP->getNumIncomingValues(); i != e; ++i) {
        if (MP->getIncomingValue(i) == Replacee)
          MP->setIncomingValue(i, Replacer);
        else
          replacedAllPhiEntries = false;
      }
    }
  }
  // Kill our dead replacee if it's dead
  if (replacedAllPhiEntries && !usedByReplacee) {
    assert(Replacee->use_empty() &&
           "Trying to remove memory access that still has uses");
    setDefiningAccess(Replacee, nullptr);
    // The call below to erase will destroy MA, so we can't change the order we
    // are doing things here
    Instruction *MemoryInst = getMemoryInst(Replacee);
    if (MemoryInst)
      InstructionToMemoryAccess.erase(MemoryInst);
    PerBlockAccesses[Replacee->getBlock()]->erase(Replacee);
  }
}

#ifndef NDEBUG
static bool onlySingleValue(MemoryPhi *MP) {
  MemoryAccess *MA = nullptr;

  for (const auto &Arg : MP->args()) {
    if (!MA)
      MA = Arg.second;
    else if (MA != Arg.second)
      return false;
  }
  return true;
}
#endif

void MemorySSA::removeMemoryAccess(MemoryAccess *MA) {
  assert(MA != LiveOnEntryDef && "Trying to remove the live on entry def");
  // We can only delete phi nodes if they are use empty
  if (MemoryPhi *MP = dyn_cast<MemoryPhi>(MA)) {
    // This code only used in assert builds
    (void)MP;
    assert(MP->use_empty() && "We can't delete memory phis that still have "
                              "uses, we don't know where the uses should "
                              "repoint to!");
    assert(onlySingleValue(MP) && "This phi still points to multiple values, "
                                  "which means it is still needed");
  } else {
    MemoryAccess *DefiningAccess = getDefiningAccess(MA);
    // Re-point the uses at our defining access
    for (auto U : MA->uses()) {
      assert(dominatesUse(DefiningAccess, U) &&
             "Definitions will not dominate uses in removal!");
      if (MemoryUse *MU = dyn_cast<MemoryUse>(U))
        MU->setDefiningAccess(DefiningAccess);
      else if (MemoryDef *MD = dyn_cast<MemoryDef>(U))
        MD->setDefiningAccess(DefiningAccess);
      else if (MemoryPhi *MP = dyn_cast<MemoryPhi>(U)) {
        for (unsigned i = 0, e = MP->getNumIncomingValues(); i != e; ++i) {
          if (MP->getIncomingValue(i) == MA)
            MP->setIncomingValue(i, DefiningAccess);
        }
      }
    }
  }

  // The call below to erase will destroy MA, so we can't change the order we
  // are doing things here
  Instruction *MemoryInst = getMemoryInst(MA);
  if (MemoryInst)
    InstructionToMemoryAccess.erase(MemoryInst);
  assert(MA->use_empty() &&
         "Trying to remove memory access that still has uses");
  setDefiningAccess(MA, nullptr);
  PerBlockAccesses[MA->getBlock()]->erase(MA);
}

void MemorySSA::print(raw_ostream &OS) const {
  MemorySSAAnnotatedWriter Writer(this);
  F.print(OS, &Writer);
}

void MemorySSA::dump(Function &F) {
  MemorySSAAnnotatedWriter Writer(this);
  F.print(dbgs(), &Writer);
}

// Verify the domination properties of MemorySSA
// This means that each definition should dominate all of its uses
void MemorySSA::verifyDomination(Function &F) {
  for (auto &B : F) {
    // Phi nodes are attached to basic blocks
    MemoryAccess *MA = getMemoryAccess(&B);
    if (MA) {
      MemoryPhi *MP = cast<MemoryPhi>(MA);
      for (const auto &U : MP->uses()) {
        BasicBlock *UseBlock;

        // Phi operands are used on edges, we simulate the right domination by
        // acting as if the use occurred at the end of the predecessor block.
        if (MemoryPhi *P = dyn_cast<MemoryPhi>(U)) {
          for (const auto &Arg : P->args()) {
            if (Arg.second == MP) {
              UseBlock = Arg.first;
              break;
            }
          }
        } else {
          UseBlock = U->getBlock();
        }

        assert(DT->dominates(MP->getBlock(), UseBlock) &&
               "Memory PHI does not dominate it's uses");
      }
    }
    for (auto &I : B) {
      MA = getMemoryAccess(&I);
      if (MA) {
        if (MemoryDef *MD = dyn_cast<MemoryDef>(MA))
          for (const auto &U : MD->uses()) {
            BasicBlock *UseBlock;
            // Things are allowed to flow to phi nodes over their predecessor
            // edge, so we ignore phi node domination for the moment
            if (MemoryPhi *P = dyn_cast<MemoryPhi>(U)) {
              for (const auto &Arg : P->args()) {
                if (Arg.second == MD) {
                  UseBlock = Arg.first;
                  break;
                }
              }
            } else {
              UseBlock = U->getBlock();
            }
            assert(DT->dominates(MD->getBlock(), UseBlock) &&
                   "Memory Def does not dominate it's uses");
          }
      }
    }
  }
}

void MemorySSA::verifyUseInDefs(MemoryAccess *Def, MemoryAccess *Use) {
  // The live on entry use may cause us to get a NULL def here
  if (Def == nullptr) {
    assert(isLiveOnEntryDef(Use) &&
           "Null def but use not point to live on entry def");
    return;
  }
  assert(Def->findUse(Use) && "Did not find use in def's use list");
}

// Verify the immediate use information, by walking all the memory
// accesses and verifying that, for each use, it appears in the
// appropriate def's use list

void MemorySSA::verifyDefUses(Function &F) {
  for (auto &B : F) {
    // Phi nodes are attached to basic blocks
    MemoryAccess *MA = getMemoryAccess(&B);
    if (MA) {
      assert(isa<MemoryPhi>(MA) &&
             "Something other than phi node on basic block");
      MemoryPhi *MP = cast<MemoryPhi>(MA);
      for (unsigned i = 0, e = MP->getNumIncomingValues(); i != e; ++i)
        verifyUseInDefs(MP->getIncomingValue(i), MP);
    }
    for (auto &I : B) {
      MA = getMemoryAccess(&I);
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

void MemoryDef::print(raw_ostream &OS) const {
  MemoryAccess *UO = getDefiningAccess();

  OS << getID() << " = "
     << "MemoryDef(";
  if (UO && UO->getID() != 0)
    OS << UO->getID();
  else
    OS << "liveOnEntry";
  OS << ")";
}

void MemoryPhi::print(raw_ostream &OS) const {
  OS << getID() << " = "
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
    if (MA->getID() != 0)
      OS << MA->getID();
    else
      OS << "liveOnEntry";
    OS << "}";
    if (i + 1 < e)
      OS << ",";
  }
  OS << ")";
}

void MemoryUse::print(raw_ostream &OS) const {
  MemoryAccess *UO = getDefiningAccess();
  OS << "MemoryUse(";
  OS << UO->getID();
  OS << ")";
}

void MemoryAccess::dump() const {
  print(dbgs());
  dbgs() << "\n";
}

char MemorySSAPrinterPass::ID = 0;

MemorySSAPrinterPass::MemorySSAPrinterPass() : FunctionPass(ID) {
  initializeMemorySSAPrinterPassPass(*PassRegistry::getPassRegistry());
}

void MemorySSAPrinterPass::releaseMemory() {
  delete MSSA;
  delete Walker;
}

void MemorySSAPrinterPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<AliasAnalysis>();
  AU.addRequired<DominatorTreeWrapperPass>();
}

bool MemorySSAPrinterPass::doInitialization(Module &M) {

  VerifyMemorySSA =
      M.getContext()
          .template getOption<bool, MemorySSAPrinterPass,
                              &MemorySSAPrinterPass::VerifyMemorySSA>();
  return false;
}

void MemorySSAPrinterPass::registerOptions() {
  OptionRegistry::registerOption<bool, MemorySSAPrinterPass,
                                 &MemorySSAPrinterPass::VerifyMemorySSA>(
      "verify-memoryssa", "Run the Memory SSA verifier", false);
}

void MemorySSAPrinterPass::print(raw_ostream &OS, const Module *M) const {
  MSSA->dump(*F);
}

bool MemorySSAPrinterPass::runOnFunction(Function &F) {
  this->F = &F;
  MSSA = new MemorySSA(F);
  AliasAnalysis *AA = &getAnalysis<AliasAnalysis>();
  DominatorTree *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  Walker = MSSA->buildMemorySSA(AA, DT);

  if (VerifyMemorySSA) {
    MSSA->verifyDefUses(F);
    MSSA->verifyDomination(F);
  }

  return false;
}

char MemorySSALazy::ID = 0;

MemorySSALazy::MemorySSALazy() : FunctionPass(ID) {
  initializeMemorySSALazyPass(*PassRegistry::getPassRegistry());
}

void MemorySSALazy::releaseMemory() { delete MSSA; }

bool MemorySSALazy::runOnFunction(Function &F) {
  MSSA = new MemorySSA(F);
  return false;
}

MemorySSAWalker::MemorySSAWalker(MemorySSA *M) : MSSA(M) {}

CachingMemorySSAWalker::CachingMemorySSAWalker(MemorySSA *M, AliasAnalysis *A)
    : MemorySSAWalker(M), AA(A) {}

CachingMemorySSAWalker::~CachingMemorySSAWalker() {
  CachedUpwardsClobberingAccess.clear();
  CachedUpwardsClobberingCall.clear();
}

struct CachingMemorySSAWalker::UpwardsMemoryQuery {
  // True if our original query started off as a call
  bool isCall;
  // The pointer location we are going to query about. This will be
  // empty if isCall is true
  AliasAnalysis::Location Loc;
  // This is the call we were querying about. This will be null if
  // isCall is false
  const Instruction *Call;
  // Set of visited Instructions for this query
  SmallPtrSet<const MemoryAccess *, 32> Visited;
  // Set of visited call accesses for this query This is separated out because
  // you can always cache and lookup the result of call queries (IE when
  // isCall == true) for every call in the chain. The calls have no AA
  // location associated with them with them, and thus, no context dependence.
  SmallPtrSet<const MemoryAccess *, 32> VisitedCalls;
};

void CachingMemorySSAWalker::doCacheInsert(const MemoryAccess *M,
                                           MemoryAccess *Result,
                                           const UpwardsMemoryQuery &Q) {
  if (Q.isCall)
    CachedUpwardsClobberingCall.insert(std::make_pair(M, Result));
  else
    CachedUpwardsClobberingAccess.insert(
        std::make_pair(std::make_pair(M, Q.Loc), Result));
}

MemoryAccess *
CachingMemorySSAWalker::doCacheLookup(const MemoryAccess *M,
                                      const UpwardsMemoryQuery &Q) {

  ++NumClobberCacheLookups;
  MemoryAccess *Result;

  if (Q.isCall)
    Result = CachedUpwardsClobberingCall.lookup(M);
  else
    Result = CachedUpwardsClobberingAccess.lookup(std::make_pair(M, Q.Loc));

  if (Result) {
    ++NumClobberCacheHits;
    return Result;
  }
  return nullptr;
}

// Get the clobbering memory access for a phi node and alias location
std::pair<MemoryAccess *, bool>
CachingMemorySSAWalker::getClobberingMemoryAccess(
    MemoryPhi *P, struct UpwardsMemoryQuery &Q) {

  bool HitVisited = false;

  ++NumClobberCacheLookups;
  auto CacheResult = doCacheLookup(P, Q);
  if (CacheResult)
    return std::make_pair(CacheResult, false);

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

  // Don't try to walk past an incomplete phi node during construction
  // This can only occur during construction.

  if (P->getNumIncomingValues() != P->getNumPreds())
    return std::make_pair(P, false);

  // If we already got here once, and didn't get to an answer (if we
  // did, it would have been cached below), we must be stuck in
  // mutually recursive phi nodes.  In that case, the correct answer
  // is "we can ignore the phi node if all the other arguments turn
  // out okay" (since it cycles between itself and the other
  // arguments).  We return true here, and are careful to make sure we
  // only pass through "true" when we are giving results
  // for the cycle itself.
  if (!Q.Visited.insert(P).second)
    return std::make_pair(P, true);

  // Look through 1 argument phi nodes
  if (P->getNumIncomingValues() == 1) {
    auto SingleResult = getClobberingMemoryAccess(P->getIncomingValue(0), Q);

    HitVisited = SingleResult.second;
    Result = SingleResult.first;
  } else {
    MemoryAccess *TargetResult = nullptr;

    // This is true if we hit ourselves from every argument
    bool AllVisited = true;
    for (unsigned i = 0; i < P->getNumIncomingValues(); ++i) {
      MemoryAccess *Arg = P->getIncomingValue(i);
      auto ArgResult = getClobberingMemoryAccess(Arg, Q);
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
  doCacheInsert(P, Result, Q);

  return std::make_pair(Result, HitVisited);
}

// For a given MemoryAccess, walk backwards using Memory SSA and find
// the MemoryAccess that actually clobbers Loc.  The second part of
// the pair we return is whether we hit a cyclic phi node.
std::pair<MemoryAccess *, bool>
CachingMemorySSAWalker::getClobberingMemoryAccess(
    MemoryAccess *MA, struct UpwardsMemoryQuery &Q) {
  MemoryAccess *CurrAccess = MA;
  while (true) {
    // Should be either a Memory Def or a Phi node at this point
    if (MemoryPhi *P = dyn_cast<MemoryPhi>(CurrAccess))
      return getClobberingMemoryAccess(P, Q);
    else {
      MemoryDef *MD = dyn_cast<MemoryDef>(CurrAccess);
      assert(MD && "Use linked to something that is not a def");
      // If we hit the top, stop
      if (MSSA->isLiveOnEntryDef(MD))
        return std::make_pair(CurrAccess, false);
      Instruction *DefMemoryInst = MD->getMemoryInst();
      assert(DefMemoryInst &&
             "Defining instruction not actually an instruction");

      // While we can do lookups, we can't sanely do inserts here unless we
      // were to track every thing we saw along the way, since we don't
      // know where we will stop.
      if (auto CacheResult = doCacheLookup(CurrAccess, Q))
        return std::make_pair(CacheResult, false);
      if (!Q.isCall) {
        // Check whether our memory location is modified by this instruction
        if (AA->getModRefInfo(DefMemoryInst, Q.Loc) & AliasAnalysis::Mod)
          break;
      } else {
        // If this is a call, try then mark it for caching
        if (ImmutableCallSite(DefMemoryInst)) {
          Q.VisitedCalls.insert(MD);
        }
        if (AA->instructionClobbersCall(DefMemoryInst, Q.Call))
          break;
      }
    }
    MemoryAccess *NextAccess = cast<MemoryDef>(CurrAccess)->getDefiningAccess();
    // Walk from def to def
    CurrAccess = NextAccess;
  }
  doCacheInsert(MA, CurrAccess, Q);
  doCacheInsert(CurrAccess, CurrAccess, Q);
  return std::make_pair(CurrAccess, false);
}

// For a given instruction, walk backwards using Memory SSA and find
// the memory access that actually clobbers this one, skipping non-aliasing
// ones along the way
MemoryAccess *
CachingMemorySSAWalker::getClobberingMemoryAccess(const Instruction *I) {
  MemoryAccess *StartingAccess = MSSA->getMemoryAccess(I);
  struct UpwardsMemoryQuery Q;

  // First extract our location, then start walking until it is
  // clobbered
  // For calls, we store the call instruction we started with in
  // Loc.Ptr
  AliasAnalysis::Location Loc(I);

  // We can't sanely do anything with a FenceInst, they conservatively
  // clobber all memory, and have no locations to get pointers from to
  // try to disambiguate
  if (isa<FenceInst>(I)) {
    return StartingAccess;
  } else if (!isa<CallInst>(I) && !isa<InvokeInst>(I)) {
    Q.isCall = false;
    Q.Loc = AA->getLocation(I);
  } else {
    Q.isCall = true;
    Q.Call = I;
  }
  auto CacheResult = doCacheLookup(StartingAccess, Q);
  if (CacheResult)
    return CacheResult;

  // If we started with a heap use, walk to the def
  if (MemoryUse *MU = dyn_cast<MemoryUse>(StartingAccess))
    StartingAccess = MU->getDefiningAccess();
  else if (MemoryDef *MD = dyn_cast<MemoryDef>(StartingAccess))
    StartingAccess = MD->getDefiningAccess();

  SmallPtrSet<MemoryAccess *, 32> Visited;
  MemoryAccess *FinalAccess =
      getClobberingMemoryAccess(StartingAccess, Q).first;
  doCacheInsert(StartingAccess, FinalAccess, Q);
  if (Q.isCall) {
    for (const auto &C : Q.VisitedCalls)
      doCacheInsert(C, FinalAccess, Q);
  }

  DEBUG(dbgs() << "Starting Memory SSA clobber for " << *I << " is ");
  DEBUG(dbgs() << *StartingAccess << "\n");
  DEBUG(dbgs() << "Final Memory SSA clobber for " << *I << " is ");
  DEBUG(dbgs() << *FinalAccess << "\n");

  return FinalAccess;
}

MemoryAccess *
DoNothingMemorySSAWalker::getClobberingMemoryAccess(const Instruction *I) {
  MemoryAccess *MA = MSSA->getMemoryAccess(I);
  if (isa<MemoryDef>(MA) || isa<MemoryPhi>(MA))
    return MA;
  MemoryUse *MU = cast<MemoryUse>(MA);
  return MU->getDefiningAccess();
}
