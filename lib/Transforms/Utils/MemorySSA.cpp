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

INITIALIZE_PASS_WITH_OPTIONS_BEGIN(MemorySSAWrapperPass, "memoryssa",
                                   "Memory SSA", false, true)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(MemorySSAWrapperPass, "memoryssa", "Memory SSA", true,
                    true);

INITIALIZE_PASS(MemorySSALazy, "memoryssalazy", "Memory SSA", true, true);

// This is a temporary (IE will be deleted once consensus is reached
// in the review) flag to determine whether we should optimize uses
// while building so they point to the nearest actual clobber
#define OPTIMIZE_USES 1

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

// We need a unique numbering for each BB.

void MemorySSA::computeBBNumbers(Function &F,
                                 DenseMap<BasicBlock *, unsigned> &BBNumbers) {
  // Assign unique ids to basic blocks
  unsigned ID = 0;
  for (auto &I : F)
    BBNumbers[&I] = ID++;
}

// This is the same algorithm as PromoteMemoryToRegister's phi
// placement algorithm.

void MemorySSA::determineInsertionPoint(
    Function &F, AccessMap &BlockAccesses,
    const SmallPtrSetImpl<BasicBlock *> &DefBlocks) {
  // Compute dominator levels and BB numbers
  DenseMap<DomTreeNode *, unsigned> DomLevels;
  computeDomLevels(DomLevels);

  DenseMap<BasicBlock *, unsigned> BBNumbers;
  computeBBNumbers(F, BBNumbers);

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

        DFBlocks.push_back(std::make_pair(BBNumbers[SuccBB], SuccBB));
        if (!DefBlocks.count(SuccBB))
          PQ.push(std::make_pair(SuccNode, SuccLevel));
      }

      for (auto &C : *Node)
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
        MemoryPhi(BB, std::distance(pred_begin(BB), pred_end(BB)), nextID++);
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
                           SmallPtrSet<BasicBlock *, 16> &Visited, UseMap &Uses,
                           MemorySSAWalker *Walker) {
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
    for (auto &L : *Accesses) {
      if (isa<MemoryPhi>(L))
        continue;

      if (MemoryUse *MU = dyn_cast<MemoryUse>(L)) {
        MU->setDefiningAccess(IncomingVal);
        auto RealVal = Walker->getClobberingMemoryAccess(MU->getMemoryInst());
        MU->setDefiningAccess(RealVal);
        addUseToMap(Uses, RealVal, MU);
      } else if (MemoryDef *MD = dyn_cast<MemoryDef>(L)) {
        MD->setDefiningAccess(IncomingVal);
        auto RealVal = Walker->getClobberingMemoryAccess(MD->getMemoryInst());
        if (RealVal == MD)
          RealVal = IncomingVal;

        MD->setDefiningAccess(RealVal);
        addUseToMap(Uses, RealVal, MD);
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

MemorySSA::MemorySSA(Function &Func)
    : F(Func), LiveOnEntryDef(nullptr), nextID(0), builtAlready(false) {}

MemorySSA::~MemorySSA() {
  InstructionToMemoryAccess.clear();
  MemoryAccessAllocator.Reset();
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
  for (auto &D : Uses) {
    std::list<MemoryAccess *> *UseList = D.second;
    MemoryAccess *User = D.first;
    User->UseList =
        MemoryAccessAllocator.Allocate<MemoryAccess *>(UseList->size());
    for (auto &U : *UseList)
      User->addUse(U);
  }
}

void MemorySSA::buildMemorySSA(AliasAnalysis *AA, DominatorTree *DT,
                               MemorySSAWalker *Walker) {

  // We don't allow updating at the moment
  // But we can't assert since the dumper does eager buildingas
  assert(!builtAlready && "We don't support updating memory ssa at this time");

  this->AA = AA;
  this->DT = DT;

  // We create an access to represent "live on entry", for things like
  // arguments or users of globals. We do not actually insert it in to
  // the IR.
  BasicBlock &StartingPoint = F.getEntryBlock();
  LiveOnEntryDef = new (MemoryAccessAllocator)
      MemoryDef(nullptr, nullptr, &StartingPoint, nextID++);

  // We temporarily maintain lists of memory accesses per-block,
  // trading time for memory. We could just look up the memory access
  // for every possible instruction in the stream.  Instead, we build
  // lists, and then throw it out once the use-def form is built.
  AccessMap PerBlockAccesses;
  SmallPtrSet<BasicBlock *, 32> DefiningBlocks;

  for (auto &B : F) {
    std::list<MemoryAccess *> *Accesses = nullptr;
    for (auto &I : B) {
      bool use = false;
      bool def = false;
      if (isa<LoadInst>(&I)) {
        use = true;
        def = false;
      } else if (isa<StoreInst>(&I)) {
        use = false;
        def = true;
      } else {
        AliasAnalysis::ModRefResult ModRef = AA->getModRefInfo(&I);
        if (ModRef & AliasAnalysis::Mod)
          def = true;
        if (ModRef & AliasAnalysis::Ref)
          use = true;
      }

      // Defs are already uses, so use && def == def
      if (use && !def) {
        MemoryUse *MU = new (MemoryAccessAllocator) MemoryUse(nullptr, &I, &B);
        InstructionToMemoryAccess.insert(std::make_pair(&I, MU));
        if (!Accesses) {
          Accesses = new std::list<MemoryAccess *>;
          PerBlockAccesses.insert(std::make_pair(&B, Accesses));
        }
        Accesses->push_back(MU);
      }
      if (def) {
        MemoryDef *MD =
            new (MemoryAccessAllocator) MemoryDef(nullptr, &I, &B, nextID++);
        InstructionToMemoryAccess.insert(std::make_pair(&I, MD));
        DefiningBlocks.insert(&B);
        if (!Accesses) {
          Accesses = new std::list<MemoryAccess *>;
          PerBlockAccesses.insert(std::make_pair(&B, Accesses));
        }
        Accesses->push_back(MD);
      }
    }
  }
  // Determine where our PHI's should go
  determineInsertionPoint(F, PerBlockAccesses, DefiningBlocks);

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
               Visited, Uses, Walker);
  } while (!RenamePassWorklist.empty());

  // At this point, we may have unreachable blocks with unreachable accesses
  // Given any uses in unreachable blocks the live on entry definition
  if (Visited.size() != F.size()) {
    for (auto &B : F)
      if (!Visited.count(&B))
        markUnreachableAsLiveOnEntry(PerBlockAccesses, &B, Uses);
  }

  // Now convert our use lists into real uses
  addUses(Uses);

  // Delete our access lists
  for (auto &D : PerBlockAccesses)
    delete D.second;

  // Densemap does not like when you delete or change the value during
  // iteration.
  std::vector<std::list<MemoryAccess *> *> UseListsToDelete;
  for (auto &D : Uses)
    UseListsToDelete.push_back(D.second);

  Uses.clear();
  for (unsigned i = 0, e = UseListsToDelete.size(); i != e; ++i) {
    delete UseListsToDelete[i];
    UseListsToDelete[i] = nullptr;
  }
  builtAlready = true;
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
  assert(std::find(Def->use_begin(), Def->use_end(), Use) != Def->use_end() &&
         "Did not find use in def's use list");
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

void MemoryDef::print(raw_ostream &OS) {
  MemoryAccess *UO = getDefiningAccess();

  OS << getID() << " = "
     << "MemoryDef(";
  OS << UO->getID() << ")";
}

void MemoryPhi::print(raw_ostream &OS) {
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
    OS << MA->getID();
    OS << "}";
    if (i + 1 < e)
      OS << ",";
  }
  OS << ")";
}

void MemoryUse::print(raw_ostream &OS) {
  MemoryAccess *UO = getDefiningAccess();
  OS << "MemoryUse(";
  OS << UO->getID();
  OS << ")";
}

char MemorySSAWrapperPass::ID = 0;

MemorySSAWrapperPass::MemorySSAWrapperPass() : FunctionPass(ID) {
  initializeMemorySSAWrapperPassPass(*PassRegistry::getPassRegistry());
}

void MemorySSAWrapperPass::releaseMemory() {
  delete MSSA;
  delete Walker;
}

void MemorySSAWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<AliasAnalysis>();
  AU.addRequired<DominatorTreeWrapperPass>();
}

bool MemorySSAWrapperPass::doInitialization(Module &M) {
  DumpMemorySSA =
      M.getContext()
          .template getOption<bool, MemorySSAWrapperPass,
                              &MemorySSAWrapperPass::DumpMemorySSA>();

  VerifyMemorySSA =
      M.getContext()
          .template getOption<bool, MemorySSAWrapperPass,
                              &MemorySSAWrapperPass::VerifyMemorySSA>();
  return false;
}

void MemorySSAWrapperPass::registerOptions() {
  OptionRegistry::registerOption<bool, MemorySSAWrapperPass,
                                 &MemorySSAWrapperPass::DumpMemorySSA>(
      "dump-memoryssa", "Dump Memory SSA after building it", false);

  OptionRegistry::registerOption<bool, MemorySSAWrapperPass,
                                 &MemorySSAWrapperPass::VerifyMemorySSA>(
      "verify-memoryssa", "Run the Memory SSA verifier", false);
}
bool MemorySSAWrapperPass::runOnFunction(Function &F) {
  MSSA = new MemorySSA(F);
  AliasAnalysis *AA = &getAnalysis<AliasAnalysis>();
  DominatorTree *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  Walker = new CachingMemorySSAWalker(MSSA, AA);

  MSSA->buildMemorySSA(AA, DT, Walker);

  if (DumpMemorySSA) {
    MSSA->print(errs());
  }

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
  CachedClobberingAccess.clear();
  CachedClobberingCall.clear();
}

struct CachingMemorySSAWalker::MemoryQuery {
  // True if our original query started off as a call
  bool isCall;
  // The pointer location we are going to query about. This will be
  // empty if isCall is true
  AliasAnalysis::Location Loc;
  // This is the call we were querying about. This will be null if
  // isCall is false
  Instruction *Call;
  // Set of visited Instructions for this query
  SmallPtrSet<const MemoryAccess *, 32> Visited;
  // Set of visited call accesses for this query
  // We separate this because we can always cache the result of calls if
  // Q.iscall is true, because they have no location
  SmallPtrSet<const MemoryAccess *, 32> VisitedCalls;
};

void CachingMemorySSAWalker::doCacheInsert(const MemoryAccess *M,
                                           MemoryAccess *Result,
                                           const MemoryQuery &Q) {

  // DEBUG(dbgs() << F->getName() << " cache insert: " << Q.isCall << "\t"
  //              << (uint64_t)M << "\t" << (uint64_t)Q.Loc.Ptr << "\t"
  //              << (uint64_t)Result << "\n");

  if (Q.isCall)
    CachedClobberingCall.insert(std::make_pair(M, Result));
  else
    CachedClobberingAccess.insert(
        std::make_pair(std::make_pair(M, Q.Loc), Result));
}

MemoryAccess *CachingMemorySSAWalker::doCacheLookup(const MemoryAccess *M,
                                                    const MemoryQuery &Q) {

  ++NumClobberCacheLookups;
  MemoryAccess *Result;

  if (Q.isCall)
    Result = CachedClobberingCall.lookup(M);
  else
    Result = CachedClobberingAccess.lookup(std::make_pair(M, Q.Loc));
  // DEBUG(dbgs() << F->getName() << " cache lookup: " << Q.isCall << "\t"
  //              << (uint64_t)M << "\t" << (uint64_t)Q.Loc.Ptr << "\t"
  //              << (uint64_t)Result << "\n");

  if (Result) {
    ++NumClobberCacheHits;
    return Result;
  }
  return nullptr;
}

// Get the clobbering memory access for a phi node and alias location
std::pair<MemoryAccess *, bool>
CachingMemorySSAWalker::getClobberingMemoryAccess(MemoryPhi *P,
                                                  struct MemoryQuery &Q) {

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

#if OPTIMIZE_USES
  // Don't try to walk past an incomplete phi node during construction
  // This can only occur during construction, and only if we are optimizing
  // uses.
  if (P->getNumIncomingValues() != P->getNumPreds())
    return std::make_pair(P, false);
#endif
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
CachingMemorySSAWalker::getClobberingMemoryAccess(MemoryAccess *MA,
                                                  struct MemoryQuery &Q) {
  MemoryAccess *CurrAccess = MA;
  while (true) {
    // If we started with a heap use, walk to the def
    if (MemoryUse *MU = dyn_cast<MemoryUse>(CurrAccess))
      CurrAccess = MU->getDefiningAccess();

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
        // If this is a call, try lookup and then mark it for caching
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
CachingMemorySSAWalker::getClobberingMemoryAccess(Instruction *I) {
  MemoryAccess *StartingAccess = MSSA->getMemoryAccess(I);
  struct MemoryQuery Q;

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
