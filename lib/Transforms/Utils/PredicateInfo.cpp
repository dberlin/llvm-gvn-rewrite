//===-- PredicateInfo.cpp - PredicateInfo Builder--------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------===//
//
// This file implements the PredicateInfo class.
//
//===----------------------------------------------------------------===//
#include "llvm/Transforms/Utils/PredicateInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/IteratedDominanceFrontier.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/PHITransAddr.h"
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
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Transforms/Scalar.h"
#include <algorithm>
#define DEBUG_TYPE "predicateinfo"
using namespace llvm;
INITIALIZE_PASS_BEGIN(PredicateInfoWrapperPass, "predicateinfo",
                      "PredicateInfo", false, true)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(PredicateInfoWrapperPass, "predicateinfo", "PredicateInfo",
                    false, true)

INITIALIZE_PASS_BEGIN(PredicateInfoPrinterLegacyPass, "print-predicateinfo",
                      "PredicateInfo Printer", false, false)
INITIALIZE_PASS_DEPENDENCY(PredicateInfoWrapperPass)
INITIALIZE_PASS_END(PredicateInfoPrinterLegacyPass, "print-predicateinfo",
                    "PredicateInfo Printer", false, false)
static cl::opt<bool> VerifyPredicateInfo(
    "verify-predicateinfo", cl::init(false), cl::Hidden,
    cl::desc("Verify PredicateInfo in legacy printer pass."));
namespace llvm {

void ComputeLiveInBlocks(BasicBlock *DefBlock,
                         SmallPtrSetImpl<BasicBlock *> &UsingBlocks,
                         SmallPtrSetImpl<BasicBlock *> &LiveInBlocks) {
  // There will be a use in the defblock, but it dies in the defblock for sure
  // since the definition is before the use.
  UsingBlocks.erase(DefBlock);
  // To determine liveness, we must iterate through the predecessors of blocks
  // where the def is live.  Blocks are added to the worklist if we need to
  // check their predecessors.  Start with all the using blocks.
  SmallVector<BasicBlock *, 64> LiveInBlockWorklist(UsingBlocks.begin(),
                                                    UsingBlocks.end());
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
      if (P == DefBlock)
        continue;

      // Otherwise it is, add to the worklist.
      LiveInBlockWorklist.push_back(P);
    }
  }
}
struct PredicateInfo::ValueDFS {

  int DFSIn = 0;
  int DFSOut = 0;
  unsigned int LocalNum = 0;
  BasicBlock *Block = nullptr;
  // Only one of these will be set.
  Value *Def = nullptr;
  Use *U = nullptr;
  bool operator<(const ValueDFS &Other) const {
    return std::tie(DFSIn, DFSOut, LocalNum, Def, U) <
           std::tie(Other.DFSIn, Other.DFSOut, Other.LocalNum, Other.Def,
                    Other.U);
  }
};

// This is a stack that contains both the value and dfs info of where
// that value is valid.
class PredicateInfo::ValueDFSStack {
public:
  ValueDFS &back() { return ValueDFSStack.back(); }
  const ValueDFS &back() const { return ValueDFSStack.back(); }

  void pop_back() { ValueDFSStack.pop_back(); }

  void push_back(ValueDFS &V) { ValueDFSStack.push_back(V); }
  bool empty() const { return ValueDFSStack.empty(); }
  bool isInScope(int DFSIn, int DFSOut) const {
    if (empty())
      return false;
    return DFSIn >= ValueDFSStack.back().DFSIn &&
           DFSOut <= ValueDFSStack.back().DFSOut;
  }

  void popUntilDFSScope(int DFSIn, int DFSOut) {

    while (!ValueDFSStack.empty() &&
           !(DFSIn >= ValueDFSStack.back().DFSIn &&
             DFSOut <= ValueDFSStack.back().DFSOut)) {
      ValueDFSStack.pop_back();
    }
  }

private:
  SmallVector<ValueDFS, 8> ValueDFSStack;
};

struct PredicateInfo::SplitInfo {
  BasicBlock *BranchBB = nullptr;
  BasicBlock *SplitBB = nullptr;
  CmpInst *Comparison = nullptr;
  bool TakenEdge = false;
  SplitInfo(BasicBlock *BranchBB, BasicBlock *SplitBB, CmpInst *Comparison,
            bool TakenEdge)
      : BranchBB(BranchBB), SplitBB(SplitBB), Comparison(Comparison),
        TakenEdge(TakenEdge) {}
  SplitInfo() {}
};

/// \brief An assembly annotator class to print PredicateInfo information in
/// comments.
class PredicateInfoAnnotatedWriter : public AssemblyAnnotationWriter {
  friend class PredicateInfo;
  const PredicateInfo *PredInfo;

public:
  PredicateInfoAnnotatedWriter(const PredicateInfo *M) : PredInfo(M) {}

  virtual void emitBasicBlockStartAnnot(const BasicBlock *BB,
                                        formatted_raw_ostream &OS) {}

  virtual void emitInstructionAnnot(const Instruction *I,
                                    formatted_raw_ostream &OS) {
    if (const auto *PI = PredInfo->getPredicateInfoFor(I))
      OS << "; Has predicate info { TakenEdge:" << PI->TakenEdge << " }\n";
  }
};

void PredicateInfo::convertUsesToDFSOrdered(
    Value *Op, SmallVectorImpl<ValueDFS> &DFSOrderedSet) {
#if 0
  ValueDFS VD;
  VD.Val = Op;

  // Arguments dominate everything
  if (isa<Argument>(Op)) {
    VD.DFSIn = -1;
    VD.DFSOut = INT_MAX;
  } else {
    DomTreeNode *DomNode = DT->getNode(cast<Instruction>(Op)->getParent());
    VD.DFSIn = DomNode->getDFSNumIn();
    VD.DFSOut = DomNode->getDFSNumOut();
  }
  DFSOrderedSet.push_back(VD);
#endif
  // Now add the uses and phi copies
  for (auto &U : Op->uses()) {
    if (auto *I = dyn_cast<Instruction>(U.getUser())) {
      ValueDFS VD;

      // Put the phi node uses in the incoming block.
      BasicBlock *IBlock;
      if (auto *PN = dyn_cast<PHINode>(I)) {
        IBlock = PN->getIncomingBlock(U);
        // Make phi node users appear last in the incoming block
        // they are from.
        VD.LocalNum = ~0U;
      } else {
        IBlock = I->getParent();
        // These come after the phi copies, but before the phi node
        // uses. Otherwise order is not important.
        VD.LocalNum = 1;
      }
      DomTreeNode *DomNode = DT->getNode(IBlock);
      VD.DFSIn = DomNode->getDFSNumIn();
      VD.DFSOut = DomNode->getDFSNumOut();
      VD.U = &U;
      DFSOrderedSet.push_back(VD);
    }
  }
}

#if 0

  This code is code to place phi nodes.  No version of ESSA actually creates phi nodes, because
  the original variable is always still live after the merge. Thus, any phi nodes seem pointless.

          // Compute live in to avoid adding using blocks unnecessarily
          // TODO: We can compute live in for all variables at once, instead of
          // this, where we are computing where *any* of them are live.
          SmallPtrSet<BasicBlock *, 8> UsingBlocks;
          for (auto *U : Op->users())
            if (auto *UI = dyn_cast<Instruction>(U))
              UsingBlocks.insert(UI->getParent());
          SmallPtrSet<BasicBlock *, 8> LiveInBlocks;
          ComputeLiveInBlocks(Comparison->getParent(), UsingBlocks,
                              LiveInBlocks);
          auto FilteredSuccs = make_filter_range(
              SuccsToProcess, [&LiveInBlocks](BasicBlock *BB) {
                return LiveInBlocks.count(BB);
              });
          if (FilteredSuccs.begin() == FilteredSuccs.end())
            continue;
        // and place phis, if any, for those copies.
        SmallVector<BasicBlock *, 32> PHIBlocks;
        PHIPlace.setLiveInBlocks(LiveInBlocks);
        PHIPlace.setDefiningBlocks(DefBlocks);
        PHIPlace.calculate(PHIBlocks);
        if (PHIBlocks.empty())
          continue;
        for (auto &Op : CmpOperands) {
          for (auto &PHIBB : PHIBlocks) {
            PHINode *PHI = PHINode::Create(
                Op->getType(),
                std::distance(pred_begin(PHIBB), pred_end(PHIBB)) + 1,
                "PredicateInfoPHI." + Op->getName(), &PHIBB->front());
            OriginalToNewMap[{Op, PHIBB}] = PHI;
            for (auto Pred : predecessors(PHIBB)) {
              auto LookupResult = OriginalToNewMap.lookup({Op, Pred});
              PHI->addIncoming(LookupResult ? LookupResult
                                            : UndefValue::get(Op->getType()),
                               Pred);
            }
          }
        }
#endif

void collectICmpOps(ICmpInst *Comparison,
                    SmallVectorImpl<Value *> &CmpOperands) {
  auto *Op0 = Comparison->getOperand(0);
  auto *Op1 = Comparison->getOperand(1);
  if (Op0 == Op1)
    return;
  CmpOperands.push_back(Comparison);
  // Only want real values, not constants
  if (isa<Instruction>(Op0) || isa<Argument>(Op0))
    CmpOperands.push_back(Op0);
  if (isa<Instruction>(Op1) || isa<Argument>(Op1))
    CmpOperands.push_back(Op1);
}

void PredicateInfo::buildPredicateInfo() {
  DT->updateDFSNumbers();
  // Collect operands to rename
  ForwardIDFCalculator PHIPlace(*DT);
  SmallPtrSet<Value *, 8> OpsToRename;
  for (auto DTN : depth_first(DT->getRootNode())) {
    SmallVector<Value *, 8> CmpOperands;
    SmallPtrSet<BasicBlock *, 8> DefBlocks;
    BasicBlock *BranchBB = DTN->getBlock();
    TerminatorInst *TI = BranchBB->getTerminator();

    if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
      if (!BI->isConditional())
        continue;
      BasicBlock *FirstBB = TI->getSuccessor(0);
      BasicBlock *SecondBB = TI->getSuccessor(1);
      bool FirstSinglePred = FirstBB->getSinglePredecessor();
      bool SecondSinglePred = SecondBB->getSinglePredecessor();
      SmallVector<BasicBlock *, 2> SuccsToProcess;
      // First make sure we have single preds for these successors
      if (FirstSinglePred)
        SuccsToProcess.push_back(FirstBB);
      if (SecondSinglePred)
        SuccsToProcess.push_back(SecondBB);
      if (SuccsToProcess.empty())
        continue;
      // Second, see if we have a comparison we support
      if (ICmpInst *Comparison = dyn_cast<ICmpInst>(BI->getCondition())) {
        collectICmpOps(Comparison, CmpOperands);
        // Process the comparison
        DefBlocks.insert(Comparison->getParent());
        // Now add our copies for our operands
        for (auto *Op : CmpOperands) {
          OpsToRename.insert(Op);
          auto &OperandInfo = getOrCreateValueInfo(Op);
          for (auto *Succ : SuccsToProcess) {
            bool TakenEdge = (Succ == FirstBB);
            OperandInfo.PossibleSplitBlocks.insert(Succ);
            OperandInfo.SplitInfos.insert(
                {Succ, {BranchBB, Succ, Comparison, TakenEdge}});
          }
        }
      }
    }
  }
  renameUses(OpsToRename);
}

void PredicateInfo::renameUses(SmallPtrSetImpl<Value *> &OpsToRename) {
  // Compute liveness, and rename in O(uses) per Op.
  for (auto *Op : OpsToRename) {
    unsigned Counter = 0;
    SmallVector<ValueDFS, 16> OrderedUses;
    const auto &ValueInfo = getValueInfo(Op);
    // Insert the possible copies into the use list.
    // They will become real copies if we find a real use for them, and never
    // created otherwise.
    for (auto PossibleCopyBlock : ValueInfo.PossibleSplitBlocks) {
      ValueDFS VD;
      DomTreeNode *DomNode = DT->getNode(PossibleCopyBlock);
      VD.DFSIn = DomNode->getDFSNumIn();
      VD.DFSOut = DomNode->getDFSNumOut();
      // Make these come first in a block
      VD.LocalNum = 0;
      VD.Block = PossibleCopyBlock;
      OrderedUses.push_back(VD);
    }

    convertUsesToDFSOrdered(Op, OrderedUses);
    std::sort(OrderedUses.begin(), OrderedUses.end());
    // Instead of the standard SSA renaming algorithm, which is O(Number of
    // instructions), and walks the entire dominator tree, we walk only the defs
    // + uses.  The standard SSA renaming algorithm does not really rely on the
    // dominator tree except to order the stack push/pops of the renaming
    // stacks, so that defs end up getting pushed before hitting the correct
    // uses.  This does not require the dominator tree, only the *order* of the
    // dominator tree. The complete and correct ordering of the defs and uses,
    // in dominator tree is contained in the DFS numbering of the dominator
    // tree. So we sort the defs and uses into the DFS ordering, and then just
    // use the renaming stack as per normal, pushing when we hit a def, popping
    // when we are out of the dfs scope for that def, and replacing any uses
    // with top of stack if it exists.
    // TODO: Use this algorithm to perform fast single-variable renaming in
    // promotememtoreg and memoryssa.
    ValueDFSStack RenameStack;
    // For each use, sorted into dfs order, push values and replaces uses with
    // top of stack, which will represent the reaching def.
    for (auto &VD : OrderedUses) {
      int MemberDFSIn = VD.DFSIn;
      int MemberDFSOut = VD.DFSOut;
      Value *MemberDef = VD.Def;
      Use *MemberUse = VD.U;
      bool PossibleCopy = !MemberDef && !MemberUse && VD.LocalNum == 0;
      if (RenameStack.empty()) {
        DEBUG(dbgs() << "Rename Stack is empty\n");
      } else {
        DEBUG(dbgs() << "Rename Stack Top DFS numbers are ("
                     << RenameStack.back().DFSIn << ","
                     << RenameStack.back().DFSOut << ")\n");
      }

      DEBUG(dbgs() << "Current DFS numbers are (" << MemberDFSIn << ","
                   << MemberDFSOut << ")\n");

      bool ShouldPush = (MemberDef || PossibleCopy);
      bool OutOfScope = !RenameStack.isInScope(MemberDFSIn, MemberDFSOut);
      if (OutOfScope || ShouldPush) {
        // Sync to our current scope.
        RenameStack.popUntilDFSScope(MemberDFSIn, MemberDFSOut);
        ShouldPush |= (MemberDef || PossibleCopy);
        if (ShouldPush) {
          RenameStack.push_back(VD);
        }
      }
      // If we get to this point, and the stack is empty we must have a use
      // with no renaming needed, just skip it.
      if (RenameStack.empty())
        continue;
      // Skip values, only want to rename the uses
      if (MemberDef || PossibleCopy)
        continue;
      ValueDFS &Result = RenameStack.back();

      // The possible copy dominated something, so materialize it
      if (!Result.Def) {
        auto SII = ValueInfo.SplitInfos.find(Result.Block);
        assert(SII != ValueInfo.SplitInfos.end());
        auto &SI = SII->second;
        PHINode *PHI = PHINode::Create(
            Op->getType(), 1, "pred." + Twine(Counter++) + "." + Op->getName(),
            &SI.SplitBB->front());
        PHI->addIncoming(Op, SI.BranchBB);
        OriginalToNewMap[{Op, SI.SplitBB}] = PHI;
        PredicateMap.insert({PHI, &SI});
        Result.Def = PHI;
      }

      DEBUG(dbgs() << "Found replacement " << *Result.Def << " for "
                   << *MemberUse->get() << " in " << *(MemberUse->getUser())
                   << "\n");
      MemberUse->set(Result.Def);
    }
  }
}

PredicateInfo::ValueInfo &PredicateInfo::getOrCreateValueInfo(Value *Operand) {
  auto OIN = ValueInfoNums.find(Operand);
  if (OIN == ValueInfoNums.end()) {
    // This will grow it
    ValueInfos.resize(ValueInfos.size() + 1);
    // This will use the new size and give us a 0 based number of the info
    auto InsertResult = ValueInfoNums.insert({Operand, ValueInfos.size() - 1});
    assert(InsertResult.second && "Value info number already existed?");
    return ValueInfos[InsertResult.first->second];
  }
  return ValueInfos[OIN->second];
}

const PredicateInfo::ValueInfo &
PredicateInfo::getValueInfo(Value *Operand) const {
  auto OINI = ValueInfoNums.lookup(Operand);
  assert(OINI != 0 && "Operand was not really in the Value Info Numbers");
  assert(OINI < ValueInfos.size() &&
         "Value Info Number greater than size of Value Info Table");
  return ValueInfos[OINI];
}

PredicateInfo::PredicateInfo(Function &F, DominatorTree *DT) : F(F), DT(DT) {
  // Push an empty operand info so that we can detect 0 as not finding one
  ValueInfos.resize(1);
  buildPredicateInfo();
}
PredicateInfo::~PredicateInfo() {}

void PredicateInfo::verifyPredicateInfo() const {}
void PredicateInfo::print(raw_ostream &OS) const {
  PredicateInfoAnnotatedWriter Writer(this);
  F.print(OS, &Writer);
}

void PredicateInfo::dump() const {
  PredicateInfoAnnotatedWriter Writer(this);
  F.print(dbgs(), &Writer);
}

char PredicateInfoPrinterLegacyPass::ID = 0;

PredicateInfoPrinterLegacyPass::PredicateInfoPrinterLegacyPass()
    : FunctionPass(ID) {
  initializePredicateInfoPrinterLegacyPassPass(
      *PassRegistry::getPassRegistry());
}

void PredicateInfoPrinterLegacyPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<PredicateInfoWrapperPass>();
  AU.addPreserved<PredicateInfoWrapperPass>();
}

bool PredicateInfoPrinterLegacyPass::runOnFunction(Function &F) {
  auto &PredInfo = getAnalysis<PredicateInfoWrapperPass>().getPredInfo();
  PredInfo.print(dbgs());
  if (VerifyPredicateInfo)
    PredInfo.verifyPredicateInfo();
  return false;
}

AnalysisKey PredicateInfoAnalysis::Key;

PredicateInfoAnalysis::Result
PredicateInfoAnalysis::run(Function &F, FunctionAnalysisManager &AM) {
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  return PredicateInfoAnalysis::Result(make_unique<PredicateInfo>(F, &DT));
}

PreservedAnalyses PredicateInfoPrinterPass::run(Function &F,
                                                FunctionAnalysisManager &AM) {
  OS << "PredicateInfo for function: " << F.getName() << "\n";
  AM.getResult<PredicateInfoAnalysis>(F).getPredInfo().print(OS);

  return PreservedAnalyses::all();
}

PreservedAnalyses PredicateInfoVerifierPass::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  AM.getResult<PredicateInfoAnalysis>(F).getPredInfo().verifyPredicateInfo();

  return PreservedAnalyses::all();
}

char PredicateInfoWrapperPass::ID = 0;

PredicateInfoWrapperPass::PredicateInfoWrapperPass() : FunctionPass(ID) {
  initializePredicateInfoWrapperPassPass(*PassRegistry::getPassRegistry());
}

void PredicateInfoWrapperPass::releaseMemory() { PredInfo.reset(); }

void PredicateInfoWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<DominatorTreeWrapperPass>();
}

bool PredicateInfoWrapperPass::runOnFunction(Function &F) {
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  PredInfo.reset(new PredicateInfo(F, &DT));
  return false;
}

void PredicateInfoWrapperPass::verifyAnalysis() const {
  PredInfo->verifyPredicateInfo();
}

void PredicateInfoWrapperPass::print(raw_ostream &OS, const Module *M) const {
  PredInfo->print(OS);
}
}
