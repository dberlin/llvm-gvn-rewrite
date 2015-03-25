//===- ADCE.cpp - Code to perform dead code elimination -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Aggressive Dead Code Elimination pass.  This pass
// optimistically assumes that all instructions are dead until proven otherwise,
// allowing it to eliminate dead computations that other DCE passes do not
// catch, particularly involving loop computations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar.h"
using namespace llvm;

#define DEBUG_TYPE "adce"

STATISTIC(NumRemoved, "Number of instructions removed");

<<<<<<< b1c22cf838a72c9cbe0398193a4908cd3c3ca924
static bool aggressiveDCE(Function& F) {
  SmallPtrSet<Instruction*, 32> Alive;
  SmallVector<Instruction*, 128> Worklist;
||||||| merged common ancestors
static bool aggressiveDCE(Function& F) {
  SmallPtrSet<Instruction*, 128> Alive;
  SmallVector<Instruction*, 128> Worklist;
=======
namespace {
struct ADCE : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  ADCE() : FunctionPass(ID) {
    initializeADCEPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }
};
}

char ADCE::ID = 0;
INITIALIZE_PASS(ADCE, "adce", "Aggressive Dead Code Elimination", false, false)

bool ADCE::runOnFunction(Function &F) {
  if (skipOptnoneFunction(F))
    return false;

  SmallPtrSet<Instruction *, 128> Alive;
  SmallVector<Instruction *, 128> Worklist;
  SmallPtrSet<BasicBlock *, 128> SeenBBs;
>>>>>>> Make per-block access lists visible to all

  // Collect the set of "root" instructions that are known live.
  for (Instruction &I : instructions(F)) {
    if (isa<TerminatorInst>(I) || isa<DbgInfoIntrinsic>(I) || I.isEHPad() ||
        I.mayHaveSideEffects()) {
      Alive.insert(&I);
      Worklist.push_back(&I);
    }
  }

  // Propagate liveness backwards to operands.
  while (!Worklist.empty()) {
    Instruction *Curr = Worklist.pop_back_val();
    for (Use &OI : Curr->operands()) {
      if (Instruction *Inst = dyn_cast<Instruction>(OI))
        if (Alive.insert(Inst).second) {
          Worklist.push_back(Inst);
          if (SeenBBs.insert(Inst->getParent()).second) {
            Alive.insert(Inst->getParent()->getTerminator());
            Worklist.push_back(Inst->getParent()->getTerminator());
          }
        }
    }
  }

  // The inverse of the live set is the dead set.  These are those instructions
  // which have no side effects and do not influence the control flow or return
  // value of the function, and may therefore be deleted safely.
  // NOTE: We reuse the Worklist vector here for memory efficiency.
  for (Instruction &I : instructions(F)) {
    if (!Alive.count(&I)) {
      if (isa<TerminatorInst>(&I)) {
        if (BranchInst *BR = dyn_cast<BranchInst>(&I)) {
          // Replace the condition with undef, otherwise leave it alone
          if (BR->isConditional())
            BR->setCondition(UndefValue::get(BR->getCondition()->getType()));
          continue;
        } else {
          new UnreachableInst(F.getContext(), I.getParent()->getTerminator());
        }
      }

      Worklist.push_back(&I);
      I.dropAllReferences();
    }
  }

  for (Instruction *&I : Worklist) {
    ++NumRemoved;
    I->eraseFromParent();
  }
  return !Worklist.empty();
}

FunctionPass *llvm::createAggressiveDCEPass() { return new ADCE(); }
