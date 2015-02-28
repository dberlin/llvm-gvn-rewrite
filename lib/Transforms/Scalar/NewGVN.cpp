//===- GVN.cpp - Eliminate redundant values and loads ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs global value numbering to eliminate fully redundant
// instructions.  It also performs simple dead load elimination.
//
// Note that this pass does the value numbering itself; it does not use the
// ValueNumbering analysis passes.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Analysis/PHITransAddr.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Transforms/Scalar/GVNExpression.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/MemorySSA.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include <vector>
using namespace llvm;
using namespace PatternMatch;
using namespace llvm::GVNExpression;

#define DEBUG_TYPE "newgvn"

STATISTIC(NumGVNInstrDeleted, "Number of instructions deleted");
STATISTIC(NumGVNBlocksDeleted, "Number of blocks deleted");
// STATISTIC(NumNewGVNPRE, "Number of instructions PRE'd");
// STATISTIC(NumNewGVNBlocks, "Number of blocks merged");
// STATISTIC(NumNewGVNSimpl, "Number of instructions simplified");
STATISTIC(NumGVNEqProp, "Number of equalities propagated");
// STATISTIC(NumPRELoad, "Number of loads PRE'd");
STATISTIC(NumGVNOpsSimplified, "Number of Expressions simplified");
STATISTIC(NumGVNPhisAllSame, "Number of PHIs whos arguments are all the same");

//===----------------------------------------------------------------------===//
//                                GVN Pass
//===----------------------------------------------------------------------===//

namespace {

// Congruence classes represent the set of expressions/instructions
// that are all the same *during some scope in the function*.
// It is very slightly flow-sensitive.
// That is, because of the way we perform equality propagation, and
// because of memory value numbering,  it is not correct to assume
// you can willy-nilly replace any member with any other at any
// point in the function.
// For any tuple (Value, BB) in the members set, it is correct to
// assume the congruence class is represented by Value in all blocks
// dominated by BB

// Every congruence class has a leader, and the leader is used to
// symbolize instructions in a canonical way (IE every operand of an
// instruction that is a member of the same congruence class will
// always be replaced with leader during symbolization).
// Each congruence class also has a defining expression,
// though the expression may be null.
// FIXME: It's not clear what use the defining expression is if you
// just use constants/etc as leaders

struct CongruenceClass {
  typedef SmallPtrSet<Value *, 4> MemberSet;
  static uint32_t nextCongruenceNum;
  uint32_t id;
  Value *leader;
  // TODO: Do we actually need this? It's not clear what purpose it
  // really serves
  Expression *expression;
  // Actual members of this class.  These are the things the same everywhere
  MemberSet members;
  typedef DenseSet<std::pair<Value *, BasicBlock *>> EquivalenceSet;

  // Noted equivalences.  These are things that are equivalence to
  // this class over certain paths.  This could be replaced with
  // proper predicate support during analysis.
  EquivalenceSet equivalences;
  bool dead;
  explicit CongruenceClass()
      : id(nextCongruenceNum++), leader(0), expression(0), dead(false) {}
  CongruenceClass(Value *Leader, Expression *E)
      : id(nextCongruenceNum++), leader(Leader), expression(E), dead(false) {}
};
uint32_t CongruenceClass::nextCongruenceNum = 0;

class NewGVN : public FunctionPass {
  MemoryDependenceAnalysis *MD;
  DominatorTree *DT;
  const DataLayout *DL;
  const TargetLibraryInfo *TLI;
  AssumptionCache *AC;
  AliasAnalysis *AA;
  MemorySSA *MSSA;
  BumpPtrAllocator ExpressionAllocator;

  // Congruence class info
  DenseMap<Value *, CongruenceClass *> ValueToClass;
  DenseMap<Value *, Expression *> ValueToExpression;
  struct ComparingExpressionInfo {
    static inline Expression *getEmptyKey() {
      intptr_t Val = -1;
      Val <<= PointerLikeTypeTraits<Expression *>::NumLowBitsAvailable;
      return reinterpret_cast<Expression *>(Val);
    }
    static inline Expression *getTombstoneKey() {
      intptr_t Val = -2;
      Val <<= PointerLikeTypeTraits<Expression *>::NumLowBitsAvailable;
      return reinterpret_cast<Expression *>(Val);
    }
    static unsigned getHashValue(const Expression *V) {
      return static_cast<unsigned>(V->getHashValue());
    }
    static bool isEqual(Expression *LHS, const Expression *RHS) {
      if (LHS == RHS)
        return true;
      if (LHS == getTombstoneKey() || RHS == getTombstoneKey() ||
          LHS == getEmptyKey() || RHS == getEmptyKey())
        return false;
      return *LHS == *RHS;
    }
  };

  typedef DenseMap<Expression *, CongruenceClass *, ComparingExpressionInfo>
      ExpressionClassMap;

  ExpressionClassMap ExpressionToClass;

  // We separate out the memory expressions to keep hashtable resizes from
  // occurring as often.
  ExpressionClassMap MemoryExpressionToClass;
  DenseSet<Expression *, ComparingExpressionInfo> UniquedExpressions;
  DenseSet<Value *> ChangedValues;
  DenseSet<std::pair<BasicBlock *, BasicBlock *>> ReachableEdges;
  DenseSet<BasicBlock *> ReachableBlocks;
  DenseSet<Instruction *> TouchedInstructions;
  DenseMap<Instruction *, uint32_t> ProcessedCount;
  CongruenceClass *InitialClass;
  std::vector<CongruenceClass *> CongruenceClasses;
  DenseMap<BasicBlock *, std::pair<int, int>> DFSBBMap;
  DenseMap<Instruction *, unsigned int> InstrLocalDFS;
  std::vector<Instruction *> InstructionsToErase;

public:
  static char ID; // Pass identification, replacement for typeid
  explicit NewGVN() : FunctionPass(ID), MD(nullptr) {
    initializeNewGVNPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

  const DataLayout *getDataLayout() const { return DL; }
  DominatorTree *getDominatorTree() const { return DT; }
  AliasAnalysis *getAliasAnalysis() const { return AA; }
  MemoryDependenceAnalysis *getMemDep() const { return MD; }

private:
  // This transformation requires dominator postdominator info
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<MemoryDependenceAnalysis>();
    AU.addRequired<MemorySSA>();
    AU.addRequired<AliasAnalysis>();

    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<AliasAnalysis>();
  }

  // expression handling
  Expression *createExpression(Instruction *, BasicBlock *);
  void setBasicExpressionInfo(Instruction *, BasicExpression *, BasicBlock *);
  Expression *createPHIExpression(Instruction *, BasicBlock *);
  Expression *createVariableExpression(Value *);
  Expression *createConstantExpression(Constant *);
  Expression *createStoreExpression(StoreInst *, MemoryAccess *, BasicBlock *);
  Expression *createLoadExpression(LoadInst *, MemoryAccess *, BasicBlock *);
  Expression *createCallExpression(CallInst *, MemoryAccess *, BasicBlock *);
  Expression *createInsertValueExpression(InsertValueInst *, BasicBlock *l);
  Expression *uniquifyExpression(Expression *);
  Expression *createCmpExpression(unsigned, Type *, CmpInst::Predicate, Value *,
                                  Value *, BasicBlock *);
  // Congruence class handling
  CongruenceClass *createCongruenceClass(Value *Leader, Expression *E) {
    CongruenceClass *result = new CongruenceClass(Leader, E);
    CongruenceClasses.push_back(result);
    return result;
  }

  CongruenceClass *createSingletonCongruenceClass(Value *Member,
                                                  BasicBlock *B) {
    CongruenceClass *CClass = createCongruenceClass(Member, NULL);
    CClass->members.insert(Member);
    ValueToClass[Member] = CClass;
    return CClass;
  }
  void initializeCongruenceClasses(Function &F);

  // Symbolic evaluation
  Expression *checkSimplificationResults(Expression *, Instruction *, Value *);
  Expression *performSymbolicEvaluation(Value *, BasicBlock *);
  Expression *performSymbolicLoadEvaluation(Instruction *, BasicBlock *);
  Expression *performSymbolicStoreEvaluation(Instruction *, BasicBlock *);
  Expression *performSymbolicCallEvaluation(Instruction *, BasicBlock *);
  Expression *performSymbolicPHIEvaluation(Instruction *, BasicBlock *);
  // Congruence finding
  Value *lookupOperandLeader(Value *, BasicBlock *);
  Value *findDominatingEquivalent(CongruenceClass *, BasicBlock *);
  void performCongruenceFinding(Value *, BasicBlock *, Expression *);
  // Predicate and reachability handling
  void updateReachableEdge(BasicBlock *, BasicBlock *);
  void processOutgoingEdges(TerminatorInst *);
  void propagateChangeInEdge(BasicBlock *);
  bool propagateEquality(Value *, Value *, BasicBlock *);
  // Instruction replacement
  unsigned replaceAllDominatedUsesWith(Value *, Value *, BasicBlock *);

  // Elimination
  struct ValueDFS;
  void convertDenseToDFSOrdered(CongruenceClass::MemberSet &,
                                std::set<ValueDFS> &);
  void convertDenseToDFSOrdered(CongruenceClass::EquivalenceSet &,
                                std::set<ValueDFS> &);

  bool eliminateInstructions(Function &);
  void replaceInstruction(Instruction *, Value *);
  void markInstructionForDeletion(Instruction *);
  void deleteInstructionsInBlock(BasicBlock *);

  // New instruction creation
  void handleNewInstruction(Instruction *);
  void markUsersTouched(Value *);
  // Utilities
  void cleanupTables();
};

char NewGVN::ID = 0;
}

// createGVNPass - The public interface to this file...
FunctionPass *llvm::createNewGVNPass() { return new NewGVN(); }

static std::string getBlockName(BasicBlock *B) {
  return DOTGraphTraits<const Function *>::getSimpleNodeLabel(B, NULL);
}

INITIALIZE_PASS_BEGIN(NewGVN, "newgvn", "Global Value Numbering", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(MemoryDependenceAnalysis)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(NewGVN, "newgvn", "Global Value Numbering", false, false)
Expression *NewGVN::createPHIExpression(Instruction *I, BasicBlock *B) {
  PHINode *PN = cast<PHINode>(I);
  PHIExpression *E = new (ExpressionAllocator) PHIExpression(I->getParent());
  E->setType(I->getType());
  E->setOpcode(I->getOpcode());

  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
    BasicBlock *B = PN->getIncomingBlock(i);
    if (!ReachableBlocks.count(B)) {
      DEBUG(dbgs() << "Skipping unreachable block " << getBlockName(B)
                   << " in PHI node " << *PN << "\n");
      continue;
    }
    if (I->getOperand(i) != I) {
      Value *Operand = lookupOperandLeader(I->getOperand(i), B);
      E->VarArgs.push_back(Operand);
    } else {
      E->VarArgs.push_back(I->getOperand(i));
    }
  }
  return E;
}

void NewGVN::setBasicExpressionInfo(Instruction *I, BasicExpression *E,
                                    BasicBlock *B) {
  E->setType(I->getType());
  E->setOpcode(I->getOpcode());

  for (auto &O : I->operands()) {
    Value *Operand = lookupOperandLeader(O, B);
    E->VarArgs.push_back(Operand);
  }
}
// This is a special function only used by equality propagation, it
// should not be called elsewhere
Expression *NewGVN::createCmpExpression(unsigned Opcode, Type *Type,
                                        CmpInst::Predicate Predicate,
                                        Value *LHS, Value *RHS, BasicBlock *B) {
  BasicExpression *E = new (ExpressionAllocator) BasicExpression();
  E->setType(Type);
  E->setOpcode((Opcode << 8) | Predicate);
  LHS = lookupOperandLeader(LHS, B);
  E->VarArgs.push_back(LHS);
  RHS = lookupOperandLeader(RHS, B);
  E->VarArgs.push_back(RHS);
  return E;
}

// Take a Value returned by simplification of Expression E/Instruction
// I, and see if it resulted in a simpler expression. If so, return
// that expression
Expression *NewGVN::checkSimplificationResults(Expression *E, Instruction *I,
                                               Value *V) {
  if (!V)
    return NULL;

  Constant *C;
  if ((C = dyn_cast<Constant>(V))) {
    DEBUG(dbgs() << "Simplified " << *I << " to "
                 << " constant " << *C << "\n");
    NumGVNOpsSimplified++;
    //    delete E;
    return createConstantExpression(C);
  }
  CongruenceClass *CC = ValueToClass.lookup(V);
  if (CC && CC->expression) {
    DEBUG(dbgs() << "Simplified " << *I << " to "
                 << " expression " << *V << "\n");
    NumGVNOpsSimplified++;
    //    delete E;
    return CC->expression;
  }
  return NULL;
}

Expression *NewGVN::createExpression(Instruction *I, BasicBlock *B) {
  BasicExpression *E = new (ExpressionAllocator) BasicExpression();
  setBasicExpressionInfo(I, E, B);

  if (I->isCommutative()) {
    // Ensure that commutative instructions that only differ by a permutation
    // of their operands get the same value number by sorting the operand value
    // numbers.  Since all commutative instructions have two operands it is more
    // efficient to sort by hand rather than using, say, std::sort.
    assert(I->getNumOperands() == 2 && "Unsupported commutative instruction!");
    if (E->VarArgs[0] > E->VarArgs[1])
      std::swap(E->VarArgs[0], E->VarArgs[1]);
  }

  // Perform simplificaiton
  // TODO: Right now we only check to see if we get a constant result.
  // We may get a less than constant, but still better, result for
  // some operations.
  // IE
  //  add 0, x -> x
  //  and x, x -> x
  // We should handle this by simply rewriting the expression.
  if (CmpInst *CI = dyn_cast<CmpInst>(I)) {
    // Sort the operand value numbers so x<y and y>x get the same value number.
    CmpInst::Predicate Predicate = CI->getPredicate();
    if (E->VarArgs[0] > E->VarArgs[1]) {
      std::swap(E->VarArgs[0], E->VarArgs[1]);
      Predicate = CmpInst::getSwappedPredicate(Predicate);
    }
    E->setOpcode((CI->getOpcode() << 8) | Predicate);
    // TODO: 10% of our time is spent in SimplifyCmpInst with pointer operands
    // TODO: Since we noop bitcasts, we may need to check types before
    // simplifying, so that we don't end up simplifying based on a wrong
    // type assumption. We should clean this up so we can use constants of the
    // wrong type

    assert(I->getOperand(0)->getType() == I->getOperand(1)->getType() &&
           "Wrong types on cmp instruction");
    if ((E->VarArgs[0]->getType() == I->getOperand(0)->getType() &&
         E->VarArgs[1]->getType() == I->getOperand(1)->getType())) {
      Value *V =
          SimplifyCmpInst(Predicate, E->VarArgs[0], E->VarArgs[1], DL, TLI, DT);
      Expression *simplifiedE;
      if ((simplifiedE = checkSimplificationResults(E, I, V)))
        return simplifiedE;
    }

  } else if (isa<SelectInst>(I)) {
    if (isa<Constant>(E->VarArgs[0]) ||
        (E->VarArgs[1]->getType() == I->getOperand(1)->getType() &&
         E->VarArgs[2]->getType() == I->getOperand(2)->getType())) {
      Value *V = SimplifySelectInst(E->VarArgs[0], E->VarArgs[1], E->VarArgs[2],
                                    DL, TLI, DT);
      Expression *simplifiedE;
      if ((simplifiedE = checkSimplificationResults(E, I, V)))
        return simplifiedE;
    }
  } else if (I->isBinaryOp()) {
    // TODO: Since we noop bitcasts, we may need to check types before
    // simplifying, so that we don't end up simplifying based on a
    // wrong type assumption
    Value *V = SimplifyBinOp(E->getOpcode(), E->VarArgs[0], E->VarArgs[1], DL,
                             TLI, DT);
    Expression *simplifiedE;
    if ((simplifiedE = checkSimplificationResults(E, I, V)))
      return simplifiedE;
  } else if (BitCastInst *BI = dyn_cast<BitCastInst>(I)) {
    Value *V = SimplifyInstruction(BI);
    Expression *simplifiedE;
    if ((simplifiedE = checkSimplificationResults(E, I, V)))
      return simplifiedE;
  } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I)) {
    // TODO: Since we noop bitcasts, we may need to check types before
    // simplifying, so that we don't end up simplifying based on a
    // wrong type assumption. We should clean this up so we can use
    // constants of the wrong type.
    if (GEP->getPointerOperandType() == E->VarArgs[0]->getType()) {
      Value *V = SimplifyGEPInst(E->VarArgs, DL, TLI, DT);
      Expression *simplifiedE;
      if ((simplifiedE = checkSimplificationResults(E, I, V)))
        return simplifiedE;
    }
  }

  return E;
}

Expression *NewGVN::uniquifyExpression(Expression *E) {
  auto P = UniquedExpressions.insert(E);
  if (!P.second) {
    return *(P.first);
  }
  return E;
}

Expression *NewGVN::createInsertValueExpression(InsertValueInst *I,
                                                BasicBlock *B) {
  InsertValueExpression *E = new (ExpressionAllocator) InsertValueExpression();
  setBasicExpressionInfo(I, E, B);
  for (auto II = I->idx_begin(), IE = I->idx_end(); II != IE; ++II)
    E->intargs.push_back(*II);
  return E;
}

Expression *NewGVN::createVariableExpression(Value *V) {
  VariableExpression *E = new (ExpressionAllocator) VariableExpression(V);
  E->setOpcode(V->getValueID());
  return E;
}

Expression *NewGVN::createConstantExpression(Constant *C) {
  ConstantExpression *E = new (ExpressionAllocator) ConstantExpression(C);
  E->setOpcode(C->getValueID());
  return E;
}

Expression *NewGVN::createCallExpression(CallInst *CI, MemoryAccess *HV,
                                         BasicBlock *B) {
  CallExpression *E = new (ExpressionAllocator) CallExpression(CI, HV);
  setBasicExpressionInfo(CI, E, B);
  return E;
}

// Find an equivalence in congruence class CC that dominates block B,
// if one exists
// TODO: Compare this against the predicate handling system in the paper

Value *NewGVN::findDominatingEquivalent(CongruenceClass *CC, BasicBlock *B) {
  // This check is much faster than doing 0 iterations of the loop below
  if (CC->equivalences.empty())
    return nullptr;

  // TODO: This can be made faster by different set ordering, if
  // necessary, or caching whether we found one
  for (auto &Member : CC->equivalences) {
    if (DT->dominates(Member.second, B))
      return Member.first;
  }
  return nullptr;
}

// lookupOperandLeader -- See if we have a congruence class and leader
// for this operand, and if so, return it. Otherwise, return the
// original operand
Value *NewGVN::lookupOperandLeader(Value *V, BasicBlock *B) {
  auto VTCI = ValueToClass.find(V);
  if (VTCI != ValueToClass.end()) {
    CongruenceClass *CC = VTCI->second;
    if (CC != InitialClass) {
      auto Equivalence = findDominatingEquivalent(CC, B);
      if (Equivalence) {
        DEBUG(dbgs() << "Found equivalence " << *Equivalence << " for value "
                     << *V << " in block " << getBlockName(B) << "\n");
        return Equivalence;
      }

      return CC->leader;
    }
  }
  return V;
}

Expression *NewGVN::createLoadExpression(LoadInst *LI, MemoryAccess *HV,
                                         BasicBlock *B) {
  LoadExpression *E = new (ExpressionAllocator) LoadExpression(LI, HV);
  E->setType(LI->getType());
  // Need opcodes to match on loads and store
  E->setOpcode(0);
  Value *Operand = lookupOperandLeader(LI->getPointerOperand(), B);
  E->VarArgs.push_back(Operand);
  // TODO: Value number heap versions. We may be able to discover
  // things alias analysis can't on it's own (IE that a store and a
  // load have the same value, and thus, it isn't clobbering the load)
  return E;
}

Expression *NewGVN::createStoreExpression(StoreInst *SI, MemoryAccess *HV,
                                          BasicBlock *B) {
  StoreExpression *E = new (ExpressionAllocator) StoreExpression(SI, HV);
  E->setType(SI->getType());
  // Need opcodes to match on loads and store
  E->setOpcode(0);
  Value *Operand = lookupOperandLeader(SI->getPointerOperand(), B);
  E->VarArgs.push_back(Operand);
  // TODO: Value number heap versions. We may be able to discover
  // things alias analysis can't on it's own (IE that a store and a
  // load have the same value, and thus, it isn't clobbering the load)
  return E;
}
Expression *NewGVN::performSymbolicStoreEvaluation(Instruction *I,
                                                   BasicBlock *B) {
  StoreInst *SI = cast<StoreInst>(I);
  Expression *E = createStoreExpression(SI, MSSA->getMemoryAccess(SI), B);
  return E;
}

Expression *NewGVN::performSymbolicLoadEvaluation(Instruction *I,
                                                  BasicBlock *B) {
  LoadInst *LI = cast<LoadInst>(I);
  if (!LI->isSimple())
    return NULL;
  MemoryAccess *HeapVersion = MSSA->getClobberingMemoryAccess(I);
  Expression *E = createLoadExpression(LI, HeapVersion, B);
  return E;
}

/// performSymbolicCallEvaluation - Evaluate read only and pure calls, and
/// create an expression result
Expression *NewGVN::performSymbolicCallEvaluation(Instruction *I,
                                                  BasicBlock *B) {
  CallInst *CI = cast<CallInst>(I);
  if (AA->doesNotAccessMemory(CI))
    return createCallExpression(CI, nullptr, B);
  else if (AA->onlyReadsMemory(CI))
    return createCallExpression(CI, MSSA->getClobberingMemoryAccess(CI), B);
  else
    return nullptr;
}

// performSymbolicPHIEvaluation - Evaluate PHI nodes symbolically, and
// create an expression result
Expression *NewGVN::performSymbolicPHIEvaluation(Instruction *I,
                                                 BasicBlock *B) {
  PHIExpression *E = cast<PHIExpression>(createPHIExpression(I, B));
  E->setOpcode(I->getOpcode());
  if (E->VarArgs.empty()) {
    DEBUG(dbgs() << "Simplified PHI node " << I << " to undef"
                 << "\n");
    //    delete E;
    return createVariableExpression(UndefValue::get(I->getType()));
  }

  Value *AllSameValue = E->VarArgs[0];

  for (unsigned i = 1, e = E->VarArgs.size(); i != e; ++i) {
    if (E->VarArgs[i] != AllSameValue) {
      AllSameValue = NULL;
      break;
    }
  }
  //
  if (AllSameValue) {
    // It's possible to have phi nodes with cycles (IE dependent on
    // other phis that are .... dependent on the original phi node), especially
    // in weird CFG's where some arguments are unreachable, or
    // uninitialized along certain paths.
    // This can cause infinite loops  during evaluation (even if you disable the
    // recursion below, you will simply ping-pong between congruence classes)
    // If a phi node symbolically evaluates to another phi node, just
    // leave it alone
    // If they are really the same, we will still eliminate them in
    // favor of each other.
    if (isa<PHINode>(AllSameValue))
      return E;
    NumGVNPhisAllSame++;
    DEBUG(dbgs() << "Simplified PHI node " << I << " to " << *AllSameValue
                 << "\n");

    //    delete E;
    return performSymbolicEvaluation(AllSameValue, B);
  }
  return E;
}

/// performSymbolicEvaluation - Substitute and symbolize the value
/// before value numbering
Expression *NewGVN::performSymbolicEvaluation(Value *V, BasicBlock *B) {
  Expression *E = NULL;
  if (Constant *C = dyn_cast<Constant>(V))
    E = createConstantExpression(C);
  else if (isa<Argument>(V) || isa<GlobalVariable>(V)) {
    E = createVariableExpression(V);
  } else {
    // TODO: memory intrinsics
    Instruction *I = cast<Instruction>(V);
    switch (I->getOpcode()) {
    // TODO: extractvalue
    case Instruction::InsertValue:
      E = createInsertValueExpression(cast<InsertValueInst>(I), B);
      break;
    case Instruction::PHI:
      E = performSymbolicPHIEvaluation(I, B);
      break;
    case Instruction::Call:
      E = performSymbolicCallEvaluation(I, B);
      break;
    case Instruction::Store:
      E = performSymbolicStoreEvaluation(I, B);
      break;
    case Instruction::Load:
      E = performSymbolicLoadEvaluation(I, B);
      break;
    case Instruction::BitCast: {
      // // Pointer bitcasts are noops, we can just make them out of whole
      // // cloth if we need to.
      // if (I->getType()->isPointerTy()) {
      //   if (Instruction *I0 = dyn_cast<Instruction>(I->getOperand(0)))
      //     return performSymbolicEvaluation(I0, I0->getParent());
      //   else
      //     return performSymbolicEvaluation(I->getOperand(0), B);
      // }
      E = createExpression(I, B);
    } break;

    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
    case Instruction::ICmp:
    case Instruction::FCmp:
    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::Select:
    case Instruction::ExtractElement:
    case Instruction::InsertElement:
    case Instruction::ShuffleVector:
    case Instruction::GetElementPtr:
      E = createExpression(I, B);
      break;
    default:
      return NULL;
    }
  }
  if (!E)
    return NULL;

  if (isa<ConstantExpression>(E) || isa<VariableExpression>(E))
    E = uniquifyExpression(E);
  return E;
}
/// replaceAllDominatedUsesWith - Replace all uses of 'From' with 'To'
/// if the use is dominated by the given basic block.  Returns the
/// number of uses that were replaced.
unsigned NewGVN::replaceAllDominatedUsesWith(Value *From, Value *To,
                                             BasicBlock *Root) {
  unsigned Count = 0;
  for (auto UI = From->use_begin(), UE = From->use_end(); UI != UE;) {
    Use &U = *UI++;

    // If From occurs as a phi node operand then the use implicitly lives in the
    // corresponding incoming block.  Otherwise it is the block containing the
    // user that must be dominated by Root.
    BasicBlock *UsingBlock;
    if (PHINode *PN = dyn_cast<PHINode>(U.getUser()))
      UsingBlock = PN->getIncomingBlock(U);
    else
      UsingBlock = cast<Instruction>(U.getUser())->getParent();

    if (DT->dominates(Root, UsingBlock)) {
      // Mark the users as touched
      if (Instruction *I = dyn_cast<Instruction>(U.getUser()))
        TouchedInstructions.insert(I);
      DEBUG(dbgs() << "Equality propagation replacing " << *From << " with "
                   << *To << " in " << *(U.getUser()) << "\n");
      U.set(To);
      ++Count;
    }
  }
  return Count;
}

/// propagateEquality - The given values are known to be equal in
/// every block dominated by 'Root'.  Exploit this, for example by
/// replacing 'LHS' with 'RHS' everywhere in the scope.  Returns
/// whether a change was made.
// FIXME: This is part of the old GVN algorithm that we have
// temporarily kept.  It should be replaced with proper predicate
// support.
// Right now it requires touching and replacing instructions in the
// middle of analysis.  Worse, it requires elimination on every
// iteration in order to maximize the equalities found, because
// the elimination it performs is fairly simple, and it expects
// something to go looking for leaders and eliminating on every
// iteration.

bool NewGVN::propagateEquality(Value *LHS, Value *RHS, BasicBlock *Root) {
  SmallVector<std::pair<Value *, Value *>, 4> Worklist;
  Worklist.push_back(std::make_pair(LHS, RHS));
  bool Changed = false;

  while (!Worklist.empty()) {
    std::pair<Value *, Value *> Item = Worklist.pop_back_val();
    LHS = Item.first;
    RHS = Item.second;

    if (LHS == RHS)
      continue;
    assert(LHS->getType() == RHS->getType() && "Equality but unequal types!");

    // Don't try to propagate equalities between constants.
    if (isa<Constant>(LHS) && isa<Constant>(RHS))
      continue;

    // Prefer a constant on the right-hand side, or an Argument if no constants.
    if (isa<Constant>(LHS) || (isa<Argument>(LHS) && !isa<Constant>(RHS)))
      std::swap(LHS, RHS);
    assert((isa<Argument>(LHS) || isa<Instruction>(LHS)) &&
           "Unexpected value!");
    assert((!isa<Instruction>(RHS) ||
            DT->properlyDominates(cast<Instruction>(RHS)->getParent(), Root)) &&
           "Instruction doesn't dominate scope!");

    // If value numbering later deduces that an instruction in the
    // scope is equal to 'LHS' then ensure it will be turned into
    // 'RHS' within the scope Root.
    CongruenceClass *CC = ValueToClass[LHS];
    assert(CC && "Should have found a congruence class");
    CC->equivalences.insert(std::make_pair(RHS, Root));

    // Replace all occurrences of 'LHS' with 'RHS' everywhere in the
    // scope.  As LHS always has at least one use that is not
    // dominated by Root, this will never do anything if LHS has only
    // one use.
    if (!LHS->hasOneUse()) {
      unsigned NumReplacements = replaceAllDominatedUsesWith(LHS, RHS, Root);
      Changed |= NumReplacements > 0;
      NumGVNEqProp += NumReplacements;
    }

    // Now try to deduce additional equalities from this one.  For
    // example, if the known equality was "(A != B)" == "false" then
    // it follows that A and B are equal in the scope.  Only boolean
    // equalities with an explicit true or false RHS are currently
    // supported.
    if (!RHS->getType()->isIntegerTy(1))
      // Not a boolean equality - bail out.
      continue;
    ConstantInt *CI = dyn_cast<ConstantInt>(RHS);
    if (!CI)
      // RHS neither 'true' nor 'false' - bail out.
      continue;
    // Whether RHS equals 'true'.  Otherwise it equals 'false'.
    bool isKnownTrue = CI->isAllOnesValue();
    bool isKnownFalse = !isKnownTrue;

    // If "A && B" is known true then both A and B are known true.  If
    // "A || B" is known false then both A and B are known false.
    Value *A, *B;
    if ((isKnownTrue && match(LHS, m_And(m_Value(A), m_Value(B)))) ||
        (isKnownFalse && match(LHS, m_Or(m_Value(A), m_Value(B))))) {
      Worklist.push_back(std::make_pair(A, RHS));
      Worklist.push_back(std::make_pair(B, RHS));
      continue;
    }
    // If we are propagating an equality like "(A == B)" == "true"
    // then also propagate the equality A == B.  When propagating a
    // comparison such as "(A >= B)" == "true", replace all instances
    // of "A < B" with "false".
    if (ICmpInst *Cmp = dyn_cast<ICmpInst>(LHS)) {
      Value *Op0 = Cmp->getOperand(0), *Op1 = Cmp->getOperand(1);

      // If "A == B" is known true, or "A != B" is known false, then replace
      // A with B everywhere in the scope.
      if ((isKnownTrue && Cmp->getPredicate() == CmpInst::ICMP_EQ) ||
          (isKnownFalse && Cmp->getPredicate() == CmpInst::ICMP_NE))
        Worklist.push_back(std::make_pair(Op0, Op1));

      // If "A >= B" is known true, replace "A < B" with false
      // everywhere.
      // Since we don't have the instruction "A < B" immediately to
      // hand, work out the value number that it would have and use
      // that to find an appropriate instruction (if any).
      CmpInst::Predicate NotPred = Cmp->getInversePredicate();
      Constant *NotVal = ConstantInt::get(Cmp->getType(), isKnownFalse);
      Expression *E = createCmpExpression(Cmp->getOpcode(), Cmp->getType(),
                                          NotPred, Op0, Op1, Cmp->getParent());

      // We cannot directly propagate into the congruence class members
      // unless it is a singleton class.
      // This is because at least one of the members may later get
      // split out of the class.

      CongruenceClass *CC = ExpressionToClass.lookup(E);
      //      delete E;
      // If we didn't find a congruence class, there is no equivalent
      // instruction already
      if (CC) {
        if (CC->members.size() == 1) {
          unsigned NumReplacements =
              replaceAllDominatedUsesWith(CC->leader, NotVal, Root);
          Changed |= NumReplacements > 0;
          NumGVNEqProp += NumReplacements;
        }
        // Ensure that any instruction in scope that gets the "A < B"
        // value number is replaced with false.

        CC->equivalences.insert(std::make_pair(NotVal, Root));
      }
      // TODO: Equality propagation - do equivalent of this
      // Ensure that any instruction in scope that gets the "A < B" value number
      // is replaced with false.
      // The leader table only tracks basic blocks, not edges. Only add to if we
      // have the simple case where the edge dominates the end.
      // if (RootDominatesEnd)
      //   addToLeaderTable(Num, NotVal, Root.getEnd());
      continue;
    }
  }

  return Changed;
}

/// isOnlyReachableViaThisEdge - There is an edge from 'Src' to 'Dst'.  Return
/// true if every path from the entry block to 'Dst' passes via this edge.  In
/// particular 'Dst' must not be reachable via another edge from 'Src'.
static bool isOnlyReachableViaThisEdge(BasicBlock *Src, BasicBlock *Dst,
                                       DominatorTree *DT) {
  // While in theory it is interesting to consider the case in which Dst has
  // more than one predecessor, because Dst might be part of a loop which is
  // only reachable from Src, in practice it is pointless since at the time
  // GVN runs all such loops have preheaders, which means that Dst will have
  // been changed to have only one predecessor, namely Src.
  BasicBlock *Pred = Dst->getSinglePredecessor();
  assert((!Pred || Pred == Src) && "No edge between these basic blocks!");
  (void)Src;
  return Pred != 0;
}

void NewGVN::markUsersTouched(Value *V) {
  // Now mark the users as touched
  for (auto &U : V->uses()) {
    Instruction *User = dyn_cast<Instruction>(U.getUser());
    assert(User && "Use of value not within an instruction?");
    TouchedInstructions.insert(User);
  }
}

/// performCongruenceFinding - Perform congruence finding on a given
/// value numbering expression
void NewGVN::performCongruenceFinding(Value *V, BasicBlock *BB, Expression *E) {
  ValueToExpression[V] = E;

  // This is guaranteed to return something, since it will at least find INITIAL
  CongruenceClass *VClass = ValueToClass[V];
  assert(VClass && "Should have found a vclass");
  // Dead classes should have been eliminated from the mapping
  assert(!VClass->dead && "Found a dead class");

  // TODO(dannyb): Double check algorithm where we are ignoring copy
  // check of "if e is a SSA variable", as LLVM has no copies.

  CongruenceClass *EClass;
  // Expressions we can't symbolize are always in their own unique
  // congruence class
  if (E == NULL) {
    // We may have already made a unique class
    if (VClass->members.size() != 1 || VClass->leader != V) {
      CongruenceClass *NewClass = createCongruenceClass(V, NULL);
      // We should always be adding the member in the below code
      EClass = NewClass;
      DEBUG(dbgs() << "Created new congruence class for " << *V
                   << " due to NULL expression\n");
    } else {
      EClass = VClass;
    }
  } else {

    ExpressionClassMap *lookupMap = &ExpressionToClass;
    if (isa<StoreExpression>(E) || isa<LoadExpression>(E))
      lookupMap = &MemoryExpressionToClass;

    auto lookupResult =
        lookupMap->insert(std::make_pair(E, (CongruenceClass *)NULL));

    // If it's not in the value table, create a new congruence class
    if (lookupResult.second) {
      CongruenceClass *NewClass = createCongruenceClass(NULL, E);
      auto place = lookupResult.first;
      place->second = NewClass;

      // Constants and variables should always be made the leader
      if (ConstantExpression *CE = dyn_cast<ConstantExpression>(E))
        NewClass->leader = CE->getConstantValue();
      else if (VariableExpression *VE = dyn_cast<VariableExpression>(E))
        NewClass->leader = VE->getVariableValue();
      else if (StoreExpression *SE = dyn_cast<StoreExpression>(E))
        NewClass->leader = SE->getStoreInst()->getValueOperand();
      else
        NewClass->leader = V;

      EClass = NewClass;
      DEBUG(dbgs() << "Created new congruence class for " << V
                   << " using expression " << *E << " at " << NewClass->id
                   << "\n");
    } else {
      EClass = lookupResult.first->second;
      assert(EClass && "Somehow don't have an eclass");

      assert(!EClass->dead && "We accidentally looked up a dead class");
    }
  }
  auto DI = ChangedValues.find(V);
  bool WasInChanged = DI != ChangedValues.end();
  if (VClass != EClass || WasInChanged) {
    DEBUG(dbgs() << "Found class " << EClass->id << " for expression " << E
                 << "\n");

    if (WasInChanged)
      ChangedValues.erase(DI);
    if (VClass != EClass) {
      DEBUG(dbgs() << "New congruence class for " << V << " is " << EClass->id
                   << "\n");
      VClass->members.erase(V);
      // assert(std::find(EClass->members.begin(), EClass->members.end(), V) ==
      // EClass->members.end() && "Tried to add something to members
      // twice!");
      EClass->members.insert(V);
      ValueToClass[V] = EClass;
      // See if we destroyed the class or need to swap leaders
      if (VClass->members.empty() && VClass != InitialClass) {
        if (VClass->expression) {
          VClass->dead = true;
          // TODO: I think this may be wrong.
          // I think we should be keeping track of the expression for
          // each instruction, not for each class, and erase the old
          // expression for the instruction when the class dies.
          ExpressionToClass.erase(VClass->expression);
          MemoryExpressionToClass.erase(VClass->expression);
        }
        // delete VClass;
      } else if (VClass->leader == V) {
        // TODO: Check what happens if expression represented the leader
        VClass->leader = *(VClass->members.begin());

        for (auto L : VClass->members) {
          if (Instruction *I = dyn_cast<Instruction>(L))
            TouchedInstructions.insert(I);
          ChangedValues.insert(L);
        }
      }
    }
    markUsersTouched(V);
  }
}

// updateReachableEdge - Process the fact that Edge (from, to) is
// reachable, including marking any newly reachable blocks and
// instructions for processing
void NewGVN::updateReachableEdge(BasicBlock *From, BasicBlock *To) {
  // Check if the Edge was reachable before
  if (ReachableEdges.insert(std::make_pair(From, To)).second) {
    // If this block wasn't reachable before, all instructions are touched
    if (ReachableBlocks.insert(To).second) {
      DEBUG(dbgs() << "Block " << getBlockName(To) << " marked reachable\n");
      for (auto &I : *To)
        TouchedInstructions.insert(&I);
    } else {
      DEBUG(dbgs() << "Block " << getBlockName(To)
                   << " was reachable, but new edge to it found\n");
      // We've made an edge reachable to an existing block, which may
      // impact predicates.
      // Otherwise, only mark the phi nodes as touched, as they are
      // the only thing that depend on new edges. Anything using their
      // values will get propagated to if necessary
      auto BI = To->begin();
      while (isa<PHINode>(BI)) {
        TouchedInstructions.insert(BI);
        ++BI;
      }
      // Propagate the change downstream.
      // TODO: I don't see how it's necessary to mark random
      // downstream instructions as suddenly touched.  They should get
      // propagated to in the normal course of business.  If we had
      // predicates, this would make sense, since the predicates do
      // not have use info, so you don't know which downstream
      // instructions were inferred based on them (though that could
      // be tracked)

      // propagateChangeInEdge(To);
    }
  }
}

//  processOutgoingEdges - Process the outgoing edges of a block for
//  reachability.
void NewGVN::processOutgoingEdges(TerminatorInst *TI) {
  // Evaluate Reachability of terminator instruction
  // Conditional branch
  BranchInst *BR;
  if ((BR = dyn_cast<BranchInst>(TI)) && BR->isConditional()) {
    Value *Cond = BR->getCondition();
    Value *CondEvaluated = NULL;
    if (Instruction *I = dyn_cast<Instruction>(Cond)) {
      Expression *E = createExpression(I, TI->getParent());
      if (ConstantExpression *CE = dyn_cast<ConstantExpression>(E)) {
        CondEvaluated = CE->getConstantValue();
      }
    } else if (isa<ConstantInt>(Cond)) {
      CondEvaluated = Cond;
    }
    ConstantInt *CI;
    BasicBlock *TrueSucc = BR->getSuccessor(0);
    BasicBlock *FalseSucc = BR->getSuccessor(1);
    if (CondEvaluated &&(CI = dyn_cast<ConstantInt>(CondEvaluated))) {
      if (CI->isOne()) {
        DEBUG(dbgs() << "Condition for Terminator " << *TI
                     << " evaluated to true\n");
        updateReachableEdge(TI->getParent(), TrueSucc);
      } else if (CI->isZero()) {
        DEBUG(dbgs() << "Condition for Terminator " << *TI
                     << " evaluated to false\n");
        updateReachableEdge(TI->getParent(), FalseSucc);
      }
    } else {
      BasicBlock *Parent = BR->getParent();
      if (isOnlyReachableViaThisEdge(Parent, TrueSucc, DT))
        propagateEquality(Cond, ConstantInt::getTrue(TrueSucc->getContext()),
                          TrueSucc);

      if (isOnlyReachableViaThisEdge(Parent, FalseSucc, DT))
        propagateEquality(Cond, ConstantInt::getFalse(FalseSucc->getContext()),
                          FalseSucc);
      updateReachableEdge(TI->getParent(), TrueSucc);
      updateReachableEdge(TI->getParent(), FalseSucc);
    }
  } else {
    // For switches, propagate the case values into the case destinations.
    if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
      Value *SwitchCond = SI->getCondition();
      BasicBlock *Parent = SI->getParent();
      for (auto i = SI->case_begin(), e = SI->case_end(); i != e; ++i) {
        BasicBlock *Dst = i.getCaseSuccessor();
        if (isOnlyReachableViaThisEdge(Parent, Dst, DT))
          propagateEquality(SwitchCond, i.getCaseValue(), Dst);
      }
    }

    for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i) {
      BasicBlock *B = TI->getSuccessor(i);
      updateReachableEdge(TI->getParent(), B);
    }
  }
}

/// propagateChangeInEdge - Propagate a change in edge reachability
// When we discover a new edge to an existing reachable block, that
// can affect the value of blocks containing phi nodes downstream.
// These blocks/instructions need to be marked for re-processing.
//
// However, it can *only* impact blocks that contain phi nodes, as
// those are the only values that would be carried from multiple
// incoming edges at once.
//
void NewGVN::propagateChangeInEdge(BasicBlock *Dest) {
  // The algorithm states that you only need to touch blocks that are
  // confluence nodes.  I also can't see why you would need to touch
  // any instructions that aren't PHI nodes.  Because we don't use
  // predicates right now, they are the ones whose value could have
  // changed as a result of a new edge becoming live, and any changes
  // to their value should propagate appropriately through the rest of
  // the block.
  DomTreeNode *DTN = DT->getNode(Dest);
  // TODO(dannyb): This differs slightly from the published algorithm,
  // verify it. The published algorithm touches all instructions, we
  // only touch the phi nodes.  This is because there should be no
  // other values that can *directly* change as a result of edge
  // reachability. If the phi node ends up changing congruence classes,
  // the users will be marked as touched anyway.  If we moved to using
  // value inferencing, there are cases we may need to touch more than
  // phi nodes.
  for (auto D : depth_first(DTN)) {
    BasicBlock *B = D->getBlock();
    if (!B->getUniquePredecessor()) {
      for (auto &I : *B) {
        if (!isa<PHINode>(&I))
          break;
        TouchedInstructions.insert(&I);
      }
    }
  }
}

void NewGVN::initializeCongruenceClasses(Function &F) {
  CongruenceClass::nextCongruenceNum = 2;
  // Initialize all other instructions to be in INITIAL class
  CongruenceClass::MemberSet InitialValues;
  for (auto &B : F) {
    for (auto &I : B) {
      InitialValues.insert(&I);
    }
  }

  InitialClass = createCongruenceClass(NULL, NULL);
  for (auto L : InitialValues)
    ValueToClass[L] = InitialClass;
  InitialClass->members.swap(InitialValues);

  // Initialize arguments to be in their own unique congruence classes
  // In an IPA-GVN, this would not be done
  for (auto &FA : F.args())
    createSingletonCongruenceClass(&FA, &F.getEntryBlock());
}

void NewGVN::cleanupTables() {

  ValueToClass.clear();
  for (unsigned i = 0, e = CongruenceClasses.size(); i != e; ++i) {
    DEBUG(dbgs() << "Congruence class " << i << " has "
                 << CongruenceClasses[i]->members.size() << " members\n");
    delete CongruenceClasses[i];

    CongruenceClasses[i] = NULL;
  }
  ExpressionAllocator.Reset();
  CongruenceClasses.clear();
  ExpressionToClass.clear();
  MemoryExpressionToClass.clear();
  UniquedExpressions.clear();
  ReachableBlocks.clear();
  ReachableEdges.clear();
  TouchedInstructions.clear();
  ProcessedCount.clear();
  DFSBBMap.clear();
  InstrLocalDFS.clear();
  ValueToExpression.clear();
  InstructionsToErase.clear();
}

/// runOnFunction - This is the main transformation entry point for a function.
bool NewGVN::runOnFunction(Function &F) {
  bool Changed = false;
  if (skipOptnoneFunction(F))
    return false;

  MD = &getAnalysis<MemoryDependenceAnalysis>();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  DataLayoutPass *DLP = getAnalysisIfAvailable<DataLayoutPass>();
  DL = DLP ? &DLP->getDataLayout() : nullptr;
  AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  TLI = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  AA = &getAnalysis<AliasAnalysis>();
  MSSA = &getAnalysis<MemorySSA>();

  uint32_t ICount = 0;

  // Count number of instructions for sizing of hash tables, and come
  // up with local dfs numbering for instructions
  for (auto &B : F) {
    ICount += F.size();
    for (auto &I : B) {
      InstrLocalDFS[&I] = ICount;
      ++ICount;
    }
  }
  // Ensure we don't end up resizing the expressionToClass map, as
  // that can be quite expensive. At most, we have one expression per
  // instruction.
  ExpressionToClass.resize(ICount * 2);
  MemoryExpressionToClass.resize(ICount + 1);
  // Initialize the touched instructions to include the entry block
  for (auto &I : F.getEntryBlock())
    TouchedInstructions.insert(&I);
  ReachableBlocks.insert(&F.getEntryBlock());

  initializeCongruenceClasses(F);

  while (!TouchedInstructions.empty()) {
    // TODO: Use two worklist method to keep ordering straight
    // TODO: or Investigate RPO numbering both blocks and instructions in the
    // same pass,
    //  and walking both lists at the same time, processing whichever has the
    //  next number in order.
    ReversePostOrderTraversal<Function *> rpoT(&F);
    for (auto &R : rpoT) {
      // TODO(Predication)
      bool blockReachable = ReachableBlocks.count(R);
      bool movedForward = false;
      for (auto BI = R->begin(), BE = R->end(); BI != BE;
           !movedForward ? BI++ : BI) {
        movedForward = false;
        auto DI = TouchedInstructions.find(BI);
        if (DI != TouchedInstructions.end()) {
          DEBUG(dbgs() << "Processing instruction " << *BI << "\n");
          TouchedInstructions.erase(DI);
          if (!blockReachable) {
            DEBUG(dbgs() << "Skipping instruction " << *BI << " because block "
                         << getBlockName(R) << " is unreachable\n");
            continue;
          }
          // This is done in case something eliminates the instruction
          // along the way.
          Instruction *I = BI++;
          movedForward = true;

          if (ProcessedCount.count(I) == 0) {
            ProcessedCount.insert(std::make_pair(I, 1));
          } else {
            ProcessedCount[I] += 1;
            assert(ProcessedCount[I] < 100 &&
                   "Seem to have processed the same instruction a lot");
          }

          if (!I->isTerminator()) {
            Expression *Symbolized = performSymbolicEvaluation(I, R);
            performCongruenceFinding(I, R, Symbolized);
          } else {
            processOutgoingEdges(dyn_cast<TerminatorInst>(I));
          }
        }
      }
    }
  }
  Changed |= eliminateInstructions(F);
  // Delete all instructions marked for deletion.
  for (unsigned i = 0, e = InstructionsToErase.size(); i != e; ++i) {
    Instruction *toErase = InstructionsToErase[i];
    if (!toErase->use_empty())
      toErase->replaceAllUsesWith(UndefValue::get(toErase->getType()));

    toErase->eraseFromParent();
    InstructionsToErase[i] = nullptr;
  }

  // Delete all unreachable blocks
  for (auto &B : F) {
    BasicBlock *BB = &B;
    if (!ReachableBlocks.count(BB)) {
      DEBUG(dbgs() << "We believe block " << getBlockName(BB)
                   << " is unreachable\n");
      deleteInstructionsInBlock(BB);
      Changed = true;
    }
  }

  cleanupTables();
  return Changed;
}
// Return true if V is a value that will always be available (IE can
// be placed anywhere) in the function.  We don't do globals here
// because they are often worse to put in place
// TODO: Separate cost from availability

static bool alwaysAvailable(Value *V) {
  return isa<Constant>(V) || isa<Argument>(V);
}

//  Get the basic block from an instruction/value
static BasicBlock *getBlockForValue(Value *V) {
  if (Instruction *I = dyn_cast<Instruction>(V))
    return I->getParent();
  return nullptr;
}
struct NewGVN::ValueDFS {
  int DFSIn;
  int DFSOut;
  int LocalNum;
  Value *Val;
  bool Equivalence;
  bool operator<(const ValueDFS &other) const {
    if (DFSIn < other.DFSIn)
      return true;
    else if (DFSIn == other.DFSIn) {
      if (DFSOut < other.DFSOut)
        return true;
      else if (DFSOut == other.DFSOut) {
        if (LocalNum < other.LocalNum)
          return true;
        else if (LocalNum == other.LocalNum)
          return !!Equivalence < !!other.Equivalence;
      }
    }
    return false;
  }
};

void NewGVN::convertDenseToDFSOrdered(CongruenceClass::MemberSet &Dense,
                                      std::set<ValueDFS> &DFSOrderedSet) {
  for (auto D : Dense) {
    BasicBlock *BB = getBlockForValue(D);
    // Constants are handled prior to ever calling this function, so
    // we should only be left with instructions as members
    assert(BB || "Should have figured out a basic block for value");
    ValueDFS VD;
    VD.Equivalence = false;
    std::pair<int, int> DFSPair = DFSBBMap[BB];
    VD.DFSIn = DFSPair.first;
    VD.DFSOut = DFSPair.second;

    // If it's an instruction, use the real local dfs number,
    if (Instruction *I = dyn_cast<Instruction>(D))
      VD.LocalNum = InstrLocalDFS[I];
    else
      llvm_unreachable("Should have been an instruction");

    VD.Val = D;
    DFSOrderedSet.insert(VD);
  }
}

void NewGVN::convertDenseToDFSOrdered(CongruenceClass::EquivalenceSet &Dense,
                                      std::set<ValueDFS> &DFSOrderedSet) {
  for (auto D : Dense) {
    std::pair<int, int> &DFSPair = DFSBBMap[D.second];
    ValueDFS VD;
    VD.DFSIn = DFSPair.first;
    VD.DFSOut = DFSPair.second;
    VD.Equivalence = true;

    // If it's an instruction, use the real local dfs number.
    // If it's a value, it *must* have come from equality propagation,
    // and thus we know it is valid for the entire block.  By giving
    // the local number as 0, it should sort before the instructions
    // in that block.
    if (Instruction *I = dyn_cast<Instruction>(D.first))
      VD.LocalNum = InstrLocalDFS[I];
    else
      VD.LocalNum = 0;

    VD.Val = D.first;
    DFSOrderedSet.insert(VD);
  }
}
static void patchReplacementInstruction(Instruction *I, Value *Repl) {
  // Patch the replacement so that it is not more restrictive than the value
  // being replaced.
  BinaryOperator *Op = dyn_cast<BinaryOperator>(I);
  BinaryOperator *ReplOp = dyn_cast<BinaryOperator>(Repl);
  if (Op && ReplOp && isa<OverflowingBinaryOperator>(Op) &&
      isa<OverflowingBinaryOperator>(ReplOp)) {
    if (ReplOp->hasNoSignedWrap() && !Op->hasNoSignedWrap())
      ReplOp->setHasNoSignedWrap(false);
    if (ReplOp->hasNoUnsignedWrap() && !Op->hasNoUnsignedWrap())
      ReplOp->setHasNoUnsignedWrap(false);
  }
  if (Instruction *ReplInst = dyn_cast<Instruction>(Repl)) {
    // FIXME: If both the original and replacement value are part of the
    // same control-flow region (meaning that the execution of one
    // guarentees the executation of the other), then we can combine the
    // noalias scopes here and do better than the general conservative
    // answer used in combineMetadata().

    // In general, GVN unifies expressions over different control-flow
    // regions, and so we need a conservative combination of the noalias
    // scopes.
    unsigned KnownIDs[] = {
        LLVMContext::MD_tbaa,    LLVMContext::MD_alias_scope,
        LLVMContext::MD_noalias, LLVMContext::MD_range,
        LLVMContext::MD_fpmath,  LLVMContext::MD_invariant_load,
    };
    combineMetadata(ReplInst, I, KnownIDs);
  }
}

static void patchAndReplaceAllUsesWith(Instruction *I, Value *Repl) {
  patchReplacementInstruction(I, Repl);
  I->replaceAllUsesWith(Repl);
}

void NewGVN::deleteInstructionsInBlock(BasicBlock *BB) {
  DEBUG(dbgs() << "  BasicBlock Dead:" << *BB);
  ++NumGVNBlocksDeleted;

  // Check to see if there are non-terminating instructions to delete.
  if (isa<TerminatorInst>(BB->begin()))
    return;

  // Delete the instructions backwards, as it has a reduced likelihood of having
  // to update as many def-use and use-def chains.
  Instruction *EndInst = BB->getTerminator(); // Last not to be deleted.
  while (EndInst != BB->begin()) {
    // Delete the next to last instruction.
    BasicBlock::iterator I = EndInst;
    Instruction *Inst = --I;
    if (!Inst->use_empty())
      Inst->replaceAllUsesWith(UndefValue::get(Inst->getType()));
    if (isa<LandingPadInst>(Inst)) {
      EndInst = Inst;
      continue;
    }
    BB->getInstList().erase(Inst);
    ++NumGVNInstrDeleted;
  }
}

void NewGVN::markInstructionForDeletion(Instruction *I) {
  DEBUG(dbgs() << "Marking " << *I << " for deletion\n");
  InstructionsToErase.push_back(I);
}

void NewGVN::replaceInstruction(Instruction *I, Value *V) {

  DEBUG(dbgs() << "Replacing " << *I << " with " << *V << "\n");
  patchAndReplaceAllUsesWith(I, V);
  // We save the actual erasing to avoid invalidating memory
  // dependencies until we are done with everything.
  markInstructionForDeletion(I);
}

namespace {
// This is a stack that contains both the value and dfs info of where
// that value is valid

class ValueDFSStack {
public:
  Value *back() const { return ValueStack.back(); }
  std::pair<int, int> dfs_back() const { return DFSStack.back(); }

  void push_back(Value *V, int DFSIn, int DFSOut) {
    ValueStack.push_back(V);
    DFSStack.push_back(std::make_pair(DFSIn, DFSOut));
  }
  bool empty() const { return DFSStack.empty(); }
  bool isInScope(int DFSIn, int DFSOut) const {
    if (empty())
      return false;
    return DFSIn >= DFSStack.back().first && DFSOut <= DFSStack.back().second;
  }

  void popUntilDFSScope(int DFSIn, int DFSOut) {
    // These two should always be in sync at this point
    assert(ValueStack.size() == DFSStack.size() &&
           "Mismatch between ValueStack and DFSStack");
    while (
        !DFSStack.empty() &&
        !(DFSIn >= DFSStack.back().first && DFSOut <= DFSStack.back().second)) {
      DFSStack.pop_back();
      ValueStack.pop_back();
    }
  }

private:
  SmallVector<Value *, 8> ValueStack;
  SmallVector<std::pair<int, int>, 8> DFSStack;
};
}

bool NewGVN::eliminateInstructions(Function &F) {
  // This is a non-standard eliminator. The normal way to eliminate is
  // to walk the dominator tree in order, keeping track of available
  // values, and eliminating them.  However, this is mildly
  // pointless. It requires doing lookups on every instruction,
  // regardless of whether we will ever eliminate it.  For
  // instructions part of a singleton congruence class, we know we
  // will never eliminate it.

  // Instead, this eliminator looks at the congruence classes directly, sorts
  // them into a DFS ordering of the dominator tree, and then we just
  // perform eliminate straight on the sets by walking the congruence
  // class members in order, and eliminate the ones dominated by the
  // last member.   This is technically O(N log N) where N = number of
  // instructions (since in theory all instructions may be in the same
  // congruence class).
  // When we find something not dominated, it becomes the new leader
  // for elimination purposes

  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI) {
    DomTreeNode *DTN = DT->getNode(FI);
    if (!DTN)
      continue;
    DFSBBMap[FI] = std::make_pair(DTN->getDFSNumIn(), DTN->getDFSNumOut());
  }

  for (unsigned i = 0, e = CongruenceClasses.size(); i != e; ++i) {
    CongruenceClass *CC = CongruenceClasses[i];
    // FIXME: We should eventually be able to replace everything still
    // in the initial class with undef, as they should be unreachable.
    // Right now, initial still contains some things we skip value
    // numbering of (UNREACHABLE's, for example)
    if (CC == InitialClass || CC->dead)
      continue;
    assert(CC->leader && "We should have had a leader");

    // If this is a leader that is always available, just replace
    // everything with it. We then update the congruence class with
    // whatever members are left.
    if (alwaysAvailable(CC->leader)) {
      SmallPtrSet<Value *, 4> MembersLeft;
      for (auto M : CC->members) {

        Value *Member = M;
        // Stores can't be replaced directly since they have no uses
        if (Member == CC->leader || isa<StoreInst>(Member)) {
          MembersLeft.insert(Member);
          continue;
        }

        DEBUG(dbgs() << "Found replacement " << *(CC->leader) << " for "
                     << *Member << "\n");
        // Due to equality propagation, these may not always be
        // instructions, they may be real values.  We don't really
        // care about trying to replace the non-instructions.
        if (Instruction *I = dyn_cast<Instruction>(Member)) {
          assert(CC->leader != I && "About to accidentally remove our leader");
          replaceInstruction(I, CC->leader);
          continue;
        } else {
          MembersLeft.insert(I);
        }
      }
      CC->members.swap(MembersLeft);

    } else {
      DEBUG(dbgs() << "Eliminating in congruence class " << CC->id << "\n");
      // If this is a singleton, we can skip it
      if (CC->members.size() != 1) {

        // This is a stack because equality replacement/etc may place
        // constants in the middle of the member list, and we want to use
        // those constant values in preference to the current leader, over
        // the scope of those constants.

        ValueDFSStack EliminationStack;

        // Convert the members and equivalences to DFS ordered sets and
        // then merge them.
        std::set<ValueDFS> DFSOrderedMembers;
        convertDenseToDFSOrdered(CC->members, DFSOrderedMembers);
        std::set<ValueDFS> DFSOrderedEquivalences;
        convertDenseToDFSOrdered(CC->equivalences, DFSOrderedEquivalences);
        std::vector<ValueDFS> DFSOrderedSet;
        set_union(DFSOrderedMembers.begin(), DFSOrderedMembers.end(),
                  DFSOrderedEquivalences.begin(), DFSOrderedEquivalences.end(),
                  std::inserter(DFSOrderedSet, DFSOrderedSet.begin()));

        for (auto &C : DFSOrderedSet) {
          int MemberDFSIn = C.DFSIn;
          int MemberDFSOut = C.DFSOut;
          Value *Member = C.Val;
          bool EquivalenceOnly = C.Equivalence;

          // We ignore stores because we can't replace them, since,
          // they have no uses
          if (isa<StoreInst>(Member))
            continue;

          if (EliminationStack.empty()) {
            DEBUG(dbgs() << "Elimination Stack is empty\n");
          } else {
            DEBUG(dbgs() << "Elimination Stack Top DFS numbers are ("
                         << EliminationStack.dfs_back().first << ","
                         << EliminationStack.dfs_back().second << ")\n");
          }
          if (isa<Constant>(Member))
            assert(isa<Constant>(CC->leader) || EquivalenceOnly);

          DEBUG(dbgs() << "Current DFS numbers are (" << MemberDFSIn << ","
                       << MemberDFSOut << ")\n");
          // First, we synchronize to our current scope, by
          // popping until we are back within a DFS scope that
          // dominates the current member.
          // Then, what happens depends on a few factors
          // If the stack is now empty, we need to push
          // If we have a constant or a local equivalence we want to
          // start using, we also push
          // Otherwise, we walk along, processing members who are
          // dominated by this scope, and eliminate them
          bool ShouldPush = EliminationStack.empty() || isa<Constant>(Member) ||
                            EquivalenceOnly;
          bool OutOfScope =
              !EliminationStack.isInScope(MemberDFSIn, MemberDFSOut);
          if (OutOfScope || ShouldPush) {
            // Sync to our current scope
            EliminationStack.popUntilDFSScope(MemberDFSIn, MemberDFSOut);
            // Push if we need to
            ShouldPush |= EliminationStack.empty();
            if (ShouldPush) {
              EliminationStack.push_back(Member, MemberDFSIn, MemberDFSOut);
            }
          }

          // Skip the case of trying to eliminate the leader, or trying
          // to eliminate equivalence-only candidates
          if (Member == CC->leader || EquivalenceOnly)
            continue;

          Value *Result = EliminationStack.back();
          if (Member == Result)
            continue;
          DEBUG(dbgs() << "Found replacement " << *Result << " for " << *Member
                       << "\n");

          // Perform actual replacement
          Instruction *I;
          if ((I = dyn_cast<Instruction>(Member)) && Member != CC->leader) {
            assert(CC->leader != I &&
                   "About to accidentally remove our leader");

            replaceInstruction(I, Result);
            CC->members.erase(Member);
          }
        }
      }
    }
  }
  return true;
}
