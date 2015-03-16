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
#include "llvm/ADT/BitVector.h"
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
  static unsigned int nextCongruenceNum;
  unsigned int id;
  Value *leader;
  Expression *expression;
  // Actual members of this class.  These are the things the same everywhere
  MemberSet members;
  // Coercible members of this class. These are loads where we can pull the
  // value out of a store. This means they need special processing during
  // elimination to do this, but they are otherwise the same as members,
  // in particular, we can eliminate one in favor of a dominating one.
  MemberSet coercible_members;

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
unsigned int CongruenceClass::nextCongruenceNum = 0;

class NewGVN : public FunctionPass {
  MemoryDependenceAnalysis *MD;
  DominatorTree *DT;
  const DataLayout *DL;
  const TargetLibraryInfo *TLI;
  AssumptionCache *AC;
  AliasAnalysis *AA;
  MemorySSA *MSSA;
  MemorySSAWalker *MSSAWalker;
  BumpPtrAllocator ExpressionAllocator;
  ArrayRecycler<Value *> ArgRecycler;

  // Congruence class info
  DenseMap<Value *, CongruenceClass *> ValueToClass;
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
  DenseSet<const BasicBlock *> ReachableBlocks;
  // This is a bitvector because, on larger functions, we may have
  // thousands of touched instructions at once (entire blocks,
  // instructions with hundreds of uses, etc).  Even with optimization
  // for when we mark whole blocks as touched, when this was a
  // SmallPtrSet or DenseSet, for some functions, we spent >20% of all
  // the time in GVN just managing this list.  The bitvector, on the
  // other hand, efficiently supports test/set/clear of both
  // individual and ranges, as well as "find next element" This
  // enables us to use it as a worklist with essentially 0 cost.
  BitVector TouchedInstructions;
  DenseMap<const BasicBlock *, std::pair<unsigned int, unsigned int>>
      BlockInstRange;
  DenseSet<const BasicBlock *> UniquePredecessorBlocks;
  DenseMap<const Instruction *, unsigned int> ProcessedCount;
  DenseMap<const BasicBlock *, unsigned int> ProcessedBlockCount;
  CongruenceClass *InitialClass;
  std::vector<CongruenceClass *> CongruenceClasses;
  DenseMap<BasicBlock *, std::pair<int, int>> DFSBBMap;
  DenseMap<const Instruction *, unsigned int> InstrDFS;
  std::vector<Instruction *> DFSToInstr;
  SmallPtrSet<Instruction *, 8> InstructionsToErase;
  // This is a mapping from Load to (offset into source, coercion source)
  DenseMap<const Value *, std::pair<unsigned, Value *>> CoercionInfo;
  // This is a mapping for loads that got widened, to the new load. This ensures
  // we coerce from the new widened load, instead of the old one. Otherwise, we
  // may try to widen the same old load multiple times.
  DenseMap<const Value *, Value *> CoercionForwarding;

public:
  static char ID; // Pass identification, replacement for typeid
  explicit NewGVN() : FunctionPass(ID), MD(nullptr) {
    initializeNewGVNPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

private:
  // This transformation requires dominator postdominator info
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<MemoryDependenceAnalysis>();
    AU.addRequired<MemorySSALazy>();
    AU.addRequired<AliasAnalysis>();

    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<AliasAnalysis>();
  }

  // expression handling
  Expression *createExpression(Instruction *, BasicBlock *);
  Expression *createBinaryExpression(unsigned, Type *, Value *, Value *,
                                     BasicBlock *);
  bool setBasicExpressionInfo(Instruction *, BasicExpression *, BasicBlock *);
  PHIExpression *createPHIExpression(Instruction *);
  VariableExpression *createVariableExpression(Value *);
  ConstantExpression *createConstantExpression(Constant *);
  Expression *createVariableOrConstant(Value *V, BasicBlock *B);
  StoreExpression *createStoreExpression(StoreInst *, MemoryAccess *,
                                         BasicBlock *);
  LoadExpression *createLoadExpression(LoadInst *, MemoryAccess *,
                                       BasicBlock *);
  CoercibleLoadExpression *createCoercibleLoadExpression(LoadInst *,
                                                         MemoryAccess *,
                                                         unsigned, Value *,
                                                         BasicBlock *);
  CallExpression *createCallExpression(CallInst *, MemoryAccess *,
                                       BasicBlock *);
  AggregateValueExpression *createAggregateValueExpression(Instruction *,
                                                           BasicBlock *l);

  Expression *uniquifyExpression(Expression *);
  BasicExpression *createCmpExpression(unsigned, Type *, CmpInst::Predicate,
                                       Value *, Value *, BasicBlock *);
  // Congruence class handling
  CongruenceClass *createCongruenceClass(Value *Leader, Expression *E) {
    CongruenceClass *result = new CongruenceClass(Leader, E);
    CongruenceClasses.emplace_back(result);
    return result;
  }

  CongruenceClass *createSingletonCongruenceClass(Value *Member) {
    CongruenceClass *CClass = createCongruenceClass(Member, NULL);
    CClass->members.insert(Member);
    ValueToClass[Member] = CClass;
    return CClass;
  }
  void initializeCongruenceClasses(Function &F);

  // Symbolic evaluation
  Expression *checkSimplificationResults(Expression *, Instruction *, Value *);
  Expression *performSymbolicEvaluation(Value *, BasicBlock *);
  Expression *performSymbolicLoadCoercion(LoadInst *, Instruction *,
                                          MemoryAccess *, BasicBlock *);
  Expression *performSymbolicLoadEvaluation(Instruction *, BasicBlock *);
  Expression *performSymbolicStoreEvaluation(Instruction *, BasicBlock *);
  Expression *performSymbolicCallEvaluation(Instruction *, BasicBlock *);
  Expression *performSymbolicPHIEvaluation(Instruction *, BasicBlock *);
  Expression *performSymbolicAggrValueEvaluation(Instruction *, BasicBlock *);
  int analyzeLoadFromClobberingStore(Type *, Value *, StoreInst *);
  int analyzeLoadFromClobberingLoad(Type *, Value *, LoadInst *);
  int analyzeLoadFromClobberingMemInst(Type *, Value *, MemIntrinsic *);
  int analyzeLoadFromClobberingWrite(Type *, Value *, Value *, uint64_t);
  // Congruence finding
  Value *lookupOperandLeader(Value *, BasicBlock *) const;
  Value *findDominatingEquivalent(CongruenceClass *, BasicBlock *) const;
  void performCongruenceFinding(Value *, Expression *);
  // Predicate and reachability handling
  void updateReachableEdge(BasicBlock *, BasicBlock *);
  void processOutgoingEdges(TerminatorInst *, BasicBlock *);
  void propagateChangeInEdge(BasicBlock *);
  bool propagateEquality(Value *, Value *, BasicBlock *);
  Value *findConditionEquivalence(Value *, BasicBlock *) const;

  // Instruction replacement
  unsigned replaceAllDominatedUsesWith(Value *, Value *, BasicBlock *);

  // Elimination
  struct ValueDFS;
  void convertDenseToDFSOrdered(CongruenceClass::MemberSet &,
                                std::vector<ValueDFS> &, bool);
  void convertDenseToDFSOrdered(CongruenceClass::EquivalenceSet &,
                                std::vector<ValueDFS> &);

  bool eliminateInstructions(Function &);
  void replaceInstruction(Instruction *, Value *);
  void markInstructionForDeletion(Instruction *);
  void deleteInstructionsInBlock(BasicBlock *);
  bool canCoerceMustAliasedValueToLoad(Value *, Type *);
  Value *coerceAvailableValueToLoadType(Value *, Type *, Instruction *);
  Value *getStoreValueForLoad(Value *, unsigned, Type *, Instruction *);
  Value *getLoadValueForLoad(LoadInst *, unsigned, Type *, Instruction *);
  Value *getMemInstValueForLoad(MemIntrinsic *, unsigned, Type *,
                                Instruction *);
  Value *coerceLoad(Value *);
  // New instruction creation
  void handleNewInstruction(Instruction *){};
  void markUsersTouched(Value *);
  // Utilities
  void cleanupTables();
  std::pair<unsigned, unsigned> assignDFSNumbers(BasicBlock *, unsigned);
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
PHIExpression *NewGVN::createPHIExpression(Instruction *I) {
  PHINode *PN = cast<PHINode>(I);
  PHIExpression *E = new (ExpressionAllocator)
      PHIExpression(PN->getNumOperands(), I->getParent());

  E->allocateArgs(ArgRecycler, ExpressionAllocator);
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
      E->args_push_back(Operand);
    } else {
      E->args_push_back(I->getOperand(i));
    }
  }
  return E;
}

// Set basic expression info (Arguments, type, opcode) for Expression
// E from Instruction I in block B

bool NewGVN::setBasicExpressionInfo(Instruction *I, BasicExpression *E,
                                    BasicBlock *B) {
  bool AllConstant = true;
  E->setType(I->getType());
  E->setOpcode(I->getOpcode());
  E->allocateArgs(ArgRecycler, ExpressionAllocator);

  for (auto &O : I->operands()) {
    Value *Operand = lookupOperandLeader(O, B);
    if (!isa<Constant>(Operand))
      AllConstant = false;
    E->args_push_back(Operand);
  }
  return AllConstant;
}
// This is a special function only used by equality propagation, it
// should not be called elsewhere
BasicExpression *NewGVN::createCmpExpression(unsigned Opcode, Type *Type,
                                             CmpInst::Predicate Predicate,
                                             Value *LHS, Value *RHS,
                                             BasicBlock *B) {
  BasicExpression *E = new (ExpressionAllocator) BasicExpression(2);
  E->allocateArgs(ArgRecycler, ExpressionAllocator);
  E->setType(Type);
  E->setOpcode((Opcode << 8) | Predicate);
  LHS = lookupOperandLeader(LHS, B);
  E->args_push_back(LHS);
  RHS = lookupOperandLeader(RHS, B);
  E->args_push_back(RHS);
  return E;
}

Expression *NewGVN::createBinaryExpression(unsigned Opcode, Type *T,
                                           Value *Arg1, Value *Arg2,
                                           BasicBlock *B) {
  BasicExpression *E = new (ExpressionAllocator) BasicExpression(2);

  E->setType(T);
  E->setOpcode(Opcode);
  E->allocateArgs(ArgRecycler, ExpressionAllocator);
  if (Instruction::isCommutative(Opcode)) {
    // Ensure that commutative instructions that only differ by a permutation
    // of their operands get the same value number by sorting the operand value
    // numbers.  Since all commutative instructions have two operands it is more
    // efficient to sort by hand rather than using, say, std::sort.
    if (Arg1 > Arg2)
      std::swap(Arg1, Arg2);
  }

  E->args_push_back(lookupOperandLeader(Arg1, B));
  E->args_push_back(lookupOperandLeader(Arg2, B));

  Value *V = SimplifyBinOp(Opcode, E->Args[0], E->Args[1], *DL, TLI, DT, AC);
  if (Expression *simplifiedE = checkSimplificationResults(E, nullptr, V))
    return simplifiedE;
  return E;
}

// Take a Value returned by simplification of Expression E/Instruction
// I, and see if it resulted in a simpler expression. If so, return
// that expression
// TODO: Once finished, this should not take an Instruction, we only
// use it for printing
Expression *NewGVN::checkSimplificationResults(Expression *E, Instruction *I,
                                               Value *V) {
  if (!V)
    return NULL;
  if (Constant *C = dyn_cast<Constant>(V)) {
#ifndef NDEBUG
    if (I)
      DEBUG(dbgs() << "Simplified " << *I << " to "
                   << " constant " << *C << "\n");
#endif
    NumGVNOpsSimplified++;
    assert(isa<BasicExpression>(E) &&
           "We should always have had a basic expression here");

    cast<BasicExpression>(E)->deallocateArgs(ArgRecycler);
    ExpressionAllocator.Deallocate(E);
    return createConstantExpression(C);
  }
  CongruenceClass *CC = ValueToClass.lookup(V);
  if (CC && CC->expression) {
#ifndef NDEBUG
    if (I)
      DEBUG(dbgs() << "Simplified " << *I << " to "
                   << " expression " << *V << "\n");

#endif
    NumGVNOpsSimplified++;
    assert(isa<BasicExpression>(E) &&
           "We should always have had a basic expression here");
    cast<BasicExpression>(E)->deallocateArgs(ArgRecycler);
    ExpressionAllocator.Deallocate(E);
    return CC->expression;
  }
  return NULL;
}

Expression *NewGVN::createExpression(Instruction *I, BasicBlock *B) {

  BasicExpression *E =
      new (ExpressionAllocator) BasicExpression(I->getNumOperands());

  bool AllConstant = setBasicExpressionInfo(I, E, B);

  if (I->isCommutative()) {
    // Ensure that commutative instructions that only differ by a permutation
    // of their operands get the same value number by sorting the operand value
    // numbers.  Since all commutative instructions have two operands it is more
    // efficient to sort by hand rather than using, say, std::sort.
    assert(I->getNumOperands() == 2 && "Unsupported commutative instruction!");
    if (E->Args[0] > E->Args[1])
      std::swap(E->Args[0], E->Args[1]);
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
    // Sort the operand value numbers so x<y and y>x get the same value
    // number.
    CmpInst::Predicate Predicate = CI->getPredicate();
    if (E->Args[0] > E->Args[1]) {
      std::swap(E->Args[0], E->Args[1]);
      Predicate = CmpInst::getSwappedPredicate(Predicate);
    }
    E->setOpcode((CI->getOpcode() << 8) | Predicate);
    // TODO: 25% of our time is spent in SimplifyCmpInst with pointer operands
    // TODO: Since we noop bitcasts, we may need to check types before
    // simplifying, so that we don't end up simplifying based on a wrong
    // type assumption. We should clean this up so we can use constants of the
    // wrong type

    assert(I->getOperand(0)->getType() == I->getOperand(1)->getType() &&
           "Wrong types on cmp instruction");
    if ((E->Args[0]->getType() == I->getOperand(0)->getType() &&
         E->Args[1]->getType() == I->getOperand(1)->getType())) {
      Value *V =
          SimplifyCmpInst(Predicate, E->Args[0], E->Args[1], *DL, TLI, DT, AC);
      if (Expression *simplifiedE = checkSimplificationResults(E, I, V))
        return simplifiedE;
    }

  } else if (isa<SelectInst>(I)) {
    if (isa<Constant>(E->Args[0]) ||
        (E->Args[1]->getType() == I->getOperand(1)->getType() &&
         E->Args[2]->getType() == I->getOperand(2)->getType())) {
      Value *V = SimplifySelectInst(E->Args[0], E->Args[1], E->Args[2], *DL,
                                    TLI, DT, AC);
      if (Expression *simplifiedE = checkSimplificationResults(E, I, V))
        return simplifiedE;
    }
  } else if (I->isBinaryOp()) {
    Value *V =
        SimplifyBinOp(E->getOpcode(), E->Args[0], E->Args[1], *DL, TLI, DT, AC);
    if (Expression *simplifiedE = checkSimplificationResults(E, I, V))
      return simplifiedE;
  } else if (BitCastInst *BI = dyn_cast<BitCastInst>(I)) {
    Value *V = SimplifyInstruction(BI, *DL, TLI, DT, AC);
    if (Expression *simplifiedE = checkSimplificationResults(E, I, V))
      return simplifiedE;
  } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I)) {
    if (GEP->getPointerOperandType() == E->Args[0]->getType()) {
      Value *V = SimplifyGEPInst(ArrayRef<Value *>(E->Args, E->args_size()),
                                 *DL, TLI, DT, AC);
      if (Expression *simplifiedE = checkSimplificationResults(E, I, V))
        return simplifiedE;
    }
  } else if (AllConstant) {
    // We don't bother trying to simplify unless all of the operands
    // were constant
    // TODO: There are a lot of Simplify*'s we could call here, if we
    // wanted to.  The original motivating case for this code was a
    // zext i1 false to i8, which we don't have an interface to
    // simplify (IE there is no SimplifyZExt)

    SmallVector<Constant *, 8> C;
    for (Value *Arg : E->arguments())
      C.emplace_back(cast<Constant>(Arg));

    Value *V =
        ConstantFoldInstOperands(E->getOpcode(), E->getType(), C, *DL, TLI);
    if (V) {
      if (Expression *simplifiedE = checkSimplificationResults(E, I, V))
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

AggregateValueExpression *
NewGVN::createAggregateValueExpression(Instruction *I, BasicBlock *B) {
  if (InsertValueInst *II = dyn_cast<InsertValueInst>(I)) {
    AggregateValueExpression *E = new (ExpressionAllocator)
        AggregateValueExpression(I->getNumOperands(), II->getNumIndices());
    setBasicExpressionInfo(I, E, B);
    E->allocateIntArgs(ExpressionAllocator);

    for (auto &Index : II->indices())
      E->int_args_push_back(Index);
    return E;

  } else if (ExtractValueInst *EI = dyn_cast<ExtractValueInst>(I)) {
    AggregateValueExpression *E = new (ExpressionAllocator)
        AggregateValueExpression(I->getNumOperands(), EI->getNumIndices());
    setBasicExpressionInfo(EI, E, B);
    E->allocateIntArgs(ExpressionAllocator);

    for (auto &Index : EI->indices())
      E->int_args_push_back(Index);
    return E;
  }
  llvm_unreachable("Unhandled type of aggregate value operation");
}

VariableExpression *NewGVN::createVariableExpression(Value *V) {
  VariableExpression *E = new (ExpressionAllocator) VariableExpression(V);
  E->setOpcode(V->getValueID());
  E = cast<VariableExpression>(uniquifyExpression(E));
  return E;
}

Expression *NewGVN::createVariableOrConstant(Value *V, BasicBlock *B) {
  Value *LeaderVal = lookupOperandLeader(V, B);
  if (Constant *C = dyn_cast<Constant>(LeaderVal))
    return createConstantExpression(C);
  return createVariableExpression(LeaderVal);
}

ConstantExpression *NewGVN::createConstantExpression(Constant *C) {
  ConstantExpression *E = new (ExpressionAllocator) ConstantExpression(C);
  E->setOpcode(C->getValueID());
  E = cast<ConstantExpression>(uniquifyExpression(E));
  return E;
}

CallExpression *NewGVN::createCallExpression(CallInst *CI, MemoryAccess *HV,
                                             BasicBlock *B) {
  CallExpression *E =
      new (ExpressionAllocator) CallExpression(CI->getNumOperands(), CI, HV);
  setBasicExpressionInfo(CI, E, B);
  return E;
}

// Find an equivalence in congruence class CC that dominates block B,
// if one exists
// TODO: Compare this against the predicate handling system in the paper

Value *NewGVN::findDominatingEquivalent(CongruenceClass *CC,
                                        BasicBlock *B) const {
  // This check is much faster than doing 0 iterations of the loop below
  if (CC->equivalences.empty())
    return nullptr;

  // TODO: This can be made faster by different set ordering, if
  // necessary, or caching whether we found one
  for (const auto &Member : CC->equivalences) {
    if (DT->dominates(Member.second, B))
      return Member.first;
  }
  return nullptr;
}

// lookupOperandLeader -- See if we have a congruence class and leader
// for this operand, and if so, return it. Otherwise, return the
// original operand
Value *NewGVN::lookupOperandLeader(Value *V, BasicBlock *B) const {
  CongruenceClass *CC = ValueToClass.lookup(V);
  if (CC && (CC != InitialClass)) {
    Value *Equivalence = findDominatingEquivalent(CC, B);
    if (Equivalence) {
      DEBUG(dbgs() << "Found equivalence " << *Equivalence << " for value "
                   << *V << " in block " << getBlockName(B) << "\n");
      return Equivalence;
    }
    return CC->leader;
  }
  return V;
}

LoadExpression *NewGVN::createLoadExpression(LoadInst *LI, MemoryAccess *DA,
                                             BasicBlock *B) {
  LoadExpression *E =
      new (ExpressionAllocator) LoadExpression(LI->getNumOperands(), LI, DA);
  E->allocateArgs(ArgRecycler, ExpressionAllocator);
  E->setType(LI->getType());
  // Give store and loads same opcode so they value number together
  E->setOpcode(0);
  Value *Operand = lookupOperandLeader(LI->getPointerOperand(), B);
  E->args_push_back(Operand);
  // TODO: Value number heap versions. We may be able to discover
  // things alias analysis can't on it's own (IE that a store and a
  // load have the same value, and thus, it isn't clobbering the load)
  return E;
}

CoercibleLoadExpression *NewGVN::createCoercibleLoadExpression(LoadInst *LI,
                                                               MemoryAccess *DA,
                                                               unsigned Offset,
                                                               Value *SrcVal,
                                                               BasicBlock *B) {
  CoercibleLoadExpression *E = new (ExpressionAllocator)
      CoercibleLoadExpression(LI->getNumOperands(), LI, DA, Offset, SrcVal);
  E->allocateArgs(ArgRecycler, ExpressionAllocator);
  E->setType(LI->getType());
  // Give store and loads same opcode so they value number together
  E->setOpcode(0);
  Value *Operand = lookupOperandLeader(LI->getPointerOperand(), B);
  E->args_push_back(Operand);
  // TODO: Value number heap versions. We may be able to discover
  // things alias analysis can't on it's own (IE that a store and a
  // load have the same value, and thus, it isn't clobbering the load)
  return E;
}

StoreExpression *NewGVN::createStoreExpression(StoreInst *SI, MemoryAccess *DA,
                                               BasicBlock *B) {
  StoreExpression *E =
      new (ExpressionAllocator) StoreExpression(SI->getNumOperands(), SI, DA);
  E->allocateArgs(ArgRecycler, ExpressionAllocator);
  E->setType(SI->getType());
  // Give store and loads same opcode so they value number together
  E->setOpcode(0);
  Value *Operand = lookupOperandLeader(SI->getPointerOperand(), B);
  E->args_push_back(Operand);
  // TODO: Value number heap versions. We may be able to discover
  // things alias analysis can't on it's own (IE that a store and a
  // load have the same value, and thus, it isn't clobbering the load)
  return E;
}

/// This function is called when we have a
/// memdep query of a load that ends up being a clobbering memory write (store,
/// memset, memcpy, memmove).  This means that the write *may* provide bits used
/// by the load but we can't be sure because the pointers don't mustalias.
///
/// Check this case to see if there is anything more we can do before we give
/// up.  This returns -1 if we have to give up, or a byte number in the stored
/// value of the piece that feeds the load.
int NewGVN::analyzeLoadFromClobberingWrite(Type *LoadTy, Value *LoadPtr,
                                           Value *WritePtr,
                                           uint64_t WriteSizeInBits) {
  // If the loaded or stored value is a first class array or struct, don't try
  // to transform them.  We need to be able to bitcast to integer.
  if (LoadTy->isStructTy() || LoadTy->isArrayTy())
    return -1;

  int64_t StoreOffset = 0, LoadOffset = 0;
  Value *StoreBase =
      GetPointerBaseWithConstantOffset(WritePtr, StoreOffset, *DL);
  Value *LoadBase = GetPointerBaseWithConstantOffset(LoadPtr, LoadOffset, *DL);
  if (StoreBase != LoadBase)
    return -1;

  // If the load and store don't overlap at all, the store doesn't provide
  // anything to the load.  In this case, they really don't alias at all, AA
  // must have gotten confused.
  uint64_t LoadSize = DL->getTypeSizeInBits(LoadTy);

  if ((WriteSizeInBits & 7) | (LoadSize & 7))
    return -1;
  uint64_t StoreSize = WriteSizeInBits >> 3; // Convert to bytes.
  LoadSize >>= 3;

  bool isAAFailure = false;
  if (StoreOffset < LoadOffset)
    isAAFailure = StoreOffset + int64_t(StoreSize) <= LoadOffset;
  else
    isAAFailure = LoadOffset + int64_t(LoadSize) <= StoreOffset;

  if (isAAFailure) {
    return -1;
  }

  // If the Load isn't completely contained within the stored bits, we don't
  // have all the bits to feed it.  We could do something crazy in the future
  // (issue a smaller load then merge the bits in) but this seems unlikely to be
  // valuable.
  if (StoreOffset > LoadOffset ||
      StoreOffset + StoreSize < LoadOffset + LoadSize)
    return -1;

  // Okay, we can do this transformation.  Return the number of bytes into the
  // store that the load is.
  return LoadOffset - StoreOffset;
}

/// This function is called when we have a
/// memdep query of a load that ends up being a clobbering store.
int NewGVN::analyzeLoadFromClobberingStore(Type *LoadTy, Value *LoadPtr,
                                           StoreInst *DepSI) {

  // Cannot handle reading from store of first-class aggregate yet.
  if (DepSI->getValueOperand()->getType()->isStructTy() ||
      DepSI->getValueOperand()->getType()->isArrayTy())
    return -1;

  Value *StorePtr = DepSI->getPointerOperand();
  uint64_t StoreSize =
      DL->getTypeSizeInBits(DepSI->getValueOperand()->getType());
  return analyzeLoadFromClobberingWrite(LoadTy, LoadPtr, StorePtr, StoreSize);
}

/// This function is called when we have a
/// memdep query of a load that ends up being clobbered by another load.  See if
/// the other load can feed into the second load.
int NewGVN::analyzeLoadFromClobberingLoad(Type *LoadTy, Value *LoadPtr,
                                          LoadInst *DepLI) {
  // Cannot handle reading from store of first-class aggregate yet.
  if (DepLI->getType()->isStructTy() || DepLI->getType()->isArrayTy())
    return -1;

  Value *DepPtr = DepLI->getPointerOperand();
  uint64_t DepSize = DL->getTypeSizeInBits(DepLI->getType());
  int Offset = analyzeLoadFromClobberingWrite(LoadTy, LoadPtr, DepPtr, DepSize);

  if (Offset != -1) {
    // If the size is too large and we will have to widen, ensure we pass the
    // widening rules below
    unsigned SrcValSize = DL->getTypeStoreSize(DepLI->getType());
    unsigned LoadSize = DL->getTypeStoreSize(LoadTy);
    if (Offset + LoadSize <= SrcValSize)
      return Offset;
  }

  // If we have a load/load clobber an DepLI can be widened to cover this load,
  // then we should widen it!
  int64_t LoadOffs = 0;
  const Value *LoadBase =
      GetPointerBaseWithConstantOffset(LoadPtr, LoadOffs, *DL);
  unsigned LoadSize = DL->getTypeStoreSize(LoadTy);

  unsigned Size = MemoryDependenceAnalysis::getLoadLoadClobberFullWidthSize(
      LoadBase, LoadOffs, LoadSize, DepLI);
  if (Size == 0)
    return -1;

  return analyzeLoadFromClobberingWrite(LoadTy, LoadPtr, DepPtr, Size * 8);
}

int NewGVN::analyzeLoadFromClobberingMemInst(Type *LoadTy, Value *LoadPtr,
                                             MemIntrinsic *MI) {
  // If the mem operation is a non-constant size, we can't handle it.
  ConstantInt *SizeCst = dyn_cast<ConstantInt>(MI->getLength());
  if (!SizeCst)
    return -1;
  uint64_t MemSizeInBits = SizeCst->getZExtValue() * 8;

  // If this is memset, we just need to see if the offset is valid in the size
  // of the memset..
  if (MI->getIntrinsicID() == Intrinsic::memset)
    return analyzeLoadFromClobberingWrite(LoadTy, LoadPtr, MI->getDest(),
                                          MemSizeInBits);

  // If we have a memcpy/memmove, the only case we can handle is if this is a
  // copy from constant memory.  In that case, we can read directly from the
  // constant memory.
  MemTransferInst *MTI = cast<MemTransferInst>(MI);

  Constant *Src = dyn_cast<Constant>(MTI->getSource());
  if (!Src)
    return -1;

  GlobalVariable *GV = dyn_cast<GlobalVariable>(GetUnderlyingObject(Src, *DL));
  if (!GV || !GV->isConstant())
    return -1;

  // See if the access is within the bounds of the transfer.
  int Offset = analyzeLoadFromClobberingWrite(LoadTy, LoadPtr, MI->getDest(),
                                              MemSizeInBits);
  if (Offset == -1)
    return Offset;

  unsigned AS = Src->getType()->getPointerAddressSpace();
  // Otherwise, see if we can constant fold a load from the constant with the
  // offset applied as appropriate.
  Src =
      ConstantExpr::getBitCast(Src, Type::getInt8PtrTy(Src->getContext(), AS));
  Constant *OffsetCst =
      ConstantInt::get(Type::getInt64Ty(Src->getContext()), (unsigned)Offset);
  Src = ConstantExpr::getGetElementPtr(Src, OffsetCst);
  Src = ConstantExpr::getBitCast(Src, PointerType::get(LoadTy, AS));
  if (ConstantFoldLoadFromConstPtr(Src, *DL))
    return Offset;
  return -1;
}

Expression *NewGVN::performSymbolicStoreEvaluation(Instruction *I,
                                                   BasicBlock *B) {
  StoreInst *SI = cast<StoreInst>(I);
  Expression *E =
      createStoreExpression(SI, MSSAWalker->getClobberingMemoryAccess(SI), B);
  return E;
}

Expression *NewGVN::performSymbolicLoadCoercion(LoadInst *LI,
                                                Instruction *DepInst,
                                                MemoryAccess *DefiningAccess,
                                                BasicBlock *B) {
  Type *LoadType = LI->getType();
  if (StoreInst *DepSI = dyn_cast<StoreInst>(DepInst)) {
    Value *LoadAddressLeader = lookupOperandLeader(LI->getPointerOperand(), B);
    Value *StoreAddressLeader =
        lookupOperandLeader(DepSI->getPointerOperand(), B);
    Value *StoreVal = DepSI->getValueOperand();
    if (StoreVal->getType() == LoadType &&
        LoadAddressLeader == StoreAddressLeader) {
      return createVariableOrConstant(DepSI->getValueOperand(), B);
    } else {
      int Offset = analyzeLoadFromClobberingStore(
          LoadType, LI->getPointerOperand(), DepSI);
      if (Offset >= 0)
        return createCoercibleLoadExpression(LI, DefiningAccess,
                                             (unsigned)Offset, DepSI, B);
    }
  } else if (LoadInst *DepLI = dyn_cast<LoadInst>(DepInst)) {
    int Offset =
        analyzeLoadFromClobberingLoad(LoadType, LI->getPointerOperand(), DepLI);
    if (Offset >= 0)
      return createCoercibleLoadExpression(LI, DefiningAccess, (unsigned)Offset,
                                           DepLI, B);
  } else if (MemIntrinsic *DepMI = dyn_cast<MemIntrinsic>(DepInst)) {
    int Offset = analyzeLoadFromClobberingMemInst(
        LoadType, LI->getPointerOperand(), DepMI);
    if (Offset >= 0)
      return createCoercibleLoadExpression(LI, DefiningAccess, (unsigned)Offset,
                                           DepMI, B);
  }
  // If this load really doesn't depend on anything, then we must be loading
  // an
  // undef value.  This can happen when loading for a fresh allocation with
  // no
  // intervening stores, for example.
  else if (isa<AllocaInst>(DepInst) || isMallocLikeFn(DepInst, TLI))
    return createConstantExpression(UndefValue::get(LoadType));

  // If this load occurs either right after a lifetime begin,
  // then the loaded value is undefined.
  else if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(DepInst)) {
    if (II->getIntrinsicID() == Intrinsic::lifetime_start)
      return createConstantExpression(UndefValue::get(LoadType));
  }
  // If this load follows a calloc (which zero initializes memory),
  // then the loaded value is zero
  else if (isCallocLikeFn(DepInst, TLI)) {
    return createConstantExpression(Constant::getNullValue(LoadType));
  }

  return nullptr;
}

Expression *NewGVN::performSymbolicLoadEvaluation(Instruction *I,
                                                  BasicBlock *B) {
  LoadInst *LI = cast<LoadInst>(I);

  // We really can't do anything with non-simple loads
  if (!LI->isSimple())
    return NULL;

  Value *LoadAddressLeader = lookupOperandLeader(LI->getPointerOperand(), B);
  // Load of undef is undef
  if (isa<UndefValue>(LoadAddressLeader))
    return createConstantExpression(UndefValue::get(LI->getType()));

  MemoryAccess *DefiningAccess = MSSAWalker->getClobberingMemoryAccess(I);

  if (!MSSA->isLiveOnEntryDef(DefiningAccess)) {
    if (MemoryDef *MD = dyn_cast<MemoryDef>(DefiningAccess)) {
      Instruction *DefiningInst = MD->getMemoryInst();
      // If the defining instruction is not reachable, replace with
      // undef
      if (!ReachableBlocks.count(DefiningInst->getParent()))
        return createConstantExpression(UndefValue::get(LI->getType()));
      Expression *CoercionResult =
          performSymbolicLoadCoercion(LI, DefiningInst, DefiningAccess, B);
      if (CoercionResult)
        return CoercionResult;
    }
  } else {
    BasicBlock *LoadBlock = LI->getParent();
    MemoryAccess *LoadAccess = MSSA->getMemoryAccess(LI);
    // Okay, so uh, we couldn't use the defining access to grab a value out of
    // See if we can reuse any of it's uses by widening a load.
    for (const MemoryAccess *MA : DefiningAccess->uses()) {
      if (MA == LoadAccess)
        continue;
      if (const MemoryUse *MU = dyn_cast<MemoryUse>(MA)) {
        Instruction *DefiningInst = MU->getMemoryInst();
        if (LoadInst *DepLI = dyn_cast<LoadInst>(DefiningInst)) {
          BasicBlock *DefiningBlock = DefiningInst->getParent();
          if (!DT->dominates(DefiningBlock, LoadBlock))
            continue;
          // Make sure the dependent load comes before the load we are trying
          // to coerce if they are in the same block
          if (InstrDFS[DepLI] >= InstrDFS[LI])
            continue;
          int Offset = analyzeLoadFromClobberingLoad(
              LI->getType(), LI->getPointerOperand(), DepLI);
          if (Offset >= 0)
            return createCoercibleLoadExpression(LI, DefiningAccess,
                                                 (unsigned)Offset, DepLI, B);
        }
      }
    }
  }

  Expression *E = createLoadExpression(LI, DefiningAccess, B);
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
    return createCallExpression(CI, MSSAWalker->getClobberingMemoryAccess(CI),
                                B);
  else
    return nullptr;
}

// performSymbolicPHIEvaluation - Evaluate PHI nodes symbolically, and
// create an expression result
Expression *NewGVN::performSymbolicPHIEvaluation(Instruction *I,
                                                 BasicBlock *B) {
  PHIExpression *E = cast<PHIExpression>(createPHIExpression(I));
  E->setOpcode(I->getOpcode());
  if (E->args_empty()) {
    DEBUG(dbgs() << "Simplified PHI node " << *I << " to undef"
                 << "\n");
    E->deallocateArgs(ArgRecycler);
    ExpressionAllocator.Deallocate(E);
    return createConstantExpression(UndefValue::get(I->getType()));
  }

  Value *AllSameValue = E->Args[0];

  for (const Value *Arg : E->arguments())
    if (Arg != AllSameValue) {
      AllSameValue = NULL;
      break;
    }

  if (AllSameValue) {
    // It's possible to have phi nodes with cycles (IE dependent on
    // other phis that are .... dependent on the original phi node),
    // especially
    // in weird CFG's where some arguments are unreachable, or
    // uninitialized along certain paths.
    // This can cause infinite loops  during evaluation (even if you disable
    // the
    // recursion below, you will simply ping-pong between congruence classes)
    // If a phi node symbolically evaluates to another phi node, just
    // leave it alone
    // If they are really the same, we will still eliminate them in
    // favor of each other.
    if (isa<PHINode>(AllSameValue))
      return E;
    NumGVNPhisAllSame++;
    DEBUG(dbgs() << "Simplified PHI node " << *I << " to " << *AllSameValue
                 << "\n");
    E->deallocateArgs(ArgRecycler);
    ExpressionAllocator.Deallocate(E);
    if (Constant *C = dyn_cast<Constant>(AllSameValue))
      return createConstantExpression(C);
    return createVariableExpression(AllSameValue);
  }
  return E;
}

Expression *NewGVN::performSymbolicAggrValueEvaluation(Instruction *I,
                                                       BasicBlock *B) {
  if (ExtractValueInst *EI = dyn_cast<ExtractValueInst>(I)) {
    IntrinsicInst *II = dyn_cast<IntrinsicInst>(EI->getAggregateOperand());
    if (II != nullptr && EI->getNumIndices() == 1 && *EI->idx_begin() == 0) {
      unsigned Opcode = 0;
      // EI might be an extract from one of our recognised intrinsics. If it
      // is we'll synthesize a semantically equivalent expression instead on
      // an extract value expression.
      switch (II->getIntrinsicID()) {
      case Intrinsic::sadd_with_overflow:
      case Intrinsic::uadd_with_overflow:
        Opcode = Instruction::Add;
        break;
      case Intrinsic::ssub_with_overflow:
      case Intrinsic::usub_with_overflow:
        Opcode = Instruction::Sub;
        break;
      case Intrinsic::smul_with_overflow:
      case Intrinsic::umul_with_overflow:
        Opcode = Instruction::Mul;
        break;
      default:
        break;
      }

      if (Opcode != 0) {
        // Intrinsic recognized. Grab its args to finish building the
        // expression.
        assert(II->getNumArgOperands() == 2 &&
               "Expect two args for recognised intrinsics.");
        return createBinaryExpression(Opcode, EI->getType(),
                                      II->getArgOperand(0),
                                      II->getArgOperand(1), B);
      }
    }
  }

  return createAggregateValueExpression(I, B);
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
    case Instruction::ExtractValue:
    case Instruction::InsertValue:
      E = performSymbolicAggrValueEvaluation(I, B);
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

    // If From occurs as a phi node operand then the use implicitly lives in
    // the
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
        TouchedInstructions.set(InstrDFS[I]);
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
  Worklist.emplace_back(LHS, RHS);
  bool Changed = false;

  while (!Worklist.empty()) {
    std::pair<Value *, Value *> Item = Worklist.pop_back_val();
    LHS = Item.first;
    RHS = Item.second;
    DEBUG(dbgs() << "Setting equivalence " << *LHS << " = " << *RHS
                 << " in blocks dominated by " << getBlockName(Root) << "\n");

    if (LHS == RHS)
      continue;
    assert(LHS->getType() == RHS->getType() && "Equality but unequal types!");

    // Don't try to propagate equalities between constants.
    if (isa<Constant>(LHS) && isa<Constant>(RHS))
      continue;

    // Prefer a constant on the right-hand side, or an Argument if no
    // constants.
    if (isa<Constant>(LHS) || (isa<Argument>(LHS) && !isa<Constant>(RHS)))
      std::swap(LHS, RHS);

    // Put the most dominating value on the RHS. This ensures that we
    // replace later things in favor of earlier things
    if (isa<Instruction>(LHS) && isa<Instruction>(RHS))
      if (InstrDFS[cast<Instruction>(LHS)] < InstrDFS[cast<Instruction>(RHS)])
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
    // FIXME: I think this can be deleted now, bootstrap with an assert
    if (!LHS->hasOneUse() && 0) {
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
      Worklist.emplace_back(A, RHS);
      Worklist.emplace_back(B, RHS);
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
        Worklist.emplace_back(Op0, Op1);

      // Handle the floating point versions of equality comparisons too.
      if ((isKnownTrue && Cmp->getPredicate() == CmpInst::FCMP_OEQ) ||
          (isKnownFalse && Cmp->getPredicate() == CmpInst::FCMP_UNE)) {

        // Floating point -0.0 and 0.0 compare equal, so we can only
        // propagate values if we know that we have a constant and that
        // its value is non-zero.

        // FIXME: We should do this optimization if 'no signed zeros' is
        // applicable via an instruction-level fast-math-flag or some other
        // indicator that relaxed FP semantics are being used.

        if (isa<ConstantFP>(Op1) && !cast<ConstantFP>(Op1)->isZero())
          Worklist.emplace_back(Op0, Op1);
      }

      // If "A >= B" is known true, replace "A < B" with false
      // everywhere.
      // Since we don't have the instruction "A < B" immediately to
      // hand, work out the value number that it would have and use
      // that to find an appropriate instruction (if any).
      CmpInst::Predicate NotPred = Cmp->getInversePredicate();
      Constant *NotVal = ConstantInt::get(Cmp->getType(), isKnownFalse);
      BasicExpression *E =
          createCmpExpression(Cmp->getOpcode(), Cmp->getType(), NotPred, Op0,
                              Op1, Cmp->getParent());

      // We cannot directly propagate into the congruence class members
      // unless it is a singleton class.
      // This is because at least one of the members may later get
      // split out of the class.

      CongruenceClass *CC = ExpressionToClass.lookup(E);
      E->deallocateArgs(ArgRecycler);
      ExpressionAllocator.Deallocate(E);

      // If we didn't find a congruence class, there is no equivalent
      // instruction already
      if (CC) {
        // FIXME: I think this can be deleted now, need to bootstrap with an
        // assert
        if (CC->members.size() == 1 && 0) {
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
      // The problem in our world is that if nothing has this value
      // and this expression never appears, we would end up with a
      // class that has no leader (this value can't be a leader for
      // all members), no members, etc.

      // Ensure that any instruction in scope that gets the "A < B" value
      // number
      // is replaced with false.
      // The leader table only tracks basic blocks, not edges. Only add to if
      // we
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
static bool isOnlyReachableViaThisEdge(BasicBlock *Src, BasicBlock *Dst) {
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
    TouchedInstructions.set(InstrDFS[User]);
  }
}

/// performCongruenceFinding - Perform congruence finding on a given
/// value numbering expression
void NewGVN::performCongruenceFinding(Value *V, Expression *E) {
  // This is guaranteed to return something, since it will at least find
  // INITIAL
  CongruenceClass *VClass = ValueToClass[V];
  assert(VClass && "Should have found a vclass");
  // Dead classes should have been eliminated from the mapping
  assert(!VClass->dead && "Found a dead class");

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
  } else if (VariableExpression *VE = dyn_cast<VariableExpression>(E)) {
    EClass = ValueToClass[VE->getVariableValue()];
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
      DEBUG(dbgs() << "Created new congruence class for " << *V
                   << " using expression " << *E << " at " << NewClass->id
                   << "\n");
    } else {
      EClass = lookupResult.first->second;
      assert(EClass && "Somehow don't have an eclass");

      assert(!EClass->dead && "We accidentally looked up a dead class");
    }
  }
  bool WasInChanged = ChangedValues.erase(V);
  if (VClass != EClass || WasInChanged) {
    DEBUG(dbgs() << "Found class " << EClass->id << " for expression " << E
                 << "\n");

    if (VClass != EClass) {
      DEBUG(dbgs() << "New congruence class for " << V << " is " << EClass->id
                   << "\n");

      if (E && isa<CoercibleLoadExpression>(E)) {
        CoercibleLoadExpression *L = cast<CoercibleLoadExpression>(E);
        VClass->coercible_members.erase(V);
        EClass->coercible_members.insert(V);
        CoercionInfo.insert(
            std::make_pair(V, std::make_pair(L->getOffset(), L->getSrc())));
      } else {
        VClass->members.erase(V);
        EClass->members.insert(V);
      }

      ValueToClass[V] = EClass;
      // See if we destroyed the class or need to swap leaders
      if ((VClass->members.empty() && VClass->coercible_members.empty()) &&
          VClass != InitialClass) {
        if (VClass->expression) {
          VClass->dead = true;

          DEBUG(dbgs() << "Erasing expression " << *E << " from table\n");

          ExpressionToClass.erase(VClass->expression);
          MemoryExpressionToClass.erase(VClass->expression);
        }
        // delete VClass;
      } else if (VClass->leader == V) {
        // TODO: Check what happens if expression represented the leader
        VClass->leader = *(VClass->members.begin());
        for (auto M : VClass->members) {
          if (Instruction *I = dyn_cast<Instruction>(M))
            TouchedInstructions.set(InstrDFS[I]);
          ChangedValues.insert(M);
        }
        for (auto EM : VClass->coercible_members) {
          if (Instruction *I = dyn_cast<Instruction>(EM))
            TouchedInstructions.set(InstrDFS[I]);
          ChangedValues.insert(EM);
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
      const auto &InstRange = BlockInstRange.lookup(To);
      TouchedInstructions.set(InstRange.first, InstRange.second);
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
        TouchedInstructions.set(InstrDFS[BI]);
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
      propagateChangeInEdge(To);
    }
  }
}

// findConditionEquivalence - Given a predicate condition (from a
// switch, cmp, or whatever) and a block, see if we know some constant
// value for it already
Value *NewGVN::findConditionEquivalence(Value *Cond, BasicBlock *B) const {
  Value *Result = lookupOperandLeader(Cond, B);
  if (isa<Constant>(Result))
    return Result;

  return nullptr;
}

//  processOutgoingEdges - Process the outgoing edges of a block for
//  reachability.
void NewGVN::processOutgoingEdges(TerminatorInst *TI, BasicBlock *B) {
  // Evaluate Reachability of terminator instruction
  // Conditional branch
  BranchInst *BR;
  if ((BR = dyn_cast<BranchInst>(TI)) && BR->isConditional()) {
    Value *Cond = BR->getCondition();
    Value *CondEvaluated = findConditionEquivalence(Cond, B);
    if (!CondEvaluated) {
      if (Instruction *I = dyn_cast<Instruction>(Cond)) {
        Expression *E = createExpression(I, B);
        if (ConstantExpression *CE = dyn_cast<ConstantExpression>(E)) {
          CondEvaluated = CE->getConstantValue();
        }
      } else if (isa<ConstantInt>(Cond)) {
        CondEvaluated = Cond;
      }
    }

    ConstantInt *CI;
    BasicBlock *TrueSucc = BR->getSuccessor(0);
    BasicBlock *FalseSucc = BR->getSuccessor(1);
    if (CondEvaluated &&(CI = dyn_cast<ConstantInt>(CondEvaluated))) {
      if (CI->isOne()) {
        DEBUG(dbgs() << "Condition for Terminator " << *TI
                     << " evaluated to true\n");
        updateReachableEdge(B, TrueSucc);
      } else if (CI->isZero()) {
        DEBUG(dbgs() << "Condition for Terminator " << *TI
                     << " evaluated to false\n");
        updateReachableEdge(B, FalseSucc);
      }
    } else {
      if (isOnlyReachableViaThisEdge(B, TrueSucc))
        propagateEquality(Cond, ConstantInt::getTrue(TrueSucc->getContext()),
                          TrueSucc);

      if (isOnlyReachableViaThisEdge(B, FalseSucc))
        propagateEquality(Cond, ConstantInt::getFalse(FalseSucc->getContext()),
                          FalseSucc);
      updateReachableEdge(B, TrueSucc);
      updateReachableEdge(B, FalseSucc);
    }
  } else if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
    // For switches, propagate the case values into the case
    // destinations.

    Value *SwitchCond = SI->getCondition();
    Value *CondEvaluated = findConditionEquivalence(SwitchCond, B);
    // See if we were able to turn this switch statement into a constant
    if (CondEvaluated && isa<ConstantInt>(CondEvaluated)) {
      ConstantInt *CondVal = cast<ConstantInt>(CondEvaluated);
      // We should be able to get case value for this
      auto CaseVal = SI->findCaseValue(CondVal);
      // Now get where it goes and mark it reachable
      BasicBlock *TargetBlock = CaseVal.getCaseSuccessor();
      updateReachableEdge(B, TargetBlock);
    } else {
      for (unsigned i = 0, e = SI->getNumSuccessors(); i != e; ++i) {
        BasicBlock *TargetBlock = SI->getSuccessor(i);
        updateReachableEdge(B, TargetBlock);
      }
    }

    // Regardless of answers, propagate equalities for case values
    for (auto i = SI->case_begin(), e = SI->case_end(); i != e; ++i) {
      BasicBlock *TargetBlock = i.getCaseSuccessor();
      if (isOnlyReachableViaThisEdge(B, TargetBlock))
        propagateEquality(SwitchCond, i.getCaseValue(), TargetBlock);
    }
  } else {
    // Otherwise this is either unconditional, or a type we have no
    // idea about. Just mark successors as reachable
    for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i) {
      BasicBlock *TargetBlock = TI->getSuccessor(i);
      updateReachableEdge(B, TargetBlock);
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
    if (!UniquePredecessorBlocks.count(B)) {
      for (auto &I : *B) {
        if (!isa<PHINode>(&I))
          break;
        TouchedInstructions.set(InstrDFS[&I]);
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
    createSingletonCongruenceClass(&FA);
}

void NewGVN::cleanupTables() {

  ValueToClass.clear();
  for (unsigned i = 0, e = CongruenceClasses.size(); i != e; ++i) {
    DEBUG(dbgs() << "Congruence class " << CongruenceClasses[i]->id << " has "
                 << CongruenceClasses[i]->members.size() << " members\n");
    delete CongruenceClasses[i];

    CongruenceClasses[i] = NULL;
  }

  ArgRecycler.clear(ExpressionAllocator);
  ExpressionAllocator.Reset();
  CongruenceClasses.clear();
  ExpressionToClass.clear();
  MemoryExpressionToClass.clear();
  UniquedExpressions.clear();
  ReachableBlocks.clear();
  ReachableEdges.clear();
  ProcessedCount.clear();
  ProcessedBlockCount.clear();
  DFSBBMap.clear();
  InstrDFS.clear();
  InstructionsToErase.clear();
  UniquePredecessorBlocks.clear();
  DFSToInstr.clear();
  BlockInstRange.clear();
  TouchedInstructions.clear();
  CoercionInfo.clear();
  CoercionForwarding.clear();
}

std::pair<unsigned, unsigned> NewGVN::assignDFSNumbers(BasicBlock *B,
                                                       unsigned Start) {
  unsigned int Count = Start;
  for (auto &I : *B) {
    InstrDFS[&I] = Count++;
    DFSToInstr.emplace_back(&I);
  }
  // All of the range functions taken half-open ranges (open on the
  // end side), we adjust endcount by 1 here so that we can use
  // standard looking loops to deal with the range, but note that
  // the end here is *1 past the real end*
  return std::make_pair(Start, Count);
}

/// runOnFunction - This is the main transformation entry point for a
/// function.
bool NewGVN::runOnFunction(Function &F) {
  bool Changed = false;
  if (skipOptnoneFunction(F))
    return false;

  MD = &getAnalysis<MemoryDependenceAnalysis>();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  DL = &F.getParent()->getDataLayout();
  AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  TLI = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  AA = &getAnalysis<AliasAnalysis>();
  MSSA = &getAnalysis<MemorySSALazy>().getMSSA();
  MSSAWalker = new CachingMemorySSAWalker(MSSA, AA);
  MSSA->buildMemorySSA(AA, DT, MSSAWalker);

  unsigned int ICount = 0;
  // Count number of instructions for sizing of hash tables, and come
  // up with a global dfs numbering for instructions
  ReversePostOrderTraversal<Function *> rpoT(&F);
  for (auto &B : rpoT) {
    const auto &BlockRange = assignDFSNumbers(B, ICount);
    BlockInstRange.insert(std::make_pair(B, BlockRange));
    ICount += BlockRange.second - BlockRange.first;
  }

  // Handle forward unreachable blocks and figure out which blocks
  // have single preds

  for (auto &B : F) {
    // Assign numbers to unreachable blocks
    if (&B != &F.getEntryBlock() && pred_empty(&B)) {
      const auto &BlockRange = assignDFSNumbers(&B, ICount);
      BlockInstRange.insert(std::make_pair(&B, BlockRange));
      ICount += BlockRange.second - BlockRange.first;
    } else if (B.getUniquePredecessor())
      UniquePredecessorBlocks.insert(&B);
  }

  TouchedInstructions.resize(ICount + 1);

  // Ensure we don't end up resizing the expressionToClass map, as
  // that can be quite expensive. At most, we have one expression per
  // instruction.
  ExpressionToClass.resize(ICount + 1);
  MemoryExpressionToClass.resize(ICount + 1);
  // Initialize the touched instructions to include the entry block
  const auto &InstRange = BlockInstRange.lookup(&F.getEntryBlock());
  TouchedInstructions.set(InstRange.first, InstRange.second);
  ReachableBlocks.insert(&F.getEntryBlock());

  initializeCongruenceClasses(F);

  // We start out in the entry block
  BasicBlock *LastBlock = &F.getEntryBlock();
  while (TouchedInstructions.any()) {
    // Walk through all the instructions in all the blocks in RPO
    for (int InstrNum = TouchedInstructions.find_first(); InstrNum != -1;
         InstrNum = TouchedInstructions.find_next(InstrNum)) {
      Instruction *I = DFSToInstr[InstrNum];
      BasicBlock *CurrBlock = I->getParent();

      // If we hit a new block, do reachability processing
      if (CurrBlock != LastBlock) {
        LastBlock = CurrBlock;
        bool BlockReachable = ReachableBlocks.count(CurrBlock);
        const auto &InstRange = BlockInstRange.lookup(CurrBlock);
        // If it's not reachable, erase any touched instructions and
        // move on
        if (!BlockReachable) {
          TouchedInstructions.reset(InstRange.first, InstRange.second);
          DEBUG(dbgs() << "Skipping instructions in block "
                       << getBlockName(CurrBlock)
                       << " because it is unreachable\n");
          continue;
        }
        // #ifndef NDEBUG
        if (ProcessedBlockCount.count(CurrBlock) == 0) {
          ProcessedBlockCount.insert(std::make_pair(CurrBlock, 1));
        } else {
          ProcessedBlockCount[CurrBlock] += 1;
          assert(ProcessedBlockCount[CurrBlock] < 100 &&
                 "Seem to have processed the same block a lot\n");
          if (ProcessedBlockCount[CurrBlock] >= 100)
            report_fatal_error("Processed block too many times");
        }
        // #endif
      }
      TouchedInstructions.reset(InstrNum);

      DEBUG(dbgs() << "Processing instruction " << *I << "\n");
      if (I->use_empty() && !I->getType()->isVoidTy()) {
        DEBUG(dbgs() << "Skipping unused instruction\n");
        continue;
      }

// This is done in case something eliminates the instruction
// along the way.

#ifndef NDEBUG
      if (ProcessedCount.count(I) == 0) {
        ProcessedCount.insert(std::make_pair(I, 1));
      } else {
        ProcessedCount[I] += 1;
        assert(ProcessedCount[I] < 100 &&
               "Seem to have processed the same instruction a lot");
      }
#endif
      if (!I->isTerminator()) {
        Expression *Symbolized = performSymbolicEvaluation(I, CurrBlock);
        performCongruenceFinding(I, Symbolized);
      } else {
        processOutgoingEdges(dyn_cast<TerminatorInst>(I), CurrBlock);
      }
    }
  }

  Changed |= eliminateInstructions(F);

  // Delete all instructions marked for deletion.
  for (Instruction *ToErase : InstructionsToErase) {
    if (!ToErase->use_empty())
      ToErase->replaceAllUsesWith(UndefValue::get(ToErase->getType()));

    ToErase->eraseFromParent();
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
  // Only one of these will be set
  Value *Val;
  Use *U;
  bool Equivalence;
  bool Coercible;

  bool operator<(const ValueDFS &other) const {
    // It's not enough that any given field be less than - we have sets
    // of fields that need to be evaluated together to give a proper ordering
    // for example, if you have
    // DFS (1, 3)
    // Val 0
    // DFS (1, 2)
    // Val 50
    // We want the second to be less than the first, but if we just go field
    // by field, we will get to Val 0 < Val 50 and say the first is less than
    // the second. We only want it to be less than if the DFS orders are equal.

    if (DFSIn < other.DFSIn)
      return true;
    else if (DFSIn == other.DFSIn) {
      if (DFSOut < other.DFSOut)
        return true;
      else if (DFSOut == other.DFSOut) {
        if (LocalNum < other.LocalNum)
          return true;
        else if (LocalNum == other.LocalNum) {
          if (!!Equivalence < !!other.Equivalence)
            return true;
          if (!!Coercible < !!other.Coercible)
            return true;
          if (Val < other.Val)
            return true;
          if (U < other.U)
            return true;
        }
      }
    }
    return false;
  }
};

void NewGVN::convertDenseToDFSOrdered(CongruenceClass::MemberSet &Dense,
                                      std::vector<ValueDFS> &DFSOrderedSet,
                                      bool Coercible) {
  for (auto D : Dense) {
    // First add the value
    BasicBlock *BB = getBlockForValue(D);
    // Constants are handled prior to ever calling this function, so
    // we should only be left with instructions as members
    assert(BB || "Should have figured out a basic block for value");
    ValueDFS VD;

    VD.Equivalence = false;
    std::pair<int, int> DFSPair = DFSBBMap[BB];
    VD.DFSIn = DFSPair.first;
    VD.DFSOut = DFSPair.second;
    VD.U = nullptr;
    VD.Val = D;
    VD.Coercible = Coercible;
    // If it's an instruction, use the real local dfs number,
    if (Instruction *I = dyn_cast<Instruction>(D))
      VD.LocalNum = InstrDFS[I];
    else
      llvm_unreachable("Should have been an instruction");

    DFSOrderedSet.push_back(VD);
    // Now add the users
    for (auto &U : D->uses()) {
      if (Instruction *I = dyn_cast<Instruction>(U.getUser())) {
        ValueDFS VD;
        VD.Equivalence = false;
        VD.Coercible = Coercible;
        // Put the phi node uses in the incoming block
        BasicBlock *IBlock;
        if (PHINode *P = dyn_cast<PHINode>(I)) {
          IBlock = P->getIncomingBlock(U);
          // Make phi node users appear last in the incoming block
          // they are from.
          VD.LocalNum = InstrDFS.size() + 1;
        } else {
          IBlock = I->getParent();
          VD.LocalNum = InstrDFS[I];
        }
        std::pair<int, int> DFSPair = DFSBBMap[IBlock];
        VD.DFSIn = DFSPair.first;
        VD.DFSOut = DFSPair.second;
        VD.U = &U;
        VD.Val = nullptr;
        DFSOrderedSet.push_back(VD);
      }
    }
  }
}

void NewGVN::convertDenseToDFSOrdered(CongruenceClass::EquivalenceSet &Dense,
                                      std::vector<ValueDFS> &DFSOrderedSet) {
  for (auto D : Dense) {
    std::pair<int, int> &DFSPair = DFSBBMap[D.second];
    ValueDFS VD;
    VD.DFSIn = DFSPair.first;
    VD.DFSOut = DFSPair.second;
    VD.Equivalence = true;
    VD.Coercible = false;

    // If it's an instruction, use the real local dfs number.
    // If it's a value, it *must* have come from equality propagation,
    // and thus we know it is valid for the entire block.  By giving
    // the local number as -1, it should sort before the instructions
    // in that block.
    if (Instruction *I = dyn_cast<Instruction>(D.first))
      VD.LocalNum = InstrDFS[I];
    else
      VD.LocalNum = -1;

    VD.Val = D.first;
    DFSOrderedSet.push_back(VD);
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

  // Delete the instructions backwards, as it has a reduced likelihood of
  // having
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
  InstructionsToErase.insert(I);
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
    ValueStack.emplace_back(V);
    DFSStack.emplace_back(DFSIn, DFSOut);
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

/// CanCoerceMustAliasedValueToLoad - Return true if
/// CoerceAvailableValueToLoadType will succeed.
bool NewGVN::canCoerceMustAliasedValueToLoad(Value *StoredVal, Type *LoadTy) {

  // If the loaded or stored value is an first class array or struct, don't
  // try
  // to transform them.  We need to be able to bitcast to integer.
  if (LoadTy->isStructTy() || LoadTy->isArrayTy() ||
      StoredVal->getType()->isStructTy() || StoredVal->getType()->isArrayTy())
    return false;

  // The store has to be at least as big as the load.
  if (DL->getTypeSizeInBits(StoredVal->getType()) <
      DL->getTypeSizeInBits(LoadTy))
    return false;

  return true;
}

/// CoerceAvailableValueToLoadType - If we saw a store of a value to memory,
/// and
/// then a load from a must-aliased pointer of a different type, try to coerce
/// the stored value.  LoadedTy is the type of the load we want to replace and
/// InsertPt is the place to insert new instructions.
///
/// If we can't do it, return null.
Value *NewGVN::coerceAvailableValueToLoadType(Value *StoredVal, Type *LoadedTy,
                                              Instruction *InsertPt) {

  if (!canCoerceMustAliasedValueToLoad(StoredVal, LoadedTy))
    return 0;

  // If this is already the right type, just return it.
  Type *StoredValTy = StoredVal->getType();

  uint64_t StoreSize = DL->getTypeSizeInBits(StoredValTy);
  uint64_t LoadSize = DL->getTypeSizeInBits(LoadedTy);

  // If the store and reload are the same size, we can always reuse it.
  if (StoreSize == LoadSize) {
    // Pointer to Pointer -> use bitcast.
    if (StoredValTy->isPointerTy() && LoadedTy->isPointerTy()) {
      Instruction *I = new BitCastInst(StoredVal, LoadedTy, "", InsertPt);
      handleNewInstruction(I);
      return I;
    }

    // Convert source pointers to integers, which can be bitcast.
    if (StoredValTy->isPointerTy()) {
      StoredValTy = DL->getIntPtrType(StoredValTy->getContext());
      Instruction *I = new PtrToIntInst(StoredVal, StoredValTy, "", InsertPt);
      StoredVal = I;
      handleNewInstruction(I);
    }

    Type *TypeToCastTo = LoadedTy;
    if (TypeToCastTo->isPointerTy())
      TypeToCastTo = DL->getIntPtrType(StoredValTy->getContext());

    if (StoredValTy != TypeToCastTo) {
      Instruction *I = new BitCastInst(StoredVal, TypeToCastTo, "", InsertPt);
      StoredVal = I;
      handleNewInstruction(I);
    }

    // Cast to pointer if the load needs a pointer type.
    if (LoadedTy->isPointerTy()) {
      Instruction *I = new IntToPtrInst(StoredVal, LoadedTy, "", InsertPt);
      StoredVal = I;
      handleNewInstruction(I);
    }
    return StoredVal;
  }

  // If the loaded value is smaller than the available value, then we can
  // extract out a piece from it.  If the available value is too small, then
  // we
  // can't do anything.
  assert(StoreSize >= LoadSize && "CanCoerceMustAliasedValueToLoad fail");

  // Convert source pointers to integers, which can be manipulated.
  if (StoredValTy->isPointerTy()) {
    StoredValTy = DL->getIntPtrType(StoredValTy->getContext());
    Instruction *I = new PtrToIntInst(StoredVal, StoredValTy, "", InsertPt);
    StoredVal = I;
    handleNewInstruction(I);
  }

  // Convert vectors and fp to integer, which can be manipulated.
  if (!StoredValTy->isIntegerTy()) {
    StoredValTy = IntegerType::get(StoredValTy->getContext(), StoreSize);
    Instruction *I = new BitCastInst(StoredVal, StoredValTy, "", InsertPt);
    StoredVal = I;
    handleNewInstruction(I);
  }

  // If this is a big-endian system, we need to shift the value down to the
  // low
  // bits so that a truncate will work.
  if (DL->isBigEndian()) {
    Constant *Val =
        ConstantInt::get(StoredVal->getType(), StoreSize - LoadSize);
    StoredVal = BinaryOperator::CreateLShr(StoredVal, Val, "tmp", InsertPt);
    if (Instruction *I = dyn_cast<Instruction>(StoredVal))
      handleNewInstruction(I);
  }

  // Truncate the integer to the right size now.
  Type *NewIntTy = IntegerType::get(StoredValTy->getContext(), LoadSize);
  Instruction *I = new TruncInst(StoredVal, NewIntTy, "trunc", InsertPt);
  StoredVal = I;
  handleNewInstruction(I);

  if (LoadedTy == NewIntTy)
    return StoredVal;

  // If the result is a pointer, inttoptr.
  if (LoadedTy->isPointerTy()) {
    I = new IntToPtrInst(StoredVal, LoadedTy, "inttoptr", InsertPt);
    handleNewInstruction(I);
    return I;
  }

  // Otherwise, bitcast.
  I = new BitCastInst(StoredVal, LoadedTy, "bitcast", InsertPt);
  handleNewInstruction(I);
  return I;
}

/// GetStoreValueForLoad - This function is called when we have a
/// memdep query of a load that ends up being a clobbering store.  This means
/// that the store provides bits used by the load but we the pointers don't
/// mustalias.  Check this case to see if there is anything more we can do
/// before we give up.
Value *NewGVN::getStoreValueForLoad(Value *SrcVal, unsigned Offset,
                                    Type *LoadTy, Instruction *InsertPt) {

  LLVMContext &Ctx = SrcVal->getType()->getContext();

  uint64_t StoreSize = (DL->getTypeSizeInBits(SrcVal->getType()) + 7) / 8;
  uint64_t LoadSize = (DL->getTypeSizeInBits(LoadTy) + 7) / 8;

  IRBuilder<> Builder(InsertPt->getParent(), InsertPt);

  // Compute which bits of the stored value are being used by the load.
  // Convert
  // to an integer type to start with.
  if (SrcVal->getType()->isPointerTy()) {
    SrcVal = Builder.CreatePtrToInt(SrcVal, DL->getIntPtrType(Ctx));
    if (Instruction *I = dyn_cast<Instruction>(SrcVal))
      handleNewInstruction(I);
  }

  if (!SrcVal->getType()->isIntegerTy()) {
    SrcVal =
        Builder.CreateBitCast(SrcVal, IntegerType::get(Ctx, StoreSize * 8));
    if (Instruction *I = dyn_cast<Instruction>(SrcVal))
      handleNewInstruction(I);
  }
  // Shift the bits to the least significant depending on endianness.
  unsigned ShiftAmt;
  if (DL->isLittleEndian())
    ShiftAmt = Offset * 8;
  else
    ShiftAmt = (StoreSize - LoadSize - Offset) * 8;

  if (ShiftAmt) {
    SrcVal = Builder.CreateLShr(SrcVal, ShiftAmt);
    if (Instruction *I = dyn_cast<Instruction>(SrcVal))
      handleNewInstruction(I);
  }
  if (LoadSize != StoreSize) {
    SrcVal = Builder.CreateTrunc(SrcVal, IntegerType::get(Ctx, LoadSize * 8));
    if (Instruction *I = dyn_cast<Instruction>(SrcVal))
      handleNewInstruction(I);
  }
  return coerceAvailableValueToLoadType(SrcVal, LoadTy, InsertPt);
}

/// GetLoadValueForLoad - This function is called when we have a
/// memdep query of a load that ends up being a clobbering load.  This means
/// that the load *may* provide bits used by the load but we can't be sure
/// because the pointers don't mustalias.  Check this case to see if there is
/// anything more we can do before we give up.
Value *NewGVN::getLoadValueForLoad(LoadInst *SrcVal, unsigned Offset,
                                   Type *LoadTy, Instruction *InsertPt) {
  // If Offset+LoadTy exceeds the size of SrcVal, then we must be wanting to
  // widen SrcVal out to a larger load.
  unsigned SrcValSize = DL->getTypeStoreSize(SrcVal->getType());
  unsigned LoadSize = DL->getTypeStoreSize(LoadTy);
  if (Offset + LoadSize > SrcValSize) {
    assert(SrcVal->isSimple() && "Cannot widen volatile/atomic load!");
    assert(SrcVal->getType()->isIntegerTy() && "Can't widen non-integer load");
    // If we have a load/load clobber an DepLI can be widened to cover this
    // load, then we should widen it to the next power of 2 size big enough!
    unsigned NewLoadSize = Offset + LoadSize;
    if (!isPowerOf2_32(NewLoadSize))
      NewLoadSize = NextPowerOf2(NewLoadSize);

    Value *PtrVal = SrcVal->getPointerOperand();

    // Insert the new load after the old load.  This ensures that subsequent
    // memdep queries will find the new load.  We can't easily remove the old
    // load completely because it is already in the value numbering table.
    IRBuilder<> Builder(SrcVal->getParent(), ++BasicBlock::iterator(SrcVal));
    Type *DestPTy = IntegerType::get(LoadTy->getContext(), NewLoadSize * 8);
    DestPTy =
        PointerType::get(DestPTy, PtrVal->getType()->getPointerAddressSpace());
    Builder.SetCurrentDebugLocation(SrcVal->getDebugLoc());
    PtrVal = Builder.CreateBitCast(PtrVal, DestPTy);
    if (Instruction *I = dyn_cast<Instruction>(PtrVal))
      handleNewInstruction(I);
    LoadInst *NewLoad = Builder.CreateLoad(PtrVal);
    NewLoad->takeName(SrcVal);
    NewLoad->setAlignment(SrcVal->getAlignment());
    handleNewInstruction(NewLoad);
    DEBUG(dbgs() << "GVN WIDENED LOAD: " << *SrcVal << "\n");
    DEBUG(dbgs() << "TO: " << *NewLoad << "\n");
    // This ensures we forward other coercions onto the new load, instead of the
    // old one
    CoercionForwarding[SrcVal] = NewLoad;

    // Replace uses of the original load with the wider load.  On a big endian
    // system, we need to shift down to get the relevant bits.
    Value *RV = NewLoad;
    if (DL->isBigEndian()) {
      RV = Builder.CreateLShr(
          RV, NewLoadSize * 8 - SrcVal->getType()->getPrimitiveSizeInBits());
      if (Instruction *I = dyn_cast<Instruction>(RV))
        handleNewInstruction(I);
    }

    RV = Builder.CreateTrunc(RV, SrcVal->getType());
    if (Instruction *I = dyn_cast<Instruction>(RV))
      handleNewInstruction(I);

    // markUsersTouched(SrcVal);
    // assert(false && "Need to debug this");
    // So, we just widened a load that we will have already gone past in
    // elimination, so in order to get rid of the uses, we have to do it here

    SrcVal->replaceAllUsesWith(RV);

    // We would like to use gvn.markInstructionForDeletion here, but we can't
    // because the load is already memoized into the leader map table that GVN
    // tracks.  It is potentially possible to remove the load from the table,
    // but then there all of the operations based on it would need to be
    // rehashed.  Just leave the dead load around.
    // FIXME: This is no longer a problem
    markInstructionForDeletion(SrcVal);
    SrcVal = NewLoad;
  }

  return getStoreValueForLoad(SrcVal, Offset, LoadTy, InsertPt);
}

/// GetMemInstValueForLoad - This function is called when we have a
/// memdep query of a load that ends up being a clobbering mem intrinsic.
Value *NewGVN::getMemInstValueForLoad(MemIntrinsic *SrcInst, unsigned Offset,
                                      Type *LoadTy, Instruction *InsertPt) {

  LLVMContext &Ctx = LoadTy->getContext();
  uint64_t LoadSize = DL->getTypeSizeInBits(LoadTy) / 8;

  IRBuilder<> Builder(InsertPt->getParent(), InsertPt);

  // We know that this method is only called when the mem transfer fully
  // provides the bits for the load.
  if (MemSetInst *MSI = dyn_cast<MemSetInst>(SrcInst)) {
    // memset(P, 'x', 1234) -> splat('x'), even if x is a variable, and
    // independently of what the offset is.
    Value *Val = MSI->getValue();
    if (LoadSize != 1) {
      Val = Builder.CreateZExt(Val, IntegerType::get(Ctx, LoadSize * 8));
      if (Instruction *I = dyn_cast<Instruction>(Val))
        handleNewInstruction(I);
    }

    Value *OneElt = Val;

    // Splat the value out to the right number of bits.
    for (unsigned NumBytesSet = 1; NumBytesSet != LoadSize;) {
      // If we can double the number of bytes set, do it.
      if (NumBytesSet * 2 <= LoadSize) {
        Value *ShVal = Builder.CreateShl(Val, NumBytesSet * 8);
        if (Instruction *I = dyn_cast<Instruction>(ShVal))
          handleNewInstruction(I);
        Val = Builder.CreateOr(Val, ShVal);
        if (Instruction *I = dyn_cast<Instruction>(Val))
          handleNewInstruction(I);
        NumBytesSet <<= 1;
        continue;
      }

      // Otherwise insert one byte at a time.
      Value *ShVal = Builder.CreateShl(Val, 1 * 8);
      if (Instruction *I = dyn_cast<Instruction>(ShVal))
        handleNewInstruction(I);

      Val = Builder.CreateOr(OneElt, ShVal);
      if (Instruction *I = dyn_cast<Instruction>(Val))
        handleNewInstruction(I);

      ++NumBytesSet;
    }

    return coerceAvailableValueToLoadType(Val, LoadTy, InsertPt);
  }

  // Otherwise, this is a memcpy/memmove from a constant global.
  MemTransferInst *MTI = cast<MemTransferInst>(SrcInst);
  Constant *Src = cast<Constant>(MTI->getSource());
  unsigned AS = Src->getType()->getPointerAddressSpace();

  // Otherwise, see if we can constant fold a load from the constant with the
  // offset applied as appropriate.
  Src = ConstantExpr::getBitCast(
      Src, llvm::Type::getInt8PtrTy(Src->getContext(), AS));
  Constant *OffsetCst =
      ConstantInt::get(Type::getInt64Ty(Src->getContext()), (unsigned)Offset);
  Src = ConstantExpr::getGetElementPtr(Src, OffsetCst);
  Src = ConstantExpr::getBitCast(Src, PointerType::get(LoadTy, AS));
  return ConstantFoldLoadFromConstPtr(Src, *DL);
}

Value *NewGVN::coerceLoad(Value *V) {
  assert(isa<LoadInst>(V) && "Trying to coerce something other than a load");
  LoadInst *LI = cast<LoadInst>(V);
  // This is an offset, source pair
  const std::pair<unsigned, Value *> &Info = CoercionInfo.lookup(LI);
  Value *Result;
  Value *RealValue = Info.second;
  // Walk all the coercion fowarding chains, in case this load has already been
  // widened into another load
  while (true) {
    auto ForwardingResult = CoercionForwarding.find(RealValue);
    if (ForwardingResult != CoercionForwarding.end())
      RealValue = ForwardingResult->second;
    else
      break;
  }

  assert(DT->dominates(cast<Instruction>(RealValue), LI) &&
         "Trying to replace a load with one that doesn't dominate it");
  if (StoreInst *DepSI = dyn_cast<StoreInst>(RealValue))
    Result = getStoreValueForLoad(DepSI->getValueOperand(), Info.first,
                                  LI->getType(), LI);
  else if (LoadInst *DepLI = dyn_cast<LoadInst>(RealValue))
    Result = getLoadValueForLoad(DepLI, Info.first, LI->getType(), LI);
  else if (MemIntrinsic *DepMI = dyn_cast<MemIntrinsic>(RealValue))
    Result = getMemInstValueForLoad(DepMI, Info.first, LI->getType(), LI);
  else
    llvm_unreachable("Unknown coercion type");

  assert(Result && "Should have been able to coerce");
  DEBUG(dbgs() << "Coerced load " << *LI << " output is " << *Result << "\n");
  return Result;
}

bool NewGVN::eliminateInstructions(Function &F) {
  // This is a non-standard eliminator. The normal way to eliminate is
  // to walk the dominator tree in order, keeping track of available
  // values, and eliminating them.  However, this is mildly
  // pointless. It requires doing lookups on every instruction,
  // regardless of whether we will ever eliminate it.  For
  // instructions part of most singleton congruence class, we know we
  // will never eliminate it.

  // Instead, this eliminator looks at the congruence classes directly, sorts
  // them into a DFS ordering of the dominator tree, and then we just
  // perform eliminate straight on the sets by walking the congruence
  // class member uses in order, and eliminate the ones dominated by the
  // last member.   This is technically O(N log N) where N = number of
  // instructions (since in theory all instructions may be in the same
  // congruence class).
  // When we find something not dominated, it becomes the new leader
  // for elimination purposes

  // First, figure out DFS numbers and get rid of unreachable phi node values
  for (auto &B : F) {
    if (!ReachableBlocks.count(&B)) {
      for (auto S : successors(&B)) {
        for (auto II = S->begin(); isa<PHINode>(II); ++II) {
          PHINode &Phi = cast<PHINode>(*II);
          DEBUG(dbgs() << "Replacing incoming value of " << *II << " for block "
                       << getBlockName(&B)
                       << " with undef due to it being unreachable\n");
          for (auto &Operand : Phi.incoming_values())
            if (Phi.getIncomingBlock(Operand) == &B)
              Operand.set(UndefValue::get(Phi.getType()));
        }
      }
    }
    DomTreeNode *DTN = DT->getNode(&B);
    if (!DTN)
      continue;
    DFSBBMap[&B] = std::make_pair(DTN->getDFSNumIn(), DTN->getDFSNumOut());
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

    // If this is a leader that is always available, and it's a
    // constant or has no equivalences, just replace everything with
    // it. We then update the congruence class with whatever members
    // are left.
    if (alwaysAvailable(CC->leader)) {
      SmallPtrSet<Value *, 4> MembersLeft;
      for (auto M : CC->members) {

        Value *Member = M;
        for (auto &Equiv : CC->equivalences)
          replaceAllDominatedUsesWith(M, Equiv.first, Equiv.second);

        // Void things have no uses we can replace
        if (Member == CC->leader || Member->getType()->isVoidTy()) {
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
      // If this is a singleton, with no equivalences, we can skip it
      if (CC->members.size() != 1 || !CC->equivalences.empty() ||
          !CC->coercible_members.empty()) {
        // If it's a singleton with equivalences, just do equivalence
        // replacement and move on
        if (CC->members.size() == 1 && 0) {
          for (auto &Equiv : CC->equivalences)
            replaceAllDominatedUsesWith(CC->leader, Equiv.first, Equiv.second);
          continue;
        }

        // This is a stack because equality replacement/etc may place
        // constants in the middle of the member list, and we want to use
        // those constant values in preference to the current leader, over
        // the scope of those constants.

        ValueDFSStack EliminationStack;

        // Convert the members and equivalences to DFS ordered sets and
        // then merge them.
        std::vector<ValueDFS> DFSOrderedSet;
        convertDenseToDFSOrdered(CC->members, DFSOrderedSet, false);
        convertDenseToDFSOrdered(CC->coercible_members, DFSOrderedSet, true);
        // During value numbering, we already proceed as if the
        // equivalences have been propagated through, but this is the
        // only place we actually do elimination (so that other passes
        // know the same thing we do)

        convertDenseToDFSOrdered(CC->equivalences, DFSOrderedSet);
        // Sort the whole thing
        sort(DFSOrderedSet.begin(), DFSOrderedSet.end());

        for (auto &C : DFSOrderedSet) {
          int MemberDFSIn = C.DFSIn;
          int MemberDFSOut = C.DFSOut;
          Value *Member = C.Val;
          Use *MemberUse = C.U;
          bool EquivalenceOnly = C.Equivalence;
          bool Coercible = C.Coercible;

          // We ignore void things because we can't get a value from
          // them.
          if (Member && Member->getType()->isVoidTy())
            continue;

          if (EliminationStack.empty()) {
            DEBUG(dbgs() << "Elimination Stack is empty\n");
          } else {
            DEBUG(dbgs() << "Elimination Stack Top DFS numbers are ("
                         << EliminationStack.dfs_back().first << ","
                         << EliminationStack.dfs_back().second << ")\n");
          }
          if (Member && isa<Constant>(Member))
            assert(isa<Constant>(CC->leader) || EquivalenceOnly);

          DEBUG(dbgs() << "Current DFS numbers are (" << MemberDFSIn << ","
                       << MemberDFSOut << ")\n");
          // First, we see if we are out of scope or empty.  If so,
          // and there equivalences, we try to replace the top of
          // stack with equivalences (if it's on the stack, it must
          // not have been eliminated yet)
          // Then we synchronize to our current scope, by
          // popping until we are back within a DFS scope that
          // dominates the current member.
          // Then, what happens depends on a few factors
          // If the stack is now empty, we need to push
          // If we have a constant or a local equivalence we want to
          // start using, we also push
          // Otherwise, we walk along, processing members who are
          // dominated by this scope, and eliminate them
          bool ShouldPush =
              Member && (EliminationStack.empty() || isa<Constant>(Member) ||
                         EquivalenceOnly);
          bool OutOfScope =
              !EliminationStack.isInScope(MemberDFSIn, MemberDFSOut);

          if (OutOfScope || ShouldPush) {
            // Sync to our current scope
            EliminationStack.popUntilDFSScope(MemberDFSIn, MemberDFSOut);
            // Push if we need to
            ShouldPush |= Member && EliminationStack.empty();
            if (ShouldPush) {
              if (Coercible)
                Member = coerceLoad(Member);
              EliminationStack.push_back(Member, MemberDFSIn, MemberDFSOut);
            }
          }

          // If we get to this point, and the stack is empty we must have a use
          // with nothing we can use to eliminate it, just skip it
          if (EliminationStack.empty())
            continue;

          // Skip the Value's, we only want to eliminate on their uses
          if (Member || EquivalenceOnly)
            continue;
          Value *Result = EliminationStack.back();

          // Don't replace our existing users with ourselves
          if (MemberUse->get() == Result)
            continue;

          DEBUG(dbgs() << "Found replacement " << *Result << " for "
                       << *MemberUse->get() << " in " << *(MemberUse->getUser())
                       << "\n");
          if (Instruction *Original = dyn_cast<Instruction>(Result))
            patchReplacementInstruction(Original, MemberUse->getUser());
          assert(isa<Instruction>(MemberUse->getUser()));
          MemberUse->set(Result);
        }
      }
    }
  }

  return true;
}
