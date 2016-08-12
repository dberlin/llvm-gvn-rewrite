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

#include "llvm/Transforms/Scalar/NewGVN.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/PHITransAddr.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/PredIteratorCache.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVNExpression.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/MemorySSA.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include <unordered_map>
#include <utility>
#include <vector>
using namespace llvm;
using namespace PatternMatch;
using namespace llvm::GVNExpression;

#define DEBUG_TYPE "newgvn"

STATISTIC(NumGVNInstrDeleted, "Number of instructions deleted");
STATISTIC(NumGVNBlocksDeleted, "Number of blocks deleted");
STATISTIC(NumGVNEqProp, "Number of equalities propagated");
STATISTIC(NumGVNOpsSimplified, "Number of Expressions simplified");
STATISTIC(NumGVNPhisAllSame, "Number of PHIs whos arguments are all the same");

//===----------------------------------------------------------------------===//
//                                GVN Pass
//===----------------------------------------------------------------------===//

// This represents a single equivalence found by equality propagation, for a
// single congruence class.
// That is, this is a thing the congruence class is also equivalent to when the
// member of the congruence class is dominated by either Edge.getEnd() (if EOnly
// is false) or Edge (if EOnly is true).
//
// For a concrete example, an Equivalence of {"constant int 0", "{block 0, block
// 1}", false}, means that any member of the congruence class is equal to
// constant int 0 if the member is dominated by block 1.

struct Equivalence {
  Value *Val;
  const BasicBlockEdge Edge;
  bool EdgeOnly;
  Equivalence(Value *V, const BasicBlockEdge E, bool EOnly)
      : Val(V), Edge(E), EdgeOnly(EOnly) {}
};

// These are equivalences that are valid for a single user.  This happens when
// we know particular edge equivalences are valid for some users, but can't
// prove they are valid for all users.
// Because we don't have control regions, when we have
// equivalences along edges to blocks with multiple predecessors, we can't add
// them to the general equivalences list, because they don't dominate a given
// root, but instead, they really represent something valid in certain control
// regions of the program, a concept that is not really expressible using the
// a standard dominator (and not postdominator) tree.
struct SingleUserEquivalence {
  User *U;
  unsigned OperandNo;
  Value *Replacement;
};

// Congruence classes represent the set of expressions/instructions
// that are all the same *during some scope in the function*.
// That is, because of the way we perform equality propagation, and
// because of memory value numbering, it is not correct to assume
// you can willy-nilly replace any member with any other at any
// point in the function.
//
// For any Value in the Member set, it is valid to replace any dominated member
// with that Value.
//
// Every congruence class has a leader, and the leader is used to
// symbolize instructions in a canonical way (IE every operand of an
// instruction that is a member of the same congruence class will
// always be replaced with leader during symbolization).
// To simplify symbolization, we keep the leader as a constant if class can be
// proved to be a constant value.
// Otherwise, the leader is a randomly chosen member of the value set, it does
// not matter which one is chosen.
// Each congruence class also has a defining expression,
// though the expression may be null.  If it exists, it can be used for forward
// propagation and reassociation of values.
//
struct CongruenceClass {
  typedef SmallPtrSet<Value *, 4> MemberSet;
  unsigned int ID;
  // Representative leader
  Value *RepLeader;
  // Defining Expression
  const Expression *DefiningExpr;
  // Actual members of this class.  These are the things the same everywhere
  MemberSet Members;
  // Coercible members of this class. These are loads where we can pull the
  // value out of a store. This means they need special processing during
  // elimination to do this, but they are otherwise the same as members,
  // in particular, we can eliminate one in favor of a dominating one.
  MemberSet CoercibleMembers;

  typedef std::list<Equivalence> EquivalenceSet;

  // See the comments on struct Equivalence and SingleUserEquivalence for a
  // discussion of what these are.

  EquivalenceSet Equivs;
  std::unordered_multimap<Value *, SingleUserEquivalence> SingleUserEquivs;

  // True if this class has no members left.  This is mainly used for assertion
  // purposes, and for skipping empty classes.
  bool Dead;

  explicit CongruenceClass(unsigned int ID)
      : ID(ID), RepLeader(0), DefiningExpr(0), Dead(false) {}
  CongruenceClass(unsigned int ID, Value *Leader, const Expression *E)
      : ID(ID), RepLeader(Leader), DefiningExpr(E), Dead(false) {}
};

class NewGVN : public FunctionPass {
  MemoryDependenceResults *MD;
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
  CongruenceClass *InitialClass;
  std::vector<CongruenceClass *> CongruenceClasses;
  unsigned int NextCongruenceNum = 0;

  // Value Mappings
  DenseMap<Value *, CongruenceClass *> ValueToClass;
  DenseMap<Value *, const Expression *> ValueToExpression;

  struct ComparingExpressionInfo : public DenseMapInfo<const Expression *> {
    static unsigned getHashValue(const Expression *V) {
      return static_cast<unsigned>(V->getHashValue());
    }
    static bool isEqual(const Expression *LHS, const Expression *RHS) {
      if (LHS == RHS)
        return true;
      if (LHS == getTombstoneKey() || RHS == getTombstoneKey() ||
          LHS == getEmptyKey() || RHS == getEmptyKey())
        return false;
      return *LHS == *RHS;
    }
  };

  struct expression_equal_to {
    bool operator()(const Expression *A, const Expression *B) const {
      if (A == B)
        return true;
      return *A == *B;
    }
  };
  struct hash_expression {
    size_t operator()(const Expression *A) const { return A->getHashValue(); }
  };

  // This multimap holds equivalences that we've detected that we haven't see
  // users in the program for, yet. If we later see a user, we will add these to
  // it's equivalences
  std::unordered_multimap<const Expression *, Equivalence, hash_expression,
                          expression_equal_to>
      PendingEquivalences;

  // Expression to class mapping
  typedef DenseMap<const Expression *, CongruenceClass *,
                   ComparingExpressionInfo>
      ExpressionClassMap;
  ExpressionClassMap ExpressionToClass;

  // Which values have changed as a result of leader changes
  SmallPtrSet<Value *, 8> ChangedValues;

  // Reachability info
  DenseSet<std::pair<BasicBlock *, BasicBlock *>> ReachableEdges;
  SmallPtrSet<const BasicBlock *, 8> ReachableBlocks;
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
  // We mark which instructions were affected by an equivalence, so we only have
  // to revisit those instructions when an edge change occurs
  BitVector InvolvedInEquivalence;
  DenseMap<const BasicBlock *, std::pair<unsigned, unsigned>> BlockInstRange;
  DenseMap<const DomTreeNode *, std::pair<unsigned, unsigned>>
      DominatedInstRange;
  // Debugging for how many times each block and instruction got processed
  DenseMap<const Value *, unsigned> ProcessedCount;

  // DFS info
  DenseMap<const BasicBlock *, std::pair<int, int>> DFSDomMap;
  DenseMap<const Instruction *, unsigned> InstrDFS;
  std::vector<Instruction *> DFSToInstr;

  // Deletion info
  SmallPtrSet<Instruction *, 8> InstructionsToErase;

public:
  static char ID; // Pass identification, replacement for typeid
  explicit NewGVN() : FunctionPass(ID), MD(nullptr) {
    initializeNewGVNPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;
  bool runGVN(Function &F, DominatorTree *DT, AssumptionCache *AC,
              TargetLibraryInfo *TLI, AliasAnalysis *AA,
              MemorySSA *MSSA);

private:
  // This transformation requires dominator postdominator info
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<MemorySSAWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();

    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }

  // expression handling
  const Expression *createExpression(Instruction *, const BasicBlock *);
  const Expression *createBinaryExpression(unsigned, Type *, Value *, Value *,
                                           const BasicBlock *);
  bool setBasicExpressionInfo(Instruction *, BasicExpression *,
                              const BasicBlock *);
  PHIExpression *createPHIExpression(Instruction *);
  const VariableExpression *createVariableExpression(Value *, bool);
  const ConstantExpression *createConstantExpression(Constant *, bool);
  const Expression *createVariableOrConstant(Value *V, const BasicBlock *B);
  const StoreExpression *createStoreExpression(StoreInst *, MemoryAccess *,
                                               const BasicBlock *);
  LoadExpression *createLoadExpression(Type *, Value *, LoadInst *,
                                       MemoryAccess *, const BasicBlock *);
  const CoercibleLoadExpression *
  createCoercibleLoadExpression(Type *, Value *, LoadInst *, MemoryAccess *,
                                unsigned, Value *, const BasicBlock *);

  const CallExpression *createCallExpression(CallInst *, MemoryAccess *,
                                             const BasicBlock *);
  const AggregateValueExpression *
  createAggregateValueExpression(Instruction *, const BasicBlock *);

  const BasicExpression *createCmpExpression(unsigned, Type *,
                                             CmpInst::Predicate, Value *,
                                             Value *, const BasicBlock *);
  // Congruence class handling
  CongruenceClass *createCongruenceClass(Value *Leader, const Expression *E) {
    CongruenceClass *result =
        new CongruenceClass(NextCongruenceNum++, Leader, E);
    CongruenceClasses.emplace_back(result);
    return result;
  }

  CongruenceClass *createSingletonCongruenceClass(Value *Member) {
    CongruenceClass *CClass = createCongruenceClass(Member, NULL);
    CClass->Members.insert(Member);
    ValueToClass[Member] = CClass;
    return CClass;
  }
  void initializeCongruenceClasses(Function &F);

  // Symbolic evaluation
  const Expression *checkSimplificationResults(Expression *, Instruction *,
                                               Value *);
  const Expression *performSymbolicEvaluation(Value *, const BasicBlock *);
  const Expression *performSymbolicLoadCoercion(Type *, Value *, LoadInst *,
                                                Instruction *, MemoryAccess *,
                                                const BasicBlock *);
  const Expression *performSymbolicLoadEvaluation(Instruction *,
                                                  const BasicBlock *);
  const Expression *performSymbolicStoreEvaluation(Instruction *,
                                                   const BasicBlock *);
  const Expression *performSymbolicCallEvaluation(Instruction *,
                                                  const BasicBlock *);
  const Expression *performSymbolicPHIEvaluation(Instruction *,
                                                 const BasicBlock *);
  const Expression *performSymbolicAggrValueEvaluation(Instruction *,
                                                       const BasicBlock *);
  int analyzeLoadFromClobberingStore(Type *, Value *, StoreInst *);
  int analyzeLoadFromClobberingLoad(Type *, Value *, LoadInst *);
  int analyzeLoadFromClobberingMemInst(Type *, Value *, MemIntrinsic *);
  int analyzeLoadFromClobberingWrite(Type *, Value *, Value *, uint64_t);
  // Congruence finding
  // Templated to allow them to work both on BB's and BB-edges
  template <class T>
  std::pair<Value *, bool> lookupOperandLeader(Value *, const User *,
                                               const T &) const;
  template <class T>
  Value *findDominatingEquivalent(CongruenceClass *, const User *,
                                  const T &) const;
  // Predicate and reachability handling
  void updateReachableEdge(BasicBlock *, BasicBlock *);
  void processOutgoingEdges(TerminatorInst *, BasicBlock *);
  void propagateChangeInEdge(BasicBlock *);
  bool propagateEquality(Value *, Value *, bool, const BasicBlockEdge &);
  bool isOnlyReachableViaThisEdge(const BasicBlockEdge &);

  void markDominatedSingleUserEquivalences(CongruenceClass *, Value *, Value *,
                                           bool, const BasicBlockEdge &);
  Value *findConditionEquivalence(Value *, BasicBlock *) const;
  const std::pair<unsigned, unsigned>
  calculateDominatedInstRange(const DomTreeNode *);

  // Instruction replacement
  unsigned replaceAllDominatedUsesWith(Value *, Value *, const BasicBlockEdge &,
                                       bool);

  // Elimination
  void markInstructionForDeletion(Instruction *);
  void deleteInstructionsInBlock(BasicBlock *);

  // Utilities
  void cleanupTables();
  std::pair<unsigned, unsigned> assignDFSNumbers(BasicBlock *, unsigned);
  void updateProcessedCount(Value *V);
};

char NewGVN::ID = 0;

// createGVNPass - The public interface to this file...
FunctionPass *llvm::createNewGVNPass() { return new NewGVN(); }

#ifndef NDEBUG
static std::string getBlockName(const BasicBlock *B) {
  return DOTGraphTraits<const Function *>::getSimpleNodeLabel(B, NULL);
}
static std::string getBlockName(const BasicBlockEdge &B) {
  return DOTGraphTraits<const Function *>::getSimpleNodeLabel(B.getEnd(), NULL);
}

#endif
INITIALIZE_PASS_BEGIN(NewGVN, "newgvn", "Global Value Numbering", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(MemorySSAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(GlobalsAAWrapperPass)
INITIALIZE_PASS_END(NewGVN, "newgvn", "Global Value Numbering", false, false)
PHIExpression *NewGVN::createPHIExpression(Instruction *I) {
  BasicBlock *PhiBlock = I->getParent();
  PHINode *PN = cast<PHINode>(I);
  PHIExpression *E = new (ExpressionAllocator)
      PHIExpression(PN->getNumOperands(), I->getParent());

  E->allocateOperands(ArgRecycler, ExpressionAllocator);
  E->setType(I->getType());
  E->setOpcode(I->getOpcode());
  bool UsedEquiv = false;
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
    BasicBlock *B = PN->getIncomingBlock(i);
    if (!ReachableBlocks.count(B)) {
      DEBUG(dbgs() << "Skipping unreachable block " << getBlockName(B)
                   << " in PHI node " << *PN << "\n");
      continue;
    }
    if (I->getOperand(i) != I) {
      const BasicBlockEdge BBE(B, PhiBlock);
      auto Operand = lookupOperandLeader(I->getOperand(i), I, BBE);
      E->ops_push_back(Operand.first);
      UsedEquiv |= Operand.second;
    } else {
      E->ops_push_back(I->getOperand(i));
    }
  }
  E->setUsedEquivalence(UsedEquiv);
  return E;
}

// Set basic expression info (Arguments, type, opcode) for Expression
// E from Instruction I in block B

bool NewGVN::setBasicExpressionInfo(Instruction *I, BasicExpression *E,
                                    const BasicBlock *B) {
  bool AllConstant = true;
  bool UsedEquiv = false;
  if (auto GEP = dyn_cast<GetElementPtrInst>(I))
    E->setType(GEP->getSourceElementType());
  else
    E->setType(I->getType());
  E->setOpcode(I->getOpcode());
  E->allocateOperands(ArgRecycler, ExpressionAllocator);

  for (auto &O : I->operands()) {
    auto Operand = lookupOperandLeader(O, I, B);
    UsedEquiv |= Operand.second;
    if (!isa<Constant>(Operand.first))
      AllConstant = false;
    E->ops_push_back(Operand.first);
  }
  E->setUsedEquivalence(UsedEquiv);
  return AllConstant;
}

// This is a special function only used by equality propagation, it
// should not be called elsewhere
const BasicExpression *NewGVN::createCmpExpression(unsigned Opcode, Type *Type,
                                                   CmpInst::Predicate Predicate,
                                                   Value *LHS, Value *RHS,
                                                   const BasicBlock *B) {
  BasicExpression *E = new (ExpressionAllocator) BasicExpression(2);
  bool UsedEquiv = false;
  E->allocateOperands(ArgRecycler, ExpressionAllocator);
  E->setType(Type);
  E->setOpcode((Opcode << 8) | Predicate);
  auto Result = lookupOperandLeader(LHS, nullptr, B);
  UsedEquiv |= Result.second;
  E->ops_push_back(Result.first);
  Result = lookupOperandLeader(RHS, nullptr, B);
  UsedEquiv |= Result.second;
  E->ops_push_back(Result.first);
  E->setUsedEquivalence(UsedEquiv);
  return E;
}

const Expression *NewGVN::createBinaryExpression(unsigned Opcode, Type *T,
                                                 Value *Arg1, Value *Arg2,
                                                 const BasicBlock *B) {
  BasicExpression *E = new (ExpressionAllocator) BasicExpression(2);

  E->setType(T);
  E->setOpcode(Opcode);
  E->allocateOperands(ArgRecycler, ExpressionAllocator);
  if (Instruction::isCommutative(Opcode)) {
    // Ensure that commutative instructions that only differ by a permutation
    // of their operands get the same value number by sorting the operand value
    // numbers.  Since all commutative instructions have two operands it is more
    // efficient to sort by hand rather than using, say, std::sort.
    if (Arg1 > Arg2)
      std::swap(Arg1, Arg2);
  }
  bool UsedEquiv = false;
  auto BinaryLeader = lookupOperandLeader(Arg1, nullptr, B);
  UsedEquiv |= BinaryLeader.second;
  E->ops_push_back(BinaryLeader.first);

  BinaryLeader = lookupOperandLeader(Arg2, nullptr, B);
  UsedEquiv |= BinaryLeader.second;
  E->ops_push_back(BinaryLeader.first);

  Value *V = SimplifyBinOp(Opcode, E->getOperand(0), E->getOperand(1), *DL, TLI,
                           DT, AC);
  if (const Expression *SimplifiedE = checkSimplificationResults(E, nullptr, V))
    return SimplifiedE;
  return E;
}

// Take a Value returned by simplification of Expression E/Instruction
// I, and see if it resulted in a simpler expression. If so, return
// that expression
// TODO: Once finished, this should not take an Instruction, we only
// use it for printing
const Expression *NewGVN::checkSimplificationResults(Expression *E,
                                                     Instruction *I, Value *V) {
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

    cast<BasicExpression>(E)->deallocateOperands(ArgRecycler);
    ExpressionAllocator.Deallocate(E);
    return createConstantExpression(C, E->usedEquivalence());
  } else if (isa<Argument>(V) || isa<GlobalVariable>(V)) {
#ifndef NDEBUG
    if (I)
      DEBUG(dbgs() << "Simplified " << *I << " to "
                   << " variable " << *V << "\n");
#endif
    cast<BasicExpression>(E)->deallocateOperands(ArgRecycler);
    ExpressionAllocator.Deallocate(E);
    return createVariableExpression(V, E->usedEquivalence());
  }

  CongruenceClass *CC = ValueToClass.lookup(V);
  if (CC && CC->DefiningExpr) {
#ifndef NDEBUG
    if (I)
      DEBUG(dbgs() << "Simplified " << *I << " to "
                   << " expression " << *V << "\n");

#endif
    NumGVNOpsSimplified++;
    assert(isa<BasicExpression>(E) &&
           "We should always have had a basic expression here");
    cast<BasicExpression>(E)->deallocateOperands(ArgRecycler);
    ExpressionAllocator.Deallocate(E);
    return CC->DefiningExpr;
  }
  return NULL;
}

const Expression *NewGVN::createExpression(Instruction *I,
                                           const BasicBlock *B) {

  BasicExpression *E =
      new (ExpressionAllocator) BasicExpression(I->getNumOperands());

  bool AllConstant = setBasicExpressionInfo(I, E, B);

  if (I->isCommutative()) {
    // Ensure that commutative instructions that only differ by a permutation
    // of their operands get the same value number by sorting the operand value
    // numbers.  Since all commutative instructions have two operands it is more
    // efficient to sort by hand rather than using, say, std::sort.
    assert(I->getNumOperands() == 2 && "Unsupported commutative instruction!");
    if (E->getOperand(0) > E->getOperand(1))
      E->swapOperands(0, 1);
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
    if (E->getOperand(0) > E->getOperand(1)) {
      E->swapOperands(0, 1);
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
    if ((E->getOperand(0)->getType() == I->getOperand(0)->getType() &&
         E->getOperand(1)->getType() == I->getOperand(1)->getType())) {
      Value *V = SimplifyCmpInst(Predicate, E->getOperand(0), E->getOperand(1),
                                 *DL, TLI, DT, AC);
      if (const Expression *SimplifiedE = checkSimplificationResults(E, I, V))
        return SimplifiedE;
    }

  } else if (isa<SelectInst>(I)) {
    if (isa<Constant>(E->getOperand(0)) ||
        (E->getOperand(1)->getType() == I->getOperand(1)->getType() &&
         E->getOperand(2)->getType() == I->getOperand(2)->getType())) {
      Value *V = SimplifySelectInst(E->getOperand(0), E->getOperand(1),
                                    E->getOperand(2), *DL, TLI, DT, AC);
      if (const Expression *SimplifiedE = checkSimplificationResults(E, I, V))
        return SimplifiedE;
    }
  } else if (I->isBinaryOp()) {
    Value *V = SimplifyBinOp(E->getOpcode(), E->getOperand(0), E->getOperand(1),
                             *DL, TLI, DT, AC);
    if (const Expression *SimplifiedE = checkSimplificationResults(E, I, V))
      return SimplifiedE;
  } else if (BitCastInst *BI = dyn_cast<BitCastInst>(I)) {
    Value *V = SimplifyInstruction(BI, *DL, TLI, DT, AC);
    if (const Expression *SimplifiedE = checkSimplificationResults(E, I, V))
      return SimplifiedE;
  } else if (isa<GetElementPtrInst>(I)) {
    Value *V = SimplifyGEPInst(E->getType(),
                               ArrayRef<Value *>(E->ops_begin(), E->ops_end()),
                               *DL, TLI, DT, AC);
    if (const Expression *SimplifiedE = checkSimplificationResults(E, I, V))
      return SimplifiedE;
  } else if (AllConstant) {
    // We don't bother trying to simplify unless all of the operands
    // were constant
    // TODO: There are a lot of Simplify*'s we could call here, if we
    // wanted to.  The original motivating case for this code was a
    // zext i1 false to i8, which we don't have an interface to
    // simplify (IE there is no SimplifyZExt)

    SmallVector<Constant *, 8> C;
    for (Value *Arg : E->operands())
      C.emplace_back(cast<Constant>(Arg));

    Value *V = ConstantFoldInstOperands(I, C, *DL, TLI);
    if (V) {
      if (const Expression *SimplifiedE = checkSimplificationResults(E, I, V))
        return SimplifiedE;
    }
  }
  return E;
}

const AggregateValueExpression *
NewGVN::createAggregateValueExpression(Instruction *I, const BasicBlock *B) {
  if (InsertValueInst *II = dyn_cast<InsertValueInst>(I)) {
    AggregateValueExpression *E = new (ExpressionAllocator)
        AggregateValueExpression(I->getNumOperands(), II->getNumIndices());
    setBasicExpressionInfo(I, E, B);
    E->allocateIntOperands(ExpressionAllocator);

    for (auto &Index : II->indices())
      E->int_ops_push_back(Index);
    return E;

  } else if (ExtractValueInst *EI = dyn_cast<ExtractValueInst>(I)) {
    AggregateValueExpression *E = new (ExpressionAllocator)
        AggregateValueExpression(I->getNumOperands(), EI->getNumIndices());
    setBasicExpressionInfo(EI, E, B);
    E->allocateIntOperands(ExpressionAllocator);

    for (auto &Index : EI->indices())
      E->int_ops_push_back(Index);
    return E;
  }
  llvm_unreachable("Unhandled type of aggregate value operation");
}

const VariableExpression *
NewGVN::createVariableExpression(Value *V, bool UsedEquivalence) {
  VariableExpression *E = new (ExpressionAllocator) VariableExpression(V);
  E->setOpcode(V->getValueID());
  E->setUsedEquivalence(UsedEquivalence);
  return E;
}

const Expression *NewGVN::createVariableOrConstant(Value *V,
                                                   const BasicBlock *B) {
  auto Leader = lookupOperandLeader(V, nullptr, B);
  if (Constant *C = dyn_cast<Constant>(Leader.first))
    return createConstantExpression(C, Leader.second);
  return createVariableExpression(Leader.first, Leader.second);
}

const ConstantExpression *
NewGVN::createConstantExpression(Constant *C, bool UsedEquivalence) {
  ConstantExpression *E = new (ExpressionAllocator) ConstantExpression(C);
  E->setOpcode(C->getValueID());
  E->setUsedEquivalence(UsedEquivalence);
  return E;
}

const CallExpression *NewGVN::createCallExpression(CallInst *CI,
                                                   MemoryAccess *HV,
                                                   const BasicBlock *B) {
  CallExpression *E =
      new (ExpressionAllocator) CallExpression(CI->getNumOperands(), CI, HV);
  setBasicExpressionInfo(CI, E, B);
  return E;
}

// Find an equivalence in congruence class CC that dominates block B,
// if one exists
// TODO: Compare this against the predicate handling system in the paper

template <class T>
Value *NewGVN::findDominatingEquivalent(CongruenceClass *CC, const User *U,
                                        const T &B) const {

  // TODO: This can be made faster by different set ordering, if
  // necessary, or caching whether we found one before and only updating it when
  // things change.
  for (const auto &Member : CC->Equivs) {
    // We can't process edge only equivalences without edges
    if (Member.EdgeOnly)
      continue;
    if (DT->dominates(Member.Edge.getEnd(), B))
      return Member.Val;
  }
  // FIXME: Use single user equivalences too

  return nullptr;
}

// If we are asked about a basic block edge, it's for an edge carried value
// (like a phi node incoming edge).  So even if it doesn't dominate the block at
// the end of the edge, if it's the same edge, that's fine too.
template <>
Value *NewGVN::findDominatingEquivalent(CongruenceClass *CC, const User *U,
                                        const BasicBlockEdge &B) const {
  // TODO: This can be made faster by different set ordering, if
  // necessary, or caching whether we found one before and only updating it when
  // things change.
  for (const auto &Member : CC->Equivs) {
    if ((Member.Edge.getStart() == B.getStart() &&
         Member.Edge.getEnd() == B.getEnd()) ||
        (!Member.EdgeOnly && DT->dominates(Member.Edge.getEnd(), B.getEnd())))
      return Member.Val;
  }
  // FIXME: Use single user equivalences too
  return nullptr;
}

// lookupOperandLeader -- See if we have a congruence class and leader
// for this operand, and if so, return it. Otherwise, return the
// original operand.  The second part of the return value is true if a
// dominating equivalence is being returned.
template <class T>
std::pair<Value *, bool> NewGVN::lookupOperandLeader(Value *V, const User *U,
                                                     const T &B) const {
  CongruenceClass *CC = ValueToClass.lookup(V);
  if (CC && (CC != InitialClass)) {
    Value *Equivalence = findDominatingEquivalent(CC, U, B);
    if (Equivalence) {
      DEBUG(dbgs() << "Found equivalence " << *Equivalence << " for value "
                   << *V << " in block " << getBlockName(B) << "\n");
      return std::make_pair(Equivalence, true);
    }
    return std::make_pair(CC->RepLeader, false);
  }
  return std::make_pair(V, false);
}

LoadExpression *NewGVN::createLoadExpression(Type *LoadType, Value *PointerOp,
                                             LoadInst *LI, MemoryAccess *DA,
                                             const BasicBlock *B) {
  LoadExpression *E = new (ExpressionAllocator) LoadExpression(1, LI, DA);
  E->allocateOperands(ArgRecycler, ExpressionAllocator);
  E->setType(LoadType);
  // Give store and loads same opcode so they value number together
  E->setOpcode(0);
  auto Operand = lookupOperandLeader(PointerOp, LI, B);
  E->ops_push_back(Operand.first);
  if (LI)
    E->setAlignment(LI->getAlignment());
  E->setUsedEquivalence(Operand.second);

  // TODO: Value number heap versions. We may be able to discover
  // things alias analysis can't on it's own (IE that a store and a
  // load have the same value, and thus, it isn't clobbering the load)
  return E;
}

const CoercibleLoadExpression *NewGVN::createCoercibleLoadExpression(
    Type *LoadType, Value *PtrOperand, LoadInst *Original, MemoryAccess *DA,
    unsigned Offset, Value *SrcVal, const BasicBlock *B) {
  CoercibleLoadExpression *E = new (ExpressionAllocator)
      CoercibleLoadExpression(1, Original, DA, Offset, SrcVal);
  E->allocateOperands(ArgRecycler, ExpressionAllocator);
  E->setType(LoadType);
  // Give store and loads same opcode so they value number together
  E->setOpcode(0);
  auto Operand = lookupOperandLeader(PtrOperand, Original, B);
  E->ops_push_back(Operand.first);
  E->setUsedEquivalence(Operand.second);
  // TODO: Value number heap versions. We may be able to discover
  // things alias analysis can't on it's own (IE that a store and a
  // load have the same value, and thus, it isn't clobbering the load)
  return E;
}

const StoreExpression *NewGVN::createStoreExpression(StoreInst *SI,
                                                     MemoryAccess *DA,
                                                     const BasicBlock *B) {
  StoreExpression *E =
      new (ExpressionAllocator) StoreExpression(SI->getNumOperands(), SI, DA);
  E->allocateOperands(ArgRecycler, ExpressionAllocator);
  E->setType(SI->getValueOperand()->getType());
  // Give store and loads same opcode so they value number together
  E->setOpcode(0);
  auto Operand = lookupOperandLeader(SI->getPointerOperand(), SI, B);
  E->ops_push_back(Operand.first);
  E->setUsedEquivalence(Operand.second);
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

  unsigned Size = MemoryDependenceResults::getLoadLoadClobberFullWidthSize(
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
  Src = ConstantExpr::getGetElementPtr(Type::getInt8Ty(Src->getContext()), Src,
                                       OffsetCst);
  Src = ConstantExpr::getBitCast(Src, PointerType::get(LoadTy, AS));
  if (ConstantFoldLoadFromConstPtr(Src, LoadTy, *DL))
    return Offset;
  return -1;
}

const Expression *NewGVN::performSymbolicStoreEvaluation(Instruction *I,
                                                         const BasicBlock *B) {
  StoreInst *SI = cast<StoreInst>(I);
  const Expression *E = createStoreExpression(SI, MSSA->getMemoryAccess(SI), B);
  return E;
}

const Expression *NewGVN::performSymbolicLoadEvaluation(Instruction *I,
                                                        const BasicBlock *B) {
  LoadInst *LI = cast<LoadInst>(I);

  // We can eliminate in favor of non-simple loads, but we won't be able to
  // eliminate them
  if (!LI->isSimple())
    return nullptr;

  Value *LoadAddressLeader =
      lookupOperandLeader(LI->getPointerOperand(), I, B).first;
  // Load of undef is undef
  if (isa<UndefValue>(LoadAddressLeader))
    return createConstantExpression(UndefValue::get(LI->getType()), false);

  MemoryAccess *DefiningAccess = MSSAWalker->getClobberingMemoryAccess(I);

  if (!MSSA->isLiveOnEntryDef(DefiningAccess)) {
    if (MemoryDef *MD = dyn_cast<MemoryDef>(DefiningAccess)) {
      Instruction *DefiningInst = MD->getMemoryInst();
      // If the defining instruction is not reachable, replace with
      // undef
      if (!ReachableBlocks.count(DefiningInst->getParent()))
        return createConstantExpression(UndefValue::get(LI->getType()), false);
    }
  } else {
    BasicBlock *LoadBlock = LI->getParent();
    MemoryAccess *LoadAccess = MSSA->getMemoryAccess(LI);
    // Okay, so uh, we couldn't use the defining access to grab a value out of
    // See if we can reuse any of it's uses by widening a load.
    for (const auto &U : DefiningAccess->users()) {
      MemoryAccess *MA = cast<MemoryAccess>(U);
      if (MA == LoadAccess)
        continue;
      if (isa<MemoryPhi>(MA))
        continue;
      Instruction *DefiningInst = cast<MemoryUseOrDef>(MA)->getMemoryInst();
      if (LoadInst *DepLI = dyn_cast<LoadInst>(DefiningInst)) {
        BasicBlock *DefiningBlock = DefiningInst->getParent();
        if (!DT->dominates(DefiningBlock, LoadBlock))
          continue;

        // Make sure the dependent load comes before the load we are trying
        // to coerce if they are in the same block
        if (InstrDFS[DepLI] >= InstrDFS[LI])
          continue;

        // Now, first make sure they really aren't identical loads, and if
        // they aren't see if the dominating one can be coerced.
        // We don't want to mark identical loads coercible, since coercible
        // loads don't value number with normal loads.
        Value *DepAddressLeader =
            lookupOperandLeader(DepLI->getPointerOperand(), DepLI, B).first;

        if (LI->getType() != DepLI->getType() ||
            DepAddressLeader != LoadAddressLeader ||
            LI->isSimple() != DepLI->isSimple()) {
          int Offset = analyzeLoadFromClobberingLoad(
              LI->getType(), LI->getPointerOperand(), DepLI);
          if (Offset >= 0)
            return createCoercibleLoadExpression(
                LI->getType(), LI->getPointerOperand(), LI, DefiningAccess,
                (unsigned)Offset, DepLI, B);
        }
      }
    }
  }

  const Expression *E = createLoadExpression(
      LI->getType(), LI->getPointerOperand(), LI, DefiningAccess, B);
  return E;
}

/// performSymbolicCallEvaluation - Evaluate read only and pure calls, and
/// create an expression result
const Expression *NewGVN::performSymbolicCallEvaluation(Instruction *I,
                                                        const BasicBlock *B) {
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
const Expression *NewGVN::performSymbolicPHIEvaluation(Instruction *I,
                                                       const BasicBlock *B) {
  PHIExpression *E = cast<PHIExpression>(createPHIExpression(I));
  if (E->ops_empty()) {
    DEBUG(dbgs() << "Simplified PHI node " << *I << " to undef"
                 << "\n");
    E->deallocateOperands(ArgRecycler);
    ExpressionAllocator.Deallocate(E);
    return createConstantExpression(UndefValue::get(I->getType()), false);
  }

  Value *AllSameValue = E->getOperand(0);

  // See if all arguments are the same, ignoring undef arguments, because we can
  // choose a value that is the same for them.
  for (const Value *Arg : E->operands())
    if (Arg != AllSameValue && !isa<UndefValue>(Arg)) {
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
    E->deallocateOperands(ArgRecycler);
    ExpressionAllocator.Deallocate(E);
    if (Constant *C = dyn_cast<Constant>(AllSameValue))
      return createConstantExpression(C, E->usedEquivalence());
    return createVariableExpression(AllSameValue, E->usedEquivalence());
  }
  return E;
}

const Expression *
NewGVN::performSymbolicAggrValueEvaluation(Instruction *I,
                                           const BasicBlock *B) {
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
const Expression *NewGVN::performSymbolicEvaluation(Value *V,
                                                    const BasicBlock *B) {
  const Expression *E = NULL;
  if (Constant *C = dyn_cast<Constant>(V))
    E = createConstantExpression(C, false);
  else if (isa<Argument>(V) || isa<GlobalVariable>(V)) {
    E = createVariableExpression(V, false);
  } else {
    // TODO: memory intrinsics
    // TODO: Some day, we should do the forward propagation and reassociation
    // parts of the algorithm.
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

/// \brief Given an edge we know does not dominate its end, go through all uses
/// of \p From, figure out if we dominate them, and if so, mark them
/// individually as equivalent to \p To.

void NewGVN::markDominatedSingleUserEquivalences(CongruenceClass *CC,
                                                 Value *From, Value *To,
                                                 bool MultipleEdgesOneReachable,
                                                 const BasicBlockEdge &Root) {

  // The MultipleEdgesOneReachable case means we know there are multiple edges
  // to the block, but *only one from us is reachable*.  This occurs if, for
  // example, we discover the value of a switch's condition is constant, and it
  // normally had multiple case values that go to the same block.  Because the
  // dominators API asserts isSingleEdge, we have to special case this, since we
  // know isSingleEdge doesn't matter.
  for (const auto &U : From->uses()) {
    bool Dominates = false;
    if (MultipleEdgesOneReachable) {
      Instruction *UserInst = cast<Instruction>(U.getUser());
      PHINode *PN = dyn_cast<PHINode>(UserInst);
      if (PN && PN->getParent() == Root.getEnd() &&
          PN->getIncomingBlock(U) == Root.getStart()) {
        Dominates = true;
      } else
        Dominates = DT->dominates(Root.getStart(), UserInst->getParent());
    } else {
      Dominates = DT->dominates(Root, U);
    }
    if (!Dominates)
      continue;
    CC->SingleUserEquivs.emplace(
        From, SingleUserEquivalence{U.getUser(), U.getOperandNo(), To});
    // Mark the users as touched
    if (Instruction *I = dyn_cast<Instruction>(U.getUser()))
      TouchedInstructions.set(InstrDFS[I]);
  }
}

/// replaceAllDominatedUsesWith - Replace all uses of 'From' with 'To'
/// if the use is dominated by the given basic block.  Returns the
/// number of uses that were replaced.
unsigned NewGVN::replaceAllDominatedUsesWith(Value *From, Value *To,
                                             const BasicBlockEdge &Root,
                                             bool EdgeEquivOnly) {
  unsigned Count = 0;
  for (auto UI = From->use_begin(), UE = From->use_end(); UI != UE;) {
    Use &U = *UI++;
    // Edge equivalents
    if (EdgeEquivOnly) {
      PHINode *PN = dyn_cast<PHINode>(U.getUser());
      if (PN && PN->getParent() == Root.getEnd() &&
          PN->getIncomingBlock(U) == Root.getStart()) {
        U.set(To);
        ++Count;
      }
    } else {
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
  }
  return Count;
}
/// There is an edge from 'Src' to 'Dst'.  Return
/// true if every path from the entry block to 'Dst' passes via this edge.  In
/// particular 'Dst' must not be reachable via another edge from 'Src'.
bool NewGVN::isOnlyReachableViaThisEdge(const BasicBlockEdge &E) {
// While in theory it is interesting to consider the case in which Dst has
// more than one predecessor, because Dst might be part of a loop which is
// only reachable from Src, in practice it is pointless since at the time
// GVN runs all such loops have preheaders, which means that Dst will have
// been changed to have only one predecessor, namely Src.
  const BasicBlock *Pred = E.getEnd()->getSinglePredecessor();
  const BasicBlock *Src = E.getStart();
  assert((!Pred || Pred == Src) && "No edge between these basic blocks!");
  (void)Src;
  return Pred != nullptr;
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

bool NewGVN::propagateEquality(Value *LHS, Value *RHS,
                               bool MultipleEdgesOneReachable,
                               const BasicBlockEdge &Root) {
  SmallVector<std::pair<Value *, Value *>, 4> Worklist;
  Worklist.emplace_back(LHS, RHS);
  bool Changed = false;

  bool RootDominatesEnd = isOnlyReachableViaThisEdge(Root);

  while (!Worklist.empty()) {
    std::pair<Value *, Value *> Item = Worklist.pop_back_val();
    LHS = Item.first;
    RHS = Item.second;
    DEBUG(dbgs() << "Setting equivalence " << *LHS << " = " << *RHS
                 << " in blocks dominated by " << getBlockName(Root.getEnd())
                 << "\n");

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

    // If value numbering later deduces that an instruction in the
    // scope is equal to 'LHS' then ensure it will be turned into
    // 'RHS' within the scope Root.
    CongruenceClass *CC = ValueToClass[LHS];
    assert(CC && "Should have found a congruence class");
    CC->Equivs.emplace_back(RHS, Root, !RootDominatesEnd);

    if (RootDominatesEnd)
      markDominatedSingleUserEquivalences(CC, LHS, RHS,
                                          MultipleEdgesOneReachable, Root);
    // Replace all occurrences of 'LHS' with 'RHS' everywhere in the
    // scope.  As LHS always has at least one use that is not
    // dominated by Root, this will never do anything if LHS has only
    // one use.
    // FIXME: I think this can be deleted now, bootstrap with an assert
    if (!LHS->hasOneUse() && 0) {
      unsigned NumReplacements =
          replaceAllDominatedUsesWith(LHS, RHS, Root, false);
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
    if (CmpInst *Cmp = dyn_cast<CmpInst>(LHS)) {
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
      const BasicExpression *E =
          createCmpExpression(Cmp->getOpcode(), Cmp->getType(), NotPred, Op0,
                              Op1, Cmp->getParent());

      // We cannot directly propagate into the congruence class members
      // unless it is a singleton class.
      // This is because at least one of the members may later get
      // split out of the class.

      CongruenceClass *CC = ExpressionToClass.lookup(E);
      // E->deallocateArgs(ArgRecycler);
      // ExpressionAllocator.Deallocate(E);

      // If we didn't find a congruence class, there is no equivalent
      // instruction already
      if (CC) {
        // FIXME: I think this can be deleted now, need to bootstrap with an
        // assert
        if (CC->Members.size() == 1 && 0) {
          unsigned NumReplacements =
              replaceAllDominatedUsesWith(CC->RepLeader, NotVal, Root, false);
          Changed |= NumReplacements > 0;
          NumGVNEqProp += NumReplacements;
        }
        // Ensure that any instruction in scope that gets the "A < B"
        // value number is replaced with false.

        CC->Equivs.emplace_back(NotVal, Root, !RootDominatesEnd);

      } else {
        // Put this in pending equivalences so if an expression shows up, we'll
        // add the equivalence
        PendingEquivalences.emplace(
            E, Equivalence{NotVal, Root, !RootDominatesEnd});
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

// updateReachableEdge - Process the fact that Edge (from, to) is
// reachable, including marking any newly reachable blocks and
// instructions for processing
void NewGVN::updateReachableEdge(BasicBlock *From, BasicBlock *To) {
  // Check if the Edge was reachable before
  if (ReachableEdges.insert({From, To}).second) {
    // If this block wasn't reachable before, all instructions are touched
    if (ReachableBlocks.insert(To).second) {
      DEBUG(dbgs() << "Block " << getBlockName(To) << " marked reachable\n");
      const auto &InstRange = BlockInstRange.lookup(To);
      TouchedInstructions.set(InstRange.first, InstRange.second);
    } else {
      DEBUG(dbgs() << "Block " << getBlockName(To)
                   << " was reachable, but new edge {" << getBlockName(From)
                   << "," << getBlockName(To) << "} to it found\n");
      // We've made an edge reachable to an existing block, which may
      // impact predicates.
      // Otherwise, only mark the phi nodes as touched, as they are
      // the only thing that depend on new edges. Anything using their
      // values will get propagated to if necessary
      auto BI = To->begin();
      while (isa<PHINode>(BI)) {
        TouchedInstructions.set(InstrDFS[&*BI]);
        ++BI;
      }
      // Propagate the change downstream.
      propagateChangeInEdge(To);
    }
  }
}

// findConditionEquivalence - Given a predicate condition (from a
// switch, cmp, or whatever) and a block, see if we know some constant
// value for it already
Value *NewGVN::findConditionEquivalence(Value *Cond, BasicBlock *B) const {
  auto Result = lookupOperandLeader(Cond, nullptr, B);
  if (isa<Constant>(Result.first))
    return Result.first;

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
        const Expression *E = createExpression(I, B);
        if (const ConstantExpression *CE = dyn_cast<ConstantExpression>(E)) {
          CondEvaluated = CE->getConstantValue();
        }
      } else if (isa<ConstantInt>(Cond)) {
        CondEvaluated = Cond;
      }
    } else {
      InvolvedInEquivalence.set(InstrDFS[TI]);
    }
    ConstantInt *CI;
    BasicBlock *TrueSucc = BR->getSuccessor(0);
    BasicBlock *FalseSucc = BR->getSuccessor(1);
    if (CondEvaluated && (CI = dyn_cast<ConstantInt>(CondEvaluated))) {
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
      propagateEquality(Cond, ConstantInt::getTrue(TrueSucc->getContext()),
                        false, {B, TrueSucc});

      propagateEquality(Cond, ConstantInt::getFalse(FalseSucc->getContext()),
                        false, {B, FalseSucc});
      updateReachableEdge(B, TrueSucc);
      updateReachableEdge(B, FalseSucc);
    }
  } else if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
    // For switches, propagate the case values into the case
    // destinations.

    // Remember how many outgoing edges there are to every successor.
    SmallDenseMap<BasicBlock *, unsigned, 16> SwitchEdges;

    bool MultipleEdgesOneReachable = false;
    Value *SwitchCond = SI->getCondition();
    Value *CondEvaluated = findConditionEquivalence(SwitchCond, B);
    // See if we were able to turn this switch statement into a constant
    if (CondEvaluated && isa<ConstantInt>(CondEvaluated)) {
      InvolvedInEquivalence.set(InstrDFS[TI]);
      ConstantInt *CondVal = cast<ConstantInt>(CondEvaluated);
      // We should be able to get case value for this
      auto CaseVal = SI->findCaseValue(CondVal);
      if (CaseVal.getCaseSuccessor() == SI->getDefaultDest()) {
        // We proved the value is outside of the range of the case.
        // We can't do anything other than mark the default dest as reachable,
        // and go home.
        updateReachableEdge(B, SI->getDefaultDest());
        return;
      }
      // Now get where it goes and mark it reachable
      BasicBlock *TargetBlock = CaseVal.getCaseSuccessor();
      updateReachableEdge(B, TargetBlock);
      unsigned WhichSucc = CaseVal.getSuccessorIndex();
      // Calculate whether our single reachable edge is really a single edge to
      // the target block.  If not, and the block has multiple predecessors, we
      // can only replace phi node values.
      for (unsigned i = 0, e = SI->getNumSuccessors(); i != e; ++i) {
        if (i == WhichSucc)
          continue;
        BasicBlock *Block = SI->getSuccessor(i);
        if (Block == TargetBlock)
          MultipleEdgesOneReachable = true;
      }
      const BasicBlockEdge E(B, TargetBlock);
      propagateEquality(SwitchCond, CaseVal.getCaseValue(),
                        MultipleEdgesOneReachable, E);
    } else {
      for (unsigned i = 0, e = SI->getNumSuccessors(); i != e; ++i) {
        BasicBlock *TargetBlock = SI->getSuccessor(i);
        ++SwitchEdges[TargetBlock];
        updateReachableEdge(B, TargetBlock);
      }

      // Regardless of answers, propagate equalities for case values
      for (auto i = SI->case_begin(), e = SI->case_end(); i != e; ++i) {
        BasicBlock *TargetBlock = i.getCaseSuccessor();
        if (SwitchEdges.lookup(TargetBlock) == 1) {
          const BasicBlockEdge E(B, TargetBlock);
          propagateEquality(SwitchCond, i.getCaseValue(),
                            MultipleEdgesOneReachable, E);
        }
      }
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

// Figure out and cache the dominated instruction range for this block. We do
// this by doing a depth first search, and figuring out the min and the max of
// the dominated instruction range.
// On the way up, we cache the ranges for all the child blocks.
// This is essentially an incremental DFS

const std::pair<unsigned, unsigned>
NewGVN::calculateDominatedInstRange(const DomTreeNode *DTN) {

  // First see if we have it, if not, we need to recalculate
  const auto &DominatedRange = DominatedInstRange.find(DTN);
  if (DominatedRange != DominatedInstRange.end()) {
    return DominatedRange->second;
  }
  // Recalculate
  SmallVector<std::pair<const DomTreeNode *, DomTreeNode::const_iterator>, 32>
      WorkStack;
  WorkStack.emplace_back(DTN, DTN->begin());
  unsigned MaxSeen = 0;
  while (!WorkStack.empty()) {
    const auto &Back = WorkStack.back();
    const DomTreeNode *Node = Back.first;
    auto ChildIt = Back.second;
    const auto &Result = BlockInstRange.lookup(Node->getBlock());
    MaxSeen = std::max(MaxSeen, Result.second);
    // If we visited all of the children of this node, "recurse" back up the
    // stack setting the ranges
    if (ChildIt == Node->end()) {
      const auto &Result = BlockInstRange.lookup(Node->getBlock());
      DominatedInstRange[DTN] = {Result.first, MaxSeen};
      WorkStack.pop_back();
      if (WorkStack.empty())
        return {Result.first, MaxSeen};
    } else {
      while (ChildIt != Node->end()) {
        // Otherwise, recursively visit this child.
        const DomTreeNode *Child = *ChildIt;
        ++WorkStack.back().second;
        const auto &LookupResult = DominatedInstRange.find(Child);
        if (LookupResult == DominatedInstRange.end()) {
          WorkStack.emplace_back(Child, Child->begin());
          break;
        } else {
          // We already have calculated this subtree
          MaxSeen = std::max(MaxSeen, LookupResult->second.second);
          ++ChildIt;
        }
      }
    }
  }
  llvm_unreachable("Should have returned a value already");
}

/// propagateChangeInEdge - Propagate a change in edge reachability
// When we discover a new edge to an existing reachable block, that
// can affect the value of instructions that used equivalences.
//
void NewGVN::propagateChangeInEdge(BasicBlock *Dest) {
  // Unlike, the published algorithm, we don't touch blocks because we aren't
  // recomputing predicates.
  // Instead, we touch all the instructions we dominate that were used in an
  // equivalence, in case they changed.
  // We incrementally compute the dominated instruction ranges, because doing it
  // up front requires a DFS walk of the dominator tree that is a complete waste
  // of time if no equivalences ever get seen.
  // Note that we expect phi nodes in the Dest block will have already been
  // touched by our caller.
  DomTreeNode *DTN = DT->getNode(Dest);

  // Note that this is an overshoot, because the inst ranges are calculated in
  // RPO order, not dominator tree order.
  const std::pair<unsigned, unsigned> Result = calculateDominatedInstRange(DTN);

  // Touch all the downstream dominated instructions that used equivalences.
  for (int InstrNum = InvolvedInEquivalence.find_next(Result.first - 1);
       InstrNum != -1 && (Result.second - InstrNum > 0);
       InstrNum = InvolvedInEquivalence.find_next(InstrNum)) {
    // TODO: We could do confluence block checks here.
    TouchedInstructions.set(InstrNum);
  }
}

// The algorithm initially places the values of the routine in the INITIAL congruence
// class. The leader of INITIAL is the undetermined value `TOP`.
// When the algorithm has finished, values still in INITIAL are unreachable.
void NewGVN::initializeCongruenceClasses(Function &F) {
  // FIXME now i can't remember why this is 2
  NextCongruenceNum = 2;
  // Initialize all other instructions to be in INITIAL class
  CongruenceClass::MemberSet InitialValues;
  for (auto &B : F)
    for (auto &I : B)
      InitialValues.insert(&I);

  InitialClass = createCongruenceClass(NULL, NULL);
  for (auto L : InitialValues)
    ValueToClass[L] = InitialClass;
  InitialClass->Members.swap(InitialValues);

  // Initialize arguments to be in their own unique congruence classes
  // In an IPA-GVN, this would not be done
  for (auto &FA : F.args())
    createSingletonCongruenceClass(&FA);
}

void NewGVN::cleanupTables() {

  ValueToClass.clear();
  for (unsigned i = 0, e = CongruenceClasses.size(); i != e; ++i) {
    DEBUG(dbgs() << "Congruence class " << CongruenceClasses[i]->ID << " has "
                 << CongruenceClasses[i]->Members.size() << " members\n");
  }

  ArgRecycler.clear(ExpressionAllocator);
  ExpressionAllocator.Reset();
  CongruenceClasses.clear();
  ExpressionToClass.clear();
  ValueToExpression.clear();
  ReachableBlocks.clear();
  ReachableEdges.clear();
  ProcessedCount.clear();
  DFSDomMap.clear();
  InstrDFS.clear();
  InstructionsToErase.clear();

  DFSToInstr.clear();
  BlockInstRange.clear();
  TouchedInstructions.clear();
  InvolvedInEquivalence.clear();
  DominatedInstRange.clear();
  PendingEquivalences.clear();
}

std::pair<unsigned, unsigned> NewGVN::assignDFSNumbers(BasicBlock *B,
                                                       unsigned Start) {
  unsigned int End = Start;
  for (auto &I : *B) {
    InstrDFS[&I] = End++;
    DFSToInstr.emplace_back(&I);
  }
  // All of the range functions taken half-open ranges (open on the end side).
  // So we do not subtract one from count, because at this point it is one
  // greater than the last instruction.

  return std::make_pair(Start, End);
}

void NewGVN::updateProcessedCount(Value *V) {
#if 1 /* NDEBUG */
  if (ProcessedCount.count(V) == 0) {
    ProcessedCount.insert({V, 1});
  } else {
    ProcessedCount[V] += 1;
    assert(ProcessedCount[V] < 100 &&
           "Seem to have processed the same Value a lot\n");
  }
#endif
}

/// runOnFunction - This is the main transformation entry point for a
/// function.
bool NewGVN::runGVN(Function &F, DominatorTree *DT, AssumptionCache *AC,
                   TargetLibraryInfo *TLI, AliasAnalysis *AA,
                   MemorySSA *MSSA) {
  bool Changed = false;
  this->DT = DT;
  this->AC = AC;
  this->TLI = TLI;
  this->AA = AA;
  this->MSSA = MSSA;
  DL = &F.getParent()->getDataLayout();
  MSSAWalker = MSSA->getWalker();

  // Count number of instructions for sizing of hash tables, and come
  // up with a global dfs numbering for instructions.
  unsigned ICount = 0;
  SmallPtrSet<BasicBlock *, 16> VisitedBlocks;

  // Note: We want RPO traversal of the blocks, which is not quite the same as
  // dominator tree order, particularly with regard whether backedges get
  // visited first or second, given a block with multiple successors.
  // If we visit in the wrong order, we will end up performing N times as many
  // iterations.
  ReversePostOrderTraversal<Function *> RPOT(&F);
  for (auto &B : RPOT) {
    VisitedBlocks.insert(B);
    const auto &BlockRange = assignDFSNumbers(B, ICount);
    BlockInstRange.insert({B, BlockRange});
    ICount += BlockRange.second - BlockRange.first;
  }

  // Handle forward unreachable blocks and figure out which blocks
  // have single preds

  for (auto &B : F) {
    // Assign numbers to unreachable blocks
    if (!VisitedBlocks.count(&B)) {
      const auto &BlockRange = assignDFSNumbers(&B, ICount);
      BlockInstRange.insert({&B, BlockRange});
      ICount += BlockRange.second - BlockRange.first;
    }
  }

  TouchedInstructions.resize(ICount + 1);
  InvolvedInEquivalence.resize(ICount + 1);
  DominatedInstRange.reserve(F.size());
  // Ensure we don't end up resizing the expressionToClass map, as
  // that can be quite expensive. At most, we have one expression per
  // instruction.
  ExpressionToClass.reserve(ICount + 1);
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
        updateProcessedCount(CurrBlock);
      }
      DEBUG(dbgs() << "Processing instruction " << *I << "\n");
      if (I->use_empty() && !I->getType()->isVoidTy()) {
        DEBUG(dbgs() << "Skipping unused instruction\n");
        if (isInstructionTriviallyDead(I, TLI))
          markInstructionForDeletion(I);
        TouchedInstructions.reset(InstrNum);
        continue;
      }
      updateProcessedCount(I);

      // This is done in case something eliminates the instruction
      // along the way.
      if (!I->isTerminator()) {
        const Expression *Symbolized = performSymbolicEvaluation(I, CurrBlock);
        if (Symbolized && Symbolized->usedEquivalence())
          InvolvedInEquivalence.set(InstrDFS[I]);
      } else {
        processOutgoingEdges(dyn_cast<TerminatorInst>(I), CurrBlock);
      }
      // Reset after processing (because we may mark ourselves as touched when
      // we propagate equalities)
      TouchedInstructions.reset(InstrNum);
    }
  }

  Changed = false;

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

bool NewGVN::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;
  return runGVN(F, &getAnalysis<DominatorTreeWrapperPass>().getDomTree(),
                &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F),
                &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(),
                &getAnalysis<AAResultsWrapperPass>().getAAResults(),
                &getAnalysis<MemorySSAWrapperPass>().getMSSA());
}

PreservedAnalyses NewGVNPass::run(Function &F,
                                  AnalysisManager<Function> &AM) {
  NewGVN Impl;

  // Apparently the order in which we get these results matter for
  // the old GVN (see Chandler's comment in GVN.cpp). I'll keep
  // the same order here, just in case.
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  auto &AA = AM.getResult<AAManager>(F);
  auto &MSSA = AM.getResult<MemorySSAAnalysis>(F).getMSSA();
  bool Changed = Impl.runGVN(F, &DT, &AC, &TLI, &AA, &MSSA);
  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<GlobalsAA>();
  return PA;
}

void NewGVN::deleteInstructionsInBlock(BasicBlock *BB) {
  DEBUG(dbgs() << "  BasicBlock Dead:" << *BB);
  ++NumGVNBlocksDeleted;

  // Check to see if there are non-terminating instructions to delete.
  if (isa<TerminatorInst>(BB->begin()))
    return;

  // Delete the instructions backwards, as it has a reduced likelihood of having
  // to update as many def-use and use-def chains.
  // Start after the terminator
  auto StartPoint = BB->rbegin();
  ++StartPoint;
  // Note that we explicitly recalculate BB->rend() on each iteration,
  // as it may change when we remove the first instruction.
  for (BasicBlock::reverse_iterator I(StartPoint); I != BB->rend();) {
    Instruction &Inst = *I;
    if (!Inst.use_empty())
      Inst.replaceAllUsesWith(UndefValue::get(Inst.getType()));
    if (isa<LandingPadInst>(Inst)) {
      ++I;
      continue;
    }

    I = BasicBlock::reverse_iterator(Inst.eraseFromParent());
    ++NumGVNInstrDeleted;
  }
}

void NewGVN::markInstructionForDeletion(Instruction *I) {
  DEBUG(dbgs() << "Marking " << *I << " for deletion\n");
  InstructionsToErase.insert(I);
}

