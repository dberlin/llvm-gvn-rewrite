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
#include "llvm/IR/PredIteratorCache.h"
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
#include <unordered_map>
#include <utility>
using namespace llvm;
using namespace PatternMatch;
using namespace llvm::GVNExpression;

#define DEBUG_TYPE "newgvn"

STATISTIC(NumGVNInstrDeleted, "Number of instructions deleted");
STATISTIC(NumGVNBlocksDeleted, "Number of blocks deleted");
STATISTIC(NumNewGVNPRE, "Number of instructions PRE'd");
// STATISTIC(NumNewGVNBlocks, "Number of blocks merged");
// STATISTIC(NumNewGVNSimpl, "Number of instructions simplified");
STATISTIC(NumGVNEqProp, "Number of equalities propagated");
STATISTIC(NumPRELoad, "Number of loads PRE'd");
STATISTIC(NumGVNOpsSimplified, "Number of Expressions simplified");
STATISTIC(NumGVNPhisAllSame, "Number of PHIs whos arguments are all the same");

static cl::opt<bool> EnablePRE("enable-pre2", cl::init(true), cl::Hidden);
static cl::opt<bool> EnableLoadPRE("enable-load-pre2", cl::init(true));

// Maximum allowed recursion depth.
static cl::opt<uint32_t>
    MaxRecurseDepth("max-recurse-depth2", cl::Hidden, cl::init(1000),
                    cl::ZeroOrMore,
                    cl::desc("Max recurse depth (default = 1000)"));

//===----------------------------------------------------------------------===//
//                                GVN Pass
//===----------------------------------------------------------------------===//

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

struct Equivalence {
  Value *Val;
  const BasicBlockEdge Edge;
  bool EdgeOnly;
  Equivalence(Value *V, const BasicBlockEdge E, bool EOnly)
      : Val(V), Edge(E), EdgeOnly(EOnly) {}
};

struct CongruenceClass {
  typedef SmallPtrSet<Value *, 4> MemberSet;
  static unsigned int nextCongruenceNum;
  unsigned int id;
  Value *leader;
  const Expression *expression;
  // Actual members of this class.  These are the things the same everywhere
  MemberSet members;
  // Coercible members of this class. These are loads where we can pull the
  // value out of a store. This means they need special processing during
  // elimination to do this, but they are otherwise the same as members,
  // in particular, we can eliminate one in favor of a dominating one.
  MemberSet coercible_members;

  typedef std::list<Equivalence> EquivalenceSet;

  // Noted equivalences.  These are things that are equivalence to
  // this class over certain paths.  This could be replaced with
  // proper predicate support during analysis.
  EquivalenceSet equivalences;
  bool dead;
  explicit CongruenceClass()
      : id(nextCongruenceNum++), leader(0), expression(0), dead(false) {}
  CongruenceClass(Value *Leader, const Expression *E)
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
  CongruenceClass *InitialClass;
  std::vector<std::unique_ptr<CongruenceClass>> CongruenceClasses;

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
                          expression_equal_to> PendingEquivalences;

  // This map holds the list of equivalences that only apply to a single
  // instruction.  Because we don't have control regions, when we have
  // equivalences along edges to blocks with multiple predecessors, we can't add
  // them to the general equivalences list, because they don't dominate a given
  // root, but instead, they really represent something valid in certain control
  // regions of the program, a concept that is not really expressible using the
  // a standard dominator (and not postdominator) tree.
  SmallDenseMap<User *, std::pair<unsigned, Value *>> SingleUserEquivalences;

  // Expression to class mapping
  typedef DenseMap<const Expression *, CongruenceClass */*,
                                                          ComparingExpressionInfo*/>
      ExpressionClassMap;
  ExpressionClassMap ExpressionToClass;

  // Uniquifying expressions
  DenseSet<const Expression *, ComparingExpressionInfo> UniquedExpressions;
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
  DenseMap<const Instruction *, unsigned> ProcessedCount;
  DenseMap<const BasicBlock *, unsigned> ProcessedBlockCount;

  // DFS info
  DenseMap<const BasicBlock *, std::pair<int, int>> DFSDomMap;
  DenseMap<const Instruction *, unsigned> InstrDFS;
  std::vector<Instruction *> DFSToInstr;

  // Deletion info
  SmallPtrSet<Instruction *, 8> InstructionsToErase;
  // This is a mapping from Load to (offset into source, coercion source)
  DenseMap<const Value *, std::pair<unsigned, Value *>> CoercionInfo;
  // This is a mapping for loads that got widened, to the new load. This ensures
  // we coerce from the new widened load, instead of the old one. Otherwise, we
  // may try to widen the same old load multiple times.
  DenseMap<const Value *, Value *> CoercionForwarding;

  // This is used by PRE to forward values when they get replaced
  // Because we don't update the expressions, ValueToExpression will point to
  // expressions which have the old arguments in them
  DenseMap<const Value *, Value *> PREValueForwarding;
  PredIteratorCache PredCache;

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
    AU.addRequired<MemorySSALazy>();
    AU.addRequired<AliasAnalysis>();

    AU.addPreserved<MemoryDependenceAnalysis>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addPreserved<AliasAnalysis>();
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

  const Expression *uniquifyExpression(const Expression *);
  const BasicExpression *createCmpExpression(unsigned, Type *,
                                             CmpInst::Predicate, Value *,
                                             Value *, const BasicBlock *);
  // Congruence class handling
  CongruenceClass *createCongruenceClass(Value *Leader, const Expression *E) {
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
  void performCongruenceFinding(Value *, const Expression *);
  // Predicate and reachability handling
  void updateReachableEdge(BasicBlock *, BasicBlock *);
  void processOutgoingEdges(TerminatorInst *, BasicBlock *);
  void propagateChangeInEdge(BasicBlock *);
  bool propagateEquality(Value *, Value *, const BasicBlockEdge &);
  void markDominatedSingleUserEquivalences(Value *, Value *,
                                           const BasicBlockEdge &);
  Value *findConditionEquivalence(Value *, BasicBlock *) const;
  std::pair<unsigned, unsigned>
  calculateDominatedInstRange(const DomTreeNode *);

  // Instruction replacement
  unsigned replaceAllDominatedUsesWith(Value *, Value *,
                                       const BasicBlockEdge &);

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

  // PRE
  struct AvailableValueInBlock {
    /// BB - The basic block in question.
    BasicBlock *BB;
    enum ValType {
      SimpleVal, // A simple offsetted value that is accessed.
      LoadVal,   // A value produced by a load.
      MemIntrin, // A memory intrinsic which is loaded from.
      UndefVal   // A UndefValue representing a value from dead block (which
                 // is not yet physically removed from the CFG).
    };

    /// V - The value that is live out of the block.
    PointerIntPair<Value *, 2, ValType> Val;

    /// Offset - The byte offset in Val that is interesting for the load
    /// coercion.
    unsigned Offset;

    static AvailableValueInBlock get(BasicBlock *BB, Value *V,
                                     unsigned Offset = 0) {
      AvailableValueInBlock Res;
      Res.BB = BB;
      Res.Val.setPointer(V);
      Res.Val.setInt(SimpleVal);
      Res.Offset = Offset;
      return Res;
    }

    static AvailableValueInBlock getMI(BasicBlock *BB, MemIntrinsic *MI,
                                       unsigned Offset = 0) {
      AvailableValueInBlock Res;
      Res.BB = BB;
      Res.Val.setPointer(MI);
      Res.Val.setInt(MemIntrin);
      Res.Offset = Offset;
      return Res;
    }

    static AvailableValueInBlock getLoad(BasicBlock *BB, LoadInst *LI,
                                         unsigned Offset = 0) {
      AvailableValueInBlock Res;
      Res.BB = BB;
      Res.Val.setPointer(LI);
      Res.Val.setInt(LoadVal);
      Res.Offset = Offset;
      return Res;
    }

    static AvailableValueInBlock getUndef(BasicBlock *BB) {
      AvailableValueInBlock Res;
      Res.BB = BB;
      Res.Val.setPointer(nullptr);
      Res.Val.setInt(UndefVal);
      Res.Offset = 0;
      return Res;
    }

    bool isSimpleValue() const { return Val.getInt() == SimpleVal; }
    bool isCoercedLoadValue() const { return Val.getInt() == LoadVal; }
    bool isMemIntrinValue() const { return Val.getInt() == MemIntrin; }
    bool isUndefValue() const { return Val.getInt() == UndefVal; }

    Value *getSimpleValue() const {
      assert(isSimpleValue() && "Wrong accessor");
      return Val.getPointer();
    }

    LoadInst *getCoercedLoadValue() const {
      assert(isCoercedLoadValue() && "Wrong accessor");
      return cast<LoadInst>(Val.getPointer());
    }

    MemIntrinsic *getMemIntrinValue() const {
      assert(isMemIntrinValue() && "Wrong accessor");
      return cast<MemIntrinsic>(Val.getPointer());
    }

    /// Emit code into this block to adjust the value defined here to the
    /// specified type. This handles various coercion cases.
    Value *MaterializeAdjustedValue(Instruction *I, NewGVN &) const;
  };
  typedef SmallVector<AvailableValueInBlock, 64> AvailValInBlkVect;
  typedef SmallVector<std::pair<BasicBlock *, const Expression *>, 64>
      UnavailBlkVect;
  typedef SmallDenseMap<const BasicBlock *, AvailableValueInBlock>
      AvailValInBlkMap;
  typedef DenseMap<const BasicBlock *, char> FullyAvailableMap;

  bool isValueFullyAvailableInBlock(BasicBlock *, FullyAvailableMap &,
                                    uint32_t);
  Value *findPRELeader(Value *, const BasicBlock *, const Value *);
  Value *findPRELeader(const Expression *, const BasicBlock *, const Value *);
  bool phiTranslateArguments(const BasicExpression *, BasicExpression *,
                             const BasicBlock *, const Value *);
  MemoryAccess *phiTranslateMemoryAccess(MemoryAccess *, const BasicBlock *);
  const Expression *phiTranslateExpression(const Expression *E, BasicBlock *,
                                           BasicBlock *, const Value *);
  BasicBlock *splitCriticalEdges(BasicBlock *, BasicBlock *);
  void analyzeAvailability(Instruction *, AvailValInBlkVect &,
                           UnavailBlkVect &);
  bool performPRE(Instruction *, AvailValInBlkVect &, UnavailBlkVect &);
  bool performPREOnClass(CongruenceClass *);
  Value *constructSSAForSet(Instruction *,
                            SmallVectorImpl<AvailableValueInBlock> &);
  void valueNumberNewInstruction(Value *);
  void valueNumberNewInstructionToValue(Value *, Value *);
  const Expression *trySimplifyPREExpression(const Expression *,
                                             const BasicBlock *);
  Value *regenerateExpression(const Expression *, BasicBlock *);
  void topoVisitCongruenceClass(CongruenceClass *,
                                SmallDenseMap<CongruenceClass *, unsigned> &,
                                SmallPtrSetImpl<CongruenceClass *> &);
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
INITIALIZE_PASS_DEPENDENCY(MemorySSALazy)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
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
  if (CC && CC->expression) {
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
    return CC->expression;
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
    Value *V = SimplifyGEPInst(ArrayRef<Value *>(E->ops_begin(), E->ops_end()),
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

    Value *V =
        ConstantFoldInstOperands(E->getOpcode(), E->getType(), C, *DL, TLI);
    if (V) {
      if (const Expression *SimplifiedE = checkSimplificationResults(E, I, V))
        return SimplifiedE;
    }
  }
  return E;
}

const Expression *NewGVN::uniquifyExpression(const Expression *E) {
  auto P = UniquedExpressions.insert(E);
  if (!P.second) {
    return *(P.first);
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
  return cast<VariableExpression>(uniquifyExpression(E));
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
  return cast<ConstantExpression>(uniquifyExpression(E));
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
  for (const auto &Member : CC->equivalences) {
    if (DT->dominates(Member.Edge, B))
      return Member.Val;
  }
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
  for (const auto &Member : CC->equivalences) {
    if ((Member.Edge.getStart() == B.getStart() &&
         Member.Edge.getEnd() == B.getEnd()) ||
        DT->dominates(Member.Edge, B.getEnd()))
      return Member.Val;
  }
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
    return std::make_pair(CC->leader, false);
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
  Src = ConstantExpr::getGetElementPtr(Type::getInt8Ty(Src->getContext()), Src,
                                       OffsetCst);
  Src = ConstantExpr::getBitCast(Src, PointerType::get(LoadTy, AS));
  if (ConstantFoldLoadFromConstPtr(Src, *DL))
    return Offset;
  return -1;
}

const Expression *NewGVN::performSymbolicStoreEvaluation(Instruction *I,
                                                         const BasicBlock *B) {
  StoreInst *SI = cast<StoreInst>(I);
  const Expression *E = createStoreExpression(SI, MSSA->getMemoryAccess(SI), B);
  return E;
}

const Expression *NewGVN::performSymbolicLoadCoercion(
    Type *LoadType, Value *LoadPtr, LoadInst *LI, Instruction *DepInst,
    MemoryAccess *DefiningAccess, const BasicBlock *B) {
  assert((!LI || LI->isSimple()) && "Not a simple load");

  if (StoreInst *DepSI = dyn_cast<StoreInst>(DepInst)) {
    Value *LoadAddressLeader = lookupOperandLeader(LoadPtr, LI, B).first;
    Value *StoreAddressLeader =
        lookupOperandLeader(DepSI->getPointerOperand(), DepSI, B).first;
    Value *StoreVal = DepSI->getValueOperand();
    if (StoreVal->getType() == LoadType &&
        LoadAddressLeader == StoreAddressLeader) {
      return createVariableOrConstant(DepSI->getValueOperand(), B);
    } else {
      int Offset = analyzeLoadFromClobberingStore(LoadType, LoadPtr, DepSI);
      if (Offset >= 0)
        return createCoercibleLoadExpression(
            LoadType, LoadPtr, LI, DefiningAccess, (unsigned)Offset, DepSI, B);
    }
  } else if (LoadInst *DepLI = dyn_cast<LoadInst>(DepInst)) {
    int Offset = analyzeLoadFromClobberingLoad(LoadType, LoadPtr, DepLI);
    if (Offset >= 0)
      return createCoercibleLoadExpression(
          LoadType, LoadPtr, LI, DefiningAccess, (unsigned)Offset, DepLI, B);
  } else if (MemIntrinsic *DepMI = dyn_cast<MemIntrinsic>(DepInst)) {
    int Offset = analyzeLoadFromClobberingMemInst(LoadType, LoadPtr, DepMI);
    if (Offset >= 0)
      return createCoercibleLoadExpression(
          LoadType, LoadPtr, LI, DefiningAccess, (unsigned)Offset, DepMI, B);
  }
  // If this load really doesn't depend on anything, then we must be loading
  // an
  // undef value.  This can happen when loading for a fresh allocation with
  // no
  // intervening stores, for example.
  else if (isa<AllocaInst>(DepInst) || isMallocLikeFn(DepInst, TLI))
    return createConstantExpression(UndefValue::get(LoadType), false);

  // If this load occurs either right after a lifetime begin,
  // then the loaded value is undefined.
  else if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(DepInst)) {
    if (II->getIntrinsicID() == Intrinsic::lifetime_start)
      return createConstantExpression(UndefValue::get(LoadType), false);
  }
  // If this load follows a calloc (which zero initializes memory),
  // then the loaded value is zero
  else if (isCallocLikeFn(DepInst, TLI)) {
    return createConstantExpression(Constant::getNullValue(LoadType), false);
  }

  return nullptr;
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
      const Expression *CoercionResult =
          performSymbolicLoadCoercion(LI->getType(), LI->getPointerOperand(),
                                      LI, DefiningInst, DefiningAccess, B);
      if (CoercionResult)
        return CoercionResult;
    }
  } else {
    BasicBlock *LoadBlock = LI->getParent();
    MemoryAccess *LoadAccess = MSSA->getMemoryAccess(LI);
    // Okay, so uh, we couldn't use the defining access to grab a value out of
    // See if we can reuse any of it's uses by widening a load.
    for (const MemoryAccess *MA : DefiningAccess->users()) {
      if (MA == LoadAccess)
        continue;
      if (isa<MemoryPhi>(MA))
        continue;
      Instruction *DefiningInst = MA->getMemoryInst();
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

  for (const Value *Arg : E->operands())
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
  return uniquifyExpression(E);
}

/// markDominatedSingleUseEquivalences - Go through all uses of From, and mark
/// them as equivalent to To if From dominates them

void NewGVN::markDominatedSingleUserEquivalences(Value *From, Value *To,
                                                 const BasicBlockEdge &Root) {
  for (const auto &U : From->uses()) {
    // If From occurs as a phi node operand then the use implicitly lives in
    // the
    // corresponding incoming block.  Otherwise it is the block containing the
    // user that must be dominated by Root.
    if (DT->dominates(Root, U)) {
      SingleUserEquivalences[U.getUser()] = {U.getOperandNo(), To};
      // Mark the users as touched
      if (Instruction *I = dyn_cast<Instruction>(U.getUser()))
        TouchedInstructions.set(InstrDFS[I]);
    }
  }
}

/// replaceAllDominatedUsesWith - Replace all uses of 'From' with 'To'
/// if the use is dominated by the given basic block.  Returns the
/// number of uses that were replaced.
unsigned NewGVN::replaceAllDominatedUsesWith(Value *From, Value *To,
                                             const BasicBlockEdge &Root) {
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
/// There is an edge from 'Src' to 'Dst'.  Return
/// true if every path from the entry block to 'Dst' passes via this edge.  In
/// particular 'Dst' must not be reachable via another edge from 'Src'.
static bool isOnlyReachableViaThisEdge(const BasicBlockEdge &E) {
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
    CC->equivalences.emplace_back(RHS, Root, !RootDominatesEnd);

    if (!RootDominatesEnd)
      markDominatedSingleUserEquivalences(LHS, RHS, Root);
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
        if (CC->members.size() == 1 && 0) {
          unsigned NumReplacements =
              replaceAllDominatedUsesWith(CC->leader, NotVal, Root);
          Changed |= NumReplacements > 0;
          NumGVNEqProp += NumReplacements;
        }
        // Ensure that any instruction in scope that gets the "A < B"
        // value number is replaced with false.

        CC->equivalences.emplace_back(NotVal, Root, !RootDominatesEnd);

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
void NewGVN::performCongruenceFinding(Value *V, const Expression *E) {

  ValueToExpression[V] = E;
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
  } else if (const VariableExpression *VE = dyn_cast<VariableExpression>(E)) {
    EClass = ValueToClass[VE->getVariableValue()];
  } else {
    auto lookupResult = ExpressionToClass.insert({E, nullptr});

    // If it's not in the value table, create a new congruence class
    if (lookupResult.second) {
      CongruenceClass *NewClass = createCongruenceClass(NULL, E);
      auto place = lookupResult.first;
      place->second = NewClass;

      // Constants and variables should always be made the leader
      if (const ConstantExpression *CE = dyn_cast<ConstantExpression>(E))
        NewClass->leader = CE->getConstantValue();
      else if (const VariableExpression *VE = dyn_cast<VariableExpression>(E))
        NewClass->leader = VE->getVariableValue();
      else if (const StoreExpression *SE = dyn_cast<StoreExpression>(E))
        NewClass->leader = SE->getStoreInst()->getValueOperand();
      else
        NewClass->leader = V;

      EClass = NewClass;
      DEBUG(dbgs() << "Created new congruence class for " << *V
                   << " using expression " << *E << " at " << NewClass->id
                   << "\n");
      DEBUG(dbgs() << "Hash value was " << E->getHashValue() << "\n");
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
        const CoercibleLoadExpression *L = cast<CoercibleLoadExpression>(E);
        VClass->coercible_members.erase(V);
        EClass->coercible_members.insert(V);
        CoercionInfo.insert(
            std::make_pair(V, std::make_pair(L->getOffset(), L->getSrc())));
      } else {
        VClass->members.erase(V);
        EClass->members.insert(V);
      }

      // See if we have any pending equivalences for this class.
      // We should only end up with pending equivalences for comparison
      // instructions.
      if (E && isa<CmpInst>(V)) {
        auto Pending = PendingEquivalences.equal_range(E);
        for (auto PI = Pending.first, PE = Pending.second; PI != PE; ++PI)
          EClass->equivalences.emplace_back(PI->second);
        PendingEquivalences.erase(Pending.first, Pending.second);
      }

      ValueToClass[V] = EClass;
      // See if we destroyed the class or need to swap leaders
      if ((VClass->members.empty() && VClass->coercible_members.empty()) &&
          VClass != InitialClass) {
        if (VClass->expression) {
          VClass->dead = true;

          DEBUG(dbgs() << "Erasing expression " << *E << " from table\n");
          // bool wasE = *E == *VClass->expression;
          ExpressionToClass.erase(VClass->expression);
          // if (wasE)
          //   lookupMap->insert({E, EClass});
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
  if (ReachableEdges.insert({From, To}).second) {
    // If this block wasn't reachable before, all instructions are touched
    if (ReachableBlocks.insert(To).second) {
      DEBUG(dbgs() << "Block " << getBlockName(To) << " marked reachable\n");
      const auto &InstRange = BlockInstRange.lookup(To);
      TouchedInstructions.set(InstRange.first, InstRange.second);
    } else {
      DEBUG(dbgs() << "Block " << getBlockName(To)
            << " was reachable, but new edge {" << getBlockName(From) << "," << getBlockName(To) << "} to it found\n");
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
      propagateEquality(Cond, ConstantInt::getTrue(TrueSucc->getContext()),
                        {B, TrueSucc});

      propagateEquality(Cond, ConstantInt::getFalse(FalseSucc->getContext()),
                        {B, FalseSucc});
      updateReachableEdge(B, TrueSucc);
      updateReachableEdge(B, FalseSucc);
    }
  } else if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
    // For switches, propagate the case values into the case
    // destinations.

    // Remember how many outgoing edges there are to every successor.
    SmallDenseMap<BasicBlock *, unsigned, 16> SwitchEdges;

    Value *SwitchCond = SI->getCondition();
    Value *CondEvaluated = findConditionEquivalence(SwitchCond, B);
    // See if we were able to turn this switch statement into a constant
    if (CondEvaluated && isa<ConstantInt>(CondEvaluated)) {
      InvolvedInEquivalence.set(InstrDFS[TI]);
      ConstantInt *CondVal = cast<ConstantInt>(CondEvaluated);
      // We should be able to get case value for this
      auto CaseVal = SI->findCaseValue(CondVal);
      // Now get where it goes and mark it reachable
      BasicBlock *TargetBlock = CaseVal.getCaseSuccessor();
      updateReachableEdge(B, TargetBlock);
      ++SwitchEdges[TargetBlock];
    } else {
      for (unsigned i = 0, e = SI->getNumSuccessors(); i != e; ++i) {
        BasicBlock *TargetBlock = SI->getSuccessor(i);
        ++SwitchEdges[TargetBlock];
        updateReachableEdge(B, TargetBlock);
      }
    }
    // Regardless of answers, propagate equalities for case values
    for (auto i = SI->case_begin(), e = SI->case_end(); i != e; ++i) {
      BasicBlock *TargetBlock = i.getCaseSuccessor();
      if (SwitchEdges.lookup(TargetBlock) == 1) {
        const BasicBlockEdge E(B, TargetBlock);
        propagateEquality(SwitchCond, i.getCaseValue(), E);
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

std::pair<unsigned, unsigned>
NewGVN::calculateDominatedInstRange(const DomTreeNode *DTN) {
  SmallVector<std::pair<const DomTreeNode *, DomTreeNode::const_iterator>, 32>
      WorkStack;
  WorkStack.emplace_back(DTN, DTN->begin());
  unsigned MaxSeen = 0;
  while (!WorkStack.empty()) {
    const auto &Back = WorkStack.back();
    const DomTreeNode *Node = Back.first;
    auto ChildIt = Back.second;
    auto Result = BlockInstRange.lookup(Node->getBlock());
    MaxSeen = std::max(MaxSeen, Result.second);
    // If we visited all of the children of this node, "recurse" back up the
    // stack setting the ranges
    if (ChildIt == Node->end()) {
      auto Result = BlockInstRange.lookup(Node->getBlock());
      DominatedInstRange[DTN] = {Result.first, MaxSeen};
      WorkStack.pop_back();
      if (WorkStack.empty())
        return std::make_pair(Result.first, MaxSeen);
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

  const auto &DominatedRange = DominatedInstRange.find(DTN);
  std::pair<unsigned, unsigned> Result;
  if (DominatedRange == DominatedInstRange.end()) {
    Result = calculateDominatedInstRange(DTN);
  } else
    Result = DominatedRange->second;

  // Touch all the downstream dominated instructions that used equivalences.
  for (int InstrNum = InvolvedInEquivalence.find_next(Result.first - 1);
       InstrNum != -1 && (Result.second - InstrNum > 0);
       InstrNum = InvolvedInEquivalence.find_next(InstrNum)) {
    // TODO: We could do confluence block checks here.
    TouchedInstructions.set(InstrNum);
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
  }

  ArgRecycler.clear(ExpressionAllocator);
  ExpressionAllocator.Reset();
  CongruenceClasses.clear();
  ExpressionToClass.clear();
  ValueToExpression.clear();
  UniquedExpressions.clear();
  ReachableBlocks.clear();
  ReachableEdges.clear();
  ProcessedCount.clear();
  ProcessedBlockCount.clear();
  DFSDomMap.clear();
  InstrDFS.clear();
  InstructionsToErase.clear();

  DFSToInstr.clear();
  BlockInstRange.clear();
  TouchedInstructions.clear();
  InvolvedInEquivalence.clear();
  CoercionInfo.clear();
  CoercionForwarding.clear();
  DominatedInstRange.clear();
  SingleUserEquivalences.clear();
  PendingEquivalences.clear();
  PredCache.clear();
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

void NewGVN::topoVisitCongruenceClass(
    CongruenceClass *CC, SmallDenseMap<CongruenceClass *, unsigned> &UsedCount,
    SmallPtrSetImpl<CongruenceClass *> &Visited) {
  Visited.insert(CC);
  // Visit the classes of the values of the operands of the leader set
  for (auto Member : CC->members) {
    if (User *U = dyn_cast<User>(Member)) {
      for (auto &I : U->operands()) {
        CongruenceClass *OperandCC = ValueToClass.lookup(I);
        if (OperandCC) {
          UsedCount[OperandCC] += 1;
          if (!Visited.count(OperandCC))
            topoVisitCongruenceClass(OperandCC, UsedCount, Visited);
        }
      }
    }
  }
}

/// runOnFunction - This is the main transformation entry point for a
/// function.
bool NewGVN::runOnFunction(Function &F) {
  bool Changed = false;
  if (skipOptnoneFunction(F))
    return false;

  MD = getAnalysisIfAvailable<MemoryDependenceAnalysis>();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  DL = &F.getParent()->getDataLayout();
  AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  TLI = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  AA = &getAnalysis<AliasAnalysis>();
  SplitAllCriticalEdges(F, CriticalEdgeSplittingOptions(AA, DT));  
  MSSA = &getAnalysis<MemorySSALazy>().getMSSA();
  MSSAWalker = MSSA->buildMemorySSA(AA, DT);

  unsigned ICount = 0;
  // Count number of instructions for sizing of hash tables, and come
  // up with a global dfs numbering for instructions
  
  SmallPtrSet<BasicBlock *, 16> VisitedBlocks;
  ReversePostOrderTraversal<Function*> RPOT(&F);
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

  UniquedExpressions.resize(ICount + 1);
  TouchedInstructions.resize(ICount + 1);
  InvolvedInEquivalence.resize(ICount + 1);

  // Ensure we don't end up resizing the expressionToClass map, as
  // that can be quite expensive. At most, we have one expression per
  // instruction.
  ExpressionToClass.resize(ICount + 1);
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
        //#ifndef NDEBUG
        if (ProcessedBlockCount.count(CurrBlock) == 0) {
          ProcessedBlockCount.insert({CurrBlock, 1});
        } else {
          ProcessedBlockCount[CurrBlock] += 1;
          assert(ProcessedBlockCount[CurrBlock] < 100 &&
                 "Seem to have processed the same block a lot\n");
          if (ProcessedBlockCount[CurrBlock] >= 100)
            report_fatal_error("Processed block too many times");
        }
        //#endif
      }
      TouchedInstructions.reset(InstrNum);

      DEBUG(dbgs() << "Processing instruction " << *I << "\n");
      if (I->use_empty() && !I->getType()->isVoidTy()) {
        DEBUG(dbgs() << "Skipping unused instruction\n");
        if (isInstructionTriviallyDead(I, TLI))
          markInstructionForDeletion(I);
        continue;
      }

// This is done in case something eliminates the instruction
// along the way.

#ifndef NDEBUG
      if (ProcessedCount.count(I) == 0) {
        ProcessedCount.insert({I, 1});
      } else {
        ProcessedCount[I] += 1;
        assert(ProcessedCount[I] < 100 &&
               "Seem to have processed the same instruction a lot");
      }
#endif
      if (!I->isTerminator()) {
        const Expression *Symbolized = performSymbolicEvaluation(I, CurrBlock);
        if (Symbolized && Symbolized->usedEquivalence())
          InvolvedInEquivalence.set(InstrDFS[I]);
        performCongruenceFinding(I, Symbolized);
      } else {
        processOutgoingEdges(dyn_cast<TerminatorInst>(I), CurrBlock);
      }
    }
  }

  Changed |= eliminateInstructions(F);

  // The ideal ordering for processing is not quite topological ordering,
  // because there are multiple roots.  It is essentially "group every vertex
  // that depends on a given vertex together after that vertex", which is not
  // the same.  It is in fact, an NP complete problem, and given that the graph
  // may be cyclic anyway, we order the congruence classes by how many things
  // depend on them.  This is a good approximation, and will cut down the number
  // of iterations.

  SmallDenseMap<CongruenceClass *, unsigned> UsedCount;
  SmallPtrSet<CongruenceClass *, 16> VisitedClasses;

  for (int i = CongruenceClasses.size() - 1; i >= 0; --i) {
    CongruenceClass *CC = CongruenceClasses[i].get();
    if (CC == InitialClass || CC->dead || VisitedClasses.count(CC))
      continue;
    topoVisitCongruenceClass(CC, UsedCount, VisitedClasses);
  }
  SmallVector<CongruenceClass *, 16> Worklist;
  for (auto &CC : CongruenceClasses)
    Worklist.push_back(CC.get());
  // std::sort(Worklist.begin(), Worklist.end(),
  //           [&UsedCount](CongruenceClass *&A, CongruenceClass *&B) {
  //             return UsedCount[A] > UsedCount[B];
  //           });

#if 0
  bool PREChanged = true;
  while (PREChanged) {
    PREChanged = false;
    // FIXME: Handle added congruence classes
    if (Worklist.size() != CongruenceClasses.size())
      Worklist.insert(Worklist.end(),
                      CongruenceClasses.begin() + (Worklist.size() - 1),
                      CongruenceClasses.end());

    for (auto CC : Worklist) {
      if (CC == InitialClass || CC->dead)
        continue;

      PREChanged |= performPREOnClass(CC);
    }
  }
  
  PREValueForwarding.clear();

  Changed |= PREChanged;
#endif
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
  bool Equivalence;
  bool Coercible;
  // Only one of these will be set
  Value *Val;
  Use *U;
  ValueDFS()
      : DFSIn(0), DFSOut(0), LocalNum(0), Equivalence(false), Coercible(false),
        Val(nullptr), U(nullptr) {}

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

    std::pair<int, int> DFSPair = DFSDomMap[BB];
    assert(DFSPair.first != -1 && DFSPair.second != -1 && "Invalid DFS Pair");
    VD.DFSIn = DFSPair.first;
    VD.DFSOut = DFSPair.second;
    VD.Val = D;
    VD.Coercible = Coercible;
    // If it's an instruction, use the real local dfs number,
    if (Instruction *I = dyn_cast<Instruction>(D))
      VD.LocalNum = InstrDFS[I];
    else
      llvm_unreachable("Should have been an instruction");

    DFSOrderedSet.emplace_back(VD);

    // Now add the users
    for (auto &U : D->uses()) {
      if (Instruction *I = dyn_cast<Instruction>(U.getUser())) {
        // If we the the same use we have an equivalence for, we will replace it
        // with the equivalence later

        const auto &LI = SingleUserEquivalences.find(I);
        if (LI != SingleUserEquivalences.end()) {
          if (U.getOperandNo() == LI->second.first)
            continue;
        }

        ValueDFS VD;
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
        std::pair<int, int> DFSPair = DFSDomMap[IBlock];
        VD.DFSIn = DFSPair.first;
        VD.DFSOut = DFSPair.second;
        VD.U = &U;
        DFSOrderedSet.emplace_back(VD);
      }
    }
  }
}

void NewGVN::convertDenseToDFSOrdered(CongruenceClass::EquivalenceSet &Dense,
                                      std::vector<ValueDFS> &DFSOrderedSet) {
  for (const auto &D : Dense) {
    // We don't have the machinery to handle edge only equivalences yet.
    // We should be using control regions to know where they are valid.
    if (D.EdgeOnly)
      continue;
    std::pair<int, int> &DFSPair = DFSDomMap[D.Edge.getEnd()];
    assert(DFSPair.first != -1 && DFSPair.second != -1 && "Invalid DFS Pair");
    ValueDFS VD;
    VD.DFSIn = DFSPair.first;
    VD.DFSOut = DFSPair.second;
    VD.Equivalence = true;

    // If it's an instruction, use the real local dfs number.
    // If it's a value, it *must* have come from equality propagation,
    // and thus we know it is valid for the entire block.  By giving
    // the local number as -1, it should sort before the instructions
    // in that block.
    if (Instruction *I = dyn_cast<Instruction>(D.Val))
      VD.LocalNum = InstrDFS[I];
    else
      VD.LocalNum = -1;

    VD.Val = D.Val;
    DFSOrderedSet.emplace_back(VD);
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

  // Compute which bits of the stored value are being used by the load.  Convert
  // to an integer type to start with.
  if (SrcVal->getType()->getScalarType()->isPointerTy()) {
    SrcVal =
        Builder.CreatePtrToInt(SrcVal, DL->getIntPtrType(SrcVal->getType()));
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
  Src = ConstantExpr::getGetElementPtr(Type::getInt8Ty(Src->getContext()), Src,
                                       OffsetCst);
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

  bool AnythingReplaced = false;

  // Since we are going to walk the domtree anyway, and we can't guarantee the
  // DFS numbers are updated, we compute some ourselves.

  DT->updateDFSNumbers();
  for (auto &B : F) {
    if (!ReachableBlocks.count(&B)) {
      for (const auto S : successors(&B)) {
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
    DomTreeNode *Node = DT->getNode(&B);
    if (Node)
      DFSDomMap[&B] = {Node->getDFSNumIn(), Node->getDFSNumOut()};
  }

  for (unsigned i = 0, e = CongruenceClasses.size(); i != e; ++i) {
    CongruenceClass *CC = CongruenceClasses[i].get();
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
          replaceAllDominatedUsesWith(M, Equiv.Val, Equiv.Edge);

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
          AnythingReplaced = true;

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
            replaceAllDominatedUsesWith(CC->leader, Equiv.Val, Equiv.Edge);
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
          if (Instruction *ReplacedInst =
                  dyn_cast<Instruction>(MemberUse->get()))
            patchReplacementInstruction(ReplacedInst, Result);
          assert(isa<Instruction>(MemberUse->getUser()));
          MemberUse->set(Result);
          AnythingReplaced = true;
        }
      }
    }
    // Cleanup the congruence class
    SmallPtrSet<Value *, 4> MembersLeft;
    for (auto MI = CC->members.begin(), ME = CC->members.end(); MI != ME;) {
      auto CurrIter = MI;
      ++MI;
      Value *Member = *CurrIter;
      if (Member->getType()->isVoidTy()) {
        MembersLeft.insert(Member);
        continue;
      }

      if (Instruction *MemberInst = dyn_cast<Instruction>(Member)) {
        if (isInstructionTriviallyDead(MemberInst)) {
          // TODO: Don't mark loads of undefs.
          markInstructionForDeletion(MemberInst);
          continue;
        }
      }
      MembersLeft.insert(Member);
    }
    CC->members.swap(MembersLeft);
    CC->coercible_members.clear();
  }
  for (auto &Equivs : SingleUserEquivalences) {
    DEBUG(dbgs() << "Using single user equivalence to replace ");
    DEBUG(dbgs() << *Equivs.first << " with " << *Equivs.second.second);
    DEBUG(dbgs() << "\n");
    Equivs.first->getOperandUse(Equivs.second.first).set(Equivs.second.second);
  }

  return AnythingReplaced;
}

Value *
NewGVN::AvailableValueInBlock::MaterializeAdjustedValue(Instruction *I,
                                                        NewGVN &gvn) const {
  if (!isa<LoadInst>(I)) {
    assert((isSimpleValue() || isUndefValue()) &&
           "Should have been a simple or undef value for a non-load!");
    if (isSimpleValue())
      return getSimpleValue();
    else
      return UndefValue::get(I->getType());
  }

  Value *Res;
  Type *LoadTy = I->getType();

  if (isSimpleValue()) {
    Res = getSimpleValue();
    if (Res->getType() != LoadTy) {
      Res = gvn.getStoreValueForLoad(Res, Offset, LoadTy, BB->getTerminator());

      DEBUG(dbgs() << "GVN COERCED NONLOCAL VAL:\nOffset: " << Offset << "  "
                   << *getSimpleValue() << '\n' << *Res << '\n' << "\n\n\n");
    }
  } else if (isCoercedLoadValue()) {
    LoadInst *Load = getCoercedLoadValue();
    if (Load->getType() == LoadTy && Offset == 0) {
      Res = Load;
    } else {
      Res = gvn.getLoadValueForLoad(Load, Offset, LoadTy, BB->getTerminator());

      DEBUG(dbgs() << "GVN COERCED NONLOCAL LOAD:\nOffset: " << Offset << "  "
                   << *getCoercedLoadValue() << '\n' << *Res << '\n'
                   << "\n\n\n");
    }
  } else if (isMemIntrinValue()) {
    Res = gvn.getMemInstValueForLoad(getMemIntrinValue(), Offset, LoadTy,
                                     BB->getTerminator());
    DEBUG(dbgs() << "GVN COERCED NONLOCAL MEM INTRIN:\nOffset: " << Offset
                 << "  " << *getMemIntrinValue() << '\n' << *Res << '\n'
                 << "\n\n\n");
  } else {
    assert(isUndefValue() && "Should be UndefVal");
    DEBUG(dbgs() << "GVN COERCED NONLOCAL Undef:\n";);
    return UndefValue::get(LoadTy);
  }
  return Res;
}

Value *NewGVN::constructSSAForSet(
    Instruction *I, SmallVectorImpl<AvailableValueInBlock> &ValuesPerBlock) {
  // Check for the fully redundant, dominating load case.  In this case, we can
  // just use the dominating value directly.
  if (ValuesPerBlock.size() == 1 &&
      DT->properlyDominates(ValuesPerBlock[0].BB, I->getParent())) {
    assert(!ValuesPerBlock[0].isUndefValue() && "Dead BB dominate this block");
    return ValuesPerBlock[0].MaterializeAdjustedValue(I, *this);
  }
  // Otherwise, we have to construct SSA form.
  SmallVector<PHINode *, 8> NewPHIs;
  SSAUpdater SSAUpdate(&NewPHIs);
  SSAUpdate.Initialize(I->getType(), I->getName());

  for (unsigned i = 0, e = ValuesPerBlock.size(); i != e; ++i) {
    const AvailableValueInBlock &AV = ValuesPerBlock[i];
    BasicBlock *BB = AV.BB;

    if (SSAUpdate.HasValueForBlock(BB))
      continue;

    SSAUpdate.AddAvailableValue(BB, AV.MaterializeAdjustedValue(I, *this));
  }

  // Perform PHI construction.
  Value *V = SSAUpdate.GetValueInMiddleOfBlock(I->getParent());

  // If new pointer PHI nodes were created, notify alias analysis.
  if (V->getType()->getScalarType()->isPointerTy()) {

    for (unsigned i = 0, e = NewPHIs.size(); i != e; ++i)
      AA->copyValue(I, NewPHIs[i]);

    // Now that we've copied information to the new PHIs, scan through
    // them again and inform alias analysis that we've added potentially
    // escaping uses to any values that are operands to these PHIs.
    for (unsigned i = 0, e = NewPHIs.size(); i != e; ++i) {
      PHINode *P = NewPHIs[i];
      for (unsigned ii = 0, ee = P->getNumIncomingValues(); ii != ee; ++ii) {
        unsigned jj = PHINode::getOperandNumForIncomingValue(ii);
        AA->addEscapingUse(P->getOperandUse(jj));
      }
    }
  }

  return V;
}

/// Split the critical edge connecting the given two blocks, and return
/// the block inserted to the critical edge.
BasicBlock *NewGVN::splitCriticalEdges(BasicBlock *Pred, BasicBlock *Succ) {
  BasicBlock *BB =
      SplitCriticalEdge(Pred, Succ, CriticalEdgeSplittingOptions(AA, DT));
  if (MD)
    MD->invalidateCachedPredecessors();
  return BB;
}

void NewGVN::valueNumberNewInstructionToValue(Value *New, Value *Old) {
  if (!ValueToClass.count(New))
    ValueToClass[New] = InitialClass;
  const Expression *NewExpr = createVariableExpression(Old, false);
  performCongruenceFinding(New, NewExpr);
}

void NewGVN::valueNumberNewInstruction(Value *V) {
  if (!ValueToClass.count(V))
    ValueToClass[V] = InitialClass;
  const Expression *NewExpr =
      performSymbolicEvaluation(V, cast<Instruction>(V)->getParent());
  performCongruenceFinding(V, NewExpr);
}

Value *NewGVN::regenerateExpression(const Expression *E, BasicBlock *BB) {
  Value *V = findPRELeader(E, BB, nullptr);
  if (V)
    return V;
  if (const LoadExpression *LE = dyn_cast<LoadExpression>(E)) {
    LoadInst *LI = new LoadInst(LE->getOperand(0), "loadpre", false,
                                LE->getAlignment(), BB->getTerminator());
    const_cast<LoadExpression *>(LE)->setLoadInst(LI);
    MSSA->addNewMemoryUse(LI, MemorySSA::InsertionPlace::End);

    return LI;
  } else if (const BasicExpression *BE = dyn_cast<BasicExpression>(E)) {
    unsigned Opcode = BE->getOpcode();
    if (Instruction::isBinaryOp(Opcode)) {
      BinaryOperator *BO = BinaryOperator::Create(
          Instruction::BinaryOps(Opcode), BE->getOperand(0), BE->getOperand(1),
          "binaryoppre", BB->getTerminator());
      // FIXME: Track NSW/NUW
      return BO;
    } else if (Opcode == Instruction::MemoryOps::GetElementPtr) {
      // It wants the pointee type, which is complex, and not equivalent to
      // getType on the original GEP
      Type *PointeeType =
          cast<SequentialType>(BE->getOperand(0)->getType()->getScalarType())
              ->getElementType();
      Value *GEP = nullptr;
      if (Value *V = SimplifyGEPInst(
              makeArrayRef(BE->ops_begin(), BE->ops_end()), *DL, TLI, DT, AC))
        GEP = V;
      else
        GEP = GetElementPtrInst::Create(
            PointeeType, BE->getOperand(0),
            makeArrayRef(BE->ops_begin(), BE->ops_end()).slice(1), "geppre",
            BB->getTerminator());
      // FIXME: Track inbounds
      return GEP;
    } else if (((Opcode & 0xff00) >> 8) == Instruction::ICmp) {
      CmpInst::Predicate Pred = (CmpInst::Predicate)(Opcode & 0xff);
      ICmpInst *Cmp = new ICmpInst(BB->getTerminator(), Pred, BE->getOperand(0),
                                   BE->getOperand(1), "icmppre");
      return Cmp;
    } else if (((Opcode & 0xff00) >> 8) == Instruction::FCmp) {
      CmpInst::Predicate Pred = (CmpInst::Predicate)(Opcode & 0xff);
      FCmpInst *Cmp = new FCmpInst(BB->getTerminator(), Pred, BE->getOperand(0),
                                   BE->getOperand(1), "fcmppre");
      return Cmp;
    } else
      llvm_unreachable("What!");
  }
  llvm_unreachable("What!");
}

bool NewGVN::performPRE(Instruction *I, AvailValInBlkVect &ValuesPerBlock,
                        UnavailBlkVect &UnavailableBlocks) {
  bool Changed = false;
  // Okay, we have *some* definitions of the value.  This means that the value
  // is available in some of our (transitive) predecessors.  Lets think about
  // doing PRE of this instruction.  This will involve inserting a new
  // instruction into the
  // predecessor when it's not available.  We could do this in general, but
  // prefer to not increase code size.  As such, we only do this when we know
  // that we only have to insert *one* load (which means we're basically moving
  // the load, not inserting a new one).

  // Decide whether PRE is profitable for this load.
  unsigned NumUnavailablePreds = UnavailableBlocks.size();
  assert(NumUnavailablePreds != 0 &&
         "Fully available value should already be eliminated!");

  // If this load is unavailable in multiple predecessors, reject it.
  // FIXME: If we could restructure the CFG, we could make a common pred with
  // all the preds that don't have an available LI and insert a new load into
  // that one block.
  if (NumUnavailablePreds > 2)
    return false;

  // Check if the load can safely be moved to all the unavailable predecessors.
  bool CanDoPRE = true;

  SmallVector<Instruction *, 8> NewInsts;
  // FIXME: Insert Can Be Avail stuff here
  if (!CanDoPRE) {
    while (!NewInsts.empty()) {
      Instruction *I = NewInsts.pop_back_val();
      if (MD)
        MD->removeInstruction(I);
      I->eraseFromParent();
    }
    // HINT: Don't revert the edge-splitting as following transformation may
    // also need to split these critical edges.
    return Changed;
  }

  // Okay, we can eliminate this load by inserting a reload in the predecessor
  // and using PHI construction to get the value in the other predecessors, do
  // it.
  DEBUG(dbgs() << "GVN REMOVING PRE INSTRUCTION: " << *I << '\n');
  DEBUG(if (!NewInsts.empty()) dbgs() << "INSERTED " << NewInsts.size()
                                      << " INSTS: " << *NewInsts.back()
                                      << '\n');

  // Assign value numbers to the new instructions.
  for (unsigned i = 0, e = NewInsts.size(); i != e; ++i) {
    valueNumberNewInstruction(NewInsts[i]);
  }

  for (auto &UnavailInfo : UnavailableBlocks)
    if (!UnavailInfo.second)
      return false;

  for (auto &UnavailInfo : UnavailableBlocks) {
    BasicBlock *UnavailablePred = UnavailInfo.first;

    Value *NewVal = regenerateExpression(UnavailInfo.second, UnavailablePred);
    valueNumberNewInstruction(NewVal);
    if (Instruction *NewInst = dyn_cast<Instruction>(NewVal)) {
      NewInst->setDebugLoc(I->getDebugLoc());
      LoadInst *LI = dyn_cast<LoadInst>(I);
      LoadInst *NewLoadInst = dyn_cast<LoadInst>(NewInst);
      // Transfer load tags
      if (LI && NewLoadInst) {
        AAMDNodes Tags;
        LI->getAAMetadata(Tags);
        if (Tags)
          NewLoadInst->setAAMetadata(Tags);
        if (MD)
          MD->invalidateCachedPointerInfo(NewLoadInst->getPointerOperand());
      }
    }

    // Add the newly created load.
    ValuesPerBlock.push_back(
        AvailableValueInBlock::get(UnavailablePred, NewVal));
    DEBUG(dbgs() << "GVN INSERTED " << *NewVal << '\n');
  }

  // Perform PHI construction.
  Value *V = constructSSAForSet(I, ValuesPerBlock);
  I->replaceAllUsesWith(V);
  PREValueForwarding[I] = V;
  // Value of V is the same as the value of the old instruction (or it would not
  // be redundant)
  valueNumberNewInstructionToValue(V, I);

  ValueToExpression[I] = ValueToExpression[V];
  // I no longer exists
  ValueToClass.lookup(I)->members.erase(I);
  if (isa<PHINode>(V))
    V->takeName(I);
  if (MD && V->getType()->getScalarType()->isPointerTy())
    MD->invalidateCachedPointerInfo(V);
  markInstructionForDeletion(I);
  ++NumPRELoad;
  return true;
}

void NewGVN::analyzeAvailability(Instruction *I,
                                 AvailValInBlkVect &ValuesPerBlock,
                                 UnavailBlkVect &UnavailableBlocks) {
  for (BasicBlock **PI = PredCache.GetPreds(I->getParent()); *PI; ++PI) {
    BasicBlock *P = *PI;
    if (!ReachableBlocks.count(P)) {
      ValuesPerBlock.push_back(AvailableValueInBlock::getUndef(P));
      continue;
    }
    const Expression *Before = ValueToExpression.lookup(I);
    // This can happen if it was too complex or complicated an expression for
    // GVN to analyze
    if (!Before)
      return;
    const Expression *E = phiTranslateExpression(Before, I->getParent(), P, I);
    Value *V = nullptr;
    if (E) {
      E = trySimplifyPREExpression(E, P);
      DEBUG(dbgs() << "After simplification, expression is " << *E << "\n");

      V = findPRELeader(E, P, I);
      // If we got a store, we can rip out the source value.
      // Otherwise, void type stuff is not acceptable
      if (V && V->getType()->isVoidTy()) {
        if (StoreInst *SI = dyn_cast<StoreInst>(V))
          V = SI->getValueOperand();
        else
          V = nullptr;
      }
    }

    if (!V)
      UnavailableBlocks.push_back({P, E});
    else
      ValuesPerBlock.push_back(AvailableValueInBlock::get(P, V));
  }
}

bool NewGVN::performPREOnClass(CongruenceClass *CC) {

  // FIXME: Only do some
  bool Changed = false;
  AvailValInBlkVect ValuesPerBlock;

  for (auto M : CC->members) {
    if (isa<PHINode>(M) || M->getType()->isVoidTy())
      continue;
    if (!isa<GetElementPtrInst>(M) && !isa<LoadInst>(M))
      continue;

    if (Instruction *I = dyn_cast<Instruction>(M)) {
      UnavailBlkVect UnavailableBlocks;
      BasicBlock *IBlock = I->getParent();
      if (PredCache.GetNumPreds(IBlock) == 0)
        continue;
      analyzeAvailability(I, ValuesPerBlock, UnavailableBlocks);

      if ((UnavailableBlocks.size() + ValuesPerBlock.size()) !=
          PredCache.GetNumPreds(IBlock))
        continue;
#if 1
      // If we have no predecessors that produce a known value for this load,
      // exit early.
      if (ValuesPerBlock.empty())
        continue;
#endif
      // Step 3: Eliminate fully redundancy.
      //
      // If all of the instructions we depend on produce a known value for this
      // load, then it is fully redundant and we can use PHI insertion to
      // compute
      // its value.  Insert PHIs and remove the fully redundant value now.
      if (UnavailableBlocks.empty()) {
        DEBUG(dbgs() << "GVN REMOVING INSTRUCTION: " << *I << '\n');

        // Perform PHI construction.
        Value *V = constructSSAForSet(I, ValuesPerBlock);
        I->replaceAllUsesWith(V);
        PREValueForwarding[I] = V;
        valueNumberNewInstruction(V);
        ValueToClass.lookup(I)->members.erase(I);
        ValueToExpression[I] = ValueToExpression[V];
        if (isa<PHINode>(V))
          V->takeName(I);
        if (MD && V->getType()->getScalarType()->isPointerTy())
          MD->invalidateCachedPointerInfo(V);
        markInstructionForDeletion(I);
        ++NumNewGVNPRE;
        Changed = true;
        continue;
      }
      // Step 4: Eliminate partial redundancy.
      if (!EnablePRE || !EnableLoadPRE)
        continue;
      Changed |= performPRE(I, ValuesPerBlock, UnavailableBlocks);
    }
  }
  return Changed;
}

// Find a leader for OP in BB.
Value *NewGVN::findPRELeader(Value *Op, const BasicBlock *BB,
                             const Value *MustDominate) {
  if (alwaysAvailable(Op))
    return Op;

  CongruenceClass *CC = ValueToClass[Op];
  if (!CC || CC == InitialClass)
    return 0;

  if (CC->leader && alwaysAvailable(CC->leader))
    return CC->leader;
  Value *Equiv = findDominatingEquivalent(CC, nullptr, BB);
  if (Equiv)
    return Equiv;

  for (auto M : CC->members) {
    if (M == MustDominate)
      continue;
    if (Instruction *I = dyn_cast<Instruction>(M))
      if (DT->dominates(I->getParent(), BB))
        return I;
  }
  return 0;
}

// Find a leader for OP in BB.
Value *NewGVN::findPRELeader(const Expression *E, const BasicBlock *BB,
                             const Value *MustDominate) {
  if (const ConstantExpression *CE = dyn_cast<ConstantExpression>(E))
    return CE->getConstantValue();
  else if (const VariableExpression *VE = dyn_cast<VariableExpression>(E))
    return findPRELeader(VE->getVariableValue(), BB, MustDominate);

  DEBUG(dbgs() << "Hash value was " << E->getHashValue() << "\n");

  const auto Result = ExpressionToClass.find(E);
  if (Result == ExpressionToClass.end())
    return 0;

  CongruenceClass *CC = Result->second;

  if (!CC || CC == InitialClass)
    return 0;

  if (CC->leader && (isa<Argument>(CC->leader) || isa<Constant>(CC->leader) ||
                     isa<GlobalValue>(CC->leader)))
    return CC->leader;
  Value *Equiv = findDominatingEquivalent(CC, nullptr, BB);
  if (Equiv)
    return Equiv;

  for (auto M : CC->members) {
    if (M == MustDominate)
      continue;
    if (Instruction *I = dyn_cast<Instruction>(M))
      if (DT->dominates(I->getParent(), BB))
        return I;
  }
  return 0;
}

MemoryAccess *NewGVN::phiTranslateMemoryAccess(MemoryAccess *MA,
                                               const BasicBlock *Pred) {
  if (MemoryPhi *MP = dyn_cast<MemoryPhi>(MA)) {
    for (auto A : MP->operands()) {
      if (A.first == Pred) {
        return A.second;
      }
    }
    // We should have found something
    return nullptr;
  }
  return MA;
}

static Value *phiTranslateValue(Value *Incoming, const BasicBlock *Pred) {
  // Back translate if defined by a phi in this block
  PHINode *P = dyn_cast<PHINode>(Incoming);
  int Index = P->getBasicBlockIndex(Pred);
  if (Index != -1) {
    return P->getIncomingValue(Index);
  }
  // Not defined by a phi in this block
  return Incoming;
}

bool NewGVN::phiTranslateArguments(const BasicExpression *From,
                                   BasicExpression *To, const BasicBlock *Pred,
                                   const Value *MustDominate) {
  for (unsigned i = 0, e = From->getNumOperands(); i != e; ++i) {
    Value *Arg = From->getOperand(i);
    Value *Forwarded = PREValueForwarding.lookup(Arg);
    if (Forwarded)
      Arg = Forwarded;
    // Fold in immediate adds
    bool processedAdd = false;
    if (Instruction *I = dyn_cast<Instruction>(Arg)) {
      if (I->getOpcode() == Instruction::Add &&
          isa<PHINode>(I->getOperand(0)) &&
          isa<ConstantInt>(I->getOperand(1))) {

        Constant *RHS = cast<ConstantInt>(I->getOperand(1));
        bool isNSW = cast<BinaryOperator>(I)->hasNoSignedWrap();
        bool isNUW = cast<BinaryOperator>(I)->hasNoUnsignedWrap();
        Value *LHS = phiTranslateValue(I->getOperand(0), Pred);
        // If the PHI translated LHS is an add of a constant, fold the
        // immediates.
        if (BinaryOperator *BOp = dyn_cast<BinaryOperator>(LHS))
          if (BOp->getOpcode() == Instruction::Add)
            if (ConstantInt *CI = dyn_cast<ConstantInt>(BOp->getOperand(1))) {
              LHS = BOp->getOperand(0);
              RHS = ConstantExpr::getAdd(RHS, CI);
              isNSW = isNUW = false;
            }
        // FIXME: Handle NSW
        const Expression *BinExpr = createBinaryExpression(
            I->getOpcode(), I->getType(), LHS, RHS, Pred);
        Value *Leader = findPRELeader(BinExpr, Pred, MustDominate);
        if (Leader) {
          Arg = Leader;
          processedAdd = true;
        }
      }
    }
    // If arg is still a phi node, and wasn't processed by the add processing
    // above, translate it now
    if (!processedAdd && isa<PHINode>(Arg)) {
      while (isa<PHINode>(Arg)) {
        Value *Before = Arg;
        Arg = phiTranslateValue(Arg, Pred);
        if (Arg == Before)
          break;
      }
    }

    Value *Leader = findPRELeader(Arg, Pred, MustDominate);
    if (Leader == nullptr)
      return false;
    To->setOperand(i, Leader);
  }
  return true;
}
const Expression *NewGVN::phiTranslateExpression(const Expression *E,
                                                 BasicBlock *Curr,
                                                 BasicBlock *Pred,
                                                 const Value *MustDominate) {
  const Expression *ResultExpr = nullptr;
  if (const LoadExpression *LE = dyn_cast<LoadExpression>(E)) {

    MemoryAccess *MA = phiTranslateMemoryAccess(LE->getDefiningAccess(), Pred);
    if (!MA)
      return nullptr;
    LoadExpression *NLE = createLoadExpression(LE->getType(), LE->getOperand(0),
                                               LE->getLoadInst(), MA, Pred);
    if (!phiTranslateArguments(LE, NLE, Pred, MustDominate))
      return nullptr;
    LoadInst *LI = LE->getLoadInst();
    AliasAnalysis::Location Loc;
    if (LI) {
      Loc = AA->getLocation(LI);
      Loc = Loc.getWithNewPtr(NLE->getOperand(0));
    } else {
      Loc.Ptr = NLE->getOperand(0);
    }
    MA = MSSAWalker->getClobberingMemoryAccess(MA, Loc);
    NLE->setDefiningAccess(MA);
    ResultExpr = NLE;
  } else if (const BasicExpression *BE = dyn_cast<BasicExpression>(E)) {
    BasicExpression *NBE =
        new (ExpressionAllocator) BasicExpression(BE->getNumOperands());
    NBE->setType(BE->getType());
    NBE->setOpcode(BE->getOpcode());
    NBE->allocateOperands(ArgRecycler, ExpressionAllocator);
    for (unsigned i = 0, e = BE->getNumOperands(); i != e; ++i)
      NBE->ops_push_back(nullptr);
    if (!phiTranslateArguments(BE, NBE, Pred, MustDominate))
      return nullptr;
    ResultExpr = NBE;
  }

  return ResultExpr;
}

const Expression *NewGVN::trySimplifyPREExpression(const Expression *E,
                                                   const BasicBlock *B) {
  const Expression *ResultExpr = E;
  // This must come first, because LoadExpression's are BasicExpressions
  if (const LoadExpression *LE = dyn_cast<LoadExpression>(E)) {
    MemoryAccess *MA = LE->getDefiningAccess();
    if (isa<MemoryDef>(MA) && !MSSA->isLiveOnEntryDef(MA)) {
      const Expression *Temp = performSymbolicLoadCoercion(
          LE->getType(), LE->getOperand(0), LE->getLoadInst(),
          MA->getMemoryInst(), MA, B);
      if (Temp)
        ResultExpr = Temp;
    }
  } else if (const BasicExpression *BE = dyn_cast<BasicExpression>(E)) {
    unsigned Opcode = BE->getOpcode();
    if (Instruction::isBinaryOp(Opcode)) {
      Value *V = SimplifyBinOp(Opcode, BE->getOperand(0), BE->getOperand(1),
                               *DL, TLI, DT, AC);
      if (V) {
        if (Constant *C = dyn_cast<Constant>(V))
          ResultExpr = createConstantExpression(C, false);
        else if (alwaysAvailable(V))
          ResultExpr = createVariableExpression(V, false);
      }
    } else if (Opcode == Instruction::GetElementPtr) {
      Value *V = SimplifyGEPInst(makeArrayRef(BE->ops_begin(), BE->ops_end()),
                                 *DL, TLI, DT, AC);
      if (V) {
        if (Constant *C = dyn_cast<Constant>(V))
          ResultExpr = createConstantExpression(C, false);
        else if (alwaysAvailable(V))
          ResultExpr = createVariableExpression(V, false);
      }
    }
  }
  return uniquifyExpression(ResultExpr);
}
