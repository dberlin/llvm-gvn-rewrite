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

#define DEBUG_TYPE "gvn"
#include "llvm/Transforms/Scalar.h"
#include "llvm/GlobalVariable.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/LLVMContext.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Analysis/PHITransAddr.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Support/PatternMatch.h"

#include <algorithm>
#include <list>
#include <set>

using namespace llvm;
using namespace PatternMatch;

STATISTIC(NumGVNInstr,  "Number of instructions deleted");
STATISTIC(NumGVNLoad,   "Number of loads deleted");
STATISTIC(NumGVNPRE,    "Number of instructions PRE'd");
STATISTIC(NumGVNBlocks, "Number of blocks merged");
STATISTIC(NumGVNSimpl,  "Number of instructions simplified");
STATISTIC(NumGVNEqProp, "Number of equalities propagated");
STATISTIC(NumPRELoad,   "Number of loads PRE'd");
STATISTIC(NumGVNPhisEqual, "Number of equivalent PHI");
STATISTIC(NumGVNPhisAllSame, "Number of PHIs whos arguments are all the same");
STATISTIC(NumGVNBinOpsSimplified, "Number of binary operations simplified");
STATISTIC(NumGVNCmpInsSimplified, "Number of comparison operations simplified");

static cl::opt<bool> EnablePRE("enable-pre",
                               cl::init(true), cl::Hidden);
static cl::opt<bool> EnableLoadPRE("enable-load-pre", cl::init(true));


// Maximum allowed recursion depth.
static cl::opt<uint32_t>
MaxRecurseDepth("max-recurse-depth", cl::Hidden, cl::init(1000), cl::ZeroOrMore,
                cl::desc("Max recurse depth (default = 1000)"));

//===----------------------------------------------------------------------===//
//                         ValueTable Class
//===----------------------------------------------------------------------===//

/// This class holds the mapping between values and value numbers.  It is used
/// as an efficient mechanism to determine the expression-wise equivalence of
/// two values.
namespace {
  static std::string getBlockName(BasicBlock *B) {
    return DOTGraphTraits<const Function*>::getSimpleNodeLabel(B, NULL);
  }
  MemoryDependenceAnalysis *MD;

  enum ExpressionType {
    ExpressionTypeBase,
    ExpressionTypeConstant,
    ExpressionTypeVariable,
    ExpressionTypeBasicStart,
    ExpressionTypeBasic,
    ExpressionTypeCall,
    ExpressionTypeInsertValue,
    ExpressionTypePhi,
    ExpressionTypeLoad,
    ExpressionTypeStore,
    ExpressionTypeBasicEnd
  };
  
  class Expression {
  private:
    void operator=(const Expression&); // Do not implement
    Expression(const Expression&); // Do not implement
  protected:
    ExpressionType etype_;
    uint32_t opcode_;
      
  public:

    uint32_t getOpcode() const {
      return opcode_;
    }

    uint32_t getOpcode() {
      return opcode_;
    }

    void setOpcode(uint32_t opcode) {
      opcode_ = opcode;
    }
    
    ExpressionType getExpressionType() const {
      return etype_;
    }
    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const Expression *) { return true; }
    
   
    Expression(uint32_t o = ~2U) : etype_(ExpressionTypeBase), opcode_(o) {}
    Expression(ExpressionType etype, uint32_t o = ~2U): etype_(etype), opcode_(o) { }
    
    virtual ~Expression() {};

    bool operator==(const Expression &other) const {
      if (opcode_ != other.opcode_)
        return false;
      if (opcode_ == ~0U || opcode_ == ~1U)
        return true;
      if (etype_ != other.etype_)
	return false;
      return equals(other);
    }

    virtual bool equals(const Expression &other) const {
      return true;
    }
    
    virtual hash_code getHashValue() const {
      return hash_combine(etype_, opcode_);
    }      
    virtual void print(raw_ostream &OS) {
      OS << "{etype = " << etype_ << ", opcode = " << opcode_ << " }";
    }
  };
  inline raw_ostream &operator <<(raw_ostream &OS, Expression &E) {
    E.print(OS);
    return OS;
  }

  class BasicExpression: public Expression {
  private:
    void operator=(const BasicExpression&); // Do not implement
    BasicExpression(const BasicExpression&); // Do not implement
  protected:
    Type *type_;
  public:
    
    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const BasicExpression *) { return true; }
    static inline bool classof(const Expression *EB) {
      ExpressionType et = EB->getExpressionType();
      return et > ExpressionTypeBasicStart && et < ExpressionTypeBasicEnd;
    }
    
    void setType(Type *T) {
      type_ = T;
    }
    Type *getType() {
      return type_;
    }
    const Type *getType() const {
      return type_;
    }
    
    SmallVector<Value*, 4> varargs;

    BasicExpression():type_(NULL)  {
      etype_ = ExpressionTypeBasic;
    };
    
    virtual ~BasicExpression() {};

    virtual bool equals(const Expression &other) const {
      const BasicExpression &OE = cast<BasicExpression>(other);
      if (type_ != OE.type_)
        return false;
      if (varargs != OE.varargs)
	return false;

      return true;      
    }
    virtual void print(raw_ostream &OS) {
      OS << "{etype = " << etype_ << ", opcode = " << opcode_ << ", varargs = {";
      for (unsigned i = 0, e = varargs.size(); i != e; ++i) {
	OS << "[" << i << "] = " << varargs[i] << "  ";
      }
      OS << "}  }";
    }

    virtual hash_code getHashValue() const {
      return hash_combine(etype_, opcode_, type_,
                          hash_combine_range(varargs.begin(),
                                             varargs.end()));
    }
  };
  class CallExpression: public BasicExpression {
  private:
    void operator=(const CallExpression&); // Do not implement
    CallExpression(const CallExpression&); // Do not implement
  protected:
    bool nomem_;
    bool readonly_;
    CallInst *callinst_;
  public:
    
  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const CallExpression *) { return true; }
    static inline bool classof(const Expression *EB) {
      return EB->getExpressionType() == ExpressionTypeCall;
    }    
    CallExpression(CallInst *CI, bool nomem = false, bool readonly = false) {
      etype_ = ExpressionTypeCall;
      callinst_ = CI;
      nomem_ = nomem;
      readonly_ = readonly;
    };
    
    virtual ~CallExpression() {};

    virtual bool equals(const Expression &other) const {
      const CallExpression &OE = cast<CallExpression>(other);
      if (type_ != OE.type_)
        return false;
      if (varargs != OE.varargs)
	return false;
      if (callinst_ != OE.callinst_) {
	MemDepResult local_dep = MD->getDependency(callinst_);
	if (!local_dep.isDef() && !local_dep.isNonLocal()) {
	  return false;
	}
	if (local_dep.isDef()) {
	  CallInst* local_cdep = cast<CallInst>(local_dep.getInst());
	  if (local_cdep != OE.callinst_) 
	    return false;
	}
	// Non-local case.
	const MemoryDependenceAnalysis::NonLocalDepInfo &deps =
	  MD->getNonLocalCallDependency(CallSite(callinst_));
	// FIXME: Move the checking logic to MemDep!
	CallInst* cdep = 0;
	
	// Check to see if we have a single dominating call instruction that is
	// identical to C.
	for (unsigned i = 0, e = deps.size(); i != e; ++i) {
	  const NonLocalDepEntry *I = &deps[i];
	  if (I->getResult().isNonLocal())
	    continue;
	  
	  // We don't handle non-definitions.  If we already have a call, reject
	  // instruction dependencies.
	  if (!I->getResult().isDef() || cdep != 0) {
	    cdep = 0;
	    break;
	  }
	  
	  CallInst *NonLocalDepCall = dyn_cast<CallInst>(I->getResult().getInst());
	  if (NonLocalDepCall && NonLocalDepCall == OE.callinst_){
	    cdep = NonLocalDepCall;
	    continue;
	  }
	  
	  cdep = 0;
	  break;
	}
	if (!cdep)
	  return false;
      }
      
      return true;      
    }
    
    virtual hash_code getHashValue() const {
      return hash_combine(etype_, opcode_, type_, nomem_, readonly_,
                          hash_combine_range(varargs.begin(),
                                             varargs.end()));
    }
  };
  class LoadExpression: public BasicExpression {
  private:
    void operator=(const LoadExpression&); // Do not implement
    LoadExpression(const LoadExpression&); // Do not implement
  public:
    
    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const LoadExpression *) { return true; }
    static inline bool classof(const Expression *EB) {
      return EB->getExpressionType() == ExpressionTypeLoad;
    }    
    LoadExpression() {
      etype_ = ExpressionTypeLoad;
    };
    
    virtual ~LoadExpression() {};

    virtual bool equals(const Expression &other) const {
      const LoadExpression &OE = cast<LoadExpression>(other);
      if (type_ != OE.type_)
        return false;
      if (varargs != OE.varargs)
	return false;

      return true;      
    }
    
    virtual hash_code getHashValue() const {
      return hash_combine(etype_, opcode_, type_,
                          hash_combine_range(varargs.begin(),
                                             varargs.end()));
    }
  };
  class StoreExpression: public BasicExpression {
  private:
    void operator=(const StoreExpression&); // Do not implement
    StoreExpression(const StoreExpression&); // Do not implement
  public:
    
    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const StoreExpression *) { return true; }
    static inline bool classof(const Expression *EB) {
      return EB->getExpressionType() == ExpressionTypeStore;
    }    
    StoreExpression() {
      etype_ = ExpressionTypeStore;
    };
    
    virtual ~StoreExpression() {};

    virtual bool equals(const Expression &other) const {
      const StoreExpression &OE = cast<StoreExpression>(other);
      if (type_ != OE.type_)
        return false;
      if (varargs != OE.varargs)
	return false;

      return true;      
    }
    
    virtual hash_code getHashValue() const {
      return hash_combine(etype_, opcode_, type_,
                          hash_combine_range(varargs.begin(),
                                             varargs.end()));
    }
  };

  class InsertValueExpression: public BasicExpression {
  private:
    void operator=(const InsertValueExpression&); // Do not implement
    InsertValueExpression(const InsertValueExpression&); // Do not implement
  public:
    
    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const InsertValueExpression *) { return true; }
    static inline bool classof(const Expression *EB) {
      return EB->getExpressionType() == ExpressionTypeInsertValue;
    }
    
    SmallVector<uint32_t, 4> intargs;

    InsertValueExpression() {
      etype_ = ExpressionTypeInsertValue;
    };
    
    virtual ~InsertValueExpression() {};

    virtual bool equals(const Expression &other) const {
      const InsertValueExpression &OE = cast<InsertValueExpression>(other);
      if (type_ != OE.type_)
        return false;
      if (varargs != OE.varargs)
	return false;
      if (intargs != OE.intargs)
	return false;

      return true;      
    }
    
    virtual hash_code getHashValue() const {
      return hash_combine(etype_, opcode_, type_,
                          hash_combine_range(varargs.begin(),
                                             varargs.end()),
                          hash_combine_range(intargs.begin(), intargs.end()));
    }
    virtual void print(raw_ostream &OS) {
      OS << "{etype = " << etype_ << ", opcode = " << opcode_ << ", varargs = {";
      for (unsigned i = 0, e = varargs.size(); i != e; ++i) {
	OS << "[" << i << "] = " << varargs[i] << "  ";
      }
      OS << "}, intargs = {";
      for (unsigned i = 0, e = intargs.size(); i != e; ++i) {
	OS << "[" << i << "] = " << intargs[i] << "  ";
      }
      OS << "}  }";
    }
  };
    
  class PHIExpression : public BasicExpression {
  public:

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const PHIExpression *) { return true; }
    static inline bool classof(const Expression *EB) {
      return EB->getExpressionType() == ExpressionTypePhi;
    }
    const BasicBlock *getBB() const {
      return bb_;
    }
    BasicBlock *getBB() {
      return bb_;
    }
    void setBB(BasicBlock *bb) {
      bb_ = bb;
    }
    
    virtual bool equals(const Expression &other) const {
      const PHIExpression &OE = cast<PHIExpression>(other);
      if (bb_ != OE.bb_) 
	return false;
      if (type_ != OE.type_)
        return false;
      if (varargs != OE.varargs)
	return false;
      return true;
    }
    
    PHIExpression():bb_(NULL) {
      etype_ = ExpressionTypePhi;
    }
    
      
    PHIExpression(BasicBlock *bb):bb_(bb) { 
      etype_ = ExpressionTypePhi;
    }

    virtual hash_code getHashValue() const {
      return hash_combine(etype_, bb_, opcode_, type_,
                          hash_combine_range(varargs.begin(),
                                             varargs.end()));
    }
    virtual void print(raw_ostream &OS) {
      OS << "{etype = " << etype_ << ", opcode = " << opcode_ << ", varargs = {";
      for (unsigned i = 0, e = varargs.size(); i != e; ++i) {
	OS << "[" << i << "] = " << varargs[i] << "  ";
      }
      OS << "}, bb = " << bb_ << "  }";
    }

    private:
    void operator=(const PHIExpression&); // Do not implement
    PHIExpression(const PHIExpression&); // Do not implement
    BasicBlock *bb_;
    
  };
  class VariableExpression : public Expression {
    public:

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const VariableExpression *) { return true; }
    static inline bool classof(const Expression *EB) {
      return EB->getExpressionType() == ExpressionTypeVariable;
    }
    const Value *getVariableValue() const {
      return variableValue_;
    }
    Value *getVariableValue() {
      return variableValue_;
    }
    void setVariableValue(Constant *V) {
      variableValue_ = V;
    }  
    virtual bool equals(const Expression &other) const {
      const VariableExpression &OC = cast<VariableExpression>(other);
      if (variableValue_ != OC.variableValue_)
	return false;
      return true;
    }
    
    VariableExpression():Expression(ExpressionTypeVariable), variableValue_(NULL) { }
    
      
    VariableExpression(Value *variableValue):Expression(ExpressionTypeVariable), variableValue_(variableValue) { }
    virtual hash_code getHashValue() const {
      return hash_combine(etype_, variableValue_->getType(), variableValue_);
    }

    private:
    void operator=(const VariableExpression&); // Do not implement
    VariableExpression(const VariableExpression&); // Do not implement

    Value* variableValue_; 
    
  };

  class ConstantExpression : public Expression {
    public:

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const ConstantExpression *) { return true; }
    static inline bool classof(const Expression *EB) {
      return EB->getExpressionType() == ExpressionTypeConstant;
    }
    const Constant *getConstantValue() const {
      return constantValue_;
    }
    Constant *getConstantValue() {
      return constantValue_;
    }
    void setConstantValue(Constant *V) {
      constantValue_ = V;
    }  
    virtual bool equals(const Expression &other) const {
      const ConstantExpression &OC = cast<ConstantExpression>(other);
      if (constantValue_ != OC.constantValue_)
	return false;
      return true;
    }
    
    ConstantExpression():Expression(ExpressionTypeConstant), constantValue_(NULL) { }
    
      
    ConstantExpression(Constant *constantValue):Expression(ExpressionTypeConstant), constantValue_(constantValue) { }
    virtual hash_code getHashValue() const {
      return hash_combine(etype_, constantValue_->getType(), constantValue_);
    }

    private:
    void operator=(const ConstantExpression&); // Do not implement
    ConstantExpression(const ConstantExpression&); // Do not implement

    Constant* constantValue_; 
    
  };

}

//===----------------------------------------------------------------------===//
//                                GVN Pass
//===----------------------------------------------------------------------===//

namespace {
  class GVN : public FunctionPass {


    static uint32_t nextCongruenceNum;    
    struct CongruenceClass {
      uint32_t id;
      //TODO(dannyb) Replace this list
      Value* leader;
      Expression* constantLeader;
      Expression* expression;
      std::set<Value*> members;
      CongruenceClass():id(nextCongruenceNum++), leader(0), expression(0) {};
      
    };
    CongruenceClass *InitialClass;
    bool NoLoads;
    AliasAnalysis *AA;
    PostDominatorTree *PDT;
    DominatorTree *DT;
    const TargetData *TD;
    const TargetLibraryInfo *TLI;
    DenseMap<BasicBlock*, uint32_t> rpoBlockNumbers;
    DenseMap<Instruction*, uint32_t> rpoInstructionNumbers;
    DenseMap<BasicBlock*, std::pair<uint32_t, uint32_t> > rpoInstructionStartEnd;
    std::vector<BasicBlock*> rpoToBlock;
    DenseSet<BasicBlock*> reachableBlocks;
    DenseSet<std::pair<BasicBlock*, BasicBlock*> > reachableEdges;
    DenseSet<Instruction*> touchedInstructions;
    // Tested: SparseBitVector, DenseSet, etc. vector<bool> is 3 times as fast
    std::vector<bool> touchedBlocks;
    std::vector<CongruenceClass*> congruenceClass;
    DenseMap<Value*, CongruenceClass*> valueToClass;
    DenseMap<Expression*,  bool> uniquedExpressions;
    
    struct ComparingExpressionInfo {
      static inline Expression* getEmptyKey() {
        intptr_t Val = -1;
        Val <<= PointerLikeTypeTraits<Expression*>::NumLowBitsAvailable;
        return reinterpret_cast<Expression*>(Val);
      }
      static inline Expression* getTombstoneKey() {
        intptr_t Val = -2;
        Val <<= PointerLikeTypeTraits<Expression*>::NumLowBitsAvailable;
        return reinterpret_cast<Expression*>(Val);
      }
      static unsigned getHashValue(const Expression *V) {
        return static_cast<unsigned>(V->getHashValue());
	
        
      }
      static bool isEqual(const Expression *LHS, const Expression *RHS) {
        if (LHS == RHS) 
          return true; 
        if (LHS == getTombstoneKey() || RHS == getTombstoneKey()
            || LHS == getEmptyKey() || RHS == getEmptyKey())
          return false;
        return (*LHS == *RHS);  
      }  
    };
    
    typedef DenseMap<Expression*, CongruenceClass*, ComparingExpressionInfo> ExpressionClassMap;
    ExpressionClassMap expressionToClass;
    DenseSet<Value*> changedValues;

    /// LeaderTable - A mapping from value numbers to lists of Value*'s that
    /// have that value number.  Use findLeader to query it.
    struct LeaderTableEntry {
      Value *Val;
      BasicBlock *BB;
      LeaderTableEntry *Next;
    };
    DenseMap<uint32_t, LeaderTableEntry> LeaderTable;
    BumpPtrAllocator TableAllocator;
    
    SmallVector<Instruction*, 8> InstrsToErase;
    Expression *createExpression(Instruction*);
    void setBasicExpressionInfo(Instruction*, BasicExpression*);
    Expression *createPHIExpression(Instruction *);
    Expression *createVariableExpression(Value *);
    Expression *createConstantExpression(Constant *);
    Expression *createCallExpression(CallInst*, bool, bool);
    Expression *createInsertValueExpression(InsertValueInst *);
    Expression *uniquifyExpression(Expression *E);

  public:
    static char ID; // Pass identification, replacement for typeid
    explicit GVN(bool noloads = false)
        : FunctionPass(ID), NoLoads(noloads) {
      initializeGVNPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F);
    
    /// markInstructionForDeletion - This removes the specified instruction from
    /// our various maps and marks it for deletion.
    void markInstructionForDeletion(Instruction *I) {
      InstrsToErase.push_back(I);
    }
    
    const TargetData *getTargetData() const { return TD; }
    DominatorTree &getDominatorTree() const { return *DT; }
    AliasAnalysis *getAliasAnalysis() const { return AA; }
    MemoryDependenceAnalysis &getMemDep() const { return *MD; }
  private:
    Expression *performSymbolicEvaluation(Value*, BasicBlock*);
    Expression *performSymbolicCallEvaluation(Instruction*, BasicBlock*);
    Expression *performSymbolicPHIEvaluation(Instruction*, BasicBlock*);

    void performCongruenceFinding(Value*, Expression*);
    void updateReachableEdge(BasicBlock*, BasicBlock*);
    void processOutgoingEdges(TerminatorInst* TI);
    void propagateChangeInEdge(BasicBlock*);
    /// addToLeaderTable - Push a new Value to the LeaderTable onto the list for
    /// its value number.
    void addToLeaderTable(uint32_t N, Value *V, BasicBlock *BB) {
      LeaderTableEntry &Curr = LeaderTable[N];
      if (!Curr.Val) {
        Curr.Val = V;
        Curr.BB = BB;
        return;
      }
      
      LeaderTableEntry *Node = TableAllocator.Allocate<LeaderTableEntry>();
      Node->Val = V;
      Node->BB = BB;
      Node->Next = Curr.Next;
      Curr.Next = Node;
    }
    
    /// removeFromLeaderTable - Scan the list of values corresponding to a given
    /// value number, and remove the given value if encountered.
    void removeFromLeaderTable(uint32_t N, Value *V, BasicBlock *BB) {
      LeaderTableEntry* Prev = 0;
      LeaderTableEntry* Curr = &LeaderTable[N];

      while (Curr->Val != V || Curr->BB != BB) {
        Prev = Curr;
        Curr = Curr->Next;
      }
      
      if (Prev) {
        Prev->Next = Curr->Next;
      } else {
        if (!Curr->Next) {
          Curr->Val = 0;
          Curr->BB = 0;
        } else {
          LeaderTableEntry* Next = Curr->Next;
          Curr->Val = Next->Val;
          Curr->BB = Next->BB;
          Curr->Next = Next->Next;
        }
      }
    }

    // List of critical edges to be split between iterations.
    SmallVector<std::pair<TerminatorInst*, unsigned>, 4> toSplit;

    // This transformation requires dominator postdominator info
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<DominatorTree>();
      AU.addRequired<PostDominatorTree>();
      AU.addRequired<TargetLibraryInfo>();
      if (!NoLoads)
        AU.addRequired<MemoryDependenceAnalysis>();
      AU.addRequired<AliasAnalysis>();
      
      AU.addPreserved<PostDominatorTree>();
      AU.addPreserved<DominatorTree>();
      AU.addPreserved<AliasAnalysis>();
    }
    

    // Helper fuctions
    // FIXME: eliminate or document these better
    bool processLoad(LoadInst *L);
    bool processInstruction(Instruction *I);
    bool processNonLocalLoad(LoadInst *L);
    bool processBlock(BasicBlock *BB);
    void dump(DenseMap<uint32_t, Value*> &d);
    bool iterateOnFunction(Function &F);
    bool performPRE(Function &F);
    Value *findLeader(BasicBlock *BB, uint32_t num);
    void cleanupGlobalSets();
    void verifyRemoved(const Instruction *I) const;
    bool splitCriticalEdges();
    unsigned replaceAllDominatedUsesWith(Value *From, Value *To,
                                         BasicBlock *Root);
    bool propagateEquality(Value *LHS, Value *RHS, BasicBlock *Root);
  };

  char GVN::ID = 0;
}

// createGVNPass - The public interface to this file...
FunctionPass *llvm::createGVNPass(bool NoLoads) {
  return new GVN(NoLoads);
}

INITIALIZE_PASS_BEGIN(GVN, "gvn", "Global Value Numbering", false, false)
INITIALIZE_PASS_DEPENDENCY(MemoryDependenceAnalysis)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfo)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(GVN, "gvn", "Global Value Numbering", false, false)

void GVN::dump(DenseMap<uint32_t, Value*>& d) {
  errs() << "{\n";
  for (DenseMap<uint32_t, Value*>::iterator I = d.begin(),
       E = d.end(); I != E; ++I) {
      errs() << I->first << "\n";
      I->second->dump();
  }
  errs() << "}\n";
}
Expression *GVN::createPHIExpression(Instruction *I) {
  PHINode *PN = cast<PHINode>(I);
  PHIExpression *E = new PHIExpression(I->getParent());
  E->setType(I->getType());
  E->setOpcode(I->getOpcode());

  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
    BasicBlock *B = PN->getIncomingBlock(i);
    if (!reachableBlocks.count(B)) {
      DEBUG(dbgs() << "Skipping unreachable block " << getBlockName(B) << " in PHI node " << *PN << "\n");
      continue;
    }
    Value *Operand = I->getOperand(i);
    DenseMap<Value*, CongruenceClass*>::iterator VTCI = valueToClass.find(Operand);
    if (VTCI != valueToClass.end()) {
      CongruenceClass *CC = VTCI->second;
      if (CC != InitialClass)
        Operand = CC->leader;
    }
    E->varargs.push_back(Operand);
  }
  return E;
}

void GVN::setBasicExpressionInfo(Instruction *I, BasicExpression *E) {
  E->setType(I->getType());
  E->setOpcode(I->getOpcode());
  
  for (Instruction::op_iterator OI = I->op_begin(), OE = I->op_end();
       OI != OE; ++OI) {
    Value *Operand = *OI;
    DenseMap<Value*, CongruenceClass*>::iterator VTCI = valueToClass.find(Operand);
    if (VTCI != valueToClass.end()) {
      CongruenceClass *CC = VTCI->second;
      if (CC != InitialClass)
        Operand = CC->leader;
    }
    E->varargs.push_back(Operand);
  }
}

Expression *GVN::createExpression(Instruction *I) {
  BasicExpression *E = new BasicExpression();
  setBasicExpressionInfo(I, E);
    
  if (I->isCommutative()) {
    // Ensure that commutative instructions that only differ by a permutation
    // of their operands get the same value number by sorting the operand value
    // numbers.  Since all commutative instructions have two operands it is more
    // efficient to sort by hand rather than using, say, std::sort.
    assert(I->getNumOperands() == 2 && "Unsupported commutative instruction!");
    if (E->varargs[0] > E->varargs[1])
      std::swap(E->varargs[0], E->varargs[1]);
  }
  
  if (CmpInst *CI = dyn_cast<CmpInst>(I)) {
    // Sort the operand value numbers so x<y and y>x get the same value number.
    CmpInst::Predicate Predicate = CI->getPredicate();
    if (E->varargs[0] > E->varargs[1]) {
      std::swap(E->varargs[0], E->varargs[1]);
      Predicate = CmpInst::getSwappedPredicate(Predicate);
    }
    E->setOpcode((CI->getOpcode() << 8) | Predicate);
    Value *V = SimplifyCmpInst(Predicate, E->varargs[0], E->varargs[1], TD, TLI, DT);
    Constant *C;
    if (V && (C = dyn_cast<Constant>(V))) {
      DEBUG(dbgs() << "Simplified " << *I << " to " << " constant " << *C << "\n");
      NumGVNCmpInsSimplified++;
      delete E;
      return createConstantExpression(C);
    }
  }

  // Handle simplifying
  if (I->isBinaryOp()) {
    Value *V = SimplifyBinOp(E->getOpcode(), E->varargs[0], E->varargs[1], TD, TLI, DT);
    Constant *C;
    if (V && (C = dyn_cast<Constant>(V))) {
      DEBUG(dbgs() << "Simplified " << *I << " to " << " constant " << *C << "\n");
      NumGVNBinOpsSimplified++;
      delete E;
      return createConstantExpression(C);
    }
  } else if (isa<GetElementPtrInst>(I)) {
    Value *V = SimplifyGEPInst(E->varargs, TD, TLI, DT);
    Constant *C;
    if (V && (C = dyn_cast<Constant>(V))) {
      DEBUG(dbgs() << "Simplified " << *I << " to " << " constant " << *C << "\n");
      NumGVNBinOpsSimplified++;
      delete E;
      return createConstantExpression(C);
    }
  }
  
  return E;
}

Expression *GVN::uniquifyExpression(Expression *E) {  
  std::pair<DenseMap<Expression*, bool>::iterator, bool> P = uniquedExpressions.insert(std::make_pair(E, true));
  if (!P.second && P.first->first != E) {
    delete E;
    return P.first->first;
  } 
  return E;
}

Expression *GVN::createInsertValueExpression(InsertValueInst *I) {
  InsertValueExpression *E = new InsertValueExpression();
  E->setType(I->getType());
  E->setOpcode(I->getOpcode());

  for (Instruction::op_iterator OI = I->op_begin(), OE = I->op_end();
       OI != OE; ++OI) {
    Value *Operand = *OI;
    DenseMap<Value*, CongruenceClass*>::iterator VTCI = valueToClass.find(Operand);
    if (VTCI != valueToClass.end()) {
      CongruenceClass *CC = VTCI->second;
      if (CC != InitialClass)
        Operand = CC->leader;
    }
    E->varargs.push_back(Operand);
  }
  for (InsertValueInst::idx_iterator II = I->idx_begin(), IE = I->idx_end();
       II != IE; ++II)
    E->intargs.push_back(*II);
  return E;
}

Expression *GVN::createVariableExpression(Value *V) {
  VariableExpression *E = new VariableExpression(V);
  // E->setType(C->getType());
  E->setOpcode(V->getValueID());
  return E;
}

Expression *GVN::createConstantExpression(Constant *C) {
  ConstantExpression *E = new ConstantExpression(C);
  // E->setType(C->getType());
  E->setOpcode(C->getValueID());
  return E;
}

Expression *GVN::createCallExpression(CallInst *CI, bool nomem, bool readonly) {
  CallExpression *E = new CallExpression(CI, nomem, readonly);
  setBasicExpressionInfo(CI, E);
  return E;
}

/// performSymbolicCallEvaluation - Evaluate read only and pure calls, and create an expression result
Expression *GVN::performSymbolicCallEvaluation(Instruction *I, BasicBlock *B) {
  CallInst *CI = cast<CallInst>(I);
  
  if (AA->doesNotAccessMemory(CI)) {
    return createCallExpression(CI, true, false);
  } else if (AA->onlyReadsMemory(CI)) {
    return createCallExpression(CI, false, true);
  } else {
    return NULL;
  }
  //   uint32_t& e = expressionNumbering[exp];
  //   if (!e) e = nextValueNumber++;
  //   valueNumbering[C] = e;
  //   return e;
  // } else if (AA->onlyReadsMemory(C)) {
  //   Expression exp = create_expression(C);
  //   uint32_t& e = expressionNumbering[exp];
  //   if (!e) {
  //     e = nextValueNumber++;
  //     valueNumbering[C] = e;
  //     return e;
  //   }
  //   if (!MD) {
  //     e = nextValueNumber++;
  //     valueNumbering[C] = e;
  //     return e;
  //   }

  //   MemDepResult local_dep = MD->getDependency(C);

  //   if (!local_dep.isDef() && !local_dep.isNonLocal()) {
  //     valueNumbering[C] =  nextValueNumber;
  //     return nextValueNumber++;
  //   }

  //   if (local_dep.isDef()) {
  //     CallInst* local_cdep = cast<CallInst>(local_dep.getInst());

  //     if (local_cdep->getNumArgOperands() != C->getNumArgOperands()) {
  //       valueNumbering[C] = nextValueNumber;
  //       return nextValueNumber++;
  //     }

  //     for (unsigned i = 0, e = C->getNumArgOperands(); i < e; ++i) {
  //       uint32_t c_vn = lookup_or_add(C->getArgOperand(i));
  //       uint32_t cd_vn = lookup_or_add(local_cdep->getArgOperand(i));
  //       if (c_vn != cd_vn) {
  //         valueNumbering[C] = nextValueNumber;
  //         return nextValueNumber++;
  //       }
  //     }

  //     uint32_t v = lookup_or_add(local_cdep);
  //     valueNumbering[C] = v;
  //     return v;
  //   }

  //   // Non-local case.
  //   const MemoryDependenceAnalysis::NonLocalDepInfo &deps =
  //     MD->getNonLocalCallDependency(CallSite(C));
  //   // FIXME: Move the checking logic to MemDep!
  //   CallInst* cdep = 0;

  //   // Check to see if we have a single dominating call instruction that is
  //   // identical to C.
  //   for (unsigned i = 0, e = deps.size(); i != e; ++i) {
  //     const NonLocalDepEntry *I = &deps[i];
  //     if (I->getResult().isNonLocal())
  //       continue;

  //     // We don't handle non-definitions.  If we already have a call, reject
  //     // instruction dependencies.
  //     if (!I->getResult().isDef() || cdep != 0) {
  //       cdep = 0;
  //       break;
  //     }

  //     CallInst *NonLocalDepCall = dyn_cast<CallInst>(I->getResult().getInst());
  //     // FIXME: All duplicated with non-local case.
  //     if (NonLocalDepCall && DT->properlyDominates(I->getBB(), C->getParent())){
  //       cdep = NonLocalDepCall;
  //       continue;
  //     }

  //     cdep = 0;
  //     break;
  //   }

  //   if (!cdep) {
  //     valueNumbering[C] = nextValueNumber;
  //     return nextValueNumber++;
  //   }

  //   if (cdep->getNumArgOperands() != C->getNumArgOperands()) {
  //     valueNumbering[C] = nextValueNumber;
  //     return nextValueNumber++;
  //   }
  //   for (unsigned i = 0, e = C->getNumArgOperands(); i < e; ++i) {
  //     uint32_t c_vn = lookup_or_add(C->getArgOperand(i));
  //     uint32_t cd_vn = lookup_or_add(cdep->getArgOperand(i));
  //     if (c_vn != cd_vn) {
  //       valueNumbering[C] = nextValueNumber;
  //       return nextValueNumber++;
  //     }
  //   }

  //   uint32_t v = lookup_or_add(cdep);
  //   valueNumbering[C] = v;
  //   return v;

  // } else {
  //   valueNumbering[C] = nextValueNumber;
  //   return nextValueNumber++;
  // }
}

// performSymbolicPHIEvaluation - Evaluate PHI nodes symbolically, and create an expression result
Expression *GVN::performSymbolicPHIEvaluation(Instruction *I, BasicBlock *B) { 
  PHIExpression *E = cast<PHIExpression>(createPHIExpression(I));
  E->setOpcode(I->getOpcode());
  if (E->varargs.empty()) {
    DEBUG(dbgs() << "Simplified PHI node " << I << " to undef" << "\n");
    delete E;
    return createVariableExpression(UndefValue::get(I->getType()));
  }
  
  Value *AllSameValue = E->varargs[0];
  
  for (unsigned i = 1, e = E->varargs.size(); i != e; ++i) {
    if (E->varargs[i] != AllSameValue) {
      AllSameValue = NULL;
      break;
    }
  }
  // 
  if (AllSameValue) {
    // It's possible to have mutually recursive phi nodes, especially in weird CFG's.
    // This can cause infinite loops (even if you disable the recursion below, you will ping-pong between congruence classes)
    // If a phi node evaluates to another phi node, just leave it alone
    if (isa<PHINode>(AllSameValue))
      return E;
    NumGVNPhisAllSame++;
    DEBUG(dbgs() << "Simplified PHI node " << I << " to " << *AllSameValue << "\n");
  
    delete E;
    return performSymbolicEvaluation(AllSameValue, B);
  }
  return E;
}


/// performSymbolicEvaluation - Substitute and symbolize the value before value numbering
Expression *GVN::performSymbolicEvaluation(Value *V, BasicBlock *B) {
  Expression *E = NULL;
  if (Constant *C = dyn_cast<Constant>(V))
    E = createConstantExpression(C);
  else if (isa<Argument>(V) || isa<GlobalVariable>(V)) {
    E = createVariableExpression(V);
  } else {
    Instruction *I = cast<Instruction>(V);
    switch (I->getOpcode()) {
      //TODO: extractvalue
    case Instruction::InsertValue:
      E = createInsertValueExpression(cast<InsertValueInst>(I));
      break;
    case Instruction::PHI: 
      E = performSymbolicPHIEvaluation(I, B);
      break;
    case Instruction::Call:
      E = performSymbolicCallEvaluation(I, B);
      break;
    // case Instruction::Store:
    //   E = performSymbolicStoreEvaluation(cast<StoreInst>(I, B));
    //   break;
    // case Instruction::Load:
    //   E = performSymbolicLoadEvaluation(cast<LoadInst>(I, B));
    //   break;
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
    case Instruction::Or :
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
    case Instruction::BitCast:
    case Instruction::Select:
    case Instruction::ExtractElement:
    case Instruction::InsertElement:
    case Instruction::ShuffleVector:
    case Instruction::GetElementPtr:
      E = createExpression(I);
      break;
    default:
      return NULL;
    }
  }
  
  return uniquifyExpression(E);
}

//   SmallVector<Value*, 4> Operands;
//   if (Instruction *I = dyn_cast<Instruction>(V)) {
//     for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
//       Value *Operand = I->getOperand(i);
//       ValueClassMap::iterator VTCI = valueToClass.find(Operand);
//       if (VTCI != valueToClass.end()) {
//         ValueToClass *CC = VTCI->second;
//         if (CC != InitialClass)
//           Operand = CC->leader;
//       }
//       Operands.push_back(Operand);
//     }
//     std::sort(Operands.begin(), Operands.end());
//     if (I->isBinaryOp()) {
//       Value *SR = SimplifyBinOp(I->getOpcode(), Operands[0], Operands[1],
//                                 TD, TLI, DT);
//       if (SR)
//         DEBUG(dbgs() << "Simplified " << V << " into " << SR << "\n");
//       V = (SR ? SR : V);
//     }
//   }
  
	
//   return V;
// }

/// performCongruenceFinding - Perform congruence finding on a given value numbering expression
void GVN::performCongruenceFinding(Value *V, Expression *E) {
  // This is guaranteed to return something, since it will at least find INITIAL
  CongruenceClass *VClass = valueToClass[V];
  //TODO(dannyb): Double check algorithm where we are ignoring copy check of "if e is a variable"
  CongruenceClass *EClass;

  // Expressions we can't symbolize are always in their own congruence class
  if (E == NULL) {
    if (VClass->members.size() != 1) {
      CongruenceClass *NewClass = new CongruenceClass();
      congruenceClass.push_back(NewClass);
      // We should always be adding it below
      // NewClass->members.push_back(V);
      NewClass->expression = NULL;
      NewClass->leader = V;
      EClass = NewClass;
      DEBUG(dbgs() << "Created new congruence class for " << V << " due to NULL expression\n");

    }
  } else {
    ExpressionClassMap::iterator VTCI = expressionToClass.find(E);
    
    // If it's not in the value table, create a new congruence class
    if (VTCI == expressionToClass.end()) {
      CongruenceClass *NewClass = new CongruenceClass();
      congruenceClass.push_back(NewClass);
      // We should always be adding it below
      // NewClass->members.push_back(V);
      NewClass->expression = E;
      expressionToClass[E] = NewClass;
      if (ConstantExpression *CE = dyn_cast<ConstantExpression>(E))
        NewClass->leader = CE->getConstantValue();
      else
        NewClass->leader = V;
      
      EClass = NewClass;
      DEBUG(dbgs() << "Created new congruence class for " << V << " using expression " << *E << " at " << NewClass->id << "\n");
    } else {
      EClass = VTCI->second;
    }
  }
  DenseSet<Value*>::iterator DI = changedValues.find(V);
  bool WasInChanged = DI != changedValues.end();
  if (VClass != EClass || WasInChanged) {
    DEBUG(dbgs() << "Found class " << EClass->id << " for expression " << E << "\n");

    if (WasInChanged)
      changedValues.erase(DI);
    if (VClass != EClass) {
      DEBUG(dbgs() << "New congruence class for " << V << " is " << EClass->id << "\n");
      VClass->members.erase(V);
      assert(std::find(EClass->members.begin(), EClass->members.end(), V) == EClass->members.end() && "Tried to add something to members twice!");
      EClass->members.insert(V);
      valueToClass[V] = EClass;
      // See if we destroyed the class or need to swap leaders
      if (VClass->members.empty() && VClass != InitialClass) {
	expressionToClass.erase(VClass->expression);
	// delete VClass;
      } else if (VClass->leader == V) {
	VClass->leader = *(VClass->members.begin());
	for (std::set<Value*>::iterator LI = VClass->members.begin(), LE = VClass->members.end();
	     LI != LE; ++LI) {
	  if (Instruction *I = dyn_cast<Instruction>(*LI))
	    touchedInstructions.insert(I);
	  changedValues.insert(*LI);
	}
      }
    }
    // Now mark the users as touched
    for (Value::use_iterator UI = V->use_begin(), UE = V->use_end();
	 UI != UE; ++UI) {
      Instruction *User = cast<Instruction>(*UI);
      touchedInstructions.insert(User);
    }
  }
}


// updateReachableEdge - Process the fact that Edge (from, to) is
// reachable, including marking any newly reachable blocks and
// instructions for processing
void GVN::updateReachableEdge(BasicBlock *From, BasicBlock *To) {
  // Check if the Edge was reachable before
  if (reachableEdges.insert(std::make_pair(From, To)).second) {
    // If this block wasn't reachable before, all instructions are touched
    if (reachableBlocks.insert(To).second) {
      DEBUG(dbgs() << "Block " << getBlockName(To) << " marked reachable\n");
      uint32_t rpoNum = rpoBlockNumbers[To];
      touchedBlocks[rpoNum] = true;
      for (BasicBlock::iterator BI = To->begin(), BE = To->end();
	   BI != BE; ++BI)
	touchedInstructions.insert(BI);
    }
  } else {
    DEBUG(dbgs() << "Block " << getBlockName(To) << " was reachable, but new edge to it found\n");
    // We've made an edge reachable to an existing block, which may impact predicates.
    // Otherwise, only mark the phi nodes as touched
    BasicBlock::iterator BI = To->begin();
    while (isa<PHINode>(BI)) {
      touchedInstructions.insert(BI);
      ++BI;
    }
    // Propagate the change downstream.
    propagateChangeInEdge(To);
  }
}

  
//  processOutgoingEdges - Process the outgoing edges of a block for reachability.
void GVN::processOutgoingEdges(TerminatorInst *TI) {
  // Evaluate Reachability of terminator instruction
  // Conditional branch
  BranchInst *BR = dyn_cast<BranchInst>(TI);
  if (BR && BR->isConditional()) {
    Value *Cond = BR->getCondition();
    Value *CondEvaluated = NULL;
    if (Instruction *I = dyn_cast<Instruction>(Cond)) {
      Expression *E = createExpression(I);
      if (ConstantExpression *CE = dyn_cast<ConstantExpression>(E)) {
	CondEvaluated = CE->getConstantValue();
      } 
      delete E;
    } else if (isa<ConstantInt>(Cond)) {
      CondEvaluated = Cond;
    }
    ConstantInt *CI;
    BasicBlock *TrueSucc = BR->getSuccessor(0);
    BasicBlock *FalseSucc = BR->getSuccessor(1);
    if (CondEvaluated && (CI = dyn_cast<ConstantInt>(CondEvaluated))) {
      if (CI->isOne()) {
	DEBUG(dbgs() << "Condition for Terminator " << *TI << " evaluated to true\n");
	updateReachableEdge(TI->getParent(), TrueSucc);
      } else if (CI->isZero()) {
	DEBUG(dbgs() << "Condition for Terminator " << *TI << " evaluated to false\n");
	updateReachableEdge(TI->getParent(), FalseSucc);
      }
    } else {
      updateReachableEdge(TI->getParent(), TrueSucc);
      updateReachableEdge(TI->getParent(), FalseSucc);
    }	  
  } else {    
    for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i) {
      BasicBlock *B = TI->getSuccessor(i); 
      updateReachableEdge(TI->getParent(), B);
      //TODO(predication)
    }
  }
  
}

/// propagateChangeInEdge - Propagate a change in edge reachability
// When we discover a new edge to an existing reachable block, that
// can affect the value of blocks containing phi nodes downstream. 
//
// However, it can *only* impact blocks that contain phi nodes, as
// those are the only values that would be carried from multiple
// incoming edges at once. 
// 
void GVN::propagateChangeInEdge(BasicBlock *Dest) {
  if (1) {
  // We have two options here.
  // We can either use the RPO numbers, and process all blocks with a
  // greater RPO number, conservatively.  Or we can use the dominator tree and post
  // dominator tree
  // We have to touch instructions in blocks we dominate
  // We only have to touch blocks we post-dominate
  //
  // The algorithm states that you only need to touch blocks that are confluence nodes.
  // I can't see why you would need to touch any nodes that aren't PHI
  // nodes.  Because we don't use predicates, they are the ones whose value could have changed as a
  // result of a new edge becoming live, and any changes to their
  // value should propagate appropriately through the rest of the block.
  DEBUG(dbgs() << "Would have processed " << rpoBlockNumbers.size() - rpoBlockNumbers[Dest] << " blocks here\n");
  uint32_t blocksProcessed = 0;
  DomTreeNode *DTN = DT->getNode(Dest);
  DomTreeNode *PDTN = PDT->getNode(Dest);
  while (PDTN) {
    BasicBlock *B = PDTN->getBlock();
    touchedBlocks[rpoBlockNumbers[B]] = true;
    blocksProcessed++;
    PDTN = PDTN->getIDom();
  }
  
  DEBUG(dbgs() << "PDT now would have processed " << blocksProcessed << " blocks\n");
  blocksProcessed = 0;
  //TODO(dannyb): This differs slightly from the published algorithm, verify it
  //The published algorithm touches all instructions, we only touch the phi nodes.
  // This is because there should be no other values that can
  //*directly* change as a result of edge reachability. If the phi
  //node ends up producing a new value, it's users will be marked as
  //touched anyway
  for (df_iterator<DomTreeNode*> DI = df_begin(DTN), DE = df_end(DTN); DI != DE; ++DI) {
    BasicBlock *B = DI->getBlock();
    if (!B->getUniquePredecessor()) {
      for (BasicBlock::iterator BI = B->begin(), BE = B->end();
	   BI != BE; ++BI) {
	if (!isa<PHINode>(BI))
	  break;
	touchedInstructions.insert(BI);    
      }    
      blocksProcessed++;
    }
  }
  DEBUG(dbgs() << "DT now would have processed " << blocksProcessed << " blocks\n");
  } else {
    for (unsigned i = rpoBlockNumbers[Dest], e = rpoBlockNumbers.size(); i != e; ++i) {
      if (!touchedBlocks[i]) {
	touchedBlocks[i] = true;
	BasicBlock *B = rpoToBlock[i];
	for (BasicBlock::iterator BI = B->begin(), BE = B->end();
	     BI != BE; ++BI)
	  touchedInstructions.insert(BI);
      }
    }
  }
}
uint32_t GVN::nextCongruenceNum = 0;

/// runOnFunction - This is the main transformation entry point for a function.
bool GVN::runOnFunction(Function& F) {
  bool Changed;
  // Merge unconditional branches, allowing PRE to catch more
  // optimization opportunities.
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ) {
    BasicBlock *BB = FI++;
    
    bool removedBlock = MergeBlockIntoPredecessor(BB, this);
    if (removedBlock) ++NumGVNBlocks;

    Changed |= removedBlock;
  }

  nextCongruenceNum = 2;
  // Initialize our rpo numbers so we can detect backwards edges
  uint32_t rpoBNumber = 0;
  uint32_t rpoINumber = 0;
  unsigned NumBasicBlocks = F.size();
  rpoToBlock.resize(NumBasicBlocks + 1);
  DEBUG(dbgs() << "Found " << NumBasicBlocks <<  " basic blocks\n");
  BasicBlock &EntryBlock = F.getEntryBlock();  
  ReversePostOrderTraversal<Function*> rpoT(&F);
  for (ReversePostOrderTraversal<Function*>::rpo_iterator RI = rpoT.begin(),
       RE = rpoT.end(); RI != RE; ++RI)
    {
      uint32_t IStart = rpoINumber;
      rpoBlockNumbers[*RI] = rpoBNumber++;
      // TODO: Use two worklist method
      // for (BasicBlock::iterator BI = EntryBlock.begin(), BE = EntryBlock.end();
      // 	   BI != BE; ++BI)
      // 	rpoInstructionNumbers[BI] = rpoINumber++;
      uint32_t IEnd = rpoINumber;
      rpoInstructionStartEnd[*RI] = std::make_pair(IStart, IEnd);
    }
  rpoToBlock.resize(rpoBNumber+1);
  for (Function::iterator BI = F.begin(), BE = F.end(); BI != BE; ++BI)
    rpoToBlock[rpoBlockNumbers[BI]] = BI;

  touchedBlocks.resize(rpoToBlock.size());
  for (unsigned i = 0, e = touchedBlocks.size(); i != e; ++i) {
    touchedBlocks[i] = false;
  }

  // Initialize the touched instructions to include the entry block
  for (BasicBlock::iterator BI = EntryBlock.begin(), BE = EntryBlock.end();
       BI != BE; ++BI)
    touchedInstructions.insert(BI);
  reachableBlocks.insert(&F.getEntryBlock());

  // Init the INITIAL class
  std::set<Value*> InitialValues;
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)  {
    for (BasicBlock::iterator BI = FI->begin(), BE = FI->end(); BI != BE; ++BI) {
      InitialValues.insert(BI);
    }
  }
  InitialClass = new CongruenceClass();
  for (std::set<Value*>::iterator LI = InitialValues.begin(), LE = InitialValues.end();
       LI != LE; ++LI)
    valueToClass[*LI] = InitialClass;
  InitialClass->members.swap(InitialValues);
  congruenceClass.push_back(InitialClass);
  if (!NoLoads)
    MD = &getAnalysis<MemoryDependenceAnalysis>();
  DT = &getAnalysis<DominatorTree>();
  PDT = &getAnalysis<PostDominatorTree>();

  TD = getAnalysisIfAvailable<TargetData>();
  TLI = &getAnalysis<TargetLibraryInfo>();
  AA = &getAnalysis<AliasAnalysis>();

  while (!touchedInstructions.empty()) {
    //TODO:We should just sort the damn touchedinstructions and touchedblocks into RPO order after every iteration
    //TODO: or Use two worklist method to keep ordering straight
    //TODO: or Investigate RPO numbering both blocks and instructions in the same pass,
    //  and walknig both lists at the same time, processing whichever has the next number in order.
    ReversePostOrderTraversal<Function*> rpoT(&F);
    for (ReversePostOrderTraversal<Function*>::rpo_iterator RI = rpoT.begin(),
           RE = rpoT.end(); RI != RE; ++RI) {
      //TODO(Predication)
      bool blockReachable = reachableBlocks.count(*RI);
      bool movedForward = false;
      for (BasicBlock::iterator BI = (*RI)->begin(), BE = (*RI)->end(); BI != BE; !movedForward ? BI++ : BI) {
	movedForward = false;
        DenseSet<Instruction*>::iterator DI = touchedInstructions.find(BI);
        if (DI != touchedInstructions.end()) {
	  DEBUG(dbgs() << "Processing instruction " << *BI << "\n");
          touchedInstructions.erase(DI);
	  if (!blockReachable) {
	    DEBUG(dbgs() << "Skipping instruction " << *BI  << " because block " << getBlockName(*RI) << " is unreachable\n");
	    continue;
	  }
	  Instruction *I = BI++;
	  movedForward = true;
	  
	  // If the instruction can be easily simplified then do so now in preference
	  // to value numbering it.  Value numbering often exposes redundancies, for
	  // example if it determines that %y is equal to %x then the instruction
	  // "%z = and i32 %x, %y" becomes "%z = and i32 %x, %x" which
	  // we now simplify.
	  //TODO:This causes us to invalidate memdep for no good
	  // reason.  The instructions *should* value number the same
	  // either way, so i'd prefer to do the elimination as a
	  // post-pass. 
	  if (0)
	  if (Value *V = SimplifyInstruction(I, TD, TLI, DT)) {
	    // Mark the uses as touched
	    for (Value::use_iterator UI = I->use_begin(), UE = I->use_end();
		 UI != UE; ++UI) {
	      Instruction *User = cast<Instruction>(*UI);
	      touchedInstructions.insert(User);
	    }

	    I->replaceAllUsesWith(V);
	    if (MD && V->getType()->isPointerTy())
	      MD->invalidateCachedPointerInfo(V);
	    DEBUG(dbgs() << "GVN removed: " << *I << '\n');
	    if (MD) MD->removeInstruction(I);
	    I->eraseFromParent();
	    DEBUG(verifyRemoved(I));
	    ++NumGVNSimpl;
	    continue;
	  }
	  
          if (!I->isTerminator()) {
	    Expression *Symbolized = performSymbolicEvaluation(I, *RI);
            performCongruenceFinding(I, Symbolized);
          } else {
            processOutgoingEdges(dyn_cast<TerminatorInst>(I));
          }
        }
      }
    }
  }

  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI) {
    if (!reachableBlocks.count(FI)) {
      DEBUG(dbgs() << "We believe block " << getBlockName(FI) << " is unreachable\n");
    }
  }

  valueToClass.clear();
  for (unsigned i = 0 , e = congruenceClass.size(); i != e; ++i) {
    delete congruenceClass[i];
    congruenceClass[i] = NULL;
  }
  
  congruenceClass.clear();
  expressionToClass.clear();
  rpoBlockNumbers.clear();
  rpoInstructionNumbers.clear();
  rpoInstructionStartEnd.clear();
  rpoToBlock.clear();
  reachableBlocks.clear();
  reachableEdges.clear();
  touchedInstructions.clear();
  touchedBlocks.clear();
  std::vector<Expression*> toDelete;
 
  for (DenseMap<Expression*, bool>::iterator I = uniquedExpressions.begin(), E = uniquedExpressions.end(); I != E; ++I) {
    toDelete.push_back(I->first);
   
  }
  uniquedExpressions.clear();
  for (unsigned i = 0, e = toDelete.size(); i != e; ++i) {
    delete toDelete[i];
    toDelete[i] = NULL;
  }
  
  
  return true;
  // bool Changed = false;
  // bool ShouldContinue = true;


  // unsigned Iteration = 0;
  // while (ShouldContinue) {
  //   DEBUG(dbgs() << "GVN iteration: " << Iteration << "\n");
  //   ShouldContinue = iterateOnFunction(F);
  //   if (splitCriticalEdges())
  //     ShouldContinue = true;
  //   Changed |= ShouldContinue;
  //   ++Iteration;
  // }

  // if (EnablePRE) {
  //   bool PREChanged = true;
  //   while (PREChanged) {
  //     PREChanged = performPRE(F);
  //     Changed |= PREChanged;
  //   }
  // }
  // // FIXME: Should perform GVN again after PRE does something.  PRE can move
  // // computations into blocks where they become fully redundant.  Note that
  // // we can't do this until PRE's critical edge splitting updates memdep.
  // // Actually, when this happens, we should just fully integrate PRE into GVN.

  // cleanupGlobalSets();

  // return Changed;
}




/// verifyRemoved - Verify that the specified instruction does not occur in our
/// internal data structures.
void GVN::verifyRemoved(const Instruction *Inst) const {
  // Walk through the value number scope to make sure the instruction isn't
  // ferreted away in it.
  for (DenseMap<uint32_t, LeaderTableEntry>::const_iterator
       I = LeaderTable.begin(), E = LeaderTable.end(); I != E; ++I) {
    const LeaderTableEntry *Node = &I->second;
    assert(Node->Val != Inst && "Inst still in value numbering scope!");
    
    while (Node->Next) {
      Node = Node->Next;
      assert(Node->Val != Inst && "Inst still in value numbering scope!");
    }
  }
}
