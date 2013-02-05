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
#include "llvm/Metadata.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Analysis/PHITransAddr.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/PatternMatch.h"
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
#include "llvm/Support/ConstantRange.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/PatternMatch.h"
#include "llvm/Support/RecyclingAllocator.h"
#include "llvm/Transforms/Utils/Local.h"


#include <algorithm>
#include <list>
#include <set>

using namespace llvm;
using namespace PatternMatch;

// Overview of GVN algorithm
// This algorithm is based on "A sparse algorithm for predicated
// global value numbering" by Karthik Gargi (PLDI 2002).
// The algorithm is modified somewhat.
// We do not perform value inference (expensive, not shown to be worth
// it)
// We do not perform the standard form of predication (ditto).


// The basic algorithm works like so:
// Create an initial congruence class
// For each instruction in the program
//   Place instruction in the initial congruence class
// Mark all blocks as unreachable
// Mark entry block as reachable
// Mark instructions in entry block as touched

//  While there are still touched instructions
//   If the touched instruction is not in a reachable block, skip it.
//   If the touched instruction is a block terminator
//      Evaluate edge reachability
//      Mark BB's with reachable edges as reachable.
//      If block is newly reachable
//        Mark instructions in block as touched
//      else if we made a new reachable edge to a block
//        Mark phi nodes in block as touched.
//
//      Propagate edge based equalities to newly reachable edge
//   else
//     Perform symbolic evaluation on the instruction
//     Lookup a congruence class for the symbolically evaluated instruction
//     If we didn't find a congruence class
//        Make a new congruence class for the instruction
//        Assign a leader to the congruence class
//     else
//        Add the instruction to the congruence class
//     If the congruence class of the instructionc changed
//       Remove the instruction from any old congruence class
//       (including updating leaders, etc)
//       Mark all users of the instruction as touched.
//
//

// Once our touched instruction queue is empty, we are done.
// We then eliminate instructions in favor of instructions with the
// same value that dominate them.

// Lastly, we perform a local type of PRE to eliminate some
// redundancies.

// Large TODO list
// 1. Make PRE a little saner
// 2. Fix pointer dependency caching in memdep
// 3. Use load + store value numbering + iteration, instead of
// equality hacks.
//
// Small TODO list
// 1. Support load of undef/lifetime start/etc = undef, as per old
// GVN.  My first thought is to just create an UndefExpression, and
// treat these loads/etc as equivalent to it, so that the right values
// get propagated around. 
// When elimination comes around, prefer undef's as leaders as we do
// with constants.
// 2. Support simplification into copies (IE and x,x -> x)
// 3. Duplicate bitcast removal

STATISTIC(NumGVNInstrDeleted,  "Number of instructions deleted");
STATISTIC(NumGVNBlocksDeleted, "Number of blocks deleted");
STATISTIC(NumGVNLoad,   "Number of loads deleted");
STATISTIC(NumGVNPRE,    "Number of instructions PRE'd");
STATISTIC(NumGVNBlocks, "Number of blocks merged");
STATISTIC(NumGVNEqProp, "Number of equalities propagated");
STATISTIC(NumPRELoad,   "Number of loads PRE'd");
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
namespace {
  typedef DenseMap<std::pair<std::pair<Value*, Value*>, BasicBlock*>, bool> DepBBQueryMap;
  DepBBQueryMap depQueryCache;
  typedef DenseMap<std::pair<Value*, Value*>, bool> DepIQueryMap;
  DepIQueryMap depIQueryCache;
  typedef DenseMap<Value *, SmallVector<NonLocalDepResult, 64> > LocDepMap;
  LocDepMap locDepCache;

  static void UpdateMemDepInfo(MemoryDependenceAnalysis *MD, Instruction *I, Value *V) {
    if (MD && V && V->getType()->isPointerTy())
      MD->invalidateCachedPointerInfo(V);
    if (MD && I && !I->isTerminator())
      MD->removeInstruction(I);
  }

  MemoryDependenceAnalysis *MD;
  const TargetData *TD;
  AliasAnalysis *AA;
  DenseSet<BasicBlock*> reachableBlocks;


/// CanCoerceMustAliasedValueToLoad - Return true if
/// CoerceAvailableValueToLoadType will succeed.
static bool CanCoerceMustAliasedValueToLoad(Value *StoredVal,
                                            Type *LoadTy,
                                            const TargetData &TD) {
  // If the loaded or stored value is an first class array or struct, don't try
  // to transform them.  We need to be able to bitcast to integer.
  if (LoadTy->isStructTy() || LoadTy->isArrayTy() ||
      StoredVal->getType()->isStructTy() ||
      StoredVal->getType()->isArrayTy())
    return false;

  // The store has to be at least as big as the load.
  if (TD.getTypeSizeInBits(StoredVal->getType()) <
        TD.getTypeSizeInBits(LoadTy))
    return false;

  return true;
}


/// AnalyzeLoadFromClobberingWrite - This function is called when we have a
/// memdep query of a load that ends up being a clobbering memory write (store,
/// memset, memcpy, memmove).  This means that the write *may* provide bits used
/// by the load but we can't be sure because the pointers don't mustalias.
///
/// Check this case to see if there is anything more we can do before we give
/// up.  This returns -1 if we have to give up, or a byte number in the stored
/// value of the piece that feeds the load.
static int AnalyzeLoadFromClobberingWrite(Type *LoadTy, Value *LoadPtr,
                                          Value *WritePtr,
                                          uint64_t WriteSizeInBits,
                                          const TargetData &TD) {
  // If the loaded or stored value is a first class array or struct, don't try
  // to transform them.  We need to be able to bitcast to integer.
  if (LoadTy->isStructTy() || LoadTy->isArrayTy())
    return -1;

  int64_t StoreOffset = 0, LoadOffset = 0;
  Value *StoreBase = GetPointerBaseWithConstantOffset(WritePtr, StoreOffset,TD);
  Value *LoadBase = GetPointerBaseWithConstantOffset(LoadPtr, LoadOffset, TD);
  if (StoreBase != LoadBase)
    return -1;

  // If the load and store are to the exact same address, they should have been
  // a must alias.  AA must have gotten confused.
  // FIXME: Study to see if/when this happens.  One case is forwarding a memset
  // to a load from the base of the memset.
#if 0
  if (LoadOffset == StoreOffset) {
    dbgs() << "STORE/LOAD DEP WITH COMMON POINTER MISSED:\n"
    << "Base       = " << *StoreBase << "\n"
    << "Store Ptr  = " << *WritePtr << "\n"
    << "Store Offs = " << StoreOffset << "\n"
    << "Load Ptr   = " << *LoadPtr << "\n";
    abort();
  }
#endif

  // If the load and store don't overlap at all, the store doesn't provide
  // anything to the load.  In this case, they really don't alias at all, AA
  // must have gotten confused.
  uint64_t LoadSize = TD.getTypeSizeInBits(LoadTy);

  if ((WriteSizeInBits & 7) | (LoadSize & 7))
    return -1;
  uint64_t StoreSize = WriteSizeInBits >> 3;  // Convert to bytes.
  LoadSize >>= 3;


  bool isAAFailure = false;
  if (StoreOffset < LoadOffset)
    isAAFailure = StoreOffset+int64_t(StoreSize) <= LoadOffset;
  else
    isAAFailure = LoadOffset+int64_t(LoadSize) <= StoreOffset;

  if (isAAFailure) {
#if 0
    dbgs() << "STORE LOAD DEP WITH COMMON BASE:\n"
    << "Base       = " << *StoreBase << "\n"
    << "Store Ptr  = " << *WritePtr << "\n"
    << "Store Offs = " << StoreOffset << "\n"
    << "Load Ptr   = " << *LoadPtr << "\n";
    abort();
#endif
    return -1;
  }

  // If the Load isn't completely contained within the stored bits, we don't
  // have all the bits to feed it.  We could do something crazy in the future
  // (issue a smaller load then merge the bits in) but this seems unlikely to be
  // valuable.
  if (StoreOffset > LoadOffset ||
      StoreOffset+StoreSize < LoadOffset+LoadSize)
    return -1;

  // Okay, we can do this transformation.  Return the number of bytes into the
  // store that the load is.
  return LoadOffset-StoreOffset;
}

/// AnalyzeLoadFromClobberingStore - This function is called when we have a
/// memdep query of a load that ends up being a clobbering store.
static int AnalyzeLoadFromClobberingStore(Type *LoadTy, Value *LoadPtr,
                                          StoreInst *DepSI,
                                          const TargetData &TD) {
  // Cannot handle reading from store of first-class aggregate yet.
  if (DepSI->getValueOperand()->getType()->isStructTy() ||
      DepSI->getValueOperand()->getType()->isArrayTy())
    return -1;

  Value *StorePtr = DepSI->getPointerOperand();
  uint64_t StoreSize =TD.getTypeSizeInBits(DepSI->getValueOperand()->getType());
  return AnalyzeLoadFromClobberingWrite(LoadTy, LoadPtr,
                                        StorePtr, StoreSize, TD);
}

/// AnalyzeLoadFromClobberingLoad - This function is called when we have a
/// memdep query of a load that ends up being clobbered by another load.  See if
/// the other load can feed into the second load.
static int AnalyzeLoadFromClobberingLoad(Type *LoadTy, Value *LoadPtr,
                                         LoadInst *DepLI, const TargetData &TD){
  // Cannot handle reading from store of first-class aggregate yet.
  if (DepLI->getType()->isStructTy() || DepLI->getType()->isArrayTy())
    return -1;

  Value *DepPtr = DepLI->getPointerOperand();
  uint64_t DepSize = TD.getTypeSizeInBits(DepLI->getType());
  int R = AnalyzeLoadFromClobberingWrite(LoadTy, LoadPtr, DepPtr, DepSize, TD);
  if (R != -1) return R;

  // If we have a load/load clobber an DepLI can be widened to cover this load,
  // then we should widen it!
  int64_t LoadOffs = 0;
  const Value *LoadBase =
    GetPointerBaseWithConstantOffset(LoadPtr, LoadOffs, TD);
  unsigned LoadSize = TD.getTypeStoreSize(LoadTy);

  unsigned Size = MemoryDependenceAnalysis::
    getLoadLoadClobberFullWidthSize(LoadBase, LoadOffs, LoadSize, DepLI, TD);
  if (Size == 0) return -1;

  return AnalyzeLoadFromClobberingWrite(LoadTy, LoadPtr, DepPtr, Size*8, TD);
}


static int AnalyzeLoadFromClobberingMemInst(Type *LoadTy, Value *LoadPtr,
                                            MemIntrinsic *MI,
                                            const TargetData &TD) {
  // If the mem operation is a non-constant size, we can't handle it.
  ConstantInt *SizeCst = dyn_cast<ConstantInt>(MI->getLength());
  if (SizeCst == 0) return -1;
  uint64_t MemSizeInBits = SizeCst->getZExtValue()*8;

  // If this is memset, we just need to see if the offset is valid in the size
  // of the memset..
  if (MI->getIntrinsicID() == Intrinsic::memset)
    return AnalyzeLoadFromClobberingWrite(LoadTy, LoadPtr, MI->getDest(),
                                          MemSizeInBits, TD);

  // If we have a memcpy/memmove, the only case we can handle is if this is a
  // copy from constant memory.  In that case, we can read directly from the
  // constant memory.
  MemTransferInst *MTI = cast<MemTransferInst>(MI);

  Constant *Src = dyn_cast<Constant>(MTI->getSource());
  if (Src == 0) return -1;

  GlobalVariable *GV = dyn_cast<GlobalVariable>(GetUnderlyingObject(Src, &TD));
  if (GV == 0 || !GV->isConstant()) return -1;

  // See if the access is within the bounds of the transfer.
  int Offset = AnalyzeLoadFromClobberingWrite(LoadTy, LoadPtr,
                                              MI->getDest(), MemSizeInBits, TD);
  if (Offset == -1)
    return Offset;

  // Otherwise, see if we can constant fold a load from the constant with the
  // offset applied as appropriate.
  Src = ConstantExpr::getBitCast(Src,
                                 llvm::Type::getInt8PtrTy(Src->getContext()));
  Constant *OffsetCst =
    ConstantInt::get(Type::getInt64Ty(Src->getContext()), (unsigned)Offset);
  Src = ConstantExpr::getGetElementPtr(Src, OffsetCst);
  Src = ConstantExpr::getBitCast(Src, PointerType::getUnqual(LoadTy));
  if (ConstantFoldLoadFromConstPtr(Src, &TD))
    return Offset;
  return -1;
}


  static std::string getBlockName(BasicBlock *B) {
    return DOTGraphTraits<const Function*>::getSimpleNodeLabel(B, NULL);
  }

  class Expression;

  struct ValueDFS {
    int dfs_in;
    int dfs_out;
    int localnum;
    Value *value;
    bool equivalence;
    bool operator <(const ValueDFS &other) const {
      if (dfs_in < other.dfs_in)
        return true;
      else if (dfs_in == other.dfs_in)
        {
          if (dfs_out < other.dfs_out)
            return true;
          else if (dfs_out == other.dfs_out)
	    {
	      if (localnum < other.localnum)
		return true;
	      else if (localnum == other.localnum)
		return !!equivalence < !!other.equivalence;
	    }
        }
      return false;
    }
  };

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
  // Each congruence class also has a canonical value expression,
  // though the expression may be null.  The expression is simply an
  // easy

  struct CongruenceClass {
    typedef DenseSet<std::pair<Value*, BasicBlock*> > MemberSet;
    static uint32_t nextCongruenceNum;
    uint32_t id;
    Value* leader;
    // TODO: Do we actually need this? It's not clear what purpose it
    // really serves
    Expression* expression;
    // Actual members of this class.  These are the things the same everywhere
    MemberSet members;
    // Noted equivalences.  These are things that are equivalence to
    // this class over certain paths.  This could be replaced with
    // proper predicate support during analysis.
    MemberSet equivalences;
    bool dead;
    explicit CongruenceClass():id(nextCongruenceNum++), leader(0), expression(0),
                               dead(false) {}
    CongruenceClass(Value *Leader, Expression *E):id(nextCongruenceNum++),
                                                  leader(Leader), expression(E),
                                                  dead(false) {}
  };

  DenseMap<Value*, CongruenceClass*> valueToClass;
  BumpPtrAllocator expressionAllocator;

  enum ExpressionType {
    ExpressionTypeBase,
    ExpressionTypeConstant,
    ExpressionTypeVariable,
    ExpressionTypeBasicStart,
    ExpressionTypeBasic,
    ExpressionTypeCall,
    ExpressionTypeInsertValue,
    ExpressionTypePhi,
    ExpressionTypeMemory,
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

    virtual ~Expression() {}

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
    // Return true if this is equivalent to the other expression, including memory dependencies
    virtual bool depequals(const Expression &other) {
      return true;
    }

    virtual hash_code getHashValue() const {
      return hash_combine(etype_, opcode_);
    }
    virtual void print(raw_ostream &OS) {
      OS << "{etype = " << etype_ << ", opcode = " << opcode_ << " }";
    }
    void *operator new(size_t s) {
      return expressionAllocator.Allocate<Expression>();
    }
    // Since this is a bump ptr allocated structure, deletes do nothing but call the destructors.
    void operator delete(void *p) {
      expressionAllocator.Deallocate(p);
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

    Type *getType() const {
      return type_;
    }

    SmallVector<Value*, 4> varargs;

    BasicExpression():type_(NULL)  {
      etype_ = ExpressionTypeBasic;
    }

    virtual ~BasicExpression() {}

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
    void *operator new(size_t s) {
      return expressionAllocator.Allocate<BasicExpression>();
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
    }

    virtual ~CallExpression() {}

    virtual bool equals(const Expression &other) const {
      // Two calls are never the same if we don't have memory dependence info
      if (!MD)
        return false;
      const CallExpression &OE = cast<CallExpression>(other);
      if (type_ != OE.type_)
        return false;
      // Calls are unequal unless they have the same arguments
      if (varargs != OE.varargs)
        return false;
      return true;
    }
    virtual bool depequals(const Expression &other);

    virtual hash_code getHashValue() const {
      return hash_combine(etype_, opcode_, type_, nomem_, readonly_,
                          hash_combine_range(varargs.begin(),
                                             varargs.end()));
    }

    virtual void print(raw_ostream &OS) {
      OS << "{etype = " << etype_ << ", opcode = " << opcode_ << ", varargs = {";
      for (unsigned i = 0, e = varargs.size(); i != e; ++i) {
        OS << "[" << i << "] = " << varargs[i] << "  ";
      }
      OS << "}";
      OS << " represents call at " << callinst_;
      OS << ", nomem = " << nomem_ << ", readonly = " << readonly_ << "}";
    }

    void *operator new(size_t s) {
      return expressionAllocator.Allocate<CallExpression>();
    }

  };
  class MemoryExpression: public BasicExpression {
  private:
    void operator=(const MemoryExpression&); // Do not implement
    MemoryExpression(const MemoryExpression&); // Do not implement
  protected:
    bool nonLocal_;
    bool isStore_;
    union {
      Instruction *inst;
      LoadInst *loadinst;
      StoreInst *storeinst;
    } inst_;

  public:

    // True if this memory expression had non local dependencies
    bool hadNonLocal() const {
      return nonLocal_;
    }

    bool isStore() const {
      return isStore_;
    }

    LoadInst *getLoadInst() const {
      assert (!isStore_ && "This is not a load memory expression");
      return inst_.loadinst;
    }

    StoreInst *getStoreInst() const {
      assert (isStore_ && "This is not a store memory expression");
      return inst_.storeinst;
    }

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const MemoryExpression *) { return true; }
    static inline bool classof(const Expression *EB) {
      return EB->getExpressionType() == ExpressionTypeMemory;
    }
    MemoryExpression(LoadInst *L) {
      etype_ = ExpressionTypeMemory;
      isStore_ = false;
      nonLocal_ = false;
      inst_.loadinst = L;
    }

    MemoryExpression(StoreInst *S) {
      etype_ = ExpressionTypeMemory;
      isStore_ = true;
      nonLocal_ = false;
      inst_.storeinst = S;
    }

    virtual ~MemoryExpression() {}

    virtual bool equals(const Expression &other) const {
      // Two loads/stores are never the same if we don't have memory dependence info
      if (!MD)
        return false;
      const MemoryExpression &OE = cast<MemoryExpression>(other);
      if (varargs != OE.varargs)
        return false;
      return true;
    }

    virtual bool depequals(const Expression &other);
    bool nonLocalEquals(LoadInst *LI, MemDepResult &Dep,
                        const MemoryExpression &OE) const;


    virtual hash_code getHashValue() const {
      return hash_combine(etype_, opcode_,
                          hash_combine_range(varargs.begin(),
                                             varargs.end()));
    }
    void *operator new(size_t s) {
      return expressionAllocator.Allocate<MemoryExpression>();
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
    }

    virtual ~InsertValueExpression() {}

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
    void *operator new(size_t s) {
      return expressionAllocator.Allocate<InsertValueExpression>();
    }

  };

  class PHIExpression : public BasicExpression {
  public:
    void *operator new(size_t s) {
      return expressionAllocator.Allocate<PHIExpression>();
    }

    /// Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const PHIExpression *) { return true; }
    static inline bool classof(const Expression *EB) {
      return EB->getExpressionType() == ExpressionTypePhi;
    }
    BasicBlock *getBB() const {
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

    Value *getVariableValue() const {
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

    VariableExpression():Expression(ExpressionTypeVariable),
                         variableValue_(NULL) { }


    VariableExpression(Value *variableValue):Expression(ExpressionTypeVariable),
                                             variableValue_(variableValue) { }
    virtual hash_code getHashValue() const {
      return hash_combine(etype_, variableValue_->getType(), variableValue_);
    }

    virtual void print(raw_ostream &OS) {
      OS << "{etype = " << etype_ << ", opcode = " << opcode_ << ", variable = " << variableValue_ << " }";
    }
    void *operator new(size_t s) {
      return expressionAllocator.Allocate<VariableExpression>();
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
    Constant *getConstantValue() const {
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

    ConstantExpression():Expression(ExpressionTypeConstant),
                         constantValue_(NULL) { }


    ConstantExpression(Constant *constantValue):Expression(ExpressionTypeConstant),
                                                constantValue_(constantValue) { }
    virtual hash_code getHashValue() const {
      return hash_combine(etype_, constantValue_->getType(), constantValue_);
    }
    void *operator new(size_t s) {
      return expressionAllocator.Allocate<ConstantExpression>();
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
    CongruenceClass *InitialClass;
    bool noLoads;
    DominatorTree *DT;
    const TargetLibraryInfo *TLI;
    DenseSet<std::pair<BasicBlock*, BasicBlock*> > reachableEdges;
    DenseSet<Instruction*> touchedInstructions;
    DenseMap<Instruction*, uint32_t> processedCount;
    std::vector<CongruenceClass*> congruenceClass;

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
      static bool isEqual(Expression *LHS, const Expression *RHS) {
        if (LHS == RHS)
          return true;
        if (LHS == getTombstoneKey() || RHS == getTombstoneKey()
            || LHS == getEmptyKey() || RHS == getEmptyKey())
          return false;
        return *LHS == *RHS && LHS->depequals(*RHS);
      }
    };

    typedef DenseMap<Expression*, CongruenceClass*, ComparingExpressionInfo> ExpressionClassMap;
    ExpressionClassMap expressionToClass;
    // We separate out the memory expressions to keep hashtable resizes from occurring as often.
    ExpressionClassMap memoryExpressionToClass;
    DenseSet<Expression*, ComparingExpressionInfo> uniquedExpressions;
    DenseSet<Expression*> expressionToDelete;
    DenseSet<Value*> changedValues;

    Value *lookupOperandLeader(Value*);
    // expression handling
    Expression *createExpression(Instruction*);
    void setBasicExpressionInfo(Instruction*, BasicExpression*);
    Expression *createPHIExpression(Instruction*);
    Expression *createVariableExpression(Value*);
    Expression *createConstantExpression(Constant*);
    Expression *createMemoryExpression(StoreInst*);
    Expression *createMemoryExpression(LoadInst*);
    Expression *createCallExpression(CallInst*, bool, bool);
    Expression *createInsertValueExpression(InsertValueInst*);
    Expression *uniquifyExpression(Expression*);
    Expression *createCmpExpression(unsigned, Type*,CmpInst::Predicate,
                                    Value*, Value*);

  public:
    static char ID; // Pass identification, replacement for typeid
    explicit GVN(bool noloads = false)
      : FunctionPass(ID), noLoads(noloads) {
      initializeGVNPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F);

    const TargetData *getTargetData() const { return TD; }
    DominatorTree &getDominatorTree() const { return *DT; }
    AliasAnalysis *getAliasAnalysis() const { return AA; }
    MemoryDependenceAnalysis &getMemDep() const { return *MD; }

    // New instruction creation
    void handleNewInstruction(Instruction*);
    void markUsersTouched(Value*);
    Value *CoerceAvailableValueToLoadType(Value*,Type*,Instruction*,const TargetData &);

    Value *GetStoreValueForLoad(Value*, unsigned, Type*, Instruction*,
                                const TargetData&);
    Value *GetLoadValueForLoad(LoadInst*, unsigned, Type*, Instruction*,
                               const TargetData&);
    Value *GetMemInstValueForLoad(MemIntrinsic*, unsigned, Type*,
                                  Instruction*, const TargetData&);

  private:
    // Congruence class handling
    CongruenceClass *createCongruenceClass(Value *Leader, Expression* E) {
      CongruenceClass *result = new CongruenceClass(Leader, E);
      congruenceClass.push_back(result);;
      return result;
    }

    CongruenceClass *createSingletonCongruenceClass(Value *Member, BasicBlock *BB) {
      CongruenceClass *CClass = createCongruenceClass(Member, NULL);
      CClass->members.insert(std::make_pair(Member, BB));
      valueToClass[Member] = CClass;
      return CClass;
    }

    DenseSet<Instruction*> instrsToErase_;
    void markInstructionForDeletion(Instruction *I) {
      DEBUG(dbgs() << "Marking " << *I << " for deletion\n");
      instrsToErase_.insert(I);
      // CongruenceClass *CC = valueToClass.lookup(I);
      // assert (!CC || CC == InitialClass);
      // valueToClass.erase(I);
    }
    void verifyRemoved(Instruction *I) {
      for (unsigned i = 0, e = congruenceClass.size(); i != e; ++i) {
        if (!congruenceClass[i] || congruenceClass[i] == InitialClass
            || congruenceClass[i]->dead)
          continue;
        CongruenceClass *CC = congruenceClass[i];
        assert (CC->leader != I && "Leader is messed up");
        assert (CC->members.count(std::make_pair(I, I->getParent())) == 0
                && "Removed instruction still a member");
      }
    }

    // Elimination
    bool eliminateInstructions(Function &);
    void convertDenseToDFSOrdered(CongruenceClass::MemberSet &, std::set<ValueDFS>&, bool);

    void replaceInstruction(Instruction*, Value*, CongruenceClass*);
    DenseMap<BasicBlock*, std::pair<int, int> > DFSBBMap;
    DenseMap<Instruction*, uint32_t> InstrLocalDFS;

    // Symbolic evaluation
    Expression *performSymbolicEvaluation(Value*, BasicBlock*);
    Expression *performSymbolicLoadEvaluation(Instruction*, BasicBlock*);
    Expression *performSymbolicStoreEvaluation(Instruction*, BasicBlock*);
    Expression *performSymbolicCallEvaluation(Instruction*, BasicBlock*);
    Expression *performSymbolicPHIEvaluation(Instruction*, BasicBlock*);
    // Congruence findin
    void performCongruenceFinding(Value*, BasicBlock*, Expression*);
    // Predicate and reachability handling
    void updateReachableEdge(BasicBlock*, BasicBlock*);
    void processOutgoingEdges(TerminatorInst* TI);
    void propagateChangeInEdge(BasicBlock*);
    // Non-local load handling
    bool processNonLocalLoad(LoadInst*);
    // Old GVN basic load emulation
    bool killUselessLoadInst(LoadInst*);
    DenseSet<LoadInst*> checkedLoads;
    
    // List of critical edges to be split between iterations.
    SmallVector<std::pair<TerminatorInst*, unsigned>, 4> toSplit;

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<DominatorTree>();
      AU.addRequired<TargetLibraryInfo>();
      if (!noLoads)
        AU.addRequired<MemoryDependenceAnalysis>();
      AU.addRequired<AliasAnalysis>();
      AU.addPreserved<DominatorTree>();
      AU.addPreserved<AliasAnalysis>();
    }


    // Helper fuctions
    // FIXME: eliminate or document these better
    void dump(DenseMap<uint32_t, Value*> &d);
    Value *findPRELeader(BasicBlock*, Value*);
    bool splitCriticalEdges();
    bool performPRE(Function &F);
    unsigned replaceAllDominatedUsesWith(Value*, Value*, BasicBlock*);
    bool propagateEquality(Value*, Value*, BasicBlock*);
  };

  char GVN::ID = 0;
}

// Test whether a load and a memory expression are going to produce
// the same value.  
// This is done by testing each non-local dependency to see if it is
// in the same congruence class, or if we can definitely extract the
// value of this load from the dependency.
// FIXME: At some point, the "value can be extracted" analysis should
// really be part of something else, and this should be straight
// dependency equality.
bool MemoryExpression::nonLocalEquals(LoadInst *LI, MemDepResult &Dep,
                                      const MemoryExpression &OE) const {
  // Check our two caches.  Note that the caches do not cause us
  // to lose any equivalences.
  // Because this is an optimistic value numbering, things do not
  // become *more equivalent*, only less equivalent.
  // As such, we only cache negative results, never positive ones
  // Positive results may become negative results in cases where
  // the congruence classes change
  // The first type of cache is a simple pairwise cache that tests
  // whether we found the instruction pair to have the same
  // dependency.
  // The second type of cache is a cache of (pointer argument, BB of
  // load instruction) -> result.

  DepIQueryMap::key_type iCacheKey(std::make_pair(LI, OE.inst_.inst));
  DepIQueryMap::const_iterator DII = depIQueryCache.find(iCacheKey);
  if (DII != depIQueryCache.end()) {
    return DII->second;
  }

  DepBBQueryMap::key_type cacheKey(std::make_pair(std::make_pair(varargs[0],
                                                                 OE.varargs[0]),
                                                  LI->getParent()));
  DepBBQueryMap::const_iterator DQI = depQueryCache.find(cacheKey);
  if (DQI != depQueryCache.end()) {
    DEBUG(dbgs() << "Dep query hit\n");
    return DQI->second;
  }
  // Check to see if all of the other deps have the same congruence class
  // as the load we are asking about, or we can reuse them.
  CongruenceClass *OIClass = valueToClass[OE.inst_.loadinst];
  if (!OIClass)
    return false;

  // We should mark the non-local ones so we can try to PRE them later
  SmallVector<NonLocalDepResult, 64> Deps;
  AliasAnalysis::Location Loc = AA->getLocation(LI);
  // TODO: This valueizes the pointer argument, which helps eliminate
  // false dependencies, and make pointer dependency testing faster by
  // making caching work better.
  // While this should be safe right now (getLocation rips the type info out
  // before we change it, so we aren't messing with what it thinks the
  // size of the access is, etc).
  // It may become unsafe in the future.
  // We should change the interface and code a bit to make this
  // explicit
  // Note that this catches significantly more loads, because we
  // avoid phi translation failures
  Loc.Ptr = varargs[0];
  Deps = locDepCache.lookup(LI);
  if (Deps.size() == 0)  {
    MD->getNonLocalPointerDependency(Loc, true, LI->getParent(), Deps);
    locDepCache[LI] = Deps;
  }

  // If we had to process more than one hundred deps to find the
  // dependencies, this load isn't worth worrying about.  Optimizing
  // it will be too expensive.
  unsigned NumDeps = Deps.size();
  if (NumDeps > 100) {
    depIQueryCache[iCacheKey] = false;
    depQueryCache[cacheKey] = false;
    return false;
  }
  if (NumDeps == 1 &&
      !Deps[0].getResult().isDef() && !Deps[0].getResult().isClobber()) {
    DEBUG(
          dbgs() << "GVN: non-local load ";
          WriteAsOperand(dbgs(), LI);
          dbgs() << " has unknown dependencies\n";
          );
    depIQueryCache[iCacheKey] = false;
    depQueryCache[cacheKey] = false;
    return false;
  }
  for (unsigned i = 0, e = NumDeps; i != e; ++i) {
    BasicBlock *DepBB = Deps[i].getBB();

    // We may have dependencies memdep has discovered but are in
    // unreachable blocks. They don't matter (block reachability
    // for things reaching this block should be completely
    // calculated by the time we get here)
    if (!reachableBlocks.count(DepBB))
      continue;
    MemDepResult DepInfo = Deps[i].getResult();
    if (!DepInfo.isDef() && !DepInfo.isClobber()) {
      depIQueryCache[iCacheKey] = false;
      depQueryCache[cacheKey] = false;
      return false;
    }
    // Make sure all dependencies are in the same congruence
    // class
    // TODO: need to be in the same congruence class or they need to be fixable.
    CongruenceClass *DepClass = valueToClass[DepInfo.getInst()];
    // It could be an unreachable dependency MemDep doesn't know
    // about.
    // If this is the case, it will still have the InitialClass as
    // its congruence class
    if (DepClass != OIClass) {
      depQueryCache[cacheKey] = false;
      return false;
    }

    if (DepInfo.isClobber()) {
      // The address being loaded in this non-local block may
      // not be the same as the pointer operand of the load
      // if PHI translation occurs.  Make sure to consider
      // the right address.
      Value *Address = Deps[i].getAddress();

      // If the dependence is to a store that writes to a
      // superset of the bits read by the load, we can
      // extract the bits we need for the load from the
      // stored value.
      if (StoreInst *DepSI = dyn_cast<StoreInst>(DepInfo.getInst())) {
        if (TD && Address) {
          int Offset = AnalyzeLoadFromClobberingStore(LI->getType(), Address,
                                                      DepSI, *TD);
          if (Offset == -1) {
            depIQueryCache[iCacheKey] = false;
            depQueryCache[cacheKey] = false;
            return false;
          }
          continue;
        }
      }

      // Check to see if we have something like this:
      //    load i32* P
      //    load i8* (P+1)
      // if we have this, we can replace the later with an
      // extraction from the former.
      if (LoadInst *DepLI = dyn_cast<LoadInst>(DepInfo.getInst())) {
        // If this is a clobber and L is the first
        // instruction in its block, then we have the first
        // instruction in the entry block.
        if (DepLI != LI && Address && TD) {
          int Offset = AnalyzeLoadFromClobberingLoad(LI->getType(),
                                                     LI->getPointerOperand(),
                                                     DepLI, *TD);
          if (Offset == -1) {
            depIQueryCache[iCacheKey] = false;
            depQueryCache[cacheKey] = false;
            return false;
          }
          continue;
        }
      }
      // If the clobbering value is a memset/memcpy/memmove, see if we can forward
      // a value on from it.
      if (MemIntrinsic *DepMI = dyn_cast<MemIntrinsic>(Dep.getInst())) {
        int Offset = AnalyzeLoadFromClobberingMemInst(LI->getType(),
                                                      LI->getPointerOperand(),
                                                      DepMI, *TD);
        if (Offset != -1) {
          depIQueryCache[iCacheKey] = false;
          depQueryCache[cacheKey] = false;
          return false;
        }
        continue;
      }
      depIQueryCache[iCacheKey] = false;
      depQueryCache[cacheKey] = false;
      return false;
    }

    // DepInfo.isDef() here
    Instruction *DepInst = DepInfo.getInst();

    if (StoreInst *S = dyn_cast<StoreInst>(DepInst)) {
      // Reject loads and stores that are to the same address
      // but are of different types if we have to.
      if (S->getValueOperand()->getType() != LI->getType()) {
        // If the stored value is larger or equal to the
        // loaded value, we can reuse it.
        if (TD == 0 || !CanCoerceMustAliasedValueToLoad(S->getValueOperand(),
                                                        LI->getType(), *TD)) {
          depIQueryCache[iCacheKey] = false;
          depQueryCache[cacheKey] = false;
          return false;
        }
      }
      continue;
    }
    // TODO: Better coercion
    if (LoadInst *LD = dyn_cast<LoadInst>(DepInst)) {
      // If the types mismatch and we can't handle it, reject reuse of the load.
      if (LD->getType() != LI->getType()) {
        // If the stored value is larger or equal to the loaded value, we can
        // reuse it.
        if (TD == 0 || !CanCoerceMustAliasedValueToLoad(LD, LI->getType(),*TD)) {
          depIQueryCache[iCacheKey] = false;
          depQueryCache[cacheKey] = false;
          return false;
        }
      }
      continue;
    }
    depIQueryCache[iCacheKey] = false;
    depQueryCache[cacheKey] = false;
    return false;
  }
  // If we got through all the dependencies, we are good to go
  return true;
}


// This is the normal memory dependency testing. We determine if two
// memory expressions have the same dependencies.  For stores, this is
// simple.  Two stores are valued the same if they store the same
// value to the same place.  For loads, this is more complicated. 
// For loads with non-local dependencies, we hand off testing to
// nonLocalEquals.
// For other loads, we check to see whether we are dependent on the
// instruction from the other memory dependency.  If so, we see if
// this and the other expression are the same except for type, or
// we can otherwise extract the value from it. 
// FIXME: At some point, the "value can be extracted" analysis should
// really be part of something else, and this should be straight
// dependency equality.
bool MemoryExpression::depequals(const Expression &other)
{
  const MemoryExpression &OE = cast<MemoryExpression>(other);
  if (!isStore_) {
    LoadInst *LI = inst_.loadinst;
    Instruction *OI = OE.inst_.inst;

    if (LI != OI) {
      MemDepResult Dep = MD->getDependency(LI);
      if (Dep.isNonLocal()) {
        nonLocal_ = true;
        return nonLocalEquals(LI, Dep, OE);
      }
      // If we weren't dependent on the other load, they aren't equal.
      // FIXME: This isn't strictly true, they could produce the same
      // value even if they were not dependent on each other, if the
      // dependency in the way is a false dependency.  This starts to
      // require fairly complicated analysis that does not belong in
      // an equals function.  
      if (Dep.getInst() != OI)
        return false;
      // If we are dependent, but it's not a straight def, see if they can be made equal.
      if (Dep.isClobber() && TD) {
        if (StoreInst *DepSI = dyn_cast<StoreInst>(Dep.getInst())) {
          int Offset = AnalyzeLoadFromClobberingStore(LI->getType(),
                                                      varargs[0],
                                                      DepSI, *TD);
          if (Offset == -1 || !CanCoerceMustAliasedValueToLoad(DepSI->getValueOperand(),
                                                               LI->getType(),
                                                               *TD))
            return false;
          return true;
        }
        if (LoadInst *DepLI = dyn_cast<LoadInst>(Dep.getInst())) {
          int Offset = AnalyzeLoadFromClobberingLoad(LI->getType(),
                                                     varargs[0],
                                                     DepLI, *TD);
          if (Offset == -1 || !CanCoerceMustAliasedValueToLoad(DepLI,
                                                               LI->getType(),
                                                               *TD))
            return false;
          return true;
        }
        // If the clobbering value is a memset/memcpy/memmove, see if we can forward
        // a value on from it.
        if (MemIntrinsic *DepMI = dyn_cast<MemIntrinsic>(Dep.getInst())) {
          int Offset = AnalyzeLoadFromClobberingMemInst(LI->getType(),
                                                        LI->getPointerOperand(),
                                                        DepMI, *TD);
          if (Offset == -1)
            return false;
          return true;
        }
        return false;
      }
      // If the load is def'd, just make sure we can coerce types.
      if (!Dep.isDef()) {
        return false;
      }
      Instruction *DepInst = Dep.getInst();
      if (StoreInst *DepSI = dyn_cast<StoreInst>(DepInst)) {
        Value *StoredVal = DepSI->getValueOperand();

        // The store and load are to a must-aliased pointer, but they may not
        // actually have the same type.  See if we know how to reuse the stored
        // value (depending on its type).
        if (StoredVal->getType() != LI->getType()) {
          if (!TD || !CanCoerceMustAliasedValueToLoad(StoredVal, LI->getType(), *TD))
            return false;
        }
        return true;
      }
      if (LoadInst *DepLI = dyn_cast<LoadInst>(DepInst)) {
        // The loads are of a must-aliased pointer, but they may not
        // actually have the same type.  See if we know how to reuse the stored
        // value (depending on its type).
        if (DepLI->getType() != LI->getType()) {
          if (!TD || !CanCoerceMustAliasedValueToLoad(DepLI, LI->getType(), *TD))
            return false;
        }
        return true;
      }
      //TODO: memintrinsic case
      return false;
    }
    return true;
  } else {
    // Two stores are the same if they store the same value to the same place.
    // TODO: Use lookup to valueize and store value of store
    if (OE.isStore_ && OE.inst_.storeinst->getValueOperand() == inst_.storeinst->getValueOperand())
      return true;
    return false;
  }

  return true;
}

bool CallExpression::depequals(const Expression &other) {
  const CallExpression &OE = cast<CallExpression>(other);
  // Given the same arguments, two calls are equal if the dependency checker says one is dependent
  // on the other
  if (callinst_ != OE.callinst_) {
    MemDepResult local_dep = MD->getDependency(callinst_);
    if (!local_dep.isDef() && !local_dep.isNonLocal()) {
      return false;
    }
    if (local_dep.isDef()) {
      CallInst* local_cdep = cast<CallInst>(local_dep.getInst());
      if (local_cdep != OE.callinst_)
        return false;
    } else {
      // True if all of the dependencies are either transparent or loads
      bool allTransparent = true;
      // Non-local case.
      const MemoryDependenceAnalysis::NonLocalDepInfo &deps =
        MD->getNonLocalCallDependency(callinst_);
      CallInst* cdep = 0;
      CongruenceClass *cclass = valueToClass[OE.callinst_];
      assert(cclass != NULL && "Somehow got a call instruction without a congruence class into the expressionToClass mapping");

      // Check to see if all non local dependencies are equal to
      // the other call or if all of them are in the same
      // congruence class as the other call instruction
      for (unsigned i = 0, e = deps.size(); i != e; ++i) {
        const NonLocalDepEntry *I = &deps[i];

        if (I->getResult().isNonLocal())
          continue;

        // Ignore clobbers by loads, since they have no impact on the call itself.
        if (I->getResult().isClobber() && isa<LoadInst>(I->getResult().getInst()))
          continue;

        allTransparent = false;
        if (!I->getResult().isDef() || cdep != 0) {
          cdep = 0;
          break;
        }
        // We need to ensure that all dependencies are in the same
        // congruence class as the other call instruction
        CallInst *NonLocalDepCall = dyn_cast<CallInst>(I->getResult().getInst());
        if (NonLocalDepCall) {
          if (!cdep) {
            if (NonLocalDepCall == OE.callinst_ || valueToClass[NonLocalDepCall] == cclass)
              cdep = NonLocalDepCall;
            else
              break;
          } else {
            CongruenceClass *NLDPClass = valueToClass[NonLocalDepCall];
            if (cclass != NLDPClass) {
              cdep = 0;
              break;
            }
          }
          continue;
        }
        cdep = 0;
        break;
      }
      if (!cdep && !allTransparent)
        return false;
    }
  }
  return true;
}

// createGVNPass - The public interface to this file...
FunctionPass *llvm::createGVNPass(bool noLoads) {
  return new GVN(noLoads);
}

INITIALIZE_PASS_BEGIN(GVN, "gvn", "Global Value Numbering", false, false)
INITIALIZE_PASS_DEPENDENCY(BreakCriticalEdges)
INITIALIZE_PASS_DEPENDENCY(MemoryDependenceAnalysis)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfo)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(GVN, "gvn", "Global Value Numbering", false, false)


/// CoerceAvailableValueToLoadType - If we saw a store of a value to memory, and
/// then a load from a must-aliased pointer of a different type, try to coerce
/// the stored value.  LoadedTy is the type of the load we want to replace and
/// InsertPt is the place to insert new instructions.
///
/// If we can't do it, return null.
Value *GVN::CoerceAvailableValueToLoadType(Value *StoredVal,
                                             Type *LoadedTy,
                                             Instruction *InsertPt,
                                             const TargetData &TD) {
  if (!CanCoerceMustAliasedValueToLoad(StoredVal, LoadedTy, TD))
    return 0;

  // If this is already the right type, just return it.
  Type *StoredValTy = StoredVal->getType();

  uint64_t StoreSize = TD.getTypeSizeInBits(StoredValTy);
  uint64_t LoadSize = TD.getTypeSizeInBits(LoadedTy);

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
      StoredValTy = TD.getIntPtrType(StoredValTy->getContext());
      Instruction *I = new PtrToIntInst(StoredVal, StoredValTy, "", InsertPt);
      StoredVal = I;
      handleNewInstruction(I);
    }

    Type *TypeToCastTo = LoadedTy;
    if (TypeToCastTo->isPointerTy())
      TypeToCastTo = TD.getIntPtrType(StoredValTy->getContext());

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
  // extract out a piece from it.  If the available value is too small, then we
  // can't do anything.
  assert(StoreSize >= LoadSize && "CanCoerceMustAliasedValueToLoad fail");

  // Convert source pointers to integers, which can be manipulated.
  if (StoredValTy->isPointerTy()) {
    StoredValTy = TD.getIntPtrType(StoredValTy->getContext());
    Instruction *I = new PtrToIntInst(StoredVal, StoredValTy, "", InsertPt);
    StoredVal = I;
    handleNewInstruction(I);
  }

  // Convert vectors and fp to integer, which can be manipulated.
  if (!StoredValTy->isIntegerTy()) {
    StoredValTy = IntegerType::get(StoredValTy->getContext(), StoreSize);
    Instruction *I =  new BitCastInst(StoredVal, StoredValTy, "", InsertPt);
    StoredVal = I;
    handleNewInstruction(I);
  }

  // If this is a big-endian system, we need to shift the value down to the low
  // bits so that a truncate will work.
  if (TD.isBigEndian()) {
    Constant *Val = ConstantInt::get(StoredVal->getType(), StoreSize-LoadSize);
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
Value *GVN::GetStoreValueForLoad(Value *SrcVal, unsigned Offset,
                                   Type *LoadTy,
                                   Instruction *InsertPt, const TargetData &TD){
  LLVMContext &Ctx = SrcVal->getType()->getContext();

  uint64_t StoreSize = (TD.getTypeSizeInBits(SrcVal->getType()) + 7) / 8;
  uint64_t LoadSize = (TD.getTypeSizeInBits(LoadTy) + 7) / 8;

  IRBuilder<> Builder(InsertPt->getParent(), InsertPt);

  // Compute which bits of the stored value are being used by the load.  Convert
  // to an integer type to start with.
  if (SrcVal->getType()->isPointerTy()) {
    SrcVal = Builder.CreatePtrToInt(SrcVal, TD.getIntPtrType(Ctx));
    if (Instruction *I = dyn_cast<Instruction>(SrcVal))
      handleNewInstruction(I);
  }

  if (!SrcVal->getType()->isIntegerTy()) {
    SrcVal = Builder.CreateBitCast(SrcVal, IntegerType::get(Ctx, StoreSize*8));
    if (Instruction *I = dyn_cast<Instruction>(SrcVal))
      handleNewInstruction(I);
  }
  // Shift the bits to the least significant depending on endianness.
  unsigned ShiftAmt;
  if (TD.isLittleEndian())
    ShiftAmt = Offset*8;
  else
    ShiftAmt = (StoreSize-LoadSize-Offset)*8;

  if (ShiftAmt) {
    SrcVal = Builder.CreateLShr(SrcVal, ShiftAmt);
    if (Instruction *I = dyn_cast<Instruction>(SrcVal))
      handleNewInstruction(I);
  }
  if (LoadSize != StoreSize) {
    SrcVal = Builder.CreateTrunc(SrcVal, IntegerType::get(Ctx, LoadSize*8));
    if (Instruction *I = dyn_cast<Instruction>(SrcVal))
      handleNewInstruction(I);
  }
  return CoerceAvailableValueToLoadType(SrcVal, LoadTy, InsertPt, TD);
}

/// GetLoadValueForLoad - This function is called when we have a
/// memdep query of a load that ends up being a clobbering load.  This means
/// that the load *may* provide bits used by the load but we can't be sure
/// because the pointers don't mustalias.  Check this case to see if there is
/// anything more we can do before we give up.
Value *GVN::GetLoadValueForLoad(LoadInst *SrcVal, unsigned Offset,
                                  Type *LoadTy, Instruction *InsertPt,
                                  const TargetData &TD) {
  // If Offset+LoadTy exceeds the size of SrcVal, then we must be wanting to
  // widen SrcVal out to a larger load.
  unsigned SrcValSize = TD.getTypeStoreSize(SrcVal->getType());
  unsigned LoadSize = TD.getTypeStoreSize(LoadTy);
  if (Offset+LoadSize > SrcValSize) {
    assert(SrcVal->isSimple() && "Cannot widen volatile/atomic load!");
    assert(SrcVal->getType()->isIntegerTy() && "Can't widen non-integer load");
    // If we have a load/load clobber an DepLI can be widened to cover this
    // load, then we should widen it to the next power of 2 size big enough!
    unsigned NewLoadSize = Offset+LoadSize;
    if (!isPowerOf2_32(NewLoadSize))
      NewLoadSize = NextPowerOf2(NewLoadSize);

    Value *PtrVal = SrcVal->getPointerOperand();

    // Insert the new load after the old load.  This ensures that subsequent
    // memdep queries will find the new load.  We can't easily remove the old
    // load completely because it is already in the value numbering table.
    IRBuilder<> Builder(SrcVal->getParent(), ++BasicBlock::iterator(SrcVal));
    Type *DestPTy =
      IntegerType::get(LoadTy->getContext(), NewLoadSize*8);
    DestPTy = PointerType::get(DestPTy,
                       cast<PointerType>(PtrVal->getType())->getAddressSpace());
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

    // Replace uses of the original load with the wider load.  On a big endian
    // system, we need to shift down to get the relevant bits.
    Value *RV = NewLoad;
    if (TD.isBigEndian()) {
      RV = Builder.CreateLShr(RV,
                    NewLoadSize*8-SrcVal->getType()->getPrimitiveSizeInBits());
      if (Instruction *I = dyn_cast<Instruction>(RV))
        handleNewInstruction(I);
    }

    RV = Builder.CreateTrunc(RV, SrcVal->getType());
    if (Instruction *I = dyn_cast<Instruction>(RV))
      handleNewInstruction(I);

    markUsersTouched(SrcVal);
    SrcVal->replaceAllUsesWith(RV);

    // We would like to use gvn.markInstructionForDeletion here, but we can't
    // because the load is already memoized into the leader map table that GVN
    // tracks.  It is potentially possible to remove the load from the table,
    // but then there all of the operations based on it would need to be
    // rehashed.  Just leave the dead load around.
    // FIXME: This is no longer a problem
    markInstructionForDeletion(SrcVal);
    UpdateMemDepInfo(MD, SrcVal, NULL);
    SrcVal = NewLoad;
  }

  return GetStoreValueForLoad(SrcVal, Offset, LoadTy, InsertPt, TD);
}

/// GetMemInstValueForLoad - This function is called when we have a
/// memdep query of a load that ends up being a clobbering mem intrinsic.
  Value *GVN::GetMemInstValueForLoad(MemIntrinsic *SrcInst, unsigned Offset,
                                     Type *LoadTy, Instruction *InsertPt,
                                     const TargetData &TD){
  LLVMContext &Ctx = LoadTy->getContext();
  uint64_t LoadSize = TD.getTypeSizeInBits(LoadTy)/8;

  IRBuilder<> Builder(InsertPt->getParent(), InsertPt);

  // We know that this method is only called when the mem transfer fully
  // provides the bits for the load.
  if (MemSetInst *MSI = dyn_cast<MemSetInst>(SrcInst)) {
    // memset(P, 'x', 1234) -> splat('x'), even if x is a variable, and
    // independently of what the offset is.
    Value *Val = MSI->getValue();
    if (LoadSize != 1) {
      Val = Builder.CreateZExt(Val, IntegerType::get(Ctx, LoadSize*8));
      if (Instruction *I = dyn_cast<Instruction>(Val))
        handleNewInstruction(I);
    }

    Value *OneElt = Val;

    // Splat the value out to the right number of bits.
    for (unsigned NumBytesSet = 1; NumBytesSet != LoadSize; ) {
      // If we can double the number of bytes set, do it.
      if (NumBytesSet*2 <= LoadSize) {
        Value *ShVal = Builder.CreateShl(Val, NumBytesSet*8);
        if (Instruction *I = dyn_cast<Instruction>(ShVal))
          handleNewInstruction(I);
        Val = Builder.CreateOr(Val, ShVal);
        if (Instruction *I = dyn_cast<Instruction>(Val))
          handleNewInstruction(I);
        NumBytesSet <<= 1;
        continue;
      }

      // Otherwise insert one byte at a time.
      Value *ShVal = Builder.CreateShl(Val, 1*8);
      if (Instruction *I = dyn_cast<Instruction>(ShVal))
        handleNewInstruction(I);

      Val = Builder.CreateOr(OneElt, ShVal);
      if (Instruction *I = dyn_cast<Instruction>(Val))
        handleNewInstruction(I);

      ++NumBytesSet;
    }

    return CoerceAvailableValueToLoadType(Val, LoadTy, InsertPt, TD);
  }

  // Otherwise, this is a memcpy/memmove from a constant global.
  MemTransferInst *MTI = cast<MemTransferInst>(SrcInst);
  Constant *Src = cast<Constant>(MTI->getSource());

  // Otherwise, see if we can constant fold a load from the constant with the
  // offset applied as appropriate.
  Src = ConstantExpr::getBitCast(Src,
                                 llvm::Type::getInt8PtrTy(Src->getContext()));
  Constant *OffsetCst =
  ConstantInt::get(Type::getInt64Ty(Src->getContext()), (unsigned)Offset);
  Src = ConstantExpr::getGetElementPtr(Src, OffsetCst);
  Src = ConstantExpr::getBitCast(Src, PointerType::getUnqual(LoadTy));
  return ConstantFoldLoadFromConstPtr(Src, &TD);
}

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
    if (I->getOperand(i) != I) {
      Value *Operand = lookupOperandLeader(I->getOperand(i));
      E->varargs.push_back(Operand);
    } else {
      E->varargs.push_back(I->getOperand(i));
    }
  }
  return E;
}

void GVN::setBasicExpressionInfo(Instruction *I, BasicExpression *E) {
  E->setType(I->getType());
  E->setOpcode(I->getOpcode());

  for (Instruction::op_iterator OI = I->op_begin(), OE = I->op_end();
       OI != OE; ++OI) {
    Value *Operand = lookupOperandLeader(*OI);
    E->varargs.push_back(Operand);
  }
}
// This is a special function only used by equality propagation, it
// should not be called elsewhere
Expression *GVN::createCmpExpression(unsigned Opcode,
                                     Type *Type,
                                     CmpInst::Predicate Predicate,
                                     Value *LHS, Value *RHS) {
  BasicExpression *E = new BasicExpression();
  E->setType(Type);
  E->setOpcode((Opcode << 8) | Predicate);
  LHS = lookupOperandLeader(LHS);
  E->varargs.push_back(LHS);
  RHS = lookupOperandLeader(RHS);
  E->varargs.push_back(RHS);
  return E;
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


  // Perform simplificaiton
  // TODO: Right now we only check to see if we get a constant result.
  // We may get a less than constant, but still better, result for
  // some operations.
  // IE
  //  add 0, x -> x 
  // and x, x -> x
  // We should handle this by simply rewriting the expression.
  if (CmpInst *CI = dyn_cast<CmpInst>(I)) {
    // Sort the operand value numbers so x<y and y>x get the same value number.
    CmpInst::Predicate Predicate = CI->getPredicate();
    if (E->varargs[0] > E->varargs[1]) {
      std::swap(E->varargs[0], E->varargs[1]);
      Predicate = CmpInst::getSwappedPredicate(Predicate);
    }
    E->setOpcode((CI->getOpcode() << 8) | Predicate);
    // TODO: 10% of our time is spent in SimplifyCmpInst with pointer operands
    //TODO: Since we noop bitcasts, we may need to check types before
    //simplifying, so that we don't end up simplifying based on a wrong
    //type assumption. We should clean this up so we can use constants of the wrong type

    assert (I->getOperand(0)->getType() == I->getOperand(1)->getType() && "Wrong types on cmp instruction");
    if ((E->varargs[0]->getType() == I->getOperand(0)->getType()
         && E->varargs[1]->getType() == I->getOperand(1)->getType())) {
      Value *V = SimplifyCmpInst(Predicate, E->varargs[0], E->varargs[1], TD, TLI, DT);
      Constant *C;
      if (V && (C = dyn_cast<Constant>(V))) {
        DEBUG(dbgs() << "Simplified " << *I << " to " << " constant " << *C << "\n");
        NumGVNCmpInsSimplified++;
        delete E;
        return createConstantExpression(C);
      }
    }

  } else if (isa<SelectInst>(I)) {
    if (isa<Constant>(E->varargs[0])
        || (E->varargs[1]->getType() == I->getOperand(1)->getType()
            && E->varargs[2]->getType() == I->getOperand(2)->getType())) {
      Value *V = SimplifySelectInst(E->varargs[0], E->varargs[1], E->varargs[2], TD, TLI, DT);
      if (V) {
        DEBUG(dbgs() << "Simplified " << *I << " to " << " " << *V << "\n");
        NumGVNCmpInsSimplified++;
        delete E;
        return performSymbolicEvaluation(V, I->getParent());
      }
    }
  } else if (I->isBinaryOp()) {
    //TODO: Since we noop bitcasts, we may need to check types before
    //simplifying, so that we don't end up simplifying based on a
    //wrong type assumption
    Value *V = SimplifyBinOp(E->getOpcode(), E->varargs[0], E->varargs[1], TD, TLI, DT);
    Constant *C;
    if (V && (C = dyn_cast<Constant>(V))) {
      DEBUG(dbgs() << "Simplified " << *I << " to " << " constant " << *C << "\n");
      NumGVNBinOpsSimplified++;
      delete E;
      return createConstantExpression(C);
    }
  } else if (BitCastInst *BI = dyn_cast<BitCastInst>(I)) {
    Value *V = SimplifyInstruction(BI);
    Constant *C;
    if (V && (C = dyn_cast<Constant>(V))) {
      DEBUG(dbgs() << "Simplified " << *I << " to " << " constant " << *C << "\n");
      NumGVNBinOpsSimplified++;
      delete E;
      return createConstantExpression(C);
    }
  } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I)) {
    //TODO: Since we noop bitcasts, we may need to check types before
    //simplifying, so that we don't end up simplifying based on a
    //wrong type assumption. We should clean this up so we can use
    //constants of the wrong type.
    if (GEP->getPointerOperandType() == E->varargs[0]->getType() ) {
      Value *V = SimplifyGEPInst(E->varargs, TD, TLI, DT);
      Constant *C;
      if (V && (C = dyn_cast<Constant>(V))) {
        DEBUG(dbgs() << "Simplified " << *I << " to " << " constant " << *C << "\n");
        NumGVNBinOpsSimplified++;
        delete E;
        return createConstantExpression(C);
      }
    }
  }

  return E;
}

Expression *GVN::uniquifyExpression(Expression *E) {
  std::pair<DenseSet<Expression *, ComparingExpressionInfo>::iterator, bool> P = uniquedExpressions.insert(E);
  if (!P.second && *(P.first) != E) {
    // We rely on the expression either never being inserted into
    // expressionsToDelete, or properly removed by our caller.
     
    assert(!expressionToDelete.count(E) && "This expression should not be in expressionsToDelete");
    delete E;
    return *(P.first);
  }
  return E;
}

Expression *GVN::createInsertValueExpression(InsertValueInst *I) {
  InsertValueExpression *E = new InsertValueExpression();
  setBasicExpressionInfo(I, E);
  for (InsertValueInst::idx_iterator II = I->idx_begin(), IE = I->idx_end();
       II != IE; ++II)
    E->intargs.push_back(*II);
  return E;
}

Expression *GVN::createVariableExpression(Value *V) {
  VariableExpression *E = new VariableExpression(V);
  E->setOpcode(V->getValueID());
  return E;
}

Expression *GVN::createConstantExpression(Constant *C) {
  ConstantExpression *E = new ConstantExpression(C);
  E->setOpcode(C->getValueID());
  return E;
}

Expression *GVN::createCallExpression(CallInst *CI, bool nomem, bool readonly) {
  CallExpression *E = new CallExpression(CI, nomem, readonly);
  setBasicExpressionInfo(CI, E);
  return E;
}

//lookupOperandLeader -- See if we have a congruence class and leader
// for this operand, and if so, return it. Otherwise, return the
// original operand
Value *GVN::lookupOperandLeader(Value *V) {
  DenseMap<Value*, CongruenceClass*>::iterator VTCI = valueToClass.find(V);
  if (VTCI != valueToClass.end()) {
    CongruenceClass *CC = VTCI->second;
    if (CC != InitialClass)
      return CC->leader;
  }
  return V;
}


Expression *GVN::createMemoryExpression(LoadInst *LI) {
  MemoryExpression *E = new MemoryExpression(LI);
  E->setType(LI->getType());
  // Need opcodes to match on loads and store
  E->setOpcode(0);
  Value *Operand = lookupOperandLeader(LI->getPointerOperand());
  E->varargs.push_back(Operand);
  return E;
}

Expression *GVN::createMemoryExpression(StoreInst *SI) {
  MemoryExpression *E = new MemoryExpression(SI);
  E->setType(SI->getType());
  // Need opcodes to match on loads and store
  E->setOpcode(0);
  Value *Operand = lookupOperandLeader(SI->getPointerOperand());
  E->varargs.push_back(Operand);
  // TODO: Set store value here!
  return E;
}

Expression *GVN::performSymbolicStoreEvaluation(Instruction *I, BasicBlock *B) {
  StoreInst *SI = cast<StoreInst>(I);
  Expression *E = createMemoryExpression(SI);
  return E;
}

Expression *GVN::performSymbolicLoadEvaluation(Instruction *I, BasicBlock *B) {
  LoadInst *LI = cast<LoadInst>(I);
  if (!LI->isSimple())
    return NULL;
  Expression *E = createMemoryExpression(LI);
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
}

// performSymbolicPHIEvaluation - Evaluate PHI nodes symbolically, and
// create an expression result 
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
    DEBUG(dbgs() << "Simplified PHI node " << I << " to " << *AllSameValue << "\n");

    delete E;
    return performSymbolicEvaluation(AllSameValue, B);
  }
  return E;
}


/// performSymbolicEvaluation - Substitute and symbolize the value
/// before value numbering 
Expression *GVN::performSymbolicEvaluation(Value *V, BasicBlock *B) {
  Expression *E = NULL;
  if (Constant *C = dyn_cast<Constant>(V))
    E = createConstantExpression(C);
  else if (isa<Argument>(V) || isa<GlobalVariable>(V)) {
    E = createVariableExpression(V);
  } else {
    //TODO: memory intrinsics
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
      if (noLoads)
        E = NULL;
      else
        E = performSymbolicCallEvaluation(I, B);
      break;
    case Instruction::Store:
      if (noLoads)
        E = NULL;
      else
        E = performSymbolicStoreEvaluation(I, B);
      break;
    case Instruction::Load:
      if (noLoads)
        E = NULL;
      else
        E = performSymbolicLoadEvaluation(I, B);
      break;
    case Instruction::BitCast: {
      // Pointer bitcasts are noops, we can just make them out of whole
      // cloth if we need to. 
      if (I->getType()->isPointerTy()){
        if (Instruction *I0 = dyn_cast<Instruction>(I->getOperand(0)))
          return performSymbolicEvaluation(I0, I0->getParent());
        else
          return performSymbolicEvaluation(I->getOperand(0), B);
      }
      E = createExpression(I);
    }
      break;

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
  if (!E)
    return NULL;

  if (isa<ConstantExpression>(E) || isa<VariableExpression>(E))
    E = uniquifyExpression(E);
  expressionToDelete.insert(E);
  return E;
}


/// replaceAllDominatedUsesWith - Replace all uses of 'From' with 'To'
/// if the use is dominated by the given basic block.  Returns the
/// number of uses that were replaced.
unsigned GVN::replaceAllDominatedUsesWith(Value *From, Value *To,
                                          BasicBlock *Root) {
  unsigned Count = 0;
  for (Value::use_iterator UI = From->use_begin(), UE = From->use_end();
       UI != UE; ) {
    Use &U = (UI++).getUse();

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
        touchedInstructions.insert(I);
      DEBUG(dbgs() << "Equality propagation replacing " << *From << " with " << *To << " in " << *(U.getUser()) << "\n");
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

bool GVN::propagateEquality(Value *LHS, Value *RHS, BasicBlock *Root) {
  SmallVector<std::pair<Value*, Value*>, 4> Worklist;
  Worklist.push_back(std::make_pair(LHS, RHS));
  bool Changed = false;

  while (!Worklist.empty()) {
    std::pair<Value*, Value*> Item = Worklist.pop_back_val();
    LHS = Item.first; RHS = Item.second;

    if (LHS == RHS) continue;
    assert(LHS->getType() == RHS->getType() && "Equality but unequal types!");

    // Don't try to propagate equalities between constants.
    if (isa<Constant>(LHS) && isa<Constant>(RHS)) continue;

    // Prefer a constant on the right-hand side, or an Argument if no constants.
    if (isa<Constant>(LHS) || (isa<Argument>(LHS) && !isa<Constant>(RHS)))
      std::swap(LHS, RHS);
    assert((isa<Argument>(LHS) || isa<Instruction>(LHS)) && "Unexpected value!");
    assert((!isa<Instruction>(RHS) ||
            DT->properlyDominates(cast<Instruction>(RHS)->getParent(), Root)) &&
           "Instruction doesn't dominate scope!");

    // If value numbering later deduces that an instruction in the
    // scope is equal to 'LHS' then ensure it will be turned into
    // 'RHS' within the scope Root.
    CongruenceClass *CC = valueToClass[LHS];
    assert (CC && "Should have found a congruence class");
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
      Expression *E = createCmpExpression(Cmp->getOpcode(), Cmp->getType(), NotPred, Op0, Op1);

      // We cannot directly propagate into the congruence class members
      // unless it is a singleton class.
      // This is because at least one of the members may later get
      // split out of the class.

      CongruenceClass *CC = expressionToClass.lookup(E);
      delete E;
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

/// performCongruenceFinding - Perform congruence finding on a given
/// value numbering expression
void GVN::performCongruenceFinding(Value *V, BasicBlock *BB, Expression *E) {
  // This is guaranteed to return something, since it will at least find INITIAL
  CongruenceClass *VClass = valueToClass[V];
  assert (VClass && "Should have found a vclass");
  // Dead classes should have been eliminated from the mapping
  assert (!VClass->dead && "Found a dead class");

  //TODO(dannyb): Double check algorithm where we are ignoring copy
  //check of "if e is a SSA variable", as LLVM has no copies.

  CongruenceClass *EClass;
  // Expressions we can't symbolize are always in their own unique
  // congruence class
  if (E == NULL) {
    // We may have already made a unique class
    if (VClass->members.size() != 1 || VClass->leader != V) {
      CongruenceClass *NewClass = createCongruenceClass(V, NULL);
      // We should always be adding the member in the below code
      EClass = NewClass;
      DEBUG(dbgs() << "Created new congruence class for " << V << " due to NULL expression\n");
    } else {
      EClass = VClass;
    }
  } else {

    ExpressionClassMap *lookupMap = isa<MemoryExpression>(E) ? &memoryExpressionToClass : &expressionToClass;
    std::pair<ExpressionClassMap::iterator, bool> lookupResult =
      lookupMap->insert(std::make_pair(E, (CongruenceClass*)NULL));

    // If it's not in the value table, create a new congruence class
    if (lookupResult.second) {
      CongruenceClass *NewClass = createCongruenceClass(NULL, E);
      ExpressionClassMap::iterator place = lookupResult.first;
      place->second = NewClass;

      // Constants and variables should always be made the leader
      if (ConstantExpression *CE = dyn_cast<ConstantExpression>(E))
        NewClass->leader = CE->getConstantValue();
      else if (VariableExpression *VE = dyn_cast<VariableExpression>(E))
        NewClass->leader = VE->getVariableValue();
      else if (MemoryExpression *ME = dyn_cast<MemoryExpression>(E)) {
        if (ME->isStore())
          NewClass->leader = ME->getStoreInst()->getValueOperand();
        else
          NewClass->leader = V;
      } else
        NewClass->leader = V;

      EClass = NewClass;
      DEBUG(dbgs() << "Created new congruence class for " << V <<
            " using expression " << *E << " at " << NewClass->id << "\n");
    } else {      
      EClass = lookupResult.first->second;
      assert(EClass && "Somehow don't have an eclass");

      assert (!EClass->dead && "We accidentally looked up a dead class");
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
      VClass->members.erase(std::make_pair(V, BB));
      // assert(std::find(EClass->members.begin(), EClass->members.end(), V) == EClass->members.end() && "Tried to add something to members twice!");
      EClass->members.insert(std::make_pair(V, BB));
      valueToClass[V] = EClass;
      // See if we destroyed the class or need to swap leaders
      if (VClass->members.empty() && VClass != InitialClass) {
        if (VClass->expression) {
          VClass->dead = true;
          // TODO: I think this may be wrong.
          // I think we should be keeping track of the expression for
          // each instruction, not for each class, and erase the old
          // expression for the instruction when the class dies.
          expressionToClass.erase(VClass->expression);
          memoryExpressionToClass.erase(VClass->expression);
        }
        // delete VClass;
      } else if (VClass->leader == V) {
        // TODO: Check what happens if expression represented the leader
        VClass->leader = VClass->members.begin()->first;
        for (CongruenceClass::MemberSet::iterator LI = VClass->members.begin(),
               LE = VClass->members.end();
             LI != LE; ++LI) {
          if (Instruction *I = dyn_cast<Instruction>(LI->first))
            touchedInstructions.insert(I);
          changedValues.insert(LI->first);
        }
      }
    }
    markUsersTouched(V);
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
      for (BasicBlock::iterator BI = To->begin(), BE = To->end();
           BI != BE; ++BI)
        touchedInstructions.insert(BI);
    } else {
      DEBUG(dbgs() << "Block " << getBlockName(To) << " was reachable, but new edge to it found\n");
      // We've made an edge reachable to an existing block, which may
      // impact predicates.
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
}


//  processOutgoingEdges - Process the outgoing edges of a block for reachability.
void GVN::processOutgoingEdges(TerminatorInst *TI) {
  // Evaluate Reachability of terminator instruction
  // Conditional branch
  BranchInst *BR;
  if ((BR = dyn_cast<BranchInst>(TI)) && BR->isConditional()) {
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
      BasicBlock *Parent = BR->getParent();
      if (isOnlyReachableViaThisEdge(Parent, TrueSucc, DT))
        propagateEquality(Cond,
                          ConstantInt::getTrue(TrueSucc->getContext()),
                          TrueSucc);

      if (isOnlyReachableViaThisEdge(Parent, FalseSucc, DT))
        propagateEquality(Cond,
                          ConstantInt::getFalse(FalseSucc->getContext()),
                          FalseSucc);
      updateReachableEdge(TI->getParent(), TrueSucc);
      updateReachableEdge(TI->getParent(), FalseSucc);
    }
  } else {
    // For switches, propagate the case values into the case destinations.
    if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
      Value *SwitchCond = SI->getCondition();
      BasicBlock *Parent = SI->getParent();
      for (SwitchInst::CaseIt i = SI->case_begin(), e = SI->case_end();
           i != e; ++i) {
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
void GVN::propagateChangeInEdge(BasicBlock *Dest) {
  // The algorithm states that you only need to touch blocks that are
  // confluence nodes.  I also can't see why you would need to touch
  // any instructions that aren't PHI nodes.  Because we don't use
  // predicates right now, they are the ones whose value could have
  // changed as a result of a new edge becoming live, and any changes
  // to their value should propagate appropriately through the rest of
  // the block.
  DomTreeNode *DTN = DT->getNode(Dest);
  //TODO(dannyb): This differs slightly from the published algorithm,
  //verify it The published algorithm touches all instructions, we
  //only touch the phi nodes.  This is because there should be no
  //other values that can *directly* change as a result of edge
  //reachability. If the phi node ends up changing congruence classes,
  //the users will be marked as touched anyway.  If we moved to using
  //value inferencing, there are cases we may need to touch more than
  //phi nodes.
  for (df_iterator<DomTreeNode*> DI = df_begin(DTN), DE = df_end(DTN); DI != DE; ++DI) {
    BasicBlock *B = DI->getBlock();
    if (!B->getUniquePredecessor()) {
      for (BasicBlock::iterator BI = B->begin(), BE = B->end();
           BI != BE; ++BI) {
        if (!isa<PHINode>(BI))
          break;
        touchedInstructions.insert(BI);
      }
    }
  }
}


static void patchReplacementInstruction(Value *Repl, Instruction *I) {
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
    SmallVector<std::pair<unsigned, MDNode*>, 4> Metadata;
    ReplInst->getAllMetadataOtherThanDebugLoc(Metadata);
    for (int i = 0, n = Metadata.size(); i < n; ++i) {
      unsigned Kind = Metadata[i].first;
      MDNode *IMD = I->getMetadata(Kind);
      MDNode *ReplMD = Metadata[i].second;
      switch(Kind) {
      default:
        ReplInst->setMetadata(Kind, NULL); // Remove unknown metadata
        break;
      case LLVMContext::MD_dbg:
        llvm_unreachable("getAllMetadataOtherThanDebugLoc returned a MD_dbg");
      case LLVMContext::MD_tbaa:
        ReplInst->setMetadata(Kind, MDNode::getMostGenericTBAA(IMD, ReplMD));
        break;
      case LLVMContext::MD_range:
        ReplInst->setMetadata(Kind, MDNode::getMostGenericRange(IMD, ReplMD));
        break;
      case LLVMContext::MD_prof:
        llvm_unreachable("MD_prof in a non terminator instruction");
        break;
      case LLVMContext::MD_fpmath:
        ReplInst->setMetadata(Kind, MDNode::getMostGenericFPMath(IMD, ReplMD));
        break;
      }
    }
  }
}

static void patchAndReplaceAllUsesWith(Value *Repl, Instruction *I) {
  patchReplacementInstruction(Repl, I);
  I->replaceAllUsesWith(Repl);
}

void GVN::replaceInstruction(Instruction *I, Value *V, CongruenceClass *ReplClass) {
  assert (ReplClass->leader != I && "About to accidentally remove our leader");
  BasicBlock *IBB = I->getParent();
  if (V->getType() != I->getType()) {
    Instruction *insertPlace = I;
    if (isa<PHINode>(I))
      insertPlace = IBB->getFirstNonPHI();
    V = CoerceAvailableValueToLoadType(V, I->getType(), insertPlace, *TD);
    assert(V && "Should have been able to coerce types!");
  }
  DEBUG(dbgs() << "Replacing " << *I << " with " << *V << "\n");
  patchAndReplaceAllUsesWith(V, I);
  // Remove the old instruction from the class member list, if it is there, so the
  // member size is correct for PRE.
  ReplClass->members.erase(std::make_pair(I, IBB));
  // We save the actual erasing to avoid invalidating memory
  // dependencies until we are done with everything.
  UpdateMemDepInfo(MD, I, V);
  markInstructionForDeletion(I);
}

namespace {

struct AvailableValueInBlock {
  /// BB - The basic block in question.
  BasicBlock *BB;
  enum ValType {
    SimpleVal,  // A simple offsetted value that is accessed.
    LoadVal,    // A value produced by a load.
    MemIntrin   // A memory intrinsic which is loaded from.
  };

  /// V - The value that is live out of the block.
  PointerIntPair<Value *, 2, ValType> Val;

  /// Offset - The byte offset in Val that is interesting for the load query.
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

  bool isSimpleValue() const { return Val.getInt() == SimpleVal; }
  bool isCoercedLoadValue() const { return Val.getInt() == LoadVal; }
  bool isMemIntrinValue() const { return Val.getInt() == MemIntrin; }

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

  /// MaterializeAdjustedValue - Emit code into this block to adjust
  /// the value defined here to the specified type.  This handles
  /// various coercion cases.
  Value *MaterializeAdjustedValue(Type *LoadTy, GVN &gvn) const {
    Value *Res;
    if (isSimpleValue()) {
      Res = getSimpleValue();
      if (Res->getType() != LoadTy) {
        const TargetData *TD = gvn.getTargetData();
        assert(TD && "Need target data to handle type mismatch case");
        Res = gvn.GetStoreValueForLoad(Res, Offset, LoadTy, BB->getTerminator(),
                                       *TD);

        DEBUG(dbgs() << "GVN COERCED NONLOCAL VAL:\nOffset: " << Offset << "  "
                     << *getSimpleValue() << '\n'
                     << *Res << '\n' << "\n\n\n");
      }
    } else if (isCoercedLoadValue()) {
      LoadInst *Load = getCoercedLoadValue();
      if (Load->getType() == LoadTy && Offset == 0) {
        Res = Load;
      } else {
        Res = gvn.GetLoadValueForLoad(Load, Offset, LoadTy, BB->getTerminator(),
                                      *TD);

        DEBUG(dbgs() << "GVN COERCED NONLOCAL LOAD:\nOffset: " << Offset << "  "
                     << *getCoercedLoadValue() << '\n'
                     << *Res << '\n' << "\n\n\n");
      }
    } else {
      const TargetData *TD = gvn.getTargetData();
      assert(TD && "Need target data to handle type mismatch case");
      Res = gvn.GetMemInstValueForLoad(getMemIntrinValue(), Offset,
                                       LoadTy, BB->getTerminator(), *TD);
      DEBUG(dbgs() << "GVN COERCED NONLOCAL MEM INTRIN:\nOffset: " << Offset
                   << "  " << *getMemIntrinValue() << '\n'
                   << *Res << '\n' << "\n\n\n");
    }
    return Res;
  }
};

} // end anonymous namespace


/// ConstructSSAForLoadSet - Given a set of loads specified by
/// ValuesPerBlock, construct SSA form, allowing us to eliminate LI.
/// This returns the value that should be used at LI's definition
/// site.
static Value *ConstructSSAForLoadSet(LoadInst *LI,
                                     SmallVectorImpl<AvailableValueInBlock> &ValuesPerBlock,
                                     GVN &gvn) {
  // Check for the fully redundant, dominating load case.  In this case, we can
  // just use the dominating value directly.
  if (ValuesPerBlock.size() == 1 &&
      gvn.getDominatorTree().properlyDominates(ValuesPerBlock[0].BB,
                                               LI->getParent()))
    return ValuesPerBlock[0].MaterializeAdjustedValue(LI->getType(), gvn);

  // Otherwise, we have to construct SSA form.
  SmallVector<PHINode*, 8> NewPHIs;
  SSAUpdater SSAUpdate(&NewPHIs);
  SSAUpdate.Initialize(LI->getType(), LI->getName());

  Type *LoadTy = LI->getType();

  for (unsigned i = 0, e = ValuesPerBlock.size(); i != e; ++i) {
    const AvailableValueInBlock &AV = ValuesPerBlock[i];
    BasicBlock *BB = AV.BB;

    if (SSAUpdate.HasValueForBlock(BB))
      continue;

    SSAUpdate.AddAvailableValue(BB, AV.MaterializeAdjustedValue(LoadTy, gvn));
  }

  // Perform PHI construction.
  Value *V = SSAUpdate.GetValueInMiddleOfBlock(LI->getParent());

  for (unsigned i = 0, e = NewPHIs.size(); i != e; ++i)
    gvn.handleNewInstruction(NewPHIs[i]);

  // If new PHI nodes were created, notify alias analysis.
  if (V->getType()->isPointerTy()) {
    AliasAnalysis *AA = gvn.getAliasAnalysis();

    for (unsigned i = 0, e = NewPHIs.size(); i != e; ++i)
      AA->copyValue(LI, NewPHIs[i]);

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

static bool isLifetimeStart(const Instruction *Inst) {
  if (const IntrinsicInst* II = dyn_cast<IntrinsicInst>(Inst))
    return II->getIntrinsicID() == Intrinsic::lifetime_start;
  return false;
}

/// IsValueFullyAvailableInBlock - Return true if we can prove that the value
/// we're analyzing is fully available in the specified block.  As we go, keep
/// track of which blocks we know are fully alive in FullyAvailableBlocks.  This
/// map is actually a tri-state map with the following values:
///   0) we know the block *is not* fully available.
///   1) we know the block *is* fully available.
///   2) we do not know whether the block is fully available or not, but we are
///      currently speculating that it will be.
///   3) we are speculating for this block and have used that to speculate for
///      other blocks.
static bool IsValueFullyAvailableInBlock(BasicBlock *BB,
                            DenseMap<BasicBlock*, char> &FullyAvailableBlocks,
                            uint32_t RecurseDepth) {
  if (RecurseDepth > MaxRecurseDepth)
    return false;

  // Optimistically assume that the block is fully available and check to see
  // if we already know about this block in one lookup.
  std::pair<DenseMap<BasicBlock*, char>::iterator, char> IV =
    FullyAvailableBlocks.insert(std::make_pair(BB, 2));

  // If the entry already existed for this block, return the precomputed value.
  if (!IV.second) {
    // If this is a speculative "available" value, mark it as being used for
    // speculation of other blocks.
    if (IV.first->second == 2)
      IV.first->second = 3;
    return IV.first->second != 0;
  }

  // Otherwise, see if it is fully available in all predecessors.
  pred_iterator PI = pred_begin(BB), PE = pred_end(BB);

  // If this block has no predecessors, it isn't live-in here.
  if (PI == PE)
    goto SpeculationFailure;

  for (; PI != PE; ++PI)
    // If the value isn't fully available in one of our predecessors, then it
    // isn't fully available in this block either.  Undo our previous
    // optimistic assumption and bail out.
    if (!IsValueFullyAvailableInBlock(*PI, FullyAvailableBlocks,RecurseDepth+1))
      goto SpeculationFailure;

  return true;

// SpeculationFailure - If we get here, we found out that this is not,
// after all, a fully-available block.  We have a problem if we
// speculated on this and used the speculation to mark other blocks as
// available.
SpeculationFailure:
  char &BBVal = FullyAvailableBlocks[BB];

  // If we didn't speculate on this, just return with it set to false.
  if (BBVal == 2) {
    BBVal = 0;
    return false;
  }

  // If we did speculate on this value, we could have blocks set to 1
  // that are incorrect.  Walk the (transitive) successors of this
  // block and mark them as 0 if set to one.
  SmallVector<BasicBlock*, 32> BBWorklist;
  BBWorklist.push_back(BB);

  do {
    BasicBlock *Entry = BBWorklist.pop_back_val();
    // Note that this sets blocks to 0 (unavailable) if they happen to not
    // already be in FullyAvailableBlocks.  This is safe.
    char &EntryVal = FullyAvailableBlocks[Entry];
    if (EntryVal == 0) continue;  // Already unavailable.

    // Mark as unavailable.
    EntryVal = 0;

    for (succ_iterator I = succ_begin(Entry), E = succ_end(Entry); I != E; ++I)
      BBWorklist.push_back(*I);
  } while (!BBWorklist.empty());

  return false;
}

void GVN::markUsersTouched(Value *V) {
  // Now mark the users as touched
  for (Value::use_iterator UI = V->use_begin(), UE = V->use_end();
       UI != UE; ++UI) {
    Instruction *User = cast<Instruction>(*UI);
    touchedInstructions.insert(User);
  }
}

void GVN::handleNewInstruction(Instruction *I) {
  valueToClass[I] = InitialClass;
  touchedInstructions.insert(I);
  InitialClass->members.insert(std::make_pair(I, I->getParent()));
}

/// processNonLocalLoad - Attempt to eliminate a load whose dependencies are
/// non-local by performing PHI construction.
// FIXME: This is one of the last vestiges of the old GVN.
// Because it performs a weird optimization not subsumed by anything
// else, we run it on singleton loads.  Realistically, this should be
// subsumed into PRE or elimination directly.  As it is, it is quite
// expensive, because it re-gets all the non-local pointer dependencies. 

bool GVN::processNonLocalLoad(LoadInst *LI) {

  // Find the non-local dependencies of the load.
  SmallVector<NonLocalDepResult, 64> Deps;
  AliasAnalysis::Location Loc = AA->getLocation(LI);

  Loc.Ptr = lookupOperandLeader(LI->getPointerOperand());
  MD->getNonLocalPointerDependency(Loc, true, LI->getParent(), Deps);
  //DEBUG(dbgs() << "INVESTIGATING NONLOCAL LOAD: "
  //             << Deps.size() << *LI << '\n');

  // If we had to process more than one hundred blocks to find the
  // dependencies, this load isn't worth worrying about.  Optimizing
  // it will be too expensive.
  unsigned NumDeps = Deps.size();
  if (NumDeps > 100)
    return false;

  // If we had a phi translation failure, we'll have a single entry which is a
  // clobber in the current block.  Reject this early.
  if (NumDeps == 1 &&
      !Deps[0].getResult().isDef() && !Deps[0].getResult().isClobber()) {
    DEBUG(
      dbgs() << "GVN: non-local load ";
      WriteAsOperand(dbgs(), LI);
      dbgs() << " has unknown dependencies\n";
    );
    return false;
  }

  // Filter out useless results (non-locals, etc).  Keep track of the blocks
  // where we have a value available in repl, also keep track of whether we see
  // dependencies that produce an unknown value for the load (such as a call
  // that could potentially clobber the load).
  SmallVector<AvailableValueInBlock, 64> ValuesPerBlock;
  SmallVector<BasicBlock*, 64> UnavailableBlocks;

  for (unsigned i = 0, e = NumDeps; i != e; ++i) {
    BasicBlock *DepBB = Deps[i].getBB();
    MemDepResult DepInfo = Deps[i].getResult();

    if (!reachableBlocks.count(DepBB))
      DEBUG(dbgs() << "Skipping dependency in unreachable block\n");

    if (!DepInfo.isDef() && !DepInfo.isClobber()) {
      UnavailableBlocks.push_back(DepBB);
      continue;
    }

    if (DepInfo.isClobber()) {
      // The address being loaded in this non-local block may not be the same as
      // the pointer operand of the load if PHI translation occurs.  Make sure
      // to consider the right address.
      Value *Address = Deps[i].getAddress();

      // If the dependence is to a store that writes to a superset of the bits
      // read by the load, we can extract the bits we need for the load from the
      // stored value.
      if (StoreInst *DepSI = dyn_cast<StoreInst>(DepInfo.getInst())) {
        if (TD && Address) {
          int Offset = AnalyzeLoadFromClobberingStore(LI->getType(), Address,
                                                      DepSI, *TD);
          if (Offset != -1) {
            ValuesPerBlock.push_back(AvailableValueInBlock::get(DepBB,
                                                       DepSI->getValueOperand(),
                                                                Offset));
            continue;
          }
        }
      }

      // Check to see if we have something like this:
      //    load i32* P
      //    load i8* (P+1)
      // if we have this, replace the later with an extraction from the former.
      if (LoadInst *DepLI = dyn_cast<LoadInst>(DepInfo.getInst())) {
        // If this is a clobber and L is the first instruction in its block, then
        // we have the first instruction in the entry block.
        if (DepLI != LI && Address && TD) {
          int Offset = AnalyzeLoadFromClobberingLoad(LI->getType(),
                                                     LI->getPointerOperand(),
                                                     DepLI, *TD);

          if (Offset != -1) {
            ValuesPerBlock.push_back(AvailableValueInBlock::getLoad(DepBB,DepLI,
                                                                    Offset));
            continue;
          }
        }
      }

      // If the clobbering value is a memset/memcpy/memmove, see if we can
      // forward a value on from it.
      if (MemIntrinsic *DepMI = dyn_cast<MemIntrinsic>(DepInfo.getInst())) {
        if (TD && Address) {
          int Offset = AnalyzeLoadFromClobberingMemInst(LI->getType(), Address,
                                                        DepMI, *TD);
          if (Offset != -1) {
            ValuesPerBlock.push_back(AvailableValueInBlock::getMI(DepBB, DepMI,
                                                                  Offset));
            continue;
          }
        }
      }

      UnavailableBlocks.push_back(DepBB);
      continue;
    }

    // DepInfo.isDef() here

    Instruction *DepInst = DepInfo.getInst();

    // Loading the allocation -> undef.
    if (isa<AllocaInst>(DepInst) || isMallocLikeFn(DepInst) ||
        // Loading immediately after lifetime begin -> undef.
        isLifetimeStart(DepInst)) {
      ValuesPerBlock.push_back(AvailableValueInBlock::get(DepBB,
                                             UndefValue::get(LI->getType())));
      continue;
    }

    if (StoreInst *S = dyn_cast<StoreInst>(DepInst)) {
      // Reject loads and stores that are to the same address but are of
      // different types if we have to.
      if (S->getValueOperand()->getType() != LI->getType()) {
        // If the stored value is larger or equal to the loaded value, we can
        // reuse it.
        if (TD == 0 || !CanCoerceMustAliasedValueToLoad(S->getValueOperand(),
                                                        LI->getType(), *TD)) {
          UnavailableBlocks.push_back(DepBB);
          continue;
        }
      }

      ValuesPerBlock.push_back(AvailableValueInBlock::get(DepBB,
                                                         S->getValueOperand()));
      continue;
    }

    if (LoadInst *LD = dyn_cast<LoadInst>(DepInst)) {
      // If the types mismatch and we can't handle it, reject reuse of the load.
      if (LD->getType() != LI->getType()) {
        // If the stored value is larger or equal to the loaded value, we can
        // reuse it.
        if (TD == 0 || !CanCoerceMustAliasedValueToLoad(LD, LI->getType(),*TD)){
          UnavailableBlocks.push_back(DepBB);
          continue;
        }
      }
      ValuesPerBlock.push_back(AvailableValueInBlock::getLoad(DepBB, LD));
      continue;
    }

    UnavailableBlocks.push_back(DepBB);
    continue;
  }

  // If we have no predecessors that produce a known value for this load, exit
  // early.
  if (ValuesPerBlock.empty()) return false;

  // If all of the instructions we depend on produce a known value for this
  // load, then it is fully redundant and we can use PHI insertion to compute
  // its value.  Insert PHIs and remove the fully redundant value now.
  if (UnavailableBlocks.empty()) {
    DEBUG(dbgs() << "GVN REMOVING NONLOCAL LOAD: " << *LI << '\n');

    // Perform PHI construction.
    Value *V = ConstructSSAForLoadSet(LI, ValuesPerBlock, *this);
    markUsersTouched(LI);
    LI->replaceAllUsesWith(V);
    if (isa<PHINode>(V))
      V->takeName(LI);
    UpdateMemDepInfo(MD, LI, V);
    markInstructionForDeletion(LI);
    ++NumGVNLoad;
    return true;
  }

  if (!EnablePRE || !EnableLoadPRE)
    return false;

  // Okay, we have *some* definitions of the value.  This means that the value
  // is available in some of our (transitive) predecessors.  Lets think about
  // doing PRE of this load.  This will involve inserting a new load into the
  // predecessor when it's not available.  We could do this in general, but
  // prefer to not increase code size.  As such, we only do this when we know
  // that we only have to insert *one* load (which means we're basically moving
  // the load, not inserting a new one).

  SmallPtrSet<BasicBlock *, 4> Blockers;
  for (unsigned i = 0, e = UnavailableBlocks.size(); i != e; ++i)
    Blockers.insert(UnavailableBlocks[i]);

  // Let's find the first basic block with more than one predecessor.  Walk
  // backwards through predecessors if needed.
  BasicBlock *LoadBB = LI->getParent();
  BasicBlock *TmpBB = LoadBB;

  bool isSinglePred = false;
  bool allSingleSucc = true;
  while (TmpBB->getSinglePredecessor()) {
    isSinglePred = true;
    TmpBB = TmpBB->getSinglePredecessor();
    if (TmpBB == LoadBB) // Infinite (unreachable) loop.
      return false;
    if (Blockers.count(TmpBB))
      return false;

    // If any of these blocks has more than one successor (i.e. if the edge we
    // just traversed was critical), then there are other paths through this
    // block along which the load may not be anticipated.  Hoisting the load
    // above this block would be adding the load to execution paths along
    // which it was not previously executed.
    if (TmpBB->getTerminator()->getNumSuccessors() != 1)
      return false;
  }

  assert(TmpBB);
  LoadBB = TmpBB;

  // FIXME: It is extremely unclear what this loop is doing, other than
  // artificially restricting loadpre.
  if (isSinglePred) {
    bool isHot = false;
    for (unsigned i = 0, e = ValuesPerBlock.size(); i != e; ++i) {
      const AvailableValueInBlock &AV = ValuesPerBlock[i];
      if (AV.isSimpleValue())
        // "Hot" Instruction is in some loop (because it dominates its dep.
        // instruction).
        if (Instruction *I = dyn_cast<Instruction>(AV.getSimpleValue()))
          if (DT->dominates(LI, I)) {
            isHot = true;
            break;
          }
    }

    // We are interested only in "hot" instructions. We don't want to do any
    // mis-optimizations here.
    if (!isHot)
      return false;
  }

  // Check to see how many predecessors have the loaded value fully
  // available.
  DenseMap<BasicBlock*, Value*> PredLoads;
  DenseMap<BasicBlock*, char> FullyAvailableBlocks;
  for (unsigned i = 0, e = ValuesPerBlock.size(); i != e; ++i)
    FullyAvailableBlocks[ValuesPerBlock[i].BB] = true;
  for (unsigned i = 0, e = UnavailableBlocks.size(); i != e; ++i)
    FullyAvailableBlocks[UnavailableBlocks[i]] = false;

  SmallVector<std::pair<TerminatorInst*, unsigned>, 4> NeedToSplit;
  for (pred_iterator PI = pred_begin(LoadBB), E = pred_end(LoadBB);
       PI != E; ++PI) {
    BasicBlock *Pred = *PI;
    if (IsValueFullyAvailableInBlock(Pred, FullyAvailableBlocks, 0)) {
      continue;
    }
    PredLoads[Pred] = 0;

    if (Pred->getTerminator()->getNumSuccessors() != 1) {
      if (isa<IndirectBrInst>(Pred->getTerminator())) {
        DEBUG(dbgs() << "COULD NOT PRE LOAD BECAUSE OF INDBR CRITICAL EDGE '"
              << Pred->getName() << "': " << *LI << '\n');
        return false;
      }

      if (LoadBB->isLandingPad()) {
        DEBUG(dbgs()
              << "COULD NOT PRE LOAD BECAUSE OF LANDING PAD CRITICAL EDGE '"
              << Pred->getName() << "': " << *LI << '\n');
        return false;
      }

      unsigned SuccNum = GetSuccessorNumber(Pred, LoadBB);
      NeedToSplit.push_back(std::make_pair(Pred->getTerminator(), SuccNum));
    }
  }

  if (!NeedToSplit.empty()) {
    toSplit.append(NeedToSplit.begin(), NeedToSplit.end());
    return false;
  }

  // Decide whether PRE is profitable for this load.
  unsigned NumUnavailablePreds = PredLoads.size();
  assert(NumUnavailablePreds != 0 &&
         "Fully available value should be eliminated above!");

  // If this load is unavailable in multiple predecessors, reject it.
  // FIXME: If we could restructure the CFG, we could make a common pred with
  // all the preds that don't have an available LI and insert a new load into
  // that one block.
  if (NumUnavailablePreds != 1)
      return false;

  // Check if the load can safely be moved to all the unavailable predecessors.
  bool CanDoPRE = true;
  SmallVector<Instruction*, 8> NewInsts;
  for (DenseMap<BasicBlock*, Value*>::iterator I = PredLoads.begin(),
         E = PredLoads.end(); I != E; ++I) {
    BasicBlock *UnavailablePred = I->first;

    // Do PHI translation to get its value in the predecessor if necessary.  The
    // returned pointer (if non-null) is guaranteed to dominate UnavailablePred.

    // If all preds have a single successor, then we know it is safe to insert
    // the load on the pred (?!?), so we can insert code to materialize the
    // pointer if it is not available.
    PHITransAddr Address(LI->getPointerOperand(), TD);
    Value *LoadPtr = 0;
    if (allSingleSucc) {
      LoadPtr = Address.PHITranslateWithInsertion(LoadBB, UnavailablePred,
                                                  *DT, NewInsts);
    } else {
      Address.PHITranslateValue(LoadBB, UnavailablePred, DT);
      LoadPtr = Address.getAddr();
    }

    // If we couldn't find or insert a computation of this phi translated value,
    // we fail PRE.
    if (LoadPtr == 0) {
      DEBUG(dbgs() << "COULDN'T INSERT PHI TRANSLATED VALUE OF: "
            << *LI->getPointerOperand() << "\n");
      CanDoPRE = false;
      break;
    }

    // Make sure it is valid to move this load here.  We have to watch out for:
    //  @1 = getelementptr (i8* p, ...
    //  test p and branch if == 0
    //  load @1
    // It is valid to have the getelementptr before the test, even if p can
    // be 0, as getelementptr only does address arithmetic.
    // If we are not pushing the value through any multiple-successor blocks
    // we do not have this case.  Otherwise, check that the load is safe to
    // put anywhere; this can be improved, but should be conservatively safe.
    if (!allSingleSucc &&
        // FIXME: REEVALUTE THIS.
        !isSafeToLoadUnconditionally(LoadPtr,
                                     UnavailablePred->getTerminator(),
                                     LI->getAlignment(), TD)) {
      CanDoPRE = false;
      break;
    }

    I->second = LoadPtr;
  }

  if (!CanDoPRE) {
    while (!NewInsts.empty()) {
      Instruction *I = NewInsts.pop_back_val();
      UpdateMemDepInfo(MD, I, NULL);
      markInstructionForDeletion(I);
    }
    return false;
  }

  // Okay, we can eliminate this load by inserting a reload in the predecessor
  // and using PHI construction to get the value in the other predecessors, do
  // it.
  DEBUG(dbgs() << "GVN REMOVING PRE LOAD: " << *LI << '\n');
  DEBUG(if (!NewInsts.empty())
          dbgs() << "INSERTED " << NewInsts.size() << " INSTS: "
                 << *NewInsts.back() << '\n');

  // Assign value numbers to the new instructions.
  for (unsigned i = 0, e = NewInsts.size(); i != e; ++i) {
    handleNewInstruction(NewInsts[i]);
  }

  for (DenseMap<BasicBlock*, Value*>::iterator I = PredLoads.begin(),
         E = PredLoads.end(); I != E; ++I) {
    BasicBlock *UnavailablePred = I->first;
    Value *LoadPtr = I->second;

    Instruction *NewLoad = new LoadInst(LoadPtr, LI->getName()+".pre", false,
                                        LI->getAlignment(),
                                        UnavailablePred->getTerminator());

    // Transfer the old load's TBAA tag to the new load.
    if (MDNode *Tag = LI->getMetadata(LLVMContext::MD_tbaa))
      NewLoad->setMetadata(LLVMContext::MD_tbaa, Tag);

    // Transfer DebugLoc.
    NewLoad->setDebugLoc(LI->getDebugLoc());

    // Add the newly created load.
    ValuesPerBlock.push_back(AvailableValueInBlock::get(UnavailablePred,
                                                        NewLoad));
    MD->invalidateCachedPointerInfo(LoadPtr);
    DEBUG(dbgs() << "GVN INSERTED " << *NewLoad << '\n');
    handleNewInstruction(NewLoad);
  }

  // Perform PHI construction.
  Value *V = ConstructSSAForLoadSet(LI, ValuesPerBlock, *this);
  markUsersTouched(LI);

  LI->replaceAllUsesWith(V);

  if (isa<PHINode>(V))
    V->takeName(LI);
  UpdateMemDepInfo(MD, LI, V);
  if (Instruction *I = dyn_cast<Instruction>(V))
    handleNewInstruction(I);
  markInstructionForDeletion(LI);
  ++NumPRELoad;
  return true;
}

static void DeleteInstructionInBlock(BasicBlock *BB) {
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


/// splitCriticalEdges - Split critical edges found during the previous
/// iteration that may enable further optimization.
bool GVN::splitCriticalEdges() {
  if (toSplit.empty())
    return false;
  do {
    std::pair<TerminatorInst*, unsigned> Edge = toSplit.pop_back_val();
    BasicBlock *NewBB = SplitCriticalEdge(Edge.first, Edge.second, this);
    reachableBlocks.insert(NewBB);
  } while (!toSplit.empty());
  if (MD) MD->invalidateCachedPredecessors();
  return true;
}


// Find a leader for OP in BB.
Value *GVN::findPRELeader(BasicBlock *BB, Value *Op) {
  CongruenceClass *CC = valueToClass[Op];
  if (!CC || CC == InitialClass)
    return 0;
  if (CC->leader && (isa<Argument>(CC->leader) || isa<Constant>(CC->leader) || isa<GlobalValue>(CC->leader)))
    return CC->leader;

  for (CongruenceClass::MemberSet::iterator SI = CC->members.begin(),
         SE = CC->members.end(); SI != SE; ++SI)
    if (DT->dominates(SI->second, BB))
      return SI->first;
  return 0;
}

/// performPRE - Perform a purely local form of PRE that looks for diamond
/// control flow patterns and attempts to perform simple PRE at the
/// join point.
bool GVN::performPRE(Function &F) {
  bool Changed = false;
  for (unsigned i = 0, e = congruenceClass.size(); i != e; ++i) {
    CongruenceClass *CC = congruenceClass[i];
    // Skip the INITIAL class, singletons, and dead classes
    if (CC == InitialClass || CC->members.size() == 1 || CC->dead)
      continue;

    std::set<ValueDFS> DFSOrderedSet;
    // TODO: Include equivalences in PRE
    convertDenseToDFSOrdered(CC->members, DFSOrderedSet, false);
    DenseMap<BasicBlock*, Value*> predMap;
    for (std::set<ValueDFS>::iterator II = DFSOrderedSet.begin(), IE = DFSOrderedSet.end(); II != IE;) {
      Value *Member = II->value;
      int MemberDFSIn = II->dfs_in;
      int MemberDFSOut = II->dfs_out;
      ++II;

      if (Instruction *CurInst = dyn_cast<Instruction>(Member)) {
        BasicBlock *BB = CurInst->getParent();
        // Nothing to PRE in the entry block.
        if (BB == &F.getEntryBlock()) continue;
        // Don't perform PRE on a landing pad.
        if (BB->isLandingPad()) continue;
        // Don't do PRE on compares. The PHI would prevent CodeGenPrepare from
        // sinking the compare again, and it would force the code generator to
        // move the i1 from processor flags or predicate registers into a general
        // purpose register.
        if (isa<CmpInst>(CurInst))
          continue;
        if (isa<AllocaInst>(CurInst) ||
            isa<TerminatorInst>(CurInst) || isa<PHINode>(CurInst) ||
            CurInst->getType()->isVoidTy() ||
            CurInst->mayReadFromMemory() || CurInst->mayHaveSideEffects() ||
            isa<DbgInfoIntrinsic>(CurInst))
          continue;
        // Look for the predecessors for PRE opportunities.  We're
        // only trying to solve the basic diamond case, where
        // a value is computed in the successor and one predecessor,
        // but not the other.  We also explicitly disallow cases
        // where the successor is its own predecessor, because they're
        // more complicated to get right.
        unsigned NumWith = 0;
        unsigned NumWithout = 0;
        BasicBlock *PREPred = 0;
        int PREPredDFSIn = 0;
        int PREPredDFSOut = 0;
        predMap.clear();

        for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB); PI != PE; ++PI) {
          BasicBlock *P = *PI;
          // We're not interested in PRE where the block is its
          // own predecessor, or in blocks with predecessors
          // that are not reachable.
          if (P == BB) {
            NumWithout = 2;
            break;
          } else if (!reachableBlocks.count(P)) {
            NumWithout = 2;
            break;
          }
          std::pair<int, int> DFSPair = DFSBBMap[P];
          ValueDFS LookupTemp;
          LookupTemp.dfs_in = DFSPair.first;
          LookupTemp.dfs_out = DFSPair.second;
          LookupTemp.localnum = -1;

          std::set<ValueDFS>::iterator ResultLookup = DFSOrderedSet.lower_bound(LookupTemp);
          Value *predV = 0;
          if (ResultLookup != DFSOrderedSet.end()) {
            // Make sure our result dominates the predecessor
            if (ResultLookup->value != Member
                && ResultLookup->dfs_in <= DFSPair.first
                && ResultLookup->dfs_out >= DFSPair.second)
              predV = ResultLookup->value;
          }

          if (predV == 0) {
            PREPred = P;
            PREPredDFSIn = DFSPair.first;
            PREPredDFSOut = DFSPair.second;
            ++NumWithout;
          } else if (predV == CurInst) {
            NumWithout = 2;
          } else {
            predMap[P] = predV;
            ++NumWith;
          }
        }

      // Don't do PRE when it might increase code size, i.e. when
      // we would need to insert instructions in more than one pred.
      if (NumWithout != 1 || NumWith == 0)
        continue;

      // Don't do PRE across indirect branch.
      if (isa<IndirectBrInst>(PREPred->getTerminator()))
        continue;

      // We can't do PRE safely on a critical edge, so instead we schedule
      // the edge to be split and perform the PRE the next time we iterate
      // on the function.
      unsigned SuccNum = GetSuccessorNumber(PREPred, BB);
      if (isCriticalEdge(PREPred->getTerminator(), SuccNum)) {
        toSplit.push_back(std::make_pair(PREPred->getTerminator(), SuccNum));
        continue;
      }

      // Instantiate the expression in the predecessor that lacked it.
      // Because we are going top-down through the block, all value numbers
      // will be available in the predecessor by the time we need them.  Any
      // that weren't originally present will have been instantiated earlier
      // in this loop.
      Instruction *PREInstr = CurInst->clone();
      bool success = true;
      for (unsigned i = 0, e = CurInst->getNumOperands(); i != e; ++i) {
        Value *Op = PREInstr->getOperand(i);
        if (isa<Argument>(Op) || isa<Constant>(Op) || isa<GlobalValue>(Op))
          continue;

        if (Value *V = findPRELeader(PREPred, Op)) {
          PREInstr->setOperand(i, V);
        } else {
          success = false;
          break;
        }
      }

      // Fail out if we encounter an operand that is not available in
      // the PRE predecessor.  This is typically because of loads which
      // are not value numbered precisely.
      if (!success) {
        delete PREInstr;
        DEBUG(verifyRemoved(PREInstr));
        continue;
      }
      DEBUG(dbgs() << "Inserting instruction " << *PREInstr << " into " << getBlockName(PREPred) << "\n");

      PREInstr->insertBefore(PREPred->getTerminator());
      PREInstr->setName(CurInst->getName() + ".pre");
      PREInstr->setDebugLoc(CurInst->getDebugLoc());
      predMap[PREPred] = PREInstr;

      valueToClass[PREInstr] = CC;
      CC->members.insert(std::make_pair(PREInstr, PREPred));
      ++NumGVNPRE;
      ValueDFS VDFS;
      VDFS.value = PREInstr;
      VDFS.dfs_in = PREPredDFSIn;
      VDFS.dfs_out = PREPredDFSOut;
      DFSOrderedSet.insert(VDFS);

      // // Update the availability map to include the new instruction.
      // addToLeaderTable(ValNo, PREInstr, PREPred);

      // Create a PHI to make the value available in this block.
      pred_iterator PB = pred_begin(BB), PE = pred_end(BB);
      PHINode* Phi = PHINode::Create(CurInst->getType(), std::distance(PB, PE),
                                     CurInst->getName() + ".pre-phi",
                                     BB->begin());
      for (pred_iterator PI = PB; PI != PE; ++PI) {
        BasicBlock *P = *PI;
        Phi->addIncoming(predMap[P], P);
      }
      VDFS.value = Phi;
      VDFS.dfs_in = MemberDFSIn;
      VDFS.dfs_out = MemberDFSOut;
      DFSOrderedSet.insert(VDFS);
      Phi->setDebugLoc(CurInst->getDebugLoc());
      replaceInstruction(CurInst, Phi, CC);
      if (Phi->getType()->isPointerTy()) {
        // Because we have added a PHI-use of the pointer value, it has now
        // "escaped" from alias analysis' perspective.  We need to inform
        // AA of this.
        for (unsigned ii = 0, ee = Phi->getNumIncomingValues(); ii != ee;
             ++ii) {
          unsigned jj = PHINode::getOperandNumForIncomingValue(ii);
          AA->addEscapingUse(Phi->getOperandUse(jj));
        }

        if (MD)
          MD->invalidateCachedPointerInfo(Phi);
      }
      DEBUG(dbgs() << "GVN PRE removed: " << *Member << '\n');
      // PRE breaks leaders, so this may fail
      // DEBUG(verifyRemoved(CurInst));
      Changed = true;
      }
    }
  }

  if (splitCriticalEdges())
    Changed = true;

  return Changed;
}

void GVN::convertDenseToDFSOrdered(CongruenceClass::MemberSet &Dense,
                                   std::set<ValueDFS> &DFSOrderedSet,
				   bool Equivalences) {
  for (CongruenceClass::MemberSet::iterator DI = Dense.begin(), DE = Dense.end();
       DI != DE; ++DI) {
    std::pair<int, int> DFSPair = DFSBBMap[DI->second];
    ValueDFS VD;
    VD.dfs_in = DFSPair.first;
    VD.dfs_out = DFSPair.second;
    VD.equivalence = Equivalences;

    // If it's an instruction, use the real local dfs number.
    // If it's a value, it *must* have come from equality propagation,
    // and thus we know it is valid for the entire block.  By giving
    // the local number as 0, it should sort before the instructions
    // in that block.
    if (Instruction *I = dyn_cast<Instruction>(DI->first))
      VD.localnum = InstrLocalDFS[I];
    else
      VD.localnum = 0;

    VD.value = DI->first;
    DFSOrderedSet.insert(VD);
  }
}

uint32_t CongruenceClass::nextCongruenceNum = 0;

bool GVN::killUselessLoadInst(LoadInst *LI) {
  if (!LI->isSimple())
    return false;
  
  if (LI->use_empty()) {	
    UpdateMemDepInfo(MD, LI, NULL);
    markInstructionForDeletion(LI);
    ++NumGVNLoad;
    return true;
  }

  // Only need to ever check these loads once
  if (!checkedLoads.insert(LI).second)
    return false;
  MemDepResult Dep = MD->getDependency(LI);
  Instruction *DepInst = Dep.getInst();
  // If we can't determine an actual dependent instruction, we can't
  // do anything
  if (Dep.isClobber() || !Dep.isDef())
    return false;
  

  // If this load really doesn't depend on anything, then we must be loading an
  // undef value.  This can happen when loading for a fresh allocation with no
  // intervening stores, for example.
  if (isa<AllocaInst>(DepInst) || isMallocLikeFn(DepInst, TLI)) {
    LI->replaceAllUsesWith(UndefValue::get(LI->getType()));
    UpdateMemDepInfo(MD, LI, NULL);
    markUsersTouched(LI);
    markInstructionForDeletion(LI);
    ++NumGVNLoad;
    return true;
  }

  // If this load occurs either right after a lifetime begin,
  // then the loaded value is undefined.
  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(DepInst)) {
    if (II->getIntrinsicID() == Intrinsic::lifetime_start) {
      LI->replaceAllUsesWith(UndefValue::get(LI->getType()));
      UpdateMemDepInfo(MD, LI, NULL);
      markUsersTouched(LI);
      markInstructionForDeletion(LI);
      ++NumGVNLoad;
      return true;
    }
  }
  return false;
  
}

/// runOnFunction - This is the main transformation entry point for a function.
bool GVN::runOnFunction(Function& F) {
  DT = &getAnalysis<DominatorTree>();

  TD = getAnalysisIfAvailable<TargetData>();
  TLI = &getAnalysis<TargetLibraryInfo>();

  bool Changed = false;

  DEBUG(dbgs() << "Starting GVN on new function " << F.getName() << "\n");

  // Merge unconditional branches, allowing PRE to catch more
  // optimization opportunities.
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ) {
    BasicBlock *BB = FI++;

    bool removedBlock = MergeBlockIntoPredecessor(BB, this);
    if (removedBlock) ++NumGVNBlocks;

    Changed |= removedBlock;
  }
  uint32_t ICount = 0;

  CongruenceClass::nextCongruenceNum = 2;
  // Count number of instructions for sizing of hash tables
  unsigned NumBasicBlocks = F.size();
  DEBUG(dbgs() << "Found " << NumBasicBlocks <<  " basic blocks\n");
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)
    for (BasicBlock::iterator BI = FI->begin(), BE = FI->end();
         BI != BE; ++BI) {
      InstrLocalDFS[BI] = ICount;
      ++ICount;
    }
  // Ensure we don't end up resizing the expressionToClass map, as
  // that can be quite expensive. At most, we have one expression per
  // instruction.
  expressionToClass.resize(ICount *2);
  memoryExpressionToClass.resize(ICount *2);


  // Initialize the touched instructions to include the entry block
  for (BasicBlock::iterator BI = F.getEntryBlock().begin(), BE = F.getEntryBlock().end();
       BI != BE; ++BI)
    touchedInstructions.insert(BI);
  reachableBlocks.insert(&F.getEntryBlock());


  // Initialize all other instructions to be in INITIAL class
  CongruenceClass::MemberSet InitialValues;
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)  {
    for (BasicBlock::iterator BI = FI->begin(), BE = FI->end(); BI != BE; ++BI) {
      InitialValues.insert(std::make_pair(BI, FI));
    }
  }

  InitialClass = createCongruenceClass(NULL, NULL);
  for (CongruenceClass::MemberSet::iterator LI = InitialValues.begin(),
         LE = InitialValues.end();
       LI != LE; ++LI)
    valueToClass[LI->first] = InitialClass;
  InitialClass->members.swap(InitialValues);

  // Initialize arguments to be in their own unique congruence classes
  // In an IPA-GVN, this would not be done
  for (Function::arg_iterator FAI = F.arg_begin(), FAE= F.arg_end(); FAI != FAE; ++FAI)
    createSingletonCongruenceClass(FAI, &F.getEntryBlock());

  if (!noLoads) {
    MD = &getAnalysis<MemoryDependenceAnalysis>();
    AA = &getAnalysis<AliasAnalysis>();
  } else {
    MD = NULL;
    AA = NULL;
  }


  while (!touchedInstructions.empty()) {
    //TODO: Use two worklist method to keep ordering straight
    //TODO: or Investigate RPO numbering both blocks and instructions in the same pass,
    //  and walking both lists at the same time, processing whichever has the next number in order.
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
          // This is done in case something eliminates the instruction
          // along the way.
          Instruction *I = BI++;
          movedForward = true;

          if (processedCount.count(I) == 0) {
            processedCount.insert(std::make_pair(I, 1));
          } else {
            processedCount[I] += 1;
            assert(processedCount[I] < 100 && "Seem to have processed the same instruction a lot");
          }

          if (!I->isTerminator()) {
	    // This duplicates what old GVN used to do with loads
            if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
	      if (killUselessLoadInst(LI)) 
		continue;
	    }

            Expression *Symbolized = performSymbolicEvaluation(I, *RI);
            performCongruenceFinding(I, *RI, Symbolized);
          } else {
            processOutgoingEdges(dyn_cast<TerminatorInst>(I));
          }
        }
      }
    }
  }
  eliminateInstructions(F);

  
  // Note: It is not currently safe to use the leader info in congruence classes
  // after PRE has finished, as PRE does not update or respect them.
  if (EnablePRE) {
    bool PREChanged = true;
    while (PREChanged) {
      PREChanged = performPRE(F);
      Changed |= PREChanged;
    }
  }
  
  // Delete all instructions marked for deletion.
  for (DenseSet<Instruction*>::iterator DI = instrsToErase_.begin(), DE = instrsToErase_.end(); DI != DE;) {
    Instruction *toErase = *DI;
    ++DI;
    if (!toErase->use_empty())
      toErase->replaceAllUsesWith(UndefValue::get(toErase->getType()));

    toErase->eraseFromParent();
  }

  // Delete all unreachable blocks
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI){
    BasicBlock *BB = FI;
    if (!reachableBlocks.count(BB)) {
      DEBUG(dbgs() << "We believe block " << getBlockName(BB) << " is unreachable\n");
      DeleteInstructionInBlock(BB);
      Changed = true;
    }
  }


 valueToClass.clear();
 for (unsigned i = 0, e = congruenceClass.size(); i != e; ++i) {
   delete congruenceClass[i];
   congruenceClass[i] = NULL;
 }

  for (DenseSet<Expression*>::iterator DI = expressionToDelete.begin(), DE = expressionToDelete.end(); DI != DE;) {
    Expression *toErase = *DI;
    ++DI;
    delete toErase;
  }

  congruenceClass.clear();
  expressionToClass.clear();
  memoryExpressionToClass.clear();
  expressionToDelete.clear();
  uniquedExpressions.clear();
  reachableBlocks.clear();
  reachableEdges.clear();
  touchedInstructions.clear();
  processedCount.clear();
  expressionAllocator.Reset();
  instrsToErase_.clear();
  depQueryCache.clear();
  depIQueryCache.clear();
  locDepCache.clear();
  DFSBBMap.clear();
  InstrLocalDFS.clear();
  checkedLoads.clear();
  return Changed;
}


// Return true if V is a value that will always be available (IE can
// be placed anywhere) in the function.
// We avoid mentioning global value

static bool alwaysAvailable(Value *V) {
  return isa<Constant>(V) || isa<Argument>(V);
}

bool GVN::eliminateInstructions(Function &F) {
  // This is a non-standard eliminator. The normal way to eliminate is
  // to walk the dominator tree in order, keeping track of available
  // values, and eliminating them.  However, this is mildly
  // pointless. It requires doing lookups on every instruction,
  // regardless of whether we will ever eliminate it.  For
  // instructions part of a singleton congruence class, we know we
  // will never eliminate it.   The dominator tree method is actually
  // fairly expensive if you use a scoped hash table that performs
  // deletion on scope exit (as LLVM's does),  as it's possible your scope exiting
  // could cause removal of O(instructions) elements.

  // Instead, this eliminator looks at the congruence classes directly, sorts
  // them into a DFS ordering of the dominator tree, and then we just
  // perform eliminate straight on the sets by walking the congruence
  // class members in order, and eliminate the ones dominated by the
  // last member.   This is technically O(N log N) where N = number of
  // instructions.  However, we skip singleton sets, and we could make it O(
  // When we find something not dominated, it becomes the new leader
  // for elimination purposes

  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI) {
    DomTreeNode *DTN = DT->getNode(FI);
    if (!DTN)
      continue;
    DFSBBMap[FI] = std::make_pair(DTN->getDFSNumIn(), DTN->getDFSNumOut());
  }

 for (unsigned i = 0, e = congruenceClass.size(); i != e; ++i) {
   CongruenceClass *CC = congruenceClass[i];
   // FIXME: We should eventually be able to replace everything still
   // in the initial class with undef, as they should be unreachable. 
   // Right now, initial still contains some things we skip value
   // numbering of (UNREACHABLE's, for example)
   if (CC == InitialClass || CC->dead)
     continue;
   // This is a stack because equality replacement/etc may place
   // constants in the middle of the member list, and we want to use
   // those constant values in preference to the current leader, over
   // the scope of those constants.
   SmallVector<Value*, 4> ValueStack;
   SmallVector<std::pair<int, int>, 64> DFSStack;
   assert (CC->leader && "We should have had a leader");

   // if (!CC->expression || !CC->leader)
   //   continue;

   // If this is a leader that is always available, just replace
   // everything with it
   if (alwaysAvailable(CC->leader)) {
     for (CongruenceClass::MemberSet::iterator CI = CC->members.begin(),
            CE = CC->members.end(); CI != CE; ) {
       Value *member = CI->first;
       ++CI;
       // Skip the member that is also us :)
       if (member != CC->leader) {
	 //
         if (isa<StoreInst>(member) || (isa<BitCastInst>(member) && !isa<Constant>(CC->leader)))
           continue;
         DEBUG(dbgs() << "Found replacement " << *(CC->leader) << " for " << *member << "\n");
         // Due to equality propagation, these may not always be
         // instructions, they may be real values.  We don't really
         // care about trying to replace the non-instructions.
         if (Instruction *I = dyn_cast<Instruction>(member))
           replaceInstruction(I, CC->leader, CC);
       }
     }
   } else {
     DEBUG(dbgs() << "Eliminating in congruence class " << CC->id << "\n");
     // If this is a singleton, the only thing we do is process
     // non-local loads.
     // TODO: processNonLocalLoad still does PRE of loads,
     // and looking or store based eliminations, and this happens
     // nowhere else.  Once that is fixed, this should be removed.
     if ((CC->members.size()  + CC->equivalences.size()) == 1) {
       MemoryExpression *ME;
       if (CC->expression && (ME = dyn_cast<MemoryExpression>(CC->expression))) {
       // This looks like a weird test, because of where hadNonLocal
       // is set (depEquals), but it will trigger if we at least
       // *compared* it with something, causing non-local to be set
       // appropriately.  If nothing else in the function was even
       // compared to this, it's probably not worth trying to eliminate.
         if (ME->hadNonLocal()) {
           Value *member = CC->members.begin()->first;
           if (LoadInst *LI = dyn_cast<LoadInst>(member)){
             processNonLocalLoad(LI);
           }
         }
       }
       continue;
     } else {
       // Convert the members and equivalences to DFS ordered sets and
       // then merge them.
       std::set<ValueDFS> DFSOrderedMembers;
       convertDenseToDFSOrdered(CC->members, DFSOrderedMembers, false);
       std::set<ValueDFS> DFSOrderedEquivalences;
       convertDenseToDFSOrdered(CC->equivalences, DFSOrderedEquivalences, true);
       std::vector<ValueDFS> DFSOrderedSet;
       set_union(DFSOrderedMembers.begin(), DFSOrderedMembers.end(),
		 DFSOrderedEquivalences.begin(), DFSOrderedEquivalences.end(),
		 std::inserter(DFSOrderedSet, DFSOrderedSet.begin()));

       for (std::vector<ValueDFS>::iterator CI = DFSOrderedSet.begin(), CE = DFSOrderedSet.end(); CI != CE;) {
         int MemberDFSIn = CI->dfs_in;
         int MemberDFSOut = CI->dfs_out;
         Value *Member = CI->value;
         bool EquivalenceOnly = CI->equivalence;
         ++CI;

         //TODO: eliminate duplicate bitcasts but not valid bitcasts.
         // Bitcasts are always going to value number to their values
         // (because of type ignoring above), so we will think we can
         // eliminate every bitcast, but we can't. We could eliminate
         // bitcasts that value number to *other bitcasts*.
	 
         if (isa<StoreInst>(Member) || (isa<BitCastInst>(Member) && !isa<Constant>(CC->leader)))
           continue;

         if (DFSStack.empty()) {
           DEBUG(dbgs() << "DFS Stack is empty\n");
         } else {
           DEBUG(dbgs() << "Stack Top DFS numbers are (" << DFSStack.back().first << "," << DFSStack.back().second << ")\n");
         }

         DEBUG(dbgs() << "Current DFS numbers are (" << MemberDFSIn << "," << MemberDFSOut <<")\n");
         // We should push whenever it's empty or there is a constant
         // or a local equivalence we want to replace with
         // We should pop until we are back within the right DFS scope
         // Walk along, processing members who are dominated by each other.
         if ((ValueStack.empty() || isa<Constant>(Member) || EquivalenceOnly)
             || !(MemberDFSIn >= DFSStack.back().first
                  && MemberDFSOut <= DFSStack.back().second)) {
           assert(ValueStack.size() == DFSStack.size() && "Mismatch between ValueStack and DFSStack");
           while (!DFSStack.empty()
                  && !(MemberDFSIn >= DFSStack.back().first  && MemberDFSOut <= DFSStack.back().second)){
             DFSStack.pop_back();
             ValueStack.pop_back();
           }

           if (DFSStack.empty() || isa<Constant>(Member)) {
             ValueStack.push_back(Member);
             DFSStack.push_back(std::make_pair(MemberDFSIn, MemberDFSOut));
           }
         } 
	 // Skip the case of trying to eliminate the leader, or trying
	 // to eliminate equivalence-only candidates
	 if (Member == CC->leader || EquivalenceOnly)
	   continue;
	 
	 Value *Result = ValueStack.back();
	 if (Member == Result)
	   continue;
	 DEBUG(dbgs() << "Found replacement " << *Result << " for " << *Member << "\n");
	 
	 // If we find a load or a store as a replacement, make sure
	 // to correct the type.  This should *always* work, since
	 // we tested whether it was possible before value numbering
	 // the same.
	 LoadInst *LI;
	 if (Result->getType() != Member->getType() && (LI = dyn_cast<LoadInst>(Member))) {
	   if (LoadInst *LIR = dyn_cast<LoadInst>(Result)) {
	     int Offset = AnalyzeLoadFromClobberingLoad(LI->getType(),
							LI->getPointerOperand(),
							LIR, *TD);
	     // If this assert is triggered, something is broken in
	     // value numbering, not elimination.
	     assert(Offset != -1 && "Should have been able to coerce load");
	     
	     Result = GetLoadValueForLoad(LIR, Offset, LI->getType(), LI, *TD);
	     
	   } else if (StoreInst *SIR = dyn_cast<StoreInst>(Result)) {
	     int Offset = AnalyzeLoadFromClobberingStore(LI->getType(),
							 LI->getPointerOperand(),
							 SIR, *TD);
	     // If this assert is triggered, something is broken in
	     // value numbering, not elimination.
	     assert(Offset != -1 && "Should have been able to coerce store");
	     Result = GetStoreValueForLoad(SIR->getValueOperand(), Offset, LI->getType(), LI, *TD);
	   }
	 }
	 
	 // Perform actual replacement
	 Instruction *I;
	 if ((I = dyn_cast<Instruction>(Member)) && Member != CC->leader)
	   replaceInstruction(I, Result, CC);
       }
     }
   }
 }
//FIXME: Calculate changed for real
 return true;
}
