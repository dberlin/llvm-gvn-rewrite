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

  MemoryDependenceAnalysis *MD;
  const TargetData *TD;

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

/// CoerceAvailableValueToLoadType - If we saw a store of a value to memory, and
/// then a load from a must-aliased pointer of a different type, try to coerce
/// the stored value.  LoadedTy is the type of the load we want to replace and
/// InsertPt is the place to insert new instructions.
///
/// If we can't do it, return null.
static Value *CoerceAvailableValueToLoadType(Value *StoredVal, 
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
    if (StoredValTy->isPointerTy() && LoadedTy->isPointerTy())
      return new BitCastInst(StoredVal, LoadedTy, "", InsertPt);
    
    // Convert source pointers to integers, which can be bitcast.
    if (StoredValTy->isPointerTy()) {
      StoredValTy = TD.getIntPtrType(StoredValTy->getContext());
      StoredVal = new PtrToIntInst(StoredVal, StoredValTy, "", InsertPt);
    }
    
    Type *TypeToCastTo = LoadedTy;
    if (TypeToCastTo->isPointerTy())
      TypeToCastTo = TD.getIntPtrType(StoredValTy->getContext());
    
    if (StoredValTy != TypeToCastTo)
      StoredVal = new BitCastInst(StoredVal, TypeToCastTo, "", InsertPt);
    
    // Cast to pointer if the load needs a pointer type.
    if (LoadedTy->isPointerTy())
      StoredVal = new IntToPtrInst(StoredVal, LoadedTy, "", InsertPt);
    
    return StoredVal;
  }
  
  // If the loaded value is smaller than the available value, then we can
  // extract out a piece from it.  If the available value is too small, then we
  // can't do anything.
  assert(StoreSize >= LoadSize && "CanCoerceMustAliasedValueToLoad fail");
  
  // Convert source pointers to integers, which can be manipulated.
  if (StoredValTy->isPointerTy()) {
    StoredValTy = TD.getIntPtrType(StoredValTy->getContext());
    StoredVal = new PtrToIntInst(StoredVal, StoredValTy, "", InsertPt);
  }
  
  // Convert vectors and fp to integer, which can be manipulated.
  if (!StoredValTy->isIntegerTy()) {
    StoredValTy = IntegerType::get(StoredValTy->getContext(), StoreSize);
    StoredVal = new BitCastInst(StoredVal, StoredValTy, "", InsertPt);
  }
  
  // If this is a big-endian system, we need to shift the value down to the low
  // bits so that a truncate will work.
  if (TD.isBigEndian()) {
    Constant *Val = ConstantInt::get(StoredVal->getType(), StoreSize-LoadSize);
    StoredVal = BinaryOperator::CreateLShr(StoredVal, Val, "tmp", InsertPt);
  }
  
  // Truncate the integer to the right size now.
  Type *NewIntTy = IntegerType::get(StoredValTy->getContext(), LoadSize);
  StoredVal = new TruncInst(StoredVal, NewIntTy, "trunc", InsertPt);
  
  if (LoadedTy == NewIntTy)
    return StoredVal;
  
  // If the result is a pointer, inttoptr.
  if (LoadedTy->isPointerTy())
    return new IntToPtrInst(StoredVal, LoadedTy, "inttoptr", InsertPt);
  
  // Otherwise, bitcast.
  return new BitCastInst(StoredVal, LoadedTy, "bitcast", InsertPt);
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
                               

/// GetStoreValueForLoad - This function is called when we have a
/// memdep query of a load that ends up being a clobbering store.  This means
/// that the store provides bits used by the load but we the pointers don't
/// mustalias.  Check this case to see if there is anything more we can do
/// before we give up.
static Value *GetStoreValueForLoad(Value *SrcVal, unsigned Offset,
                                   Type *LoadTy,
                                   Instruction *InsertPt, const TargetData &TD){
  LLVMContext &Ctx = SrcVal->getType()->getContext();
  
  uint64_t StoreSize = (TD.getTypeSizeInBits(SrcVal->getType()) + 7) / 8;
  uint64_t LoadSize = (TD.getTypeSizeInBits(LoadTy) + 7) / 8;
  
  IRBuilder<> Builder(InsertPt->getParent(), InsertPt);
  
  // Compute which bits of the stored value are being used by the load.  Convert
  // to an integer type to start with.
  if (SrcVal->getType()->isPointerTy())
    SrcVal = Builder.CreatePtrToInt(SrcVal, TD.getIntPtrType(Ctx));
  if (!SrcVal->getType()->isIntegerTy())
    SrcVal = Builder.CreateBitCast(SrcVal, IntegerType::get(Ctx, StoreSize*8));
  
  // Shift the bits to the least significant depending on endianness.
  unsigned ShiftAmt;
  if (TD.isLittleEndian())
    ShiftAmt = Offset*8;
  else
    ShiftAmt = (StoreSize-LoadSize-Offset)*8;
  
  if (ShiftAmt)
    SrcVal = Builder.CreateLShr(SrcVal, ShiftAmt);
  
  if (LoadSize != StoreSize)
    SrcVal = Builder.CreateTrunc(SrcVal, IntegerType::get(Ctx, LoadSize*8));
  
  return CoerceAvailableValueToLoadType(SrcVal, LoadTy, InsertPt, TD);
}

/// GetLoadValueForLoad - This function is called when we have a
/// memdep query of a load that ends up being a clobbering load.  This means
/// that the load *may* provide bits used by the load but we can't be sure
/// because the pointers don't mustalias.  Check this case to see if there is
/// anything more we can do before we give up.
static Value *GetLoadValueForLoad(LoadInst *SrcVal, unsigned Offset,
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
    LoadInst *NewLoad = Builder.CreateLoad(PtrVal);
    NewLoad->takeName(SrcVal);
    NewLoad->setAlignment(SrcVal->getAlignment());

    DEBUG(dbgs() << "GVN WIDENED LOAD: " << *SrcVal << "\n");
    DEBUG(dbgs() << "TO: " << *NewLoad << "\n");
    
    // Replace uses of the original load with the wider load.  On a big endian
    // system, we need to shift down to get the relevant bits.
    Value *RV = NewLoad;
    if (TD.isBigEndian())
      RV = Builder.CreateLShr(RV,
                    NewLoadSize*8-SrcVal->getType()->getPrimitiveSizeInBits());
    RV = Builder.CreateTrunc(RV, SrcVal->getType());
    SrcVal->replaceAllUsesWith(RV);
    
    // We would like to use gvn.markInstructionForDeletion here, but we can't
    // because the load is already memoized into the leader map table that GVN
    // tracks.  It is potentially possible to remove the load from the table,
    // but then there all of the operations based on it would need to be
    // rehashed.  Just leave the dead load around.
    MD->removeInstruction(SrcVal);
    SrcVal = NewLoad;
  }
  
  return GetStoreValueForLoad(SrcVal, Offset, LoadTy, InsertPt, TD);
}


  static std::string getBlockName(BasicBlock *B) {
    return DOTGraphTraits<const Function*>::getSimpleNodeLabel(B, NULL);
  }

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
    
    Type *getType() const {
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
	} else {
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
      }
      return true;      
    }
    
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
  };
  class MemoryExpression: public BasicExpression {
  private:
    void operator=(const MemoryExpression&); // Do not implement
    MemoryExpression(const MemoryExpression&); // Do not implement
  protected:
    bool isStore_;
    union {
      Instruction *inst;
      LoadInst *loadinst;
      StoreInst *storeinst;
    } inst_;
    
  public:
    
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
      inst_.loadinst = L;
    };

    MemoryExpression(StoreInst *S) {
      etype_ = ExpressionTypeMemory;
      isStore_ = true;
      inst_.storeinst = S;
    };
    
    virtual ~MemoryExpression() {};

    virtual bool equals(const Expression &other) const {
      const MemoryExpression &OE = cast<MemoryExpression>(other);
      if (varargs != OE.varargs)
	return false;
      
      if (!isStore_) {
	LoadInst *LI = inst_.loadinst;
	Instruction *OI = OE.inst_.inst;
	
	if (LI != OI) {
	  MemDepResult Dep = MD->getDependency(LI);
	  if (Dep.getInst() != OI)
	    return false;
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
	    return false;
	  }
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
	   // Non-local case
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
    
    virtual hash_code getHashValue() const {
      return hash_combine(etype_, opcode_,
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
    
    VariableExpression():Expression(ExpressionTypeVariable), variableValue_(NULL) { }
    
      
    VariableExpression(Value *variableValue):Expression(ExpressionTypeVariable), variableValue_(variableValue) { }
    virtual hash_code getHashValue() const {
      return hash_combine(etype_, variableValue_->getType(), variableValue_);
    }

    virtual void print(raw_ostream &OS) {
      OS << "{etype = " << etype_ << ", opcode = " << opcode_ << ", variable = " << variableValue_ << " }";
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
    const TargetLibraryInfo *TLI;
    DenseMap<BasicBlock*, uint32_t> rpoBlockNumbers;
    DenseMap<Instruction*, uint32_t> rpoInstructionNumbers;
    DenseMap<BasicBlock*, std::pair<uint32_t, uint32_t> > rpoInstructionStartEnd;
    std::vector<BasicBlock*> rpoToBlock;
    DenseSet<BasicBlock*> reachableBlocks;
    DenseSet<std::pair<BasicBlock*, BasicBlock*> > reachableEdges;
    DenseSet<Instruction*> touchedInstructions;
    DenseMap<Instruction*, uint32_t> processedCount;
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
    
    Value *lookupOperandLeader(Value*);
    
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

  public:
    static char ID; // Pass identification, replacement for typeid
    explicit GVN(bool noloads = false)
        : FunctionPass(ID), NoLoads(noloads) {
      initializeGVNPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F);
    
    const TargetData *getTargetData() const { return TD; }
    DominatorTree &getDominatorTree() const { return *DT; }
    AliasAnalysis *getAliasAnalysis() const { return AA; }
    MemoryDependenceAnalysis &getMemDep() const { return *MD; }
  private:
    Expression *performSymbolicEvaluation(Value*, BasicBlock*);
    Expression *performSymbolicLoadEvaluation(Instruction*, BasicBlock*);
    Expression *performSymbolicStoreEvaluation(Instruction*, BasicBlock*);
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
    void dump(DenseMap<uint32_t, Value*> &d);
    Value *findLeader(BasicBlock *BB, uint32_t num);
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
    // TODO: 10% of our time is spent in SimplifyCmpInst with pointer operands
    //TODO: Since we noop bitcasts, we may need to check types before
    //simplifying, so that we don't end up simplifying based on a wrong
    //type assumption.

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
  } else if (isa<GetElementPtrInst>(I)) {
    //TODO: Since we noop bitcasts, we may need to check types before
    //simplifying, so that we don't end up simplifying based on a
    //wrong type assumption
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
  setBasicExpressionInfo(I, E);
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

//lookupOperandLeader -- See if we have a congruence class and leader for this operand, and if so, return it.
// Otherwise, return the original operand
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
      E = performSymbolicCallEvaluation(I, B);
      break;
    case Instruction::Store:
      E = performSymbolicStoreEvaluation(I, B);
      break;
    case Instruction::Load:
      E = performSymbolicLoadEvaluation(I, B);
      break;
    case Instruction::BitCast: {
      //Pointer bitcasts are noops, we can just make them out of whole cloth if we need to. 
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

  // Expressions we can't symbolize are always in their own unique congruence class
  if (E == NULL) {
    // We may have already made a unique class
    if (VClass->members.size() != 1 || VClass->leader != V) {
      CongruenceClass *NewClass = new CongruenceClass();
      congruenceClass.push_back(NewClass);
      // We should always be adding it below
      // NewClass->members.push_back(V);
      NewClass->expression = NULL;
      NewClass->leader = V;
      EClass = NewClass;
      DEBUG(dbgs() << "Created new congruence class for " << V << " due to NULL expression\n");

    } else {
      EClass = VClass;
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

      // Constants and variables should always be made the leader
      if (ConstantExpression *CE = dyn_cast<ConstantExpression>(E))
        NewClass->leader = CE->getConstantValue();
      else if (VariableExpression *VE = dyn_cast<VariableExpression>(E))
	NewClass->leader = VE->getVariableValue();
      else if (MemoryExpression *ME = dyn_cast<MemoryExpression>(E)) {
	if (ME->isStore())
	  NewClass->leader = lookupOperandLeader(ME->getStoreInst()->getValueOperand());
	else 
	  NewClass->leader = V;
      } else
        NewClass->leader = V;
      
      EClass = NewClass;
      DEBUG(dbgs() << "Created new congruence class for " << V << " using expression " << *E << " at " << NewClass->id << "\n");
    } else {
      EClass = VTCI->second;
      MemoryExpression *ME = dyn_cast<MemoryExpression>(E);
      MemoryExpression *ClassME;
      if (ME && EClass->expression && (ClassME = dyn_cast<MemoryExpression>(EClass->expression))) {
      }	
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
	if (VClass->expression)
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
       for (BasicBlock::iterator BI = EntryBlock.begin(), BE = EntryBlock.end();
	    BI != BE; ++BI)
	 ++ICount;
       
      // 	rpoInstructionNumbers[BI] = rpoINumber++;
      uint32_t IEnd = rpoINumber;
      // rpoInstructionStartEnd[*RI] = std::make_pair(IStart, IEnd);
    }
  // Ensure we don't end up resizing the expressionToClass map, as that can be quite expensive
  expressionToClass.resize(ICount);
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
	  if (processedCount.count(I) == 0) {
	    processedCount.insert(std::make_pair(I, 1));
	  } else {
	    processedCount[I] += 1;
	    assert(processedCount[I] < 100 && "Seem to have processed the same instruction a lot");
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
