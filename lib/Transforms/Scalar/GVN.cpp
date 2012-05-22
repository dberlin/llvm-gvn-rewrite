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
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Support/PatternMatch.h"
#include "llvm/Support/RecyclingAllocator.h"
#include "llvm/Transforms/Utils/Local.h"


#include <algorithm>
#include <list>
#include <set>

using namespace llvm;
using namespace PatternMatch;

STATISTIC(NumGVNInstrDeleted,  "Number of instructions deleted");
STATISTIC(NumGVNBlocksDeleted, "Number of blocks deleted");
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
  DenseSet<Instruction*> instrsToErase_;
  static void markInstructionForDeletion(Instruction *I) {
    instrsToErase_.insert(I);
  }

  static void UpdateMemDepInfo(MemoryDependenceAnalysis *MD, Instruction *I, Value *V) {
    if (MD && V && V->getType()->isPointerTy())
      MD->invalidateCachedPointerInfo(V);
    if (MD && I && !I->isTerminator())
      MD->removeInstruction(I);
  }

  struct CongruenceClass;
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
    // FIXME: This is no longer a problem
    UpdateMemDepInfo(MD, SrcVal, NULL);
    SrcVal = NewLoad;
  }

  return GetStoreValueForLoad(SrcVal, Offset, LoadTy, InsertPt, TD);
}

/// GetMemInstValueForLoad - This function is called when we have a
/// memdep query of a load that ends up being a clobbering mem intrinsic.
static Value *GetMemInstValueForLoad(MemIntrinsic *SrcInst, unsigned Offset,
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
    if (LoadSize != 1)
      Val = Builder.CreateZExt(Val, IntegerType::get(Ctx, LoadSize*8));

    Value *OneElt = Val;

    // Splat the value out to the right number of bits.
    for (unsigned NumBytesSet = 1; NumBytesSet != LoadSize; ) {
      // If we can double the number of bytes set, do it.
      if (NumBytesSet*2 <= LoadSize) {
        Value *ShVal = Builder.CreateShl(Val, NumBytesSet*8);
        Val = Builder.CreateOr(Val, ShVal);
        NumBytesSet <<= 1;
        continue;
      }

      // Otherwise insert one byte at a time.
      Value *ShVal = Builder.CreateShl(Val, 1*8);
      Val = Builder.CreateOr(OneElt, ShVal);
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

  static std::string getBlockName(BasicBlock *B) {
    return DOTGraphTraits<const Function*>::getSimpleNodeLabel(B, NULL);
  }

  class Expression;

  static uint32_t nextCongruenceNum = 0;
  struct CongruenceClass {
    uint32_t id;
    Value* leader;
    Expression* expression;
    std::set<std::pair<std::pair<int, int>, Value*> > members;
    bool dead;
    CongruenceClass():id(nextCongruenceNum++), leader(0), expression(0), dead(false) {};

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
    // Return true if this is equivalent to the other expression, including memory dependencies
    virtual bool depequals(const Expression &other) const {
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
    };

    virtual ~CallExpression() {};

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
    virtual bool depequals (const Expression &other) const {
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
	  // Non-local case.
	  const MemoryDependenceAnalysis::NonLocalDepInfo &deps =
	    MD->getNonLocalCallDependency(CallSite(callinst_));
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

	    // We don't handle non-definitions.  If we already have a
	    // call, reject instruction dependencies.
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

    void *operator new(size_t s) {
      return expressionAllocator.Allocate<CallExpression>();
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

    bool nonLocalEquals(LoadInst *LI, MemDepResult &Dep,
			const MemoryExpression &OE) const {
      // We should mark the non-local ones so we can try to PRE them later
      SmallVector<NonLocalDepResult, 64> Deps;
      AliasAnalysis::Location Loc = AA->getLocation(LI);
      MD->getNonLocalPointerDependency(Loc, true, LI->getParent(), Deps);
      // If we had to process more than one hundred blocks to find the
      // dependencies, this load isn't worth worrying about.  Optimizing
      // it will be too expensive.
      unsigned NumDeps = Deps.size();
      if (NumDeps > 100)
	return false;
      if (NumDeps == 1 &&
	  !Deps[0].getResult().isDef() && !Deps[0].getResult().isClobber()) {
	DEBUG(
	      dbgs() << "GVN: non-local load ";
	      WriteAsOperand(dbgs(), LI);
	      dbgs() << " has unknown dependencies\n";
	      );
	return false;
      }
      // Check to see if all of the other deps have the same congruence class
      // as the load we are asking about, or we can reuse them.
      CongruenceClass *OIClass = valueToClass[OE.inst_.loadinst];
      if (!OIClass)
	return false;
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
	  return false;
	}
	// Make sure all dependencies are in the same congruence
	// class
	CongruenceClass *DepClass = valueToClass[DepInfo.getInst()];
	// It could be an unreachable dependency MemDep doesn't know
	// about.
	// If this is the case, it will still have the InitialClass as
	// its congruence class
	if (DepClass != OIClass)
	  return false;

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
		return false;
	      }
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
		return false;
	      }
	    }
	  }
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
	      return false;
	    }
	  }
	  continue;
	}

	if (LoadInst *LD = dyn_cast<LoadInst>(DepInst)) {
	  // If the types mismatch and we can't handle it, reject reuse of the load.
	  if (LD->getType() != LI->getType()) {
	    // If the stored value is larger or equal to the loaded value, we can
	    // reuse it.
	    if (TD == 0 || !CanCoerceMustAliasedValueToLoad(LD, LI->getType(),*TD)){
	      return false;
	    }
	  }
	  continue;
	}
	return false;
      }
      // If we got through all the dependencies, we are good to go
      return true;
    }
    virtual bool equals(const Expression &other) const {
      // Two loads/stores are never the same if we don't have memory dependence info
      if (!MD)
	return false;
      const MemoryExpression &OE = cast<MemoryExpression>(other);
      if (varargs != OE.varargs)
	return false;
      return true;
    }

    virtual bool depequals(const Expression &other) const {
      const MemoryExpression &OE = cast<MemoryExpression>(other);
      if (!isStore_) {
	LoadInst *LI = inst_.loadinst;
	Instruction *OI = OE.inst_.inst;

	if (LI != OI) {
	  MemDepResult Dep = MD->getDependency(LI);
	  if (Dep.isNonLocal()) {
	    return 0;
	    return nonLocalEquals(LI, Dep, OE);
	  }
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
	    // TODO: MemIntrinsic case
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
	  //TODO: memintrisic case
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

    VariableExpression():Expression(ExpressionTypeVariable), variableValue_(NULL) { }


    VariableExpression(Value *variableValue):Expression(ExpressionTypeVariable), variableValue_(variableValue) { }
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

    ConstantExpression():Expression(ExpressionTypeConstant), constantValue_(NULL) { }


    ConstantExpression(Constant *constantValue):Expression(ExpressionTypeConstant), constantValue_(constantValue) { }
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
typedef RecyclingAllocator<BumpPtrAllocator,
			       ScopedHashTableVal<CongruenceClass*, Value*> > AllocatorTy;
    typedef ScopedHashTable<CongruenceClass*, Value*,  DenseMapInfo<CongruenceClass*>,
			    AllocatorTy> ScopedHTType;
    typedef ScopedHTType::ScopeTy ScopeType;
    bool noLoads;
    DominatorTree *DT;
    PostDominatorTree *PDT;
    const TargetLibraryInfo *TLI;
    DenseMap<BasicBlock*, uint32_t> rpoBlockNumbers;
    DenseMap<Instruction*, uint32_t> rpoInstructionNumbers;
    DenseMap<BasicBlock*, std::pair<uint32_t, uint32_t> > rpoInstructionStartEnd;
    std::vector<BasicBlock*> rpoToBlock;
    DenseSet<std::pair<BasicBlock*, BasicBlock*> > reachableEdges;
    DenseSet<Instruction*> touchedInstructions;
    DenseMap<Instruction*, uint32_t> processedCount;
    // Tested: SparseBitVector, DenseSet, etc. vector<bool> is 3 times as fast
    std::vector<bool> touchedBlocks;
    std::vector<CongruenceClass*> congruenceClass;
    DenseMap<BasicBlock*, ScopeType*> ScopeMap;
    ScopedHTType availableTable;

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


    /// LeaderTable - A mapping from value numbers to lists of Value*'s that
    /// have that value number.  Use findLeader to query it.
    struct LeaderTableEntry {
      Value *Val;
      BasicBlock *BB;
      LeaderTableEntry *Next;
    };
    DenseMap<uint32_t, LeaderTableEntry> leaderTable;
    BumpPtrAllocator tableAllocator;

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
  private:
    void ReplaceInstruction(Instruction*, Value*, CongruenceClass*);
    // Symbolic evaluation
    Expression *performSymbolicEvaluation(Value*, BasicBlock*);
    Expression *performSymbolicLoadEvaluation(Instruction*, BasicBlock*);
    Expression *performSymbolicStoreEvaluation(Instruction*, BasicBlock*);
    Expression *performSymbolicCallEvaluation(Instruction*, BasicBlock*);
    Expression *performSymbolicPHIEvaluation(Instruction*, BasicBlock*);
    // Congruence findin
    void performCongruenceFinding(Value*, Expression*, DomTreeNode*);
    // Predicate and reachability handling
    void updateReachableEdge(BasicBlock*, BasicBlock*);
    void processOutgoingEdges(TerminatorInst* TI);
    void propagateChangeInEdge(BasicBlock*);

    bool processNonLocalLoad(LoadInst *);

    // Dominator based replacement handling
    void EnterScope(BasicBlock*);
    void ExitScope(BasicBlock*);
    bool ProcessBlock(BasicBlock*);
    void ProcessTerminatorInst(TerminatorInst *TI);
    void ExitScopeIfDone(DomTreeNode*, DenseMap<DomTreeNode*, unsigned>&,
			 DenseMap<DomTreeNode*, DomTreeNode*> &);

    /// addToLeaderTable - Push a new Value to the LeaderTable onto the list for
    /// its value number.
    void addToLeaderTable(uint32_t N, Value *V, BasicBlock *BB) {
      LeaderTableEntry &Curr = leaderTable[N];
      if (!Curr.Val) {
        Curr.Val = V;
        Curr.BB = BB;
        return;
      }

      LeaderTableEntry *Node = tableAllocator.Allocate<LeaderTableEntry>();
      Node->Val = V;
      Node->BB = BB;
      Node->Next = Curr.Next;
      Curr.Next = Node;
    }

    /// removeFromLeaderTable - Scan the list of values corresponding to a given
    /// value number, and remove the given value if encountered.
    void removeFromLeaderTable(uint32_t N, Value *V, BasicBlock *BB) {
      LeaderTableEntry* Prev = 0;
      LeaderTableEntry* Curr = &leaderTable[N];

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
      if (!noLoads)
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
    bool splitCriticalEdges();
    unsigned replaceAllDominatedUsesWith(Value *From, Value *To,
                                         BasicBlock *Root);
    bool propagateEquality(Value *LHS, Value *RHS, BasicBlock *Root);
  };

  char GVN::ID = 0;
}


// createGVNPass - The public interface to this file...
FunctionPass *llvm::createGVNPass(bool noLoads) {
  return new GVN(noLoads);
}

INITIALIZE_PASS_BEGIN(GVN, "gvn", "Global Value Numbering", false, false)
INITIALIZE_PASS_DEPENDENCY(MemoryDependenceAnalysis)
INITIALIZE_PASS_DEPENDENCY(DominatorTree)
INITIALIZE_PASS_DEPENDENCY(PostDominatorTree)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfo)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(GVN, "gvn", "Global Value Numbering", false, false)

// findLeader - In order to find a leader for a given value number at a
// specific basic block, we first obtain the list of all Values for that number,
// and then scan the list to find one whose block dominates the block in
// question.  This is fast because dominator tree queries consist of only
// a few comparisons of DFS numbers.
Value *GVN::findLeader(BasicBlock *BB, uint32_t num) {
  LeaderTableEntry Vals = leaderTable[num];
  if (!Vals.Val) return 0;

  Value *Val = 0;
  if (DT->dominates(Vals.BB, BB)) {
    Val = Vals.Val;
    if (isa<Constant>(Val)) return Val;
  }

  LeaderTableEntry* Next = Vals.Next;
  while (Next) {
    if (DT->dominates(Next->BB, BB)) {
      if (isa<Constant>(Next->Val)) return Next->Val;
      if (!Val) Val = Next->Val;
    }

    Next = Next->Next;
  }

  return Val;
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
    //type assumption. We should clean this up so we can use constants of the wrong type


    if (E->varargs[0]->getType() == E->varargs[1]->getType()) {
      Value *V = SimplifyCmpInst(Predicate, E->varargs[0], E->varargs[1], TD, TLI, DT);
      Constant *C;
      if (V && (C = dyn_cast<Constant>(V))) {
	DEBUG(dbgs() << "Simplified " << *I << " to " << " constant " << *C << "\n");
	NumGVNCmpInsSimplified++;
	delete E;
	return createConstantExpression(C);
      }
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
  expressionToDelete.insert(E);

  if (isa<ConstantExpression>(E) || isa<VariableExpression>(E))
    E = uniquifyExpression(E);
  return E;
}

/// performCongruenceFinding - Perform congruence finding on a given value numbering expression
void GVN::performCongruenceFinding(Value *V, Expression *E, DomTreeNode *DTN) {
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
    ExpressionClassMap *lookupMap = isa<MemoryExpression>(E) ? &memoryExpressionToClass : &expressionToClass;
    std::pair<ExpressionClassMap::iterator, bool> lookupResult =
      lookupMap->insert(std::make_pair(E, (CongruenceClass*)NULL));

    // If it's not in the value table, create a new congruence class
    if (lookupResult.second) {
      CongruenceClass *NewClass = new CongruenceClass();
      congruenceClass.push_back(NewClass);
      // We should always be adding it below
      // NewClass->members.push_back(V);
      NewClass->expression = E;
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
      DEBUG(dbgs() << "Created new congruence class for " << V << " using expression " << *E << " at " << NewClass->id << "\n");
    } else {
      EClass = lookupResult.first->second;
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
      int dfs_in = DTN->getDFSNumIn();
      int dfs_out = DTN->getDFSNumOut();
      VClass->members.erase(std::make_pair(std::make_pair(dfs_in, dfs_out), V));
      // assert(std::find(EClass->members.begin(), EClass->members.end(), V) == EClass->members.end() && "Tried to add something to members twice!");
      EClass->members.insert(std::make_pair(std::make_pair(dfs_in, dfs_out), V));
      valueToClass[V] = EClass;
      // See if we destroyed the class or need to swap leaders
      if (VClass->members.empty() && VClass != InitialClass) {
	if (VClass->expression) {
	  VClass->dead = true;
	  expressionToClass.erase(VClass->expression);
	  memoryExpressionToClass.erase(VClass->expression);
	}
	// delete VClass;
      } else if (VClass->leader == V) {
	VClass->leader = VClass->members.begin()->second;
	for (std::set<std::pair<std::pair<int, int>, Value* > >::iterator LI = VClass->members.begin(),
	       LE = VClass->members.end();
	     LI != LE; ++LI) {
	  if (Instruction *I = dyn_cast<Instruction>(LI->second))
	    touchedInstructions.insert(I);
	  changedValues.insert(LI->second);
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
  // I also can't see why you would need to touch any instructions that aren't PHI
  // nodes.  Because we don't use predicates right now, they are the ones whose
  // value could have changed as a result of a new edge becoming live,
  // and any changes to their value should propagate appropriately
  // through the rest of the block.
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

void GVN::EnterScope(BasicBlock *BB) {
  DEBUG(dbgs() << "Entering: " << BB->getName() << '\n');
  ScopeType *Scope = new ScopeType(availableTable);
  ScopeMap[BB] = Scope;
}

void GVN::ExitScope(BasicBlock *BB) {
  DEBUG(dbgs() << "Exiting: " << BB->getName() << '\n');
  DenseMap<BasicBlock*, ScopeType*>::iterator SI = ScopeMap.find(BB);
  assert(SI != ScopeMap.end());
  ScopeMap.erase(SI);
  delete SI->second;
}

// ExitScopeIfDone - Destroy scope for the BB that corresponds to the given
/// dominator tree node if its a leaf or all of its children are done. Walk
/// up the dominator tree to destroy ancestors which are now done.
void GVN::ExitScopeIfDone(DomTreeNode *Node,
			  DenseMap<DomTreeNode*, unsigned> &OpenChildren,
			  DenseMap<DomTreeNode*, DomTreeNode*> &ParentMap) {
  if (OpenChildren[Node])
    return;

  // Pop scope.
  ExitScope(Node->getBlock());

  // Now traverse upwards to pop ancestors whose offsprings are all done.
  while (DomTreeNode *Parent = ParentMap[Node]) {
    unsigned Left = --OpenChildren[Parent];
    if (Left != 0)
      break;
    ExitScope(Parent->getBlock());
    Node = Parent;
  }
}

void GVN::ProcessTerminatorInst(TerminatorInst *TI){
  BranchInst *BR = dyn_cast<BranchInst>(TI);
  if (BR && BR->isConditional()) {
    Value *Cond = BR->getCondition();
    // See if it was already simplified for us. This should be the
    // normal cases for ones we can remove
    if (isa<Constant>(Cond)) {
      return;
    }
    // Otherwise
    CongruenceClass *CondClass = valueToClass[Cond];
    if (CondClass && CondClass->expression) {
      assert (!isa<ConstantExpression>(CondClass->expression) && "Should have been folded already");
    }
  }
}

void GVN::ReplaceInstruction(Instruction *I, Value *V, CongruenceClass *ReplClass) {
  assert (ReplClass->leader != I && "About to accidentally remove our leader");
  if (V->getType() != I->getType()) {
    Instruction *insertPlace = I;
    if (isa<PHINode>(I))
      insertPlace = I->getParent()->getFirstNonPHI();
    V = CoerceAvailableValueToLoadType(V, I->getType(), insertPlace, *TD);
    assert(V && "Should have been able to coerce types!");
  }
  DEBUG(dbgs() << "Replacing " << *I << " with " << *V << "\n");
  I->replaceAllUsesWith(V);
  // Remove the old instruction from the class member list, so the
  // member size is correct for PRE.
  DomTreeNode *DTN = DT->getNode(I->getParent());
  int dfs_in = DTN->getDFSNumIn();
  int dfs_out = DTN->getDFSNumOut();
  
  ReplClass->members.erase(std::make_pair(std::make_pair(dfs_in, dfs_out), I));
  // We save the actual erasing to avoid invalidating memory
  // dependencies until we are done with everything.
  UpdateMemDepInfo(MD, I, V);
  markInstructionForDeletion(I);
}

bool GVN::ProcessBlock(BasicBlock *BB) {
  bool Changed = false;
  if (!reachableBlocks.count(BB)) {
    DEBUG(dbgs() << "Skipping unreachable block " << getBlockName(BB) << " during elimination\n");
    return false;
  }

  for (BasicBlock::iterator II = BB->begin(), IE = BB->end();  II != IE; ) {
    Instruction *I = &*II;
    ++II;
    // Terminators
    if (I->isTerminator()) {
      ProcessTerminatorInst(cast<TerminatorInst>(I));
      continue;
    }  else if (isa<StoreInst>(I)) {
      // Eliminating stores is possible, but we'd rather not do the computation
      continue;
    } else if (isa<BitCastInst>(I)) {
      continue;
    }

    bool isLoad = isa<LoadInst>(I);


    CongruenceClass *IClass = valueToClass[I];
    // We may not find a congruence class if it is, for example, a
    // bitcast instruction we just inserted
    if (!IClass)
      continue;

    if (IClass->expression && IClass->leader && IClass->leader != I) {
      if (Constant *C = dyn_cast<Constant>(IClass->leader)){
	DEBUG(dbgs() << "Found constant replacement " << *C << " for " << *I << "\n");
	ReplaceInstruction(I, C, IClass);
	continue;
      } else if (Argument *A = dyn_cast<Argument>(IClass->leader)) {
	DEBUG(dbgs() << "Found argument replacement " << *A << " for " << *I << "\n");
	ReplaceInstruction(I, A, IClass);
	continue;
      }
    }

    // Skip instructions with singleton congruence classes
    // TODO unless they are nonlocal loads.
    if (0 && IClass->members.size() <= 1)
      continue;
    Value *Result = availableTable.lookup(IClass);

    if (Result){
      if (Result != I) {
	DEBUG(dbgs() << "Found replacement " << *Result << " for " << *I << "\n");
	LoadInst *LI;
	if (I->getType() != Result->getType() && (LI = dyn_cast<LoadInst>(I))) {
	  if (LoadInst *LIR = dyn_cast<LoadInst>(Result)) {
	    int Offset = AnalyzeLoadFromClobberingLoad(LI->getType(),
						       LI->getPointerOperand(),
						       LIR,
						       *TD);
	    assert(Offset != -1 && "Should have been able to coerce load");

	    Result = GetLoadValueForLoad(LIR, Offset, LI->getType(), LI, *TD);

	  } else if (StoreInst *SIR = dyn_cast<StoreInst>(Result)) {
	    int Offset = AnalyzeLoadFromClobberingStore(LI->getType(),
							LI->getPointerOperand(),
							SIR, *TD);
	    assert(Offset != -1 && "Should have been able to coerce store");
	    Result = GetStoreValueForLoad(SIR->getValueOperand(), Offset, LI->getType(), LI, *TD);
	  }
	}
	ReplaceInstruction(I, Result, IClass);
      }
    } else {
      if (LoadInst *LI = dyn_cast<LoadInst>(I)){
	MemDepResult local_dep = MD->getDependency(LI);
	if (local_dep.isNonLocal())
	  processNonLocalLoad(LI);
      }
      availableTable.insert(IClass, I);
    }
  }
  return Changed;
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

  /// MaterializeAdjustedValue - Emit code into this block to adjust the value
  /// defined here to the specified type.  This handles various coercion cases.
  Value *MaterializeAdjustedValue(Type *LoadTy, GVN &gvn) const {
    Value *Res;
    if (isSimpleValue()) {
      Res = getSimpleValue();
      if (Res->getType() != LoadTy) {
        const TargetData *TD = gvn.getTargetData();
        assert(TD && "Need target data to handle type mismatch case");
        Res = GetStoreValueForLoad(Res, Offset, LoadTy, BB->getTerminator(),
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
        Res = GetLoadValueForLoad(Load, Offset, LoadTy, BB->getTerminator(),
                                  *TD);

        DEBUG(dbgs() << "GVN COERCED NONLOCAL LOAD:\nOffset: " << Offset << "  "
                     << *getCoercedLoadValue() << '\n'
                     << *Res << '\n' << "\n\n\n");
      }
    } else {
      const TargetData *TD = gvn.getTargetData();
      assert(TD && "Need target data to handle type mismatch case");
      Res = GetMemInstValueForLoad(getMemIntrinValue(), Offset,
                                   LoadTy, BB->getTerminator(), *TD);
      DEBUG(dbgs() << "GVN COERCED NONLOCAL MEM INTRIN:\nOffset: " << Offset
                   << "  " << *getMemIntrinValue() << '\n'
                   << *Res << '\n' << "\n\n\n");
    }
    return Res;
  }
};

} // end anonymous namespace


/// ConstructSSAForLoadSet - Given a set of loads specified by ValuesPerBlock,
/// construct SSA form, allowing us to eliminate LI.  This returns the value
/// that should be used at LI's definition site.
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

// SpeculationFailure - If we get here, we found out that this is not, after
// all, a fully-available block.  We have a problem if we speculated on this and
// used the speculation to mark other blocks as available.
SpeculationFailure:
  char &BBVal = FullyAvailableBlocks[BB];

  // If we didn't speculate on this, just return with it set to false.
  if (BBVal == 2) {
    BBVal = 0;
    return false;
  }

  // If we did speculate on this value, we could have blocks set to 1 that are
  // incorrect.  Walk the (transitive) successors of this block and mark them as
  // 0 if set to one.
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


/// processNonLocalLoad - Attempt to eliminate a load whose dependencies are
/// non-local by performing PHI construction.
bool GVN::processNonLocalLoad(LoadInst *LI) {
  // Find the non-local dependencies of the load.
  SmallVector<NonLocalDepResult, 64> Deps;
  AliasAnalysis::Location Loc = AA->getLocation(LI);
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
    if (isa<AllocaInst>(DepInst) || isMalloc(DepInst) ||
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
    // FIXME: We really _ought_ to insert these value numbers into their
    // parent's availability map.  However, in doing so, we risk getting into
    // ordering issues.  If a block hasn't been processed yet, we would be
    // marking a value as AVAIL-IN, which isn't what we intend.
    // VN.lookup_or_add(NewInsts[i]);
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
  }

  // Perform PHI construction.
  Value *V = ConstructSSAForLoadSet(LI, ValuesPerBlock, *this);
  LI->replaceAllUsesWith(V);
  if (isa<PHINode>(V))
    V->takeName(LI);
  UpdateMemDepInfo(MD, LI, V);
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
    SplitCriticalEdge(Edge.first, Edge.second, this);
  } while (!toSplit.empty());
  if (MD) MD->invalidateCachedPredecessors();
  return true;
}

// uint32_t GVN::nextCongruenceNum = 0;

/// runOnFunction - This is the main transformation entry point for a function.
bool GVN::runOnFunction(Function& F) {
  DT = &getAnalysis<DominatorTree>();
  PDT = &getAnalysis<PostDominatorTree>();

  TD = getAnalysisIfAvailable<TargetData>();
  TLI = &getAnalysis<TargetLibraryInfo>();

  // Split all critical edges to ensure maximal removal
  splitCriticalEdges();

  DenseMap<DomTreeNode*, unsigned> OpenChildren;
  DenseMap<DomTreeNode*, DomTreeNode*> ParentMap;
  SmallVector<DomTreeNode*, 32> Scopes;
  SmallVector<DomTreeNode*, 8> Worklist;

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
  // Ensure we don't end up resizing the expressionToClass map, as
  // that can be quite expensive. At most, we have one expression per
  // instruction.
  expressionToClass.resize(ICount);
  memoryExpressionToClass.resize(ICount * 8);
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
  std::set<std::pair<std::pair<int, int>, Value* > > InitialValues;
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)  {
    DomTreeNode *DTN = DT->getNode(FI);
    // In the unreachable block case, just ignore things
    if (!DTN)
      continue;
    
    int dfs_in = DTN->getDFSNumIn();
    int dfs_out = DTN->getDFSNumOut();
    for (BasicBlock::iterator BI = FI->begin(), BE = FI->end(); BI != BE; ++BI) {
      InitialValues.insert(std::make_pair(std::make_pair(dfs_in, dfs_out), BI));
    }
  }
  InitialClass = new CongruenceClass();
  for (std::set<std::pair<std::pair<int, int>, Value* > >::iterator LI = InitialValues.begin(), LE = InitialValues.end();
       LI != LE; ++LI)
    valueToClass[LI->second] = InitialClass;
  InitialClass->members.swap(InitialValues);
  congruenceClass.push_back(InitialClass);
  if (!noLoads)
    {
      MD = &getAnalysis<MemoryDependenceAnalysis>();
      AA = &getAnalysis<AliasAnalysis>();
    }
  else
    {
      MD = NULL;
      AA = NULL;
    }



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
      DomTreeNode *DTN = DT->getNode(*RI);
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
	    DEBUG(dbgs() << "GVN removed: " << *I << '\n');
	    UpdateMemDepInfo(MD, I, V);
	    markInstructionForDeletion(I);
	    ++NumGVNSimpl;
	    continue;
	  }

	  // if (I->use_empty()) {
	  //   UpdateMemDepInfo(MD, I, NULL);
	  //   markInstructionForDeletion(I);
	  //   continue;
	  // }

	  if (processedCount.count(I) == 0) {
	    processedCount.insert(std::make_pair(I, 1));
	  } else {
	    processedCount[I] += 1;
	    assert(processedCount[I] < 100 && "Seem to have processed the same instruction a lot");
	  }

          if (!I->isTerminator()) {
	    Expression *Symbolized = performSymbolicEvaluation(I, *RI);
            performCongruenceFinding(I, Symbolized, DTN);
          } else {
            processOutgoingEdges(dyn_cast<TerminatorInst>(I));
          }
        }
      }
    }
  }

  // Perform dominator based elimination.

  Worklist.push_back(DT->getRootNode());
  do {
    DomTreeNode *Node;
    Node = Worklist.pop_back_val();
    Scopes.push_back(Node);
    const std::vector<DomTreeNode*> &Children = Node->getChildren();
    unsigned NumChildren = Children.size();
    OpenChildren[Node] = NumChildren;
    for (unsigned i = 0; i != NumChildren; ++i) {
      DomTreeNode *Child = Children[i];
      ParentMap[Child] = Node;
      Worklist.push_back(Child);
    }
  } while (!Worklist.empty());
#if 0
 for (unsigned i = 0, e = Scopes.size(); i != e; ++i) {
    DomTreeNode *Node = Scopes[i];
    BasicBlock *BB = Node->getBlock();
    EnterScope(BB);
    Changed |= ProcessBlock(BB);
    // If it's a leaf node, it's done. Traverse upwards to pop ancestors.
    ExitScopeIfDone(Node, OpenChildren, ParentMap);
  }
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI){
    BasicBlock *BB = FI;
    if (!reachableBlocks.count(BB)) {
      DEBUG(dbgs() << "We believe block " << getBlockName(BB) << " is unreachable\n");
      DeleteInstructionInBlock(BB);
      Changed = true;
    }
  }

  //TODO: Perform PRE.
  for (DenseSet<Instruction*>::iterator DI = instrsToErase_.begin(), DE = instrsToErase_.end(); DI != DE;) {
    Instruction *toErase = *DI;
    ++DI;
    // if (!toErase->use_empty()) {
    //   for (Value::use_iterator UI = toErase->use_begin(), UE = toErase->use_end();
    // 	 UI != UE; ++UI) {
    // 	assert (instrsToErase_.count(cast<Instruction>(*UI)) && "trying to removing something without also deleting it's uses");
    //   }
    // }
    
    // toErase->eraseFromParent();
  }
#else

  // This is a mildly crazy eliminator. The normal way to eliminate is
  // to walk the dominator tree in order, keeping track of available
  // values, and eliminating them.  However, we keep track of the
  // dominator tree dfs numbers in each value, and by keeping the set
  // sorted in dfs number order, we don't need to walk anything to
  // eliminate
  // Instead, we walk the congruence class members in order,
  // and eliminate the ones dominated by the last member.
  // When we find something not dominated, it becomes the new leader
  // for elimination purposes
 for (unsigned i = 0, e = congruenceClass.size(); i != e; ++i) {
   CongruenceClass *CC = congruenceClass[i];
   if (CC != InitialClass && !CC->dead) {
     int lastdfs_in = 0;
     int lastdfs_out = 0;
     Value *lastval = NULL;
     if (0 && CC->members.size() == 1)
       continue;
     if (CC->expression && CC->leader) {
       if (isa<Constant>(CC->leader) || isa<Argument>(CC->leader)) {
	 for (std::set<std::pair<std::pair<int, int>, Value*> >::const_iterator CI = CC->members.begin(), CE = CC->members.end(); CI != CE; ) {
	   Value *member = CI->second;
	   ++CI;
	   
	   if (member != CC->leader) {
	     if (isa<StoreInst>(member) || isa<BitCastInst>(member))
	       continue;
	     DEBUG(dbgs() << "Found replacement " << *(CC->leader) << " for " << *member << "\n");
	     ReplaceInstruction(cast<Instruction>(CI->second), CC->leader, CC);
	   }
	 }
       } else {
	 for (std::set<std::pair<std::pair<int, int>, Value*> >::const_iterator CI = CC->members.begin(), CE = CC->members.end(); CI != CE;) {
	   int currdfs_in = CI->first.first;
	   int currdfs_out = CI->first.second;
	   Value *member = CI->second;
	   ++CI;
	   
	   if (isa<StoreInst>(member) || isa<BitCastInst>(member))
	     continue;
	   DEBUG(dbgs() << "Last DFS numbers are (" << lastdfs_in << "," << lastdfs_out << ")\n");
	   DEBUG(dbgs() << "Current DFS numbers are (" << currdfs_in << "," << currdfs_out <<")\n");
	   // Walk along, processing members who are dominated by each other.
	   if (lastval == NULL || !(currdfs_in >= lastdfs_in && currdfs_out <= lastdfs_out)) {
	     lastval = member;
	     lastdfs_in = currdfs_in;
	     lastdfs_out = currdfs_out;
	   } else {
	     DEBUG(dbgs() << "Found replacement " << *lastval << " for " << *member << "\n");
	     ReplaceInstruction(cast<Instruction>(member), lastval, CC);
	   }
	 } 
       }
     }
   }
 }
 
       
#endif
 ScopeMap.clear();
 Scopes.clear();
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
  rpoBlockNumbers.clear();
  rpoInstructionNumbers.clear();
  rpoInstructionStartEnd.clear();
  rpoToBlock.clear();
  reachableBlocks.clear();
  reachableEdges.clear();
  touchedInstructions.clear();
  touchedBlocks.clear();
  processedCount.clear();
  expressionAllocator.Reset();
  leaderTable.clear();
  tableAllocator.Reset();
  instrsToErase_.clear();
  return Changed;
}
