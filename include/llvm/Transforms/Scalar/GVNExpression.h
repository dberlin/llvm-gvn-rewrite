//===- GVNExpression.h - GVN Expression classes -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// The header file for the GVN pass that contains expression handling
/// classes
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_SCALAR_GVNEXPRESSION_H
#define LLVM_TRANSFORMS_SCALAR_GVNEXPRESSION_H
#include "llvm/ADT/Hashing.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ArrayRecycler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace llvm {
class MemoryAccess;

namespace GVNExpression {

enum ExpressionType {
  ExpressionTypeBase,
  ExpressionTypeConstant,
  ExpressionTypeVariable,
  ExpressionTypeBasicStart,
  ExpressionTypeBasic,
  ExpressionTypeCall,
  ExpressionTypeAggregateValue,
  ExpressionTypePhi,
  ExpressionTypeLoad,
  ExpressionTypeCoercibleLoad,
  ExpressionTypeStore,
  ExpressionTypeBasicEnd
};
class Expression {

private:
  void operator=(const Expression &) = delete;
  Expression(const Expression &) = delete;

protected:
  ExpressionType EType;
  unsigned int Opcode;
  bool UsedEquivalence;

public:
  unsigned int getOpcode() const { return Opcode; }

  void setOpcode(unsigned int opcode) { Opcode = opcode; }

  ExpressionType getExpressionType() const { return EType; }
  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Expression *) { return true; }

  Expression(unsigned int o = ~2U)
      : EType(ExpressionTypeBase), Opcode(o), UsedEquivalence(false) {}
  Expression(ExpressionType etype, unsigned int o = ~2U)
      : EType(etype), Opcode(o), UsedEquivalence(false) {}

  virtual ~Expression() {}

  bool operator==(const Expression &Other) const {
    if (Opcode != Other.Opcode)
      return false;
    if (Opcode == ~0U || Opcode == ~1U)
      return true;
    // Compare etype for anything but load and store
    if (getExpressionType() != ExpressionTypeLoad &&
        getExpressionType() != ExpressionTypeStore &&
        getExpressionType() != Other.getExpressionType())
      return false;

    return equals(Other);
  }
  bool usedEquivalence() const { return UsedEquivalence; }
  void setUsedEquivalence(bool V) { UsedEquivalence = V; }

  virtual bool equals(const Expression &other) const { return true; }

  virtual hash_code getHashValue() const {
    return hash_combine(EType, Opcode, UsedEquivalence);
  }
  virtual void printInternal(raw_ostream &OS, bool printEType) const {
    if (printEType)
      OS << "etype = " << EType << ",";
    OS << "opcode = " << Opcode << ", ";
    OS << "UsedEquivalence = " << UsedEquivalence << ", ";
  }

  void print(raw_ostream &OS) const {
    OS << "{ ";
    printInternal(OS, true);
    OS << "}";
  }
  void dump() const { print(dbgs()); }
};
inline raw_ostream &operator<<(raw_ostream &OS, const Expression &E) {
  E.print(OS);
  return OS;
}

class BasicExpression : public Expression {
private:
  void operator=(const BasicExpression &) = delete;
  BasicExpression(const BasicExpression &) = delete;
  BasicExpression() = delete;
  typedef ArrayRecycler<Value *> RecyclerType;
  typedef RecyclerType::Capacity RecyclerCapacity;

protected:
  Value **Operands;
  unsigned int MaxOperands;
  unsigned int NumOperands;
  Type *ValueType;

public:
  typedef Value **op_iterator;
  typedef Value *const *const_ops_iterator;

  /// \brief Swap two operands. Used during GVN to put commutative operands in
  /// order.
  inline void swapOperands(unsigned First, unsigned Second) {
    std::swap(Operands[First], Operands[Second]);
  }
  inline Value *getOperand(unsigned N) const {
    assert(Operands && "Operands not allocated");
    assert(N < NumOperands && "Operand out of range");
    return Operands[N];
  }

  inline void setOperand(unsigned N, Value *V) {
    assert(Operands && "Operands not allocated before setting");
    assert(N < NumOperands && "Operand out of range");
    Operands[N] = V;
  }
  inline unsigned int getNumOperands() const { return NumOperands; }

  inline op_iterator ops_begin() { return Operands; }
  inline op_iterator ops_end() { return Operands + NumOperands; }
  inline const_ops_iterator ops_begin() const { return Operands; }
  inline const_ops_iterator ops_end() const { return Operands + NumOperands; }
  inline iterator_range<op_iterator> operands() {
    return iterator_range<op_iterator>(ops_begin(), ops_end());
  }

  inline iterator_range<const_ops_iterator> operands() const {
    return iterator_range<const_ops_iterator>(ops_begin(), ops_end());
  }

  inline void ops_push_back(Value *Arg) {
    assert(NumOperands < MaxOperands && "Tried to add too many operands");
    assert(Operands && "Operandss not allocated before pushing");
    Operands[NumOperands++] = Arg;
  }
  inline bool ops_empty() const { return getNumOperands() == 0; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const BasicExpression *) { return true; }
  static inline bool classof(const Expression *EB) {
    ExpressionType et = EB->getExpressionType();
    return et > ExpressionTypeBasicStart && et < ExpressionTypeBasicEnd;
  }

  void allocateOperands(RecyclerType &Recycler, BumpPtrAllocator &Allocator) {
    assert(!Operands && "Operands already allocated");
    Operands = Recycler.allocate(RecyclerCapacity::get(MaxOperands), Allocator);
  }
  void deallocateOperands(RecyclerType &Recycler) {
    Recycler.deallocate(RecyclerCapacity::get(MaxOperands), Operands);
  }

  void setType(Type *T) { ValueType = T; }

  Type *getType() const { return ValueType; }

  BasicExpression(unsigned int NumOperands)
      : BasicExpression(NumOperands, ExpressionTypeBasic) {}
  BasicExpression(unsigned int NumOperands, ExpressionType ET)
      : Expression(ET), Operands(nullptr), MaxOperands(NumOperands),
        NumOperands(0), ValueType(nullptr) {}

  virtual ~BasicExpression() {}

  virtual bool equals(const Expression &Other) const {
    const BasicExpression &OE = cast<BasicExpression>(Other);
    if (Opcode != OE.Opcode)
      return false;
    if (ValueType != OE.ValueType)
      return false;
    if (NumOperands != OE.NumOperands)
      return false;
    if (!std::equal(ops_begin(), ops_end(), OE.ops_begin()))
      return false;
    return true;
  }
  virtual void printInternal(raw_ostream &OS, bool printEType) const {
    if (printEType)
      OS << "ExpressionTypeBasic, ";

    this->Expression::printInternal(OS, false);
    OS << "operands = {";
    for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
      OS << "[" << i << "] = ";
      Operands[i]->printAsOperand(OS);
      OS << "  ";
    }
    OS << "} ";
  }

  virtual hash_code getHashValue() const {
    return hash_combine(EType, Opcode, ValueType,
                        hash_combine_range(ops_begin(), ops_end()));
  }
};
class CallExpression final : public BasicExpression {
private:
  void operator=(const CallExpression &) = delete;
  CallExpression(const CallExpression &) = delete;
  CallExpression() = delete;

protected:
  CallInst *Call;
  MemoryAccess *DefiningAccess;

public:
  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const CallExpression *) { return true; }
  static inline bool classof(const Expression *EB) {
    return EB->getExpressionType() == ExpressionTypeCall;
  }
  CallExpression(unsigned int NumOperands, CallInst *C, MemoryAccess *DA)
      : BasicExpression(NumOperands, ExpressionTypeCall), Call(C),
        DefiningAccess(DA) {}

  virtual ~CallExpression() {}

  virtual bool equals(const Expression &Other) const {
    if (!this->BasicExpression::equals(Other))
      return false;
    const CallExpression &OE = cast<CallExpression>(Other);
    if (DefiningAccess != OE.DefiningAccess)
      return false;
    return true;
  }

  virtual hash_code getHashValue() const {
    return hash_combine(this->BasicExpression::getHashValue(), DefiningAccess);
  }

  virtual void printInternal(raw_ostream &OS, bool printEType) const {
    if (printEType)
      OS << "ExpressionTypeCall, ";
    this->BasicExpression::printInternal(OS, false);
    OS << " represents call at " << Call;
  }
};
class LoadExpression : public BasicExpression {
private:
  void operator=(const LoadExpression &) = delete;
  LoadExpression(const LoadExpression &) = delete;
  LoadExpression() = delete;

protected:
  LoadInst *Load;
  MemoryAccess *DefiningAccess;
  unsigned Alignment;

  LoadExpression(enum ExpressionType EType, unsigned int NumOperands,
                 LoadInst *L, MemoryAccess *DA)
      : BasicExpression(NumOperands, EType), Load(L), DefiningAccess(DA) {
    Alignment = L ? L->getAlignment() : 0;
  }

public:
  LoadInst *getLoadInst() const { return Load; }
  void setLoadInst(LoadInst *L) { Load = L; }

  MemoryAccess *getDefiningAccess() const { return DefiningAccess; }
  void setDefiningAccess(MemoryAccess *MA) { DefiningAccess = MA; }
  unsigned getAlignment() const { return Alignment; }
  void setAlignment(unsigned Align) { Alignment = Align; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const LoadExpression *) { return true; }
  static inline bool classof(const Expression *EB) {
    return EB->getExpressionType() >= ExpressionTypeLoad &&
           EB->getExpressionType() <= ExpressionTypeCoercibleLoad;
  }

  LoadExpression(unsigned int NumOperands, LoadInst *L, MemoryAccess *DA)
      : LoadExpression(ExpressionTypeLoad, NumOperands, L, DA) {}

  virtual ~LoadExpression() {}

  virtual bool equals(const Expression &Other) const;

  virtual hash_code getHashValue() const {
    return hash_combine(Opcode, ValueType, DefiningAccess,
                        hash_combine_range(ops_begin(), ops_end()));
  }

  virtual void printInternal(raw_ostream &OS, bool printEType) const {
    if (printEType)
      OS << "ExpressionTypeLoad, ";
    this->BasicExpression::printInternal(OS, false);
    OS << " represents Load at " << Load;
    OS << " with DefiningAccess " << DefiningAccess;
  }
};
class CoercibleLoadExpression final : public LoadExpression {
private:
  void operator=(const CoercibleLoadExpression &) = delete;
  CoercibleLoadExpression(const CoercibleLoadExpression &) = delete;
  CoercibleLoadExpression() = delete;

  // Offset into the value we can coerce from;
  unsigned int Offset;
  // Value we can coerce from
  Value *Src;

public:
  unsigned int getOffset() const { return Offset; }
  void setOffset(unsigned int O) { Offset = O; }
  Value *getSrc() const { return Src; }
  void setSrc(Value *S) { Src = S; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const CoercibleLoadExpression *) { return true; }
  static inline bool classof(const Expression *EB) {
    return EB->getExpressionType() == ExpressionTypeCoercibleLoad;
  }

  CoercibleLoadExpression(unsigned int NumOperands, LoadInst *L,
                          MemoryAccess *DA, unsigned int O, Value *S)
      : LoadExpression(ExpressionTypeCoercibleLoad, NumOperands, L, DA),
        Offset(O), Src(S) {}

  virtual ~CoercibleLoadExpression() {}
  virtual bool equals(const Expression &Other) const {
    // Unlike normal loads, coercible loads are equal if they have the same src,
    // offset, and type, because that is what we are going to pull the value
    // from. The rest of the load arguments don't actually matter since we've
    // already analyzed that they are "the same enough" for us to do coercion.
    if (!isa<CoercibleLoadExpression>(Other))
      return false;

    const CoercibleLoadExpression &OE = cast<CoercibleLoadExpression>(Other);
    if (ValueType != OE.ValueType)
      return false;
    if (Src != OE.Src)
      return false;
    if (Offset != OE.Offset)
      return false;

    return true;
  }
  virtual hash_code getHashValue() const {
    return hash_combine(ValueType, Offset, Src);
  }
  virtual void printInternal(raw_ostream &OS, bool printEType) const {
    if (printEType)
      OS << "ExpressionTypeCoercibleLoad, ";
    this->LoadExpression::printInternal(OS, false);
    OS << " represents CoercibleLoad at " << Load << " with Src " << Src
       << " and offset " << Offset;
  }
};

class StoreExpression final : public BasicExpression {
private:
  void operator=(const StoreExpression &) = delete;
  StoreExpression(const StoreExpression &) = delete;
  StoreExpression() = delete;

protected:
  StoreInst *Store;
  MemoryAccess *DefiningAccess;

public:
  StoreInst *getStoreInst() const { return Store; }
  MemoryAccess *getDefiningAccess() const { return DefiningAccess; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const StoreExpression *) { return true; }
  static inline bool classof(const Expression *EB) {
    return EB->getExpressionType() == ExpressionTypeStore;
  }
  StoreExpression(unsigned int NumOperands, StoreInst *S, MemoryAccess *DA)
      : BasicExpression(NumOperands, ExpressionTypeStore), Store(S),
        DefiningAccess(DA) {}

  virtual ~StoreExpression() {}

  virtual bool equals(const Expression &Other) const;

  virtual void printInternal(raw_ostream &OS, bool printEType) const {
    if (printEType)
      OS << "ExpressionTypeStore, ";
    this->BasicExpression::printInternal(OS, false);
    OS << " represents Store at " << Store;
  }

  virtual hash_code getHashValue() const {
    return hash_combine(Opcode, ValueType, DefiningAccess,
                        hash_combine_range(ops_begin(), ops_end()));
  }
};

class AggregateValueExpression final : public BasicExpression {
private:
  void operator=(const AggregateValueExpression &) = delete;
  AggregateValueExpression(const AggregateValueExpression &) = delete;
  AggregateValueExpression() = delete;

  unsigned int MaxIntOperands;
  unsigned int NumIntOperands;
  unsigned int *IntOperands;

public:
  typedef unsigned int *int_arg_iterator;
  typedef const unsigned int *const_int_arg_iterator;

  inline int_arg_iterator int_ops_begin() { return IntOperands; }
  inline int_arg_iterator int_ops_end() { return IntOperands + NumIntOperands; }
  inline const_int_arg_iterator int_ops_begin() const { return IntOperands; }
  inline const_int_arg_iterator int_ops_end() const {
    return IntOperands + NumIntOperands;
  }
  inline unsigned int int_ops_size() const { return NumIntOperands; }
  inline bool int_ops_empty() const { return NumIntOperands == 0; }
  inline void int_ops_push_back(unsigned int IntOperand) {
    assert(NumIntOperands < MaxIntOperands &&
           "Tried to add too many int operands");
    assert(IntOperands && "Operands not allocated before pushing");
    IntOperands[NumIntOperands++] = IntOperand;
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const AggregateValueExpression *) { return true; }
  static inline bool classof(const Expression *EB) {
    return EB->getExpressionType() == ExpressionTypeAggregateValue;
  }

  AggregateValueExpression(unsigned int NumOperands,
                           unsigned int NumIntOperands)
      : BasicExpression(NumOperands, ExpressionTypeAggregateValue),
        MaxIntOperands(NumIntOperands), NumIntOperands(0),
        IntOperands(nullptr) {}

  virtual ~AggregateValueExpression() {}
  virtual void allocateIntOperands(BumpPtrAllocator &Allocator) {
    assert(!IntOperands && "Operands already allocated");
    IntOperands = Allocator.Allocate<unsigned int>(MaxIntOperands);
  }

  virtual bool equals(const Expression &Other) const {
    if (!this->BasicExpression::equals(Other))
      return false;
    const AggregateValueExpression &OE = cast<AggregateValueExpression>(Other);
    if (NumIntOperands != OE.NumIntOperands)
      return false;
    if (!std::equal(int_ops_begin(), int_ops_end(), OE.int_ops_begin()))
      return false;

    return true;
  }

  virtual hash_code getHashValue() const {
    return hash_combine(this->BasicExpression::getHashValue(),
                        hash_combine_range(int_ops_begin(), int_ops_end()));
  }
  virtual void printInternal(raw_ostream &OS, bool printEType) const {
    if (printEType)
      OS << "ExpressionTypeAggregateValue, ";
    this->BasicExpression::printInternal(OS, false);
    OS << ", intoperands = {";
    for (unsigned i = 0, e = int_ops_size(); i != e; ++i) {
      OS << "[" << i << "] = " << IntOperands[i] << "  ";
    }
    OS << "}";
  }
};

class PHIExpression : public BasicExpression {
public:
  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const PHIExpression *) { return true; }
  static inline bool classof(const Expression *EB) {
    return EB->getExpressionType() == ExpressionTypePhi;
  }
  BasicBlock *getBB() const { return BB; }

  void setBB(BasicBlock *bb) { BB = bb; }

  virtual bool equals(const Expression &Other) const {
    if (!this->BasicExpression::equals(Other))
      return false;
    const PHIExpression &OE = cast<PHIExpression>(Other);
    if (BB != OE.BB)
      return false;
    return true;
  }

  PHIExpression(unsigned int NumOperands, BasicBlock *B)
      : BasicExpression(NumOperands, ExpressionTypePhi), BB(B) {}

  virtual ~PHIExpression() {}

  virtual hash_code getHashValue() const {
    return hash_combine(this->BasicExpression::getHashValue(), BB);
  }
  virtual void printInternal(raw_ostream &OS, bool printEType) const {
    if (printEType)
      OS << "ExpressionTypePhi, ";
    this->BasicExpression::printInternal(OS, false);
    OS << "bb = " << BB;
  }

private:
  void operator=(const PHIExpression &) = delete;
  PHIExpression(const PHIExpression &) = delete;
  PHIExpression() = delete;
  BasicBlock *BB;
};
class VariableExpression : public Expression {
public:
  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const VariableExpression *) { return true; }
  static inline bool classof(const Expression *EB) {
    return EB->getExpressionType() == ExpressionTypeVariable;
  }

  Value *getVariableValue() const { return VariableValue; }
  void setVariableValue(Value *V) { VariableValue = V; }
  virtual bool equals(const Expression &Other) const {
    const VariableExpression &OC = cast<VariableExpression>(Other);
    if (VariableValue != OC.VariableValue)
      return false;
    return true;
  }

  VariableExpression(Value *V)
      : Expression(ExpressionTypeVariable), VariableValue(V) {}
  virtual hash_code getHashValue() const {
    return hash_combine(EType, VariableValue->getType(), VariableValue);
  }

  virtual void printInternal(raw_ostream &OS, bool printEType) const {
    if (printEType)
      OS << "ExpressionTypeVariable, ";
    this->Expression::printInternal(OS, false);
    OS << " variable = " << *VariableValue;
  }

private:
  void operator=(const VariableExpression &) = delete;
  VariableExpression(const VariableExpression &) = delete;
  VariableExpression() = delete;

  Value *VariableValue;
};
class ConstantExpression : public Expression {
public:
  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ConstantExpression *) { return true; }
  static inline bool classof(const Expression *EB) {
    return EB->getExpressionType() == ExpressionTypeConstant;
  }
  Constant *getConstantValue() const { return ConstantValue; }

  void setConstantValue(Constant *V) { ConstantValue = V; }
  virtual bool equals(const Expression &Other) const {
    const ConstantExpression &OC = cast<ConstantExpression>(Other);
    if (ConstantValue != OC.ConstantValue)
      return false;
    return true;
  }

  ConstantExpression()
      : Expression(ExpressionTypeConstant), ConstantValue(NULL) {}

  ConstantExpression(Constant *constantValue)
      : Expression(ExpressionTypeConstant), ConstantValue(constantValue) {}
  virtual hash_code getHashValue() const {
    return hash_combine(EType, ConstantValue->getType(), ConstantValue);
  }
  virtual void printInternal(raw_ostream &OS, bool printEType) const {
    if (printEType)
      OS << "ExpressionTypeConstant, ";
    this->Expression::printInternal(OS, false);
    OS << " constant = " << *ConstantValue;
  }

private:
  void operator=(const ConstantExpression &) = delete;
  ConstantExpression(const ConstantExpression &) = delete;

  Constant *ConstantValue;
};

bool LoadExpression::equals(const Expression &Other) const {
  if (!isa<LoadExpression>(Other) && !isa<StoreExpression>(Other))
    return false;
  if (!this->BasicExpression::equals(Other))
    return false;
  if (const LoadExpression *OtherL = dyn_cast<LoadExpression>(&Other)) {
    if (DefiningAccess != OtherL->getDefiningAccess())
      return false;
  } else if (const StoreExpression *OtherS =
                 dyn_cast<StoreExpression>(&Other)) {
    if (DefiningAccess != OtherS->getDefiningAccess())
      return false;
  }

  return true;
}
bool StoreExpression::equals(const Expression &Other) const {
  if (!isa<LoadExpression>(Other) && !isa<StoreExpression>(Other))
    return false;
  if (!this->BasicExpression::equals(Other))
    return false;
  if (const LoadExpression *OtherL = dyn_cast<LoadExpression>(&Other)) {
    if (DefiningAccess != OtherL->getDefiningAccess())
      return false;
  } else if (const StoreExpression *OtherS =
                 dyn_cast<StoreExpression>(&Other)) {
    if (DefiningAccess != OtherS->getDefiningAccess())
      return false;
  }

  return true;
}
}
}

#endif
