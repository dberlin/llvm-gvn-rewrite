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
#include <algorithm>
#include "llvm/ADT/Hashing.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ArrayRecycler.h"
#include "llvm/Support/raw_ostream.h"

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

public:
  unsigned int getOpcode() const { return Opcode; }

  void setOpcode(unsigned int opcode) { Opcode = opcode; }

  ExpressionType getExpressionType() const { return EType; }
  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Expression *) { return true; }

  Expression(unsigned int o = ~2U) : EType(ExpressionTypeBase), Opcode(o) {}
  Expression(ExpressionType etype, unsigned int o = ~2U)
      : EType(etype), Opcode(o) {}

  virtual ~Expression() {}

  bool operator==(const Expression &Other) const {
    if (Opcode != Other.Opcode)
      return false;
    if (Opcode == ~0U || Opcode == ~1U)
      return true;
    if (EType != Other.EType)
      return false;
    return equals(Other);
  }

  virtual bool equals(const Expression &other) const { return true; }

  virtual hash_code getHashValue() const { return hash_combine(EType, Opcode); }
  virtual void printInternal(raw_ostream &OS, bool printEType) {
    if (printEType)
      OS << "etype = " << EType << ",";
    OS << "opcode = " << Opcode << ", ";
  }

  void print(raw_ostream &OS) {
    OS << "{ ";
    printInternal(OS, true);
    OS << "}";
  }
};
inline raw_ostream &operator<<(raw_ostream &OS, Expression &E) {
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
  unsigned int MaxArgs;
  unsigned int NumArgs;
  Type *ValueType;

public:
  Value **Args;
  typedef Value **arg_iterator;
  typedef Value *const *const_arg_iterator;

  inline arg_iterator args_begin() { return Args; }
  inline arg_iterator args_end() { return Args + NumArgs; }
  inline const_arg_iterator args_begin() const { return Args; }
  inline const_arg_iterator args_end() const { return Args + NumArgs; }
  inline iterator_range<arg_iterator> arguments() {
    return iterator_range<arg_iterator>(args_begin(), args_end());
  }

  inline iterator_range<const_arg_iterator> arguments() const {
    return iterator_range<const_arg_iterator>(args_begin(), args_end());
  }

  inline unsigned int args_size() const { return NumArgs; }
  inline void args_push_back(Value *Arg) {
    assert(NumArgs < MaxArgs && "Tried to add too many args");
    assert(Args && "Args not allocated before pushing");
    Args[NumArgs++] = Arg;
  }
  inline bool args_empty() const { return args_size() == 0; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const BasicExpression *) { return true; }
  static inline bool classof(const Expression *EB) {
    ExpressionType et = EB->getExpressionType();
    return et > ExpressionTypeBasicStart && et < ExpressionTypeBasicEnd;
  }

  void allocateArgs(RecyclerType &Recycler, BumpPtrAllocator &Allocator) {
    assert(!Args && "Args already allocated");
    Args = Recycler.allocate(RecyclerCapacity::get(MaxArgs), Allocator);
  }
  void deallocateArgs(RecyclerType &Recycler) {
    Recycler.deallocate(RecyclerCapacity::get(MaxArgs), Args);
  }

  void setType(Type *T) { ValueType = T; }

  Type *getType() const { return ValueType; }

  BasicExpression(unsigned int NumArgs)
      : BasicExpression(NumArgs, ExpressionTypeBasic) {}
  BasicExpression(unsigned int NumArgs, ExpressionType ET)
      : Expression(ET), MaxArgs(NumArgs), NumArgs(0), ValueType(nullptr),
        Args(nullptr) {}

  virtual ~BasicExpression() {}

  virtual bool equals(const Expression &Other) const {
    const BasicExpression &OE = cast<BasicExpression>(Other);
    if (ValueType != OE.ValueType)
      return false;
    if (NumArgs != OE.NumArgs)
      return false;
    if (!std::equal(args_begin(), args_end(), OE.args_begin()))
      return false;
    return true;
  }
  virtual void printInternal(raw_ostream &OS, bool printEType) {
    if (printEType)
      OS << "ExpressionTypeBasic, ";

    this->Expression::printInternal(OS, false);
    OS << "args = {";
    for (unsigned i = 0, e = args_size(); i != e; ++i) {
      OS << "[" << i << "] = " << Args[i] << "  ";
    }
    OS << "} ";
  }

  virtual hash_code getHashValue() const {
    return hash_combine(EType, Opcode, ValueType,
                        hash_combine_range(args_begin(), args_end()));
  }
};
class CallExpression : public BasicExpression {
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
  CallExpression(unsigned int NumArgs, CallInst *C, MemoryAccess *DA)
      : BasicExpression(NumArgs, ExpressionTypeCall), Call(C),
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

  virtual void printInternal(raw_ostream &OS, bool printEType) {
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

  LoadExpression(enum ExpressionType EType, unsigned int NumArgs, LoadInst *L,
                 MemoryAccess *DA)
      : BasicExpression(NumArgs, EType), Load(L), DefiningAccess(DA) {}

public:
  LoadInst *getLoadInst() const { return Load; }

  MemoryAccess *getDefiningAccess() const { return DefiningAccess; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const LoadExpression *) { return true; }
  static inline bool classof(const Expression *EB) {
    return EB->getExpressionType() >= ExpressionTypeLoad &&
           EB->getExpressionType() <= ExpressionTypeCoercibleLoad;
  }

  LoadExpression(unsigned int NumArgs, LoadInst *L, MemoryAccess *DA)
      : LoadExpression(ExpressionTypeLoad, NumArgs, L, DA) {}

  virtual ~LoadExpression() {}

  virtual bool equals(const Expression &Other) const {
    if (!this->BasicExpression::equals(Other))
      return false;
    const LoadExpression &OE = cast<LoadExpression>(Other);
    if (DefiningAccess != OE.DefiningAccess)
      return false;
    return true;
  }

  virtual hash_code getHashValue() const {
    return hash_combine(this->BasicExpression::getHashValue(), DefiningAccess);
  }

  virtual void printInternal(raw_ostream &OS, bool printEType) {
    if (printEType)
      OS << "ExpressionTypeLoad, ";
    this->BasicExpression::printInternal(OS, false);
    OS << " represents Load at " << Load;
  }
};
class CoercibleLoadExpression : public LoadExpression {
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

  CoercibleLoadExpression(unsigned int NumArgs, LoadInst *L, MemoryAccess *DA,
                          unsigned int O, Value *S)
      : LoadExpression(ExpressionTypeCoercibleLoad, NumArgs, L, DA), Offset(O),
        Src(S) {}

  virtual ~CoercibleLoadExpression() {}
  virtual bool equals(const Expression &Other) const {
    // Unlike normal loads, coercible loads are equal if they have the same src,
    // offset, and type, because that is what we are going to pull the value
    // from. The rest of the load arguments don't actually matter since we've
    // already analyzed that they are "the same enough" for us to do coercion.
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
  virtual void printInternal(raw_ostream &OS, bool printEType) {
    if (printEType)
      OS << "ExpressionTypeCoercibleLoad, ";
    this->LoadExpression::printInternal(OS, false);
    OS << " represents CoercibleLoad at " << Load << " with Src " << Src
       << " and offset " << Offset;
  }
};

class StoreExpression : public BasicExpression {
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
  StoreExpression(unsigned int NumArgs, StoreInst *S, MemoryAccess *DA)
      : BasicExpression(NumArgs, ExpressionTypeStore), Store(S),
        DefiningAccess(DA) {}

  virtual ~StoreExpression() {}

  virtual bool equals(const Expression &Other) const {
    if (!this->BasicExpression::equals(Other))
      return false;
    const StoreExpression &OE = cast<StoreExpression>(Other);
    if (DefiningAccess != OE.DefiningAccess)
      return false;
    return true;
  }

  virtual void printInternal(raw_ostream &OS, bool printEType) {
    if (printEType)
      OS << "ExpressionTypeStore, ";
    this->BasicExpression::printInternal(OS, false);
    OS << " represents Store at " << Store;
  }

  virtual hash_code getHashValue() const {
    return hash_combine(this->BasicExpression::getHashValue(), DefiningAccess);
  }
};

class AggregateValueExpression : public BasicExpression {
private:
  void operator=(const AggregateValueExpression &) = delete;
  AggregateValueExpression(const AggregateValueExpression &) = delete;
  AggregateValueExpression() = delete;

  unsigned int MaxIntArgs;
  unsigned int NumIntArgs;
  unsigned int *IntArgs;

public:
  typedef unsigned int *int_arg_iterator;
  typedef const unsigned int *const_int_arg_iterator;

  inline int_arg_iterator int_args_begin() { return IntArgs; }
  inline int_arg_iterator int_args_end() { return IntArgs + NumIntArgs; }
  inline const_int_arg_iterator int_args_begin() const { return IntArgs; }
  inline const_int_arg_iterator int_args_end() const {
    return IntArgs + NumIntArgs;
  }
  inline unsigned int int_args_size() const { return NumIntArgs; }
  inline bool int_args_empty() const { return NumIntArgs == 0; }
  inline void int_args_push_back(unsigned int IntArg) {
    assert(NumIntArgs < MaxIntArgs && "Tried to add too many int args");
    assert(IntArgs && "Args not allocated before pushing");
    IntArgs[NumIntArgs++] = IntArg;
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const AggregateValueExpression *) { return true; }
  static inline bool classof(const Expression *EB) {
    return EB->getExpressionType() == ExpressionTypeAggregateValue;
  }

  AggregateValueExpression(unsigned int NumArgs, unsigned int NumIntArgs)
      : BasicExpression(NumArgs, ExpressionTypeAggregateValue),
        MaxIntArgs(NumIntArgs), NumIntArgs(0), IntArgs(nullptr) {}

  virtual ~AggregateValueExpression() {}
  virtual void allocateIntArgs(BumpPtrAllocator &Allocator) {
    assert(!IntArgs && "Args already allocated");
    IntArgs = Allocator.Allocate<unsigned int>(MaxIntArgs);
  }

  virtual bool equals(const Expression &Other) const {
    if (!this->BasicExpression::equals(Other))
      return false;
    const AggregateValueExpression &OE = cast<AggregateValueExpression>(Other);
    if (NumIntArgs != OE.NumIntArgs)
      return false;
    if (!std::equal(int_args_begin(), int_args_end(), OE.int_args_begin()))
      return false;

    return true;
  }

  virtual hash_code getHashValue() const {
    return hash_combine(this->BasicExpression::getHashValue(),
                        hash_combine_range(int_args_begin(), int_args_end()));
  }
  virtual void printInternal(raw_ostream &OS, bool printEType) {
    if (printEType)
      OS << "ExpressionTypeAggregateValue, ";
    this->BasicExpression::printInternal(OS, false);
    OS << ", intargs = {";
    for (unsigned i = 0, e = int_args_size(); i != e; ++i) {
      OS << "[" << i << "] = " << IntArgs[i] << "  ";
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

  PHIExpression(unsigned int NumArgs, BasicBlock *B)
      : BasicExpression(NumArgs, ExpressionTypePhi), BB(B) {}

  virtual ~PHIExpression() {}

  virtual hash_code getHashValue() const {
    return hash_combine(this->BasicExpression::getHashValue(), BB);
  }
  virtual void printInternal(raw_ostream &OS, bool printEType) {
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

  virtual void printInternal(raw_ostream &OS, bool printEType) {
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
  virtual void printInternal(raw_ostream &OS, bool printEType) {
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
}
}
#endif
