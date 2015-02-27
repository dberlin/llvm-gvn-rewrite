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
  ExpressionTypeInsertValue,
  ExpressionTypePhi,
  ExpressionTypeLoad,
  ExpressionTypeStore,
  ExpressionTypeBasicEnd
};
class Expression {

private:
  void operator=(const Expression &); // Do not implement
  Expression(const Expression &);     // Do not implement
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

  bool operator==(const Expression &other) const {
    if (Opcode != other.Opcode)
      return false;
    if (Opcode == ~0U || Opcode == ~1U)
      return true;
    if (EType != other.EType)
      return false;
    return equals(other);
  }

  virtual bool equals(const Expression &other) const { return true; }

  virtual hash_code getHashValue() const { return hash_combine(EType, Opcode); }
  virtual void print(raw_ostream &OS) {
    OS << "{etype = " << EType << ", opcode = " << Opcode << " }";
  }
};
inline raw_ostream &operator<<(raw_ostream &OS, Expression &E) {
  E.print(OS);
  return OS;
}

class BasicExpression : public Expression {
private:
  void operator=(const BasicExpression &);  // Do not implement
  BasicExpression(const BasicExpression &); // Do not implement
protected:
  Type *ValueType;

public:
  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const BasicExpression *) { return true; }
  static inline bool classof(const Expression *EB) {
    ExpressionType et = EB->getExpressionType();
    return et > ExpressionTypeBasicStart && et < ExpressionTypeBasicEnd;
  }

  void setType(Type *T) { ValueType = T; }

  Type *getType() const { return ValueType; }

  SmallVector<Value *, 4> VarArgs;

  BasicExpression() : ValueType(NULL) { EType = ExpressionTypeBasic; }

  virtual ~BasicExpression() {}

  virtual bool equals(const Expression &other) const {
    const BasicExpression &OE = cast<BasicExpression>(other);
    if (ValueType != OE.ValueType)
      return false;
    if (VarArgs != OE.VarArgs)
      return false;

    return true;
  }
  virtual void print(raw_ostream &OS) {
    OS << "{etype = " << EType << ", opcode = " << Opcode << ", VarArgs = {";
    for (unsigned i = 0, e = VarArgs.size(); i != e; ++i) {
      OS << "[" << i << "] = " << VarArgs[i] << "  ";
    }
    OS << "}  }";
  }

  virtual hash_code getHashValue() const {
    return hash_combine(EType, Opcode, ValueType,
                        hash_combine_range(VarArgs.begin(), VarArgs.end()));
  }
};
class CallExpression : public BasicExpression {
private:
  void operator=(const CallExpression &); // Do not implement
  CallExpression(const CallExpression &); // Do not implement
protected:
  CallInst *CI;
  MemoryAccess *HeapVersion;

public:
  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const CallExpression *) { return true; }
  static inline bool classof(const Expression *EB) {
    return EB->getExpressionType() == ExpressionTypeCall;
  }
  CallExpression(CallInst *C, MemoryAccess *HV) {

    EType = ExpressionTypeCall;
    CI = C;
    HeapVersion = HV;
  }

  virtual ~CallExpression() {}

  virtual bool equals(const Expression &other) const {
    // Two calls are never the same if we don't have memory dependence info
    const CallExpression &OE = cast<CallExpression>(other);
    if (ValueType != OE.ValueType)
      return false;
    // Calls are unequal unless they have the same arguments
    if (VarArgs != OE.VarArgs)
      return false;
    if (HeapVersion != OE.HeapVersion)
      return false;
    return true;
  }

  virtual hash_code getHashValue() const {
    return hash_combine(EType, Opcode, ValueType,
                        hash_combine_range(VarArgs.begin(), VarArgs.end()));
  }

  virtual void print(raw_ostream &OS) {
    OS << "{etype = " << EType << ", opcode = " << Opcode << ", VarArgs = {";
    for (unsigned i = 0, e = VarArgs.size(); i != e; ++i) {
      OS << "[" << i << "] = " << VarArgs[i] << "  ";
    }
    OS << "}";
    OS << " represents call at " << CI << "}";
  }
};
class LoadExpression : public BasicExpression {
private:
  void operator=(const LoadExpression &); // Do not implement
  LoadExpression(const LoadExpression &); // Do not implement
protected:
  LoadInst *LI;
  MemoryAccess *HeapVersion;

public:
  LoadInst *getLoadInst() const { return LI; }

  MemoryAccess *getHeapVersion() const { return HeapVersion; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const LoadExpression *) { return true; }
  static inline bool classof(const Expression *EB) {
    return EB->getExpressionType() == ExpressionTypeLoad;
  }

  LoadExpression(LoadInst *L, MemoryAccess *HV) {
    EType = ExpressionTypeLoad;
    LI = L;
    HeapVersion = HV;
  }

  virtual ~LoadExpression() {}

  virtual bool equals(const Expression &other) const {
    const LoadExpression &OE = cast<LoadExpression>(other);
    if (VarArgs != OE.VarArgs)
      return false;
    if (HeapVersion != OE.HeapVersion)
      return false;
    return true;
  }

  virtual hash_code getHashValue() const {
    return hash_combine(EType, Opcode, HeapVersion,
                        hash_combine_range(VarArgs.begin(), VarArgs.end()));
  }
};

class StoreExpression : public BasicExpression {
private:
  void operator=(const StoreExpression &);  // Do not implement
  StoreExpression(const StoreExpression &); // Do not implement
protected:
  StoreInst *SI;
  MemoryAccess *HeapVersion;

public:
  StoreInst *getStoreInst() const { return SI; }
  MemoryAccess *getHeapVersion() const { return HeapVersion; }

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const StoreExpression *) { return true; }
  static inline bool classof(const Expression *EB) {
    return EB->getExpressionType() == ExpressionTypeStore;
  }
  StoreExpression(StoreInst *S, MemoryAccess *HV) {
    EType = ExpressionTypeStore;
    SI = S;
    HeapVersion = HV;
  }

  virtual ~StoreExpression() {}

  virtual bool equals(const Expression &other) const {
    const StoreExpression &OE = cast<StoreExpression>(other);
    if (VarArgs != OE.VarArgs)
      return false;
    if (HeapVersion != OE.HeapVersion)
      return false;
    return true;
  }

  virtual hash_code getHashValue() const {
    return hash_combine(EType, Opcode, HeapVersion,
                        hash_combine_range(VarArgs.begin(), VarArgs.end()));
  }
};

class InsertValueExpression : public BasicExpression {
private:
  void operator=(const InsertValueExpression &);        // Do not implement
  InsertValueExpression(const InsertValueExpression &); // Do not implement
public:
  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const InsertValueExpression *) { return true; }
  static inline bool classof(const Expression *EB) {
    return EB->getExpressionType() == ExpressionTypeInsertValue;
  }

  SmallVector<unsigned int, 4> intargs;

  InsertValueExpression() { EType = ExpressionTypeInsertValue; }

  virtual ~InsertValueExpression() {}

  virtual bool equals(const Expression &other) const {
    const InsertValueExpression &OE = cast<InsertValueExpression>(other);
    if (ValueType != OE.ValueType)
      return false;
    if (VarArgs != OE.VarArgs)
      return false;
    if (intargs != OE.intargs)
      return false;

    return true;
  }

  virtual hash_code getHashValue() const {
    return hash_combine(EType, Opcode, ValueType,
                        hash_combine_range(VarArgs.begin(), VarArgs.end()),
                        hash_combine_range(intargs.begin(), intargs.end()));
  }
  virtual void print(raw_ostream &OS) {
    OS << "{etype = " << EType << ", opcode = " << Opcode << ", VarArgs = {";
    for (unsigned i = 0, e = VarArgs.size(); i != e; ++i) {
      OS << "[" << i << "] = " << VarArgs[i] << "  ";
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
  BasicBlock *getBB() const { return BB; }

  void setBB(BasicBlock *bb) { BB = bb; }

  virtual bool equals(const Expression &other) const {
    const PHIExpression &OE = cast<PHIExpression>(other);
    if (BB != OE.BB)
      return false;
    if (ValueType != OE.ValueType)
      return false;
    if (VarArgs != OE.VarArgs)
      return false;
    return true;
  }

  PHIExpression() : BB(NULL) { EType = ExpressionTypePhi; }

  PHIExpression(BasicBlock *bb) : BB(bb) { EType = ExpressionTypePhi; }
  virtual ~PHIExpression() {}

  virtual hash_code getHashValue() const {
    return hash_combine(EType, BB, Opcode, ValueType,
                        hash_combine_range(VarArgs.begin(), VarArgs.end()));
  }
  virtual void print(raw_ostream &OS) {
    OS << "{etype = " << EType << ", opcode = " << Opcode << ", VarArgs = {";
    for (unsigned i = 0, e = VarArgs.size(); i != e; ++i) {
      OS << "[" << i << "] = " << VarArgs[i] << "  ";
    }
    OS << "}, bb = " << BB << "  }";
  }

private:
  void operator=(const PHIExpression &); // Do not implement
  PHIExpression(const PHIExpression &);  // Do not implement
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
  virtual bool equals(const Expression &other) const {
    const VariableExpression &OC = cast<VariableExpression>(other);
    if (VariableValue != OC.VariableValue)
      return false;
    return true;
  }

  VariableExpression()
      : Expression(ExpressionTypeVariable), VariableValue(NULL) {}

  VariableExpression(Value *variableValue)
      : Expression(ExpressionTypeVariable), VariableValue(variableValue) {}
  virtual hash_code getHashValue() const {
    return hash_combine(EType, VariableValue->getType(), VariableValue);
  }

  virtual void print(raw_ostream &OS) {
    OS << "{etype = " << EType << ", opcode = " << Opcode
       << ", variable = " << VariableValue << " }";
  }

private:
  void operator=(const VariableExpression &);     // Do not implement
  VariableExpression(const VariableExpression &); // Do not implement

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
  virtual bool equals(const Expression &other) const {
    const ConstantExpression &OC = cast<ConstantExpression>(other);
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

private:
  void operator=(const ConstantExpression &);     // Do not implement
  ConstantExpression(const ConstantExpression &); // Do not implement

  Constant *ConstantValue;
};
}
}
#endif
