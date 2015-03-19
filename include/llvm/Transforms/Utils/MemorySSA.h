//===- MemorySSA.h - Build Memory SSA ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes an interface to building/using memory SSA to walk memory
// instructions using a use/def graph

// Memory SSA class builds an SSA form that links together memory
// access instructions such loads, stores, and clobbers (atomics,
// calls, etc), so they can be walked easily.  Additionally, it does a
// trivial form of "heap versioning" Every time the memory state
// changes in the program, we generate a new heap version It generates
// MemoryDef/Uses/Phis that are overlayed on top of the existing
// instructions

// As a trivial example,
// define i32 @main() #0 {
// entry:
//   %call = call noalias i8* @_Znwm(i64 4) #2
//   %0 = bitcast i8* %call to i32*
//   %call1 = call noalias i8* @_Znwm(i64 4) #2
//   %1 = bitcast i8* %call1 to i32*
//   store i32 5, i32* %0, align 4
//   store i32 7, i32* %1, align 4
//   %2 = load i32* %0, align 4
//   %3 = load i32* %1, align 4
//   %add = add nsw i32 %2, %3
//   ret i32 %add
// }
// Will become
//  define i32 @main() #0 {
// entry:
//   ; 1 = MemoryDef(0)
//   %call = call noalias i8* @_Znwm(i64 4) #3
//   %2 = bitcast i8* %call to i32*
//   ; 2 = MemoryDef(1)
//   %call1 = call noalias i8* @_Znwm(i64 4) #3
//   %4 = bitcast i8* %call1 to i32*
//   ; 3 = MemoryDef(2)
//   store i32 5, i32* %2, align 4
//   ; 4 = MemoryDef(3)
//   store i32 7, i32* %4, align 4
//   ; MemoryUse(4)
//   %7 = load i32* %2, align 4
//   ; MemoryUse(4)
//   %8 = load i32* %4, align 4
//   %add = add nsw i32 %7, %8
//   ret i32 %add
// }
// Given this form, all the stores that could ever effect the load
// at %8 can be gotten by using the memory use associated with it,
// and walking from use to def until you hit the top of the function.

// Each def also has a list of uses
// Also note that it does not attempt any disambiguation, it is simply
// linking together the instructions.

//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_UTILS_MEMORYSSA_H
#define LLVM_TRANSFORMS_UTILS_MEMORYSSA_H
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Pass.h"
#include <list>
namespace llvm {

class BasicBlock;
class DominatorTree;
class Function;

class MemoryAccess {
public:
  enum AccessType { AccessUse, AccessDef, AccessPhi };

  // Methods for support type inquiry through isa, cast, and
  // dyn_cast
  static inline bool classof(const MemoryAccess *) { return true; }

  AccessType getAccessType() const { return AccessType; }
  virtual ~MemoryAccess() {}
  BasicBlock *getBlock() const { return Block; }

  virtual void print(raw_ostream &OS) = 0;

  typedef SmallPtrSet<MemoryAccess *, 8> UseListType;
  typedef UseListType::iterator iterator;
  typedef UseListType::const_iterator const_iterator;

  iterator use_begin() { return Uses.begin(); }
  iterator use_end() { return Uses.end(); }
  iterator_range<iterator> uses() {
    return iterator_range<iterator>(use_begin(), use_end());
  }

  const_iterator use_begin() const { return Uses.begin(); }
  const_iterator use_end() const { return Uses.end(); }
  iterator_range<const_iterator> uses() const {
    return iterator_range<const_iterator>(use_begin(), use_end());
  }

  virtual unsigned int getID() const { return 0; }

protected:
  friend class MemorySSA;
  friend class MemoryUse;
  friend class MemoryPhi;

  /// Add a use of this memory access to our list of uses.
  // Note: We depend on being able to add the same use multiple times and not
  // have it end up in our use list multiple times
  void addUse(MemoryAccess *Use) { Uses.insert(Use); }

  /// Remove a use of this memory access from our list of uses
  void removeUse(MemoryAccess *Use) { Uses.erase(Use); }
  // Return true if Use is one of the uses of this memory access
  bool findUse(MemoryAccess *Use) { return Uses.count(Use); }

  MemoryAccess(AccessType AT, BasicBlock *BB) : AccessType(AT), Block(BB) {}

private:
  MemoryAccess(const MemoryAccess &);
  void operator=(const MemoryAccess &);
  AccessType AccessType;
  BasicBlock *Block;
  UseListType Uses;
};

static inline raw_ostream &operator<<(raw_ostream &OS, MemoryAccess &MA) {
  MA.print(OS);
  return OS;
}

/// This class represents an instruction that uses memory
class MemoryUse : public MemoryAccess {
public:
  MemoryUse(MemoryAccess *DMA, Instruction *MI, BasicBlock *BB)
      : MemoryUse(DMA, AccessUse, MI, BB) {}

  MemoryAccess *getDefiningAccess() const { return DefiningAccess; }
  /// Replace our defining access with a new one.
  // This function updates use lists.
  void setDefiningAccess(MemoryAccess *DMA) {
    if (DefiningAccess != DMA) {
      if (DefiningAccess)
        DefiningAccess->removeUse(this);
      if (DMA)
        DMA->addUse(this);
    }
    DefiningAccess = DMA;
  }
  Instruction *getMemoryInst() const { return MemoryInst; }
  void setMemoryInst(Instruction *MI) { MemoryInst = MI; }

  static inline bool classof(const MemoryUse *) { return true; }
  static inline bool classof(const MemoryAccess *MA) {
    return MA->getAccessType() == AccessUse;
  }
  virtual void print(raw_ostream &OS);

protected:
  MemoryUse(MemoryAccess *DMA, enum AccessType AT, Instruction *MI,
            BasicBlock *BB)
      : MemoryAccess(AT, BB), DefiningAccess(DMA), MemoryInst(MI) {}

private:
  MemoryAccess *DefiningAccess;
  Instruction *MemoryInst;
};

/// This class represents an instruction that defines *or* clobbers memory
// All defs also have a use
class MemoryDef : public MemoryUse {
public:
  MemoryDef(MemoryAccess *DMA, Instruction *MI, BasicBlock *BB, unsigned Ver)
      : MemoryUse(DMA, AccessDef, MI, BB), ID(Ver) {}

  static inline bool classof(const MemoryDef *) { return true; }
  static inline bool classof(const MemoryUse *MA) {
    return MA->getAccessType() == AccessDef;
  }

  static inline bool classof(const MemoryAccess *MA) {
    return MA->getAccessType() == AccessDef;
  }
  virtual void print(raw_ostream &OS);
  typedef MemoryAccess **iterator;
  typedef const MemoryAccess **const_iterator;

protected:
  friend class MemorySSA;
  // For debugging only. This gets used to give memory accesses pretty numbers
  // when printing them out

  virtual unsigned int getID() const { return ID; }

private:
  const unsigned int ID;
};

/// This class represents Phis for memory accesses. These have the same
/// semantics
/// as regular Phis, except only one Phi will ever exist in a given basic block
class MemoryPhi : public MemoryAccess {
public:
  MemoryPhi(BasicBlock *BB, unsigned int NP, unsigned int Ver)
      : MemoryAccess(AccessPhi, BB), ID(Ver), NumPreds(NP) {
    Args.reserve(NumPreds);
  }
  // This is the number of actual predecessors
  unsigned int getNumPreds() { return NumPreds; }
  // This is the number of incoming values filled in right now
  // During construction, we differentiate between this and NumPreds to know
  // when the PHI node is fully constructed.
  unsigned int getNumIncomingValues() { return Args.size(); }
  void setIncomingValue(unsigned int v, MemoryAccess *MA) {
    std::pair<BasicBlock *, MemoryAccess *> &Val = Args[v];
    // We need to update use lists.  Because our Uses are not to specific
    // operands, but instead to this MemoryAccess, and because a given memory
    // access may appear multiple times in the Phi argument list, we need to be
    // careful not to remove the use of this phi, from MA, until we check to
    // make sure MA does not appear elsewhere in the Phi argument list.
    if (Val.second != MA) {
      if (Val.second) {
        bool existsElsewhere = false;
        for (unsigned i = 0, e = Args.size(); i != e; ++i) {
          if (i == v)
            continue;
          if (Args[i].second == Val.second)
            existsElsewhere = true;
        }
        if (!existsElsewhere)
          Val.second->removeUse(this);
      }
      MA->addUse(this);
      Val.second = MA;
    }
  }
  MemoryAccess *getIncomingValue(unsigned int v) { return Args[v].second; }
  void setIncomingBlock(unsigned int v, BasicBlock *BB) {
    std::pair<BasicBlock *, MemoryAccess *> &Val = Args[v];
    Val.first = BB;
  }
  BasicBlock *getIncomingBlock(unsigned int v) { return Args[v].first; }

  typedef SmallVector<std::pair<BasicBlock *, MemoryAccess *>, 8> ArgsType;
  typedef ArgsType::const_iterator const_arg_iterator;

  inline const_arg_iterator args_begin() const { return Args.begin(); }
  inline const_arg_iterator args_end() const { return Args.end(); }
  inline iterator_range<const_arg_iterator> args() const {
    return iterator_range<const_arg_iterator>(args_begin(), args_end());
  }

  static inline bool classof(const MemoryPhi *) { return true; }
  static inline bool classof(const MemoryAccess *MA) {
    return MA->getAccessType() == AccessPhi;
  }

  virtual void print(raw_ostream &OS);

protected:
  friend class MemorySSA;
  // MemorySSA currently cannot handle edge additions or deletions (but can
  // handle direct replacement).  This is protected to ensure people don't try.

  void addIncoming(MemoryAccess *MA, BasicBlock *BB) {
    Args.push_back(std::make_pair(BB, MA));
    MA->addUse(this);
  }

  // For debugging only. This gets used to give memory accesses pretty numbers
  // when printing them out
  unsigned int getID() const { return ID; }

private:
  // For debugging only
  const unsigned ID;
  unsigned NumPreds;
  ArgsType Args;
};

class MemorySSAWalker;
class MemorySSA {

private:
  AliasAnalysis *AA;
  DominatorTree *DT;
  Function &F;

  // Memory SSA mappings
  DenseMap<const Value *, MemoryAccess *> InstructionToMemoryAccess;
  MemoryAccess *LiveOnEntryDef;

  // Memory SSA building info
  typedef DenseMap<BasicBlock *, std::list<MemoryAccess *> *> AccessMap;
  unsigned int nextID;
  bool builtAlready;
  std::vector<MemoryAccess *> AllAccesses;

public:
  MemorySSA(Function &);
  ~MemorySSA();
  // Memory SSA related stuff
  void buildMemorySSA(AliasAnalysis *, DominatorTree *, MemorySSAWalker *);
  // Given a memory using/clobbering/etc instruction, get the
  // MemorySSA access associaed with it.  If you hand it a basic block
  // it will give you the memory phi node that exists for that block,
  // if there is one.
  MemoryAccess *getMemoryAccess(const Value *) const;
  void dump(Function &);
  void print(raw_ostream &) const;
  inline bool isLiveOnEntryDef(const MemoryAccess *MA) const {
    return MA == LiveOnEntryDef;
  }
  inline const MemoryAccess *getLiveOnEntryDef() const {
    assert(LiveOnEntryDef && "Live on entry def not initialized yet");
    return LiveOnEntryDef;
  }

protected:
  // Used by memory ssa annotater, dumpers, and wrapper pass
  friend class MemorySSAAnnotatedWriter;
  friend class MemorySSAWrapperPass;
  void verifyDefUses(Function &F);
  void verifyDomination(Function &F);

private:
  void verifyUseInDefs(MemoryAccess *, MemoryAccess *);

  void
  determineInsertionPoint(Function &F, AccessMap &BlockAccesses,
                          const SmallPtrSetImpl<BasicBlock *> &DefiningBlocks);
  void computeDomLevels(DenseMap<DomTreeNode *, unsigned> &DomLevels);
  void markUnreachableAsLiveOnEntry(AccessMap &BlockAccesses, BasicBlock *BB);

  struct RenamePassData {
    BasicBlock *BB;
    BasicBlock *Pred;
    MemoryAccess *MA;

    RenamePassData() : BB(nullptr), Pred(nullptr), MA(nullptr) {}

    RenamePassData(BasicBlock *B, BasicBlock *P, MemoryAccess *M)
        : BB(B), Pred(P), MA(M) {}
    void swap(RenamePassData &RHS) {
      std::swap(BB, RHS.BB);
      std::swap(Pred, RHS.Pred);
      std::swap(MA, RHS.MA);
    }
  };

  void renamePass(BasicBlock *BB, BasicBlock *Pred, MemoryAccess *IncomingVal,
                  AccessMap &BlockAccesses,
                  std::vector<RenamePassData> &Worklist,
                  SmallPtrSet<BasicBlock *, 16> &Visited, MemorySSAWalker *);
};

// This pass does eager building of MemorySSA. It is used by the tests to be
// able to build, dump, and verify Memory SSA. It should not really be used in
// normal cases, you should use MemorySSALazyPass instead.

class MemorySSAWrapperPass : public FunctionPass {
public:
  MemorySSAWrapperPass();

  static char ID;
  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &) override;
  void releaseMemory() override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  static void registerOptions();
  MemorySSA &getMSSA() { return *MSSA; }

private:
  bool DumpMemorySSA;
  bool VerifyMemorySSA;

  MemorySSA *MSSA;
  MemorySSAWalker *Walker;
};

class MemorySSALazy : public FunctionPass {
public:
  MemorySSALazy();

  static char ID;
  bool runOnFunction(Function &) override;
  void releaseMemory() override;
  MemorySSA &getMSSA() { return *MSSA; }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

private:
  MemorySSA *MSSA;
};

/// This is the generic walker interface for walkers of MemorySSA.  Walkers are
/// used to be able to further disambiguate the def-use chains MemorySSA gives
/// you.
class MemorySSAWalker {
public:
  MemorySSAWalker(MemorySSA *);
  virtual ~MemorySSAWalker() {}

  typedef SmallVector<MemoryAccess *, 8> MemoryAccessSet;
  // Given a memory defining/using/clobbering instruction, calling this will
  // give you the nearest dominating clobbering Memory Access (by skipping
  // non-aliasing def links).  Note that this will return a single accesses, and
  // it must dominate the Instruction, so if an argument of a MemoryPhi node
  // clobbers Instruction, it will return the memoryphi node, *not* the
  // argument.
  virtual MemoryAccess *getClobberingMemoryAccess(const Instruction *) = 0;

  // Given a memory defining/using/clobbering instruction, calling this will
  // give you the set of nearest clobbering accesses.  They are not guaranteed
  // to dominate an instruction.  The main difference between this and the above
  // is that if a phi's argument clobbers the instruction, the set will include
  // the nearest clobbering access of all of phi arguments, instead of the phi.
  // virtual MemoryAccessSet getClobberingMemoryAccesses(const Instruction *) =
  // 0;

protected:
  MemorySSA *MSSA;
};

// This walker does no alias queries, or anything else. It simply returns the
// links as they were constructed by the builder
class DoNothingMemorySSAWalker final : public MemorySSAWalker {
public:
  MemoryAccess *getClobberingMemoryAccess(const Instruction *) override;
};

// This walker does real AA walks, and caching of lookups.
class CachingMemorySSAWalker final : public MemorySSAWalker {
public:
  CachingMemorySSAWalker(MemorySSA *, AliasAnalysis *);
  virtual ~CachingMemorySSAWalker();
  MemoryAccess *getClobberingMemoryAccess(const Instruction *) override;

protected:
  struct MemoryQuery;
  MemoryAccess *doCacheLookup(const MemoryAccess *, const MemoryQuery &);
  void doCacheInsert(const MemoryAccess *, MemoryAccess *, const MemoryQuery &);

private:
  std::pair<MemoryAccess *, bool>
  getClobberingMemoryAccess(MemoryPhi *Phi, struct MemoryQuery &);
  std::pair<MemoryAccess *, bool>
  getClobberingMemoryAccess(MemoryAccess *, struct MemoryQuery &);

  DenseMap<std::pair<const MemoryAccess *, AliasAnalysis::Location>,
           MemoryAccess *> CachedClobberingAccess;
  DenseMap<const MemoryAccess *, MemoryAccess *> CachedClobberingCall;
  AliasAnalysis *AA;
};
}

#endif
