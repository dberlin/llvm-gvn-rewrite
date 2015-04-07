//===- MemorySSA.h - Build Memory SSA ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// \file
// \brief This file exposes an interface to building/using memory SSA to
// walk memory instructions using a use/def graph

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
#include "llvm/ADT/ilist_node.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Pass.h"
#include <list>
namespace llvm {

class BasicBlock;
class DominatorTree;
class Function;

// \brief The base for all memory accesses. All memory accesses in a
// block are linked together using an intrusive list.
class MemoryAccess : public ilist_node<MemoryAccess> {
public:
  enum AccessType { AccessUse, AccessDef, AccessPhi };

  // Methods for support type inquiry through isa, cast, and
  // dyn_cast
  static inline bool classof(const MemoryAccess *) { return true; }

  virtual ~MemoryAccess();
  BasicBlock *getBlock() const { return Block; }

  /// \brief Get the instruction that this MemoryAccess represents.
  /// This may be null in the case of phi nodes.
  virtual Instruction *getMemoryInst() const = 0;

  /// \brief Set the instruction that this MemoryUse represents.
  virtual void setMemoryInst(Instruction *MI) = 0;

  /// \brief Get the access that produces the memory state used by this access.
  virtual MemoryAccess *getDefiningAccess() const = 0;

  /// \brief Replace our defining access with a new one.
  ///
  /// This function updates use lists.
  virtual void setDefiningAccess(MemoryAccess *) = 0;

  virtual void print(raw_ostream &OS) const {};
  virtual void dump() const;

  typedef SmallPtrSet<MemoryAccess *, 8> UseListType;
  typedef UseListType::iterator iterator;
  typedef UseListType::const_iterator const_iterator;

  unsigned use_size() const { return Uses.size(); }
  bool use_empty() const { return Uses.empty(); }
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

protected:
  friend class MemorySSA;
  friend class MemoryUse;
  friend class MemoryDef;
  friend class MemoryPhi;
  AccessType getAccessType() const { return AccessType; }

  /// \brief Add a use of this memory access to our list of uses.
  ///
  /// Note: We depend on being able to add the same use multiple times and not
  /// have it end up in our use list multiple times.
  void addUse(MemoryAccess *Use) { Uses.insert(Use); }

  /// \brief Remove a use of this memory access from our list of uses.
  void removeUse(MemoryAccess *Use) { Uses.erase(Use); }

  /// \brief Return true if \p Use is one of the uses of this memory access.
  bool findUse(MemoryAccess *Use) { return Uses.count(Use); }

  MemoryAccess(AccessType AT, BasicBlock *BB) : AccessType(AT), Block(BB) {}

  /// \brief Used internally to give IDs to MemoryAccesses for printing
  virtual unsigned int getID() const { return 0; }

private:
  MemoryAccess(const MemoryAccess &);
  void operator=(const MemoryAccess &);
  AccessType AccessType;
  BasicBlock *Block;
  UseListType Uses;
};

template <>
struct ilist_traits<MemoryAccess> : public ilist_default_traits<MemoryAccess> {
  // See details of the instruction class for why this trick works
  MemoryAccess *createSentinel() const {
    return static_cast<MemoryAccess *>(&Sentinel);
  }

  static void destroySentinel(MemoryAccess *) {}

  MemoryAccess *provideInitialHead() const { return createSentinel(); }
  MemoryAccess *ensureHead(MemoryAccess *) const { return createSentinel(); }
  static void noteHead(MemoryAccess *, MemoryAccess *) {}

private:
  mutable ilist_half_node<MemoryAccess> Sentinel;
};

static inline raw_ostream &operator<<(raw_ostream &OS, MemoryAccess &MA) {
  MA.print(OS);
  return OS;
}

/// \brief Represents read-only accesses to memory
class MemoryUse : public MemoryAccess {
public:
  MemoryUse(MemoryAccess *DMA, Instruction *MI, BasicBlock *BB)
      : MemoryUse(DMA, AccessUse, MI, BB) {}

  virtual MemoryAccess *getDefiningAccess() const final {
    return DefiningAccess;
  }

  virtual void setDefiningAccess(MemoryAccess *DMA) final {
    if (DefiningAccess != DMA) {
      if (DefiningAccess)
        DefiningAccess->removeUse(this);
      if (DMA)
        DMA->addUse(this);
    }
    DefiningAccess = DMA;
  }
  virtual Instruction *getMemoryInst() const final { return MemoryInst; }

  virtual void setMemoryInst(Instruction *MI) final { MemoryInst = MI; }

  static inline bool classof(const MemoryUse *) { return true; }
  static inline bool classof(const MemoryAccess *MA) {
    return MA->getAccessType() == AccessUse;
  }
  virtual void print(raw_ostream &OS) const;

protected:
  MemoryUse(MemoryAccess *DMA, enum AccessType AT, Instruction *MI,
            BasicBlock *BB)
      : MemoryAccess(AT, BB), DefiningAccess(nullptr), MemoryInst(MI) {
    setDefiningAccess(DMA);
  }

private:
  MemoryAccess *DefiningAccess;
  Instruction *MemoryInst;
};

/// \brief Represents a read-write access to memory, whether it is real, or a
/// clobber.
///
/// Note that, in order to provide def-def chains, all defs also have a use
/// associated with them.
class MemoryDef final : public MemoryUse {
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
  virtual void print(raw_ostream &OS) const;

protected:
  friend class MemorySSA;
  // For debugging only. This gets used to give memory accesses pretty numbers
  // when printing them out

  virtual unsigned int getID() const final { return ID; }

private:
  const unsigned int ID;
};

/// \brief Represents phi nodes for memory accesses.
///
/// These have the same semantic as regular phi nodes, with the exception that
/// only one phi will ever exist in a given basic block.

class MemoryPhi final : public MemoryAccess {
public:
  MemoryPhi(BasicBlock *BB, unsigned int NP, unsigned int Ver)
      : MemoryAccess(AccessPhi, BB), ID(Ver), NumPreds(NP) {
    Args.reserve(NumPreds);
  }

  virtual Instruction *getMemoryInst() const final { return nullptr; }

  virtual void setMemoryInst(Instruction *MI) final {}

  virtual MemoryAccess *getDefiningAccess() const {
    llvm_unreachable("MemoryPhi's do not have a single defining access");
  }
  virtual void setDefiningAccess(MemoryAccess *) final {
    llvm_unreachable("MemoryPhi's do not have a single defining access");
  }

  /// \brief This is the number of actual predecessors this phi node has.
  unsigned int getNumPreds() const { return NumPreds; }

  /// \brief This is the number of incoming values currently in use
  ///
  /// During SSA construction, we differentiate between this and NumPreds to
  /// know when the PHI node is fully constructed.
  unsigned int getNumIncomingValues() const { return Args.size(); }

  /// \brief Set the memory access of argument \p v of this phi node to be \p MA
  ///
  /// This function updates use lists.
  void setIncomingValue(unsigned int v, MemoryAccess *MA) {
    std::pair<BasicBlock *, MemoryAccess *> &Val = Args[v];
    // We need to update use lists.  Because our uses are not to specific
    // operands, but instead to this MemoryAccess, and because a given memory
    // access may appear multiple times in the phi argument list, we need to be
    // careful not to remove the use of this phi, from MA, until we check to
    // make sure MA does not appear elsewhere in the phi argument list.
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

  MemoryAccess *getIncomingValue(unsigned int v) const {
    return Args[v].second;
  }
  void setIncomingBlock(unsigned int v, BasicBlock *BB) {
    std::pair<BasicBlock *, MemoryAccess *> &Val = Args[v];
    Val.first = BB;
  }
  BasicBlock *getIncomingBlock(unsigned int v) const { return Args[v].first; }

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

  virtual void print(raw_ostream &OS) const;

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
  virtual unsigned int getID() const final { return ID; }

private:
  // For debugging only
  const unsigned ID;
  unsigned NumPreds;
  ArgsType Args;
};

class MemorySSAWalker;

/// \brief Encapsulates MemorySSA, including all data associated with memory
/// accesses.
class MemorySSA {
public:
  MemorySSA(Function &);
  ~MemorySSA();

  /// \brief Build Memory SSA, and return the walker we used during building,
  /// for later reuse.  If MemorySSA is already built, just return the walker.
  MemorySSAWalker *buildMemorySSA(AliasAnalysis *, DominatorTree *);

  /// \brief Given a memory using/clobbering/etc instruction, get the MemorySSA
  /// access associaed with it.  If passed a basic block gets the memory phi
  /// node that exists for that block, if there is one.
  MemoryAccess *getMemoryAccess(const Value *) const;
  void dump(Function &);
  void print(raw_ostream &) const;

  /// \brief Return true if \p MA represents the live on entry value
  ///
  /// Loads and stores from pointer arguments and other global values may be
  /// defined by memory operations that do not occur in the current function, so
  /// they may be live on entry to the function.  MemorySSA represents such
  /// memory state by the live on entry definition, which is guaranteed to
  /// occurr before any other memory access in the function.
  inline bool isLiveOnEntryDef(const MemoryAccess *MA) const {
    return MA == LiveOnEntryDef;
  }

  inline MemoryAccess *getLiveOnEntryDef() const {
    assert(LiveOnEntryDef && "Live on entry def not initialized yet");
    return LiveOnEntryDef;
  }

  typedef ilist<MemoryAccess> AccessListType;

  /// \brief Return the list of MemoryAccess's for a given basic block.
  ///
  /// This list is not modifiable by the user.
  const AccessListType *getBlockAccesses(const BasicBlock *BB) {
    return PerBlockAccesses[BB];
  }

  /// \brief Remove a MemoryAccess from MemorySSA, including updating all
  // definitions and uses.
  void removeMemoryAccess(MemoryAccess *);

  /// \brief Replace one MemoryAccess with another, including updating all
  /// definitions and uses.
  void replaceMemoryAccess(MemoryAccess *Replacee, MemoryAccess *Replacer);

  enum InsertionPlace { Beginning, End };

  /// \brief Replace a MemoryAccess with a new access, created based on
  /// instruction \p Replacer - this does not perform generic SSA updates, so it
  /// only works if the new access dominates the old accesses uses.
  ///
  /// This version places the access at either the end or the beginning of \p
  /// Replacer's block, depending on the value of \p Where.
  ///
  /// \returns the new access that was created.
  MemoryAccess *replaceMemoryAccessWithNewAccess(MemoryAccess *Replacee,
                                                 Instruction *Replacer,
                                                 enum InsertionPlace Where);
  /// \brief Replace a MemoryAccess with a new access, created based on
  /// instruction \p Replacer - this does not perform generic SSA updates, so it
  /// only works if the new access dominates the old accesses uses.
  ///
  /// This version places the access before the place that \p Where points to.
  MemoryAccess *
  replaceMemoryAccessWithNewAccess(MemoryAccess *Replacee,
                                   Instruction *Replacer,
                                   const AccessListType::iterator &Where);

  /// \brief Add a new MemoryUse for \p Use at the beginning or end of a block.
  ///
  /// \returns The new memory access that was created.
  MemoryAccess *addNewMemoryUse(Instruction *Use, enum InsertionPlace Where);

protected:
  // Used by Memory SSA annotater, dumpers, and wrapper pass
  friend class MemorySSAAnnotatedWriter;
  friend class MemorySSAPrinterPass;
  void verifyDefUses(Function &F);
  void verifyDomination(Function &F);

private:
  void verifyUseInDefs(MemoryAccess *, MemoryAccess *);
  typedef DenseMap<const BasicBlock *, AccessListType *> AccessMap;

  void
  determineInsertionPoint(AccessMap &BlockAccesses,
                          const SmallPtrSetImpl<BasicBlock *> &DefiningBlocks);
  void computeDomLevels(DenseMap<DomTreeNode *, unsigned> &DomLevels);
  void markUnreachableAsLiveOnEntry(AccessMap &BlockAccesses, BasicBlock *BB);
  bool dominatesUse(MemoryAccess *, MemoryAccess *) const;
  void removeFromLookups(MemoryAccess *);
  MemoryAccess *createNewAccess(Instruction *);
  MemoryAccess *findDominatingDef(Instruction *, enum InsertionPlace);

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
  AliasAnalysis *AA;
  DominatorTree *DT;
  Function &F;

  // Memory SSA mappings
  DenseMap<const Value *, MemoryAccess *> InstructionToMemoryAccess;
  AccessMap PerBlockAccesses;
  MemoryAccess *LiveOnEntryDef;

  // Memory SSA building info
  unsigned int nextID;
  bool builtAlready;
  MemorySSAWalker *Walker;
};

// This pass does eager building and then printing of MemorySSA. It is used by
// the tests to be able to build, dump, and verify Memory SSA.

class MemorySSAPrinterPass : public FunctionPass {
public:
  MemorySSAPrinterPass();

  static char ID;
  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &) override;
  void releaseMemory() override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void print(raw_ostream &OS, const Module *M) const override;
  static void registerOptions();
  MemorySSA &getMSSA() { return *MSSA; }

private:
  bool VerifyMemorySSA;

  MemorySSA *MSSA;
  MemorySSAWalker *Walker;
  Function *F;
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

/// \brief This is the generic walker interface for walkers of MemorySSA.
/// Walkers are used to be able to further disambiguate the def-use chains
/// MemorySSA gives you.
class MemorySSAWalker {
public:
  MemorySSAWalker(MemorySSA *);
  virtual ~MemorySSAWalker() {}

  typedef SmallVector<MemoryAccess *, 8> MemoryAccessSet;
  /// \brief Given a memory defining/using/clobbering instruction, calling this
  /// will give you the nearest dominating clobbering MemoryAccess (by skipping
  /// non-aliasing def links).
  ///
  /// Note that this will return a single access, and it must dominate the
  /// Instruction, so if an argument of a MemoryPhi node clobbers thenstruction,
  /// it will return the memoryphi node, *not* the argument.
  virtual MemoryAccess *getClobberingMemoryAccess(const Instruction *) = 0;

  /// \brief Given a potentially clobbering memory access and a new location,
  /// calling this will give you the nearest dominating clobbering MemoryAccess
  /// (by skipping non-aliasing def links).
  ///
  /// This version of the function is mainly used to disambiguate phi translated
  /// pointers, where the value of a pointer may have changed from the initial
  /// memory access.  Note that this expects to be handed either a memory use,
  /// or an already potentially clobbering access.  Unlike the above API, if
  /// given a MemoryDef that clobbers the pointer as the starting access, it
  /// will return that MemoryDef, whereas the above would return the clobber
  /// starting from the use side of  the memory def.
  virtual MemoryAccess *
  getClobberingMemoryAccess(MemoryAccess *, AliasAnalysis::Location &) = 0;

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

/// \brief A MemorySSAWalker that does no alias queries, or anything else. It
/// simply returns the links as they were constructed by the builder.
class DoNothingMemorySSAWalker final : public MemorySSAWalker {
public:
  MemoryAccess *getClobberingMemoryAccess(const Instruction *) override;
  MemoryAccess *getClobberingMemoryAccess(MemoryAccess *,
                                          AliasAnalysis::Location &) override;
};

/// \brief A MemorySSAWalker that does real AA walks and caching of lookups.
class CachingMemorySSAWalker final : public MemorySSAWalker {
public:
  CachingMemorySSAWalker(MemorySSA *, AliasAnalysis *);
  virtual ~CachingMemorySSAWalker();
  MemoryAccess *getClobberingMemoryAccess(const Instruction *) override;
  MemoryAccess *getClobberingMemoryAccess(MemoryAccess *,
                                          AliasAnalysis::Location &) override;

protected:
  struct UpwardsMemoryQuery;
  MemoryAccess *doCacheLookup(const MemoryAccess *, const UpwardsMemoryQuery &);
  void doCacheInsert(const MemoryAccess *, MemoryAccess *,
                     const UpwardsMemoryQuery &);

private:
  std::pair<MemoryAccess *, bool>
  getClobberingMemoryAccess(MemoryPhi *Phi, struct UpwardsMemoryQuery &);
  std::pair<MemoryAccess *, bool>
  getClobberingMemoryAccess(MemoryAccess *, struct UpwardsMemoryQuery &);

  DenseMap<std::pair<const MemoryAccess *, AliasAnalysis::Location>,
           MemoryAccess *> CachedUpwardsClobberingAccess;
  DenseMap<const MemoryAccess *, MemoryAccess *> CachedUpwardsClobberingCall;
  AliasAnalysis *AA;
};
}

#endif
