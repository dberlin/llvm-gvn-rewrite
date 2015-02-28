//===- MemorySSA.h - Build Memory SSA ----------------------------*- C++
//-*-===//
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
#include "llvm/ADT/UniqueVector.h"
#include "llvm/Support/Allocator.h"
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

  virtual void print(raw_ostream &OS, UniqueVector<MemoryAccess *> &SlotInfo) {}
  typedef MemoryAccess **iterator;
  typedef MemoryAccess **const const_iterator;

  // The use list is immutable because it is allocated in a
  // BumpPtrAllocator

  const_iterator use_begin() const { return UseList; }
  const_iterator use_end() const { return UseList + NumUses; }

protected:
  friend class MemorySSA;
  // We automatically allocate the right amount of space
  void addUse(MemoryAccess *Use) { UseList[NumUses++] = Use; }
  MemoryAccess(AccessType AT, BasicBlock *BB)
      : AccessType(AT), Block(BB), NumUses(0), UseList(nullptr) {}

private:
  MemoryAccess(const MemoryAccess &);
  void operator=(const MemoryAccess &);
  AccessType AccessType;
  BasicBlock *Block;
  unsigned int NumUses;
  MemoryAccess **UseList;
};

class MemoryUse : public MemoryAccess {
public:
  MemoryUse(MemoryAccess *DMA, Instruction *MI, BasicBlock *BB)
      : MemoryUse(DMA, AccessUse, MI, BB) {}

  MemoryAccess *getDefiningAccess() const { return DefiningAccess; }
  void setDefiningAccess(MemoryAccess *DMA) { DefiningAccess = DMA; }
  Instruction *getMemoryInst() const { return MemoryInst; }
  void setMemoryInst(Instruction *MI) { MemoryInst = MI; }

  static inline bool classof(const MemoryUse *) { return true; }
  static inline bool classof(const MemoryAccess *MA) {
    return MA->getAccessType() == AccessUse;
  }
  virtual void print(raw_ostream &OS, UniqueVector<MemoryAccess *> &SlotInfo);

protected:
  MemoryUse(MemoryAccess *DMA, enum AccessType AT, Instruction *MI,
            BasicBlock *BB)
      : MemoryAccess(AT, BB), DefiningAccess(DMA), MemoryInst(MI) {}

private:
  MemoryAccess *DefiningAccess;
  Instruction *MemoryInst;
};

// All defs also have a use
class MemoryDef : public MemoryUse {
public:
  MemoryDef(MemoryAccess *DMA, Instruction *MI, BasicBlock *BB)
      : MemoryUse(DMA, AccessDef, MI, BB) {}

  static inline bool classof(const MemoryDef *) { return true; }
  static inline bool classof(const MemoryUse *MA) {
    return MA->getAccessType() == AccessDef;
  }

  static inline bool classof(const MemoryAccess *MA) {
    return MA->getAccessType() == AccessDef;
  }
  virtual void print(raw_ostream &OS, UniqueVector<MemoryAccess *> &SlotInfo);
  typedef MemoryAccess **iterator;
  typedef const MemoryAccess **const_iterator;
};
class MemoryPhi : public MemoryAccess {
public:
  MemoryPhi(BasicBlock *BB, unsigned int NumPreds)
      : MemoryAccess(AccessPhi, BB) {
    Args.reserve(NumPreds);
  }
  unsigned int getNumIncomingValues() { return Args.size(); }
  void addIncoming(MemoryAccess *MA, BasicBlock *BB) {
    Args.push_back(std::make_pair(BB, MA));
  }
  void setIncomingValue(unsigned int v, MemoryAccess *MA) {
    std::pair<BasicBlock *, MemoryAccess *> &Val = Args[v];
    Val.second = MA;
  }
  MemoryAccess *getIncomingValue(unsigned int v) { return Args[v].second; }
  void setIncomingBlock(unsigned int v, BasicBlock *BB) {
    std::pair<BasicBlock *, MemoryAccess *> &Val = Args[v];
    Val.first = BB;
  }
  BasicBlock *getIncomingBlock(unsigned int v) { return Args[v].first; }
  static inline bool classof(const MemoryPhi *) { return true; }
  static inline bool classof(const MemoryAccess *MA) {
    return MA->getAccessType() == AccessPhi;
  }

  virtual void print(raw_ostream &OS, UniqueVector<MemoryAccess *> &SlotInfo);

private:
  SmallVector<std::pair<BasicBlock *, MemoryAccess *>, 8> Args;
};

class MemorySSA : public FunctionPass {

private:
  AliasAnalysis *AA;
  DominatorTree *DT;
  BumpPtrAllocator MemoryAccessAllocator;
  Function *F;

  // Memory SSA mappings
  DenseMap<const Value *, MemoryAccess *> InstructionToMemoryAccess;
  DenseMap<std::pair<MemoryAccess *, AliasAnalysis::Location>, MemoryAccess *>
      CachedClobberingAccess;

  // Memory SSA building info
  MemoryAccess *LiveOnEntryDef;

  typedef DenseMap<BasicBlock *, std::list<MemoryAccess *> *> AccessMap;

public:
  MemorySSA();
  ~MemorySSA();
  static char ID;

  bool runOnFunction(Function &) override;

  void releaseMemory() override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  // Memory SSA related stuff
  void buildMemorySSA(Function &F);
  // Given a memory defining/using/clobbering instruction, give you
  // the nearest dominating clobbering Memory Access (by skipping non-aliasing
  // def links)
  MemoryAccess *getClobberingMemoryAccess(Instruction *);
  // Given a memory using/clobbering/etc instruction, get the
  // MemorySSA access associaed with it
  MemoryAccess *getMemoryAccess(const Value *I) const;
  void dump(Function &F);
  void print(raw_ostream &OS, const Module *M) const override;

private:
  bool isLiveOnEntryDef(const MemoryAccess *MA) const;
  void verifyUseInDefs(MemoryAccess *Def, MemoryAccess *Use);
  typedef DenseMap<MemoryAccess *, std::list<MemoryAccess *> *> UseMap;
  std::pair<MemoryAccess *, bool>
  getClobberingMemoryAccess(MemoryPhi *Phi, const AliasAnalysis::Location &,
                            SmallPtrSet<MemoryAccess *, 32> &);
  std::pair<MemoryAccess *, bool>
  getClobberingMemoryAccess(MemoryAccess *, const AliasAnalysis::Location &,
                            SmallPtrSet<MemoryAccess *, 32> &);

  void computeLiveInBlocks(const AccessMap &BlockAccesses,
                           const SmallPtrSetImpl<BasicBlock *> &DefBlocks,
                           const SmallVector<BasicBlock *, 32> &UseBlocks,
                           SmallPtrSetImpl<BasicBlock *> &LiveInBlocks);
  void
  determineInsertionPoint(Function &F, AccessMap &BlockAccesses,
                          const SmallPtrSetImpl<BasicBlock *> &DefiningBlocks,
                          const SmallVector<BasicBlock *, 32> &UsingBlocks);
  void computeDomLevels(DenseMap<DomTreeNode *, unsigned> &DomLevels);
  void computeBBNumbers(Function &F,
                        DenseMap<BasicBlock *, unsigned> &BBNumbers);
  void markUnreachableAsLiveOnEntry(AccessMap &BlockAccesses, BasicBlock *BB,
                                    UseMap &Uses);
  void addUses(UseMap &Uses);
  void addUseToMap(UseMap &, MemoryAccess *, MemoryAccess *);
  void verifyDefUses(Function &F);

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
                  SmallPtrSet<BasicBlock *, 16> &Visited, UseMap &Uses);
};
}
#endif
