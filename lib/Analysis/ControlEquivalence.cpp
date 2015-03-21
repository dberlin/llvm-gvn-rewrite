//===- ControlEquivalence.cpp - Compute Control Equivalence ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
#include "llvm/Analysis/ControlEquivalence.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include <utility>

using namespace llvm;
char ControlEquivalence::ID = 0;

INITIALIZE_PASS(ControlEquivalence, "controlequiv",
                "Control Equivalence Construction", true, true);

bool ControlEquivalence::runOnFunction(Function &F) {
  DFSNumber = 0;
  ClassNumber = 1;

  BlockData.resize(F.size());
  for (auto &B : F)
    BlockData.insert({&B, ControlBlockData()});

  return false;
}

void ControlEquivalence::releaseMemory() { BlockData.clear(); }

// print - Show contents in human readable format...
void ControlEquivalence::print(raw_ostream &O, const Module *M) const {}
