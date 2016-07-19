//===- NewGVN.h - Eliminate redundant values and loads ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides the interface for LLVM's Global Value Numbering pass
/// which eliminates fully redundant instructions. It also does somewhat Ad-Hoc
/// PRE and dead load elimination.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_NEWGVN_H
#define LLVM_TRANSFORMS_SCALAR_NEWGVN_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class NewGVNPass : public PassInfoMixin<NewGVNPass> {
public:
  /// \brief Run the pass over the function.
  PreservedAnalyses run(Function &F, AnalysisManager<Function> &AM);
};
}

#endif // LLVM_TRANSFORMS_SCALAR_NEWGVN_H

