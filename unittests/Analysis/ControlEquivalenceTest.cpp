//===- ControlEquivalenceTest.cpp - Control Equivalence tests -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ControlEquivalence.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"
#include <map>
#include <unordered_set>
#include <string>

using namespace llvm;

namespace {

// This fixture assists in running the ControlEquivalence analysis
// and ensuring it produces the correct answer each time.
class ControlEquivalenceTest : public testing::Test {
protected:
  ControlEquivalenceTest() : M(nullptr) {}
  std::unique_ptr<Module> M;

  void ParseAssembly(const char *Assembly) {
    SMDiagnostic Error;
    M = parseAssemblyString(Assembly, Error, getGlobalContext());

    std::string errMsg;
    raw_string_ostream os(errMsg);
    Error.print("", os);

    // A failure here means that the test itself is buggy.
    if (!M)
      report_fatal_error(os.str().c_str());
  }
  typedef std::vector<std::vector<std::string>> ResultType;

  // Pass in the equivalence classes you expect
  // So for example, "a", ["b", "c"], "d"
  // means we expect a and d to be in unique equivalence classes, and b and c to
  // share one
  void ExpectEquivalence(const ResultType &ExpectedResults) {
    static char ID;
    class ControlEquivalenceTestPass : public FunctionPass {
    public:
      ControlEquivalenceTestPass(const ResultType &ExpectedResults)
          : FunctionPass(ID), Expectation(ExpectedResults) {}

      static int initialize() {
        PassInfo *PI = new PassInfo("isPotentiallyReachable testing pass", "",
                                    &ID, nullptr, true, true);
        PassRegistry::getPassRegistry()->registerPass(*PI, false);
        initializeControlEquivalencePass(*PassRegistry::getPassRegistry());
        return 0;
      }

      void getAnalysisUsage(AnalysisUsage &AU) const {
        AU.setPreservesAll();
        AU.addRequired<ControlEquivalence>();
      }

      bool runOnFunction(Function &F) {
        ControlEquivalence *CE = &getAnalysis<ControlEquivalence>();
        std::map<std::string, unsigned> Results;
        for (auto &BB : F) {
          unsigned int ClassNum = CE->getClassNumber(&BB);
          assert(BB.hasName() && "Need to name all BasicBlocks");
          // We do not ever assign class number 0, if we see it, it means the
          // algorithm broke
          EXPECT_NE(ClassNum, (unsigned)0);
          Results[BB.getName().str()] = ClassNum;
        }
        std::unordered_set<unsigned> SeenIds;
        for (const auto &V : Expectation) {
          unsigned int LookingFor = 0;
          // This is a vector, check each member of the vector has the same
          // id, and it's not one we've seen before.
          for (const auto &Member : V) {
            EXPECT_EQ(Results.count(Member), 1u);
            if (LookingFor == 0) {
              LookingFor = Results[Member];
              EXPECT_EQ(SeenIds.count(LookingFor), 0u);
            } else {
              EXPECT_EQ(Results[Member], LookingFor);
            }
          }
          SeenIds.insert(LookingFor);
        }
        // Make sure we verified did not end up with extra classes
        EXPECT_EQ(SeenIds.size(), Expectation.size());

        return false;
      }
      const ResultType &Expectation;
    };

    static int initialize = ControlEquivalenceTestPass::initialize();
    (void)initialize;

    ControlEquivalenceTestPass *P =
        new ControlEquivalenceTestPass(ExpectedResults);
    legacy::PassManager PM;
    PM.add(P);
    PM.run(*M);
  }
};

TEST_F(ControlEquivalenceTest, StraightLineTest) {
  ParseAssembly(" define void @teststraightline() {\n"
                " start:\n"
                "   br label %next\n"
                " next:\n"
                "   br label %returnit\n"
                " returnit:\n"
                "   ret void\n"
                " }");
  ExpectEquivalence({{"start", "next", "returnit"}});
}
TEST_F(ControlEquivalenceTest, DiamondTest) {
  ParseAssembly("define void @testdiamond() {\n"
                "start:\n"
                "  br i1 true, label %same, label %different\n"
                "\n"
                "same: \n"
                "  br label %returnit\n"
                "\n"
                "different:\n"
                "  br label %returnit\n"
                "\n"
                "returnit:\n"
                "  ret void\n"
                "}");
  ExpectEquivalence({{"start", "returnit"}, {"same"}, {"different"}});
}
TEST_F(ControlEquivalenceTest, SplitWithMultipleReturnTest) {
  ParseAssembly("define void @testsplitwithmultiplereturn() {\n"
                "start:\n"
                "  br i1 true, label %same, label %different\n"
                "\n"
                "same:\n"
                "  ret void\n"
                "\n"
                "different:\n"
                "  ret void\n"
                "}");
  ExpectEquivalence({{"start"}, {"same"}, {"different"}});
}
TEST_F(ControlEquivalenceTest, DoubleDiamondTest) {
  ParseAssembly(" define void @testdiamond() {\n"
                " start:\n"
                "  br i1 true, label %same, label %different\n"
                " same:\n"
                "  br i1 true, label %samepart1, label %samepart2\n"
                " samepart1:\n"
                "  br label %samemergepoint\n"
                " samepart2:\n"
                "  br label %samemergepoint\n"
                " samemergepoint:\n"
                "  br label %returnit\n"
                " different:\n"
                "  br label %returnit\n"
                " returnit:\n"
                "     ret void\n"
                " }");
  ExpectEquivalence({{"start", "returnit"},
                     {"same", "samemergepoint"},
                     {"different"},
                     {"samepart1"},
                     {"samepart2"}});
}
}
