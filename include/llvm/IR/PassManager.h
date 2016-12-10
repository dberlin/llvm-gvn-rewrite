//===- PassManager.h - Pass management infrastructure -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This header defines various interfaces for pass management in LLVM. There
/// is no "pass" interface in LLVM per se. Instead, an instance of any class
/// which supports a method to 'run' it over a unit of IR can be used as
/// a pass. A pass manager is generally a tool to collect a sequence of passes
/// which run over a particular IR construct, and run each of them in sequence
/// over each such construct in the containing IR construct. As there is no
/// containing IR construct for a Module, a manager for passes over modules
/// forms the base case which runs its managed passes in sequence over the
/// single module provided.
///
/// The core IR library provides managers for running passes over
/// modules and functions.
///
/// * FunctionPassManager can run over a Module, runs each pass over
///   a Function.
/// * ModulePassManager must be directly run, runs each pass over the Module.
///
/// Note that the implementations of the pass managers use concept-based
/// polymorphism as outlined in the "Value Semantics and Concept-based
/// Polymorphism" talk (or its abbreviated sibling "Inheritance Is The Base
/// Class of Evil") by Sean Parent:
/// * http://github.com/sean-parent/sean-parent.github.com/wiki/Papers-and-Presentations
/// * http://www.youtube.com/watch?v=_BpMYeUFXv8
/// * http://channel9.msdn.com/Events/GoingNative/2013/Inheritance-Is-The-Base-Class-of-Evil
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_PASSMANAGER_H
#define LLVM_IR_PASSMANAGER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManagerInternal.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TypeName.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/type_traits.h"
#include <list>
#include <memory>
#include <vector>

namespace llvm {

/// A special type used by analysis passes to provide an address that
/// identifies that particular analysis pass type.
///
/// Analysis passes should have a static data member of this type and derive
/// from the \c AnalysisInfoMixin to get a static ID method used to identify
/// the analysis in the pass management infrastructure.
struct alignas(8) AnalysisKey {};

/// \brief An abstract set of preserved analyses following a transformation pass
/// run.
///
/// When a transformation pass is run, it can return a set of analyses whose
/// results were preserved by that transformation. The default set is "none",
/// and preserving analyses must be done explicitly.
///
/// There is also an explicit all state which can be used (for example) when
/// the IR is not mutated at all.
class PreservedAnalyses {
public:
  /// \brief Convenience factory function for the empty preserved set.
  static PreservedAnalyses none() { return PreservedAnalyses(); }

  /// \brief Construct a special preserved set that preserves all passes.
  static PreservedAnalyses all() {
    PreservedAnalyses PA;
    PA.PreservedAnalysisIDs.insert(&AllAnalysesKey);
    return PA;
  }

  /// \brief Mark a particular pass as preserved, adding it to the set.
  template <typename PassT> void preserve() { preserve(PassT::ID()); }

  /// \brief Mark an abstract ID as preserved, adding it to the set.
  void preserve(AnalysisKey *ID) {
    if (!areAllPreserved())
      PreservedAnalysisIDs.insert(ID);
  }

  /// \brief Intersect this set with another in place.
  ///
  /// This is a mutating operation on this preserved set, removing all
  /// preserved passes which are not also preserved in the argument.
  void intersect(const PreservedAnalyses &Arg) {
    if (Arg.areAllPreserved())
      return;
    if (areAllPreserved()) {
      PreservedAnalysisIDs = Arg.PreservedAnalysisIDs;
      return;
    }
    for (auto ID : PreservedAnalysisIDs)
      if (!Arg.PreservedAnalysisIDs.count(ID))
        PreservedAnalysisIDs.erase(ID);
  }

  /// \brief Intersect this set with a temporary other set in place.
  ///
  /// This is a mutating operation on this preserved set, removing all
  /// preserved passes which are not also preserved in the argument.
  void intersect(PreservedAnalyses &&Arg) {
    if (Arg.areAllPreserved())
      return;
    if (areAllPreserved()) {
      PreservedAnalysisIDs = std::move(Arg.PreservedAnalysisIDs);
      return;
    }
    for (auto ID : PreservedAnalysisIDs)
      if (!Arg.PreservedAnalysisIDs.count(ID))
        PreservedAnalysisIDs.erase(ID);
  }

  /// \brief Query whether a pass is marked as preserved by this set.
  template <typename PassT> bool preserved() const {
    return preserved(PassT::ID());
  }

  /// \brief Query whether an abstract pass ID is marked as preserved by this
  /// set.
  bool preserved(AnalysisKey *ID) const {
    return PreservedAnalysisIDs.count(&AllAnalysesKey) ||
           PreservedAnalysisIDs.count(ID);
  }

  /// \brief Query whether all of the analyses in the set are preserved.
  bool preserved(const PreservedAnalyses& Arg) {
    if (Arg.areAllPreserved())
      return areAllPreserved();
    for (auto ID : Arg.PreservedAnalysisIDs)
      if (!preserved(ID))
        return false;
    return true;
  }

  /// \brief Test whether all passes are preserved.
  ///
  /// This is used primarily to optimize for the case of no changes which will
  /// common in many scenarios.
  bool areAllPreserved() const {
    return PreservedAnalysisIDs.count(&AllAnalysesKey);
  }

private:
  // A special key used to indicate all analyses.
  static AnalysisKey AllAnalysesKey;

  SmallPtrSet<AnalysisKey *, 2> PreservedAnalysisIDs;
};

// Forward declare the analysis manager template.
template <typename IRUnitT, typename... ExtraArgTs> class AnalysisManager;

/// A CRTP mix-in to automatically provide informational APIs needed for
/// passes.
///
/// This provides some boiler plate for types that are passes.
template <typename DerivedT> struct PassInfoMixin {
  /// Returns the name of the derived pass type.
  static StringRef name() {
    StringRef Name = getTypeName<DerivedT>();
    if (Name.startswith("llvm::"))
      Name = Name.drop_front(strlen("llvm::"));
    return Name;
  }
};

/// A CRTP mix-in to automatically provide informational APIs needed for
/// analysis passes.
///
/// This provides some boiler plate for types that are analysis passes. It
/// automatically mixes in \c PassInfoMixin and adds informational APIs
/// specifically used for analyses.
template <typename DerivedT>
struct AnalysisInfoMixin : PassInfoMixin<DerivedT> {
  /// Returns an opaque, unique ID for this analysis type.
  ///
  /// This ID is a pointer type that is guaranteed to be 8-byte aligned and
  /// thus suitable for use in sets, maps, and other data structures optimized
  /// for pointer-like types using the alignment-provided low bits.
  ///
  /// Note that this requires the derived type provide a static \c AnalysisKey
  /// member called \c Key.
  ///
  /// FIXME: The only reason the derived type needs to provide this rather than
  /// this mixin providing it is due to broken implementations which cannot
  /// correctly unique a templated static so that they have the same addresses
  /// for each instantiation and are definitively emitted once for each
  /// instantiation. The only currently known platform with this limitation are
  /// Windows DLL builds, specifically building each part of LLVM as a DLL. If
  /// we ever remove that build configuration, this mixin can provide the
  /// static key as well.
  static AnalysisKey *ID() { return &DerivedT::Key; }
};

/// A class template to provide analysis sets for IR units.
///
/// Analyses operate on units of IR. It is useful to be able to talk about
/// preservation of all analyses for a given unit of IR as a set. This class
/// template can be used with the \c PreservedAnalyses API for that purpose and
/// the \c AnalysisManager will automatically check and use this set to skip
/// invalidation events.
///
/// Note that you must provide an explicit instantiation declaration and
/// definition for this template in order to get the correct behavior on
/// Windows. Otherwise, the address of SetKey will not be stable.
template <typename IRUnitT>
class AllAnalysesOn {
public:
  static AnalysisKey *ID() { return &SetKey; }

private:
  static AnalysisKey SetKey;
};

template <typename IRUnitT> AnalysisKey AllAnalysesOn<IRUnitT>::SetKey;

extern template class AllAnalysesOn<Module>;
extern template class AllAnalysesOn<Function>;

/// \brief Manages a sequence of passes over units of IR.
///
/// A pass manager contains a sequence of passes to run over units of IR. It is
/// itself a valid pass over that unit of IR, and when over some given IR will
/// run each pass in sequence. This is the primary and most basic building
/// block of a pass pipeline.
///
/// If it is run with an \c AnalysisManager<IRUnitT> argument, it will propagate
/// that analysis manager to each pass it runs, as well as calling the analysis
/// manager's invalidation routine with the PreservedAnalyses of each pass it
/// runs.
template <typename IRUnitT,
          typename AnalysisManagerT = AnalysisManager<IRUnitT>,
          typename... ExtraArgTs>
class PassManager : public PassInfoMixin<
                        PassManager<IRUnitT, AnalysisManagerT, ExtraArgTs...>> {
public:
  /// \brief Construct a pass manager.
  ///
  /// It can be passed a flag to get debug logging as the passes are run.
  explicit PassManager(bool DebugLogging = false) : DebugLogging(DebugLogging) {}

  // FIXME: These are equivalent to the default move constructor/move
  // assignment. However, using = default triggers linker errors due to the
  // explicit instantiations below. Find away to use the default and remove the
  // duplicated code here.
  PassManager(PassManager &&Arg)
      : Passes(std::move(Arg.Passes)),
        DebugLogging(std::move(Arg.DebugLogging)) {}
  PassManager &operator=(PassManager &&RHS) {
    Passes = std::move(RHS.Passes);
    DebugLogging = std::move(RHS.DebugLogging);
    return *this;
  }

  /// \brief Run all of the passes in this manager over the IR.
  PreservedAnalyses run(IRUnitT &IR, AnalysisManagerT &AM,
                        ExtraArgTs... ExtraArgs) {
    PreservedAnalyses PA = PreservedAnalyses::all();

    if (DebugLogging)
      dbgs() << "Starting " << getTypeName<IRUnitT>() << " pass manager run.\n";

    for (unsigned Idx = 0, Size = Passes.size(); Idx != Size; ++Idx) {
      if (DebugLogging)
        dbgs() << "Running pass: " << Passes[Idx]->name() << " on "
               << IR.getName() << "\n";

      PreservedAnalyses PassPA = Passes[Idx]->run(IR, AM, ExtraArgs...);

      // Update the analysis manager as each pass runs and potentially
      // invalidates analyses.
      AM.invalidate(IR, PassPA);

      // Finally, we intersect the preserved analyses to compute the aggregate
      // preserved set for this pass manager.
      PA.intersect(std::move(PassPA));

      // FIXME: Historically, the pass managers all called the LLVM context's
      // yield function here. We don't have a generic way to acquire the
      // context and it isn't yet clear what the right pattern is for yielding
      // in the new pass manager so it is currently omitted.
      //IR.getContext().yield();
    }

    // Invaliadtion was handled after each pass in the above loop for the
    // current unit of IR. Therefore, the remaining analysis results in the
    // AnalysisManager are preserved. We mark this with a set so that we don't
    // need to inspect each one individually.
    PA.preserve<AllAnalysesOn<IRUnitT>>();

    if (DebugLogging)
      dbgs() << "Finished " << getTypeName<IRUnitT>() << " pass manager run.\n";

    return PA;
  }

  template <typename PassT> void addPass(PassT Pass) {
    typedef detail::PassModel<IRUnitT, PassT, PreservedAnalyses,
                              AnalysisManagerT, ExtraArgTs...>
        PassModelT;
    Passes.emplace_back(new PassModelT(std::move(Pass)));
  }

private:
  typedef detail::PassConcept<IRUnitT, AnalysisManagerT, ExtraArgTs...>
      PassConceptT;

  std::vector<std::unique_ptr<PassConceptT>> Passes;

  /// \brief Flag indicating whether we should do debug logging.
  bool DebugLogging;
};

extern template class PassManager<Module>;
/// \brief Convenience typedef for a pass manager over modules.
typedef PassManager<Module> ModulePassManager;

extern template class PassManager<Function>;
/// \brief Convenience typedef for a pass manager over functions.
typedef PassManager<Function> FunctionPassManager;

/// \brief A generic analysis pass manager with lazy running and caching of
/// results.
///
/// This analysis manager can be used for any IR unit where the address of the
/// IR unit sufficies as its identity. It manages the cache for a unit of IR via
/// the address of each unit of IR cached.
template <typename IRUnitT, typename... ExtraArgTs> class AnalysisManager {
public:
  class Invalidator;

private:
  // Now that we've defined our invalidator, we can build types for the concept
  // types.
  typedef detail::AnalysisResultConcept<IRUnitT, PreservedAnalyses, Invalidator>
      ResultConceptT;
  typedef detail::AnalysisPassConcept<IRUnitT, PreservedAnalyses, Invalidator,
                                      ExtraArgTs...>
      PassConceptT;

  /// \brief List of function analysis pass IDs and associated concept pointers.
  ///
  /// Requires iterators to be valid across appending new entries and arbitrary
  /// erases. Provides the analysis ID to enable finding iterators to a given entry
  /// in maps below, and provides the storage for the actual result concept.
  typedef std::list<std::pair<AnalysisKey *, std::unique_ptr<ResultConceptT>>>
      AnalysisResultListT;

  /// \brief Map type from IRUnitT pointer to our custom list type.
  typedef DenseMap<IRUnitT *, AnalysisResultListT> AnalysisResultListMapT;

  /// \brief Map type from a pair of analysis ID and IRUnitT pointer to an
  /// iterator into a particular result list which is where the actual result
  /// is stored.
  typedef DenseMap<std::pair<AnalysisKey *, IRUnitT *>,
                   typename AnalysisResultListT::iterator>
      AnalysisResultMapT;

public:
  /// API to communicate dependencies between analyses during invalidation.
  ///
  /// When an analysis result embeds handles to other analysis results, it
  /// needs to be invalidated both when its own information isn't preserved and
  /// if any of those embedded analysis results end up invalidated. We pass in
  /// an \c Invalidator object from the analysis manager in order to let the
  /// analysis results themselves define the dependency graph on the fly. This
  /// avoids building an explicit data structure representation of the
  /// dependencies between analysis results.
  class Invalidator {
  public:
    /// Trigger the invalidation of some other analysis pass if not already
    /// handled and return whether it will in fact be invalidated.
    ///
    /// This is expected to be called from within a given analysis result's \c
    /// invalidate method to trigger a depth-first walk of all inter-analysis
    /// dependencies. The same \p IR unit and \p PA passed to that result's \c
    /// invalidate method should in turn be provided to this routine.
    ///
    /// The first time this is called for a given analysis pass, it will
    /// trigger the corresponding result's \c invalidate method to be called.
    /// Subsequent calls will use a cache of the results of that initial call.
    /// It is an error to form cyclic dependencies between analysis results.
    ///
    /// This returns true if the given analysis pass's result is invalid and
    /// any dependecies on it will become invalid as a result.
    template <typename PassT>
    bool invalidate(IRUnitT &IR, const PreservedAnalyses &PA) {
      AnalysisKey *ID = PassT::ID();

      // If we've already visited this pass, return true if it was invalidated
      // and false otherwise.
      auto IMapI = IsResultInvalidated.find(ID);
      if (IMapI != IsResultInvalidated.end())
        return IMapI->second;

      // Otherwise look up the result object.
      auto RI = Results.find({ID, &IR});
      assert(RI != Results.end() &&
             "Trying to invalidate a dependent result that isn't in the "
             "manager's cache is always an error, likely due to a stale result "
             "handle!");

      typedef detail::AnalysisResultModel<IRUnitT, PassT,
                                          typename PassT::Result,
                                          PreservedAnalyses, Invalidator>
          ResultModelT;
      auto &ResultModel = static_cast<ResultModelT &>(*RI->second->second);

      // Insert into the map whether the result should be invalidated and
      // return that. Note that we cannot re-use IMapI and must do a fresh
      // insert here as calling the invalidate routine could (recursively)
      // insert things into the map making any iterator or reference invalid.
      bool Inserted;
      std::tie(IMapI, Inserted) = IsResultInvalidated.insert(
          {ID, ResultModel.invalidate(IR, PA, *this)});
      (void)Inserted;
      assert(Inserted && "Should not have already inserted this ID, likely "
                         "indicates a dependency cycle!");
      return IMapI->second;
    }

  private:
    friend class AnalysisManager;

    Invalidator(SmallDenseMap<AnalysisKey *, bool, 8> &IsResultInvalidated,
                const AnalysisResultMapT &Results)
        : IsResultInvalidated(IsResultInvalidated), Results(Results) {}

    SmallDenseMap<AnalysisKey *, bool, 8> &IsResultInvalidated;
    const AnalysisResultMapT &Results;
  };

  /// \brief Construct an empty analysis manager.
  ///
  /// A flag can be passed to indicate that the manager should perform debug
  /// logging.
  AnalysisManager(bool DebugLogging = false) : DebugLogging(DebugLogging) {}
  AnalysisManager(AnalysisManager &&) = default;
  AnalysisManager &operator=(AnalysisManager &&) = default;

  /// \brief Returns true if the analysis manager has an empty results cache.
  bool empty() const {
    assert(AnalysisResults.empty() == AnalysisResultLists.empty() &&
           "The storage and index of analysis results disagree on how many "
           "there are!");
    return AnalysisResults.empty();
  }

  /// \brief Clear any results for a single unit of IR.
  ///
  /// This doesn't invalidate but directly clears the results. It is useful
  /// when the IR is being removed and we want to clear out all the memory
  /// pinned for it.
  void clear(IRUnitT &IR) {
    if (DebugLogging)
      dbgs() << "Clearing all analysis results for: " << IR.getName() << "\n";

    auto ResultsListI = AnalysisResultLists.find(&IR);
    if (ResultsListI == AnalysisResultLists.end())
      return;
    // Clear the map pointing into the results list.
    for (auto &IDAndResult : ResultsListI->second)
      AnalysisResults.erase({IDAndResult.first, &IR});

    // And actually destroy and erase the results associated with this IR.
    AnalysisResultLists.erase(ResultsListI);
  }

  /// \brief Clear the analysis result cache.
  ///
  /// This routine allows cleaning up when the set of IR units itself has
  /// potentially changed, and thus we can't even look up a a result and
  /// invalidate it directly. Notably, this does *not* call invalidate
  /// functions as there is nothing to be done for them.
  void clear() {
    AnalysisResults.clear();
    AnalysisResultLists.clear();
  }

  /// \brief Get the result of an analysis pass for this module.
  ///
  /// If there is not a valid cached result in the manager already, this will
  /// re-run the analysis to produce a valid result.
  template <typename PassT>
  typename PassT::Result &getResult(IRUnitT &IR, ExtraArgTs... ExtraArgs) {
    assert(AnalysisPasses.count(PassT::ID()) &&
           "This analysis pass was not registered prior to being queried");
    ResultConceptT &ResultConcept =
        getResultImpl(PassT::ID(), IR, ExtraArgs...);
    typedef detail::AnalysisResultModel<IRUnitT, PassT, typename PassT::Result,
                                        PreservedAnalyses, Invalidator>
        ResultModelT;
    return static_cast<ResultModelT &>(ResultConcept).Result;
  }

  /// \brief Get the cached result of an analysis pass for this module.
  ///
  /// This method never runs the analysis.
  ///
  /// \returns null if there is no cached result.
  template <typename PassT>
  typename PassT::Result *getCachedResult(IRUnitT &IR) const {
    assert(AnalysisPasses.count(PassT::ID()) &&
           "This analysis pass was not registered prior to being queried");

    ResultConceptT *ResultConcept = getCachedResultImpl(PassT::ID(), IR);
    if (!ResultConcept)
      return nullptr;

    typedef detail::AnalysisResultModel<IRUnitT, PassT, typename PassT::Result,
                                        PreservedAnalyses, Invalidator>
        ResultModelT;
    return &static_cast<ResultModelT *>(ResultConcept)->Result;
  }

  /// \brief Register an analysis pass with the manager.
  ///
  /// The argument is a callable whose result is a pass. This allows passing in
  /// a lambda to construct the pass.
  ///
  /// The pass type registered is the result type of calling the argument. If
  /// that pass has already been registered, then the argument will not be
  /// called and this function will return false. Otherwise, the pass type
  /// becomes registered, with the instance provided by calling the argument
  /// once, and this function returns true.
  ///
  /// While this returns whether or not the pass type was already registered,
  /// there in't an independent way to query that as that would be prone to
  /// risky use when *querying* the analysis manager. Instead, the only
  /// supported use case is avoiding duplicate registry of an analysis. This
  /// interface also lends itself to minimizing the number of times we have to
  /// do lookups for analyses or construct complex passes only to throw them
  /// away.
  template <typename PassBuilderT>
  bool registerPass(PassBuilderT &&PassBuilder) {
    typedef decltype(PassBuilder()) PassT;
    typedef detail::AnalysisPassModel<IRUnitT, PassT, PreservedAnalyses,
                                      Invalidator, ExtraArgTs...>
        PassModelT;

    auto &PassPtr = AnalysisPasses[PassT::ID()];
    if (PassPtr)
      // Already registered this pass type!
      return false;

    // Construct a new model around the instance returned by the builder.
    PassPtr.reset(new PassModelT(PassBuilder()));
    return true;
  }

  /// \brief Invalidate a specific analysis pass for an IR module.
  ///
  /// Note that the analysis result can disregard invalidation.
  template <typename PassT> void invalidate(IRUnitT &IR) {
    assert(AnalysisPasses.count(PassT::ID()) &&
           "This analysis pass was not registered prior to being invalidated");
    invalidateImpl(PassT::ID(), IR);
  }

  /// \brief Invalidate analyses cached for an IR unit.
  ///
  /// Walk through all of the analyses pertaining to this unit of IR and
  /// invalidate them unless they are preserved by the PreservedAnalyses set.
  void invalidate(IRUnitT &IR, const PreservedAnalyses &PA) {
    // Short circuit for common cases of all analyses being preserved.
    if (PA.areAllPreserved() || PA.preserved<AllAnalysesOn<IRUnitT>>())
      return;

    if (DebugLogging)
      dbgs() << "Invalidating all non-preserved analyses for: " << IR.getName()
             << "\n";

    // Track whether each pass's result is invalidated. Memoize the results
    // using the IsResultInvalidated map.
    SmallDenseMap<AnalysisKey *, bool, 8> IsResultInvalidated;
    Invalidator Inv(IsResultInvalidated, AnalysisResults);
    AnalysisResultListT &ResultsList = AnalysisResultLists[&IR];
    for (auto &AnalysisResultPair : ResultsList) {
      // This is basically the same thing as Invalidator::invalidate, but we
      // can't call it here because we're operating on the type-erased result.
      // Moreover if we instead called invalidate() directly, it would do an
      // unnecessary look up in ResultsList.
      AnalysisKey *ID = AnalysisResultPair.first;
      auto &Result = *AnalysisResultPair.second;

      auto IMapI = IsResultInvalidated.find(ID);
      if (IMapI != IsResultInvalidated.end())
        // This result was already handled via the Invalidator.
        continue;

      // Try to invalidate the result, giving it the Invalidator so it can
      // recursively query for any dependencies it has and record the result.
      // Note that we cannot re-use 'IMapI' here or pre-insert the ID as the
      // invalidate method may insert things into the map as well, invalidating
      // any iterator or pointer.
      bool Inserted =
          IsResultInvalidated.insert({ID, Result.invalidate(IR, PA, Inv)})
              .second;
      (void)Inserted;
      assert(Inserted && "Should never have already inserted this ID, likely "
                         "indicates a cycle!");
    }

    // Now erase the results that were marked above as invalidated.
    if (!IsResultInvalidated.empty()) {
      for (auto I = ResultsList.begin(), E = ResultsList.end(); I != E;) {
        AnalysisKey *ID = I->first;
        if (!IsResultInvalidated.lookup(ID)) {
          ++I;
          continue;
        }

        if (DebugLogging)
          dbgs() << "Invalidating analysis: " << this->lookUpPass(ID).name()
                 << "\n";

        I = ResultsList.erase(I);
        AnalysisResults.erase({ID, &IR});
      }
    }

    if (ResultsList.empty())
      AnalysisResultLists.erase(&IR);
  }

private:
  /// \brief Look up a registered analysis pass.
  PassConceptT &lookUpPass(AnalysisKey *ID) {
    typename AnalysisPassMapT::iterator PI = AnalysisPasses.find(ID);
    assert(PI != AnalysisPasses.end() &&
           "Analysis passes must be registered prior to being queried!");
    return *PI->second;
  }

  /// \brief Look up a registered analysis pass.
  const PassConceptT &lookUpPass(AnalysisKey *ID) const {
    typename AnalysisPassMapT::const_iterator PI = AnalysisPasses.find(ID);
    assert(PI != AnalysisPasses.end() &&
           "Analysis passes must be registered prior to being queried!");
    return *PI->second;
  }

  /// \brief Get an analysis result, running the pass if necessary.
  ResultConceptT &getResultImpl(AnalysisKey *ID, IRUnitT &IR,
                                ExtraArgTs... ExtraArgs) {
    typename AnalysisResultMapT::iterator RI;
    bool Inserted;
    std::tie(RI, Inserted) = AnalysisResults.insert(std::make_pair(
        std::make_pair(ID, &IR), typename AnalysisResultListT::iterator()));

    // If we don't have a cached result for this function, look up the pass and
    // run it to produce a result, which we then add to the cache.
    if (Inserted) {
      auto &P = this->lookUpPass(ID);
      if (DebugLogging)
        dbgs() << "Running analysis: " << P.name() << "\n";
      AnalysisResultListT &ResultList = AnalysisResultLists[&IR];
      ResultList.emplace_back(ID, P.run(IR, *this, ExtraArgs...));

      // P.run may have inserted elements into AnalysisResults and invalidated
      // RI.
      RI = AnalysisResults.find({ID, &IR});
      assert(RI != AnalysisResults.end() && "we just inserted it!");

      RI->second = std::prev(ResultList.end());
    }

    return *RI->second->second;
  }

  /// \brief Get a cached analysis result or return null.
  ResultConceptT *getCachedResultImpl(AnalysisKey *ID, IRUnitT &IR) const {
    typename AnalysisResultMapT::const_iterator RI =
        AnalysisResults.find({ID, &IR});
    return RI == AnalysisResults.end() ? nullptr : &*RI->second->second;
  }

  /// \brief Invalidate a function pass result.
  void invalidateImpl(AnalysisKey *ID, IRUnitT &IR) {
    typename AnalysisResultMapT::iterator RI =
        AnalysisResults.find({ID, &IR});
    if (RI == AnalysisResults.end())
      return;

    if (DebugLogging)
      dbgs() << "Invalidating analysis: " << this->lookUpPass(ID).name()
             << "\n";
    AnalysisResultLists[&IR].erase(RI->second);
    AnalysisResults.erase(RI);
  }

  /// \brief Map type from module analysis pass ID to pass concept pointer.
  typedef DenseMap<AnalysisKey *, std::unique_ptr<PassConceptT>> AnalysisPassMapT;

  /// \brief Collection of module analysis passes, indexed by ID.
  AnalysisPassMapT AnalysisPasses;

  /// \brief Map from function to a list of function analysis results.
  ///
  /// Provides linear time removal of all analysis results for a function and
  /// the ultimate storage for a particular cached analysis result.
  AnalysisResultListMapT AnalysisResultLists;

  /// \brief Map from an analysis ID and function to a particular cached
  /// analysis result.
  AnalysisResultMapT AnalysisResults;

  /// \brief A flag indicating whether debug logging is enabled.
  bool DebugLogging;
};

extern template class AnalysisManager<Module>;
/// \brief Convenience typedef for the Module analysis manager.
typedef AnalysisManager<Module> ModuleAnalysisManager;

extern template class AnalysisManager<Function>;
/// \brief Convenience typedef for the Function analysis manager.
typedef AnalysisManager<Function> FunctionAnalysisManager;

/// \brief A module analysis which acts as a proxy for a function analysis
/// manager.
///
/// This primarily proxies invalidation information from the module analysis
/// manager and module pass manager to a function analysis manager. You should
/// never use a function analysis manager from within (transitively) a module
/// pass manager unless your parent module pass has received a proxy result
/// object for it.
///
/// Note that the proxy's result is a move-only object and represents ownership
/// of the validity of the analyses in the \c FunctionAnalysisManager it
/// provides.
template <typename AnalysisManagerT, typename IRUnitT, typename... ExtraArgTs>
class InnerAnalysisManagerProxy
    : public AnalysisInfoMixin<
          InnerAnalysisManagerProxy<AnalysisManagerT, IRUnitT>> {
public:
  class Result {
  public:
    explicit Result(AnalysisManagerT &InnerAM) : InnerAM(&InnerAM) {}
    Result(Result &&Arg) : InnerAM(std::move(Arg.InnerAM)) {
      // We have to null out the analysis manager in the moved-from state
      // because we are taking ownership of the responsibilty to clear the
      // analysis state.
      Arg.InnerAM = nullptr;
    }
    Result &operator=(Result &&RHS) {
      InnerAM = RHS.InnerAM;
      // We have to null out the analysis manager in the moved-from state
      // because we are taking ownership of the responsibilty to clear the
      // analysis state.
      RHS.InnerAM = nullptr;
      return *this;
    }
    ~Result() {
      // InnerAM is cleared in a moved from state where there is nothing to do.
      if (!InnerAM)
        return;

      // Clear out the analysis manager if we're being destroyed -- it means we
      // didn't even see an invalidate call when we got invalidated.
      InnerAM->clear();
    }

    /// \brief Accessor for the analysis manager.
    AnalysisManagerT &getManager() { return *InnerAM; }

    /// \brief Handler for invalidation of the outer IR unit.
    ///
    /// If this analysis itself is preserved, then we assume that the set of \c
    /// IR units that the inner analysis manager controls hasn't changed and
    /// thus we don't need to invalidate *all* cached data associated with any
    /// \c IRUnitT* in the \c AnalysisManagerT.
    ///
    /// Regardless of whether this analysis is marked as preserved, all of the
    /// analyses in the \c AnalysisManagerT are potentially invalidated (for
    /// the relevant inner set of their IR units) based on the set of preserved
    /// analyses.
    ///
    /// Because this needs to understand the mapping from one IR unit to an
    /// inner IR unit, this method isn't defined in the primary template.
    /// Instead, each specialization of this template will need to provide an
    /// explicit specialization of this method to handle that particular pair
    /// of IR unit and inner AnalysisManagerT.
    bool invalidate(
        IRUnitT &IR, const PreservedAnalyses &PA,
        typename AnalysisManager<IRUnitT, ExtraArgTs...>::Invalidator &Inv);

  private:
    AnalysisManagerT *InnerAM;
  };

  explicit InnerAnalysisManagerProxy(AnalysisManagerT &InnerAM)
      : InnerAM(&InnerAM) {}

  /// \brief Run the analysis pass and create our proxy result object.
  ///
  /// This doesn't do any interesting work, it is primarily used to insert our
  /// proxy result object into the module analysis cache so that we can proxy
  /// invalidation to the function analysis manager.
  ///
  /// In debug builds, it will also assert that the analysis manager is empty
  /// as no queries should arrive at the function analysis manager prior to
  /// this analysis being requested.
  Result run(IRUnitT &IR, AnalysisManager<IRUnitT, ExtraArgTs...> &AM,
             ExtraArgTs...) {
    return Result(*InnerAM);
  }

private:
  friend AnalysisInfoMixin<
      InnerAnalysisManagerProxy<AnalysisManagerT, IRUnitT>>;
  static AnalysisKey Key;

  AnalysisManagerT *InnerAM;
};

template <typename AnalysisManagerT, typename IRUnitT, typename... ExtraArgTs>
AnalysisKey
    InnerAnalysisManagerProxy<AnalysisManagerT, IRUnitT, ExtraArgTs...>::Key;

/// Provide the \c FunctionAnalysisManager to \c Module proxy.
typedef InnerAnalysisManagerProxy<FunctionAnalysisManager, Module>
    FunctionAnalysisManagerModuleProxy;

/// Specialization of the invalidate method for the \c
/// FunctionAnalysisManagerModuleProxy's result.
template <>
bool FunctionAnalysisManagerModuleProxy::Result::invalidate(
    Module &M, const PreservedAnalyses &PA,
    ModuleAnalysisManager::Invalidator &Inv);

// Ensure the \c FunctionAnalysisManagerModuleProxy is provided as an extern
// template.
extern template class InnerAnalysisManagerProxy<FunctionAnalysisManager,
                                                Module>;

/// \brief A function analysis which acts as a proxy for a module analysis
/// manager.
///
/// This primarily provides an accessor to a parent module analysis manager to
/// function passes. Only the const interface of the module analysis manager is
/// provided to indicate that once inside of a function analysis pass you
/// cannot request a module analysis to actually run. Instead, the user must
/// rely on the \c getCachedResult API.
///
/// This proxy *doesn't* manage the invalidation in any way. That is handled by
/// the recursive return path of each layer of the pass manager and the
/// returned PreservedAnalysis set.
template <typename AnalysisManagerT, typename IRUnitT, typename... ExtraArgTs>
class OuterAnalysisManagerProxy
    : public AnalysisInfoMixin<
          OuterAnalysisManagerProxy<AnalysisManagerT, IRUnitT>> {
public:
  /// \brief Result proxy object for \c OuterAnalysisManagerProxy.
  class Result {
  public:
    explicit Result(const AnalysisManagerT &AM) : AM(&AM) {}

    const AnalysisManagerT &getManager() const { return *AM; }

    /// \brief Handle invalidation by ignoring it, this pass is immutable.
    bool invalidate(
        IRUnitT &, const PreservedAnalyses &,
        typename AnalysisManager<IRUnitT, ExtraArgTs...>::Invalidator &) {
      return false;
    }

  private:
    const AnalysisManagerT *AM;
  };

  OuterAnalysisManagerProxy(const AnalysisManagerT &AM) : AM(&AM) {}

  /// \brief Run the analysis pass and create our proxy result object.
  /// Nothing to see here, it just forwards the \c AM reference into the
  /// result.
  Result run(IRUnitT &, AnalysisManager<IRUnitT, ExtraArgTs...> &,
             ExtraArgTs...) {
    return Result(*AM);
  }

private:
  friend AnalysisInfoMixin<
      OuterAnalysisManagerProxy<AnalysisManagerT, IRUnitT>>;
  static AnalysisKey Key;

  const AnalysisManagerT *AM;
};

template <typename AnalysisManagerT, typename IRUnitT, typename... ExtraArgTs>
AnalysisKey
    OuterAnalysisManagerProxy<AnalysisManagerT, IRUnitT, ExtraArgTs...>::Key;

extern template class OuterAnalysisManagerProxy<ModuleAnalysisManager,
                                                Function>;
/// Provide the \c ModuleAnalysisManager to \c Fucntion proxy.
typedef OuterAnalysisManagerProxy<ModuleAnalysisManager, Function>
    ModuleAnalysisManagerFunctionProxy;

/// \brief Trivial adaptor that maps from a module to its functions.
///
/// Designed to allow composition of a FunctionPass(Manager) and
/// a ModulePassManager. Note that if this pass is constructed with a pointer
/// to a \c ModuleAnalysisManager it will run the
/// \c FunctionAnalysisManagerModuleProxy analysis prior to running the function
/// pass over the module to enable a \c FunctionAnalysisManager to be used
/// within this run safely.
///
/// Function passes run within this adaptor can rely on having exclusive access
/// to the function they are run over. They should not read or modify any other
/// functions! Other threads or systems may be manipulating other functions in
/// the module, and so their state should never be relied on.
/// FIXME: Make the above true for all of LLVM's actual passes, some still
/// violate this principle.
///
/// Function passes can also read the module containing the function, but they
/// should not modify that module outside of the use lists of various globals.
/// For example, a function pass is not permitted to add functions to the
/// module.
/// FIXME: Make the above true for all of LLVM's actual passes, some still
/// violate this principle.
template <typename FunctionPassT>
class ModuleToFunctionPassAdaptor
    : public PassInfoMixin<ModuleToFunctionPassAdaptor<FunctionPassT>> {
public:
  explicit ModuleToFunctionPassAdaptor(FunctionPassT Pass)
      : Pass(std::move(Pass)) {}

  /// \brief Runs the function pass across every function in the module.
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    // Setup the function analysis manager from its proxy.
    FunctionAnalysisManager &FAM =
        AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

    PreservedAnalyses PA = PreservedAnalyses::all();
    for (Function &F : M) {
      if (F.isDeclaration())
        continue;

      PreservedAnalyses PassPA = Pass.run(F, FAM);

      // We know that the function pass couldn't have invalidated any other
      // function's analyses (that's the contract of a function pass), so
      // directly handle the function analysis manager's invalidation here.
      FAM.invalidate(F, PassPA);

      // Then intersect the preserved set so that invalidation of module
      // analyses will eventually occur when the module pass completes.
      PA.intersect(std::move(PassPA));
    }

    // By definition we preserve the proxy. We also preserve all analyses on
    // Function units. This precludes *any* invalidation of function analyses
    // by the proxy, but that's OK because we've taken care to invalidate
    // analyses in the function analysis manager incrementally above.
    PA.preserve<AllAnalysesOn<Function>>();
    PA.preserve<FunctionAnalysisManagerModuleProxy>();
    return PA;
  }

private:
  FunctionPassT Pass;
};

/// \brief A function to deduce a function pass type and wrap it in the
/// templated adaptor.
template <typename FunctionPassT>
ModuleToFunctionPassAdaptor<FunctionPassT>
createModuleToFunctionPassAdaptor(FunctionPassT Pass) {
  return ModuleToFunctionPassAdaptor<FunctionPassT>(std::move(Pass));
}

/// \brief A template utility pass to force an analysis result to be available.
///
/// If there are extra arguments at the pass's run level there may also be
/// extra arguments to the analysis manager's \c getResult routine. We can't
/// guess how to effectively map the arguments from one to the other, and so
/// this specialization just ignores them.
///
/// Specific patterns of run-method extra arguments and analysis manager extra
/// arguments will have to be defined as appropriate specializations.
template <typename AnalysisT, typename IRUnitT,
          typename AnalysisManagerT = AnalysisManager<IRUnitT>,
          typename... ExtraArgTs>
struct RequireAnalysisPass
    : PassInfoMixin<RequireAnalysisPass<AnalysisT, IRUnitT, AnalysisManagerT,
                                        ExtraArgTs...>> {
  /// \brief Run this pass over some unit of IR.
  ///
  /// This pass can be run over any unit of IR and use any analysis manager
  /// provided they satisfy the basic API requirements. When this pass is
  /// created, these methods can be instantiated to satisfy whatever the
  /// context requires.
  PreservedAnalyses run(IRUnitT &Arg, AnalysisManagerT &AM,
                        ExtraArgTs &&... Args) {
    (void)AM.template getResult<AnalysisT>(Arg,
                                           std::forward<ExtraArgTs>(Args)...);

    return PreservedAnalyses::all();
  }
};

/// \brief A template utility pass to force an analysis result to be
/// invalidated.
///
/// This is a no-op pass which simply forces a specific analysis result to be
/// invalidated when it is run.
template <typename AnalysisT>
struct InvalidateAnalysisPass
    : PassInfoMixin<InvalidateAnalysisPass<AnalysisT>> {
  /// \brief Run this pass over some unit of IR.
  ///
  /// This pass can be run over any unit of IR and use any analysis manager
  /// provided they satisfy the basic API requirements. When this pass is
  /// created, these methods can be instantiated to satisfy whatever the
  /// context requires.
  template <typename IRUnitT, typename AnalysisManagerT, typename... ExtraArgTs>
  PreservedAnalyses run(IRUnitT &Arg, AnalysisManagerT &AM, ExtraArgTs &&...) {
    // We have to directly invalidate the analysis result as we can't
    // enumerate all other analyses and use the preserved set to control it.
    AM.template invalidate<AnalysisT>(Arg);

    return PreservedAnalyses::all();
  }
};

/// \brief A utility pass that does nothing but preserves no analyses.
///
/// As a consequence fo not preserving any analyses, this pass will force all
/// analysis passes to be re-run to produce fresh results if any are needed.
struct InvalidateAllAnalysesPass : PassInfoMixin<InvalidateAllAnalysesPass> {
  /// \brief Run this pass over some unit of IR.
  template <typename IRUnitT, typename AnalysisManagerT, typename... ExtraArgTs>
  PreservedAnalyses run(IRUnitT &, AnalysisManagerT &, ExtraArgTs &&...) {
    return PreservedAnalyses::none();
  }
};

/// A utility pass template that simply runs another pass multiple times.
///
/// This can be useful when debugging or testing passes. It also serves as an
/// example of how to extend the pass manager in ways beyond composition.
template <typename PassT>
class RepeatedPass : public PassInfoMixin<RepeatedPass<PassT>> {
public:
  RepeatedPass(int Count, PassT P) : Count(Count), P(std::move(P)) {}

  template <typename IRUnitT, typename AnalysisManagerT, typename... Ts>
  PreservedAnalyses run(IRUnitT &Arg, AnalysisManagerT &AM, Ts &&... Args) {
    auto PA = PreservedAnalyses::all();
    for (int i = 0; i < Count; ++i)
      PA.intersect(P.run(Arg, AM, std::forward<Ts>(Args)...));
    return PA;
  }

private:
  int Count;
  PassT P;
};

template <typename PassT>
RepeatedPass<PassT> createRepeatedPass(int Count, PassT P) {
  return RepeatedPass<PassT>(Count, std::move(P));
}

}

#endif
