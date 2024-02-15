/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2021 Tobias Heuer <tobias.heuer@kit.edu>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/

#pragma once

#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/partition/refinement/i_refiner.h"
#include "mt-kahypar/partition/refinement/flows/quotient_graph.h"
#include "mt-kahypar/partition/refinement/flows/refiner_adapter.h"
#include "mt-kahypar/partition/refinement/flows/problem_construction.h"
#include "mt-kahypar/partition/refinement/gains/gain_cache_ptr.h"
#include "mt-kahypar/parallel/atomic_wrapper.h"
#include "mt-kahypar/utils/utilities.h"

namespace mt_kahypar {

namespace {

  static constexpr size_t PROGRESS_BAR_SIZE = 50;

  template<typename F>
  std::string progress_bar(const size_t value, const size_t max, const F& f) {
    const double percentage = static_cast<double>(value) / std::max(max,UL(1));
    const size_t ticks = PROGRESS_BAR_SIZE * percentage;
    std::stringstream pbar_str;
    pbar_str << "|"
             << f(percentage) << std::string(ticks, '|') << END
             << std::string(PROGRESS_BAR_SIZE - ticks, ' ')
             << "| " << std::setprecision(2) << (100.0 * percentage) << "% (" << value << ")";
    return pbar_str.str();
  }
}

template<typename GraphAndGainTypes>
class FlowRefinementScheduler final : public IRefiner {

  static constexpr bool debug = false;
  static constexpr bool enable_heavy_assert = false;

  using TypeTraits = typename GraphAndGainTypes::TypeTraits;
  using PartitionedHypergraph = typename GraphAndGainTypes::PartitionedHypergraph;
  using GainCache = typename GraphAndGainTypes::GainCache;
  using AttributedGains = typename GraphAndGainTypes::AttributedGains;

  struct RefinementStats {
    RefinementStats(utils::Stats& stats) :
      _stats(stats),
      num_refinements(0),
      num_improvements(0),
      num_time_limits(0),
      correct_expected_improvement(0),
      zero_gain_improvement(0),
      failed_updates_due_to_conflicting_moves(0),
      failed_updates_due_to_conflicting_moves_without_rollback(0),
      failed_updates_due_to_balance_constraint(0),
      total_improvement(0) { }

    void reset() {
      num_refinements.store(0);
      num_improvements.store(0);
      num_time_limits.store(0);
      correct_expected_improvement.store(0);
      zero_gain_improvement.store(0);
      failed_updates_due_to_conflicting_moves.store(0);
      failed_updates_due_to_conflicting_moves_without_rollback.store(0);
      failed_updates_due_to_balance_constraint.store(0);
      total_improvement.store(0);
    }

    void update_global_stats();

    utils::Stats& _stats;
    CAtomic<int64_t> num_refinements;
    CAtomic<int64_t> num_improvements;
    CAtomic<int64_t> num_time_limits;
    CAtomic<int64_t> correct_expected_improvement;
    CAtomic<int64_t> zero_gain_improvement;
    CAtomic<int64_t> failed_updates_due_to_conflicting_moves;
    CAtomic<int64_t> failed_updates_due_to_conflicting_moves_without_rollback;
    CAtomic<int64_t> failed_updates_due_to_balance_constraint;
    CAtomic<HyperedgeWeight> total_improvement;
  };

  struct PartWeightUpdateResult {
    bool is_balanced = true;
    PartitionID overloaded_block = kInvalidPartition;
    HypernodeWeight overload_weight = 0;
  };

  friend std::ostream & operator<< (std::ostream& str, const RefinementStats& stats) {
    str << "\n";
    str << "Total Improvement                   = " << stats.total_improvement << "\n";
    str << "Number of Flow-Based Refinements    = " << stats.num_refinements << "\n";
    str << "+ No Improvements                   = "
        << progress_bar(stats.num_refinements - stats.num_improvements, stats.num_refinements,
            [&](const double percentage) { return percentage > 0.9 ? RED : percentage > 0.75 ? YELLOW : GREEN; }) << "\n";
    str << "+ Number of Improvements            = "
        << progress_bar(stats.num_improvements, stats.num_refinements,
            [&](const double percentage) { return percentage < 0.05 ? RED : percentage < 0.15 ? YELLOW : GREEN; }) << "\n";
    str << "  + Correct Expected Improvements   = "
        << progress_bar(stats.correct_expected_improvement, stats.num_improvements,
            [&](const double percentage) { return percentage > 0.9 ? GREEN : percentage > 0.75 ? YELLOW : RED; }) << "\n";
    str << "  + Incorrect Expected Improvements = "
        << progress_bar(stats.num_improvements - stats.correct_expected_improvement, stats.num_improvements,
            [&](const double percentage) { return percentage < 0.1 ? GREEN : percentage < 0.25 ? YELLOW : RED; }) << "\n";
    str << "  + Zero-Gain Improvements          = "
        << progress_bar(stats.zero_gain_improvement, stats.num_improvements,
            [&](const double) { return WHITE; }) << "\n";
    str << "+ Failed due to Balance Constraint  = "
        << progress_bar(stats.failed_updates_due_to_balance_constraint, stats.num_refinements,
            [&](const double percentage) { return percentage < 0.01 ? GREEN : percentage < 0.05 ? YELLOW : RED; }) << "\n";
    str << "+ Failed due to Conflicting Moves   = "
        << progress_bar(stats.failed_updates_due_to_conflicting_moves, stats.num_refinements,
            [&](const double percentage) { return percentage < 0.01 ? GREEN : percentage < 0.05 ? YELLOW : RED; }) << "\n";
    str << "+ Time Limits                       = "
        << progress_bar(stats.num_time_limits, stats.num_refinements,
            [&](const double percentage) { return percentage < 0.0025 ? GREEN : percentage < 0.01 ? YELLOW : RED; }) << "\n";
    str << "---------------------------------------------------------------";
    return str;
  }

public:
  FlowRefinementScheduler(const HypernodeID num_hypernodes,
                          const HyperedgeID num_hyperedges,
                          const Context& context,
                          GainCache& gain_cache) :
    _phg(nullptr),
    _context(context),
    _gain_cache(gain_cache),
    _current_k(context.partition.k),
    _quotient_graph(num_hyperedges, context),
    _refiner(num_hyperedges, context),
    _constructor(num_hypernodes, num_hyperedges, context),
    _was_moved(num_hypernodes, uint8_t(false)),
    _part_weights_lock(),
    _part_weights(context.partition.k, 0),
    _max_part_weights(context.partition.k, 0),
    _stats(utils::Utilities::instance().getStats(context.utility_id)),
    _apply_moves_lock() { }

  FlowRefinementScheduler(const HypernodeID num_hypernodes,
                          const HyperedgeID num_hyperedges,
                          const Context& context,
                          gain_cache_t gain_cache) :
    FlowRefinementScheduler(num_hypernodes, num_hyperedges, context,
      GainCachePtr::cast<GainCache>(gain_cache)) { }

  FlowRefinementScheduler(const FlowRefinementScheduler&) = delete;
  FlowRefinementScheduler(FlowRefinementScheduler&&) = delete;

  FlowRefinementScheduler & operator= (const FlowRefinementScheduler &) = delete;
  FlowRefinementScheduler & operator= (FlowRefinementScheduler &&) = delete;

  /**
   * Applies the sequence of vertex moves to the partitioned hypergraph.
   * The method ensures that the move sequence does not violate
   * the balance constaint and not worsen solution quality.
   * Returns, improvement in solution quality.
   */
  HyperedgeWeight applyMoves(const SearchID search_id,
                             MoveSequence& sequence);

  /**
   * Returns the current weight of each block.
   * Note, we do not want that the underlying refiner (ILP and Flows)
   * see partially updated part weight information. Thus, we perform
   * part weight updates for a move sequence as a transaction, which
   * we protect with a spin lock.
   */
  vec<HypernodeWeight> partWeights() {
    _part_weights_lock.lock();
    vec<HypernodeWeight> _copy_part_weights(_part_weights);
    _part_weights_lock.unlock();
    return _copy_part_weights;
  }

private:
  bool refineImpl(mt_kahypar_partitioned_hypergraph_t& phg,
                  const vec<HypernodeID>& refinement_nodes,
                  Metrics& metrics,
                  double time_limit) final;

  void initializeImpl(mt_kahypar_partitioned_hypergraph_t& phg) final;

  void resizeDataStructuresForCurrentK();

  PartWeightUpdateResult partWeightUpdate(const vec<HypernodeWeight>& part_weight_deltas,
                                          const bool rollback);

  std::string blocksOfSearch(const SearchID search_id) {
    const BlockPair blocks = _quotient_graph.getBlockPair(search_id);
    return "(" + std::to_string(blocks.i) + "," + std::to_string(blocks.j) + ")";
  }

  PartitionedHypergraph* _phg;
  const Context& _context;
  GainCache& _gain_cache;
  PartitionID _current_k;

  // ! Contains information of all cut hyperedges between the
  // ! blocks of the partition
  QuotientGraph<TypeTraits> _quotient_graph;

  // ! Maintains the flow refiner instances
  FlowRefinerAdapter<TypeTraits> _refiner;

  // ! Responsible for construction of an flow problems
  ProblemConstruction<TypeTraits> _constructor;

  // ! For each vertex it store wheather the corresponding vertex
  // ! was moved or not
  vec<uint8_t> _was_moved;

  // ! Maintains the part weights of each block
  SpinLock _part_weights_lock;
  vec<HypernodeWeight> _part_weights;
  vec<HypernodeWeight> _max_part_weights;

  // ! Contains refinement statistics
  RefinementStats _stats;

  SpinLock _apply_moves_lock;
};

}  // namespace kahypar
