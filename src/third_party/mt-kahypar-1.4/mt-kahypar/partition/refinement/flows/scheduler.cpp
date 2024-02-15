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

#include "mt-kahypar/partition/refinement/flows/scheduler.h"

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/partition/metrics.h"
#include "mt-kahypar/partition/refinement/gains/gain_definitions.h"
#include "mt-kahypar/io/partitioning_output.h"
#include "mt-kahypar/utils/utilities.h"
#include "mt-kahypar/utils/cast.h"

namespace mt_kahypar {

template<typename GraphAndGainTypes>
void FlowRefinementScheduler<GraphAndGainTypes>::RefinementStats::update_global_stats() {
  _stats.update_stat("num_flow_refinements",
    num_refinements.load(std::memory_order_relaxed));
  _stats.update_stat("num_flow_improvement",
    num_improvements.load(std::memory_order_relaxed));
  _stats.update_stat("num_time_limits",
    num_time_limits.load(std::memory_order_relaxed));
  _stats.update_stat("correct_expected_improvement",
    correct_expected_improvement.load(std::memory_order_relaxed));
  _stats.update_stat("zero_gain_improvement",
    zero_gain_improvement.load(std::memory_order_relaxed));
  _stats.update_stat("failed_updates_due_to_conflicting_moves",
    failed_updates_due_to_conflicting_moves.load(std::memory_order_relaxed));
  _stats.update_stat("failed_updates_due_to_conflicting_moves_without_rollback",
    failed_updates_due_to_conflicting_moves_without_rollback.load(std::memory_order_relaxed));
  _stats.update_stat("failed_updates_due_to_balance_constraint",
    failed_updates_due_to_balance_constraint.load(std::memory_order_relaxed));
  _stats.update_stat("total_flow_refinement_improvement",
    total_improvement.load(std::memory_order_relaxed));
}

template<typename GraphAndGainTypes>
bool FlowRefinementScheduler<GraphAndGainTypes>::refineImpl(
                mt_kahypar_partitioned_hypergraph_t& hypergraph,
                const parallel::scalable_vector<HypernodeID>&,
                Metrics& best_metrics,
                const double)  {
  PartitionedHypergraph& phg = utils::cast<PartitionedHypergraph>(hypergraph);
  ASSERT(_phg == &phg);
  _quotient_graph.setObjective(best_metrics.quality);

  std::atomic<HyperedgeWeight> overall_delta(0);
  utils::Timer& timer = utils::Utilities::instance().getTimer(_context.utility_id);
  tbb::parallel_for(UL(0), _refiner.numAvailableRefiner(), [&](const size_t i) {
    while ( i < std::max(UL(1), static_cast<size_t>(
        std::ceil(_context.refinement.flows.parallel_searches_multiplier *
            _quotient_graph.numActiveBlockPairs()))) ) {
      SearchID search_id = _quotient_graph.requestNewSearch(_refiner);
      if ( search_id != QuotientGraph<TypeTraits>::INVALID_SEARCH_ID ) {
        DBG << "Start search" << search_id
            << "( Blocks =" << blocksOfSearch(search_id)
            << ", Refiner =" << i << ")";
        timer.start_timer("region_growing", "Grow Region", true);
        const Subhypergraph sub_hg =
          _constructor.construct(search_id, _quotient_graph, phg);
        _quotient_graph.finalizeConstruction(search_id);
        timer.stop_timer("region_growing");

        HyperedgeWeight delta = 0;
        bool improved_solution = false;
        if ( sub_hg.numNodes() > 0 ) {
          ++_stats.num_refinements;
          MoveSequence sequence = _refiner.refine(search_id, phg, sub_hg);

          if ( !sequence.moves.empty() ) {
            timer.start_timer("apply_moves", "Apply Moves", true);
            delta = applyMoves(search_id, sequence);
            overall_delta -= delta;
            improved_solution = sequence.state == MoveSequenceState::SUCCESS && delta > 0;
            timer.stop_timer("apply_moves");
          } else if ( sequence.state == MoveSequenceState::TIME_LIMIT ) {
            ++_stats.num_time_limits;
            DBG << RED << "Search" << search_id << "reaches the time limit ( Time Limit ="
                << _refiner.timeLimit() << "s )" << END;
          }
        }
        _quotient_graph.finalizeSearch(search_id, improved_solution ? delta : 0);
        _refiner.finalizeSearch(search_id);
        DBG << "End search" << search_id
            << "( Blocks =" << blocksOfSearch(search_id)
            << ", Refiner =" << i
            << ", Running Time =" << _refiner.runningTime(search_id) << ")";
      } else {
        break;
      }
    }
    _refiner.terminateRefiner();
    DBG << RED << "Refiner" << i << "terminates!" << END;
  });

  DBG << _stats;

  ASSERT([&]() {
    for ( PartitionID i = 0; i < _context.partition.k; ++i ) {
      if ( _part_weights[i] != phg.partWeight(i) ) {
        LOG << V(_part_weights[i]) << V(phg.partWeight(i));
        return false;
      }
    }
    return true;
  }(), "Concurrent part weight updates failed!");

  // Update metrics statistics
  HEAVY_REFINEMENT_ASSERT(best_metrics.quality + overall_delta == metrics::quality(phg, _context),
    V(best_metrics.quality) << V(overall_delta) << V(metrics::quality(phg, _context)));
  best_metrics.quality += overall_delta;
  best_metrics.imbalance = metrics::imbalance(phg, _context);
  _stats.update_global_stats();

  // Update Gain Cache
  if ( _context.forceGainCacheUpdates() && _gain_cache.isInitialized() ) {
    phg.doParallelForAllNodes([&](const HypernodeID& hn) {
      if ( _was_moved[hn] ) {
        _gain_cache.recomputeInvalidTerms(phg, hn);
        _was_moved[hn] = uint8_t(false);
      }
    });
  }

  HEAVY_REFINEMENT_ASSERT(phg.checkTrackedPartitionInformation(_gain_cache));
  _phg = nullptr;
  return overall_delta.load(std::memory_order_relaxed) < 0;
}

template<typename GraphAndGainTypes>
void FlowRefinementScheduler<GraphAndGainTypes>::initializeImpl(mt_kahypar_partitioned_hypergraph_t& hypergraph)  {
  PartitionedHypergraph& phg = utils::cast<PartitionedHypergraph>(hypergraph);
  _phg = &phg;
  resizeDataStructuresForCurrentK();

  // Initialize Part Weights
  for ( PartitionID i = 0; i < _context.partition.k; ++i ) {
    _part_weights[i] = phg.partWeight(i);
    _max_part_weights[i] = std::max(
      phg.partWeight(i), _context.partition.max_part_weights[i]);
  }

  _stats.reset();
  utils::Timer& timer = utils::Utilities::instance().getTimer(_context.utility_id);
  timer.start_timer("initialize_quotient_graph", "Initialize Quotient Graph");
  _quotient_graph.initialize(phg);
  timer.stop_timer("initialize_quotient_graph");

  const size_t max_parallism = _context.refinement.flows.num_parallel_searches;
  DBG << "Initial Active Block Pairs =" << _quotient_graph.numActiveBlockPairs()
      << ", Initial Num Threads =" << max_parallism;
  _refiner.initialize(max_parallism);
}

template<typename GraphAndGainTypes>
void FlowRefinementScheduler<GraphAndGainTypes>::resizeDataStructuresForCurrentK() {
  if ( _current_k != _context.partition.k ) {
    _current_k = _context.partition.k;
    // Note that in general changing the number of blocks should not resize
    // any data structure as we initialize the scheduler with the final
    // number of blocks. This is just a fallback if someone changes this in the future.
    if ( static_cast<size_t>(_current_k) > _part_weights.size() ) {
      _part_weights.resize(_current_k);
      _max_part_weights.resize(_current_k);
    }
    _quotient_graph.changeNumberOfBlocks(_current_k);
    _constructor.changeNumberOfBlocks(_current_k);
  }
}

namespace {

struct NewCutHyperedge {
  HyperedgeID he;
  PartitionID block;
};

template<typename PartitionedHypergraph, typename GainCache, typename F>
bool changeNodePart(PartitionedHypergraph& phg,
                    GainCache& gain_cache,
                    const HypernodeID hn,
                    const PartitionID from,
                    const PartitionID to,
                    const F& objective_delta,
                    const bool gain_cache_update) {
  bool success = false;
  if ( gain_cache_update && gain_cache.isInitialized()) {
    success = phg.changeNodePart(gain_cache, hn, from, to,
      std::numeric_limits<HypernodeWeight>::max(), []{}, objective_delta);
  } else {
    success = phg.changeNodePart(hn, from, to,
      std::numeric_limits<HypernodeWeight>::max(), []{}, objective_delta);
  }
  ASSERT(success);
  return success;
}

template<typename PartitionedHypergraph, typename GainCache, typename F>
void applyMoveSequence(PartitionedHypergraph& phg,
                       GainCache& gain_cache,
                       const MoveSequence& sequence,
                       const F& objective_delta,
                       const bool gain_cache_update,
                       vec<uint8_t>& was_moved,
                       vec<NewCutHyperedge>& new_cut_hes) {
  for ( const Move& move : sequence.moves ) {
    ASSERT(move.from == phg.partID(move.node));
    if ( move.from != move.to ) {
      changeNodePart(phg, gain_cache, move.node, move.from,
        move.to, objective_delta, gain_cache_update);
      was_moved[move.node] = uint8_t(true);
      // If move increases the pin count of some hyperedges in block 'move.to' to one 1
      // we set the corresponding block here.
      int i = new_cut_hes.size() - 1;
      while ( i >= 0 && new_cut_hes[i].block == kInvalidPartition ) {
        new_cut_hes[i].block = move.to;
        --i;
      }
    }
  }
}

template<typename PartitionedHypergraph, typename GainCache, typename F>
void revertMoveSequence(PartitionedHypergraph& phg,
                        GainCache& gain_cache,
                        const MoveSequence& sequence,
                        const F& objective_delta,
                        const bool gain_cache_update) {
  for ( const Move& move : sequence.moves ) {
    if ( move.from != move.to ) {
      ASSERT(phg.partID(move.node) == move.to);
      changeNodePart(phg, gain_cache, move.node, move.to,
        move.from, objective_delta, gain_cache_update);
    }
  }
}

template<typename TypeTraits>
void addCutHyperedgesToQuotientGraph(QuotientGraph<TypeTraits>& quotient_graph,
                                     const vec<NewCutHyperedge>& new_cut_hes) {
  for ( const NewCutHyperedge& new_cut_he : new_cut_hes ) {
    ASSERT(new_cut_he.block != kInvalidPartition);
    quotient_graph.addNewCutHyperedge(new_cut_he.he, new_cut_he.block);
  }
}

} // namespace

template<typename GraphAndGainTypes>
HyperedgeWeight FlowRefinementScheduler<GraphAndGainTypes>::applyMoves(const SearchID search_id, MoveSequence& sequence) {
  unused(search_id);
  ASSERT(_phg);

  // TODO: currently we lock the applyMoves method
  // => find something smarter here
  _apply_moves_lock.lock();

  // Compute Part Weight Deltas
  vec<HypernodeWeight> part_weight_deltas(_context.partition.k, 0);
  for ( Move& move : sequence.moves ) {
    move.from = _phg->partID(move.node);
    if ( move.from != move.to ) {
      const HypernodeWeight node_weight = _phg->nodeWeight(move.node);
      part_weight_deltas[move.from] -= node_weight;
      part_weight_deltas[move.to] += node_weight;
    }
  }

  HyperedgeWeight improvement = 0;
  vec<NewCutHyperedge> new_cut_hes;
  auto delta_func = [&](const SynchronizedEdgeUpdate& sync_update) {
    improvement -= AttributedGains::gain(sync_update);

    // Collect hyperedges with new blocks in its connectivity set
    if ( sync_update.pin_count_in_to_part_after == 1 ) {
      // the corresponding block will be set in applyMoveSequence(...) function
      new_cut_hes.emplace_back(NewCutHyperedge { sync_update.he, kInvalidPartition });
    }
  };

  // Update part weights atomically
  PartWeightUpdateResult update_res = partWeightUpdate(part_weight_deltas, false);
  if ( update_res.is_balanced ) {
    // Apply move sequence to partition
    applyMoveSequence(*_phg, _gain_cache, sequence, delta_func,
      _context.forceGainCacheUpdates(), _was_moved, new_cut_hes);

    if ( improvement < 0 ) {
      update_res = partWeightUpdate(part_weight_deltas, true);
      if ( update_res.is_balanced ) {
        // Move sequence worsen solution quality => Rollback
        DBG << RED << "Move sequence worsen solution quality ("
            << "Expected Improvement =" << sequence.expected_improvement
            << ", Real Improvement =" << improvement
            << ", Search ID =" << search_id << ")" << END;
        revertMoveSequence(*_phg, _gain_cache, sequence, delta_func, _context.forceGainCacheUpdates());
        ++_stats.failed_updates_due_to_conflicting_moves;
        sequence.state = MoveSequenceState::WORSEN_SOLUTION_QUALITY;
      } else {
        // Rollback would violate balance constraint => Worst Case
        ++_stats.failed_updates_due_to_conflicting_moves_without_rollback;
        sequence.state = MoveSequenceState::WORSEN_SOLUTION_QUALITY_WITHOUT_ROLLBACK;
        DBG << RED << "Rollback of move sequence violated balance constraint ( Moved Nodes ="
            << sequence.moves.size()
            << ", Expected Improvement =" << sequence.expected_improvement
            << ", Real Improvement =" << improvement
            << ", Search ID =" << search_id << ")" << END;
      }
    } else {
      ++_stats.num_improvements;
      _stats.correct_expected_improvement += (improvement == sequence.expected_improvement);
      _stats.zero_gain_improvement += (improvement == 0);
      sequence.state = MoveSequenceState::SUCCESS;
      DBG << ( improvement > 0 ? GREEN : "" ) << "SUCCESS -"
          << "Moved Nodes =" << sequence.moves.size()
          << ", Expected Improvement =" << sequence.expected_improvement
          << ", Real Improvement =" << improvement
          << ", Search ID =" << search_id << ( improvement > 0 ? END : "" );
    }
  } else {
    ++_stats.failed_updates_due_to_balance_constraint;
    sequence.state = MoveSequenceState::VIOLATES_BALANCE_CONSTRAINT;
    DBG << RED << "Move sequence violated balance constraint ( Moved Nodes ="
        << sequence.moves.size()
        << ", Expected Improvement =" << sequence.expected_improvement
        << ", Search ID =" << search_id << ")" << END;
  }

  _apply_moves_lock.unlock();

  if ( sequence.state == MoveSequenceState::SUCCESS && improvement > 0 ) {
    addCutHyperedgesToQuotientGraph(_quotient_graph, new_cut_hes);
    _stats.total_improvement += improvement;
  }

  return improvement;
}

template<typename GraphAndGainTypes>
typename FlowRefinementScheduler<GraphAndGainTypes>::PartWeightUpdateResult
FlowRefinementScheduler<GraphAndGainTypes>::partWeightUpdate(const vec<HypernodeWeight>& part_weight_deltas,
                                                          const bool rollback) {
  const HypernodeWeight multiplier = rollback ? -1 : 1;
  PartWeightUpdateResult res;
  _part_weights_lock.lock();
  PartitionID i = 0;
  for ( ; i < _context.partition.k; ++i ) {
    if ( _part_weights[i] + multiplier * part_weight_deltas[i] > _max_part_weights[i] ) {
      DBG << "Move sequence violated balance constraint of block" << i
          << "(Max =" << _max_part_weights[i]
          << ", Actual =" << (_part_weights[i] + multiplier * part_weight_deltas[i]) << ")";
      res.is_balanced = false;
      res.overloaded_block = i;
      res.overload_weight = ( _part_weights[i] + multiplier *
        part_weight_deltas[i] ) - _max_part_weights[i];
      // Move Sequence Violates Balance Constraint => Rollback
      --i;
      for ( ; i >= 0; --i ) {
        _part_weights[i] -= multiplier * part_weight_deltas[i];
      }
      break;
    }
    _part_weights[i] += multiplier * part_weight_deltas[i];
  }
  _part_weights_lock.unlock();
  return res;
}

namespace {
#define FLOW_REFINEMENT_SCHEDULER(X) FlowRefinementScheduler<X>
}

INSTANTIATE_CLASS_WITH_VALID_TRAITS(FLOW_REFINEMENT_SCHEDULER)

}
