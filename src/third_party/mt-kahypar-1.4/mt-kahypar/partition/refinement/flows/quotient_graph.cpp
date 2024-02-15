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


#include "mt-kahypar/partition/refinement/flows/quotient_graph.h"

#include <queue>

#include "tbb/parallel_sort.h"

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/datastructures/sparse_map.h"


namespace mt_kahypar {

template<typename TypeTraits>
void QuotientGraph<TypeTraits>::QuotientGraphEdge::add_hyperedge(const HyperedgeID he,
                                                                 const HyperedgeWeight weight) {
  cut_hes.push_back(he);
  cut_he_weight += weight;
  ++num_cut_hes;
}

template<typename TypeTraits>
void QuotientGraph<TypeTraits>::QuotientGraphEdge::reset() {
  cut_hes.clear();
  ownership.store(INVALID_SEARCH_ID, std::memory_order_relaxed);
  is_in_queue.store(false, std::memory_order_relaxed);
  num_cut_hes.store(0, std::memory_order_relaxed);
  cut_he_weight.store(0, std::memory_order_relaxed);
}

template<typename TypeTraits>
bool QuotientGraph<TypeTraits>::ActiveBlockSchedulingRound::popBlockPairFromQueue(BlockPair& blocks) {
  blocks.i = kInvalidPartition;
  blocks.j = kInvalidPartition;
  if ( _unscheduled_blocks.try_pop(blocks) ) {
    _quotient_graph[blocks.i][blocks.j].markAsNotInQueue();
  }
  return blocks.i != kInvalidPartition && blocks.j != kInvalidPartition;
}

template<typename TypeTraits>
void QuotientGraph<TypeTraits>::ActiveBlockSchedulingRound::finalizeSearch(const BlockPair& blocks,
                                                                           const HyperedgeWeight improvement,
                                                                           bool& block_0_becomes_active,
                                                                           bool& block_1_becomes_active) {
  _round_improvement += improvement;
  --_remaining_blocks;
  if ( improvement > 0 ) {
    _active_blocks_lock.lock();
    block_0_becomes_active = !_active_blocks[blocks.i];
    block_1_becomes_active = !_active_blocks[blocks.j];
    _active_blocks[blocks.i] = true;
    _active_blocks[blocks.j] = true;
    _active_blocks_lock.unlock();
  }
}

template<typename TypeTraits>
bool QuotientGraph<TypeTraits>::ActiveBlockSchedulingRound::pushBlockPairIntoQueue(const BlockPair& blocks) {
  QuotientGraphEdge& qg_edge = _quotient_graph[blocks.i][blocks.j];
  if ( qg_edge.markAsInQueue() ) {
    _unscheduled_blocks.push(blocks);
    ++_remaining_blocks;
    return true;
  } else {
    return false;
  }
}

template<typename TypeTraits>
void QuotientGraph<TypeTraits>::ActiveBlockScheduler::initialize(const vec<uint8_t>& active_blocks,
                                                                 const bool is_input_hypergraph) {
  reset();
  _is_input_hypergraph = is_input_hypergraph;

  HyperedgeWeight best_total_improvement = 1;
  for ( PartitionID i = 0; i < _context.partition.k; ++i ) {
    for ( PartitionID j = i + 1; j < _context.partition.k; ++j ) {
      best_total_improvement = std::max(best_total_improvement,
        _quotient_graph[i][j].total_improvement.load(std::memory_order_relaxed));
    }
  }

  vec<BlockPair> active_block_pairs;
  for ( PartitionID i = 0; i < _context.partition.k; ++i ) {
    for ( PartitionID j = i + 1; j < _context.partition.k; ++j ) {
      if ( isActiveBlockPair(i, j) && ( active_blocks[i] || active_blocks[j] ) ) {
        active_block_pairs.push_back( BlockPair { i, j } );
      }
    }
  }

  if ( active_block_pairs.size() > 0 ) {
    std::sort(active_block_pairs.begin(), active_block_pairs.end(),
      [&](const BlockPair& lhs, const BlockPair& rhs) {
        return _quotient_graph[lhs.i][lhs.j].total_improvement >
          _quotient_graph[rhs.i][rhs.j].total_improvement ||
          ( _quotient_graph[lhs.i][lhs.j].total_improvement ==
            _quotient_graph[rhs.i][rhs.j].total_improvement &&
            _quotient_graph[lhs.i][lhs.j].cut_he_weight >
            _quotient_graph[rhs.i][rhs.j].cut_he_weight );
      });
    _rounds.emplace_back(_context, _quotient_graph);
    ++_num_rounds;
    for ( const BlockPair& blocks : active_block_pairs ) {
      DBG << "Schedule blocks (" << blocks.i << "," << blocks.j << ") in round 1 ("
          << "Total Improvement =" << _quotient_graph[blocks.i][blocks.j].total_improvement << ","
          << "Cut Weight =" << _quotient_graph[blocks.i][blocks.j].cut_he_weight << ")";
      _rounds.back().pushBlockPairIntoQueue(blocks);
    }
  }
}

template<typename TypeTraits>
bool QuotientGraph<TypeTraits>::ActiveBlockScheduler::popBlockPairFromQueue(BlockPair& blocks, size_t& round) {
  bool success = false;
  round = _first_active_round;
  while ( !_terminate && round < _num_rounds ) {
    success = _rounds[round].popBlockPairFromQueue(blocks);
    if ( success ) {
      break;
    }
    ++round;
  }

  if ( success && round == _num_rounds - 1 ) {
    _round_lock.lock();
    if ( round == _num_rounds - 1 ) {
      // There must always be a next round available such that we can
      // reschedule block pairs that become active.
      _rounds.emplace_back(_context, _quotient_graph);
      ++_num_rounds;
    }
    _round_lock.unlock();
  }

  return success;
}

template<typename TypeTraits>
void QuotientGraph<TypeTraits>::ActiveBlockScheduler::finalizeSearch(const BlockPair& blocks,
                                                         const size_t round,
                                                         const HyperedgeWeight improvement) {
  ASSERT(round < _rounds.size());
  bool block_0_becomes_active = false;
  bool block_1_becomes_active = false;
  _rounds[round].finalizeSearch(blocks, improvement,
    block_0_becomes_active, block_1_becomes_active);

  if ( block_0_becomes_active ) {
    // If blocks.i becomes active, we push all adjacent blocks into the queue of the next round
    ASSERT(round + 1 < _rounds.size());
    for ( PartitionID other = 0; other < _context.partition.k; ++other ) {
      if ( blocks.i != other ) {
        const PartitionID block_0 = std::min(blocks.i, other);
        const PartitionID block_1 = std::max(blocks.i, other);
        if ( isActiveBlockPair(block_0, block_1) ) {
          DBG << "Schedule blocks (" << block_0 << "," << block_1 << ") in round" << (round + 2) << " ("
              << "Total Improvement =" << _quotient_graph[block_0][block_1].total_improvement << ","
              << "Cut Weight =" << _quotient_graph[block_0][block_1].cut_he_weight << ")";
          _rounds[round + 1].pushBlockPairIntoQueue(BlockPair { block_0, block_1 });
        }
      }
    }
  }

  if ( block_1_becomes_active ) {
    // If blocks.j becomes active, we push all adjacent blocks into the queue of the next round
    ASSERT(round + 1 < _rounds.size());
    for ( PartitionID other = 0; other < _context.partition.k; ++other ) {
      if ( blocks.j != other ) {
        const PartitionID block_0 = std::min(blocks.j, other);
        const PartitionID block_1 = std::max(blocks.j, other);
        if ( isActiveBlockPair(block_0, block_1) ) {
          DBG << "Schedule blocks (" << block_0 << "," << block_1 << ") in round" << (round + 2) << " ("
              << "Total Improvement =" << _quotient_graph[block_0][block_1].total_improvement << ","
              << "Cut Weight =" << _quotient_graph[block_0][block_1].cut_he_weight << ")";
          _rounds[round + 1].pushBlockPairIntoQueue(BlockPair { block_0, block_1 });
        }
      }
    }
  }

  // Special case
  if ( improvement > 0 && !_quotient_graph[blocks.i][blocks.j].isInQueue() && isActiveBlockPair(blocks.i, blocks.j) &&
       ( _rounds[round].isActive(blocks.i) || _rounds[round].isActive(blocks.j) ) ) {
        // The active block scheduling strategy works in multiple rounds and each contain a separate queue
        // to store active block pairs. A block pair is only allowed to be contained in one queue.
        // If a block becomes active, we schedule all quotient graph edges incident to the block in
        // the next round. However, there could be some edges that are already contained in a queue of
        // a previous round, which are then not scheduled in the next round. If this edge is scheduled and
        // leads to an improvement, we schedule it in the next round here.
        DBG << "Schedule blocks (" << blocks.i << "," << blocks.j << ") in round" << (round + 2) << " ("
            << "Total Improvement =" << _quotient_graph[blocks.i][blocks.j].total_improvement << ","
            << "Cut Weight =" << _quotient_graph[blocks.i][blocks.j].cut_he_weight << ")";
        _rounds[round + 1].pushBlockPairIntoQueue(BlockPair { blocks.i, blocks.j });
  }

  if ( round == _first_active_round && _rounds[round].numRemainingBlocks() == 0 ) {
    _round_lock.lock();
    // We consider a round as finished, if the previous round is also finished and there
    // are no remaining blocks in the queue of that round.
    while ( _first_active_round < _rounds.size() &&
            _rounds[_first_active_round].numRemainingBlocks() == 0 ) {
      DBG << GREEN << "Round" << (_first_active_round + 1) << "terminates with improvement"
          << _rounds[_first_active_round].roundImprovement() << "("
          << "Minimum Required Improvement =" << _min_improvement_per_round << ")" << END;
      // We require that minimum improvement per round must be greater than a threshold,
      // otherwise we terminate early
      _terminate = _rounds[_first_active_round].roundImprovement() < _min_improvement_per_round;
      ++_first_active_round;
    }
    _round_lock.unlock();
  }
}

template<typename TypeTraits>
bool QuotientGraph<TypeTraits>::ActiveBlockScheduler::isActiveBlockPair(const PartitionID i,
                                                                        const PartitionID j) const {
  const bool skip_small_cuts = !_is_input_hypergraph &&
    _context.refinement.flows.skip_small_cuts;
  const bool contains_enough_cut_hes =
    (skip_small_cuts && _quotient_graph[i][j].cut_he_weight > 10) ||
    (!skip_small_cuts && _quotient_graph[i][j].cut_he_weight > 0);
  const bool is_promising_blocks_pair =
    !_context.refinement.flows.skip_unpromising_blocks ||
      ( _first_active_round == 0 || _quotient_graph[i][j].num_improvements_found > 0 );
  return contains_enough_cut_hes && is_promising_blocks_pair;
}

template<typename TypeTraits>
SearchID QuotientGraph<TypeTraits>::requestNewSearch(FlowRefinerAdapter<TypeTraits>& refiner) {
  ASSERT(_phg);
  SearchID search_id = INVALID_SEARCH_ID;
  BlockPair blocks { kInvalidPartition, kInvalidPartition };
  size_t round = 0;
  bool success = _active_block_scheduler.popBlockPairFromQueue(blocks, round);
  _register_search_lock.lock();

  const SearchID tmp_search_id = _searches.size();
  if ( success && _quotient_graph[blocks.i][blocks.j].acquire(tmp_search_id) ) {
    ++_num_active_searches;
    // Create new search
    search_id = tmp_search_id;
    _searches.emplace_back(blocks, round);
    _register_search_lock.unlock();

    // Associate refiner with search id
    success = refiner.registerNewSearch(search_id, *_phg);
    ASSERT(success); unused(success);
  } else {
    _register_search_lock.unlock();
    if ( success ) {
      _active_block_scheduler.finalizeSearch(blocks, round, 0);
    }
  }
  return search_id;
}

template<typename TypeTraits>
void QuotientGraph<TypeTraits>::addNewCutHyperedge(const HyperedgeID he,
                                                   const PartitionID block) {
  ASSERT(_phg);
  ASSERT(_phg->pinCountInPart(he, block) > 0);
  // Add hyperedge he as a cut hyperedge to each block pair that contains 'block'
  for ( const PartitionID& other_block : _phg->connectivitySet(he) ) {
    if ( other_block != block ) {
      _quotient_graph[std::min(block, other_block)][std::max(block, other_block)]
        .add_hyperedge(he, _phg->edgeWeight(he));
    }
  }
}

template<typename TypeTraits>
void QuotientGraph<TypeTraits>::finalizeConstruction(const SearchID search_id) {
  ASSERT(search_id < _searches.size());
  _searches[search_id].is_finalized = true;
  const BlockPair& blocks = _searches[search_id].blocks;
  _quotient_graph[blocks.i][blocks.j].release(search_id);
}

template<typename TypeTraits>
void QuotientGraph<TypeTraits>::finalizeSearch(const SearchID search_id,
                                               const HyperedgeWeight total_improvement) {
  ASSERT(_phg);
  ASSERT(search_id < _searches.size());
  ASSERT(_searches[search_id].is_finalized);

  const BlockPair& blocks = _searches[search_id].blocks;
  QuotientGraphEdge& qg_edge = _quotient_graph[blocks.i][blocks.j];
  if ( total_improvement > 0 ) {
    // If the search improves the quality of the partition, we reinsert
    // all hyperedges that were used by the search and are still cut.
    ++qg_edge.num_improvements_found;
    qg_edge.total_improvement += total_improvement;
  }
  // In case the block pair becomes active,
  // we reinsert it into the queue
  _active_block_scheduler.finalizeSearch(
    blocks, _searches[search_id].round, total_improvement);
  --_num_active_searches;
}

template<typename TypeTraits>
void QuotientGraph<TypeTraits>::initialize(const PartitionedHypergraph& phg) {
  _phg = &phg;

  // Reset internal members
  resetQuotientGraphEdges();
  _num_active_searches.store(0, std::memory_order_relaxed);
  _searches.clear();

  // Find all cut hyperedges between the blocks
  tbb::enumerable_thread_specific<HyperedgeID> local_num_hes(0);
  phg.doParallelForAllEdges([&](const HyperedgeID he) {
    ++local_num_hes.local();
    const HyperedgeWeight edge_weight = phg.edgeWeight(he);
    for ( const PartitionID i : phg.connectivitySet(he) ) {
      for ( const PartitionID j : phg.connectivitySet(he) ) {
        if ( i < j ) {
          _quotient_graph[i][j].add_hyperedge(he, edge_weight);
        }
      }
    }
  });
  _current_num_edges = local_num_hes.combine(std::plus<HyperedgeID>());

  // Initalize block scheduler queue
  vec<uint8_t> active_blocks(_context.partition.k, true);
  _active_block_scheduler.initialize(active_blocks, isInputHypergraph());
}

template<typename TypeTraits>
void QuotientGraph<TypeTraits>::changeNumberOfBlocks(const PartitionID new_k) {
  // Reset improvement history as the number of blocks had changed
  for ( size_t i = 0; i < _quotient_graph.size(); ++i ) {
    for ( size_t j = 0; j < _quotient_graph.size(); ++j ) {
      _quotient_graph[i][j].num_improvements_found.store(0, std::memory_order_relaxed);
      _quotient_graph[i][j].total_improvement.store(0, std::memory_order_relaxed);
    }
  }

  if ( static_cast<size_t>(new_k) > _quotient_graph.size() ) {
    _quotient_graph.clear();
    _quotient_graph.assign(new_k, vec<QuotientGraphEdge>(new_k));
  }
}

template<typename TypeTraits>
size_t QuotientGraph<TypeTraits>::numActiveBlockPairs() const {
  return _active_block_scheduler.numRemainingBlocks() + _num_active_searches;
}

template<typename TypeTraits>
void QuotientGraph<TypeTraits>::resetQuotientGraphEdges() {
  for ( PartitionID i = 0; i < _context.partition.k; ++i ) {
    for ( PartitionID j = i + 1; j < _context.partition.k; ++j ) {
      _quotient_graph[i][j].reset();
    }
  }
}

INSTANTIATE_CLASS_WITH_TYPE_TRAITS(QuotientGraph)

} // namespace mt_kahypar