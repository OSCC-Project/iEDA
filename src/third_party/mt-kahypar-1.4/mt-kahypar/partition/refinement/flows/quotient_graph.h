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

#include "tbb/concurrent_queue.h"
#include "tbb/concurrent_vector.h"
#include "tbb/enumerable_thread_specific.h"

#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/partition/refinement/flows/refiner_adapter.h"
#include "mt-kahypar/parallel/atomic_wrapper.h"
#include "mt-kahypar/utils/randomize.h"

namespace mt_kahypar {

struct BlockPair {
  PartitionID i = kInvalidPartition;
  PartitionID j = kInvalidPartition;
};

template<typename TypeTraits>
class QuotientGraph {

  static constexpr bool debug = false;
  static constexpr bool enable_heavy_assert = false;

  using PartitionedHypergraph = typename TypeTraits::PartitionedHypergraph;

  // ! Represents an edge of the quotient graph
  struct QuotientGraphEdge {
    QuotientGraphEdge() :
      blocks(),
      ownership(INVALID_SEARCH_ID),
      is_in_queue(false),
      cut_hes(),
      num_cut_hes(0),
      cut_he_weight(0),
      num_improvements_found(0),
      total_improvement(0) { }

    // ! Adds a cut hyperedge to this quotient graph edge
    void add_hyperedge(const HyperedgeID he,
                       const HyperedgeWeight weight);

    void reset();

    // ! Returns true, if quotient graph edge is acquired by a search
    bool isAcquired() const {
      return ownership.load() != INVALID_SEARCH_ID;
    }

    // ! Tries to acquire quotient graph edge with corresponding search id
    bool acquire(const SearchID search_id) {
      SearchID expected = INVALID_SEARCH_ID;
      SearchID desired = search_id;
      return ownership.compare_exchange_strong(expected, desired);
    }

    // ! Releases quotient graph edge
    void release(const SearchID search_id) {
      unused(search_id);
      ASSERT(ownership.load() == search_id);
      ownership.store(INVALID_SEARCH_ID);
    }

    bool isInQueue() const {
      return is_in_queue.load(std::memory_order_relaxed);
    }

    // ! Marks quotient graph edge as in queue. Queued edges are scheduled
    // ! for refinement.
    bool markAsInQueue() {
      bool expected = false;
      bool desired = true;
      return is_in_queue.compare_exchange_strong(expected, desired);
    }

    // ! Marks quotient graph edge as nnot in queue
    bool markAsNotInQueue() {
      bool expected = true;
      bool desired = false;
      return is_in_queue.compare_exchange_strong(expected, desired);
    }

    // ! Block pair this quotient graph edge represents
    BlockPair blocks;
    // ! Atomic that contains the search currently constructing
    // ! a problem on this block pair
    CAtomic<SearchID> ownership;
    // ! True, if block is contained in block scheduler queue
    CAtomic<bool> is_in_queue;
    // ! Cut hyperedges of block pair
    tbb::concurrent_vector<HyperedgeID> cut_hes;
    // ! Number of cut hyperedges
    CAtomic<size_t> num_cut_hes;
    // ! Current weight of all cut hyperedges
    CAtomic<HyperedgeWeight> cut_he_weight;
    // ! Number of improvements found on this block pair
    CAtomic<size_t> num_improvements_found;
    // ! Total improvement found on this block pair
    CAtomic<HyperedgeWeight> total_improvement;
  };

  /**
   * Maintains the block pair of a round of the active block scheduling strategy
   */
  class ActiveBlockSchedulingRound {

   public:
    explicit ActiveBlockSchedulingRound(const Context& context,
                                        vec<vec<QuotientGraphEdge>>& quotient_graph) :
      _context(context),
      _quotient_graph(quotient_graph),
      _unscheduled_blocks(),
      _round_improvement(0),
      _active_blocks_lock(),
      _active_blocks(context.partition.k, false),
      _remaining_blocks(0) { }

    // ! Pops a block pair from the queue.
    // ! Returns true, if a block pair was successfully popped from the queue.
    // ! The corresponding block pair will be stored in blocks.
    bool popBlockPairFromQueue(BlockPair& blocks);

    // ! Pushes a block pair into the queue.
    // ! Return true, if the block pair was successfully pushed into the queue.
    // ! Note, that a block pair is only allowed to be contained in one queue
    // ! (there are multiple active rounds).
    bool pushBlockPairIntoQueue(const BlockPair& blocks);

    // ! Signals that the search on the corresponding block pair terminated.
    void finalizeSearch(const BlockPair& blocks,
                        const HyperedgeWeight improvement,
                        bool& block_0_becomes_active,
                        bool& block_1_becomes_active);

    HyperedgeWeight roundImprovement() const {
      return _round_improvement.load(std::memory_order_relaxed);
    }

    bool isActive(const PartitionID block) const {
      ASSERT(block < _context.partition.k);
      return _active_blocks[block];
    }

    size_t numRemainingBlocks() const {
      return _remaining_blocks;
    }

   const Context& _context;
   // ! Quotient graph
    vec<vec<QuotientGraphEdge>>& _quotient_graph;
    // ! Queue that contains all unscheduled block pairs of the current round
    tbb::concurrent_queue<BlockPair> _unscheduled_blocks;
    // ! Current improvement made in this round
    CAtomic<HyperedgeWeight> _round_improvement;
    // Active blocks for next round
    SpinLock _active_blocks_lock;
    vec<uint8_t> _active_blocks;
    // Remaining active block pairs in the current round.
    CAtomic<size_t> _remaining_blocks;
  };

  /**
   * Implements the active block scheduling strategy.
   * The active block scheduling strategy proceeds in rounds. In each round,
   * all active edges of the quotient graph are scheduled for refinement.
   * A edge is called active, if at least of the blocks is active and a block
   * is called active if a refinement involving that block in the previous round
   * leads to an improvement. In the sequential active block scheduling strategy
   * the rounds acts as synchronization barriers. However, to achieve better scalibility
   * we immediatly schedule an edge in the next round once we find an improvement.
   * Thus, there can be multiple active searches that process block pairs from different
   * rounds. However, block pairs from earlier rounds have an higher priority to be
   * scheduled.
   */
  class ActiveBlockScheduler {

   public:
    explicit ActiveBlockScheduler(const Context& context,
                                  vec<vec<QuotientGraphEdge>>& quotient_graph) :
      _context(context),
      _quotient_graph(quotient_graph),
      _num_rounds(0),
      _rounds(),
      _min_improvement_per_round(0),
      _terminate(false),
      _round_lock(),
      _first_active_round(0),
      _is_input_hypergraph(false) { }

    // ! Initialize the first round of the active block scheduling strategy
    void initialize(const vec<uint8_t>& active_blocks,
                    const bool is_input_hypergraph);

    // ! Pops a block pair from the queue.
    // ! Returns true, if a block pair was successfully popped from the queue.
    // ! The corresponding block pair and the round to which this blocks corresponds
    // ! to are stored in blocks and round.
    bool popBlockPairFromQueue(BlockPair& blocks, size_t& round);

    // ! Signals that the search on the corresponding block pair terminated.
    // ! If one the two blocks become active, we immediatly schedule all edges
    // ! adjacent in the quotient graph in the next round of active block scheduling
    void finalizeSearch(const BlockPair& blocks,
                        const size_t round,
                        const HyperedgeWeight improvement);

    size_t numRemainingBlocks() const {
      size_t num_remaining_blocks = 0;
      for ( size_t i = _first_active_round; i < _num_rounds; ++i ) {
        num_remaining_blocks += _rounds[i].numRemainingBlocks();
      }
      return num_remaining_blocks;
    }

    void setObjective(const HyperedgeWeight objective) {
      _min_improvement_per_round =
        _context.refinement.flows.min_relative_improvement_per_round * objective;
    }

   private:

    void reset() {
      _num_rounds.store(0);
      _rounds.clear();
      _first_active_round = 0;
      _terminate = false;
    }

    bool isActiveBlockPair(const PartitionID i,
                           const PartitionID j) const;

    const Context& _context;
    // ! Quotient graph
    vec<vec<QuotientGraphEdge>>& _quotient_graph;
    // Contains all active block scheduling rounds
    CAtomic<size_t> _num_rounds;
    tbb::concurrent_vector<ActiveBlockSchedulingRound> _rounds;
    // ! Minimum improvement per round to continue with next round
    HyperedgeWeight _min_improvement_per_round;
    // ! If true, then search is immediatly terminated
    bool _terminate;
    // ! First Active Round
    SpinLock _round_lock;
    size_t _first_active_round;
    // ! Indicate if the current hypergraph represents the input hypergraph
    bool _is_input_hypergraph;
  };

  // Contains information required by a local search
  struct Search {
    explicit Search(const BlockPair& blocks, const size_t round) :
      blocks(blocks),
      round(round),
      is_finalized(false) { }

    // ! Block pair on which this search operates on
    BlockPair blocks;
    // ! Round of active block scheduling
    size_t round;
    // ! Flag indicating if construction of the corresponding search
    // ! is finalized
    bool is_finalized;
  };

public:
  static constexpr SearchID INVALID_SEARCH_ID = std::numeric_limits<SearchID>::max();

  explicit QuotientGraph(const HyperedgeID num_hyperedges,
                         const Context& context) :
    _phg(nullptr),
    _context(context),
    _initial_num_edges(num_hyperedges),
    _current_num_edges(kInvalidHyperedge),
    _quotient_graph(context.partition.k,
      vec<QuotientGraphEdge>(context.partition.k)),
    _register_search_lock(),
    _active_block_scheduler(context, _quotient_graph),
    _num_active_searches(0),
    _searches() {
    for ( PartitionID i = 0; i < _context.partition.k; ++i ) {
      for ( PartitionID j = i + 1; j < _context.partition.k; ++j ) {
        _quotient_graph[i][j].blocks.i = i;
        _quotient_graph[i][j].blocks.j = j;
      }
    }
  }

  QuotientGraph(const QuotientGraph&) = delete;
  QuotientGraph(QuotientGraph&&) = delete;

  QuotientGraph & operator= (const QuotientGraph &) = delete;
  QuotientGraph & operator= (QuotientGraph &&) = delete;

  /**
   * Returns a new search id which is associated with a certain number
   * of block pairs. The corresponding search can request hyperedges
   * with the search id that are cut between the corresponding blocks
   * associated with the search. If there are currently no block pairs
   * available then INVALID_SEARCH_ID is returned.
   */
  SearchID requestNewSearch(FlowRefinerAdapter<TypeTraits>& refiner);

  // ! Returns the block pair on which the corresponding search operates on
  BlockPair getBlockPair(const SearchID search_id) const {
    ASSERT(search_id < _searches.size());
    return _searches[search_id].blocks;
  }

  // ! Number of block pairs used by the corresponding search
  size_t numBlockPairs(const SearchID) const {
    return 1;
  }

  template<typename F>
  void doForAllCutHyperedgesOfSearch(const SearchID search_id, const F& f) {
    const BlockPair& blocks = _searches[search_id].blocks;
    const size_t num_cut_hes = _quotient_graph[blocks.i][blocks.j].num_cut_hes.load();
    std::shuffle(_quotient_graph[blocks.i][blocks.j].cut_hes.begin(),
                 _quotient_graph[blocks.i][blocks.j].cut_hes.begin() + num_cut_hes,
                 utils::Randomize::instance().getGenerator());
    for ( size_t i = 0; i < num_cut_hes; ++i ) {
      const HyperedgeID he = _quotient_graph[blocks.i][blocks.j].cut_hes[i];
      if ( _phg->pinCountInPart(he, blocks.i) > 0 && _phg->pinCountInPart(he, blocks.j) > 0 ) {
        f(he);
      }
    }
  }


  /**
   * Notifies the quotient graph that hyperedge he contains
   * a new block, which was previously not contained. The thread
   * that increases the pin count of hyperedge he in the corresponding
   * block to 1 is responsible to call this function.
   */
  void addNewCutHyperedge(const HyperedgeID he,
                          const PartitionID block);

  /**
   * Notify the quotient graph that the construction of the corresponding
   * search is completed. The corresponding block pairs associated with the
   * search are made available again for other searches.
   */
  void finalizeConstruction(const SearchID search_id);

  /**
   * Notify the quotient graph that the corrseponding search terminated.
   * If the search improves the quality of the partition (success == true),
   * we reinsert all hyperedges that were used throughout the construction
   * and are still cut between the corresponding block.
   */
  void finalizeSearch(const SearchID search_id,
                      const HyperedgeWeight total_improvement);

  // ! Initializes the quotient graph. This includes to find
  // ! all cut hyperedges between all block pairs
  void initialize(const PartitionedHypergraph& phg);

  void setObjective(const HyperedgeWeight objective) {
    _active_block_scheduler.setObjective(objective);
  }

  size_t numActiveBlockPairs() const;

  // ! Only for testing
  HyperedgeWeight getCutHyperedgeWeightOfBlockPair(const PartitionID i, const PartitionID j) const {
    ASSERT(i < j);
    ASSERT(0 <= i && i < _context.partition.k);
    ASSERT(0 <= j && j < _context.partition.k);
    return _quotient_graph[i][j].cut_he_weight;
  }

  void changeNumberOfBlocks(const PartitionID new_k);

 private:

  void resetQuotientGraphEdges();

  bool isInputHypergraph() const {
    return _current_num_edges == _initial_num_edges;
  }

  const PartitionedHypergraph* _phg;
  const Context& _context;
  const HypernodeID _initial_num_edges;
  HypernodeID _current_num_edges;

  // ! Each edge contains stats and the cut hyperedges
  // ! of the block pair which its represents.
  vec<vec<QuotientGraphEdge>> _quotient_graph;

  SpinLock _register_search_lock;
  // ! Queue that contains all block pairs.
  ActiveBlockScheduler _active_block_scheduler;

  // ! Number of active searches
  CAtomic<size_t> _num_active_searches;
  // ! Information about searches that are currently running
  tbb::concurrent_vector<Search> _searches;
};

}  // namespace kahypar
