/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2023 Lars Gottesbüren <lars.gottesbueren@kit.edu>
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

#include "mt-kahypar/datastructures/priority_queue.h"
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/partition/metrics.h"
#include "mt-kahypar/partition/refinement/i_refiner.h"
#include "mt-kahypar/partition/refinement/i_rebalancer.h"
#include "mt-kahypar/partition/refinement/gains/gain_cache_ptr.h"

namespace mt_kahypar {

namespace rebalancer {
  struct GuardedPQ {
    GuardedPQ(PosT *handles, size_t num_nodes) : pq(handles, num_nodes) { }
    SpinLock lock;
    ds::MaxHeap<float, HypernodeID> pq;
    float top_key = std::numeric_limits<float>::min();
    void reset() {
      pq.clear();
      top_key = std::numeric_limits<float>::min();
    }
  };

  struct NodeState {
    uint8_t state = 0;

    bool canMove() const { return state == 1; }

    bool isLocked() const { return state == 2; }

    bool wasMoved() const { return state == 3; }

    // Returns true if the node is marked as movable, is not locked and taking the lock now succeeds
    bool tryLock() {
      uint8_t expected = 1;
      return state == 1 && __atomic_compare_exchange_n(&state, &expected, 2, false, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED);
    }

    void unlock() { __atomic_store_n(&state, 1, __ATOMIC_RELEASE); }

    void markAsMovedAndUnlock() { __atomic_store_n(&state, 3, __ATOMIC_RELEASE); }

    void markAsMovable() { state = 1; }

    void reset() { state = 0; }
  };

} // namespace rebalancer


template <typename GraphAndGainTypes>
class AdvancedRebalancer final : public IRebalancer {
private:
  using PartitionedHypergraph = typename GraphAndGainTypes::PartitionedHypergraph;
  using GainCache = typename GraphAndGainTypes::GainCache;
  using GainCalculator = typename GraphAndGainTypes::GainComputation;
  using AttributedGains = typename GraphAndGainTypes::AttributedGains;

  static constexpr bool debug = false;
  static constexpr bool enable_heavy_assert = false;

public:

  explicit AdvancedRebalancer(HypernodeID num_nodes,
                        const Context& context,
                        GainCache& gain_cache);

  explicit AdvancedRebalancer(HypernodeID num_nodes,
                        const Context& context,
                        gain_cache_t gain_cache);

private:
  bool refineImpl(mt_kahypar_partitioned_hypergraph_t& hypergraph,
                  const vec<HypernodeID>& refinement_nodes,
                  Metrics& best_metrics,
                  double);

  void initializeImpl(mt_kahypar_partitioned_hypergraph_t& hypergraph) final;

  bool refineAndOutputMovesImpl(mt_kahypar_partitioned_hypergraph_t& hypergraph,
                                const vec<HypernodeID>& refinement_nodes,
                                vec<vec<Move>>& moves_by_part,
                                Metrics& best_metrics,
                                const double);

  bool refineAndOutputMovesLinearImpl(mt_kahypar_partitioned_hypergraph_t& hypergraph,
                                      const vec<HypernodeID>& refinement_nodes,
                                      vec<Move>& moves,
                                      Metrics& best_metrics,
                                      const double);

  bool refineInternalParallel(mt_kahypar_partitioned_hypergraph_t& hypergraph,
                              vec<vec<Move>>* moves_by_part,
                              vec<Move>* moves_linear,
                              Metrics& best_metric);

  const Context& _context;
  GainCache& _gain_cache;
  PartitionID _current_k;
  GainCalculator _gain;


  void insertNodesInOverloadedBlocks(mt_kahypar_partitioned_hypergraph_t& hypergraph);
  std::pair<int64_t, size_t> findMoves(mt_kahypar_partitioned_hypergraph_t& hypergraph);

  ds::Array<Move> _moves;
  vec<rebalancer::GuardedPQ> _pqs;
  vec<PartitionID> _overloaded_blocks;
  vec<uint8_t> _is_overloaded;
  ds::Array<PartitionID> _target_part;
  ds::Array<PosT> _pq_handles;
  ds::Array<int> _pq_id;
  ds::Array<rebalancer::NodeState> _node_state;
};

}  // namespace mt_kahypar
