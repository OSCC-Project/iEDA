/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2020 Lars Gottesbüren <lars.gottesbueren@kit.edu>
 * Copyright (C) 2019 Tobias Heuer <tobias.heuer@kit.edu>
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

#include <queue>

#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/partition/metrics.h"
#include "mt-kahypar/partition/refinement/i_refiner.h"
#include "mt-kahypar/partition/refinement/i_rebalancer.h"
#include "mt-kahypar/partition/refinement/gains/km1/km1_gain_computation.h"
#include "mt-kahypar/partition/refinement/gains/cut/cut_gain_computation.h"
#include "mt-kahypar/partition/refinement/gains/gain_cache_ptr.h"
#include "mt-kahypar/utils/cast.h"

namespace mt_kahypar {
template <typename GraphAndGainTypes>
class SimpleRebalancer final : public IRebalancer {
 private:
  using PartitionedHypergraph = typename GraphAndGainTypes::PartitionedHypergraph;
  using GainCache = typename GraphAndGainTypes::GainCache;
  using GainCalculator = typename GraphAndGainTypes::GainComputation;
  using AtomicWeight = parallel::IntegralAtomicWrapper<HypernodeWeight>;

  static constexpr bool debug = false;
  static constexpr bool enable_heavy_assert = false;

  static constexpr Gain MIN_PQ_GAIN_THRESHOLD = 5;

public:

  struct MoveGainComparator {
    bool operator()(const Move& lhs, const Move& rhs) {
      return lhs.gain > rhs.gain || (lhs.gain == rhs.gain && lhs.node < rhs.node);
    }
  };

  using MovePQ = std::priority_queue<Move, vec<Move>, MoveGainComparator>;

  struct IndexedMovePQ {
    explicit IndexedMovePQ(const size_t idx) :
      idx(idx),
      pq() { }

    size_t idx;
    MovePQ pq;
  };

  explicit SimpleRebalancer(const Context& context) :
    _context(context),
    _current_k(context.partition.k),
    _gain(context),
    _part_weights(_context.partition.k) { }

  explicit SimpleRebalancer(HypernodeID , const Context& context, GainCache&) :
    SimpleRebalancer(context) { }

  explicit SimpleRebalancer(HypernodeID num_nodes, const Context& context, gain_cache_t gain_cache) :
    SimpleRebalancer(num_nodes, context, GainCachePtr::cast<GainCache>(gain_cache)) {}

  SimpleRebalancer(const SimpleRebalancer&) = delete;
  SimpleRebalancer(SimpleRebalancer&&) = delete;

  SimpleRebalancer & operator= (const SimpleRebalancer &) = delete;
  SimpleRebalancer & operator= (SimpleRebalancer &&) = delete;

  bool refineImpl(mt_kahypar_partitioned_hypergraph_t& hypergraph,
                  const vec<HypernodeID>&,
                  Metrics& best_metrics,
                  double) final ;

  void initializeImpl(mt_kahypar_partitioned_hypergraph_t&) final { }

  bool refineAndOutputMovesImpl(mt_kahypar_partitioned_hypergraph_t&,
                                const vec<HypernodeID>&,
                                vec<vec<Move>>&,
                                Metrics&,
                                const double) override final {
    ERR("simple rebalancer can not be used for unconstrained refinement");
  }

  bool refineAndOutputMovesLinearImpl(mt_kahypar_partitioned_hypergraph_t&,
                                      const vec<HypernodeID>&,
                                      vec<Move>&,
                                      Metrics&,
                                      const double) override final {
    ERR("simple rebalancer can not be used for unconstrained refinement");
  }

  vec<Move> repairEmptyBlocks(PartitionedHypergraph& phg);

private:

  template<typename F>
  bool moveVertex(PartitionedHypergraph& phg,
                  const HypernodeID hn,
                  const Move& move,
                  const F& objective_delta) {
    ASSERT(phg.partID(hn) == move.from);
    const PartitionID from = move.from;
    const PartitionID to = move.to;
    const HypernodeWeight node_weight = phg.nodeWeight(hn);
    if ( from != to ) {
      // Before moving, we ensure that the block we move the vertex to does
      // not become overloaded
      _part_weights[to] += node_weight;
      if ( _part_weights[to] <= _context.partition.max_part_weights[to] ) {
        if ( phg.changeNodePart(hn, from, to, objective_delta) ) {
          DBG << "Moved vertex" << hn << "from block" << from << "to block" << to
              << "with gain" << move.gain;
          _part_weights[from] -= node_weight;
          return true;
        }
      }
      _part_weights[to] -= node_weight;
    }
    return false;
  }


  void resizeDataStructuresForCurrentK() {
    // If the number of blocks changes, we resize data structures
    // (can happen during deep multilevel partitioning)
    if ( _current_k != _context.partition.k ) {
      _current_k = _context.partition.k;
      _gain.changeNumberOfBlocks(_current_k);
      _part_weights = parallel::scalable_vector<AtomicWeight>(_context.partition.k);
    }
  }

  const Context& _context;
  PartitionID _current_k;
  GainCalculator _gain;
  parallel::scalable_vector<AtomicWeight> _part_weights;
};

}  // namespace kahypar
