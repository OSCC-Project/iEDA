/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2021 Noah Wahl <noah.wahl@kit.edu>
 * Copyright (C) 2021 Tobias Heuer <tobias.heuer@kit.edu>
 * Copyright (C) 2021 Lars Gottesbüren <lars.gottesbueren@kit.edu>
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

#include <sstream>

#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/partition/refinement/i_refiner.h"
#include "mt-kahypar/partition/coarsening/coarsening_commons.h"
#include "mt-kahypar/partition/refinement/flows/scheduler.h"
#include "mt-kahypar/partition/refinement/gains/gain_cache_ptr.h"
#include "mt-kahypar/utils/utilities.h"
#include "mt-kahypar/partition/metrics.h"
#include "mt-kahypar/partition/factories.h"
#include "mt-kahypar/utils/cast.h"

namespace mt_kahypar {

template<typename TypeTraits>
class UncoarsenerBase {

 protected:
  static constexpr bool debug = false;

  using Hypergraph = typename TypeTraits::Hypergraph;
  using PartitionedHypergraph = typename TypeTraits::PartitionedHypergraph;

 public:
  UncoarsenerBase(Hypergraph& hypergraph,
                  const Context& context,
                  UncoarseningData<TypeTraits>& uncoarseningData) :
          _hg(hypergraph),
          _context(context),
          _timer(utils::Utilities::instance().getTimer(context.utility_id)),
          _uncoarseningData(uncoarseningData),
          _gain_cache(gain_cache_t {nullptr, GainPolicy::none}),
          _label_propagation(nullptr),
          _fm(nullptr),
          _flows(nullptr),
          _rebalancer(nullptr) {}

  UncoarsenerBase(const UncoarsenerBase&) = delete;
  UncoarsenerBase(UncoarsenerBase&&) = delete;
  UncoarsenerBase & operator= (const UncoarsenerBase &) = delete;
  UncoarsenerBase & operator= (UncoarsenerBase &&) = delete;

  virtual ~UncoarsenerBase() {
    GainCachePtr::deleteGainCache(_gain_cache);
  };

 protected:
  Hypergraph& _hg;
  const Context& _context;
  utils::Timer& _timer;
  UncoarseningData<TypeTraits>& _uncoarseningData;
  gain_cache_t _gain_cache;
  std::unique_ptr<IRefiner> _label_propagation;
  std::unique_ptr<IRefiner> _fm;
  std::unique_ptr<IRefiner> _flows;
  std::unique_ptr<IRebalancer> _rebalancer;

 protected:

  double refinementTimeLimit(const Context& context, const double time) {
    if ( context.refinement.fm.time_limit_factor != std::numeric_limits<double>::max() ) {
      const double time_limit_factor = std::max(1.0,  context.refinement.fm.time_limit_factor * context.partition.k);
      return std::max(5.0, time_limit_factor * time);
    } else {
      return std::numeric_limits<double>::max();
    }
  }

  Metrics initializeMetrics(PartitionedHypergraph& phg) {
    Metrics m = { metrics::quality(phg, _context),  metrics::imbalance(phg, _context) };

    int64_t num_nodes = phg.initialNumNodes();
    int64_t num_edges = Hypergraph::is_graph ? phg.initialNumEdges() / 2 : phg.initialNumEdges();
    utils::Stats& stats = utils::Utilities::instance().getStats(_context.utility_id);
    stats.add_stat("initial_num_nodes", num_nodes);
    stats.add_stat("initial_num_edges", num_edges);
    std::stringstream ss;
    ss << "initial_" << _context.partition.objective;
    stats.add_stat(ss.str(), metrics::quality(phg, _context));
    if ( _context.partition.objective != Objective::cut ) {
      stats.add_stat("initial_cut", metrics::quality(phg, Objective::cut));
    }
    if ( _context.partition.objective != Objective::km1 ) {
      stats.add_stat("initial_km1", metrics::quality(phg, Objective::km1));
    }
    stats.add_stat("initial_imbalance", m.imbalance);
    return m;
  }

  void initializeRefinementAlgorithms() {
    _gain_cache = GainCachePtr::constructGainCache(_context);
    // refinement algorithms require access to the rebalancer
    _rebalancer = RebalancerFactory::getInstance().createObject(
      _context.refinement.rebalancer, _hg.initialNumNodes(), _context, _gain_cache);
    _label_propagation = LabelPropagationFactory::getInstance().createObject(
      _context.refinement.label_propagation.algorithm,
      _hg.initialNumNodes(), _hg.initialNumEdges(), _context, _gain_cache, *_rebalancer);
    _fm = FMFactory::getInstance().createObject(
      _context.refinement.fm.algorithm,
      _hg.initialNumNodes(), _hg.initialNumEdges(), _context, _gain_cache, *_rebalancer);
    _flows = FlowSchedulerFactory::getInstance().createObject(
      _context.refinement.flows.algorithm,
      _hg.initialNumNodes(), _hg.initialNumEdges(), _context, _gain_cache);
  }
};
}
