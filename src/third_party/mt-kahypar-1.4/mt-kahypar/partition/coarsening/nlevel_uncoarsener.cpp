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

#include "mt-kahypar/partition/coarsening/nlevel_uncoarsener.h"

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/utils/progress_bar.h"
#include "mt-kahypar/io/partitioning_output.h"
#include "mt-kahypar/utils/utilities.h"
#include "mt-kahypar/utils/cast.h"

namespace mt_kahypar {

  template<typename TypeTraits>
  void NLevelUncoarsener<TypeTraits>::initializeImpl() {
    // Initialize n-level batch uncontraction hierarchy
    _timer.start_timer("create_batch_uncontraction_hierarchy", "Create n-Level Hierarchy");
    _hierarchy = _hg.createBatchUncontractionHierarchy(_context.refinement.max_batch_size);
    ASSERT(_uncoarseningData.removed_hyperedges_batches.size() == _hierarchy.size() - 1);
    _timer.stop_timer("create_batch_uncontraction_hierarchy");

    ASSERT(_uncoarseningData.is_finalized);
    _uncoarseningData.compactified_phg->setTargetGraph(_target_graph);
    _current_metrics = Base::initializeMetrics(*_uncoarseningData.compactified_phg);
    _stats.current_number_of_nodes = _uncoarseningData.compactified_hg->initialNumNodes();
    Base::initializeRefinementAlgorithms();

    if (_context.type == ContextType::main) {
      _context.initial_km1 = _current_metrics.quality;
    }

    // For initial partitioning, we compactify the node IDs of smallest hypergraph to
    // a consecutive range. This step project the partition from the compactified hypergraph
    // to the node IDs of the input hypergraph.
    _timer.start_timer("initialize_partition", "Initialize Partition");
    *_uncoarseningData.partitioned_hg = PartitionedHypergraph(_context.partition.k, _hg, parallel_tag_t());
    _uncoarseningData.partitioned_hg->doParallelForAllNodes([&](const HypernodeID hn) {
      ASSERT(static_cast<size_t>(hn) < _uncoarseningData.compactified_hn_mapping.size());
      const HypernodeID compactified_hn = _uncoarseningData.compactified_hn_mapping[hn];
      const PartitionID block_id = _uncoarseningData.compactified_phg->partID(compactified_hn);
      ASSERT(block_id != kInvalidPartition && block_id < _context.partition.k);
      _uncoarseningData.partitioned_hg->setOnlyNodePart(hn, block_id);
    });
    _uncoarseningData.partitioned_hg->initializePartition();
    _uncoarseningData.partitioned_hg->setTargetGraph(_target_graph);

    // Initialize Gain Cache
    if ( _context.refinement.fm.algorithm == FMAlgorithm::kway_fm ) {
      GainCachePtr::initializeGainCache(
        *_uncoarseningData.partitioned_hg, _gain_cache);
    }

    ASSERT(metrics::quality(*_uncoarseningData.compactified_phg, _context) ==
           metrics::quality(*_uncoarseningData.partitioned_hg, _context),
           V(metrics::quality(*_uncoarseningData.compactified_phg, _context)) <<
           V(metrics::quality(*_uncoarseningData.partitioned_hg, _context)));
    ASSERT(metrics::imbalance(*_uncoarseningData.compactified_phg, _context) ==
           metrics::imbalance(*_uncoarseningData.partitioned_hg, _context),
           V(metrics::imbalance(*_uncoarseningData.compactified_phg, _context)) <<
           V(metrics::imbalance(*_uncoarseningData.partitioned_hg, _context)));
    _timer.stop_timer("initialize_partition");

    // Enable progress bar if verbose output is enabled
    if ( _context.partition.verbose_output && _context.partition.enable_progress_bar && !debug ) {
      _progress.enable();
      _progress.setObjective(_current_metrics.quality);
      _progress += _uncoarseningData.compactified_hg->initialNumNodes();
    }

    // Initialize Refiner
    mt_kahypar_partitioned_hypergraph_t phg =
      utils::partitioned_hg_cast(*_uncoarseningData.partitioned_hg);
    if ( _rebalancer ) {
      _rebalancer->initialize(phg);
    }
    if ( _label_propagation ) {
      _label_propagation->initialize(phg);
    }
    if ( _fm ) {
      _fm->initialize(phg);
    }

    ASSERT(_uncoarseningData.round_coarsening_times.size() == _uncoarseningData.removed_hyperedges_batches.size());
    _uncoarseningData.round_coarsening_times.push_back(_uncoarseningData.round_coarsening_times.size() > 0 ?
      _uncoarseningData.round_coarsening_times.back() : std::numeric_limits<double>::max()); // Sentinel

    if ( _timer.isEnabled() ) {
      _timer.disable();
      _is_timer_disabled = true;
    }
  }

  template<typename TypeTraits>
  bool NLevelUncoarsener<TypeTraits>::isTopLevelImpl() const {
    return _hierarchy.empty();
  }

  template<typename TypeTraits>
  void NLevelUncoarsener<TypeTraits>::projectToNextLevelAndRefineImpl() {
    BatchVector& batches = _hierarchy.back();

    // Uncontracts all batches from one coarsening pass. One coarsening pass iterates over all
    // nodes and contracts each node onto another node. Afterwards, we remove all single-pin and
    // identical nets. The following loop reverts all contractions and restores single-pin and
    // identical nets.
    while ( !batches.empty() ) {
      const Batch& batch = batches.back();
      if ( batch.size() > 0 ) {
        HEAVY_REFINEMENT_ASSERT(metrics::quality(*_uncoarseningData.partitioned_hg, _context) == _current_metrics.quality,
          V(_current_metrics.quality) << V(metrics::quality(*_uncoarseningData.partitioned_hg, _context)));

        // Performs batch uncontraction operation
        _timer.start_timer("batch_uncontractions", "Batch Uncontractions", false, _force_measure_timings);
        GainCachePtr::uncontract(*_uncoarseningData.partitioned_hg, batch, _gain_cache);
        _timer.stop_timer("batch_uncontractions", _force_measure_timings);

        HEAVY_REFINEMENT_ASSERT(_hg.verifyIncidenceArrayAndIncidentNets());
        HEAVY_REFINEMENT_ASSERT(GainCachePtr::checkTrackedPartitionInformation(*_uncoarseningData.partitioned_hg, _gain_cache));
        HEAVY_REFINEMENT_ASSERT(metrics::quality(*_uncoarseningData.partitioned_hg, _context) == _current_metrics.quality,
          V(_current_metrics.quality) << V(metrics::quality(*_uncoarseningData.partitioned_hg, _context)));

        // Extracts all border vertices of the current batch
        _timer.start_timer("collect_border_vertices", "Collect Border Vertices", false, _force_measure_timings);
        tbb::parallel_for(UL(0), batch.size(), [&](const size_t i) {
          const Memento& memento = batch[i];
          if ( !_border_vertices_of_batch[memento.u] && _uncoarseningData.partitioned_hg->isBorderNode(memento.u) ) {
            _border_vertices_of_batch.set(memento.u, true);
            _tmp_refinement_nodes.stream(memento.u);
          }
          if ( !_border_vertices_of_batch[memento.v] && _uncoarseningData.partitioned_hg->isBorderNode(memento.v) ) {
            _border_vertices_of_batch.set(memento.v, true);
            _tmp_refinement_nodes.stream(memento.v);
          }
        });
        _timer.stop_timer("collect_border_vertices", _force_measure_timings);

        // We perform localized refinement around the uncontracted nodes if the current number
        // of border nodes is greater than a predefined threshold.
        if ( _tmp_refinement_nodes.size() >= _stats.min_num_border_vertices ) {
          localizedRefine(*_uncoarseningData.partitioned_hg);
        }

        ++_stats.num_batches;
        _stats.total_batch_sizes += batch.size();
        // Update Progress Bar
        _progress.setObjective(_current_metrics.quality);
        _progress += batch.size();
        _stats.current_number_of_nodes += batch.size();
      }
      batches.pop_back();
    }

    // Perform localized refinement on the remaining nodes such that we do
    // not miss any improvement.
    if ( _tmp_refinement_nodes.size() > 0 ) {
      localizedRefine(*_uncoarseningData.partitioned_hg);
    }

    // Restore single-pin and identical nets
    if ( !_uncoarseningData.removed_hyperedges_batches.empty() ) {
      _timer.start_timer("restore_single_pin_and_parallel_nets", "Restore Single Pin and Parallel Nets", false, _force_measure_timings);
      GainCachePtr::restoreSinglePinAndParallelNets(*_uncoarseningData.partitioned_hg,
        _uncoarseningData.removed_hyperedges_batches.back(), _gain_cache);
      _uncoarseningData.removed_hyperedges_batches.pop_back();
      _timer.stop_timer("restore_single_pin_and_parallel_nets", _force_measure_timings);
      HEAVY_REFINEMENT_ASSERT(_hg.verifyIncidenceArrayAndIncidentNets());
      HEAVY_REFINEMENT_ASSERT(GainCachePtr::checkTrackedPartitionInformation(*_uncoarseningData.partitioned_hg, _gain_cache));

      // After restoring all single-pin and identical-nets, we perform an additional
      // refinement step on all border nodes.
      IUncoarsener<TypeTraits>::refine();
      _progress.setObjective(_current_metrics.quality);
      _uncoarseningData.round_coarsening_times.pop_back();
    }

    _hierarchy.pop_back();

    if ( _hierarchy.empty() ) {
      // After we reach the top-level hypergraph, we perform an additional
      // refinement step on all border nodes.
      const HyperedgeWeight objective_before = _current_metrics.quality;
      const double time_limit = Base::refinementTimeLimit(_context, _uncoarseningData.round_coarsening_times.back());
      globalRefine(*_uncoarseningData.partitioned_hg, time_limit);
      _uncoarseningData.round_coarsening_times.pop_back();
      ASSERT(_uncoarseningData.round_coarsening_times.size() == 0);
      const HyperedgeWeight objective_after = _current_metrics.quality;
      if ( _context.partition.verbose_output && objective_after < objective_before ) {
        LOG << GREEN << "Top-Level Refinment improved objective from"
        << objective_before << "to" << objective_after << END;
      }

      if ( _is_timer_disabled ) {
        _timer.enable();
      }
    }
  }

  template<typename TypeTraits>
  void NLevelUncoarsener<TypeTraits>::refineImpl() {
    const double time_limit = Base::refinementTimeLimit(_context, _uncoarseningData.round_coarsening_times.back());
    globalRefine(*_uncoarseningData.partitioned_hg, time_limit);
  }

  template<typename TypeTraits>
  void NLevelUncoarsener<TypeTraits>::rebalancingImpl() {
    // If we reach the top-level hypergraph and the partition is still imbalanced,
    // we use a rebalancing algorithm to restore balance.
    if ( _context.type == ContextType::main && !metrics::isBalanced(*_uncoarseningData.partitioned_hg, _context)) {
      const HyperedgeWeight quality_before = _current_metrics.quality;
      if ( _context.partition.verbose_output ) {
        LOG << RED << "Partition is imbalanced (Current Imbalance:"
        << metrics::imbalance(*_uncoarseningData.partitioned_hg, _context) << ") ->"
        << "Rebalancer is activated" << END;

        LOG << "Part weights: (violations in red)";
        io::printPartWeightsAndSizes(*_uncoarseningData.partitioned_hg, _context);
      }

        // Preform rebalancing
      _timer.start_timer("rebalance", "Rebalance");
       mt_kahypar_partitioned_hypergraph_t phg =
        utils::partitioned_hg_cast(*_uncoarseningData.partitioned_hg);
      _rebalancer->refine(phg, {}, _current_metrics, 0.0);
      _timer.stop_timer("rebalance");

      const HyperedgeWeight quality_after = _current_metrics.quality;
      if ( _context.partition.verbose_output ) {
        const HyperedgeWeight quality_delta = quality_after - quality_before;
        if ( quality_delta > 0 ) {
          LOG << RED << "Rebalancer worsen solution quality by" << quality_delta
          << "(Current Imbalance:" << metrics::imbalance(*_uncoarseningData.partitioned_hg, _context) << ")" << END;
        } else {
          LOG << GREEN << "Rebalancer improves solution quality by" << abs(quality_delta)
          << "(Current Imbalance:" << metrics::imbalance(*_uncoarseningData.partitioned_hg, _context) << ")" << END;
        }
      }
    }

    ASSERT(metrics::quality(*_uncoarseningData.partitioned_hg, _context) == _current_metrics.quality,
      V(_current_metrics.quality) << V(metrics::quality(*_uncoarseningData.partitioned_hg, _context)));
  }

  template<typename TypeTraits>
  HyperedgeWeight NLevelUncoarsener<TypeTraits>::getObjectiveImpl() const {
    return _current_metrics.quality;
  }

  template<typename TypeTraits>
  void NLevelUncoarsener<TypeTraits>::updateMetricsImpl() {
    _current_metrics = Base::initializeMetrics(*_uncoarseningData.partitioned_hg);
    _progress.setObjective(_current_metrics.quality);
  }

  template<typename TypeTraits>
  typename TypeTraits::PartitionedHypergraph& NLevelUncoarsener<TypeTraits>::currentPartitionedHypergraphImpl() {
    return *_uncoarseningData.partitioned_hg;
  }

  template<typename TypeTraits>
  HypernodeID NLevelUncoarsener<TypeTraits>::currentNumberOfNodesImpl() const {
    return _stats.current_number_of_nodes;
  }

  template<typename TypeTraits>
  typename TypeTraits::PartitionedHypergraph&& NLevelUncoarsener<TypeTraits>::movePartitionedHypergraphImpl() {
    ASSERT(isTopLevelImpl());
    return std::move(*_uncoarseningData.partitioned_hg);
  }

  template<typename TypeTraits>
  void NLevelUncoarsener<TypeTraits>::localizedRefine(PartitionedHypergraph& partitioned_hypergraph) {
    // Copy all border nodes into one vector
    vec<HypernodeID> refinement_nodes = _tmp_refinement_nodes.copy_parallel();
    _tmp_refinement_nodes.clear_parallel();
    _border_vertices_of_batch.reset();

    if ( debug && _context.type == ContextType::main ) {
      io::printHypergraphInfo(partitioned_hypergraph.hypergraph(),
        _context, "Refinement Hypergraph", false);
      DBG << "Start Refinement - objective = " << _current_metrics.quality
          << ", imbalance = " << _current_metrics.imbalance;
    }

    bool improvement_found = true;
    mt_kahypar_partitioned_hypergraph_t phg = utils::partitioned_hg_cast(partitioned_hypergraph);
    while( improvement_found ) {
      improvement_found = false;

      if ( _label_propagation && _context.refinement.label_propagation.algorithm != LabelPropagationAlgorithm::do_nothing ) {
        _timer.start_timer("label_propagation", "Label Propagation", false, _force_measure_timings);
        improvement_found |= _label_propagation->refine(phg,
          refinement_nodes, _current_metrics, std::numeric_limits<double>::max());
        _timer.stop_timer("label_propagation", _force_measure_timings);
      }

      if ( _fm && _context.refinement.fm.algorithm != FMAlgorithm::do_nothing ) {
        _timer.start_timer("fm", "FM", false, _force_measure_timings);
        improvement_found |= _fm->refine(phg,
          refinement_nodes, _current_metrics, std::numeric_limits<double>::max());
        _timer.stop_timer("fm", _force_measure_timings);
      }

      if ( _context.type == ContextType::main ) {
        ASSERT(_current_metrics.quality == metrics::quality(partitioned_hypergraph, _context.partition.objective),
            "Actual metric" << V(metrics::quality(partitioned_hypergraph, _context)) <<
            "does not match the metric updated by the refiners" << V(_current_metrics.quality));
      }

      if ( !_context.refinement.refine_until_no_improvement ) {
        break;
      }
    }

    if ( _context.type == ContextType::main) {
      DBG << "--------------------------------------------------\n";
    }
  }

  template<typename TypeTraits>
  void NLevelUncoarsener<TypeTraits>::globalRefine(PartitionedHypergraph& partitioned_hypergraph,
                                       const double time_limit) {

    auto applyGlobalFMParameters = [&](const FMParameters& fm, const NLevelGlobalFMParameters global_fm){
      NLevelGlobalFMParameters tmp_global_fm;
      tmp_global_fm.num_seed_nodes = fm.num_seed_nodes;
      tmp_global_fm.obey_minimal_parallelism = fm.obey_minimal_parallelism;
      fm.num_seed_nodes = global_fm.num_seed_nodes;
      fm.obey_minimal_parallelism = global_fm.obey_minimal_parallelism;
      return tmp_global_fm;
    };

    if ( _context.refinement.global_fm.use_global_fm ) {
      if ( debug && _context.type == ContextType::main ) {
        io::printHypergraphInfo(partitioned_hypergraph.hypergraph(),
          _context, "Refinement Hypergraph", false);
        DBG << "Start Refinement - objective = " << _current_metrics.quality
            << ", imbalance = " << _current_metrics.imbalance;
      }

      // Enable Timings
      bool was_enabled = false;
      if ( !_timer.isEnabled() &&
           _context.type == ContextType::main ) {
        _timer.enable();
        was_enabled = true;
      }

      // Apply global FM parameters to FM context and temporary store old fm context
      _timer.start_timer("global_refinement", "Global Refinement");
      NLevelGlobalFMParameters tmp_global_fm = applyGlobalFMParameters(
        _context.refinement.fm, _context.refinement.global_fm);
      bool improvement_found = true;
      mt_kahypar_partitioned_hypergraph_t phg = utils::partitioned_hg_cast(partitioned_hypergraph);
      while( improvement_found ) {
        improvement_found = false;
        const HyperedgeWeight metric_before = _current_metrics.quality;

        if ( _fm && _context.refinement.fm.algorithm != FMAlgorithm::do_nothing ) {
          _timer.start_timer("fm", "FM");
          improvement_found |= _fm->refine(phg, {}, _current_metrics, time_limit);
          _timer.stop_timer("fm");
        }

        if ( _flows && _context.refinement.flows.algorithm != FlowAlgorithm::do_nothing ) {
          _timer.start_timer("initialize_flow_scheduler", "Initialize Flow Scheduler");
          _flows->initialize(phg);
          _timer.stop_timer("initialize_flow_scheduler");

          _timer.start_timer("flow_refinement_scheduler", "Flow Refinement Scheduler");
          improvement_found |= _flows->refine(phg, {}, _current_metrics, time_limit);
          _timer.stop_timer("flow_refinement_scheduler");
        }

        if ( _context.type == ContextType::main ) {
          ASSERT(_current_metrics.quality == metrics::quality(partitioned_hypergraph, _context.partition.objective),
              "Actual metric" << V(metrics::quality(partitioned_hypergraph, _context)) <<
              "does not match the metric updated by the refiners" << V(_current_metrics.quality));
        }

        const HyperedgeWeight metric_after = _current_metrics.quality;
        const double relative_improvement = 1.0 -
          static_cast<double>(metric_after) / metric_before;
        if ( !_context.refinement.global_fm.refine_until_no_improvement ||
            relative_improvement <= _context.refinement.relative_improvement_threshold ) {
          break;
        }
      }
      // Reset FM context
      applyGlobalFMParameters(_context.refinement.fm, tmp_global_fm);
      _timer.stop_timer("global_refinement");

      if ( was_enabled ) {
        _timer.disable();
      }

      if ( _context.type == ContextType::main) {
        DBG << "--------------------------------------------------\n";
      }
    }
  }

  INSTANTIATE_CLASS_WITH_TYPE_TRAITS(NLevelUncoarsener)

}
