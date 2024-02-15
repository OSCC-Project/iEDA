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

#include "mt-kahypar/partition/refinement/flows/parallel_construction.h"

#include "kahypar-resources/utils/math.h"

#include "tbb/concurrent_queue.h"

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/parallel/stl/scalable_queue.h"
#include "mt-kahypar/partition/refinement/gains/gain_definitions.h"

namespace mt_kahypar {

template<typename GraphAndGainTypes>
typename ParallelConstruction<GraphAndGainTypes>::TmpHyperedge
ParallelConstruction<GraphAndGainTypes>::DynamicIdenticalNetDetection::get(const size_t he_hash,
                                                                        const vec<whfc::Node>& pins) {
  const size_t bucket_idx = he_hash % _hash_buckets.size();
  if ( __atomic_load_n(&_hash_buckets[bucket_idx].threshold, __ATOMIC_RELAXED) == _threshold ) {
    // There exists already some hyperedges with the same hash
    for ( const ThresholdHyperedge& tmp : _hash_buckets[bucket_idx].identical_nets ) {
      // Check if there is some hyperedge equal to he
      const TmpHyperedge& tmp_e = tmp.e;
      if ( tmp.threshold == _threshold && tmp_e.hash == he_hash &&
           _flow_hg.tmpPinCount(tmp_e.bucket, tmp_e.e) == pins.size() ) {
        bool is_identical = true;
        size_t idx = 0;
        for ( const whfc::FlowHypergraph::Pin& u : _flow_hg.tmpPinsOf(tmp_e.bucket, tmp_e.e) ) {
          if ( u.pin != pins[idx++] ) {
            is_identical = false;
            break;
          }
        }
        if ( is_identical ) {
          return tmp_e;
        }
      }
    }
  }
  return TmpHyperedge { 0, std::numeric_limits<size_t>::max(), whfc::invalidHyperedge };
}

template<typename GraphAndGainTypes>
void ParallelConstruction<GraphAndGainTypes>::DynamicIdenticalNetDetection::add(const TmpHyperedge& tmp_he) {
  const size_t bucket_idx = tmp_he.hash % _hash_buckets.size();
  uint32_t expected = __atomic_load_n(&_hash_buckets[bucket_idx].threshold, __ATOMIC_RELAXED);
  uint32_t desired = _threshold - 1;
  while ( __atomic_load_n(&_hash_buckets[bucket_idx].threshold, __ATOMIC_RELAXED) < _threshold ) {
    if ( expected < desired &&
        __atomic_compare_exchange(&_hash_buckets[bucket_idx].threshold,
          &expected, &desired, false, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED) ) {
      _hash_buckets[bucket_idx].identical_nets.clear();
      __atomic_store_n(&_hash_buckets[bucket_idx].threshold, _threshold, __ATOMIC_RELAXED);
    }
  }
  _hash_buckets[bucket_idx].identical_nets.push_back(ThresholdHyperedge { tmp_he, _threshold });
}

template<typename GraphAndGainTypes>
FlowProblem ParallelConstruction<GraphAndGainTypes>::constructFlowHypergraph(const PartitionedHypergraph& phg,
                                                                          const Subhypergraph& sub_hg,
                                                                          const PartitionID block_0,
                                                                          const PartitionID block_1,
                                                                          vec<HypernodeID>& whfc_to_node) {
  FlowProblem flow_problem;
  const double density = static_cast<double>(phg.initialNumEdges()) / phg.initialNumNodes();
  const double avg_he_size = static_cast<double>(phg.initialNumPins()) / phg.initialNumEdges();
  if ( density >= 0.5 && avg_he_size <= 100 ) {
    // This algorithm iterates over all hyperedges and checks for all pins if
    // they are contained in the flow problem. Algorithm could have overheads, if
    // only a small portion of each hyperedge is contained in the flow hypergraph.
    flow_problem = constructDefault(phg, sub_hg, block_0, block_1, whfc_to_node);
  } else {
    // This is a construction algorithm optimized for hypergraphs with large hyperedges.
    // Algorithm constructs a temporary pin list, therefore it could have overheads
    // for hypergraphs with small hyperedges.
    flow_problem = constructOptimizedForLargeHEs(phg, sub_hg, block_0, block_1, whfc_to_node);
  }

  if ( _flow_hg.nodeWeight(flow_problem.source) == 0 ||
       _flow_hg.nodeWeight(flow_problem.sink) == 0 ) {
    // Source or sink not connected to vertices in the flow problem
    flow_problem.non_removable_cut = 0;
    flow_problem.total_cut = 0;
  } else {
    _flow_hg.finalizeParallel();

    if ( _context.refinement.flows.determine_distance_from_cut ) {
      // Determine the distance of each node contained in the flow network from the cut.
      // This technique improves piercing decision within the WHFC framework.
      determineDistanceFromCut(phg, flow_problem.source,
        flow_problem.sink, block_0, block_1, whfc_to_node);
    }
  }

  DBG << "Flow Hypergraph [ Nodes =" << _flow_hg.numNodes()
      << ", Edges =" << _flow_hg.numHyperedges()
      << ", Pins =" << _flow_hg.numPins()
      << ", Blocks = (" << block_0 << "," << block_1 << ") ]";

  return flow_problem;
}

template<typename GraphAndGainTypes>
FlowProblem ParallelConstruction<GraphAndGainTypes>::constructFlowHypergraph(const PartitionedHypergraph& phg,
                                                                          const Subhypergraph& sub_hg,
                                                                          const PartitionID block_0,
                                                                          const PartitionID block_1,
                                                                          vec<HypernodeID>& whfc_to_node,
                                                                          const bool default_construction) {
  FlowProblem flow_problem;
  if ( default_construction ) {
    // This algorithm iterates over all hyperedges and checks for all pins if
    // they are contained in the flow problem. Algorithm could have overheads, if
    // only a small portion of each hyperedge is contained in the flow hypergraph.
    flow_problem = constructDefault(phg, sub_hg, block_0, block_1, whfc_to_node);
  } else {
    // This is a construction algorithm optimized for hypergraphs with large hyperedges.
    // Algorithm constructs a temporary pin list, therefore it could have overheads
    // for hypergraphs with small hyperedges.
    flow_problem = constructOptimizedForLargeHEs(phg, sub_hg, block_0, block_1, whfc_to_node);
  }

  if ( _flow_hg.nodeWeight(flow_problem.source) == 0 ||
       _flow_hg.nodeWeight(flow_problem.sink) == 0 ) {
    // Source or sink not connected to vertices in the flow problem
    flow_problem.non_removable_cut = 0;
    flow_problem.total_cut = 0;
  } else {
    _flow_hg.finalizeParallel();

    if ( _context.refinement.flows.determine_distance_from_cut ) {
      // Determine the distance of each node contained in the flow network from the cut.
      // This technique improves piercing decision within the WHFC framework.
      determineDistanceFromCut(phg, flow_problem.source,
        flow_problem.sink, block_0, block_1, whfc_to_node);
    }
  }

  DBG << "Flow Hypergraph [ Nodes =" << _flow_hg.numNodes()
      << ", Edges =" << _flow_hg.numHyperedges()
      << ", Pins =" << _flow_hg.numPins()
      << ", Blocks = (" << block_0 << "," << block_1 << ") ]";

  return flow_problem;
}

template<typename GraphAndGainTypes>
FlowProblem ParallelConstruction<GraphAndGainTypes>::constructDefault(const PartitionedHypergraph& phg,
                                                                   const Subhypergraph& sub_hg,
                                                                   const PartitionID block_0,
                                                                   const PartitionID block_1,
                                                                   vec<HypernodeID>& whfc_to_node) {
  ASSERT(block_0 != kInvalidPartition && block_1 != kInvalidPartition);
  FlowProblem flow_problem;
  flow_problem.total_cut = 0;
  flow_problem.non_removable_cut = 0;
  _node_to_whfc.clear();

  tbb::parallel_invoke([&]() {
    _node_to_whfc.clear();
    _node_to_whfc.setMaxSize(sub_hg.numNodes());
  }, [&] {
    whfc_to_node.resize(sub_hg.numNodes() + 2);
  }, [&] {
    _flow_hg.allocateNodes(sub_hg.numNodes() + 2);
  }, [&] {
    _identical_nets.reset();
  });

  if ( _context.refinement.flows.determine_distance_from_cut ) {
    _cut_hes.clear();
  }

  // Add refinement nodes to flow network
  tbb::parallel_invoke([&] {
    // Add source nodes
    flow_problem.source = whfc::Node(0);
    whfc_to_node[flow_problem.source] = kInvalidHypernode;
    _flow_hg.nodeWeight(flow_problem.source) = whfc::NodeWeight(
      std::max(0, phg.partWeight(block_0) - sub_hg.weight_of_block_0));
    tbb::parallel_for(UL(0), sub_hg.nodes_of_block_0.size(), [&](const size_t i) {
      const HypernodeID hn = sub_hg.nodes_of_block_0[i];
      const whfc::Node u(1 + i);
      whfc_to_node[u] = hn;
      _node_to_whfc[hn] = u;
      _flow_hg.nodeWeight(u) = whfc::NodeWeight(phg.nodeWeight(hn));
    });
  }, [&] {
    // Add sink nodes
    flow_problem.sink = whfc::Node(sub_hg.nodes_of_block_0.size() + 1);
    whfc_to_node[flow_problem.sink] = kInvalidHypernode;
    _flow_hg.nodeWeight(flow_problem.sink) = whfc::NodeWeight(
      std::max(0, phg.partWeight(block_1) - sub_hg.weight_of_block_1));
    tbb::parallel_for(UL(0), sub_hg.nodes_of_block_1.size(), [&](const size_t i) {
      const HypernodeID hn = sub_hg.nodes_of_block_1[i];
      const whfc::Node u(flow_problem.sink + 1 + i);
      whfc_to_node[u] = hn;
      _node_to_whfc[hn] = u;
      _flow_hg.nodeWeight(u) = whfc::NodeWeight(phg.nodeWeight(hn));
    });
  });
  flow_problem.weight_of_block_0 = _flow_hg.nodeWeight(flow_problem.source) + sub_hg.weight_of_block_0;
  flow_problem.weight_of_block_1 = _flow_hg.nodeWeight(flow_problem.sink) + sub_hg.weight_of_block_1;

  const HyperedgeID max_hyperedges = sub_hg.hes.size();
  const HypernodeID max_pins = sub_hg.num_pins + max_hyperedges;
  _flow_hg.allocateHyperedgesAndPins(max_hyperedges, max_pins);

  // Add hyperedge to flow network and configure source and sink
  auto push_into_tmp_pins = [&](vec<whfc::Node>& tmp_pins, const whfc::Node pin,
                                size_t& current_hash, const bool is_source_or_sink) {
    tmp_pins.push_back(pin);
    current_hash += kahypar::math::hash(pin);
    if ( is_source_or_sink ) {
      // According to Lars: Adding to source or sink to the start of
      // each pin list improves running time
      std::swap(tmp_pins[0], tmp_pins.back());
    }
  };

  _flow_hg.setNumCSRBuckets(NUM_CSR_BUCKETS);
  const size_t step = max_hyperedges / NUM_CSR_BUCKETS + (max_hyperedges % NUM_CSR_BUCKETS != 0);
  tbb::parallel_for(UL(0), NUM_CSR_BUCKETS, [&](const size_t idx) {
    const size_t start = std::min(step * idx, static_cast<size_t>(max_hyperedges));
    const size_t end = std::min(step * (idx + 1), static_cast<size_t>(max_hyperedges));
    const size_t num_hes = end - start;
    size_t num_pins = 0;
    for ( size_t i = start; i < end; ++i ) {
      const HyperedgeID he = sub_hg.hes[i];
      num_pins += phg.edgeSize(he) + 1;
    }
    _flow_hg.initializeCSRBucket(idx, num_hes, num_pins);

    whfc::Hyperedge e(0);
    size_t pin_idx = 0;
    vec<whfc::Node>& tmp_pins = _tmp_pins.local();
    for ( size_t i = start; i < end; ++i ) {
      const HyperedgeID he = sub_hg.hes[i];
      if ( !FlowNetworkConstruction::dropHyperedge(phg, he, block_0, block_1) ) {
        tmp_pins.clear();
        size_t he_hash = 0;
        bool connectToSource = FlowNetworkConstruction::connectToSource(phg, he, block_0, block_1);
        bool connectToSink = FlowNetworkConstruction::connectToSink(phg, he, block_0, block_1);
        const HyperedgeWeight he_weight = FlowNetworkConstruction::capacity(phg, _context, he, block_0, block_1);
        if ( ( phg.pinCountInPart(he, block_0) > 0 && phg.pinCountInPart(he, block_1) > 0 ) ||
               FlowNetworkConstruction::isCut(phg, he, block_0, block_1) ) {
          __atomic_fetch_add(&flow_problem.total_cut, he_weight, __ATOMIC_RELAXED);
        }
        for ( const HypernodeID& pin : phg.pins(he) ) {
          whfc::Node* whfc_pin = _node_to_whfc.get_if_contained(pin);
          if ( whfc_pin ) {
            push_into_tmp_pins(tmp_pins, *whfc_pin, he_hash, false);
          } else {
            const PartitionID pin_block = phg.partID(pin);
            connectToSource |= pin_block == block_0;
            connectToSink |= pin_block == block_1;
          }
        }

        const bool empty_hyperedge = tmp_pins.size() == 0;
        const bool connected_to_source_and_sink = connectToSource && connectToSink;
        if ( connected_to_source_and_sink ) {
          // Hyperedge is connected to source and sink which means we can not remove it
          // from the cut with the current flow problem => remove he from flow problem
          __atomic_fetch_add(&flow_problem.non_removable_cut, he_weight, __ATOMIC_RELAXED);
        } else if ( !empty_hyperedge ) {
          if ( connectToSource ) {
            push_into_tmp_pins(tmp_pins, flow_problem.source, he_hash, true);
          } else if ( connectToSink ) {
            push_into_tmp_pins(tmp_pins, flow_problem.sink, he_hash, true);
          }

          // Sort pins for identical net detection
          std::sort( tmp_pins.begin() +
                   ( tmp_pins[0] == flow_problem.source ||
                     tmp_pins[0] == flow_problem.sink), tmp_pins.end());

          if ( tmp_pins.size() > 1 ) {
            const TmpHyperedge identical_net = _identical_nets.get(he_hash, tmp_pins);
            if ( identical_net.e == whfc::invalidHyperedge ) {
              const size_t pin_start = pin_idx;
              const size_t pin_end = pin_start + tmp_pins.size();
              for ( size_t i = 0; i < tmp_pins.size(); ++i ) {
                _flow_hg.addPin(tmp_pins[i], idx, pin_idx++);
              }
              TmpHyperedge tmp_e { he_hash, idx, e++ };
              if ( _context.refinement.flows.determine_distance_from_cut &&
                  phg.pinCountInPart(he, block_0) > 0 && phg.pinCountInPart(he, block_1) > 0 ) {
                _cut_hes.push_back(tmp_e);
              }
              _flow_hg.finishHyperedge(tmp_e.e, he_weight, idx, pin_start, pin_end);
              _identical_nets.add(tmp_e);
            } else {
              // Current hyperedge is identical to an already added
              __atomic_fetch_add(&_flow_hg.capacity(identical_net.bucket, identical_net.e), he_weight, __ATOMIC_RELAXED);
            }
          }
        }
      }
    }
  });

  tbb::parallel_for(UL(0), NUM_CSR_BUCKETS, [&](const size_t idx) {
    _flow_hg.finalizeCSRBucket(idx);
  });
  _flow_hg.finalizeHyperedges();

  return flow_problem;
}

template<typename GraphAndGainTypes>
FlowProblem ParallelConstruction<GraphAndGainTypes>::constructOptimizedForLargeHEs(const PartitionedHypergraph& phg,
                                                                                const Subhypergraph& sub_hg,
                                                                                const PartitionID block_0,
                                                                                const PartitionID block_1,
                                                                                vec<HypernodeID>& whfc_to_node) {
  ASSERT(block_0 != kInvalidPartition && block_1 != kInvalidPartition);
  FlowProblem flow_problem;
  flow_problem.total_cut = 0;
  flow_problem.non_removable_cut = 0;
  _node_to_whfc.clear();

  tbb::parallel_invoke([&]() {
    _he_to_whfc.clear();
    _he_to_whfc.setMaxSize(sub_hg.hes.size());
    tbb::parallel_for(UL(0), sub_hg.hes.size(), [&](const size_t i) {
      const HyperedgeID he = sub_hg.hes[i];
      _he_to_whfc[he] = i;
    });
  }, [&] {
    whfc_to_node.resize(sub_hg.numNodes() + 2);
  }, [&] {
    _flow_hg.allocateNodes(sub_hg.numNodes() + 2);
  }, [&] {
    _identical_nets.reset();
  });

  if ( _context.refinement.flows.determine_distance_from_cut ) {
    _cut_hes.clear();
  }

  // Add refinement nodes to flow network
  const size_t num_buckets = _pins.numBuckets();
  const HyperedgeID max_hyperedges = sub_hg.hes.size();
  const size_t hes_per_bucket = max_hyperedges / num_buckets + (max_hyperedges % num_buckets != 0);
  tbb::parallel_invoke([&] {
    // Add source nodes
    flow_problem.source = whfc::Node(0);
    whfc_to_node[flow_problem.source] = kInvalidHypernode;
    _flow_hg.nodeWeight(flow_problem.source) = whfc::NodeWeight(
      std::max(0, phg.partWeight(block_0) - sub_hg.weight_of_block_0));
    tbb::parallel_for(UL(0), sub_hg.nodes_of_block_0.size(), [&](const size_t i) {
      const HypernodeID hn = sub_hg.nodes_of_block_0[i];
      const whfc::Node u(1 + i);
      whfc_to_node[u] = hn;
      _flow_hg.nodeWeight(u) = whfc::NodeWeight(phg.nodeWeight(hn));
      for ( const HyperedgeID& he : phg.incidentEdges(hn) ) {
        ASSERT(_he_to_whfc.get_if_contained(he) != nullptr);
        const HyperedgeID e = _he_to_whfc[he];
        _pins.insert(e / hes_per_bucket, TmpPin { e, u, block_0 });
      }
    });
  }, [&] {
    // Add sink nodes
    flow_problem.sink = whfc::Node(sub_hg.nodes_of_block_0.size() + 1);
    whfc_to_node[flow_problem.sink] = kInvalidHypernode;
    _flow_hg.nodeWeight(flow_problem.sink) = whfc::NodeWeight(
      std::max(0, phg.partWeight(block_1) - sub_hg.weight_of_block_1));
    tbb::parallel_for(UL(0), sub_hg.nodes_of_block_1.size(), [&](const size_t i) {
      const HypernodeID hn = sub_hg.nodes_of_block_1[i];
      const whfc::Node u(flow_problem.sink + 1 + i);
      whfc_to_node[u] = hn;
      _flow_hg.nodeWeight(u) = whfc::NodeWeight(phg.nodeWeight(hn));
      for ( const HyperedgeID& he : phg.incidentEdges(hn) ) {
        ASSERT(_he_to_whfc.get_if_contained(he) != nullptr);
        const HyperedgeID e = _he_to_whfc[he];
        _pins.insert(e / hes_per_bucket, TmpPin { e, u, block_1 });
      }
    });
  });
  flow_problem.weight_of_block_0 = _flow_hg.nodeWeight(flow_problem.source) + sub_hg.weight_of_block_0;
  flow_problem.weight_of_block_1 = _flow_hg.nodeWeight(flow_problem.sink) + sub_hg.weight_of_block_1;

  const HypernodeID max_pins = sub_hg.num_pins + max_hyperedges;
  _flow_hg.allocateHyperedgesAndPins(max_hyperedges, max_pins);
  _flow_hg.setNumCSRBuckets(num_buckets);

  _pins.doParallelForAllBuckets([&](const size_t idx) {
    vec<TmpPin>& pins_of_bucket = _pins.getBucket(idx);
    if ( pins_of_bucket.size() > 0 ) {
      std::sort(pins_of_bucket.begin(), pins_of_bucket.end(),
        [&](const TmpPin& lhs, const TmpPin& rhs ) {
          return lhs.e < rhs.e || (lhs.e == rhs.e && lhs.pin < rhs.pin);
        });

      HyperedgeID last_he = kInvalidHyperedge;
      size_t num_hes = 1;
      size_t num_pins = 0;
      for ( const TmpPin& pin : pins_of_bucket ) {
        if ( pin.e != last_he ) {
          ++num_hes;
          last_he = pin.e;
        }
        ++num_pins;
      }
      num_pins += num_hes;
      _flow_hg.initializeCSRBucket(idx, num_hes, num_pins);

      whfc::Hyperedge current_he(0);
      size_t pin_idx = 0;
      vec<whfc::Node>& tmp_pins = _tmp_pins.local();
      size_t start_idx = 0;
      last_he = pins_of_bucket[start_idx].e;
      HypernodeID pin_count_in_block_0 = 0;
      HypernodeID pin_count_in_block_1 = 0;
      auto add_hyperedge = [&](const size_t end_idx) {
        ASSERT(start_idx < end_idx);
        tmp_pins.clear();
        const HyperedgeID he = sub_hg.hes[last_he];
        if ( !FlowNetworkConstruction::dropHyperedge(phg, he, block_0, block_1) ) {
          const HyperedgeWeight he_weight = FlowNetworkConstruction::capacity(phg, _context, he, block_0, block_1);
          const HypernodeID actual_pin_count_block_0 = phg.pinCountInPart(he, block_0);
          const HypernodeID actual_pin_count_block_1 = phg.pinCountInPart(he, block_1);
          bool connect_to_source = FlowNetworkConstruction::connectToSource(phg, he, block_0, block_1);
          bool connect_to_sink = FlowNetworkConstruction::connectToSink(phg, he, block_0, block_1);
          connect_to_source |= pin_count_in_block_0 < actual_pin_count_block_0;
          connect_to_sink |= pin_count_in_block_1 < actual_pin_count_block_1;
          if ( ( actual_pin_count_block_0 > 0 && actual_pin_count_block_1 > 0 ) ||
                 FlowNetworkConstruction::isCut(phg, he, block_0, block_1) ) {
            __atomic_fetch_add(&flow_problem.total_cut, he_weight, __ATOMIC_RELAXED);
          }

          if ( connect_to_source && connect_to_sink ) {
            // Hyperedge is connected to source and sink which means we can not remove it
            // from the cut with the current flow problem => remove he from flow problem
            __atomic_fetch_add(&flow_problem.non_removable_cut, he_weight, __ATOMIC_RELAXED);
          } else {
            // Add hyperedge to flow network and configure source and sink
            size_t hash = 0;
            if ( connect_to_source ) {
              tmp_pins.push_back(flow_problem.source);
              hash += kahypar::math::hash(flow_problem.source);
            } else if ( connect_to_sink ) {
              tmp_pins.push_back(flow_problem.sink);
              hash += kahypar::math::hash(flow_problem.sink);
            }
            for ( size_t i = start_idx; i < end_idx; ++i ) {
              tmp_pins.push_back(pins_of_bucket[i].pin);
              hash += kahypar::math::hash(pins_of_bucket[i].pin);
            }

            if ( tmp_pins.size() > 1 ) {
              const TmpHyperedge identical_net = _identical_nets.get(hash, tmp_pins);
              if ( identical_net.e == whfc::invalidHyperedge ) {
                const size_t pin_start = pin_idx;
                const size_t pin_end = pin_start + tmp_pins.size();
                for ( const whfc::Node& pin : tmp_pins ) {
                  _flow_hg.addPin(pin, idx, pin_idx++);
                }
                TmpHyperedge tmp_e { hash, idx, current_he++ };
                if ( _context.refinement.flows.determine_distance_from_cut &&
                    actual_pin_count_block_0 > 0 && actual_pin_count_block_1 > 0 ) {
                  _cut_hes.push_back(tmp_e);
                }
                _flow_hg.finishHyperedge(tmp_e.e, he_weight, idx, pin_start, pin_end);
                _identical_nets.add(tmp_e);
              } else {
                // Current hyperedge is identical to an already added
                __atomic_fetch_add(&_flow_hg.capacity(identical_net.bucket, identical_net.e), he_weight, __ATOMIC_RELAXED);
              }
            }
          }
        }
      };
      for ( size_t i = 0; i < pins_of_bucket.size(); ++i ) {
        if ( last_he != pins_of_bucket[i].e ) {
          add_hyperedge(i);
          start_idx = i;
          last_he = pins_of_bucket[i].e;
          pin_count_in_block_0 = 0;
          pin_count_in_block_1 = 0;
        }
        pin_count_in_block_0 += pins_of_bucket[i].block == block_0;
        pin_count_in_block_1 += pins_of_bucket[i].block == block_1;
      }
      add_hyperedge(pins_of_bucket.size());
    } else {
      _flow_hg.initializeCSRBucket(idx, 0, 0);
    }
    _pins.clear(idx);
  });

  tbb::parallel_for(UL(0), num_buckets, [&](const size_t idx) {
    _flow_hg.finalizeCSRBucket(idx);
  });
  _flow_hg.finalizeHyperedges();

  return flow_problem;
}

namespace {
template<typename T>
class BFSQueue {

 public:
  explicit BFSQueue(const size_t num_threads) :
    _q(num_threads) { }

  bool empty() {
    bool is_empty = true;
    for ( size_t i = 0; i < _q.size(); ++i ) {
      is_empty &= _q[i].empty();
    }
    return is_empty;
  }

  bool empty(const size_t i) {
    ASSERT(i < _q.size());
    return _q[i].empty();
  }

  void push(const T elem, const size_t i) {
    ASSERT(i < _q.size());
    return _q[i].push(elem);
  }

  T front(const size_t i) {
    ASSERT(i < _q.size());
    return _q[i].front();
  }

  void pop(const size_t i) {
    ASSERT(i < _q.size());
    return _q[i].pop();
  }

 private:
  vec<parallel::scalable_queue<T>> _q;
};
}

template<typename GraphAndGainTypes>
void ParallelConstruction<GraphAndGainTypes>::determineDistanceFromCut(const PartitionedHypergraph& phg,
                                                                    const whfc::Node source,
                                                                    const whfc::Node sink,
                                                                    const PartitionID block_0,
                                                                    const PartitionID block_1,
                                                                    const vec<HypernodeID>& whfc_to_node) {
  auto& distances = _hfc.cs.border_nodes.distance;
  distances.assign(_flow_hg.numNodes(), whfc::HopDistance(0));
  _visited_hns.resize(_flow_hg.numNodes() + _flow_hg.numHyperedges());
  _visited_hns.reset();
  _visited_hns.set(source, true);
  _visited_hns.set(sink, true);

  // Initialize bfs queue with vertices contained in cut hyperedges
  size_t q_idx = 0;

  const size_t num_threads = std::thread::hardware_concurrency();
  vec<BFSQueue<whfc::Node>> q(2, BFSQueue<whfc::Node>(num_threads));
  tbb::parallel_for(UL(0), _cut_hes.size(), [&](const size_t i) {
    const int thread_idx = tbb::this_task_arena::current_thread_index();
    const whfc::Hyperedge he = _flow_hg.originalHyperedgeID(_cut_hes[i].bucket, _cut_hes[i].e);
    for ( const whfc::FlowHypergraph::Pin& pin : _flow_hg.pinsOf(he) ) {
      if ( _visited_hns.compare_and_set_to_true(pin.pin) ) {
        q[q_idx].push(pin.pin, thread_idx);
      }
    }
    _visited_hns.set(_flow_hg.numNodes() + he, true);
  });

  // Perform BFS to determine distance of each vertex from cut
  whfc::HopDistance dist(1);
  whfc::HopDistance max_dist_source(0);
  whfc::HopDistance max_dist_sink(0);
  while ( !q[q_idx].empty() ) {
    bool reached_source_side = false;
    bool reached_sink_side = false;
    tbb::parallel_for(UL(0), num_threads, [&](const size_t idx) {
      while ( !q[q_idx].empty(idx) ) {
        whfc::Node u = q[q_idx].front(idx);
        q[q_idx].pop(idx);
        const PartitionID block_of_u = phg.partID(whfc_to_node[u]);
        if ( block_of_u == block_0 ) {
          distances[u] = -dist;
          reached_source_side = true;
        } else if ( block_of_u == block_1 ) {
          distances[u] = dist;
          reached_sink_side = true;
        }

        for ( const whfc::FlowHypergraph::InHe& in_he : _flow_hg.hyperedgesOf(u) ) {
          const whfc::Hyperedge he = in_he.e;
          if ( _visited_hns.compare_and_set_to_true(_flow_hg.numNodes() + he) ) {
            for ( const whfc::FlowHypergraph::Pin& pin : _flow_hg.pinsOf(he) ) {
              if ( _visited_hns.compare_and_set_to_true(pin.pin) ) {
                q[1 - q_idx].push(pin.pin, idx);
              }
            }
          }
        }
      }
    });

    if ( reached_source_side ) max_dist_source = dist;
    if ( reached_sink_side ) max_dist_sink = dist;

    ASSERT(q[q_idx].empty());
    q_idx = 1 - q_idx;
    ++dist;
  }
  distances[source] = -(max_dist_source + 1);
  distances[sink] = max_dist_sink + 1;
}

namespace {
#define PARALLEL_CONSTRUCTION(X) ParallelConstruction<X>
}

INSTANTIATE_CLASS_WITH_VALID_TRAITS(PARALLEL_CONSTRUCTION)

} // namespace mt_kahypar
