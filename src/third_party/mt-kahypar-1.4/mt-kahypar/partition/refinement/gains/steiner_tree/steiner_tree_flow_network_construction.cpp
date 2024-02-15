/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2023 Tobias Heuer <tobias.heuer@kit.edu>
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

#include "mt-kahypar/partition/refinement/gains/steiner_tree/steiner_tree_flow_network_construction.h"

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/partition/mapping/target_graph.h"
#include "mt-kahypar/utils/exception.h"

namespace mt_kahypar {

namespace {
HyperedgeWeight capacity_for_cut_edge(const SteinerTreeFlowValuePolicy policy,
                                      const HyperedgeWeight gain_0,
                                      const HyperedgeWeight gain_1) {
  switch ( policy ) {
    case SteinerTreeFlowValuePolicy::lower_bound: return std::min(gain_0, gain_1);
    case SteinerTreeFlowValuePolicy::upper_bound: return std::max(gain_0, gain_1);
    case SteinerTreeFlowValuePolicy::UNDEFINED:
      throw InvalidParameterException(
        "Steiner tree flow value policy is undefined");
      return 0;
  }
  return 0;
}
} // namespace

template<typename PartitionedHypergraph>
HyperedgeWeight SteinerTreeFlowNetworkConstruction::capacity(const PartitionedHypergraph& phg,
                                                             const Context& context,
                                                             const HyperedgeID he,
                                                             const PartitionID block_0,
                                                             const PartitionID block_1)  {
  ASSERT(phg.hasTargetGraph());
  const TargetGraph& target_graph = *phg.targetGraph();
  const HyperedgeWeight edge_weight = phg.edgeWeight(he);
  const HypernodeID pin_count_block_0 = phg.pinCountInPart(he, block_0);
  const HypernodeID pin_count_block_1 = phg.pinCountInPart(he, block_1);
  ds::Bitset& connectivity_set = phg.deepCopyOfConnectivitySet(he);
  const HyperedgeWeight current_distance = target_graph.distance(connectivity_set);
  if ( pin_count_block_0 > 0 && pin_count_block_1 == 0 ) {
    // Hyperedge is non-cut
    // => we use gain for making the hyperedge cut as capacity to get a lower bound for the
    // actual improvement
    HyperedgeWeight distance_with_block_1 = 0;
    if ( pin_count_block_0 == 1 ) {
      distance_with_block_1 = target_graph.distanceAfterExchangingBlocks(connectivity_set, block_0, block_1);
    } else {
      distance_with_block_1 = target_graph.distanceWithBlock(connectivity_set, block_1);
    }
    return std::abs(current_distance - distance_with_block_1) * edge_weight;
  } else if ( pin_count_block_0 == 0 && pin_count_block_1 > 0 ) {
    // Hyperedge is non-cut
    // => we use gain for making the hyperedge cut as capacity to get a lower bound for the
    // actual improvement
    HyperedgeWeight distance_with_block_0 = 0;
    if ( pin_count_block_1 == 1 ) {
      distance_with_block_0 = target_graph.distanceAfterExchangingBlocks(connectivity_set, block_1, block_0);
    } else {
      distance_with_block_0 = target_graph.distanceWithBlock(connectivity_set, block_0);
    }
    return std::abs(current_distance - distance_with_block_0) * edge_weight;
  } else {
    // Hyperedge is cut
    // => does we either use min(gain_0, gain_1) to compute a lower bound for the actual improvement or
    // max(gain_0,gain_1) to compute an uppter bound for the actual improvement.
    const HyperedgeWeight distance_without_block_0 = target_graph.distanceWithoutBlock(connectivity_set, block_0);
    const HyperedgeWeight distance_without_block_1 = target_graph.distanceWithoutBlock(connectivity_set, block_1);
    const HyperedgeWeight gain_0 = (current_distance - distance_without_block_0) * edge_weight;
    const HyperedgeWeight gain_1 = (current_distance - distance_without_block_1) * edge_weight;
    return capacity_for_cut_edge(context.refinement.flows.steiner_tree_policy, gain_0, gain_1);
  }
}

template<typename PartitionedHypergraph>
bool SteinerTreeFlowNetworkConstruction::connectToSource(const PartitionedHypergraph& partitioned_hg,
                                                         const HyperedgeID he,
                                                         const PartitionID block_0,
                                                         const PartitionID block_1) {
  ASSERT(partitioned_hg.hasTargetGraph());
  const HypernodeID pin_count_block_0 = partitioned_hg.pinCountInPart(he, block_0);
  const HypernodeID pin_count_block_1 = partitioned_hg.pinCountInPart(he, block_1);
  const TargetGraph& target_graph = *partitioned_hg.targetGraph();
  if ( pin_count_block_0 > 0 && pin_count_block_1 == 0 ) {
    ds::Bitset& connectivity_set = partitioned_hg.deepCopyOfConnectivitySet(he);
    const HyperedgeWeight current_distance = target_graph.distance(connectivity_set);
    const HyperedgeWeight distance_after_exchange =
      target_graph.distanceAfterExchangingBlocks(connectivity_set, block_0, block_1);
    if ( current_distance < distance_after_exchange ) {
      // If all nodes from block_0 would move to block_1, we would worsen the steiner tree metric,
      // even though the connectivity of the hyperedge does not change. To model this percurlarity in the flow network,
      // we add the corresponding hyperedge to the source.
      return true;
    }
  }
  if ( pin_count_block_0 == 0 && pin_count_block_1 == 1 ) {
    ds::Bitset& connectivity_set = partitioned_hg.deepCopyOfConnectivitySet(he);
    const HyperedgeWeight current_distance = target_graph.distance(connectivity_set);
    const HyperedgeWeight distance_after_exchange =
      target_graph.distanceAfterExchangingBlocks(connectivity_set, block_1, block_0);
    if ( current_distance > distance_after_exchange ) {
      return true;
    }
  }
  return false;
}


template<typename PartitionedHypergraph>
bool SteinerTreeFlowNetworkConstruction::connectToSink(const PartitionedHypergraph& partitioned_hg,
                                                       const HyperedgeID he,
                                                       const PartitionID block_0,
                                                       const PartitionID block_1) {
  ASSERT(partitioned_hg.hasTargetGraph());
  const HypernodeID pin_count_block_0 = partitioned_hg.pinCountInPart(he, block_0);
  const HypernodeID pin_count_block_1 = partitioned_hg.pinCountInPart(he, block_1);
  const TargetGraph& target_graph = *partitioned_hg.targetGraph();
  if ( partitioned_hg.pinCountInPart(he, block_0) == 0 && partitioned_hg.pinCountInPart(he, block_1) > 0 ) {
    ds::Bitset& connectivity_set = partitioned_hg.deepCopyOfConnectivitySet(he);
    const HyperedgeWeight current_distance = target_graph.distance(connectivity_set);
    const HyperedgeWeight distance_after_exchange =
      target_graph.distanceAfterExchangingBlocks(connectivity_set, block_1, block_0);
    if ( current_distance < distance_after_exchange ) {
      // If all nodes from block_1 would move to block_0, we would worsen the steiner tree metric,
      // even though the connectivity of the hyperedge does not change. To model this percurlarity in the flow network,
      // we add the corresponding hyperedge to the sink.
      return true;
    }
  }
  if ( pin_count_block_0 == 1 && pin_count_block_1 == 0 ) {
    ds::Bitset& connectivity_set = partitioned_hg.deepCopyOfConnectivitySet(he);
    const HyperedgeWeight current_distance = target_graph.distance(connectivity_set);
    const HyperedgeWeight distance_after_exchange =
      target_graph.distanceAfterExchangingBlocks(connectivity_set, block_0, block_1);
    if ( current_distance > distance_after_exchange ) {
      return true;
    }
  }
  return false;
}

template<typename PartitionedHypergraph>
bool SteinerTreeFlowNetworkConstruction::isCut(const PartitionedHypergraph& partitioned_hg,
                                               const HyperedgeID he,
                                               const PartitionID block_0,
                                               const PartitionID block_1) {
  ASSERT(partitioned_hg.hasTargetGraph());
  const HypernodeID pin_count_block_0 = partitioned_hg.pinCountInPart(he, block_0);
  const HypernodeID pin_count_block_1 = partitioned_hg.pinCountInPart(he, block_1);
  const TargetGraph& target_graph = *partitioned_hg.targetGraph();
  if ( pin_count_block_0 == 0 && pin_count_block_1 == 1 ) {
    ds::Bitset& connectivity_set = partitioned_hg.deepCopyOfConnectivitySet(he);
    const HyperedgeWeight current_distance = target_graph.distance(connectivity_set);
    const HyperedgeWeight distance_after_exchange =
      target_graph.distanceAfterExchangingBlocks(connectivity_set, block_1, block_0);
    if ( current_distance > distance_after_exchange ) {
      return true;
    }
  }
  if ( pin_count_block_0 == 1 && pin_count_block_1 == 0 ) {
    ds::Bitset& connectivity_set = partitioned_hg.deepCopyOfConnectivitySet(he);
    const HyperedgeWeight current_distance = target_graph.distance(connectivity_set);
    const HyperedgeWeight distance_after_exchange =
      target_graph.distanceAfterExchangingBlocks(connectivity_set, block_0, block_1);
    if ( current_distance > distance_after_exchange ) {
      return true;
    }
  }
  return false;
}

namespace {
#define STEINER_TREE_CAPACITY(X) HyperedgeWeight SteinerTreeFlowNetworkConstruction::capacity(  \
  const X&, const Context&, const HyperedgeID, const PartitionID, const PartitionID)
#define STEINER_TREE_CONNECT_TO_SOURCE(X) bool SteinerTreeFlowNetworkConstruction::connectToSource(  \
  const X&, const HyperedgeID, const PartitionID, const PartitionID)
#define STEINER_TREE_CONNECT_TO_SINK(X) bool SteinerTreeFlowNetworkConstruction::connectToSink(  \
  const X&, const HyperedgeID, const PartitionID, const PartitionID)
#define STEINER_TREE_IS_CUT(X) bool SteinerTreeFlowNetworkConstruction::isCut(  \
  const X&, const HyperedgeID, const PartitionID, const PartitionID)
}

INSTANTIATE_FUNC_WITH_PARTITIONED_HG(STEINER_TREE_CAPACITY)
INSTANTIATE_FUNC_WITH_PARTITIONED_HG(STEINER_TREE_CONNECT_TO_SOURCE)
INSTANTIATE_FUNC_WITH_PARTITIONED_HG(STEINER_TREE_CONNECT_TO_SINK)
INSTANTIATE_FUNC_WITH_PARTITIONED_HG(STEINER_TREE_IS_CUT)

}  // namespace mt_kahypar
