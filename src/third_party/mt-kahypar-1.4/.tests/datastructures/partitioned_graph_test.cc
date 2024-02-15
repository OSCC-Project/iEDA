/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2019 Tobias Heuer <tobias.heuer@kit.edu>
 * Copyright (C) 2021 Nikolai Maas <nikolai.maas@student.kit.edu>
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

#include <atomic>

#include "gmock/gmock.h"


#include "tests/definitions.h"
#include "mt-kahypar/datastructures/static_graph.h"
#include "mt-kahypar/datastructures/static_graph_factory.h"
#include "mt-kahypar/datastructures/partitioned_graph.h"
#include "mt-kahypar/partition/refinement/gains/cut_for_graphs/cut_gain_cache_for_graphs.h"
#include "mt-kahypar/partition/refinement/gains/cut/cut_attributed_gains.h"
#include "mt-kahypar/partition/metrics.h"

using ::testing::Test;

namespace mt_kahypar {
namespace ds {

template<typename TypeTraits>
class APartitionedGraph : public Test {

 public:
  using Hypergraph = typename TypeTraits::Hypergraph;
  using PartitionedGraph = typename TypeTraits::PartitionedHypergraph;
  using Factory = typename Hypergraph::Factory;

  APartitionedGraph() :
    hypergraph(Factory::construct(7 , 6,
      { {1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6} }, nullptr, nullptr, true)),
    partitioned_hypergraph(3, hypergraph),
    gain_cache() {
    initializePartition();
  }

  void initializePartition() {
    if ( hypergraph.nodeIsEnabled(0) ) partitioned_hypergraph.setNodePart(0, 0);
    if ( hypergraph.nodeIsEnabled(1) ) partitioned_hypergraph.setNodePart(1, 0);
    if ( hypergraph.nodeIsEnabled(2) ) partitioned_hypergraph.setNodePart(2, 0);
    if ( hypergraph.nodeIsEnabled(3) ) partitioned_hypergraph.setNodePart(3, 1);
    if ( hypergraph.nodeIsEnabled(4) ) partitioned_hypergraph.setNodePart(4, 1);
    if ( hypergraph.nodeIsEnabled(5) ) partitioned_hypergraph.setNodePart(5, 2);
    if ( hypergraph.nodeIsEnabled(6) ) partitioned_hypergraph.setNodePart(6, 2);
  }

  void verifyPartitionPinCountsAndConnectivity(const HyperedgeID he,
                                               const std::vector<HypernodeID>& expected_pin_counts) {
    ASSERT(expected_pin_counts.size() == static_cast<size_t>(partitioned_hypergraph.k()));
    for (PartitionID block = 0; block < 3; ++block) {
      const HypernodeID pin_count = partitioned_hypergraph.pinCountInPart(he, block);
      ASSERT_EQ(expected_pin_counts[block], pin_count) << V(he) << V(block);
    }
    HypernodeID expected_connectivity = 0;
    for (const PartitionID& block : partitioned_hypergraph.connectivitySet(he)) {
      ASSERT_TRUE(expected_pin_counts[block] > 0) << V(he) << V(block);
      expected_connectivity++;
    }
    ASSERT_EQ(expected_connectivity, partitioned_hypergraph.connectivity(he));
  }

  void verifyGains(const HypernodeID node, const std::vector<HyperedgeWeight>& expected_gains) {
    ASSERT(expected_gains.size() == static_cast<size_t>(partitioned_hypergraph.k()));
    const PartitionID part_id = partitioned_hypergraph.partID(node);
    for (PartitionID block = 0; block < 3; ++block) {
      if (block != part_id) {
        ASSERT_EQ(expected_gains[block], gain_cache.gain(node, part_id, block)) << V(node) << V(block);
      }
    }
  }

  void verifyPins(const Hypergraph& hg,
                  const std::vector<HyperedgeID> hyperedges,
                  const std::vector< std::set<HypernodeID> >& references,
                  bool log = false) {
    ASSERT(hyperedges.size() == references.size());
    for (size_t i = 0; i < hyperedges.size(); ++i) {
      const HyperedgeID he = hyperedges[i];
      const std::set<HypernodeID>& reference = references[i];
      size_t count = 0;
      for (const HypernodeID& pin : hg.pins(he)) {
        if (log) LOG << V(he) << V(pin);
        ASSERT_TRUE(reference.find(pin) != reference.end()) << V(he) << V(pin);
        count++;
      }
      ASSERT_EQ(count, reference.size());
    }
  }

  HyperedgeWeight compute_km1() {
    HyperedgeWeight km1 = 0;
    for (const HyperedgeID& he : partitioned_hypergraph.edges()) {
      km1 += std::max(partitioned_hypergraph.connectivity(he) - 1, 0) * partitioned_hypergraph.edgeWeight(he);
    }
    return km1;
  }

  void verifyAllKm1GainValues() {
    for ( const HypernodeID hn : hypergraph.nodes() ) {
      const PartitionID from = partitioned_hypergraph.partID(hn);
      for ( PartitionID to = 0; to < partitioned_hypergraph.k(); ++to ) {
        if ( from != to ) {
          const HyperedgeWeight km1_before = compute_km1();
          const HyperedgeWeight km1_gain = gain_cache.gain(hn, from, to);
          partitioned_hypergraph.changeNodePart(hn, from, to);
          const HyperedgeWeight km1_after = compute_km1();
          ASSERT_EQ(km1_gain, km1_before - km1_after);
          partitioned_hypergraph.changeNodePart(hn, to, from);
        }
      }
    }
  }

  Hypergraph hypergraph;
  PartitionedGraph partitioned_hypergraph;
  GraphCutGainCache gain_cache;
};

template <class F1, class F2>
void executeConcurrent(const F1& f1, const F2& f2) {
  std::atomic<int> cnt(0);
  tbb::parallel_invoke([&] {
    cnt++;
    while (cnt < 2) { }
    f1();
  }, [&] {
    cnt++;
    while (cnt < 2) { }
    f2();
  });
}
TYPED_TEST_CASE(APartitionedGraph, tests::GraphTestTypeTraits);

TYPED_TEST(APartitionedGraph, HasCorrectPartWeightAndSizes) {
  ASSERT_EQ(3, this->partitioned_hypergraph.partWeight(0));
  ASSERT_EQ(2, this->partitioned_hypergraph.partWeight(1));
  ASSERT_EQ(2, this->partitioned_hypergraph.partWeight(2));
}

TYPED_TEST(APartitionedGraph, HasCorrectPartWeightsIfOnlyOneThreadPerformsModifications) {
  ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(0, 0, 1));

  ASSERT_EQ(2, this->partitioned_hypergraph.partWeight(0));
  ASSERT_EQ(3, this->partitioned_hypergraph.partWeight(1));
  ASSERT_EQ(2, this->partitioned_hypergraph.partWeight(2));
}

TYPED_TEST(APartitionedGraph, PerformsConcurrentMovesWhereAllSucceed) {
  executeConcurrent([&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(0, 0, 1));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(3, 1, 2));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(2, 0, 2));
  }, [&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(5, 2, 1));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(6, 2, 0));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(4, 1, 2));
  });

  ASSERT_EQ(2, this->partitioned_hypergraph.partWeight(0));
  ASSERT_EQ(2, this->partitioned_hypergraph.partWeight(1));
  ASSERT_EQ(3, this->partitioned_hypergraph.partWeight(2));
}



TYPED_TEST(APartitionedGraph, HasCorrectInitialPartitionPinCounts) {
  // edge 1 - 2
  this->verifyPartitionPinCountsAndConnectivity(0, { 2, 0, 0 });
  this->verifyPartitionPinCountsAndConnectivity(2, { 2, 0, 0 });
  // edge 1 - 4
  this->verifyPartitionPinCountsAndConnectivity(1, { 1, 1, 0 });
  this->verifyPartitionPinCountsAndConnectivity(5, { 1, 1, 0 });
  // edge 2 - 3
  this->verifyPartitionPinCountsAndConnectivity(3, { 1, 1, 0 });
  this->verifyPartitionPinCountsAndConnectivity(4, { 1, 1, 0 });
  // edge 4 - 5
  this->verifyPartitionPinCountsAndConnectivity(6, { 0, 1, 1 });
  this->verifyPartitionPinCountsAndConnectivity(8, { 0, 1, 1 });
  // edge 4 - 6
  this->verifyPartitionPinCountsAndConnectivity(7, { 0, 1, 1 });
  this->verifyPartitionPinCountsAndConnectivity(10, { 0, 1, 1 });
  // edge 5 - 6
  this->verifyPartitionPinCountsAndConnectivity(9, { 0, 0, 2 });
  this->verifyPartitionPinCountsAndConnectivity(11, { 0, 0, 2 });
}

TYPED_TEST(APartitionedGraph, HasCorrectPartitionPinCountsIfTwoNodesMovesConcurrent) {
  executeConcurrent([&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(2, 0, 1));
  }, [&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(1, 0, 2));
  });

  // edge 1 - 2
  this->verifyPartitionPinCountsAndConnectivity(0, { 0, 1, 1 });
  this->verifyPartitionPinCountsAndConnectivity(2, { 0, 1, 1 });
  // edge 1 - 4
  this->verifyPartitionPinCountsAndConnectivity(1, { 0, 1, 1 });
  this->verifyPartitionPinCountsAndConnectivity(5, { 0, 1, 1 });
  // edge 2 - 3
  this->verifyPartitionPinCountsAndConnectivity(3, { 0, 2, 0 });
  this->verifyPartitionPinCountsAndConnectivity(4, { 0, 2, 0 });
}

TYPED_TEST(APartitionedGraph, HasCorrectPartitionPinCountsIfAllNodesMovesConcurrent) {
  executeConcurrent([&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(0, 0, 1));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(4, 1, 2));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(6, 2, 1));
  }, [&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(1, 0, 2));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(3, 1, 0));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(5, 2, 1));
  });

  // edge 1 - 2
  this->verifyPartitionPinCountsAndConnectivity(0, { 1, 0, 1 });
  this->verifyPartitionPinCountsAndConnectivity(2, { 1, 0, 1 });
  // edge 1 - 4
  this->verifyPartitionPinCountsAndConnectivity(1, { 0, 0, 2 });
  this->verifyPartitionPinCountsAndConnectivity(5, { 0, 0, 2 });
  // edge 2 - 3
  this->verifyPartitionPinCountsAndConnectivity(3, { 2, 0, 0 });
  this->verifyPartitionPinCountsAndConnectivity(4, { 2, 0, 0 });
  // edge 4 - 5
  this->verifyPartitionPinCountsAndConnectivity(6, { 0, 1, 1 });
  this->verifyPartitionPinCountsAndConnectivity(8, { 0, 1, 1 });
  // edge 4 - 6
  this->verifyPartitionPinCountsAndConnectivity(7, { 0, 1, 1 });
  this->verifyPartitionPinCountsAndConnectivity(10, { 0, 1, 1 });
  // edge 5 - 6
  this->verifyPartitionPinCountsAndConnectivity(9, { 0, 2, 0 });
  this->verifyPartitionPinCountsAndConnectivity(11, { 0, 2, 0 });
}

TYPED_TEST(APartitionedGraph, HasCorrectInitialBorderNodes) {
  ASSERT_FALSE(this->partitioned_hypergraph.isBorderNode(0));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(1));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(2));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(3));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(4));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(5));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(6));

  ASSERT_EQ(0, this->partitioned_hypergraph.numIncidentCutHyperedges(0));
  ASSERT_EQ(1, this->partitioned_hypergraph.numIncidentCutHyperedges(1));
  ASSERT_EQ(1, this->partitioned_hypergraph.numIncidentCutHyperedges(2));
  ASSERT_EQ(1, this->partitioned_hypergraph.numIncidentCutHyperedges(3));
  ASSERT_EQ(3, this->partitioned_hypergraph.numIncidentCutHyperedges(4));
  ASSERT_EQ(1, this->partitioned_hypergraph.numIncidentCutHyperedges(5));
  ASSERT_EQ(1, this->partitioned_hypergraph.numIncidentCutHyperedges(6));
}


TYPED_TEST(APartitionedGraph, ExtractBlockZero) {
  auto extracted_hg = this->partitioned_hypergraph.extract(0, nullptr, true, true);
  auto& hg = extracted_hg.hg;
  auto& mapping = extracted_hg.hn_mapping;

  ASSERT_EQ(3, hg.initialNumNodes());
  ASSERT_EQ(2, hg.initialNumEdges());
  ASSERT_EQ(2, hg.initialNumPins());
  ASSERT_EQ(2, hg.maxEdgeSize());

  this->verifyPins(hg, {0, 1},
    { {mapping[1], mapping[2]}, {mapping[1], mapping[2]} });
}


TYPED_TEST(APartitionedGraph, ExtractBlockOne) {
  auto extracted_hg = this->partitioned_hypergraph.extract(1, nullptr, true, true);
  auto& hg = extracted_hg.hg;

  ASSERT_EQ(2, hg.initialNumNodes());
  ASSERT_EQ(0, hg.initialNumEdges());
  ASSERT_EQ(0, hg.initialNumPins());
}

TYPED_TEST(APartitionedGraph, ExtractBlockTwo) {
  auto extracted_hg = this->partitioned_hypergraph.extract(2, nullptr, true, true);
  auto& hg = extracted_hg.hg;
  auto& mapping = extracted_hg.hn_mapping;

  ASSERT_EQ(2, hg.initialNumNodes());
  ASSERT_EQ(2, hg.initialNumEdges());
  ASSERT_EQ(2, hg.initialNumPins());
  ASSERT_EQ(2, hg.maxEdgeSize());

  this->verifyPins(hg, {0, 1},
    { {mapping[5], mapping[6]}, {mapping[5], mapping[6]} });
}

TYPED_TEST(APartitionedGraph, ExtractsAllBlocks) {
  auto extracted_hg = this->partitioned_hypergraph.extractAllBlocks(3, nullptr, true, true);
  auto& graphs = extracted_hg.first;
  auto& mapping = extracted_hg.second;

  ASSERT_EQ(3, graphs[0].hg.initialNumNodes());
  ASSERT_EQ(2, graphs[0].hg.initialNumEdges());
  ASSERT_EQ(2, graphs[0].hg.initialNumPins());
  ASSERT_EQ(2, graphs[0].hg.maxEdgeSize());

  this->verifyPins(graphs[0].hg, {0, 1},
    { {mapping[1], mapping[2]}, {mapping[1], mapping[2]} });

  ASSERT_EQ(2, graphs[1].hg.initialNumNodes());
  ASSERT_EQ(0, graphs[1].hg.initialNumEdges());
  ASSERT_EQ(0, graphs[1].hg.initialNumPins());
  ASSERT_EQ(2, graphs[1].hg.maxEdgeSize());

  this->verifyPins(graphs[1].hg, {}, { });

  ASSERT_EQ(2, graphs[2].hg.initialNumNodes());
  ASSERT_EQ(2, graphs[2].hg.initialNumEdges());
  ASSERT_EQ(2, graphs[2].hg.initialNumPins());
  ASSERT_EQ(2, graphs[2].hg.maxEdgeSize());

  this->verifyPins(graphs[2].hg, {0, 1},
    { {mapping[5], mapping[6]}, {mapping[5], mapping[6]} });
}

TYPED_TEST(APartitionedGraph, ExtractBlockZeroWithCommunityInformation) {
  this->hypergraph.setCommunityID(0, 0);
  this->hypergraph.setCommunityID(1, 1);
  this->hypergraph.setCommunityID(2, 2);
  this->hypergraph.setCommunityID(3, 4);
  this->hypergraph.setCommunityID(4, 3);
  this->hypergraph.setCommunityID(5, 4);
  this->hypergraph.setCommunityID(6, 5);

  auto extracted_hg = this->partitioned_hypergraph.extract(0, nullptr, true, true);
  auto& hg = extracted_hg.hg;
  auto& mapping = extracted_hg.hn_mapping;

  ASSERT_EQ(0, hg.communityID(mapping[0]));
  ASSERT_EQ(1, hg.communityID(mapping[1]));
  ASSERT_EQ(2, hg.communityID(mapping[2]));
}

TYPED_TEST(APartitionedGraph, ComputesPartInfoCorrectlyIfNodePartsAreSetOnly) {
  this->partitioned_hypergraph.resetPartition();
  this->partitioned_hypergraph.setOnlyNodePart(0, 0);
  this->partitioned_hypergraph.setOnlyNodePart(1, 0);
  this->partitioned_hypergraph.setOnlyNodePart(2, 0);
  this->partitioned_hypergraph.setOnlyNodePart(3, 1);
  this->partitioned_hypergraph.setOnlyNodePart(4, 1);
  this->partitioned_hypergraph.setOnlyNodePart(5, 2);
  this->partitioned_hypergraph.setOnlyNodePart(6, 2);
  this->partitioned_hypergraph.initializePartition();

  ASSERT_EQ(3, this->partitioned_hypergraph.partWeight(0));
  ASSERT_EQ(2, this->partitioned_hypergraph.partWeight(1));
  ASSERT_EQ(2, this->partitioned_hypergraph.partWeight(2));
}

TYPED_TEST(APartitionedGraph, ComputesGainsCorrectly) {
  this->gain_cache.initializeGainCache(this->partitioned_hypergraph);

  this->verifyGains(0, {0, 0, 0});
  this->verifyGains(1, {0, 0, -1});
  this->verifyGains(2, {0, 0, -1});
  this->verifyGains(3, {1, 0, 0});
  this->verifyGains(4, {1, 0, 2});
  this->verifyGains(5, {-1, 0, 0});
  this->verifyGains(6, {-1, 0, 0});
}

TYPED_TEST(APartitionedGraph, ComputesDeltaAndGainsCorrectlyIfAllNodesMoveConcurrently) {
  this->gain_cache.initializeGainCache(this->partitioned_hypergraph);

  CAtomic<HyperedgeWeight> delta(0);
  auto delta_fun = [&](const SynchronizedEdgeUpdate& sync_update) {
      delta.fetch_add(CutAttributedGains::gain(sync_update));
  };

  executeConcurrent([&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(this->gain_cache, 4, 1, 2, 5, []{}, delta_fun));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(this->gain_cache, 2, 0, 2, 5, []{}, delta_fun));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(this->gain_cache, 3, 1, 2, 5, []{}, delta_fun));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(this->gain_cache, 4, 2, 0, 5, []{}, delta_fun));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(this->gain_cache, 2, 2, 1, 5, []{}, delta_fun));
  }, [&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(this->gain_cache, 5, 2, 0, 5, []{}, delta_fun));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(this->gain_cache, 1, 0, 2, 5, []{}, delta_fun));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(this->gain_cache, 6, 2, 0, 5, []{}, delta_fun));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(this->gain_cache, 0, 0, 2, 5, []{}, delta_fun));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(this->gain_cache, 1, 2, 1, 5, []{}, delta_fun));
  });

  ASSERT_EQ(-2, delta.load());
  this->verifyGains(0, {0, 0, 0});
  this->verifyGains(1, {0, 0, -1});
  this->verifyGains(2, {-1, 0, 0});
  this->verifyGains(3, {0, 1, 0});
  this->verifyGains(4, {0, -1, -2});
  this->verifyGains(5, {0, -2, -2});
  this->verifyGains(6, {0, -2, -2});
}

}  // namespace ds
}  // namespace mt_kahypar
