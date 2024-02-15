/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
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



#include <atomic>
#include <mt-kahypar/parallel/tbb_initializer.h>

#include "gmock/gmock.h"

#include "tests/definitions.h"

using ::testing::Test;

namespace mt_kahypar {
namespace ds {


template<typename TypeTraits>
class APartitionedHypergraph : public Test {

 public:
 using Hypergraph = typename TypeTraits::Hypergraph;
 using PartitionedHypergraph = typename TypeTraits::PartitionedHypergraph;
 using Factory = typename Hypergraph::Factory;

  APartitionedHypergraph() :
    hypergraph(Factory::construct(
      7 , 4, { {0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6} })),
    partitioned_hypergraph(3, hypergraph, parallel_tag_t()) {
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

  void verifyPartitionPinCounts(const HyperedgeID he,
                                const std::vector<HypernodeID>& expected_pin_counts) {
    ASSERT(expected_pin_counts.size() == static_cast<size_t>(partitioned_hypergraph.k()));
    for (PartitionID block = 0; block < 3; ++block) {
      ASSERT_EQ(expected_pin_counts[block], partitioned_hypergraph.pinCountInPart(he, block)) << V(he) << V(block);
    }
  }

  void verifyConnectivitySet(const HyperedgeID he,
                             const std::set<PartitionID>& connectivity_set) {
    ASSERT_EQ(connectivity_set.size(), partitioned_hypergraph.connectivity(he)) << V(he);
    PartitionID connectivity = 0;
    for (const PartitionID& block : partitioned_hypergraph.connectivitySet(he)) {
      ASSERT_TRUE(connectivity_set.find(block) != connectivity_set.end()) << V(he) << V(block);
      ++connectivity;
    }
    ASSERT_EQ(connectivity_set.size(), connectivity) << V(he);
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
          const HyperedgeWeight km1_gain = partitioned_hypergraph.km1Gain(hn, from, to);
          partitioned_hypergraph.changeNodePart(hn, from, to);
          const HyperedgeWeight km1_after = compute_km1();
          ASSERT_EQ(km1_gain, km1_before - km1_after);
          partitioned_hypergraph.changeNodePart(hn, to, from);
        }
      }
    }
  }

  Hypergraph hypergraph;
  PartitionedHypergraph partitioned_hypergraph;
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

TYPED_TEST_CASE(APartitionedHypergraph, tests::HypergraphTestTypeTraits);

TYPED_TEST(APartitionedHypergraph, HasCorrectPartWeightAndSizes) {
  ASSERT_EQ(3, this->partitioned_hypergraph.partWeight(0));
  ASSERT_EQ(2, this->partitioned_hypergraph.partWeight(1));
  ASSERT_EQ(2, this->partitioned_hypergraph.partWeight(2));
}

TYPED_TEST(APartitionedHypergraph, HasCorrectPartWeightsIfOnlyOneThreadPerformsModifications) {
  ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(0, 0, 1));

  ASSERT_EQ(2, this->partitioned_hypergraph.partWeight(0));
  ASSERT_EQ(3, this->partitioned_hypergraph.partWeight(1));
  ASSERT_EQ(2, this->partitioned_hypergraph.partWeight(2));
}

TYPED_TEST(APartitionedHypergraph, PerformsConcurrentMovesWhereAllSucceed) {
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



TYPED_TEST(APartitionedHypergraph, HasCorrectInitialPartitionPinCounts) {
  this->verifyPartitionPinCounts(0, { 2, 0, 0 });
  this->verifyPartitionPinCounts(1, { 2, 2, 0 });
  this->verifyPartitionPinCounts(2, { 0, 2, 1 });
  this->verifyPartitionPinCounts(3, { 1, 0, 2 });
}

TYPED_TEST(APartitionedHypergraph, HasCorrectPartitionPinCountsIfTwoNodesMovesConcurrent1) {
  executeConcurrent([&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(0, 0, 1));
  }, [&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(1, 0, 2));
  });

  this->verifyPartitionPinCounts(0, { 1, 1, 0 });
  this->verifyPartitionPinCounts(1, { 0, 3, 1 });
  this->verifyPartitionPinCounts(2, { 0, 2, 1 });
  this->verifyPartitionPinCounts(3, { 1, 0, 2 });
}

TYPED_TEST(APartitionedHypergraph, HasCorrectPartitionPinCountsIfTwoNodesMovesConcurrent2) {
  executeConcurrent([&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(3, 1, 2));
  }, [&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(6, 2, 0));
  });

  this->verifyPartitionPinCounts(0, { 2, 0, 0 });
  this->verifyPartitionPinCounts(1, { 2, 1, 1 });
  this->verifyPartitionPinCounts(2, { 1, 1, 1 });
  this->verifyPartitionPinCounts(3, { 2, 0, 1 });
}

TYPED_TEST(APartitionedHypergraph, HasCorrectPartitionPinCountsIfTwoNodesMovesConcurrent3) {
  executeConcurrent([&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(3, 1, 2));
  }, [&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(4, 1, 2));
  });

  this->verifyPartitionPinCounts(0, { 2, 0, 0 });
  this->verifyPartitionPinCounts(1, { 2, 0, 2 });
  this->verifyPartitionPinCounts(2, { 0, 0, 3 });
  this->verifyPartitionPinCounts(3, { 1, 0, 2 });
}

TYPED_TEST(APartitionedHypergraph, HasCorrectPartitionPinCountsIfTwoNodesMovesConcurrent4) {
  executeConcurrent([&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(2, 0, 2));
  }, [&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(5, 2, 0));
  });

  this->verifyPartitionPinCounts(0, { 1, 0, 1 });
  this->verifyPartitionPinCounts(1, { 2, 2, 0 });
  this->verifyPartitionPinCounts(2, { 0, 2, 1 });
  this->verifyPartitionPinCounts(3, { 1, 0, 2 });
}

TYPED_TEST(APartitionedHypergraph, HasCorrectPartitionPinCountsIfTwoNodesMovesConcurrent5) {
  executeConcurrent([&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(0, 0, 1));
  }, [&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(6, 2, 1));
  });

  this->verifyPartitionPinCounts(0, { 1, 1, 0 });
  this->verifyPartitionPinCounts(1, { 1, 3, 0 });
  this->verifyPartitionPinCounts(2, { 0, 3, 0 });
  this->verifyPartitionPinCounts(3, { 1, 1, 1 });
}

TYPED_TEST(APartitionedHypergraph, HasCorrectPartitionPinCountsIfAllNodesMovesConcurrent) {
  executeConcurrent([&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(0, 0, 1));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(2, 0, 2));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(4, 1, 0));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(6, 2, 1));
  }, [&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(1, 0, 2));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(3, 1, 0));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(5, 2, 1));
  });

  this->verifyPartitionPinCounts(0, { 0, 1, 1 });
  this->verifyPartitionPinCounts(1, { 2, 1, 1 });
  this->verifyPartitionPinCounts(2, { 2, 1, 0 });
  this->verifyPartitionPinCounts(3, { 0, 2, 1 });
}


TYPED_TEST(APartitionedHypergraph, HasCorrectConnectivitySetIfTwoNodesMovesConcurrent1) {
  executeConcurrent([&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(6, 2, 0));
  }, [&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(0, 0, 1));
  });

  this->verifyConnectivitySet(0, { 0, 1 });
  this->verifyConnectivitySet(1, { 0, 1 });
  this->verifyConnectivitySet(2, { 0, 1 });
  this->verifyConnectivitySet(3, { 0, 2 });
}

TYPED_TEST(APartitionedHypergraph, HasCorrectConnectivitySetIfTwoNodesMovesConcurrent2) {
  executeConcurrent([&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(5, 2, 0));
  }, [&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(2, 0, 2));
  });

  this->verifyConnectivitySet(0, { 0, 2 });
  this->verifyConnectivitySet(1, { 0, 1 });
  this->verifyConnectivitySet(2, { 1, 2 });
  this->verifyConnectivitySet(3, { 0, 2 });
}

TYPED_TEST(APartitionedHypergraph, HasCorrectConnectivitySetIfTwoNodesMovesConcurrent3) {
  executeConcurrent([&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(0, 0, 1));
  }, [&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(2, 0, 1));
  });

  this->verifyConnectivitySet(0, { 1 });
  this->verifyConnectivitySet(1, { 0, 1 });
  this->verifyConnectivitySet(2, { 1, 2 });
  this->verifyConnectivitySet(3, { 1, 2 });
}

TYPED_TEST(APartitionedHypergraph, HasCorrectConnectivitySetIfTwoNodesMovesConcurrent4) {
  executeConcurrent([&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(4, 1, 0));
  }, [&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(3, 1, 0));
  });

  this->verifyConnectivitySet(0, { 0 });
  this->verifyConnectivitySet(1, { 0 });
  this->verifyConnectivitySet(2, { 0, 2 });
  this->verifyConnectivitySet(3, { 0, 2 });
}

TYPED_TEST(APartitionedHypergraph, HasCorrectConnectivitySetIfTwoNodesMovesConcurrent5) {
  executeConcurrent([&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(1, 0, 2));
  }, [&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(3, 1, 2));
  });

  this->verifyConnectivitySet(0, { 0 });
  this->verifyConnectivitySet(1, { 0, 1, 2 });
  this->verifyConnectivitySet(2, { 1, 2 });
  this->verifyConnectivitySet(3, { 0, 2 });
}

TYPED_TEST(APartitionedHypergraph, HasCorrectConnectivitySetIfAllNodesMovesConcurrent) {
  executeConcurrent([&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(0, 0, 1));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(3, 1, 0));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(6, 2, 0));
  }, [&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(1, 0, 2));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(2, 0, 1));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(4, 1, 0));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(5, 2, 1));
  });

  this->verifyConnectivitySet(0, { 1 });
  this->verifyConnectivitySet(1, { 0, 1, 2 });
  this->verifyConnectivitySet(2, { 0 });
  this->verifyConnectivitySet(3, { 0, 1 });
}

TYPED_TEST(APartitionedHypergraph, HasCorrectInitialBorderNodes) {
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(0));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(1));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(2));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(3));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(4));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(5));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(6));

  ASSERT_EQ(1, this->partitioned_hypergraph.numIncidentCutHyperedges(0));
  ASSERT_EQ(1, this->partitioned_hypergraph.numIncidentCutHyperedges(1));
  ASSERT_EQ(1, this->partitioned_hypergraph.numIncidentCutHyperedges(2));
  ASSERT_EQ(2, this->partitioned_hypergraph.numIncidentCutHyperedges(3));
  ASSERT_EQ(2, this->partitioned_hypergraph.numIncidentCutHyperedges(4));
  ASSERT_EQ(1, this->partitioned_hypergraph.numIncidentCutHyperedges(5));
  ASSERT_EQ(2, this->partitioned_hypergraph.numIncidentCutHyperedges(6));
}

TYPED_TEST(APartitionedHypergraph, HasCorrectBorderNodesIfNodesAreMovingConcurrently1) {
  executeConcurrent([&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(4, 1, 0));
  }, [&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(3, 1, 0));
  });

  ASSERT_FALSE(this->partitioned_hypergraph.isBorderNode(0));
  ASSERT_FALSE(this->partitioned_hypergraph.isBorderNode(1));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(2));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(3));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(4));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(5));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(6));

  ASSERT_EQ(0, this->partitioned_hypergraph.numIncidentCutHyperedges(0));
  ASSERT_EQ(0, this->partitioned_hypergraph.numIncidentCutHyperedges(1));
  ASSERT_EQ(1, this->partitioned_hypergraph.numIncidentCutHyperedges(2));
  ASSERT_EQ(1, this->partitioned_hypergraph.numIncidentCutHyperedges(3));
  ASSERT_EQ(1, this->partitioned_hypergraph.numIncidentCutHyperedges(4));
  ASSERT_EQ(1, this->partitioned_hypergraph.numIncidentCutHyperedges(5));
  ASSERT_EQ(2, this->partitioned_hypergraph.numIncidentCutHyperedges(6));
}

TYPED_TEST(APartitionedHypergraph, HasCorrectBorderNodesIfNodesAreMovingConcurrently2) {
  executeConcurrent([&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(3, 1, 0));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(1, 0, 1));
  }, [&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(4, 1, 0));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(2, 0, 1));
  });

  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(0));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(1));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(2));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(3));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(4));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(5));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(6));

  ASSERT_EQ(2, this->partitioned_hypergraph.numIncidentCutHyperedges(0));
  ASSERT_EQ(1, this->partitioned_hypergraph.numIncidentCutHyperedges(1));
  ASSERT_EQ(2, this->partitioned_hypergraph.numIncidentCutHyperedges(2));
  ASSERT_EQ(2, this->partitioned_hypergraph.numIncidentCutHyperedges(3));
  ASSERT_EQ(2, this->partitioned_hypergraph.numIncidentCutHyperedges(4));
  ASSERT_EQ(1, this->partitioned_hypergraph.numIncidentCutHyperedges(5));
  ASSERT_EQ(2, this->partitioned_hypergraph.numIncidentCutHyperedges(6));
}

TYPED_TEST(APartitionedHypergraph, HasCorrectBorderNodesIfNodesAreMovingConcurrently3) {
  executeConcurrent([&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(6, 2, 0));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(3, 1, 0));
  }, [&] {
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(5, 2, 0));
    ASSERT_TRUE(this->partitioned_hypergraph.changeNodePart(4, 1, 0));
  });

  ASSERT_FALSE(this->partitioned_hypergraph.isBorderNode(0));
  ASSERT_FALSE(this->partitioned_hypergraph.isBorderNode(1));
  ASSERT_FALSE(this->partitioned_hypergraph.isBorderNode(2));
  ASSERT_FALSE(this->partitioned_hypergraph.isBorderNode(3));
  ASSERT_FALSE(this->partitioned_hypergraph.isBorderNode(4));
  ASSERT_FALSE(this->partitioned_hypergraph.isBorderNode(5));
  ASSERT_FALSE(this->partitioned_hypergraph.isBorderNode(6));

  ASSERT_EQ(0, this->partitioned_hypergraph.numIncidentCutHyperedges(0));
  ASSERT_EQ(0, this->partitioned_hypergraph.numIncidentCutHyperedges(1));
  ASSERT_EQ(0, this->partitioned_hypergraph.numIncidentCutHyperedges(2));
  ASSERT_EQ(0, this->partitioned_hypergraph.numIncidentCutHyperedges(3));
  ASSERT_EQ(0, this->partitioned_hypergraph.numIncidentCutHyperedges(4));
  ASSERT_EQ(0, this->partitioned_hypergraph.numIncidentCutHyperedges(5));
  ASSERT_EQ(0, this->partitioned_hypergraph.numIncidentCutHyperedges(6));
}



TYPED_TEST(APartitionedHypergraph, ExtractBlockZeroWithCutNetSplitting) {
  auto extracted_hg = this->partitioned_hypergraph.extract(0, nullptr, true, true);
  auto& hg = extracted_hg.hg;
  auto& hn_mapping = extracted_hg.hn_mapping;

  ASSERT_EQ(3, hg.initialNumNodes());
  ASSERT_EQ(2, hg.initialNumEdges());
  ASSERT_EQ(4, hg.initialNumPins());
  ASSERT_EQ(2, hg.maxEdgeSize());

  auto map_from_original_to_extracted_hg = [&](const HypernodeID hn) {
    return hn_mapping[hn];
  };
  parallel::scalable_vector<HypernodeID> node_id = {
    map_from_original_to_extracted_hg(0),
    map_from_original_to_extracted_hg(1),
    map_from_original_to_extracted_hg(2)
  };

  this->verifyPins(hg, {0, 1},
    { {node_id[0], node_id[2]}, {node_id[0], node_id[1]} });
}


TYPED_TEST(APartitionedHypergraph, ExtractBlockOneWithCutNetSplitting) {
  auto extracted_hg = this->partitioned_hypergraph.extract(1, nullptr, true, true);
  auto& hg = extracted_hg.hg;
  auto& hn_mapping = extracted_hg.hn_mapping;

  ASSERT_EQ(2, hg.initialNumNodes());
  ASSERT_EQ(2, hg.initialNumEdges());
  ASSERT_EQ(4, hg.initialNumPins());
  ASSERT_EQ(2, hg.maxEdgeSize());

  auto map_from_original_to_extracted_hg = [&](const HypernodeID hn) {
    return hn_mapping[hn];
  };
  parallel::scalable_vector<HypernodeID> node_id = {
    map_from_original_to_extracted_hg(3),
    map_from_original_to_extracted_hg(4)
  };

  this->verifyPins(hg, {0, 1},
    { {node_id[0], node_id[1]}, {node_id[0], node_id[1]} });
}

TYPED_TEST(APartitionedHypergraph, ExtractBlockTwoWithCutNetSplitting) {
  auto extracted_hg = this->partitioned_hypergraph.extract(2, nullptr, true, true);
  auto& hg = extracted_hg.hg;
  auto& hn_mapping = extracted_hg.hn_mapping;

  ASSERT_EQ(2, hg.initialNumNodes());
  ASSERT_EQ(1, hg.initialNumEdges());
  ASSERT_EQ(2, hg.initialNumPins());
  ASSERT_EQ(2, hg.maxEdgeSize());

  auto map_from_original_to_extracted_hg = [&](const HypernodeID hn) {
    return hn_mapping[hn];
  };
  parallel::scalable_vector<HypernodeID> node_id = {
    map_from_original_to_extracted_hg(5),
    map_from_original_to_extracted_hg(6)
  };

  this->verifyPins(hg, {0},
    { {node_id[0], node_id[1]} });
}



TYPED_TEST(APartitionedHypergraph, ExtractBlockZeroWithCutNetRemoval) {
  auto extracted_hg = this->partitioned_hypergraph.extract(0, nullptr, false, true);
  auto& hg = extracted_hg.hg;
  auto& hn_mapping = extracted_hg.hn_mapping;

  ASSERT_EQ(3, hg.initialNumNodes());
  ASSERT_EQ(1, hg.initialNumEdges());
  ASSERT_EQ(2, hg.initialNumPins());
  ASSERT_EQ(2, hg.maxEdgeSize());

  auto map_from_original_to_extracted_hg = [&](const HypernodeID hn) {
    return hn_mapping[hn];
  };
  parallel::scalable_vector<HypernodeID> node_id = {
    map_from_original_to_extracted_hg(0),
    map_from_original_to_extracted_hg(1),
    map_from_original_to_extracted_hg(2)
  };
  parallel::scalable_vector<HypernodeID> edge_id = { 0 };

  this->verifyPins(hg, {0},
    { {node_id[0], node_id[2]} });
}

TYPED_TEST(APartitionedHypergraph, ExtractBlockOneWithCutNetRemoval) {
  this->partitioned_hypergraph.changeNodePart(6, 2, 1);
  auto extracted_hg = this->partitioned_hypergraph.extract(1, nullptr, false, true);
  auto& hg = extracted_hg.hg;
  auto& hn_mapping = extracted_hg.hn_mapping;

  ASSERT_EQ(3, hg.initialNumNodes());
  ASSERT_EQ(1, hg.initialNumEdges());
  ASSERT_EQ(3, hg.initialNumPins());
  ASSERT_EQ(3, hg.maxEdgeSize());

  auto map_from_original_to_extracted_hg = [&](const HypernodeID hn) {
    return hn_mapping[hn];
  };
  parallel::scalable_vector<HypernodeID> node_id = {
    map_from_original_to_extracted_hg(3),
    map_from_original_to_extracted_hg(4),
    map_from_original_to_extracted_hg(6)
  };

  this->verifyPins(hg, {0},
    { {node_id[0], node_id[1], node_id[2]} });
}

TYPED_TEST(APartitionedHypergraph, ExtractBlockTwoWithCutNetRemoval) {
  this->partitioned_hypergraph.changeNodePart(2, 0, 2);
  auto extracted_hg = this->partitioned_hypergraph.extract(2, nullptr, false, true);
  auto& hg = extracted_hg.hg;
  auto& hn_mapping = extracted_hg.hn_mapping;

  ASSERT_EQ(3, hg.initialNumNodes());
  ASSERT_EQ(1, hg.initialNumEdges());
  ASSERT_EQ(3, hg.initialNumPins());
  ASSERT_EQ(3, hg.maxEdgeSize());

  auto map_from_original_to_extracted_hg = [&](const HypernodeID hn) {
    return hn_mapping[hn];
  };
  parallel::scalable_vector<HypernodeID> node_id = {
    map_from_original_to_extracted_hg(2),
    map_from_original_to_extracted_hg(5),
    map_from_original_to_extracted_hg(6)
  };

  this->verifyPins(hg, {0},
    { {node_id[0], node_id[1], node_id[2]} });
}

TYPED_TEST(APartitionedHypergraph, ExtractAllBlockBlocksWithCutNetSplitting) {
  auto extracted_hg = this->partitioned_hypergraph.extractAllBlocks(3, nullptr, true, true);
  auto& hypergraphs = extracted_hg.first;
  vec<HypernodeID>& hn_mapping = extracted_hg.second;

  ASSERT_EQ(3, hypergraphs[0].hg.initialNumNodes());
  ASSERT_EQ(2, hypergraphs[0].hg.initialNumEdges());
  ASSERT_EQ(4, hypergraphs[0].hg.initialNumPins());
  ASSERT_EQ(2, hypergraphs[0].hg.maxEdgeSize());

  parallel::scalable_vector<HypernodeID> node_id = {
    hn_mapping[0], hn_mapping[1], hn_mapping[2] };
  this->verifyPins(hypergraphs[0].hg, {0, 1},
    { { node_id[0], node_id[2] }, { node_id[0], node_id[1] } });

  ASSERT_EQ(2, hypergraphs[1].hg.initialNumNodes());
  ASSERT_EQ(2, hypergraphs[1].hg.initialNumEdges());
  ASSERT_EQ(4, hypergraphs[1].hg.initialNumPins());
  ASSERT_EQ(2, hypergraphs[1].hg.maxEdgeSize());

  node_id = { hn_mapping[3], hn_mapping[4] };
  this->verifyPins(hypergraphs[1].hg, {0, 1},
    { { node_id[0], node_id[1] }, { node_id[0], node_id[1] } });

  ASSERT_EQ(2, hypergraphs[2].hg.initialNumNodes());
  ASSERT_EQ(1, hypergraphs[2].hg.initialNumEdges());
  ASSERT_EQ(2, hypergraphs[2].hg.initialNumPins());
  ASSERT_EQ(2, hypergraphs[2].hg.maxEdgeSize());

  node_id = { hn_mapping[5], hn_mapping[6] };
  this->verifyPins(hypergraphs[2].hg, { 0 },
    { { node_id[0], node_id[1] } });
}

TYPED_TEST(APartitionedHypergraph, ExtractAllBlockBlocksWithCutNetRemoval) {
  this->partitioned_hypergraph.changeNodePart(6, 2, 1);
  auto extracted_hg = this->partitioned_hypergraph.extractAllBlocks(3, nullptr, false, true);
  auto& hypergraphs = extracted_hg.first;
  vec<HypernodeID>& hn_mapping = extracted_hg.second;

  ASSERT_EQ(3, hypergraphs[0].hg.initialNumNodes());
  ASSERT_EQ(1, hypergraphs[0].hg.initialNumEdges());
  ASSERT_EQ(2, hypergraphs[0].hg.initialNumPins());
  ASSERT_EQ(2, hypergraphs[0].hg.maxEdgeSize());

  parallel::scalable_vector<HypernodeID> node_id = {
    hn_mapping[0], hn_mapping[1], hn_mapping[2] };
  this->verifyPins(hypergraphs[0].hg, {0},
    { { node_id[0], node_id[2] } });

  ASSERT_EQ(3, hypergraphs[1].hg.initialNumNodes());
  ASSERT_EQ(1, hypergraphs[1].hg.initialNumEdges());
  ASSERT_EQ(3, hypergraphs[1].hg.initialNumPins());
  ASSERT_EQ(3, hypergraphs[1].hg.maxEdgeSize());

  node_id = { hn_mapping[3], hn_mapping[4], hn_mapping[6] };
  this->verifyPins(hypergraphs[1].hg, {0},
    { { node_id[0], node_id[1], node_id[2] } });

  ASSERT_EQ(1, hypergraphs[2].hg.initialNumNodes());
  ASSERT_EQ(0, hypergraphs[2].hg.initialNumEdges());
  ASSERT_EQ(0, hypergraphs[2].hg.initialNumPins());
  ASSERT_EQ(0, hypergraphs[2].hg.maxEdgeSize());

  this->verifyPins(hypergraphs[2].hg, { }, { });
}

TYPED_TEST(APartitionedHypergraph, ExtractBlockZeroWithCommunityInformation) {
  this->hypergraph.setCommunityID(0, 0);
  this->hypergraph.setCommunityID(1, 1);
  this->hypergraph.setCommunityID(2, 0);
  this->hypergraph.setCommunityID(3, 2);
  this->hypergraph.setCommunityID(4, 3);
  this->hypergraph.setCommunityID(5, 4);
  this->hypergraph.setCommunityID(6, 5);

  auto extracted_hg = this->partitioned_hypergraph.extract(0, nullptr, true, true);
  auto& hg = extracted_hg.hg;
  auto& hn_mapping = extracted_hg.hn_mapping;

  auto map_from_original_to_extracted_hg = [&](const HypernodeID hn) {
    return hn_mapping[hn];
  };

  ASSERT_EQ(0, hg.communityID(map_from_original_to_extracted_hg(0)));
  ASSERT_EQ(1, hg.communityID(map_from_original_to_extracted_hg(1)));
  ASSERT_EQ(0, hg.communityID(map_from_original_to_extracted_hg(2)));
}

TYPED_TEST(APartitionedHypergraph, ExtractBlockOneWithCommunityInformation) {
  this->hypergraph.setCommunityID(0, 0);
  this->hypergraph.setCommunityID(1, 1);
  this->hypergraph.setCommunityID(2, 0);
  this->hypergraph.setCommunityID(3, 2);
  this->hypergraph.setCommunityID(4, 3);
  this->hypergraph.setCommunityID(5, 4);
  this->hypergraph.setCommunityID(6, 5);

  auto extracted_hg = this->partitioned_hypergraph.extract(1, nullptr, true, true);
  auto& hg = extracted_hg.hg;
  auto& hn_mapping = extracted_hg.hn_mapping;

  auto map_from_original_to_extracted_hg = [&](const HypernodeID hn) {
    return hn_mapping[hn];
  };

  ASSERT_EQ(2, hg.communityID(map_from_original_to_extracted_hg(3)));
  ASSERT_EQ(3, hg.communityID(map_from_original_to_extracted_hg(4)));
}

TYPED_TEST(APartitionedHypergraph, ExtractBlockTwoWithCommunityInformation) {
  this->hypergraph.setCommunityID(0, 0);
  this->hypergraph.setCommunityID(1, 1);
  this->hypergraph.setCommunityID(2, 0);
  this->hypergraph.setCommunityID(3, 2);
  this->hypergraph.setCommunityID(4, 3);
  this->hypergraph.setCommunityID(5, 4);
  this->hypergraph.setCommunityID(6, 5);

  auto extracted_hg = this->partitioned_hypergraph.extract(2, nullptr, true, true);
  auto& hg = extracted_hg.hg;
  auto& hn_mapping = extracted_hg.hn_mapping;

  auto map_from_original_to_extracted_hg = [&](const HypernodeID hn) {
    return hn_mapping[hn];
  };

  ASSERT_EQ(4, hg.communityID(map_from_original_to_extracted_hg(5)));
  ASSERT_EQ(5, hg.communityID(map_from_original_to_extracted_hg(6)));
}


TYPED_TEST(APartitionedHypergraph, ComputesPartInfoCorrectIfNodePartsAreSetOnly) {
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

TYPED_TEST(APartitionedHypergraph, SetPinCountsInPartCorrectIfNodePartsAreSetOnly) {
  this->partitioned_hypergraph.resetPartition();
  this->partitioned_hypergraph.setOnlyNodePart(0, 0);
  this->partitioned_hypergraph.setOnlyNodePart(1, 0);
  this->partitioned_hypergraph.setOnlyNodePart(2, 0);
  this->partitioned_hypergraph.setOnlyNodePart(3, 1);
  this->partitioned_hypergraph.setOnlyNodePart(4, 1);
  this->partitioned_hypergraph.setOnlyNodePart(5, 2);
  this->partitioned_hypergraph.setOnlyNodePart(6, 2);
  this->partitioned_hypergraph.initializePartition();

  this->verifyPartitionPinCounts(0, { 2, 0, 0 });
  this->verifyPartitionPinCounts(1, { 2, 2, 0 });
  this->verifyPartitionPinCounts(2, { 0, 2, 1 });
  this->verifyPartitionPinCounts(3, { 1, 0, 2 });
}

TYPED_TEST(APartitionedHypergraph, ComputesConnectivitySetCorrectIfNodePartsAreSetOnly) {
  this->partitioned_hypergraph.resetPartition();
  this->partitioned_hypergraph.setOnlyNodePart(0, 0);
  this->partitioned_hypergraph.setOnlyNodePart(1, 0);
  this->partitioned_hypergraph.setOnlyNodePart(2, 0);
  this->partitioned_hypergraph.setOnlyNodePart(3, 1);
  this->partitioned_hypergraph.setOnlyNodePart(4, 1);
  this->partitioned_hypergraph.setOnlyNodePart(5, 2);
  this->partitioned_hypergraph.setOnlyNodePart(6, 2);
  this->partitioned_hypergraph.initializePartition();

  this->verifyConnectivitySet(0, { 0 });
  this->verifyConnectivitySet(1, { 0, 1 });
  this->verifyConnectivitySet(2, { 1, 2 });
  this->verifyConnectivitySet(3, { 0, 2 });
}

TYPED_TEST(APartitionedHypergraph, ComputesBorderNodesCorrectIfNodePartsAreSetOnly) {
  this->partitioned_hypergraph.resetPartition();
  this->partitioned_hypergraph.setOnlyNodePart(0, 0);
  this->partitioned_hypergraph.setOnlyNodePart(1, 0);
  this->partitioned_hypergraph.setOnlyNodePart(2, 0);
  this->partitioned_hypergraph.setOnlyNodePart(3, 1);
  this->partitioned_hypergraph.setOnlyNodePart(4, 1);
  this->partitioned_hypergraph.setOnlyNodePart(5, 2);
  this->partitioned_hypergraph.setOnlyNodePart(6, 2);
  this->partitioned_hypergraph.initializePartition();

  ASSERT_EQ(1, this->partitioned_hypergraph.numIncidentCutHyperedges(0));
  ASSERT_EQ(1, this->partitioned_hypergraph.numIncidentCutHyperedges(1));
  ASSERT_EQ(1, this->partitioned_hypergraph.numIncidentCutHyperedges(2));
  ASSERT_EQ(2, this->partitioned_hypergraph.numIncidentCutHyperedges(3));
  ASSERT_EQ(2, this->partitioned_hypergraph.numIncidentCutHyperedges(4));
  ASSERT_EQ(1, this->partitioned_hypergraph.numIncidentCutHyperedges(5));
  ASSERT_EQ(2, this->partitioned_hypergraph.numIncidentCutHyperedges(6));

  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(0));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(1));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(2));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(3));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(4));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(5));
  ASSERT_TRUE(this->partitioned_hypergraph.isBorderNode(6));
}

}  // namespace ds
}  // namespace mt_kahypar
