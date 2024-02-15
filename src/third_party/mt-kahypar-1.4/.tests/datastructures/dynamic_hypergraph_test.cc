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

#include "gmock/gmock.h"

#include <atomic>

#include "mt-kahypar/definitions.h"
#include "tests/datastructures/hypergraph_fixtures.h"
#include "mt-kahypar/datastructures/dynamic_hypergraph.h"
#include "mt-kahypar/datastructures/dynamic_hypergraph_factory.h"
#include "mt-kahypar/utils/randomize.h"

namespace mt_kahypar {
namespace ds {

using ADynamicHypergraph = HypergraphFixture<DynamicHypergraph>;

template<typename F, typename K>
void executeParallel(const F& f1, const K& f2) {
  std::atomic<size_t> cnt(0);
  tbb::parallel_invoke([&] {
    ++cnt;
    while ( cnt < 2 ) { }
    f1();
  }, [&] {
    ++cnt;
    while ( cnt < 2 ) { }
    f2();
  });
}

TEST_F(ADynamicHypergraph, HasCorrectStats) {
  ASSERT_EQ(7,  hypergraph.initialNumNodes());
  ASSERT_EQ(4,  hypergraph.initialNumEdges());
  ASSERT_EQ(12, hypergraph.initialNumPins());
  ASSERT_EQ(12, hypergraph.initialTotalVertexDegree());
  ASSERT_EQ(7,  hypergraph.totalWeight());
  ASSERT_EQ(4,  hypergraph.maxEdgeSize());
}

TEST_F(ADynamicHypergraph, HasCorrectInitialNodeIterator) {
  HypernodeID expected_hn = 0;
  for ( const HypernodeID& hn : hypergraph.nodes() ) {
    ASSERT_EQ(expected_hn++, hn);
  }
  ASSERT_EQ(7, expected_hn);
}

TEST_F(ADynamicHypergraph, HasCorrectNodeIteratorIfVerticesAreDisabled) {
  hypergraph.disableHypernode(0);
  hypergraph.disableHypernode(5);
  const std::vector<HypernodeID> expected_iter =
    { 1, 2, 3, 4, 6 };
  HypernodeID pos = 0;
  for ( const HypernodeID& hn : hypergraph.nodes() ) {
    ASSERT_EQ(expected_iter[pos++], hn);
  }
  ASSERT_EQ(expected_iter.size(), pos);
}

TEST_F(ADynamicHypergraph, HasCorrectInitialEdgeIterator) {
  HyperedgeID expected_he = 0;
  for ( const HyperedgeID& he : hypergraph.edges() ) {
    ASSERT_EQ(expected_he++, he);
  }
  ASSERT_EQ(4, expected_he);
}

TEST_F(ADynamicHypergraph, HasCorrectEdgeIteratorIfVerticesAreDisabled) {
  hypergraph.disableHyperedge(0);
  hypergraph.disableHyperedge(2);
  const std::vector<HyperedgeID> expected_iter = { 1, 3 };
  HypernodeID pos = 0;
  for ( const HyperedgeID& he : hypergraph.edges() ) {
    ASSERT_EQ(expected_iter[pos++], he);
  }
  ASSERT_EQ(expected_iter.size(), pos);
}

TEST_F(ADynamicHypergraph, IteratesParallelOverAllNodes) {
  std::vector<uint8_t> visited(7, false);
  hypergraph.doParallelForAllNodes([&](const HypernodeID hn) {
      visited[hn] = true;
    });

  for ( size_t i = 0; i < visited.size(); ++i ) {
    ASSERT_TRUE(visited[i]) << i;
  }
}

TEST_F(ADynamicHypergraph, IteratesParallelOverAllEdges) {
  std::vector<uint8_t> visited(4, false);
  hypergraph.doParallelForAllEdges([&](const HyperedgeID he) {
      visited[he] = true;
    });

  for ( size_t i = 0; i < visited.size(); ++i ) {
    ASSERT_TRUE(visited[i]) << i;
  }
}

TEST_F(ADynamicHypergraph, VerifiesIncidentNets1) {
  verifyIncidentNets(0, { 0, 1 });
}

TEST_F(ADynamicHypergraph, VerifiesIncidentNets2) {
  verifyIncidentNets(2, { 0, 3 });
}

TEST_F(ADynamicHypergraph, VerifiesIncidentNets3) {
  verifyIncidentNets(3, { 1, 2 });
}

TEST_F(ADynamicHypergraph, VerifiesIncidentNets4) {
  verifyIncidentNets(6, { 2, 3 });
}

TEST_F(ADynamicHypergraph, VerifiesPinsOfHyperedges) {
  verifyPins({ 0, 1, 2, 3 },
    { {0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6} });
}

TEST_F(ADynamicHypergraph, VerifiesVertexWeights) {
  for ( const HypernodeID& hn : hypergraph.nodes() ) {
    ASSERT_EQ(1, hypergraph.nodeWeight(hn));
  }
}

TEST_F(ADynamicHypergraph, ModifiesNodeWeight) {
  hypergraph.setNodeWeight(0, 2);
  hypergraph.setNodeWeight(6, 2);
  ASSERT_EQ(2, hypergraph.nodeWeight(0));
  ASSERT_EQ(2, hypergraph.nodeWeight(6));
  hypergraph.updateTotalWeight(parallel_tag_t());
  ASSERT_EQ(9, hypergraph.totalWeight());
}


TEST_F(ADynamicHypergraph, VerifiesVertexDegrees) {
  ASSERT_EQ(2, hypergraph.nodeDegree(0));
  ASSERT_EQ(1, hypergraph.nodeDegree(1));
  ASSERT_EQ(2, hypergraph.nodeDegree(2));
  ASSERT_EQ(2, hypergraph.nodeDegree(3));
  ASSERT_EQ(2, hypergraph.nodeDegree(4));
  ASSERT_EQ(1, hypergraph.nodeDegree(5));
  ASSERT_EQ(2, hypergraph.nodeDegree(6));
}

TEST_F(ADynamicHypergraph, RemovesVertices) {
  hypergraph.removeHypernode(0);
  hypergraph.removeHypernode(5);
  ASSERT_EQ(2, hypergraph.numRemovedHypernodes());
}

TEST_F(ADynamicHypergraph, VerifiesEdgeWeights) {
  for ( const HyperedgeID& he : hypergraph.edges() ) {
    ASSERT_EQ(1, hypergraph.edgeWeight(he));
  }
}

TEST_F(ADynamicHypergraph, ModifiesEdgeWeight) {
  hypergraph.setEdgeWeight(0, 2);
  hypergraph.setEdgeWeight(2, 2);
  ASSERT_EQ(2, hypergraph.edgeWeight(0));
  ASSERT_EQ(2, hypergraph.edgeWeight(2));
}

TEST_F(ADynamicHypergraph, VerifiesEdgeSizes) {
  ASSERT_EQ(2, hypergraph.edgeSize(0));
  ASSERT_EQ(4, hypergraph.edgeSize(1));
  ASSERT_EQ(3, hypergraph.edgeSize(2));
  ASSERT_EQ(3, hypergraph.edgeSize(3));
}

TEST_F(ADynamicHypergraph, SetsCommunityIDsForEachVertex) {
  hypergraph.setCommunityID(0, 1);
  hypergraph.setCommunityID(1, 1);
  hypergraph.setCommunityID(2, 1);
  hypergraph.setCommunityID(3, 2);
  hypergraph.setCommunityID(4, 2);
  hypergraph.setCommunityID(5, 3);
  hypergraph.setCommunityID(6, 3);

  ASSERT_EQ(1, hypergraph.communityID(0));
  ASSERT_EQ(1, hypergraph.communityID(1));
  ASSERT_EQ(1, hypergraph.communityID(2));
  ASSERT_EQ(2, hypergraph.communityID(3));
  ASSERT_EQ(2, hypergraph.communityID(4));
  ASSERT_EQ(3, hypergraph.communityID(5));
  ASSERT_EQ(3, hypergraph.communityID(6));
}


TEST_F(ADynamicHypergraph, RemovesAHyperedgeFromTheHypergraph1) {
  hypergraph.removeEdge(0);
  verifyIncidentNets(0, { 1 });
  verifyIncidentNets(2, { 3 });
  for ( const HyperedgeID& he : hypergraph.edges() ) {
    ASSERT_NE(0, he);
  }
}

TEST_F(ADynamicHypergraph, RemovesAHyperedgeFromTheHypergraph2) {
  hypergraph.removeEdge(1);
  verifyIncidentNets(0, { 0 });
  verifyIncidentNets(1, { });
  verifyIncidentNets(3, { 2 });
  verifyIncidentNets(4, { 2 });
  for ( const HyperedgeID& he : hypergraph.edges() ) {
    ASSERT_NE(1, he);
  }
}

TEST_F(ADynamicHypergraph, RemovesAHyperedgeFromTheHypergraph3) {
  hypergraph.removeEdge(2);
  verifyIncidentNets(3, { 1 });
  verifyIncidentNets(4, { 1 });
  verifyIncidentNets(6, { 3 });
  for ( const HyperedgeID& he : hypergraph.edges() ) {
    ASSERT_NE(2, he);
  }
}

TEST_F(ADynamicHypergraph, RemovesAHyperedgeFromTheHypergraph4) {
  hypergraph.removeEdge(3);
  verifyIncidentNets(2, { 0 });
  verifyIncidentNets(5, { });
  verifyIncidentNets(6, { 2 });
  for ( const HyperedgeID& he : hypergraph.edges() ) {
    ASSERT_NE(3, he);
  }
}

TEST_F(ADynamicHypergraph, ComparesStatsIfCopiedParallel) {
  DynamicHypergraph copy_hg = hypergraph.copy(parallel_tag_t());
  ASSERT_EQ(hypergraph.initialNumNodes(), copy_hg.initialNumNodes());
  ASSERT_EQ(hypergraph.initialNumEdges(), copy_hg.initialNumEdges());
  ASSERT_EQ(hypergraph.initialNumPins(), copy_hg.initialNumPins());
  ASSERT_EQ(hypergraph.initialTotalVertexDegree(), copy_hg.initialTotalVertexDegree());
  ASSERT_EQ(hypergraph.totalWeight(), copy_hg.totalWeight());
  ASSERT_EQ(hypergraph.maxEdgeSize(), copy_hg.maxEdgeSize());
}

TEST_F(ADynamicHypergraph, ComparesStatsIfCopiedSequential) {
  DynamicHypergraph copy_hg = hypergraph.copy();
  ASSERT_EQ(hypergraph.initialNumNodes(), copy_hg.initialNumNodes());
  ASSERT_EQ(hypergraph.initialNumEdges(), copy_hg.initialNumEdges());
  ASSERT_EQ(hypergraph.initialNumPins(), copy_hg.initialNumPins());
  ASSERT_EQ(hypergraph.initialTotalVertexDegree(), copy_hg.initialTotalVertexDegree());
  ASSERT_EQ(hypergraph.totalWeight(), copy_hg.totalWeight());
  ASSERT_EQ(hypergraph.maxEdgeSize(), copy_hg.maxEdgeSize());
}

TEST_F(ADynamicHypergraph, ComparesIncidentNetsIfCopiedParallel) {
  DynamicHypergraph copy_hg = hypergraph.copy(parallel_tag_t());
  verifyIncidentNets(copy_hg, 0, { 0, 1 });
  verifyIncidentNets(copy_hg, 1, { 1 });
  verifyIncidentNets(copy_hg, 2, { 0, 3 });
  verifyIncidentNets(copy_hg, 3, { 1, 2 });
  verifyIncidentNets(copy_hg, 4, { 1, 2 });
  verifyIncidentNets(copy_hg, 5, { 3 });
  verifyIncidentNets(copy_hg, 6, { 2, 3 });
}

TEST_F(ADynamicHypergraph, ComparesIncidentNetsIfCopiedSequential) {
  DynamicHypergraph copy_hg = hypergraph.copy();
  verifyIncidentNets(copy_hg, 0, { 0, 1 });
  verifyIncidentNets(copy_hg, 1, { 1 });
  verifyIncidentNets(copy_hg, 2, { 0, 3 });
  verifyIncidentNets(copy_hg, 3, { 1, 2 });
  verifyIncidentNets(copy_hg, 4, { 1, 2 });
  verifyIncidentNets(copy_hg, 5, { 3 });
  verifyIncidentNets(copy_hg, 6, { 2, 3 });
}

TEST_F(ADynamicHypergraph, ComparesPinsOfHyperedgesIfCopiedParallel) {
  DynamicHypergraph copy_hg = hypergraph.copy(parallel_tag_t());
  verifyPins(copy_hg, { 0, 1, 2, 3 },
    { {0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6} });
}

TEST_F(ADynamicHypergraph, ComparesPinsOfHyperedgesIfCopiedSequential) {
  DynamicHypergraph copy_hg = hypergraph.copy();
  verifyPins(copy_hg, { 0, 1, 2, 3 },
    { {0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6} });
}

TEST_F(ADynamicHypergraph, ComparesCommunityIdsIfCopiedParallel) {
  assignCommunityIds();
  DynamicHypergraph copy_hg = hypergraph.copy(parallel_tag_t());
  ASSERT_EQ(hypergraph.communityID(0), copy_hg.communityID(0));
  ASSERT_EQ(hypergraph.communityID(1), copy_hg.communityID(1));
  ASSERT_EQ(hypergraph.communityID(2), copy_hg.communityID(2));
  ASSERT_EQ(hypergraph.communityID(3), copy_hg.communityID(3));
  ASSERT_EQ(hypergraph.communityID(4), copy_hg.communityID(4));
  ASSERT_EQ(hypergraph.communityID(5), copy_hg.communityID(5));
  ASSERT_EQ(hypergraph.communityID(6), copy_hg.communityID(6));
}

TEST_F(ADynamicHypergraph, ComparesCommunityIdsIfCopiedSequential) {
  assignCommunityIds();
  DynamicHypergraph copy_hg = hypergraph.copy();
  ASSERT_EQ(hypergraph.communityID(0), copy_hg.communityID(0));
  ASSERT_EQ(hypergraph.communityID(1), copy_hg.communityID(1));
  ASSERT_EQ(hypergraph.communityID(2), copy_hg.communityID(2));
  ASSERT_EQ(hypergraph.communityID(3), copy_hg.communityID(3));
  ASSERT_EQ(hypergraph.communityID(4), copy_hg.communityID(4));
  ASSERT_EQ(hypergraph.communityID(5), copy_hg.communityID(5));
  ASSERT_EQ(hypergraph.communityID(6), copy_hg.communityID(6));
}

TEST_F(ADynamicHypergraph, RegistersAContraction1) {
  ASSERT_TRUE(hypergraph.registerContraction(1, 0));
  ASSERT_EQ(1, hypergraph.contractionTree(0));
  ASSERT_EQ(1, hypergraph.contractionTree(1));
  ASSERT_EQ(0, hypergraph.pendingContractions(0));
  ASSERT_EQ(1, hypergraph.pendingContractions(1));
}

TEST_F(ADynamicHypergraph, RegistersAContraction2) {
  ASSERT_TRUE(hypergraph.registerContraction(4, 3));
  ASSERT_EQ(4, hypergraph.contractionTree(3));
  ASSERT_EQ(4, hypergraph.contractionTree(4));
  ASSERT_EQ(0, hypergraph.pendingContractions(3));
  ASSERT_EQ(1, hypergraph.pendingContractions(4));
}

TEST_F(ADynamicHypergraph, RegistersAContraction3) {
  ASSERT_TRUE(hypergraph.registerContraction(4, 3));
  ASSERT_TRUE(hypergraph.registerContraction(4, 2));
  ASSERT_EQ(4, hypergraph.contractionTree(2));
  ASSERT_EQ(4, hypergraph.contractionTree(3));
  ASSERT_EQ(4, hypergraph.contractionTree(4));
  ASSERT_EQ(0, hypergraph.pendingContractions(2));
  ASSERT_EQ(0, hypergraph.pendingContractions(3));
  ASSERT_EQ(2, hypergraph.pendingContractions(4));
}

TEST_F(ADynamicHypergraph, RegistersAContraction4) {
  ASSERT_TRUE(hypergraph.registerContraction(4, 3));
  ASSERT_TRUE(hypergraph.registerContraction(4, 2));
  ASSERT_TRUE(hypergraph.registerContraction(6, 4));
  ASSERT_EQ(4, hypergraph.contractionTree(2));
  ASSERT_EQ(4, hypergraph.contractionTree(3));
  ASSERT_EQ(6, hypergraph.contractionTree(4));
  ASSERT_EQ(6, hypergraph.contractionTree(6));
  ASSERT_EQ(0, hypergraph.pendingContractions(2));
  ASSERT_EQ(0, hypergraph.pendingContractions(3));
  ASSERT_EQ(2, hypergraph.pendingContractions(4));
  ASSERT_EQ(1, hypergraph.pendingContractions(6));
}

TEST_F(ADynamicHypergraph, RegistersAContraction5) {
  ASSERT_TRUE(hypergraph.registerContraction(6, 4));
  ASSERT_TRUE(hypergraph.registerContraction(4, 3));
  ASSERT_TRUE(hypergraph.registerContraction(4, 2));
  ASSERT_EQ(6, hypergraph.contractionTree(2));
  ASSERT_EQ(6, hypergraph.contractionTree(3));
  ASSERT_EQ(6, hypergraph.contractionTree(4));
  ASSERT_EQ(6, hypergraph.contractionTree(6));
  ASSERT_EQ(0, hypergraph.pendingContractions(2));
  ASSERT_EQ(0, hypergraph.pendingContractions(3));
  ASSERT_EQ(0, hypergraph.pendingContractions(4));
  ASSERT_EQ(3, hypergraph.pendingContractions(6));
}

TEST_F(ADynamicHypergraph, RegistersAContraction6) {
  ASSERT_TRUE(hypergraph.registerContraction(4, 3));
  ASSERT_TRUE(hypergraph.registerContraction(6, 4));
  ASSERT_TRUE(hypergraph.registerContraction(4, 2));
  ASSERT_EQ(4, hypergraph.contractionTree(2));
  ASSERT_EQ(4, hypergraph.contractionTree(3));
  ASSERT_EQ(6, hypergraph.contractionTree(4));
  ASSERT_EQ(6, hypergraph.contractionTree(6));
  ASSERT_EQ(0, hypergraph.pendingContractions(2));
  ASSERT_EQ(0, hypergraph.pendingContractions(3));
  ASSERT_EQ(2, hypergraph.pendingContractions(4));
  ASSERT_EQ(1, hypergraph.pendingContractions(6));
}

TEST_F(ADynamicHypergraph, RegistersAContraction7) {
  ASSERT_TRUE(hypergraph.registerContraction(1, 0));
  ASSERT_TRUE(hypergraph.registerContraction(2, 1));
  ASSERT_TRUE(hypergraph.registerContraction(3, 2));
  ASSERT_EQ(1, hypergraph.contractionTree(0));
  ASSERT_EQ(2, hypergraph.contractionTree(1));
  ASSERT_EQ(3, hypergraph.contractionTree(2));
  ASSERT_EQ(3, hypergraph.contractionTree(3));
  ASSERT_EQ(0, hypergraph.pendingContractions(0));
  ASSERT_EQ(1, hypergraph.pendingContractions(1));
  ASSERT_EQ(1, hypergraph.pendingContractions(2));
  ASSERT_EQ(1, hypergraph.pendingContractions(3));
}

TEST_F(ADynamicHypergraph, RegistersAContractionThatInducesACycle1) {
  ASSERT_TRUE(hypergraph.registerContraction(1, 0));
  ASSERT_FALSE(hypergraph.registerContraction(0, 1));
}

TEST_F(ADynamicHypergraph, RegistersAContractionThatInducesACycle2) {
  ASSERT_TRUE(hypergraph.registerContraction(1, 0));
  ASSERT_TRUE(hypergraph.registerContraction(2, 1));
  hypergraph.decrementPendingContractions(1);
  ASSERT_FALSE(hypergraph.registerContraction(0, 2));
}

TEST_F(ADynamicHypergraph, RegistersAContractionThatInducesACycle3) {
  ASSERT_TRUE(hypergraph.registerContraction(1, 0));
  ASSERT_TRUE(hypergraph.registerContraction(2, 1));
  ASSERT_TRUE(hypergraph.registerContraction(3, 2));
  hypergraph.decrementPendingContractions(1);
  hypergraph.decrementPendingContractions(2);
  ASSERT_FALSE(hypergraph.registerContraction(0, 3));
}

TEST_F(ADynamicHypergraph, RegistersAContractionThatInducesACycle4) {
  ASSERT_TRUE(hypergraph.registerContraction(4, 3));
  ASSERT_TRUE(hypergraph.registerContraction(4, 2));
  ASSERT_TRUE(hypergraph.registerContraction(6, 4));
  ASSERT_FALSE(hypergraph.registerContraction(2, 6));
}

TEST_F(ADynamicHypergraph, RegistersAContractionThatInducesACycle5) {
  ASSERT_TRUE(hypergraph.registerContraction(4, 3));
  ASSERT_TRUE(hypergraph.registerContraction(4, 2));
  ASSERT_TRUE(hypergraph.registerContraction(6, 4));
  ASSERT_TRUE(hypergraph.registerContraction(5, 6));
  ASSERT_FALSE(hypergraph.registerContraction(4, 5));
}

TEST_F(ADynamicHypergraph, RegisterContractionsInParallel1) {
  executeParallel([&] {
    ASSERT_TRUE(hypergraph.registerContraction(1, 0));
  }, [&] {
    ASSERT_TRUE(hypergraph.registerContraction(2, 1));
  });

  ASSERT_TRUE(
    // In case (0,1) is executed before (1,2)
    ( hypergraph.contractionTree(0) == 1 &&
      hypergraph.contractionTree(1) == 2 &&
      hypergraph.contractionTree(2) == 2 &&
      hypergraph.pendingContractions(0) == 0 &&
      hypergraph.pendingContractions(1) == 1 &&
      hypergraph.pendingContractions(2) == 1 ) ||
    // In case (1,2) is executed before (0,1)
    ( hypergraph.contractionTree(0) == 2 &&
      hypergraph.contractionTree(1) == 2 &&
      hypergraph.contractionTree(2) == 2 &&
      hypergraph.pendingContractions(0) == 0 &&
      hypergraph.pendingContractions(1) == 0 &&
      hypergraph.pendingContractions(2) == 2 )
  ) << V(hypergraph.contractionTree(0)) << " "
    << V(hypergraph.contractionTree(1)) << " "
    << V(hypergraph.contractionTree(2)) << " "
    << V(hypergraph.contractionTree(3));
}

TEST_F(ADynamicHypergraph, RegisterContractionsInParallel2) {
  executeParallel([&] {
    ASSERT_TRUE(hypergraph.registerContraction(2, 0));
  }, [&] {
    ASSERT_TRUE(hypergraph.registerContraction(2, 1));
  });

  ASSERT_EQ(2, hypergraph.contractionTree(0));
  ASSERT_EQ(2, hypergraph.contractionTree(1));
  ASSERT_EQ(2, hypergraph.contractionTree(2));
  ASSERT_EQ(0, hypergraph.pendingContractions(0));
  ASSERT_EQ(0, hypergraph.pendingContractions(1));
  ASSERT_EQ(2, hypergraph.pendingContractions(2));
}

TEST_F(ADynamicHypergraph, RegisterContractionsInParallel3) {
  executeParallel([&] {
    ASSERT_TRUE(hypergraph.registerContraction(2, 0));
  }, [&] {
    ASSERT_TRUE(hypergraph.registerContraction(3, 2));
    ASSERT_TRUE(hypergraph.registerContraction(2, 1));
  });

  ASSERT_TRUE(
    // In case (0,2) is executed before (2,3)
    ( hypergraph.contractionTree(0) == 2 &&
      hypergraph.contractionTree(1) == 2 &&
      hypergraph.contractionTree(2) == 3 &&
      hypergraph.contractionTree(3) == 3 &&
      hypergraph.pendingContractions(0) == 0 &&
      hypergraph.pendingContractions(1) == 0 &&
      hypergraph.pendingContractions(2) == 2 &&
      hypergraph.pendingContractions(3) == 1 ) ||
    // In case (2,3) is executed before (0,2)
    ( hypergraph.contractionTree(0) == 3 &&
      hypergraph.contractionTree(1) == 3 &&
      hypergraph.contractionTree(2) == 3 &&
      hypergraph.contractionTree(3) == 3 &&
      hypergraph.pendingContractions(0) == 0 &&
      hypergraph.pendingContractions(1) == 0 &&
      hypergraph.pendingContractions(2) == 0 &&
      hypergraph.pendingContractions(3) == 3  )
  ) << V(hypergraph.contractionTree(0)) << " "
    << V(hypergraph.contractionTree(1)) << " "
    << V(hypergraph.contractionTree(2)) << " "
    << V(hypergraph.contractionTree(3));
}

TEST_F(ADynamicHypergraph, RegisterContractionsInParallel4) {
  executeParallel([&] {
    ASSERT_TRUE(hypergraph.registerContraction(2, 0)); // (0)
    ASSERT_TRUE(hypergraph.registerContraction(4, 3)); // (1)
  }, [&] {
    ASSERT_TRUE(hypergraph.registerContraction(3, 2)); // (2)
    ASSERT_TRUE(hypergraph.registerContraction(2, 1)); // (3)
  });

  ASSERT_TRUE(
    // Execution order 0, 1, 2, 3
    ( hypergraph.contractionTree(0) == 2 &&
      hypergraph.contractionTree(1) == 2 &&
      hypergraph.contractionTree(2) == 4 &&
      hypergraph.contractionTree(3) == 4 &&
      hypergraph.contractionTree(4) == 4 &&
      hypergraph.pendingContractions(0) == 0 &&
      hypergraph.pendingContractions(1) == 0 &&
      hypergraph.pendingContractions(2) == 2 &&
      hypergraph.pendingContractions(3) == 0 &&
      hypergraph.pendingContractions(4) == 2) ||
    // Execution order 0, 2, 1, 3
    ( hypergraph.contractionTree(0) == 2 &&
      hypergraph.contractionTree(1) == 4 &&
      hypergraph.contractionTree(2) == 3 &&
      hypergraph.contractionTree(3) == 4 &&
      hypergraph.contractionTree(4) == 4 &&
      hypergraph.pendingContractions(0) == 0 &&
      hypergraph.pendingContractions(1) == 0 &&
      hypergraph.pendingContractions(2) == 1 &&
      hypergraph.pendingContractions(3) == 1 &&
      hypergraph.pendingContractions(4) == 2) ||
    // Execution order 0, 2, 3, 1
    ( hypergraph.contractionTree(0) == 2 &&
      hypergraph.contractionTree(1) == 2 &&
      hypergraph.contractionTree(2) == 3 &&
      hypergraph.contractionTree(3) == 4 &&
      hypergraph.contractionTree(4) == 4 &&
      hypergraph.pendingContractions(0) == 0 &&
      hypergraph.pendingContractions(1) == 0 &&
      hypergraph.pendingContractions(2) == 2 &&
      hypergraph.pendingContractions(3) == 1 &&
      hypergraph.pendingContractions(4) == 1) ||
    // Execution order 2, 0, 1, 3 or 2, 0, 3, 1 or 2, 3, 0, 1
    ( hypergraph.contractionTree(0) == 3 &&
      hypergraph.contractionTree(1) == 3 &&
      hypergraph.contractionTree(2) == 3 &&
      hypergraph.contractionTree(3) == 4 &&
      hypergraph.contractionTree(4) == 4 &&
      hypergraph.pendingContractions(0) == 0 &&
      hypergraph.pendingContractions(1) == 0 &&
      hypergraph.pendingContractions(2) == 0 &&
      hypergraph.pendingContractions(3) == 3 &&
      hypergraph.pendingContractions(4) == 1)
  ) << V(hypergraph.contractionTree(0)) << " "
    << V(hypergraph.contractionTree(1)) << " "
    << V(hypergraph.contractionTree(2)) << " "
    << V(hypergraph.contractionTree(3)) << " "
    << V(hypergraph.contractionTree(4));
}

TEST_F(ADynamicHypergraph, RegisterContractionsThatInducesACycleInParallel1) {
  bool succeded_1 = false;
  bool succeded_2 = false;
  executeParallel([&] {
    succeded_1 = hypergraph.registerContraction(0, 1);
  }, [&] {
    succeded_2 = hypergraph.registerContraction(1, 0);
  });

  ASSERT_TRUE((succeded_1 && !succeded_2) || (!succeded_1 && succeded_2));
  if ( succeded_1 ) {
    ASSERT_EQ(0, hypergraph.contractionTree(0));
    ASSERT_EQ(0, hypergraph.contractionTree(1));
    ASSERT_EQ(1, hypergraph.pendingContractions(0));
    ASSERT_EQ(0, hypergraph.pendingContractions(1));
  } else {
    ASSERT_EQ(1, hypergraph.contractionTree(0));
    ASSERT_EQ(1, hypergraph.contractionTree(1));
    ASSERT_EQ(0, hypergraph.pendingContractions(0));
    ASSERT_EQ(1, hypergraph.pendingContractions(1));
  }
}

TEST_F(ADynamicHypergraph, RegisterContractionsThatInducesACycleInParallel2) {
  ASSERT_TRUE(hypergraph.registerContraction(1, 0));
  ASSERT_TRUE(hypergraph.registerContraction(3, 2));

  bool succeded_1 = false;
  bool succeded_2 = false;
  executeParallel([&] {
    succeded_1 = hypergraph.registerContraction(0, 3);
  }, [&] {
    succeded_2 = hypergraph.registerContraction(2, 1);
  });

  ASSERT_TRUE((succeded_1 && !succeded_2) || (!succeded_1 && succeded_2));
  if ( succeded_1 ) {
    ASSERT_EQ(1, hypergraph.contractionTree(0));
    ASSERT_EQ(1, hypergraph.contractionTree(1));
    ASSERT_EQ(3, hypergraph.contractionTree(2));
    ASSERT_EQ(1, hypergraph.contractionTree(3));
    ASSERT_EQ(0, hypergraph.pendingContractions(0));
    ASSERT_EQ(2, hypergraph.pendingContractions(1));
    ASSERT_EQ(0, hypergraph.pendingContractions(2));
    ASSERT_EQ(1, hypergraph.pendingContractions(3));
  } else {
    ASSERT_EQ(1, hypergraph.contractionTree(0));
    ASSERT_EQ(3, hypergraph.contractionTree(1));
    ASSERT_EQ(3, hypergraph.contractionTree(2));
    ASSERT_EQ(3, hypergraph.contractionTree(3));
    ASSERT_EQ(0, hypergraph.pendingContractions(0));
    ASSERT_EQ(1, hypergraph.pendingContractions(1));
    ASSERT_EQ(0, hypergraph.pendingContractions(2));
    ASSERT_EQ(2, hypergraph.pendingContractions(3));
  }
}

TEST_F(ADynamicHypergraph, RegisterContractionsThatInducesACycleInParallel3) {
  ASSERT_TRUE(hypergraph.registerContraction(1, 0));
  ASSERT_TRUE(hypergraph.registerContraction(4, 3));

  bool succeded_1 = false;
  bool succeded_2 = false;
  bool succeded_3 = false;
  executeParallel([&] {
    succeded_1 = hypergraph.registerContraction(2, 1);
  }, [&] {
    succeded_2 = hypergraph.registerContraction(3, 2);
    succeded_3 = hypergraph.registerContraction(0, 4);
  });

  const size_t num_succeded = succeded_1 + succeded_2 + succeded_3;
  ASSERT_EQ(2, num_succeded);
  if ( succeded_1 ) {
  ASSERT_TRUE(
    // In case (1,2) is executed before (2,3)
    ( hypergraph.contractionTree(0) == 1 &&
      hypergraph.contractionTree(1) == 2 &&
      hypergraph.contractionTree(2) == 4 &&
      hypergraph.contractionTree(3) == 4 &&
      hypergraph.contractionTree(4) == 4 &&
      hypergraph.pendingContractions(0) == 0 &&
      hypergraph.pendingContractions(1) == 1 &&
      hypergraph.pendingContractions(2) == 1 &&
      hypergraph.pendingContractions(3) == 0 &&
      hypergraph.pendingContractions(4) == 2 ) ||
    // In case (2,3) is executed before (1,2)
    ( hypergraph.contractionTree(0) == 1 &&
      hypergraph.contractionTree(1) == 4 &&
      hypergraph.contractionTree(2) == 4 &&
      hypergraph.contractionTree(3) == 4 &&
      hypergraph.contractionTree(4) == 4 &&
      hypergraph.pendingContractions(0) == 0 &&
      hypergraph.pendingContractions(1) == 1 &&
      hypergraph.pendingContractions(2) == 0 &&
      hypergraph.pendingContractions(3) == 0 &&
      hypergraph.pendingContractions(4) == 3 )
  ) << V(hypergraph.contractionTree(0)) << " "
    << V(hypergraph.contractionTree(1)) << " "
    << V(hypergraph.contractionTree(2)) << " "
    << V(hypergraph.contractionTree(3)) << " "
    << V(hypergraph.contractionTree(4));
  } else {
    ASSERT_EQ(1, hypergraph.contractionTree(0));
    ASSERT_EQ(1, hypergraph.contractionTree(1));
    ASSERT_EQ(4, hypergraph.contractionTree(2));
    ASSERT_EQ(4, hypergraph.contractionTree(3));
    ASSERT_EQ(1, hypergraph.contractionTree(4));
    ASSERT_EQ(0, hypergraph.pendingContractions(0));
    ASSERT_EQ(2, hypergraph.pendingContractions(1));
    ASSERT_EQ(0, hypergraph.pendingContractions(2));
    ASSERT_EQ(0, hypergraph.pendingContractions(3));
    ASSERT_EQ(2, hypergraph.pendingContractions(4));
  }
}


using MementoVector = parallel::scalable_vector<Memento>;

void assertEqual(MementoVector actual, MementoVector expected) {
  auto compare = [&](const Memento& lhs, const Memento& rhs) {
    return lhs.u < rhs.u || (lhs.u == rhs.u && lhs.v < rhs.v);
  };
  std::sort(actual.begin(), actual.end(), compare);
  std::sort(expected.begin(), expected.end(), compare);

  ASSERT_EQ(actual.size(), expected.size());
  for ( size_t i = 0; i < actual.size(); ++i ) {
    ASSERT_EQ(actual[i].u, expected[i].u);
    ASSERT_EQ(actual[i].v, expected[i].v);
  }
}

bool assertEqualToOneAlternative(MementoVector actual,
                                 MementoVector alternative_1,
                                 MementoVector alternative_2) {
  auto compare = [&](const Memento& lhs, const Memento& rhs) {
    return lhs.u < rhs.u || (lhs.u == rhs.u && lhs.v < rhs.v);
  };
  std::sort(actual.begin(), actual.end(), compare);
  std::sort(alternative_1.begin(), alternative_1.end(), compare);
  std::sort(alternative_2.begin(), alternative_2.end(), compare);

  bool equal_to_alternative_1 = actual.size() == alternative_1.size();
  bool equal_to_alternative_2 = actual.size() == alternative_2.size();
  if ( equal_to_alternative_1 ) {
    for ( size_t i = 0; i < actual.size(); ++i ) {
      equal_to_alternative_1 = equal_to_alternative_1 &&
        actual[i].u == alternative_1[i].u && actual[i].v == alternative_1[i].v;
    }
  }

  if ( equal_to_alternative_2 ) {
    for ( size_t i = 0; i < actual.size(); ++i ) {
      equal_to_alternative_2 = equal_to_alternative_2 &&
        actual[i].u == alternative_2[i].u && actual[i].v == alternative_2[i].v;
    }
  }

  const size_t num_equal = UI64(equal_to_alternative_1) + UI64(equal_to_alternative_2);
  if ( num_equal == 1 ) {
    return true;
  } else {
    return false;
  }
}

TEST_F(ADynamicHypergraph, PerformsAContraction1) {
  ASSERT_TRUE(hypergraph.registerContraction(1, 0));
  hypergraph.contract(0);

  ASSERT_FALSE(hypergraph.nodeIsEnabled(0));
  ASSERT_EQ(2, hypergraph.nodeWeight(1));
  ASSERT_EQ(0, hypergraph.pendingContractions(1));

  verifyIncidentNets(1, {0, 1});
  verifyPins({ 0, 1, 2, 3 },
    { {1, 2}, {1, 3, 4}, {3, 4, 6}, {2, 5, 6} });
}

TEST_F(ADynamicHypergraph, PerformsAContraction2) {
  ASSERT_TRUE(hypergraph.registerContraction(1, 0));
  ASSERT_TRUE(hypergraph.registerContraction(2, 1));
  hypergraph.contract(1);
  hypergraph.contract(0);

  ASSERT_FALSE(hypergraph.nodeIsEnabled(0));
  ASSERT_FALSE(hypergraph.nodeIsEnabled(1));
  ASSERT_EQ(3, hypergraph.nodeWeight(2));
  ASSERT_EQ(0, hypergraph.pendingContractions(2));

  verifyIncidentNets(2, {0, 1, 3});
  verifyPins({ 0, 1, 2, 3 },
    { {2}, {2, 3, 4}, {3, 4, 6}, {2, 5, 6} });
}

TEST_F(ADynamicHypergraph, PerformsAContraction3) {
  ASSERT_TRUE(hypergraph.registerContraction(2, 0));
  ASSERT_TRUE(hypergraph.registerContraction(2, 1));
  ASSERT_TRUE(hypergraph.registerContraction(3, 2));

  hypergraph.contract(1);
  ASSERT_FALSE(hypergraph.nodeIsEnabled(1));
  ASSERT_EQ(2, hypergraph.nodeWeight(2));
  ASSERT_EQ(1, hypergraph.pendingContractions(2));

  hypergraph.contract(0);
  ASSERT_FALSE(hypergraph.nodeIsEnabled(0));
  ASSERT_FALSE(hypergraph.nodeIsEnabled(2));
  ASSERT_EQ(4, hypergraph.nodeWeight(3));
  ASSERT_EQ(0, hypergraph.pendingContractions(3));

  verifyIncidentNets(3, {0, 1, 2, 3});
  verifyPins({ 0, 1, 2, 3 },
    { {3}, {3, 4}, {3, 4, 6}, {3, 5, 6} });
}

TEST_F(ADynamicHypergraph, PerformsAContraction4) {
  ASSERT_TRUE(hypergraph.registerContraction(2, 0));
  ASSERT_TRUE(hypergraph.registerContraction(2, 1));
  ASSERT_TRUE(hypergraph.registerContraction(3, 2));
  ASSERT_TRUE(hypergraph.registerContraction(3, 4));

  hypergraph.contract(1);
  hypergraph.contract(4);
  ASSERT_FALSE(hypergraph.nodeIsEnabled(1));
  ASSERT_FALSE(hypergraph.nodeIsEnabled(4));
  ASSERT_EQ(2, hypergraph.nodeWeight(2));
  ASSERT_EQ(2, hypergraph.nodeWeight(3));
  ASSERT_EQ(1, hypergraph.pendingContractions(2));
  ASSERT_EQ(1, hypergraph.pendingContractions(3));

  hypergraph.contract(0);
  ASSERT_FALSE(hypergraph.nodeIsEnabled(0));
  ASSERT_FALSE(hypergraph.nodeIsEnabled(2));
  ASSERT_EQ(5, hypergraph.nodeWeight(3));
  ASSERT_EQ(0, hypergraph.pendingContractions(3));

  verifyIncidentNets(3, {0, 1, 2, 3});
  verifyPins({ 0, 1, 2, 3 },
    { {3}, {3}, {3, 6}, {3, 5, 6} });
}

TEST_F(ADynamicHypergraph, PerformsAContraction5) {
  ASSERT_TRUE(hypergraph.registerContraction(2, 0));
  ASSERT_TRUE(hypergraph.registerContraction(2, 1));
  ASSERT_TRUE(hypergraph.registerContraction(3, 2));
  ASSERT_TRUE(hypergraph.registerContraction(3, 4));
  ASSERT_TRUE(hypergraph.registerContraction(6, 3));

  hypergraph.contract(1);
  hypergraph.contract(4);
  hypergraph.contract(0);
  ASSERT_EQ(6, hypergraph.nodeWeight(6));

  verifyIncidentNets(6, {0, 1, 2, 3});
  verifyPins({ 0, 1, 2, 3 },
    { {6}, {6}, {6}, {5, 6} });
}

TEST_F(ADynamicHypergraph, PerformsAContractionWithWeightGreaterThanMaxNodeWeight1) {
  ASSERT_TRUE(hypergraph.registerContraction(1, 0));
  ASSERT_EQ(1, hypergraph.contractionTree(0));
  ASSERT_EQ(1, hypergraph.pendingContractions(1));
  hypergraph.contract(0, 1);
  ASSERT_TRUE(hypergraph.nodeIsEnabled(0));
  ASSERT_TRUE(hypergraph.nodeIsEnabled(1));
  ASSERT_EQ(0, hypergraph.contractionTree(0));
  ASSERT_EQ(0, hypergraph.pendingContractions(1));
}

TEST_F(ADynamicHypergraph, PerformsAContractionWithWeightGreaterThanMaxNodeWeight2) {
  ASSERT_TRUE(hypergraph.registerContraction(1, 0));
  hypergraph.contract(0, 2);

  ASSERT_TRUE(hypergraph.registerContraction(2, 1));
  ASSERT_TRUE(hypergraph.registerContraction(3, 2));
  ASSERT_EQ(2, hypergraph.contractionTree(1));
  ASSERT_EQ(3, hypergraph.contractionTree(2));
  ASSERT_EQ(1, hypergraph.pendingContractions(2));
  ASSERT_EQ(1, hypergraph.pendingContractions(3));
  hypergraph.contract(1, 2);
  ASSERT_EQ(1, hypergraph.contractionTree(1));
  ASSERT_EQ(0, hypergraph.pendingContractions(3));

  verifyIncidentNets(1, {0, 1});
  verifyIncidentNets(3, {0, 1, 2, 3});
  verifyPins({ 0, 1, 2, 3 },
    { {1, 3}, {1, 3, 4}, {3, 4, 6}, {3, 5, 6} });
}

TEST_F(ADynamicHypergraph, PerformAContractionsInParallel1) {
  ASSERT_TRUE(hypergraph.registerContraction(2, 0));
  ASSERT_TRUE(hypergraph.registerContraction(2, 1));
  executeParallel([&] {
    hypergraph.contract(0);
  }, [&] {
    hypergraph.contract(1);
  });
  ASSERT_FALSE(hypergraph.nodeIsEnabled(0));
  ASSERT_FALSE(hypergraph.nodeIsEnabled(1));
  ASSERT_EQ(3, hypergraph.nodeWeight(2));
  ASSERT_EQ(0, hypergraph.pendingContractions(2));

  verifyIncidentNets(2, {0, 1, 3});
  verifyPins({ 0, 1, 2, 3 },
    { {2}, {2, 3, 4}, {3, 4, 6}, {2, 5, 6} });
}

TEST_F(ADynamicHypergraph, PerformAContractionsInParallel2) {
  ASSERT_TRUE(hypergraph.registerContraction(2, 0));
  ASSERT_TRUE(hypergraph.registerContraction(2, 1));
  ASSERT_TRUE(hypergraph.registerContraction(3, 2));
  executeParallel([&] {
    hypergraph.contract(0);
  }, [&] {
    hypergraph.contract(1);
  });
  ASSERT_FALSE(hypergraph.nodeIsEnabled(0));
  ASSERT_FALSE(hypergraph.nodeIsEnabled(1));
  ASSERT_FALSE(hypergraph.nodeIsEnabled(2));
  ASSERT_EQ(4, hypergraph.nodeWeight(3));
  ASSERT_EQ(0, hypergraph.pendingContractions(3));

  verifyIncidentNets(3, {0, 1, 2, 3});
  verifyPins({ 0, 1, 2, 3 },
    { {3}, {3, 4}, {3, 4, 6}, {3, 5, 6} });
}

TEST_F(ADynamicHypergraph, PerformAContractionsInParallel3) {
  ASSERT_TRUE(hypergraph.registerContraction(2, 0));
  ASSERT_TRUE(hypergraph.registerContraction(2, 1));
  ASSERT_TRUE(hypergraph.registerContraction(3, 2));
  ASSERT_TRUE(hypergraph.registerContraction(3, 4));
  ASSERT_TRUE(hypergraph.registerContraction(6, 3));
  std::atomic<size_t> cnt(2);
  executeParallel([&] {
    hypergraph.contract(0);
    ++cnt;
  }, [&] {
    hypergraph.contract(1);
    ++cnt;
    while ( cnt < 2 ) { }
    hypergraph.contract(4);
  });
  ASSERT_FALSE(hypergraph.nodeIsEnabled(0));
  ASSERT_FALSE(hypergraph.nodeIsEnabled(1));
  ASSERT_FALSE(hypergraph.nodeIsEnabled(2));
  ASSERT_FALSE(hypergraph.nodeIsEnabled(3));
  ASSERT_FALSE(hypergraph.nodeIsEnabled(4));
  ASSERT_EQ(6, hypergraph.nodeWeight(6));
  ASSERT_EQ(0, hypergraph.pendingContractions(6));

  verifyIncidentNets(6, {0, 1, 2, 3});
  verifyPins({ 0, 1, 2, 3 },
    { {6}, {6}, {6}, {5, 6} });
}

void verifyBatchUncontractionHierarchy(ContractionTree& tree,
                                       const VersionedBatchVector& versioned_batches,
                                       const size_t batch_size,
                                       const size_t num_versions = 1) {
  tree.finalize(num_versions);
  std::vector<bool> enabled_vertices(tree.num_hypernodes(), false);
  size_t expected_uncontractions = 0;
  for ( const HypernodeID& root : tree.roots() ) {
    enabled_vertices[root] = true;
    expected_uncontractions += tree.subtreeSize(root);
  }

  parallel::scalable_vector<size_t> expected_uncontractions_of_version(num_versions, 0);
  for ( size_t version = 0; version < num_versions; ++version ) {
    for ( HypernodeID hn = 0; hn < tree.num_hypernodes(); ++hn ) {
      tree.doForEachChildOfVersion(hn, version, [&](const HypernodeID) {
        ++expected_uncontractions_of_version[version];
      });
    }
  }

  size_t actual_uncontractions = 0;
  for ( int version = versioned_batches.size() - 1; version >= 0; --version ) {
    size_t actual_uncontractions_of_version = 0;
    const BatchVector& batches = versioned_batches[version];
    for ( int i = batches.size() - 1; i >= 0; --i ) {
      actual_uncontractions_of_version += batches[i].size();
      ASSERT_LE(batches[i].size(), batch_size);
      for ( const Memento& memento : batches[i] ) {
        ASSERT_TRUE(enabled_vertices[memento.u]) << "Memento: (" << memento.u << "," << memento.v << ")";
        ASSERT_FALSE(enabled_vertices[memento.v]) << "Memento: (" << memento.u << "," << memento.v << ")";
      }
      for ( const Memento& memento : batches[i] ) {
        enabled_vertices[memento.v] = true;
      }
    }
    ASSERT_EQ(expected_uncontractions_of_version[version], actual_uncontractions_of_version);
    actual_uncontractions += actual_uncontractions_of_version;
  }
  ASSERT_EQ(expected_uncontractions, actual_uncontractions);
}

TEST_F(ADynamicHypergraph, CreateBatchUncontractionHierarchy1) {
  ContractionTree tree;
  tree.initialize(7);
  tree.setParent(1, 0);
  tree.setParent(2, 0);
  tree.setParent(4, 3);
  tree.setParent(5, 3);
  tree.setParent(6, 1);
  auto versioned_batches = hypergraph.createBatchUncontractionHierarchy(tree.copy(), 2);
  ASSERT_EQ(1, versioned_batches.size());
  verifyBatchUncontractionHierarchy(tree, versioned_batches, 2);
}

TEST_F(ADynamicHypergraph, CreateBatchUncontractionHierarchy2) {
  ContractionTree tree;
  tree.initialize(10);
  tree.setParent(1, 0);
  tree.setParent(2, 1);
  tree.setParent(3, 2);
  tree.setParent(4, 3);
  tree.setParent(5, 4);
  tree.setParent(7, 6);
  tree.setParent(8, 7);
  tree.setParent(9, 8);
  auto versioned_batches = hypergraph.createBatchUncontractionHierarchy(tree.copy(), 3);
  ASSERT_EQ(1, versioned_batches.size());
  verifyBatchUncontractionHierarchy(tree, versioned_batches, 3);
}

TEST_F(ADynamicHypergraph, CreateBatchUncontractionHierarchy3) {
  ContractionTree tree;
  // Complete binary tree
  tree.initialize(15);
  tree.setParent(1, 0);
  tree.setParent(2, 0);
  tree.setParent(3, 1);
  tree.setParent(4, 1);
  tree.setParent(5, 2);
  tree.setParent(6, 2);
  tree.setParent(7, 3);
  tree.setParent(8, 3);
  tree.setParent(9, 4);
  tree.setParent(10, 4);
  tree.setParent(11, 5);
  tree.setParent(12, 5);
  tree.setParent(13, 6);
  tree.setParent(14, 6);
  auto versioned_batches = hypergraph.createBatchUncontractionHierarchy(tree.copy(), 4);
  ASSERT_EQ(1, versioned_batches.size());
  verifyBatchUncontractionHierarchy(tree, versioned_batches, 4);
}

TEST_F(ADynamicHypergraph, CreateBatchUncontractionHierarchy4) {
  ContractionTree tree;
  // 4 chains of size 4
  tree.initialize(16);
  tree.setParent(1, 0);
  tree.setParent(2, 1);
  tree.setParent(3, 2);
  tree.setParent(5, 4);
  tree.setParent(6, 5);
  tree.setParent(7, 6);
  tree.setParent(9, 8);
  tree.setParent(10, 9);
  tree.setParent(11, 10);
  tree.setParent(13, 12);
  tree.setParent(14, 13);
  tree.setParent(15, 14);
  auto versioned_batches = hypergraph.createBatchUncontractionHierarchy(tree.copy(), 4);
  ASSERT_EQ(1, versioned_batches.size());
  for ( size_t i = 0; i < versioned_batches.size(); ++i ) {
    ASSERT_EQ(4, versioned_batches.back()[i].size());
  }
  verifyBatchUncontractionHierarchy(tree, versioned_batches, 4);
}

TEST_F(ADynamicHypergraph, CreateBatchUncontractionHierarchy5) {
  ContractionTree tree;
  // 3 complete binary trees
  tree.initialize(23);
  tree.setParent(1, 0);
  tree.setParent(2, 0);
  tree.setParent(3, 1);
  tree.setParent(4, 1);
  tree.setParent(5, 2);
  tree.setParent(6, 2);
  tree.setParent(8, 7);
  tree.setParent(9, 7);
  tree.setParent(10, 8);
  tree.setParent(11, 8);
  tree.setParent(14, 9);
  tree.setParent(15, 9);
  tree.setParent(17, 16);
  tree.setParent(18, 16);
  tree.setParent(19, 17);
  tree.setParent(20, 17);
  tree.setParent(21, 18);
  tree.setParent(22, 18);
  auto versioned_batches = hypergraph.createBatchUncontractionHierarchy(tree.copy(), 6);
  ASSERT_EQ(1, versioned_batches.size());
  verifyBatchUncontractionHierarchy(tree, versioned_batches, 6);
}


// TODO(heuer): test fails sporadically on CI -> further investigation and fix required
// TEST_F(ADynamicHypergraph, CreateBatchUncontractionHierarchy6) {
//   ContractionTree tree;
//   // Binary Tree where each left child has exactly two childrens
//   tree.initialize(15);
//   tree.setParent(1, 0);
//   tree.setParent(2, 0);
//   tree.setParent(3, 1);
//   tree.setParent(4, 1);
//   tree.setParent(5, 3);
//   tree.setParent(6, 3);
//   tree.setParent(7, 5);
//   tree.setParent(8, 5);
//   tree.setParent(9, 7);
//   tree.setParent(10, 7);
//   tree.setParent(11, 9);
//   tree.setParent(12, 9);
//   tree.setParent(13, 11);
//   tree.setParent(14, 11);
//   auto versioned_batches = hypergraph.createBatchUncontractionHierarchy(tree.copy(), 4);
//   ASSERT_EQ(1, versioned_batches.size());
//   for ( size_t i = 0; i < versioned_batches.size(); ++i ) {
//     ASSERT_EQ(2, versioned_batches.back()[i].size());
//   }
//   verifyBatchUncontractionHierarchy(tree, versioned_batches, 4);
// }

TEST_F(ADynamicHypergraph, CreateBatchUncontractionHierarchyWithDifferentVersions1) {
  ContractionTree tree;
  tree.initialize(7);
  tree.setParent(1, 0, 1);
  tree.setParent(2, 0, 1);
  tree.setParent(4, 3, 0);
  tree.setParent(5, 3, 0);
  tree.setParent(6, 1, 0);
  auto versioned_batches = hypergraph.createBatchUncontractionHierarchy(tree.copy(), 2, 2);
  ASSERT_EQ(2, versioned_batches.size());
  verifyBatchUncontractionHierarchy(tree, versioned_batches, 2, 2);
}

TEST_F(ADynamicHypergraph, CreateBatchUncontractionHierarchyWithDifferentVersions2) {
  ContractionTree tree;
  tree.initialize(5);
  tree.setParent(1, 0, 1);
  tree.setParent(2, 0, 1);
  tree.setParent(3, 1, 0);
  tree.setParent(4, 1, 0);
  auto versioned_batches = hypergraph.createBatchUncontractionHierarchy(tree.copy(), 2, 2);
  ASSERT_EQ(2, versioned_batches.size());
  verifyBatchUncontractionHierarchy(tree, versioned_batches, 2, 2);
}

TEST_F(ADynamicHypergraph, CreateBatchUncontractionHierarchyWithDifferentVersions3) {
  ContractionTree tree;
  tree.initialize(7);
  tree.setParent(1, 0, 2);
  tree.setParent(2, 0, 1);
  tree.setParent(3, 1, 0);
  tree.setParent(4, 1, 1);
  tree.setParent(5, 2, 0);
  tree.setParent(6, 2, 1);
  auto versioned_batches = hypergraph.createBatchUncontractionHierarchy(tree.copy(), 2, 3);
  ASSERT_EQ(3, versioned_batches.size());
  verifyBatchUncontractionHierarchy(tree, versioned_batches, 2, 3);
}

TEST_F(ADynamicHypergraph, CreateBatchUncontractionHierarchyWithDifferentVersions4) {
  ContractionTree tree;
  tree.initialize(6);
  tree.setParent(1, 0, 2);
  tree.setParent(2, 0, 1);
  tree.setParent(4, 3, 0);
  tree.setParent(5, 3, 1);
  auto versioned_batches = hypergraph.createBatchUncontractionHierarchy(tree.copy(), 2, 3);
  ASSERT_EQ(3, versioned_batches.size());
  verifyBatchUncontractionHierarchy(tree, versioned_batches, 2, 3);
}

TEST_F(ADynamicHypergraph, CreateBatchUncontractionHierarchyWithDifferentVersions5) {
  ContractionTree tree;
  tree.initialize(10);
  tree.setParent(1, 0, 4);
  tree.setParent(2, 0, 4);
  tree.setParent(3, 1, 1);
  tree.setParent(4, 2, 2);
  tree.setParent(6, 5, 3);
  tree.setParent(7, 5, 4);
  tree.setParent(8, 6, 0);
  tree.setParent(9, 7, 2);
  auto versioned_batches = hypergraph.createBatchUncontractionHierarchy(tree.copy(), 2, 5);
  ASSERT_EQ(5, versioned_batches.size());
  verifyBatchUncontractionHierarchy(tree, versioned_batches, 2, 5);
}

TEST_F(ADynamicHypergraph, CreateBatchUncontractionHierarchyWithEmptyVersionBatch) {
  ContractionTree tree;
  tree.initialize(6);
  tree.setParent(1, 0, 2);
  tree.setParent(2, 0, 0);
  tree.setParent(4, 3, 0);
  tree.setParent(5, 3, 0);
  auto versioned_batches = hypergraph.createBatchUncontractionHierarchy(tree.copy(), 2, 3);
  ASSERT_EQ(3, versioned_batches.size());
  verifyBatchUncontractionHierarchy(tree, versioned_batches, 2, 3);
}

void verifyEqualityOfDynamicHypergraphs(const DynamicHypergraph& expected_hypergraph,
                                 const DynamicHypergraph& actual_hypergraph) {
  parallel::scalable_vector<HyperedgeID> expected_incident_edges;
  parallel::scalable_vector<HyperedgeID> actual_incident_edges;
  for ( const HypernodeID& hn : expected_hypergraph.nodes() ) {
    ASSERT_TRUE(actual_hypergraph.nodeIsEnabled(hn));
    ASSERT_EQ(expected_hypergraph.nodeWeight(hn), actual_hypergraph.nodeWeight(hn));
    ASSERT_EQ(expected_hypergraph.nodeDegree(hn), actual_hypergraph.nodeDegree(hn));
    for ( const HyperedgeID he : expected_hypergraph.incidentEdges(hn) ) {
      expected_incident_edges.push_back(he);
    }
    for ( const HyperedgeID he : actual_hypergraph.incidentEdges(hn) ) {
      actual_incident_edges.push_back(he);
    }
    std::sort(expected_incident_edges.begin(), expected_incident_edges.end());
    std::sort(actual_incident_edges.begin(), actual_incident_edges.end());
    ASSERT_EQ(expected_incident_edges.size(), actual_incident_edges.size());
    for ( size_t i = 0; i < expected_incident_edges.size(); ++i ) {
      ASSERT_EQ(expected_incident_edges[i], actual_incident_edges[i]);
    }
    expected_incident_edges.clear();
    actual_incident_edges.clear();
  }

  parallel::scalable_vector<HypernodeID> expected_pins;
  parallel::scalable_vector<HypernodeID> actual_pins;
  for ( const HyperedgeID& he : expected_hypergraph.edges() ) {
    for ( const HyperedgeID he : expected_hypergraph.pins(he) ) {
      expected_pins.push_back(he);
    }
    for ( const HyperedgeID he : actual_hypergraph.pins(he) ) {
      actual_pins.push_back(he);
    }
    std::sort(expected_pins.begin(), expected_pins.end());
    std::sort(actual_pins.begin(), actual_pins.end());
    ASSERT_EQ(expected_pins.size(), actual_pins.size());
    for ( size_t i = 0; i < expected_pins.size(); ++i ) {
      ASSERT_EQ(expected_pins[i], actual_pins[i]);
    }
    expected_pins.clear();
    actual_pins.clear();
  }
}

void verifyBatchUncontractions(DynamicHypergraph& hypergraph,
                               const parallel::scalable_vector<Memento>& contractions,
                               const size_t batch_size) {
  DynamicHypergraph expected_hypergraph = hypergraph.copy();

  // Perform contractions
  for ( const Memento& memento : contractions ) {
    hypergraph.registerContraction(memento.u, memento.v);
    hypergraph.contract(memento.v);
  }

  auto versioned_batches = hypergraph.createBatchUncontractionHierarchy(batch_size);

  while ( !versioned_batches.empty() ) {
    BatchVector& batches = versioned_batches.back();
    while ( !batches.empty() ) {
      const parallel::scalable_vector<Memento> batch = batches.back();
      hypergraph.uncontract(batch);
      batches.pop_back();
    }
    versioned_batches.pop_back();
  }

  verifyEqualityOfDynamicHypergraphs(expected_hypergraph, hypergraph);
}

TEST_F(ADynamicHypergraph, PerformsBatchUncontractions1) {
  verifyBatchUncontractions(hypergraph,
    { Memento { 0, 2 }, Memento { 3, 4 }, Memento { 5, 6 } }, 3);
}

TEST_F(ADynamicHypergraph, PerformsBatchUncontractions2) {
  verifyBatchUncontractions(hypergraph,
    { Memento { 1, 0 }, Memento { 2, 1 }, Memento { 3, 2 } }, 3);
}

TEST_F(ADynamicHypergraph, PerformsBatchUncontractions3) {
  verifyBatchUncontractions(hypergraph,
    { Memento { 1, 0 }, Memento { 1, 2 }, Memento { 3, 1 },
      Memento { 4, 6 }, Memento { 4, 5 } }, 3);
}

TEST_F(ADynamicHypergraph, PerformsBatchUncontractions4) {
  verifyBatchUncontractions(hypergraph,
    { Memento { 5, 6 }, Memento { 4, 5 }, Memento { 3, 4 },
      Memento { 2, 3 }, Memento { 1, 2 }, Memento { 0, 1 } }, 3);
}


TEST_F(ADynamicHypergraph, PerformsBatchUncontractions5) {
  verifyBatchUncontractions(hypergraph,
    { Memento { 2, 6 }, Memento { 2, 5 }, Memento { 1, 3 },
      Memento { 1, 4 }, Memento { 0, 1 }, Memento { 0, 2 } }, 2);
}

TEST_F(ADynamicHypergraph, RemovesSinglePinAndParallelNets1) {
  const parallel::scalable_vector<Memento> contractions =
   { Memento { 0, 2 }, Memento { 0, 1 }, Memento { 3, 6 }, Memento { 4, 5 } };

  for ( const Memento& memento : contractions ) {
    hypergraph.registerContraction(memento.u, memento.v);
    hypergraph.contract(memento.v);
  }

  verifyPins( { 0, 1, 2, 3 },
    { { 0 }, { 0, 3, 4 }, { 3, 4 }, { 0, 3, 4 } } );

  using ParallelHyperedge = typename DynamicHypergraph::ParallelHyperedge;
  auto removed_hyperedges = hypergraph.removeSinglePinAndParallelHyperedges();
  std::sort(removed_hyperedges.begin(), removed_hyperedges.end(),
    [&](const ParallelHyperedge& lhs, const ParallelHyperedge& rhs) {
      return lhs.removed_hyperedge < rhs.removed_hyperedge;
    });

  ASSERT_EQ(2, removed_hyperedges.size());
  ASSERT_EQ(0, removed_hyperedges[0].removed_hyperedge);
  ASSERT_EQ(3, removed_hyperedges[1].removed_hyperedge);
  ASSERT_FALSE(hypergraph.edgeIsEnabled(0));
  ASSERT_FALSE(hypergraph.edgeIsEnabled(3));
  ASSERT_EQ(2, hypergraph.edgeWeight(1));
  verifyPins( { 1, 2 },
    { { 0, 3, 4 }, { 3, 4 } } );
  verifyIncidentNets(0, { 1 });
  verifyIncidentNets(3, { 1, 2 });
  verifyIncidentNets(4, { 1, 2 });
}

TEST_F(ADynamicHypergraph, RemovesSinglePinAndParallelNets2) {
  const parallel::scalable_vector<Memento> contractions =
   { Memento { 0, 2 }, Memento { 1, 5 }, Memento { 6, 3 }, Memento { 6, 4 } };

  for ( const Memento& memento : contractions ) {
    hypergraph.registerContraction(memento.u, memento.v);
    hypergraph.contract(memento.v);
  }

  verifyPins( { 0, 1, 2, 3 },
    { { 0 }, { 0, 1, 6 }, { 6 }, { 0, 1, 6 } } );

  using ParallelHyperedge = typename DynamicHypergraph::ParallelHyperedge;
  auto removed_hyperedges = hypergraph.removeSinglePinAndParallelHyperedges();
  std::sort(removed_hyperedges.begin(), removed_hyperedges.end(),
    [&](const ParallelHyperedge& lhs, const ParallelHyperedge& rhs) {
      return lhs.removed_hyperedge < rhs.removed_hyperedge;
    });

  ASSERT_EQ(3, removed_hyperedges.size());
  ASSERT_EQ(0, removed_hyperedges[0].removed_hyperedge);
  ASSERT_EQ(2, removed_hyperedges[1].removed_hyperedge);
  ASSERT_EQ(3, removed_hyperedges[2].removed_hyperedge);
  ASSERT_FALSE(hypergraph.edgeIsEnabled(0));
  ASSERT_FALSE(hypergraph.edgeIsEnabled(2));
  ASSERT_FALSE(hypergraph.edgeIsEnabled(3));
  ASSERT_EQ(2, hypergraph.edgeWeight(1));
  verifyPins( { 1 },
    { { 0, 1, 6 } } );
  verifyIncidentNets(0, { 1 });
  verifyIncidentNets(1, { 1 });
  verifyIncidentNets(6, { 1 });
}

TEST_F(ADynamicHypergraph, RestoreSinglePinAndParallelNets1) {
  const parallel::scalable_vector<Memento> contractions =
   { Memento { 0, 2 }, Memento { 0, 1 }, Memento { 3, 6 }, Memento { 4, 5 } };

  for ( const Memento& memento : contractions ) {
    hypergraph.registerContraction(memento.u, memento.v);
    hypergraph.contract(memento.v);
  }

  auto removed_hyperedges = hypergraph.removeSinglePinAndParallelHyperedges();
  hypergraph.restoreSinglePinAndParallelNets(removed_hyperedges);

  verifyIncidentNets(0, { 0, 1, 3 });
  verifyIncidentNets(3, { 1, 2, 3 });
  verifyIncidentNets(4, { 1, 2, 3 });
  verifyPins( { 0, 1, 2, 3 },
    { { 0 }, { 0, 3, 4 }, { 3, 4 }, { 0, 3, 4 } } );
  ASSERT_EQ(1, hypergraph.edgeWeight(0));
  ASSERT_EQ(1, hypergraph.edgeWeight(1));
  ASSERT_EQ(1, hypergraph.edgeWeight(2));
  ASSERT_EQ(1, hypergraph.edgeWeight(3));
}

TEST_F(ADynamicHypergraph, RestoresSinglePinAndParallelNets2) {
  const parallel::scalable_vector<Memento> contractions =
   { Memento { 0, 2 }, Memento { 1, 5 }, Memento { 6, 3 }, Memento { 6, 4 } };

  for ( const Memento& memento : contractions ) {
    hypergraph.registerContraction(memento.u, memento.v);
    hypergraph.contract(memento.v);
  }

  auto removed_hyperedges = hypergraph.removeSinglePinAndParallelHyperedges();
  hypergraph.restoreSinglePinAndParallelNets(removed_hyperedges);

  verifyIncidentNets(0, { 0, 1, 3 });
  verifyIncidentNets(1, { 1, 3 });
  verifyIncidentNets(6, { 1, 2, 3 });
  verifyPins( { 0, 1, 2, 3 },
    { { 0 }, { 0, 1, 6 }, { 6 }, { 0, 1, 6 } } );
  ASSERT_EQ(1, hypergraph.edgeWeight(0));
  ASSERT_EQ(1, hypergraph.edgeWeight(1));
  ASSERT_EQ(1, hypergraph.edgeWeight(2));
  ASSERT_EQ(1, hypergraph.edgeWeight(3));
}

TEST_F(ADynamicHypergraph, GeneratesACompactifiedHypergraph1) {
  const parallel::scalable_vector<Memento> contractions =
   { Memento { 0, 2 } };

  for ( const Memento& memento : contractions ) {
    hypergraph.registerContraction(memento.u, memento.v);
    hypergraph.contract(memento.v);
  }
  hypergraph.removeSinglePinAndParallelHyperedges();

  auto res = DynamicHypergraphFactory::compactify(hypergraph);
  DynamicHypergraph& compactified_hg = res.first;
  ASSERT_EQ(6, compactified_hg.initialNumNodes());
  ASSERT_EQ(3, compactified_hg.initialNumEdges());
  verifyPins(compactified_hg, {0, 1, 2},
    { {0, 1, 2, 3}, {2, 3, 5}, {0, 4, 5} } );
}

TEST_F(ADynamicHypergraph, GeneratesACompactifiedHypergraph2) {
  const parallel::scalable_vector<Memento> contractions =
   { Memento { 0, 2 }, Memento { 1, 5 }, Memento { 6, 3 }, Memento { 6, 4 } };

  for ( const Memento& memento : contractions ) {
    hypergraph.registerContraction(memento.u, memento.v);
    hypergraph.contract(memento.v);
  }
  hypergraph.removeSinglePinAndParallelHyperedges();

  auto res = DynamicHypergraphFactory::compactify(hypergraph);
  DynamicHypergraph& compactified_hg = res.first;
  ASSERT_EQ(3, compactified_hg.initialNumNodes());
  ASSERT_EQ(1, compactified_hg.initialNumEdges());
  ASSERT_EQ(2, compactified_hg.edgeWeight(0));
  ASSERT_EQ(2, compactified_hg.nodeWeight(0));
  ASSERT_EQ(2, compactified_hg.nodeWeight(1));
  ASSERT_EQ(3, compactified_hg.nodeWeight(2));
  verifyPins(compactified_hg, {0}, { {0, 1, 2} });
}

} // namespace ds
} // namespace mt_kahypar