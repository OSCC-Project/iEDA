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


#include "gmock/gmock.h"

#include "tests/datastructures/hypergraph_fixtures.h"
#include "mt-kahypar/definitions.h"
#include "mt-kahypar/datastructures/static_graph.h"
#include "mt-kahypar/datastructures/static_graph_factory.h"

using ::testing::Test;

namespace mt_kahypar {
namespace ds {

using AStaticGraph = HypergraphFixture<StaticGraph, true>;

TEST_F(AStaticGraph, HasCorrectStats) {
  ASSERT_EQ(7,  hypergraph.initialNumNodes());
  ASSERT_EQ(12,  hypergraph.initialNumEdges());
  ASSERT_EQ(12, hypergraph.initialNumPins());
  ASSERT_EQ(12, hypergraph.initialTotalVertexDegree());
  ASSERT_EQ(7,  hypergraph.totalWeight());
  ASSERT_EQ(2,  hypergraph.maxEdgeSize());
}

TEST_F(AStaticGraph, HasCorrectInitialNodeIterator) {
  HypernodeID expected_hn = 0;
  for ( const HypernodeID& hn : hypergraph.nodes() ) {
    ASSERT_EQ(expected_hn++, hn);
  }
  ASSERT_EQ(7, expected_hn);
}

TEST_F(AStaticGraph, HasCorrectNodeIteratorIfVerticesAreDisabled) {
  hypergraph.removeDegreeZeroHypernode(0);
  const std::vector<HypernodeID> expected_iter =
    { 1, 2, 3, 4, 5, 6 };
  HypernodeID pos = 0;
  for ( const HypernodeID& hn : hypergraph.nodes() ) {
    ASSERT_EQ(expected_iter[pos++], hn);
  }
  ASSERT_EQ(expected_iter.size(), pos);
}

TEST_F(AStaticGraph, HasCorrectInitialEdgeIterator) {
  // Note that each hyperedge is represented as two edges
  HyperedgeID expected_he = 0;
  for ( const HyperedgeID& he : hypergraph.edges() ) {
    ASSERT_EQ(expected_he++, he);
  }
  ASSERT_EQ(12, expected_he);
}

TEST_F(AStaticGraph, IteratesParallelOverAllNodes) {
  std::vector<uint8_t> visited(7, false);
  hypergraph.doParallelForAllNodes([&](const HypernodeID hn) {
      visited[hn] = true;
    });

  for ( size_t i = 0; i < visited.size(); ++i ) {
    ASSERT_TRUE(visited[i]) << i;
  }
}

TEST_F(AStaticGraph, VerifiesIncidentNets1) {
  verifyIncidentNets(0, { });
}

TEST_F(AStaticGraph, VerifiesIncidentNets2) {
  verifyIncidentNets(1, { 0, 1 });
}

TEST_F(AStaticGraph, VerifiesIncidentNets3) {
  verifyIncidentNets(2, { 2, 3 });
}

TEST_F(AStaticGraph, VerifiesIncidentNets4) {
  verifyIncidentNets(6, { 10, 11 });
}

TEST_F(AStaticGraph, VerifiesPinsOfHyperedges) {
  verifyPins({ 0, 1, 3, 6, 7, 9 },
    { {1, 2}, {1, 4}, {2, 3}, {4, 5}, {4, 6}, {5, 6} });
}

TEST_F(AStaticGraph, VerifiesVertexWeights) {
  for ( const HypernodeID& hn : hypergraph.nodes() ) {
    ASSERT_EQ(1, hypergraph.nodeWeight(hn));
  }
}

TEST_F(AStaticGraph, ModifiesNodeWeight) {
  hypergraph.setNodeWeight(0, 2);
  hypergraph.setNodeWeight(6, 2);
  ASSERT_EQ(2, hypergraph.nodeWeight(0));
  ASSERT_EQ(2, hypergraph.nodeWeight(6));
    hypergraph.computeAndSetTotalNodeWeight(parallel_tag_t());
  ASSERT_EQ(9, hypergraph.totalWeight());
}

TEST_F(AStaticGraph, VerifiesVertexDegrees) {
  ASSERT_EQ(0, hypergraph.nodeDegree(0));
  ASSERT_EQ(2, hypergraph.nodeDegree(1));
  ASSERT_EQ(2, hypergraph.nodeDegree(2));
  ASSERT_EQ(1, hypergraph.nodeDegree(3));
  ASSERT_EQ(3, hypergraph.nodeDegree(4));
  ASSERT_EQ(2, hypergraph.nodeDegree(5));
  ASSERT_EQ(2, hypergraph.nodeDegree(6));
}

TEST_F(AStaticGraph, VerifiesEdgeIDs) {
  ASSERT_EQ(0, hypergraph.uniqueEdgeID(0));
  ASSERT_EQ(2, hypergraph.uniqueEdgeID(1));
  ASSERT_EQ(0, hypergraph.uniqueEdgeID(2));
  ASSERT_EQ(1, hypergraph.uniqueEdgeID(3));
  ASSERT_EQ(1, hypergraph.uniqueEdgeID(4));
  ASSERT_EQ(2, hypergraph.uniqueEdgeID(5));
  ASSERT_EQ(3, hypergraph.uniqueEdgeID(6));
  ASSERT_EQ(4, hypergraph.uniqueEdgeID(7));
  ASSERT_EQ(3, hypergraph.uniqueEdgeID(8));
  ASSERT_EQ(5, hypergraph.uniqueEdgeID(9));
  ASSERT_EQ(4, hypergraph.uniqueEdgeID(10));
  ASSERT_EQ(5, hypergraph.uniqueEdgeID(11));
}

TEST_F(AStaticGraph, RemovesVertices) {
  hypergraph.removeDegreeZeroHypernode(0);
  ASSERT_EQ(1, hypergraph.numRemovedHypernodes());
}

TEST_F(AStaticGraph, VerifiesEdgeWeights) {
  for ( const HyperedgeID& he : hypergraph.edges() ) {
    ASSERT_EQ(1, hypergraph.edgeWeight(he));
  }
}

TEST_F(AStaticGraph, ModifiesEdgeWeight) {
  hypergraph.setEdgeWeight(0, 2);
  hypergraph.setEdgeWeight(2, 2);
  ASSERT_EQ(2, hypergraph.edgeWeight(0));
  ASSERT_EQ(2, hypergraph.edgeWeight(2));
}

TEST_F(AStaticGraph, VerifiesEdgeSizes) {
  for ( const HyperedgeID& he : hypergraph.edges() ) {
    ASSERT_EQ(2, hypergraph.edgeSize(he));
  }
}

TEST_F(AStaticGraph, SetsCommunityIDsForEachVertex) {
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

TEST_F(AStaticGraph, ComparesStatsIfCopiedParallel) {
  StaticGraph copy_hg = hypergraph.copy(parallel_tag_t());
  ASSERT_EQ(hypergraph.initialNumNodes(), copy_hg.initialNumNodes());
  ASSERT_EQ(hypergraph.initialNumEdges(), copy_hg.initialNumEdges());
  ASSERT_EQ(hypergraph.initialNumPins(), copy_hg.initialNumPins());
  ASSERT_EQ(hypergraph.initialTotalVertexDegree(), copy_hg.initialTotalVertexDegree());
  ASSERT_EQ(hypergraph.totalWeight(), copy_hg.totalWeight());
  ASSERT_EQ(hypergraph.maxEdgeSize(), copy_hg.maxEdgeSize());
}

TEST_F(AStaticGraph, ComparesStatsIfCopiedSequential) {
  StaticGraph copy_hg = hypergraph.copy();
  ASSERT_EQ(hypergraph.initialNumNodes(), copy_hg.initialNumNodes());
  ASSERT_EQ(hypergraph.initialNumEdges(), copy_hg.initialNumEdges());
  ASSERT_EQ(hypergraph.initialNumPins(), copy_hg.initialNumPins());
  ASSERT_EQ(hypergraph.initialTotalVertexDegree(), copy_hg.initialTotalVertexDegree());
  ASSERT_EQ(hypergraph.totalWeight(), copy_hg.totalWeight());
  ASSERT_EQ(hypergraph.maxEdgeSize(), copy_hg.maxEdgeSize());
}

TEST_F(AStaticGraph, ComparesIncidentNetsIfCopiedParallel) {
  StaticGraph copy_hg = hypergraph.copy(parallel_tag_t());
  verifyIncidentNets(copy_hg, 0, { });
  verifyIncidentNets(copy_hg, 1, { 0, 1 });
  verifyIncidentNets(copy_hg, 2, { 2, 3 });
  verifyIncidentNets(copy_hg, 3, { 4 });
  verifyIncidentNets(copy_hg, 4, { 5, 6, 7 });
  verifyIncidentNets(copy_hg, 5, { 8, 9 });
  verifyIncidentNets(copy_hg, 6, { 10, 11 });
}

TEST_F(AStaticGraph, ComparesIncidentNetsIfCopiedSequential) {
  StaticGraph copy_hg = hypergraph.copy();
  verifyIncidentNets(copy_hg, 0, { });
  verifyIncidentNets(copy_hg, 1, { 0, 1 });
  verifyIncidentNets(copy_hg, 2, { 2, 3 });
  verifyIncidentNets(copy_hg, 3, { 4 });
  verifyIncidentNets(copy_hg, 4, { 5, 6, 7 });
  verifyIncidentNets(copy_hg, 5, { 8, 9 });
  verifyIncidentNets(copy_hg, 6, { 10, 11 });
}

TEST_F(AStaticGraph, ComparesPinsOfHyperedgesIfCopiedParallel) {
  StaticGraph copy_hg = hypergraph.copy(parallel_tag_t());
  verifyPins(copy_hg, { 0, 1, 3, 6, 7, 9 },
    { {1, 2}, {1, 4}, {2, 3}, {4, 5}, {4, 6}, {5, 6} });
}

TEST_F(AStaticGraph, ComparesPinsOfHyperedgesIfCopiedSequential) {
  StaticGraph copy_hg = hypergraph.copy();
  verifyPins(copy_hg, { 0, 1, 3, 6, 7, 9 },
    { {1, 2}, {1, 4}, {2, 3}, {4, 5}, {4, 6}, {5, 6} });
}

TEST_F(AStaticGraph, ComparesCommunityIdsIfCopiedParallel) {
  assignCommunityIds();
  StaticGraph copy_hg = hypergraph.copy(parallel_tag_t());
  ASSERT_EQ(hypergraph.communityID(0), copy_hg.communityID(0));
  ASSERT_EQ(hypergraph.communityID(1), copy_hg.communityID(1));
  ASSERT_EQ(hypergraph.communityID(2), copy_hg.communityID(2));
  ASSERT_EQ(hypergraph.communityID(3), copy_hg.communityID(3));
  ASSERT_EQ(hypergraph.communityID(4), copy_hg.communityID(4));
  ASSERT_EQ(hypergraph.communityID(5), copy_hg.communityID(5));
  ASSERT_EQ(hypergraph.communityID(6), copy_hg.communityID(6));
}

TEST_F(AStaticGraph, ComparesCommunityIdsIfCopiedSequential) {
  assignCommunityIds();
  StaticGraph copy_hg = hypergraph.copy();
  ASSERT_EQ(hypergraph.communityID(0), copy_hg.communityID(0));
  ASSERT_EQ(hypergraph.communityID(1), copy_hg.communityID(1));
  ASSERT_EQ(hypergraph.communityID(2), copy_hg.communityID(2));
  ASSERT_EQ(hypergraph.communityID(3), copy_hg.communityID(3));
  ASSERT_EQ(hypergraph.communityID(4), copy_hg.communityID(4));
  ASSERT_EQ(hypergraph.communityID(5), copy_hg.communityID(5));
  ASSERT_EQ(hypergraph.communityID(6), copy_hg.communityID(6));
}

TEST_F(AStaticGraph, ContractsCommunities1) {
  parallel::scalable_vector<HypernodeID> c_mapping = {1, 4, 1, 1, 5, 4, 5};
  StaticGraph c_graph = hypergraph.contract(c_mapping);

  // Verify Mapping
  ASSERT_EQ(0, c_mapping[0]);
  ASSERT_EQ(1, c_mapping[1]);
  ASSERT_EQ(0, c_mapping[2]);
  ASSERT_EQ(0, c_mapping[3]);
  ASSERT_EQ(2, c_mapping[4]);
  ASSERT_EQ(1, c_mapping[5]);
  ASSERT_EQ(2, c_mapping[6]);

  // Verify Stats
  ASSERT_EQ(3, c_graph.initialNumNodes());
  ASSERT_EQ(4, c_graph.initialNumEdges());
  ASSERT_EQ(7, c_graph.totalWeight());

  // Verify Vertex Weights
  ASSERT_EQ(3, c_graph.nodeWeight(0));
  ASSERT_EQ(2, c_graph.nodeWeight(1));
  ASSERT_EQ(2, c_graph.nodeWeight(2));

  // Verify Edge Weights
  ASSERT_EQ(1, c_graph.edgeWeight(0));
  ASSERT_EQ(3, c_graph.edgeWeight(2));

  // Verify Edge IDs
  ASSERT_EQ(0, c_graph.uniqueEdgeID(0));
  ASSERT_EQ(0, c_graph.uniqueEdgeID(1));
  ASSERT_EQ(1, c_graph.uniqueEdgeID(2));
  ASSERT_EQ(1, c_graph.uniqueEdgeID(3));

  // Verify Graph Structure - note that each edge has two IDs
  verifyIncidentNets(c_graph, 0, { 0 });
  verifyIncidentNets(c_graph, 1, { 1, 2 });
  verifyIncidentNets(c_graph, 2, { 3 });
  verifyPins(c_graph, { 0 }, { {0, 1} });
  verifyPins(c_graph, { 1 }, { {0, 1} });
  verifyPins(c_graph, { 2 }, { {1, 2} });
  verifyPins(c_graph, { 3 }, { {1, 2} });
}

TEST_F(AStaticGraph, ContractsCommunities2) {
  parallel::scalable_vector<HypernodeID> c_mapping = {0, 1, 2, 2, 2, 2, 2};
  StaticGraph c_graph = hypergraph.contract(c_mapping);

  // Verify Mapping
  ASSERT_EQ(0, c_mapping[0]);
  ASSERT_EQ(1, c_mapping[1]);
  ASSERT_EQ(2, c_mapping[2]);
  ASSERT_EQ(2, c_mapping[3]);
  ASSERT_EQ(2, c_mapping[4]);
  ASSERT_EQ(2, c_mapping[5]);
  ASSERT_EQ(2, c_mapping[6]);

  // Verify Stats
  ASSERT_EQ(3, c_graph.initialNumNodes());
  ASSERT_EQ(2, c_graph.initialNumEdges());
  ASSERT_EQ(7, c_graph.totalWeight());

  // Verify Vertex Weights
  ASSERT_EQ(1, c_graph.nodeWeight(0));
  ASSERT_EQ(1, c_graph.nodeWeight(1));
  ASSERT_EQ(5, c_graph.nodeWeight(2));

  // Verify Edge Weights
  ASSERT_EQ(2, c_graph.edgeWeight(0));

  // Verify Edge IDs
  ASSERT_EQ(0, c_graph.uniqueEdgeID(0));
  ASSERT_EQ(0, c_graph.uniqueEdgeID(1));

  // Verify Graph Structure - note that each edge has two IDs
  verifyIncidentNets(c_graph, 0, {});
  verifyIncidentNets(c_graph, 1, { 0 });
  verifyIncidentNets(c_graph, 2, { 1 });
  verifyPins(c_graph, { 0 }, { {1, 2} });
  verifyPins(c_graph, { 1 }, { {1, 2} });
}

}
} // namespace mt_kahypar