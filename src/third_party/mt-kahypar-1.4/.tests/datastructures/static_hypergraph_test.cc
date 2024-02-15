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

#include "tests/datastructures/hypergraph_fixtures.h"
#include "mt-kahypar/definitions.h"
#include "mt-kahypar/datastructures/static_hypergraph.h"
#include "mt-kahypar/datastructures/static_hypergraph_factory.h"

using ::testing::Test;

namespace mt_kahypar {
namespace ds {

using AStaticHypergraph = HypergraphFixture<StaticHypergraph>;

TEST_F(AStaticHypergraph, HasCorrectStats) {
  ASSERT_EQ(7,  hypergraph.initialNumNodes());
  ASSERT_EQ(4,  hypergraph.initialNumEdges());
  ASSERT_EQ(12, hypergraph.initialNumPins());
  ASSERT_EQ(12, hypergraph.initialTotalVertexDegree());
  ASSERT_EQ(7,  hypergraph.totalWeight());
  ASSERT_EQ(4,  hypergraph.maxEdgeSize());
}

TEST_F(AStaticHypergraph, HasCorrectInitialNodeIterator) {
  HypernodeID expected_hn = 0;
  for ( const HypernodeID& hn : hypergraph.nodes() ) {
    ASSERT_EQ(expected_hn++, hn);
  }
  ASSERT_EQ(7, expected_hn);
}

TEST_F(AStaticHypergraph, HasCorrectNodeIteratorIfVerticesAreDisabled) {
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

TEST_F(AStaticHypergraph, HasCorrectInitialEdgeIterator) {
  HyperedgeID expected_he = 0;
  for ( const HyperedgeID& he : hypergraph.edges() ) {
    ASSERT_EQ(expected_he++, he);
  }
  ASSERT_EQ(4, expected_he);
}

TEST_F(AStaticHypergraph, HasCorrectEdgeIteratorIfVerticesAreDisabled) {
  hypergraph.disableHyperedge(0);
  hypergraph.disableHyperedge(2);
  const std::vector<HyperedgeID> expected_iter = { 1, 3 };
  HypernodeID pos = 0;
  for ( const HyperedgeID& he : hypergraph.edges() ) {
    ASSERT_EQ(expected_iter[pos++], he);
  }
  ASSERT_EQ(expected_iter.size(), pos);
}

TEST_F(AStaticHypergraph, IteratesParallelOverAllNodes) {
  std::vector<uint8_t> visited(7, false);
  hypergraph.doParallelForAllNodes([&](const HypernodeID hn) {
      visited[hn] = true;
    });

  for ( size_t i = 0; i < visited.size(); ++i ) {
    ASSERT_TRUE(visited[i]) << i;
  }
}

TEST_F(AStaticHypergraph, IteratesParallelOverAllEdges) {
  std::vector<uint8_t> visited(4, false);
  hypergraph.doParallelForAllEdges([&](const HyperedgeID he) {
      visited[he] = true;
    });

  for ( size_t i = 0; i < visited.size(); ++i ) {
    ASSERT_TRUE(visited[i]) << i;
  }
}

TEST_F(AStaticHypergraph, VerifiesIncidentNets1) {
  verifyIncidentNets(0, { 0, 1 });
}

TEST_F(AStaticHypergraph, VerifiesIncidentNets2) {
  verifyIncidentNets(2, { 0, 3 });
}

TEST_F(AStaticHypergraph, VerifiesIncidentNets3) {
  verifyIncidentNets(3, { 1, 2 });
}

TEST_F(AStaticHypergraph, VerifiesIncidentNets4) {
  verifyIncidentNets(6, { 2, 3 });
}

TEST_F(AStaticHypergraph, VerifiesPinsOfHyperedges) {
  verifyPins({ 0, 1, 2, 3 },
    { {0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6} });
}

TEST_F(AStaticHypergraph, VerifiesVertexWeights) {
  for ( const HypernodeID& hn : hypergraph.nodes() ) {
    ASSERT_EQ(1, hypergraph.nodeWeight(hn));
  }
}

TEST_F(AStaticHypergraph, ModifiesNodeWeight) {
  hypergraph.setNodeWeight(0, 2);
  hypergraph.setNodeWeight(6, 2);
  ASSERT_EQ(2, hypergraph.nodeWeight(0));
  ASSERT_EQ(2, hypergraph.nodeWeight(6));
  hypergraph.computeAndSetTotalNodeWeight(parallel_tag_t());
  ASSERT_EQ(9, hypergraph.totalWeight());
}


TEST_F(AStaticHypergraph, VerifiesVertexDegrees) {
  ASSERT_EQ(2, hypergraph.nodeDegree(0));
  ASSERT_EQ(1, hypergraph.nodeDegree(1));
  ASSERT_EQ(2, hypergraph.nodeDegree(2));
  ASSERT_EQ(2, hypergraph.nodeDegree(3));
  ASSERT_EQ(2, hypergraph.nodeDegree(4));
  ASSERT_EQ(1, hypergraph.nodeDegree(5));
  ASSERT_EQ(2, hypergraph.nodeDegree(6));
}

TEST_F(AStaticHypergraph, RemovesVertices) {
  hypergraph.removeHypernode(0);
  hypergraph.removeHypernode(5);
  ASSERT_EQ(2, hypergraph.numRemovedHypernodes());
}

TEST_F(AStaticHypergraph, VerifiesEdgeWeights) {
  for ( const HyperedgeID& he : hypergraph.edges() ) {
    ASSERT_EQ(1, hypergraph.edgeWeight(he));
  }
}

TEST_F(AStaticHypergraph, ModifiesEdgeWeight) {
  hypergraph.setEdgeWeight(0, 2);
  hypergraph.setEdgeWeight(2, 2);
  ASSERT_EQ(2, hypergraph.edgeWeight(0));
  ASSERT_EQ(2, hypergraph.edgeWeight(2));
}

TEST_F(AStaticHypergraph, VerifiesEdgeSizes) {
  ASSERT_EQ(2, hypergraph.edgeSize(0));
  ASSERT_EQ(4, hypergraph.edgeSize(1));
  ASSERT_EQ(3, hypergraph.edgeSize(2));
  ASSERT_EQ(3, hypergraph.edgeSize(3));
}

TEST_F(AStaticHypergraph, SetsCommunityIDsForEachVertex) {
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

TEST_F(AStaticHypergraph, RemovesAHyperedgeFromTheHypergraph1) {
  hypergraph.removeEdge(0);
  verifyIncidentNets(0, { 1 });
  verifyIncidentNets(2, { 3 });
  for ( const HyperedgeID& he : hypergraph.edges() ) {
    ASSERT_NE(0, he);
  }
}

TEST_F(AStaticHypergraph, RemovesAHyperedgeFromTheHypergraph2) {
  hypergraph.removeEdge(1);
  verifyIncidentNets(0, { 0 });
  verifyIncidentNets(1, { });
  verifyIncidentNets(3, { 2 });
  verifyIncidentNets(4, { 2 });
  for ( const HyperedgeID& he : hypergraph.edges() ) {
    ASSERT_NE(1, he);
  }
}

TEST_F(AStaticHypergraph, RemovesAHyperedgeFromTheHypergraph3) {
  hypergraph.removeEdge(2);
  verifyIncidentNets(3, { 1 });
  verifyIncidentNets(4, { 1 });
  verifyIncidentNets(6, { 3 });
  for ( const HyperedgeID& he : hypergraph.edges() ) {
    ASSERT_NE(2, he);
  }
}

TEST_F(AStaticHypergraph, RemovesAHyperedgeFromTheHypergraph4) {
  hypergraph.removeEdge(3);
  verifyIncidentNets(2, { 0 });
  verifyIncidentNets(5, { });
  verifyIncidentNets(6, { 2 });
  for ( const HyperedgeID& he : hypergraph.edges() ) {
    ASSERT_NE(3, he);
  }
}

TEST_F(AStaticHypergraph, ComparesStatsIfCopiedParallel) {
  StaticHypergraph copy_hg = hypergraph.copy(parallel_tag_t());
  ASSERT_EQ(hypergraph.initialNumNodes(), copy_hg.initialNumNodes());
  ASSERT_EQ(hypergraph.initialNumEdges(), copy_hg.initialNumEdges());
  ASSERT_EQ(hypergraph.initialNumPins(), copy_hg.initialNumPins());
  ASSERT_EQ(hypergraph.initialTotalVertexDegree(), copy_hg.initialTotalVertexDegree());
  ASSERT_EQ(hypergraph.totalWeight(), copy_hg.totalWeight());
  ASSERT_EQ(hypergraph.maxEdgeSize(), copy_hg.maxEdgeSize());
}

TEST_F(AStaticHypergraph, ComparesStatsIfCopiedSequential) {
  StaticHypergraph copy_hg = hypergraph.copy();
  ASSERT_EQ(hypergraph.initialNumNodes(), copy_hg.initialNumNodes());
  ASSERT_EQ(hypergraph.initialNumEdges(), copy_hg.initialNumEdges());
  ASSERT_EQ(hypergraph.initialNumPins(), copy_hg.initialNumPins());
  ASSERT_EQ(hypergraph.initialTotalVertexDegree(), copy_hg.initialTotalVertexDegree());
  ASSERT_EQ(hypergraph.totalWeight(), copy_hg.totalWeight());
  ASSERT_EQ(hypergraph.maxEdgeSize(), copy_hg.maxEdgeSize());
}

TEST_F(AStaticHypergraph, ComparesIncidentNetsIfCopiedParallel) {
  StaticHypergraph copy_hg = hypergraph.copy(parallel_tag_t());
  verifyIncidentNets(copy_hg, 0, { 0, 1 });
  verifyIncidentNets(copy_hg, 1, { 1 });
  verifyIncidentNets(copy_hg, 2, { 0, 3 });
  verifyIncidentNets(copy_hg, 3, { 1, 2 });
  verifyIncidentNets(copy_hg, 4, { 1, 2 });
  verifyIncidentNets(copy_hg, 5, { 3 });
  verifyIncidentNets(copy_hg, 6, { 2, 3 });
}

TEST_F(AStaticHypergraph, ComparesIncidentNetsIfCopiedSequential) {
  StaticHypergraph copy_hg = hypergraph.copy();
  verifyIncidentNets(copy_hg, 0, { 0, 1 });
  verifyIncidentNets(copy_hg, 1, { 1 });
  verifyIncidentNets(copy_hg, 2, { 0, 3 });
  verifyIncidentNets(copy_hg, 3, { 1, 2 });
  verifyIncidentNets(copy_hg, 4, { 1, 2 });
  verifyIncidentNets(copy_hg, 5, { 3 });
  verifyIncidentNets(copy_hg, 6, { 2, 3 });
}

TEST_F(AStaticHypergraph, ComparesPinsOfHyperedgesIfCopiedParallel) {
  StaticHypergraph copy_hg = hypergraph.copy(parallel_tag_t());
  verifyPins(copy_hg, { 0, 1, 2, 3 },
    { {0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6} });
}

TEST_F(AStaticHypergraph, ComparesPinsOfHyperedgesIfCopiedSequential) {
  StaticHypergraph copy_hg = hypergraph.copy();
  verifyPins(copy_hg, { 0, 1, 2, 3 },
    { {0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6} });
}

TEST_F(AStaticHypergraph, ComparesCommunityIdsIfCopiedParallel) {
  assignCommunityIds();
  StaticHypergraph copy_hg = hypergraph.copy(parallel_tag_t());
  ASSERT_EQ(hypergraph.communityID(0), copy_hg.communityID(0));
  ASSERT_EQ(hypergraph.communityID(1), copy_hg.communityID(1));
  ASSERT_EQ(hypergraph.communityID(2), copy_hg.communityID(2));
  ASSERT_EQ(hypergraph.communityID(3), copy_hg.communityID(3));
  ASSERT_EQ(hypergraph.communityID(4), copy_hg.communityID(4));
  ASSERT_EQ(hypergraph.communityID(5), copy_hg.communityID(5));
  ASSERT_EQ(hypergraph.communityID(6), copy_hg.communityID(6));
}

TEST_F(AStaticHypergraph, ComparesCommunityIdsIfCopiedSequential) {
  assignCommunityIds();
  StaticHypergraph copy_hg = hypergraph.copy();
  ASSERT_EQ(hypergraph.communityID(0), copy_hg.communityID(0));
  ASSERT_EQ(hypergraph.communityID(1), copy_hg.communityID(1));
  ASSERT_EQ(hypergraph.communityID(2), copy_hg.communityID(2));
  ASSERT_EQ(hypergraph.communityID(3), copy_hg.communityID(3));
  ASSERT_EQ(hypergraph.communityID(4), copy_hg.communityID(4));
  ASSERT_EQ(hypergraph.communityID(5), copy_hg.communityID(5));
  ASSERT_EQ(hypergraph.communityID(6), copy_hg.communityID(6));
}

TEST_F(AStaticHypergraph, ContractsCommunities1) {
  parallel::scalable_vector<HypernodeID> c_mapping = {1, 4, 1, 5, 5, 4, 5};
  StaticHypergraph c_hypergraph = hypergraph.contract(c_mapping);

  // Verify Mapping
  ASSERT_EQ(0, c_mapping[0]);
  ASSERT_EQ(1, c_mapping[1]);
  ASSERT_EQ(0, c_mapping[2]);
  ASSERT_EQ(2, c_mapping[3]);
  ASSERT_EQ(2, c_mapping[4]);
  ASSERT_EQ(1, c_mapping[5]);
  ASSERT_EQ(2, c_mapping[6]);

  // Verify Stats
  ASSERT_EQ(3, c_hypergraph.initialNumNodes());
  ASSERT_EQ(1, c_hypergraph.initialNumEdges());
  ASSERT_EQ(3, c_hypergraph.initialNumPins());
  ASSERT_EQ(7, c_hypergraph.totalWeight());
  ASSERT_EQ(3, c_hypergraph.maxEdgeSize());

  // Verify Vertex Weights
  ASSERT_EQ(2, c_hypergraph.nodeWeight(0));
  ASSERT_EQ(2, c_hypergraph.nodeWeight(1));
  ASSERT_EQ(3, c_hypergraph.nodeWeight(2));

  // Verify Hyperedge Weights
  ASSERT_EQ(2, c_hypergraph.edgeWeight(0));

  // Verify Hypergraph Structure
  verifyIncidentNets(c_hypergraph, 0, { 0 });
  verifyIncidentNets(c_hypergraph, 1, { 0 });
  verifyIncidentNets(c_hypergraph, 2, { 0 });
  verifyPins(c_hypergraph, { 0 }, { {0, 1, 2} });
}

TEST_F(AStaticHypergraph, ContractsCommunities2) {
  parallel::scalable_vector<HypernodeID> c_mapping = {1, 4, 1, 5, 5, 6, 5};
  StaticHypergraph c_hypergraph = hypergraph.contract(c_mapping);

  // Verify Mapping
  ASSERT_EQ(0, c_mapping[0]);
  ASSERT_EQ(1, c_mapping[1]);
  ASSERT_EQ(0, c_mapping[2]);
  ASSERT_EQ(2, c_mapping[3]);
  ASSERT_EQ(2, c_mapping[4]);
  ASSERT_EQ(3, c_mapping[5]);
  ASSERT_EQ(2, c_mapping[6]);

  // Verify Stats
  ASSERT_EQ(4, c_hypergraph.initialNumNodes());
  ASSERT_EQ(2, c_hypergraph.initialNumEdges());
  ASSERT_EQ(6, c_hypergraph.initialNumPins());
  ASSERT_EQ(7, c_hypergraph.totalWeight());
  ASSERT_EQ(3, c_hypergraph.maxEdgeSize());

  // Verify Vertex Weights
  ASSERT_EQ(2, c_hypergraph.nodeWeight(0));
  ASSERT_EQ(1, c_hypergraph.nodeWeight(1));
  ASSERT_EQ(3, c_hypergraph.nodeWeight(2));
  ASSERT_EQ(1, c_hypergraph.nodeWeight(3));

  // Verify Hyperedge Weights
  ASSERT_EQ(1, c_hypergraph.edgeWeight(0));
  ASSERT_EQ(1, c_hypergraph.edgeWeight(1));

  // Verify Hypergraph Structure
  verifyIncidentNets(c_hypergraph, 0, { 0, 1 });
  verifyIncidentNets(c_hypergraph, 1, { 0 });
  verifyIncidentNets(c_hypergraph, 2, { 0, 1 });
  verifyIncidentNets(c_hypergraph, 3, { 1 });
  verifyPins(c_hypergraph, { 0, 1 }, { {0, 1, 2}, {0, 2, 3} });
}

TEST_F(AStaticHypergraph, ContractsCommunities3) {
  parallel::scalable_vector<HypernodeID> c_mapping = {2, 2, 0, 5, 5, 1, 1};
  StaticHypergraph c_hypergraph = hypergraph.contract(c_mapping);

  // Verify Mapping
  ASSERT_EQ(2, c_mapping[0]);
  ASSERT_EQ(2, c_mapping[1]);
  ASSERT_EQ(0, c_mapping[2]);
  ASSERT_EQ(3, c_mapping[3]);
  ASSERT_EQ(3, c_mapping[4]);
  ASSERT_EQ(1, c_mapping[5]);
  ASSERT_EQ(1, c_mapping[6]);

  // Verify Stats
  ASSERT_EQ(4, c_hypergraph.initialNumNodes());
  ASSERT_EQ(4, c_hypergraph.initialNumEdges());
  ASSERT_EQ(8, c_hypergraph.initialNumPins());
  ASSERT_EQ(7, c_hypergraph.totalWeight());
  ASSERT_EQ(2, c_hypergraph.maxEdgeSize());

  // Verify Vertex Weights
  ASSERT_EQ(1, c_hypergraph.nodeWeight(0));
  ASSERT_EQ(2, c_hypergraph.nodeWeight(1));
  ASSERT_EQ(2, c_hypergraph.nodeWeight(2));
  ASSERT_EQ(2, c_hypergraph.nodeWeight(3));

  // Verify Hyperedge Weights
  ASSERT_EQ(1, c_hypergraph.edgeWeight(0));
  ASSERT_EQ(1, c_hypergraph.edgeWeight(1));
  ASSERT_EQ(1, c_hypergraph.edgeWeight(2));
  ASSERT_EQ(1, c_hypergraph.edgeWeight(3));

  // Verify Hypergraph Structure
  verifyIncidentNets(c_hypergraph, 0, { 0, 3 });
  verifyIncidentNets(c_hypergraph, 1, { 2, 3 });
  verifyIncidentNets(c_hypergraph, 2, { 0, 1 });
  verifyIncidentNets(c_hypergraph, 3, { 1, 2 });
  verifyPins(c_hypergraph, { 0, 1, 2, 3 },
    { {0, 2}, {2, 3}, {1, 3}, {0, 1} });
}

TEST_F(AStaticHypergraph, ContractsCommunitiesWithDisabledHypernodes) {
  hypergraph.disableHypernode(0);
  hypergraph.disableHypernode(6);
  hypergraph.computeAndSetTotalNodeWeight(parallel_tag_t());

  parallel::scalable_vector<HypernodeID> c_mapping = {0, 1, 1, 2, 2, 2, 6};
  StaticHypergraph c_hypergraph = hypergraph.contract(c_mapping);

  // Verify Mapping
  ASSERT_EQ(0, c_mapping[1]);
  ASSERT_EQ(0, c_mapping[2]);
  ASSERT_EQ(1, c_mapping[3]);
  ASSERT_EQ(1, c_mapping[4]);
  ASSERT_EQ(1, c_mapping[5]);

  // Verify Stats
  ASSERT_EQ(2, c_hypergraph.initialNumNodes());
  ASSERT_EQ(1, c_hypergraph.initialNumEdges());
  ASSERT_EQ(2, c_hypergraph.initialNumPins());
  ASSERT_EQ(5, c_hypergraph.totalWeight());
  ASSERT_EQ(2, c_hypergraph.maxEdgeSize());

  // Verify Vertex Weights
  ASSERT_EQ(2, c_hypergraph.nodeWeight(0));
  ASSERT_EQ(3, c_hypergraph.nodeWeight(1));

  // Verify Hyperedge Weights
  ASSERT_EQ(2, c_hypergraph.edgeWeight(0));

  // Verify Hypergraph Structure
  verifyIncidentNets(c_hypergraph, 0, { 0 });
  verifyIncidentNets(c_hypergraph, 1, { 0 });
  verifyPins(c_hypergraph, { 0 }, { {0, 1} });
}

TEST_F(AStaticHypergraph, ContractsCommunitiesWithDisabledHyperedges) {
  hypergraph.disableHyperedge(3);

  parallel::scalable_vector<HypernodeID> c_mapping = {0, 0, 0, 1, 1, 2, 3};
  StaticHypergraph c_hypergraph = hypergraph.contract(c_mapping);

  // Verify Mapping
  ASSERT_EQ(0, c_mapping[0]);
  ASSERT_EQ(0, c_mapping[1]);
  ASSERT_EQ(0, c_mapping[2]);
  ASSERT_EQ(1, c_mapping[3]);
  ASSERT_EQ(1, c_mapping[4]);
  ASSERT_EQ(2, c_mapping[5]);
  ASSERT_EQ(3, c_mapping[6]);

  // Verify Stats
  ASSERT_EQ(4, c_hypergraph.initialNumNodes());
  ASSERT_EQ(2, c_hypergraph.initialNumEdges());
  ASSERT_EQ(4, c_hypergraph.initialNumPins());
  ASSERT_EQ(7, c_hypergraph.totalWeight());
  ASSERT_EQ(2, c_hypergraph.maxEdgeSize());

  // Verify Vertex Weights
  ASSERT_EQ(3, c_hypergraph.nodeWeight(0));
  ASSERT_EQ(2, c_hypergraph.nodeWeight(1));
  ASSERT_EQ(1, c_hypergraph.nodeWeight(2));
  ASSERT_EQ(1, c_hypergraph.nodeWeight(3));

  // Verify Hyperedge Weights
  ASSERT_EQ(1, c_hypergraph.edgeWeight(0));
  ASSERT_EQ(1, c_hypergraph.edgeWeight(1));

  // Verify Hypergraph Structure
  verifyIncidentNets(c_hypergraph, 0, { 0 });
  verifyIncidentNets(c_hypergraph, 1, { 0, 1 });
  verifyIncidentNets(c_hypergraph, 2, { });
  verifyIncidentNets(c_hypergraph, 3, { 1 });
  verifyPins(c_hypergraph, { 0, 1 },
    { {0, 1}, {1, 3} });
}

TEST_F(AStaticHypergraph, ContractCommunitiesIfCommunityInformationAreAvailable) {
  assignCommunityIds();
  parallel::scalable_vector<HypernodeID> c_mapping = {0, 0, 1, 2, 2, 3, 3};
  StaticHypergraph c_hypergraph = hypergraph.contract(c_mapping);

  // Verify Community Ids
  ASSERT_EQ(0, c_hypergraph.communityID(0));
  ASSERT_EQ(0, c_hypergraph.communityID(1));
  ASSERT_EQ(1, c_hypergraph.communityID(2));
  ASSERT_EQ(2, c_hypergraph.communityID(3));

}


}
} // namespace mt_kahypar