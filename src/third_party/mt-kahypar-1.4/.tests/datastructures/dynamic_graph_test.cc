/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2019 Tobias Heuer <tobias.heuer@kit.edu>
 * Copyright (C) 2022 Nikolai Maas <nikolai.maas@student.kit.edu>
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
#include "mt-kahypar/datastructures/dynamic_graph.h"
#include "mt-kahypar/datastructures/dynamic_graph_factory.h"
#include "mt-kahypar/utils/randomize.h"

namespace mt_kahypar {
namespace ds {

using ADynamicGraph = HypergraphFixture<DynamicGraph, true>;

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

TEST_F(ADynamicGraph, HasCorrectStats) {
  ASSERT_EQ(7,  hypergraph.initialNumNodes());
  ASSERT_EQ(12,  hypergraph.initialNumEdges());
  ASSERT_EQ(12, hypergraph.initialNumPins());
  ASSERT_EQ(12, hypergraph.initialTotalVertexDegree());
  ASSERT_EQ(7,  hypergraph.totalWeight());
  ASSERT_EQ(2,  hypergraph.maxEdgeSize());
}

TEST_F(ADynamicGraph, HasCorrectInitialNodeIterator) {
  HypernodeID expected_hn = 0;
  for ( const HypernodeID& hn : hypergraph.nodes() ) {
    ASSERT_EQ(expected_hn++, hn);
  }
  ASSERT_EQ(7, expected_hn);
}

TEST_F(ADynamicGraph, HasCorrectNodeIteratorIfVerticesAreDisabled) {
  hypergraph.removeDegreeZeroHypernode(0);
  hypergraph.disableHypernode(5);
  const std::vector<HypernodeID> expected_iter =
    { 1, 2, 3, 4, 6 };
  HypernodeID pos = 0;
  for ( const HypernodeID& hn : hypergraph.nodes() ) {
    ASSERT_EQ(expected_iter[pos++], hn);
  }
  ASSERT_EQ(expected_iter.size(), pos);
}

TEST_F(ADynamicGraph, HasCorrectInitialEdgeIterator) {
  std::vector<HyperedgeID> expected_iter = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  HypernodeID pos = 0;
  for ( const HyperedgeID& he : hypergraph.edges() ) {
    ASSERT_EQ(expected_iter[pos++], he);
  }
  ASSERT_EQ(expected_iter.size(), pos);
}

TEST_F(ADynamicGraph, VerifiesIncidentEdges) {
  verifyIncidentNets(0, { });
  verifyIncidentNets(1, { 0, 1 });
  verifyIncidentNets(2, { 2, 3 });
  verifyIncidentNets(3, { 4 });
  verifyIncidentNets(4, { 5, 6, 7 });
  verifyIncidentNets(5, { 8, 9 });
  verifyIncidentNets(6, { 10, 11 });
}

TEST_F(ADynamicGraph, VerifiesPinsOfEdges) {
  verifyPins({ 0, 1, 3, 6, 7, 9 },
    { {1, 2}, {1, 4}, {2, 3}, {4, 5}, {4, 6}, {5, 6} });
}

TEST_F(ADynamicGraph, VerifiesVertexWeights) {
  for ( const HypernodeID& hn : hypergraph.nodes() ) {
    ASSERT_EQ(1, hypergraph.nodeWeight(hn));
  }
}

TEST_F(ADynamicGraph, IteratesParallelOverAllNodes) {
  std::vector<uint8_t> visited(7, false);
  hypergraph.doParallelForAllNodes([&](const HypernodeID hn) {
      visited[hn] = true;
    });

  for ( size_t i = 0; i < visited.size(); ++i ) {
    ASSERT_TRUE(visited[i]) << i;
  }
}

TEST_F(ADynamicGraph, IteratesParallelOverAllEdges) {
  std::vector<uint8_t> visited(12, false);
  hypergraph.doParallelForAllEdges([&](const HyperedgeID he) {
      visited[he] = true;
    });

  for ( size_t i = 0; i < visited.size(); ++i ) {
    ASSERT_TRUE(visited[i]) << i;
  }
}

TEST_F(ADynamicGraph, ModifiesNodeWeight) {
  hypergraph.setNodeWeight(0, 2);
  hypergraph.setNodeWeight(6, 2);
  ASSERT_EQ(2, hypergraph.nodeWeight(0));
  ASSERT_EQ(2, hypergraph.nodeWeight(6));
  hypergraph.updateTotalWeight(parallel_tag_t());
  ASSERT_EQ(9, hypergraph.totalWeight());
}


TEST_F(ADynamicGraph, VerifiesVertexDegrees) {
  ASSERT_EQ(0, hypergraph.nodeDegree(0));
  ASSERT_EQ(2, hypergraph.nodeDegree(1));
  ASSERT_EQ(2, hypergraph.nodeDegree(2));
  ASSERT_EQ(1, hypergraph.nodeDegree(3));
  ASSERT_EQ(3, hypergraph.nodeDegree(4));
  ASSERT_EQ(2, hypergraph.nodeDegree(5));
  ASSERT_EQ(2, hypergraph.nodeDegree(6));
}

TEST_F(ADynamicGraph, RemovesVertices) {
  hypergraph.removeHypernode(0);
  hypergraph.removeHypernode(5);
  ASSERT_EQ(2, hypergraph.numRemovedHypernodes());
}

TEST_F(ADynamicGraph, VerifiesEdgeWeights) {
  for ( const HyperedgeID& he : hypergraph.edges() ) {
    ASSERT_EQ(1, hypergraph.edgeWeight(he));
  }
}

TEST_F(ADynamicGraph, ModifiesEdgeWeight) {
  hypergraph.setEdgeWeight(0, 2);
  hypergraph.setEdgeWeight(2, 2);
  ASSERT_EQ(2, hypergraph.edgeWeight(0));
  ASSERT_EQ(2, hypergraph.edgeWeight(2));
}

TEST_F(ADynamicGraph, VerifiesEdgeSizes) {
  for ( const HyperedgeID& he : hypergraph.edges() ) {
    ASSERT_EQ(2, hypergraph.edgeSize(he));
  }
}

TEST_F(ADynamicGraph, SetsCommunityIDsForEachVertex) {
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

TEST_F(ADynamicGraph, ComparesStatsIfCopiedParallel) {
  DynamicGraph copy_hg = hypergraph.copy(parallel_tag_t());
  ASSERT_EQ(hypergraph.initialNumNodes(), copy_hg.initialNumNodes());
  ASSERT_EQ(hypergraph.initialNumEdges(), copy_hg.initialNumEdges());
  ASSERT_EQ(hypergraph.initialNumPins(), copy_hg.initialNumPins());
  ASSERT_EQ(hypergraph.initialTotalVertexDegree(), copy_hg.initialTotalVertexDegree());
  ASSERT_EQ(hypergraph.totalWeight(), copy_hg.totalWeight());
  ASSERT_EQ(hypergraph.maxEdgeSize(), copy_hg.maxEdgeSize());
}

TEST_F(ADynamicGraph, ComparesStatsIfCopiedSequential) {
  DynamicGraph copy_hg = hypergraph.copy();
  ASSERT_EQ(hypergraph.initialNumNodes(), copy_hg.initialNumNodes());
  ASSERT_EQ(hypergraph.initialNumEdges(), copy_hg.initialNumEdges());
  ASSERT_EQ(hypergraph.initialNumPins(), copy_hg.initialNumPins());
  ASSERT_EQ(hypergraph.initialTotalVertexDegree(), copy_hg.initialTotalVertexDegree());
  ASSERT_EQ(hypergraph.totalWeight(), copy_hg.totalWeight());
  ASSERT_EQ(hypergraph.maxEdgeSize(), copy_hg.maxEdgeSize());
}

TEST_F(ADynamicGraph, ComparesIncidentEdgesIfCopiedParallel) {
  DynamicGraph copy_hg = hypergraph.copy(parallel_tag_t());
  verifyIncidentNets(copy_hg, 0, { });
  verifyIncidentNets(copy_hg, 1, { 0, 1 });
  verifyIncidentNets(copy_hg, 2, { 2, 3 });
  verifyIncidentNets(copy_hg, 3, { 4 });
  verifyIncidentNets(copy_hg, 4, { 5, 6, 7 });
  verifyIncidentNets(copy_hg, 5, { 8, 9 });
  verifyIncidentNets(copy_hg, 6, { 10, 11 });
}

TEST_F(ADynamicGraph, ComparesIncidentEdgesIfCopiedSequential) {
  DynamicGraph copy_hg = hypergraph.copy();
  verifyIncidentNets(copy_hg, 0, { });
  verifyIncidentNets(copy_hg, 1, { 0, 1 });
  verifyIncidentNets(copy_hg, 2, { 2, 3 });
  verifyIncidentNets(copy_hg, 3, { 4 });
  verifyIncidentNets(copy_hg, 4, { 5, 6, 7 });
  verifyIncidentNets(copy_hg, 5, { 8, 9 });
  verifyIncidentNets(copy_hg, 6, { 10, 11 });
}

TEST_F(ADynamicGraph, RegistersAContraction1) {
  ASSERT_TRUE(hypergraph.registerContraction(1, 0));
  ASSERT_EQ(1, hypergraph.contractionTree(0));
  ASSERT_EQ(1, hypergraph.contractionTree(1));
  ASSERT_EQ(0, hypergraph.pendingContractions(0));
  ASSERT_EQ(1, hypergraph.pendingContractions(1));
}

TEST_F(ADynamicGraph, RegistersAContraction2) {
  ASSERT_TRUE(hypergraph.registerContraction(4, 3));
  ASSERT_TRUE(hypergraph.registerContraction(4, 2));
  ASSERT_EQ(4, hypergraph.contractionTree(2));
  ASSERT_EQ(4, hypergraph.contractionTree(3));
  ASSERT_EQ(4, hypergraph.contractionTree(4));
  ASSERT_EQ(0, hypergraph.pendingContractions(2));
  ASSERT_EQ(0, hypergraph.pendingContractions(3));
  ASSERT_EQ(2, hypergraph.pendingContractions(4));
}

TEST_F(ADynamicGraph, RegistersAContraction3) {
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

TEST_F(ADynamicGraph, RegistersAContraction4) {
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

TEST_F(ADynamicGraph, RegistersAContractionThatInducesACycle1) {
  ASSERT_TRUE(hypergraph.registerContraction(1, 0));
  ASSERT_FALSE(hypergraph.registerContraction(0, 1));
}

TEST_F(ADynamicGraph, RegistersAContractionThatInducesACycle2) {
  ASSERT_TRUE(hypergraph.registerContraction(1, 0));
  ASSERT_TRUE(hypergraph.registerContraction(2, 1));
  ASSERT_TRUE(hypergraph.registerContraction(3, 2));
  hypergraph.decrementPendingContractions(1);
  hypergraph.decrementPendingContractions(2);
  ASSERT_FALSE(hypergraph.registerContraction(0, 3));
}

TEST_F(ADynamicGraph, RegistersAContractionThatInducesACycle3) {
  ASSERT_TRUE(hypergraph.registerContraction(4, 3));
  ASSERT_TRUE(hypergraph.registerContraction(4, 2));
  ASSERT_TRUE(hypergraph.registerContraction(6, 4));
  ASSERT_TRUE(hypergraph.registerContraction(5, 6));
  ASSERT_FALSE(hypergraph.registerContraction(4, 5));
}

TEST_F(ADynamicGraph, RegisterContractionsInParallel) {
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

void verifyNeighbors(const HypernodeID u,
                     const DynamicGraph& graph,
                     const std::set<HypernodeID>& _expected_neighbors,
                     bool strict = false) {
  size_t num_neighbors = 0;
  size_t degree = 0;
  std::vector<bool> actual_neighbors(graph.initialNumNodes(), false);
  for ( const HyperedgeID& he : graph.incidentEdges(u) ) {
    const HypernodeID neighbor = graph.edge(he).target;
    ASSERT_NE(_expected_neighbors.find(neighbor), _expected_neighbors.end())
      << "Vertex " << neighbor << " should not be neighbor of vertex " << u;
    ASSERT_EQ(u, graph.edge(he).source)
      << "Source of " << he << " (target: " << graph.edge(he).target << ") should be "
      << u << " but is " << graph.edge(he).source;
    ASSERT_TRUE(!strict || !actual_neighbors[neighbor])
      << "Vertex " << u << " contain duplicate edge with target " << neighbor;
    if (!actual_neighbors[neighbor]) {
      ++num_neighbors;
    }
    ++degree;
    actual_neighbors[neighbor] = true;
  }
  ASSERT_EQ(num_neighbors, _expected_neighbors.size());
  ASSERT_EQ(degree, graph.nodeDegree(u));
  ASSERT_TRUE(!strict || num_neighbors == degree);
}

TEST_F(ADynamicGraph, PerformsContractions1) {
  ASSERT_TRUE(hypergraph.registerContraction(1, 0));
  hypergraph.contract(0);

  ASSERT_FALSE(hypergraph.nodeIsEnabled(0));
  ASSERT_EQ(2, hypergraph.nodeWeight(1));
  ASSERT_EQ(0, hypergraph.pendingContractions(1));

  verifyNeighbors(1, hypergraph, { 2, 4 });
}

TEST_F(ADynamicGraph, PerformsContractions2) {
  ASSERT_TRUE(hypergraph.registerContraction(1, 0));
  ASSERT_TRUE(hypergraph.registerContraction(1, 2));
  ASSERT_TRUE(hypergraph.registerContraction(4, 5));

  hypergraph.contract(0);
  ASSERT_FALSE(hypergraph.nodeIsEnabled(0));
  ASSERT_EQ(2, hypergraph.nodeWeight(1));
  ASSERT_EQ(1, hypergraph.pendingContractions(1));
  hypergraph.contract(5);
  ASSERT_FALSE(hypergraph.nodeIsEnabled(5));
  ASSERT_EQ(2, hypergraph.nodeWeight(4));
  ASSERT_EQ(0, hypergraph.pendingContractions(4));
  hypergraph.contract(2);
  ASSERT_FALSE(hypergraph.nodeIsEnabled(2));
  ASSERT_EQ(3, hypergraph.nodeWeight(1));
  ASSERT_EQ(0, hypergraph.pendingContractions(1));

  verifyNeighbors(1, hypergraph, { 1, 3, 4 });
  verifyNeighbors(4, hypergraph, { 1, 4, 6 });
}

TEST_F(ADynamicGraph, PerformsAContractionWithWeightGreaterThanMaxNodeWeight) {
  ASSERT_TRUE(hypergraph.registerContraction(1, 0));
  ASSERT_EQ(1, hypergraph.contractionTree(0));
  ASSERT_EQ(1, hypergraph.pendingContractions(1));
  hypergraph.contract(0, 1);
  ASSERT_TRUE(hypergraph.nodeIsEnabled(0));
  ASSERT_TRUE(hypergraph.nodeIsEnabled(1));
  ASSERT_EQ(0, hypergraph.contractionTree(0));
  ASSERT_EQ(0, hypergraph.pendingContractions(1));
}

void verifyEqualityOfDynamicGraphs(DynamicGraph& expected_graph,
                                   DynamicGraph& actual_graph) {
  expected_graph.sortIncidentEdges();
  actual_graph.sortIncidentEdges();
  parallel::scalable_vector<HyperedgeID> expected_incident_edges;
  parallel::scalable_vector<HyperedgeID> actual_incident_edges;
  for ( const HypernodeID& hn : expected_graph.nodes() ) {
    ASSERT_TRUE(actual_graph.nodeIsEnabled(hn));
    ASSERT_EQ(expected_graph.nodeWeight(hn), actual_graph.nodeWeight(hn));
    ASSERT_EQ(expected_graph.nodeDegree(hn), actual_graph.nodeDegree(hn));
    for ( const HyperedgeID he : expected_graph.incidentEdges(hn) ) {
      expected_incident_edges.push_back(he);
    }
    for ( const HyperedgeID he : actual_graph.incidentEdges(hn) ) {
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
  for ( const HyperedgeID& he : expected_graph.edges() ) {
    ASSERT_EQ(expected_graph.edgeSource(he), actual_graph.edgeSource(he));
    ASSERT_EQ(expected_graph.edgeTarget(he), actual_graph.edgeTarget(he));
  }
}

void verifyBatchUncontractions(DynamicGraph& graph,
                               const parallel::scalable_vector<Memento>& contractions,
                               const size_t batch_size) {
  DynamicGraph expected_graph = graph.copy();

  // Perform contractions
  for ( const Memento& memento : contractions ) {
    graph.registerContraction(memento.u, memento.v);
    graph.contract(memento.v);
  }

  auto versioned_batches = graph.createBatchUncontractionHierarchy(batch_size);

  while ( !versioned_batches.empty() ) {
    BatchVector& batches = versioned_batches.back();
    while ( !batches.empty() ) {
      const parallel::scalable_vector<Memento> batch = batches.back();
      graph.uncontract(batch, [](const HyperedgeID&) { return false; });
      batches.pop_back();
    }
    versioned_batches.pop_back();
  }

  verifyEqualityOfDynamicGraphs(expected_graph, graph);
}

TEST_F(ADynamicGraph, PerformsBatchUncontractions1) {
  verifyBatchUncontractions(hypergraph,
    { Memento { 0, 2 }, Memento { 3, 4 }, Memento { 5, 6 } }, 3);
}

TEST_F(ADynamicGraph, PerformsBatchUncontractions2) {
  verifyBatchUncontractions(hypergraph,
    { Memento { 1, 0 }, Memento { 2, 1 }, Memento { 3, 2 } }, 3);
}

TEST_F(ADynamicGraph, PerformsBatchUncontractions3) {
  verifyBatchUncontractions(hypergraph,
    { Memento { 1, 0 }, Memento { 1, 2 }, Memento { 3, 1 },
      Memento { 4, 6 }, Memento { 4, 5 } }, 3);
}

TEST_F(ADynamicGraph, PerformsBatchUncontractions4) {
  verifyBatchUncontractions(hypergraph,
    { Memento { 5, 6 }, Memento { 4, 5 }, Memento { 3, 4 },
      Memento { 2, 3 }, Memento { 1, 2 }, Memento { 0, 1 } }, 3);
}

TEST_F(ADynamicGraph, PerformsBatchUncontractions5) {
  verifyBatchUncontractions(hypergraph,
    { Memento { 2, 6 }, Memento { 2, 5 }, Memento { 1, 3 },
      Memento { 1, 4 }, Memento { 0, 1 }, Memento { 0, 2 } }, 2);
}

TEST_F(ADynamicGraph, GeneratesACompactifiedHypergraph) {
  const parallel::scalable_vector<Memento> contractions =
   { Memento { 0, 3 }, Memento { 1, 5 }, Memento { 6, 2 }, Memento { 6, 4 } };

  for ( const Memento& memento : contractions ) {
    hypergraph.registerContraction(memento.u, memento.v);
    hypergraph.contract(memento.v);
  }
  hypergraph.removeSinglePinAndParallelHyperedges();

  auto res = DynamicGraphFactory::compactify(hypergraph);
  DynamicGraph& compactified_hg = res.first;
  compactified_hg.sortIncidentEdges();
  ASSERT_EQ(3, compactified_hg.initialNumNodes());
  ASSERT_EQ(4, compactified_hg.initialNumEdges());
  ASSERT_EQ(1, compactified_hg.edgeWeight(0));
  ASSERT_EQ(4, compactified_hg.edgeWeight(1));
  ASSERT_EQ(1, compactified_hg.edgeWeight(2));
  ASSERT_EQ(4, compactified_hg.edgeWeight(3));
  ASSERT_EQ(2, compactified_hg.nodeWeight(0));
  ASSERT_EQ(2, compactified_hg.nodeWeight(1));
  ASSERT_EQ(3, compactified_hg.nodeWeight(2));
  verifyPins(compactified_hg, {0}, { {0, 2} });
  verifyPins(compactified_hg, {1}, { {1, 2} });
}

} // namespace ds
} // namespace mt_kahypar
