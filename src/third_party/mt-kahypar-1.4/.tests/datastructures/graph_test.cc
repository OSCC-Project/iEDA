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
#include "mt-kahypar/datastructures/graph.h"

using ::testing::Test;

namespace mt_kahypar::ds {

using AGraph = HypergraphFixture<ds::StaticHypergraph>;
using TestGraph = Graph<ds::StaticHypergraph>;

void verifyArcIterator(const TestGraph& graph,
                       const NodeID u,
                       const std::vector<NodeID>& arcs,
                       const std::vector<ArcWeight>& weights,
                       const size_t n = std::numeric_limits<size_t>::max()) {
  size_t size = 0;
  std::vector<bool> vis(arcs.size(), false);
  for ( const Arc& arc : graph.arcsOf(u, n) ) {
    for ( size_t pos = 0; pos < arcs.size(); ++pos ) {
      if ( arcs[pos] == arc.head ) {
        ASSERT_FALSE(vis[pos]);
        ASSERT_EQ(arcs[pos], arc.head);
        ASSERT_EQ(weights[pos], arc.weight);
        vis[pos] = true;
        ++size;
      }
    }
  }
  ASSERT_EQ(arcs.size(), size);
}

TEST_F(AGraph, HasCorrectNumNodesAndArcs) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::uniform);
  ASSERT_EQ(11, graph.numNodes());
  ASSERT_EQ(24, graph.numArcs());
  ASSERT_EQ(4,  graph.max_degree());
}

TEST_F(AGraph, IteratesOverAllNodes) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::uniform);
  std::vector<bool> vis(graph.numNodes(), false);
  for ( const NodeID& u : graph.nodes() ) {
    ASSERT_LE(u, graph.numNodes() - 1);
    vis[u] = true;
  }

  for ( size_t i = 0; i < vis.size(); ++i ) {
    ASSERT_TRUE(vis[i]);
  }
}

TEST_F(AGraph, VerifyTotalVolumeForUniformEdgeWeight) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::uniform);
  ASSERT_EQ(24, graph.totalVolume());
}

TEST_F(AGraph, VerifyNodeVolumeForUniformEdgeWeight) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::uniform);
  ASSERT_EQ(2, graph.nodeVolume(0));
  ASSERT_EQ(1, graph.nodeVolume(1));
  ASSERT_EQ(2, graph.nodeVolume(3));
  ASSERT_EQ(1, graph.nodeVolume(5));
  ASSERT_EQ(4, graph.nodeVolume(8));
  ASSERT_EQ(3, graph.nodeVolume(10));
}

TEST_F(AGraph, VerifyNodeVolumeForNonUniformEdgeWeight) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::non_uniform);
  ASSERT_EQ(0.75, graph.nodeVolume(0));
  ASSERT_EQ(0.25, graph.nodeVolume(1));
  ASSERT_EQ(0.25 + ( 1.0 / 3.0 ), graph.nodeVolume(3));
  ASSERT_EQ(1.0 / 3.0, graph.nodeVolume(5));
  ASSERT_EQ(1.0, graph.nodeVolume(8));
  ASSERT_EQ(1.0, graph.nodeVolume(10));
}

TEST_F(AGraph, WithCorrectVertexDegrees) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::degree);
  ASSERT_EQ(2, graph.degree(0));
  ASSERT_EQ(1, graph.degree(1));
  ASSERT_EQ(2, graph.degree(2));
  ASSERT_EQ(2, graph.degree(3));
  ASSERT_EQ(2, graph.degree(4));
  ASSERT_EQ(1, graph.degree(5));
  ASSERT_EQ(2, graph.degree(6));
  ASSERT_EQ(2, graph.degree(7));
  ASSERT_EQ(4, graph.degree(8));
  ASSERT_EQ(3, graph.degree(9));
  ASSERT_EQ(3, graph.degree(10));
}

TEST_F(AGraph, HasCorrectAdjacentVertices1a) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::uniform);
  verifyArcIterator(graph, 0, {7, 8}, {1.0, 1.0});
}

TEST_F(AGraph, HasCorrectAdjacentVertices1b) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::non_uniform);
  verifyArcIterator(graph, 0, {7, 8}, {0.5, 0.25});
}

TEST_F(AGraph, HasCorrectAdjacentVertices1c) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::degree);
  verifyArcIterator(graph, 0, {7, 8}, {1.0, 0.5});
}

TEST_F(AGraph, HasCorrectAdjacentVertices1d) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::uniform);
  verifyArcIterator(graph, 0, {7}, {1.0}, 1);
}

TEST_F(AGraph, HasCorrectAdjacentVertices2a) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::uniform);
  verifyArcIterator(graph, 2, {7, 10}, {1.0, 1.0});
}

TEST_F(AGraph, HasCorrectAdjacentVertices2b) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::non_uniform);
  verifyArcIterator(graph, 2, {7, 10}, {0.5, 1.0 / 3.0});
}

TEST_F(AGraph, HasCorrectAdjacentVertices2c) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::degree);
  verifyArcIterator(graph, 2, {7, 10}, {1.0, 2.0 / 3.0});
}

TEST_F(AGraph, HasCorrectAdjacentVertices2d) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::uniform);
  verifyArcIterator(graph, 2, {7}, {1.0}, 1);
}

TEST_F(AGraph, HasCorrectAdjacentVertices3a) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::uniform);
  verifyArcIterator(graph, 5, {10}, {1.0});
}

TEST_F(AGraph, HasCorrectAdjacentVertices3b) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::non_uniform);
  verifyArcIterator(graph, 5, {10}, {1.0 / 3.0});
}

TEST_F(AGraph, HasCorrectAdjacentVertices3c) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::degree);
  verifyArcIterator(graph, 5, {10}, {1.0 / 3.0});
}

TEST_F(AGraph, HasCorrectAdjacentVertices3d) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::uniform);
  verifyArcIterator(graph, 5, {10}, {1.0}, 2);
}

TEST_F(AGraph, HasCorrectAdjacentVertices4a) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::uniform);
  verifyArcIterator(graph, 6, {9, 10}, {1.0, 1.0});
}

TEST_F(AGraph, HasCorrectAdjacentVertices4b) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::non_uniform);
  verifyArcIterator(graph, 6, {9, 10}, {1.0 / 3.0, 1.0 / 3.0});
}

TEST_F(AGraph, HasCorrectAdjacentVertices4c) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::degree);
  verifyArcIterator(graph, 6, {9, 10}, {2.0 / 3.0, 2.0 / 3.0});
}

TEST_F(AGraph, HasCorrectAdjacentVertices4d) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::uniform);
  verifyArcIterator(graph, 6, {9, 10}, {1.0, 1.0}, 1000);
}

TEST_F(AGraph, HasCorrectAdjacentVertices5a) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::uniform);
  verifyArcIterator(graph, 7, {0, 2}, {1.0, 1.0});
}

TEST_F(AGraph, HasCorrectAdjacentVertices5b) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::non_uniform);
  verifyArcIterator(graph, 7, {0, 2}, {0.5, 0.5});
}

TEST_F(AGraph, HasCorrectAdjacentVertices5c) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::degree);
  verifyArcIterator(graph, 7, {0, 2}, {1.0, 1.0});
}

TEST_F(AGraph, HasCorrectAdjacentVertices5d) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::uniform);
  verifyArcIterator(graph, 7, {0}, {1.0}, 1);
}

TEST_F(AGraph, HasCorrectAdjacentVertices6a) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::uniform);
  verifyArcIterator(graph, 8, {0, 1, 3, 4}, {1.0, 1.0, 1.0, 1.0});
}

TEST_F(AGraph, HasCorrectAdjacentVertices6b) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::non_uniform);
  verifyArcIterator(graph, 8, {0, 1, 3, 4}, {0.25, 0.25, 0.25, 0.25});
}

TEST_F(AGraph, HasCorrectAdjacentVertices6c) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::degree);
  verifyArcIterator(graph, 8, {0, 1, 3, 4}, {0.5, 0.25, 0.5, 0.5});
}

TEST_F(AGraph, HasCorrectAdjacentVertices6d) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::uniform);
  verifyArcIterator(graph, 8, {0, 1}, {1.0, 1.0}, 2);
}

TEST_F(AGraph, ConstructsAHypergraphWhichIsAGraph) {
  using HypergraphFactory = typename ds::StaticHypergraph::Factory;
  ds::StaticHypergraph graph_hg = HypergraphFactory::construct(
    5, 6, { { 0, 1 }, { 0, 2 }, {1, 2}, { 2, 3 }, { 2, 4 }, { 3, 4 } } );
  TestGraph graph(graph_hg, LouvainEdgeWeight::uniform, true);
  ASSERT_EQ(4, graph.max_degree());
  verifyArcIterator(graph, 0, {1, 2}, {1.0, 1.0});
  verifyArcIterator(graph, 1, {0, 2}, {1.0, 1.0});
  verifyArcIterator(graph, 2, {0, 1, 3, 4}, {1.0, 1.0, 1.0, 1.0});
  verifyArcIterator(graph, 3, {2, 4}, {1.0, 1.0});
  verifyArcIterator(graph, 4, {2, 3}, {1.0, 1.0});
}


Clustering clustering(const std::vector<PartitionID>& communities) {
  Clustering c(communities.size());
  for ( size_t i = 0; i < communities.size(); ++i ) {
    c[i] = communities[i];
  }
  return c;
}

TEST_F(AGraph, ContractCommunities1) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::uniform);
  Clustering communities = clustering( { 3, 3, 3, 2, 2, 4, 4, 3, 3, 2, 4 } );
  TestGraph coarse_graph = graph.contract(communities, false);

  ASSERT_EQ(graph.totalVolume(), coarse_graph.totalVolume());
  ASSERT_EQ(2, coarse_graph.max_degree());
  ASSERT_EQ(7,  coarse_graph.nodeVolume(0));
  ASSERT_EQ(11, coarse_graph.nodeVolume(1));
  ASSERT_EQ(6,  coarse_graph.nodeVolume(2));

  verifyArcIterator(coarse_graph, 0, {1, 2}, {2, 1});
  verifyArcIterator(coarse_graph, 1, {0, 2}, {2, 1});
  verifyArcIterator(coarse_graph, 2, {0, 1}, {1, 1});
}

TEST_F(AGraph, ContractCommunities2) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::uniform);
  Clustering communities = clustering( { 7, 7, 2, 9, 9, 2, 2, 7, 9, 9, 2 } );
  TestGraph coarse_graph = graph.contract(communities, false);

  ASSERT_EQ(graph.totalVolume(), coarse_graph.totalVolume());
  ASSERT_EQ(2, coarse_graph.max_degree());
  ASSERT_EQ(8,  coarse_graph.nodeVolume(0));
  ASSERT_EQ(5,  coarse_graph.nodeVolume(1));
  ASSERT_EQ(11, coarse_graph.nodeVolume(2));

  verifyArcIterator(coarse_graph, 0, {1, 2}, {1, 1});
  verifyArcIterator(coarse_graph, 1, {0, 2}, {1, 2});
  verifyArcIterator(coarse_graph, 2, {0, 1}, {1, 2});
}

TEST_F(AGraph, ContractCommunities3) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::uniform);
  Clustering communities = clustering( { 5, 5, 7, 3, 3, 9, 9, 7, 5, 3, 9 });
  TestGraph coarse_graph = graph.contract(communities, false);

  ASSERT_EQ(graph.totalVolume(), coarse_graph.totalVolume());
  ASSERT_EQ(2, coarse_graph.max_degree());
  ASSERT_EQ(7, coarse_graph.nodeVolume(0));
  ASSERT_EQ(7, coarse_graph.nodeVolume(1));
  ASSERT_EQ(4, coarse_graph.nodeVolume(2));
  ASSERT_EQ(6, coarse_graph.nodeVolume(3));

  verifyArcIterator(coarse_graph, 0, {1, 3}, {2, 1});
  verifyArcIterator(coarse_graph, 1, {0, 2}, {2, 1});
  verifyArcIterator(coarse_graph, 2, {1, 3}, {1, 1});
  verifyArcIterator(coarse_graph, 3, {0, 2}, {1, 1});
}

TEST_F(AGraph, ContractCommunities4) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::uniform);
  Clustering communities = clustering({ 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5 });
  TestGraph coarse_graph = graph.contract(communities, false);

  ASSERT_EQ(graph.totalVolume(), coarse_graph.totalVolume());
  ASSERT_EQ(3, coarse_graph.max_degree());
  ASSERT_EQ(3, coarse_graph.nodeVolume(0));
  ASSERT_EQ(4, coarse_graph.nodeVolume(1));
  ASSERT_EQ(3, coarse_graph.nodeVolume(2));
  ASSERT_EQ(2, coarse_graph.nodeVolume(3));
  ASSERT_EQ(6, coarse_graph.nodeVolume(4));
  ASSERT_EQ(6, coarse_graph.nodeVolume(5));

  verifyArcIterator(coarse_graph, 0, {4},       {3});
  verifyArcIterator(coarse_graph, 1, {4, 5},    {2, 2});
  verifyArcIterator(coarse_graph, 2, {4, 5},    {1, 2});
  verifyArcIterator(coarse_graph, 3, {5},       {2});
  verifyArcIterator(coarse_graph, 4, {0, 1, 2}, {3, 2, 1});
  verifyArcIterator(coarse_graph, 5, {1, 2, 3}, {2, 2, 2});
}

TEST_F(AGraph, ContractCommunities5) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::uniform);
  Clustering communities = clustering({ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
  TestGraph coarse_graph = graph.contract(communities, false);

  ASSERT_EQ(graph.totalVolume(), coarse_graph.totalVolume());
  ASSERT_EQ(0,  coarse_graph.max_degree());
  ASSERT_EQ(1,  coarse_graph.numNodes());
  ASSERT_EQ(0,  coarse_graph.numArcs());
  ASSERT_EQ(24, coarse_graph.nodeVolume(0));
  ASSERT_EQ(0,  coarse_graph.degree(0));
}

TEST_F(AGraph, HasSameTotalVolumeAfterTwoContractions) {
  TestGraph graph(hypergraph, LouvainEdgeWeight::uniform);
  Clustering communities = clustering( { 3, 3, 3, 2, 2, 4, 4, 3, 3, 2, 4 } );
  TestGraph coarse_graph = graph.contract(communities, false);
  communities = clustering( { 0, 1, 2 } );
  TestGraph coarse_coarse_graph = coarse_graph.contract(communities, false);

  ASSERT_EQ(coarse_coarse_graph.totalVolume(), coarse_graph.totalVolume());
  ASSERT_EQ(7,  coarse_coarse_graph.nodeVolume(0));
  ASSERT_EQ(11, coarse_coarse_graph.nodeVolume(1));
  ASSERT_EQ(6,  coarse_coarse_graph.nodeVolume(2));
}

} // namespace mt_kahypar::ds