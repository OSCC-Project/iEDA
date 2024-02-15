/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2020 Tobias Heuer <tobias.heuer@kit.edu>
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

#include "mt-kahypar/datastructures/dynamic_adjacency_array.h"
#include "mt-kahypar/parallel/atomic_wrapper.h"

using ::testing::Test;

namespace mt_kahypar {
namespace ds {

void verifyNeighbors(const HypernodeID u,
                     const HypernodeID num_nodes,
                     const DynamicAdjacencyArray& adjacency_array,
                     const std::set<HypernodeID>& _expected_neighbors,
                     bool strict = false) {
  size_t num_neighbors = 0;
  size_t degree = 0;
  std::vector<bool> actual_neighbors(num_nodes, false);
  for ( const HyperedgeID& he : adjacency_array.incidentEdges(u) ) {
    const HypernodeID neighbor = adjacency_array.edge(he).target;
    ASSERT_NE(_expected_neighbors.find(neighbor), _expected_neighbors.end())
      << "Vertex " << neighbor << " should not be neighbor of vertex " << u;
    ASSERT_EQ(u, adjacency_array.edge(he).source)
      << "Source of " << he << " (target: " << adjacency_array.edge(he).target << ") should be "
      << u << " but is " << adjacency_array.edge(he).source;
    ASSERT_TRUE(!strict || !actual_neighbors[neighbor])
      << "Vertex " << u << " contain duplicate edge with target " << neighbor;
    if (!actual_neighbors[neighbor]) {
      ++num_neighbors;
    }
    ++degree;
    actual_neighbors[neighbor] = true;
  }
  ASSERT_EQ(num_neighbors, _expected_neighbors.size());
  ASSERT_EQ(degree, adjacency_array.nodeDegree(u));
  ASSERT_TRUE(!strict || num_neighbors == degree);
}

void verifyEdges(HyperedgeID expected_num_edges, const DynamicAdjacencyArray& adjacency_array,
                 const std::set<std::pair<HypernodeID, HypernodeID>>& _expected_edges) {
  size_t num_edges = 0;
  for ( const HyperedgeID& he : adjacency_array.edges() ) {
    const HypernodeID source = adjacency_array.edge(he).source;
    const HypernodeID target = adjacency_array.edge(he).target;
    ASSERT_TRUE(_expected_edges.find({source, target}) != _expected_edges.end() ||
                _expected_edges.find({target, source}) != _expected_edges.end())
      << "Edge (" << source << ", " << target << ") is invalid.";
    ++num_edges;
  }
  ASSERT_EQ(num_edges, expected_num_edges);
}

void verifyEdgeWeight(const DynamicAdjacencyArray& adjacency_array, HyperedgeID source,
                      HyperedgeID target, HyperedgeWeight weight) {
  bool found = false;
  for (HyperedgeID edge: adjacency_array.edges()) {
    const auto& e = adjacency_array.edge(edge);
    if ((e.source == source && e.target == target) || (e.target == source && e.source == target)) {
      ASSERT_EQ(weight, e.weight);
      found = true;
    }
  }
  ASSERT_TRUE(found);
}

TEST(ADynamicAdjacencyArray, VerifyInitialEdges) {
  DynamicAdjacencyArray adjacency_array(
    7, {{1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6}});
  verifyEdges(12, adjacency_array, { {1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6} });
}

TEST(ADynamicAdjacencyArray, VerifyEdgesAfterContractions1) {
  DynamicAdjacencyArray adjacency_array(
    7, {{1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6}});
  adjacency_array.contract(2, 3);
  adjacency_array.contract(4, 6);
  verifyEdges(12, adjacency_array, { {1, 2}, {1, 4}, {4, 5}, {2, 2}, {4, 4} });
}

TEST(ADynamicAdjacencyArray, VerifyEdgesAfterContractions2) {
  DynamicAdjacencyArray adjacency_array(
    7, {{1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6}});
  adjacency_array.contract(1, 4);
  adjacency_array.contract(1, 5);
  verifyEdges(12, adjacency_array, {{1, 1}, {1, 2}, {2, 3}, {1, 6} });
}

TEST(ADynamicAdjacencyArray, VerifyEdgesAfterContractions3) {
  DynamicAdjacencyArray adjacency_array(
    7, {{1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6}});
  adjacency_array.contract(2, 1);
  adjacency_array.contract(4, 6);
  adjacency_array.contract(2, 3);
  adjacency_array.contract(4, 0);
  adjacency_array.contract(2, 4);
  verifyEdges(10, adjacency_array, { {2, 2}, {2, 5} });
  adjacency_array.contract(2, 5);
  verifyEdges(10, adjacency_array, { {2, 2} });
}

TEST(ADynamicAdjacencyArray, VerifyInitialNeighborsOfEachVertex) {
  DynamicAdjacencyArray adjacency_array(
    7, {{1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6}});
  verifyNeighbors(0, 7, adjacency_array, { });
  verifyNeighbors(1, 7, adjacency_array, { 2, 4 });
  verifyNeighbors(2, 7, adjacency_array, { 1, 3 });
  verifyNeighbors(3, 7, adjacency_array, { 2 });
  verifyNeighbors(4, 7, adjacency_array, { 1, 5, 6 });
  verifyNeighbors(5, 7, adjacency_array, { 4, 6 });
  verifyNeighbors(6, 7, adjacency_array, { 4, 5 });
}

TEST(ADynamicAdjacencyArray, ContractTwoVertices1) {
  DynamicAdjacencyArray adjacency_array(
    7, {{1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6}});
  adjacency_array.contract(0, 1);
  verifyNeighbors(0, 7, adjacency_array, { 2, 4 });
}

TEST(ADynamicAdjacencyArray, ContractTwoVertices2) {
  DynamicAdjacencyArray adjacency_array(
    7, {{1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6}});
  adjacency_array.contract(1, 2);
  verifyNeighbors(1, 7, adjacency_array, { 1, 3, 4 });
}

TEST(ADynamicAdjacencyArray, ContractTwoVertices3) {
  DynamicAdjacencyArray adjacency_array(
    7, {{1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6}});
  adjacency_array.contract(1, 5);
  verifyNeighbors(1, 7, adjacency_array, { 2, 4, 6 });
  verifyNeighbors(2, 7, adjacency_array, { 1, 3 });
  verifyNeighbors(4, 7, adjacency_array, { 1, 6 });
  verifyNeighbors(6, 7, adjacency_array, { 1, 4 });
}

TEST(ADynamicAdjacencyArray, ContractSeveralVertices1) {
  DynamicAdjacencyArray adjacency_array(
    7, {{1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6}});
  adjacency_array.contract(0, 1);
  adjacency_array.contract(0, 2);
  verifyNeighbors(0, 7, adjacency_array, { 0, 3, 4 });
}

TEST(ADynamicAdjacencyArray, ContractSeveralVertices2) {
  DynamicAdjacencyArray adjacency_array(
    7, {{1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6}});
  adjacency_array.contract(1, 0);
  adjacency_array.contract(1, 2);
  verifyNeighbors(1, 7, adjacency_array, { 1, 3, 4 });
  adjacency_array.contract(4, 5);
  adjacency_array.contract(4, 6);
  verifyNeighbors(4, 7, adjacency_array, { 1, 4 });
  adjacency_array.contract(1, 3);
  adjacency_array.contract(1, 4);
  verifyNeighbors(1, 7, adjacency_array, { 1 });
}

TEST(ADynamicAdjacencyArray, UncontractTwoVertices1) {
  DynamicAdjacencyArray adjacency_array(
    7, {{1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6}});
  adjacency_array.contract(1, 2);
  adjacency_array.uncontract(1, 2);
  verifyNeighbors(1, 7, adjacency_array, { 2, 4 });
  verifyNeighbors(2, 7, adjacency_array, { 1, 3 });
}

TEST(ADynamicAdjacencyArray, UncontractTwoVertices2) {
  DynamicAdjacencyArray adjacency_array(
    7, {{1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6}});
  adjacency_array.contract(1, 5);
  adjacency_array.uncontract(1, 5);
  verifyNeighbors(1, 7, adjacency_array, { 2, 4 });
  verifyNeighbors(5, 7, adjacency_array, { 4, 6 });
  verifyNeighbors(2, 7, adjacency_array, { 1, 3 });
  verifyNeighbors(4, 7, adjacency_array, { 1, 5, 6 });
  verifyNeighbors(6, 7, adjacency_array, { 4, 5 });
}

TEST(ADynamicAdjacencyArray, UncontractSeveralVertices1) {
  DynamicAdjacencyArray adjacency_array(
    7, {{1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6}});
  adjacency_array.contract(1, 2);
  adjacency_array.contract(1, 0);
  adjacency_array.contract(4, 5);
  adjacency_array.contract(4, 6);
  adjacency_array.contract(4, 3);
  adjacency_array.contract(4, 1);
  verifyNeighbors(4, 7, adjacency_array, { 4 });
  adjacency_array.uncontract(4, 1);
  verifyNeighbors(1, 7, adjacency_array, { 1, 4 });
  verifyNeighbors(4, 7, adjacency_array, { 1, 4 });
  adjacency_array.uncontract(4, 3);
  verifyNeighbors(4, 7, adjacency_array, { 1, 4 });
  verifyNeighbors(3, 7, adjacency_array, { 1 });
  adjacency_array.uncontract(4, 6);
  verifyNeighbors(4, 7, adjacency_array, { 1, 4, 6 });
  verifyNeighbors(6, 7, adjacency_array, { 4 });
  adjacency_array.uncontract(4, 5);
  verifyNeighbors(4, 7, adjacency_array, { 1, 5, 6 });
  verifyNeighbors(5, 7, adjacency_array, { 4, 6 });
  adjacency_array.uncontract(1, 0);
  verifyNeighbors(0, 7, adjacency_array, { });
  verifyNeighbors(1, 7, adjacency_array, { 1, 3, 4 });
  adjacency_array.uncontract(1, 2);
  verifyNeighbors(0, 7, adjacency_array, { });
  verifyNeighbors(1, 7, adjacency_array, { 2, 4 });
  verifyNeighbors(2, 7, adjacency_array, { 1, 3 });
  verifyNeighbors(3, 7, adjacency_array, { 2 });
  verifyNeighbors(4, 7, adjacency_array, { 1, 5, 6 });
  verifyNeighbors(5, 7, adjacency_array, { 4, 6 });
  verifyNeighbors(6, 7, adjacency_array, { 4, 5 });
}

TEST(ADynamicAdjacencyArray, UncontractSeveralVertices2) {
  DynamicAdjacencyArray adjacency_array(
    7, {{1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6}});
  adjacency_array.contract(3, 1);
  adjacency_array.contract(3, 4);
  adjacency_array.contract(5, 6);
  adjacency_array.contract(3, 5);
  adjacency_array.contract(0, 2);
  adjacency_array.contract(0, 3);
  verifyNeighbors(0, 7, adjacency_array, { 0 });
  adjacency_array.uncontract(0, 3);
  verifyNeighbors(0, 7, adjacency_array, { 3 });
  verifyNeighbors(3, 7, adjacency_array, { 0, 3 });
  adjacency_array.uncontract(0, 2);
  verifyNeighbors(0, 7, adjacency_array, { });
  verifyNeighbors(2, 7, adjacency_array, { 3 });
  adjacency_array.uncontract(3, 5);
  verifyNeighbors(3, 7, adjacency_array, { 2, 3, 5 });
  verifyNeighbors(5, 7, adjacency_array, { 3, 5 });
  adjacency_array.uncontract(5, 6);
  verifyNeighbors(5, 7, adjacency_array, { 3, 6 });
  verifyNeighbors(6, 7, adjacency_array, { 3, 5 });
  adjacency_array.uncontract(3, 4);
  verifyNeighbors(3, 7, adjacency_array, { 2, 4 });
  verifyNeighbors(4, 7, adjacency_array, { 3, 5, 6 });
  adjacency_array.uncontract(3, 1);
  verifyNeighbors(0, 7, adjacency_array, { });
  verifyNeighbors(1, 7, adjacency_array, { 2, 4 });
  verifyNeighbors(2, 7, adjacency_array, { 1, 3 });
  verifyNeighbors(3, 7, adjacency_array, { 2 });
  verifyNeighbors(4, 7, adjacency_array, { 1, 5, 6 });
  verifyNeighbors(5, 7, adjacency_array, { 4, 6 });
  verifyNeighbors(6, 7, adjacency_array, { 4, 5 });
}

TEST(ADynamicAdjacencyArray, UncontractsVerticesInDifferentOrder1) {
  DynamicAdjacencyArray adjacency_array(
    7, {{1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6}});
  adjacency_array.contract(0, 1);
  adjacency_array.contract(0, 2);
  adjacency_array.uncontract(0, 1);
  adjacency_array.uncontract(0, 2);
  verifyNeighbors(0, 7, adjacency_array, { });
  verifyNeighbors(1, 7, adjacency_array, { 2, 4 });
  verifyNeighbors(2, 7, adjacency_array, { 1, 3 });
  ASSERT_EQ(0, adjacency_array.nodeDegree(0));
  ASSERT_EQ(2, adjacency_array.nodeDegree(1));
  ASSERT_EQ(2, adjacency_array.nodeDegree(2));
}

TEST(ADynamicAdjacencyArray, UncontractsVerticesInDifferentOrder2) {
  DynamicAdjacencyArray adjacency_array(
    7, {{1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6}});
  adjacency_array.contract(4, 5);
  adjacency_array.contract(4, 6);
  adjacency_array.uncontract(4, 5);
  adjacency_array.uncontract(4, 6);
  verifyNeighbors(4, 7, adjacency_array, { 1, 5, 6 });
  verifyNeighbors(5, 7, adjacency_array, { 4, 6 });
  verifyNeighbors(6, 7, adjacency_array, { 4, 5 });
  ASSERT_EQ(3, adjacency_array.nodeDegree(4));
  ASSERT_EQ(2, adjacency_array.nodeDegree(5));
  ASSERT_EQ(2, adjacency_array.nodeDegree(6));
}

TEST(ADynamicAdjacencyArray, UncontractsVerticesInDifferentOrder3) {
  DynamicAdjacencyArray adjacency_array(
    7, {{1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6}});
  adjacency_array.contract(3, 2);
  adjacency_array.contract(3, 1);
  adjacency_array.contract(3, 4);
  adjacency_array.uncontract(3, 2);
  adjacency_array.uncontract(3, 1);
  adjacency_array.uncontract(3, 4);
  verifyNeighbors(1, 7, adjacency_array, { 2, 4 });
  verifyNeighbors(2, 7, adjacency_array, { 1, 3 });
  verifyNeighbors(3, 7, adjacency_array, { 2 });
  verifyNeighbors(4, 7, adjacency_array, { 1, 5, 6 });
  ASSERT_EQ(2, adjacency_array.nodeDegree(1));
  ASSERT_EQ(2, adjacency_array.nodeDegree(2));
  ASSERT_EQ(1, adjacency_array.nodeDegree(3));
  ASSERT_EQ(3, adjacency_array.nodeDegree(4));
}

TEST(ADynamicAdjacencyArray, RemovesParrallelEdges1) {
  DynamicAdjacencyArray adjacency_array(
    7, {{1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6}});
  adjacency_array.contract(2, 4);
  adjacency_array.removeSinglePinAndParallelEdges();
  verifyNeighbors(1, 7, adjacency_array, { 2 }, true);
  verifyNeighbors(2, 7, adjacency_array, { 1, 3, 5, 6 }, true);
  verifyEdgeWeight(adjacency_array, 1, 2, 2);
}

TEST(ADynamicAdjacencyArray, RemovesParrallelEdges2) {
  DynamicAdjacencyArray adjacency_array(
    7, {{1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6}});
  adjacency_array.contract(5, 6);
  adjacency_array.removeSinglePinAndParallelEdges();
  verifyNeighbors(5, 7, adjacency_array, { 4 }, true);
  verifyNeighbors(4, 7, adjacency_array, { 1, 5 }, true);
  verifyEdgeWeight(adjacency_array, 4, 5, 2);
}

TEST(ADynamicAdjacencyArray, RemovesParrallelEdges3) {
  DynamicAdjacencyArray adjacency_array(
    7, {{1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6}});
  adjacency_array.contract(4, 2);
  adjacency_array.contract(4, 5);
  adjacency_array.removeSinglePinAndParallelEdges();
  verifyNeighbors(1, 7, adjacency_array, { 4 }, true);
  verifyNeighbors(3, 7, adjacency_array, { 4 }, true);
  verifyNeighbors(4, 7, adjacency_array, { 1, 3, 6 }, true);
  verifyNeighbors(6, 7, adjacency_array, { 4 }, true);
  verifyEdgeWeight(adjacency_array, 1, 4, 2);
  verifyEdgeWeight(adjacency_array, 4, 6, 2);
}

TEST(ADynamicAdjacencyArray, RemovesParrallelEdges4) {
  DynamicAdjacencyArray adjacency_array(
    7, {{1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6}});
  adjacency_array.contract(2, 4);
  adjacency_array.contract(5, 1);
  adjacency_array.contract(2, 0);
  adjacency_array.contract(5, 3);
  adjacency_array.contract(2, 6);
  adjacency_array.removeSinglePinAndParallelEdges();
  verifyNeighbors(2, 7, adjacency_array, { 5 }, true);
  verifyNeighbors(5, 7, adjacency_array, { 2 }, true);
  verifyEdgeWeight(adjacency_array, 2, 5, 5);
}

TEST(ADynamicAdjacencyArray, RestoresParrallelEdges1) {
  DynamicAdjacencyArray adjacency_array(
    7, {{1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6}});
  adjacency_array.contract(2, 4);
  auto edges_to_restore = adjacency_array.removeSinglePinAndParallelEdges();
  adjacency_array.restoreSinglePinAndParallelEdges(edges_to_restore);
  adjacency_array.uncontract(2, 4);
  verifyNeighbors(0, 7, adjacency_array, { });
  verifyNeighbors(1, 7, adjacency_array, { 2, 4 });
  verifyNeighbors(2, 7, adjacency_array, { 1, 3 });
  verifyNeighbors(3, 7, adjacency_array, { 2 });
  verifyNeighbors(4, 7, adjacency_array, { 1, 5, 6 });
  verifyNeighbors(5, 7, adjacency_array, { 4, 6 });
  verifyNeighbors(6, 7, adjacency_array, { 4, 5 });

  for (HyperedgeID e: adjacency_array.edges()) {
    ASSERT_EQ(1, adjacency_array.edge(e).weight);
  }
}

TEST(ADynamicAdjacencyArray, RestoresParrallelEdges2) {
  DynamicAdjacencyArray adjacency_array(
    7, {{1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6}});
  adjacency_array.contract(2, 4);
  adjacency_array.contract(5, 1);
  auto edges_to_restore1 = adjacency_array.removeSinglePinAndParallelEdges();
  adjacency_array.contract(2, 0);
  adjacency_array.contract(5, 3);
  auto edges_to_restore2 = adjacency_array.removeSinglePinAndParallelEdges();
  adjacency_array.contract(2, 6);
  adjacency_array.contract(5, 2);
  auto edges_to_restore3 = adjacency_array.removeSinglePinAndParallelEdges();
  adjacency_array.restoreSinglePinAndParallelEdges(edges_to_restore3);
  adjacency_array.uncontract(5, 2);
  adjacency_array.uncontract(2, 6);
  adjacency_array.restoreSinglePinAndParallelEdges(edges_to_restore2);
  adjacency_array.uncontract(5, 3);
  adjacency_array.uncontract(2, 0);
  adjacency_array.restoreSinglePinAndParallelEdges(edges_to_restore1);
  adjacency_array.uncontract(5, 1);
  adjacency_array.uncontract(2, 4);

  verifyNeighbors(0, 7, adjacency_array, { });
  verifyNeighbors(1, 7, adjacency_array, { 2, 4 });
  verifyNeighbors(2, 7, adjacency_array, { 1, 3 });
  verifyNeighbors(3, 7, adjacency_array, { 2 });
  verifyNeighbors(4, 7, adjacency_array, { 1, 5, 6 });
  verifyNeighbors(5, 7, adjacency_array, { 4, 6 });
  verifyNeighbors(6, 7, adjacency_array, { 4, 5 });

  for (HyperedgeID e: adjacency_array.edges()) {
    ASSERT_EQ(1, adjacency_array.edge(e).weight);
  }
}

}  // namespace ds
}  // namespace mt_kahypar
