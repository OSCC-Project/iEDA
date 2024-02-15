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

#include "tests/definitions.h"
#include "mt-kahypar/io/hypergraph_factory.h"
#include "mt-kahypar/partition/context_enum_classes.h"

using ::testing::Test;

namespace mt_kahypar {
namespace io {

template<typename Hypergraph>
class AnInputReader : public Test {

 public:
  AnInputReader() :
    hypergraph() { }

  void readHypergraph(const std::string& filename, const FileFormat format) {
    hypergraph = readInputFile<Hypergraph>(filename, format, true);
  }

  void verifyIncidentNets(const std::vector< std::set<HyperedgeID> >& references) {
    ASSERT(hypergraph.initialNumNodes() == references.size());
    for (HypernodeID hn = 0; hn < hypergraph.initialNumNodes(); ++hn) {
      const std::set<HyperedgeID>& reference = references[hn];
      size_t count = 0;
      for (const HyperedgeID& he : hypergraph.incidentEdges(hn)) {
        ASSERT_TRUE(reference.find(he) != reference.end()) << V(hn) << V(he);
        count++;
      }
      ASSERT_EQ(count, reference.size());
    }
  }


  void verifyPins(const std::vector< std::set<HypernodeID> >& references) {
    ASSERT(hypergraph.initialNumEdges() == references.size());
    for (HyperedgeID he = 0; he < hypergraph.initialNumEdges(); ++he) {
      const std::set<HypernodeID>& reference = references[he];
      size_t count = 0;
      for (const HypernodeID& pin : hypergraph.pins(he)) {
        ASSERT_TRUE(reference.find(pin) != reference.end()) << V(he) << V(pin);
        count++;
      }
      ASSERT_EQ(count, reference.size());
    }
  }

  void verifyNeighbors(const std::vector<std::set<HypernodeID>> references) {
    ASSERT(hypergraph.initialNumNodes() == references.size());
    for ( const HypernodeID& hn : hypergraph.nodes() ) {
      size_t cnt = 0;
      const std::set<HypernodeID>& reference = references[hn];
      for ( const HyperedgeID& he : hypergraph.incidentEdges(hn) ) {
        ASSERT_EQ(hypergraph.edgeSize(he), 2);
        for ( const HypernodeID& pin : hypergraph.pins(he) ) {
          if ( pin != hn ) {
            ASSERT_TRUE(reference.find(pin) != reference.end()) << V(he) << V(pin);
            ++cnt;
          }
        }
      }
      ASSERT_EQ(cnt, reference.size()) << V(hn);
    }
  }

  void verifyNeighborsAndEdgeWeights(const std::vector<std::set<std::pair<HypernodeID,HyperedgeWeight>>> references) {
    ASSERT(hypergraph.initialNumNodes() == references.size());
    for ( const HypernodeID& hn : hypergraph.nodes() ) {
      size_t cnt = 0;
      const std::set<std::pair<HypernodeID,HyperedgeWeight>>& reference = references[hn];
      for ( const HyperedgeID& he : hypergraph.incidentEdges(hn) ) {
        ASSERT_EQ(hypergraph.edgeSize(he), 2);
        const HyperedgeWeight w = hypergraph.edgeWeight(he);
        for ( const HypernodeID& target : hypergraph.pins(he) ) {
          if ( target != hn ) {
            ASSERT_TRUE(reference.find({target, w}) != reference.end()) << V(he) << V(hn) << V(target) << V(w);
            ++cnt;
          }
        }
      }
      ASSERT_EQ(cnt, reference.size()) << V(hn);
    }
  }

  Hypergraph hypergraph;
};

template<typename Hypergraph>
using AHypergraphReader = AnInputReader<Hypergraph>;
TYPED_TEST_CASE(AHypergraphReader, tests::HypergraphTestTypes);

template<typename Graph>
using AGraphReader = AnInputReader<Graph>;
TYPED_TEST_CASE(AGraphReader, tests::GraphAndHypergraphTestTypes);

TYPED_TEST(AHypergraphReader, ReadsAnUnweightedHypergraph) {
  this->readHypergraph("../tests/instances/unweighted_hypergraph.hgr", FileFormat::hMetis);

  // Verify Incident Nets
  this->verifyIncidentNets(
    { { 0, 1 }, { 1 }, { 0, 3 }, { 1, 2 },
      {1, 2}, { 3 }, { 2, 3 } });

  // Verify Pins
  this->verifyPins({ { 0, 2 }, { 0, 1, 3, 4 },
    { 3, 4, 6 }, { 2, 5, 6 } });

  // Verify Node Weights
  ASSERT_EQ(1, this->hypergraph.nodeWeight(0));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(1));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(2));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(3));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(4));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(5));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(6));

  // Verify Edge Weights
  ASSERT_EQ(1, this->hypergraph.edgeWeight(0));
  ASSERT_EQ(1, this->hypergraph.edgeWeight(1));
  ASSERT_EQ(1, this->hypergraph.edgeWeight(2));
  ASSERT_EQ(1, this->hypergraph.edgeWeight(3));
}

TYPED_TEST(AHypergraphReader, ReadsAnHypergraphWithEdgeWeights) {
  this->readHypergraph("../tests/instances/hypergraph_with_edge_weights.hgr", FileFormat::hMetis);

  // Verify Incident Nets
  this->verifyIncidentNets(
    { { 0, 1 }, { 1 }, { 0, 3 }, { 1, 2 },
      {1, 2}, { 3 }, { 2, 3 } });

  // Verify Pins
  this->verifyPins({ { 0, 2 }, { 0, 1, 3, 4 },
    { 3, 4, 6 }, { 2, 5, 6 } });

  // Verify Node Weights
  ASSERT_EQ(1, this->hypergraph.nodeWeight(0));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(1));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(2));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(3));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(4));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(5));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(6));

  // Verify Edge Weights
  ASSERT_EQ(4, this->hypergraph.edgeWeight(0));
  ASSERT_EQ(2, this->hypergraph.edgeWeight(1));
  ASSERT_EQ(3, this->hypergraph.edgeWeight(2));
  ASSERT_EQ(8, this->hypergraph.edgeWeight(3));
}

TYPED_TEST(AHypergraphReader, ReadsAnHypergraphWithNodeWeights) {
  this->readHypergraph("../tests/instances/hypergraph_with_node_weights.hgr", FileFormat::hMetis);

  // Verify Incident Nets
  this->verifyIncidentNets(
    { { 0, 1 }, { 1 }, { 0, 3 }, { 1, 2 },
      {1, 2}, { 3 }, { 2, 3 } });

  // Verify Pins
  this->verifyPins({ { 0, 2 }, { 0, 1, 3, 4 },
    { 3, 4, 6 }, { 2, 5, 6 } });

  // Verify Node Weights
  ASSERT_EQ(5, this->hypergraph.nodeWeight(0));
  ASSERT_EQ(8, this->hypergraph.nodeWeight(1));
  ASSERT_EQ(2, this->hypergraph.nodeWeight(2));
  ASSERT_EQ(3, this->hypergraph.nodeWeight(3));
  ASSERT_EQ(4, this->hypergraph.nodeWeight(4));
  ASSERT_EQ(9, this->hypergraph.nodeWeight(5));
  ASSERT_EQ(8, this->hypergraph.nodeWeight(6));

  // Verify Edge Weights
  ASSERT_EQ(1, this->hypergraph.edgeWeight(0));
  ASSERT_EQ(1, this->hypergraph.edgeWeight(1));
  ASSERT_EQ(1, this->hypergraph.edgeWeight(2));
  ASSERT_EQ(1, this->hypergraph.edgeWeight(3));
}

TYPED_TEST(AHypergraphReader, ReadsAnHypergraphWithNodeAndEdgeWeights) {
  this->readHypergraph("../tests/instances/hypergraph_with_node_and_edge_weights.hgr", FileFormat::hMetis);

  // Verify Incident Nets
  this->verifyIncidentNets(
    { { 0, 1 }, { 1 }, { 0, 3 }, { 1, 2 },
      {1, 2}, { 3 }, { 2, 3 } });

  // Verify Pins
  this->verifyPins({ { 0, 2 }, { 0, 1, 3, 4 },
    { 3, 4, 6 }, { 2, 5, 6 } });

  // Verify Node Weights
  ASSERT_EQ(5, this->hypergraph.nodeWeight(0));
  ASSERT_EQ(8, this->hypergraph.nodeWeight(1));
  ASSERT_EQ(2, this->hypergraph.nodeWeight(2));
  ASSERT_EQ(3, this->hypergraph.nodeWeight(3));
  ASSERT_EQ(4, this->hypergraph.nodeWeight(4));
  ASSERT_EQ(9, this->hypergraph.nodeWeight(5));
  ASSERT_EQ(8, this->hypergraph.nodeWeight(6));

  // Verify Edge Weights
  ASSERT_EQ(4, this->hypergraph.edgeWeight(0));
  ASSERT_EQ(2, this->hypergraph.edgeWeight(1));
  ASSERT_EQ(3, this->hypergraph.edgeWeight(2));
  ASSERT_EQ(8, this->hypergraph.edgeWeight(3));
}

TYPED_TEST(AGraphReader, ReadsAMetisGraph) {
  this->readHypergraph("../tests/instances/unweighted_graph.graph", FileFormat::Metis);

  // Verify Neighbors
  this->verifyNeighbors(
    { { 1, 2, 4 },
      { 0, 2, 3 },
      { 0, 1, 3, 4 },
      { 1, 2, 5, 6 },
      { 0, 2, 5 },
      { 3, 4, 6 },
      { 3, 5 },
      { } } );

  // Verify Node Weights
  ASSERT_EQ(1, this->hypergraph.nodeWeight(0));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(1));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(2));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(3));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(4));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(5));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(6));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(7));

  // Verify Edge Weights
  for ( HyperedgeID e : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10} ) {
    ASSERT_EQ(1, this->hypergraph.edgeWeight(e));
  }
}

TYPED_TEST(AGraphReader, ReadsAMetisGraphWithNodeWeights) {
  this->readHypergraph("../tests/instances/graph_with_node_weights.graph", FileFormat::Metis);

  // Verify Neighbors
  this->verifyNeighbors(
    { { 1, 2, 4 },
      { 0, 2, 3 },
      { 0, 1, 3, 4 },
      { 1, 2, 5, 6 },
      { 0, 2, 5 },
      { 3, 4, 6 },
      { 3, 5 },
      { } } );

  // Verify Node Weights
  ASSERT_EQ(4, this->hypergraph.nodeWeight(0));
  ASSERT_EQ(2, this->hypergraph.nodeWeight(1));
  ASSERT_EQ(5, this->hypergraph.nodeWeight(2));
  ASSERT_EQ(3, this->hypergraph.nodeWeight(3));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(4));
  ASSERT_EQ(6, this->hypergraph.nodeWeight(5));
  ASSERT_EQ(2, this->hypergraph.nodeWeight(6));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(7));

  // Verify Edge Weights
  for ( HyperedgeID e : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10} ) {
    ASSERT_EQ(1, this->hypergraph.edgeWeight(e));
  }
}

TYPED_TEST(AGraphReader, ReadsAMetisGraphWithEdgeWeights) {
  this->readHypergraph("../tests/instances/graph_with_edge_weights.graph", FileFormat::Metis);

  // Verify Neighbors and Edge Weights
  this->verifyNeighborsAndEdgeWeights(
    { { { 1, 1 }, { 2, 2 }, { 4, 1 } },
      { { 0, 1 }, { 2, 2 }, { 3, 1 } },
      { { 0, 2 }, { 1, 2 }, { 3, 2 }, { 4, 3 } },
      { { 1, 1 }, { 2, 2 }, { 5, 2 }, { 6, 5 } },
      { { 0, 1 }, { 2, 3 }, { 5, 2 } },
      { { 3, 2 }, { 4, 2 }, { 6, 6 } },
      { { 3, 5 }, { 5, 6 } },
      { } } );

  // Verify Node Weights
  ASSERT_EQ(1, this->hypergraph.nodeWeight(0));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(1));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(2));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(3));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(4));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(5));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(6));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(7));
}

TYPED_TEST(AGraphReader, ReadsAMetisGraphWithNodeAndEdgeWeights) {
  this->readHypergraph("../tests/instances/graph_with_node_and_edge_weights.graph", FileFormat::Metis);

  // Verify Neighbors and Edge Weights
  this->verifyNeighborsAndEdgeWeights(
    { { { 1, 1 }, { 2, 2 }, { 4, 1 } },
      { { 0, 1 }, { 2, 2 }, { 3, 1 } },
      { { 0, 2 }, { 1, 2 }, { 3, 2 }, { 4, 3 } },
      { { 1, 1 }, { 2, 2 }, { 5, 2 }, { 6, 5 } },
      { { 0, 1 }, { 2, 3 }, { 5, 2 } },
      { { 3, 2 }, { 4, 2 }, { 6, 6 } },
      { { 3, 5 }, { 5, 6 } },
      { } } );

  // Verify Node Weights
  ASSERT_EQ(4, this->hypergraph.nodeWeight(0));
  ASSERT_EQ(2, this->hypergraph.nodeWeight(1));
  ASSERT_EQ(5, this->hypergraph.nodeWeight(2));
  ASSERT_EQ(3, this->hypergraph.nodeWeight(3));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(4));
  ASSERT_EQ(6, this->hypergraph.nodeWeight(5));
  ASSERT_EQ(2, this->hypergraph.nodeWeight(6));
  ASSERT_EQ(1, this->hypergraph.nodeWeight(7));
}

}  // namespace io
}  // namespace mt_kahypar
