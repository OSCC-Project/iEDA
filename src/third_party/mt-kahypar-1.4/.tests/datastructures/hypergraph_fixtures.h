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

#pragma once

#include "gmock/gmock.h"

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/parallel/hardware_topology.h"
#include "mt-kahypar/parallel/tbb_initializer.h"
#include "tests/parallel/topology_mock.h"

using ::testing::Test;

namespace mt_kahypar {
namespace ds {

static auto identity = [](const HypernodeID& id) { return id; };

template<typename Hypergraph, bool useGraphStructure = false>
class HypergraphFixture : public Test {

  using HypergraphFactory = typename Hypergraph::Factory;

 public:
  HypergraphFixture() :
    hypergraph(useGraphStructure ?
      HypergraphFactory::construct(
        7 , 6, { {1, 2}, {2, 3}, {1, 4}, {4, 5}, {4, 6}, {5, 6} }, nullptr, nullptr, true) :
      HypergraphFactory::construct(
        7 , 4, { {0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6} }, nullptr, nullptr, true)) {
  }

  template <typename K = decltype(identity)>
  void verifyIncidentNets(const Hypergraph& hg,
                          const HypernodeID hn,
                          const std::set<HypernodeID>& reference,
                          K map_func = identity,
                          bool log = false) {
    size_t count = 0;
    for (const HyperedgeID& he : hg.incidentEdges(hn)) {
      if (log) LOG << V(he) << V(map_func(he));
      ASSERT_TRUE(reference.find(map_func(he)) != reference.end()) << V(map_func(he));
      count++;
    }
    ASSERT_EQ(count, reference.size());
  }

  template <typename K = decltype(identity)>
  void verifyIncidentNets(const HypernodeID hn,
                          const std::set<HypernodeID>& reference,
                          K map_func = identity,
                          bool log = false) {
    verifyIncidentNets(hypergraph, hn, reference, map_func, log);
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

  void verifyPins(const std::vector<HyperedgeID> hyperedges,
                  const std::vector< std::set<HypernodeID> >& references,
                  bool log = false) {
    verifyPins(hypergraph, hyperedges, references, log);
  }

  void assignCommunityIds() {
    hypergraph.setCommunityID(0, 0);
    hypergraph.setCommunityID(1, 0);
    hypergraph.setCommunityID(2, 0);
    hypergraph.setCommunityID(3, 1);
    hypergraph.setCommunityID(4, 1);
    hypergraph.setCommunityID(5, 2);
    hypergraph.setCommunityID(6, 2);
  }

  Hypergraph hypergraph;
};

}  // namespace ds
}  // namespace mt_kahypar
