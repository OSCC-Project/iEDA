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

#include "mt-kahypar/datastructures/incident_net_array.h"
#include "mt-kahypar/parallel/atomic_wrapper.h"

using ::testing::Test;

namespace mt_kahypar {
namespace ds {

void verifyIncidentNets(const HypernodeID u,
                        const HyperedgeID num_hyperedges,
                        const IncidentNetArray& incident_nets,
                        const std::set<HyperedgeID>& _expected_incident_nets) {
  size_t num_incident_edges = 0;
  std::vector<bool> actual_incident_edges(num_hyperedges, false);
  for ( const HyperedgeID& he : incident_nets.incidentEdges(u) ) {
    ASSERT_TRUE(_expected_incident_nets.find(he) != _expected_incident_nets.end())
      << "Hyperedge " << he << " should be not part of incident nets of vertex " << u;
    ASSERT_FALSE(actual_incident_edges[he])
      << "Hyperedge " << he << " occurs more than once in incident nets of vertex " << u;
    actual_incident_edges[he] = true;
    ++num_incident_edges;
  }
  ASSERT_EQ(num_incident_edges, _expected_incident_nets.size());
  ASSERT_EQ(num_incident_edges, incident_nets.nodeDegree(u));
}

kahypar::ds::FastResetFlagArray<> createFlagArray(const HyperedgeID num_hyperedges,
                                                  const std::vector<HyperedgeID>& contained_hes) {
  kahypar::ds::FastResetFlagArray<> flag_array(num_hyperedges);
  for ( const HyperedgeID& he : contained_hes ) {
    flag_array.set(he, true);
  }
  return flag_array;
}

TEST(AIncidentNetArray, VerifyInitialIncidentNetsOfEachVertex) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  verifyIncidentNets(0, 4, incident_nets, { 0, 1 });
  verifyIncidentNets(1, 4, incident_nets, { 1 });
  verifyIncidentNets(2, 4, incident_nets, { 0, 3 });
  verifyIncidentNets(3, 4, incident_nets, { 1, 2 });
  verifyIncidentNets(4, 4, incident_nets, { 1, 2 });
  verifyIncidentNets(5, 4, incident_nets, { 3 });
  verifyIncidentNets(6, 4, incident_nets, { 2, 3 });
}

TEST(AIncidentNetArray, ContractTwoVertices1) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.contract(3, 4, createFlagArray(4, { 1, 2 }));
  verifyIncidentNets(3, 4, incident_nets, { 1, 2 });
}

TEST(AIncidentNetArray, ContractTwoVertices2) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.contract(0, 2, createFlagArray(4, { 0 }));
  verifyIncidentNets(0, 4, incident_nets, { 0, 1, 3 });
}

TEST(AIncidentNetArray, ContractTwoVertices3) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.contract(0, 6, createFlagArray(4, { }));
  verifyIncidentNets(0, 4, incident_nets, { 0, 1, 2, 3 });
}

TEST(AIncidentNetArray, ContractSeveralVertices1) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.contract(3, 4, createFlagArray(4, { 1, 2 }));
  incident_nets.contract(3, 0, createFlagArray(4, { 1 }));
  verifyIncidentNets(3, 4, incident_nets, { 0, 1, 2 });
}

TEST(AIncidentNetArray, ContractSeveralVertices2) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.contract(1, 5, createFlagArray(4, { }));
  incident_nets.contract(4, 1, createFlagArray(4, { 1 }));
  verifyIncidentNets(4, 4, incident_nets, { 1, 2, 3 });
}

TEST(AIncidentNetArray, ContractSeveralVertices3) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.contract(0, 3, createFlagArray(4, { 1 }));
  incident_nets.contract(0, 5, createFlagArray(4, { }));
  incident_nets.contract(0, 6, createFlagArray(4, { 2, 3 }));
  verifyIncidentNets(0, 4, incident_nets, { 0, 1, 2, 3 });
}

TEST(AIncidentNetArray, ContractSeveralVertices4) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.contract(0, 2, createFlagArray(4, { 0 }));
  incident_nets.contract(3, 4, createFlagArray(4, { 1, 2 }));
  incident_nets.contract(5, 6, createFlagArray(4, { 3 }));
  verifyIncidentNets(0, 4, incident_nets, { 0, 1, 3 });
  verifyIncidentNets(3, 4, incident_nets, { 1, 2 });
  verifyIncidentNets(5, 4, incident_nets, { 2, 3 });
  incident_nets.contract(0, 3, createFlagArray(4, { 1 }));
  verifyIncidentNets(0, 4, incident_nets, { 0, 1, 2, 3 });
  incident_nets.contract(0, 5, createFlagArray(4, { 2, 3 }));
  verifyIncidentNets(0, 4, incident_nets, { 0, 1, 2, 3 });
}

TEST(AIncidentNetArray, UncontractTwoVertices1) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.contract(3, 4, createFlagArray(4, { 1, 2 }));
  incident_nets.uncontract(3, 4);
  verifyIncidentNets(3, 4, incident_nets, { 1, 2 });
  verifyIncidentNets(4, 4, incident_nets, { 1, 2 });
}

TEST(AIncidentNetArray, UnontractTwoVertices2) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.contract(0, 2, createFlagArray(4, { 0 }));
  incident_nets.uncontract(0, 2);
  verifyIncidentNets(0, 4, incident_nets, { 0, 1 });
  verifyIncidentNets(2, 4, incident_nets, { 0, 3 });
}

TEST(AIncidentNetArray, UncontractTwoVertices3) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.contract(0, 6, createFlagArray(4, { }));
  incident_nets.uncontract(0, 6);
  verifyIncidentNets(0, 4, incident_nets, { 0, 1 });
  verifyIncidentNets(6, 4, incident_nets, { 2, 3 });
}

TEST(AIncidentNetArray, UncontractSeveralVertices1) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.contract(3, 4, createFlagArray(4, { 1, 2 }));
  incident_nets.contract(3, 0, createFlagArray(4, { 1 }));
  incident_nets.uncontract(3, 0);
  verifyIncidentNets(0, 4, incident_nets, { 0, 1 });
  verifyIncidentNets(3, 4, incident_nets, { 1, 2 });
  incident_nets.uncontract(3, 4);
  verifyIncidentNets(0, 4, incident_nets, { 0, 1 });
  verifyIncidentNets(3, 4, incident_nets, { 1, 2 });
  verifyIncidentNets(4, 4, incident_nets, { 1, 2 });
}

TEST(AIncidentNetArray, UncontractSeveralVertices2) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.contract(1, 5, createFlagArray(4, { }));
  incident_nets.contract(4, 1, createFlagArray(4, { 1 }));
  incident_nets.uncontract(4, 1);
  verifyIncidentNets(1, 4, incident_nets, { 1, 3 });
  verifyIncidentNets(4, 4, incident_nets, { 1, 2 });
  incident_nets.uncontract(1, 5);
  verifyIncidentNets(1, 4, incident_nets, { 1 });
  verifyIncidentNets(4, 4, incident_nets, { 1, 2 });
  verifyIncidentNets(5, 4, incident_nets, { 3 });
}

TEST(AIncidentNetArray, UncontractSeveralVertices3) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.contract(0, 3, createFlagArray(4, { 1 }));
  incident_nets.contract(0, 5, createFlagArray(4, { }));
  incident_nets.contract(0, 6, createFlagArray(4, { 2, 3 }));
  incident_nets.uncontract(0, 6);
  verifyIncidentNets(0, 4, incident_nets, { 0, 1, 2, 3 });
  verifyIncidentNets(6, 4, incident_nets, { 2, 3 });
  incident_nets.uncontract(0, 5);
  verifyIncidentNets(0, 4, incident_nets, { 0, 1, 2 });
  verifyIncidentNets(5, 4, incident_nets, { 3 });
  verifyIncidentNets(6, 4, incident_nets, { 2, 3 });
  incident_nets.uncontract(0, 3);
  verifyIncidentNets(0, 4, incident_nets, { 0, 1 });
  verifyIncidentNets(3, 4, incident_nets, { 1, 2 });
  verifyIncidentNets(5, 4, incident_nets, { 3 });
  verifyIncidentNets(6, 4, incident_nets, { 2, 3 });
}

TEST(AIncidentNetArray, UncontractSeveralVertices4) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.contract(0, 2, createFlagArray(4, { 0 }));
  incident_nets.contract(3, 4, createFlagArray(4, { 1, 2 }));
  incident_nets.contract(5, 6, createFlagArray(4, { 3 }));
  incident_nets.contract(0, 3, createFlagArray(4, { 1 }));
  incident_nets.contract(0, 5, createFlagArray(4, { 2, 3 }));
  incident_nets.uncontract(0, 5);
  verifyIncidentNets(0, 4, incident_nets, { 0, 1, 2, 3 });
  verifyIncidentNets(5, 4, incident_nets, { 2, 3 });
  incident_nets.uncontract(0, 3);
  verifyIncidentNets(0, 4, incident_nets, { 0, 1, 3 });
  verifyIncidentNets(3, 4, incident_nets, { 1, 2 });
  verifyIncidentNets(5, 4, incident_nets, { 2, 3 });
  incident_nets.uncontract(5, 6);
  verifyIncidentNets(0, 4, incident_nets, { 0, 1, 3 });
  verifyIncidentNets(3, 4, incident_nets, { 1, 2 });
  verifyIncidentNets(5, 4, incident_nets, { 3 });
  verifyIncidentNets(6, 4, incident_nets, { 2, 3 });
  incident_nets.uncontract(3, 4);
  verifyIncidentNets(0, 4, incident_nets, { 0, 1, 3 });
  verifyIncidentNets(3, 4, incident_nets, { 1, 2 });
  verifyIncidentNets(4, 4, incident_nets, { 1, 2 });
  verifyIncidentNets(5, 4, incident_nets, { 3 });
  verifyIncidentNets(6, 4, incident_nets, { 2, 3 });
  incident_nets.uncontract(0, 2);
  verifyIncidentNets(0, 4, incident_nets, { 0, 1 });
  verifyIncidentNets(2, 4, incident_nets, { 0, 3 });
  verifyIncidentNets(3, 4, incident_nets, { 1, 2 });
  verifyIncidentNets(4, 4, incident_nets, { 1, 2 });
  verifyIncidentNets(5, 4, incident_nets, { 3 });
  verifyIncidentNets(6, 4, incident_nets, { 2, 3 });
}

using OwnershipVector = parallel::scalable_vector<SpinLock>;

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

TEST(AIncidentNetArray, ContractsParallel1) {
  OwnershipVector acquired_hns(7);
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});

  executeParallel([&] {
    incident_nets.contract(0, 1, createFlagArray(4, { 1 }),
      [&](const HypernodeID u) {
        acquired_hns[u].lock();
      }, [&](const HypernodeID u) {
        acquired_hns[u].unlock();
      });
  }, [&] {
    incident_nets.contract(0, 5, createFlagArray(4, {}),
      [&](const HypernodeID u) {
        acquired_hns[u].lock();
      }, [&](const HypernodeID u) {
        acquired_hns[u].unlock();
      });
  });

  verifyIncidentNets(0, 4, incident_nets, { 0, 1, 3 });
}

TEST(AIncidentNetArray, ContractsParallel2) {
  OwnershipVector acquired_hns(7);
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});

  executeParallel([&] {
    incident_nets.contract(0, 1, createFlagArray(4, { 1 }),
      [&](const HypernodeID u) {
        acquired_hns[u].lock();
      }, [&](const HypernodeID u) {
        acquired_hns[u].unlock();
      });
  }, [&] {
    incident_nets.contract(0, 6, createFlagArray(4, {}),
      [&](const HypernodeID u) {
        acquired_hns[u].lock();
      }, [&](const HypernodeID u) {
        acquired_hns[u].unlock();
      });
  });

  verifyIncidentNets(0, 4, incident_nets, { 0, 1, 2, 3 });
}

TEST(AIncidentNetArray, UnontractsParallel1) {
  OwnershipVector acquired_hns(7);
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.contract(0, 1, createFlagArray(4, { 1 }));
  incident_nets.contract(0, 5, createFlagArray(4, {}));

  executeParallel([&] {
    incident_nets.uncontract(0, 1,
      [&](const HypernodeID u) {
        acquired_hns[u].lock();
      }, [&](const HypernodeID u) {
        acquired_hns[u].unlock();
      });
  }, [&] {
    incident_nets.uncontract(0, 5,
      [&](const HypernodeID u) {
        acquired_hns[u].lock();
      }, [&](const HypernodeID u) {
        acquired_hns[u].unlock();
      });
  });

  verifyIncidentNets(0, 4, incident_nets, { 0, 1 });
  verifyIncidentNets(1, 4, incident_nets, { 1 });
  verifyIncidentNets(5, 4, incident_nets, { 3 });
}

TEST(AIncidentNetArray, UnontractsParallel2) {
  OwnershipVector acquired_hns(7);
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.contract(0, 1, createFlagArray(4, { 1 }));
  incident_nets.contract(0, 6, createFlagArray(4, {}));

  executeParallel([&] {
    incident_nets.uncontract(0, 1,
      [&](const HypernodeID u) {
        acquired_hns[u].lock();
      }, [&](const HypernodeID u) {
        acquired_hns[u].unlock();
      });
  }, [&] {
    incident_nets.uncontract(0, 6,
      [&](const HypernodeID u) {
        acquired_hns[u].lock();
      }, [&](const HypernodeID u) {
        acquired_hns[u].unlock();
      });
  });

  verifyIncidentNets(0, 4, incident_nets, { 0, 1 });
  verifyIncidentNets(1, 4, incident_nets, { 1 });
  verifyIncidentNets(6, 4, incident_nets, { 2, 3 });
}

TEST(AIncidentNetArray, RemovesIncidentNets1) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.removeIncidentNets(0, createFlagArray(4, { 1 }));
  verifyIncidentNets(0, 4, incident_nets, { 0 });
}

TEST(AIncidentNetArray, RemovesIncidentNets2) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.removeIncidentNets(5, createFlagArray(4, { 3 }));
  verifyIncidentNets(5, 4, incident_nets, { });
}

TEST(AIncidentNetArray, RemovesIncidentNets3) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.removeIncidentNets(3, createFlagArray(4, { 1 }));
  incident_nets.removeIncidentNets(4, createFlagArray(4, { 1 }));
  verifyIncidentNets(3, 4, incident_nets, { 2 });
  verifyIncidentNets(4, 4, incident_nets, { 2 });
}

TEST(AIncidentNetArray, RemovesIncidentNetsAfterContraction1) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.contract(0, 2, createFlagArray(4, { 0 }));
  incident_nets.removeIncidentNets(0, createFlagArray(4, { 1 }));
  verifyIncidentNets(0, 4, incident_nets, { 0, 3 });
}

TEST(AIncidentNetArray, RemovesIncidentNetsAfterContraction2) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.contract(0, 2, createFlagArray(4, { 0 }));
  incident_nets.contract(0, 6, createFlagArray(4, { 3 }));
  incident_nets.removeIncidentNets(0, createFlagArray(4, { 1, 3 }));
  verifyIncidentNets(0, 4, incident_nets, { 0, 2 });
}

TEST(AIncidentNetArray, RestoreIncidentNets1) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.removeIncidentNets(0, createFlagArray(4, { 1 }));
  incident_nets.restoreIncidentNets(0);
  verifyIncidentNets(0, 4, incident_nets, { 0, 1 });
}

TEST(AIncidentNetArray, RestoreIncidentNets2) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.removeIncidentNets(5, createFlagArray(4, { 3 }));
  incident_nets.restoreIncidentNets(5);
  verifyIncidentNets(5, 4, incident_nets, { 3 });
}

TEST(AIncidentNetArray, RestoreIncidentNets3) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.removeIncidentNets(3, createFlagArray(4, { 1 }));
  incident_nets.removeIncidentNets(4, createFlagArray(4, { 1 }));
  incident_nets.restoreIncidentNets(3);
  incident_nets.restoreIncidentNets(4);
  verifyIncidentNets(3, 4, incident_nets, { 1, 2 });
  verifyIncidentNets(4, 4, incident_nets, { 1, 2 });
}

TEST(AIncidentNetArray, RestoreIncidentNetsAfterContraction1) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.contract(0, 2, createFlagArray(4, { 0 }));
  incident_nets.removeIncidentNets(0, createFlagArray(4, { 1 }));
  incident_nets.restoreIncidentNets(0);
  verifyIncidentNets(0, 4, incident_nets, { 0, 1, 3 });
  incident_nets.uncontract(0, 2);
  verifyIncidentNets(0, 4, incident_nets, { 0, 1 });
  verifyIncidentNets(2, 4, incident_nets, { 0, 3 });
}

TEST(AIncidentNetArray, RestoreIncidentNetsAfterContraction2) {
  IncidentNetArray incident_nets(
    7, {{0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6}});
  incident_nets.contract(0, 2, createFlagArray(4, { 0 }));
  incident_nets.contract(0, 6, createFlagArray(4, { 3 }));
  incident_nets.removeIncidentNets(0, createFlagArray(4, { 1, 3 }));
  incident_nets.restoreIncidentNets(0);
  verifyIncidentNets(0, 4, incident_nets, { 0, 1, 2, 3 });
  incident_nets.uncontract(0, 6);
  incident_nets.uncontract(0, 2);
  verifyIncidentNets(0, 4, incident_nets, { 0, 1 });
  verifyIncidentNets(2, 4, incident_nets, { 0, 3 });
  verifyIncidentNets(6, 4, incident_nets, { 2, 3 });
}


}  // namespace ds
}  // namespace mt_kahypar
