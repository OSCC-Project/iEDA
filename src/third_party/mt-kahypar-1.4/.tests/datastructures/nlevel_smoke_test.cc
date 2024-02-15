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
#include <set>

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/datastructures/fixed_vertex_support.h"
#include "mt-kahypar/partition/refinement/gains/gain_cache_ptr.h"
#include "mt-kahypar/partition/metrics.h"
#include "mt-kahypar/utils/randomize.h"

namespace mt_kahypar {
namespace ds {

template<typename Hypergraph>
void verifyEqualityOfHypergraphs(const Hypergraph& e_hypergraph,
                                 const Hypergraph& a_hypergraph) {
  Hypergraph expected_hypergraph = e_hypergraph.copy();
  Hypergraph actual_hypergraph = a_hypergraph.copy();
  if constexpr ( Hypergraph::is_graph ) {
    expected_hypergraph.sortIncidentEdges();
    actual_hypergraph.sortIncidentEdges();
  }

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

    if constexpr ( Hypergraph::is_graph ) {
      for ( size_t i = 0; i < expected_incident_edges.size(); ++i ) {
        HyperedgeID exp = expected_incident_edges[i];
        HyperedgeID act = actual_incident_edges[i];
        ASSERT_EQ(expected_hypergraph.edgeSource(exp), actual_hypergraph.edgeSource(act));
        ASSERT_EQ(expected_hypergraph.edgeTarget(exp), actual_hypergraph.edgeTarget(act));
        ASSERT_EQ(expected_hypergraph.edgeWeight(exp), actual_hypergraph.edgeWeight(act));
      }
    } else {
      for ( size_t i = 0; i < expected_incident_edges.size(); ++i ) {
        ASSERT_EQ(expected_incident_edges[i], actual_incident_edges[i]);
      }
    }
    expected_incident_edges.clear();
    actual_incident_edges.clear();
  }

  if constexpr ( !Hypergraph::is_graph ) {
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
}

template<typename PartitionedHypergraph>
HyperedgeWeight compute_km1(PartitionedHypergraph& partitioned_hypergraph) {
  HyperedgeWeight km1 = 0;
  for (const HyperedgeID& he : partitioned_hypergraph.edges()) {
    km1 += std::max(partitioned_hypergraph.connectivity(he) - 1, 0) * partitioned_hypergraph.edgeWeight(he);
  }
  return PartitionedHypergraph::is_graph ? km1 / 2 : km1;
}

template<typename PartitionedHypergraph, typename GainCache>
void verifyGainCache(PartitionedHypergraph& partitioned_hypergraph,
                     GainCache& gain_cache) {
  const PartitionID k = partitioned_hypergraph.k();
  utils::Randomize& rand = utils::Randomize::instance();
  HyperedgeWeight km1_before = compute_km1(partitioned_hypergraph);
  HyperedgeWeight expected_gain = 0;
  for ( const HypernodeID& hn : partitioned_hypergraph.nodes() ) {
    const PartitionID from = partitioned_hypergraph.partID(hn);
    PartitionID to = rand.getRandomInt(0, k - 1, THREAD_ID);
    if ( from == to ) to = (to + 1) % k;
    expected_gain += gain_cache.gain(hn, from, to);
    partitioned_hypergraph.changeNodePart(gain_cache, hn, from, to);
  }
  HyperedgeWeight km1_after = compute_km1(partitioned_hypergraph);
  ASSERT_EQ(expected_gain, km1_before - km1_after) << V(expected_gain) << V(km1_before) << V(km1_after);
}

template<typename PartitionedHypergraph>
void verifyNumIncidentCutHyperedges(const PartitionedHypergraph& partitioned_hypergraph) {
  partitioned_hypergraph.doParallelForAllNodes([&](const HypernodeID& hn) {
    HypernodeID expected_num_cut_hyperedges = 0;
    for ( const HyperedgeID& he : partitioned_hypergraph.incidentEdges(hn) ) {
      if ( partitioned_hypergraph.connectivity(he) > 1 ) {
        ++expected_num_cut_hyperedges;
      }
    }
    ASSERT_EQ(expected_num_cut_hyperedges, partitioned_hypergraph.numIncidentCutHyperedges(hn));
  });
}

template<typename Hypergraph>
void verifyFixedVertices(const Hypergraph& hypergraph,
                         const ds::FixedVertexSupport<Hypergraph>& expected_fixed_vertices,
                         const ds::FixedVertexSupport<Hypergraph>& actual_fixed_vertices) {
  hypergraph.doParallelForAllNodes([&](const HypernodeID& hn) {
    ASSERT_EQ(expected_fixed_vertices.fixedVertexBlock(hn),
              actual_fixed_vertices.fixedVertexBlock(hn));
  });

  for ( PartitionID block = 0; block < expected_fixed_vertices.numBlocks(); ++block ) {
    ASSERT_EQ(expected_fixed_vertices.fixedVertexBlockWeight(block),
              actual_fixed_vertices.fixedVertexBlockWeight(block));
  }
}

template<typename Hypergraph>
Hypergraph generateRandomHypergraph(const HypernodeID num_hypernodes,
                                    const HyperedgeID num_hyperedges,
                                    const HypernodeID max_edge_size) {
  using Factory = typename Hypergraph::Factory;
  parallel::scalable_vector<parallel::scalable_vector<HypernodeID>> hyperedges;
  utils::Randomize& rand = utils::Randomize::instance();

  std::set<std::pair<HypernodeID, HypernodeID>> graph_edges;
  if constexpr ( Hypergraph::is_graph ) {
    for ( size_t i = 0; i < num_hypernodes; ++i ) {
      graph_edges.insert({i, i});
    }
  }

  for ( size_t i = 0; i < num_hyperedges; ++i ) {
    parallel::scalable_vector<HypernodeID> net;
    if constexpr ( Hypergraph::is_graph ) {
      unused(max_edge_size);
      std::pair<HypernodeID, HypernodeID> edge{rand.getRandomInt(0, num_hypernodes - 1, THREAD_ID),
                                               rand.getRandomInt(0, num_hypernodes - 1, THREAD_ID)};
      graph_edges.insert({edge});
      graph_edges.insert({edge.first, edge.second});
      net.push_back(edge.first);
      net.push_back(edge.second);
    } else {
      const size_t edge_size = rand.getRandomInt(2, max_edge_size, THREAD_ID);
      for ( size_t i = 0; i < edge_size; ++i ) {
        const HypernodeID pin = rand.getRandomInt(0, num_hypernodes - 1, THREAD_ID);
        if ( std::find(net.begin(), net.end(), pin) == net.end() ) {
          net.push_back(pin);
        }
      }
    }
    hyperedges.emplace_back(std::move(net));
  }
  return Factory::construct(num_hypernodes, num_hyperedges, hyperedges);
}

template<typename Hypergraph>
void addRandomFixedVertices(Hypergraph& hypergraph,
                            const PartitionID k,
                            const double percentage_fixed_vertices) {
  ds::FixedVertexSupport<Hypergraph> fixed_vertices(hypergraph.initialNumNodes(), k);
  fixed_vertices.setHypergraph(&hypergraph);
  utils::Randomize& rand = utils::Randomize::instance();
  const int threshold = percentage_fixed_vertices * 1000;
  for ( const HypernodeID& hn : hypergraph.nodes() ) {
    const bool is_fixed = rand.getRandomInt(0, 1000, THREAD_ID) <= threshold;
    if ( is_fixed ) {
      fixed_vertices.fixToBlock(hn, rand.getRandomInt(0, k - 1, THREAD_ID));
    }
  }
  hypergraph.addFixedVertexSupport(std::move(fixed_vertices));
}

BatchVector generateRandomContractions(const HypernodeID num_hypernodes,
                                       const HypernodeID num_contractions,
                                       const bool multi_versioned = true) {
 ASSERT(num_contractions < num_hypernodes);
 HypernodeID tmp_num_contractions = num_contractions;
 BatchVector contractions;
 parallel::scalable_vector<HypernodeID> active_hns(num_hypernodes);
 std::iota(active_hns.begin(), active_hns.end(), 0);
 utils::Randomize& rand = utils::Randomize::instance();
 const int cpu_id = THREAD_ID;
 while ( tmp_num_contractions > 0 ) {
   HypernodeID current_num_contractions = tmp_num_contractions;
   if ( multi_versioned && current_num_contractions > 25 ) current_num_contractions /= 2;
   contractions.emplace_back();
   for ( size_t i = 0; i < current_num_contractions; ++i ) {
    ASSERT(active_hns.size() >= 2UL);
    int idx_1 = rand.getRandomInt(0, static_cast<int>(active_hns.size() - 1), cpu_id);
    int idx_2 = rand.getRandomInt(0, static_cast<int>(active_hns.size() - 1), cpu_id);
      if ( idx_1 == idx_2 ) {
        idx_2 = (idx_2 + 1) % active_hns.size();
      }
      contractions.back().push_back(Memento { active_hns[idx_1], active_hns[idx_2] });
      std::swap(active_hns[idx_2], active_hns.back());
      active_hns.pop_back();
    }
   tmp_num_contractions -= current_num_contractions;
 }
 return contractions;
}

template<typename PartitionedHypergraph>
void generateRandomPartition(PartitionedHypergraph& partitioned_hypergraph) {
  const PartitionID k = partitioned_hypergraph.k();
  utils::Randomize& rand = utils::Randomize::instance();
  partitioned_hypergraph.doParallelForAllNodes([&](const HypernodeID& hn) {
    if ( partitioned_hypergraph.isFixed(hn) ) {
      partitioned_hypergraph.setOnlyNodePart(hn, partitioned_hypergraph.fixedVertexBlock(hn));
    } else {
      partitioned_hypergraph.setOnlyNodePart(hn, rand.getRandomInt(0, k - 1, THREAD_ID));
    }
  });
}

template<typename Hypergraph, typename PartitionedHypergraph, typename GainCache>
Hypergraph simulateNLevel(Hypergraph& hypergraph,
                          PartitionedHypergraph& partitioned_hypergraph,
                          GainCache& gain_cache,
                          const BatchVector& contraction_batches,
                          const size_t batch_size,
                          const bool parallel,
                          utils::Timer& timer) {
  using ParallelHyperedge = typename Hypergraph::ParallelHyperedge;
  using Factory = typename Hypergraph::Factory;

  auto timer_key = [&](const std::string& key) {
    if ( parallel ) {
      return key + "_parallel";
    } else {
      return key;
    }
  };

  parallel::scalable_vector<parallel::scalable_vector<ParallelHyperedge>> removed_hyperedges;
  for ( size_t i = 0; i < contraction_batches.size(); ++i ) {
    timer.start_timer(timer_key("contractions"), "Contractions");
    const parallel::scalable_vector<Memento>& contractions = contraction_batches[i];
    if ( parallel ) {
      tbb::parallel_for(UL(0), contractions.size(), [&](const size_t j) {
        const Memento& memento = contractions[j];
        hypergraph.registerContraction(memento.u, memento.v);
        hypergraph.contract(memento.v);
      });
    } else {
      for ( size_t j = 0; j < contractions.size(); ++j ) {
        const Memento& memento = contractions[j];
        hypergraph.registerContraction(memento.u, memento.v);
        hypergraph.contract(memento.v);
      }
    }
    timer.stop_timer(timer_key("contractions"));

    timer.start_timer(timer_key("remove_parallel_nets"), "Parallel Net Detection");
    removed_hyperedges.emplace_back(hypergraph.removeSinglePinAndParallelHyperedges());
    timer.stop_timer(timer_key("remove_parallel_nets"));
  }

  timer.start_timer(timer_key("copy_coarsest_hypergraph"), "Copy Coarsest Hypergraph");
  Hypergraph coarsest_hypergraph;
  if ( parallel ) {
    coarsest_hypergraph = hypergraph.copy(parallel_tag_t());
  } else {
    coarsest_hypergraph = hypergraph.copy();
  }
  timer.stop_timer(timer_key("copy_coarsest_hypergraph"));


  timer.start_timer(timer_key("initial_partition"), "Initial Partition");

  {
    timer.start_timer(timer_key("compactify_hypergraph"), "Compactify Hypergraph");
    auto res = Factory::compactify(hypergraph);
    Hypergraph& compactified_hg = res.first;
    auto& hn_mapping = res.second;
    PartitionedHypergraph compactified_phg(
      partitioned_hypergraph.k(), compactified_hg, parallel_tag_t());
    timer.stop_timer(timer_key("compactify_hypergraph"));

    timer.start_timer(timer_key("generate_random_partition"), "Generate Random Partition");
    generateRandomPartition(compactified_phg);
    timer.stop_timer(timer_key("generate_random_partition"));

    timer.start_timer(timer_key("project_partition"), "Project Partition");
    partitioned_hypergraph.doParallelForAllNodes([&](const HypernodeID hn) {
      partitioned_hypergraph.setOnlyNodePart(hn, compactified_phg.partID(hn_mapping[hn]));
    });
    timer.stop_timer(timer_key("project_partition"));
  }

  timer.start_timer(timer_key("initialize_partition"), "Initialize Partition");
  partitioned_hypergraph.initializePartition();
  timer.stop_timer(timer_key("initialize_partition"));

  timer.start_timer(timer_key("initialize_gain_cache"), "Initialize Initialize Gain Cache");
  gain_cache.initializeGainCache(partitioned_hypergraph);
  timer.stop_timer(timer_key("initialize_gain_cache"));

  timer.stop_timer(timer_key("initial_partition"));

  timer.start_timer(timer_key("create_batch_uncontraction_hierarchy"), "Create n-Level Hierarchy");
  const size_t tmp_batch_size = parallel ? batch_size : 1;
  auto versioned_batches = hypergraph.createBatchUncontractionHierarchy(tmp_batch_size);
  timer.stop_timer(timer_key("create_batch_uncontraction_hierarchy"));

  timer.start_timer(timer_key("batch_uncontractions"), "Batch Uncontractions");
  while ( !versioned_batches.empty() ) {
    BatchVector& batches = versioned_batches.back();
    while ( !batches.empty() ) {
      const Batch& batch = batches.back();
      if ( !batch.empty() ) {
        partitioned_hypergraph.uncontract(batch, gain_cache);
      }
      batches.pop_back();
    }
    versioned_batches.pop_back();

    if ( !removed_hyperedges.empty() ) {
      timer.start_timer(timer_key("restore_parallel_nets"), "Restore Parallel Nets");
      partitioned_hypergraph.restoreSinglePinAndParallelNets(removed_hyperedges.back(), gain_cache);
      removed_hyperedges.pop_back();
      timer.stop_timer(timer_key("restore_parallel_nets"));
    }
  }

  timer.stop_timer(timer_key("batch_uncontractions"));

  return coarsest_hypergraph;
}

#ifdef KAHYPAR_ENABLE_HIGHEST_QUALITY_FEATURES
TEST(ANlevelHypergraph, SimulatesContractionsAndBatchUncontractions) {
  using Hypergraph = typename DynamicHypergraphTypeTraits::Hypergraph;
  using PartitionedHypergraph = typename DynamicHypergraphTypeTraits::PartitionedHypergraph;
  const HypernodeID num_hypernodes = 10000;
  const HypernodeID num_hyperedges = Hypergraph::is_graph ? 40000 : 10000;
  const HypernodeID max_edge_size = 30;
  const HypernodeID num_contractions = 9950;
  const size_t batch_size = 100;
  const bool show_timings = false;
  const bool debug = false;
  utils::Timer timer;
  timer.showDetailedTimings(true);

  if ( debug ) LOG << "Generate Random Hypergraph";
  Hypergraph original_hypergraph = generateRandomHypergraph<Hypergraph>(num_hypernodes, num_hyperedges, max_edge_size);
  Hypergraph sequential_hg = original_hypergraph.copy(parallel_tag_t());
  PartitionedHypergraph sequential_phg(4, sequential_hg, parallel_tag_t());
  Hypergraph parallel_hg = original_hypergraph.copy(parallel_tag_t());
  PartitionedHypergraph parallel_phg(4, parallel_hg, parallel_tag_t());

  if ( debug ) LOG << "Determine random contractions";
  BatchVector contractions = generateRandomContractions(num_hypernodes, num_contractions);

  if ( debug ) LOG << "Simulate n-Level sequentially";
  timer.start_timer("sequential_n_level", "Sequential n-Level");
  Km1GainCache gain_cache_seq;
  Hypergraph coarsest_sequential_hg = simulateNLevel(
    sequential_hg, sequential_phg, gain_cache_seq, contractions, 1, false, timer);
  timer.stop_timer("sequential_n_level");

  if ( debug ) LOG << "Simulate n-Level in parallel";
  timer.start_timer("parallel_n_level", "Parallel n-Level");
  Km1GainCache gain_cache_par;
  Hypergraph coarsest_parallel_hg = simulateNLevel(
    parallel_hg, parallel_phg, gain_cache_par, contractions, batch_size, true, timer);
  timer.stop_timer("parallel_n_level");

  if ( debug ) LOG << "Verify equality of hypergraphs";
  verifyEqualityOfHypergraphs(coarsest_sequential_hg, coarsest_parallel_hg);
  verifyEqualityOfHypergraphs(original_hypergraph, sequential_hg);
  verifyEqualityOfHypergraphs(original_hypergraph, parallel_hg);

  if ( debug ) LOG << "Verify gain cache of hypergraphs";
  verifyGainCache(sequential_phg, gain_cache_seq);
  verifyGainCache(parallel_phg, gain_cache_par);

  if ( debug ) LOG << "Verify number of incident cut hyperedges";
  verifyNumIncidentCutHyperedges(sequential_phg);
  verifyNumIncidentCutHyperedges(parallel_phg);

  if ( show_timings ) {
    LOG << timer;
  }
}

TEST(ANlevelHypergraph, SimulatesContractionsAndBatchUncontractionsWithFixedVertices) {
  using Hypergraph = typename DynamicHypergraphTypeTraits::Hypergraph;
  using PartitionedHypergraph = typename DynamicHypergraphTypeTraits::PartitionedHypergraph;
  const HypernodeID num_hypernodes = 10000;
  const HypernodeID num_hyperedges = Hypergraph::is_graph ? 40000 : 10000;
  const HypernodeID max_edge_size = 30;
  const HypernodeID num_contractions = 9950;
  const size_t batch_size = 100;
  const double fixed_vertex_percentage = 0.02;
  const bool show_timings = false;
  const bool debug = false;
  utils::Timer timer;
  timer.showDetailedTimings(true);

  if ( debug ) LOG << "Generate Random Hypergraph with Fixed Vertices";
  Hypergraph original_hypergraph = generateRandomHypergraph<Hypergraph>(num_hypernodes, num_hyperedges, max_edge_size);
  addRandomFixedVertices(original_hypergraph, 4, fixed_vertex_percentage);
  ds::FixedVertexSupport<Hypergraph> original_fixed_vertices =
    original_hypergraph.copyOfFixedVertexSupport();
  Hypergraph sequential_hg = original_hypergraph.copy(parallel_tag_t());
  PartitionedHypergraph sequential_phg(4, sequential_hg, parallel_tag_t());
  Hypergraph parallel_hg = original_hypergraph.copy(parallel_tag_t());
  PartitionedHypergraph parallel_phg(4, parallel_hg, parallel_tag_t());

  if ( debug ) LOG << "Determine random contractions";
  BatchVector contractions = generateRandomContractions(num_hypernodes, num_contractions);

  if ( debug ) LOG << "Simulate n-Level sequentially";
  timer.start_timer("sequential_n_level", "Sequential n-Level");
  Km1GainCache gain_cache_seq;
  simulateNLevel(sequential_hg, sequential_phg, gain_cache_seq, contractions, 1, false, timer);
  timer.stop_timer("sequential_n_level");

  if ( debug ) LOG << "Simulate n-Level in parallel";
  timer.start_timer("parallel_n_level", "Parallel n-Level");
  Km1GainCache gain_cache_par;
  simulateNLevel(parallel_hg, parallel_phg, gain_cache_par, contractions, batch_size, true, timer);
  timer.stop_timer("parallel_n_level");

  if ( debug ) LOG << "Verify equality of original and sequential fixed vertex support";
  ds::FixedVertexSupport<Hypergraph> sequential_fixed_vertices =
    sequential_hg.copyOfFixedVertexSupport();
  verifyFixedVertices(original_hypergraph, original_fixed_vertices, sequential_fixed_vertices);

  if ( debug ) LOG << "Verify equality of original and parallel fixed vertex support";
  ds::FixedVertexSupport<Hypergraph> parallel_fixed_vertices =
    parallel_hg.copyOfFixedVertexSupport();
  verifyFixedVertices(original_hypergraph, original_fixed_vertices, parallel_fixed_vertices);

  if ( show_timings ) {
    LOG << timer;
  }
}

TEST(ANlevelHypergraph, SimulatesParallelContractionsAndAccessToHypergraph) {
  using Hypergraph = typename DynamicHypergraphTypeTraits::Hypergraph;
  const HypernodeID num_hypernodes = 10000;
  const HypernodeID num_hyperedges = Hypergraph::is_graph ? 40000 : 10000;
  const HypernodeID max_edge_size = 30;
  const HypernodeID num_contractions = 9950;
  const bool show_timings = false;
  const bool debug = false;
  utils::Timer timer;
  timer.showDetailedTimings(true);

  if ( debug ) LOG << "Generate Random Hypergraph";
  Hypergraph hypergraph = generateRandomHypergraph<Hypergraph>(num_hypernodes, num_hyperedges, max_edge_size);
  Hypergraph tmp_hypergraph = hypergraph.copy(parallel_tag_t());

  if ( debug ) LOG << "Determine random contractions";
  BatchVector contractions = generateRandomContractions(num_hypernodes, num_contractions, false);

  if ( debug ) LOG << "Perform Parallel Contractions With Parallel Access";
  bool terminate = false;
  timer.start_timer("contractions_with_access", "Contractions With Access");
  tbb::parallel_invoke([&] {
    while ( !terminate ) {
      // Iterate over all vertices of the hypergraph in parallel
      hypergraph.doParallelForAllNodes([&](const HypernodeID hn) {
        RatingType rating = 0;
        for ( const HyperedgeID& he : hypergraph.incidentEdges(hn) ) {
          const HyperedgeWeight edge_weight = hypergraph.edgeWeight(he);
          for ( const HypernodeID& pin : hypergraph.pins(he) ) {
            const HyperedgeID node_degree = hypergraph.nodeDegree(pin);
            const HypernodeWeight node_weight = hypergraph.nodeWeight(pin);
            if ( hypergraph.communityID(hn) == hypergraph.communityID(pin) ) {
              rating += static_cast<RatingType>(edge_weight * node_degree) / node_weight;
            }
          }
        }
      });
    }
  }, [&] {
    // Perform contractions in parallel
    tbb::parallel_for(UL(0), contractions.back().size(), [&](const size_t i) {
      const Memento& memento = contractions.back()[i];
      hypergraph.registerContraction(memento.u, memento.v);
      hypergraph.contract(memento.v);
    });
    terminate = true;
  });
  timer.stop_timer("contractions_with_access");

  if ( debug ) LOG << "Perform Parallel Contractions Without Parallel Access";
  timer.start_timer("contractions_without_access", "Contractions Without Access");
  tbb::parallel_for(UL(0), contractions.back().size(), [&](const size_t i) {
    const Memento& memento = contractions.back()[i];
    tmp_hypergraph.registerContraction(memento.u, memento.v);
    tmp_hypergraph.contract(memento.v);
  });
  timer.stop_timer("contractions_without_access");

  if ( show_timings ) {
    LOG << timer;
  }
}

#ifdef KAHYPAR_ENABLE_GRAPH_PARTITIONING_FEATURES
TEST(ANlevelGraph, SimulatesContractionsAndBatchUncontractions) {
  using Hypergraph = typename DynamicGraphTypeTraits::Hypergraph;
  using PartitionedHypergraph = typename DynamicGraphTypeTraits::PartitionedHypergraph;
  const HypernodeID num_hypernodes = 10000;
  const HypernodeID num_hyperedges = Hypergraph::is_graph ? 40000 : 10000;
  const HypernodeID max_edge_size = 30;
  const HypernodeID num_contractions = 9950;
  const size_t batch_size = 100;
  const bool show_timings = false;
  const bool debug = false;
  utils::Timer timer;
  timer.showDetailedTimings(true);

  if ( debug ) LOG << "Generate Random Hypergraph";
  Hypergraph original_hypergraph = generateRandomHypergraph<Hypergraph>(num_hypernodes, num_hyperedges, max_edge_size);
  Hypergraph sequential_hg = original_hypergraph.copy(parallel_tag_t());
  PartitionedHypergraph sequential_phg(4, sequential_hg, parallel_tag_t());
  Hypergraph parallel_hg = original_hypergraph.copy(parallel_tag_t());
  PartitionedHypergraph parallel_phg(4, parallel_hg, parallel_tag_t());

  if ( debug ) LOG << "Determine random contractions";
  BatchVector contractions = generateRandomContractions(num_hypernodes, num_contractions);

  if ( debug ) LOG << "Simulate n-Level sequentially";
  timer.start_timer("sequential_n_level", "Sequential n-Level");
  GraphCutGainCache gain_cache_seq;
  Hypergraph coarsest_sequential_hg = simulateNLevel(
    sequential_hg, sequential_phg, gain_cache_seq, contractions, 1, false, timer);
  timer.stop_timer("sequential_n_level");

  if ( debug ) LOG << "Simulate n-Level in parallel";
  timer.start_timer("parallel_n_level", "Parallel n-Level");
  GraphCutGainCache gain_cache_par;
  Hypergraph coarsest_parallel_hg = simulateNLevel(
    parallel_hg, parallel_phg, gain_cache_par, contractions, batch_size, true, timer);
  timer.stop_timer("parallel_n_level");

  if ( debug ) LOG << "Verify equality of hypergraphs";
  verifyEqualityOfHypergraphs(coarsest_sequential_hg, coarsest_parallel_hg);
  verifyEqualityOfHypergraphs(original_hypergraph, sequential_hg);
  verifyEqualityOfHypergraphs(original_hypergraph, parallel_hg);

  if ( debug ) LOG << "Verify gain cache of hypergraphs";
  verifyGainCache(sequential_phg, gain_cache_seq);
  verifyGainCache(parallel_phg, gain_cache_par);

  if ( debug ) LOG << "Verify number of incident cut hyperedges";
  verifyNumIncidentCutHyperedges(sequential_phg);
  verifyNumIncidentCutHyperedges(parallel_phg);

  if ( show_timings ) {
    LOG << timer;
  }
}

TEST(ANlevelGraph, SimulatesContractionsAndBatchUncontractionsWithFixedVertices) {
  using Hypergraph = typename DynamicHypergraphTypeTraits::Hypergraph;
  using PartitionedHypergraph = typename DynamicHypergraphTypeTraits::PartitionedHypergraph;
  const HypernodeID num_hypernodes = 10000;
  const HypernodeID num_hyperedges = Hypergraph::is_graph ? 40000 : 10000;
  const HypernodeID max_edge_size = 30;
  const HypernodeID num_contractions = 9950;
  const size_t batch_size = 100;
  const double fixed_vertex_percentage = 0.02;
  const bool show_timings = false;
  const bool debug = false;
  utils::Timer timer;
  timer.showDetailedTimings(true);

  if ( debug ) LOG << "Generate Random Hypergraph with Fixed Vertices";
  Hypergraph original_hypergraph = generateRandomHypergraph<Hypergraph>(num_hypernodes, num_hyperedges, max_edge_size);
  addRandomFixedVertices(original_hypergraph, 4, fixed_vertex_percentage);
  ds::FixedVertexSupport<Hypergraph> original_fixed_vertices =
    original_hypergraph.copyOfFixedVertexSupport();
  Hypergraph sequential_hg = original_hypergraph.copy(parallel_tag_t());
  PartitionedHypergraph sequential_phg(4, sequential_hg, parallel_tag_t());
  Hypergraph parallel_hg = original_hypergraph.copy(parallel_tag_t());
  PartitionedHypergraph parallel_phg(4, parallel_hg, parallel_tag_t());

  if ( debug ) LOG << "Determine random contractions";
  BatchVector contractions = generateRandomContractions(num_hypernodes, num_contractions);

  if ( debug ) LOG << "Simulate n-Level sequentially";
  timer.start_timer("sequential_n_level", "Sequential n-Level");
  Km1GainCache gain_cache_seq;
  simulateNLevel(sequential_hg, sequential_phg, gain_cache_seq, contractions, 1, false, timer);
  timer.stop_timer("sequential_n_level");

  if ( debug ) LOG << "Simulate n-Level in parallel";
  timer.start_timer("parallel_n_level", "Parallel n-Level");
  Km1GainCache gain_cache_par;
  simulateNLevel(parallel_hg, parallel_phg, gain_cache_par, contractions, batch_size, true, timer);
  timer.stop_timer("parallel_n_level");

  if ( debug ) LOG << "Verify equality of original and sequential fixed vertex support";
  ds::FixedVertexSupport<Hypergraph> sequential_fixed_vertices =
    sequential_hg.copyOfFixedVertexSupport();
  verifyFixedVertices(original_hypergraph, original_fixed_vertices, sequential_fixed_vertices);

  if ( debug ) LOG << "Verify equality of original and parallel fixed vertex support";
  ds::FixedVertexSupport<Hypergraph> parallel_fixed_vertices =
    parallel_hg.copyOfFixedVertexSupport();
  verifyFixedVertices(original_hypergraph, original_fixed_vertices, parallel_fixed_vertices);

  if ( show_timings ) {
    LOG << timer;
  }
}

TEST(ANlevelGraph, SimulatesParallelContractionsAndAccessToHypergraph) {
  using Hypergraph = typename DynamicGraphTypeTraits::Hypergraph;
  const HypernodeID num_hypernodes = 10000;
  const HypernodeID num_hyperedges = Hypergraph::is_graph ? 40000 : 10000;
  const HypernodeID max_edge_size = 30;
  const HypernodeID num_contractions = 9950;
  const bool show_timings = false;
  const bool debug = false;
  utils::Timer timer;
  timer.showDetailedTimings(true);

  if ( debug ) LOG << "Generate Random Hypergraph";
  Hypergraph hypergraph = generateRandomHypergraph<Hypergraph>(num_hypernodes, num_hyperedges, max_edge_size);
  Hypergraph tmp_hypergraph = hypergraph.copy(parallel_tag_t());

  if ( debug ) LOG << "Determine random contractions";
  BatchVector contractions = generateRandomContractions(num_hypernodes, num_contractions, false);

  if ( debug ) LOG << "Perform Parallel Contractions With Parallel Access";
  bool terminate = false;
  timer.start_timer("contractions_with_access", "Contractions With Access");
  tbb::parallel_invoke([&] {
    while ( !terminate ) {
      // Iterate over all vertices of the hypergraph in parallel
      hypergraph.doParallelForAllNodes([&](const HypernodeID hn) {
        RatingType rating = 0;
        for ( const HyperedgeID& he : hypergraph.incidentEdges(hn) ) {
          const HyperedgeWeight edge_weight = hypergraph.edgeWeight(he);
          for ( const HypernodeID& pin : hypergraph.pins(he) ) {
            const HyperedgeID node_degree = hypergraph.nodeDegree(pin);
            const HypernodeWeight node_weight = hypergraph.nodeWeight(pin);
            if ( hypergraph.communityID(hn) == hypergraph.communityID(pin) ) {
              rating += static_cast<RatingType>(edge_weight * node_degree) / node_weight;
            }
          }
        }
      });
    }
  }, [&] {
    // Perform contractions in parallel
    tbb::parallel_for(UL(0), contractions.back().size(), [&](const size_t i) {
      const Memento& memento = contractions.back()[i];
      hypergraph.registerContraction(memento.u, memento.v);
      hypergraph.contract(memento.v);
    });
    terminate = true;
  });
  timer.stop_timer("contractions_with_access");

  if ( debug ) LOG << "Perform Parallel Contractions Without Parallel Access";
  timer.start_timer("contractions_without_access", "Contractions Without Access");
  tbb::parallel_for(UL(0), contractions.back().size(), [&](const size_t i) {
    const Memento& memento = contractions.back()[i];
    tmp_hypergraph.registerContraction(memento.u, memento.v);
    tmp_hypergraph.contract(memento.v);
  });
  timer.stop_timer("contractions_without_access");

  if ( show_timings ) {
    LOG << timer;
  }
}
#endif
#endif

} // namespace ds
} // namespace mt_kahypar