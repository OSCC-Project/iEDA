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
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/partition/preprocessing/community_detection/parallel_louvain.h"
#include "mt-kahypar/io/hypergraph_factory.h"

using ::testing::Test;

namespace mt_kahypar::community_detection {

namespace {
  using TypeTraits = StaticHypergraphTypeTraits;
  using Hypergraph = typename TypeTraits::Hypergraph;
}

class ALouvain : public ds::HypergraphFixture<Hypergraph> {

 using Base = ds::HypergraphFixture<Hypergraph>;

 public:

  ALouvain() :
    Base(),
    graph(nullptr),
    context(),
    karate_club_hg(),
    karate_club_graph(nullptr) {
    context.partition.graph_filename = "../tests/instances/karate_club.graph.hgr";
    context.preprocessing.community_detection.edge_weight_function = LouvainEdgeWeight::uniform;
    context.preprocessing.community_detection.max_pass_iterations = 100;
    context.preprocessing.community_detection.min_vertex_move_fraction = 0.0001;
    context.shared_memory.num_threads = 1;

    graph = std::make_unique<Graph<Hypergraph>>(hypergraph, LouvainEdgeWeight::uniform);
    karate_club_hg = io::readInputFile<Hypergraph>(
      context.partition.graph_filename, FileFormat::hMetis, true);
    karate_club_graph = std::make_unique<Graph<Hypergraph>>(karate_club_hg, LouvainEdgeWeight::uniform, true);
  }

  using Base::hypergraph;
  std::unique_ptr<Graph<Hypergraph>> graph;
  Context context;
  Hypergraph karate_club_hg;
  std::unique_ptr<Graph<Hypergraph>> karate_club_graph;
};

ds::Clustering clustering(const std::vector<PartitionID>& communities) {
  ds::Clustering c(communities.size());
  for ( size_t i = 0; i < communities.size(); ++i ) {
    c[i] = communities[i];
  }
  return c;
}

TEST_F(ALouvain, ComputesMaxGainMove1) {
  ParallelLocalMovingModularity<Hypergraph> plm(context, graph->numNodes());
  ds::Clustering communities = clustering( { 0, 1, 0, 2, 3, 4, 5, 1, 2, 3, 4 } );
  plm.initializeClusterVolumes(*graph, communities);
  PartitionID to = plm.computeMaxGainCluster(
    *graph, communities, 7, plm.non_sampling_incident_cluster_weights.local());
  ASSERT_EQ(0, to);
}

TEST_F(ALouvain, ComputesMaxGainMove2) {
  ParallelLocalMovingModularity<Hypergraph> plm(context, graph->numNodes());
  ds::Clustering communities = clustering( { 0, 1, 0, 3, 3, 4, 5, 1, 2, 3, 4 } );
  plm.initializeClusterVolumes(*graph, communities);
  PartitionID to = plm.computeMaxGainCluster(
    *graph, communities, 8, plm.non_sampling_incident_cluster_weights.local());
  ASSERT_EQ(3, to);
}

TEST_F(ALouvain, ComputesMaxGainMove3) {
  ParallelLocalMovingModularity<Hypergraph> plm(context, graph->numNodes());
  ds::Clustering communities = clustering( { 0, 1, 0, 2, 3, 4, 5, 1, 2, 3, 4 } );
  plm.initializeClusterVolumes(*graph, communities);
  PartitionID to = plm.computeMaxGainCluster(
    *graph, communities, 8, plm.non_sampling_incident_cluster_weights.local());
  ASSERT_EQ(2, to);
}

TEST_F(ALouvain, ComputesMaxGainMove4) {
  ParallelLocalMovingModularity<Hypergraph> plm(context, graph->numNodes());
  ds::Clustering communities = clustering( { 0, 1, 0, 2, 3, 4, 5, 1, 2, 3, 4 } );
  plm.initializeClusterVolumes(*graph, communities);
  PartitionID to = plm.computeMaxGainCluster(
    *graph, communities, 9, plm.non_sampling_incident_cluster_weights.local());
  ASSERT_EQ(3, to);
}

TEST_F(ALouvain, ComputesMaxGainMove5) {
  ParallelLocalMovingModularity<Hypergraph> plm(context, graph->numNodes());
  ds::Clustering communities = clustering( { 0, 1, 0, 2, 2, 4, 5, 1, 2, 3, 4 } );
  plm.initializeClusterVolumes(*graph, communities);
  PartitionID to = plm.computeMaxGainCluster(
    *graph, communities, 9, plm.non_sampling_incident_cluster_weights.local());
  ASSERT_EQ(2, to);
}

TEST_F(ALouvain, ComputesMaxGainMove6) {
  ParallelLocalMovingModularity<Hypergraph> plm(context, graph->numNodes());
  ds::Clustering communities = clustering( { 0, 1, 0, 2, 2, 4, 5, 1, 2, 3, 4 } );
  plm.initializeClusterVolumes(*graph, communities);
  PartitionID to = plm.computeMaxGainCluster(
    *graph, communities, 10, plm.non_sampling_incident_cluster_weights.local());
  ASSERT_EQ(4, to);
}

TEST_F(ALouvain, ComputesMaxGainMove7) {
  ParallelLocalMovingModularity<Hypergraph> plm(context, graph->numNodes());
  ds::Clustering communities = clustering( { 0, 1, 0, 2, 2, 4, 0, 1, 2, 3, 4 } );
  plm.initializeClusterVolumes(*graph, communities);
  PartitionID to = plm.computeMaxGainCluster(
    *graph, communities, 10, plm.non_sampling_incident_cluster_weights.local());
  ASSERT_EQ(0, to);
}

TEST_F(ALouvain, ComputesMaxGainMove8) {
  ParallelLocalMovingModularity<Hypergraph> plm(context, graph->numNodes());
  ds::Clustering communities = clustering( { 0, 1, 0, 2, 2, 4, 0, 1, 1, 3, 4 } );
  plm.initializeClusterVolumes(*graph, communities);
  PartitionID to = plm.computeMaxGainCluster(
    *graph, communities, 0, plm.non_sampling_incident_cluster_weights.local());
  ASSERT_EQ(1, to);
}

TEST_F(ALouvain, ComputesMaxGainMove9) {
  ParallelLocalMovingModularity<Hypergraph> plm(context, graph->numNodes());
  ds::Clustering communities = clustering( { 0, 1, 0, 2, 2, 4, 0, 1, 3, 3, 4 } );
  plm.initializeClusterVolumes(*graph, communities);
  PartitionID to = plm.computeMaxGainCluster(
    *graph, communities, 4, plm.non_sampling_incident_cluster_weights.local());
  ASSERT_EQ(3, to);
}

TEST_F(ALouvain, ComputesMaxGainMove10) {
  ParallelLocalMovingModularity<Hypergraph> plm(context, graph->numNodes());
  ds::Clustering communities = clustering( { 0, 1, 0, 2, 2, 0, 4, 1, 3, 3, 4 } );
  plm.initializeClusterVolumes(*graph, communities);
  PartitionID to = plm.computeMaxGainCluster(
    *graph, communities, 6, plm.non_sampling_incident_cluster_weights.local());
  ASSERT_EQ(4, to);
}

TEST_F(ALouvain, KarateClubTest) {
    tbb::task_arena sequential_arena(1);
#ifdef KAHYPAR_TRAVIS_BUILD
    ds::Clustering communities(0);
    sequential_arena.execute([&] {
      communities = run_parallel_louvain(*karate_club_graph, context, true);
    });
#else
    ds::Clustering communities = sequential_arena.execute([&] {
      return run_parallel_louvain(*karate_club_graph, context, true);
    });
#endif
  ds::Clustering expected_comm = { 1, 1, 1, 1, 0, 0, 0, 1, 3, 1, 0, 1, 1, 1, 3, 3, 0, 1,
                                             3, 1, 3, 1, 3, 2, 2, 2, 3, 2, 2, 3, 3, 2, 3, 3 };

  karate_club_graph = std::make_unique<Graph<Hypergraph>>(
    karate_club_hg, LouvainEdgeWeight::uniform, true);
  ASSERT_EQ(expected_comm, communities);
  ASSERT_EQ(metrics::modularity(*karate_club_graph, communities),
            metrics::modularity(*karate_club_graph, expected_comm));
}

}  // namespace mt_kahypar
