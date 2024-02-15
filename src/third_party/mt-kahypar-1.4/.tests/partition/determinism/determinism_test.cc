/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2021 Lars Gottesbüren <lars.gottesbueren@kit.edu>
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

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/io/hypergraph_factory.h"

#include "mt-kahypar/partition/initial_partitioning/bfs_initial_partitioner.h"
#include "mt-kahypar/partition/coarsening/deterministic_multilevel_coarsener.h"
#include "mt-kahypar/partition/refinement/deterministic/deterministic_label_propagation.h"
#include "mt-kahypar/partition/refinement/gains/gain_definitions.h"
#include "mt-kahypar/partition/preprocessing/community_detection/parallel_louvain.h"
#include "mt-kahypar/utils/cast.h"

using ::testing::Test;

namespace mt_kahypar {

namespace {
  using TypeTraits = StaticHypergraphTypeTraits;
  using Hypergraph = typename TypeTraits::Hypergraph;
  using PartitionedHypergraph = typename TypeTraits::PartitionedHypergraph;
}
class DeterminismTest : public Test {

public:
  DeterminismTest() :
          hypergraph(),
          partitioned_hypergraph(),
          context(),
          metrics() {
    context.partition.graph_filename = "../tests/instances/powersim.mtx.hgr";
    context.partition.mode = Mode::direct;
    context.partition.preset_type = PresetType::deterministic;
    context.partition.instance_type = InstanceType::hypergraph;
    context.partition.partition_type = PartitionedHypergraph::TYPE;
    context.partition.epsilon = 0.25;
    context.partition.verbose_output = false;
    context.partition.k = 8;

    // Shared Memory
    context.shared_memory.num_threads = std::thread::hardware_concurrency();

    // Initial Partitioning
    context.initial_partitioning.mode = Mode::recursive_bipartitioning;
    context.initial_partitioning.runs = 1;
    context.initial_partitioning.population_size = 16;

    context.partition.deterministic = true;

    // preprocessing
    context.preprocessing.community_detection.num_sub_rounds_deterministic = 16;
    context.preprocessing.community_detection.max_pass_iterations = 5;
    context.preprocessing.community_detection.min_vertex_move_fraction = 0.01;

    // coarsening
    context.coarsening.num_sub_rounds_deterministic = 3;
    context.coarsening.contraction_limit = 320;
    context.coarsening.max_allowed_node_weight = 30;
    context.coarsening.minimum_shrink_factor = 1.0;
    context.coarsening.maximum_shrink_factor = 4.0;

    // refinement
    context.refinement.deterministic_refinement.num_sub_rounds_sync_lp = 2;
    context.refinement.label_propagation.maximum_iterations = 5;
    context.refinement.deterministic_refinement.use_active_node_set = false;

    context.partition.objective = Objective::km1;
    context.partition.gain_policy = GainPolicy::km1;

    // Read hypergraph
    hypergraph = io::readInputFile<Hypergraph>(
      context.partition.graph_filename, FileFormat::hMetis, true);
    partitioned_hypergraph = PartitionedHypergraph(
            context.partition.k, hypergraph, parallel_tag_t());
    context.setupPartWeights(hypergraph.totalWeight());
  }

  void initialPartition() {
    Context ip_context(context);
    ip_context.refinement.label_propagation.algorithm = LabelPropagationAlgorithm::do_nothing;
    InitialPartitioningDataContainer<TypeTraits> ip_data(partitioned_hypergraph, ip_context);
    ip_data_container_t* ip_data_ptr = ip::to_pointer(ip_data);
    BFSInitialPartitioner<TypeTraits> initial_partitioner(
      InitialPartitioningAlgorithm::bfs, ip_data_ptr, ip_context, 420, 0);
    initial_partitioner.partition();
    ip_data.apply();
    metrics.quality = metrics::quality(partitioned_hypergraph, ip_context);
    metrics.imbalance = metrics::imbalance(partitioned_hypergraph, context);
  }

  void performRepeatedRefinement() {
    initialPartition();
    vec<PartitionID> initial_partition(hypergraph.initialNumNodes());
    for (HypernodeID u : hypergraph.nodes()) {
      initial_partition[u] = partitioned_hypergraph.partID(u);
    }

    vec<PartitionID> first(hypergraph.initialNumNodes());
    for (size_t i = 0; i < num_repetitions; ++i) {
      partitioned_hypergraph.resetPartition();
      for (HypernodeID u : hypergraph.nodes()) {
        partitioned_hypergraph.setNodePart(u, initial_partition[u]);
      }

      mt_kahypar_partitioned_hypergraph_t phg = utils::partitioned_hg_cast(partitioned_hypergraph);
      DeterministicLabelPropagationRefiner<GraphAndGainTypes<TypeTraits, Km1GainTypes>> refiner(
        hypergraph.initialNumNodes(), hypergraph.initialNumEdges(), context);
      refiner.initialize(phg);
      vec<HypernodeID> dummy_refinement_nodes;
      Metrics my_metrics = metrics;
      refiner.refine(phg, dummy_refinement_nodes, my_metrics, 0.0);

      if (i == 0) {
        for (HypernodeID u : hypergraph.nodes()) {
          first[u] = partitioned_hypergraph.partID(u);
        }
      } else {
        for (HypernodeID u : hypergraph.nodes()) {
          ASSERT_EQ(first[u], partitioned_hypergraph.partID(u));
        }
      }
    }
  }

  Hypergraph hypergraph;
  PartitionedHypergraph partitioned_hypergraph;
  Context context;
  Metrics metrics;
  static constexpr size_t num_repetitions = 5;
};

TEST_F(DeterminismTest, Preprocessing) {
  context.preprocessing.community_detection.low_memory_contraction = true;

  LouvainEdgeWeight edge_weight_type;
  if (static_cast<double>(hypergraph.initialNumEdges()) /
      static_cast<double>(hypergraph.initialNumNodes()) < 0.75) {
    edge_weight_type = LouvainEdgeWeight::degree;
  } else {
    edge_weight_type = LouvainEdgeWeight::uniform;
  }

  Graph<Hypergraph> graph(hypergraph, edge_weight_type);
  ds::Clustering first;
  for (size_t i = 0; i < num_repetitions; ++i) {
    ds::Clustering communities = community_detection::run_parallel_louvain(graph, context);
    if (i == 0) {
      first = std::move(communities);
    } else {
      ASSERT_EQ(first, communities);
    }
  }
}

TEST_F(DeterminismTest, Coarsening) {
  Hypergraph first;
  for (size_t i = 0; i < num_repetitions; ++i) {
    UncoarseningData<TypeTraits> uncoarseningData(false, hypergraph, context);
    uncoarsening_data_t* data_ptr = uncoarsening::to_pointer(uncoarseningData);
    mt_kahypar_hypergraph_t hg = utils::hypergraph_cast(hypergraph);
    DeterministicMultilevelCoarsener<TypeTraits> coarsener(hg, context, data_ptr);
    coarsener.coarsen();
    if (i == 0) {
      mt_kahypar_hypergraph_t first_hg = coarsener.coarsestHypergraph();
      first = utils::cast<Hypergraph>(first_hg).copy();
    } else {
      mt_kahypar_hypergraph_t other_hg = coarsener.coarsestHypergraph();
      const Hypergraph& other = utils::cast<Hypergraph>(other_hg);
      ASSERT_EQ(other.initialNumNodes(), first.initialNumNodes());
      ASSERT_EQ(other.initialNumEdges(), first.initialNumEdges());
      ASSERT_EQ(other.initialNumPins(), first.initialNumPins());
      vec<HyperedgeID> inets_first, inets_other;
      for (HypernodeID u : first.nodes()) {
        for (HyperedgeID e : first.incidentEdges(u)) inets_first.push_back(e);
        for (HyperedgeID e : other.incidentEdges(u)) inets_other.push_back(e);
        ASSERT_EQ(inets_first, inets_other);
        inets_first.clear(); inets_other.clear();
      }

      vec<HypernodeID> pins_first, pins_other;
      for (HyperedgeID e : first.edges()) {
        for (HypernodeID v : first.pins(e)) pins_first.push_back(v);
        for (HypernodeID v : other.pins(e)) pins_other.push_back(v);
        ASSERT_EQ(pins_first, pins_other);
        pins_first.clear(); pins_other.clear();
      }
    }
  }
}

TEST_F(DeterminismTest, Refinement) {
  performRepeatedRefinement();
}

TEST_F(DeterminismTest, RefinementOnSmallImbalance) {
  context.partition.epsilon = 0.03;
  context.setupPartWeights(hypergraph.totalWeight());
  performRepeatedRefinement();
}

TEST_F(DeterminismTest, RefinementWithActiveNodeSet) {
  context.refinement.deterministic_refinement.use_active_node_set = true;
  performRepeatedRefinement();
}

TEST_F(DeterminismTest, RefinementK2) {
  context.partition.k = 2;
  partitioned_hypergraph = PartitionedHypergraph(
          context.partition.k, hypergraph, parallel_tag_t());
  context.setupPartWeights(hypergraph.totalWeight());
  performRepeatedRefinement();
}

TEST_F(DeterminismTest, RefinementOnCoarseHypergraph) {
  UncoarseningData<TypeTraits> uncoarseningData(false, hypergraph, context);
  uncoarsening_data_t* data_ptr = uncoarsening::to_pointer(uncoarseningData);
  mt_kahypar_hypergraph_t hg = utils::hypergraph_cast(hypergraph);
  DeterministicMultilevelCoarsener<TypeTraits> coarsener(hg, context, data_ptr);
  coarsener.coarsen();
  hypergraph = utils::cast<Hypergraph>(coarsener.coarsestHypergraph()).copy();
  partitioned_hypergraph = PartitionedHypergraph(
          context.partition.k, hypergraph, parallel_tag_t());
  performRepeatedRefinement();
}

}  // namespace mt_kahypar
