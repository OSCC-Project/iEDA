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
#include "mt-kahypar/io/hypergraph_factory.h"
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/partition/initial_partitioning/bfs_initial_partitioner.h"
#include "mt-kahypar/partition/refinement/fm/sequential_twoway_fm_refiner.h"
#include "mt-kahypar/utils/randomize.h"

using ::testing::Test;

namespace mt_kahypar {

class ATwoWayFmRefiner : public Test {

  using TypeTraits = StaticHypergraphTypeTraits;
  using Hypergraph = typename TypeTraits::Hypergraph;
  using PartitionedHypergraph = typename TypeTraits::PartitionedHypergraph;

 public:
  ATwoWayFmRefiner() :
    hypergraph(),
    partitioned_hypergraph(),
    context(),
    refiner(nullptr),
    metrics() {
    context.partition.mode = Mode::direct;
    context.partition.objective = Objective::cut;
    context.partition.gain_policy = GainPolicy::km1; /* change this once cut policy available */
    context.partition.epsilon = 0.25;
    context.partition.k = 2;
    context.partition.verbose_output = false;

    // Shared Memory
    context.shared_memory.num_threads = 1;

    // Initial Partitioning
    context.initial_partitioning.mode = Mode::deep_multilevel;
    context.initial_partitioning.runs = 1;

    // Read hypergraph
    hypergraph = io::readInputFile<Hypergraph>(
      "../tests/instances/contracted_ibm01.hgr", FileFormat::hMetis, true);
    partitioned_hypergraph = PartitionedHypergraph(
      context.partition.k, hypergraph, parallel_tag_t());
    context.setupPartWeights(hypergraph.totalWeight());
    initialPartition();

    refiner = std::make_unique<SequentialTwoWayFmRefiner<TypeTraits>>(partitioned_hypergraph, context);
    prng = std::make_unique<std::mt19937>(420);
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

  Hypergraph hypergraph;
  PartitionedHypergraph partitioned_hypergraph;
  Context context;
  std::unique_ptr<SequentialTwoWayFmRefiner<TypeTraits>> refiner;
  Metrics metrics;
  std::unique_ptr<std::mt19937> prng;
};

static constexpr double EPS = 0.05;

TEST_F(ATwoWayFmRefiner, UpdatesImbalanceCorrectly) {
  refiner->refine(metrics, *prng);
  ASSERT_DOUBLE_EQ(metrics::imbalance(partitioned_hypergraph, context), metrics.imbalance);
}

TEST_F(ATwoWayFmRefiner, DoesNotViolateBalanceConstraint) {
  refiner->refine(metrics, *prng);
  ASSERT_LE(metrics.imbalance, context.partition.epsilon + EPS);
}

TEST_F(ATwoWayFmRefiner, UpdatesMetricsCorrectly) {
  refiner->refine(metrics, *prng);
  ASSERT_EQ(metrics::quality(partitioned_hypergraph, context), metrics.quality);
}

TEST_F(ATwoWayFmRefiner, DoesNotWorsenSolutionQuality) {
  HyperedgeWeight objective_before = metrics::quality(partitioned_hypergraph, context);
    refiner->refine(metrics, *prng);
  ASSERT_LE(metrics.quality, objective_before);
}

TEST_F(ATwoWayFmRefiner, DoesProduceCorrectMetricIfExecutedSeveralTimes) {
  refiner->refine(metrics, *prng);
  ASSERT_EQ(metrics::quality(partitioned_hypergraph, context), metrics.quality);

  partitioned_hypergraph.resetPartition();
  initialPartition();
  refiner->refine(metrics, *prng);
  ASSERT_EQ(metrics::quality(partitioned_hypergraph, context), metrics.quality);
}
}  // namespace mt_kahypar
