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

#include "tbb/parallel_invoke.h"

#include "tests/datastructures/hypergraph_fixtures.h"
#include "mt-kahypar/utils/utilities.h"
#include "mt-kahypar/io/hypergraph_factory.h"
#include "mt-kahypar/datastructures/fixed_vertex_support.h"
#include "mt-kahypar/partition/initial_partitioning/random_initial_partitioner.h"
#include "mt-kahypar/partition/initial_partitioning/bfs_initial_partitioner.h"
#include "mt-kahypar/partition/initial_partitioning/greedy_initial_partitioner.h"
#include "mt-kahypar/partition/initial_partitioning/label_propagation_initial_partitioner.h"
#include "mt-kahypar/partition/initial_partitioning/policies/gain_computation_policy.h"
#include "mt-kahypar/partition/initial_partitioning/policies/pq_selection_policy.h"
#include "mt-kahypar/utils/randomize.h"

using ::testing::Test;

namespace mt_kahypar {

template<typename TypeTraitsT,
         template<typename> typename InitialPartitioner,
         InitialPartitioningAlgorithm algorithm,
         PartitionID k, size_t runs>
struct TestConfig {
  using TypeTraits = TypeTraitsT;
  using InitialPartitionerTask = InitialPartitioner<TypeTraits>;
  static constexpr InitialPartitioningAlgorithm ALGORITHM = algorithm;
  static constexpr PartitionID K = k;
  static constexpr size_t RUNS = runs;
};

template<typename Config>
class AFlatInitialPartitionerTest : public Test {

 public:
  using TypeTraits = typename Config::TypeTraits;
  using InitialPartitioner = typename Config::InitialPartitionerTask;
  using Hypergraph = typename TypeTraits::Hypergraph;
  using PartitionedHypergraph = typename TypeTraits::PartitionedHypergraph;

  AFlatInitialPartitionerTest() :
    hypergraph(),
    partitioned_hypergraph(),
    context(),
    ip_data(nullptr) {
    context.partition.k = Config::K;
    context.partition.epsilon = 0.2;
    context.partition.objective = Objective::km1;
    context.partition.gain_policy = GainPolicy::km1;
    context.initial_partitioning.lp_initial_block_size = 5;
    context.initial_partitioning.lp_maximum_iterations = 100;
    hypergraph = io::readInputFile<Hypergraph>(
      "../tests/instances/test_instance.hgr", FileFormat::hMetis, true);
    partitioned_hypergraph = PartitionedHypergraph(
      context.partition.k, hypergraph, parallel_tag_t());
    context.setupPartWeights(hypergraph.totalWeight());
    context.refinement.label_propagation.algorithm = LabelPropagationAlgorithm::do_nothing;
    context.initial_partitioning.refinement.label_propagation.algorithm = LabelPropagationAlgorithm::do_nothing;
    utils::Utilities::instance().getTimer(context.utility_id).disable();
  }

  void execute() {
    ip_data = std::make_unique<InitialPartitioningDataContainer<TypeTraits>>(partitioned_hypergraph, context);
    tbb::task_group tg;
    const int seed = 420;
    ip_data_container_t* ip_data_ptr = ip::to_pointer(*ip_data);
    for ( size_t i = 0; i < Config::RUNS; ++i ) {
      tg.run([&, i] {
        InitialPartitioner ip(Config::ALGORITHM, ip_data_ptr, context, seed + i, i);
        ip.partition();
      });
    }
    tg.wait();
    ip_data->apply();
  }

  void addFixedVertices(const double percentage,
                        const PartitionID default_block = kInvalidPartition) {
    ds::FixedVertexSupport<Hypergraph> fixed_vertices(
      hypergraph.initialNumNodes(), context.partition.k);
    fixed_vertices.setHypergraph(&hypergraph);

    const int threshold = percentage * 1000;
    utils::Randomize& rand = utils::Randomize::instance();
    for ( const HypernodeID& hn : hypergraph.nodes() ) {
      int rnd = rand.getRandomInt(0, 1000, THREAD_ID);
      if ( rnd <= threshold ) {
        const PartitionID block = default_block == kInvalidPartition ?
          rand.getRandomInt(0, context.partition.k - 1, THREAD_ID) : default_block;
        fixed_vertices.fixToBlock(hn, block);
      }
    }
    hypergraph.addFixedVertexSupport(std::move(fixed_vertices));
  }

  Hypergraph hypergraph;
  PartitionedHypergraph partitioned_hypergraph;
  Context context;
  std::unique_ptr<InitialPartitioningDataContainer<TypeTraits>> ip_data;
};

template<typename TypeTraits>
using GreedyRoundRobinFMInitialPartitioner = GreedyInitialPartitioner<TypeTraits, CutGainPolicy, RoundRobinPQSelectionPolicy>;
template<typename TypeTraits>
using GreedyGlobalFMInitialPartitioner = GreedyInitialPartitioner<TypeTraits, CutGainPolicy, GlobalPQSelectionPolicy>;
template<typename TypeTraits>
using GreedySequentialFMInitialPartitioner = GreedyInitialPartitioner<TypeTraits, CutGainPolicy, SequentialPQSelectionPolicy>;
template<typename TypeTraits>
using GreedyRoundRobinMaxNetInitialPartitioner = GreedyInitialPartitioner<TypeTraits, MaxNetGainPolicy, RoundRobinPQSelectionPolicy>;
template<typename TypeTraits>
using GreedyGlobalMaxNetInitialPartitioner = GreedyInitialPartitioner<TypeTraits, MaxNetGainPolicy, GlobalPQSelectionPolicy>;
template<typename TypeTraits>
using GreedySequentialMaxNetInitialPartitioner = GreedyInitialPartitioner<TypeTraits, MaxNetGainPolicy, SequentialPQSelectionPolicy>;

typedef ::testing::Types<TestConfig<StaticHypergraphTypeTraits, RandomInitialPartitioner, InitialPartitioningAlgorithm::random, 2, 1>,
                         TestConfig<StaticHypergraphTypeTraits, RandomInitialPartitioner, InitialPartitioningAlgorithm::random, 2, 2>,
                         TestConfig<StaticHypergraphTypeTraits, RandomInitialPartitioner, InitialPartitioningAlgorithm::random, 2, 5>,
                         TestConfig<StaticHypergraphTypeTraits, RandomInitialPartitioner, InitialPartitioningAlgorithm::random, 3, 1>,
                         TestConfig<StaticHypergraphTypeTraits, RandomInitialPartitioner, InitialPartitioningAlgorithm::random, 3, 2>,
                         TestConfig<StaticHypergraphTypeTraits, RandomInitialPartitioner, InitialPartitioningAlgorithm::random, 3, 5>,
                         TestConfig<StaticHypergraphTypeTraits, RandomInitialPartitioner, InitialPartitioningAlgorithm::random, 4, 1>,
                         TestConfig<StaticHypergraphTypeTraits, RandomInitialPartitioner, InitialPartitioningAlgorithm::random, 4, 2>,
                         TestConfig<StaticHypergraphTypeTraits, RandomInitialPartitioner, InitialPartitioningAlgorithm::random, 4, 5>,
                         TestConfig<StaticHypergraphTypeTraits, RandomInitialPartitioner, InitialPartitioningAlgorithm::random, 5, 1>,
                         TestConfig<StaticHypergraphTypeTraits, RandomInitialPartitioner, InitialPartitioningAlgorithm::random, 5, 2>,
                         TestConfig<StaticHypergraphTypeTraits, RandomInitialPartitioner, InitialPartitioningAlgorithm::random, 5, 5>,
                         TestConfig<StaticHypergraphTypeTraits, BFSInitialPartitioner, InitialPartitioningAlgorithm::bfs, 2, 1>,
                         TestConfig<StaticHypergraphTypeTraits, BFSInitialPartitioner, InitialPartitioningAlgorithm::bfs, 2, 2>,
                         TestConfig<StaticHypergraphTypeTraits, BFSInitialPartitioner, InitialPartitioningAlgorithm::bfs, 2, 5>,
                         TestConfig<StaticHypergraphTypeTraits, BFSInitialPartitioner, InitialPartitioningAlgorithm::bfs, 3, 1>,
                         TestConfig<StaticHypergraphTypeTraits, BFSInitialPartitioner, InitialPartitioningAlgorithm::bfs, 3, 2>,
                         TestConfig<StaticHypergraphTypeTraits, BFSInitialPartitioner, InitialPartitioningAlgorithm::bfs, 3, 5>,
                         TestConfig<StaticHypergraphTypeTraits, BFSInitialPartitioner, InitialPartitioningAlgorithm::bfs, 4, 1>,
                         TestConfig<StaticHypergraphTypeTraits, BFSInitialPartitioner, InitialPartitioningAlgorithm::bfs, 4, 2>,
                         TestConfig<StaticHypergraphTypeTraits, BFSInitialPartitioner, InitialPartitioningAlgorithm::bfs, 4, 5>,
                         TestConfig<StaticHypergraphTypeTraits, BFSInitialPartitioner, InitialPartitioningAlgorithm::bfs, 5, 1>,
                         TestConfig<StaticHypergraphTypeTraits, BFSInitialPartitioner, InitialPartitioningAlgorithm::bfs, 5, 2>,
                         TestConfig<StaticHypergraphTypeTraits, BFSInitialPartitioner, InitialPartitioningAlgorithm::bfs, 5, 5>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyRoundRobinFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_round_robin_fm, 2, 1>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyRoundRobinFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_round_robin_fm, 2, 2>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyRoundRobinFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_round_robin_fm, 2, 5>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyRoundRobinFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_round_robin_fm, 3, 1>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyRoundRobinFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_round_robin_fm, 3, 2>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyRoundRobinFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_round_robin_fm, 3, 5>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyRoundRobinFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_round_robin_fm, 4, 1>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyRoundRobinFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_round_robin_fm, 4, 2>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyRoundRobinFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_round_robin_fm, 4, 5>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyRoundRobinFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_round_robin_fm, 5, 1>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyRoundRobinFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_round_robin_fm, 5, 2>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyRoundRobinFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_round_robin_fm, 5, 5>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyGlobalFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_global_fm, 2, 1>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyGlobalFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_global_fm, 2, 2>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyGlobalFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_global_fm, 2, 5>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyGlobalFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_global_fm, 3, 1>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyGlobalFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_global_fm, 3, 2>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyGlobalFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_global_fm, 3, 5>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyGlobalFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_global_fm, 4, 1>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyGlobalFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_global_fm, 4, 2>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyGlobalFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_global_fm, 4, 5>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyGlobalFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_global_fm, 5, 1>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyGlobalFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_global_fm, 5, 2>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyGlobalFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_global_fm, 5, 5>,
                         TestConfig<StaticHypergraphTypeTraits, GreedySequentialFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_sequential_fm, 2, 1>,
                         TestConfig<StaticHypergraphTypeTraits, GreedySequentialFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_sequential_fm, 2, 2>,
                         TestConfig<StaticHypergraphTypeTraits, GreedySequentialFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_sequential_fm, 2, 5>,
                         TestConfig<StaticHypergraphTypeTraits, GreedySequentialFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_sequential_fm, 3, 1>,
                         TestConfig<StaticHypergraphTypeTraits, GreedySequentialFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_sequential_fm, 3, 2>,
                         TestConfig<StaticHypergraphTypeTraits, GreedySequentialFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_sequential_fm, 3, 5>,
                         TestConfig<StaticHypergraphTypeTraits, GreedySequentialFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_sequential_fm, 4, 1>,
                         TestConfig<StaticHypergraphTypeTraits, GreedySequentialFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_sequential_fm, 4, 2>,
                         TestConfig<StaticHypergraphTypeTraits, GreedySequentialFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_sequential_fm, 4, 5>,
                         TestConfig<StaticHypergraphTypeTraits, GreedySequentialFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_sequential_fm, 5, 1>,
                         TestConfig<StaticHypergraphTypeTraits, GreedySequentialFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_sequential_fm, 5, 2>,
                         TestConfig<StaticHypergraphTypeTraits, GreedySequentialFMInitialPartitioner, InitialPartitioningAlgorithm::greedy_sequential_fm, 5, 5>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyRoundRobinMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_round_robin_max_net, 2, 1>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyRoundRobinMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_round_robin_max_net, 2, 2>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyRoundRobinMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_round_robin_max_net, 2, 5>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyRoundRobinMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_round_robin_max_net, 3, 1>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyRoundRobinMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_round_robin_max_net, 3, 2>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyRoundRobinMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_round_robin_max_net, 3, 5>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyRoundRobinMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_round_robin_max_net, 4, 1>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyRoundRobinMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_round_robin_max_net, 4, 2>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyRoundRobinMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_round_robin_max_net, 4, 5>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyRoundRobinMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_round_robin_max_net, 5, 1>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyRoundRobinMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_round_robin_max_net, 5, 2>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyRoundRobinMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_round_robin_max_net, 5, 5>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyGlobalMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_global_max_net, 2, 1>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyGlobalMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_global_max_net, 2, 2>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyGlobalMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_global_max_net, 2, 5>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyGlobalMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_global_max_net, 3, 1>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyGlobalMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_global_max_net, 3, 2>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyGlobalMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_global_max_net, 3, 5>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyGlobalMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_global_max_net, 4, 1>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyGlobalMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_global_max_net, 4, 2>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyGlobalMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_global_max_net, 4, 5>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyGlobalMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_global_max_net, 5, 1>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyGlobalMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_global_max_net, 5, 2>,
                         TestConfig<StaticHypergraphTypeTraits, GreedyGlobalMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_global_max_net, 5, 5>,
                         TestConfig<StaticHypergraphTypeTraits, GreedySequentialMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_sequential_max_net, 2, 1>,
                         TestConfig<StaticHypergraphTypeTraits, GreedySequentialMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_sequential_max_net, 2, 2>,
                         TestConfig<StaticHypergraphTypeTraits, GreedySequentialMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_sequential_max_net, 2, 5>,
                         TestConfig<StaticHypergraphTypeTraits, GreedySequentialMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_sequential_max_net, 3, 1>,
                         TestConfig<StaticHypergraphTypeTraits, GreedySequentialMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_sequential_max_net, 3, 2>,
                         TestConfig<StaticHypergraphTypeTraits, GreedySequentialMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_sequential_max_net, 3, 5>,
                         TestConfig<StaticHypergraphTypeTraits, GreedySequentialMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_sequential_max_net, 4, 1>,
                         TestConfig<StaticHypergraphTypeTraits, GreedySequentialMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_sequential_max_net, 4, 2>,
                         TestConfig<StaticHypergraphTypeTraits, GreedySequentialMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_sequential_max_net, 4, 5>,
                         TestConfig<StaticHypergraphTypeTraits, GreedySequentialMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_sequential_max_net, 5, 1>,
                         TestConfig<StaticHypergraphTypeTraits, GreedySequentialMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_sequential_max_net, 5, 2>,
                         TestConfig<StaticHypergraphTypeTraits, GreedySequentialMaxNetInitialPartitioner, InitialPartitioningAlgorithm::greedy_sequential_max_net, 5, 5>,
                         TestConfig<StaticHypergraphTypeTraits, LabelPropagationInitialPartitioner, InitialPartitioningAlgorithm::label_propagation, 2, 1>,
                         TestConfig<StaticHypergraphTypeTraits, LabelPropagationInitialPartitioner, InitialPartitioningAlgorithm::label_propagation, 2, 2>,
                         TestConfig<StaticHypergraphTypeTraits, LabelPropagationInitialPartitioner, InitialPartitioningAlgorithm::label_propagation, 2, 5>,
                         TestConfig<StaticHypergraphTypeTraits, LabelPropagationInitialPartitioner, InitialPartitioningAlgorithm::label_propagation, 3, 1>,
                         TestConfig<StaticHypergraphTypeTraits, LabelPropagationInitialPartitioner, InitialPartitioningAlgorithm::label_propagation, 3, 2>,
                         TestConfig<StaticHypergraphTypeTraits, LabelPropagationInitialPartitioner, InitialPartitioningAlgorithm::label_propagation, 3, 5>,
                         TestConfig<StaticHypergraphTypeTraits, LabelPropagationInitialPartitioner, InitialPartitioningAlgorithm::label_propagation, 4, 1>,
                         TestConfig<StaticHypergraphTypeTraits, LabelPropagationInitialPartitioner, InitialPartitioningAlgorithm::label_propagation, 4, 2>,
                         TestConfig<StaticHypergraphTypeTraits, LabelPropagationInitialPartitioner, InitialPartitioningAlgorithm::label_propagation, 4, 5>,
                         TestConfig<StaticHypergraphTypeTraits, LabelPropagationInitialPartitioner, InitialPartitioningAlgorithm::label_propagation, 5, 1>,
                         TestConfig<StaticHypergraphTypeTraits, LabelPropagationInitialPartitioner, InitialPartitioningAlgorithm::label_propagation, 5, 2>,
                         TestConfig<StaticHypergraphTypeTraits, LabelPropagationInitialPartitioner, InitialPartitioningAlgorithm::label_propagation, 5, 5> > TestConfigs;

TYPED_TEST_CASE(AFlatInitialPartitionerTest, TestConfigs);

TYPED_TEST(AFlatInitialPartitionerTest, HasValidImbalance) {
  this->execute();

  ASSERT_LE(metrics::imbalance(this->partitioned_hypergraph, this->context),
            this->context.partition.epsilon);
}

TYPED_TEST(AFlatInitialPartitionerTest, AssginsEachHypernode) {
  this->execute();

  for ( const HypernodeID& hn : this->hypergraph.nodes() ) {
    ASSERT_NE(this->partitioned_hypergraph.partID(hn), -1);
  }
}

TYPED_TEST(AFlatInitialPartitionerTest, HasNoSignificantLowPartitionWeights) {
  this->execute();

  // Each block should have a weight greater or equal than 20% of the average
  // block weight.
  for ( PartitionID block = 0; block < this->context.partition.k; ++block ) {
    ASSERT_GE(this->partitioned_hypergraph.partWeight(block),
              this->context.partition.perfect_balance_part_weights[block] / 5);
  }
}

TYPED_TEST(AFlatInitialPartitionerTest, CanHandleFixedVertices) {
  this->addFixedVertices(0.25 /* 25% of the nodes are fixed */);
  this->execute();

  for ( const HypernodeID& hn : this->hypergraph.nodes() ) {
    if ( this->hypergraph.isFixed(hn) ) {
      ASSERT_EQ(this->hypergraph.fixedVertexBlock(hn),
        this->partitioned_hypergraph.partID(hn));
    }
  }

  ASSERT_LE(metrics::imbalance(this->partitioned_hypergraph, this->context),
            this->context.partition.epsilon);
}

TYPED_TEST(AFlatInitialPartitionerTest, CanHandleFixedVerticesInOnlyOneBlock) {
  this->addFixedVertices(0.05 /* 5% of the nodes are fixed to block 0 */, 0);
  this->execute();

  for ( const HypernodeID& hn : this->hypergraph.nodes() ) {
    if ( this->hypergraph.isFixed(hn) ) {
      ASSERT_EQ(this->hypergraph.fixedVertexBlock(hn),
        this->partitioned_hypergraph.partID(hn));
    }
  }

  ASSERT_LE(metrics::imbalance(this->partitioned_hypergraph, this->context),
            this->context.partition.epsilon);
}


}  // namespace mt_kahypar
