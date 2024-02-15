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

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/utils/utilities.h"
#include "mt-kahypar/io/hypergraph_factory.h"
#include "mt-kahypar/partition/initial_partitioning/pool_initial_partitioner.h"
#include "mt-kahypar/partition/metrics.h"

using ::testing::Test;

namespace mt_kahypar {

template<typename TypeTraitsT, PartitionID k, size_t runs>
struct TestConfig {
  using TypeTraits = TypeTraitsT;
  static constexpr PartitionID K = k;
  static constexpr size_t RUNS = runs;
};

template<typename Config>
class APoolInitialPartitionerTest : public Test {

  using TypeTraits = typename Config::TypeTraits;
  using Hypergraph = typename TypeTraits::Hypergraph;
  using PartitionedHypergraph = typename TypeTraits::PartitionedHypergraph;

 public:
  APoolInitialPartitionerTest() :
    hypergraph(),
    partitioned_hypergraph(),
    context() {
    context.partition.k = Config::K;
    context.partition.epsilon = 0.2;
    #ifdef KAHYPAR_ENABLE_HIGHEST_QUALITY_FEATURES
    context.partition.preset_type = Hypergraph::is_static_hypergraph ?
      PresetType::default_preset : PresetType::highest_quality;
    #else
    context.partition.preset_type = PresetType::default_preset;
    #endif
    context.partition.instance_type = InstanceType::hypergraph;
    context.partition.partition_type = PartitionedHypergraph::TYPE;
    context.partition.objective = Objective::km1;
    context.partition.gain_policy = GainPolicy::km1;
    context.initial_partitioning.runs = Config::RUNS;
    context.refinement.label_propagation.algorithm =
      LabelPropagationAlgorithm::label_propagation;
    context.initial_partitioning.refinement.label_propagation.algorithm =
      LabelPropagationAlgorithm::label_propagation;
    hypergraph = io::readInputFile<Hypergraph>(
      "../tests/instances/test_instance.hgr", FileFormat::hMetis, true);
    partitioned_hypergraph = PartitionedHypergraph(
      context.partition.k, hypergraph, parallel_tag_t());
    context.setupPartWeights(hypergraph.totalWeight());
    utils::Utilities::instance().getTimer(context.utility_id).disable();
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

  void bipartition() {
    Pool<TypeTraits>::bipartition(partitioned_hypergraph, context);
  }

  static void SetUpTestSuite() {
    TBBInitializer::instance(HardwareTopology::instance().num_cpus());
  }

  Hypergraph hypergraph;
  PartitionedHypergraph partitioned_hypergraph;
  Context context;
};

typedef ::testing::Types<TestConfig<StaticHypergraphTypeTraits, 2, 1>,
                         TestConfig<StaticHypergraphTypeTraits, 2, 2>,
                         TestConfig<StaticHypergraphTypeTraits, 2, 5>,
                         TestConfig<StaticHypergraphTypeTraits, 3, 1>,
                         TestConfig<StaticHypergraphTypeTraits, 3, 2>,
                         TestConfig<StaticHypergraphTypeTraits, 3, 5>,
                         TestConfig<StaticHypergraphTypeTraits, 4, 1>,
                         TestConfig<StaticHypergraphTypeTraits, 4, 2>,
                         TestConfig<StaticHypergraphTypeTraits, 4, 5>,
                         TestConfig<StaticHypergraphTypeTraits, 5, 1>,
                         TestConfig<StaticHypergraphTypeTraits, 5, 2>,
                         TestConfig<StaticHypergraphTypeTraits, 5, 5>
                         ENABLE_HIGHEST_QUALITY(COMMA TestConfig<DynamicHypergraphTypeTraits COMMA 2 COMMA 1>)
                         ENABLE_HIGHEST_QUALITY(COMMA TestConfig<DynamicHypergraphTypeTraits COMMA 2 COMMA 2>)
                         ENABLE_HIGHEST_QUALITY(COMMA TestConfig<DynamicHypergraphTypeTraits COMMA 2 COMMA 5>)
                         ENABLE_HIGHEST_QUALITY(COMMA TestConfig<DynamicHypergraphTypeTraits COMMA 3 COMMA 1>)
                         ENABLE_HIGHEST_QUALITY(COMMA TestConfig<DynamicHypergraphTypeTraits COMMA 3 COMMA 2>)
                         ENABLE_HIGHEST_QUALITY(COMMA TestConfig<DynamicHypergraphTypeTraits COMMA 3 COMMA 5>)
                         ENABLE_HIGHEST_QUALITY(COMMA TestConfig<DynamicHypergraphTypeTraits COMMA 4 COMMA 1>)
                         ENABLE_HIGHEST_QUALITY(COMMA TestConfig<DynamicHypergraphTypeTraits COMMA 4 COMMA 2>)
                         ENABLE_HIGHEST_QUALITY(COMMA TestConfig<DynamicHypergraphTypeTraits COMMA 4 COMMA 5>)
                         ENABLE_HIGHEST_QUALITY(COMMA TestConfig<DynamicHypergraphTypeTraits COMMA 5 COMMA 1>)
                         ENABLE_HIGHEST_QUALITY(COMMA TestConfig<DynamicHypergraphTypeTraits COMMA 5 COMMA 2>)
                         ENABLE_HIGHEST_QUALITY(COMMA TestConfig<DynamicHypergraphTypeTraits COMMA 5 COMMA 5>) > TestConfigs;

TYPED_TEST_CASE(APoolInitialPartitionerTest, TestConfigs);

TYPED_TEST(APoolInitialPartitionerTest, HasValidImbalance) {
  this->bipartition();
  ASSERT_LE(metrics::imbalance(this->partitioned_hypergraph, this->context),
            this->context.partition.epsilon);
}

TYPED_TEST(APoolInitialPartitionerTest, AssginsEachHypernode) {
  this->bipartition();
  for ( const HypernodeID& hn : this->partitioned_hypergraph.nodes() ) {
    ASSERT_NE(this->partitioned_hypergraph.partID(hn), -1);
  }
}

TYPED_TEST(APoolInitialPartitionerTest, HasNoSignificantLowPartitionWeights) {
  this->bipartition();
  // Each block should have a weight greater or equal than 20% of the average
  // block weight.
  for ( PartitionID block = 0; block < this->context.partition.k; ++block ) {
    ASSERT_GE(this->partitioned_hypergraph.partWeight(block),
              this->context.partition.perfect_balance_part_weights[block] / 5);
  }
}

TYPED_TEST(APoolInitialPartitionerTest, CanHandleFixedVertices) {
  this->addFixedVertices(0.25);
  this->bipartition();

  for ( const HypernodeID& hn : this->hypergraph.nodes() ) {
    if ( this->hypergraph.isFixed(hn) ) {
      ASSERT_EQ(this->hypergraph.fixedVertexBlock(hn),
        this->partitioned_hypergraph.partID(hn));
    }
  }

  ASSERT_LE(metrics::imbalance(this->partitioned_hypergraph, this->context),
            this->context.partition.epsilon);
}

TYPED_TEST(APoolInitialPartitionerTest, CanHandleFixedVerticesInOnlyOneBlock) {
  this->addFixedVertices(0.05, 0);
  this->bipartition();

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
