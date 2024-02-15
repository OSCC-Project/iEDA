/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2020 Lars Gottesbüren <lars.gottesbueren@kit.edu>
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
#include "mt-kahypar/partition/refinement/fm/fm_commons.h"
#include "mt-kahypar/partition/refinement/fm/multitry_kway_fm.h"
#include "mt-kahypar/partition/refinement/gains/gain_definitions.h"
#include "mt-kahypar/partition/refinement/fm/strategies/gain_cache_strategy.h"
#include "mt-kahypar/partition/initial_partitioning/bfs_initial_partitioner.h"
#include "mt-kahypar/partition/refinement/rebalancing/advanced_rebalancer.h"

using ::testing::Test;

namespace mt_kahypar {

template <typename TypeTraitsT, PartitionID k, FMAlgorithm alg>
struct TestConfig {
  using TypeTraits = TypeTraitsT;
  static constexpr PartitionID K = k;
  static constexpr FMAlgorithm ALG = alg;
};

template<typename Config>
class MultiTryFMTest : public Test {

 public:
  using TypeTraits = typename Config::TypeTraits;
  using Hypergraph = typename TypeTraits::Hypergraph;
  using PartitionedHypergraph = typename TypeTraits::PartitionedHypergraph;
  using Refiner = MultiTryKWayFM<GraphAndGainTypes<TypeTraits, Km1GainTypes>>;

  MultiTryFMTest() :
          hypergraph(),
          partitioned_hypergraph(),
          context(),
          gain_cache(),
          refiner(nullptr),
          metrics() {
    TBBInitializer::instance(std::thread::hardware_concurrency());
    context.partition.graph_filename = "../tests/instances/contracted_ibm01.hgr";
    context.partition.graph_community_filename = "../tests/instances/contracted_ibm01.hgr.community";
    context.partition.mode = Mode::direct;
    context.partition.epsilon = 0.25;
    context.partition.k = Config::K;
    #ifdef KAHYPAR_ENABLE_HIGHEST_QUALITY_FEATURES
    context.partition.preset_type = Hypergraph::is_static_hypergraph ?
      PresetType::default_preset : PresetType::highest_quality;
    #else
    context.partition.preset_type = PresetType::default_preset;
    #endif
    context.partition.instance_type = InstanceType::hypergraph;
    context.partition.partition_type = PartitionedHypergraph::TYPE;
    context.partition.verbose_output = false;

    // Shared Memory
    context.shared_memory.original_num_threads = std::thread::hardware_concurrency();
    context.shared_memory.num_threads = std::thread::hardware_concurrency();

    // Initial Partitioning
    context.initial_partitioning.mode = Mode::deep_multilevel;
    context.initial_partitioning.runs = 1;

    context.refinement.fm.algorithm = Config::ALG;
    context.refinement.fm.multitry_rounds = 10;
    if (context.refinement.fm.algorithm == FMAlgorithm::unconstrained_fm) {
      context.refinement.fm.unconstrained_rounds = 10;
      context.refinement.fm.imbalance_penalty_min = 0.5;
      context.refinement.fm.imbalance_penalty_max = 0.5;
    }
    context.refinement.fm.num_seed_nodes = 5;
    context.refinement.fm.rollback_balance_violation_factor = 1.0;

    context.partition.objective = Objective::km1;
    context.partition.gain_policy = GainPolicy::km1;

    // Read hypergraph
    hypergraph = io::readInputFile<Hypergraph>(
      "../tests/instances/contracted_unweighted_ibm01.hgr", FileFormat::hMetis, true);
    partitioned_hypergraph = PartitionedHypergraph(
            context.partition.k, hypergraph, parallel_tag_t());
    context.setupPartWeights(hypergraph.totalWeight());
    initialPartition();

    rebalancer = std::make_unique<AdvancedRebalancer<GraphAndGainTypes<TypeTraits, Km1GainTypes>>>(
      hypergraph.initialNumNodes(), context, gain_cache);
    refiner = std::make_unique<Refiner>(hypergraph.initialNumNodes(),
      hypergraph.initialNumEdges(), context, gain_cache, *rebalancer);
    mt_kahypar_partitioned_hypergraph_t phg = utils::partitioned_hg_cast(partitioned_hypergraph);
    refiner->initialize(phg);
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
    metrics.quality = metrics::quality(partitioned_hypergraph, context);
    metrics.imbalance = metrics::imbalance(partitioned_hypergraph, context);
  }

  Hypergraph hypergraph;
  PartitionedHypergraph partitioned_hypergraph;
  Context context;
  Km1GainCache gain_cache;
  std::unique_ptr<Refiner> refiner;
  std::unique_ptr<IRebalancer> rebalancer;
  Metrics metrics;
};


typedef ::testing::Types<TestConfig<StaticHypergraphTypeTraits, 2, FMAlgorithm::kway_fm>,
                         TestConfig<StaticHypergraphTypeTraits, 4, FMAlgorithm::kway_fm>,
                         TestConfig<StaticHypergraphTypeTraits, 8, FMAlgorithm::kway_fm>,
                         TestConfig<StaticHypergraphTypeTraits, 128, FMAlgorithm::kway_fm>
                         ENABLE_HIGHEST_QUALITY(COMMA TestConfig<DynamicHypergraphTypeTraits COMMA 2 COMMA FMAlgorithm::kway_fm>)
                         ENABLE_HIGHEST_QUALITY(COMMA TestConfig<DynamicHypergraphTypeTraits COMMA 4 COMMA FMAlgorithm::kway_fm>)
                         ENABLE_HIGHEST_QUALITY(COMMA TestConfig<DynamicHypergraphTypeTraits COMMA 8 COMMA FMAlgorithm::kway_fm>)
                         ENABLE_HIGHEST_QUALITY(COMMA TestConfig<DynamicHypergraphTypeTraits COMMA 128 COMMA FMAlgorithm::kway_fm>),
                         // unconstrained
                         TestConfig<StaticHypergraphTypeTraits, 2, FMAlgorithm::unconstrained_fm>,
                         TestConfig<StaticHypergraphTypeTraits, 4, FMAlgorithm::unconstrained_fm>,
                         TestConfig<StaticHypergraphTypeTraits, 8, FMAlgorithm::unconstrained_fm>,
                         TestConfig<StaticHypergraphTypeTraits, 128, FMAlgorithm::unconstrained_fm>
                         ENABLE_HIGHEST_QUALITY(COMMA TestConfig<DynamicHypergraphTypeTraits COMMA 2 COMMA FMAlgorithm::unconstrained_fm>)
                         ENABLE_HIGHEST_QUALITY(COMMA TestConfig<DynamicHypergraphTypeTraits COMMA 4 COMMA FMAlgorithm::unconstrained_fm>)
                         ENABLE_HIGHEST_QUALITY(COMMA TestConfig<DynamicHypergraphTypeTraits COMMA 8 COMMA FMAlgorithm::unconstrained_fm>)
                         ENABLE_HIGHEST_QUALITY(COMMA TestConfig<DynamicHypergraphTypeTraits COMMA 128 COMMA FMAlgorithm::unconstrained_fm>) > TestConfigs;

TYPED_TEST_CASE(MultiTryFMTest, TestConfigs);

TYPED_TEST(MultiTryFMTest, UpdatesImbalanceCorrectly) {
  mt_kahypar_partitioned_hypergraph_t phg = utils::partitioned_hg_cast(this->partitioned_hypergraph);
  this->refiner->refine(phg, {}, this->metrics, std::numeric_limits<double>::max());
  ASSERT_DOUBLE_EQ(metrics::imbalance(this->partitioned_hypergraph, this->context), this->metrics.imbalance);
}


TYPED_TEST(MultiTryFMTest, DoesNotViolateBalanceConstraint) {
  mt_kahypar_partitioned_hypergraph_t phg = utils::partitioned_hg_cast(this->partitioned_hypergraph);
  this->refiner->refine(phg, {}, this->metrics, std::numeric_limits<double>::max());
  ASSERT_LE(this->metrics.imbalance, this->context.partition.epsilon);
}

TYPED_TEST(MultiTryFMTest, UpdatesMetricsCorrectly) {
  mt_kahypar_partitioned_hypergraph_t phg = utils::partitioned_hg_cast(this->partitioned_hypergraph);
  this->refiner->refine(phg, {}, this->metrics, std::numeric_limits<double>::max());
  ASSERT_EQ(metrics::quality(this->partitioned_hypergraph, this->context.partition.objective),
            this->metrics.quality);
}

TYPED_TEST(MultiTryFMTest, DoesNotWorsenSolutionQuality) {
  HyperedgeWeight objective_before = metrics::quality(this->partitioned_hypergraph, this->context.partition.objective);
  mt_kahypar_partitioned_hypergraph_t phg = utils::partitioned_hg_cast(this->partitioned_hypergraph);
  this->refiner->refine(phg, {}, this->metrics, std::numeric_limits<double>::max());
  ASSERT_LE(this->metrics.quality, objective_before);
}

TYPED_TEST(MultiTryFMTest, AlsoWorksWithNonDefaultFeatures) {
  this->context.refinement.fm.obey_minimal_parallelism = true;
  this->context.refinement.fm.rollback_parallel = false;
  HyperedgeWeight objective_before = metrics::quality(this->partitioned_hypergraph, this->context.partition.objective);
  mt_kahypar_partitioned_hypergraph_t phg = utils::partitioned_hg_cast(this->partitioned_hypergraph);
  this->refiner->refine(phg, {}, this->metrics, std::numeric_limits<double>::max());
  ASSERT_LE(this->metrics.quality, objective_before);
  ASSERT_EQ(metrics::quality(this->partitioned_hypergraph, this->context.partition.objective),
            this->metrics.quality);
  ASSERT_LE(this->metrics.imbalance, this->context.partition.epsilon);
  ASSERT_DOUBLE_EQ(metrics::imbalance(this->partitioned_hypergraph, this->context), this->metrics.imbalance);
}

TYPED_TEST(MultiTryFMTest, WorksWithRefinementNodes) {
  parallel::scalable_vector<HypernodeID> refinement_nodes;
  for (HypernodeID u = 0; u < this->partitioned_hypergraph.initialNumNodes(); ++u) {
    refinement_nodes.push_back(u);
  }
  HyperedgeWeight objective_before = metrics::quality(this->partitioned_hypergraph, this->context.partition.objective);
  mt_kahypar_partitioned_hypergraph_t phg = utils::partitioned_hg_cast(this->partitioned_hypergraph);
  this->refiner->refine(phg, refinement_nodes, this->metrics, std::numeric_limits<double>::max());
  ASSERT_LE(this->metrics.quality, objective_before);
  ASSERT_EQ(metrics::quality(this->partitioned_hypergraph, this->context.partition.objective),
            this->metrics.quality);
  ASSERT_LE(this->metrics.imbalance, this->context.partition.epsilon);
  ASSERT_DOUBLE_EQ(metrics::imbalance(this->partitioned_hypergraph, this->context), this->metrics.imbalance);

  std::stringstream buffer;
  std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());  // redirect std::cout to discard output
  this->refiner->printMemoryConsumption();
  std::cout.rdbuf(old);                                   // and reset again
}

TYPED_TEST(MultiTryFMTest, ChangesTheNumberOfBlocks) {
  using PartitionedHypergraph = typename TestFixture::PartitionedHypergraph;
  HyperedgeWeight objective_before = metrics::quality(this->partitioned_hypergraph, this->context.partition.objective);
  mt_kahypar_partitioned_hypergraph_t phg = utils::partitioned_hg_cast(this->partitioned_hypergraph);
  this->refiner->refine(phg, {}, this->metrics, std::numeric_limits<double>::max());
  ASSERT_LE(this->metrics.quality, objective_before);

  // Initialize partition with smaller K
  const PartitionID old_k = this->context.partition.k;
  this->context.partition.k = std::max(old_k / 2, 2);
  this->context.setupPartWeights(this->hypergraph.totalWeight());
  PartitionedHypergraph phg_with_new_k(
    this->context.partition.k, this->hypergraph, mt_kahypar::parallel_tag_t());
  this->partitioned_hypergraph.doParallelForAllNodes([&](const HypernodeID hn) {
    // create a semi-random partition
    const PartitionID block = this->partitioned_hypergraph.partID(hn);
    phg_with_new_k.setOnlyNodePart(hn, (block + hn) % this->context.partition.k);
  });
  phg_with_new_k.initializePartition();
  this->metrics.quality = metrics::quality(phg_with_new_k, this->context);
  this->metrics.imbalance = metrics::imbalance(phg_with_new_k, this->context);

  objective_before = metrics::quality(phg_with_new_k, this->context.partition.objective);
  mt_kahypar_partitioned_hypergraph_t phg_new_k = utils::partitioned_hg_cast(phg_with_new_k);
  this->gain_cache.reset();
  this->refiner->initialize(phg_new_k);
  this->rebalancer->initialize(phg_new_k);
  this->refiner->refine(phg_new_k, {}, this->metrics, std::numeric_limits<double>::max());
  ASSERT_LE(this->metrics.quality, objective_before);
  ASSERT_EQ(metrics::quality(phg_with_new_k, this->context.partition.objective),
            this->metrics.quality);
}

TEST(UnconstrainedFMDataTest, CorrectlyComputesPenalty) {
  using TypeTraits = StaticHypergraphTypeTraits;
  using Hypergraph = typename TypeTraits::Hypergraph;
  using PartitionedHypergraph = typename TypeTraits::PartitionedHypergraph;
  using HypergraphFactory = typename Hypergraph::Factory;

  Context context;
  context.partition.k = 2;

  // use a super-heavy edge to trigger the fallback case
  std::vector<HyperedgeWeight> he_weights{ 1, 10000 };
  Hypergraph hg = HypergraphFactory::construct(4, 2, { {0, 1}, {2, 3} }, he_weights.data());
  PartitionedHypergraph phg(2, hg);
  phg.setOnlyNodePart(0, 0);
  phg.setOnlyNodePart(1, 0);
  phg.setOnlyNodePart(2, 1);
  phg.setOnlyNodePart(3, 1);
  phg.initializePartition();

  Km1GainCache gain_cache;
  gain_cache.initializeGainCache(phg);

  UnconstrainedFMData ufm_data(4);
  ufm_data.initialize<GraphAndGainTypes<TypeTraits, Km1GainTypes>>(context, phg, gain_cache);

  ASSERT_EQ(0, ufm_data.estimatePenaltyForImbalancedMove(0, -1, -1));
  ASSERT_LE(1.0, ufm_data.estimatePenaltyForImbalancedMove(0, 0, 1));
  ASSERT_GE(1.5, ufm_data.estimatePenaltyForImbalancedMove(0, 0, 1));
  ASSERT_LE(2.0, ufm_data.estimatePenaltyForImbalancedMove(0, 0, 2));
  ASSERT_GE(3.0, ufm_data.estimatePenaltyForImbalancedMove(0, 0, 2));

  ASSERT_EQ(0, ufm_data.estimatePenaltyForImbalancedMove(1, -1, -1));
  ASSERT_LE(10000, ufm_data.estimatePenaltyForImbalancedMove(1, 0, 1));
  ASSERT_GE(15000, ufm_data.estimatePenaltyForImbalancedMove(1, 0, 1));
  ASSERT_LE(20000, ufm_data.estimatePenaltyForImbalancedMove(1, 0, 2));
  ASSERT_GE(30000, ufm_data.estimatePenaltyForImbalancedMove(1, 0, 2));
}

}  // namespace mt_kahypar
