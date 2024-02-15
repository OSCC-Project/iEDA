/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2021 Tobias Heuer <tobias.heuer@kit.edu>
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
#include "mt-kahypar/io/hypergraph_factory.h"
#include "mt-kahypar/io/hypergraph_io.h"
#include "mt-kahypar/partition/refinement/flows/scheduler.h"
#include "mt-kahypar/partition/refinement/gains/km1/km1_gain_computation.h"
#include "mt-kahypar/partition/refinement/gains/gain_definitions.h"
#include "tests/partition/refinement/flow_refiner_mock.h"

using ::testing::Test;

#define MOVE(HN, FROM, TO) Move { FROM, TO, HN, 0 }

namespace mt_kahypar {

namespace {
  using TypeTraits = StaticHypergraphTypeTraits;
  using Hypergraph = typename TypeTraits::Hypergraph;
  using HypergraphFactory = typename Hypergraph::Factory;
  using PartitionedHypergraph = typename TypeTraits::PartitionedHypergraph;
}

class AFlowRefinementScheduler : public Test {
 public:
  AFlowRefinementScheduler() :
    hg(HypergraphFactory::construct(7 , 4,
      { {0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6} }, nullptr, nullptr, true)),
    phg(2, hg, parallel_tag_t()),
    context() {
    context.partition.k = 2;
    context.partition.perfect_balance_part_weights.assign(2, 3);
    context.partition.max_part_weights.assign(2, 4);
    context.partition.objective = Objective::km1;

    context.shared_memory.num_threads = 2;
    context.refinement.flows.algorithm = FlowAlgorithm::mock;
    context.refinement.flows.parallel_searches_multiplier = 1.0;
    context.refinement.flows.max_bfs_distance = 2;

    phg.setOnlyNodePart(0, 0);
    phg.setOnlyNodePart(1, 0);
    phg.setOnlyNodePart(2, 0);
    phg.setOnlyNodePart(3, 0);
    phg.setOnlyNodePart(4, 1);
    phg.setOnlyNodePart(5, 1);
    phg.setOnlyNodePart(6, 1);
    phg.initializePartition();
  }

  Hypergraph hg;
  PartitionedHypergraph phg;
  Context context;
};

template <class F, class K>
void executeConcurrent(F f1, K f2) {
  std::atomic<int> cnt(0);

  tbb::parallel_invoke([&] {
    cnt++;
    while (cnt < 2) { }
    f1();
  }, [&] {
    cnt++;
    while (cnt < 2) { }
    f2();
  });
}

void verifyPartWeights(const vec<HypernodeWeight> actual_weights,
                       const vec<HypernodeWeight> expected_weights) {
  ASSERT_EQ(actual_weights.size(), expected_weights.size());
  for ( size_t i = 0; i < actual_weights.size(); ++i ) {
    ASSERT_EQ(actual_weights[i], expected_weights[i]);
  }
}

TEST_F(AFlowRefinementScheduler, MovesOneVertex) {
  Km1GainCache gain_cache;
  FlowRefinementScheduler<GraphAndGainTypes<TypeTraits, Km1GainTypes>> refiner(
    hg.initialNumNodes(), hg.initialNumEdges(), context, gain_cache);
  mt_kahypar_partitioned_hypergraph_t partitioned_hg = utils::partitioned_hg_cast(phg);
  refiner.initialize(partitioned_hg);
  MoveSequence sequence { { MOVE(3, 0, 1) }, 1 };

  const HyperedgeWeight improvement = refiner.applyMoves(
    QuotientGraph<TypeTraits>::INVALID_SEARCH_ID, sequence);
  ASSERT_EQ(sequence.state, MoveSequenceState::SUCCESS);
  ASSERT_EQ(improvement, sequence.expected_improvement);
  ASSERT_EQ(1, phg.partID(3));
  verifyPartWeights(refiner.partWeights(), { 3, 4 });
}

TEST_F(AFlowRefinementScheduler, MovesVerticesWithIntermediateBalanceViolation) {
  Km1GainCache gain_cache;
  FlowRefinementScheduler<GraphAndGainTypes<TypeTraits, Km1GainTypes>> refiner(
    hg.initialNumNodes(), hg.initialNumEdges(), context, gain_cache);
  mt_kahypar_partitioned_hypergraph_t partitioned_hg = utils::partitioned_hg_cast(phg);
  refiner.initialize(partitioned_hg);
  MoveSequence sequence { { MOVE(5, 1, 0), MOVE(1, 0, 1), MOVE(3, 0, 1) }, 1 };

  const HyperedgeWeight improvement = refiner.applyMoves(
    QuotientGraph<TypeTraits>::INVALID_SEARCH_ID, sequence);
  ASSERT_EQ(sequence.state, MoveSequenceState::SUCCESS);
  ASSERT_EQ(improvement, sequence.expected_improvement);
  ASSERT_EQ(1, phg.partID(1));
  ASSERT_EQ(1, phg.partID(3));
  ASSERT_EQ(0, phg.partID(5));
  verifyPartWeights(refiner.partWeights(), { 3, 4 });
}

TEST_F(AFlowRefinementScheduler, MovesAVertexThatWorsenSolutionQuality) {
  Km1GainCache gain_cache;
  FlowRefinementScheduler<GraphAndGainTypes<TypeTraits, Km1GainTypes>> refiner(
    hg.initialNumNodes(), hg.initialNumEdges(), context, gain_cache);
  mt_kahypar_partitioned_hypergraph_t partitioned_hg = utils::partitioned_hg_cast(phg);
  refiner.initialize(partitioned_hg);
  MoveSequence sequence { { MOVE(0, 0, 1) }, 1 };

  const HyperedgeWeight improvement = refiner.applyMoves(
    QuotientGraph<TypeTraits>::INVALID_SEARCH_ID, sequence);
  ASSERT_EQ(sequence.state, MoveSequenceState::WORSEN_SOLUTION_QUALITY);
  ASSERT_EQ(improvement, 0);
  ASSERT_EQ(0, phg.partID(0));
  verifyPartWeights(refiner.partWeights(), { 4, 3 });
}

TEST_F(AFlowRefinementScheduler, MovesAVertexThatViolatesBalanceConstraint) {
  Km1GainCache gain_cache;
  FlowRefinementScheduler<GraphAndGainTypes<TypeTraits, Km1GainTypes>> refiner(
    hg.initialNumNodes(), hg.initialNumEdges(), context, gain_cache);
  mt_kahypar_partitioned_hypergraph_t partitioned_hg = utils::partitioned_hg_cast(phg);
  refiner.initialize(partitioned_hg);
  MoveSequence sequence { { MOVE(4, 1, 0) }, 1 };

  const HyperedgeWeight improvement = refiner.applyMoves(
    QuotientGraph<TypeTraits>::INVALID_SEARCH_ID, sequence);
  ASSERT_EQ(sequence.state, MoveSequenceState::VIOLATES_BALANCE_CONSTRAINT);
  ASSERT_EQ(improvement, 0);
  ASSERT_EQ(1, phg.partID(4));
  verifyPartWeights(refiner.partWeights(), { 4, 3 });
}

TEST_F(AFlowRefinementScheduler, MovesTwoVerticesConcurrently) {
  context.partition.max_part_weights.assign(2, 5);
  Km1GainCache gain_cache;
  FlowRefinementScheduler<GraphAndGainTypes<TypeTraits, Km1GainTypes>> refiner(
    hg.initialNumNodes(), hg.initialNumEdges(), context, gain_cache);
  mt_kahypar_partitioned_hypergraph_t partitioned_hg = utils::partitioned_hg_cast(phg);
  refiner.initialize(partitioned_hg);

  MoveSequence sequence_1 { { MOVE(3, 0, 1) }, 1 };
  MoveSequence sequence_2 { { MOVE(5, 1, 0) }, 0 };
  HypernodeWeight improvement_1 = 0, improvement_2 = 0;
  executeConcurrent([&] {
    improvement_1 = refiner.applyMoves(
      QuotientGraph<TypeTraits>::INVALID_SEARCH_ID, sequence_1);
    ASSERT_EQ(sequence_1.state, MoveSequenceState::SUCCESS);
    ASSERT_EQ(improvement_1, sequence_1.expected_improvement);
    ASSERT_EQ(1, phg.partID(3));
  }, [&] {
    improvement_2 = refiner.applyMoves(
      QuotientGraph<TypeTraits>::INVALID_SEARCH_ID, sequence_2);
    ASSERT_EQ(sequence_2.state, MoveSequenceState::SUCCESS);
    ASSERT_EQ(improvement_2, sequence_2.expected_improvement);
    ASSERT_EQ(0, phg.partID(5));
  });

  verifyPartWeights(refiner.partWeights(), { 4, 3 });
}

TEST_F(AFlowRefinementScheduler, MovesTwoVerticesConcurrentlyWhereOneViolateBalanceConstraint) {
  Km1GainCache gain_cache;
  FlowRefinementScheduler<GraphAndGainTypes<TypeTraits, Km1GainTypes>> refiner(
    hg.initialNumNodes(), hg.initialNumEdges(), context, gain_cache);
  mt_kahypar_partitioned_hypergraph_t partitioned_hg = utils::partitioned_hg_cast(phg);
  refiner.initialize(partitioned_hg);

  MoveSequence sequence_1 { { MOVE(3, 0, 1) }, 1 };
  MoveSequence sequence_2 { { MOVE(1, 0, 1) }, 0 };
  HypernodeWeight improvement_1 = 0, improvement_2 = 0;
  executeConcurrent([&] {
    improvement_1 = refiner.applyMoves(
      QuotientGraph<TypeTraits>::INVALID_SEARCH_ID, sequence_1);
  }, [&] {
    improvement_2 = refiner.applyMoves(
      QuotientGraph<TypeTraits>::INVALID_SEARCH_ID, sequence_2);
  });

  ASSERT_TRUE(sequence_1.state == MoveSequenceState::VIOLATES_BALANCE_CONSTRAINT ||
              sequence_2.state == MoveSequenceState::VIOLATES_BALANCE_CONSTRAINT);
  ASSERT_TRUE(sequence_1.state == MoveSequenceState::SUCCESS ||
              sequence_2.state == MoveSequenceState::SUCCESS);
  if ( sequence_1.state == MoveSequenceState::SUCCESS ) {
    ASSERT_EQ(improvement_1, sequence_1.expected_improvement);
    ASSERT_EQ(1, phg.partID(3));
    ASSERT_EQ(improvement_2, 0);
    ASSERT_EQ(0, phg.partID(1));
  } else {
    ASSERT_EQ(improvement_1, 0);
    ASSERT_EQ(0, phg.partID(3));
    ASSERT_EQ(improvement_2, sequence_2.expected_improvement);
    ASSERT_EQ(1, phg.partID(1));
  }
  verifyPartWeights(refiner.partWeights(), { 3, 4 });
}

class AFlowRefinementEndToEnd : public Test {

  using GainCalculator = Km1GainComputation;

 public:
  AFlowRefinementEndToEnd() :
    hg(),
    phg(),
    context(),
    max_part_weights(8, 200),
    mover(nullptr) {

    context.partition.graph_filename = "../tests/instances/ibm01.hgr";
    context.partition.k = 8;
    context.partition.epsilon = 0.03;
    context.partition.mode = Mode::direct;
    context.partition.objective = Objective::km1;
    context.shared_memory.num_threads = std::thread::hardware_concurrency();
    context.refinement.flows.algorithm = FlowAlgorithm::mock;
    context.refinement.flows.parallel_searches_multiplier = 1.0;
    context.refinement.flows.max_bfs_distance = 2;

    // Read hypergraph
    hg = io::readInputFile<Hypergraph>(
      context.partition.graph_filename, FileFormat::hMetis, true);
    phg = PartitionedHypergraph(
      context.partition.k, hg, parallel_tag_t());
    context.setupPartWeights(hg.totalWeight());

    // Read Partition
    std::vector<PartitionID> partition;
    io::readPartitionFile("../tests/instances/ibm01.hgr.part8", partition);
    phg.doParallelForAllNodes([&](const HypernodeID& hn) {
      phg.setOnlyNodePart(hn, partition[hn]);
    });
    phg.initializePartition();

    FlowRefinerMockControl::instance().reset();

    mover = std::make_unique<GainCalculator>(context);
    // Refine solution with simple label propagation
    FlowRefinerMockControl::instance().refine_func = [&](const PartitionedHypergraph& phg,
                                                         const Subhypergraph& sub_hg,
                                                         const size_t) {
      MoveSequence sequence { {}, 0 };
      vec<HypernodeID> nodes;
      nodes.insert(nodes.end(), sub_hg.nodes_of_block_0.begin(), sub_hg.nodes_of_block_0.end());
      nodes.insert(nodes.end(), sub_hg.nodes_of_block_1.begin(), sub_hg.nodes_of_block_1.end());
      for ( const HypernodeID& hn : nodes ) {
        Move move = mover->computeMaxGainMove(phg, hn);
        if ( move.from != move.to ) {
          sequence.moves.emplace_back(std::move(move));
          sequence.expected_improvement -= move.gain;
        }
      }
      return sequence;
    };

    // Move approx. 0.5% of the vertices randomly to a different block
    double p = 0.05;
    phg.doParallelForAllNodes([&](const HypernodeID& hn) {
      const int rand_int = utils::Randomize::instance().getRandomInt(0, 100, THREAD_ID);
      if ( rand_int <= p * 100 ) {
        const PartitionID from = phg.partID(hn);
        PartitionID to = utils::Randomize::instance().getRandomInt(
            0, context.partition.k - 1, THREAD_ID);
        while ( from == to ) {
          to = utils::Randomize::instance().getRandomInt(
            0, context.partition.k - 1, THREAD_ID);
        }
        phg.changeNodePart(hn, from, to, context.partition.max_part_weights[to], []{ }, NOOP_FUNC);
      }
    });

    utils::Utilities::instance().getTimer(context.utility_id).clear();
    utils::Utilities::instance().getStats(context.utility_id).clear();
  }

  Hypergraph hg;
  PartitionedHypergraph phg;
  Context context;
  vec<HypernodeWeight> max_part_weights;
  std::unique_ptr<GainCalculator> mover;
};

TEST_F(AFlowRefinementEndToEnd, SmokeTestWithTwoBlocksPerRefiner) {
  const bool debug = false;
  Km1GainCache gain_cache;
  FlowRefinementScheduler<GraphAndGainTypes<TypeTraits, Km1GainTypes>> scheduler(
    hg.initialNumNodes(), hg.initialNumEdges(), context, gain_cache);

  Metrics metrics;
  metrics.quality = metrics::quality(phg, context);
  metrics.imbalance = metrics::imbalance(phg, context);

  if ( debug ) {
    LOG << "Start Solution km1 =" << metrics.quality;
  }

  mt_kahypar_partitioned_hypergraph_t partitioned_hg = utils::partitioned_hg_cast(phg);
  scheduler.initialize(partitioned_hg);
  scheduler.refine(partitioned_hg, {}, metrics, 0.0);

  if ( debug ) {
    LOG << "Final Solution km1 =" << metrics.quality;
  }

  if ( debug ) {
    utils::Utilities::instance().getTimer(context.utility_id).showDetailedTimings(true);
    LOG << utils::Utilities::instance().getTimer(context.utility_id);
    LOG << utils::Utilities::instance().getStats(context.utility_id);
  }

  ASSERT_EQ(metrics::quality(phg, Objective::km1), metrics.quality);
  ASSERT_EQ(metrics::imbalance(phg, context), metrics.imbalance);
  for ( PartitionID i = 0; i < context.partition.k; ++i ) {
    ASSERT_LE(phg.partWeight(i), context.partition.max_part_weights[i]);
  }
}

TEST_F(AFlowRefinementEndToEnd, SmokeTestWithFourBlocksPerRefiner) {
  const bool debug = false;
  FlowRefinerMockControl::instance().max_num_blocks = 4;
  Km1GainCache gain_cache;
  FlowRefinementScheduler<GraphAndGainTypes<TypeTraits, Km1GainTypes>> scheduler(
    hg.initialNumNodes(), hg.initialNumEdges(), context, gain_cache);

  Metrics metrics;
  metrics.quality = metrics::quality(phg, context);
  metrics.imbalance = metrics::imbalance(phg, context);

  if ( debug ) {
    LOG << "Start Solution km1 =" << metrics.quality;
  }

  mt_kahypar_partitioned_hypergraph_t partitioned_hg = utils::partitioned_hg_cast(phg);
  scheduler.initialize(partitioned_hg);
  scheduler.refine(partitioned_hg, {}, metrics, 0.0);

  if ( debug ) {
    LOG << "Final Solution km1 =" << metrics.quality;
  }

  if ( debug ) {
    utils::Utilities::instance().getTimer(context.utility_id).showDetailedTimings(true);
    LOG << utils::Utilities::instance().getTimer(context.utility_id);
    LOG << utils::Utilities::instance().getStats(context.utility_id);
  }

  ASSERT_EQ(metrics::quality(phg, Objective::km1), metrics.quality);
  ASSERT_EQ(metrics::imbalance(phg, context), metrics.imbalance);
  for ( PartitionID i = 0; i < context.partition.k; ++i ) {
    ASSERT_LE(phg.partWeight(i), context.partition.max_part_weights[i]);
  }
}

}