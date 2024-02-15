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
#include "tests/partition/refinement/flow_refiner_mock.h"
#include "mt-kahypar/partition/refinement/flows/refiner_adapter.h"

using ::testing::Test;

#define MOVE(HN, FROM, TO) Move { FROM, TO, HN, 0 }

namespace mt_kahypar {

namespace {
  using TypeTraits = StaticHypergraphTypeTraits;
  using Hypergraph = typename TypeTraits::Hypergraph;
  using HypergraphFactory = typename Hypergraph::Factory;
  using PartitionedHypergraph = typename TypeTraits::PartitionedHypergraph;
}

class AFlowRefinerAdapter : public Test {
 public:
  AFlowRefinerAdapter() :
    hg(HypergraphFactory::construct(7 , 4, {
       {0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6} }, nullptr, nullptr, true)),
    phg(2, hg, parallel_tag_t()),
    context() {
    context.partition.k = 2;
    context.partition.perfect_balance_part_weights.assign(2, 3);
    context.partition.max_part_weights.assign(2, 4);
    context.partition.objective = Objective::km1;
    context.shared_memory.num_threads = 8;
    context.refinement.flows.algorithm = FlowAlgorithm::mock;

    FlowRefinerMockControl::instance().reset();

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
  std::unique_ptr<FlowRefinerAdapter<TypeTraits>> refiner;
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

TEST_F(AFlowRefinerAdapter, FailsToRegisterMoreSearchesIfAllAreUsed) {
  refiner = std::make_unique<FlowRefinerAdapter<TypeTraits>>(hg.initialNumEdges(), context);
  refiner->initialize(2);

  ASSERT_TRUE(refiner->registerNewSearch(0, phg));
  ASSERT_TRUE(refiner->registerNewSearch(1, phg));
  ASSERT_FALSE(refiner->registerNewSearch(2, phg));
}

TEST_F(AFlowRefinerAdapter, UseCorrectNumberOfThreadsForSearch1) {
  refiner = std::make_unique<FlowRefinerAdapter<TypeTraits>>(hg.initialNumEdges(), context);
  refiner->initialize(2);
  ASSERT_EQ(2, refiner->numAvailableRefiner());
  ASSERT_EQ(0, refiner->numUsedThreads());

  FlowRefinerMockControl::instance().refine_func =
    [&](const PartitionedHypergraph&, const Subhypergraph&, const size_t num_threads) -> MoveSequence {
      EXPECT_EQ(context.shared_memory.num_threads / 2, num_threads);
      EXPECT_EQ(context.shared_memory.num_threads / 2, refiner->numUsedThreads());
      return MoveSequence { {}, 0 };
    };
  ASSERT_TRUE(refiner->registerNewSearch(0, phg));
  refiner->refine(0, phg, {});
  refiner->finalizeSearch(0);
}

TEST_F(AFlowRefinerAdapter, UseCorrectNumberOfThreadsForSearch2) {
  refiner = std::make_unique<FlowRefinerAdapter<TypeTraits>>(hg.initialNumEdges(), context);
  refiner->initialize(2);
  ASSERT_EQ(2, refiner->numAvailableRefiner());
  ASSERT_EQ(0, refiner->numUsedThreads());

  std::atomic<size_t> cnt(0);
  FlowRefinerMockControl::instance().refine_func =
    [&](const PartitionedHypergraph&, const Subhypergraph&, const size_t num_threads) -> MoveSequence {
      EXPECT_EQ(context.shared_memory.num_threads / 2, num_threads);
      EXPECT_EQ(context.shared_memory.num_threads / 2, refiner->numUsedThreads());
      ++cnt;
      while ( cnt < 2 ) { }
      return MoveSequence { {}, 0 };
    };
  ASSERT_TRUE(refiner->registerNewSearch(0, phg));
  FlowRefinerMockControl::instance().refine_func =
    [&](const PartitionedHypergraph&, const Subhypergraph&, const size_t num_threads) -> MoveSequence {
      EXPECT_EQ(context.shared_memory.num_threads / 2, num_threads);
      EXPECT_EQ(context.shared_memory.num_threads, refiner->numUsedThreads());
      ++cnt;
      return MoveSequence { {}, 0 };
    };
  ASSERT_TRUE(refiner->registerNewSearch(1, phg));
  executeConcurrent([&] {
    refiner->refine(0, phg, {});
  }, [&] {
    while ( cnt < 1 ) { }
    refiner->refine(1, phg, {});
  });
  refiner->finalizeSearch(0);
  refiner->finalizeSearch(1);
}

TEST_F(AFlowRefinerAdapter, UsesMoreThreadsIfOneRefinerTermiantes) {
  refiner = std::make_unique<FlowRefinerAdapter<TypeTraits>>(hg.initialNumEdges(), context);
  refiner->initialize(2);
  ASSERT_EQ(2, refiner->numAvailableRefiner());
  ASSERT_EQ(0, refiner->numUsedThreads());
  refiner->terminateRefiner();

  FlowRefinerMockControl::instance().refine_func =
    [&](const PartitionedHypergraph&, const Subhypergraph&, const size_t num_threads) -> MoveSequence {
      EXPECT_EQ(context.shared_memory.num_threads, num_threads);
      EXPECT_EQ(context.shared_memory.num_threads, refiner->numUsedThreads());
      return MoveSequence { {}, 0 };
    };
  ASSERT_TRUE(refiner->registerNewSearch(0, phg));
  refiner->refine(0, phg, {});
  refiner->finalizeSearch(0);
}

}