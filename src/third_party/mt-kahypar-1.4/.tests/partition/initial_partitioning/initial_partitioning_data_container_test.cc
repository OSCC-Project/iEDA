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
#include "mt-kahypar/definitions.h"
#include "mt-kahypar/utils/utilities.h"
#include "mt-kahypar/partition/initial_partitioning/initial_partitioning_data_container.h"

using ::testing::Test;

namespace mt_kahypar {

namespace {
using TypeTraits = StaticHypergraphTypeTraits;
using Hypergraph = typename StaticHypergraphTypeTraits::Hypergraph;
using PartitionedHypergraph = typename StaticHypergraphTypeTraits::PartitionedHypergraph;
}

class AInitialPartitioningDataContainer : public ds::HypergraphFixture<Hypergraph> {
 private:
  using Base = ds::HypergraphFixture<Hypergraph>;

 public:
  AInitialPartitioningDataContainer() :
    Base(),
    context() {
    context.partition.k = 2;
    context.partition.epsilon = 0.2;
    context.partition.objective = Objective::km1;
    context.partition.gain_policy = GainPolicy::km1;
    // Max Part Weight = 4
    context.setupPartWeights(hypergraph.totalWeight());
    utils::Utilities::instance().getTimer(context.utility_id).disable();
  }

  using Base::hypergraph;
  Context context;
};

TEST_F(AInitialPartitioningDataContainer, ReturnsAnUnassignedLocalHypernode1) {
  PartitionedHypergraph partitioned_hypergraph(
    context.partition.k, hypergraph);
  InitialPartitioningDataContainer<TypeTraits> ip_data(
    partitioned_hypergraph, context, true);
  PartitionedHypergraph& local_hg = ip_data.local_partitioned_hypergraph();
  std::mt19937 prng(420);
  ip_data.reset_unassigned_hypernodes(prng);
  ASSERT_EQ(-1, local_hg.partID(ip_data.get_unassigned_hypernode()));
}

TEST_F(AInitialPartitioningDataContainer, ReturnsAnUnassignedLocalHypernode2) {
  PartitionedHypergraph partitioned_hypergraph(
    context.partition.k, hypergraph);
  InitialPartitioningDataContainer<TypeTraits> ip_data(
    partitioned_hypergraph, context, true);
  PartitionedHypergraph& local_hg = ip_data.local_partitioned_hypergraph();
  std::mt19937 prng(420);
  ip_data.reset_unassigned_hypernodes(prng);

  size_t num_hypernodes_to_assign = 2;
  size_t assigned_hypernodes = 0;
  for ( const HypernodeID& hn : local_hg.nodes() ) {
    local_hg.setNodePart(hn, 0);
    ++assigned_hypernodes;
    if ( assigned_hypernodes == num_hypernodes_to_assign ) {
      break;
    }
  }

  ASSERT_EQ(-1, local_hg.partID(ip_data.get_unassigned_hypernode()));
}

TEST_F(AInitialPartitioningDataContainer, ReturnsAnUnassignedLocalHypernode3) {
  PartitionedHypergraph partitioned_hypergraph(
    context.partition.k, hypergraph);
  InitialPartitioningDataContainer<TypeTraits> ip_data(
    partitioned_hypergraph, context, true);
  PartitionedHypergraph& local_hg = ip_data.local_partitioned_hypergraph();
  std::mt19937 prng(420);
  ip_data.reset_unassigned_hypernodes(prng);

  size_t num_hypernodes_to_assign = 4;
  size_t assigned_hypernodes = 0;
  for ( const HypernodeID& hn : local_hg.nodes() ) {
    local_hg.setNodePart(hn, 0);
    ++assigned_hypernodes;
    if ( assigned_hypernodes == num_hypernodes_to_assign ) {
      break;
    }
  }

  ASSERT_EQ(-1, local_hg.partID(ip_data.get_unassigned_hypernode()));
}

TEST_F(AInitialPartitioningDataContainer, ReturnsAnUnassignedLocalHypernode4) {
  PartitionedHypergraph partitioned_hypergraph(
    context.partition.k, hypergraph);
  InitialPartitioningDataContainer<TypeTraits> ip_data(
    partitioned_hypergraph, context, true);
  PartitionedHypergraph& local_hg = ip_data.local_partitioned_hypergraph();
  std::mt19937 prng(420);
  ip_data.reset_unassigned_hypernodes(prng);

  size_t num_hypernodes_to_assign = 6;
  size_t assigned_hypernodes = 0;
  for ( const HypernodeID& hn : local_hg.nodes() ) {
    local_hg.setNodePart(hn, 0);
    ++assigned_hypernodes;
    if ( assigned_hypernodes == num_hypernodes_to_assign ) {
      break;
    }
  }

  ASSERT_EQ(-1, local_hg.partID(ip_data.get_unassigned_hypernode()));
}

TEST_F(AInitialPartitioningDataContainer, ReturnsInvalidHypernodeIfAllHypernodesAreAssigned) {
  PartitionedHypergraph partitioned_hypergraph(
    context.partition.k, hypergraph);
  InitialPartitioningDataContainer<TypeTraits> ip_data(
    partitioned_hypergraph, context, true);
  PartitionedHypergraph& local_hg = ip_data.local_partitioned_hypergraph();
  std::mt19937 prng(420);
  ip_data.reset_unassigned_hypernodes(prng);

  for ( const HypernodeID& hn : local_hg.nodes() ) {
    local_hg.setNodePart(hn, 0);
  }

  ASSERT_EQ(std::numeric_limits<HypernodeID>::max(),
            ip_data.get_unassigned_hypernode());
}

TEST_F(AInitialPartitioningDataContainer, ReturnsValidUnassignedHypernodeIfPartitionIsResetted) {
  PartitionedHypergraph partitioned_hypergraph(
    context.partition.k, hypergraph);
  InitialPartitioningDataContainer<TypeTraits> ip_data(
    partitioned_hypergraph, context, true);
  PartitionedHypergraph& local_hg = ip_data.local_partitioned_hypergraph();
  std::mt19937 prng(420);
  ip_data.reset_unassigned_hypernodes(prng);
  for ( const HypernodeID& hn : local_hg.nodes() ) {
    local_hg.setNodePart(hn, 0);
  }

  ASSERT_EQ(std::numeric_limits<HypernodeID>::max(),
            ip_data.get_unassigned_hypernode());

  local_hg.resetPartition();
  ip_data.reset_unassigned_hypernodes(prng);
  ASSERT_EQ(-1, local_hg.partID(ip_data.get_unassigned_hypernode()));
}

TEST_F(AInitialPartitioningDataContainer, AppliesPartitionToHypergraph) {
  PartitionedHypergraph partitioned_hypergraph(
    context.partition.k, hypergraph);
  InitialPartitioningDataContainer<TypeTraits> ip_data(
    partitioned_hypergraph, context, true);
  PartitionedHypergraph& local_hg = ip_data.local_partitioned_hypergraph();

  // Cut = 2
  local_hg.setNodePart(0, 0);
  local_hg.setNodePart(1, 0);
  local_hg.setNodePart(2, 0);
  local_hg.setNodePart(3, 1);
  local_hg.setNodePart(4, 1);
  local_hg.setNodePart(5, 1);
  local_hg.setNodePart(6, 1);
  std::mt19937 prng(420);
  ip_data.commit(InitialPartitioningAlgorithm::random, prng, 0);

  ip_data.apply();

  ASSERT_EQ(2, metrics::quality(partitioned_hypergraph, context.partition.objective));
  ASSERT_EQ(0, partitioned_hypergraph.partID(0));
  ASSERT_EQ(0, partitioned_hypergraph.partID(1));
  ASSERT_EQ(0, partitioned_hypergraph.partID(2));
  ASSERT_EQ(1, partitioned_hypergraph.partID(3));
  ASSERT_EQ(1, partitioned_hypergraph.partID(4));
  ASSERT_EQ(1, partitioned_hypergraph.partID(5));
  ASSERT_EQ(1, partitioned_hypergraph.partID(6));
}

TEST_F(AInitialPartitioningDataContainer, AppliesBestPartitionToHypergraph) {
  PartitionedHypergraph partitioned_hypergraph(
    context.partition.k, hypergraph);
  InitialPartitioningDataContainer<TypeTraits> ip_data(
    partitioned_hypergraph, context, true);
  PartitionedHypergraph& local_hg = ip_data.local_partitioned_hypergraph();

  // Cut = 3
  local_hg.setNodePart(0, 0);
  local_hg.setNodePart(1, 0);
  local_hg.setNodePart(2, 0);
  local_hg.setNodePart(3, 0);
  local_hg.setNodePart(4, 1);
  local_hg.setNodePart(5, 1);
  local_hg.setNodePart(6, 1);
  std::mt19937 prng(420);
  ip_data.commit(InitialPartitioningAlgorithm::random, prng, 0);

  // Cut = 2
  local_hg.setNodePart(0, 0);
  local_hg.setNodePart(1, 0);
  local_hg.setNodePart(2, 0);
  local_hg.setNodePart(3, 1);
  local_hg.setNodePart(4, 1);
  local_hg.setNodePart(5, 1);
  local_hg.setNodePart(6, 1);
  ip_data.commit(InitialPartitioningAlgorithm::random, prng, 1);

  ip_data.apply();

  ASSERT_EQ(2, metrics::quality(partitioned_hypergraph, context.partition.objective));
  ASSERT_EQ(0, partitioned_hypergraph.partID(0));
  ASSERT_EQ(0, partitioned_hypergraph.partID(1));
  ASSERT_EQ(0, partitioned_hypergraph.partID(2));
  ASSERT_EQ(1, partitioned_hypergraph.partID(3));
  ASSERT_EQ(1, partitioned_hypergraph.partID(4));
  ASSERT_EQ(1, partitioned_hypergraph.partID(5));
  ASSERT_EQ(1, partitioned_hypergraph.partID(6));
}

TEST_F(AInitialPartitioningDataContainer, AppliesBestPartitionWithImbalancedPartitionToHypergraph1) {
  PartitionedHypergraph partitioned_hypergraph(
    context.partition.k, hypergraph);
  InitialPartitioningDataContainer<TypeTraits> ip_data(
    partitioned_hypergraph, context, true);
  PartitionedHypergraph& local_hg = ip_data.local_partitioned_hypergraph();

  // Cut = 1, but imbalanced
  local_hg.setNodePart(0, 1);
  local_hg.setNodePart(1, 0);
  local_hg.setNodePart(2, 1);
  local_hg.setNodePart(3, 1);
  local_hg.setNodePart(4, 1);
  local_hg.setNodePart(5, 1);
  local_hg.setNodePart(6, 1);
  ip_data.commit(InitialPartitioningAlgorithm::random);

  // Cut = 2
  local_hg.setNodePart(0, 0);
  local_hg.setNodePart(1, 0);
  local_hg.setNodePart(2, 0);
  local_hg.setNodePart(3, 1);
  local_hg.setNodePart(4, 1);
  local_hg.setNodePart(5, 1);
  local_hg.setNodePart(6, 1);
  ip_data.commit(InitialPartitioningAlgorithm::random);

  ip_data.apply();

  ASSERT_EQ(2, metrics::quality(partitioned_hypergraph, context.partition.objective));
  ASSERT_EQ(0, partitioned_hypergraph.partID(0));
  ASSERT_EQ(0, partitioned_hypergraph.partID(1));
  ASSERT_EQ(0, partitioned_hypergraph.partID(2));
  ASSERT_EQ(1, partitioned_hypergraph.partID(3));
  ASSERT_EQ(1, partitioned_hypergraph.partID(4));
  ASSERT_EQ(1, partitioned_hypergraph.partID(5));
  ASSERT_EQ(1, partitioned_hypergraph.partID(6));
}

TEST_F(AInitialPartitioningDataContainer, AppliesBestPartitionWithImbalancedPartitionToHypergraph2) {
  PartitionedHypergraph partitioned_hypergraph(
    context.partition.k, hypergraph);
  InitialPartitioningDataContainer<TypeTraits> ip_data(
    partitioned_hypergraph, context, true);
  PartitionedHypergraph& local_hg = ip_data.local_partitioned_hypergraph();

  // Cut = 1, but imbalanced
  local_hg.setNodePart(0, 1);
  local_hg.setNodePart(1, 0);
  local_hg.setNodePart(2, 1);
  local_hg.setNodePart(3, 1);
  local_hg.setNodePart(4, 1);
  local_hg.setNodePart(5, 1);
  local_hg.setNodePart(6, 1);
  ip_data.commit(InitialPartitioningAlgorithm::random);

  // Cut = 2, also imbalanced but better balance
  local_hg.setNodePart(0, 0);
  local_hg.setNodePart(1, 0);
  local_hg.setNodePart(2, 1);
  local_hg.setNodePart(3, 1);
  local_hg.setNodePart(4, 1);
  local_hg.setNodePart(5, 1);
  local_hg.setNodePart(6, 1);
  ip_data.commit(InitialPartitioningAlgorithm::random);

  ip_data.apply();

  ASSERT_EQ(2, metrics::quality(partitioned_hypergraph, context.partition.objective));
  ASSERT_EQ(0, partitioned_hypergraph.partID(0));
  ASSERT_EQ(0, partitioned_hypergraph.partID(1));
  ASSERT_EQ(1, partitioned_hypergraph.partID(2));
  ASSERT_EQ(1, partitioned_hypergraph.partID(3));
  ASSERT_EQ(1, partitioned_hypergraph.partID(4));
  ASSERT_EQ(1, partitioned_hypergraph.partID(5));
  ASSERT_EQ(1, partitioned_hypergraph.partID(6));
}

TEST_F(AInitialPartitioningDataContainer, AppliesBestPartitionWithImbalancedPartitionToHypergraph3) {
  PartitionedHypergraph partitioned_hypergraph(
    context.partition.k, hypergraph);
  InitialPartitioningDataContainer<TypeTraits> ip_data(
    partitioned_hypergraph, context, true);
  PartitionedHypergraph& local_hg = ip_data.local_partitioned_hypergraph();

  // Cut = 1
  local_hg.setNodePart(0, 1);
  local_hg.setNodePart(1, 0);
  local_hg.setNodePart(2, 1);
  local_hg.setNodePart(3, 1);
  local_hg.setNodePart(4, 1);
  local_hg.setNodePart(5, 1);
  local_hg.setNodePart(6, 1);
  ip_data.commit(InitialPartitioningAlgorithm::random);

  // Cut = 2
  local_hg.setNodePart(0, 0);
  local_hg.setNodePart(1, 1);
  local_hg.setNodePart(2, 1);
  local_hg.setNodePart(3, 1);
  local_hg.setNodePart(4, 1);
  local_hg.setNodePart(5, 1);
  local_hg.setNodePart(6, 1);
  ip_data.commit(InitialPartitioningAlgorithm::random);

  ip_data.apply();

  ASSERT_EQ(1, metrics::quality(partitioned_hypergraph, context.partition.objective));
  ASSERT_EQ(1, partitioned_hypergraph.partID(0));
  ASSERT_EQ(0, partitioned_hypergraph.partID(1));
  ASSERT_EQ(1, partitioned_hypergraph.partID(2));
  ASSERT_EQ(1, partitioned_hypergraph.partID(3));
  ASSERT_EQ(1, partitioned_hypergraph.partID(4));
  ASSERT_EQ(1, partitioned_hypergraph.partID(5));
  ASSERT_EQ(1, partitioned_hypergraph.partID(6));
}

TEST_F(AInitialPartitioningDataContainer, AppliesBestPartitionToHypergraphInParallel1) {
  PartitionedHypergraph partitioned_hypergraph(
    context.partition.k, hypergraph);
  InitialPartitioningDataContainer<TypeTraits> ip_data(
    partitioned_hypergraph, context, true);

  std::atomic<size_t> cnt(0);
  tbb::parallel_invoke([&] {
    ++cnt;
    while(cnt < 2) { }
    PartitionedHypergraph& local_hg = ip_data.local_partitioned_hypergraph();
    // Cut = 3
    local_hg.setNodePart(0, 0);
    local_hg.setNodePart(1, 0);
    local_hg.setNodePart(2, 0);
    local_hg.setNodePart(3, 0);
    local_hg.setNodePart(4, 1);
    local_hg.setNodePart(5, 1);
    local_hg.setNodePart(6, 1);
    ip_data.commit(InitialPartitioningAlgorithm::random);
  }, [&] {
    ++cnt;
    while(cnt < 2) { }
    PartitionedHypergraph& local_hg = ip_data.local_partitioned_hypergraph();
    // Cut = 2
    local_hg.setNodePart(0, 0);
    local_hg.setNodePart(1, 0);
    local_hg.setNodePart(2, 0);
    local_hg.setNodePart(3, 1);
    local_hg.setNodePart(4, 1);
    local_hg.setNodePart(5, 1);
    local_hg.setNodePart(6, 1);
    ip_data.commit(InitialPartitioningAlgorithm::random);
  });

  ip_data.apply();

  ASSERT_EQ(2, metrics::quality(partitioned_hypergraph, context.partition.objective));
  ASSERT_EQ(0, partitioned_hypergraph.partID(0));
  ASSERT_EQ(0, partitioned_hypergraph.partID(1));
  ASSERT_EQ(0, partitioned_hypergraph.partID(2));
  ASSERT_EQ(1, partitioned_hypergraph.partID(3));
  ASSERT_EQ(1, partitioned_hypergraph.partID(4));
  ASSERT_EQ(1, partitioned_hypergraph.partID(5));
  ASSERT_EQ(1, partitioned_hypergraph.partID(6));
}

TEST_F(AInitialPartitioningDataContainer, AppliesBestPartitionToHypergraphInParallel2) {
  PartitionedHypergraph partitioned_hypergraph(
    context.partition.k, hypergraph);
  InitialPartitioningDataContainer<TypeTraits> ip_data(
    partitioned_hypergraph, context, true);

  std::atomic<size_t> cnt(0);
  tbb::parallel_invoke([&] {
    ++cnt;
    while(cnt < 2) { }
    PartitionedHypergraph& local_hg = ip_data.local_partitioned_hypergraph();
    // Cut = 1, but imbalanced
    local_hg.setNodePart(0, 1);
    local_hg.setNodePart(1, 0);
    local_hg.setNodePart(2, 1);
    local_hg.setNodePart(3, 1);
    local_hg.setNodePart(4, 1);
    local_hg.setNodePart(5, 1);
    local_hg.setNodePart(6, 1);
    ip_data.commit(InitialPartitioningAlgorithm::random);
  }, [&] {
    ++cnt;
    while(cnt < 2) { }
    PartitionedHypergraph& local_hg = ip_data.local_partitioned_hypergraph();
    // Cut = 2, but balanced
    local_hg.setNodePart(0, 0);
    local_hg.setNodePart(1, 0);
    local_hg.setNodePart(2, 0);
    local_hg.setNodePart(3, 1);
    local_hg.setNodePart(4, 1);
    local_hg.setNodePart(5, 1);
    local_hg.setNodePart(6, 1);
    ip_data.commit(InitialPartitioningAlgorithm::random);
  });

  ip_data.apply();

  ASSERT_EQ(2, metrics::quality(partitioned_hypergraph, context.partition.objective));
  ASSERT_EQ(0, partitioned_hypergraph.partID(0));
  ASSERT_EQ(0, partitioned_hypergraph.partID(1));
  ASSERT_EQ(0, partitioned_hypergraph.partID(2));
  ASSERT_EQ(1, partitioned_hypergraph.partID(3));
  ASSERT_EQ(1, partitioned_hypergraph.partID(4));
  ASSERT_EQ(1, partitioned_hypergraph.partID(5));
  ASSERT_EQ(1, partitioned_hypergraph.partID(6));
}

TEST_F(AInitialPartitioningDataContainer, AppliesBestPartitionToHypergraphInParallel3) {
  PartitionedHypergraph partitioned_hypergraph(
    context.partition.k, hypergraph);
  InitialPartitioningDataContainer<TypeTraits> ip_data(
    partitioned_hypergraph, context, true);

  std::atomic<size_t> cnt(0);
  tbb::parallel_invoke([&] {
    ++cnt;
    while(cnt < 2) { }
    PartitionedHypergraph& local_hg = ip_data.local_partitioned_hypergraph();
    // Cut = 3
    local_hg.setNodePart(0, 0);
    local_hg.setNodePart(1, 0);
    local_hg.setNodePart(2, 0);
    local_hg.setNodePart(3, 0);
    local_hg.setNodePart(4, 1);
    local_hg.setNodePart(5, 1);
    local_hg.setNodePart(6, 1);
    ip_data.commit(InitialPartitioningAlgorithm::random);
  }, [&] {
    ++cnt;
    while(cnt < 2) { }
    PartitionedHypergraph& local_hg = ip_data.local_partitioned_hypergraph();
    // Cut = 2
    local_hg.setNodePart(0, 0);
    local_hg.setNodePart(1, 0);
    local_hg.setNodePart(2, 1);
    local_hg.setNodePart(3, 0);
    local_hg.setNodePart(4, 0);
    local_hg.setNodePart(5, 1);
    local_hg.setNodePart(6, 1);
    ip_data.commit(InitialPartitioningAlgorithm::random);
  });

  ip_data.apply();

  ASSERT_EQ(2, metrics::quality(partitioned_hypergraph, context.partition.objective));
  ASSERT_EQ(0, partitioned_hypergraph.partID(0));
  ASSERT_EQ(0, partitioned_hypergraph.partID(1));
  ASSERT_EQ(1, partitioned_hypergraph.partID(2));
  ASSERT_EQ(0, partitioned_hypergraph.partID(3));
  ASSERT_EQ(0, partitioned_hypergraph.partID(4));
  ASSERT_EQ(1, partitioned_hypergraph.partID(5));
  ASSERT_EQ(1, partitioned_hypergraph.partID(6));
}


}  // namespace mt_kahypar
