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

#include <atomic>
#include <mt-kahypar/parallel/tbb_initializer.h>

#include "gmock/gmock.h"

#include "tests/datastructures/hypergraph_fixtures.h"
#include "mt-kahypar/definitions.h"
#include "mt-kahypar/datastructures/hypergraph_common.h"
#include "mt-kahypar/datastructures/delta_partitioned_hypergraph.h"
#include "mt-kahypar/partition/refinement/gains/km1/km1_gain_cache.h"

using ::testing::Test;

namespace mt_kahypar {
namespace ds {

class ADeltaPartitionedHypergraph : public Test {

 using Hypergraph = typename StaticHypergraphTypeTraits::Hypergraph;
 using HypergraphFactory = typename Hypergraph::Factory;
 using PartitionedHypergraph = typename StaticHypergraphTypeTraits::PartitionedHypergraph;
 using DeltaPartitionedHypergraph = typename PartitionedHypergraph::DeltaPartition<true>;
 using GainCache = Km1GainCache;
 using DeltaGainCache = DeltaKm1GainCache;

 public:

  ADeltaPartitionedHypergraph() :
    hg(HypergraphFactory::construct(
      7 , 4, { {0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6} })),
    phg(3, hg, parallel_tag_t()),
    context(),
    gain_cache(),
    delta_phg(nullptr),
    delta_gain_cache(nullptr) {
    phg.setOnlyNodePart(0, 0);
    phg.setOnlyNodePart(1, 0);
    phg.setOnlyNodePart(2, 0);
    phg.setOnlyNodePart(3, 1);
    phg.setOnlyNodePart(4, 1);
    phg.setOnlyNodePart(5, 2);
    phg.setOnlyNodePart(6, 2);
    phg.initializePartition();
    gain_cache.initializeGainCache(phg);

    context.partition.k = 3;
    delta_phg = std::make_unique<DeltaPartitionedHypergraph>(context);
    delta_gain_cache = std::make_unique<DeltaGainCache>(gain_cache);
    delta_phg->setPartitionedHypergraph(&phg);
  }

  void verifyPinCounts(const HyperedgeID he,
                       const std::vector<HypernodeID>& expected_pin_counts) {
    ASSERT(expected_pin_counts.size() == static_cast<size_t>(phg.k()));
    for (PartitionID block = 0; block < 3; ++block) {
      ASSERT_EQ(expected_pin_counts[block], delta_phg->pinCountInPart(he, block)) << V(he) << V(block);
    }
  }
  void verifyConnectivitySet(const HyperedgeID he,
                             const std::vector<PartitionID>& expected_connectivity_set) {
    ASSERT_EQ(delta_phg->connectivity(he), static_cast<PartitionID>(expected_connectivity_set.size()));
    size_t idx = 0;
    for ( const PartitionID block : delta_phg->connectivitySet(he) ) {
      ASSERT_EQ(block, expected_connectivity_set[idx++]);
    }
    ASSERT_EQ(idx, expected_connectivity_set.size());
  }

  void verifyBenefitTerm(const HypernodeID hn,
                         const std::vector<HypernodeID>& expected_penalties) {
    ASSERT(expected_penalties.size() == static_cast<size_t>(phg.k()));
    for (PartitionID block = 0; block < 3; ++block) {
      ASSERT_EQ(expected_penalties[block], delta_gain_cache->benefitTerm(hn, block)) << V(hn) << V(block);
    }
  }

  void changeNodePartWithGainCacheUpdate(const HypernodeID hn,
                                         const PartitionID from,
                                         const PartitionID to) {
    auto delta_gain_update = [&](const SynchronizedEdgeUpdate& sync_update) {
      delta_gain_cache->deltaGainUpdate(*delta_phg, sync_update);
    };
    delta_phg->changeNodePart(hn, from, to, 1000, delta_gain_update);
  }

  Hypergraph hg;
  PartitionedHypergraph phg;
  Context context;
  GainCache gain_cache;
  std::unique_ptr<DeltaPartitionedHypergraph> delta_phg;
  std::unique_ptr<DeltaGainCache> delta_gain_cache;
};

TEST_F(ADeltaPartitionedHypergraph, VerifiesInitialPinCounts) {
  verifyPinCounts(0, { 2, 0, 0 });
  verifyPinCounts(1, { 2, 2, 0 });
  verifyPinCounts(2, { 0, 2, 1 });
  verifyPinCounts(3, { 1, 0, 2 });
}

TEST_F(ADeltaPartitionedHypergraph, VerifiesInitialConnectivitySets) {
  verifyConnectivitySet(0, { 0 });
  verifyConnectivitySet(1, { 0, 1 });
  verifyConnectivitySet(2, { 1, 2 });
  verifyConnectivitySet(3, { 0, 2 });
}

TEST_F(ADeltaPartitionedHypergraph, VerifiesInitialPenaltyTerms) {
  ASSERT_EQ(2, delta_gain_cache->penaltyTerm(0, kInvalidPartition));
  ASSERT_EQ(1, delta_gain_cache->penaltyTerm(1, kInvalidPartition));
  ASSERT_EQ(1, delta_gain_cache->penaltyTerm(2, kInvalidPartition));
  ASSERT_EQ(2, delta_gain_cache->penaltyTerm(3, kInvalidPartition));
  ASSERT_EQ(2, delta_gain_cache->penaltyTerm(4, kInvalidPartition));
  ASSERT_EQ(1, delta_gain_cache->penaltyTerm(5, kInvalidPartition));
  ASSERT_EQ(1, delta_gain_cache->penaltyTerm(6, kInvalidPartition));
}

TEST_F(ADeltaPartitionedHypergraph, VerifiesInitialBenefitTerms) {
  verifyBenefitTerm(0, { 2, 1, 0 });
  verifyBenefitTerm(1, { 1, 1, 0 });
  verifyBenefitTerm(2, { 2, 0, 1 });
  verifyBenefitTerm(3, { 1, 2, 1 });
  verifyBenefitTerm(4, { 1, 2, 1 });
  verifyBenefitTerm(5, { 1, 0, 1 });
  verifyBenefitTerm(6, { 1, 1, 2 });
}
TEST_F(ADeltaPartitionedHypergraph, MovesAVertex1) {
  changeNodePartWithGainCacheUpdate(1, 0, 1);
  ASSERT_EQ(0, phg.partID(1));
  ASSERT_EQ(1, delta_phg->partID(1));

  // Verify Pin Counts and Connectivity Set
  verifyPinCounts(1, { 1, 3, 0 });
  verifyConnectivitySet(1, { 0, 1 });

  // Verify Move From Benefit
  ASSERT_EQ(1, delta_gain_cache->penaltyTerm(0, kInvalidPartition));
  ASSERT_EQ(2, delta_gain_cache->penaltyTerm(3, kInvalidPartition));
  ASSERT_EQ(2, delta_gain_cache->penaltyTerm(4, kInvalidPartition));

  // Verify Move To Penalty
  verifyBenefitTerm(0, { 2, 1, 0 });
  verifyBenefitTerm(3, { 1, 2, 1 });
  verifyBenefitTerm(4, { 1, 2, 1 });
}

TEST_F(ADeltaPartitionedHypergraph, MovesAVertex2) {
  changeNodePartWithGainCacheUpdate(6, 2, 1);
  ASSERT_EQ(2, phg.partID(6));
  ASSERT_EQ(1, delta_phg->partID(6));

  // Verify Pin Counts and Connectivity Set
  verifyPinCounts(2, { 0, 3, 0 });
  verifyPinCounts(3, { 1, 1, 1 });
  verifyConnectivitySet(2, { 1 });
  verifyConnectivitySet(3, { 0, 1, 2 });

  // Verify Move From Benefit
  ASSERT_EQ(1, delta_gain_cache->penaltyTerm(2, kInvalidPartition));
  ASSERT_EQ(2, delta_gain_cache->penaltyTerm(3, kInvalidPartition));
  ASSERT_EQ(2, delta_gain_cache->penaltyTerm(4, kInvalidPartition));
  ASSERT_EQ(0, delta_gain_cache->penaltyTerm(5, kInvalidPartition));

  // Verify Move To Penalty
  verifyBenefitTerm(2, { 2, 1, 1 });
  verifyBenefitTerm(3, { 1, 2, 0 });
  verifyBenefitTerm(4, { 1, 2, 0 });
  verifyBenefitTerm(5, { 1, 1, 1 });
}

TEST_F(ADeltaPartitionedHypergraph, MovesSeveralVertices) {
  changeNodePartWithGainCacheUpdate(6, 2, 1);
  changeNodePartWithGainCacheUpdate(2, 0, 1);
  changeNodePartWithGainCacheUpdate(5, 2, 1);
  ASSERT_EQ(0, phg.partID(2));
  ASSERT_EQ(2, phg.partID(5));
  ASSERT_EQ(2, phg.partID(6));
  ASSERT_EQ(1, delta_phg->partID(2));
  ASSERT_EQ(1, delta_phg->partID(5));
  ASSERT_EQ(1, delta_phg->partID(6));

  // Verify Pin Counts and Connectivity Set
  verifyPinCounts(0, { 1, 1, 0 });
  verifyPinCounts(1, { 2, 2, 0 });
  verifyPinCounts(2, { 0, 3, 0 });
  verifyPinCounts(3, { 0, 3, 0 });
  verifyConnectivitySet(0, { 0, 1 });
  verifyConnectivitySet(1, { 0, 1 });
  verifyConnectivitySet(2, { 1 });
  verifyConnectivitySet(3, { 1 });

  // Verify Move From Benefit
  ASSERT_EQ(1, delta_gain_cache->penaltyTerm(0, kInvalidPartition));
  ASSERT_EQ(1, delta_gain_cache->penaltyTerm(1, kInvalidPartition));
  ASSERT_EQ(2, delta_gain_cache->penaltyTerm(3, kInvalidPartition));
  ASSERT_EQ(2, delta_gain_cache->penaltyTerm(4, kInvalidPartition));

  // Verify Move To Penalty
  verifyBenefitTerm(0, { 2, 2, 0 });
  verifyBenefitTerm(1, { 1, 1, 0 });
  verifyBenefitTerm(2, { 1, 2, 0 });
  verifyBenefitTerm(3, { 1, 2, 0 });
  verifyBenefitTerm(4, { 1, 2, 0 });
  verifyBenefitTerm(5, { 0, 1, 0 });
  verifyBenefitTerm(6, { 0, 2, 0 });
}

} // namespace ds
} // namespace mt_kahypar
