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
#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/partition/refinement/gains/km1/km1_gain_computation.h"
#include "mt-kahypar/partition/refinement/gains/cut/cut_gain_computation.h"

using ::testing::Test;

namespace mt_kahypar {

namespace {
  using TypeTraits = StaticHypergraphTypeTraits;
  using Hypergraph = typename TypeTraits::Hypergraph;
  using PartitionedHypergraph = typename TypeTraits::PartitionedHypergraph;
}

template <typename GainCalculator, PartitionID K>
class AGainPolicy : public Test {
 public:
  using HypergraphFactory = typename Hypergraph::Factory;

  AGainPolicy() :
    hg(HypergraphFactory::construct(7 , 4, { {0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6} })),
    context(),
    gain(nullptr) {
    context.partition.k = K;
    context.partition.max_part_weights.assign(K, std::numeric_limits<HypernodeWeight>::max());
    gain = std::make_unique<GainCalculator>(context, true  /* disable randomization */);
    hypergraph = PartitionedHypergraph(K, hg, parallel_tag_t());
  }

  void assignPartitionIDs(const std::vector<PartitionID>& part_ids) {
    HypernodeID hn = 0;
    for (const PartitionID& part : part_ids) {
      ASSERT(part < K);
      hypergraph.setNodePart(hn++, part);
    }
  }

  Hypergraph hg;
  PartitionedHypergraph hypergraph;
  Context context;
  std::unique_ptr<GainCalculator> gain;
};

using AKm1PolicyK2 = AGainPolicy<Km1GainComputation, 2>;

TEST_F(AKm1PolicyK2, ComputesCorrectMoveGainForVertex1) {
  assignPartitionIDs({ 1, 0, 0, 0, 0, 1, 1 });
  Move move = gain->computeMaxGainMove(hypergraph, 0);
  ASSERT_EQ(1, move.from);
  ASSERT_EQ(0, move.to);
  ASSERT_EQ(-2, move.gain);
}

TEST_F(AKm1PolicyK2, ComputesCorrectObjectiveDelta1) {
  assignPartitionIDs({ 1, 0, 0, 0, 0, 1, 1 });
  ASSERT_TRUE(hypergraph.changeNodePart(0, 1, 0,
    [&](const SynchronizedEdgeUpdate& sync_update) {
      gain->computeDeltaForHyperedge(sync_update);
    }));
  ASSERT_EQ(-2, gain->delta());
}

TEST_F(AKm1PolicyK2, ComputesCorrectMoveGainForVertex2) {
  assignPartitionIDs({ 0, 0, 0, 1, 0, 1, 1 });
  Move move = gain->computeMaxGainMove(hypergraph, 3);
  ASSERT_EQ(1, move.from);
  ASSERT_EQ(0, move.to);
  ASSERT_EQ(-1, move.gain);
}

TEST_F(AKm1PolicyK2, ComputesCorrectObjectiveDelta2) {
  assignPartitionIDs({ 0, 0, 0, 1, 0, 1, 1 });
  ASSERT_TRUE(hypergraph.changeNodePart(3, 1, 0,
    [&](const SynchronizedEdgeUpdate& sync_update) {
      gain->computeDeltaForHyperedge(sync_update);
    }));
  ASSERT_EQ(-1, gain->delta());
}

TEST_F(AKm1PolicyK2, ComputesCorrectMoveGainForVertex3) {
  assignPartitionIDs({ 0, 0, 0, 0, 0, 1, 1 });
  Move move = gain->computeMaxGainMove(hypergraph, 4);
  ASSERT_EQ(0, move.from);
  ASSERT_EQ(0, move.to);
  ASSERT_EQ(0, move.gain);
}

using ACutPolicyK2 = AGainPolicy<CutGainComputation, 2>;

TEST_F(ACutPolicyK2, ComputesCorrectMoveGainForVertex1) {
  assignPartitionIDs({ 1, 0, 0, 0, 0, 1, 1 });
  Move move = gain->computeMaxGainMove(hypergraph, 0);
  ASSERT_EQ(1, move.from);
  ASSERT_EQ(0, move.to);
  ASSERT_EQ(-2, move.gain);
}

TEST_F(ACutPolicyK2, ComputesCorrectObjectiveDelta1) {
  assignPartitionIDs({ 1, 0, 0, 0, 0, 1, 1 });
  ASSERT_TRUE(hypergraph.changeNodePart(0, 1, 0,
    [&](const SynchronizedEdgeUpdate& sync_update) {
      gain->computeDeltaForHyperedge(sync_update);
    }));
  ASSERT_EQ(-2, gain->delta());
}

TEST_F(ACutPolicyK2, ComputesCorrectMoveGainForVertex2) {
  assignPartitionIDs({ 0, 0, 0, 1, 0, 1, 1 });
  Move move = gain->computeMaxGainMove(hypergraph, 3);
  ASSERT_EQ(1, move.from);
  ASSERT_EQ(0, move.to);
  ASSERT_EQ(-1, move.gain);
}

TEST_F(ACutPolicyK2, ComputesCorrectObjectiveDelta2) {
  assignPartitionIDs({ 0, 0, 0, 1, 0, 1, 1 });
  ASSERT_TRUE(hypergraph.changeNodePart(3, 1, 0,
    [&](const SynchronizedEdgeUpdate& sync_update) {
      gain->computeDeltaForHyperedge(sync_update);
    }));
  ASSERT_EQ(-1, gain->delta());
}

TEST_F(ACutPolicyK2, ComputesCorrectMoveGainForVertex3) {
  assignPartitionIDs({ 0, 0, 0, 0, 0, 1, 1 });
  Move move = gain->computeMaxGainMove(hypergraph, 4);
  ASSERT_EQ(0, move.from);
  ASSERT_EQ(0, move.to);
  ASSERT_EQ(0, move.gain);
}

using AKm1PolicyK4 = AGainPolicy<Km1GainComputation, 4>;

TEST_F(AKm1PolicyK4, ComputesCorrectMoveGainForVertex1) {
  assignPartitionIDs({ 0, 1, 2, 3, 3, 1, 2 });
  Move move = gain->computeMaxGainMove(hypergraph, 0);
  ASSERT_EQ(0, move.from);
  ASSERT_EQ(2, move.to);
  ASSERT_EQ(-1, move.gain);
}

TEST_F(AKm1PolicyK4, ComputesCorrectObjectiveDelta1) {
  assignPartitionIDs({ 0, 1, 2, 3, 3, 1, 2 });
  ASSERT_TRUE(hypergraph.changeNodePart(0, 0, 1,
    [&](const SynchronizedEdgeUpdate& sync_update) {
      gain->computeDeltaForHyperedge(sync_update);
    }));
  ASSERT_EQ(-1, gain->delta());
}

TEST_F(AKm1PolicyK4, ComputesCorrectMoveGainForVertex2) {
  assignPartitionIDs({ 0, 3, 1, 2, 2, 0, 3 });
  Move move = gain->computeMaxGainMove(hypergraph, 6);
  ASSERT_EQ(3, move.from);
  ASSERT_EQ(2, move.to);
  ASSERT_EQ(-1, move.gain);
}

TEST_F(AKm1PolicyK4, ComputesCorrectObjectiveDelta2) {
  assignPartitionIDs({ 0, 3, 1, 2, 2, 0, 3 });
  ASSERT_TRUE(hypergraph.changeNodePart(6, 3, 0,
    [&](const SynchronizedEdgeUpdate& sync_update) {
      gain->computeDeltaForHyperedge(sync_update);
    }));
  ASSERT_EQ(-1, gain->delta());
}

TEST_F(AKm1PolicyK4, ComputesCorrectMoveGainForVertex3) {
  assignPartitionIDs({ 0, 3, 1, 2, 2, 0, 3 });
  Move move = gain->computeMaxGainMove(hypergraph, 3);
  ASSERT_EQ(2, move.from);
  ASSERT_EQ(2, move.to);
  ASSERT_EQ(0, move.gain);
}

using ACutPolicyK4 = AGainPolicy<CutGainComputation, 4>;

TEST_F(ACutPolicyK4, ComputesCorrectMoveGainForVertex1) {
  assignPartitionIDs({ 0, 1, 2, 3, 3, 1, 2 });
  Move move = gain->computeMaxGainMove(hypergraph, 0);
  ASSERT_EQ(0, move.from);
  ASSERT_EQ(2, move.to);
  ASSERT_EQ(-1, move.gain);
}

TEST_F(ACutPolicyK4, ComputesCorrectObjectiveDelta1) {
  assignPartitionIDs({ 0, 1, 2, 3, 3, 1, 2 });
  ASSERT_TRUE(hypergraph.changeNodePart(0, 0, 2,
    [&](const SynchronizedEdgeUpdate& sync_update) {
      gain->computeDeltaForHyperedge(sync_update);
    }));
  ASSERT_EQ(-1, gain->delta());
}

TEST_F(ACutPolicyK4, ComputesCorrectMoveGainForVertex2) {
  assignPartitionIDs({ 0, 3, 1, 2, 2, 0, 3 });
  Move move = gain->computeMaxGainMove(hypergraph, 6);
  ASSERT_EQ(3, move.from);
  ASSERT_EQ(2, move.to);
  ASSERT_EQ(-1, move.gain);
}

TEST_F(ACutPolicyK4, ComputesCorrectObjectiveDelta2) {
  assignPartitionIDs({ 0, 3, 1, 2, 2, 0, 3 });
  ASSERT_TRUE(hypergraph.changeNodePart(6, 3, 2,
    [&](const SynchronizedEdgeUpdate& sync_update) {
      gain->computeDeltaForHyperedge(sync_update);
    }));
  ASSERT_EQ(-1, gain->delta());
}

TEST_F(ACutPolicyK4, ComputesCorrectMoveGainForVertex3) {
  assignPartitionIDs({ 0, 3, 1, 2, 2, 0, 3 });
  Move move = gain->computeMaxGainMove(hypergraph, 3);
  ASSERT_EQ(2, move.from);
  ASSERT_EQ(2, move.to);
  ASSERT_EQ(0, move.gain);
}
}  // namespace mt_kahypar
