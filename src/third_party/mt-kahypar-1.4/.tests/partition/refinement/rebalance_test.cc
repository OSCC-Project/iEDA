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

#include <functional>
#include <random>


#include "gmock/gmock.h"

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/io/hypergraph_factory.h"
#include "mt-kahypar/partition/refinement/rebalancing/simple_rebalancer.h"
#include "mt-kahypar/partition/refinement/gains/gain_definitions.h"

using ::testing::Test;

namespace mt_kahypar {

namespace {
  using TypeTraits = StaticHypergraphTypeTraits;
  using Hypergraph = typename TypeTraits::Hypergraph;
  using PartitionedHypergraph = typename TypeTraits::PartitionedHypergraph;
  using Km1Rebalancer = SimpleRebalancer<GraphAndGainTypes<TypeTraits, Km1GainTypes>>;
}


TEST(RebalanceTests, HeapSortWithMoveGainComparator) {
  vec<Move> moves;
  vec<Gain> gains = { 52, 12, 72, -154, 2672, -717, 1346, -7111, -113461, 136682, 3833 };

  for (HypernodeID i = 0; i < gains.size(); ++i) {
    moves.push_back(Move{-1, -1 , i, gains[i]});
  }

  std::make_heap(moves.begin(), moves.end(), Km1Rebalancer::MoveGainComparator());
  for (size_t i = 0; i < moves.size(); ++i) {
    std::pop_heap(moves.begin(), moves.end() - i, Km1Rebalancer::MoveGainComparator());
  }

  // assert that moves is sorted descendingly
  ASSERT_TRUE(std::is_sorted(moves.begin(), moves.end(),
                         [](const Move& lhs, const Move& rhs) {
    return lhs.gain > rhs.gain;
  }) );
}

TEST(RebalanceTests, FindsMoves) {
  PartitionID k = 8;
  Context context;
  context.partition.k = k;
  context.partition.epsilon = 0.03;
  Hypergraph hg = io::readInputFile<Hypergraph>(
    "../tests/instances/contracted_ibm01.hgr", FileFormat::hMetis,
    true /* enable stable construction */);
  context.setupPartWeights(hg.totalWeight());
  PartitionedHypergraph phg = PartitionedHypergraph(k, hg);

  HypernodeID nodes_per_part = hg.initialNumNodes() / (k-4);
  ASSERT(hg.initialNumNodes() % (k - 4) == 0);
  for (PartitionID i = 0; i < k - 4; ++i) {
    for (HypernodeID u = i * nodes_per_part; u < (i+1) * nodes_per_part; ++u) {
      phg.setOnlyNodePart(u, i);
    }
  }
  phg.initializePartition();
  Km1GainCache gain_cache;
  gain_cache.initializeGainCache(phg);

  Km1Rebalancer rebalancer(context);
  vec<Move> moves_to_empty_blocks = rebalancer.repairEmptyBlocks(phg);

  ASSERT_EQ(moves_to_empty_blocks.size(), 4);

  for (Move& m : moves_to_empty_blocks) {
    ASSERT_EQ(gain_cache.gain(m.node, m.from, m.to), m.gain);
    Gain recomputed_gain = gain_cache.recomputeBenefitTerm(phg, m.node, m.to) -
      gain_cache.recomputePenaltyTerm(phg, m.node);
    if (recomputed_gain == 0) {
      ASSERT_TRUE([&]() {
        for (HyperedgeID e : phg.incidentEdges(m.node)) {
          if (phg.pinCountInPart(e, m.from) != 1) {
            return false;
          }
        }
        return true;
      }());
    }
    ASSERT_EQ(m.gain, recomputed_gain);
    ASSERT_EQ(m.from, phg.partID(m.node));
    ASSERT_EQ(phg.partWeight(m.to), 0);
    ASSERT_GE(m.to, k - 4);
    ASSERT_LT(m.from, k - 4);
    phg.changeNodePart(gain_cache, m.node, m.from, m.to);
  }

  moves_to_empty_blocks = rebalancer.repairEmptyBlocks(phg);
  ASSERT_EQ(moves_to_empty_blocks.size(), 0);
}

}