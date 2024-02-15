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

#include "mt-kahypar/macros.h"


#include "mt-kahypar/definitions.h"
#include "mt-kahypar/io/hypergraph_factory.h"

#include "mt-kahypar/partition/refinement/fm/global_rollback.h"
#include "mt-kahypar/partition/refinement/gains/gain_definitions.h"

#include "mt-kahypar/partition/metrics.h"

using ::testing::Test;

namespace mt_kahypar {

namespace {
  using TypeTraits = StaticHypergraphTypeTraits;
  using Hypergraph = typename TypeTraits::Hypergraph;
  using PartitionedHypergraph = typename TypeTraits::PartitionedHypergraph;
}


TEST(RollbackTests, GainRecalculationAndRollsbackCorrectly) {
  Hypergraph hg = io::readInputFile<Hypergraph>(
    "../tests/instances/twocenters.hgr", FileFormat::hMetis, true);
  PartitionID k = 2;

  PartitionedHypergraph phg(k, hg);
  phg.setNodePart(0, 1);
  phg.setNodePart(1, 1);
  for (HypernodeID u = 4; u < 12; ++u) {
    phg.setNodePart(u, 0);
  }
  phg.setNodePart(2, 0);
  phg.setNodePart(3, 0);
  for (HypernodeID u = 12; u < 20; ++u) {
    phg.setNodePart(u, 1);
  }
  Km1GainCache gain_cache;
  gain_cache.initializeGainCache(phg);

  Context context;
  context.partition.k = k;
  context.setupPartWeights(phg.totalWeight());
  context.partition.max_part_weights = { std::numeric_limits<HypernodeWeight>::max(), std::numeric_limits<HypernodeWeight>::max()};
  context.refinement.fm.rollback_balance_violation_factor = 0.0;


  FMSharedData sharedData(hg.initialNumNodes(), false);

  GlobalRollback<GraphAndGainTypes<TypeTraits, Km1GainTypes>> grb(
    hg.initialNumEdges(), context, gain_cache);
  auto performMove = [&](Move m) {
    if (phg.changeNodePart(gain_cache, m.node, m.from, m.to)) {
      sharedData.moveTracker.insertMove(m);
    }
  };

  ASSERT_EQ(3, gain_cache.gain(0, 1, 0));
  performMove({1, 0,  0,  3});
  ASSERT_EQ(3, gain_cache.gain(1, 1, 0));
  performMove({1, 0,  1,  3});
  ASSERT_EQ(1, gain_cache.gain(2, 0, 1));
  performMove({0, 1,  2,  1});
  ASSERT_EQ(1, gain_cache.gain(3, 0, 1));
  performMove({0, 1,  3,  1});

  ASSERT_EQ(gain_cache.gain(4, 0, 1), -1);
  performMove({0, 1, 4, -1});
  ASSERT_EQ(gain_cache.gain(5, 0, 1), 0);
  performMove({0, 1, 5, 0});

  vec<HypernodeWeight> dummy_part_weights(k, 0);
  grb.revertToBestPrefix(phg, sharedData, dummy_part_weights);
  // revert last two moves
  ASSERT_EQ(phg.partID(4), 0);
  ASSERT_EQ(phg.partID(5), 0);
  ASSERT_EQ(metrics::quality(phg, Objective::km1, false), 2);
}


TEST(RollbackTests, GainRecalculation2) {
  Hypergraph hg = io::readInputFile<Hypergraph>(
    "../tests/instances/twocenters.hgr", FileFormat::hMetis, true);
  PartitionID k = 2;
  PartitionedHypergraph phg(k, hg);
  phg.setNodePart(0, 1);
  phg.setNodePart(1, 1);
  for (HypernodeID u = 4; u < 12; ++u) {
    phg.setNodePart(u, 0);
  }
  phg.setNodePart(2, 0);
  phg.setNodePart(3, 0);
  for (HypernodeID u = 12; u < 20; ++u) {
    phg.setNodePart(u, 1);
  }
  Km1GainCache gain_cache;
  gain_cache.initializeGainCache(phg);

  Context context;
  context.partition.k = k;
  context.setupPartWeights(phg.totalWeight());
  context.partition.max_part_weights = { std::numeric_limits<HypernodeWeight>::max(), std::numeric_limits<HypernodeWeight>::max()};
  context.refinement.fm.rollback_balance_violation_factor = 0.0;

  FMSharedData sharedData(hg.initialNumNodes(), false);

  GlobalRollback<GraphAndGainTypes<TypeTraits, Km1GainTypes>> grb(
    hg.initialNumEdges(), context, gain_cache);

  auto performUpdates = [&](Move& m) {
   sharedData.moveTracker.insertMove(m);
  };

  vec<Gain> expected_gains = { 3, 1 };

  ASSERT_EQ(gain_cache.gain(2, 0, 1), 3);
  Move move_2 = { 0, 1, 2, 3 };
  phg.changeNodePart(gain_cache, move_2.node, move_2.from, move_2.to);

  ASSERT_EQ(gain_cache.gain(0, 1, 0), 1);
  Move move_0 = { 1, 0, 0, 1 };
  phg.changeNodePart(gain_cache, move_0.node, move_0.from, move_0.to);

  performUpdates(move_0);
  performUpdates(move_2);

  grb.recalculateGains(phg, sharedData);
  grb.verifyGains(phg, sharedData);
}

}   // namespace mt_kahypar
