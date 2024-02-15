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

#include "tests/definitions.h"
#include "mt-kahypar/io/hypergraph_factory.h"
#include "mt-kahypar/partition/refinement/gains/km1/km1_gain_cache.h"

using ::testing::Test;

namespace mt_kahypar {
namespace ds {

template<typename TypeTraits>
class AGainUpdate : public Test {

  using Hypergraph = typename TypeTraits::Hypergraph;
  using PartitionedHypergraph = typename TypeTraits::PartitionedHypergraph;

 public:
  AGainUpdate() :
    hg(io::readInputFile<Hypergraph>(
      "../tests/instances/twocenters.hgr", FileFormat::hMetis, true)),
    phg(),
    gain_cache() {
    phg = PartitionedHypergraph(2, hg);
  }

  Hypergraph hg;
  PartitionedHypergraph phg;
  Km1GainCache gain_cache;
};

TYPED_TEST_CASE(AGainUpdate, tests::HypergraphTestTypeTraits);

TYPED_TEST(AGainUpdate, Example1) {
  this->phg.setNodePart(0, 0);
  this->phg.setNodePart(1, 0);
  for (HypernodeID u = 4; u < 12; ++u) {
    this->phg.setNodePart(u, 0);
  }

  this->phg.setNodePart(2, 1);
  this->phg.setNodePart(3, 1);
  for (HypernodeID u = 12; u < 20; ++u) {
    this->phg.setNodePart(u, 1);
  }

  ASSERT_EQ(this->phg.partWeight(0), this->phg.partWeight(1));
  ASSERT_EQ(this->phg.partWeight(0), 10);

  this->gain_cache.initializeGainCache(this->phg);
  ASSERT_EQ(this->gain_cache.gain(0, this->phg.partID(0), 1), -1);
  ASSERT_EQ(this->gain_cache.penaltyTerm(0, kInvalidPartition), 2);
  ASSERT_EQ(this->gain_cache.benefitTerm(0, 1), 1);

  ASSERT_EQ(this->gain_cache.gain(2, this->phg.partID(2), 0), -1);

  ASSERT_EQ(this->gain_cache.gain(4, this->phg.partID(4), 1), -1);
  ASSERT_EQ(this->gain_cache.gain(6, this->phg.partID(6), 1), -2);

  ASSERT_EQ(this->gain_cache.gain(12, this->phg.partID(12), 0), -1);
  ASSERT_EQ(this->gain_cache.gain(14, this->phg.partID(14), 0), -2);

  this->phg.changeNodePart(this->gain_cache, 8, 0, 1);

  this->gain_cache.recomputeInvalidTerms(this->phg, 8);  // nodes are allowed to move once before moveFromPenalty must be recomputed
  ASSERT_EQ(this->gain_cache.gain(8, 1, 0), 2);

  ASSERT_EQ(this->gain_cache.gain(6, 0, 1), 0);
}


}  // namespace ds
}  // namespace mt_kahypar
