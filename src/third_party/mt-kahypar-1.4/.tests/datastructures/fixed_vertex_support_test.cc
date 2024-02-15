/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2023 Tobias Heuer <tobias.heuer@kit.edu>
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
#include "tbb/task_group.h"

#include "mt-kahypar/datastructures/static_hypergraph.h"
#include "mt-kahypar/datastructures/static_hypergraph_factory.h"
#include "mt-kahypar/datastructures/fixed_vertex_support.h"

using ::testing::Test;

namespace mt_kahypar {
namespace ds {


class AFixedVertexSupport : public Test {

 public:
  using Hypergraph = StaticHypergraph;
  using Factory = typename Hypergraph::Factory;

  AFixedVertexSupport() :
    hypergraph(Factory::construct(
      7 , 4, { {0, 2}, {0, 1, 3, 4}, {3, 4, 6}, {2, 5, 6} })),
    fixed_vertices(7, 3) {
    fixed_vertices.setHypergraph(&hypergraph);
    fixed_vertices.fixToBlock(0, 0);
    fixed_vertices.fixToBlock(2, 0);
    fixed_vertices.fixToBlock(4, 1);
    fixed_vertices.fixToBlock(6, 2);
  }

  void verifyFixedVertices(const std::string& desc, const vec<PartitionID>& expected) {
    HypernodeWeight total_weight = 0;
    vec<HypernodeWeight> block_weight(3, 0);
    for ( const HypernodeID& hn : hypergraph.nodes() ) {
      const bool is_fixed = expected[hn] != kInvalidPartition;
      ASSERT_EQ(is_fixed, fixed_vertices.isFixed(hn)) << V(hn) << " " << V(desc);
      ASSERT_EQ(expected[hn], fixed_vertices.fixedVertexBlock(hn)) << V(hn) << " " << V(desc);
      if ( is_fixed ) {
        total_weight += hypergraph.nodeWeight(hn);
        block_weight[expected[hn]] += hypergraph.nodeWeight(hn);
      }
    }
    ASSERT_EQ(total_weight, fixed_vertices.totalFixedVertexWeight()) << V(desc);
    for ( PartitionID i = 0; i < 3; ++i ) {
      ASSERT_EQ(block_weight[i], fixed_vertices.fixedVertexBlockWeight(i)) << V(i) << " " << V(desc);
    }
  }

  void verifyFixedVertices(const vec<PartitionID>& expected) {
    verifyFixedVertices("", expected);
  }

  Hypergraph hypergraph;
  FixedVertexSupport<Hypergraph> fixed_vertices;
};

template <class F, class K>
void runParallel(F f1, K f2) {
  std::atomic<int> cnt(0);
  tbb::task_group group;

  group.run([&] {
        cnt++;
        while (cnt < 2) { }
        f1();
      });

  group.run([&] {
        cnt++;
        while (cnt < 2) { }
        f2();
      });

  group.wait();
}

TEST_F(AFixedVertexSupport, CheckIfNodesAreFixed) {
  ASSERT_TRUE(fixed_vertices.hasFixedVertices());
  ASSERT_TRUE(fixed_vertices.isFixed(0));
  ASSERT_FALSE(fixed_vertices.isFixed(1));
  ASSERT_TRUE(fixed_vertices.isFixed(2));
  ASSERT_FALSE(fixed_vertices.isFixed(3));
  ASSERT_TRUE(fixed_vertices.isFixed(4));
  ASSERT_FALSE(fixed_vertices.isFixed(5));
  ASSERT_TRUE(fixed_vertices.isFixed(6));
}

TEST_F(AFixedVertexSupport, CheckFixedVertexBlocks) {
  ASSERT_EQ(0, fixed_vertices.fixedVertexBlock(0));
  ASSERT_EQ(kInvalidPartition, fixed_vertices.fixedVertexBlock(1));
  ASSERT_EQ(0, fixed_vertices.fixedVertexBlock(2));
  ASSERT_EQ(kInvalidPartition, fixed_vertices.fixedVertexBlock(3));
  ASSERT_EQ(1, fixed_vertices.fixedVertexBlock(4));
  ASSERT_EQ(kInvalidPartition, fixed_vertices.fixedVertexBlock(5));
  ASSERT_EQ(2, fixed_vertices.fixedVertexBlock(6));
}

TEST_F(AFixedVertexSupport, CheckFixedVertexBlockWeights) {
  ASSERT_EQ(4, fixed_vertices.totalFixedVertexWeight());
  ASSERT_EQ(2, fixed_vertices.fixedVertexBlockWeight(0));
  ASSERT_EQ(1, fixed_vertices.fixedVertexBlockWeight(1));
  ASSERT_EQ(1, fixed_vertices.fixedVertexBlockWeight(2));
}

TEST_F(AFixedVertexSupport, ContractFreeOntoFreeVertex) {
  ASSERT_TRUE(fixed_vertices.contract(3, 5));
  ASSERT_FALSE(fixed_vertices.isFixed(3));
  ASSERT_FALSE(fixed_vertices.isFixed(5));
}

TEST_F(AFixedVertexSupport, ContractFreeOntoFixedVertex) {
  ASSERT_TRUE(fixed_vertices.contract(0, 3));
  ASSERT_TRUE(fixed_vertices.isFixed(3));
  ASSERT_EQ(0, fixed_vertices.fixedVertexBlock(3));
  ASSERT_TRUE(fixed_vertices.isFixed(0));
  ASSERT_EQ(0, fixed_vertices.fixedVertexBlock(0));
  ASSERT_EQ(3, fixed_vertices.fixedVertexBlockWeight(0));
  ASSERT_EQ(5, fixed_vertices.totalFixedVertexWeight());
}

TEST_F(AFixedVertexSupport, ContractFixedOntoFixedVertex1) {
  ASSERT_TRUE(fixed_vertices.contract(0, 2));
  ASSERT_TRUE(fixed_vertices.isFixed(0));
  ASSERT_EQ(0, fixed_vertices.fixedVertexBlock(0));
  ASSERT_TRUE(fixed_vertices.isFixed(2));
  ASSERT_EQ(0, fixed_vertices.fixedVertexBlock(2));
}

TEST_F(AFixedVertexSupport, ContractFixedOntoFixedVertex2) {
  ASSERT_FALSE(fixed_vertices.contract(0, 6));
  ASSERT_TRUE(fixed_vertices.isFixed(0));
  ASSERT_EQ(0, fixed_vertices.fixedVertexBlock(0));
  ASSERT_TRUE(fixed_vertices.isFixed(6));
  ASSERT_EQ(2, fixed_vertices.fixedVertexBlock(6));
}

TEST_F(AFixedVertexSupport, ContractFixedOntoFreeVertex) {
  ASSERT_TRUE(fixed_vertices.contract(1, 4));
  ASSERT_TRUE(fixed_vertices.isFixed(1));
  ASSERT_EQ(1, fixed_vertices.fixedVertexBlock(1));
  ASSERT_EQ(2, fixed_vertices.fixedVertexBlockWeight(1));
  ASSERT_EQ(5, fixed_vertices.totalFixedVertexWeight());
}

TEST_F(AFixedVertexSupport, UnontractFreeOntoFreeVertex) {
  ASSERT_TRUE(fixed_vertices.contract(3, 5));
  fixed_vertices.uncontract(3, 5);
  ASSERT_FALSE(fixed_vertices.isFixed(3));
  ASSERT_FALSE(fixed_vertices.isFixed(5));
}

TEST_F(AFixedVertexSupport, UnontractFreeOntoFixedVertex) {
  ASSERT_TRUE(fixed_vertices.contract(0, 3));
  fixed_vertices.uncontract(0, 3);
  ASSERT_FALSE(fixed_vertices.isFixed(3));
  ASSERT_EQ(kInvalidPartition, fixed_vertices.fixedVertexBlock(3));
  ASSERT_TRUE(fixed_vertices.isFixed(0));
  ASSERT_EQ(0, fixed_vertices.fixedVertexBlock(0));
  ASSERT_EQ(2, fixed_vertices.fixedVertexBlockWeight(0));
  ASSERT_EQ(4, fixed_vertices.totalFixedVertexWeight());
}

TEST_F(AFixedVertexSupport, UncontractFixedOntoFixedVertex) {
  ASSERT_TRUE(fixed_vertices.contract(0, 2));
  fixed_vertices.uncontract(0, 2);
  ASSERT_TRUE(fixed_vertices.isFixed(0));
  ASSERT_EQ(0, fixed_vertices.fixedVertexBlock(0));
  ASSERT_TRUE(fixed_vertices.isFixed(2));
  ASSERT_EQ(0, fixed_vertices.fixedVertexBlock(2));
  ASSERT_EQ(2, fixed_vertices.fixedVertexBlockWeight(0));
  ASSERT_EQ(4, fixed_vertices.totalFixedVertexWeight());
}

TEST_F(AFixedVertexSupport, UncontractFixedOntoFreeVertex) {
  ASSERT_TRUE(fixed_vertices.contract(1, 4));
  fixed_vertices.uncontract(1, 4);
  ASSERT_FALSE(fixed_vertices.isFixed(1));
  ASSERT_EQ(kInvalidPartition, fixed_vertices.fixedVertexBlock(1));
  ASSERT_TRUE(fixed_vertices.isFixed(4));
  ASSERT_EQ(1, fixed_vertices.fixedVertexBlock(4));
  ASSERT_EQ(1, fixed_vertices.fixedVertexBlockWeight(1));
  ASSERT_EQ(4, fixed_vertices.totalFixedVertexWeight());
}

TEST_F(AFixedVertexSupport, ContractSeveralFixedVertices1) {
  ASSERT_TRUE(fixed_vertices.contract(1, 4));
  ASSERT_TRUE(fixed_vertices.contract(2, 3));
  ASSERT_TRUE(fixed_vertices.contract(5, 6));
  verifyFixedVertices({ 0, 1, 0, 0, 1, 2, 2 });
}

TEST_F(AFixedVertexSupport, ContractSeveralFixedVertices2) {
  ASSERT_TRUE(fixed_vertices.contract(3, 5));
  ASSERT_TRUE(fixed_vertices.contract(2, 0));
  ASSERT_TRUE(fixed_vertices.contract(1, 2));
  verifyFixedVertices({ 0, 0, 0, kInvalidPartition, 1, kInvalidPartition, 2 });
}

TEST_F(AFixedVertexSupport, ContractSeveralFixedVertices3) {
  ASSERT_TRUE(fixed_vertices.contract(2, 0));
  ASSERT_TRUE(fixed_vertices.contract(1, 2));
  ASSERT_TRUE(fixed_vertices.contract(3, 1));
  verifyFixedVertices({ 0, 0, 0, 0, 1, kInvalidPartition, 2 });
}

TEST_F(AFixedVertexSupport, ContractSeveralFixedVertices4) {
  ASSERT_TRUE(fixed_vertices.contract(2, 3));
  ASSERT_TRUE(fixed_vertices.contract(2, 1));
  ASSERT_TRUE(fixed_vertices.contract(0, 2));
  verifyFixedVertices({ 0, 0, 0, 0, 1, kInvalidPartition, 2 });
}

TEST_F(AFixedVertexSupport, ContractSeveralFixedVertices5) {
  ASSERT_TRUE(fixed_vertices.contract(2, 0));
  ASSERT_TRUE(fixed_vertices.contract(2, 1));
  ASSERT_TRUE(fixed_vertices.contract(3, 2));
  verifyFixedVertices({ 0, 0, 0, 0, 1, kInvalidPartition, 2 });
}

TEST_F(AFixedVertexSupport, ContractSeveralFixedVertices6) {
  ASSERT_TRUE(fixed_vertices.contract(1, 2));
  ASSERT_TRUE(fixed_vertices.contract(1, 0));
  ASSERT_TRUE(fixed_vertices.contract(3, 1));
  verifyFixedVertices({ 0, 0, 0, 0, 1, kInvalidPartition, 2 });
}

TEST_F(AFixedVertexSupport, ContractSeveralFixedVertices7) {
  ASSERT_TRUE(fixed_vertices.contract(1, 2));
  ASSERT_TRUE(fixed_vertices.contract(3, 1));
  ASSERT_TRUE(fixed_vertices.contract(0, 3));
  verifyFixedVertices({ 0, 0, 0, 0, 1, kInvalidPartition, 2 });
}

TEST_F(AFixedVertexSupport, ContractSeveralFixedVertices8) {
  ASSERT_TRUE(fixed_vertices.contract(1, 2));
  ASSERT_TRUE(fixed_vertices.contract(0, 1));
  ASSERT_TRUE(fixed_vertices.contract(3, 0));
  verifyFixedVertices({ 0, 0, 0, 0, 1, kInvalidPartition, 2 });
}

TEST_F(AFixedVertexSupport, UncontractSeveralFixedVertices1) {
  ASSERT_TRUE(fixed_vertices.contract(1, 4));
  ASSERT_TRUE(fixed_vertices.contract(2, 3));
  ASSERT_TRUE(fixed_vertices.contract(5, 6));
  verifyFixedVertices("After contractions", { 0, 1, 0, 0, 1, 2, 2 });

  fixed_vertices.uncontract(5, 6);
  verifyFixedVertices("First uncontraction",
    { 0, 1, 0, 0, 1, kInvalidPartition, 2 });

  fixed_vertices.uncontract(2, 3);
  verifyFixedVertices("Second uncontraction",
    { 0, 1, 0, kInvalidPartition, 1, kInvalidPartition, 2 });

  fixed_vertices.uncontract(1, 4);
  verifyFixedVertices("Third uncontraction",
    { 0, kInvalidPartition, 0, kInvalidPartition, 1, kInvalidPartition, 2 });
}

TEST_F(AFixedVertexSupport, UncontractSeveralFixedVertices2) {
  ASSERT_TRUE(fixed_vertices.contract(3, 5));
  ASSERT_TRUE(fixed_vertices.contract(2, 0));
  ASSERT_TRUE(fixed_vertices.contract(1, 2));
  verifyFixedVertices("After contractions",
    { 0, 0, 0, kInvalidPartition, 1, kInvalidPartition, 2 });

  fixed_vertices.uncontract(1, 2);
  verifyFixedVertices("First uncontraction",
    { 0, kInvalidPartition, 0, kInvalidPartition, 1, kInvalidPartition, 2 });

  fixed_vertices.uncontract(2, 0);
  verifyFixedVertices("Second uncontraction",
    { 0, kInvalidPartition, 0, kInvalidPartition, 1, kInvalidPartition, 2 });

  fixed_vertices.uncontract(3, 5);
  verifyFixedVertices("Third uncontraction",
    { 0, kInvalidPartition, 0, kInvalidPartition, 1, kInvalidPartition, 2 });
}

TEST_F(AFixedVertexSupport, UncontractSeveralFixedVertices3) {
  ASSERT_TRUE(fixed_vertices.contract(2, 0));
  ASSERT_TRUE(fixed_vertices.contract(1, 2));
  ASSERT_TRUE(fixed_vertices.contract(3, 1));
  verifyFixedVertices("After contractions", { 0, 0, 0, 0, 1, kInvalidPartition, 2 });

  fixed_vertices.uncontract(3, 1);
  verifyFixedVertices("First uncontraction",
    { 0, 0, 0, kInvalidPartition, 1, kInvalidPartition, 2 });

  fixed_vertices.uncontract(1, 2);
  verifyFixedVertices("Second uncontraction",
    { 0, kInvalidPartition, 0, kInvalidPartition, 1, kInvalidPartition, 2 });

  fixed_vertices.uncontract(2, 0);
  verifyFixedVertices("Third uncontraction",
    { 0, kInvalidPartition, 0, kInvalidPartition, 1, kInvalidPartition, 2 });
}

TEST_F(AFixedVertexSupport, UncontractSeveralFixedVertices4) {
  ASSERT_TRUE(fixed_vertices.contract(2, 3));
  ASSERT_TRUE(fixed_vertices.contract(2, 1));
  ASSERT_TRUE(fixed_vertices.contract(0, 2));
  verifyFixedVertices("After contractions", { 0, 0, 0, 0, 1, kInvalidPartition, 2 });

  fixed_vertices.uncontract(0, 2);
  verifyFixedVertices("First uncontraction",
    { 0, 0, 0, 0, 1, kInvalidPartition, 2 });

  fixed_vertices.uncontract(2, 1);
  verifyFixedVertices("Second uncontraction",
    { 0, kInvalidPartition, 0, 0, 1, kInvalidPartition, 2 });

  fixed_vertices.uncontract(2, 3);
  verifyFixedVertices("Third uncontraction",
    { 0, kInvalidPartition, 0, kInvalidPartition, 1, kInvalidPartition, 2 });
}

TEST_F(AFixedVertexSupport, UncontractSeveralFixedVertices5) {
  ASSERT_TRUE(fixed_vertices.contract(2, 0));
  ASSERT_TRUE(fixed_vertices.contract(2, 1));
  ASSERT_TRUE(fixed_vertices.contract(3, 2));
  verifyFixedVertices("After contractions", { 0, 0, 0, 0, 1, kInvalidPartition, 2 });

  fixed_vertices.uncontract(3, 2);
  verifyFixedVertices("First uncontraction",
    { 0, 0, 0, kInvalidPartition, 1, kInvalidPartition, 2 });

  fixed_vertices.uncontract(2, 1);
  verifyFixedVertices("Second uncontraction",
    { 0, kInvalidPartition, 0, kInvalidPartition, 1, kInvalidPartition, 2 });

  fixed_vertices.uncontract(2, 0);
  verifyFixedVertices("Third uncontraction",
    { 0, kInvalidPartition, 0, kInvalidPartition, 1, kInvalidPartition, 2 });
}

TEST_F(AFixedVertexSupport, UnontractSeveralFixedVertices6) {
  ASSERT_TRUE(fixed_vertices.contract(1, 2));
  ASSERT_TRUE(fixed_vertices.contract(1, 0));
  ASSERT_TRUE(fixed_vertices.contract(3, 1));
  verifyFixedVertices("After contractions", { 0, 0, 0, 0, 1, kInvalidPartition, 2 });

  fixed_vertices.uncontract(3, 1);
  verifyFixedVertices("First uncontraction",
    { 0, 0, 0, kInvalidPartition, 1, kInvalidPartition, 2 });

  fixed_vertices.uncontract(1, 0);
  verifyFixedVertices("Second uncontraction",
    { 0, 0, 0, kInvalidPartition, 1, kInvalidPartition, 2 });

  fixed_vertices.uncontract(1, 2);
  verifyFixedVertices("Third uncontraction",
    { 0, kInvalidPartition, 0, kInvalidPartition, 1, kInvalidPartition, 2 });
}

TEST_F(AFixedVertexSupport, UncontractSeveralFixedVertices7) {
  ASSERT_TRUE(fixed_vertices.contract(1, 2));
  ASSERT_TRUE(fixed_vertices.contract(3, 1));
  ASSERT_TRUE(fixed_vertices.contract(0, 3));
  verifyFixedVertices("After contractions", { 0, 0, 0, 0, 1, kInvalidPartition, 2 });

  fixed_vertices.uncontract(0, 3);
  verifyFixedVertices("First uncontraction",
    { 0, 0, 0, 0, 1, kInvalidPartition, 2 });

  fixed_vertices.uncontract(3, 1);
  verifyFixedVertices("Second uncontraction",
    { 0, 0, 0, kInvalidPartition, 1, kInvalidPartition, 2 });

  fixed_vertices.uncontract(1, 2);
  verifyFixedVertices("Third uncontraction",
    { 0, kInvalidPartition, 0, kInvalidPartition, 1, kInvalidPartition, 2 });
}

TEST_F(AFixedVertexSupport, UncontractSeveralFixedVertices8) {
  ASSERT_TRUE(fixed_vertices.contract(1, 2));
  ASSERT_TRUE(fixed_vertices.contract(0, 1));
  ASSERT_TRUE(fixed_vertices.contract(3, 0));
  verifyFixedVertices("After contractions", { 0, 0, 0, 0, 1, kInvalidPartition, 2 });

  fixed_vertices.uncontract(3, 0);
  verifyFixedVertices("First uncontraction",
    { 0, 0, 0, kInvalidPartition, 1, kInvalidPartition, 2 });

  fixed_vertices.uncontract(0, 1);
  verifyFixedVertices("Second uncontraction",
    { 0, 0, 0, kInvalidPartition, 1, kInvalidPartition, 2 });

  fixed_vertices.uncontract(1, 2);
  verifyFixedVertices("Third uncontraction",
    { 0, kInvalidPartition, 0, kInvalidPartition, 1, kInvalidPartition, 2 });
}

TEST_F(AFixedVertexSupport, PerformsParallelContractionsAndUncontractions1) {
  runParallel([&] { fixed_vertices.contract(0, 2); },
              [&] { fixed_vertices.contract(0, 1); });
  verifyFixedVertices("After contractions",
    { 0, 0, 0, kInvalidPartition, 1, kInvalidPartition, 2 });

  runParallel([&] { fixed_vertices.uncontract(0, 2); },
              [&] { fixed_vertices.uncontract(0, 1); });
  verifyFixedVertices("After uncontractions",
    { 0, kInvalidPartition, 0, kInvalidPartition, 1, kInvalidPartition, 2 });
}

TEST_F(AFixedVertexSupport, PerformsParallelContractionsAndUncontractions2) {
  runParallel([&] { fixed_vertices.contract(1, 2); },
              [&] { fixed_vertices.contract(1, 0); });
  verifyFixedVertices("After contractions",
    { 0, 0, 0, kInvalidPartition, 1, kInvalidPartition, 2 });

  runParallel([&] { fixed_vertices.uncontract(1, 2); },
              [&] { fixed_vertices.uncontract(1, 0); });
  verifyFixedVertices("After uncontractions",
    { 0, kInvalidPartition, 0, kInvalidPartition, 1, kInvalidPartition, 2 });
}

TEST_F(AFixedVertexSupport, PerformContractionWithMaximumAllowedBlockWeight1) {
  fixed_vertices.setMaxBlockWeight({ 2, 1, 1 });
  ASSERT_FALSE(fixed_vertices.contract(0, 1));
  verifyFixedVertices({ 0, kInvalidPartition, 0, kInvalidPartition, 1, kInvalidPartition, 2 });
}

TEST_F(AFixedVertexSupport, PerformContractionWithMaximumAllowedBlockWeight2) {
  fixed_vertices.setMaxBlockWeight({ 2, 1, 1 });
  ASSERT_TRUE(fixed_vertices.contract(0, 2));
  verifyFixedVertices({ 0, kInvalidPartition, 0, kInvalidPartition, 1, kInvalidPartition, 2 });
}

TEST_F(AFixedVertexSupport, PerformContractionWithMaximumAllowedBlockWeight3) {
  fixed_vertices.setMaxBlockWeight({ 3, 1, 1 });
  ASSERT_TRUE(fixed_vertices.contract(0, 1));
  ASSERT_FALSE(fixed_vertices.contract(0, 3));
  verifyFixedVertices({ 0, 0, 0, kInvalidPartition, 1, kInvalidPartition, 2 });
}

TEST_F(AFixedVertexSupport, PerformParallelContractionWithMaximumAllowedBlockWeight1) {
  fixed_vertices.setMaxBlockWeight({ 3, 1, 1 });
  runParallel([&] { ASSERT_TRUE(fixed_vertices.contract(1, 2)); },
              [&] { ASSERT_TRUE(fixed_vertices.contract(1, 0)); });
  verifyFixedVertices({ 0, 0, 0, kInvalidPartition, 1, kInvalidPartition, 2 });
}

TEST_F(AFixedVertexSupport, PerformParallelContractionWithMaximumAllowedBlockWeight2) {
  fixed_vertices.setMaxBlockWeight({ 3, 1, 1 });
  std::atomic<size_t> successful_contractions(0);
  runParallel([&] { successful_contractions += fixed_vertices.contract(0, 3); },
              [&] { successful_contractions += fixed_vertices.contract(0, 1); });
  ASSERT_EQ(1, successful_contractions.load(std::memory_order_relaxed));
  if ( fixed_vertices.isFixed(1) ) {
    verifyFixedVertices({ 0, 0, 0, kInvalidPartition, 1, kInvalidPartition, 2 });
  } else {
    verifyFixedVertices({ 0, kInvalidPartition, 0, 0, 1, kInvalidPartition, 2 });
  }
}

TEST_F(AFixedVertexSupport, PerformParallelContractionWithMaximumAllowedBlockWeight3) {
  fixed_vertices.setMaxBlockWeight({ 3, 1, 1 });
  std::atomic<size_t> successful_contractions(0);
  runParallel([&] { successful_contractions += fixed_vertices.contract(0, 3); },
              [&] { successful_contractions += fixed_vertices.contract(2, 1); });
  ASSERT_EQ(1, successful_contractions.load(std::memory_order_relaxed));
  if ( fixed_vertices.isFixed(1) ) {
    verifyFixedVertices({ 0, 0, 0, kInvalidPartition, 1, kInvalidPartition, 2 });
  } else {
    verifyFixedVertices({ 0, kInvalidPartition, 0, 0, 1, kInvalidPartition, 2 });
  }
}

}  // namespace ds
}  // namespace mt_kahypar
