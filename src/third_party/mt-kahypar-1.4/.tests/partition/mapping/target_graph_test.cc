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

#include "tbb/task_group.h"

#include "mt-kahypar/datastructures/static_graph_factory.h"
#include "mt-kahypar/partition/mapping/target_graph.h"

using ::testing::Test;

namespace mt_kahypar {

class ATargetGraph : public Test {

  using UnsafeBlock = ds::StaticBitset::Block;

 public:
  ATargetGraph() :
    graph(nullptr) {

    /**
     * Target Graph:
     *        1           2           4
     * 0  -------- 1  -------- 2  -------- 3
     * |           |           |           |
     * | 3         | 2         | 1         | 1
     * |      3    |      2    |      1    |
     * 4  -------- 5  -------- 6  -------- 7
     * |           |           |           |
     * | 1         | 1         | 3         | 2
     * |      2    |      4    |      2    |
     * 8  -------- 9  -------- 10 -------- 11
     * |           |           |           |
     * | 1         | 2         | 2         | 2
     * |      1    |      1    |      2    |
     * 12 -------- 13 -------- 14 -------- 15
    */
    vec<HyperedgeWeight> edge_weights =
      { 1, 2, 4,
        3, 2, 1, 1,
        3, 2, 1,
        1, 1, 3, 2,
        2, 4, 2,
        1, 2, 2, 2,
        1, 1, 2 };
    graph = std::make_unique<TargetGraph>(
      ds::StaticGraphFactory::construct(16, 24,
        { { 0, 1 }, { 1, 2 }, { 2, 3 },
          { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 },
          { 4, 5 }, { 5, 6 }, { 6, 7 },
          { 4, 8 }, { 5, 9 }, { 6, 10 }, { 7, 11 },
          { 8, 9 }, { 9, 10 }, { 10, 11 },
          { 8, 12 }, { 9, 13 }, { 10, 14 }, { 11, 15 },
          { 12, 13 }, { 13, 14 }, { 14, 15 } },
          edge_weights.data()));

  }


  HyperedgeWeight distance(const vec<PartitionID>& connectivity_set) {
    ds::Bitset bitset = getBitset(connectivity_set);;
    ds::StaticBitset con_set(bitset.numBlocks(), bitset.data());
    return graph->distance(con_set);
  }

  HyperedgeWeight distanceWithBlock(const vec<PartitionID>& connectivity_set,
                                    const PartitionID block) {
    ds::Bitset con_set = getBitset(connectivity_set);
    return graph->distanceWithBlock(con_set, block);
  }

  HyperedgeWeight distanceWithoutBlock(const vec<PartitionID>& connectivity_set,
                                       const PartitionID block) {
    ds::Bitset con_set = getBitset(connectivity_set);
    return graph->distanceWithoutBlock(con_set, block);
  }

  HyperedgeWeight distanceAfterExchangingBlocks(const vec<PartitionID>& connectivity_set,
                                                const PartitionID removed_block,
                                                const PartitionID added_block) {
    ds::Bitset con_set = getBitset(connectivity_set);
    return graph->distanceAfterExchangingBlocks(con_set, removed_block, added_block);
  }

  std::unique_ptr<TargetGraph> graph;

 private:
  ds::Bitset getBitset(const vec<PartitionID>& connectivity_set) {
    ds::Bitset bitset(graph->numBlocks());
    for ( const PartitionID block : connectivity_set ) {
      bitset.set(block);
    }
    return bitset;
  }

  void setBit(UnsafeBlock& bits, size_t pos) {
    bits |= (UL(1) << pos);
  }
};

template <class F, class K>
void executeConcurrent(F f1, K f2) {
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

TEST_F(ATargetGraph, HasCorrectNumberOfBlocks) {
  ASSERT_EQ(16, graph->numBlocks());
}

TEST_F(ATargetGraph, ComputesAllShortestPaths) {
  graph->precomputeDistances(2);
  ASSERT_EQ(0, graph->distance(0, 0));
  ASSERT_EQ(0, graph->distance(1, 1));
  ASSERT_EQ(1, graph->distance(0, 1));
  ASSERT_EQ(3, graph->distance(0, 2));
  ASSERT_EQ(6, graph->distance(0, 3));
  ASSERT_EQ(6, graph->distance(3, 0));
  ASSERT_EQ(6, graph->distance(1, 10));
  ASSERT_EQ(7, graph->distance(8, 11));
  ASSERT_EQ(7, graph->distance(12, 7));
  ASSERT_EQ(3, graph->distance(9, 14));
  ASSERT_EQ(5, graph->distance(6, 15));
  ASSERT_EQ(2, graph->distance(3, 6));
  ASSERT_EQ(7, graph->distance(4, 3));
}

TEST_F(ATargetGraph, ComputesAllShortestPathsWithConnectivitySet) {
  graph->precomputeDistances(2);
  ASSERT_EQ(0, distance({ 0 }));
  ASSERT_EQ(0, distance({ 1 }));
  ASSERT_EQ(1, distance({ 0, 1 }));
  ASSERT_EQ(3, distance({ 0, 2 }));
  ASSERT_EQ(6, distance({ 0, 3 }));
  ASSERT_EQ(6, distance({ 3, 0 }));
  ASSERT_EQ(6, distance({ 1, 10 }));
  ASSERT_EQ(7, distance({ 8, 11 }));
  ASSERT_EQ(7, distance({ 12, 7 }));
  ASSERT_EQ(3, distance({ 9, 14 }));
  ASSERT_EQ(5, distance({ 6, 15 }));
  ASSERT_EQ(2, distance({ 3, 6 }));
  ASSERT_EQ(7, distance({ 4, 3 }));
}

TEST_F(ATargetGraph, ComputesAllSteinerTreesUpToSizeThree) {
  graph->precomputeDistances(3);
  ASSERT_EQ(8, distance({ 0, 3, 9 }));
  ASSERT_EQ(8, distance({ 1, 3, 10 }));
  ASSERT_EQ(8, distance({ 2, 8, 13 }));
  ASSERT_EQ(7, distance({ 8, 11, 13 }));
  ASSERT_EQ(10, distance({ 0, 3, 15 }));
  ASSERT_EQ(3, distance({ 0, 1, 2 }));
  ASSERT_EQ(7, distance({ 6, 11, 14 }));
  ASSERT_EQ(4, distance({ 12, 14, 15 }));
  ASSERT_EQ(3, distance({ 8, 13, 14 }));
  ASSERT_EQ(5, distance({ 9, 10, 14 }));
}

TEST_F(ATargetGraph, ComputeDistancesWithAnAdditionalBlock) {
  graph->precomputeDistances(3);
  ASSERT_EQ(8, distanceWithBlock({ 0, 3 }, 9));
  ASSERT_EQ(8, distanceWithBlock({ 1, 3 }, 10));
  ASSERT_EQ(8, distanceWithBlock({ 2, 8 }, 13));
  ASSERT_EQ(7, distanceWithBlock({ 8, 11 }, 13));
  ASSERT_EQ(10, distanceWithBlock({ 0, 3 }, 15));
  ASSERT_EQ(3, distanceWithBlock({ 0, 1 }, 2));
  ASSERT_EQ(7, distanceWithBlock({ 6, 11 }, 14));
  ASSERT_EQ(4, distanceWithBlock({ 14, 15 }, 12));
  ASSERT_EQ(3, distanceWithBlock({ 8, 14 }, 13));
  ASSERT_EQ(5, distanceWithBlock({ 10, 14 }, 9));
}

TEST_F(ATargetGraph, ComputeDistancesWithoutAnBlock) {
  graph->precomputeDistances(3);
  ASSERT_EQ(8, distanceWithoutBlock({ 0, 2, 3, 9 }, 2));
  ASSERT_EQ(8, distanceWithoutBlock({ 1, 3, 7, 10 }, 7));
  ASSERT_EQ(8, distanceWithoutBlock({ 2, 8, 12, 13 }, 12));
  ASSERT_EQ(7, distanceWithoutBlock({ 8, 11, 13, 15 }, 15));
  ASSERT_EQ(10, distanceWithoutBlock({ 0, 3, 4, 15 }, 4));
  ASSERT_EQ(3, distanceWithoutBlock({ 0, 1, 2, 3 }, 3));
  ASSERT_EQ(7, distanceWithoutBlock({ 0, 6, 11, 14 }, 0));
  ASSERT_EQ(4, distanceWithoutBlock({ 10, 12, 14, 15 }, 10));
  ASSERT_EQ(3, distanceWithoutBlock({ 4, 8, 13, 14 }, 4));
  ASSERT_EQ(5, distanceWithoutBlock({ 9, 10, 13, 14 }, 13));
}

TEST_F(ATargetGraph, ComputeDistancesAfterExchangingBlocks) {
  graph->precomputeDistances(3);
  ASSERT_EQ(8, distanceAfterExchangingBlocks({ 0, 4, 9 }, 4, 3));
  ASSERT_EQ(8, distanceAfterExchangingBlocks({ 1, 3, 12 }, 12, 10));
  ASSERT_EQ(8, distanceAfterExchangingBlocks({ 1, 8, 13 }, 1, 2));
  ASSERT_EQ(7, distanceAfterExchangingBlocks({ 8, 9, 13 }, 9, 11));
  ASSERT_EQ(10, distanceAfterExchangingBlocks({ 2, 3, 15 }, 2, 0));
  ASSERT_EQ(3, distanceAfterExchangingBlocks({ 0, 1, 12 }, 12, 2));
  ASSERT_EQ(7, distanceAfterExchangingBlocks({ 4, 11, 14 }, 4, 6));
  ASSERT_EQ(4, distanceAfterExchangingBlocks({ 5, 14, 15 }, 5, 12));
  ASSERT_EQ(3, distanceAfterExchangingBlocks({ 1, 13, 14 }, 1, 8));
  ASSERT_EQ(5, distanceAfterExchangingBlocks({ 9, 10, 12 }, 12, 14));
}

TEST_F(ATargetGraph, ComputesAllSteinerTreesUpToSizeFour) {
  graph->precomputeDistances(4);
  ASSERT_EQ(10, distance({ 0, 3, 9, 11 }));
  ASSERT_EQ(8, distance({ 5, 8, 10, 13 }));
  ASSERT_EQ(11, distance({ 1, 3, 10, 15 }));
  ASSERT_EQ(9, distance({ 3, 9, 11, 15 }));
  ASSERT_EQ(10, distance({ 2, 4, 10 ,12 }));
  ASSERT_EQ(6, distance({ 0, 1, 2, 3 }));
  ASSERT_EQ(14, distance({ 0, 3, 12, 15 }));
  ASSERT_EQ(11, distance({ 0, 3, 9, 14 }));
}

TEST_F(ATargetGraph, ComputeDistanceBetweenNonPrecomputedSets) {
  graph->precomputeDistances(2);
  ASSERT_EQ(8, distance({ 0, 5, 9, 10 }));
  ASSERT_EQ(13, distance({ 0, 3, 10, 14 }));
  ASSERT_EQ(13, distance({ 0, 4, 6, 13, 15 }));
  ASSERT_EQ(10, distance({ 1, 5, 8, 10, 12 }));
  ASSERT_EQ(15, distance({ 2, 3, 4, 8, 10, 14, 15 }));
}

TEST_F(ATargetGraph, ComputeDistanceBetweenNonPrecomputedSetsConcurrently) {
  graph->precomputeDistances(2);
  executeConcurrent([&] {
    ASSERT_EQ(8, distance({ 0, 5, 9, 10 }));
    ASSERT_EQ(13, distance({ 0, 3, 10, 14 }));
  }, [&] {
    ASSERT_EQ(13, distance({ 0, 4, 6, 13, 15 }));
    ASSERT_EQ(10, distance({ 1, 5, 8, 10, 12 }));
    ASSERT_EQ(15, distance({ 2, 3, 4, 8, 10, 14, 15 }));
  });
}

TEST_F(ATargetGraph, UsesACachedDistanceForNonPrecomputedSets) {
  graph->precomputeDistances(2);
  ASSERT_EQ(8, distance({ 0, 5, 9, 10 }));
  ASSERT_EQ(8, distance({ 0, 5, 9, 10 }));
}

TEST_F(ATargetGraph, InsertsIntoCacheConcurrentlyForNonPrecomputedSets) {
  graph->precomputeDistances(2);
  executeConcurrent([&] {
    ASSERT_EQ(8, distance({ 0, 5, 9, 10 }));
    ASSERT_EQ(8, distance({ 0, 5, 9, 10 }));
  }, [&] {
    ASSERT_EQ(13, distance({ 0, 4, 6, 13, 15 }));
    ASSERT_EQ(13, distance({ 0, 4, 6, 13, 15 }));
  });
  ASSERT_EQ(8, distance({ 0, 5, 9, 10 }));
  ASSERT_EQ(13, distance({ 0, 4, 6, 13, 15 }));
}


}  // namespace mt_kahypar
