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
#include "mt-kahypar/datastructures/priority_queue.h"



using ::testing::Test;

namespace mt_kahypar {
namespace ds {


namespace QuadHeap {
  using EMaxHeap = ExclusiveHandleHeap<MaxHeap<int, int>>;

  TEST(APriorityQueue, ReturnsMax) {
    EMaxHeap h(400);
    h.insert(3, 4);
    h.insert(2, 5);
    h.insert(1, 1);
    ASSERT_TRUE(h.top() == 2);
    ASSERT_TRUE(h.topKey() == 5);

    h.deleteTop();
    ASSERT_TRUE(h.top() == 3);
    ASSERT_TRUE(h.topKey() == 4);
  }

  TEST(APriorityQueue, IncreaseKeyWorks) {
    EMaxHeap h(400);
    h.insert(3, 4);
    h.insert(2, 5);
    h.insert(1, 1);

    h.increaseKey(1, 500);
    ASSERT_TRUE(h.top() == 1);
    ASSERT_TRUE(h.topKey() == 500);
    ASSERT_TRUE(h.keyOf(1) == 500);
  }

  TEST(APriorityQueue, DecreaseKeyWorks) {
    EMaxHeap h(400);
    h.insert(3, 42);
    h.insert(2, 54);
    h.insert(1, 100);

    h.decreaseKey(1, 2);
    ASSERT_TRUE(h.top() == 2);
    ASSERT_TRUE(h.topKey() == 54);
    ASSERT_TRUE(h.keyOf(1) == 2);
  }

  TEST(APriorityQueue, AdjustKeyWorks) {
    EMaxHeap h(400);
    h.insert(3, 42);
    h.insert(2, 54);
    h.insert(1, 100);
    h.insert(0, 5000);
    ASSERT_EQ(h.top(), 0);

    h.adjustKey(1, 500000);
    ASSERT_TRUE(h.top() == 1);
    ASSERT_TRUE(h.topKey() == 500000);
    ASSERT_TRUE(h.keyOf(1) == 500000);
  }

  TEST(APriorityQueue, RemoveDoesRemove) {
    EMaxHeap h(400);
    h.insert(3, 42);
    h.insert(2, 54);
    h.insert(1, 100);

    ASSERT_TRUE(h.contains(2));
    h.remove(2);
    ASSERT_FALSE(h.contains(2));
  }

  TEST(APriorityQueue, RemoveLeavesRestIntact) {
    EMaxHeap h(400);
    h.insert(5, 10);
    h.insert(2, 11);
    h.insert(1, 12);
    h.insert(4, 9);
    h.insert(0, 8);
    h.insert(6, 14);
    h.insert(7, 13);

    ASSERT_TRUE(h.contains(2));
    h.remove(2);
    ASSERT_FALSE(h.contains(2));

    std::vector<int> expected_id_order = {6, 7, 1, 5, 4, 0};
    ASSERT_EQ(expected_id_order.size(), h.size());
    size_t i = 0;
    while (!h.empty()) {
      ASSERT_EQ(h.top(), expected_id_order[i++]);
      h.deleteTop();
    }
    ASSERT_TRUE(h.empty());
  }

  TEST(APriorityQueue, HeapSort) {
    size_t n = 50000;
    EMaxHeap h(n);
    std::vector<std::pair<int, int>> kv_pairs;
    std::mt19937 rng(420);
    std::uniform_int_distribution dist(0, 1000000);
    for (size_t i = 0; i < n; ++i) {
      kv_pairs.emplace_back(dist(rng), i);
    }
    std::shuffle(kv_pairs.begin(), kv_pairs.end(), rng);

    for (auto& x : kv_pairs) {
      h.insert(x.second, x.first);
    }

    std::sort(kv_pairs.begin(), kv_pairs.end(), std::greater<std::pair<int, int>>());
    size_t i = 0;
    while (!h.empty()) {
      ASSERT_EQ(h.topKey(), kv_pairs[i].first);
      i++;
      h.deleteTop();
    }
    ASSERT_EQ(i, n);
  }
}

namespace BinaryHeap {
  using EMaxHeap = ExclusiveHandleHeap<Heap<int, int, std::less<int>, 2>>;

  TEST(APriorityQueue, ReturnsMax) {
    EMaxHeap h(400);
    h.insert(3, 4);
    h.insert(2, 5);
    h.insert(1, 1);
    ASSERT_TRUE(h.top() == 2);
    ASSERT_TRUE(h.topKey() == 5);
    h.deleteTop();
    ASSERT_EQ(h.top(), 3);
    ASSERT_EQ(h.topKey(), 4);
  }


  TEST(APriorityQueue, IncreaseKeyWorks) {
    EMaxHeap h(400);
    h.insert(3, 4);
    h.insert(2, 5);
    h.insert(1, 1);
    h.increaseKey(1, 500);
    ASSERT_TRUE(h.top() == 1);
    ASSERT_TRUE(h.topKey() == 500);
    ASSERT_TRUE(h.keyOf(1) == 500);
  }

  TEST(APriorityQueue, DecreaseKeyWorks) {
    EMaxHeap h(400);
    h.insert(3, 42);
    h.insert(2, 54);
    h.insert(1, 100);
    h.decreaseKey(1, 2);
    ASSERT_EQ(h.top(), 2);
    ASSERT_EQ(h.topKey(), 54);
    ASSERT_EQ(h.keyOf(1), 2);
  }

  TEST(APriorityQueue, AdjustKeyWorks) {
    EMaxHeap h(400);
    h.insert(3, 42);
    h.insert(2, 54);
    h.insert(1, 100);
    h.insert(0, 5000);
    ASSERT_EQ(h.top(), 0);

    h.adjustKey(1, 500000);
    ASSERT_TRUE(h.top() == 1);
    ASSERT_TRUE(h.topKey() == 500000);
    ASSERT_TRUE(h.keyOf(1) == 500000);
  }

  TEST(APriorityQueue, RemoveDoesRemove) {
    EMaxHeap h(400);
    h.insert(3, 42);
    h.insert(2, 54);
    h.insert(1, 100);

    ASSERT_TRUE(h.contains(2));
    h.remove(2);
    ASSERT_FALSE(h.contains(2));
  }

  TEST(APriorityQueue, RemoveLeavesRestIntact) {
    EMaxHeap h(400);
    h.insert(5, 10);
    h.insert(2, 11);
    h.insert(1, 12);
    h.insert(4, 9);
    h.insert(0, 8);
    h.insert(6, 14);
    h.insert(7, 13);

    ASSERT_TRUE(h.contains(2));
    h.remove(2);
    ASSERT_FALSE(h.contains(2));

    std::vector<int> expected_id_order = {6, 7, 1, 5, 4, 0};
    ASSERT_EQ(expected_id_order.size(), h.size());
    size_t i = 0;
    while (!h.empty()) {
      ASSERT_EQ(h.top(), expected_id_order[i++]);
      h.deleteTop();
    }
    ASSERT_TRUE(h.empty());
  }

  TEST(APriorityQueue, HeapSort) {
    size_t n = 50000;
    EMaxHeap h(n);
    std::vector<std::pair<int, int>> kv_pairs;
    std::mt19937 rng(420);
    std::uniform_int_distribution dist(0, 1000000);
    for (size_t i = 0; i < n; ++i) {
      kv_pairs.emplace_back(dist(rng), i);
    }
    std::shuffle(kv_pairs.begin(), kv_pairs.end(), rng);

    for (auto& x : kv_pairs) {
      h.insert(x.second, x.first);
    }

    std::sort(kv_pairs.begin(), kv_pairs.end(), std::greater<std::pair<int, int>>());
    size_t i = 0;
    while (!h.empty()) {
      ASSERT_EQ(h.topKey(), kv_pairs[i].first);
      i++;
      h.deleteTop();
    }
    ASSERT_EQ(i, n);
  }

}

}  // namespace ds
}  // namespace mt_kahypar
