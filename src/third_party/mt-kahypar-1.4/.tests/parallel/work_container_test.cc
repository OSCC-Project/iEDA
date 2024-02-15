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

#include "gmock/gmock.h"

#include <mt-kahypar/parallel/work_stack.h>
#include <thread>

using ::testing::Test;

namespace mt_kahypar {
namespace parallel {

size_t n = 100000;


TEST(WorkContainer, HasCorrectSizeAfterParallelInsertionAndDeletion) {
  int m = 75000;
  WorkContainer<int> cdc(std::thread::hardware_concurrency());
  tbb::parallel_for(0, m, [&](int i) {
    cdc.safe_push(i, tbb::this_task_arena::current_thread_index());
  });
  ASSERT_EQ(cdc.unsafe_size(), m);

  tbb::enumerable_thread_specific<int> counters;
  tbb::task_group tg;
  int num_tasks = 7;
  for (int i = 0; i < num_tasks; ++i) {
    tg.run([&]() {
      int res = 0;
      int& lc = counters.local();
      while (cdc.try_pop(res, tbb::this_task_arena::current_thread_index())) {
        lc++;
      }
    });
  }
  tg.wait();

  int overall = counters.combine(std::plus<int>());
  ASSERT_EQ(overall, m);

  cdc.clear();
  ASSERT_EQ(cdc.unsafe_size(), 0);
}

TEST(WorkContainer, ClearWorks) {
  WorkContainer<int> cdc(std::thread::hardware_concurrency());
  cdc.safe_push(5, tbb::this_task_arena::current_thread_index());
  cdc.safe_push(420, tbb::this_task_arena::current_thread_index());
  ASSERT_EQ(cdc.unsafe_size(), 2);
  cdc.clear();
  ASSERT_TRUE(cdc.unsafe_size() == 0);
}


TEST(WorkContainer, WorkStealingWorks) {
  WorkContainer<int> cdc(std::thread::hardware_concurrency());

  std::atomic<size_t> stage { 0 };
  size_t steals = 0;
  size_t own_pops = 0;

  int m = 99999;

  std::thread producer([&] {
    int thread_id = 0;
    for (int i = 0; i < m; ++i) {
      cdc.safe_push(i, thread_id);
    }

    stage.fetch_add(1, std::memory_order_acquire);

    int own_element;
    while (cdc.try_pop(own_element, thread_id)) {
      own_pops++;
    }
  });

  std::thread consumer([&] {
    int thread_id = 1;
    while (stage.load(std::memory_order_acquire) < 1) { } //spin

    int stolen_element;
    while (cdc.try_pop(stolen_element, thread_id)) {
      steals++;
    }
  });

  consumer.join();
  producer.join();

  ASSERT_EQ(steals + own_pops, m);
}

}  // namespace parallel
}  // namespace mt_kahypar
