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
#include <cstdlib>
#include <mt-kahypar/macros.h>

#include "gmock/gmock.h"
#include "tbb/task_group.h"
#include "tbb/parallel_invoke.h"

#include "mt-kahypar/datastructures/sparse_map.h"
#include "mt-kahypar/datastructures/concurrent_flat_map.h"

using ::testing::Test;

namespace mt_kahypar {
namespace ds {

template<typename MapType>
struct ADynamicSparseMap : public Test {
  MapType map;
};

using DynamicSparseMapTestTypes = ::testing::Types<DynamicSparseMap<size_t, size_t>, DynamicFlatMap<size_t, size_t>>;

TYPED_TEST_CASE(ADynamicSparseMap, DynamicSparseMapTestTypes);

TYPED_TEST(ADynamicSparseMap, AddsSeveralElements) {
  auto& map = this->map;
  map.initialize(1);
  map[4] = 5;
  map[8] = 1;
  map[1] = 4;
  ASSERT_EQ(3, map.size());
  ASSERT_EQ(5, map[4]);
  ASSERT_EQ(1, map[8]);
  ASSERT_EQ(4, map[1]);
}

TYPED_TEST(ADynamicSparseMap, ModifiesAnExistingValue) {
  auto& map = this->map;
  map.initialize(1);
  map[4] = 5;
  map[8] = 1;
  map[1] = 4;
  ++map[1];
  ASSERT_EQ(3, map.size());
  ASSERT_EQ(5, map[4]);
  ASSERT_EQ(1, map[8]);
  ASSERT_EQ(5, map[1]);
}

TYPED_TEST(ADynamicSparseMap, IsForcedToGrow) {
  const size_t initial_capacity = 256;
  auto& map = this->map;
  map.initialize(initial_capacity);
  const size_t n = map.capacity();
  for ( size_t i = 0; i < (2 * n) / 5; ++i ) {
    map[i] = i;
  }
  ASSERT_EQ(initial_capacity, map.capacity());
  ASSERT_EQ((2 * n) / 5, map.size());

  // Forces map to dynamically grow
  map[n] = n;

  ASSERT_EQ(2 * initial_capacity, map.capacity());
  ASSERT_EQ((2 * n) / 5 + 1, map.size());
  ASSERT_EQ(n, map[n]++);
  for ( size_t i = 0; i < (2 * n) / 5; ++i ) {
    ASSERT_EQ(i, map[i]++);
  }

  ASSERT_EQ(n + 1, map[n]);
  for ( size_t i = 0; i < (2 * n) / 5; ++i ) {
    ASSERT_EQ(i + 1, map[i]);
  }
}

}  // namespace ds
}  // namespace mt_kahypar
