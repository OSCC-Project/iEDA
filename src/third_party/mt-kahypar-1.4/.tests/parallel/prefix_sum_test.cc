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

#include "mt-kahypar/parallel/parallel_prefix_sum.h"

#include <random>
#include <algorithm>

using ::testing::Test;

namespace mt_kahypar {

  TEST(PrefixSumTest, AllZeroes) {
    size_t n = 1 << 19;
    vec<size_t> in(n, 0);
    vec<size_t> out(n, 420);

    for (size_t i = 0; i < n; ++i) {
      ASSERT_EQ(in[i], 0);
      ASSERT_EQ(out[i], 420);
    }

    parallel_prefix_sum(in.begin(), in.end(), out.begin(), std::plus<size_t>(), 0);

    for (size_t i = 0; i < n; ++i) {
      if (in[i] != out[i]) {
        LOG << V(i) << V(in[i]) << V(out[i]) << V(n);
        FAIL();
      }
    }
  }

  TEST(PrefixSumTest, MatchesSequential) {
    size_t n = 1 << 19;
    vec<size_t> in(n, 0);

    std::mt19937 rng(420);
    std::generate(in.begin(), in.end(), rng);

    vec<size_t> out_parallel(n, 420);
    parallel_prefix_sum(in.begin(), in.end(), out_parallel.begin(), std::plus<size_t>(), 0);

    vec<size_t> out_custom_seq;
    sequential_prefix_sum(in.begin(), in.end(), std::back_inserter(out_custom_seq), 0, std::plus<size_t>());

    vec<size_t> out_stl;
    std::partial_sum(in.begin(), in.end(), std::back_inserter(out_stl), std::plus<size_t>());

    ASSERT_EQ(out_custom_seq, out_parallel);
    ASSERT_EQ(out_parallel, out_stl);
  }

  TEST(PrefixSumTest, WorksInplace) {
    size_t n = 1 << 19;
    vec<size_t> in(n, 0);
    std::mt19937 rng(421);
    std::generate(in.begin(), in.end(), rng);
    vec<size_t> in_stl = in;

    parallel_prefix_sum(in.begin(), in.end(), in.begin(), std::plus<size_t>(), 0);
    std::partial_sum(in_stl.begin(), in_stl.end(), in_stl.begin(), std::plus<size_t>());

    ASSERT_EQ(in, in_stl);
  }

  TEST(PrefixSumTest, WorksInplaceSmall) {
    size_t n = 1 << 12;
    vec<size_t> in(n, 0);
    std::mt19937 rng(421);
    std::generate(in.begin(), in.end(), rng);
    vec<size_t> in_stl = in;

    parallel_prefix_sum(in.begin(), in.end(), in.begin(), std::plus<size_t>(), 0);
    std::partial_sum(in_stl.begin(), in_stl.end(), in_stl.begin(), std::plus<size_t>());

    ASSERT_EQ(in, in_stl);
  }

}