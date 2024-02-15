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

#include <numeric>
#include <algorithm>

#include "gmock/gmock.h"

#include "mt-kahypar/parallel/stl/scalable_vector.h"
#include "mt-kahypar/datastructures/static_bitset.h"

using ::testing::Test;

namespace mt_kahypar {
namespace ds {

using Block = StaticBitset::Block;

void set_one_bits(Bitset& bitset,
                  const vec<PartitionID>& one_bits) {
  for ( const size_t pos : one_bits ) {
    bitset.set(pos);
  }
}

void verify_iterator(const StaticBitset& bitset,
                     const vec<PartitionID>& expected) {
  ASSERT_EQ(static_cast<size_t>(bitset.popcount()), expected.size());
  size_t cnt = 0;
  for ( const PartitionID& block : bitset ) {
    ASSERT_EQ(block, expected[cnt++]);
  }
  ASSERT_EQ(cnt, expected.size());
}

TEST(AStaticBitset, CountNumberOfOneBits1) {
  Bitset bits(64);
  set_one_bits(bits, { 0, 1, 5, 7 });
  StaticBitset bitset(bits.numBlocks(), bits.data());
  ASSERT_EQ(4, bitset.popcount());
}

TEST(AStaticBitset, CountNumberOfOneBits2) {
  Bitset bits(64);
  set_one_bits(bits, { 0, 1, 5, 7, 16, 24, 30, 31, 46, 63 });
  StaticBitset bitset(bits.numBlocks(), bits.data());
  ASSERT_EQ(10, bitset.popcount());
}

TEST(AStaticBitset, CountNumberOfOneBits3) {
  Bitset bits(128);
  set_one_bits(bits, {0, 1, 5, 7, 64, 65, 69, 71, 80, 88, 94, 95, 110, 127});
  StaticBitset bitset(bits.numBlocks(), bits.data());
  ASSERT_EQ(14, bitset.popcount());
}

TEST(AStaticBitset, CountNumberOfOneBits4) {
  Bitset bits(192);
  set_one_bits(bits, {0, 1, 5, 7, 64, 65, 69, 71, 80, 88,
                      94, 95, 110, 127, 151, 152, 153, 154, 155,  183});
  StaticBitset bitset(bits.numBlocks(), bits.data());
  ASSERT_EQ(20, bitset.popcount());
}

TEST(AStaticBitset, VerifyIterator1) {
  vec<PartitionID> expected = { 0, 1, 5, 7 };
  Bitset bits(64);
  set_one_bits(bits, expected);
  StaticBitset bitset(bits.numBlocks(), bits.data());
  verify_iterator(bitset, expected);
}

TEST(AStaticBitset, VerifyIterator2) {
  vec<PartitionID> expected = { 0, 1, 5, 7, 16, 24, 30, 31, 46, 63 };
  Bitset bits(64);
  set_one_bits(bits, expected);
  StaticBitset bitset(bits.numBlocks(), bits.data());
  verify_iterator(bitset, expected);
}

TEST(AStaticBitset, VerifyIterator3) {
  vec<PartitionID> expected = { 0, 1, 5, 7, 64, 65, 69, 71, 80, 88, 94, 95, 110, 127 };
  Bitset bits(128);
  set_one_bits(bits, expected);
  StaticBitset bitset(bits.numBlocks(), bits.data());
  verify_iterator(bitset, expected);
}

TEST(AStaticBitset, VerifyIterator4) {
  vec<PartitionID> expected = { 0, 1, 5, 7, 64, 65, 69, 71, 80, 88,
                                94, 95, 110, 127, 151, 152, 153, 154, 155,  183 };
  Bitset bits(192);
  set_one_bits(bits, expected);
  StaticBitset bitset(bits.numBlocks(), bits.data());
  verify_iterator(bitset, expected);
}

TEST(AStaticBitset, PerformsXOROperation1) {
  Bitset bits_1(64);
  set_one_bits(bits_1, { 0, 3, 6, 25 });
  Bitset bits_2(64);
  set_one_bits(bits_2, { 3, 6 });
  StaticBitset bitset_1(bits_1.numBlocks(), bits_1.data());
  StaticBitset bitset_2(bits_2.numBlocks(), bits_2.data());
  Bitset res = bitset_1 ^ bitset_2;
  StaticBitset res_bitset(res.numBlocks(), res.data());
  verify_iterator(res_bitset, { 0, 25 });
}

TEST(AStaticBitset, PerformsXOROperation2) {
  Bitset bits_1(128);
  set_one_bits(bits_1, { 0, 3, 6, 25, 65, 85, 121 });
  Bitset bits_2(128);
  set_one_bits(bits_2, { 3, 6, 65, 121 });
  StaticBitset bitset_1(bits_1.numBlocks(), bits_1.data());
  StaticBitset bitset_2(bits_2.numBlocks(), bits_2.data());
  Bitset res = bitset_1 ^ bitset_2;
  StaticBitset res_bitset(res.numBlocks(), res.data());
  verify_iterator(res_bitset, { 0, 25, 85 });
}

}  // namespace ds
}  // namespace mt_kahypar
