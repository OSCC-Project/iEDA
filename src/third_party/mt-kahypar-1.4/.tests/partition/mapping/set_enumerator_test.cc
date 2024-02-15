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

#include "mt-kahypar/partition/mapping/set_enumerator.h"

using ::testing::Test;

namespace mt_kahypar {

template<typename Iterator>
void verifyIterator(Iterator& it,
                    const vec<vec<PartitionID>>& expected) {
  size_t cnt = 0;
  for ( const auto& set : it ) {
    size_t i = 0;
    for ( const PartitionID block : set ) {
      ASSERT_EQ(block, expected[cnt][i]) << V(cnt) << V(i);
      ++i;
    }
    ASSERT_EQ(i, expected[cnt].size());
    ++cnt;
  }
  ASSERT_EQ(cnt, expected.size());
}

TEST(ASetEnumerator, IteratesOverAllSetsOfSizeTwo) {
  SetEnumerator sets(5, 2);
  verifyIterator(sets, { { 0, 1 }, { 0, 2 }, { 0, 3 }, { 0, 4 },
                         { 1, 2 }, { 1, 3 }, { 1, 4 }, { 2, 3 },
                         { 2, 4 }, { 3, 4 } });
}

TEST(ASetEnumerator, IteratesOverAllSetsOfSizeThree) {
  SetEnumerator sets(5, 3);
  verifyIterator(sets, { { 0, 1, 2 }, { 0, 1, 3 }, { 0, 1, 4 },
                         { 0, 2, 3 }, { 0, 2, 4 }, { 0, 3, 4 },
                         { 1, 2, 3 }, { 1, 2, 4 }, { 1, 3, 4 },
                         { 2, 3, 4 } });
}

TEST(ASetEnumerator, IteratesOverAllSetsOfSizeFour) {
  SetEnumerator sets(5, 4);
  verifyIterator(sets, { { 0, 1, 2, 3 }, { 0, 1, 2, 4 },
                         { 0, 1, 3, 4 }, { 0, 2, 3, 4 },
                         { 1, 2, 3, 4 } });
}

TEST(ASubsetEnumerator, IteratesOverAllSubsets1) {
  ds::Bitset bits(8);
  bits.set(0);
  bits.set(1);
  ds::StaticBitset bitset(bits.numBlocks(), bits.data());
  SubsetEnumerator subsets(8, bitset);
  verifyIterator(subsets, { { 0 }, { 1 } });
}

TEST(ASubsetEnumerator, IteratesOverAllSubsets2) {
  ds::Bitset bits(8);
  bits.set(1);
  bits.set(3);
  bits.set(5);
  ds::StaticBitset bitset(bits.numBlocks(), bits.data());
  SubsetEnumerator subsets(8, bitset);
  verifyIterator(subsets, { { 1 }, { 3 }, { 1, 3 }, { 5 },
                            { 1, 5 }, { 3, 5 } });
}

TEST(ASubsetEnumerator, IteratesOverAllSubsets3) {
  ds::Bitset bits(8);
  bits.set(1);
  bits.set(3);
  bits.set(5);
  bits.set(7);
  ds::StaticBitset bitset(bits.numBlocks(), bits.data());
  SubsetEnumerator subsets(8, bitset);
  verifyIterator(subsets, { { 1 }, { 3 }, { 1, 3 }, { 5 },
                            { 1, 5 }, { 3, 5 }, { 1, 3, 5 },
                            { 7 }, { 1, 7 }, { 3, 7 }, { 1, 3, 7 },
                            { 5, 7 }, { 1, 5, 7 }, { 3, 5, 7 } });
}

}  // namespace mt_kahypar
