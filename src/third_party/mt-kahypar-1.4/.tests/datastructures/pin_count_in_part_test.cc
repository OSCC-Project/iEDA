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

#include "mt-kahypar/datastructures/pin_count_in_part.h"
#ifdef KAHYPAR_ENABLE_LARGE_K_PARTITIONING_FEATURES
#include "mt-kahypar/datastructures/sparse_pin_counts.h"
#endif

using ::testing::Test;

namespace mt_kahypar {
namespace ds {

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

template<typename PinCounts>
class APinCountDataStructure : public Test {

 public:
  APinCountDataStructure() :
    pin_count() { }

  void initialize(const HyperedgeID num_hyperedges,
                  const PartitionID k,
                  const HypernodeID max_value) {
    pin_count.initialize(num_hyperedges, k, max_value);
  }

  PinCounts pin_count;
};

#ifdef KAHYPAR_ENABLE_LARGE_K_PARTITIONING_FEATURES
using PinCountTestTypes =
  ::testing::Types<PinCountInPart, SparsePinCounts>;
#else
using PinCountTestTypes =
  ::testing::Types<PinCountInPart>;
#endif

TYPED_TEST_CASE(APinCountDataStructure, PinCountTestTypes);

TYPED_TEST(APinCountDataStructure, IsZeroInitialized_k32_Max2) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 32;
  const HypernodeID max_value = 2;
  this->initialize(num_hyperedges, k, max_value);

  for ( HyperedgeID he = 0; he < num_hyperedges; ++he ) {
    for ( PartitionID block = 0; block < k; ++block ) {
      ASSERT_EQ(0, this->pin_count.pinCountInPart(he, block));
    }
  }
}

TYPED_TEST(APinCountDataStructure, SetsPinCountPart1_k32_Max2) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 32;
  const HypernodeID max_value = 2;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.setPinCountInPart(4, 2, 2);
  ASSERT_EQ(2, this->pin_count.pinCountInPart(4, 2));
}

TYPED_TEST(APinCountDataStructure, SetsPinCountPart2_k32_Max2) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 32;
  const HypernodeID max_value = 2;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.setPinCountInPart(4, 31, 1);
  ASSERT_EQ(1, this->pin_count.pinCountInPart(4, 31));
}

TYPED_TEST(APinCountDataStructure, SetsPinCountPart3_k32_Max2) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 32;
  const HypernodeID max_value = 2;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.setPinCountInPart(32, 4, 2);
  this->pin_count.setPinCountInPart(32, 5, 1);
  ASSERT_EQ(2, this->pin_count.pinCountInPart(32, 4));
  ASSERT_EQ(1, this->pin_count.pinCountInPart(32, 5));
}

TYPED_TEST(APinCountDataStructure, SetsPinCountPart4_k32_Max2) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 32;
  const HypernodeID max_value = 2;
  this->initialize(num_hyperedges, k, max_value);

  std::vector<HypernodeID> expected_pin_count(k, 0);
  for ( PartitionID block = 0; block < k; ++block ) {
    expected_pin_count[block] = rand() % max_value;
    this->pin_count.setPinCountInPart(16, block, expected_pin_count[block]);
  }

  for ( PartitionID block = 0; block < k; ++block ) {
    ASSERT_EQ(expected_pin_count[block], this->pin_count.pinCountInPart(16, block));
  }
}

TYPED_TEST(APinCountDataStructure, SetsPinCountPart5_k32_Max2) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 32;
  const HypernodeID max_value = 2;
  this->initialize(num_hyperedges, k, max_value);

  std::vector<std::vector<HypernodeID>> expected_pin_count(
    num_hyperedges, std::vector<HypernodeID>(k, 0));
  for ( HyperedgeID he = 0; he < num_hyperedges; ++he ) {
    for ( PartitionID block = 0; block < k; ++block ) {
      expected_pin_count[he][block] = rand() % max_value;
      this->pin_count.setPinCountInPart(he, block, expected_pin_count[he][block]);
    }
  }

  for ( HyperedgeID he = 0; he < num_hyperedges; ++he ) {
    for ( PartitionID block = 0; block < k; ++block ) {
      ASSERT_EQ(expected_pin_count[he][block], this->pin_count.pinCountInPart(he, block));
    }
  }
}

TYPED_TEST(APinCountDataStructure, IncrementsPinCountInPart1_k32_Max2) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 32;
  const HypernodeID max_value = 2;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.incrementPinCountInPart(5, 31);
  ASSERT_EQ(1, this->pin_count.pinCountInPart(5, 31));
}

TYPED_TEST(APinCountDataStructure, IncrementsPinCountInPart2_k32_Max2) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 32;
  const HypernodeID max_value = 2;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.incrementPinCountInPart(5, 31);
  this->pin_count.incrementPinCountInPart(5, 30);
  ASSERT_EQ(1, this->pin_count.pinCountInPart(5, 31));
  ASSERT_EQ(1, this->pin_count.pinCountInPart(5, 30));
}

TYPED_TEST(APinCountDataStructure, IncrementsPinCountInPart3_k32_Max2) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 32;
  const HypernodeID max_value = 2;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.incrementPinCountInPart(5, 31);
  this->pin_count.incrementPinCountInPart(5, 31);
  ASSERT_EQ(2, this->pin_count.pinCountInPart(5, 31));
}

TYPED_TEST(APinCountDataStructure, DecrementsPinCountInPart1_k32_Max2) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 32;
  const HypernodeID max_value = 2;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.setPinCountInPart(5, 31, 2);
  this->pin_count.decrementPinCountInPart(5, 31);
  ASSERT_EQ(1, this->pin_count.pinCountInPart(5, 31));
}

TYPED_TEST(APinCountDataStructure, DecrementsPinCountInPart2_k32_Max2) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 32;
  const HypernodeID max_value = 2;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.setPinCountInPart(5, 31, 2);
  this->pin_count.setPinCountInPart(5, 30, 1);
  this->pin_count.decrementPinCountInPart(5, 31);
  this->pin_count.decrementPinCountInPart(5, 30);
  ASSERT_EQ(1, this->pin_count.pinCountInPart(5, 31));
  ASSERT_EQ(0, this->pin_count.pinCountInPart(5, 30));
}

TYPED_TEST(APinCountDataStructure, DecrementsPinCountInPart3_k32_Max2) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 32;
  const HypernodeID max_value = 2;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.setPinCountInPart(5, 31, 2);
  this->pin_count.decrementPinCountInPart(5, 31);
  this->pin_count.decrementPinCountInPart(5, 31);
  ASSERT_EQ(0, this->pin_count.pinCountInPart(5, 31));
}

TYPED_TEST(APinCountDataStructure, ModifyTwoHyperedgesConcurrently1_k32_Max2) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 32;
  const HypernodeID max_value = 2;
  this->initialize(num_hyperedges, k, max_value);

  executeConcurrent([&] {
    this->pin_count.setPinCountInPart(5, 4, 2);
  }, [&] {
    this->pin_count.setPinCountInPart(6, 1, 1);
  });

  ASSERT_EQ(2, this->pin_count.pinCountInPart(5, 4));
  ASSERT_EQ(1, this->pin_count.pinCountInPart(6, 1));
}

TYPED_TEST(APinCountDataStructure, ModifyTwoHyperedgesConcurrently2_k32_Max2) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 32;
  const HypernodeID max_value = 2;
  this->initialize(num_hyperedges, k, max_value);

  executeConcurrent([&] {
    this->pin_count.setPinCountInPart(5, 4, 2);
    this->pin_count.decrementPinCountInPart(5, 4);
  }, [&] {
    this->pin_count.setPinCountInPart(6, 1, 1);
    this->pin_count.incrementPinCountInPart(6, 1);
  });

  ASSERT_EQ(1, this->pin_count.pinCountInPart(5, 4));
  ASSERT_EQ(2, this->pin_count.pinCountInPart(6, 1));
}

TYPED_TEST(APinCountDataStructure, ModifyTwoHyperedgesConcurrently3_k32_Max2) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 32;
  const HypernodeID max_value = 2;
  this->initialize(num_hyperedges, k, max_value);

  executeConcurrent([&] {
    this->pin_count.setPinCountInPart(5, 4, 2);
    this->pin_count.decrementPinCountInPart(5, 4);
    this->pin_count.setPinCountInPart(7, 19, 2);
  }, [&] {
    this->pin_count.setPinCountInPart(6, 1, 1);
    this->pin_count.incrementPinCountInPart(6, 1);
    this->pin_count.setPinCountInPart(4, 18, 1);
  });

  ASSERT_EQ(1, this->pin_count.pinCountInPart(4, 18));
  ASSERT_EQ(1, this->pin_count.pinCountInPart(5, 4));
  ASSERT_EQ(2, this->pin_count.pinCountInPart(6, 1));
  ASSERT_EQ(2, this->pin_count.pinCountInPart(7, 19));
}

TYPED_TEST(APinCountDataStructure, IsZeroInitialized_k20_Max8) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 20;
  const HypernodeID max_value = 8;
  this->initialize(num_hyperedges, k, max_value);

  for ( HyperedgeID he = 0; he < num_hyperedges; ++he ) {
    for ( PartitionID block = 0; block < k; ++block ) {
      ASSERT_EQ(0, this->pin_count.pinCountInPart(he, block));
    }
  }
}

TYPED_TEST(APinCountDataStructure, SetsPinCountPart1_k20_Max8) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 20;
  const HypernodeID max_value = 8;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.setPinCountInPart(4, 2, 8);
  ASSERT_EQ(8, this->pin_count.pinCountInPart(4, 2));
}


TYPED_TEST(APinCountDataStructure, SetsPinCountPart2_k20_Max8) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 20;
  const HypernodeID max_value = 8;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.setPinCountInPart(4, 19, 7);
  ASSERT_EQ(7, this->pin_count.pinCountInPart(4, 19));
}

TYPED_TEST(APinCountDataStructure, SetsPinCountPart3_k20_Max8) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 20;
  const HypernodeID max_value = 8;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.setPinCountInPart(32, 4, 7);
  this->pin_count.setPinCountInPart(32, 5, 6);
  ASSERT_EQ(7, this->pin_count.pinCountInPart(32, 4));
  ASSERT_EQ(6, this->pin_count.pinCountInPart(32, 5));
}

TYPED_TEST(APinCountDataStructure, SetsPinCountPart4_k20_Max8) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 20;
  const HypernodeID max_value = 8;
  this->initialize(num_hyperedges, k, max_value);

  std::vector<HypernodeID> expected_pin_count(k, 0);
  for ( PartitionID block = 0; block < k; ++block ) {
    expected_pin_count[block] = rand() % max_value;
    this->pin_count.setPinCountInPart(16, block, expected_pin_count[block]);
  }

  for ( PartitionID block = 0; block < k; ++block ) {
    ASSERT_EQ(expected_pin_count[block], this->pin_count.pinCountInPart(16, block));
  }
}

TYPED_TEST(APinCountDataStructure, SetsPinCountPart5_k20_Max8) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 20;
  const HypernodeID max_value = 8;
  this->initialize(num_hyperedges, k, max_value);

  std::vector<std::vector<HypernodeID>> expected_pin_count(
    num_hyperedges, std::vector<HypernodeID>(k, 0));
  for ( HyperedgeID he = 0; he < num_hyperedges; ++he ) {
    for ( PartitionID block = 0; block < k; ++block ) {
      expected_pin_count[he][block] = rand() % max_value;
      this->pin_count.setPinCountInPart(he, block, expected_pin_count[he][block]);
    }
  }

  for ( HyperedgeID he = 0; he < num_hyperedges; ++he ) {
    for ( PartitionID block = 0; block < k; ++block ) {
      ASSERT_EQ(expected_pin_count[he][block], this->pin_count.pinCountInPart(he, block));
    }
  }
}

TYPED_TEST(APinCountDataStructure, IncrementsPinCountInPart1_k20_Max8) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 20;
  const HypernodeID max_value = 8;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.incrementPinCountInPart(5, 19);
  ASSERT_EQ(1, this->pin_count.pinCountInPart(5, 19));
}

TYPED_TEST(APinCountDataStructure, IncrementsPinCountInPart2_k20_Max8) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 20;
  const HypernodeID max_value = 8;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.incrementPinCountInPart(5, 19);
  this->pin_count.incrementPinCountInPart(5, 19);
  this->pin_count.incrementPinCountInPart(5, 18);
  ASSERT_EQ(2, this->pin_count.pinCountInPart(5, 19));
  ASSERT_EQ(1, this->pin_count.pinCountInPart(5, 18));
}

TYPED_TEST(APinCountDataStructure, IncrementsPinCountInPart3_k20_Max8) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 20;
  const HypernodeID max_value = 8;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.incrementPinCountInPart(5, 19);
  this->pin_count.incrementPinCountInPart(5, 19);
  this->pin_count.incrementPinCountInPart(5, 19);
  this->pin_count.incrementPinCountInPart(5, 19);
  this->pin_count.incrementPinCountInPart(5, 19);
  ASSERT_EQ(5, this->pin_count.pinCountInPart(5, 19));
}

TYPED_TEST(APinCountDataStructure, DecrementsPinCountInPart1_k20_Max8) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 20;
  const HypernodeID max_value = 8;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.setPinCountInPart(5, 19, 5);
  this->pin_count.decrementPinCountInPart(5, 19);
  ASSERT_EQ(4, this->pin_count.pinCountInPart(5, 19));
}

TYPED_TEST(APinCountDataStructure, DecrementsPinCountInPart2_k20_Max8) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 20;
  const HypernodeID max_value = 8;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.setPinCountInPart(5, 19, 5);
  this->pin_count.setPinCountInPart(5, 18, 4);
  this->pin_count.decrementPinCountInPart(5, 19);
  this->pin_count.decrementPinCountInPart(5, 18);
  ASSERT_EQ(4, this->pin_count.pinCountInPart(5, 19));
  ASSERT_EQ(3, this->pin_count.pinCountInPart(5, 18));
}

TYPED_TEST(APinCountDataStructure, DecrementsPinCountInPart3_k20_Max8) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 20;
  const HypernodeID max_value = 8;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.setPinCountInPart(5, 19, 5);
  this->pin_count.decrementPinCountInPart(5, 19);
  this->pin_count.decrementPinCountInPart(5, 19);
  this->pin_count.decrementPinCountInPart(5, 19);
  this->pin_count.decrementPinCountInPart(5, 19);
  ASSERT_EQ(1, this->pin_count.pinCountInPart(5, 19));
}

TYPED_TEST(APinCountDataStructure, ModifyTwoHyperedgesConcurrently1_k20_Max8) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 20;
  const HypernodeID max_value = 8;
  this->initialize(num_hyperedges, k, max_value);

  executeConcurrent([&] {
    this->pin_count.setPinCountInPart(5, 4, 2);
  }, [&] {
    this->pin_count.setPinCountInPart(6, 1, 1);
  });

  ASSERT_EQ(2, this->pin_count.pinCountInPart(5, 4));
  ASSERT_EQ(1, this->pin_count.pinCountInPart(6, 1));
}

TYPED_TEST(APinCountDataStructure, ModifyTwoHyperedgesConcurrently2_k20_Max8) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 20;
  const HypernodeID max_value = 8;
  this->initialize(num_hyperedges, k, max_value);

  executeConcurrent([&] {
    this->pin_count.setPinCountInPart(5, 4, 2);
    this->pin_count.decrementPinCountInPart(5, 4);
  }, [&] {
    this->pin_count.setPinCountInPart(6, 1, 1);
    this->pin_count.incrementPinCountInPart(6, 1);
  });

  ASSERT_EQ(1, this->pin_count.pinCountInPart(5, 4));
  ASSERT_EQ(2, this->pin_count.pinCountInPart(6, 1));
}

TYPED_TEST(APinCountDataStructure, ModifyTwoHyperedgesConcurrently3_k20_Max8) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 20;
  const HypernodeID max_value = 8;
  this->initialize(num_hyperedges, k, max_value);

  executeConcurrent([&] {
    this->pin_count.setPinCountInPart(5, 4, 8);
    this->pin_count.decrementPinCountInPart(5, 4);
    this->pin_count.setPinCountInPart(7, 19, 7);
  }, [&] {
    this->pin_count.setPinCountInPart(6, 1, 6);
    this->pin_count.incrementPinCountInPart(6, 1);
    this->pin_count.setPinCountInPart(4, 18, 4);
  });

  ASSERT_EQ(4, this->pin_count.pinCountInPart(4, 18));
  ASSERT_EQ(7, this->pin_count.pinCountInPart(5, 4));
  ASSERT_EQ(7, this->pin_count.pinCountInPart(6, 1));
  ASSERT_EQ(7, this->pin_count.pinCountInPart(7, 19));
}

TYPED_TEST(APinCountDataStructure, IsZeroInitialized_k30_Max30) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 30;
  const HypernodeID max_value = 30;
  this->initialize(num_hyperedges, k, max_value);

  for ( HyperedgeID he = 0; he < num_hyperedges; ++he ) {
    for ( PartitionID block = 0; block < k; ++block ) {
      ASSERT_EQ(0, this->pin_count.pinCountInPart(he, block));
    }
  }
}

TYPED_TEST(APinCountDataStructure, SetsPinCountPart1_k30_Max30) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 30;
  const HypernodeID max_value = 30;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.setPinCountInPart(4, 2, 30);
  ASSERT_EQ(30, this->pin_count.pinCountInPart(4, 2));
}


TYPED_TEST(APinCountDataStructure, SetsPinCountPart2_k30_Max30) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 30;
  const HypernodeID max_value = 30;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.setPinCountInPart(4, 19, 29);
  ASSERT_EQ(29, this->pin_count.pinCountInPart(4, 19));
}

TYPED_TEST(APinCountDataStructure, SetsPinCountPart3_k30_Max30) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 30;
  const HypernodeID max_value = 30;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.setPinCountInPart(32, 4, 23);
  this->pin_count.setPinCountInPart(32, 5, 22);
  ASSERT_EQ(23, this->pin_count.pinCountInPart(32, 4));
  ASSERT_EQ(22, this->pin_count.pinCountInPart(32, 5));
}

TYPED_TEST(APinCountDataStructure, SetsPinCountPart4_k30_Max30) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 30;
  const HypernodeID max_value = 30;
  this->initialize(num_hyperedges, k, max_value);

  std::vector<HypernodeID> expected_pin_count(k, 0);
  for ( PartitionID block = 0; block < k; ++block ) {
    expected_pin_count[block] = rand() % max_value;
    this->pin_count.setPinCountInPart(16, block, expected_pin_count[block]);
  }

  for ( PartitionID block = 0; block < k; ++block ) {
    ASSERT_EQ(expected_pin_count[block], this->pin_count.pinCountInPart(16, block));
  }
}

TYPED_TEST(APinCountDataStructure, SetsPinCountPart5_k30_Max30) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 30;
  const HypernodeID max_value = 30;
  this->initialize(num_hyperedges, k, max_value);

  std::vector<std::vector<HypernodeID>> expected_pin_count(
    num_hyperedges, std::vector<HypernodeID>(k, 0));
  for ( HyperedgeID he = 0; he < num_hyperedges; ++he ) {
    for ( PartitionID block = 0; block < k; ++block ) {
      expected_pin_count[he][block] = rand() % max_value;
      this->pin_count.setPinCountInPart(he, block, expected_pin_count[he][block]);
    }
  }

  for ( HyperedgeID he = 0; he < num_hyperedges; ++he ) {
    for ( PartitionID block = 0; block < k; ++block ) {
      ASSERT_EQ(expected_pin_count[he][block], this->pin_count.pinCountInPart(he, block));
    }
  }
}

TYPED_TEST(APinCountDataStructure, IncrementsPinCountInPart1_k30_Max30) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 30;
  const HypernodeID max_value = 30;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.incrementPinCountInPart(5, 19);
  ASSERT_EQ(1, this->pin_count.pinCountInPart(5, 19));
}

TYPED_TEST(APinCountDataStructure, IncrementsPinCountInPart2_k30_Max30) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 30;
  const HypernodeID max_value = 30;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.incrementPinCountInPart(5, 19);
  this->pin_count.incrementPinCountInPart(5, 19);
  this->pin_count.incrementPinCountInPart(5, 18);
  ASSERT_EQ(2, this->pin_count.pinCountInPart(5, 19));
  ASSERT_EQ(1, this->pin_count.pinCountInPart(5, 18));
}

TYPED_TEST(APinCountDataStructure, IncrementsPinCountInPart3_k30_Max30) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 30;
  const HypernodeID max_value = 30;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.incrementPinCountInPart(5, 19);
  this->pin_count.incrementPinCountInPart(5, 19);
  this->pin_count.incrementPinCountInPart(5, 19);
  this->pin_count.incrementPinCountInPart(5, 19);
  this->pin_count.incrementPinCountInPart(5, 19);
  this->pin_count.incrementPinCountInPart(5, 19);
  this->pin_count.incrementPinCountInPart(5, 19);
  this->pin_count.incrementPinCountInPart(5, 19);
  this->pin_count.incrementPinCountInPart(5, 19);
  ASSERT_EQ(9, this->pin_count.pinCountInPart(5, 19));
}

TYPED_TEST(APinCountDataStructure, DecrementsPinCountInPart1_k30_Max30) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 30;
  const HypernodeID max_value = 30;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.setPinCountInPart(5, 19, 20);
  this->pin_count.decrementPinCountInPart(5, 19);
  ASSERT_EQ(19, this->pin_count.pinCountInPart(5, 19));
}

TYPED_TEST(APinCountDataStructure, DecrementsPinCountInPart2_k30_Max30) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 30;
  const HypernodeID max_value = 30;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.setPinCountInPart(5, 19, 20);
  this->pin_count.setPinCountInPart(5, 18, 19);
  this->pin_count.decrementPinCountInPart(5, 19);
  this->pin_count.decrementPinCountInPart(5, 18);
  ASSERT_EQ(19, this->pin_count.pinCountInPart(5, 19));
  ASSERT_EQ(18, this->pin_count.pinCountInPart(5, 18));
}

TYPED_TEST(APinCountDataStructure, DecrementsPinCountInPart3_k30_Max30) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 30;
  const HypernodeID max_value = 30;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.setPinCountInPart(5, 19, 20);
  this->pin_count.decrementPinCountInPart(5, 19);
  this->pin_count.decrementPinCountInPart(5, 19);
  this->pin_count.decrementPinCountInPart(5, 19);
  this->pin_count.decrementPinCountInPart(5, 19);
  this->pin_count.decrementPinCountInPart(5, 19);
  this->pin_count.decrementPinCountInPart(5, 19);
  this->pin_count.decrementPinCountInPart(5, 19);
  this->pin_count.decrementPinCountInPart(5, 19);
  this->pin_count.decrementPinCountInPart(5, 19);
  this->pin_count.decrementPinCountInPart(5, 19);
  ASSERT_EQ(10, this->pin_count.pinCountInPart(5, 19));
}

TYPED_TEST(APinCountDataStructure, ModifyTwoHyperedgesConcurrently1_k30_Max30) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 30;
  const HypernodeID max_value = 30;
  this->initialize(num_hyperedges, k, max_value);

  executeConcurrent([&] {
    this->pin_count.setPinCountInPart(5, 4, 2);
  }, [&] {
    this->pin_count.setPinCountInPart(6, 1, 1);
  });

  ASSERT_EQ(2, this->pin_count.pinCountInPart(5, 4));
  ASSERT_EQ(1, this->pin_count.pinCountInPart(6, 1));
}

TYPED_TEST(APinCountDataStructure, ModifyTwoHyperedgesConcurrently2_k30_Max30) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 30;
  const HypernodeID max_value = 30;
  this->initialize(num_hyperedges, k, max_value);

  executeConcurrent([&] {
    this->pin_count.setPinCountInPart(5, 4, 2);
    this->pin_count.decrementPinCountInPart(5, 4);
  }, [&] {
    this->pin_count.setPinCountInPart(6, 1, 1);
    this->pin_count.incrementPinCountInPart(6, 1);
  });

  ASSERT_EQ(1, this->pin_count.pinCountInPart(5, 4));
  ASSERT_EQ(2, this->pin_count.pinCountInPart(6, 1));
}

TYPED_TEST(APinCountDataStructure, ModifyTwoHyperedgesConcurrently3_k30_Max30) {
  const HyperedgeID num_hyperedges = 100;
  const PartitionID k = 30;
  const HypernodeID max_value = 30;
  this->initialize(num_hyperedges, k, max_value);

  executeConcurrent([&] {
    this->pin_count.setPinCountInPart(5, 4, 20);
    this->pin_count.decrementPinCountInPart(5, 4);
    this->pin_count.setPinCountInPart(7, 19, 30);
  }, [&] {
    this->pin_count.setPinCountInPart(6, 1, 26);
    this->pin_count.incrementPinCountInPart(6, 1);
    this->pin_count.setPinCountInPart(4, 18, 25);
  });

  ASSERT_EQ(25, this->pin_count.pinCountInPart(4, 18));
  ASSERT_EQ(19, this->pin_count.pinCountInPart(5, 4));
  ASSERT_EQ(27, this->pin_count.pinCountInPart(6, 1));
  ASSERT_EQ(30, this->pin_count.pinCountInPart(7, 19));
}

TYPED_TEST(APinCountDataStructure, MakesASnapshot1) {
  const HyperedgeID num_hyperedges = 4;
  const PartitionID k = 8;
  const HypernodeID max_value = 17;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.setPinCountInPart(2, 0, 2);
  this->pin_count.setPinCountInPart(2, 2, 4);
  this->pin_count.setPinCountInPart(2, 5, 3);
  this->pin_count.setPinCountInPart(2, 7, 1);
  PinCountSnapshot& snapshot = this->pin_count.snapshot(2);
  EXPECT_EQ(2, snapshot.pinCountInPart(0));
  EXPECT_EQ(0, snapshot.pinCountInPart(1));
  EXPECT_EQ(4, snapshot.pinCountInPart(2));
  EXPECT_EQ(0, snapshot.pinCountInPart(3));
  EXPECT_EQ(0, snapshot.pinCountInPart(4));
  EXPECT_EQ(3, snapshot.pinCountInPart(5));
  EXPECT_EQ(0, snapshot.pinCountInPart(6));
  EXPECT_EQ(1, snapshot.pinCountInPart(7));
  snapshot.decrementPinCountInPart(7);
  snapshot.setPinCountInPart(4,2);
  EXPECT_EQ(2, snapshot.pinCountInPart(0));
  EXPECT_EQ(0, snapshot.pinCountInPart(1));
  EXPECT_EQ(4, snapshot.pinCountInPart(2));
  EXPECT_EQ(0, snapshot.pinCountInPart(3));
  EXPECT_EQ(2, snapshot.pinCountInPart(4));
  EXPECT_EQ(3, snapshot.pinCountInPart(5));
  EXPECT_EQ(0, snapshot.pinCountInPart(6));
  EXPECT_EQ(0, snapshot.pinCountInPart(7));
}

TYPED_TEST(APinCountDataStructure, MakesASnapshot2) {
  const HyperedgeID num_hyperedges = 4;
  const PartitionID k = 8;
  const HypernodeID max_value = 17;
  this->initialize(num_hyperedges, k, max_value);

  this->pin_count.setPinCountInPart(2, 0, 2);
  this->pin_count.setPinCountInPart(2, 2, 4);
  this->pin_count.setPinCountInPart(2, 5, 3);
  this->pin_count.setPinCountInPart(2, 7, 1);
  this->pin_count.setPinCountInPart(3, 1, 3);
  this->pin_count.setPinCountInPart(3, 3, 1);
  this->pin_count.setPinCountInPart(3, 4, 2);
  this->pin_count.setPinCountInPart(3, 6, 4);
  PinCountSnapshot& snapshot_1 = this->pin_count.snapshot(2);
  EXPECT_EQ(2, snapshot_1.pinCountInPart(0));
  EXPECT_EQ(0, snapshot_1.pinCountInPart(1));
  EXPECT_EQ(4, snapshot_1.pinCountInPart(2));
  EXPECT_EQ(0, snapshot_1.pinCountInPart(3));
  EXPECT_EQ(0, snapshot_1.pinCountInPart(4));
  EXPECT_EQ(3, snapshot_1.pinCountInPart(5));
  EXPECT_EQ(0, snapshot_1.pinCountInPart(6));
  EXPECT_EQ(1, snapshot_1.pinCountInPart(7));
  PinCountSnapshot& snapshot_2 = this->pin_count.snapshot(3);
  EXPECT_EQ(0, snapshot_2.pinCountInPart(0));
  EXPECT_EQ(3, snapshot_2.pinCountInPart(1));
  EXPECT_EQ(0, snapshot_2.pinCountInPart(2));
  EXPECT_EQ(1, snapshot_2.pinCountInPart(3));
  EXPECT_EQ(2, snapshot_2.pinCountInPart(4));
  EXPECT_EQ(0, snapshot_2.pinCountInPart(5));
  EXPECT_EQ(4, snapshot_2.pinCountInPart(6));
  EXPECT_EQ(0, snapshot_2.pinCountInPart(7));
}


#ifdef KAHYPAR_ENABLE_LARGE_K_PARTITIONING_FEATURES

using SparsePinCountsAsConnectivitySet = APinCountDataStructure<SparsePinCounts>;

void add(const HyperedgeID he, SparsePinCounts& conn_set, const std::set<PartitionID>& ids) {
  for (const PartitionID& id : ids) {
    conn_set.incrementPinCountInPart(he, id);
  }
}

void remove(const HyperedgeID he, SparsePinCounts& conn_set, const std::set<PartitionID>& ids) {
  for (const PartitionID& id : ids) {
    const HypernodeID pin_count = conn_set.pinCountInPart(he, id);
    for ( HypernodeID i = 0; i < pin_count; ++i ) {
      conn_set.decrementPinCountInPart(he, id);
    }
  }
}

void verify(const HyperedgeID he,
            const SparsePinCounts& conn_set,
            const PartitionID k,
            const std::set<PartitionID>& contained) {
  // Verify bitset in connectivity set
  ASSERT_EQ(contained.size(), conn_set.connectivity(he));
  for (PartitionID i = 0; i < k; ++i) {
    if (contained.find(i) != contained.end()) {
      ASSERT_TRUE(conn_set.contains(he, i)) << V(i);
    } else {
      ASSERT_FALSE(conn_set.contains(he, i)) << V(i);
    }
  }

  // Verify iterator
  size_t connectivity = 0;
  for (const PartitionID id : conn_set.connectivitySet(he)) {
    ASSERT_TRUE(contained.find(id) != contained.end()) << V(id);
    ++connectivity;
  }
  ASSERT_EQ(contained.size(), connectivity);
}

TEST_F(SparsePinCountsAsConnectivitySet, IsCorrectInitialized) {
  initialize(1, 32, 0);
  verify(0, pin_count, 32, { });
}

TEST_F(SparsePinCountsAsConnectivitySet, AddOnePartition1) {
  initialize(1, 32, 0);
  pin_count.incrementPinCountInPart(0, 2);
  verify(0, pin_count, 32, { 2 });
}

TEST_F(SparsePinCountsAsConnectivitySet, AddOnePartition2) {
  initialize(1, 32, 0);
  pin_count.incrementPinCountInPart(0, 14);
  verify(0, pin_count, 32, { 14 });
}

TEST_F(SparsePinCountsAsConnectivitySet, AddOnePartition3) {
  initialize(1, 32, 0);
  pin_count.incrementPinCountInPart(0, 23);
  verify(0, pin_count, 32, { 23 });
}

TEST_F(SparsePinCountsAsConnectivitySet, AddOnePartition4) {
  initialize(1, 32, 0);
  pin_count.incrementPinCountInPart(0, 30);
  verify(0, pin_count, 32, { 30 });
}

TEST_F(SparsePinCountsAsConnectivitySet, AddTwoPartitions1) {
  initialize(1, 32, 0);
  std::set<PartitionID> added = { 5, 31 };
  add(0, pin_count, added);
  verify(0, pin_count, 32, added);
}

TEST_F(SparsePinCountsAsConnectivitySet, AddTwoPartitions2) {
  initialize(1, 32, 0);
  std::set<PartitionID> added = { 14, 24 };
  add(0, pin_count, added);
  verify(0, pin_count, 32, added);
}

TEST_F(SparsePinCountsAsConnectivitySet, AddTwoPartitions3) {
  initialize(1, 32, 0);
  std::set<PartitionID> added = { 7, 16 };
  add(0, pin_count, added);
  verify(0, pin_count, 32, added);
}

TEST_F(SparsePinCountsAsConnectivitySet, AddSeveralPartitions1) {
  initialize(1, 32, 0);
  std::set<PartitionID> added = { 0, 1, 5, 14, 24, 27, 31 };
  add(0, pin_count, added);
  verify(0, pin_count, 32, added);
}

TEST_F(SparsePinCountsAsConnectivitySet, AddSeveralPartitions2) {
  initialize(1, 32, 0);
  std::set<PartitionID> added = { 5, 6, 7, 11, 13, 15, 24, 28, 30 };
  add(0, pin_count, added);
  verify(0, pin_count, 32, added);
}

TEST_F(SparsePinCountsAsConnectivitySet, AddSeveralPartitions3) {
  initialize(1, 32, 0);
  std::set<PartitionID> added = { 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };
  add(0, pin_count, added);
  verify(0, pin_count, 32, added);
}

TEST_F(SparsePinCountsAsConnectivitySet, AddTwoPartitionsAndRemoveOne1) {
  initialize(1, 32, 0);
  std::set<PartitionID> added = { 5, 31 };
  std::set<PartitionID> removed = { 31 };
  add(0, pin_count, added);
  remove(0, pin_count, removed);
  for (const PartitionID id : removed) {
    added.erase(id);
  }
  verify(0, pin_count, 32, added);
}

TEST_F(SparsePinCountsAsConnectivitySet, AddTwoPartitionsAndRemoveOne2) {
  initialize(1, 32, 0);
  std::set<PartitionID> added = { 16, 17 };
  std::set<PartitionID> removed = { 16 };
  add(0, pin_count, added);
  remove(0, pin_count, removed);
  for (const PartitionID id : removed) {
    added.erase(id);
  }
  verify(0, pin_count, 32, added);
}

TEST_F(SparsePinCountsAsConnectivitySet, AddTwoPartitionsAndRemoveOne3) {
  initialize(1, 32, 0);
  std::set<PartitionID> added = { 7, 21 };
  std::set<PartitionID> removed = { 7 };
  add(0, pin_count, added);
  remove(0, pin_count, removed);
  for (const PartitionID id : removed) {
    added.erase(id);
  }
  verify(0, pin_count, 32, added);
}

TEST_F(SparsePinCountsAsConnectivitySet, AddTwoPartitionsAndRemoveOne4) {
  initialize(1, 32, 0);
  std::set<PartitionID> added = { 25, 27 };
  std::set<PartitionID> removed = { 27 };
  add(0, pin_count, added);
  remove(0, pin_count, removed);
  for (const PartitionID id : removed) {
    added.erase(id);
  }
  verify(0, pin_count, 32, added);
}

TEST_F(SparsePinCountsAsConnectivitySet, AddSeveralPartitionsAndRemoveSeveral1) {
  initialize(1, 32, 0);
  std::set<PartitionID> added = { 1, 13, 15, 23, 24, 30 };
  std::set<PartitionID> removed = { 13, 15, 23 };
  add(0, pin_count, added);
  remove(0, pin_count, removed);
  for (const PartitionID id : removed) {
    added.erase(id);
  }
  verify(0, pin_count, 32, added);
}

TEST_F(SparsePinCountsAsConnectivitySet, AddSeveralPartitionsAndRemoveSeveral2) {
  initialize(1, 32, 0);
  std::set<PartitionID> added = { 2, 5, 6, 14, 15, 21, 23, 29 };
  std::set<PartitionID> removed = { 5, 14, 21, 29 };
  add(0, pin_count, added);
  remove(0, pin_count, removed);
  for (const PartitionID id : removed) {
    added.erase(id);
  }
  verify(0, pin_count, 32, added);
}

TEST_F(SparsePinCountsAsConnectivitySet, AddSeveralPartitionsAndRemoveSeveral3) {
  initialize(1, 32, 0);
  std::set<PartitionID> added = { 0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29 };
  std::set<PartitionID> removed = { 5, 6, 7, 24, 25, 26, 27 };
  add(0, pin_count, added);
  remove(0, pin_count, removed);
  for (const PartitionID id : removed) {
    added.erase(id);
  }
  verify(0, pin_count, 32, added);
}


TEST_F(SparsePinCountsAsConnectivitySet, AddConcurrentTwoPartitions1) {
  initialize(2, 32, 0);
  executeConcurrent([&] {
      pin_count.incrementPinCountInPart(0, 7);
    }, [&] {
      pin_count.incrementPinCountInPart(1, 16);
    });
  verify(0, pin_count, 32, { 7 });
  verify(1, pin_count, 32, { 16 });
}

TEST_F(SparsePinCountsAsConnectivitySet, AddConcurrentTwoPartitions2) {
  initialize(2, 32, 0);
  executeConcurrent([&] {
      pin_count.incrementPinCountInPart(0, 4);
    }, [&] {
      pin_count.incrementPinCountInPart(1, 5);
    });
  verify(0, pin_count, 32, { 4 });
  verify(1, pin_count, 32, { 5 });
}

TEST_F(SparsePinCountsAsConnectivitySet, AddConcurrentTwoPartitions3) {
  initialize(2, 32, 0);
  executeConcurrent([&] {
      pin_count.incrementPinCountInPart(0, 12);
    }, [&] {
      pin_count.incrementPinCountInPart(1, 14);
    });
  verify(0, pin_count, 32, { 12 });
  verify(1, pin_count, 32, { 14 });
}

TEST_F(SparsePinCountsAsConnectivitySet, AddConcurrentTwoPartitions4) {
  initialize(2, 32, 0);
  executeConcurrent([&] {
      pin_count.incrementPinCountInPart(0, 31);
    }, [&] {
      pin_count.incrementPinCountInPart(1, 31);
    });
  verify(0, pin_count, 32, { 31 });
  verify(1, pin_count, 32, { 31 });
}

TEST_F(SparsePinCountsAsConnectivitySet, AddConcurrentSeveralPartitions1) {
  initialize(2, 32, 0);
  executeConcurrent([&] {
      add(0, pin_count, { 1, 15, 24 });
    }, [&] {
      add(1, pin_count, { 13, 23, 30 });
    });
  verify(0, pin_count, 32, { 1, 15, 24 });
  verify(1, pin_count, 32, { 13, 23, 30 });
}

TEST_F(SparsePinCountsAsConnectivitySet, AddConcurrentSeveralPartitions2) {
  initialize(2, 32, 0);
  executeConcurrent([&] {
      add(0, pin_count, { 0, 15, 24, 28, 31 });
    }, [&] {
      add(1, pin_count, { 4, 23, 25, 30 });
    });
  verify(0, pin_count, 32, { 0, 15, 24, 28, 31 });
  verify(1, pin_count, 32, { 4, 23, 25, 30 });
}

TEST_F(SparsePinCountsAsConnectivitySet, AddConcurrentSeveralPartitions3) {
  initialize(2, 32, 0);
  executeConcurrent([&] {
      add(0, pin_count, { 6, 8, 10, 15, 17, 24, 26 });
    }, [&] {
      add(1, pin_count, { 7, 9, 14, 16, 23, 25 });
    });
  verify(0, pin_count, 32, { 6, 8, 10, 15, 17, 24, 26 });
  verify(1, pin_count, 32, { 7, 9, 14, 16, 23, 25 });
}

TEST_F(SparsePinCountsAsConnectivitySet, AddAndRemoveOnePartitionConcurrently1) {
  initialize(2, 32, 0);
  pin_count.incrementPinCountInPart(0, 4);
  executeConcurrent([&] {
        pin_count.decrementPinCountInPart(0, 4);
      }, [&] {
        pin_count.incrementPinCountInPart(1, 4);
      });
  verify(0, pin_count, 32, { });
  verify(1, pin_count, 32, { 4 });
}

TEST_F(SparsePinCountsAsConnectivitySet, AddAndRemoveOnePartitionConcurrently2) {
  initialize(2, 32, 0);
  pin_count.incrementPinCountInPart(0, 9);
  executeConcurrent([&] {
        pin_count.decrementPinCountInPart(0, 9);
      }, [&] {
        pin_count.incrementPinCountInPart(1, 24);
      });
  verify(0, pin_count, 32, { });
  verify(1, pin_count, 32, { 24 });
}

TEST_F(SparsePinCountsAsConnectivitySet, IteratesThroughPartitionsAndSimultanouslyAddElements) {
  initialize(1, 32, 0);
  add(0, pin_count, { 1, 2, 6, 10, 15, 22, 24, 31 });

  std::atomic<size_t> cnt(0);

  executeConcurrent([&] {
        while (cnt < 1);
        // Add 0. Change should be not visibile in iterator
        pin_count.incrementPinCountInPart(0, 0);
        ++cnt;

        while (cnt < 3);
        // Add 12. Change should be not visibile in iterator
        pin_count.incrementPinCountInPart(0, 12);
        ++cnt;

        while (cnt < 5);
        // Add 14. Change should be not visibile in iterator
        pin_count.incrementPinCountInPart(0, 14);
        ++cnt;

        while (cnt < 7);
        // Add 30. Change should be not visibile in iterator
        pin_count.incrementPinCountInPart(0, 30);
        ++cnt;
      }, [&] {
        std::vector<PartitionID> expected = { 1, 2, 6, 10, 15, 22, 24, 31 };
        size_t i = 0;
        for (const PartitionID& id : pin_count.connectivitySet(0)) {
          if (i == 1) {
            ++cnt;
            while (cnt < 2);
            // Wait until thread 1 adds 0
          }

          if (i == 2) {
            ++cnt;
            while (cnt < 4);
            // Wait until thread 1 adds 12
          }

          if (i == 5) {
            ++cnt;
            while (cnt < 6);
            // Wait until thread 1 adds 14
          }

          if (i == 7) {
            ++cnt;
            while (cnt < 8);
            // Wait until thread 1 adds 30
          }

          EXPECT_EQ(expected[i], id);
          i++;
        }
        ASSERT_EQ(expected.size(), i);
      });

  verify(0, pin_count, 32, { 0, 1, 2, 6, 10, 12, 14, 15, 22, 24, 30, 31 });
}

TEST_F(SparsePinCountsAsConnectivitySet, IteratesThroughPartitionsAndSimultanouslyRemoveElements) {
  initialize(1, 32, 0);
  add(0, pin_count, { 1, 2, 6, 10, 15, 22, 24, 31 });

  std::atomic<size_t> cnt(0);

  executeConcurrent([&] {
        while (cnt < 1);
        // Iteration currently waits at part 2. Removing part 1 will not be visible.
        // Removing part 1 will swap it with part 31 which will then also not visible
        // for the iterator
        // Connectivity Set State -> { 31, 2 |iter| 6, 10, 15, 22, 24 }
        pin_count.decrementPinCountInPart(0, 1);
        ++cnt;

        while (cnt < 3);
        // Remove 15. Since iteration waits at part 6, the change will be visible.
        // Connectivity Set State -> { 31, 2, 6 |iter| 10, 24, 22 }
        pin_count.decrementPinCountInPart(0, 15);
        ++cnt;

        while (cnt < 5);
        // Remove 31. Iteration waits at part 24.
        // This will swap part 22 and 31. Therefore, block 22 will be not visible
        // for the iterator
        // Connectivity Set State -> { 22, 2, 6, 10, 24 |iter| }
        pin_count.decrementPinCountInPart(0, 31);
        ++cnt;
      }, [&] {
        // Note that the iterator will not see all blocks contained in the hyperedge
        // due to concurrent write operation. When we remove a block, we swap it to
        // end. The block at the end is not visible to the iterator if it already
        // iterated over the removed block. E.g. in this example block 22 will not
        // be visible for the iterator.
        std::vector<PartitionID> expected = { 1, 2, 6, 10, 24 };
        size_t i = 0;
        for (const PartitionID& id : pin_count.connectivitySet(0)) {
          if (i == 1) {
            ++cnt;
            while (cnt < 2);
            // Wait at part 2 until thread 1 removes part 1
          }

          if (i == 2) {
            ++cnt;
            while (cnt < 4);
            // Wait at part 6 until thread 1 removes part 15
          }

          if (i == 4) {
            ++cnt;
            while (cnt < 6);
            // Wait at part 24 until thread 1 removes 31
          }

          EXPECT_EQ(expected[i], id);
          i++;
        }
        ASSERT_EQ(expected.size(), i);
      });

  verify(0, pin_count, 32, { 2, 6, 10, 22, 24 });
}

#endif

}  // namespace ds
}  // namespace mt_kahypar
