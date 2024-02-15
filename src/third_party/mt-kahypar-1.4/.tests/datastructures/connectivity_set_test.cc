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
#include <mt-kahypar/macros.h>

#include "gmock/gmock.h"
#include "tbb/task_group.h"

#include "mt-kahypar/datastructures/connectivity_set.h"
#include "mt-kahypar/datastructures/delta_connectivity_set.h"

using ::testing::Test;

namespace mt_kahypar {
namespace ds {
using PartitionID = int32_t;

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

template<typename ConnectivitySet>
void add(ConnectivitySet& conn_set, const std::set<PartitionID>& ids) {
  for (const PartitionID& id : ids) {
    conn_set.add(0, id);
  }
}

template<typename ConnectivitySet>
void remove(ConnectivitySet& conn_set, const std::set<PartitionID>& ids) {
  for (const PartitionID& id : ids) {
    conn_set.remove(0, id);
  }
}

template<typename ConnectivitySet>
void verify(const ConnectivitySet& conn_set,
            const PartitionID k,
            const std::set<PartitionID>& contained) {
  // Verify bitset in connectivity set
  ASSERT_EQ(contained.size(), conn_set.connectivity(0));
  for (PartitionID i = 0; i < k; ++i) {
    if (contained.find(i) != contained.end()) {
      ASSERT_TRUE(conn_set.contains(0, i)) << V(i);
    } else {
      ASSERT_FALSE(conn_set.contains(0, i)) << V(i);
    }
  }

  // Verify iterator
  size_t connectivity = 0;
  for (const PartitionID id : conn_set.connectivitySet(0)) {
    ASSERT_TRUE(contained.find(id) != contained.end()) << V(id);
    ++connectivity;
  }
  ASSERT_EQ(contained.size(), connectivity);
}

TEST(AConnectivitySet, IsCorrectInitialized) {
  ConnectivitySets conn_set(1, 32);
  verify(conn_set, 32, { });
}

TEST(AConnectivitySet, AddOnePartition1) {
  ConnectivitySets conn_set(1, 32);
  conn_set.add(0, 2);
  verify(conn_set, 32, { 2 });
}

TEST(AConnectivitySet, AddOnePartition2) {
  ConnectivitySets conn_set(1, 32);
  conn_set.add(0, 14);
  verify(conn_set, 32, { 14 });
}

TEST(AConnectivitySet, AddOnePartition3) {
  ConnectivitySets conn_set(1, 32);
  conn_set.add(0, 23);
  verify(conn_set, 32, { 23 });
}

TEST(AConnectivitySet, AddOnePartition4) {
  ConnectivitySets conn_set(1, 32);;
  conn_set.add(0, 30);
  verify(conn_set, 32, { 30 });
}

TEST(AConnectivitySet, AddTwoPartitions1) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 5, 31 };
  add(conn_set, added);
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddTwoPartitions2) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 14, 24 };
  add(conn_set, added);
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddTwoPartitions3) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 7, 16 };
  add(conn_set, added);
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddSeveralPartitions1) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 0, 1, 5, 14, 24, 27, 31 };
  add(conn_set, added);
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddSeveralPartitions2) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 5, 6, 7, 11, 13, 15, 24, 28, 30 };
  add(conn_set, added);
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddSeveralPartitions3) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };
  add(conn_set, added);
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddTwoPartitionsAndRemoveOne1) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 5, 31 };
  std::set<PartitionID> removed = { 31 };
  add(conn_set, added);
  remove(conn_set, removed);
  for (const PartitionID id : removed) {
    added.erase(id);
  }
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddTwoPartitionsAndRemoveOne2) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 16, 17 };
  std::set<PartitionID> removed = { 16 };
  add(conn_set, added);
  remove(conn_set, removed);
  for (const PartitionID id : removed) {
    added.erase(id);
  }
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddTwoPartitionsAndRemoveOne3) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 7, 21 };
  std::set<PartitionID> removed = { 7 };
  add(conn_set, added);
  remove(conn_set, removed);
  for (const PartitionID id : removed) {
    added.erase(id);
  }
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddTwoPartitionsAndRemoveOne4) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 25, 27 };
  std::set<PartitionID> removed = { 27 };
  add(conn_set, added);
  remove(conn_set, removed);
  for (const PartitionID id : removed) {
    added.erase(id);
  }
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddSeveralPartitionsAndRemoveSeveral1) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 1, 13, 15, 23, 24, 30 };
  std::set<PartitionID> removed = { 13, 15, 23 };
  add(conn_set, added);
  remove(conn_set, removed);
  for (const PartitionID id : removed) {
    added.erase(id);
  }
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddSeveralPartitionsAndRemoveSeveral2) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 2, 5, 6, 14, 15, 21, 23, 29 };
  std::set<PartitionID> removed = { 5, 14, 21, 29 };
  add(conn_set, added);
  remove(conn_set, removed);
  for (const PartitionID id : removed) {
    added.erase(id);
  }
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddSeveralPartitionsAndRemoveSeveral3) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29 };
  std::set<PartitionID> removed = { 5, 6, 7, 24, 25, 26, 27 };
  add(conn_set, added);
  remove(conn_set, removed);
  for (const PartitionID id : removed) {
    added.erase(id);
  }
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddConcurrentTwoPartitions1) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 7, 16 };
  executeConcurrent([&] {
        conn_set.add(0, 7);
      }, [&] {
        conn_set.add(0, 16);
      });
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddConcurrentTwoPartitions2) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 4, 5 };
  executeConcurrent([&] {
        conn_set.add(0, 4);
      }, [&] {
        conn_set.add(0, 5);
      });
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddConcurrentTwoPartitions3) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 12, 14 };
  executeConcurrent([&] {
        conn_set.add(0, 12);
      }, [&] {
        conn_set.add(0, 14);
      });
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddConcurrentTwoPartitions4) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 30, 31 };
  executeConcurrent([&] {
        conn_set.add(0, 30);
      }, [&] {
        conn_set.add(0, 31);
      });
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddConcurrentSeveralPartitions1) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 1, 13, 15, 23, 24, 30 };
  executeConcurrent([&] {
        add(conn_set, { 1, 15, 24 });
      }, [&] {
        add(conn_set, { 13, 23, 30 });
      });
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddConcurrentSeveralPartitions2) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 0, 4, 15, 23, 24, 25, 28, 30, 31 };
  executeConcurrent([&] {
        add(conn_set, { 0, 15, 24, 28, 31 });
      }, [&] {
        add(conn_set, { 4, 23, 25, 30 });
      });
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddConcurrentSeveralPartitions3) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 6, 7, 8, 9, 10, 14, 15, 16, 17, 23, 24, 25, 26 };
  executeConcurrent([&] {
        add(conn_set, { 6, 8, 10, 15, 17, 24, 26 });
      }, [&] {
        add(conn_set, { 7, 9, 14, 16, 23, 25 });
      });
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddAndRemoveOnePartitionConcurrently1) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 3 };
  conn_set.add(0, 4);
  executeConcurrent([&] {
        conn_set.add(0, 3);
      }, [&] {
        conn_set.remove(0, 4);
      });
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddAndRemoveOnePartitionConcurrently2) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 24 };
  conn_set.add(0, 9);
  executeConcurrent([&] {
        conn_set.add(0, 24);
      }, [&] {
        conn_set.remove(0, 9);
      });
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddAndRemoveOnePartitionConcurrently3) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 15 };
  conn_set.add(0, 31);
  executeConcurrent([&] {
        conn_set.add(0, 15);
      }, [&] {
        conn_set.remove(0, 31);
      });
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddAndRemoveOnePartitionConcurrently4) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 13 };
  conn_set.add(0, 14);
  executeConcurrent([&] {
        conn_set.add(0, 13);
      }, [&] {
        conn_set.remove(0, 14);
      });
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddAndRemoveSeveralPartitionsConcurrently1) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 1, 4, 5, 15, 16, 21, 30 };
  add(conn_set, { 2, 3, 14, 17, 23, 24 });
  executeConcurrent([&] {
        conn_set.add(0, 1);
        conn_set.remove(0, 3);
        conn_set.add(0, 5);
        conn_set.remove(0, 14);
        conn_set.add(0, 16);
        conn_set.remove(0, 23);
        conn_set.add(0, 30);
      }, [&] {
        conn_set.remove(0, 2);
        conn_set.add(0, 4);
        conn_set.remove(0, 17);
        conn_set.add(0, 15);
        conn_set.remove(0, 24);
        conn_set.add(0, 21);
      });
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddAndRemoveSeveralPartitionsConcurrently2) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 4, 5, 8, 10, 18, 21, 22, 27, 31 };
  add(conn_set, { 1, 2, 5, 17, 19, 20, 23, 30 });
  executeConcurrent([&] {
        conn_set.add(0, 4);
        conn_set.remove(0, 1);
        conn_set.add(0, 8);
        conn_set.remove(0, 5);
        conn_set.add(0, 18);
        conn_set.remove(0, 19);
        conn_set.add(0, 22);
        conn_set.remove(0, 23);
        conn_set.add(0, 31);
      }, [&] {
        conn_set.remove(0, 2);
        conn_set.add(0, 5);
        conn_set.remove(0, 17);
        conn_set.add(0, 10);
        conn_set.remove(0, 20);
        conn_set.add(0, 21);
        conn_set.remove(0, 30);
        conn_set.add(0, 27);
      });
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddAndRemoveSeveralPartitionsConcurrently3) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 13, 14, 15, 17, 18, 22, 23, 24, 25, 26, 27 };
  add(conn_set, { 6, 7, 8, 9, 10, 19, 20, 21, 28, 29, 30 });
  executeConcurrent([&] {
        conn_set.add(0, 13);
        conn_set.remove(0, 6);
        conn_set.add(0, 15);
        conn_set.remove(0, 8);
        conn_set.add(0, 18);
        conn_set.remove(0, 10);
        conn_set.add(0, 23);
        conn_set.remove(0, 20);
        conn_set.add(0, 25);
        conn_set.remove(0, 28);
        conn_set.add(0, 27);
        conn_set.remove(0, 30);
      }, [&] {
        conn_set.remove(0, 7);
        conn_set.add(0, 14);
        conn_set.remove(0, 9);
        conn_set.add(0, 17);
        conn_set.remove(0, 19);
        conn_set.add(0, 22);
        conn_set.remove(0, 21);
        conn_set.add(0, 24);
        conn_set.remove(0, 29);
        conn_set.add(0, 26);
      });
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, AddAndRemoveSamePartitionConcurrently1) {
  ConnectivitySets conn_set(1, 32);;
  executeConcurrent([&] {
        conn_set.add(0, 15);
      }, [&] {
        conn_set.remove(0, 15);
      });
  verify(conn_set, 32, { });
}

TEST(AConnectivitySet, AddAndRemoveSamePartitionConcurrently2) {
  ConnectivitySets conn_set(1, 32);;
  conn_set.add(0, 15);
  executeConcurrent([&] {
        conn_set.add(0, 15);
      }, [&] {
        conn_set.remove(0, 15);
      });
  verify(conn_set, 32, { 15 });
}

TEST(AConnectivitySet, AddAndRemoveSamePartitionConcurrently3) {
  ConnectivitySets conn_set(1, 32);;
  executeConcurrent([&] {
        conn_set.add(0, 6);
      }, [&] {
        conn_set.remove(0, 6);
      });
  verify(conn_set, 32, { });
}

TEST(AConnectivitySet, AddAndRemoveSamePartitionConcurrently4) {
  ConnectivitySets conn_set(1, 32);;
  conn_set.add(0, 6);
  executeConcurrent([&] {
        conn_set.add(0, 6);
      }, [&] {
        conn_set.remove(0, 6);
      });
  verify(conn_set, 32, { 6 });
}

TEST(AConnectivitySet, AddAndRemoveSamePartitionConcurrently5) {
  ConnectivitySets conn_set(1, 32);;
  executeConcurrent([&] {
        conn_set.add(0, 22);
      }, [&] {
        conn_set.remove(0, 22);
      });
  verify(conn_set, 32, { });
}

TEST(AConnectivitySet, AddAndRemoveSamePartitionConcurrently6) {
  ConnectivitySets conn_set(1, 32);;
  conn_set.add(0, 22);
  executeConcurrent([&] {
        conn_set.add(0, 22);
      }, [&] {
        conn_set.remove(0, 22);
      });
  verify(conn_set, 32, { 22 });
}

TEST(AConnectivitySet, AddAndRemoveSeveralSamePartitionsConcurrently) {
  ConnectivitySets conn_set(1, 32);;
  std::set<PartitionID> added = { 13, 14, 15, 16, 18, 22, 23 };
  add(conn_set, { 13, 14, 15, 17, 18, 22, 23, 24, 25, 26, 27 });
  executeConcurrent([&] {
        conn_set.add(0, 16);
        conn_set.remove(0, 13);
        conn_set.add(0, 24);
        conn_set.remove(0, 24);
        conn_set.remove(0, 26);
        conn_set.remove(0, 17);
      }, [&] {
        conn_set.add(0, 13);
        conn_set.remove(0, 16);
        conn_set.add(0, 16);
        conn_set.remove(0, 24);
        conn_set.remove(0, 25);
        conn_set.add(0, 26);
        conn_set.remove(0, 26);
        conn_set.remove(0, 27);
      });
  verify(conn_set, 32, added);
}

TEST(AConnectivitySet, IteratesThroughPartitionsAndSimultanouslyAddElements) {
  ConnectivitySets conn_set(1, 32);
  add(conn_set, { 1, 2, 6, 10, 15, 22, 24, 31 });

  std::atomic<size_t> cnt(0);

  executeConcurrent([&] {
        while (cnt < 1);
        // Add 0. Since the iterator is at 1 already, the change is not visible
        conn_set.add(0, 0);
        ++cnt;

        while (cnt < 3);
        // Add 12. Since the iteration currently waits at part 2, the change is visible
        conn_set.add(0, 12);
        ++cnt;

        while (cnt < 5);
        // Add 14. Since the iteration currently waits at part 15 the change is not visible
        conn_set.add(0, 14);
        ++cnt;

        while (cnt < 7);
        // Add 30. Change is visible, since the iteration currently waits at part 24
        conn_set.add(0, 30);
        ++cnt;
      }, [&] {
        std::vector<PartitionID> expected = { 1, 2, 6, 10, 12, 15, 22, 24, 30, 31 };
        size_t i = 0;
        for (const PartitionID& id : conn_set.connectivitySet(0)) {
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

          ASSERT_EQ(expected[i], id);
          i++;
        }
        ASSERT_EQ(expected.size(), i);
      });

  std::vector<PartitionID> expected = { 0, 1, 2, 6, 10, 12, 14, 15, 22, 24, 30, 31 };
  size_t i = 0;
  for (const PartitionID& id : conn_set.connectivitySet(0)) {
    ASSERT_EQ(expected[i++], id);
  }
}

TEST(AConnectivitySet, IteratesThroughPartitionsAndSimultanouslyRemoveElements) {
  ConnectivitySets conn_set(1, 32);
  add(conn_set, { 1, 2, 6, 10, 15, 22, 24, 31 });

  std::atomic<size_t> cnt(0);

  executeConcurrent([&] {
        while (cnt < 1);
        // Iteration currently waits at part 2. Removing part 1 will not be visible.
        conn_set.remove(0, 1);
        ++cnt;

        while (cnt < 3);
        // Remove 15. Since iteration waits at part 6, the change will be visible.
        conn_set.remove(0, 15);
        ++cnt;

        while (cnt < 5);
        // Remove 31. Iteration waits at part 22, so the change will be visible
        conn_set.remove(0, 31);
        ++cnt;
      }, [&] {
        std::vector<PartitionID> expected = { 1, 2, 6, 10, 22, 24 };
        size_t i = 0;
        for (const PartitionID& id : conn_set.connectivitySet(0)) {
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
            // Wait at part 22 until thread 1 removes 31
          }

          ASSERT_EQ(expected[i], id);
          i++;
        }
        ASSERT_EQ(expected.size(), i);
      });

  std::vector<PartitionID> expected = { 2, 6, 10, 22, 24 };
  size_t i = 0;
  for (const PartitionID& id : conn_set.connectivitySet(0)) {
    ASSERT_EQ(expected[i++], id);
  }
}

TEST(AConnectivitySet, IteratesThroughPartitionsAndSimultanouslyAddAndRemoveElements) {
  ConnectivitySets conn_set(1, 32);
  add(conn_set, { 1, 2, 6, 10, 15, 22, 24, 28, 31 });

  std::atomic<size_t> cnt(0);

  executeConcurrent([&] {
      while (cnt < 1);
      // Add 0. Since the iterator is at 1 already, the change is not visible
      conn_set.add(0, 0);
      ++cnt;

      while (cnt < 3);
      // Remove 15 and add 12. Iterator waits at part 6, so change will be visible
      conn_set.remove(0, 15);
      conn_set.add(0, 12);
      ++cnt;

      while (cnt < 5);
      // Iterator waits at part 22. --> adding 14 is not visible, removing 24 is
      conn_set.add(0, 14);
      conn_set.remove(0, 24);
      ++cnt;

      while (cnt < 7);
      // Add 29 and 30 and remove 31. Iterator waits at part 28, so the changes will be visible
      conn_set.add(0, 29);
      conn_set.add(0, 30);
      conn_set.remove(0, 31);
      ++cnt;
    }, [&] {
      std::vector<PartitionID> expected = { 1, 2, 6, 10, 12, 22, 28, 29, 30 };
      size_t i = 0;
      for (const PartitionID& id : conn_set.connectivitySet(0)) {
        if (i == 1) {
          ++cnt;
          while (cnt < 2);
          // Wait at part 2 until thread 1 adds 0
        }

        if (i == 2) {
          ++cnt;
          while (cnt < 4);
          // Wait at part 6 until thread 1 removes 15 and adds 16
        }

        if (i == 5) {
          ++cnt;
          while (cnt < 6);
          // Wait at part 22 until thread 1 adds 14 and removes 24
        }

        if (i == 6) {
          ++cnt;
          while (cnt < 8);
          // Wait at part 28 until thread 1 adds 29, 30 and removes 31
        }

        ASSERT_EQ(expected[i++], id);
      }
      ASSERT_EQ(expected.size(), i);
  });

  std::vector<PartitionID> expected = { 0, 1, 2, 6, 10, 12, 14, 22, 28, 29, 30 };
  size_t i = 0;
  for (const PartitionID& id : conn_set.connectivitySet(0)) {
    ASSERT_EQ(expected[i++], id);
  }
}

TEST(ADeltaConnectivitySet, IsEqualToConnectivitySetWhenInitialized) {
  ConnectivitySets con_set(1, 32);
  DeltaConnectivitySet<ConnectivitySets> delta_con_set(32);
  delta_con_set.setConnectivitySet(&con_set);
  std::set<PartitionID> added = { 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };
  add(con_set, added);

  verify(con_set, 32, added);
  verify(delta_con_set, 32, added);
}

TEST(ADeltaConnectivitySet, AddsSomeBlocksToConnectivitySet) {
  ConnectivitySets con_set(1, 32);
  DeltaConnectivitySet<ConnectivitySets> delta_con_set(32);
  delta_con_set.setConnectivitySet(&con_set);
  add(con_set, { 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 });
  add(delta_con_set, { 29, 30, 31 });

  verify(con_set, 32, { 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 });
  verify(delta_con_set, 32, { 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 29, 30, 31 });
}

TEST(ADeltaConnectivitySet, RemovesSomeBlocksFromConnectivitySet) {
  ConnectivitySets con_set(1, 32);
  DeltaConnectivitySet<ConnectivitySets> delta_con_set(32);
  delta_con_set.setConnectivitySet(&con_set);
  add(con_set, { 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 });
  remove(delta_con_set, { 11, 12, 13 });

  verify(con_set, 32, { 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 });
  verify(delta_con_set, 32, { 9, 10, 14, 15, 16, 17, 18 });
}

TEST(ADeltaConnectivitySet, AddAndRemovesSomeBlocksFromConnectivitySet) {
  ConnectivitySets con_set(1, 32);
  DeltaConnectivitySet<ConnectivitySets> delta_con_set(32);
  delta_con_set.setConnectivitySet(&con_set);
  add(con_set, { 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 });
  remove(delta_con_set, { 11, 12, 13 });
  add(delta_con_set, { 29, 30, 31 });

  verify(con_set, 32, { 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 });
  verify(delta_con_set, 32, { 9, 10, 14, 15, 16, 17, 18, 29, 30, 31 });
}

}  // namespace ds
}  // namespace mt_kahypar
