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

#include "gmock/gmock.h"
#include "tbb/task_group.h"

#include "mt-kahypar/parallel/memory_pool.h"
#include "mt-kahypar/parallel/hardware_topology.h"
#include "mt-kahypar/parallel/tbb_initializer.h"
#include "tests/parallel/topology_mock.h"

using ::testing::Test;

namespace mt_kahypar {
namespace parallel {

using TopoMock = mt_kahypar::parallel::TopologyMock<2>;
using HwTopology = mt_kahypar::parallel::HardwareTopology<TopoMock, parallel::topology_t, parallel::node_t>;
using TBB = mt_kahypar::parallel::TBBInitializer<HwTopology, false>;


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

static void setupMemoryPool(const bool optimize_allocations) {
  MemoryPool::instance().deactivate_minimum_allocation_size();
  MemoryPool::instance().register_memory_group("TEST_GROUP_1", 1);
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_1", "TEST_CHUNK_1", 5, sizeof(size_t));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_1", "TEST_CHUNK_2", 5, sizeof(int));
  MemoryPool::instance().register_memory_group("TEST_GROUP_2", 2);
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_2", "TEST_CHUNK_1", 5, sizeof(double));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_2", "TEST_CHUNK_2", 5, sizeof(size_t));
  MemoryPool::instance().allocate_memory_chunks(optimize_allocations);
}

TEST(AMemoryPool, AllocatesMemory) {
  setupMemoryPool(false);

  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_1"));
  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_2"));
  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_2", "TEST_CHUNK_1"));
  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_2", "TEST_CHUNK_2"));

  MemoryPool::instance().free_memory_chunks();
}

TEST(AMemoryPool, CanHandleARequest) {
  setupMemoryPool(false);

  ASSERT_EQ(MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_1"),
            MemoryPool::instance().request_mem_chunk("TEST_GROUP_1", "TEST_CHUNK_1", 5, sizeof(size_t)));

  MemoryPool::instance().free_memory_chunks();
}

TEST(AMemoryPool, CanHandleSeveralRequests) {
  setupMemoryPool(false);

  ASSERT_EQ(MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_1"),
            MemoryPool::instance().request_mem_chunk("TEST_GROUP_1", "TEST_CHUNK_1", 5, sizeof(size_t)));
  ASSERT_EQ(MemoryPool::instance().mem_chunk("TEST_GROUP_2", "TEST_CHUNK_2"),
            MemoryPool::instance().request_mem_chunk("TEST_GROUP_2", "TEST_CHUNK_2", 5, sizeof(size_t)));

  MemoryPool::instance().free_memory_chunks();
}

TEST(AMemoryPool, ReturnsNullptrIfMemoryChunkAlreadyRequested) {
  setupMemoryPool(false);

  ASSERT_EQ(MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_1"),
            MemoryPool::instance().request_mem_chunk("TEST_GROUP_1", "TEST_CHUNK_1", 5, sizeof(size_t)));
  ASSERT_EQ(nullptr,
            MemoryPool::instance().request_mem_chunk("TEST_GROUP_1", "TEST_CHUNK_1", 5, sizeof(size_t)));

  MemoryPool::instance().free_memory_chunks();
}


TEST(AMemoryPool, ReturnsNullptrOnOverallocation) {
  setupMemoryPool(false);

  ASSERT_EQ(nullptr,
            MemoryPool::instance().request_mem_chunk("TEST_GROUP_1", "TEST_CHUNK_1", 10, sizeof(size_t)));

  MemoryPool::instance().free_memory_chunks();
}

TEST(AMemoryPool, ReturnsNullptrIfMemoryChunkIsNotAvailable) {
  setupMemoryPool(false);

  ASSERT_EQ(nullptr,
            MemoryPool::instance().request_mem_chunk("TEST_GROUP_1", "TEST_CHUNK_3", 5, sizeof(size_t)));

  MemoryPool::instance().free_memory_chunks();
}

TEST(AMemoryPool, RequestMemoryAfterRelease) {
  setupMemoryPool(false);

  ASSERT_EQ(MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_1"),
            MemoryPool::instance().request_mem_chunk("TEST_GROUP_1", "TEST_CHUNK_1", 5, sizeof(size_t)));
  MemoryPool::instance().release_mem_chunk("TEST_GROUP_1", "TEST_CHUNK_1");
  ASSERT_EQ(MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_1"),
            MemoryPool::instance().request_mem_chunk("TEST_GROUP_1", "TEST_CHUNK_1", 5, sizeof(size_t)));

  MemoryPool::instance().free_memory_chunks();
}

TEST(AMemoryPool, OnlyOneRequestSucceedsOnConcurrentAccess) {
  setupMemoryPool(false);

  char* chunk_1 = nullptr;
  char* chunk_2 = nullptr;
  executeConcurrent([&] {
    chunk_1 = MemoryPool::instance().request_mem_chunk("TEST_GROUP_1", "TEST_CHUNK_1", 5, sizeof(size_t));
  }, [&] {
    chunk_2 = MemoryPool::instance().request_mem_chunk("TEST_GROUP_1", "TEST_CHUNK_1", 5, sizeof(size_t));
  });

  char* expected_chunk = MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_1");
  if ( chunk_1 ) {
    ASSERT_EQ(nullptr, chunk_2);
    ASSERT_EQ(expected_chunk, chunk_1);
  } else {
    ASSERT_EQ(nullptr, chunk_1);
    ASSERT_EQ(expected_chunk, chunk_2);
  }

  MemoryPool::instance().free_memory_chunks();
}

TEST(AMemoryPool, AllocatesMemoryWithOptimizedAllocationUsage) {
  MemoryPool::instance().register_memory_group("TEST_GROUP_1", 1);
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_1", "TEST_CHUNK_1", 10, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_1", "TEST_CHUNK_2", 3, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_1", "TEST_CHUNK_3", 5, sizeof(char));
  MemoryPool::instance().register_memory_group("TEST_GROUP_2", 2);
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_2", "TEST_CHUNK_1", 4, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_2", "TEST_CHUNK_2", 14, sizeof(char));
  MemoryPool::instance().register_memory_group("TEST_GROUP_3", 3);
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_3", "TEST_CHUNK_1", 10, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_3", "TEST_CHUNK_2", 10, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_3", "TEST_CHUNK_3", 4, sizeof(char));
  MemoryPool::instance().allocate_memory_chunks(true);

  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_1"));
  ASSERT_EQ(14, MemoryPool::instance().size_in_bytes("TEST_GROUP_1", "TEST_CHUNK_1"));
  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_2"));
  ASSERT_EQ(4, MemoryPool::instance().size_in_bytes("TEST_GROUP_1", "TEST_CHUNK_2"));
  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_3"));
  ASSERT_EQ(10, MemoryPool::instance().size_in_bytes("TEST_GROUP_1", "TEST_CHUNK_3"));

  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_2", "TEST_CHUNK_1"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_2", "TEST_CHUNK_1"));
  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_2", "TEST_CHUNK_2"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_2", "TEST_CHUNK_2"));

  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_3", "TEST_CHUNK_1"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_3", "TEST_CHUNK_1"));
  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_3", "TEST_CHUNK_2"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_3", "TEST_CHUNK_2"));
  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_3", "TEST_CHUNK_3"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_3", "TEST_CHUNK_3"));

  MemoryPool::instance().free_memory_chunks();
}

TEST(AMemoryPool, MemoryIsTransferedToNextGroupIfGroupIsReleased1) {
  MemoryPool::instance().register_memory_group("TEST_GROUP_1", 1);
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_1", "TEST_CHUNK_1", 10, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_1", "TEST_CHUNK_2", 3, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_1", "TEST_CHUNK_3", 5, sizeof(char));
  MemoryPool::instance().register_memory_group("TEST_GROUP_2", 2);
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_2", "TEST_CHUNK_1", 4, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_2", "TEST_CHUNK_2", 14, sizeof(char));
  MemoryPool::instance().register_memory_group("TEST_GROUP_3", 3);
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_3", "TEST_CHUNK_1", 10, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_3", "TEST_CHUNK_2", 10, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_3", "TEST_CHUNK_3", 4, sizeof(char));
  MemoryPool::instance().allocate_memory_chunks(true);

  MemoryPool::instance().release_mem_group("TEST_GROUP_1");

  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_1"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_1", "TEST_CHUNK_1"));
  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_2"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_1", "TEST_CHUNK_2"));
  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_3"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_1", "TEST_CHUNK_3"));

  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_2", "TEST_CHUNK_1"));
  ASSERT_EQ(10, MemoryPool::instance().size_in_bytes("TEST_GROUP_2", "TEST_CHUNK_1"));
  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_2", "TEST_CHUNK_2"));
  ASSERT_EQ(14, MemoryPool::instance().size_in_bytes("TEST_GROUP_2", "TEST_CHUNK_2"));

  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_3", "TEST_CHUNK_1"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_3", "TEST_CHUNK_1"));
  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_3", "TEST_CHUNK_2"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_3", "TEST_CHUNK_2"));
  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_3", "TEST_CHUNK_3"));
  ASSERT_EQ(4, MemoryPool::instance().size_in_bytes("TEST_GROUP_3", "TEST_CHUNK_3"));

  MemoryPool::instance().free_memory_chunks();
}

TEST(AMemoryPool, MemoryIsTransferedToNextGroupIfGroupIsReleased2) {
  MemoryPool::instance().register_memory_group("TEST_GROUP_1", 1);
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_1", "TEST_CHUNK_1", 10, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_1", "TEST_CHUNK_2", 3, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_1", "TEST_CHUNK_3", 5, sizeof(char));
  MemoryPool::instance().register_memory_group("TEST_GROUP_2", 2);
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_2", "TEST_CHUNK_1", 4, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_2", "TEST_CHUNK_2", 14, sizeof(char));
  MemoryPool::instance().register_memory_group("TEST_GROUP_3", 3);
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_3", "TEST_CHUNK_1", 10, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_3", "TEST_CHUNK_2", 10, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_3", "TEST_CHUNK_3", 4, sizeof(char));
  MemoryPool::instance().allocate_memory_chunks(true);

  MemoryPool::instance().release_mem_group("TEST_GROUP_1");
  MemoryPool::instance().release_mem_group("TEST_GROUP_2");

  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_1"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_1", "TEST_CHUNK_1"));
  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_2"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_1", "TEST_CHUNK_2"));
  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_3"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_1", "TEST_CHUNK_3"));

  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_2", "TEST_CHUNK_1"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_2", "TEST_CHUNK_1"));
  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_2", "TEST_CHUNK_2"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_2", "TEST_CHUNK_2"));

  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_3", "TEST_CHUNK_1"));
  ASSERT_EQ(10, MemoryPool::instance().size_in_bytes("TEST_GROUP_3", "TEST_CHUNK_1"));
  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_3", "TEST_CHUNK_2"));
  ASSERT_EQ(14, MemoryPool::instance().size_in_bytes("TEST_GROUP_3", "TEST_CHUNK_2"));
  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_3", "TEST_CHUNK_3"));
  ASSERT_EQ(4, MemoryPool::instance().size_in_bytes("TEST_GROUP_3", "TEST_CHUNK_3"));

  MemoryPool::instance().free_memory_chunks();
}

TEST(AMemoryPool, RequestsAnUnsedChunk1) {
  MemoryPool::instance().deactivate_round_robin_assignment();
  MemoryPool::instance().register_memory_group("TEST_GROUP_1", 1);
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_1", "TEST_CHUNK_1", 10, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_1", "TEST_CHUNK_2", 3, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_1", "TEST_CHUNK_3", 5, sizeof(char));
  MemoryPool::instance().register_memory_group("TEST_GROUP_2", 2);
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_2", "TEST_CHUNK_1", 4, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_2", "TEST_CHUNK_2", 14, sizeof(char));
  MemoryPool::instance().register_memory_group("TEST_GROUP_3", 3);
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_3", "TEST_CHUNK_1", 10, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_3", "TEST_CHUNK_2", 10, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_3", "TEST_CHUNK_3", 4, sizeof(char));
  MemoryPool::instance().allocate_memory_chunks(true);

  ASSERT_NE(nullptr, MemoryPool::instance().request_unused_mem_chunk(5, sizeof(char), false));
  ASSERT_NE(nullptr, MemoryPool::instance().request_unused_mem_chunk(4, sizeof(char), false));
  ASSERT_NE(nullptr, MemoryPool::instance().request_unused_mem_chunk(1, sizeof(char), false));
  ASSERT_EQ(nullptr, MemoryPool::instance().request_unused_mem_chunk(1, sizeof(char), false));

  MemoryPool::instance().free_memory_chunks();
}

TEST(AMemoryPool, RequestsAnUnsedChunk2) {
  MemoryPool::instance().deactivate_round_robin_assignment();
  MemoryPool::instance().register_memory_group("TEST_GROUP_1", 1);
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_1", "TEST_CHUNK_1", 10, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_1", "TEST_CHUNK_2", 3, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_1", "TEST_CHUNK_3", 5, sizeof(char));
  MemoryPool::instance().register_memory_group("TEST_GROUP_2", 2);
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_2", "TEST_CHUNK_1", 4, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_2", "TEST_CHUNK_2", 14, sizeof(char));
  MemoryPool::instance().register_memory_group("TEST_GROUP_3", 3);
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_3", "TEST_CHUNK_1", 10, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_3", "TEST_CHUNK_2", 10, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_3", "TEST_CHUNK_3", 4, sizeof(char));
  MemoryPool::instance().allocate_memory_chunks(true);

  MemoryPool::instance().release_mem_group("TEST_GROUP_1");

  ASSERT_NE(nullptr, MemoryPool::instance().request_unused_mem_chunk(6, sizeof(char), false));
  ASSERT_EQ(nullptr, MemoryPool::instance().request_unused_mem_chunk(1, sizeof(char), false));

  MemoryPool::instance().free_memory_chunks();
}

TEST(AMemoryPool, RequestsAnUnsedChunk3) {
  MemoryPool::instance().deactivate_round_robin_assignment();
  MemoryPool::instance().register_memory_group("TEST_GROUP_1", 1);
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_1", "TEST_CHUNK_1", 10, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_1", "TEST_CHUNK_2", 3, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_1", "TEST_CHUNK_3", 5, sizeof(char));
  MemoryPool::instance().register_memory_group("TEST_GROUP_2", 2);
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_2", "TEST_CHUNK_1", 4, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_2", "TEST_CHUNK_2", 14, sizeof(char));
  MemoryPool::instance().allocate_memory_chunks(true);

  ASSERT_NE(nullptr, MemoryPool::instance().request_unused_mem_chunk(4, sizeof(char), false));
  ASSERT_EQ(nullptr, MemoryPool::instance().request_unused_mem_chunk(1, sizeof(char), false));

  MemoryPool::instance().release_mem_group("TEST_GROUP_1");

  ASSERT_NE(nullptr, MemoryPool::instance().request_unused_mem_chunk(3, sizeof(char), false));
  ASSERT_NE(nullptr, MemoryPool::instance().request_unused_mem_chunk(1, sizeof(char), false));
  ASSERT_EQ(nullptr, MemoryPool::instance().request_unused_mem_chunk(1, sizeof(char), false));

  MemoryPool::instance().free_memory_chunks();
}

TEST(AMemoryPool, MemoryPoolIsCorrectlyResetted) {
  MemoryPool::instance().register_memory_group("TEST_GROUP_1", 1);
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_1", "TEST_CHUNK_1", 10, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_1", "TEST_CHUNK_2", 3, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_1", "TEST_CHUNK_3", 5, sizeof(char));
  MemoryPool::instance().register_memory_group("TEST_GROUP_2", 2);
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_2", "TEST_CHUNK_1", 4, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_2", "TEST_CHUNK_2", 14, sizeof(char));
  MemoryPool::instance().register_memory_group("TEST_GROUP_3", 3);
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_3", "TEST_CHUNK_1", 10, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_3", "TEST_CHUNK_2", 10, sizeof(char));
  MemoryPool::instance().register_memory_chunk("TEST_GROUP_3", "TEST_CHUNK_3", 4, sizeof(char));
  MemoryPool::instance().allocate_memory_chunks(true);

  MemoryPool::instance().release_mem_group("TEST_GROUP_1");
  MemoryPool::instance().release_mem_group("TEST_GROUP_2");

  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_1"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_1", "TEST_CHUNK_1"));
  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_2"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_1", "TEST_CHUNK_2"));
  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_3"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_1", "TEST_CHUNK_3"));

  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_2", "TEST_CHUNK_1"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_2", "TEST_CHUNK_1"));
  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_2", "TEST_CHUNK_2"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_2", "TEST_CHUNK_2"));

  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_3", "TEST_CHUNK_1"));
  ASSERT_EQ(10, MemoryPool::instance().size_in_bytes("TEST_GROUP_3", "TEST_CHUNK_1"));
  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_3", "TEST_CHUNK_2"));
  ASSERT_EQ(14, MemoryPool::instance().size_in_bytes("TEST_GROUP_3", "TEST_CHUNK_2"));
  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_3", "TEST_CHUNK_3"));
  ASSERT_EQ(4, MemoryPool::instance().size_in_bytes("TEST_GROUP_3", "TEST_CHUNK_3"));

  MemoryPool::instance().reset();

  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_1"));
  ASSERT_EQ(14, MemoryPool::instance().size_in_bytes("TEST_GROUP_1", "TEST_CHUNK_1"));
  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_2"));
  ASSERT_EQ(4, MemoryPool::instance().size_in_bytes("TEST_GROUP_1", "TEST_CHUNK_2"));
  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_3"));
  ASSERT_EQ(10, MemoryPool::instance().size_in_bytes("TEST_GROUP_1", "TEST_CHUNK_3"));

  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_2", "TEST_CHUNK_1"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_2", "TEST_CHUNK_1"));
  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_2", "TEST_CHUNK_2"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_2", "TEST_CHUNK_2"));

  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_3", "TEST_CHUNK_1"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_3", "TEST_CHUNK_1"));
  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_3", "TEST_CHUNK_2"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_3", "TEST_CHUNK_2"));
  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_3", "TEST_CHUNK_3"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_3", "TEST_CHUNK_3"));

  MemoryPool::instance().release_mem_group("TEST_GROUP_1");

  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_1"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_1", "TEST_CHUNK_1"));
  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_2"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_1", "TEST_CHUNK_2"));
  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_3"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_1", "TEST_CHUNK_3"));

  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_2", "TEST_CHUNK_1"));
  ASSERT_EQ(10, MemoryPool::instance().size_in_bytes("TEST_GROUP_2", "TEST_CHUNK_1"));
  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_2", "TEST_CHUNK_2"));
  ASSERT_EQ(14, MemoryPool::instance().size_in_bytes("TEST_GROUP_2", "TEST_CHUNK_2"));

  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_3", "TEST_CHUNK_1"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_3", "TEST_CHUNK_1"));
  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_3", "TEST_CHUNK_2"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_3", "TEST_CHUNK_2"));
  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_3", "TEST_CHUNK_3"));
  ASSERT_EQ(4, MemoryPool::instance().size_in_bytes("TEST_GROUP_3", "TEST_CHUNK_3"));

  MemoryPool::instance().release_mem_group("TEST_GROUP_2");

  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_1"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_1", "TEST_CHUNK_1"));
  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_2"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_1", "TEST_CHUNK_2"));
  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_1", "TEST_CHUNK_3"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_1", "TEST_CHUNK_3"));

  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_2", "TEST_CHUNK_1"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_2", "TEST_CHUNK_1"));
  ASSERT_EQ(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_2", "TEST_CHUNK_2"));
  ASSERT_EQ(0, MemoryPool::instance().size_in_bytes("TEST_GROUP_2", "TEST_CHUNK_2"));

  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_3", "TEST_CHUNK_1"));
  ASSERT_EQ(10, MemoryPool::instance().size_in_bytes("TEST_GROUP_3", "TEST_CHUNK_1"));
  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_3", "TEST_CHUNK_2"));
  ASSERT_EQ(14, MemoryPool::instance().size_in_bytes("TEST_GROUP_3", "TEST_CHUNK_2"));
  ASSERT_NE(nullptr, MemoryPool::instance().mem_chunk("TEST_GROUP_3", "TEST_CHUNK_3"));
  ASSERT_EQ(4, MemoryPool::instance().size_in_bytes("TEST_GROUP_3", "TEST_CHUNK_3"));

  MemoryPool::instance().free_memory_chunks();
}


}  // namespace parallel
}  // namespace mt_kahypar
