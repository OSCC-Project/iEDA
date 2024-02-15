/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2019 Lars Gottesbüren <lars.gottesbueren@kit.edu>
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

#pragma once

#include <atomic>
#include <type_traits>
#include <limits>
#include <cassert>

#include "tbb/enumerable_thread_specific.h"

#include "mt-kahypar/parallel/stl/scalable_vector.h"
#include "mt-kahypar/datastructures/array.h"
#include "mt-kahypar/datastructures/static_bitset.h"
#include "mt-kahypar/parallel/atomic_wrapper.h"
#include "mt-kahypar/utils/bit_ops.h"
#include "mt-kahypar/utils/memory_tree.h"
#include "mt-kahypar/utils/range.h"

#include "mt-kahypar/macros.h"
#include "hypergraph_common.h"

namespace mt_kahypar {
namespace ds {

/**
 *      The connectivity set of a hyperedge is the set of parts of the partition, that it has pins in.
 *      For each hyperedge we maintain its connectivity set in a packed format (std::vector<uint64_t>)
 *      and implement the necessary bitset functionality ourselves, i.e. add, remove, contains, clear, iteration.
 *      That is because we want atomic updates to support safe parallel modification of the partition.
 *      Adding/removing a part are both implemented as a toggle of the corresponding bit, in case an add and
 *      a remove operation are interweaved. However, this means the user must ensure that no two threads simultaneously try
 *      to add a part. One correct way is to keep an atomic count of pins for each hyperedge and part. Then only the thread
 *      raising the counter from zero to one performs the add, and only the thread decreasing the counter from one to zero
 *      performs the removal.
 */
class ConnectivitySets {
public:

  static constexpr bool debug = false;

  static constexpr int BITS_PER_BLOCK = StaticBitset::BITS_PER_BLOCK;
  using UnsafeBlock = StaticBitset::Block;
  using Iterator = typename StaticBitset::const_iterator;

  ConnectivitySets() :
    _k(0),
    _num_hyperedges(0),
    _num_blocks_per_hyperedge(0),
    _bits(),
    _deep_copy_bitset(),
    _shallow_copy_bitset() { }

  ConnectivitySets(const HyperedgeID num_hyperedges,
                   const PartitionID k,
                   const bool assign_parallel = true) :
    _k(k),
    _num_hyperedges(num_hyperedges),
    _num_blocks_per_hyperedge(k / BITS_PER_BLOCK + (k % BITS_PER_BLOCK != 0)),
    _bits(),
    _deep_copy_bitset(),
    _shallow_copy_bitset() {
      if ( num_hyperedges > 0 ) {
        _bits.resize("Refinement", "connectivity_set",
          static_cast<size_t>(num_hyperedges) * _num_blocks_per_hyperedge, true, assign_parallel);
      }
    }

  IteratorRange<Iterator> connectivitySet(const HyperedgeID he) const {
    return IteratorRange<Iterator>(
      Iterator(_num_blocks_per_hyperedge, _bits.data() + he * _num_blocks_per_hyperedge, -1),
      Iterator(_num_blocks_per_hyperedge, _bits.data() +
        he * _num_blocks_per_hyperedge, _num_blocks_per_hyperedge * BITS_PER_BLOCK));
  }

  void add(const HyperedgeID he, const PartitionID p) {
    toggle(he, p);
  }

  void remove(const HyperedgeID he, const PartitionID p) {
    toggle(he, p);
  }

  bool contains(const HyperedgeID he, const PartitionID p) const {
    const size_t div = p / BITS_PER_BLOCK;
    const size_t rem = p % BITS_PER_BLOCK;
    const size_t pos = static_cast<size_t>(he) * _num_blocks_per_hyperedge + div;
    return __atomic_load_n(&_bits[pos], __ATOMIC_RELAXED) & (UnsafeBlock(1) << rem);
  }

  // not threadsafe
  void clear(const HyperedgeID he) {
    const size_t start = static_cast<size_t>(he) * _num_blocks_per_hyperedge;
    const size_t end = ( static_cast<size_t>(he) + 1 ) * _num_blocks_per_hyperedge;
    for (size_t i = start; i < end; ++i) {
      __atomic_store_n(&_bits[i], 0, __ATOMIC_RELAXED);
    }
  }

  void reset(const bool reset_parallel = false) {
    if ( reset_parallel ) {
      tbb::parallel_for(UL(0), _bits.size(), [&](const size_t i) {
        __atomic_store_n(&_bits[i], 0, __ATOMIC_RELAXED);
      });
    } else {
      for (size_t i = 0; i < _bits.size(); ++i) {
        __atomic_store_n(&_bits[i], 0, __ATOMIC_RELAXED);
      }
    }
  }

  PartitionID connectivity(const HyperedgeID he) const {
    PartitionID conn = 0;
    const size_t start = static_cast<size_t>(he) * _num_blocks_per_hyperedge;
    const size_t end = ( static_cast<size_t>(he) + 1 ) * _num_blocks_per_hyperedge;
    for (size_t i = start; i < end; ++i) {
      conn += utils::popcount_64(__atomic_load_n(&_bits[i], __ATOMIC_RELAXED));
    }
    return conn;
  }

  // Creates a shallow copy of the connectivity set of hyperedge he
  StaticBitset& shallowCopy(const HyperedgeID he) const {
    StaticBitset& shallow_copy = _shallow_copy_bitset.local();
    shallow_copy.set(_num_blocks_per_hyperedge,
      &_bits[UL(he) * _num_blocks_per_hyperedge]);
    return shallow_copy;
  }

  // Creates a deep copy of the connectivity set of hyperedge he
  Bitset& deepCopy(const HyperedgeID he) const {
    Bitset& deep_copy = _deep_copy_bitset.local();
    deep_copy.copy(_num_blocks_per_hyperedge,
      &_bits[UL(he) * _num_blocks_per_hyperedge]);
    return deep_copy;
  }

  void freeInternalData() {
    parallel::free(_bits);
  }

  void memoryConsumption(utils::MemoryTreeNode* parent) const {
    ASSERT(parent);
    parent->addChild("Connectivity Bit Vector", sizeof(UnsafeBlock) * _bits.size());
  }

  static size_t num_elements(const HyperedgeID num_hyperedges,
                             const PartitionID k) {
    return static_cast<size_t>(num_hyperedges) * (k / BITS_PER_BLOCK + (k % BITS_PER_BLOCK != 0));
  }

private:
	void toggle(const HyperedgeID he, const PartitionID p) {
	  assert(p < _k);
	  assert(he < _num_hyperedges);
    const size_t div = p / BITS_PER_BLOCK, rem = p % BITS_PER_BLOCK;
    const size_t idx = static_cast<size_t>(he) * _num_blocks_per_hyperedge + div;
    assert(idx < _bits.size());
    __atomic_xor_fetch(&_bits[idx], UnsafeBlock(1) << rem, __ATOMIC_RELAXED);
	}

	PartitionID _k;
	HyperedgeID _num_hyperedges;
	PartitionID _num_blocks_per_hyperedge;
	Array<UnsafeBlock> _bits;

  // Bitsets to create shallow and deep copies of the connectivity set
  mutable tbb::enumerable_thread_specific<Bitset> _deep_copy_bitset;
  mutable tbb::enumerable_thread_specific<StaticBitset> _shallow_copy_bitset;
};



}  // namespace ds
}  // namespace mt_kahypar
