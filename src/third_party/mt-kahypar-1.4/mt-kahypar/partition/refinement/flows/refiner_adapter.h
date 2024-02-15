/*******************************************************************************
 * MIT License
 *
 * This file is part of Mt-KaHyPar.
 *
 * Copyright (C) 2021 Tobias Heuer <tobias.heuer@kit.edu>
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

#include "tbb/concurrent_vector.h"
#include "tbb/concurrent_queue.h"

#include "mt-kahypar/partition/context.h"
#include "mt-kahypar/partition/refinement/flows/i_flow_refiner.h"
#include "mt-kahypar/parallel/stl/scalable_vector.h"
#include "mt-kahypar/parallel/atomic_wrapper.h"

namespace mt_kahypar {

template<typename TypeTraits>
class FlowRefinerAdapter {

  static constexpr bool debug = false;
  static constexpr bool enable_heavy_assert = false;
  static constexpr size_t INVALID_REFINER_IDX = std::numeric_limits<size_t>::max();

  using PartitionedHypergraph = typename TypeTraits::PartitionedHypergraph;

  struct ActiveSearch {
    size_t refiner_idx;
    HighResClockTimepoint start;
    double running_time;
    bool reaches_time_limit;
  };

  struct ThreadOrganizer {
    ThreadOrganizer() :
      lock(),
      num_threads(0),
      num_used_threads(0),
      num_parallel_refiners(0),
      num_active_refiners(0) { }

    size_t acquireFreeThreads() {
      lock.lock();
      const size_t num_threads_per_search =
        std::max(UL(1), static_cast<size_t>(std::ceil(
          static_cast<double>(num_threads - num_used_threads) /
          ( num_parallel_refiners - num_active_refiners ) )));
      const size_t num_free_threads = std::min(
        num_threads_per_search, num_threads - num_used_threads);
      ++num_active_refiners;
      num_used_threads += num_free_threads;
      lock.unlock();
      return num_free_threads;
    }

    void releaseThreads(const size_t num_threads) {
      lock.lock();
      ASSERT(num_threads <= num_used_threads);
      ASSERT(num_active_refiners);
      num_used_threads -= num_threads;
      --num_active_refiners;
      lock.unlock();
    }

    void terminateRefiner() {
      lock.lock();
      --num_parallel_refiners;
      lock.unlock();
    }

    SpinLock lock;
    size_t num_threads;
    size_t num_used_threads;
    size_t num_parallel_refiners;
    size_t num_active_refiners;
  };

public:
  explicit FlowRefinerAdapter(const HyperedgeID num_hyperedges,
                              const Context& context) :
    _num_hyperedges(num_hyperedges),
    _context(context),
    _unused_refiners(),
    _refiner(),
    _search_lock(),
    _active_searches(),
    _threads(),
    _num_parallel_refiners(0),
    _num_refinements(0),
    _average_running_time(0.0) {
    for ( size_t i = 0; i < _context.shared_memory.num_threads; ++i ) {
      _refiner.emplace_back(nullptr);
    }
  }

  FlowRefinerAdapter(const FlowRefinerAdapter&) = delete;
  FlowRefinerAdapter(FlowRefinerAdapter&&) = delete;

  FlowRefinerAdapter & operator= (const FlowRefinerAdapter &) = delete;
  FlowRefinerAdapter & operator= (FlowRefinerAdapter &&) = delete;

  void initialize(const size_t max_parallelism);

  // ! Associates a refiner with a search id.
  // ! Returns true, if there is an idle refiner left.
  bool registerNewSearch(const SearchID search_id,
                         const PartitionedHypergraph& phg);

  MoveSequence refine(const SearchID search_id,
                      const PartitionedHypergraph& phg,
                      const Subhypergraph& sub_hg);

  // ! Returns the maximum number of blocks which is allowed to be
  // ! contained in the problem of the refiner associated with
  // ! corresponding search id
  PartitionID maxNumberOfBlocks(const SearchID search_id);

  // ! Makes the refiner associated with the corresponding search id
  // ! available again
  void finalizeSearch(const SearchID search_id);

  void terminateRefiner() {
    _threads.terminateRefiner();
  }

  size_t numAvailableRefiner() const {
    return _num_parallel_refiners;
  }

  double runningTime(const SearchID search_id) const {
    ASSERT(static_cast<size_t>(search_id) < _active_searches.size());
    return _active_searches[search_id].running_time;
  }

  double timeLimit() const {
    return shouldSetTimeLimit() ?
      std::max(_context.refinement.flows.time_limit_factor *
        _average_running_time, 0.1) : std::numeric_limits<double>::max();
  }

  // ! Only for testing
  size_t numUsedThreads() const {
    return _threads.num_used_threads;
  }

private:
  std::unique_ptr<IFlowRefiner> initializeRefiner();

  bool shouldSetTimeLimit() const {
    return _num_refinements > static_cast<size_t>(_context.partition.k) &&
      _context.refinement.flows.time_limit_factor > 1.0;
  }

  const HyperedgeID _num_hyperedges;
  const Context& _context;

  // ! Indices of unused refiners
  tbb::concurrent_queue<size_t> _unused_refiners;
  // ! Available refiners
  vec<std::unique_ptr<IFlowRefiner>> _refiner;
  // ! Mapping from search id to refiner
  SpinLock _search_lock;
  tbb::concurrent_vector<ActiveSearch> _active_searches;

  ThreadOrganizer _threads;
  size_t _num_parallel_refiners;

  size_t _num_refinements;
  double _average_running_time;

};

}  // namespace kahypar
