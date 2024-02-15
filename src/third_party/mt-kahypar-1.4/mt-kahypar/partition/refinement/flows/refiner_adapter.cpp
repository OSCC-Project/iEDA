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

#include "mt-kahypar/partition/refinement/flows/refiner_adapter.h"

#include "mt-kahypar/definitions.h"
#include "mt-kahypar/partition/factories.h"
#include "mt-kahypar/utils/cast.h"

namespace mt_kahypar {

namespace {
  #define NOW std::chrono::high_resolution_clock::now()
  #define RUNNING_TIME(X) std::chrono::duration<double>(NOW - X).count();
}

template<typename TypeTraits>
bool FlowRefinerAdapter<TypeTraits>::registerNewSearch(const SearchID search_id,
                                                       const PartitionedHypergraph& phg) {
  bool success = true;
  size_t refiner_idx = INVALID_REFINER_IDX;
  if ( _unused_refiners.try_pop(refiner_idx) ) {
    // Note, search id are usually consecutive starting from 0.
    // However, this function is not called in increasing search id order.
    _search_lock.lock();
    while ( static_cast<size_t>(search_id) >= _active_searches.size() ) {
      _active_searches.push_back(ActiveSearch { INVALID_REFINER_IDX, NOW, 0.0, false });
    }
    _search_lock.unlock();

    if ( !_refiner[refiner_idx] ) {
      // Lazy initialization of refiner
      _refiner[refiner_idx] = initializeRefiner();
    }

    _active_searches[search_id].refiner_idx = refiner_idx;
    _active_searches[search_id].start = NOW;
    mt_kahypar_partitioned_hypergraph_const_t partitioned_hg =
      utils::partitioned_hg_const_cast(phg);
    _refiner[refiner_idx]->initialize(partitioned_hg);
    _refiner[refiner_idx]->updateTimeLimit(timeLimit());
  } else {
    success = false;
  }
  return success;
}

template<typename TypeTraits>
MoveSequence FlowRefinerAdapter<TypeTraits>::refine(const SearchID search_id,
                                                    const PartitionedHypergraph& phg,
                                                    const Subhypergraph& sub_hg) {
  ASSERT(static_cast<size_t>(search_id) < _active_searches.size());
  ASSERT(_active_searches[search_id].refiner_idx != INVALID_REFINER_IDX);

  // Perform refinement
  mt_kahypar_partitioned_hypergraph_const_t partitioned_hg =
    utils::partitioned_hg_const_cast(phg);
  const size_t refiner_idx = _active_searches[search_id].refiner_idx;
  const size_t num_free_threads = _threads.acquireFreeThreads();
  _refiner[refiner_idx]->setNumThreadsForSearch(num_free_threads);
  MoveSequence moves = _refiner[refiner_idx]->refine(partitioned_hg, sub_hg, _active_searches[search_id].start);
  _threads.releaseThreads(num_free_threads);
  _active_searches[search_id].reaches_time_limit = moves.state == MoveSequenceState::TIME_LIMIT;
  return moves;
}

template<typename TypeTraits>
PartitionID FlowRefinerAdapter<TypeTraits>::maxNumberOfBlocks(const SearchID search_id) {
  ASSERT(static_cast<size_t>(search_id) < _active_searches.size());
  ASSERT(_active_searches[search_id].refiner_idx != INVALID_REFINER_IDX);
  const size_t refiner_idx = _active_searches[search_id].refiner_idx;
  return _refiner[refiner_idx]->maxNumberOfBlocksPerSearch();
}

template<typename TypeTraits>
void FlowRefinerAdapter<TypeTraits>::finalizeSearch(const SearchID search_id) {
  ASSERT(static_cast<size_t>(search_id) < _active_searches.size());
  const double running_time = RUNNING_TIME(_active_searches[search_id].start);
  _active_searches[search_id].running_time = running_time;

  //Update average running time
  _search_lock.lock();
  if ( !_active_searches[search_id].reaches_time_limit ) {
    _average_running_time = (running_time + _num_refinements *
      _average_running_time) / static_cast<double>(_num_refinements + 1);
    ++_num_refinements;
  }
  _search_lock.unlock();

  // Search position of refiner associated with the search id
  if ( shouldSetTimeLimit() ) {
    for ( size_t idx = 0; idx < _refiner.size(); ++idx ) {
      if ( _refiner[idx] ) {
        _refiner[idx]->updateTimeLimit(timeLimit());
      }
    }
  }

  ASSERT(_active_searches[search_id].refiner_idx != INVALID_REFINER_IDX);
  _unused_refiners.push(_active_searches[search_id].refiner_idx);
  _active_searches[search_id].refiner_idx = INVALID_REFINER_IDX;
}

template<typename TypeTraits>
void FlowRefinerAdapter<TypeTraits>::initialize(const size_t max_parallelism) {
  _num_parallel_refiners = max_parallelism;
  _threads.num_threads = _context.shared_memory.num_threads;
  _threads.num_parallel_refiners = max_parallelism;
  _threads.num_active_refiners = 0;
  _threads.num_used_threads = 0;

  _unused_refiners.clear();
  for ( size_t i = 0; i < numAvailableRefiner(); ++i ) {
    _unused_refiners.push(i);
  }
  _active_searches.clear();
  _num_refinements = 0;
  _average_running_time = 0.0;
}

template<typename TypeTraits>
std::unique_ptr<IFlowRefiner> FlowRefinerAdapter<TypeTraits>::initializeRefiner() {
  return FlowRefinementFactory::getInstance().createObject(
    _context.refinement.flows.algorithm, _num_hyperedges, _context);
}

INSTANTIATE_CLASS_WITH_TYPE_TRAITS(FlowRefinerAdapter)

}
