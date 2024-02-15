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

#include <unordered_map>
#include <thread>
#include <mutex>
#include <sstream>

#ifdef __linux__
#include <sched.h>
#elif _WIN32
#include <winbase.h>
#endif

#undef __TBB_ARENA_OBSERVER
#define __TBB_ARENA_OBSERVER true
#include "tbb/task_scheduler_observer.h"
#undef __TBB_ARENA_OBSERVER

#include "mt-kahypar/macros.h"
#include "mt-kahypar/utils/randomize.h"
#include "mt-kahypar/utils/exception.h"

namespace mt_kahypar {
namespace parallel {

template <typename HwTopology>
class ThreadPinningObserver : public tbb::task_scheduler_observer {
  using Base = tbb::task_scheduler_observer;

  static constexpr bool debug = false;

 public:

  // Observer is pinned to a task arena and is responsible for pinning
  // joining threads to the corresponding numa node.
  explicit ThreadPinningObserver(tbb::task_arena& arena,
                                 const int numa_node,
                                 const std::vector<int>& cpus) :
    Base(arena),
    _num_cpus(HwTopology::instance().num_cpus()),
    _numa_node(numa_node),
    _is_global_thread_pool(false),
    _cpus(cpus),
    _mutex(),
    _cpu_before() {
    ASSERT(cpus.size() > 0 && cpus.size() == static_cast<size_t>(arena.max_concurrency()),
      V(cpus.size()) << V(arena.max_concurrency()));

    // In case we have a task arena of size one, it can sometimes happen that master threads
    // joins the arena regardless if there are an other worker thread already inside.
    // In order to have enough CPU available for this special case, we request a backup
    // CPU from the hardware topology here.
    if ( _cpus.size() == 1 ) {
      _cpus.push_back(HwTopology::instance().get_backup_cpu(_numa_node, _cpus[0]));
    }

    #ifdef KAHYPAR_ENABLE_THREAD_PINNING
    #ifndef MT_KAHYPAR_LIBRARY_MODE
    observe(true); // Enable thread pinning
    #endif
    #endif
  }

  // Observer is pinned to the global task arena and is reponsible for
  // pinning threads to unique CPU id.
  explicit ThreadPinningObserver(const std::vector<int>& cpus) :
    Base(),
    _num_cpus(HwTopology::instance().num_cpus()),
    _numa_node(-1),
    _is_global_thread_pool(true),
    _cpus(cpus),
    _mutex(),
    _cpu_before() {
    if ( _cpus.size() == 1 ) {
      _cpus.push_back(HwTopology::instance().get_backup_cpu(0, _cpus[0]));
    }

    #ifdef KAHYPAR_ENABLE_THREAD_PINNING
    #ifndef MT_KAHYPAR_LIBRARY_MODE
    observe(true); // Enable thread pinning
    #endif
    #endif
  }


  ThreadPinningObserver(const ThreadPinningObserver&) = delete;
  ThreadPinningObserver & operator= (const ThreadPinningObserver &) = delete;

  ThreadPinningObserver(ThreadPinningObserver&& other) :
    _num_cpus(other._num_cpus),
    _numa_node(other._numa_node),
    _is_global_thread_pool(other._is_global_thread_pool),
    _cpus(other._cpus),
    _mutex(),
    _cpu_before(std::move(other._cpu_before)) { }

  ThreadPinningObserver & operator= (ThreadPinningObserver &&) = delete;

  void on_scheduler_entry(bool) override {
    const int slot = tbb::this_task_arena::current_thread_index();
    ASSERT(static_cast<size_t>(slot) < _cpus.size(), V(slot) << V(_cpus.size()));

    if ( slot >= static_cast<int>(_cpus.size()) ) {
      std::stringstream thread_id;
      thread_id << std::this_thread::get_id();
      throw SystemException(
        "Thread " + thread_id.str() + " entered the global task arena "
        "in a slot that should not exist (Slot = " + std::to_string(slot) + ", Max Slots = " + std::to_string(_cpus.size()) +
        ", slots are 0-indexed). This bug only occurs in older versions of TBB. "
        "We recommend upgrading TBB to the newest version.");
    }

    DBG << pin_thread_message(_cpus[slot]);
    if(!_is_global_thread_pool) {
      std::thread::id thread_id = std::this_thread::get_id();
      int current_cpu = THREAD_ID;
      std::lock_guard<std::mutex> lock(_mutex);
      _cpu_before[thread_id] = current_cpu;
    }
    pin_thread_to_cpu(_cpus[slot]);
  }

  void on_scheduler_exit(bool) override {
    if (!_is_global_thread_pool) {
      std::thread::id thread_id = std::this_thread::get_id();
      int cpu_before = -1;
      {
        std::lock_guard<std::mutex> lock(_mutex);
        DBG << unpin_thread_message();
        auto it = _cpu_before.find(thread_id);
        if ( it != _cpu_before.end() ) {
          cpu_before = it->second;
          _cpu_before.erase(thread_id);
        }
      }
      if ( cpu_before != -1 ) {
        pin_thread_to_cpu(cpu_before);
      }
    } else {
          DBG << "Thread with PID" << std::this_thread::get_id()
            << "leaves GLOBAL task arena";
    }
  }

 private:

  void pin_thread_to_cpu(const int cpu_id) {
    #ifdef __linux__
    const size_t size = CPU_ALLOC_SIZE(_num_cpus);
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu_id, &mask);
    const int err = sched_setaffinity(0, size, &mask);
    #elif _WIN32
    auto mask = (static_cast<DWORD_PTR>(1) << cpu_id);
    const int err = SetThreadAffinityMask(GetCurrentThread(), mask) == 0;
    #endif

    if (err) {
      const int error = errno;
      throw SystemException(
        "Failed to set thread affinity to cpu" + std::to_string(cpu_id) + "." + strerror(error));
    }

    ASSERT(THREAD_ID == cpu_id);
    DBG << "Thread with PID" << std::this_thread::get_id()
        << "successfully pinned to CPU" << cpu_id;
  }

  std::string pin_thread_message(const int cpu_id) {
    std::stringstream ss;
    ss << "Assign thread with PID " << std::this_thread::get_id()
       << " to CPU " << cpu_id;
    if ( _numa_node != -1 ) {
      ss << " on NUMA node " << _numa_node;
    } else {
      ss << " in GLOBAL task arena";
    }
    return ss.str();
  }

  std::string unpin_thread_message() {
    std::stringstream ss;
    ss << "Unassign thread with PID " << std::this_thread::get_id()
       << " on CPU " << THREAD_ID;
    if ( _numa_node != -1 ) {
      ss << " from NUMA node " << _numa_node;
    } else {
      ss << " from GLOBAL task arena";
    }
    ss << " (Assign thread to its last CPU: "
       << _cpu_before[std::this_thread::get_id()] << ")";
    return ss.str();
  }

  const int _num_cpus;
  const int _numa_node;
  const bool _is_global_thread_pool;
  std::vector<int> _cpus;

  std::mutex _mutex;
  std::unordered_map<std::thread::id, int> _cpu_before;
};
}  // namespace parallel
}  // namespace mt_kahypar
