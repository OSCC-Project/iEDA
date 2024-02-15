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

#include <hwloc.h>
#include <mutex>
#include <memory>
#include <shared_mutex>
#include <functional>

#include "tbb/task_arena.h"
#include "tbb/task_group.h"
#include "tbb/global_control.h"

#include "mt-kahypar/macros.h"
#include "mt-kahypar/parallel/thread_pinning_observer.h"

namespace mt_kahypar {
namespace parallel {
/**
 * Creates number of NUMA nodes TBB task arenas. Each task arena is pinned
 * to a unique NUMA node. Each task arena can then be used to execute tasks
 * on specific NUMA node.
 */
template <typename HwTopology, bool is_numa_aware>
class TBBInitializer {

  static constexpr bool debug = false;

  using Self = TBBInitializer<HwTopology, is_numa_aware>;
  using ThreadPinningObserver = mt_kahypar::parallel::ThreadPinningObserver<HwTopology>;

 public:
  TBBInitializer(const TBBInitializer&) = delete;
  TBBInitializer & operator= (const TBBInitializer &) = delete;

  TBBInitializer(TBBInitializer&&) = delete;
  TBBInitializer & operator= (TBBInitializer &&) = delete;

  static TBBInitializer& instance(const size_t num_threads = std::thread::hardware_concurrency()) {
    static TBBInitializer instance(num_threads);
    return instance;
  }

  int total_number_of_threads() const {
    return _num_threads;
  }

  int number_of_used_cpus_on_numa_node(const int node) const {
    ASSERT(static_cast<size_t>(node) < _numa_node_to_cpu_id.size());
    return _numa_node_to_cpu_id[node].size();
  }

  int num_used_numa_nodes() const {
    return _numa_node_to_cpu_id.size();
  }

  hwloc_cpuset_t used_cpuset() const {
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    for ( const auto& numa_node : _numa_node_to_cpu_id ) {
      for ( const int cpu_id : numa_node ) {
        hwloc_bitmap_set(cpuset, cpu_id);
      }
    }
    return cpuset;
  }

  void terminate() {
    if ( _global_observer ) {
      _global_observer->observe(false);
    }
  }

 private:
  explicit TBBInitializer(const int num_threads) :
    _num_threads(num_threads),
    _gc(tbb::global_control::max_allowed_parallelism, num_threads),
    _global_observer(nullptr),
    _cpus(),
    _numa_node_to_cpu_id() {
    HwTopology& topology = HwTopology::instance();
    int num_numa_nodes = topology.num_numa_nodes();
    DBG << "Initialize TBB with" << num_threads << "threads";
    _cpus = topology.get_all_cpus();
    // Sort cpus in the following order
    // 1.) Non-hyperthread first
    // 2.) Increasing order of numa node
    // 3.) Increasing order of cpu id
    // ...
    std::sort(_cpus.begin(), _cpus.end(),
              [&](const int& lhs, const int& rhs) {
          int node_lhs = topology.numa_node_of_cpu(lhs);
          int node_rhs = topology.numa_node_of_cpu(rhs);
          bool is_hyperthread_lhs = topology.is_hyperthread(lhs);
          bool is_hyperthread_rhs = topology.is_hyperthread(rhs);
          return is_hyperthread_lhs < is_hyperthread_rhs ||
          (is_hyperthread_lhs == is_hyperthread_rhs && node_lhs < node_rhs) ||
          (is_hyperthread_lhs == is_hyperthread_rhs && node_lhs == node_rhs && lhs < rhs);
        });
    // ... this ensure that we first pop nodes in hyperthreading
    while (static_cast<int>(_cpus.size()) > _num_threads) {
      _cpus.pop_back();
    }
    _global_observer = std::make_unique<ThreadPinningObserver>(_cpus);

    _numa_node_to_cpu_id.resize(num_numa_nodes);
    for ( const int cpu_id : _cpus ) {
      int node = topology.numa_node_of_cpu(cpu_id);
      ASSERT(node < static_cast<int>(_numa_node_to_cpu_id.size()));
      _numa_node_to_cpu_id[node].push_back(cpu_id);
    }
    while( !_numa_node_to_cpu_id.empty() && _numa_node_to_cpu_id.back().empty() ) {
      _numa_node_to_cpu_id.pop_back();
    }
  }

  int _num_threads;
  tbb::global_control _gc;
  std::unique_ptr<ThreadPinningObserver> _global_observer;
  std::vector<int> _cpus;
  std::vector<std::vector<int>> _numa_node_to_cpu_id;
};
}  // namespace parallel
}  // namespace mt_kahypar
