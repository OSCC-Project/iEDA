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
#include <thread>
#include <vector>
#include <algorithm>
#include <random>

#include "mt-kahypar/macros.h"

#include "mt-kahypar/parallel/hwloc_topology.h"
#include "mt-kahypar/utils/exception.h"

namespace mt_kahypar {
namespace parallel {
/**
 * Class represents the hardware topology of the system.
 * Internally it uses hwloc library to find numa nodes and corresponding
 * cpus. Furthermore, it implements functionalities to pin logical threads
 * to cpus of a specific numa node.
 *
 * Template parameters can be replaced in order to mock hardware topology and
 * simulate a NUMA on a UMA system.
 */
template <typename HwTopology = HwlocTopology,
          typename Topology = hwloc_topology_t,
          typename Node = hwloc_obj_t>
class HardwareTopology {
 private:
  static constexpr bool debug = false;

  using Self = HardwareTopology<HwTopology, Topology, Node>;

  struct Cpu {
    int cpu_id;
    bool is_hyperthread;
  };

  class NumaNode {
   public:
    NumaNode(Node node) :
      _node_id(node->os_index),
      _num_cores(0),
      _cpuset(node->cpuset),
      _cpus(),
      _mutex() {
      for (const int cpu_id :  HwTopology::get_cpus_of_numa_node_without_hyperthreads(node)) {
        _cpus.emplace_back(Cpu { cpu_id, false });
        _num_cores++;
      }
      for (const int cpu_id :  HwTopology::get_cpus_of_numa_node_only_hyperthreads(node)) {
        _cpus.emplace_back(Cpu { cpu_id, true });
      }
    }

    NumaNode(const NumaNode&) = delete;
    NumaNode & operator= (const NumaNode &) = delete;

    NumaNode(NumaNode&& other) :
      _node_id(other._node_id),
      _num_cores(other._num_cores),
      _cpuset(std::move(other._cpuset)),
      _cpus(std::move(other._cpus)),
      _mutex() { }

    int get_id() const {
      return _node_id;
    }

    hwloc_cpuset_t get_cpuset() const {
      return _cpuset;
    }

    std::vector<int> cpus() {
      std::vector<int> cpus;
      for (const Cpu& cpu : _cpus) {
        cpus.push_back(cpu.cpu_id);
      }
      return cpus;
    }

    size_t num_cores_on_numa_node() const {
      return _num_cores;
    }

    size_t num_cpus_on_numa_node() const {
      return _cpus.size();
    }

    bool is_hyperthread(const int cpu_id) {
      size_t pos = 0;
      for ( ; pos < _cpus.size(); ++pos) {
        if (_cpus[pos].cpu_id == cpu_id) {
          break;
        }
      }
      ASSERT(pos < _cpus.size(), "CPU" << cpu_id << "not found on numa node" << _node_id);
      return _cpus[pos].is_hyperthread;
    }

    int get_backup_cpu(const int except_cpu) {
      std::lock_guard<std::mutex> lock(_mutex);
      int cpu_id = -1;
      if ( _cpus.size() > 1 ) {
        std::mt19937 rng(420);
        std::shuffle(_cpus.begin(), _cpus.end(), rng);
        if ( _cpus[0].cpu_id != except_cpu ) {
          cpu_id = _cpus[0].cpu_id;
        } else {
          cpu_id = _cpus[1].cpu_id;
        }
      }
      return cpu_id;
    }

   private:
    int _node_id;
    size_t _num_cores;
    hwloc_cpuset_t _cpuset;
    std::vector<Cpu> _cpus;
    std::mutex _mutex;
  };

 public:
  HardwareTopology(const HardwareTopology&) = delete;
  HardwareTopology & operator= (const HardwareTopology &) = delete;

  HardwareTopology(HardwareTopology&&) = delete;
  HardwareTopology & operator= (HardwareTopology &&) = delete;

  ~HardwareTopology() {
    HwTopology::destroy_topology(_topology);
  }

  static HardwareTopology& instance() {
    static HardwareTopology instance;
    return instance;
  }

  size_t num_numa_nodes() const {
    return _numa_nodes.size();
  }

  size_t num_cpus() const {
    return _num_cpus;
  }

  int numa_node_of_cpu(const int cpu_id) const {
    ASSERT(cpu_id < (int)_cpu_to_numa_node.size());
    ASSERT(_cpu_to_numa_node[cpu_id] != std::numeric_limits<int>::max());
    return _cpu_to_numa_node[cpu_id];
  }

  bool is_hyperthread(const int cpu_id) {
    int node = numa_node_of_cpu(cpu_id);
    return _numa_nodes[node].is_hyperthread(cpu_id);
  }

  // ! Number of Cores on NUMA node
  int num_cores_on_numa_node(const int node) const {
    ASSERT(node < (int)_numa_nodes.size());
    ASSERT(_numa_nodes[node].get_id() == node);
    return _numa_nodes[node].num_cores_on_numa_node();
  }

  // ! Number of CPUs on NUMA node
  int num_cpus_on_numa_node(const int node) const {
    ASSERT(node < (int)_numa_nodes.size());
    ASSERT(_numa_nodes[node].get_id() == node);
    return _numa_nodes[node].num_cpus_on_numa_node();
  }

  // ! CPU bitmap of NUMA node
  hwloc_cpuset_t get_cpuset_of_numa_node(int node) const {
    ASSERT(node < (int)_numa_nodes.size());
    ASSERT(_numa_nodes[node].get_id() == node);
    return _numa_nodes[node].get_cpuset();
  }

  // ! List of CPUs of NUMA node
  std::vector<int> get_cpus_of_numa_node(int node) {
    ASSERT(node < (int)_numa_nodes.size());
    ASSERT(_numa_nodes[node].get_id() == node);
    return _numa_nodes[node].cpus();
  }

  // ! List of all available CPUs
  std::vector<int> get_all_cpus() {
    std::vector<int> cpus;
    for ( size_t node = 0; node < num_numa_nodes(); ++node ) {
      for ( const int cpu_id : _numa_nodes[node].cpus() ) {
        cpus.push_back(cpu_id);
      }
    }
    return cpus;
  }

  // ! Returns a CPU on a NUMA node that differs from CPU except_cpu
  int get_backup_cpu(const int node, const int except_cpu) {
    ASSERT(node < (int)_numa_nodes.size());
    int cpu_id = _numa_nodes[node].get_backup_cpu(except_cpu);
    if ( cpu_id == -1 ) {
      #ifndef KAHYPAR_TRAVIS_BUILD
      throw SystemException("Your system has not enough cpus to execute MT-KaHyPar (> 1)");
      #else
      // Handling special case:
      // Travis CI has only two cpus, when mocking a numa architecture
      // of two nodes, we have to search for a backup node on a different
      // numa node. Note, this is only enabled in DEBUG mode.
      cpu_id = except_cpu;
      #endif
    }
    return cpu_id;
  }

  // ! Set membind policy to interleaved allocations on used NUMA nodes
  // ! covered by cpuset
  void activate_interleaved_membind_policy(hwloc_cpuset_t cpuset) const {
    hwloc_set_membind(_topology, cpuset, HWLOC_MEMBIND_INTERLEAVE, HWLOC_MEMBIND_MIGRATE);
  }

 private:
  HardwareTopology() :
    _num_cpus(0),
    _topology(),
    _numa_nodes(),
    _cpu_to_numa_node(std::thread::hardware_concurrency(),
      std::numeric_limits<int>::max()) {
    HwTopology::initialize(_topology);
    init_numa_nodes();
  }

  void init_numa_nodes() {
    Node node = HwTopology::get_first_numa_node(_topology);
    while (node != nullptr) {
      _numa_nodes.emplace_back(node);
      node = node->next_cousin;
      for (const int cpu_id : _numa_nodes.back().cpus()) {
        ASSERT(cpu_id < (int)_cpu_to_numa_node.size());
        _cpu_to_numa_node[cpu_id] = _numa_nodes.back().get_id();
        ++_num_cpus;
      }
    }
  }


  size_t _num_cpus;
  Topology _topology;
  std::vector<NumaNode> _numa_nodes;
  std::vector<int> _cpu_to_numa_node;
};

}  // namespace parallel
}  // namespace mt_kahypar
