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

#pragma once

#include <algorithm>
#include <hwloc.h>
#include <numeric>
#include <thread>
#include <vector>

#include "mt-kahypar/parallel/hardware_topology.h"
#include "mt-kahypar/macros.h"

namespace mt_kahypar {
namespace parallel {
using numa_to_cpu_t = std::vector<std::vector<int> >;

struct Cpu {
  int cpu_id;
  bool is_hyperthread;
};

struct Node {
  explicit Node(const int node) :
    os_index(node),
    cpuset(),
    next_cousin(),
    cpus() {
    cpuset = hwloc_bitmap_alloc();
  }

  Node(const Node& other) :
    os_index(other.os_index),
    cpuset(hwloc_bitmap_dup(other.cpuset)),
    next_cousin(other.next_cousin),
    cpus(other.cpus) { }

  Node & operator= (const Node& other) {
    os_index = other.os_index;
    cpuset = hwloc_bitmap_dup(other.cpuset);
    next_cousin = other.next_cousin;
    cpus = other.cpus;
    return *this;
  }

  ~Node() {
    hwloc_bitmap_free(cpuset);
  }

  void init_cpuset(const std::vector<int>& cpu_ids) {
    for (const int cpu_id : cpu_ids) {
      hwloc_bitmap_set(cpuset, cpu_id);
      cpus.emplace_back(Cpu { cpu_id,
        HardwareTopology<>::instance().is_hyperthread(cpu_id) });
    }
  }

  int os_index;
  hwloc_cpuset_t cpuset;
  Node* next_cousin;
  std::vector<Cpu> cpus;
};

class Topology {
 public:
  explicit Topology(const numa_to_cpu_t& numa_to_cpu) :
    _nodes() {
    init(numa_to_cpu);
  }

  Node* get_node(const int node) {
    ASSERT(node < (int)_nodes.size());
    return &_nodes[node];
  }

 private:
  void init(const numa_to_cpu_t& numa_to_cpu) {
    int numa_nodes = numa_to_cpu.size();
    for (int node = 0; node < numa_nodes; ++node) {
      _nodes.emplace_back(node);
      _nodes.back().init_cpuset(numa_to_cpu[node]);
    }
    for (int node = 0; node < numa_nodes - 1; ++node) {
      _nodes[node].next_cousin = &_nodes[node + 1];
    }
  }

  std::vector<Node> _nodes;
};

static numa_to_cpu_t split_physical_cpus_into_numa_nodes(const int num_numa_nodes) {
  ASSERT(num_numa_nodes > 0);
  numa_to_cpu_t numa_to_cpu(num_numa_nodes);

  std::vector<int> cpus;
  for ( const int cpu_id : HardwareTopology<>::instance().get_all_cpus() ) {
    cpus.push_back(cpu_id);
  }
  std::sort(cpus.begin(), cpus.end(),
    [&](const int lhs, const int rhs) {
      const bool is_hyperthread_lhs = HardwareTopology<>::instance().is_hyperthread(lhs);
      const bool is_hyperthread_rhs = HardwareTopology<>::instance().is_hyperthread(rhs);
      return is_hyperthread_lhs < is_hyperthread_rhs ||
             ( is_hyperthread_lhs == is_hyperthread_rhs && lhs < rhs );
    });

  ASSERT(num_numa_nodes <= (int)cpus.size());
  int current_numa_node = 0;
  for ( const int cpu_id : cpus ) {
    numa_to_cpu[current_numa_node].push_back(cpu_id);
    current_numa_node = (current_numa_node + 1) % num_numa_nodes;
  }

  return numa_to_cpu;
}

using node_t = Node *;
using topology_t = Topology *;

template <int NUM_NUMA_NODES>
class TopologyMock {
 public:
  static void initialize(topology_t& topology) {
    topology = new Topology(split_physical_cpus_into_numa_nodes(NUM_NUMA_NODES));
  }

  static node_t get_first_numa_node(topology_t topology) {
    return topology->get_node(0);
  }

  static std::vector<int> get_cpus_of_numa_node_without_hyperthreads(node_t node) {
    std::vector<int> cpus;
    for (const Cpu& cpu : node->cpus) {
      if (!cpu.is_hyperthread) {
        cpus.push_back(cpu.cpu_id);
      }
    }
    return cpus;
  }

  static std::vector<int> get_cpus_of_numa_node_only_hyperthreads(node_t node) {
    std::vector<int> cpus;
    for (const Cpu& cpu : node->cpus) {
      if (cpu.is_hyperthread) {
        cpus.push_back(cpu.cpu_id);
      }
    }
    return cpus;
  }

  static void destroy_topology(topology_t) { }

 private:
  TopologyMock() { }
};
}  // namespace parallel
}  // namespace mt_kahypar
