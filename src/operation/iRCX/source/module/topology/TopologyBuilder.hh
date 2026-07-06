// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "TopoPool.hh"
namespace ircx {

class LayoutData;
class Net;
class SpecialNet;


class TopologyBuilder {
 public:
  explicit TopologyBuilder(TopoPool& topologies) : topo_pool_(&topologies) {}
  TopologyBuilder() = delete;
  ~TopologyBuilder() = default;

  // Result of building a single net's topology.
  // Holds all data until it can be merged into the shared TopoPool.
  struct NetTopo {
    std::vector<TopoNode> nodes;
    std::vector<TopoEdge> edges;
  };

  // Build all regular-net topologies. Phase 1 builds each net in parallel into
  // independent local storage; Phase 2 serially merges results into the shared
  // contiguous pools, preserving cache-friendly layout.
  void build_all(const LayoutData& ld) const;

  // Build a topology for the special net (power/ground) and store it in
  // topo_pool_'s dedicated special_edge_pool(). Each per-layer rectangle of every
  // segment and Patch becomes one TopoEdge with net_id = kSpecialNetId
  // (no nodes; graph connectivity is not needed for special nets).
  // Must be called after build_all().
  void build_special(const LayoutData& ld) const;

 private:
  // Build one net's topology into independent local storage (no shared state).
  // Safe to call from multiple threads simultaneously on different nets.
  NetTopo build_one_(const Net& net) const;

 private:
  TopoPool* topo_pool_{nullptr};
};

}  // namespace ircx
