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

#include "TopoPool.hh"
namespace ircx
{

class MetalDensity
{
 public:
  MetalDensity() = default;
  ~MetalDensity() = default;

  void set_topo_pool(const TopoPool* v) { topo_pool_ = v; }

  void clear()
  {
    topo_pool_ = nullptr;
    polyset_map_.clear();
  }

  // Build the per-layer polygon set from all edges (regular + special).
  // Special-net edges are stored in a dedicated special_edge_pool() in TopoPool.
  void build()
  {
    polyset_map_.clear();

    // regular net edges
    for (const auto& edge : topo_pool_->edge_pool()) {
      polyset_map_[edge.layer_id()] += edge.shape();
    }

    // special net edges (power/ground)
    for (const auto& edge : topo_pool_->special_edge_pool()) {
      polyset_map_[edge.layer_id()] += edge.shape();
    }
  }

  double cal_density(Size layer, const GtlRectI& box) const
  {
    const auto it = polyset_map_.find(layer);
    if (it == polyset_map_.end()) return 0.0;

    return static_cast<double>(gtl::area(it->second & box)) /
           static_cast<double>(gtl::area(box));
  }

  MetalDensity(const MetalDensity&) = delete;
  MetalDensity(MetalDensity&&) = delete;
  auto operator=(const MetalDensity&) -> MetalDensity& = delete;
  auto operator=(MetalDensity&&) -> MetalDensity& = delete;

 private:
  const TopoPool* topo_pool_{nullptr};
  std::map<Size, GtlPolySetI> polyset_map_;
};

} // namespace ircx
