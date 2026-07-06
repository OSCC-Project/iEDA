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

#include <cmath>
#include <utility>

#include "LayerTable.hh"
#include "TopoPool.hh"
#include "NetEtchProfile.hh"
#include "ProcessCorner.hpp"
#include "RCXData.hh"
namespace ircx {

class WidthModel {
 public:
  WidthModel() = default;
  ~WidthModel() = default;

  void set_topo_pool(const TopoPool* v) { topo_pool_ = v; }
  void set_layer_table(const LayerTable* v) { layer_table_ = v; }
  void set_corner_data(const std::vector<RCXData::CornerData>* v) { corner_data_ = v; }

  void apply_width_variation(Size corner_idx, Size net_idx, NetEtchProfile& etch_profile) const {
    const auto& corner = *(*corner_data_)[corner_idx].process_corner;
    const auto net_edges = topo_pool_->net_edges(net_idx);
    const Size edge_count = net_edges.size();
    
    for (Size edge_idx = 0; edge_idx < edge_count; ++edge_idx) {
      const TopoEdge& edge = net_edges[edge_idx];
      if (edge.is_via()) continue;

      const Size process_layer_id = layer_table_->design_to_process_id(edge.layer_id());
      auto* conductor_layer = corner.get_layers()->find_conductor_layer(process_layer_id);
      if (!conductor_layer) continue;

      std::span<EdgeEtchInterval> edge_etch_intervals = etch_profile.edgeIntervals(edge_idx);
      for (EdgeEtchInterval& etch_interval : edge_etch_intervals) {
        for (const auto& etch_table : conductor_layer->get_etch_vws_list()) { // TODO: 还不完善，只考虑了vws这种情况，甚至没考虑是否CAPACITIVE_ONLY
          // Low-side etch interval: lo_spacing is edge-to-edge in microns.
          Micron low_side_etch = 0;
          auto low_side_etch_opt =
              etch_table.query_interpolation(etch_interval.width, etch_interval.lo_spacing);
          if (low_side_etch_opt) low_side_etch = low_side_etch_opt.value();

          // High-side etch interval.
          Micron high_side_etch = 0;
          auto high_side_etch_opt =
              etch_table.query_interpolation(etch_interval.width, etch_interval.hi_spacing);
          if (high_side_etch_opt) high_side_etch = high_side_etch_opt.value();

          // Positive etch interval shrinks the conductor: center shifts, width decreases.
          etch_interval.center += 0.5 * low_side_etch - 0.5 * high_side_etch;
          etch_interval.width  -= low_side_etch + high_side_etch;
        }
      }
    }
  }

 private:
  const TopoPool* topo_pool_{nullptr};
  const LayerTable* layer_table_{nullptr};
  const std::vector<RCXData::CornerData>* corner_data_{nullptr};
};

}
