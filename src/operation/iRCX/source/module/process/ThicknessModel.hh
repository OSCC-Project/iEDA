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

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

#include "LayerTable.hh"
#include "LayoutData.hh"
#include "TopoPool.hh"
#include "NetEtchProfile.hh"
#include "MetalDensity.hh"
#include "ProcessCorner.hpp"
#include "RCXData.hh"
namespace ircx {

class ThicknessModel {
 public:
  ThicknessModel() = default;
  ~ThicknessModel() = default;

  void set_layout_data(const LayoutData* v) { layout_data_ = v; }
  void set_layer_table(const LayerTable* v) { layer_table_ = v; }
  void set_topo_pool(const TopoPool* v) { topo_pool_ = v; }
  void set_corner_data(const std::vector<RCXData::CornerData>* v) { corner_data_ = v; }
  void set_metal_density(const MetalDensity* v) { metal_density_ = v; }

  void apply_thickness_variation(Size corner_idx, Size net_idx, NetEtchProfile& etch_profile) const {
    const auto& corner = *(*corner_data_)[corner_idx].process_corner;
    const auto net_edges = topo_pool_->net_edges(net_idx);
    const Size edge_count = net_edges.size();

    for (Size edge_idx = 0; edge_idx < edge_count; ++edge_idx) {
      const TopoEdge& edge = net_edges[edge_idx];
      if (edge.is_via()) continue; // TODO: 针对via的thickness变化

      const Size process_layer_id = layer_table_->design_to_process_id(edge.layer_id());
      auto* conductor_layer = corner.get_layers()->find_conductor_layer(process_layer_id);
      if (!conductor_layer) continue;

      std::span<EdgeEtchInterval> edge_etch_intervals = etch_profile.edgeIntervals(edge_idx);
      if (edge_etch_intervals.empty()) continue;

      const PBTVCache pbtv_cache = build_pbtv_cache_(corner, *conductor_layer);
      PBTVScratch scratch;

      if (pbtv_cache.valid()) {
        const double effective_density = cal_effective_density_(edge, *conductor_layer);
        fill_polynomial_terms_(effective_density, *pbtv_cache.density_orders, scratch.density_terms);
      }

      for (EdgeEtchInterval& etch_interval : edge_etch_intervals) {
        // Height (no variation: read directly from layer)
        etch_interval.height = static_cast<Micron>(conductor_layer->get_height());

        // Thickness (POLYNOMIAL_BASED_THICKNESS_VARIATION)
        model_thickness_(etch_interval, *conductor_layer, pbtv_cache, scratch);
      }
    }
  }

 private:
  struct PBTVCache {
    const ::itf::itfiPBTV* table{nullptr};
    const std::vector<int>* density_orders{nullptr};
    const std::vector<int>* width_orders{nullptr};
    const std::vector<float>* width_ranges{nullptr};

    bool valid() const
    {
      return table != nullptr && density_orders != nullptr && width_orders != nullptr
             && width_ranges != nullptr && !density_orders->empty() && !width_orders->empty();
    }
  };

  struct PBTVScratch {
    std::vector<float> density_terms;
    std::vector<float> width_terms;
  };

  // POLYNOMIAL_BASED_THICKNESS_VARIATION
  // Fills d.thickness using the PBTV polynomial model.
  void model_thickness_(
      EdgeEtchInterval& etch,
      const ::itf::LayerConductor& cdt,
      const PBTVCache& pbtv_cache,
      PBTVScratch& scratch) const
  {
    const float t_nom = cdt.get_thickness();
    if (!pbtv_cache.valid() || scratch.density_terms.empty()) {
      etch.thickness = static_cast<Micron>(t_nom);
      return;
    }

    // const float rt_deff = cal_rt_deff_(
    //     scratch.density_terms, static_cast<double>(etch.width), pbtv_cache, scratch.width_terms);
    // const float rt_ws = 0.0f;   // f(W, S)   — not yet modeled
    // const float rt_siw = 0.0f;  // f(SiW)    — not yet modeled

    // etch.thickness = static_cast<Micron>(t_nom * (1.0f + rt_deff + rt_ws + rt_siw));
    etch.thickness = static_cast<Micron>(t_nom);
  }

  // DENSITY_BOX_WEIGHTING_FACTOR
  // Compute density-weighted effective density at the edge center.
  double cal_effective_density_(
      const TopoEdge& edge,
      const ::itf::LayerConductor& cdt) const
  {
    const Size process_layer_id = layer_table_->design_to_process_id(edge.layer_id());

    double effective_density = 0.0;
    GtlRectI density_box;

    const GtlPointI center_point = edge.center();
    const Dbu center_x_dbu = geom::x(center_point);
    const Dbu center_y_dbu = geom::y(center_point);


    for (const auto& [box_size, weight] : cdt.get_density_box_weighting_factor()) {
      const Dbu half_box_size_dbu = static_cast<Dbu>(box_size * layout_data_->dbu_per_micron);
      density_box = GtlRectI(center_x_dbu - half_box_size_dbu, center_y_dbu - half_box_size_dbu,
                             center_x_dbu + half_box_size_dbu, center_y_dbu + half_box_size_dbu);

      effective_density += metal_density_->cal_density(process_layer_id, density_box) * weight;
    }

    return effective_density;
  }

  // cal_rt_deff
  // Returns relative thickness change as a function of effective density.
  float cal_rt_deff_(
      std::span<const float> density_terms,
      double width,
      const PBTVCache& pbtv_cache,
      std::vector<float>& width_terms) const
  {
    if (!pbtv_cache.valid()) {
      return 0.0f;
    }

    fill_polynomial_terms_(width, *pbtv_cache.width_orders, width_terms);
    if (width_terms.empty()) {
      return 0.0f;
    }

    const size_t table_idx = select_width_table_(width, *pbtv_cache.width_ranges);
    const auto& coeffs = pbtv_cache.table->get_polynomial_coefficients_list(table_idx);
    if (coeffs.size() != density_terms.size() * width_terms.size()) {
      return 0.0f;
    }

    // Coefficients are stored row-major with density as the outer index.
    float rt_deff = 0.0f;
    size_t coeff_idx = 0;
    for (size_t density_idx = 0; density_idx < density_terms.size(); ++density_idx) {
      const float density_term = density_terms[density_idx];
      for (size_t width_idx = 0; width_idx < width_terms.size(); ++width_idx, ++coeff_idx) {
        rt_deff += density_term * coeffs[coeff_idx] * width_terms[width_idx];
      }
    }

    return rt_deff;
  }

  PBTVCache build_pbtv_cache_(
      const ::itf::ProcessCorner& pc,
      const ::itf::LayerConductor& cdt) const
  {
    if (!pc.get_use_si_density()) {
      return {};
    }

    const auto& table = cdt.get_PBTV();
    if (table.get_polynomial_coefficients_list_size() == 0) {
      return {};  // PBTV not available for this layer
    }

    const auto& d_orders = table.get_density_polynomial_orders();
    const auto& w_orders = table.get_width_polynomial_orders();
    if (d_orders.empty() || w_orders.empty()) {
      return {};
    }

    PBTVCache cache;
    cache.table = &table;
    cache.density_orders = &d_orders;
    cache.width_orders = &w_orders;
    cache.width_ranges = &table.get_width_ranges();
    return cache;
  }

  void fill_polynomial_terms_(
      double value,
      const std::vector<int>& orders,
      std::vector<float>& terms) const
  {
    terms.resize(orders.size());
    for (size_t term_idx = 0; term_idx < orders.size(); ++term_idx) {
      const int order = orders[term_idx];
      if (order == 0) {
        terms[term_idx] = 1.0f;
      } else if (order == 1) {
        terms[term_idx] = static_cast<float>(value);
      } else {
        terms[term_idx] = static_cast<float>(std::pow(value, order));
      }
    }
  }

  size_t select_width_table_(
      double width,
      const std::vector<float>& width_ranges) const
  {
    auto it = std::upper_bound(width_ranges.begin(), width_ranges.end(),
                               static_cast<float>(width));
    return static_cast<size_t>(std::distance(width_ranges.begin(), it));
  }

  const LayoutData* layout_data_{nullptr};
  const LayerTable* layer_table_{nullptr};
  const TopoPool* topo_pool_{nullptr};
  const std::vector<RCXData::CornerData>* corner_data_{nullptr};

  // built here
  const MetalDensity* metal_density_{nullptr};
};

}
