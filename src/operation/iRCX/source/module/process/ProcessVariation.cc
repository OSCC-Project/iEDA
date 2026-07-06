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
#include <limits>
#include <omp.h>

#include "ProcessVariation.hh"
#include "TopoPool.hh"
#include "ThicknessModel.hh"
#include "Types.hh"
#include "WidthModel.hh"
#include "log/Log.hh"
namespace ircx {

void ProcessVariation::reset()
{
  layout_data_ = nullptr;
  net_environments_ = nullptr;
  corner_net_etch_pools_ = nullptr;
  layer_table_ = nullptr;
  topo_pool_ = nullptr;
  corner_data_ = nullptr;
  metal_density_.clear();
  corner_num_ = 0;
  net_num_ = 0;
}

bool ProcessVariation::buildEtchProfiles()
{
  if (!layout_data_) {
    LOG_ERROR << "build process variation failed: LayoutData not initialized.";
    return false;
  }
  if (!net_environments_) {
    LOG_ERROR << "build process variation failed: net environments not set.";
    return false;
  }
  if (!corner_net_etch_pools_) {
    LOG_ERROR << "build process variation failed: etch profiles not set.";
    return false;
  }
  if (!layer_table_) {
    LOG_ERROR << "build process variation failed: LayerTable not initialized.";
    return false;
  }
  if (!topo_pool_) {
    LOG_ERROR << "build process variation failed: TopoPool not initialized.";
    return false;
  }
  if (corner_data_ == nullptr || corner_data_->empty()) {
    LOG_ERROR << "build process variation failed: process corners not set.";
    return false;
  }
  for (const auto& corner : *corner_data_) {
    if (corner.process_corner == nullptr) {
      LOG_ERROR << "build process variation failed: null process corner "
                << corner.name << ".";
      return false;
    }
  }

  initMetalDensity();
  initEtchIntervals();

  WidthModel wm;
  wm.set_topo_pool(topo_pool_);
  wm.set_layer_table(layer_table_);
  wm.set_corner_data(corner_data_);
  for (Size corner_idx = 0; corner_idx < corner_num_; ++corner_idx) {
    #pragma omp parallel for schedule(dynamic)
    for (Size net_idx = 0; net_idx < net_num_; ++net_idx) {
      wm.apply_width_variation(
          corner_idx,
          net_idx,
          corner_net_etch_pools_->at({corner_idx, net_idx}));
    }
  }

  ThicknessModel tm;
  tm.set_layout_data(layout_data_);
  tm.set_topo_pool(topo_pool_);
  tm.set_layer_table(layer_table_);
  tm.set_corner_data(corner_data_);
  tm.set_metal_density(&metal_density_);

  for (Size corner_idx = 0; corner_idx < corner_num_; ++corner_idx) {
    #pragma omp parallel for schedule(dynamic)
    for (Size net_idx = 0; net_idx < net_num_; ++net_idx) {
      tm.apply_thickness_variation(
          corner_idx,
          net_idx,
          corner_net_etch_pools_->at({corner_idx, net_idx}));
    }
  }

  return true;
}

void ProcessVariation::initMetalDensity()
{
  metal_density_.set_topo_pool(topo_pool_);
  metal_density_.build();
}

void ProcessVariation::initEtchIntervals()
{
  net_num_ = layout_data_->regular_net_count();
  corner_num_ = corner_data_->size();
  corner_net_etch_pools_->init(corner_num_, net_num_);

  const std::vector<NetEnvironment>& net_environments = *net_environments_;

  Micron micron_per_dbu = unit::to_micron(1, layout_data_->dbu_per_micron);

  for (Size corner_idx = 0; corner_idx < corner_num_; ++corner_idx) {
    #pragma omp parallel for schedule(dynamic)
    for (Size net_idx = 0; net_idx < net_num_; ++net_idx) {
      const auto net_edges = topo_pool_->net_edges(net_idx);
      const Size edge_count = net_edges.size();

      NetEtchProfile& etch_profile = corner_net_etch_pools_->at({corner_idx, net_idx});
      const NetEnvironment& environment = net_environments[net_idx];

      for (Size edge_idx = 0; edge_idx < edge_count; ++edge_idx) {
        const TopoEdge& edge = net_edges[edge_idx];

        if (edge.is_via()) {
          etch_profile.appendEdgeIntervals({});  // placeholder to keep index aligned with TopoPool
          continue;
        }

        const std::span<const EdgeEnvironmentInterval> env_intervals =
            environment.edgeIntervals(edge_idx);
        const Size interval_count = env_intervals.size();
        std::vector<EdgeEtchInterval> etch_intervals(interval_count);

        for (Size interval_idx = 0; interval_idx < interval_count; ++interval_idx) {
          const EdgeEnvironmentInterval& env_interval = env_intervals[interval_idx];
          EdgeEtchInterval& etch_interval = etch_intervals[interval_idx];

          etch_interval.a0     = env_interval.a0 * micron_per_dbu;
          etch_interval.a1     = env_interval.a1 * micron_per_dbu;
          etch_interval.center = edge.coord() * micron_per_dbu;
          etch_interval.width  = edge.width() * micron_per_dbu;

          // convert center-to-center spacing (EdgeEnvironmentInterval) to edge-to-edge gap (EdgeEtchInterval)
          if (env_interval.lo_adjacent)
            etch_interval.lo_spacing =
                (env_interval.lo_spacing - env_interval.lo_adjacent->half_width() - edge.half_width())
                * micron_per_dbu;
          if (env_interval.hi_adjacent)
            etch_interval.hi_spacing =
                (env_interval.hi_spacing - env_interval.hi_adjacent->half_width() - edge.half_width())
                * micron_per_dbu;

          etch_interval.thickness = 0;
          etch_interval.height    = 0;
        }

        etch_profile.appendEdgeIntervals(std::move(etch_intervals));
      }
    }
  }
}

} // namespace ircx
