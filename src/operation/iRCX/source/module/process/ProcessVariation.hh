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

#include <vector>

#include "CornerNetPool.hh"
#include "Types.hh"
#include "NetEtchProfile.hh"
#include "NetEnvironment.hh"
#include "MetalDensity.hh"
#include "RCXData.hh"
namespace ircx {
  class LayoutData;
  class LayerTable;
  class MetalDensity;
  class TopoPool;
}

namespace ircx {

class ProcessVariation final
{
 public:
  ProcessVariation() = default;
  ~ProcessVariation() = default;

  void set_layout_data(const LayoutData* v) { layout_data_ = v; }
  void set_net_environments(const std::vector<NetEnvironment>* v) { net_environments_ = v; }
  void set_corner_net_etch_pools(CornerNetPool<NetEtchProfile>* v) { corner_net_etch_pools_ = v; }
  void set_layer_table(const LayerTable* v) { layer_table_ = v; }
  void set_topo_pool(const TopoPool* v) { topo_pool_ = v; }
  void set_corner_data(const std::vector<RCXData::CornerData>* v) { corner_data_ = v; }

  Size corner_num() const { return corner_num_; }

  // other built data
  const MetalDensity* metal_density() const { return &metal_density_; }

  // entry points
  void reset();
  bool buildEtchProfiles();

  ProcessVariation(const ProcessVariation&) = delete;
  ProcessVariation(ProcessVariation&&) = delete;
  auto operator=(const ProcessVariation&) -> ProcessVariation& = delete;
  auto operator=(ProcessVariation&&) -> ProcessVariation& = delete;

 private:
  void initMetalDensity();
  void initEtchIntervals();

  // set from outside
  const LayoutData* layout_data_{nullptr};
  const std::vector<NetEnvironment>* net_environments_{nullptr};
  CornerNetPool<NetEtchProfile>* corner_net_etch_pools_{nullptr};
  const LayerTable* layer_table_{nullptr};
  const TopoPool* topo_pool_{nullptr};
  const std::vector<RCXData::CornerData>* corner_data_{nullptr};

  // built here
  MetalDensity metal_density_;

  Size corner_num_{0};

  Size net_num_{0};
};

} // namespace ircx
