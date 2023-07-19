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

#include "LayerCoord.hpp"
#include "RTAPI.hpp"
#include "VRSourceType.hpp"

namespace irt {

class VRGCell
{
 public:
  VRGCell() = default;
  ~VRGCell() = default;
  // getter
  PlanarRect& get_real_rect() { return _real_rect; }
  std::map<VRSourceType, std::map<irt_int, std::vector<LayerRect>>>& get_source_net_rect_map() { return _source_net_rect_map; }
  std::map<VRSourceType, void*>& get_source_region_query_map() { return _source_region_query_map; }
  // setter
  void set_real_rect(const PlanarRect& real_rect) { _real_rect = real_rect; }
  // function
  void addRect(VRSourceType vr_source_type, irt_int net_idx, const LayerRect& rect)
  {
    _source_net_rect_map[vr_source_type][net_idx].push_back(rect);
    RTAPI_INST.addEnvRectList(_source_region_query_map[vr_source_type], rect);
  }

 private:
  PlanarRect _real_rect;
  /**
   * VRSourceType::kBlockage 存储blockage
   * VRSourceType::kPanelResult 存储net布线的结果
   */
  std::map<VRSourceType, std::map<irt_int, std::vector<LayerRect>>> _source_net_rect_map;
  std::map<VRSourceType, void*> _source_region_query_map;
};

}  // namespace irt
