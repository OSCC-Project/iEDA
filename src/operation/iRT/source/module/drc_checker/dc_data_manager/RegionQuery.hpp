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

#include "LayerRect.hpp"
#include "RTU.hpp"

namespace irt {

class RegionQuery
{
 public:
  RegionQuery() = default;
  ~RegionQuery()
  {
    _routing_info_rect_map.clear();
    _cut_info_rect_map.clear();
  }
  // getter
  std::map<irt_int, std::map<BaseInfo, std::set<LayerRect, CmpLayerRectByXASC>, CmpBaseInfo>>& get_routing_info_rect_map()
  {
    return _routing_info_rect_map;
  }
  std::map<irt_int, std::map<BaseInfo, std::set<LayerRect, CmpLayerRectByXASC>, CmpBaseInfo>>& get_cut_info_rect_map()
  {
    return _cut_info_rect_map;
  }
  //////////////////////////////////////////////////////////////////////////
  /////////////////////////////// drc-check ////////////////////////////////
  std::map<irt_int, std::map<irt_int, std::map<LayerRect, std::vector<BaseShape*>, CmpLayerRectByLayerASC>>>& get_routing_net_shape_map()
  {
    return _routing_net_shape_map;
  }
  std::map<irt_int, std::map<irt_int, std::map<LayerRect, std::vector<BaseShape*>, CmpLayerRectByLayerASC>>>& get_cut_net_shape_map()
  {
    return _cut_net_shape_map;
  }
  BaseRegion& get_base_region() { return _base_region; }
  std::map<irt_int, bgi::rtree<std::pair<BGRectInt, BaseShape*>, bgi::quadratic<16>>>& get_routing_region_map()
  {
    return _base_region.get_routing_region_map();
  }
  std::map<irt_int, bgi::rtree<std::pair<BGRectInt, BaseShape*>, bgi::quadratic<16>>>& get_cut_region_map()
  {
    return _base_region.get_cut_region_map();
  }
  /////////////////////////////// drc-check ////////////////////////////////
  //////////////////////////////////////////////////////////////////////////

 private:
  std::map<irt_int, std::map<BaseInfo, std::set<LayerRect, CmpLayerRectByXASC>, CmpBaseInfo>> _routing_info_rect_map;
  std::map<irt_int, std::map<BaseInfo, std::set<LayerRect, CmpLayerRectByXASC>, CmpBaseInfo>> _cut_info_rect_map;
  //////////////////////////////////////////////////////////////////////////
  /////////////////////////////// drc-check ////////////////////////////////
  std::map<irt_int, std::map<irt_int, std::map<LayerRect, std::vector<BaseShape*>, CmpLayerRectByLayerASC>>> _routing_net_shape_map;
  std::map<irt_int, std::map<irt_int, std::map<LayerRect, std::vector<BaseShape*>, CmpLayerRectByLayerASC>>> _cut_net_shape_map;
  BaseRegion _base_region;
  /////////////////////////////// drc-check ////////////////////////////////
  //////////////////////////////////////////////////////////////////////////
};

}  // namespace irt
