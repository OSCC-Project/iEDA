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

#include "AccessPoint.hpp"
#include "Violation.hpp"

namespace irt {

class GCell
{
 public:
  GCell() = default;
  ~GCell() = default;
  // getter
  std::map<bool, std::map<irt_int, std::map<irt_int, std::set<EXTLayerRect*>>>>& get_type_layer_net_fixed_rect_map()
  {
    return _type_layer_net_fixed_rect_map;
  }
  std::map<irt_int, std::set<AccessPoint*>>& get_net_access_point_map() { return _net_access_point_map; }
  std::map<irt_int, std::map<Orientation, irt_int>>& get_routing_orien_supply_map() { return _routing_orien_supply_map; }
  std::map<irt_int, std::set<Segment<LayerCoord>*>>& get_net_result_map() { return _net_result_map; }
  std::map<irt_int, std::set<EXTLayerRect*>>& get_net_patch_map() { return _net_patch_map; }
  std::set<Violation*>& get_violation_set() { return _violation_set; }
  // setter
  void set_type_layer_net_fixed_rect_map(
      const std::map<bool, std::map<irt_int, std::map<irt_int, std::set<EXTLayerRect*>>>>& type_layer_net_fixed_rect_map)
  {
    _type_layer_net_fixed_rect_map = type_layer_net_fixed_rect_map;
  }
  void set_net_access_point_map(const std::map<irt_int, std::set<AccessPoint*>>& net_access_point_map)
  {
    _net_access_point_map = net_access_point_map;
  }
  void set_routing_orien_supply_map(const std::map<irt_int, std::map<Orientation, irt_int>>& routing_orien_supply_map)
  {
    _routing_orien_supply_map = routing_orien_supply_map;
  }
  void set_net_result_map(const std::map<irt_int, std::set<Segment<LayerCoord>*>>& net_result_map) { _net_result_map = net_result_map; }
  void set_net_patch_map(const std::map<irt_int, std::set<EXTLayerRect*>>& net_patch_map) { _net_patch_map = net_patch_map; }
  void set_violation_set(const std::set<Violation*>& violation_set) { _violation_set = violation_set; }
  // function

 private:
  // blockage & pin_shape 如果是routing则true，cut则false
  std::map<bool, std::map<irt_int, std::map<irt_int, std::set<EXTLayerRect*>>>> _type_layer_net_fixed_rect_map;
  // access point 只有routing层有
  std::map<irt_int, std::set<AccessPoint*>> _net_access_point_map;
  // global supply 只有routing层有
  std::map<irt_int, std::map<Orientation, irt_int>> _routing_orien_supply_map;
  // routing result
  std::map<irt_int, std::set<Segment<LayerCoord>*>> _net_result_map;
  // patch shape 只有routing层有
  std::map<irt_int, std::set<EXTLayerRect*>> _net_patch_map;
  // violation region
  std::set<Violation*> _violation_set;
};

}  // namespace irt
