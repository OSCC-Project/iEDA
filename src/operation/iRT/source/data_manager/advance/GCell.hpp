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

class GCell : public PlanarRect
{
 public:
  GCell() = default;
  ~GCell() = default;
  // getter
  std::map<bool, std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>>>& get_type_layer_net_fixed_rect_map() { return _type_layer_net_fixed_rect_map; }
  std::map<int32_t, std::set<AccessPoint*, CmpAccessPoint>>& get_net_access_point_map() { return _net_access_point_map; }
  std::map<int32_t, std::map<int32_t, std::set<Segment<LayerCoord>*>>>& get_net_pin_access_result_map() { return _net_pin_access_result_map; }
  std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>>& get_net_pin_access_patch_map() { return _net_pin_access_patch_map; }
  double get_boundary_wire_unit() const { return _boundary_wire_unit; }
  double get_internal_wire_unit() const { return _internal_wire_unit; }
  double get_internal_via_unit() const { return _internal_via_unit; }
  std::map<int32_t, std::map<Orientation, int32_t>>& get_routing_orient_supply_map() { return _routing_orient_supply_map; }
  std::map<int32_t, std::map<int32_t, std::set<Orientation>>>& get_routing_ignore_net_orient_map() { return _routing_ignore_net_orient_map; }
  std::map<int32_t, std::set<Segment<LayerCoord>*>>& get_net_global_result_map() { return _net_global_result_map; }
  std::map<int32_t, std::set<Segment<LayerCoord>*>>& get_net_detailed_result_map() { return _net_detailed_result_map; }
  std::map<int32_t, std::set<EXTLayerRect*>>& get_net_detailed_patch_map() { return _net_detailed_patch_map; }
  std::set<Violation*, CmpViolation>& get_violation_set() { return _violation_set; }
  // setter
  void set_type_layer_net_fixed_rect_map(const std::map<bool, std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>>>& type_layer_net_fixed_rect_map)
  {
    _type_layer_net_fixed_rect_map = type_layer_net_fixed_rect_map;
  }
  void set_net_access_point_map(const std::map<int32_t, std::set<AccessPoint*, CmpAccessPoint>>& net_access_point_map)
  {
    _net_access_point_map = net_access_point_map;
  }
  void set_net_pin_access_result_map(const std::map<int32_t, std::map<int32_t, std::set<Segment<LayerCoord>*>>>& net_pin_access_result_map)
  {
    _net_pin_access_result_map = net_pin_access_result_map;
  }
  void set_net_pin_access_patch_map(const std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>>& net_pin_access_patch_map)
  {
    _net_pin_access_patch_map = net_pin_access_patch_map;
  }
  void set_boundary_wire_unit(const double boundary_wire_unit) { _boundary_wire_unit = boundary_wire_unit; }
  void set_internal_wire_unit(const double internal_wire_unit) { _internal_wire_unit = internal_wire_unit; }
  void set_internal_via_unit(const double internal_via_unit) { _internal_via_unit = internal_via_unit; }
  void set_routing_orient_supply_map(const std::map<int32_t, std::map<Orientation, int32_t>>& routing_orient_supply_map)
  {
    _routing_orient_supply_map = routing_orient_supply_map;
  }
  void set_routing_ignore_net_orient_map(const std::map<int32_t, std::map<int32_t, std::set<Orientation>>>& routing_ignore_net_orient_map)
  {
    _routing_ignore_net_orient_map = routing_ignore_net_orient_map;
  }
  void set_net_global_result_map(const std::map<int32_t, std::set<Segment<LayerCoord>*>>& net_global_result_map)
  {
    _net_global_result_map = net_global_result_map;
  }
  void set_net_detailed_result_map(const std::map<int32_t, std::set<Segment<LayerCoord>*>>& net_detailed_result_map)
  {
    _net_detailed_result_map = net_detailed_result_map;
  }
  void set_net_detailed_patch_map(const std::map<int32_t, std::set<EXTLayerRect*>>& net_detailed_patch_map)
  {
    _net_detailed_patch_map = net_detailed_patch_map;
  }
  void set_violation_set(const std::set<Violation*, CmpViolation>& violation_set) { _violation_set = violation_set; }
  // function

 private:
  // obstacle & pin_shape
  std::map<bool, std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>>> _type_layer_net_fixed_rect_map;
  // access point
  std::map<int32_t, std::set<AccessPoint*, CmpAccessPoint>> _net_access_point_map;
  // access routing result
  std::map<int32_t, std::map<int32_t, std::set<Segment<LayerCoord>*>>> _net_pin_access_result_map;
  // access routing patch
  std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>> _net_pin_access_patch_map;
  // global demand unit
  double _boundary_wire_unit = -1;
  double _internal_wire_unit = -1;
  double _internal_via_unit = -1;
  // global supply
  std::map<int32_t, std::map<Orientation, int32_t>> _routing_orient_supply_map;
  // global ignore net orient
  std::map<int32_t, std::map<int32_t, std::set<Orientation>>> _routing_ignore_net_orient_map;
  // global routing result
  std::map<int32_t, std::set<Segment<LayerCoord>*>> _net_global_result_map;
  // detailed routing result
  std::map<int32_t, std::set<Segment<LayerCoord>*>> _net_detailed_result_map;
  // detailed routing patch
  std::map<int32_t, std::set<EXTLayerRect*>> _net_detailed_patch_map;
  // violation
  std::set<Violation*, CmpViolation> _violation_set;
};

}  // namespace irt
