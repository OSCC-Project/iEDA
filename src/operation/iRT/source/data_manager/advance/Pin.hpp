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
#include "EXTLayerRect.hpp"
#include "PlanarCoord.hpp"
#include "RTHeader.hpp"

namespace irt {

class Pin
{
 public:
  Pin() = default;
  ~Pin() = default;
  // getter
  int32_t get_pin_idx() const { return _pin_idx; }
  std::string& get_pin_name() { return _pin_name; }
  std::vector<EXTLayerRect>& get_routing_shape_list() { return _routing_shape_list; }
  std::vector<EXTLayerRect>& get_cut_shape_list() { return _cut_shape_list; }
  bool get_is_driven() const { return _is_driven; }
  AccessPoint& get_origin_access_point() { return _origin_access_point; }
  AccessPoint& get_extend_access_point() { return _extend_access_point; }
  AccessPoint& get_access_point() { return _access_point; }
  // setter
  void set_pin_idx(const int32_t pin_idx) { _pin_idx = pin_idx; }
  void set_pin_name(const std::string& pin_name) { _pin_name = pin_name; }
  void set_routing_shape_list(const std::vector<EXTLayerRect>& routing_shape_list) { _routing_shape_list = routing_shape_list; }
  void set_cut_shape_list(const std::vector<EXTLayerRect>& cut_shape_list) { _cut_shape_list = cut_shape_list; }
  void set_is_driven(const bool is_driven) { _is_driven = is_driven; }
  void set_origin_access_point(const AccessPoint& origin_access_point) { _origin_access_point = origin_access_point; }
  void set_extend_access_point(const AccessPoint& extend_access_point) { _extend_access_point = extend_access_point; }
  void set_access_point(const AccessPoint& access_point) { _access_point = access_point; }
  // function

 private:
  int32_t _pin_idx = -1;
  std::string _pin_name;
  std::vector<EXTLayerRect> _routing_shape_list;
  std::vector<EXTLayerRect> _cut_shape_list;
  bool _is_driven = false;
  AccessPoint _origin_access_point;
  AccessPoint _extend_access_point;
  AccessPoint _access_point;
};

}  // namespace irt
