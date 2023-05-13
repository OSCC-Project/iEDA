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

#include "RTU.hpp"

namespace irt {

class PAModelStat
{
 public:
  PAModelStat() = default;
  ~PAModelStat() = default;
  // getter
  irt_int get_total_pin_num() { return _total_pin_num; }
  irt_int get_track_grid_pin_num() { return _track_grid_pin_num; }
  irt_int get_track_center_pin_num() { return _track_center_pin_num; }
  irt_int get_shape_center_pin_num() { return _shape_center_pin_num; }
  irt_int get_total_port_num() { return _total_port_num; }
  std::map<irt_int, irt_int>& get_layer_port_num_map() { return _layer_port_num_map; }
  irt_int get_total_access_point_num() { return _total_access_point_num; }
  std::map<irt_int, irt_int>& get_layer_access_point_num_map() { return _layer_access_point_num_map; }
  // setter
  // function
  void addTotalPinNum(const irt_int pin_num) { _total_pin_num += pin_num; }
  void addTrackGridPinNum(const irt_int pin_num) { _track_grid_pin_num += pin_num; }
  void addTrackCenterPinNum(const irt_int pin_num) { _track_center_pin_num += pin_num; }
  void addShapeCenterPinNum(const irt_int pin_num) { _shape_center_pin_num += pin_num; }
  void addTotalPortNum(const irt_int port_num) { _total_port_num += port_num; }
  void addTotalAccessPointNum(const irt_int access_point) { _total_access_point_num += access_point; }

 private:
  irt_int _total_pin_num = 0;
  irt_int _track_grid_pin_num = 0;
  irt_int _track_center_pin_num = 0;
  irt_int _shape_center_pin_num = 0;
  irt_int _total_port_num = 0;
  std::map<irt_int, irt_int> _layer_port_num_map;
  irt_int _total_access_point_num = 0;
  std::map<irt_int, irt_int> _layer_access_point_num_map;
};

}  // namespace irt
