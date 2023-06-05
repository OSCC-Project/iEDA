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

class VRModelStat
{
 public:
  VRModelStat() = default;
  ~VRModelStat() = default;
  // getter
  std::map<irt_int, double>& get_routing_wire_length_map() { return _routing_wire_length_map; }
  std::map<irt_int, irt_int>& get_cut_via_number_map() { return _cut_via_number_map; }
  std::map<irt_int, double>& get_routing_net_and_obs_violation_area_map() { return _routing_net_and_obs_violation_area_map; }
  std::map<irt_int, double>& get_routing_net_and_net_violation_area_map() { return _routing_net_and_net_violation_area_map; }
  double get_total_wire_length() { return _total_wire_length; }
  irt_int get_total_via_number() { return _total_via_number; }
  double get_total_net_and_obs_violation_area() { return _total_net_and_obs_violation_area; }
  double get_total_net_and_net_violation_area() { return _total_net_and_net_violation_area; }
  // setter
  void set_total_wire_length(const double total_wire_length) { _total_wire_length = total_wire_length; }
  void set_total_via_number(const irt_int total_via_number) { _total_via_number = total_via_number; }
  void set_total_net_and_obs_violation_area(const double total_net_and_obs_violation_area)
  {
    _total_net_and_obs_violation_area = total_net_and_obs_violation_area;
  }
  void set_total_net_and_net_violation_area(const double total_net_and_net_violation_area)
  {
    _total_net_and_net_violation_area = total_net_and_net_violation_area;
  }
  // function

 private:
  std::map<irt_int, double> _routing_wire_length_map;
  std::map<irt_int, irt_int> _cut_via_number_map;
  std::map<irt_int, double> _routing_net_and_obs_violation_area_map;
  std::map<irt_int, double> _routing_net_and_net_violation_area_map;
  double _total_wire_length = 0;
  irt_int _total_via_number = 0;
  double _total_net_and_obs_violation_area = 0;
  double _total_net_and_net_violation_area = 0;
};

}  // namespace irt
