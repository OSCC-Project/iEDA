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

class TAModelStat
{
 public:
  TAModelStat() = default;
  ~TAModelStat() = default;
  // getter
  double get_total_wire_length() { return _total_wire_length; }
  std::map<irt_int, double>& get_layer_wire_length_map() { return _layer_wire_length_map; }
  double get_total_net_and_net_violation_area() { return _total_net_and_net_violation_area; }
  std::map<irt_int, double>& get_layer_net_and_net_violation_area_map() { return _layer_net_and_net_violation_area_map; }
  double get_total_net_and_obs_violation_area() { return _total_net_and_obs_violation_area; }
  std::map<irt_int, double>& get_layer_net_and_obs_violation_area_map() { return _layer_net_and_obs_violation_area_map; }
  // setter
  // function
  void addTotalWireLength(const double wire_length) { _total_wire_length += wire_length; }
  void addTotalNetAndNetViolation(const double violation_area) { _total_net_and_net_violation_area += violation_area; }
  void addTotalNetAndObsViolation(const double violation_area) { _total_net_and_obs_violation_area += violation_area; }

 private:
  double _total_wire_length = 0;
  std::map<irt_int, double> _layer_wire_length_map;
  double _total_net_and_net_violation_area = 0;
  std::map<irt_int, double> _layer_net_and_net_violation_area_map;
  double _total_net_and_obs_violation_area = 0;
  std::map<irt_int, double> _layer_net_and_obs_violation_area_map;
};

}  // namespace irt
