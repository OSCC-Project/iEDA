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

class DRBoxStat
{
 public:
  DRBoxStat() = default;
  ~DRBoxStat() = default;
  // getter
  std::map<irt_int, double>& get_routing_wire_length_map() { return _routing_wire_length_map; }
  std::map<irt_int, irt_int>& get_cut_via_number_map() { return _cut_via_number_map; }
  std::map<irt_int, double>& get_routing_net_and_obs_violation_area_map() { return _routing_net_and_obs_violation_area_map; }
  std::map<irt_int, double>& get_routing_net_and_net_violation_area_map() { return _routing_net_and_net_violation_area_map; }
  // setter
  // function

 private:
  std::map<irt_int, double> _routing_wire_length_map;
  std::map<irt_int, irt_int> _cut_via_number_map;
  std::map<irt_int, double> _routing_net_and_obs_violation_area_map;
  std::map<irt_int, double> _routing_net_and_net_violation_area_map;
};

}  // namespace irt
