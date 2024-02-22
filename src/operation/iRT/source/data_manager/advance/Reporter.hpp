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

class Reporter
{
 public:
  Reporter() = default;
  ~Reporter() = default;
  /////////////////////////////////////////////
  // **********   RT     ********** //
  // **********   PinAccessor     ********** //
  std::map<irt_int, irt_int> pa_routing_access_point_map;
  std::map<AccessPointType, irt_int> pa_type_access_point_map;
  irt_int pa_total_access_point_num = 0;
  // ********     SupplyAnalyzer    ******** //
  std::map<irt_int, irt_int> sa_routing_supply_map;
  irt_int sa_total_supply_num = 0;
  // **********   InitialRouter    ********** //
  std::map<irt_int, irt_int> ir_routing_demand_map;
  irt_int ir_total_demand_num = 0;
  std::map<irt_int, irt_int> ir_routing_overflow_map;
  irt_int ir_total_overflow_num = 0;
  std::map<irt_int, double> ir_routing_wire_length_map;
  double ir_total_wire_length = 0;
  std::map<irt_int, irt_int> ir_cut_via_num_map;
  irt_int ir_total_via_num = 0;
  std::unordered_map<std::string, std::vector<double>> ir_timing;
  // **********   GlobalRouter    ********** //
  // **********   TrackAssigner   ********** //
  std::map<irt_int, double> ta_routing_wire_length_map;
  double ta_total_wire_length = 0;
  std::map<irt_int, irt_int> ta_routing_violation_map;
  irt_int ta_total_violation_num = 0;
  // **********  DetailedRouter   ********** //
  std::map<irt_int, double> dr_routing_wire_length_map;
  double dr_total_wire_length = 0;
  std::map<irt_int, irt_int> dr_cut_via_num_map;
  irt_int dr_total_via_num = 0;
  std::map<irt_int, double> dr_routing_patch_map;
  double dr_total_patch = 0;
  std::map<irt_int, irt_int> dr_routing_violation_map;
  irt_int dr_total_violation_num = 0;
  std::unordered_map<std::string, std::vector<double>> dr_timing;
  /////////////////////////////////////////////
};

}  // namespace irt
