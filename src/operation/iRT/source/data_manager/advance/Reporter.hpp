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

#include "RTHeader.hpp"

namespace irt {

class Reporter
{
 public:
  Reporter() = default;
  ~Reporter() = default;
  /////////////////////////////////////////////
  // **********   RT     ********** //
  // **********   PinAccessor     ********** //
  std::map<int32_t, int32_t> pa_routing_access_point_map;
  std::map<AccessPointType, int32_t> pa_type_access_point_map;
  int32_t pa_total_access_point_num = 0;
  // ********     SupplyAnalyzer    ******** //
  std::map<int32_t, int32_t> sa_routing_supply_map;
  int32_t sa_total_supply_num = 0;
  // **********   InitialRouter    ********** //
  std::map<int32_t, int32_t> ir_routing_demand_map;
  int32_t ir_total_demand_num = 0;
  std::map<int32_t, int32_t> ir_routing_overflow_map;
  int32_t ir_total_overflow_num = 0;
  std::map<int32_t, double> ir_routing_wire_length_map;
  double ir_total_wire_length = 0;
  std::map<int32_t, int32_t> ir_cut_via_num_map;
  int32_t ir_total_via_num = 0;
  std::map<std::string, std::vector<double>> ir_timing;
  // **********   GlobalRouter    ********** //
  std::map<int32_t, int32_t> gr_routing_demand_map;
  int32_t gr_total_demand_num = 0;
  std::map<int32_t, int32_t> gr_routing_overflow_map;
  int32_t gr_total_overflow_num = 0;
  std::map<int32_t, double> gr_routing_wire_length_map;
  double gr_total_wire_length = 0;
  std::map<int32_t, int32_t> gr_cut_via_num_map;
  int32_t gr_total_via_num = 0;
  std::map<std::string, std::vector<double>> gr_timing;
  // **********   TrackAssigner   ********** //
  std::map<int32_t, double> ta_routing_wire_length_map;
  double ta_total_wire_length = 0;
  std::map<int32_t, int32_t> ta_routing_violation_map;
  int32_t ta_total_violation_num = 0;
  // **********  DetailedRouter   ********** //
  std::map<int32_t, double> dr_routing_wire_length_map;
  double dr_total_wire_length = 0;
  std::map<int32_t, int32_t> dr_cut_via_num_map;
  int32_t dr_total_via_num = 0;
  std::map<int32_t, double> dr_routing_patch_map;
  double dr_total_patch = 0;
  std::map<int32_t, int32_t> dr_routing_violation_map;
  int32_t dr_total_violation_num = 0;
  std::map<std::string, std::vector<double>> dr_timing;
  /////////////////////////////////////////////
};

}  // namespace irt
