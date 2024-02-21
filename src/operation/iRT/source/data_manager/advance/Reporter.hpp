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
  // 

  // ir
  std::map<std::string, irt_int> ir_routing_supply_map;
  std::map<std::string, irt_int> ir_routing_demand_map;
  std::map<std::string, double> ir_routing_wire_length_map;
  std::map<std::string, irt_int> ir_cut_via_num_map;
  std::map<std::string, std::vector<double>> ir_net_timing_map;
  // gr
  // dr
  std::map<std::string, std::vector<double>> dr_routing_wire_length_map;
  std::map<std::string, std::vector<double>> dr_cut_via_num_map;
  std::map<std::string, std::vector<double>> dr_net_timing_map;
  /////////////////////////////////////////////
};

}  // namespace irt
