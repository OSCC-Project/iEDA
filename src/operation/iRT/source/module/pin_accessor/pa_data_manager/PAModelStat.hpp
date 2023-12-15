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

#include "PASourceType.hpp"
#include "RTU.hpp"

namespace irt {

class PAModelStat
{
 public:
  PAModelStat() = default;
  ~PAModelStat() = default;
  // getter
  std::map<AccessPointType, irt_int>& get_type_pin_num_map() { return _type_pin_num_map; }
  std::map<irt_int, irt_int>& get_routing_port_num_map() { return _routing_port_num_map; }
  std::map<irt_int, irt_int>& get_routing_access_point_num_map() { return _routing_access_point_num_map; }
  std::map<PASourceType, std::map<irt_int, std::map<std::string, std::vector<ViolationInfo>>>>& get_source_routing_drc_violation_map()
  {
    return _source_routing_drc_violation_map;
  }
  std::map<PASourceType, std::map<irt_int, std::map<std::string, std::vector<ViolationInfo>>>>& get_source_cut_drc_violation_map()
  {
    return _source_cut_drc_violation_map;
  }
  irt_int get_total_pin_num() { return _total_pin_num; }
  irt_int get_total_port_num() { return _total_port_num; }
  irt_int get_total_access_point_num() { return _total_access_point_num; }
  irt_int get_total_drc_number() { return _total_drc_number; }
  // setter
  void set_total_pin_num(const irt_int total_pin_num) { _total_pin_num = total_pin_num; }
  void set_total_port_num(const irt_int total_port_num) { _total_port_num = total_port_num; }
  void set_total_access_point_num(const irt_int total_access_point_num) { _total_access_point_num = total_access_point_num; }
  void set_total_drc_number(const irt_int total_drc_number) { _total_drc_number = total_drc_number; }
  // function

 private:
  std::map<AccessPointType, irt_int> _type_pin_num_map;
  std::map<irt_int, irt_int> _routing_port_num_map;
  std::map<irt_int, irt_int> _routing_access_point_num_map;
  std::map<PASourceType, std::map<irt_int, std::map<std::string, std::vector<ViolationInfo>>>> _source_routing_drc_violation_map;
  std::map<PASourceType, std::map<irt_int, std::map<std::string, std::vector<ViolationInfo>>>> _source_cut_drc_violation_map;
  irt_int _total_pin_num = 0;
  irt_int _total_port_num = 0;
  irt_int _total_access_point_num = 0;
  irt_int _total_drc_number = 0;
};

}  // namespace irt
