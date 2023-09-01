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
#include "VRSourceType.hpp"

namespace irt {

class VRModelStat
{
 public:
  VRModelStat() = default;
  ~VRModelStat() = default;
  // getter
  std::map<irt_int, double>& get_routing_wire_length_map() { return _routing_wire_length_map; }
  std::map<irt_int, double>& get_routing_prefer_wire_length_map() { return _routing_prefer_wire_length_map; }
  std::map<irt_int, double>& get_routing_nonprefer_wire_length_map() { return _routing_nonprefer_wire_length_map; }
  std::map<irt_int, irt_int>& get_routing_patch_number_map() { return _routing_patch_number_map; }
  std::map<irt_int, irt_int>& get_cut_via_number_map() { return _cut_via_number_map; }
  std::vector<double>& get_resource_overflow_list() { return _resource_overflow_list; }
  std::map<VRSourceType, std::map<std::string, irt_int>>& get_source_drc_number_map() { return _source_drc_number_map; }
  std::map<std::string, irt_int>& get_drc_number_map() { return _drc_number_map; }
  std::map<std::string, irt_int>& get_source_number_map() { return _source_number_map; }
  double get_total_wire_length() { return _total_wire_length; }
  double get_total_prefer_wire_length() { return _total_prefer_wire_length; }
  double get_total_nonprefer_wire_length() { return _total_nonprefer_wire_length; }
  irt_int get_total_patch_number() { return _total_patch_number; }
  irt_int get_total_via_number() { return _total_via_number; }
  irt_int get_total_drc_number() { return _total_drc_number; }
  // setter
  void set_total_wire_length(const double total_wire_length) { _total_wire_length = total_wire_length; }
  void set_total_prefer_wire_length(const double total_prefer_wire_length) { _total_prefer_wire_length = total_prefer_wire_length; }
  void set_total_nonprefer_wire_length(const double total_nonprefer_wire_length)
  {
    _total_nonprefer_wire_length = total_nonprefer_wire_length;
  }
  void set_total_patch_number(const irt_int total_patch_number) { _total_patch_number = total_patch_number; }
  void set_total_via_number(const irt_int total_via_number) { _total_via_number = total_via_number; }
  void set_total_drc_number(const irt_int total_drc_number) { _total_drc_number = total_drc_number; }
  // function

 private:
  std::map<irt_int, double> _routing_wire_length_map;
  std::map<irt_int, double> _routing_prefer_wire_length_map;
  std::map<irt_int, double> _routing_nonprefer_wire_length_map;
  std::map<irt_int, irt_int> _routing_patch_number_map;
  std::map<irt_int, irt_int> _cut_via_number_map;
  std::vector<double> _resource_overflow_list;
  std::map<VRSourceType, std::map<std::string, irt_int>> _source_drc_number_map;
  std::map<std::string, irt_int> _drc_number_map;
  std::map<std::string, irt_int> _source_number_map;
  double _total_wire_length = 0;
  double _total_prefer_wire_length = 0;
  double _total_nonprefer_wire_length = 0;
  irt_int _total_patch_number = 0;
  irt_int _total_via_number = 0;
  irt_int _total_drc_number = 0;
};

}  // namespace irt
