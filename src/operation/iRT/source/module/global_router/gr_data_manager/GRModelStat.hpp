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

class GRModelStat
{
 public:
  GRModelStat() = default;
  ~GRModelStat() = default;
  // getter
  std::map<irt_int, double>& get_routing_wire_length_map() { return _routing_wire_length_map; }
  std::map<irt_int, irt_int>& get_cut_via_number_map() { return _cut_via_number_map; }
  std::map<irt_int, std::vector<double>>& get_layer_resource_overflow_map() { return _layer_resource_overflow_map; }
  std::map<irt_int, std::vector<double>>& get_layer_access_overflow_map() { return _layer_access_overflow_map; }
  double get_total_wire_length() const { return _total_wire_length; }
  irt_int get_total_via_number() const { return _total_via_number; }
  irt_int get_total_resource_overflow_number() const { return _total_resource_overflow_number; }
  irt_int get_total_access_overflow_number() const { return _total_access_overflow_number; }
  double get_max_resource_overflow() { return _max_resource_overflow; }
  double get_max_access_overflow() { return _max_access_overflow; }
  // setter
  void set_total_wire_length(const double total_wire_length) { _total_wire_length = total_wire_length; }
  void set_total_via_number(const irt_int total_via_number) { _total_via_number = total_via_number; }
  void set_total_resource_overflow_number(const irt_int total_resource_overflow_number)
  {
    _total_resource_overflow_number = total_resource_overflow_number;
  }
  void set_total_access_overflow_number(const irt_int total_access_overflow_number)
  {
    _total_access_overflow_number = total_access_overflow_number;
  }
  void set_max_resource_overflow(const double max_resource_overflow) { _max_resource_overflow = max_resource_overflow; }
  void set_max_access_overflow(const double max_access_overflow) { _max_access_overflow = max_access_overflow; }
  // function

 private:
  std::map<irt_int, double> _routing_wire_length_map;
  std::map<irt_int, irt_int> _cut_via_number_map;
  std::map<irt_int, std::vector<double>> _layer_resource_overflow_map;
  std::map<irt_int, std::vector<double>> _layer_access_overflow_map;
  double _total_wire_length = 0;
  irt_int _total_via_number = 0;
  irt_int _total_resource_overflow_number = 0;
  irt_int _total_access_overflow_number = 0;
  double _max_resource_overflow = 0;
  double _max_access_overflow = 0;
};

}  // namespace irt
