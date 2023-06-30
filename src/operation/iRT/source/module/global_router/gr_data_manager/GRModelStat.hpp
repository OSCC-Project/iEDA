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
  std::vector<double>& get_overflow_list() { return _overflow_list; }
  double get_total_wire_length() const { return _total_wire_length; }
  irt_int get_total_via_number() const { return _total_via_number; }
  double get_max_overflow() { return _max_overflow; }
  // setter
  void set_total_wire_length(const double total_wire_length) { _total_wire_length = total_wire_length; }
  void set_total_via_number(const irt_int total_via_number) { _total_via_number = total_via_number; }
  void set_max_overflow(const double max_overflow) { _max_overflow = max_overflow; }
  // function

 private:
  std::map<irt_int, double> _routing_wire_length_map;
  std::map<irt_int, irt_int> _cut_via_number_map;
  std::vector<double> _overflow_list;
  double _total_wire_length = 0;
  irt_int _total_via_number = 0;
  double _max_overflow = 0;
};

}  // namespace irt
