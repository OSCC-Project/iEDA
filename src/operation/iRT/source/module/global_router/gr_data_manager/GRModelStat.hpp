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
  double get_total_wire_length() const { return _total_wire_length; }
  std::map<irt_int, irt_int>& get_cut_via_number_map() { return _cut_via_number_map; }
  irt_int get_total_via_number() const { return _total_via_number; }
  std::vector<double>& get_wire_overflow_list() { return _wire_overflow_list; }
  double get_max_wire_overflow() { return _max_wire_overflow; }
  std::vector<double>& get_via_overflow_list() { return _via_overflow_list; }
  double get_max_via_overflow() { return _max_via_overflow; }
  // setter
  void set_total_wire_length(const double total_wire_length) { _total_wire_length = total_wire_length; }
  void set_total_via_number(const double total_via_number) { _total_via_number = total_via_number; }
  void set_max_wire_overflow(const double max_wire_overflow) { _max_wire_overflow = max_wire_overflow; }
  void set_max_via_overflow(const double max_via_overflow) { _max_via_overflow = max_via_overflow; }
  // function

 private:
  std::map<irt_int, double> _routing_wire_length_map;
  double _total_wire_length = 0;
  std::map<irt_int, irt_int> _cut_via_number_map;
  irt_int _total_via_number = 0;
  std::vector<double> _wire_overflow_list;
  double _max_wire_overflow = 0;
  std::vector<double> _via_overflow_list;
  double _max_via_overflow = 0;
};

}  // namespace irt
