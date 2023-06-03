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

#include <map>
#include <string>
#include <vector>

#include "RTUtil.hpp"

namespace irt {

class EGRStat
{
 public:
  EGRStat() = default;
  ~EGRStat() = default;
  // getter
  std::vector<std::map<irt_int, irt_int, std::greater<irt_int>>>& get_overflow_map_list() { return _overflow_map_list; }
  std::map<irt_int, irt_int, std::greater<irt_int>>& get_total_overflow_map() { return _total_overflow_map; }
  irt_int& get_total_track_overflow() { return _total_track_overflow; }
  std::vector<double>& get_wire_length_list() { return _wire_length_list; }
  std::vector<irt_int>& get_via_num_list() { return _via_num_list; }
  double& get_total_wire_length() { return _total_wire_length; }
  irt_int& get_total_via_num() { return _total_via_num; }

  // setter
  void set_overflow_map_list(const std::vector<std::map<irt_int, irt_int, std::greater<irt_int>>>& overflow_map_list)
  {
    _overflow_map_list = overflow_map_list;
  }
  void set_total_overflow_map(const std::map<irt_int, irt_int, std::greater<irt_int>>& total_overflow_map)
  {
    _total_overflow_map = total_overflow_map;
  }
  void set_total_track_overflow(const irt_int& total_track_overflow) { _total_track_overflow = total_track_overflow; }
  void set_wire_length_list(const std::vector<double>& wire_length_list) { _wire_length_list = wire_length_list; }
  void set_via_num_list(const std::vector<irt_int>& via_num_list) { _via_num_list = via_num_list; }
  void set_total_wire_length(const double& total_wire_length) { _total_wire_length = total_wire_length; }
  void set_total_via_num(const irt_int& total_via_num) { _total_via_num = total_via_num; }

 private:
  std::vector<std::map<irt_int, irt_int, std::greater<irt_int>>> _overflow_map_list;
  std::map<irt_int, irt_int, std::greater<irt_int>> _total_overflow_map;
  irt_int _total_track_overflow = 0;
  std::vector<double> _wire_length_list;
  std::vector<irt_int> _via_num_list;
  double _total_wire_length = 0;
  irt_int _total_via_num = 0;
};

}  // namespace irt
