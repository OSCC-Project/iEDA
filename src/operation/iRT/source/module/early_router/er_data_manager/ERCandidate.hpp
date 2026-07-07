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

#include "PlanarCoord.hpp"
#include "RTHeader.hpp"
#include "Segment.hpp"

namespace irt {

class ERCandidate
{
 public:
  ERCandidate() = default;
  ERCandidate(int32_t topo_idx, const std::vector<Segment<PlanarCoord>>& routing_segment_list, int32_t total_corner_num, int32_t total_wire_length,
              bool is_path_blocked, double total_overflow_cost)
  {
    _topo_idx = topo_idx;
    _routing_segment_list = routing_segment_list;
    _total_corner_num = total_corner_num;
    _total_wire_length = total_wire_length;
    _is_path_blocked = is_path_blocked;
    _total_overflow_cost = total_overflow_cost;
  }
  ~ERCandidate() = default;
  // getter
  int32_t get_topo_idx() const { return _topo_idx; }
  std::vector<Segment<PlanarCoord>>& get_routing_segment_list() { return _routing_segment_list; }
  int32_t get_total_corner_num() const { return _total_corner_num; }
  int32_t get_total_wire_length() const { return _total_wire_length; }
  bool get_is_path_blocked() const { return _is_path_blocked; }
  double get_total_overflow_cost() const { return _total_overflow_cost; }
  // setter
  void set_topo_idx(const int32_t topo_idx) { _topo_idx = topo_idx; }
  void set_routing_segment_list(const std::vector<Segment<PlanarCoord>>& routing_segment_list) { _routing_segment_list = routing_segment_list; }
  void set_total_corner_num(const int32_t total_corner_num) { _total_corner_num = total_corner_num; }
  void set_total_wire_length(const int32_t total_wire_length) { _total_wire_length = total_wire_length; }
  void set_is_path_blocked(const bool is_path_blocked) { _is_path_blocked = is_path_blocked; }
  void set_total_overflow_cost(const double total_overflow_cost) { _total_overflow_cost = total_overflow_cost; }
  // function

 private:
  int32_t _topo_idx = -1;
  std::vector<Segment<PlanarCoord>> _routing_segment_list;
  int32_t _total_corner_num = 0;
  int32_t _total_wire_length = 0;
  bool _is_path_blocked = false;
  double _total_overflow_cost = 0.0;
};

}  // namespace irt
