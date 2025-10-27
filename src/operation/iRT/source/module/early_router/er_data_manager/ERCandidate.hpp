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
  ERCandidate(int32_t topo_idx, const std::vector<Segment<PlanarCoord>>& routing_segment_list, int32_t total_length, bool is_blocked, double total_cost)
  {
    _topo_idx = topo_idx;
    _routing_segment_list = routing_segment_list;
    _total_length = total_length;
    _is_blocked = is_blocked;
    _total_cost = total_cost;
  }
  ~ERCandidate() = default;
  // getter
  int32_t get_topo_idx() const { return _topo_idx; }
  std::vector<Segment<PlanarCoord>>& get_routing_segment_list() { return _routing_segment_list; }
  int32_t get_total_length() const { return _total_length; }
  bool get_is_blocked() const { return _is_blocked; }
  double get_total_cost() const { return _total_cost; }
  // setter
  void set_topo_idx(const int32_t topo_idx) { _topo_idx = topo_idx; }
  void set_routing_segment_list(const std::vector<Segment<PlanarCoord>>& routing_segment_list) { _routing_segment_list = routing_segment_list; }
  void set_total_length(const int32_t total_length) { _total_length = total_length; }
  void set_is_blocked(const bool is_blocked) { _is_blocked = is_blocked; }
  void set_total_cost(const double total_cost) { _total_cost = total_cost; }
  // function

 private:
  int32_t _topo_idx = -1;
  std::vector<Segment<PlanarCoord>> _routing_segment_list;
  int32_t _total_length = 0;
  bool _is_blocked = false;
  double _total_cost = 0.0;
};

}  // namespace irt
