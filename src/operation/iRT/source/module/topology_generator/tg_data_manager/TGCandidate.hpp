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

struct TGCandidateCost
{
  int32_t total_corner_num = 0;
  int32_t total_wire_length = 0;
  bool is_path_blocked = false;
  double total_usage_cost = 0.0;
  double total_saturation_cost = 0.0;
  double total_hotspot_cost = 0.0;
  double total_overflow_cost = 0.0;
  double total_overflow = 0.0;
  double max_usage_ratio = 0.0;
  int32_t saturation_node_num = 0;
  int32_t hotspot_node_num = 0;
  int32_t overflow_node_num = 0;

  double getTotalCost() const { return total_usage_cost + total_saturation_cost + total_hotspot_cost + total_overflow_cost; }
};

class TGCandidate
{
 public:
  TGCandidate() = default;
  TGCandidate(int32_t topo_idx, const std::vector<Segment<PlanarCoord>>& routing_segment_list, int32_t total_corner_num, int32_t total_wire_length,
              bool is_path_blocked, double total_overflow_cost, double total_usage_cost = 0.0, double total_overflow = 0.0,
              int32_t overflow_node_num = 0)
  {
    _topo_idx = topo_idx;
    _routing_segment_list = routing_segment_list;
    _total_corner_num = total_corner_num;
    _total_wire_length = total_wire_length;
    _is_path_blocked = is_path_blocked;
    _total_usage_cost = total_usage_cost;
    _total_overflow_cost = total_overflow_cost;
    _total_overflow = total_overflow;
    _overflow_node_num = overflow_node_num;
  }
  ~TGCandidate() = default;
  // getter
  int32_t get_topo_idx() const { return _topo_idx; }
  std::vector<Segment<PlanarCoord>>& get_routing_segment_list() { return _routing_segment_list; }
  int32_t get_total_corner_num() const { return _total_corner_num; }
  int32_t get_total_wire_length() const { return _total_wire_length; }
  bool get_is_path_blocked() const { return _is_path_blocked; }
  double get_total_usage_cost() const { return _total_usage_cost; }
  double get_total_saturation_cost() const { return _total_saturation_cost; }
  double get_total_hotspot_cost() const { return _total_hotspot_cost; }
  double get_total_overflow_cost() const { return _total_overflow_cost; }
  double get_total_overflow() const { return _total_overflow; }
  double get_max_usage_ratio() const { return _max_usage_ratio; }
  int32_t get_saturation_node_num() const { return _saturation_node_num; }
  int32_t get_hotspot_node_num() const { return _hotspot_node_num; }
  int32_t get_overflow_node_num() const { return _overflow_node_num; }
  double get_total_cost() const { return _total_usage_cost + _total_saturation_cost + _total_hotspot_cost + _total_overflow_cost; }
  // setter
  void set_topo_idx(const int32_t topo_idx) { _topo_idx = topo_idx; }
  void set_routing_segment_list(const std::vector<Segment<PlanarCoord>>& routing_segment_list) { _routing_segment_list = routing_segment_list; }
  void set_total_corner_num(const int32_t total_corner_num) { _total_corner_num = total_corner_num; }
  void set_total_wire_length(const int32_t total_wire_length) { _total_wire_length = total_wire_length; }
  void set_is_path_blocked(const bool is_path_blocked) { _is_path_blocked = is_path_blocked; }
  void set_total_usage_cost(const double total_usage_cost) { _total_usage_cost = total_usage_cost; }
  void set_total_saturation_cost(const double total_saturation_cost) { _total_saturation_cost = total_saturation_cost; }
  void set_total_hotspot_cost(const double total_hotspot_cost) { _total_hotspot_cost = total_hotspot_cost; }
  void set_total_overflow_cost(const double total_overflow_cost) { _total_overflow_cost = total_overflow_cost; }
  void set_total_overflow(const double total_overflow) { _total_overflow = total_overflow; }
  void set_max_usage_ratio(const double max_usage_ratio) { _max_usage_ratio = max_usage_ratio; }
  void set_saturation_node_num(const int32_t saturation_node_num) { _saturation_node_num = saturation_node_num; }
  void set_hotspot_node_num(const int32_t hotspot_node_num) { _hotspot_node_num = hotspot_node_num; }
  void set_overflow_node_num(const int32_t overflow_node_num) { _overflow_node_num = overflow_node_num; }
  void set_candidate_cost(const TGCandidateCost& candidate_cost)
  {
    _total_corner_num = candidate_cost.total_corner_num;
    _total_wire_length = candidate_cost.total_wire_length;
    _is_path_blocked = candidate_cost.is_path_blocked;
    _total_usage_cost = candidate_cost.total_usage_cost;
    _total_saturation_cost = candidate_cost.total_saturation_cost;
    _total_hotspot_cost = candidate_cost.total_hotspot_cost;
    _total_overflow_cost = candidate_cost.total_overflow_cost;
    _total_overflow = candidate_cost.total_overflow;
    _max_usage_ratio = candidate_cost.max_usage_ratio;
    _saturation_node_num = candidate_cost.saturation_node_num;
    _hotspot_node_num = candidate_cost.hotspot_node_num;
    _overflow_node_num = candidate_cost.overflow_node_num;
  }
  // function

 private:
  int32_t _topo_idx = -1;
  std::vector<Segment<PlanarCoord>> _routing_segment_list;
  int32_t _total_corner_num = 0;
  int32_t _total_wire_length = 0;
  bool _is_path_blocked = false;
  double _total_usage_cost = 0.0;
  double _total_saturation_cost = 0.0;
  double _total_hotspot_cost = 0.0;
  double _total_overflow_cost = 0.0;
  double _total_overflow = 0.0;
  double _max_usage_ratio = 0.0;
  int32_t _saturation_node_num = 0;
  int32_t _hotspot_node_num = 0;
  int32_t _overflow_node_num = 0;
};

}  // namespace irt
