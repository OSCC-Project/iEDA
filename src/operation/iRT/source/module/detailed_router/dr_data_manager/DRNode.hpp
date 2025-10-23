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

#include "Direction.hpp"
#include "LayerCoord.hpp"
#include "Orientation.hpp"
#include "RTHeader.hpp"
#include "Utility.hpp"

namespace irt {

#if 1  // astar
enum class DRNodeState
{
  kNone = 0,
  kOpen = 1,
  kClose = 2
};
#endif

class DRNode : public LayerCoord
{
 public:
  DRNode() = default;
  ~DRNode() = default;
  // getter
  std::map<Orientation, DRNode*>& get_neighbor_node_map() { return _neighbor_node_map; }
  std::map<Orientation, std::set<int32_t>>& get_orient_fixed_rect_map() { return _orient_fixed_rect_map; }
  std::map<Orientation, std::set<int32_t>>& get_orient_routed_rect_map() { return _orient_routed_rect_map; }
  std::map<Orientation, int32_t>& get_orient_violation_number_map() { return _orient_violation_number_map; }
  // setter
  void set_neighbor_node_map(const std::map<Orientation, DRNode*>& neighbor_node_map) { _neighbor_node_map = neighbor_node_map; }
  void set_orient_fixed_rect_map(const std::map<Orientation, std::set<int32_t>>& orient_fixed_rect_map) { _orient_fixed_rect_map = orient_fixed_rect_map; }
  void set_orient_routed_rect_map(const std::map<Orientation, std::set<int32_t>>& orient_routed_rect_map) { _orient_routed_rect_map = orient_routed_rect_map; }
  void set_orient_violation_number_map(const std::map<Orientation, int32_t>& orient_violation_number_map)
  {
    _orient_violation_number_map = orient_violation_number_map;
  }
  // function
  DRNode* getNeighborNode(Orientation orientation)
  {
    DRNode* neighbor_node = nullptr;
    if (RTUTIL.exist(_neighbor_node_map, orientation)) {
      neighbor_node = _neighbor_node_map[orientation];
    }
    return neighbor_node;
  }
  double getFixedRectCost(int32_t net_idx, Orientation orientation, double fixed_rect_unit)
  {
    int32_t fixed_rect_num = 0;
    if (RTUTIL.exist(_orient_fixed_rect_map, orientation)) {
      std::set<int32_t>& net_set = _orient_fixed_rect_map[orientation];
      fixed_rect_num = static_cast<int32_t>(net_set.size());
      if (RTUTIL.exist(net_set, net_idx)) {
        fixed_rect_num--;
      }
      if (fixed_rect_num < 0) {
        RTLOG.error(Loc::current(), "The fixed_rect_num < 0!");
      }
    }
    double cost = 0;
    if (fixed_rect_num > 0) {
      cost = fixed_rect_unit;
    }
    return cost;
  }
  double getRoutedRectCost(int32_t net_idx, Orientation orientation, double routed_rect_unit)
  {
    int32_t routed_rect_num = 0;
    if (RTUTIL.exist(_orient_routed_rect_map, orientation)) {
      std::set<int32_t>& net_set = _orient_routed_rect_map[orientation];
      routed_rect_num = static_cast<int32_t>(net_set.size());
      if (RTUTIL.exist(net_set, net_idx)) {
        routed_rect_num--;
      }
      if (routed_rect_num < 0) {
        RTLOG.error(Loc::current(), "The routed_rect_num < 0!");
      }
    }
    double cost = 0;
    if (routed_rect_num > 0) {
      cost = routed_rect_unit;
    }
    return cost;
  }
  double getViolationCost(Orientation orientation, double violation_unit)
  {
    int32_t violation_num = 0;
    if (RTUTIL.exist(_orient_violation_number_map, orientation)) {
      violation_num = _orient_violation_number_map[orientation];
    }
    double cost = 0;
    if (violation_num > 0) {
      cost = violation_unit;
    }
    return cost;
  }
#if 1  // astar
  // single task
  std::set<Direction>& get_direction_set() { return _direction_set; }
  void set_direction_set(std::set<Direction>& direction_set) { _direction_set = direction_set; }
  // single path
  DRNodeState& get_state() { return _state; }
  DRNode* get_parent_node() const { return _parent_node; }
  double get_known_cost() const { return _known_cost; }
  double get_estimated_cost() const { return _estimated_cost; }
  void set_state(DRNodeState state) { _state = state; }
  void set_parent_node(DRNode* parent_node) { _parent_node = parent_node; }
  void set_known_cost(const double known_cost) { _known_cost = known_cost; }
  void set_estimated_cost(const double estimated_cost) { _estimated_cost = estimated_cost; }
  // function
  bool isNone() { return _state == DRNodeState::kNone; }
  bool isOpen() { return _state == DRNodeState::kOpen; }
  bool isClose() { return _state == DRNodeState::kClose; }
  double getTotalCost() { return (_known_cost + _estimated_cost); }
#endif

 private:
  std::map<Orientation, DRNode*> _neighbor_node_map;
  // obstacle & pin_shape
  std::map<Orientation, std::set<int32_t>> _orient_fixed_rect_map;
  // net_result
  std::map<Orientation, std::set<int32_t>> _orient_routed_rect_map;
  // violation
  std::map<Orientation, int32_t> _orient_violation_number_map;
#if 1  // astar
  // single task
  std::set<Direction> _direction_set;
  // single path
  DRNodeState _state = DRNodeState::kNone;
  DRNode* _parent_node = nullptr;
  double _known_cost = 0.0;  // include curr
  double _estimated_cost = 0.0;
#endif
};

#if 1  // astar
struct CmpDRNodeCost
{
  bool operator()(DRNode* a, DRNode* b)
  {
    if (RTUTIL.equalDoubleByError(a->getTotalCost(), b->getTotalCost(), RT_ERROR)) {
      if (RTUTIL.equalDoubleByError(a->get_estimated_cost(), b->get_estimated_cost(), RT_ERROR)) {
        return a->get_neighbor_node_map().size() < b->get_neighbor_node_map().size();
      } else {
        return a->get_estimated_cost() > b->get_estimated_cost();
      }
    } else {
      return a->getTotalCost() > b->getTotalCost();
    }
  }
};
#endif

}  // namespace irt
