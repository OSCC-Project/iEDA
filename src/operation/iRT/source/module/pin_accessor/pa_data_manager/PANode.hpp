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
enum class PANodeState
{
  kNone = 0,
  kOpen = 1,
  kClose = 2
};
#endif

class PANode : public LayerCoord
{
 public:
  PANode() = default;
  ~PANode() = default;
  // getter
  std::map<Orientation, PANode*>& get_neighbor_node_map() { return _neighbor_node_map; }
  std::map<Orientation, std::set<int32_t>>& get_orient_fixed_rect_map() { return _orient_fixed_rect_map; }
  std::map<Orientation, std::set<int32_t>>& get_orient_routed_rect_map() { return _orient_routed_rect_map; }
  std::map<Orientation, int32_t>& get_orient_violation_number_map() { return _orient_violation_number_map; }
  // setter
  void set_neighbor_node_map(const std::map<Orientation, PANode*>& neighbor_node_map) { _neighbor_node_map = neighbor_node_map; }
  void set_orient_fixed_rect_map(const std::map<Orientation, std::set<int32_t>>& orient_fixed_rect_map) { _orient_fixed_rect_map = orient_fixed_rect_map; }
  void set_orient_routed_rect_map(const std::map<Orientation, std::set<int32_t>>& orient_routed_rect_map) { _orient_routed_rect_map = orient_routed_rect_map; }
  void set_orient_violation_number_map(const std::map<Orientation, int32_t>& orient_violation_number_map)
  {
    _orient_violation_number_map = orient_violation_number_map;
  }
  // function
  PANode* getNeighborNode(Orientation orientation)
  {
    PANode* neighbor_node = nullptr;
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
  // single path
  PANodeState& get_state() { return _state; }
  PANode* get_parent_node() const { return _parent_node; }
  double get_known_cost() const { return _known_cost; }
  double get_estimated_cost() const { return _estimated_cost; }
  void set_state(PANodeState state) { _state = state; }
  void set_parent_node(PANode* parent_node) { _parent_node = parent_node; }
  void set_known_cost(const double known_cost) { _known_cost = known_cost; }
  void set_estimated_cost(const double estimated_cost) { _estimated_cost = estimated_cost; }
  // function
  bool isNone() { return _state == PANodeState::kNone; }
  bool isOpen() { return _state == PANodeState::kOpen; }
  bool isClose() { return _state == PANodeState::kClose; }
  double getTotalCost() { return (_known_cost + _estimated_cost); }
#endif

 private:
  std::map<Orientation, PANode*> _neighbor_node_map;
  // obstacle & pin_shape
  std::map<Orientation, std::set<int32_t>> _orient_fixed_rect_map;
  // net_result
  std::map<Orientation, std::set<int32_t>> _orient_routed_rect_map;
  // violation
  std::map<Orientation, int32_t> _orient_violation_number_map;
#if 1  // astar
  // single path
  PANodeState _state = PANodeState::kNone;
  PANode* _parent_node = nullptr;
  double _known_cost = 0.0;  // include curr
  double _estimated_cost = 0.0;
#endif
};

#if 1  // astar
struct CmpPANodeCost
{
  bool operator()(PANode* a, PANode* b)
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
