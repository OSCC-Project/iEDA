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
#include "RTU.hpp"
#include "RTUtil.hpp"

namespace irt {

#if 1  // astar
enum class TANodeState
{
  kNone = 0,
  kOpen = 1,
  kClose = 2
};
#endif

class TANode : public LayerCoord
{
 public:
  TANode() = default;
  ~TANode() = default;
  // getter
  std::map<Orientation, TANode*>& get_neighbor_node_map() { return _neighbor_node_map; }
  std::map<Orientation, std::set<irt_int>>& get_orien_fixed_rect_map() { return _orien_fixed_rect_map; }
  std::map<Orientation, std::set<irt_int>>& get_orien_routed_rect_map() { return _orien_routed_rect_map; }
  std::map<Orientation, irt_int>& get_orien_violation_number_map() { return _orien_violation_number_map; }
  // setter
  void set_neighbor_node_map(const std::map<Orientation, TANode*>& neighbor_node_map) { _neighbor_node_map = neighbor_node_map; }
  void set_orien_fixed_rect_map(const std::map<Orientation, std::set<irt_int>>& orien_fixed_rect_map)
  {
    _orien_fixed_rect_map = orien_fixed_rect_map;
  }
  void set_orien_routed_rect_map(const std::map<Orientation, std::set<irt_int>>& orien_routed_rect_map)
  {
    _orien_routed_rect_map = orien_routed_rect_map;
  }
  void set_orien_violation_number_map(const std::map<Orientation, irt_int>& orien_violation_number_map)
  {
    _orien_violation_number_map = orien_violation_number_map;
  }
  // function
  TANode* getNeighborNode(Orientation orientation)
  {
    TANode* neighbor_node = nullptr;
    if (RTUtil::exist(_neighbor_node_map, orientation)) {
      neighbor_node = _neighbor_node_map[orientation];
    }
    return neighbor_node;
  }
  double getFixedRectCost(irt_int net_idx, Orientation orientation, double fixed_rect_cost)
  {
    irt_int fixed_rect_num = 0;
    if (RTUtil::exist(_orien_fixed_rect_map, orientation)) {
      std::set<irt_int>& net_set = _orien_fixed_rect_map[orientation];
      fixed_rect_num = static_cast<irt_int>(net_set.size());
      if (RTUtil::exist(net_set, net_idx)) {
        fixed_rect_num--;
      }
      if (fixed_rect_num < 0) {
        LOG_INST.error(Loc::current(), "The fixed_rect_num < 0!");
      }
    }
    double cost = 0;
    cost = (fixed_rect_num * fixed_rect_cost);
    return cost;
  }
  double getRoutedRectCost(irt_int net_idx, Orientation orientation, double routed_rect_cost)
  {
    irt_int routed_rect_num = 0;
    if (RTUtil::exist(_orien_routed_rect_map, orientation)) {
      std::set<irt_int>& net_set = _orien_routed_rect_map[orientation];
      routed_rect_num = static_cast<irt_int>(net_set.size());
      if (RTUtil::exist(net_set, net_idx)) {
        routed_rect_num--;
      }
      if (routed_rect_num < 0) {
        LOG_INST.error(Loc::current(), "The routed_rect_num < 0!");
      }
    }
    double cost = 0;
    cost = (routed_rect_num * routed_rect_cost);
    return cost;
  }
  double getViolationCost(Orientation orientation, double violation_cost)
  {
    irt_int violation_num = 0;
    if (RTUtil::exist(_orien_violation_number_map, orientation)) {
      violation_num = _orien_violation_number_map[orientation];
    }
    double cost = 0;
    cost = (violation_num * violation_cost);
    return cost;
  }
#if 1  // astar
  // single task
  std::set<Direction>& get_direction_set() { return _direction_set; }
  void set_direction_set(std::set<Direction>& direction_set) { _direction_set = direction_set; }
  // single path
  TANodeState& get_state() { return _state; }
  TANode* get_parent_node() const { return _parent_node; }
  double get_known_cost() const { return _known_cost; }
  double get_estimated_cost() const { return _estimated_cost; }
  void set_state(TANodeState state) { _state = state; }
  void set_parent_node(TANode* parent_node) { _parent_node = parent_node; }
  void set_known_cost(const double known_cost) { _known_cost = known_cost; }
  void set_estimated_cost(const double estimated_cost) { _estimated_cost = estimated_cost; }
  // function
  bool isNone() { return _state == TANodeState::kNone; }
  bool isOpen() { return _state == TANodeState::kOpen; }
  bool isClose() { return _state == TANodeState::kClose; }
  double getTotalCost() { return (_known_cost + _estimated_cost); }
#endif

 private:
  std::map<Orientation, TANode*> _neighbor_node_map;
  // blockage & pin_shape
  std::map<Orientation, std::set<irt_int>> _orien_fixed_rect_map;
  // net_result & patch
  std::map<Orientation, std::set<irt_int>> _orien_routed_rect_map;
  // violation
  std::map<Orientation, irt_int> _orien_violation_number_map;
#if 1  // astar
  // single task
  std::set<Direction> _direction_set;
  // single path
  TANodeState _state = TANodeState::kNone;
  TANode* _parent_node = nullptr;
  double _known_cost = 0.0;  // include curr
  double _estimated_cost = 0.0;
#endif
};

#if 1  // astar
struct CmpTANodeCost
{
  bool operator()(TANode* a, TANode* b)
  {
    if (RTUtil::equalDoubleByError(a->getTotalCost(), b->getTotalCost(), DBL_ERROR)) {
      if (RTUtil::equalDoubleByError(a->get_estimated_cost(), b->get_estimated_cost(), DBL_ERROR)) {
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
