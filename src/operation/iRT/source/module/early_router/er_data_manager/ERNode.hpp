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
enum class ERNodeState
{
  kNone = 0,
  kOpen = 1,
  kClose = 2
};
#endif

class ERNode : public LayerCoord
{
 public:
  ERNode() = default;
  ~ERNode() = default;
  // getter
  std::map<Orientation, ERNode*>& get_neighbor_node_map() { return _neighbor_node_map; }
  std::map<Orientation, int32_t>& get_orient_supply_map() { return _orient_supply_map; }
  std::map<Orientation, int32_t>& get_orient_demand_map() { return _orient_demand_map; }
  // setter
  void set_neighbor_node_map(const std::map<Orientation, ERNode*>& neighbor_node_map) { _neighbor_node_map = neighbor_node_map; }
  void set_orient_supply_map(const std::map<Orientation, int32_t>& orient_supply_map) { _orient_supply_map = orient_supply_map; }
  void set_orient_demand_map(const std::map<Orientation, int32_t>& orient_demand_map) { _orient_demand_map = orient_demand_map; }
  // function
  ERNode* getNeighborNode(Orientation orientation)
  {
    ERNode* neighbor_node = nullptr;
    if (RTUTIL.exist(_neighbor_node_map, orientation)) {
      neighbor_node = _neighbor_node_map[orientation];
    }
    return neighbor_node;
  }
  double getOverflowCost(Orientation orientation, double overflow_cost)
  {
    double cost = 0;
    if (orientation != Orientation::kAbove && orientation != Orientation::kBelow) {
      int32_t node_demand = 0;
      if (RTUTIL.exist(_orient_demand_map, orientation)) {
        node_demand = _orient_demand_map[orientation];
      }
      int32_t node_supply = 0;
      if (RTUTIL.exist(_orient_supply_map, orientation)) {
        node_supply = _orient_supply_map[orientation];
      }
      cost += (calcCost(node_demand + 1, node_supply) * overflow_cost);
    }
    return cost;
  }
  double calcCost(double demand, double supply)
  {
    double cost = 0;
    if (demand == supply) {
      cost = 1;
    } else if (demand > supply) {
      cost = std::pow(demand - supply + 1, 2);
    } else if (demand < supply) {
      cost = std::pow(demand / supply, 2);
    }
    return cost;
  }
  void updateDemand(std::set<Orientation> orient_set, ChangeType change_type)
  {
    for (const Orientation& orient : orient_set) {
      if (orient == Orientation::kEast || orient == Orientation::kWest || orient == Orientation::kSouth || orient == Orientation::kNorth) {
        _orient_demand_map[orient] += (change_type == ChangeType::kAdd ? 1 : -1);
      }
    }
  }
#if 1  // astar
  // single path
  ERNodeState& get_state() { return _state; }
  ERNode* get_parent_node() const { return _parent_node; }
  double get_known_cost() const { return _known_cost; }
  double get_estimated_cost() const { return _estimated_cost; }
  void set_state(ERNodeState state) { _state = state; }
  void set_parent_node(ERNode* parent_node) { _parent_node = parent_node; }
  void set_known_cost(const double known_cost) { _known_cost = known_cost; }
  void set_estimated_cost(const double estimated_cost) { _estimated_cost = estimated_cost; }
  // function
  bool isNone() { return _state == ERNodeState::kNone; }
  bool isOpen() { return _state == ERNodeState::kOpen; }
  bool isClose() { return _state == ERNodeState::kClose; }
  double getTotalCost() { return (_known_cost + _estimated_cost); }
#endif

 private:
  std::map<Orientation, ERNode*> _neighbor_node_map;
  std::map<Orientation, int32_t> _orient_supply_map;
  std::map<Orientation, int32_t> _orient_demand_map;
#if 1  // astar
  // single path
  ERNodeState _state = ERNodeState::kNone;
  ERNode* _parent_node = nullptr;
  double _known_cost = 0.0;  // include curr
  double _estimated_cost = 0.0;
#endif
};

#if 1  // astar
struct CmpERNodeCost
{
  bool operator()(ERNode* a, ERNode* b)
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
