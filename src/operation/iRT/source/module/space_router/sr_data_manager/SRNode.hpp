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
enum class SRNodeState
{
  kNone = 0,
  kOpen = 1,
  kClose = 2
};
#endif

class SRNode : public LayerCoord
{
 public:
  SRNode() = default;
  ~SRNode() = default;
  // getter
  double get_boundary_wire_unit() const { return _boundary_wire_unit; }
  double get_internal_wire_unit() const { return _internal_wire_unit; }
  double get_internal_via_unit() const { return _internal_via_unit; }
  std::map<Orientation, SRNode*>& get_neighbor_node_map() { return _neighbor_node_map; }
  std::map<Orientation, int32_t>& get_orient_supply_map() { return _orient_supply_map; }
  std::map<Orientation, std::set<int32_t>>& get_orient_net_map() { return _orient_net_map; }
  std::map<int32_t, std::set<Orientation>>& get_net_orient_map() { return _net_orient_map; }
  // setter
  void set_boundary_wire_unit(const double boundary_wire_unit) { _boundary_wire_unit = boundary_wire_unit; }
  void set_internal_wire_unit(const double internal_wire_unit) { _internal_wire_unit = internal_wire_unit; }
  void set_internal_via_unit(const double internal_via_unit) { _internal_via_unit = internal_via_unit; }
  void set_neighbor_node_map(const std::map<Orientation, SRNode*>& neighbor_node_map) { _neighbor_node_map = neighbor_node_map; }
  void set_orient_supply_map(const std::map<Orientation, int32_t>& orient_supply_map) { _orient_supply_map = orient_supply_map; }
  void set_orient_net_map(const std::map<Orientation, std::set<int32_t>>& orient_net_map) { _orient_net_map = orient_net_map; }
  void set_net_orient_map(const std::map<int32_t, std::set<Orientation>>& net_orient_map) { _net_orient_map = net_orient_map; }
  // function
  SRNode* getNeighborNode(Orientation orientation)
  {
    SRNode* neighbor_node = nullptr;
    if (RTUTIL.exist(_neighbor_node_map, orientation)) {
      neighbor_node = _neighbor_node_map[orientation];
    }
    return neighbor_node;
  }
  double getOverflowCost(int32_t curr_net_idx, Orientation orientation, double overflow_unit)
  {
    if (!validDemandUnit()) {
      RTLOG.error(Loc::current(), "The demand unit is error!");
    }
    double boundary_overflow = 0;
    if (orientation == Orientation::kEast || orientation == Orientation::kWest || orientation == Orientation::kSouth || orientation == Orientation::kNorth) {
      double boundary_demand = 0;
      if (RTUTIL.exist(_orient_net_map, orientation)) {
        std::set<int32_t>& net_set = _orient_net_map[orientation];
        boundary_demand += (static_cast<double>(net_set.size()) * _boundary_wire_unit);
        if (RTUTIL.exist(net_set, curr_net_idx)) {
          boundary_demand -= _boundary_wire_unit;
        }
      }
      double boundary_supply = 0;
      if (RTUTIL.exist(_orient_supply_map, orientation)) {
        boundary_supply = (_orient_supply_map[orientation] * _boundary_wire_unit);
      }
      boundary_overflow = calcCost(boundary_demand + _boundary_wire_unit, boundary_supply);
    }
    double internal_overflow = 0;
    {
      double internal_demand = 0;
      for (auto& [orient, net_set] : _orient_net_map) {
        if (orient == Orientation::kAbove || orient == Orientation::kBelow) {
          continue;
        }
        internal_demand += (static_cast<double>(net_set.size()) * _internal_wire_unit);
        if (RTUTIL.exist(net_set, curr_net_idx)) {
          internal_demand -= _internal_wire_unit;
        }
      }
      for (auto& [net_idx, orient_set] : _net_orient_map) {
        if (net_idx == curr_net_idx) {
          continue;
        }
        if (RTUTIL.exist(orient_set, Orientation::kAbove) || RTUTIL.exist(orient_set, Orientation::kBelow)) {
          internal_demand += _internal_via_unit;
        }
      }
      double internal_supply = 0;
      for (auto& [orient, supply] : _orient_supply_map) {
        internal_supply += (supply * _internal_wire_unit);
      }
      if (orientation == Orientation::kEast || orientation == Orientation::kWest || orientation == Orientation::kSouth || orientation == Orientation::kNorth) {
        internal_overflow = calcCost(internal_demand + _internal_wire_unit, internal_supply);
      } else if (orientation == Orientation::kAbove || orientation == Orientation::kBelow) {
        internal_overflow = calcCost(internal_demand + _internal_via_unit, internal_supply);
      }
    }
    double cost = 0;
    cost += (overflow_unit * (boundary_overflow + internal_overflow));
    return cost;
  }
  bool validDemandUnit()
  {
    if (_boundary_wire_unit <= 0) {
      return false;
    }
    if (_internal_wire_unit <= 0) {
      return false;
    }
    if (_internal_via_unit <= 0) {
      return false;
    }
    return true;
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
  double getDemand()
  {
    if (!validDemandUnit()) {
      RTLOG.error(Loc::current(), "The demand unit is error!");
    }
    double boundary_demand = 0;
    for (Orientation orient : {Orientation::kEast, Orientation::kWest, Orientation::kSouth, Orientation::kNorth}) {
      if (RTUTIL.exist(_orient_net_map, orient)) {
        boundary_demand += (static_cast<double>(_orient_net_map[orient].size()) * _boundary_wire_unit);
      }
    }
    double internal_demand = 0;
    for (Orientation orient : {Orientation::kEast, Orientation::kWest, Orientation::kSouth, Orientation::kNorth}) {
      if (RTUTIL.exist(_orient_net_map, orient)) {
        internal_demand += (static_cast<double>(_orient_net_map[orient].size()) * _internal_wire_unit);
      }
    }
    for (auto& [net_idx, orient_set] : _net_orient_map) {
      if (RTUTIL.exist(orient_set, Orientation::kAbove) || RTUTIL.exist(orient_set, Orientation::kBelow)) {
        internal_demand += _internal_via_unit;
      }
    }
    return (boundary_demand + internal_demand);
  }
  double getOverflow()
  {
    if (!validDemandUnit()) {
      RTLOG.error(Loc::current(), "The demand unit is error!");
    }
    double boundary_overflow = 0;
    for (Orientation orient : {Orientation::kEast, Orientation::kWest, Orientation::kSouth, Orientation::kNorth}) {
      double boundary_demand = 0;
      if (RTUTIL.exist(_orient_net_map, orient)) {
        boundary_demand = (static_cast<double>(_orient_net_map[orient].size()) * _boundary_wire_unit);
      }
      double boundary_supply = 0;
      if (RTUTIL.exist(_orient_supply_map, orient)) {
        boundary_supply = (_orient_supply_map[orient] * _boundary_wire_unit);
      }
      boundary_overflow += std::max(0.0, boundary_demand - boundary_supply);
    }
    double internal_overflow = 0;
    {
      double internal_demand = 0;
      for (Orientation orient : {Orientation::kEast, Orientation::kWest, Orientation::kSouth, Orientation::kNorth}) {
        if (RTUTIL.exist(_orient_net_map, orient)) {
          internal_demand += (static_cast<double>(_orient_net_map[orient].size()) * _internal_wire_unit);
        }
      }
      for (auto& [net_idx, orient_set] : _net_orient_map) {
        if (RTUTIL.exist(orient_set, Orientation::kAbove) || RTUTIL.exist(orient_set, Orientation::kBelow)) {
          internal_demand += _internal_via_unit;
        }
      }
      double internal_supply = 0;
      for (auto& [orient, supply] : _orient_supply_map) {
        internal_supply += (supply * _internal_wire_unit);
      }
      internal_overflow += std::max(0.0, internal_demand - internal_supply);
    }
    return (boundary_overflow + internal_overflow);
  }
  std::set<int32_t> getOverflowNetSet()
  {
    if (!validDemandUnit()) {
      RTLOG.error(Loc::current(), "The demand unit is error!");
    }
    std::set<int32_t> overflow_net_set;
    for (Orientation orient : {Orientation::kEast, Orientation::kWest, Orientation::kSouth, Orientation::kNorth}) {
      double boundary_demand = 0;
      if (RTUTIL.exist(_orient_net_map, orient)) {
        boundary_demand = (static_cast<double>(_orient_net_map[orient].size()) * _boundary_wire_unit);
      }
      double boundary_supply = 0;
      if (RTUTIL.exist(_orient_supply_map, orient)) {
        boundary_supply = (_orient_supply_map[orient] * _boundary_wire_unit);
      }
      if (boundary_demand - boundary_supply > 0) {
        overflow_net_set.insert(_orient_net_map[orient].begin(), _orient_net_map[orient].end());
      }
    }
    {
      double internal_demand = 0;
      for (Orientation orient : {Orientation::kEast, Orientation::kWest, Orientation::kSouth, Orientation::kNorth}) {
        if (RTUTIL.exist(_orient_net_map, orient)) {
          internal_demand += (static_cast<double>(_orient_net_map[orient].size()) * _internal_wire_unit);
        }
      }
      for (auto& [net_idx, orient_set] : _net_orient_map) {
        if (RTUTIL.exist(orient_set, Orientation::kAbove) || RTUTIL.exist(orient_set, Orientation::kBelow)) {
          internal_demand += _internal_via_unit;
        }
      }
      double internal_supply = 0;
      for (auto& [orient, supply] : _orient_supply_map) {
        internal_supply += (supply * _internal_wire_unit);
      }
      if (internal_demand - internal_supply > 0) {
        for (auto& [net_idx, orient_set] : _net_orient_map) {
          overflow_net_set.insert(net_idx);
        }
      }
    }
    return overflow_net_set;
  }
  void updateDemand(int32_t net_idx, std::set<Orientation> orient_set, ChangeType change_type)
  {
    for (const Orientation& orient : orient_set) {
      if (change_type == ChangeType::kAdd) {
        _orient_net_map[orient].insert(net_idx);
        _net_orient_map[net_idx].insert(orient);
      } else {
        _orient_net_map[orient].erase(net_idx);
        if (_orient_net_map[orient].empty()) {
          _orient_net_map.erase(orient);
        }
        _net_orient_map[net_idx].erase(orient);
        if (_net_orient_map[net_idx].empty()) {
          _net_orient_map.erase(net_idx);
        }
      }
    }
  }
#if 1  // astar
  // single path
  SRNodeState& get_state() { return _state; }
  SRNode* get_parent_node() const { return _parent_node; }
  double get_known_cost() const { return _known_cost; }
  double get_estimated_cost() const { return _estimated_cost; }
  void set_state(SRNodeState state) { _state = state; }
  void set_parent_node(SRNode* parent_node) { _parent_node = parent_node; }
  void set_known_cost(const double known_cost) { _known_cost = known_cost; }
  void set_estimated_cost(const double estimated_cost) { _estimated_cost = estimated_cost; }
  // function
  bool isNone() { return _state == SRNodeState::kNone; }
  bool isOpen() { return _state == SRNodeState::kOpen; }
  bool isClose() { return _state == SRNodeState::kClose; }
  double getTotalCost() { return (_known_cost + _estimated_cost); }
#endif

 private:
  double _boundary_wire_unit = -1;
  double _internal_wire_unit = -1;
  double _internal_via_unit = -1;
  std::map<Orientation, SRNode*> _neighbor_node_map;
  std::map<Orientation, int32_t> _orient_supply_map;
  std::map<Orientation, std::set<int32_t>> _orient_net_map;
  std::map<int32_t, std::set<Orientation>> _net_orient_map;
#if 1  // astar
  // single path
  SRNodeState _state = SRNodeState::kNone;
  SRNode* _parent_node = nullptr;
  double _known_cost = 0.0;  // include curr
  double _estimated_cost = 0.0;
#endif
};

#if 1  // astar
struct CmpSRNodeCost
{
  bool operator()(SRNode* a, SRNode* b)
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
