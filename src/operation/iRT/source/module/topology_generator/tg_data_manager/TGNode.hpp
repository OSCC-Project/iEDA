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

class TGNode : public PlanarCoord
{
 public:
  TGNode() = default;
  ~TGNode() = default;
  // getter
  double get_boundary_wire_unit() const { return _boundary_wire_unit; }
  double get_internal_wire_unit() const { return _internal_wire_unit; }
  double get_internal_via_unit() const { return _internal_via_unit; }
  std::map<Orientation, TGNode*>& get_neighbor_node_map() { return _neighbor_node_map; }
  std::map<Orientation, int32_t>& get_orient_supply_map() { return _orient_supply_map; }
  std::map<Orientation, std::set<int32_t>>& get_orient_net_map() { return _orient_net_map; }
  std::map<int32_t, std::set<Orientation>>& get_net_orient_map() { return _net_orient_map; }
  // setter
  void set_boundary_wire_unit(const double boundary_wire_unit) { _boundary_wire_unit = boundary_wire_unit; }
  void set_internal_wire_unit(const double internal_wire_unit) { _internal_wire_unit = internal_wire_unit; }
  void set_internal_via_unit(const double internal_via_unit) { _internal_via_unit = internal_via_unit; }
  void set_neighbor_node_map(const std::map<Orientation, TGNode*>& neighbor_node_map) { _neighbor_node_map = neighbor_node_map; }
  void set_orient_supply_map(const std::map<Orientation, int32_t>& orient_supply_map) { _orient_supply_map = orient_supply_map; }
  void set_orient_net_map(const std::map<Orientation, std::set<int32_t>>& orient_net_map) { _orient_net_map = orient_net_map; }
  void set_net_orient_map(const std::map<int32_t, std::set<Orientation>>& net_orient_map) { _net_orient_map = net_orient_map; }
  // function
  TGNode* getNeighborNode(Orientation orientation)
  {
    TGNode* neighbor_node = nullptr;
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
  void updateDemand(int32_t net_idx, std::set<Orientation> orient_set, ChangeType change_type)
  {
    for (const Orientation& orient : orient_set) {
      if (change_type == ChangeType::kAdd) {
        _orient_net_map[orient].insert(net_idx);
        _net_orient_map[net_idx].insert(orient);
      } else {
        _orient_net_map[orient].erase(net_idx);
        _net_orient_map[net_idx].erase(orient);
      }
    }
  }

 private:
  double _boundary_wire_unit = -1;
  double _internal_wire_unit = -1;
  double _internal_via_unit = -1;
  std::map<Orientation, TGNode*> _neighbor_node_map;
  std::map<Orientation, int32_t> _orient_supply_map;
  std::map<Orientation, std::set<int32_t>> _orient_net_map;
  std::map<int32_t, std::set<Orientation>> _net_orient_map;
};

}  // namespace irt
