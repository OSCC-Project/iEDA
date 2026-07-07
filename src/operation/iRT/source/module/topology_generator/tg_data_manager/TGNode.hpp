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

struct TGNodeCost
{
  double usage_cost = 0.0;
  double saturation_cost = 0.0;
  double hotspot_cost = 0.0;
  double overflow_cost = 0.0;
  double overflow = 0.0;
  double max_usage_ratio = 0.0;
  int32_t saturation_orient_num = 0;
  int32_t hotspot_orient_num = 0;
  int32_t overflow_orient_num = 0;

  double getTotalCost() const { return usage_cost + saturation_cost + hotspot_cost + overflow_cost; }
  void addCost(const TGNodeCost& cost)
  {
    usage_cost += cost.usage_cost;
    saturation_cost += cost.saturation_cost;
    hotspot_cost += cost.hotspot_cost;
    overflow_cost += cost.overflow_cost;
    overflow += cost.overflow;
    max_usage_ratio = std::max(max_usage_ratio, cost.max_usage_ratio);
    saturation_orient_num += cost.saturation_orient_num;
    hotspot_orient_num += cost.hotspot_orient_num;
    overflow_orient_num += cost.overflow_orient_num;
  }
};

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
  std::map<int32_t, std::set<Orientation>>& get_ignore_net_orient_map() { return _ignore_net_orient_map; }
  std::map<Orientation, std::set<int32_t>>& get_orient_net_map() { return _orient_net_map; }
  std::map<int32_t, std::set<Orientation>>& get_net_orient_map() { return _net_orient_map; }
  // setter
  void set_boundary_wire_unit(const double boundary_wire_unit) { _boundary_wire_unit = boundary_wire_unit; }
  void set_internal_wire_unit(const double internal_wire_unit) { _internal_wire_unit = internal_wire_unit; }
  void set_internal_via_unit(const double internal_via_unit) { _internal_via_unit = internal_via_unit; }
  void set_neighbor_node_map(const std::map<Orientation, TGNode*>& neighbor_node_map) { _neighbor_node_map = neighbor_node_map; }
  void set_orient_supply_map(const std::map<Orientation, int32_t>& orient_supply_map) { _orient_supply_map = orient_supply_map; }
  void set_ignore_net_orient_map(const std::map<int32_t, std::set<Orientation>>& ignore_net_orient_map) { _ignore_net_orient_map = ignore_net_orient_map; }
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
  TGNodeCost getCost(int32_t net_idx, Direction direction, double overflow_unit,
                     const std::set<Orientation>* extra_orient_set = nullptr)
  {
    if (!validDemandUnit()) {
      RTLOG.error(Loc::current(), "The demand unit is error!");
    }
    std::map<Orientation, std::set<int32_t>> orient_net_map = _orient_net_map;
    std::map<int32_t, std::set<Orientation>> net_orient_map = _net_orient_map;
    if (extra_orient_set) {
      for (Orientation orient : *extra_orient_set) {
        orient_net_map[orient].insert(net_idx);
        net_orient_map[net_idx].insert(orient);
      }
    }
    if (direction == Direction::kHorizontal) {
      for (Orientation orient : {Orientation::kEast, Orientation::kWest}) {
        orient_net_map[orient].insert(net_idx);
        net_orient_map[net_idx].insert(orient);
      }
    } else if (direction == Direction::kVertical) {
      for (Orientation orient : {Orientation::kSouth, Orientation::kNorth}) {
        orient_net_map[orient].insert(net_idx);
        net_orient_map[net_idx].insert(orient);
      }
    } else {
      RTLOG.error(Loc::current(), "The direction is error!");
    }
    TGNodeCost node_cost;
    for (Orientation orient : {Orientation::kEast, Orientation::kWest, Orientation::kSouth, Orientation::kNorth}) {
      double boundary_demand = 0;
      if (RTUTIL.exist(orient_net_map, orient)) {
        for (int32_t demand_net_idx : orient_net_map[orient]) {
          if (RTUTIL.exist(_ignore_net_orient_map, demand_net_idx) && RTUTIL.exist(_ignore_net_orient_map[demand_net_idx], orient)) {
            continue;
          }
          boundary_demand += _boundary_wire_unit;
        }
      }
      double boundary_supply = 0;
      if (RTUTIL.exist(_orient_supply_map, orient)) {
        boundary_supply = _orient_supply_map[orient];
      }
      node_cost.addCost(calcCost(boundary_demand, boundary_supply, overflow_unit));
    }
    {
      double internal_demand = 0;
      for (Orientation orient : {Orientation::kEast, Orientation::kWest, Orientation::kSouth, Orientation::kNorth}) {
        if (RTUTIL.exist(orient_net_map, orient)) {
          for (int32_t demand_net_idx : orient_net_map[orient]) {
            if (RTUTIL.exist(_ignore_net_orient_map, demand_net_idx) && RTUTIL.exist(_ignore_net_orient_map[demand_net_idx], orient)) {
              continue;
            }
            internal_demand += _internal_wire_unit;
          }
        }
      }
      double internal_supply = 0;
      for (auto& [orient, supply] : _orient_supply_map) {
        internal_supply += supply;
      }
      node_cost.addCost(calcCost(internal_demand, internal_supply, overflow_unit));
    }
    return node_cost;
  }
  double getOverflowCost(int32_t net_idx, Direction direction, double overflow_unit,
                         const std::set<Orientation>* extra_orient_set = nullptr)
  {
    return getCost(net_idx, direction, overflow_unit, extra_orient_set).getTotalCost();
  }
  bool validDemandUnit()
  {
    if (_boundary_wire_unit <= 0) {
      return false;
    }
    if (_internal_wire_unit <= 0) {
      return false;
    }
    return true;
  }
  TGNodeCost calcCost(double demand, double supply, double overflow_unit)
  {
    constexpr double kSaturationStartRatio = 0.8;
    constexpr double kHotspotStartRatio = 0.9;
    constexpr double kFullSupplyPenaltyScale = 1.0;
    constexpr double kHotspotPenaltyScale = 2.0;

    TGNodeCost cost;
    if (supply <= 0) {
      if (demand <= 0) {
        return cost;
      }
      cost.max_usage_ratio = demand + 1.0;
      cost.overflow = demand;
      cost.overflow_orient_num = 1;
      cost.overflow_cost = overflow_unit * std::pow(cost.overflow + 1, 4);
      return cost;
    }

    double usage_ratio = demand / supply;
    cost.max_usage_ratio = usage_ratio;
    if (demand > supply) {
      cost.overflow = demand - supply;
      cost.overflow_orient_num = 1;
      cost.overflow_cost = overflow_unit * std::pow(cost.overflow + 1, 4);
      return cost;
    }

    cost.usage_cost = overflow_unit * std::pow(usage_ratio, 4);
    if (usage_ratio >= kSaturationStartRatio) {
      double saturation_ratio = (usage_ratio - kSaturationStartRatio) / (1.0 - kSaturationStartRatio);
      cost.saturation_orient_num = 1;
      cost.saturation_cost = overflow_unit * std::pow(saturation_ratio, 2);
      if (usage_ratio >= 1.0) {
        cost.saturation_cost += overflow_unit * kFullSupplyPenaltyScale;
      }
    }
    if (usage_ratio >= kHotspotStartRatio) {
      double hotspot_ratio = (usage_ratio - kHotspotStartRatio) / (1.0 - kHotspotStartRatio);
      cost.hotspot_orient_num = 1;
      cost.hotspot_cost = overflow_unit * kHotspotPenaltyScale * std::pow(hotspot_ratio, 2);
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
        for (int32_t demand_net_idx : _orient_net_map[orient]) {
          if (RTUTIL.exist(_ignore_net_orient_map, demand_net_idx) && RTUTIL.exist(_ignore_net_orient_map[demand_net_idx], orient)) {
            continue;
          }
          boundary_demand += _boundary_wire_unit;
        }
      }
    }
    double internal_demand = 0;
    for (Orientation orient : {Orientation::kEast, Orientation::kWest, Orientation::kSouth, Orientation::kNorth}) {
      if (RTUTIL.exist(_orient_net_map, orient)) {
        for (int32_t demand_net_idx : _orient_net_map[orient]) {
          if (RTUTIL.exist(_ignore_net_orient_map, demand_net_idx) && RTUTIL.exist(_ignore_net_orient_map[demand_net_idx], orient)) {
            continue;
          }
          internal_demand += _internal_wire_unit;
        }
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
        for (int32_t demand_net_idx : _orient_net_map[orient]) {
          if (RTUTIL.exist(_ignore_net_orient_map, demand_net_idx) && RTUTIL.exist(_ignore_net_orient_map[demand_net_idx], orient)) {
            continue;
          }
          boundary_demand += _boundary_wire_unit;
        }
      }
      double boundary_supply = 0;
      if (RTUTIL.exist(_orient_supply_map, orient)) {
        boundary_supply = _orient_supply_map[orient];
      }
      boundary_overflow += std::max(0.0, boundary_demand - boundary_supply);
    }
    double internal_overflow = 0;
    {
      double internal_demand = 0;
      for (Orientation orient : {Orientation::kEast, Orientation::kWest, Orientation::kSouth, Orientation::kNorth}) {
        if (RTUTIL.exist(_orient_net_map, orient)) {
          for (int32_t demand_net_idx : _orient_net_map[orient]) {
            if (RTUTIL.exist(_ignore_net_orient_map, demand_net_idx) && RTUTIL.exist(_ignore_net_orient_map[demand_net_idx], orient)) {
              continue;
            }
            internal_demand += _internal_wire_unit;
          }
        }
      }
      double internal_supply = 0;
      for (auto& [orient, supply] : _orient_supply_map) {
        internal_supply += supply;
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

 private:
  double _boundary_wire_unit = -1;
  double _internal_wire_unit = -1;
  double _internal_via_unit = -1;
  std::map<Orientation, TGNode*> _neighbor_node_map;
  std::map<Orientation, int32_t> _orient_supply_map;
  std::map<int32_t, std::set<Orientation>> _ignore_net_orient_map;
  std::map<Orientation, std::set<int32_t>> _orient_net_map;
  std::map<int32_t, std::set<Orientation>> _net_orient_map;
};

}  // namespace irt
