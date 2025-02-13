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
  std::map<Orientation, TGNode*>& get_neighbor_node_map() { return _neighbor_node_map; }
  std::map<Orientation, int32_t>& get_orient_supply_map() { return _orient_supply_map; }
  std::map<Orientation, int32_t>& get_orient_demand_map() { return _orient_demand_map; }
  // setter
  void set_neighbor_node_map(const std::map<Orientation, TGNode*>& neighbor_node_map) { _neighbor_node_map = neighbor_node_map; }
  void set_orient_supply_map(const std::map<Orientation, int32_t>& orient_supply_map) { _orient_supply_map = orient_supply_map; }
  void set_orient_demand_map(const std::map<Orientation, int32_t>& orient_demand_map) { _orient_demand_map = orient_demand_map; }
  // function
  TGNode* getNeighborNode(Orientation orientation)
  {
    TGNode* neighbor_node = nullptr;
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

 private:
  std::map<Orientation, TGNode*> _neighbor_node_map;
  std::map<Orientation, int32_t> _orient_supply_map;
  std::map<Orientation, int32_t> _orient_demand_map;
};

}  // namespace irt
