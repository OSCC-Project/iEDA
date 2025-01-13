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

class VRNode : public LayerCoord
{
 public:
  VRNode() = default;
  ~VRNode() = default;
  // getter
  std::map<Orientation, VRNode*>& get_neighbor_node_map() { return _neighbor_node_map; }
  std::map<Orientation, std::set<int32_t>>& get_orient_fixed_rect_map() { return _orient_fixed_rect_map; }
  std::map<Orientation, std::set<int32_t>>& get_orient_routed_rect_map() { return _orient_routed_rect_map; }
  std::map<Orientation, int32_t>& get_orient_violation_number_map() { return _orient_violation_number_map; }
  // setter
  void set_neighbor_node_map(const std::map<Orientation, VRNode*>& neighbor_node_map) { _neighbor_node_map = neighbor_node_map; }
  void set_orient_fixed_rect_map(const std::map<Orientation, std::set<int32_t>>& orient_fixed_rect_map)
  {
    _orient_fixed_rect_map = orient_fixed_rect_map;
  }
  void set_orient_routed_rect_map(const std::map<Orientation, std::set<int32_t>>& orient_routed_rect_map)
  {
    _orient_routed_rect_map = orient_routed_rect_map;
  }
  void set_orient_violation_number_map(const std::map<Orientation, int32_t>& orient_violation_number_map)
  {
    _orient_violation_number_map = orient_violation_number_map;
  }
  // function
  VRNode* getNeighborNode(Orientation orientation)
  {
    VRNode* neighbor_node = nullptr;
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

 private:
  std::map<Orientation, VRNode*> _neighbor_node_map;
  // obstacle & pin_shape
  std::map<Orientation, std::set<int32_t>> _orient_fixed_rect_map;
  // net_result
  std::map<Orientation, std::set<int32_t>> _orient_routed_rect_map;
  // violation
  std::map<Orientation, int32_t> _orient_violation_number_map;
};

}  // namespace irt
