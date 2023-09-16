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

#include "DRSourceType.hpp"
#include "Direction.hpp"
#include "LayerCoord.hpp"
#include "Orientation.hpp"
#include "RTU.hpp"
#include "RTUtil.hpp"

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
  std::map<Orientation, DRNode*>& get_neighbor_ptr_map() { return _neighbor_ptr_map; }
  std::map<DRSourceType, std::map<Orientation, std::set<irt_int>>>& get_source_orien_net_map() { return _source_orien_net_map; }
  std::map<Orientation, double>& get_orien_history_cost_map() { return _orien_history_cost_map; }
  // setter
  void set_neighbor_ptr_map(const std::map<Orientation, DRNode*>& neighbor_ptr_map) { _neighbor_ptr_map = neighbor_ptr_map; }
  void set_source_orien_net_map(const std::map<DRSourceType, std::map<Orientation, std::set<irt_int>>>& source_orien_net_map)
  {
    _source_orien_net_map = source_orien_net_map;
  }
  void set_orien_history_cost_map(const std::map<Orientation, double>& orien_history_cost_map)
  {
    _orien_history_cost_map = orien_history_cost_map;
  }
  // function
  DRNode* getNeighborNode(Orientation orientation)
  {
    DRNode* neighbor_node = nullptr;
    if (RTUtil::exist(_neighbor_ptr_map, orientation)) {
      neighbor_node = _neighbor_ptr_map[orientation];
    }
    return neighbor_node;
  }
  double getCost(irt_int net_idx, Orientation orientation)
  {
    double dr_layout_shape_unit = DM_INST.getConfig().dr_layout_shape_unit;
    double dr_reserved_via_unit = DM_INST.getConfig().dr_reserved_via_unit;

    double cost = 0;
    for (DRSourceType dr_source_type : {DRSourceType::kLayoutShape, DRSourceType::kReservedVia}) {
      irt_int violation_net_num = 0;
      if (RTUtil::exist(_source_orien_net_map, dr_source_type)) {
        std::map<Orientation, std::set<irt_int>>& orien_net_map = _source_orien_net_map[dr_source_type];
        if (RTUtil::exist(orien_net_map, orientation)) {
          std::set<irt_int>& net_set = orien_net_map[orientation];
          if (net_set.size() >= 2) {
            violation_net_num = static_cast<irt_int>(net_set.size());
          } else {
            violation_net_num = RTUtil::exist(net_set, net_idx) ? 0 : 1;
          }
        }
      }
      switch (dr_source_type) {
        case DRSourceType::kLayoutShape:
          cost += (dr_layout_shape_unit * violation_net_num);
          break;
        case DRSourceType::kReservedVia:
          cost += (dr_reserved_via_unit * violation_net_num);
          break;
        default:
          break;
      }
    }
    if (RTUtil::exist(_orien_history_cost_map, orientation)) {
      cost += _orien_history_cost_map[orientation];
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
  std::map<Orientation, DRNode*> _neighbor_ptr_map;
  std::map<DRSourceType, std::map<Orientation, std::set<irt_int>>> _source_orien_net_map;
  std::map<Orientation, double> _orien_history_cost_map;
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
    if (RTUtil::equalDoubleByError(a->getTotalCost(), b->getTotalCost(), DBL_ERROR)) {
      if (RTUtil::equalDoubleByError(a->get_estimated_cost(), b->get_estimated_cost(), DBL_ERROR)) {
        return a->get_neighbor_ptr_map().size() < b->get_neighbor_ptr_map().size();
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
