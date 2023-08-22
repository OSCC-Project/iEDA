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

#include "LayerCoord.hpp"
#include "Orientation.hpp"
#include "RTU.hpp"
#include "RTUtil.hpp"
#include "TARouteStrategy.hpp"
#include "TASourceType.hpp"

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
  std::map<Orientation, TANode*>& get_neighbor_ptr_map() { return _neighbor_ptr_map; }
  std::map<TASourceType, std::map<Orientation, std::set<irt_int>>>& get_source_orien_task_map() { return _source_orien_task_map; }
  // setter
  void set_neighbor_ptr_map(const std::map<Orientation, TANode*>& neighbor_ptr_map) { _neighbor_ptr_map = neighbor_ptr_map; }
  void set_source_orien_task_map(const std::map<TASourceType, std::map<Orientation, std::set<irt_int>>>& source_orien_task_map)
  {
    _source_orien_task_map = source_orien_task_map;
  }
  // function
  TANode* getNeighborNode(Orientation orientation)
  {
    TANode* neighbor_node = nullptr;
    if (RTUtil::exist(_neighbor_ptr_map, orientation)) {
      neighbor_node = _neighbor_ptr_map[orientation];
    }
    return neighbor_node;
  }
  bool isOBS(irt_int task_idx, Orientation orientation, TARouteStrategy ta_route_strategy)
  {
    bool is_obs = false;
    if (ta_route_strategy == TARouteStrategy::kIgnoringBlockAndPin) {
      return is_obs;
    }
    if (!is_obs) {
      if (RTUtil::exist(_source_orien_task_map, TASourceType::kBlockAndPin)) {
        std::map<irt::Orientation, std::set<irt_int>>& orien_task_map = _source_orien_task_map[TASourceType::kBlockAndPin];
        if (RTUtil::exist(orien_task_map, orientation)) {
          std::set<irt_int>& task_set = orien_task_map[orientation];
          if (task_set.size() >= 2) {
            is_obs = true;
          } else {
            is_obs = RTUtil::exist(task_set, task_idx) ? false : true;
          }
        }
      }
    }
    return is_obs;
  }
  double getCost(irt_int task_idx, Orientation orientation)
  {
    double cost = 0;
    for (TASourceType ta_source_type : {TASourceType::kEnclosure, TASourceType::kOtherPanel, TASourceType::kSelfPanel}) {
      bool add_cost = false;
      if (RTUtil::exist(_source_orien_task_map, ta_source_type)) {
        std::map<irt::Orientation, std::set<irt_int>>& orien_task_map = _source_orien_task_map[ta_source_type];
        if (RTUtil::exist(orien_task_map, orientation)) {
          std::set<irt_int>& task_set = orien_task_map[orientation];
          if (task_set.size() >= 2) {
            add_cost = true;
          } else {
            add_cost = RTUtil::exist(task_set, task_idx) ? false : true;
          }
        }
      }
      if (add_cost) {
        switch (ta_source_type) {
          case TASourceType::kEnclosure:
            cost += 20;
            break;
          case TASourceType::kOtherPanel:
            cost += 10;
            break;
          case TASourceType::kSelfPanel:
            cost += 5;
            break;
          default:
            break;
        }
      }
    }
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
  std::map<Orientation, TANode*> _neighbor_ptr_map;
  std::map<TASourceType, std::map<Orientation, std::set<irt_int>>> _source_orien_task_map;
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
