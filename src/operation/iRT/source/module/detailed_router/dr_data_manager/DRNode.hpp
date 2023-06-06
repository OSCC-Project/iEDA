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

#include "DRRouteStrategy.hpp"
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
  irt_int get_fence_violation_cost() const { return _fence_violation_cost; }
  std::map<Orientation, DRNode*>& get_neighbor_ptr_map() { return _neighbor_ptr_map; }
  std::map<Orientation, std::set<irt_int>>& get_obs_task_map() { return _obs_task_map; }
  std::map<Orientation, std::set<irt_int>>& get_fence_task_map() { return _fence_task_map; }
  std::map<Orientation, std::set<irt_int>>& get_env_task_map() { return _env_task_map; }
  std::queue<irt_int>& get_task_queue() { return _task_queue; }
  // setter
  void set_fence_violation_cost(const irt_int fence_violation_cost) { _fence_violation_cost = fence_violation_cost; }
  void set_neighbor_ptr_map(const std::map<Orientation, DRNode*>& neighbor_ptr_map) { _neighbor_ptr_map = neighbor_ptr_map; }
  void set_obs_task_map(const std::map<Orientation, std::set<irt_int>>& obs_task_map) { _obs_task_map = obs_task_map; }
  void set_fence_task_map(const std::map<Orientation, std::set<irt_int>>& fence_task_map) { _fence_task_map = fence_task_map; }
  void set_env_task_map(const std::map<Orientation, std::set<irt_int>>& env_task_map) { _env_task_map = env_task_map; }
  void set_task_queue(const std::queue<irt_int>& task_queue) { _task_queue = task_queue; }
  // function
  DRNode* getNeighborNode(Orientation orientation)
  {
    DRNode* neighbor_node = nullptr;
    if (RTUtil::exist(_neighbor_ptr_map, orientation)) {
      neighbor_node = _neighbor_ptr_map[orientation];
    }
    return neighbor_node;
  }
  bool isOBS(irt_int task_idx, Orientation orientation, DRRouteStrategy dr_route_strategy)
  {
    bool is_obs = false;
    if (dr_route_strategy == DRRouteStrategy::kIgnoringOBS) {
      return is_obs;
    }
    if (RTUtil::exist(_obs_task_map, orientation)) {
      if (_obs_task_map[orientation].size() >= 2) {
        is_obs = true;
      } else {
        is_obs = RTUtil::exist(_obs_task_map[orientation], task_idx) ? false : true;
      }
    }
    if (dr_route_strategy == DRRouteStrategy::kIgnoringFence) {
      return is_obs;
    }
    if (!is_obs) {
      if (RTUtil::exist(_fence_task_map, orientation)) {
        if (_fence_task_map[orientation].size() >= 2) {
          is_obs = true;
        } else {
          is_obs = RTUtil::exist(_fence_task_map[orientation], task_idx) ? false : true;
        }
      }
    }
    if (dr_route_strategy == DRRouteStrategy::kIgnoringENV) {
      return is_obs;
    }
    if (!is_obs) {
      if (RTUtil::exist(_env_task_map, orientation)) {
        if (_env_task_map[orientation].size() >= 2) {
          is_obs = true;
        } else {
          is_obs = RTUtil::exist(_env_task_map[orientation], task_idx) ? false : true;
        }
      }
    }

    return is_obs;
  }
  double getCost(irt_int task_idx, Orientation orientation)
  {
    irt_int fence_violation_num = 0;
    if (RTUtil::exist(_fence_task_map, orientation)) {
      std::set<irt_int>& task_idx_set = _fence_task_map[orientation];
      if (task_idx_set.size() >= 2) {
        fence_violation_num += static_cast<irt_int>(task_idx_set.size());
      } else {
        fence_violation_num += RTUtil::exist(task_idx_set, task_idx) ? 0 : 1;
      }
    }
    if (RTUtil::exist(_env_task_map, orientation)) {
      fence_violation_num += static_cast<irt_int>(_env_task_map[orientation].size());
    }
    return static_cast<double>(fence_violation_num * _fence_violation_cost);
  }
  void addEnv(irt_int task_idx, Orientation orientation) { _env_task_map[orientation].insert(task_idx); }
  void addDemand(irt_int task_idx) { _task_queue.push(task_idx); }
#if 1  // astar
  std::set<Orientation>& get_orientation_set() { return _orientation_set; }
  DRNodeState& get_state() { return _state; }
  DRNode* get_parent_node() const { return _parent_node; }
  double get_known_cost() const { return _known_cost; }
  double get_estimated_cost() const { return _estimated_cost; }
  void set_orientation_set(std::set<Orientation>& orientation_set) { _orientation_set = orientation_set; }
  void set_state(DRNodeState state) { _state = state; }
  void set_parent_node(DRNode* parent_node) { _parent_node = parent_node; }
  void set_known_cost(const double known_cost) { _known_cost = known_cost; }
  void set_estimated_cost(const double estimated_cost) { _estimated_cost = estimated_cost; }
  bool isNone() { return _state == DRNodeState::kNone; }
  bool isOpen() { return _state == DRNodeState::kOpen; }
  bool isClose() { return _state == DRNodeState::kClose; }
  double getTotalCost() { return (_known_cost + _estimated_cost); }
#endif

 private:
  irt_int _fence_violation_cost = 0;
  std::map<Orientation, DRNode*> _neighbor_ptr_map;
  std::map<Orientation, std::set<irt_int>> _obs_task_map;
  std::map<Orientation, std::set<irt_int>> _fence_task_map;
  std::map<Orientation, std::set<irt_int>> _env_task_map;
  std::queue<irt_int> _task_queue;
#if 1  // astar
  // single net
  std::set<Orientation> _orientation_set;
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
