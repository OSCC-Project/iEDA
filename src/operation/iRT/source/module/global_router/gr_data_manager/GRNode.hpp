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

#include "GRRouteStrategy.hpp"
#include "LayerCoord.hpp"

namespace irt {

#if 1  // astar
enum class GRNodeState
{
  kNone = 0,
  kOpen = 1,
  kClose = 2
};
#endif

class GRNode : public LayerCoord
{
 public:
  GRNode() = default;
  ~GRNode() = default;

  // getter
  PlanarRect& get_real_rect() { return _real_rect; }
  std::map<Orientation, GRNode*>& get_neighbor_ptr_map() { return _neighbor_ptr_map; }
  std::map<irt_int, std::vector<PlanarRect>>& get_net_blockage_map() { return _net_blockage_map; }
  std::map<irt_int, std::vector<PlanarRect>>& get_net_fence_region_map() { return _net_fence_region_map; }
  irt_int get_single_wire_area() const { return _single_wire_area; }
  irt_int get_single_via_area() const { return _single_via_area; }
  irt_int get_wire_area_supply() const { return _wire_area_supply; }
  irt_int get_via_area_supply() const { return _via_area_supply; }
  irt_int get_wire_area_demand() const { return _wire_area_demand; }
  irt_int get_via_area_demand() const { return _via_area_demand; }
  std::map<irt_int, std::set<Orientation>>& get_net_access_map() { return _net_access_map; }
  std::queue<irt_int>& get_net_queue() { return _net_queue; }
  // setter
  void set_real_rect(const PlanarRect& real_rect) { _real_rect = real_rect; }
  void set_neighbor_ptr_map(const std::map<Orientation, GRNode*>& neighbor_ptr_map) { _neighbor_ptr_map = neighbor_ptr_map; }
  void set_net_blockage_map(const std::map<irt_int, std::vector<PlanarRect>>& net_blockage_map) { _net_blockage_map = net_blockage_map; }
  void set_net_fence_region_map(const std::map<irt_int, std::vector<PlanarRect>>& net_fence_region_map) { _net_fence_region_map = net_fence_region_map; }
  void set_single_wire_area(const irt_int single_wire_area) { _single_wire_area = single_wire_area; }
  void set_single_via_area(const irt_int single_via_area) { _single_via_area = single_via_area; }
  void set_wire_area_supply(const irt_int wire_area_supply) { _wire_area_supply = wire_area_supply; }
  void set_via_area_supply(const irt_int via_area_supply) { _via_area_supply = via_area_supply; }
  void set_wire_area_demand(const irt_int wire_area_demand) { _wire_area_demand = wire_area_demand; }
  void set_via_area_demand(const irt_int via_area_demand) { _via_area_demand = via_area_demand; }
  void set_net_access_map(const std::map<irt_int, std::set<Orientation>>& net_access_map) { _net_access_map = net_access_map; }
  void set_net_queue(const std::queue<irt_int>& net_queue) { _net_queue = net_queue; }
  // function
  GRNode* getNeighborNode(Orientation orientation)
  {
    GRNode* neighbor_node = nullptr;
    if (RTUtil::exist(_neighbor_ptr_map, orientation)) {
      neighbor_node = _neighbor_ptr_map[orientation];
    }
    return neighbor_node;
  }
  bool isOBS(irt_int net_idx, Orientation orientation, GRRouteStrategy gr_route_strategy)
  {
    bool is_obs = false;
    if (gr_route_strategy == GRRouteStrategy::kIgnoringOBS) {
      return is_obs;
    }
    if (RTUtil::exist(_net_access_map, net_idx)) {
      // net在node中有引导，但是方向不对，视为障碍
      is_obs = !RTUtil::exist(_net_access_map[net_idx], orientation);
    } else {
      if (orientation == Orientation::kUp || orientation == Orientation::kDown) {
        // wire剩余可以给via
        is_obs = ((_wire_area_supply - _wire_area_demand + _via_area_supply - _via_area_demand) < _single_via_area);
      } else {
        // via剩余不可转wire
        is_obs = ((_wire_area_supply - _wire_area_demand + std::min(_via_area_supply - _via_area_demand, 0)) < _single_wire_area);
      }
    }
    return is_obs;
  }
  double getCost(irt_int net_idx, Orientation orientation)
  {
    double cost = 0;
    if (RTUtil::exist(_net_access_map, net_idx)) {
      // net在node中有引导，但是方向不对，视为障碍
      cost += !RTUtil::exist(_net_access_map[net_idx], orientation) ? 1 : 0;
    } else {
      if (orientation == Orientation::kUp || orientation == Orientation::kDown) {
        // wire剩余可以给via
        cost += RTUtil::sigmoid(_via_area_demand, _wire_area_supply - _wire_area_demand + _via_area_supply);
      } else {
        // via剩余不可转wire
        cost += RTUtil::sigmoid(_wire_area_demand, _wire_area_supply + std::min(_via_area_supply - _via_area_demand, 0));
      }
    }
    if (!RTUtil::exist(_net_fence_region_map, net_idx)) {
      cost += static_cast<double>(_net_fence_region_map.size());
    }
    return cost;
  }
  void addDemand(irt_int net_idx, std::set<Orientation> orientation_set)
  {
    if (RTUtil::exist(orientation_set, Orientation::kEast) || RTUtil::exist(orientation_set, Orientation::kWest)
        || RTUtil::exist(orientation_set, Orientation::kSouth) || RTUtil::exist(orientation_set, Orientation::kNorth)) {
      _wire_area_demand += _single_wire_area;
      _net_queue.push(net_idx);
    } else if (RTUtil::exist(orientation_set, Orientation::kUp) || RTUtil::exist(orientation_set, Orientation::kDown)) {
      _via_area_demand += _single_via_area;
      _net_queue.push(net_idx);
    }
  }
#if 1  // astar
  std::set<Orientation>& get_orientation_set() { return _orientation_set; }
  GRNodeState& get_state() { return _state; }
  GRNode* get_parent_node() const { return _parent_node; }
  double get_known_cost() const { return _known_cost; }
  double get_estimated_cost() const { return _estimated_cost; }
  void set_orientation_set(std::set<Orientation>& orientation_set) { _orientation_set = orientation_set; }
  void set_state(GRNodeState state) { _state = state; }
  void set_parent_node(GRNode* parent_node) { _parent_node = parent_node; }
  void set_known_cost(const double known_cost) { _known_cost = known_cost; }
  void set_estimated_cost(const double estimated_cost) { _estimated_cost = estimated_cost; }
  bool isNone() { return _state == GRNodeState::kNone; }
  bool isOpen() { return _state == GRNodeState::kOpen; }
  bool isClose() { return _state == GRNodeState::kClose; }
  double getTotalCost() { return (_known_cost + _estimated_cost); }
#endif

 private:
  PlanarRect _real_rect;
  std::map<Orientation, GRNode*> _neighbor_ptr_map;
  std::map<irt_int, std::vector<PlanarRect>> _net_blockage_map;
  std::map<irt_int, std::vector<PlanarRect>> _net_fence_region_map;
  irt_int _single_wire_area = 0;
  irt_int _single_via_area = 0;
  irt_int _wire_area_supply = 0;
  irt_int _via_area_supply = 0;
  irt_int _wire_area_demand = 0;
  irt_int _via_area_demand = 0;
  /**
   * 路线引导
   *  当对应net出现在引导中时，必须按照引导的方向布线，否则视为障碍
   *  当对应net不在引导中时，视为普通线网布线
   */
  std::map<irt_int, std::set<Orientation>> _net_access_map;
  std::queue<irt_int> _net_queue;
#if 1  // astar
  // single net
  std::set<Orientation> _orientation_set;
  // single path
  GRNodeState _state = GRNodeState::kNone;
  GRNode* _parent_node = nullptr;
  double _known_cost = 0.0;  // include curr
  double _estimated_cost = 0.0;
#endif
};

#if 1  // astar
struct CmpGRNodeCost
{
  bool operator()(GRNode* a, GRNode* b)
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
