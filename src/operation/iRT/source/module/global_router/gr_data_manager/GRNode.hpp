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

#include "DRCChecker.hpp"
#include "GRSourceType.hpp"
#include "LayerCoord.hpp"
#include "RegionQuery.hpp"

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
  PlanarRect& get_base_region() { return _base_region; }
  std::map<Orientation, GRNode*>& get_neighbor_ptr_map() { return _neighbor_ptr_map; }
  std::map<GRSourceType, RegionQuery*>& get_source_region_query_map() { return _source_region_query_map; }
  irt_int get_whole_wire_demand() const { return _whole_wire_demand; }
  irt_int get_whole_via_demand() const { return _whole_via_demand; }
  std::map<irt_int, std::map<Orientation, irt_int>>& get_net_orien_wire_demand_map() { return _net_orien_wire_demand_map; }
  std::map<Orientation, irt_int>& get_orien_access_supply_map() { return _orien_access_supply_map; }
  std::map<Orientation, irt_int>& get_orien_access_demand_map() { return _orien_access_demand_map; }
  irt_int get_resource_supply() const { return _resource_supply; }
  irt_int get_resource_demand() const { return _resource_demand; }
  std::map<Orientation, double>& get_history_orien_access_cost_map() { return _history_orien_access_cost_map; }
  double get_history_resource_cost() const { return _history_resource_cost; }
  std::set<irt_int>& get_passed_net_set() { return _passed_net_set; }
  // setter
  void set_base_region(const PlanarRect& base_region) { _base_region = base_region; }
  void set_neighbor_ptr_map(const std::map<Orientation, GRNode*>& neighbor_ptr_map) { _neighbor_ptr_map = neighbor_ptr_map; }
  void set_source_region_query_map(const std::map<GRSourceType, RegionQuery*>& source_region_query_map)
  {
    _source_region_query_map = source_region_query_map;
  }
  void set_whole_wire_demand(const irt_int whole_wire_demand) { _whole_wire_demand = whole_wire_demand; }
  void set_whole_via_demand(const irt_int whole_via_demand) { _whole_via_demand = whole_via_demand; }
  void set_net_orien_wire_demand_map(const std::map<irt_int, std::map<Orientation, irt_int>>& net_orien_wire_demand_map)
  {
    _net_orien_wire_demand_map = net_orien_wire_demand_map;
  }
  void set_orien_access_supply_map(const std::map<Orientation, irt_int>& orien_access_supply_map)
  {
    _orien_access_supply_map = orien_access_supply_map;
  }
  void set_orien_access_demand_map(const std::map<Orientation, irt_int>& orien_access_demand_map)
  {
    _orien_access_demand_map = orien_access_demand_map;
  }
  void set_resource_supply(const irt_int resource_supply) { _resource_supply = resource_supply; }
  void set_resource_demand(const irt_int resource_demand) { _resource_demand = resource_demand; }
  void set_history_orien_access_cost_map(const std::map<Orientation, double>& history_orien_access_cost_map)
  {
    _history_orien_access_cost_map = history_orien_access_cost_map;
  }
  void set_history_resource_cost(const double history_resource_cost) { _history_resource_cost = history_resource_cost; }
  void set_passed_net_set(const std::set<irt_int>& passed_net_set) { _passed_net_set = passed_net_set; }
  // function
  GRNode* getNeighborNode(Orientation orientation)
  {
    GRNode* neighbor_node = nullptr;
    if (RTUtil::exist(_neighbor_ptr_map, orientation)) {
      neighbor_node = _neighbor_ptr_map[orientation];
    }
    return neighbor_node;
  }
  RegionQuery* getRegionQuery(GRSourceType gr_source_type)
  {
    RegionQuery*& region_query = _source_region_query_map[gr_source_type];
    if (region_query == nullptr) {
      region_query = DC_INST.initRegionQuery();
    }
    return region_query;
  }
  double getCost(irt_int net_idx, Orientation orientation)
  {
    double cost = 0;
    if (orientation != Orientation::kUp && orientation != Orientation::kDown) {
      // 对于平面来说 需要先判断方向
      irt_int access_supply = 0;
      irt_int access_demand = 0;
      if (RTUtil::exist(_orien_access_supply_map, orientation)) {
        access_supply = _orien_access_supply_map[orientation];
      }
      if (RTUtil::exist(_orien_access_demand_map, orientation)) {
        access_demand = _orien_access_demand_map[orientation];
      }
      cost += calcCost(1 + access_demand, access_supply);
      if (RTUtil::exist(_history_orien_access_cost_map, orientation)) {
        cost += _history_orien_access_cost_map[orientation];
      }
    }
    if (orientation != Orientation::kUp && orientation != Orientation::kDown) {
      // 需要区分是整根线还是线网内demand
      irt_int wire_demand = _whole_wire_demand;
      if (RTUtil::exist(_net_orien_wire_demand_map, net_idx)) {
        if (RTUtil::exist(_net_orien_wire_demand_map[net_idx], orientation)) {
          wire_demand = _net_orien_wire_demand_map[net_idx][orientation];
        }
      }
      cost += calcCost(wire_demand + _resource_demand, _resource_supply);
      cost += _history_resource_cost;
    } else {
      // 对于up和down来说 只有via_demand
      cost += calcCost(_whole_via_demand + _resource_demand, _resource_supply);
      cost += _history_resource_cost;
    }
    return cost;
  }
  double calcCost(irt_int demand, irt_int supply)
  {
    double cost = 0;
    if (supply != 0) {
      cost = static_cast<double>(demand) / supply;
    } else {
      cost = static_cast<double>(demand);
    }
    cost = std::max(static_cast<double>(0), 1 + std::log10(cost));
    return cost;
  }
  void updateDemand(irt_int net_idx, std::set<Orientation> orien_set, ChangeType change_type)
  {
    if (RTUtil::exist(orien_set, Orientation::kEast) || RTUtil::exist(orien_set, Orientation::kWest)
        || RTUtil::exist(orien_set, Orientation::kSouth) || RTUtil::exist(orien_set, Orientation::kNorth)) {
      orien_set.erase(Orientation::kUp);
      orien_set.erase(Orientation::kDown);
      if (orien_set.size() > 2) {
        LOG_INST.error(Loc::current(), "The size of orien_set > 2!");
      }
      bool has_net_demand = false;
      irt_int wire_demand = 0;
      if (RTUtil::exist(_net_orien_wire_demand_map, net_idx)) {
        for (Orientation orientation : orien_set) {
          if (RTUtil::exist(_net_orien_wire_demand_map[net_idx], orientation)) {
            wire_demand += _net_orien_wire_demand_map[net_idx][orientation];
            has_net_demand = true;
          }
        }
      }
      if (!has_net_demand) {
        wire_demand = _whole_wire_demand;
      }
      if (change_type == ChangeType::kAdd) {
        _resource_demand += wire_demand;
      } else if (change_type == ChangeType::kDel) {
        _resource_demand -= wire_demand;
      }

      for (Orientation orientation : orien_set) {
        if (change_type == ChangeType::kAdd) {
          _orien_access_demand_map[orientation]++;
        } else if (change_type == ChangeType::kDel) {
          _orien_access_demand_map[orientation]--;
        }
      }
    } else if (RTUtil::exist(orien_set, Orientation::kUp) || RTUtil::exist(orien_set, Orientation::kDown)) {
      if (change_type == ChangeType::kAdd) {
        _resource_demand += _whole_via_demand;
      } else if (change_type == ChangeType::kDel) {
        _resource_demand -= _whole_via_demand;
      }
    }
    if (change_type == ChangeType::kAdd) {
      _passed_net_set.insert(net_idx);
    } else if (change_type == ChangeType::kDel) {
      _passed_net_set.erase(net_idx);
    }
  }
#if 1  // astar
  // single net
  std::set<Direction>& get_direction_set() { return _direction_set; }
  void set_direction_set(std::set<Direction>& direction_set) { _direction_set = direction_set; }
  // single path
  GRNodeState& get_state() { return _state; }
  GRNode* get_parent_node() const { return _parent_node; }
  double get_known_cost() const { return _known_cost; }
  double get_estimated_cost() const { return _estimated_cost; }
  void set_state(GRNodeState state) { _state = state; }
  void set_parent_node(GRNode* parent_node) { _parent_node = parent_node; }
  void set_known_cost(const double known_cost) { _known_cost = known_cost; }
  void set_estimated_cost(const double estimated_cost) { _estimated_cost = estimated_cost; }
  // function
  bool isNone() { return _state == GRNodeState::kNone; }
  bool isOpen() { return _state == GRNodeState::kOpen; }
  bool isClose() { return _state == GRNodeState::kClose; }
  double getTotalCost() { return (_known_cost + _estimated_cost); }
#endif

 private:
  PlanarRect _base_region;
  std::map<Orientation, GRNode*> _neighbor_ptr_map;
  std::map<GRSourceType, RegionQuery*> _source_region_query_map;
  /**
   * gcell 布线结果该算多少demand?
   *
   * _whole_via_demand  一个完整的gr_via所需要的资源(以当前层最小面积做为参考)，不是真via
   * _whole_wire_demand 一个完整的贯穿gcell的wire，中间布线结果用这个，包括T字或十字
   * _net_orien_wire_demand_map 布线端点处使用此资源，减少直接使用whole_wire_demand的浪费
   */
  irt_int _whole_wire_demand = 0;
  irt_int _whole_via_demand = 0;
  std::map<irt_int, std::map<Orientation, irt_int>> _net_orien_wire_demand_map;
  /**
   * gcell 入口控制
   *
   * _orien_access_supply_map 方向与对应的入口数
   * _orien_access_demand_map 方向与消耗的入口数
   */
  std::map<Orientation, irt_int> _orien_access_supply_map;
  std::map<Orientation, irt_int> _orien_access_demand_map;
  /**
   * gcell 集成资源
   *
   * _resource_supply 能使用的供给
   * _resource_demand 能使用的需求
   */
  irt_int _resource_supply = 0;
  irt_int _resource_demand = 0;
  /**
   * gcell 历史代价
   */
  std::map<Orientation, double> _history_orien_access_cost_map;
  double _history_resource_cost = 0.0;
  /**
   * gcell 从此node经过的线网
   */
  std::set<irt_int> _passed_net_set;
#if 1  // astar
  // single net
  std::set<Direction> _direction_set;
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
