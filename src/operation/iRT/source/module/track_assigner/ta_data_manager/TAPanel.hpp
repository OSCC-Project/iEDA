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
#include "LayerRect.hpp"
#include "RTAPI.hpp"
#include "RTU.hpp"
#include "RegionQuery.hpp"
#include "ScaleAxis.hpp"
#include "TANode.hpp"
#include "TAPanelId.hpp"
#include "TAPanelStat.hpp"
#include "TASourceType.hpp"
#include "TATask.hpp"

namespace irt {

class TAPanel : public LayerRect
{
 public:
  TAPanel() = default;
  ~TAPanel() = default;
  // getter
  TAPanelId& get_ta_panel_id() { return _ta_panel_id; }
  ScaleAxis& get_panel_track_axis() { return _panel_track_axis; }
  std::map<TASourceType, RegionQuery*>& get_source_region_query_map() { return _source_region_query_map; }
  std::vector<TATask>& get_ta_task_list() { return _ta_task_list; }
  std::map<irt_int, std::vector<irt_int>>& get_net_task_map() { return _net_task_map; }
  GridMap<TANode>& get_ta_node_map() { return _ta_node_map; }
  TAPanelStat& get_ta_panel_stat() { return _ta_panel_stat; }
  irt_int get_curr_iter() { return _curr_iter; }
  // setter
  void set_ta_panel_id(const TAPanelId& ta_panel_id) { _ta_panel_id = ta_panel_id; }
  void set_panel_track_axis(const ScaleAxis& panel_track_axis) { _panel_track_axis = panel_track_axis; }
  void set_source_region_query_map(const std::map<TASourceType, RegionQuery*>& source_region_query_map)
  {
    _source_region_query_map = source_region_query_map;
  }
  void set_ta_task_list(const std::vector<TATask>& ta_task_list) { _ta_task_list = ta_task_list; }
  void set_net_task_map(const std::map<irt_int, std::vector<irt_int>>& net_task_map) { _net_task_map = net_task_map; }
  void set_ta_node_map(const GridMap<TANode>& ta_node_map) { _ta_node_map = ta_node_map; }
  void set_ta_panel_stat(const TAPanelStat& ta_panel_stat) { _ta_panel_stat = ta_panel_stat; }
  void set_curr_iter(const irt_int curr_iter) { _curr_iter = curr_iter; }
  // function
  RegionQuery* getRegionQuery(TASourceType ta_source_type)
  {
    if (ta_source_type == TASourceType::kUnknownPanel) {
      LOG_INST.error(Loc::current(), "The ta_source_type is uncategorized!");
    }
    RegionQuery*& region_query = _source_region_query_map[ta_source_type];
    if (region_query == nullptr) {
      region_query = DC_INST.initRegionQuery();
    }
    return region_query;
  }
#if 1  // astar
  // config
  double get_wire_unit() const { return _wire_unit; }
  double get_corner_unit() const { return _corner_unit; }
  double get_via_unit() const { return _via_unit; }
  void set_wire_unit(const double wire_unit) { _wire_unit = wire_unit; }
  void set_corner_unit(const double corner_unit) { _corner_unit = corner_unit; }
  void set_via_unit(const double via_unit) { _via_unit = via_unit; }
  // single task
  const irt_int get_curr_net_idx() const { return _ta_task_ref->get_origin_net_idx(); }
  const irt_int get_curr_task_idx() const { return _ta_task_ref->get_task_idx(); }
  const PlanarRect& get_curr_bounding_box() const { return _ta_task_ref->get_bounding_box(); }
  const std::map<LayerCoord, double, CmpLayerCoordByXASC>& get_curr_coord_cost_map() const { return _ta_task_ref->get_coord_cost_map(); }
  PlanarRect& get_routing_region() { return _routing_region; }
  std::vector<std::vector<TANode*>>& get_start_node_comb_list() { return _start_node_comb_list; }
  std::vector<std::vector<TANode*>>& get_end_node_comb_list() { return _end_node_comb_list; }
  std::set<Orientation>& get_routing_offset_set() { return _routing_offset_set; }
  std::vector<TANode*>& get_path_node_list() { return _path_node_list; }
  std::vector<TANode*>& get_single_task_visited_node_list() { return _single_task_visited_node_list; }
  std::vector<Segment<LayerCoord>>& get_routing_segment_list() { return _routing_segment_list; }
  void set_ta_task_ref(TATask* ta_task_ref) { _ta_task_ref = ta_task_ref; }
  void set_routing_region(const PlanarRect& routing_region) { _routing_region = routing_region; }
  void set_start_node_comb_list(const std::vector<std::vector<TANode*>>& start_node_comb_list)
  {
    _start_node_comb_list = start_node_comb_list;
  }
  void set_end_node_comb_list(const std::vector<std::vector<TANode*>>& end_node_comb_list) { _end_node_comb_list = end_node_comb_list; }
  void set_routing_offset_set(const std::set<Orientation>& routing_offset_set) { _routing_offset_set = routing_offset_set; }
  void set_path_node_list(const std::vector<TANode*>& path_node_list) { _path_node_list = path_node_list; }
  void set_single_task_visited_node_list(const std::vector<TANode*>& single_task_visited_node_list)
  {
    _single_task_visited_node_list = single_task_visited_node_list;
  }
  void set_routing_segment_list(const std::vector<Segment<LayerCoord>>& routing_segment_list)
  {
    _routing_segment_list = routing_segment_list;
  }
  // single path
  TARouteStrategy& get_ta_route_strategy() { return _ta_route_strategy; }
  std::priority_queue<TANode*, std::vector<TANode*>, CmpTANodeCost>& get_open_queue() { return _open_queue; }
  std::vector<TANode*>& get_single_path_visited_node_list() { return _single_path_visited_node_list; }
  TANode* get_path_head_node() { return _path_head_node; }
  irt_int get_end_node_comb_idx() const { return _end_node_comb_idx; }
  void set_ta_route_strategy(const TARouteStrategy& ta_route_strategy) { _ta_route_strategy = ta_route_strategy; }
  void set_open_queue(const std::priority_queue<TANode*, std::vector<TANode*>, CmpTANodeCost>& open_queue) { _open_queue = open_queue; }
  void set_single_path_visited_node_list(const std::vector<TANode*>& single_path_visited_node_list)
  {
    _single_path_visited_node_list = single_path_visited_node_list;
  }
  void set_path_head_node(TANode* path_head_node) { _path_head_node = path_head_node; }
  void set_end_node_comb_idx(const irt_int end_node_comb_idx) { _end_node_comb_idx = end_node_comb_idx; }
#endif

 private:
  TAPanelId _ta_panel_id;
  ScaleAxis _panel_track_axis;
  std::map<TASourceType, RegionQuery*> _source_region_query_map;
  std::vector<TATask> _ta_task_list;
  std::map<irt_int, std::vector<irt_int>> _net_task_map;
  GridMap<TANode> _ta_node_map;
  TAPanelStat _ta_panel_stat;
  irt_int _curr_iter = -1;
#if 1  // astar
  // config
  double _wire_unit = 1;
  double _corner_unit = 1;
  double _via_unit = 1;
  // single task
  TATask* _ta_task_ref = nullptr;
  PlanarRect _routing_region;
  std::vector<std::vector<TANode*>> _start_node_comb_list;
  std::vector<std::vector<TANode*>> _end_node_comb_list;
  std::set<Orientation> _routing_offset_set;
  std::vector<TANode*> _path_node_list;
  std::vector<TANode*> _single_task_visited_node_list;
  std::vector<Segment<LayerCoord>> _routing_segment_list;
  // single path
  TARouteStrategy _ta_route_strategy = TARouteStrategy::kNone;
  std::priority_queue<TANode*, std::vector<TANode*>, CmpTANodeCost> _open_queue;
  std::vector<TANode*> _single_path_visited_node_list;
  TANode* _path_head_node = nullptr;
  irt_int _end_node_comb_idx = -1;
#endif
};

}  // namespace irt