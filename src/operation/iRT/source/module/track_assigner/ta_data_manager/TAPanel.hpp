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

#include "LayerRect.hpp"
#include "RTAPI.hpp"
#include "ScaleAxis.hpp"
#include "TANode.hpp"
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
  irt_int get_panel_idx() const { return _panel_idx; }
  std::map<TASourceType, std::map<irt_int, std::vector<LayerRect>>>& get_source_net_rect_map() { return _source_net_rect_map; }
  std::map<TASourceType, void*>& get_source_region_query_map() { return _source_region_query_map; }
  ScaleAxis& get_panel_scale_axis() { return _panel_scale_axis; }
  std::vector<TATask>& get_ta_task_list() { return _ta_task_list; }
  GridMap<TANode>& get_ta_node_map() { return _ta_node_map; }
  TAPanelStat& get_ta_panel_stat() { return _ta_panel_stat; }
  // setter
  void set_panel_idx(const irt_int panel_idx) { _panel_idx = panel_idx; }
  void set_source_net_rect_map(const std::map<TASourceType, std::map<irt_int, std::vector<LayerRect>>>& source_net_rect_map)
  {
    _source_net_rect_map = source_net_rect_map;
  }
  void set_source_region_query_map(const std::map<TASourceType, void*>& source_region_query_map)
  {
    _source_region_query_map = source_region_query_map;
  }
  void set_panel_scale_axis(const ScaleAxis& panel_scale_axis) { _panel_scale_axis = panel_scale_axis; }
  void set_ta_task_list(const std::vector<TATask>& ta_task_list) { _ta_task_list = ta_task_list; }
  void set_ta_node_map(const GridMap<TANode>& ta_node_map) { _ta_node_map = ta_node_map; }
  // function
  bool skipAssigning() { return _ta_task_list.empty(); }
  void addRect(TASourceType ta_source_type, irt_int net_idx, const LayerRect& rect)
  {
    _source_net_rect_map[ta_source_type][net_idx].push_back(rect);
    RTAPI_INST.addEnvRectList(_source_region_query_map[ta_source_type], rect);
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
  const irt_int get_curr_task_idx() const { return _ta_task_ref->get_task_idx(); }
  const PlanarRect& get_curr_bounding_box() const { return _ta_task_ref->get_bounding_box(); }
  const std::map<LayerCoord, double, CmpLayerCoordByXASC>& get_curr_coord_cost_map() const { return _ta_task_ref->get_coord_cost_map(); }
  PlanarRect& get_routing_region() { return _routing_region; }
  std::vector<std::vector<TANode*>>& get_start_node_comb_list() { return _start_node_comb_list; }
  std::vector<std::vector<TANode*>>& get_end_node_comb_list() { return _end_node_comb_list; }
  std::set<Orientation>& get_routing_offset_set() { return _routing_offset_set; }
  std::vector<TANode*>& get_path_node_comb() { return _path_node_comb; }
  std::vector<Segment<TANode*>>& get_node_segment_list() { return _node_segment_list; }
  void set_ta_task_ref(TATask* ta_task_ref) { _ta_task_ref = ta_task_ref; }
  void set_routing_region(const PlanarRect& routing_region) { _routing_region = routing_region; }
  void set_start_node_comb_list(const std::vector<std::vector<TANode*>>& start_node_comb_list)
  {
    _start_node_comb_list = start_node_comb_list;
  }
  void set_end_node_comb_list(const std::vector<std::vector<TANode*>>& end_node_comb_list) { _end_node_comb_list = end_node_comb_list; }
  void set_routing_offset_set(const std::set<Orientation>& routing_offset_set) { _routing_offset_set = routing_offset_set; }
  void set_path_node_comb(const std::vector<TANode*>& path_node_comb) { _path_node_comb = path_node_comb; }
  void set_node_segment_list(const std::vector<Segment<TANode*>>& node_segment_list) { _node_segment_list = node_segment_list; }
  // single path
  TARouteStrategy& get_ta_route_strategy() { return _ta_route_strategy; }
  std::priority_queue<TANode*, std::vector<TANode*>, CmpTANodeCost>& get_open_queue() { return _open_queue; }
  std::vector<TANode*>& get_visited_node_list() { return _visited_node_list; }
  TANode* get_path_head_node() { return _path_head_node; }
  irt_int get_end_node_comb_idx() const { return _end_node_comb_idx; }
  void set_ta_route_strategy(const TARouteStrategy& ta_route_strategy) { _ta_route_strategy = ta_route_strategy; }
  void set_open_queue(const std::priority_queue<TANode*, std::vector<TANode*>, CmpTANodeCost>& open_queue) { _open_queue = open_queue; }
  void set_visited_node_list(const std::vector<TANode*>& visited_node_list) { _visited_node_list = visited_node_list; }
  void set_path_head_node(TANode* path_head_node) { _path_head_node = path_head_node; }
  void set_end_node_comb_idx(const irt_int end_node_comb_idx) { _end_node_comb_idx = end_node_comb_idx; }
#endif

 private:
  irt_int _panel_idx = -1;
  /**
   * TASourceType::kBlockage 存储blockage
   * TASourceType::kOtherPanelResult 存储其他panel的结果
   * TASourceType::kSelfPanelResult 存储自己panel的结果
   */
  std::map<TASourceType, std::map<irt_int, std::vector<LayerRect>>> _source_net_rect_map;
  std::map<TASourceType, void*> _source_region_query_map;
  ScaleAxis _panel_scale_axis;
  std::vector<TATask> _ta_task_list;
  GridMap<TANode> _ta_node_map;
  TAPanelStat _ta_panel_stat;
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
  std::vector<TANode*> _path_node_comb;
  std::vector<Segment<TANode*>> _node_segment_list;
  // single path
  TARouteStrategy _ta_route_strategy = TARouteStrategy::kNone;
  std::priority_queue<TANode*, std::vector<TANode*>, CmpTANodeCost> _open_queue;
  std::vector<TANode*> _visited_node_list;
  TANode* _path_head_node = nullptr;
  irt_int _end_node_comb_idx = -1;
#endif
};

}  // namespace irt