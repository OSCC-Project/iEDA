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
#include "PriorityQueue.hpp"
#include "RTAPI.hpp"
#include "RTU.hpp"
#include "RegionQuery.hpp"
#include "ScaleAxis.hpp"
#include "TANode.hpp"
#include "TAPanelId.hpp"
#include "TATask.hpp"
#include "TAParameter.hpp"

namespace irt {

class TAPanel
{
 public:
  TAPanel() = default;
  ~TAPanel() = default;
  // getter
  EXTLayerRect& get_panel_rect() { return _panel_rect; }
  TAPanelId& get_ta_panel_id() { return _ta_panel_id; }
  TAParameter* get_curr_ta_parameter() { return _curr_ta_parameter; }
  std::vector<TATask*>& get_ta_task_list() { return _ta_task_list; }
  std::map<irt_int, std::set<EXTLayerRect*>>& get_net_fixed_rect_map() { return _net_fixed_rect_map; }
  std::vector<Violation>& get_violation_list() { return _violation_list; }
  ScaleAxis& get_panel_track_axis() { return _panel_track_axis; }
  GridMap<TANode>& get_ta_node_map() { return _ta_node_map; }
  // setter
  void set_panel_rect(const EXTLayerRect& panel_rect) { _panel_rect = panel_rect; }
  void set_ta_panel_id(const TAPanelId& ta_panel_id) { _ta_panel_id = ta_panel_id; }
  void set_curr_ta_parameter(TAParameter* curr_ta_parameter) { _curr_ta_parameter = curr_ta_parameter; }
  void set_ta_task_list(const std::vector<TATask*>& ta_task_list) { _ta_task_list = ta_task_list; }
  void set_net_fixed_rect_map(const std::map<irt_int, std::set<EXTLayerRect*>>& net_fixed_rect_map)
  {
    _net_fixed_rect_map = net_fixed_rect_map;
  }
  void set_violation_list(const std::vector<Violation>& violation_list) { _violation_list = violation_list; }
  void set_panel_track_axis(const ScaleAxis& panel_track_axis) { _panel_track_axis = panel_track_axis; }
  void set_ta_node_map(const GridMap<TANode>& ta_node_map) { _ta_node_map = ta_node_map; }
  // function
#if 1  // astar
  // single task
  const irt_int get_curr_net_idx() const { return _curr_net_idx; }
  PlanarRect& get_routing_region() { return _routing_region; }
  std::set<Orientation>& get_routing_offset_set() { return _routing_offset_set; }
  std::vector<std::vector<TANode*>>& get_start_node_list_list() { return _start_node_list_list; }
  std::vector<std::vector<TANode*>>& get_end_node_list_list() { return _end_node_list_list; }
  std::vector<TANode*>& get_path_node_list() { return _path_node_list; }
  std::vector<TANode*>& get_single_task_visited_node_list() { return _single_task_visited_node_list; }
  std::vector<Segment<LayerCoord>>& get_routing_segment_list() { return _routing_segment_list; }
  void set_curr_net_idx(const irt_int curr_net_idx) { _curr_net_idx = curr_net_idx; }
  void set_routing_region(const PlanarRect& routing_region) { _routing_region = routing_region; }
  void set_routing_offset_set(const std::set<Orientation>& routing_offset_set) { _routing_offset_set = routing_offset_set; }
  void set_start_node_list_list(const std::vector<std::vector<TANode*>>& start_node_list_list)
  {
    _start_node_list_list = start_node_list_list;
  }
  void set_end_node_list_list(const std::vector<std::vector<TANode*>>& end_node_list_list) { _end_node_list_list = end_node_list_list; }
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
  PriorityQueue<TANode*, std::vector<TANode*>, CmpTANodeCost>& get_open_queue() { return _open_queue; }
  std::vector<TANode*>& get_single_path_visited_node_list() { return _single_path_visited_node_list; }
  TANode* get_path_head_node() { return _path_head_node; }
  irt_int get_end_node_comb_idx() const { return _end_node_comb_idx; }
  void set_open_queue(const PriorityQueue<TANode*, std::vector<TANode*>, CmpTANodeCost>& open_queue) { _open_queue = open_queue; }
  void set_single_path_visited_node_list(const std::vector<TANode*>& single_path_visited_node_list)
  {
    _single_path_visited_node_list = single_path_visited_node_list;
  }
  void set_path_head_node(TANode* path_head_node) { _path_head_node = path_head_node; }
  void set_end_node_comb_idx(const irt_int end_node_comb_idx) { _end_node_comb_idx = end_node_comb_idx; }
#endif

 private:
  EXTLayerRect _panel_rect;
  TAPanelId _ta_panel_id;
  TAParameter* _curr_ta_parameter = nullptr;
  std::vector<TATask*> _ta_task_list;
  std::map<irt_int, std::set<EXTLayerRect*>> _net_fixed_rect_map;
  std::vector<Violation> _violation_list;
  ScaleAxis _panel_track_axis;
  GridMap<TANode> _ta_node_map;
#if 1  // astar
  // single task
  irt_int _curr_net_idx = -1;
  PlanarRect _routing_region;
  std::set<Orientation> _routing_offset_set;
  std::vector<std::vector<TANode*>> _start_node_list_list;
  std::vector<std::vector<TANode*>> _end_node_list_list;
  std::vector<TANode*> _path_node_list;
  std::vector<TANode*> _single_task_visited_node_list;
  std::vector<Segment<LayerCoord>> _routing_segment_list;
  // single path
  PriorityQueue<TANode*, std::vector<TANode*>, CmpTANodeCost> _open_queue;
  std::vector<TANode*> _single_path_visited_node_list;
  TANode* _path_head_node = nullptr;
  irt_int _end_node_comb_idx = -1;
#endif
};

}  // namespace irt