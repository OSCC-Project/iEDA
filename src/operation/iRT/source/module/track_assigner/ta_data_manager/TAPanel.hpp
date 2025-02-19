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
#include "RTHeader.hpp"
#include "ScaleAxis.hpp"
#include "TAComParam.hpp"
#include "TANode.hpp"
#include "TAPanelId.hpp"
#include "TATask.hpp"

namespace irt {

class TAPanel
{
 public:
  TAPanel() = default;
  ~TAPanel() = default;
  // getter
  EXTLayerRect& get_panel_rect() { return _panel_rect; }
  TAPanelId& get_ta_panel_id() { return _ta_panel_id; }
  TAComParam* get_ta_com_param() { return _ta_com_param; }
  std::vector<TATask*>& get_ta_task_list() { return _ta_task_list; }
  std::map<int32_t, std::set<EXTLayerRect*>>& get_net_fixed_rect_map() { return _net_fixed_rect_map; }
  std::map<int32_t, std::map<int32_t, std::vector<LayerRect>>>& get_net_pin_access_result_map() { return _net_pin_access_result_map; }
  std::map<int32_t, std::vector<LayerRect>>& get_net_detailed_result_map() { return _net_detailed_result_map; }
  std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>>>>& get_net_task_result_map() { return _net_task_result_map; }
  std::vector<Violation>& get_violation_list() { return _violation_list; }
  ScaleAxis& get_panel_track_axis() { return _panel_track_axis; }
  GridMap<TANode>& get_ta_node_map() { return _ta_node_map; }
  // setter
  void set_panel_rect(const EXTLayerRect& panel_rect) { _panel_rect = panel_rect; }
  void set_ta_panel_id(const TAPanelId& ta_panel_id) { _ta_panel_id = ta_panel_id; }
  void set_ta_com_param(TAComParam* ta_com_param) { _ta_com_param = ta_com_param; }
  void set_ta_task_list(const std::vector<TATask*>& ta_task_list) { _ta_task_list = ta_task_list; }
  void set_net_fixed_rect_map(const std::map<int32_t, std::set<EXTLayerRect*>>& net_fixed_rect_map) { _net_fixed_rect_map = net_fixed_rect_map; }
  void set_net_pin_access_result_map(const std::map<int32_t, std::map<int32_t, std::vector<LayerRect>>>& net_pin_access_result_map)
  {
    _net_pin_access_result_map = net_pin_access_result_map;
  }
  void set_net_detailed_result_map(const std::map<int32_t, std::vector<LayerRect>>& net_detailed_result_map)
  {
    _net_detailed_result_map = net_detailed_result_map;
  }
  void set_violation_list(const std::vector<Violation>& violation_list) { _violation_list = violation_list; }
  void set_panel_track_axis(const ScaleAxis& panel_track_axis) { _panel_track_axis = panel_track_axis; }
  void set_ta_node_map(const GridMap<TANode>& ta_node_map) { _ta_node_map = ta_node_map; }
  // function
#if 1  // astar
  // single task
  TATask* get_curr_ta_task() { return _curr_ta_task; }
  std::vector<std::vector<TANode*>>& get_start_node_list_list() { return _start_node_list_list; }
  std::vector<std::vector<TANode*>>& get_end_node_list_list() { return _end_node_list_list; }
  std::vector<TANode*>& get_path_node_list() { return _path_node_list; }
  std::vector<TANode*>& get_single_task_visited_node_list() { return _single_task_visited_node_list; }
  std::vector<Segment<LayerCoord>>& get_routing_segment_list() { return _routing_segment_list; }
  void set_curr_ta_task(TATask* curr_ta_task) { _curr_ta_task = curr_ta_task; }
  void set_start_node_list_list(const std::vector<std::vector<TANode*>>& start_node_list_list) { _start_node_list_list = start_node_list_list; }
  void set_end_node_list_list(const std::vector<std::vector<TANode*>>& end_node_list_list) { _end_node_list_list = end_node_list_list; }
  void set_path_node_list(const std::vector<TANode*>& path_node_list) { _path_node_list = path_node_list; }
  void set_single_task_visited_node_list(const std::vector<TANode*>& single_task_visited_node_list)
  {
    _single_task_visited_node_list = single_task_visited_node_list;
  }
  void set_routing_segment_list(const std::vector<Segment<LayerCoord>>& routing_segment_list) { _routing_segment_list = routing_segment_list; }
  // single path
  PriorityQueue<TANode*, std::vector<TANode*>, CmpTANodeCost>& get_open_queue() { return _open_queue; }
  std::vector<TANode*>& get_single_path_visited_node_list() { return _single_path_visited_node_list; }
  TANode* get_path_head_node() { return _path_head_node; }
  int32_t get_end_node_list_idx() const { return _end_node_list_idx; }
  void set_open_queue(const PriorityQueue<TANode*, std::vector<TANode*>, CmpTANodeCost>& open_queue) { _open_queue = open_queue; }
  void set_single_path_visited_node_list(const std::vector<TANode*>& single_path_visited_node_list)
  {
    _single_path_visited_node_list = single_path_visited_node_list;
  }
  void set_path_head_node(TANode* path_head_node) { _path_head_node = path_head_node; }
  void set_end_node_list_idx(const int32_t end_node_list_idx) { _end_node_list_idx = end_node_list_idx; }
#endif

 private:
  EXTLayerRect _panel_rect;
  TAPanelId _ta_panel_id;
  TAComParam* _ta_com_param = nullptr;
  std::vector<TATask*> _ta_task_list;
  std::map<int32_t, std::set<EXTLayerRect*>> _net_fixed_rect_map;
  std::map<int32_t, std::map<int32_t, std::vector<LayerRect>>> _net_pin_access_result_map;
  std::map<int32_t, std::vector<LayerRect>> _net_detailed_result_map;
  std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>>>> _net_task_result_map;
  std::vector<Violation> _violation_list;
  ScaleAxis _panel_track_axis;
  GridMap<TANode> _ta_node_map;
#if 1  // astar
  // single task
  TATask* _curr_ta_task = nullptr;
  std::vector<std::vector<TANode*>> _start_node_list_list;
  std::vector<std::vector<TANode*>> _end_node_list_list;
  std::vector<TANode*> _path_node_list;
  std::vector<TANode*> _single_task_visited_node_list;
  std::vector<Segment<LayerCoord>> _routing_segment_list;
  // single path
  PriorityQueue<TANode*, std::vector<TANode*>, CmpTANodeCost> _open_queue;
  std::vector<TANode*> _single_path_visited_node_list;
  TANode* _path_head_node = nullptr;
  int32_t _end_node_list_idx = -1;
#endif
};

}  // namespace irt