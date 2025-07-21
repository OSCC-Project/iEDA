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
#include "LayerRect.hpp"
#include "PABoxId.hpp"
#include "PAIterParam.hpp"
#include "PANode.hpp"
#include "PATask.hpp"
#include "PriorityQueue.hpp"
#include "ScaleAxis.hpp"
#include "Violation.hpp"

namespace irt {

class PABox
{
 public:
  PABox() = default;
  ~PABox() = default;
  // getter
  EXTPlanarRect& get_box_rect() { return _box_rect; }
  PABoxId& get_pa_box_id() { return _pa_box_id; }
  PAIterParam* get_pa_iter_param() { return _pa_iter_param; }
  bool get_initial_routing() const { return _initial_routing; }
  std::vector<PATask*>& get_pa_task_list() { return _pa_task_list; }
  std::map<bool, std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>>>& get_type_layer_net_fixed_rect_map() { return _type_layer_net_fixed_rect_map; }
  std::map<int32_t, std::map<int32_t, std::set<Segment<LayerCoord>*>>>& get_net_pin_access_result_map() { return _net_pin_access_result_map; }
  std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>>>>& get_net_task_access_result_map() { return _net_task_access_result_map; }
  std::vector<Violation>& get_violation_list() { return _violation_list; }
  ScaleAxis& get_box_track_axis() { return _box_track_axis; }
  std::vector<GridMap<PANode>>& get_layer_node_map() { return _layer_node_map; }
  std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>>>>& get_best_net_task_access_result_map() { return _best_net_task_access_result_map; }
  std::vector<Violation>& get_best_violation_list() { return _best_violation_list; }
  // setter
  void set_box_rect(const EXTPlanarRect& box_rect) { _box_rect = box_rect; }
  void set_pa_box_id(const PABoxId& pa_box_id) { _pa_box_id = pa_box_id; }
  void set_pa_iter_param(PAIterParam* pa_iter_param) { _pa_iter_param = pa_iter_param; }
  void set_initial_routing(const bool initial_routing) { _initial_routing = initial_routing; }
  void set_pa_task_list(const std::vector<PATask*>& pa_task_list) { _pa_task_list = pa_task_list; }
  void set_type_layer_net_fixed_rect_map(const std::map<bool, std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>>>& type_layer_net_fixed_rect_map)
  {
    _type_layer_net_fixed_rect_map = type_layer_net_fixed_rect_map;
  }
  void set_net_pin_access_result_map(const std::map<int32_t, std::map<int32_t, std::set<Segment<LayerCoord>*>>>& net_pin_access_result_map)
  {
    _net_pin_access_result_map = net_pin_access_result_map;
  }
  void set_net_task_access_result_map(const std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>>>>& net_task_access_result_map)
  {
    _net_task_access_result_map = net_task_access_result_map;
  }
  void set_violation_list(const std::vector<Violation>& violation_list) { _violation_list = violation_list; }
  void set_box_track_axis(const ScaleAxis& box_track_axis) { _box_track_axis = box_track_axis; }
  void set_layer_node_map(const std::vector<GridMap<PANode>>& layer_node_map) { _layer_node_map = layer_node_map; }
  void set_best_net_task_access_result_map(const std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>>>>& best_net_task_access_result_map)
  {
    _best_net_task_access_result_map = best_net_task_access_result_map;
  }
  void set_best_violation_list(const std::vector<Violation>& best_violation_list) { _best_violation_list = best_violation_list; }
  // function
#if 1  // astar
  // single task
  PATask* get_curr_pa_task() { return _curr_pa_task; }
  std::vector<std::vector<PANode*>>& get_start_node_list_list() { return _start_node_list_list; }
  std::vector<std::vector<PANode*>>& get_end_node_list_list() { return _end_node_list_list; }
  std::vector<PANode*>& get_path_node_list() { return _path_node_list; }
  std::vector<PANode*>& get_single_task_visited_node_list() { return _single_task_visited_node_list; }
  std::vector<Segment<LayerCoord>>& get_routing_segment_list() { return _routing_segment_list; }
  void set_curr_pa_task(PATask* curr_pa_task) { _curr_pa_task = curr_pa_task; }
  void set_start_node_list_list(const std::vector<std::vector<PANode*>>& start_node_list_list) { _start_node_list_list = start_node_list_list; }
  void set_end_node_list_list(const std::vector<std::vector<PANode*>>& end_node_list_list) { _end_node_list_list = end_node_list_list; }
  void set_path_node_list(const std::vector<PANode*>& path_node_list) { _path_node_list = path_node_list; }
  void set_single_task_visited_node_list(const std::vector<PANode*>& single_task_visited_node_list)
  {
    _single_task_visited_node_list = single_task_visited_node_list;
  }
  void set_routing_segment_list(const std::vector<Segment<LayerCoord>>& routing_segment_list) { _routing_segment_list = routing_segment_list; }
  // single path
  PriorityQueue<PANode*, std::vector<PANode*>, CmpPANodeCost>& get_open_queue() { return _open_queue; }
  std::vector<PANode*>& get_single_path_visited_node_list() { return _single_path_visited_node_list; }
  PANode* get_path_head_node() { return _path_head_node; }
  int32_t get_end_node_list_idx() const { return _end_node_list_idx; }
  void set_open_queue(const PriorityQueue<PANode*, std::vector<PANode*>, CmpPANodeCost>& open_queue) { _open_queue = open_queue; }
  void set_single_path_visited_node_list(const std::vector<PANode*>& single_path_visited_node_list)
  {
    _single_path_visited_node_list = single_path_visited_node_list;
  }
  void set_path_head_node(PANode* path_head_node) { _path_head_node = path_head_node; }
  void set_end_node_list_idx(const int32_t end_node_list_idx) { _end_node_list_idx = end_node_list_idx; }
#endif

 private:
  EXTPlanarRect _box_rect;
  PABoxId _pa_box_id;
  PAIterParam* _pa_iter_param = nullptr;
  bool _initial_routing = true;
  std::vector<PATask*> _pa_task_list;
  std::map<bool, std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>>> _type_layer_net_fixed_rect_map;
  std::map<int32_t, std::map<int32_t, std::set<Segment<LayerCoord>*>>> _net_pin_access_result_map;
  std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>>>> _net_task_access_result_map;
  std::vector<Violation> _violation_list;
  ScaleAxis _box_track_axis;
  std::vector<GridMap<PANode>> _layer_node_map;
  std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>>>> _best_net_task_access_result_map;
  std::vector<Violation> _best_violation_list;
#if 1  // astar
  // single task
  PATask* _curr_pa_task = nullptr;
  std::vector<std::vector<PANode*>> _start_node_list_list;
  std::vector<std::vector<PANode*>> _end_node_list_list;
  std::vector<PANode*> _path_node_list;
  std::vector<PANode*> _single_task_visited_node_list;
  std::vector<Segment<LayerCoord>> _routing_segment_list;
  // single path
  PriorityQueue<PANode*, std::vector<PANode*>, CmpPANodeCost> _open_queue;
  std::vector<PANode*> _single_path_visited_node_list;
  PANode* _path_head_node = nullptr;
  int32_t _end_node_list_idx = -1;
#endif
};

}  // namespace irt
