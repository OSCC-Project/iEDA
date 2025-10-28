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
#include "OpenQueue.hpp"
#include "SRBoxId.hpp"
#include "SRIterParam.hpp"
#include "SRNode.hpp"
#include "SRTask.hpp"
#include "ScaleAxis.hpp"
#include "Violation.hpp"

namespace irt {

class SRBox
{
 public:
  SRBox() = default;
  ~SRBox() = default;
  // getter
  EXTPlanarRect& get_box_rect() { return _box_rect; }
  SRBoxId& get_sr_box_id() { return _sr_box_id; }
  SRIterParam* get_sr_iter_param() { return _sr_iter_param; }
  bool get_initial_routing() const { return _initial_routing; }
  std::map<int32_t, std::vector<Segment<LayerCoord>>>& get_net_task_global_result_map() { return _net_task_global_result_map; }
  std::vector<SRTask*>& get_sr_task_list() { return _sr_task_list; }
  double get_total_overflow() const { return _total_overflow; }
  std::vector<std::set<int32_t>>& get_overflow_net_set_list() { return _overflow_net_set_list; }
  ScaleAxis& get_box_track_axis() { return _box_track_axis; }
  std::vector<GridMap<SRNode>>& get_layer_node_map() { return _layer_node_map; }
  std::map<int32_t, std::vector<Segment<LayerCoord>>>& get_best_net_task_global_result_map() { return _best_net_task_global_result_map; }
  double get_best_total_overflow() const { return _best_total_overflow; }
  // setter
  void set_box_rect(const EXTPlanarRect& box_rect) { _box_rect = box_rect; }
  void set_sr_box_id(const SRBoxId& sr_box_id) { _sr_box_id = sr_box_id; }
  void set_sr_iter_param(SRIterParam* sr_iter_param) { _sr_iter_param = sr_iter_param; }
  void set_initial_routing(const bool initial_routing) { _initial_routing = initial_routing; }
  void set_net_task_global_result_map(const std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_task_global_result_map)
  {
    _net_task_global_result_map = net_task_global_result_map;
  }
  void set_sr_task_list(const std::vector<SRTask*>& sr_task_list) { _sr_task_list = sr_task_list; }
  void set_total_overflow(const double total_overflow) { _total_overflow = total_overflow; }
  void set_overflow_net_set_list(const std::vector<std::set<int32_t>>& overflow_net_set_list) { _overflow_net_set_list = overflow_net_set_list; }
  void set_box_track_axis(const ScaleAxis& box_track_axis) { _box_track_axis = box_track_axis; }
  void set_layer_node_map(const std::vector<GridMap<SRNode>>& layer_node_map) { _layer_node_map = layer_node_map; }
  void set_best_net_task_global_result_map(const std::map<int32_t, std::vector<Segment<LayerCoord>>>& best_net_task_global_result_map)
  {
    _best_net_task_global_result_map = best_net_task_global_result_map;
  }
  void set_best_total_overflow(const double best_total_overflow) { _best_total_overflow = best_total_overflow; }
  // function
#if 1  // astar
  // single task
  SRTask* get_curr_sr_task() { return _curr_sr_task; }
  std::vector<std::vector<SRNode*>>& get_start_node_list_list() { return _start_node_list_list; }
  std::vector<std::vector<SRNode*>>& get_end_node_list_list() { return _end_node_list_list; }
  std::vector<SRNode*>& get_path_node_list() { return _path_node_list; }
  std::vector<SRNode*>& get_single_task_visited_node_list() { return _single_task_visited_node_list; }
  std::vector<Segment<LayerCoord>>& get_routing_segment_list() { return _routing_segment_list; }
  void set_curr_sr_task(SRTask* curr_sr_task) { _curr_sr_task = curr_sr_task; }
  void set_start_node_list_list(const std::vector<std::vector<SRNode*>>& start_node_list_list) { _start_node_list_list = start_node_list_list; }
  void set_end_node_list_list(const std::vector<std::vector<SRNode*>>& end_node_list_list) { _end_node_list_list = end_node_list_list; }
  void set_path_node_list(const std::vector<SRNode*>& path_node_list) { _path_node_list = path_node_list; }
  void set_single_task_visited_node_list(const std::vector<SRNode*>& single_task_visited_node_list)
  {
    _single_task_visited_node_list = single_task_visited_node_list;
  }
  void set_routing_segment_list(const std::vector<Segment<LayerCoord>>& routing_segment_list) { _routing_segment_list = routing_segment_list; }
  // single path
  OpenQueue<SRNode>& get_open_queue() { return _open_queue; }
  std::vector<SRNode*>& get_single_path_visited_node_list() { return _single_path_visited_node_list; }
  SRNode* get_path_head_node() { return _path_head_node; }
  int32_t get_end_node_list_idx() const { return _end_node_list_idx; }
  void set_open_queue(const OpenQueue<SRNode>& open_queue) { _open_queue = open_queue; }
  void set_single_path_visited_node_list(const std::vector<SRNode*>& single_path_visited_node_list)
  {
    _single_path_visited_node_list = single_path_visited_node_list;
  }
  void set_path_head_node(SRNode* path_head_node) { _path_head_node = path_head_node; }
  void set_end_node_list_idx(const int32_t end_node_list_idx) { _end_node_list_idx = end_node_list_idx; }
#endif

 private:
  EXTPlanarRect _box_rect;
  SRBoxId _sr_box_id;
  SRIterParam* _sr_iter_param = nullptr;
  bool _initial_routing = true;
  std::map<int32_t, std::vector<Segment<LayerCoord>>> _net_task_global_result_map;
  std::vector<SRTask*> _sr_task_list;
  double _total_overflow = 0;
  std::vector<std::set<int32_t>> _overflow_net_set_list;
  ScaleAxis _box_track_axis;
  std::vector<GridMap<SRNode>> _layer_node_map;
  std::map<int32_t, std::vector<Segment<LayerCoord>>> _best_net_task_global_result_map;
  double _best_total_overflow = 0;
#if 1  // astar
  // single task
  SRTask* _curr_sr_task = nullptr;
  std::vector<std::vector<SRNode*>> _start_node_list_list;
  std::vector<std::vector<SRNode*>> _end_node_list_list;
  std::vector<SRNode*> _path_node_list;
  std::vector<SRNode*> _single_task_visited_node_list;
  std::vector<Segment<LayerCoord>> _routing_segment_list;
  // single path
  OpenQueue<SRNode> _open_queue;
  std::vector<SRNode*> _single_path_visited_node_list;
  SRNode* _path_head_node = nullptr;
  int32_t _end_node_list_idx = -1;
#endif
};

}  // namespace irt
