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

#include "GRBoxId.hpp"
#include "GRIterParam.hpp"
#include "GRNode.hpp"
#include "GRTask.hpp"
#include "LayerCoord.hpp"
#include "LayerRect.hpp"
#include "PriorityQueue.hpp"
#include "ScaleAxis.hpp"
#include "Violation.hpp"

namespace irt {

class GRBox
{
 public:
  GRBox() = default;
  ~GRBox() = default;
  // getter
  EXTPlanarRect& get_box_rect() { return _box_rect; }
  GRBoxId& get_gr_box_id() { return _gr_box_id; }
  GRIterParam* get_gr_iter_param() { return _gr_iter_param; }
  std::vector<GRTask*>& get_gr_task_list() { return _gr_task_list; }
  std::map<int32_t, std::vector<Segment<LayerCoord>>>& get_net_task_global_result_map() { return _net_task_global_result_map; }
  int32_t get_total_overflow() const { return _total_overflow; }
  std::vector<std::set<int32_t>>& get_overflow_net_set_list() { return _overflow_net_set_list; }
  ScaleAxis& get_box_track_axis() { return _box_track_axis; }
  std::vector<GridMap<GRNode>>& get_layer_node_map() { return _layer_node_map; }
  std::map<int32_t, std::vector<Segment<LayerCoord>>>& get_best_net_task_global_result_map() { return _best_net_task_global_result_map; }
  int32_t get_best_total_overflow() const { return _best_total_overflow; }
  // setter
  void set_box_rect(const EXTPlanarRect& box_rect) { _box_rect = box_rect; }
  void set_gr_box_id(const GRBoxId& gr_box_id) { _gr_box_id = gr_box_id; }
  void set_gr_iter_param(GRIterParam* gr_iter_param) { _gr_iter_param = gr_iter_param; }
  void set_gr_task_list(const std::vector<GRTask*>& gr_task_list) { _gr_task_list = gr_task_list; }
  void set_net_task_global_result_map(const std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_task_global_result_map)
  {
    _net_task_global_result_map = net_task_global_result_map;
  }
  void set_total_overflow(const int32_t total_overflow) { _total_overflow = total_overflow; }
  void set_overflow_net_set_list(const std::vector<std::set<int32_t>>& overflow_net_set_list) { _overflow_net_set_list = overflow_net_set_list; }
  void set_box_track_axis(const ScaleAxis& box_track_axis) { _box_track_axis = box_track_axis; }
  void set_layer_node_map(const std::vector<GridMap<GRNode>>& layer_node_map) { _layer_node_map = layer_node_map; }
  void set_best_net_task_global_result_map(const std::map<int32_t, std::vector<Segment<LayerCoord>>>& best_net_task_global_result_map)
  {
    _best_net_task_global_result_map = best_net_task_global_result_map;
  }
  void set_best_total_overflow(const int32_t best_total_overflow) { _best_total_overflow = best_total_overflow; }
  // function
#if 1  // astar
  // single task
  GRTask* get_curr_gr_task() { return _curr_gr_task; }
  std::vector<std::vector<GRNode*>>& get_start_node_list_list() { return _start_node_list_list; }
  std::vector<std::vector<GRNode*>>& get_end_node_list_list() { return _end_node_list_list; }
  std::vector<GRNode*>& get_path_node_list() { return _path_node_list; }
  std::vector<GRNode*>& get_single_task_visited_node_list() { return _single_task_visited_node_list; }
  std::vector<Segment<LayerCoord>>& get_routing_segment_list() { return _routing_segment_list; }
  void set_curr_gr_task(GRTask* curr_gr_task) { _curr_gr_task = curr_gr_task; }
  void set_start_node_list_list(const std::vector<std::vector<GRNode*>>& start_node_list_list) { _start_node_list_list = start_node_list_list; }
  void set_end_node_list_list(const std::vector<std::vector<GRNode*>>& end_node_list_list) { _end_node_list_list = end_node_list_list; }
  void set_path_node_list(const std::vector<GRNode*>& path_node_list) { _path_node_list = path_node_list; }
  void set_single_task_visited_node_list(const std::vector<GRNode*>& single_task_visited_node_list)
  {
    _single_task_visited_node_list = single_task_visited_node_list;
  }
  void set_routing_segment_list(const std::vector<Segment<LayerCoord>>& routing_segment_list) { _routing_segment_list = routing_segment_list; }
  // single path
  PriorityQueue<GRNode*, std::vector<GRNode*>, CmpGRNodeCost>& get_open_queue() { return _open_queue; }
  std::vector<GRNode*>& get_single_path_visited_node_list() { return _single_path_visited_node_list; }
  GRNode* get_path_head_node() { return _path_head_node; }
  int32_t get_end_node_list_idx() const { return _end_node_list_idx; }
  void set_open_queue(const PriorityQueue<GRNode*, std::vector<GRNode*>, CmpGRNodeCost>& open_queue) { _open_queue = open_queue; }
  void set_single_path_visited_node_list(const std::vector<GRNode*>& single_path_visited_node_list)
  {
    _single_path_visited_node_list = single_path_visited_node_list;
  }
  void set_path_head_node(GRNode* path_head_node) { _path_head_node = path_head_node; }
  void set_end_node_list_idx(const int32_t end_node_list_idx) { _end_node_list_idx = end_node_list_idx; }
#endif

 private:
  EXTPlanarRect _box_rect;
  GRBoxId _gr_box_id;
  GRIterParam* _gr_iter_param = nullptr;
  std::vector<GRTask*> _gr_task_list;
  std::map<int32_t, std::vector<Segment<LayerCoord>>> _net_task_global_result_map;
  int32_t _total_overflow = 0;
  std::vector<std::set<int32_t>> _overflow_net_set_list;
  ScaleAxis _box_track_axis;
  std::vector<GridMap<GRNode>> _layer_node_map;
  std::map<int32_t, std::vector<Segment<LayerCoord>>> _best_net_task_global_result_map;
  int32_t _best_total_overflow = 0;
#if 1  // astar
  // single task
  GRTask* _curr_gr_task = nullptr;
  std::vector<std::vector<GRNode*>> _start_node_list_list;
  std::vector<std::vector<GRNode*>> _end_node_list_list;
  std::vector<GRNode*> _path_node_list;
  std::vector<GRNode*> _single_task_visited_node_list;
  std::vector<Segment<LayerCoord>> _routing_segment_list;
  // single path
  PriorityQueue<GRNode*, std::vector<GRNode*>, CmpGRNodeCost> _open_queue;
  std::vector<GRNode*> _single_path_visited_node_list;
  GRNode* _path_head_node = nullptr;
  int32_t _end_node_list_idx = -1;
#endif
};

}  // namespace irt
