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
#include "PABoxId.hpp"
#include "PAIterParam.hpp"
#include "PANode.hpp"
#include "PAPatch.hpp"
#include "PAShadow.hpp"
#include "PATask.hpp"
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
  std::map<bool, std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>>>& get_type_layer_net_fixed_rect_map() { return _type_layer_net_fixed_rect_map; }
  std::map<int32_t, std::set<AccessPoint*, CmpAccessPoint>>& get_net_access_point_map() { return _net_access_point_map; }
  std::map<int32_t, std::map<int32_t, std::set<Segment<LayerCoord>*>>>& get_net_pin_access_result_map() { return _net_pin_access_result_map; }
  std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>>>>& get_net_task_access_result_map() { return _net_task_access_result_map; }
  std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>>& get_net_pin_access_patch_map() { return _net_pin_access_patch_map; }
  std::map<int32_t, std::map<int32_t, std::vector<EXTLayerRect>>>& get_net_task_access_patch_map() { return _net_task_access_patch_map; }
  std::vector<PATask*>& get_pa_task_list() { return _pa_task_list; }
  std::vector<Violation>& get_route_violation_list() { return _route_violation_list; }
  ScaleAxis& get_box_track_axis() { return _box_track_axis; }
  std::vector<GridMap<PANode>>& get_layer_node_map() { return _layer_node_map; }
  std::vector<PAShadow>& get_layer_shadow_map() { return _layer_shadow_map; }
  std::map<int32_t, std::pair<std::set<int32_t>, std::set<int32_t>>>& get_layer_axis_map() { return _layer_axis_map; }
  std::map<PAPin*, AccessPoint>& get_pin_access_point_map() { return _pin_access_point_map; }
  std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>>>>& get_best_net_task_access_result_map() { return _best_net_task_access_result_map; }
  std::map<int32_t, std::map<int32_t, std::vector<EXTLayerRect>>>& get_best_net_task_access_patch_map() { return _best_net_task_access_patch_map; }
  std::map<PAPin*, AccessPoint>& get_best_pin_access_point_map() { return _best_pin_access_point_map; }
  std::vector<Violation>& get_best_route_violation_list() { return _best_route_violation_list; }
  // setter
  void set_box_rect(const EXTPlanarRect& box_rect) { _box_rect = box_rect; }
  void set_pa_box_id(const PABoxId& pa_box_id) { _pa_box_id = pa_box_id; }
  void set_pa_iter_param(PAIterParam* pa_iter_param) { _pa_iter_param = pa_iter_param; }
  void set_initial_routing(const bool initial_routing) { _initial_routing = initial_routing; }
  void set_type_layer_net_fixed_rect_map(const std::map<bool, std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>>>& type_layer_net_fixed_rect_map)
  {
    _type_layer_net_fixed_rect_map = type_layer_net_fixed_rect_map;
  }
  void set_net_access_point_map(const std::map<int32_t, std::set<AccessPoint*, CmpAccessPoint>>& net_access_point_map)
  {
    _net_access_point_map = net_access_point_map;
  }
  void set_net_pin_access_result_map(const std::map<int32_t, std::map<int32_t, std::set<Segment<LayerCoord>*>>>& net_pin_access_result_map)
  {
    _net_pin_access_result_map = net_pin_access_result_map;
  }
  void set_net_task_access_result_map(const std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>>>>& net_task_access_result_map)
  {
    _net_task_access_result_map = net_task_access_result_map;
  }
  void set_net_pin_access_patch_map(const std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>>& net_pin_access_patch_map)
  {
    _net_pin_access_patch_map = net_pin_access_patch_map;
  }
  void set_net_task_access_patch_map(const std::map<int32_t, std::map<int32_t, std::vector<EXTLayerRect>>>& net_task_access_patch_map)
  {
    _net_task_access_patch_map = net_task_access_patch_map;
  }
  void set_pa_task_list(const std::vector<PATask*>& pa_task_list) { _pa_task_list = pa_task_list; }
  void set_route_violation_list(const std::vector<Violation>& route_violation_list) { _route_violation_list = route_violation_list; }
  void set_box_track_axis(const ScaleAxis& box_track_axis) { _box_track_axis = box_track_axis; }
  void set_layer_node_map(const std::vector<GridMap<PANode>>& layer_node_map) { _layer_node_map = layer_node_map; }
  void set_layer_shadow_map(const std::vector<PAShadow>& layer_shadow_map) { _layer_shadow_map = layer_shadow_map; }
  void set_layer_axis_map(const std::map<int32_t, std::pair<std::set<int32_t>, std::set<int32_t>>>& layer_axis_map) { _layer_axis_map = layer_axis_map; }
  void set_pin_access_point_map(const std::map<PAPin*, AccessPoint>& pin_access_point_map) { _pin_access_point_map = pin_access_point_map; }
  void set_best_net_task_access_result_map(const std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>>>>& best_net_task_access_result_map)
  {
    _best_net_task_access_result_map = best_net_task_access_result_map;
  }
  void set_best_net_task_access_patch_map(const std::map<int32_t, std::map<int32_t, std::vector<EXTLayerRect>>>& best_net_task_access_patch_map)
  {
    _best_net_task_access_patch_map = best_net_task_access_patch_map;
  }
  void set_best_pin_access_point_map(const std::map<PAPin*, AccessPoint>& best_pin_access_point_map) { _best_pin_access_point_map = best_pin_access_point_map; }
  void set_best_route_violation_list(const std::vector<Violation>& best_route_violation_list) { _best_route_violation_list = best_route_violation_list; }
  // function
#if 1  // astar
  // single task
  PATask* get_curr_route_task() { return _curr_route_task; }
  std::vector<std::vector<PANode*>>& get_start_node_list_list() { return _start_node_list_list; }
  std::vector<std::vector<PANode*>>& get_end_node_list_list() { return _end_node_list_list; }
  std::vector<PANode*>& get_path_node_list() { return _path_node_list; }
  std::vector<PANode*>& get_single_task_visited_node_list() { return _single_task_visited_node_list; }
  std::vector<Segment<LayerCoord>>& get_routing_segment_list() { return _routing_segment_list; }
  void set_curr_route_task(PATask* curr_route_task) { _curr_route_task = curr_route_task; }
  void set_start_node_list_list(const std::vector<std::vector<PANode*>>& start_node_list_list) { _start_node_list_list = start_node_list_list; }
  void set_end_node_list_list(const std::vector<std::vector<PANode*>>& end_node_list_list) { _end_node_list_list = end_node_list_list; }
  void set_path_node_list(const std::vector<PANode*>& path_node_list) { _path_node_list = path_node_list; }
  void set_single_task_visited_node_list(const std::vector<PANode*>& single_task_visited_node_list)
  {
    _single_task_visited_node_list = single_task_visited_node_list;
  }
  void set_routing_segment_list(const std::vector<Segment<LayerCoord>>& routing_segment_list) { _routing_segment_list = routing_segment_list; }
  // single path
  OpenQueue<PANode>& get_open_queue() { return _open_queue; }
  std::vector<PANode*>& get_single_path_visited_node_list() { return _single_path_visited_node_list; }
  PANode* get_path_head_node() { return _path_head_node; }
  int32_t get_end_node_list_idx() const { return _end_node_list_idx; }
  void set_open_queue(const OpenQueue<PANode>& open_queue) { _open_queue = open_queue; }
  void set_single_path_visited_node_list(const std::vector<PANode*>& single_path_visited_node_list)
  {
    _single_path_visited_node_list = single_path_visited_node_list;
  }
  void set_path_head_node(PANode* path_head_node) { _path_head_node = path_head_node; }
  void set_end_node_list_idx(const int32_t end_node_list_idx) { _end_node_list_idx = end_node_list_idx; }
#endif
#if 1  // zstar
  // single task
  PATask* get_curr_patch_task() { return _curr_patch_task; }
  std::vector<EXTLayerRect>& get_routing_patch_list() { return _routing_patch_list; }
  std::vector<Violation>& get_patch_violation_list() { return _patch_violation_list; }
  std::set<Violation, CmpViolation>& get_tried_fix_violation_set() { return _tried_fix_violation_set; }
  void set_curr_patch_task(PATask* curr_patch_task) { _curr_patch_task = curr_patch_task; }
  void set_routing_patch_list(const std::vector<EXTLayerRect>& routing_patch_list) { _routing_patch_list = routing_patch_list; }
  void set_patch_violation_list(const std::vector<Violation>& patch_violation_list) { _patch_violation_list = patch_violation_list; }
  void set_tried_fix_violation_set(const std::set<Violation, CmpViolation>& tried_fix_violation_set) { _tried_fix_violation_set = tried_fix_violation_set; }
  // single violation
  Violation& get_curr_patch_violation() { return _curr_patch_violation; }
  void set_curr_patch_violation(const Violation& curr_patch_violation) { _curr_patch_violation = curr_patch_violation; }
#endif

 private:
  EXTPlanarRect _box_rect;
  PABoxId _pa_box_id;
  PAIterParam* _pa_iter_param = nullptr;
  bool _initial_routing = true;
  std::map<bool, std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>>> _type_layer_net_fixed_rect_map;
  std::map<int32_t, std::set<AccessPoint*, CmpAccessPoint>> _net_access_point_map;
  std::map<int32_t, std::map<int32_t, std::set<Segment<LayerCoord>*>>> _net_pin_access_result_map;
  std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>>>> _net_task_access_result_map;
  std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>> _net_pin_access_patch_map;
  std::map<int32_t, std::map<int32_t, std::vector<EXTLayerRect>>> _net_task_access_patch_map;
  std::vector<PATask*> _pa_task_list;
  std::vector<Violation> _route_violation_list;
  ScaleAxis _box_track_axis;
  std::vector<GridMap<PANode>> _layer_node_map;
  std::vector<PAShadow> _layer_shadow_map;
  std::map<int32_t, std::pair<std::set<int32_t>, std::set<int32_t>>> _layer_axis_map;
  std::map<PAPin*, AccessPoint> _pin_access_point_map;
  std::map<int32_t, std::map<int32_t, std::vector<Segment<LayerCoord>>>> _best_net_task_access_result_map;
  std::map<int32_t, std::map<int32_t, std::vector<EXTLayerRect>>> _best_net_task_access_patch_map;
  std::map<PAPin*, AccessPoint> _best_pin_access_point_map;
  std::vector<Violation> _best_route_violation_list;
#if 1  // astar
  // single task
  PATask* _curr_route_task = nullptr;
  std::vector<std::vector<PANode*>> _start_node_list_list;
  std::vector<std::vector<PANode*>> _end_node_list_list;
  std::vector<PANode*> _path_node_list;
  std::vector<PANode*> _single_task_visited_node_list;
  std::vector<Segment<LayerCoord>> _routing_segment_list;
  // single path
  OpenQueue<PANode> _open_queue;
  std::vector<PANode*> _single_path_visited_node_list;
  PANode* _path_head_node = nullptr;
  int32_t _end_node_list_idx = -1;
#endif
#if 1  // zstar
  // single task
  PATask* _curr_patch_task = nullptr;
  std::vector<EXTLayerRect> _routing_patch_list;
  std::vector<Violation> _patch_violation_list;
  std::set<Violation, CmpViolation> _tried_fix_violation_set;
  // single violation
  Violation _curr_patch_violation;
#endif
};

}  // namespace irt
