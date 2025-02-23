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
#include "PriorityQueue.hpp"
#include "ScaleAxis.hpp"
#include "VRBoxId.hpp"
#include "VRIterParam.hpp"
#include "VRNode.hpp"
#include "VRTask.hpp"
#include "Violation.hpp"

namespace irt {

class VRBox
{
 public:
  VRBox() = default;
  ~VRBox() = default;
  // getter
  EXTPlanarRect& get_box_rect() { return _box_rect; }
  VRBoxId& get_vr_box_id() { return _vr_box_id; }
  VRIterParam* get_vr_iter_param() { return _vr_iter_param; }
  std::vector<VRTask*>& get_vr_task_list() { return _vr_task_list; }
  std::map<bool, std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>>>& get_type_layer_net_fixed_rect_map() { return _type_layer_net_fixed_rect_map; }
  std::map<int32_t, std::set<Segment<LayerCoord>*>>& get_net_final_result_map() { return _net_final_result_map; }
  std::map<int32_t, std::set<EXTLayerRect*>>& get_net_final_patch_map() { return _net_final_patch_map; }
  std::map<int32_t, std::vector<Segment<LayerCoord>>>& get_net_task_final_result_map() { return _net_task_final_result_map; }
  std::map<int32_t, std::vector<EXTLayerRect>>& get_net_task_final_patch_map() { return _net_task_final_patch_map; }
  std::vector<Violation>& get_violation_list() { return _violation_list; }
  ScaleAxis& get_box_track_axis() { return _box_track_axis; }
  std::map<bool, std::map<int32_t, std::map<int32_t, std::set<PlanarRect, CmpPlanarRectByXASC>>>>& get_graph_type_layer_net_fixed_rect_map()
  {
    return _graph_type_layer_net_fixed_rect_map;
  }
  std::map<bool, std::map<int32_t, std::map<int32_t, std::set<PlanarRect, CmpPlanarRectByXASC>>>>& get_graph_type_layer_net_routed_rect_map()
  {
    return _graph_type_layer_net_routed_rect_map;
  }
  std::map<int32_t, std::set<PlanarRect, CmpPlanarRectByXASC>>& get_graph_layer_violation_map() { return _graph_layer_violation_map; }
  // setter
  void set_box_rect(const EXTPlanarRect& box_rect) { _box_rect = box_rect; }
  void set_vr_box_id(const VRBoxId& vr_box_id) { _vr_box_id = vr_box_id; }
  void set_vr_iter_param(VRIterParam* vr_iter_param) { _vr_iter_param = vr_iter_param; }
  void set_vr_task_list(const std::vector<VRTask*>& vr_task_list) { _vr_task_list = vr_task_list; }
  void set_type_layer_net_fixed_rect_map(const std::map<bool, std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>>>& type_layer_net_fixed_rect_map)
  {
    _type_layer_net_fixed_rect_map = type_layer_net_fixed_rect_map;
  }
  void set_net_final_result_map(const std::map<int32_t, std::set<Segment<LayerCoord>*>>& net_final_result_map) { _net_final_result_map = net_final_result_map; }
  void set_net_final_patch_map(const std::map<int32_t, std::set<EXTLayerRect*>>& net_final_patch_map) { _net_final_patch_map = net_final_patch_map; }
  void set_net_task_final_result_map(const std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_task_final_result_map)
  {
    _net_task_final_result_map = net_task_final_result_map;
  }
  void set_net_task_final_patch_map(const std::map<int32_t, std::vector<EXTLayerRect>>& net_task_final_patch_map)
  {
    _net_task_final_patch_map = net_task_final_patch_map;
  }
  void set_violation_list(const std::vector<Violation>& violation_list) { _violation_list = violation_list; }
  void set_box_track_axis(const ScaleAxis& box_track_axis) { _box_track_axis = box_track_axis; }
  void set_graph_type_layer_net_fixed_rect_map(
      const std::map<bool, std::map<int32_t, std::map<int32_t, std::set<PlanarRect, CmpPlanarRectByXASC>>>>& graph_type_layer_net_fixed_rect_map)
  {
    _graph_type_layer_net_fixed_rect_map = graph_type_layer_net_fixed_rect_map;
  }
  void set_graph_type_layer_net_routed_rect_map(
      const std::map<bool, std::map<int32_t, std::map<int32_t, std::set<PlanarRect, CmpPlanarRectByXASC>>>>& graph_type_layer_net_routed_rect_map)
  {
    _graph_type_layer_net_routed_rect_map = graph_type_layer_net_routed_rect_map;
  }
  void set_graph_layer_violation_map(const std::map<int32_t, std::set<PlanarRect, CmpPlanarRectByXASC>>& graph_layer_violation_map)
  {
    _graph_layer_violation_map = graph_layer_violation_map;
  }
  // function
#if 1
  // single task
  VRTask* get_curr_vr_task() { return _curr_vr_task; }
  std::vector<Segment<LayerCoord>>& get_routing_segment_list() { return _routing_segment_list; }
  std::vector<EXTLayerRect>& get_routing_patch_list() { return _routing_patch_list; }
  void set_curr_vr_task(VRTask* curr_vr_task) { _curr_vr_task = curr_vr_task; }
  void set_routing_segment_list(const std::vector<Segment<LayerCoord>>& routing_segment_list) { _routing_segment_list = routing_segment_list; }
  void set_routing_patch_list(const std::vector<EXTLayerRect>& routing_patch_list) { _routing_patch_list = routing_patch_list; }
#endif
 private:
  EXTPlanarRect _box_rect;
  VRBoxId _vr_box_id;
  VRIterParam* _vr_iter_param = nullptr;
  std::map<bool, std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>>> _type_layer_net_fixed_rect_map;
  std::vector<Violation> _violation_list;
  std::vector<VRTask*> _vr_task_list;
  std::map<int32_t, std::set<Segment<LayerCoord>*>> _net_final_result_map;
  std::map<int32_t, std::set<EXTLayerRect*>> _net_final_patch_map;
  std::map<int32_t, std::vector<Segment<LayerCoord>>> _net_task_final_result_map;
  std::map<int32_t, std::vector<EXTLayerRect>> _net_task_final_patch_map;
  ScaleAxis _box_track_axis;
  std::map<bool, std::map<int32_t, std::map<int32_t, std::set<PlanarRect, CmpPlanarRectByXASC>>>> _graph_type_layer_net_fixed_rect_map;
  std::map<bool, std::map<int32_t, std::map<int32_t, std::set<PlanarRect, CmpPlanarRectByXASC>>>> _graph_type_layer_net_routed_rect_map;
  std::map<int32_t, std::set<PlanarRect, CmpPlanarRectByXASC>> _graph_layer_violation_map;
#if 1
  // single task
  VRTask* _curr_vr_task = nullptr;
  std::vector<Segment<LayerCoord>> _routing_segment_list;
  std::vector<EXTLayerRect> _routing_patch_list;
#endif
};

}  // namespace irt
