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
  std::map<bool, std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>>>& get_type_layer_net_fixed_rect_map()
  {
    return _type_layer_net_fixed_rect_map;
  }
  std::map<int32_t, std::set<Segment<LayerCoord>*>>& get_net_final_result_map() { return _net_final_result_map; }
  std::map<int32_t, std::set<EXTLayerRect*>>& get_net_final_patch_map() { return _net_final_patch_map; }
  std::map<int32_t, std::vector<Segment<LayerCoord>>>& get_net_task_final_result_map() { return _net_task_final_result_map; }
  std::map<int32_t, std::vector<EXTLayerRect>>& get_net_task_final_patch_map() { return _net_task_final_patch_map; }
  std::vector<Violation>& get_violation_list() { return _violation_list; }
  ScaleAxis& get_box_track_axis() { return _box_track_axis; }
  std::vector<GridMap<VRNode>>& get_layer_node_map() { return _layer_node_map; }
  std::map<int32_t, std::vector<Segment<LayerCoord>>>& get_best_net_task_final_result_map() { return _best_net_task_final_result_map; }
  std::map<int32_t, std::vector<EXTLayerRect>>& get_best_net_task_final_patch_map() { return _best_net_task_final_patch_map; }
  std::vector<Violation>& get_best_violation_list() { return _best_violation_list; }
  // setter
  void set_box_rect(const EXTPlanarRect& box_rect) { _box_rect = box_rect; }
  void set_vr_box_id(const VRBoxId& vr_box_id) { _vr_box_id = vr_box_id; }
  void set_vr_iter_param(VRIterParam* vr_iter_param) { _vr_iter_param = vr_iter_param; }
  void set_vr_task_list(const std::vector<VRTask*>& vr_task_list) { _vr_task_list = vr_task_list; }
  void set_type_layer_net_fixed_rect_map(
      const std::map<bool, std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>>>& type_layer_net_fixed_rect_map)
  {
    _type_layer_net_fixed_rect_map = type_layer_net_fixed_rect_map;
  }
  void set_net_final_result_map(const std::map<int32_t, std::set<Segment<LayerCoord>*>>& net_final_result_map)
  {
    _net_final_result_map = net_final_result_map;
  }
  void set_net_final_patch_map(const std::map<int32_t, std::set<EXTLayerRect*>>& net_final_patch_map)
  {
    _net_final_patch_map = net_final_patch_map;
  }
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
  void set_layer_node_map(const std::vector<GridMap<VRNode>>& layer_node_map) { _layer_node_map = layer_node_map; }
  void set_best_net_task_final_result_map(const std::map<int32_t, std::vector<Segment<LayerCoord>>>& best_net_task_final_result_map)
  {
    _best_net_task_final_result_map = best_net_task_final_result_map;
  }
  void set_best_net_task_final_patch_map(const std::map<int32_t, std::vector<EXTLayerRect>>& best_net_task_final_patch_map)
  {
    _best_net_task_final_patch_map = best_net_task_final_patch_map;
  }
  void set_best_violation_list(const std::vector<Violation>& best_violation_list) { _best_violation_list = best_violation_list; }
  // function
 private:
  EXTPlanarRect _box_rect;
  VRBoxId _vr_box_id;
  VRIterParam* _vr_iter_param = nullptr;
  std::vector<VRTask*> _vr_task_list;
  std::map<bool, std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>>> _type_layer_net_fixed_rect_map;
  std::map<int32_t, std::set<Segment<LayerCoord>*>> _net_final_result_map;
  std::map<int32_t, std::set<EXTLayerRect*>> _net_final_patch_map;
  std::map<int32_t, std::vector<Segment<LayerCoord>>> _net_task_final_result_map;
  std::map<int32_t, std::vector<EXTLayerRect>> _net_task_final_patch_map;
  std::vector<Violation> _violation_list;
  ScaleAxis _box_track_axis;
  std::vector<GridMap<VRNode>> _layer_node_map;
  std::map<int32_t, std::vector<Segment<LayerCoord>>> _best_net_task_final_result_map;
  std::map<int32_t, std::vector<EXTLayerRect>> _best_net_task_final_patch_map;
  std::vector<Violation> _best_violation_list;
};

}  // namespace irt
