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

#include "DRBoxStat.hpp"
#include "DRNode.hpp"
#include "DRSourceType.hpp"
#include "DRTask.hpp"
#include "LayerCoord.hpp"
#include "LayerRect.hpp"
#include "RTAPI.hpp"
#include "ScaleAxis.hpp"

namespace irt {

class DRBox : public DRSpaceRegion
{
 public:
  DRBox() = default;
  ~DRBox() = default;
  // getter
  PlanarCoord& get_grid_coord() { return _grid_coord; }
  std::map<DRSourceType, std::map<irt_int, std::vector<LayerRect>>>& get_source_net_rect_map() { return _source_net_rect_map; }
  std::map<DRSourceType, void*>& get_source_region_query_map() { return _source_region_query_map; }
  ScaleAxis& get_box_scale_axis() { return _box_scale_axis; }
  std::vector<DRTask>& get_dr_task_list() { return _dr_task_list; }
  std::vector<GridMap<DRNode>>& get_layer_node_map() { return _layer_node_map; }
  DRBoxStat& get_dr_box_stat() { return _dr_box_stat; }
  // setter
  void set_grid_coord(const PlanarCoord& grid_coord) { _grid_coord = grid_coord; }
  void set_source_net_rect_map(const std::map<DRSourceType, std::map<irt_int, std::vector<LayerRect>>>& source_net_rect_map)
  {
    _source_net_rect_map = source_net_rect_map;
  }
  void set_source_region_query_map(const std::map<DRSourceType, void*>& source_region_query_map)
  {
    _source_region_query_map = source_region_query_map;
  }
  void set_box_scale_axis(const ScaleAxis& box_scale_axis) { _box_scale_axis = box_scale_axis; }
  void set_dr_task_list(const std::vector<DRTask>& dr_task_list) { _dr_task_list = dr_task_list; }
  void set_layer_node_map(const std::vector<GridMap<DRNode>>& layer_node_map) { _layer_node_map = layer_node_map; }
  // function
  bool skipRouting() { return _dr_task_list.empty(); }
  void addRect(DRSourceType dr_source_type, irt_int net_idx, const LayerRect& rect)
  {
    _source_net_rect_map[dr_source_type][net_idx].push_back(rect);
    RTAPI_INST.addEnvRectList(_source_region_query_map[dr_source_type], rect);
  }
#if 1  // astar
  double get_wire_unit() const { return _wire_unit; }
  double get_corner_unit() const { return _corner_unit; }
  double get_via_unit() const { return _via_unit; }
  const irt_int get_curr_task_idx() const { return _dr_task_ref->get_task_idx(); }
  const DRSpaceRegion& get_curr_bounding_box() const { return _dr_task_ref->get_bounding_box(); }
  const std::map<LayerCoord, double, CmpLayerCoordByXASC>& get_curr_coord_cost_map() const { return _dr_task_ref->get_coord_cost_map(); }
  DRSpaceRegion& get_routing_region() { return _routing_region; }
  std::vector<std::vector<DRNode*>>& get_start_node_comb_list() { return _start_node_comb_list; }
  std::vector<std::vector<DRNode*>>& get_end_node_comb_list() { return _end_node_comb_list; }
  std::vector<DRNode*>& get_path_node_comb() { return _path_node_comb; }
  std::vector<Segment<DRNode*>>& get_node_segment_list() { return _node_segment_list; }
  DRRouteStrategy& get_dr_route_strategy() { return _dr_route_strategy; }
  std::priority_queue<DRNode*, std::vector<DRNode*>, CmpDRNodeCost>& get_open_queue() { return _open_queue; }
  std::vector<DRNode*>& get_visited_node_list() { return _visited_node_list; }
  DRNode* get_path_head_node() { return _path_head_node; }
  irt_int get_end_node_comb_idx() const { return _end_node_comb_idx; }
  void set_wire_unit(const double wire_unit) { _wire_unit = wire_unit; }
  void set_corner_unit(const double corner_unit) { _corner_unit = corner_unit; }
  void set_via_unit(const double via_unit) { _via_unit = via_unit; }
  void set_dr_task_ref(DRTask* dr_task_ref) { _dr_task_ref = dr_task_ref; }
  void set_routing_region(const DRSpaceRegion& routing_region) { _routing_region = routing_region; }
  void set_start_node_comb_list(const std::vector<std::vector<DRNode*>>& start_node_comb_list)
  {
    _start_node_comb_list = start_node_comb_list;
  }
  void set_end_node_comb_list(const std::vector<std::vector<DRNode*>>& end_node_comb_list) { _end_node_comb_list = end_node_comb_list; }
  void set_path_node_comb(const std::vector<DRNode*>& path_node_comb) { _path_node_comb = path_node_comb; }
  void set_node_segment_list(const std::vector<Segment<DRNode*>>& node_segment_list) { _node_segment_list = node_segment_list; }
  void set_dr_route_strategy(const DRRouteStrategy& dr_route_strategy) { _dr_route_strategy = dr_route_strategy; }
  void set_open_queue(const std::priority_queue<DRNode*, std::vector<DRNode*>, CmpDRNodeCost>& open_queue) { _open_queue = open_queue; }
  void set_visited_node_list(const std::vector<DRNode*>& visited_node_list) { _visited_node_list = visited_node_list; }
  void set_path_head_node(DRNode* path_head_node) { _path_head_node = path_head_node; }
  void set_end_node_comb_idx(const irt_int end_node_comb_idx) { _end_node_comb_idx = end_node_comb_idx; }
#endif

 private:
  PlanarCoord _grid_coord;
  /**
   * DRSourceType::kBlockage 存储blockage
   * DRSourceType::kPanelResult 存储ta的结果
   * DRSourceType::kOtherBoxResult 存储其他box的结果
   * DRSourceType::kSelfBoxResult 存储自己box的结果
   */
  std::map<DRSourceType, std::map<irt_int, std::vector<LayerRect>>> _source_net_rect_map;
  std::map<DRSourceType, void*> _source_region_query_map;
  ScaleAxis _box_scale_axis;
  std::vector<DRTask> _dr_task_list;
  std::vector<GridMap<DRNode>> _layer_node_map;
  DRBoxStat _dr_box_stat;
#if 1  // astar
  // config
  double _wire_unit = 1;
  double _corner_unit = 1;
  double _via_unit = 1;
  // single task
  DRTask* _dr_task_ref = nullptr;
  DRSpaceRegion _routing_region;
  std::vector<std::vector<DRNode*>> _start_node_comb_list;
  std::vector<std::vector<DRNode*>> _end_node_comb_list;
  std::vector<DRNode*> _path_node_comb;
  std::vector<Segment<DRNode*>> _node_segment_list;
  // single path
  DRRouteStrategy _dr_route_strategy = DRRouteStrategy::kNone;
  std::priority_queue<DRNode*, std::vector<DRNode*>, CmpDRNodeCost> _open_queue;
  std::vector<DRNode*> _visited_node_list;
  DRNode* _path_head_node = nullptr;
  irt_int _end_node_comb_idx = -1;
#endif
};

}  // namespace irt
