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

#include "DRBoxId.hpp"
#include "DRBoxStat.hpp"
#include "DRCChecker.hpp"
#include "DRNode.hpp"
#include "DRSourceType.hpp"
#include "DRTask.hpp"
#include "LayerCoord.hpp"
#include "LayerRect.hpp"
#include "RTAPI.hpp"
#include "RegionQuery.hpp"
#include "ScaleAxis.hpp"

namespace irt {

class DRBox : public SpaceRegion
{
 public:
  DRBox() = default;
  ~DRBox() = default;
  // getter
  DRBoxId& get_dr_box_id() { return _dr_box_id; }
  ScaleAxis& get_box_track_axis() { return _box_track_axis; }
  std::map<DRSourceType, RegionQuery*>& get_source_region_query_map() { return _source_region_query_map; }
  std::vector<DRTask>& get_dr_task_list() { return _dr_task_list; }
  std::map<irt_int, std::vector<irt_int>>& get_net_task_map() { return _net_task_map; }
  std::vector<GridMap<DRNode>>& get_layer_node_map() { return _layer_node_map; }
  std::vector<std::vector<irt_int>>& get_task_order_list_list() { return _task_order_list_list; }
  DRBoxStat& get_dr_box_stat() { return _dr_box_stat; }
  irt_int get_curr_iter() { return _curr_iter; }
  // setter
  void set_dr_box_id(const DRBoxId& dr_box_id) { _dr_box_id = dr_box_id; }
  void set_box_track_axis(const ScaleAxis& box_track_axis) { _box_track_axis = box_track_axis; }
  void set_source_region_query_map(const std::map<DRSourceType, RegionQuery*>& source_region_query_map)
  {
    _source_region_query_map = source_region_query_map;
  }
  void set_dr_task_list(const std::vector<DRTask>& dr_task_list) { _dr_task_list = dr_task_list; }
  void set_net_task_map(const std::map<irt_int, std::vector<irt_int>>& net_task_map) { _net_task_map = net_task_map; }
  void set_layer_node_map(const std::vector<GridMap<DRNode>>& layer_node_map) { _layer_node_map = layer_node_map; }
  void set_task_order_list_list(const std::vector<std::vector<irt_int>>& task_order_list_list)
  {
    _task_order_list_list = task_order_list_list;
  }
  void set_dr_box_stat(const DRBoxStat& dr_box_stat) { _dr_box_stat = dr_box_stat; }
  void set_curr_iter(const irt_int curr_iter) { _curr_iter = curr_iter; }
  // function
  RegionQuery* getRegionQuery(DRSourceType dr_source_type)
  {
    RegionQuery*& region_query = _source_region_query_map[dr_source_type];
    if (region_query == nullptr) {
      region_query = DC_INST.initRegionQuery();
    }
    return region_query;
  }
#if 1  // astar
  // single task
  const irt_int get_curr_net_idx() const { return _dr_task_ref->get_origin_net_idx(); }
  const irt_int get_curr_task_idx() const { return _dr_task_ref->get_task_idx(); }
  const SpaceRegion& get_curr_bounding_box() const { return _dr_task_ref->get_bounding_box(); }
  const std::map<LayerCoord, double, CmpLayerCoordByXASC>& get_curr_coord_cost_map() const { return _dr_task_ref->get_coord_cost_map(); }
  SpaceRegion& get_routing_region() { return _routing_region; }
  std::vector<std::vector<DRNode*>>& get_start_node_list_list() { return _start_node_list_list; }
  std::vector<std::vector<DRNode*>>& get_end_node_list_list() { return _end_node_list_list; }
  std::vector<DRNode*>& get_path_node_list() { return _path_node_list; }
  std::vector<DRNode*>& get_single_task_visited_node_list() { return _single_task_visited_node_list; }
  std::vector<Segment<LayerCoord>>& get_routing_segment_list() { return _routing_segment_list; }
  void set_dr_task_ref(DRTask* dr_task_ref) { _dr_task_ref = dr_task_ref; }
  void set_routing_region(const SpaceRegion& routing_region) { _routing_region = routing_region; }
  void set_start_node_list_list(const std::vector<std::vector<DRNode*>>& start_node_list_list)
  {
    _start_node_list_list = start_node_list_list;
  }
  void set_end_node_list_list(const std::vector<std::vector<DRNode*>>& end_node_list_list) { _end_node_list_list = end_node_list_list; }
  void set_path_node_list(const std::vector<DRNode*>& path_node_list) { _path_node_list = path_node_list; }
  void set_single_task_visited_node_list(const std::vector<DRNode*>& single_task_visited_node_list)
  {
    _single_task_visited_node_list = single_task_visited_node_list;
  }
  void set_routing_segment_list(const std::vector<Segment<LayerCoord>>& routing_segment_list)
  {
    _routing_segment_list = routing_segment_list;
  }
  // single path
  std::priority_queue<DRNode*, std::vector<DRNode*>, CmpDRNodeCost>& get_open_queue() { return _open_queue; }
  std::vector<DRNode*>& get_single_path_visited_node_list() { return _single_path_visited_node_list; }
  DRNode* get_path_head_node() { return _path_head_node; }
  irt_int get_end_node_comb_idx() const { return _end_node_comb_idx; }
  void set_open_queue(const std::priority_queue<DRNode*, std::vector<DRNode*>, CmpDRNodeCost>& open_queue) { _open_queue = open_queue; }
  void set_single_path_visited_node_list(const std::vector<DRNode*>& single_path_visited_node_list)
  {
    _single_path_visited_node_list = single_path_visited_node_list;
  }
  void set_path_head_node(DRNode* path_head_node) { _path_head_node = path_head_node; }
  void set_end_node_comb_idx(const irt_int end_node_comb_idx) { _end_node_comb_idx = end_node_comb_idx; }
#endif

 private:
  DRBoxId _dr_box_id;
  ScaleAxis _box_track_axis;
  std::map<DRSourceType, RegionQuery*> _source_region_query_map;
  std::vector<DRTask> _dr_task_list;
  std::map<irt_int, std::vector<irt_int>> _net_task_map;
  std::vector<GridMap<DRNode>> _layer_node_map;
  /**
   * _task_order_list_list.back()作为即将要跑的序
   */
  std::vector<std::vector<irt_int>> _task_order_list_list;
  DRBoxStat _dr_box_stat;
  irt_int _curr_iter = -1;
#if 1  // astar
  // single task
  DRTask* _dr_task_ref = nullptr;
  SpaceRegion _routing_region;
  std::vector<std::vector<DRNode*>> _start_node_list_list;
  std::vector<std::vector<DRNode*>> _end_node_list_list;
  std::vector<DRNode*> _path_node_list;
  std::vector<DRNode*> _single_task_visited_node_list;
  std::vector<Segment<LayerCoord>> _routing_segment_list;
  // single path
  std::priority_queue<DRNode*, std::vector<DRNode*>, CmpDRNodeCost> _open_queue;
  std::vector<DRNode*> _single_path_visited_node_list;
  DRNode* _path_head_node = nullptr;
  irt_int _end_node_comb_idx = -1;
#endif
};

}  // namespace irt
