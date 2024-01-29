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

#include "IRModelStat.hpp"
#include "IRNet.hpp"
#include "IRNode.hpp"
#include "IRTask.hpp"
#include "GridMap.hpp"
#include "PriorityQueue.hpp"

namespace irt {

class IRModel
{
 public:
  IRModel() = default;
  ~IRModel() = default;
  // getter
  std::vector<GridMap<IRNode>>& get_layer_node_map() { return _layer_node_map; }
  std::vector<IRNet>& get_ir_net_list() { return _ir_net_list; }
  std::vector<std::vector<irt_int>>& get_net_order_list_list() { return _net_order_list_list; }
  std::map<LayerCoord, std::set<Orientation>, CmpLayerCoordByXASC>& get_history_grid_access_orien_map()
  {
    return _history_grid_access_orien_map;
  }
  std::set<LayerCoord, CmpLayerCoordByXASC>& get_ripup_grid_set() { return _ripup_grid_set; }
  IRModelStat& get_ir_model_stat() { return _ir_model_stat; }
  irt_int get_curr_iter() { return _curr_iter; }
  // setter
  void set_layer_node_map(const std::vector<GridMap<IRNode>>& layer_node_map) { _layer_node_map = layer_node_map; }
  void set_ir_net_list(const std::vector<IRNet>& ir_net_list) { _ir_net_list = ir_net_list; }
  void set_net_order_list_list(const std::vector<std::vector<irt_int>>& net_order_list_list) { _net_order_list_list = net_order_list_list; }
  void set_history_grid_access_orien_map(
      const std::map<LayerCoord, std::set<Orientation>, CmpLayerCoordByXASC>& history_grid_access_orien_map)
  {
    _history_grid_access_orien_map = history_grid_access_orien_map;
  }
  void set_ripup_grid_set(const std::set<LayerCoord, CmpLayerCoordByXASC>& ripup_grid_set) { _ripup_grid_set = ripup_grid_set; }
  void set_ir_model_stat(const IRModelStat& ir_model_stat) { _ir_model_stat = ir_model_stat; }
  void set_curr_iter(const irt_int curr_iter) { _curr_iter = curr_iter; }
#if 1  // astar
  // single net
  const irt_int get_curr_net_idx() const { return _ir_net_ref->get_net_idx(); }
  const PlanarRect& get_curr_bounding_box() const { return _ir_net_ref->get_bounding_box().get_grid_rect(); }
  const GridMap<double>& get_curr_cost_map() const { return _ir_net_ref->get_ra_cost_map(); }
  PlanarRect& get_routing_region() { return _routing_region; }
  std::vector<IRTask>& get_ir_task_list() { return _ir_task_list; }
  std::vector<Segment<IRNode*>>& get_node_segment_list() { return _node_segment_list; }
  void set_ir_net_ref(IRNet* ir_net_ref) { _ir_net_ref = ir_net_ref; }
  void set_routing_region(const PlanarRect& routing_region) { _routing_region = routing_region; }
  void set_ir_task_list(const std::vector<IRTask>& ir_task_list) { _ir_task_list = ir_task_list; }
  void set_node_segment_list(const std::vector<Segment<IRNode*>>& node_segment_list) { _node_segment_list = node_segment_list; }
  // single task
  std::vector<IRGroup>& get_start_group_list() { return _start_group_list; }
  std::vector<IRGroup>& get_end_group_list() { return _end_group_list; }
  IRGroup& get_path_group() { return _path_group; }
  void set_start_group_list(const std::vector<IRGroup>& start_group_list) { _start_group_list = start_group_list; }
  void set_end_group_list(const std::vector<IRGroup>& end_group_list) { _end_group_list = end_group_list; }
  void set_path_group(const IRGroup& path_group) { _path_group = path_group; }
  // single path
  PriorityQueue<IRNode*, std::vector<IRNode*>, CmpIRNodeCost>& get_open_queue() { return _open_queue; }
  std::vector<IRNode*>& get_visited_node_list() { return _visited_node_list; }
  IRNode* get_path_head_node() { return _path_head_node; }
  irt_int get_end_group_idx() const { return _end_group_idx; }
  void set_open_queue(const PriorityQueue<IRNode*, std::vector<IRNode*>, CmpIRNodeCost>& open_queue) { _open_queue = open_queue; }
  void set_visited_node_list(const std::vector<IRNode*>& visited_node_list) { _visited_node_list = visited_node_list; }
  void set_path_head_node(IRNode* path_head_node) { _path_head_node = path_head_node; }
  void set_end_group_idx(const irt_int end_group_idx) { _end_group_idx = end_group_idx; }
#endif

 private:
  std::vector<GridMap<IRNode>> _layer_node_map;
  std::vector<IRNet> _ir_net_list;
  /**
   * _net_order_list_list.back()作为即将要跑的序
   */
  std::vector<std::vector<irt_int>> _net_order_list_list;
  /**
   * 对_history_grid_access_orien_map内的方向加history_cost
   */
  std::map<LayerCoord, std::set<Orientation>, CmpLayerCoordByXASC> _history_grid_access_orien_map;
  /**
   * 对_ripup_grid_access_orien_map内通过的线网进行拆线
   */
  std::set<LayerCoord, CmpLayerCoordByXASC> _ripup_grid_set;
  IRModelStat _ir_model_stat;
  irt_int _curr_iter = -1;
#if 1  // astar
  // single net
  IRNet* _ir_net_ref = nullptr;
  PlanarRect _routing_region;
  std::vector<IRTask> _ir_task_list;
  std::vector<Segment<IRNode*>> _node_segment_list;
  // single task
  std::vector<IRGroup> _start_group_list;
  std::vector<IRGroup> _end_group_list;
  IRGroup _path_group;
  // single path
  PriorityQueue<IRNode*, std::vector<IRNode*>, CmpIRNodeCost> _open_queue;
  std::vector<IRNode*> _visited_node_list;
  IRNode* _path_head_node = nullptr;
  irt_int _end_group_idx = -1;
#endif
};

}  // namespace irt
