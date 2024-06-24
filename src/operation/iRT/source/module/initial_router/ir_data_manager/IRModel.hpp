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

#include "IRNet.hpp"
#include "IRNode.hpp"
#include "IRParameter.hpp"
#include "IRTopo.hpp"
#include "PriorityQueue.hpp"

namespace irt {

class IRModel
{
 public:
  IRModel() = default;
  ~IRModel() = default;
  // getter
  std::vector<IRNet>& get_ir_net_list() { return _ir_net_list; }
  IRParameter& get_ir_parameter() { return _ir_parameter; }
  std::vector<IRNet*>& get_ir_task_list() { return _ir_task_list; }
  std::vector<GridMap<IRNode>>& get_layer_node_map() { return _layer_node_map; }
  // setter
  void set_ir_net_list(const std::vector<IRNet>& ir_net_list) { _ir_net_list = ir_net_list; }
  void set_ir_parameter(const IRParameter& ir_parameter) { _ir_parameter = ir_parameter; }
  void set_ir_task_list(const std::vector<IRNet*>& ir_task_list) { _ir_task_list = ir_task_list; }
  void set_layer_node_map(const std::vector<GridMap<IRNode>>& layer_node_map) { _layer_node_map = layer_node_map; }
  // function
#if 1  // astar
  // single topo
  IRTopo* get_curr_ir_topo() { return _curr_ir_topo; }
  std::vector<std::vector<IRNode*>>& get_start_node_list_list() { return _start_node_list_list; }
  std::vector<std::vector<IRNode*>>& get_end_node_list_list() { return _end_node_list_list; }
  std::vector<IRNode*>& get_path_node_list() { return _path_node_list; }
  std::vector<IRNode*>& get_single_topo_visited_node_list() { return _single_topo_visited_node_list; }
  std::vector<Segment<LayerCoord>>& get_routing_segment_list() { return _routing_segment_list; }
  void set_curr_ir_topo(IRTopo* curr_ir_topo) { _curr_ir_topo = curr_ir_topo; }
  void set_start_node_list_list(const std::vector<std::vector<IRNode*>>& start_node_list_list)
  {
    _start_node_list_list = start_node_list_list;
  }
  void set_end_node_list_list(const std::vector<std::vector<IRNode*>>& end_node_list_list) { _end_node_list_list = end_node_list_list; }
  void set_path_node_list(const std::vector<IRNode*>& path_node_list) { _path_node_list = path_node_list; }
  void set_single_topo_visited_node_list(const std::vector<IRNode*>& single_topo_visited_node_list)
  {
    _single_topo_visited_node_list = single_topo_visited_node_list;
  }
  void set_routing_segment_list(const std::vector<Segment<LayerCoord>>& routing_segment_list)
  {
    _routing_segment_list = routing_segment_list;
  }
  // single path
  PriorityQueue<IRNode*, std::vector<IRNode*>, CmpIRNodeCost>& get_open_queue() { return _open_queue; }
  std::vector<IRNode*>& get_single_path_visited_node_list() { return _single_path_visited_node_list; }
  IRNode* get_path_head_node() { return _path_head_node; }
  int32_t get_end_node_list_idx() const { return _end_node_list_idx; }
  void set_open_queue(const PriorityQueue<IRNode*, std::vector<IRNode*>, CmpIRNodeCost>& open_queue) { _open_queue = open_queue; }
  void set_single_path_visited_node_list(const std::vector<IRNode*>& single_path_visited_node_list)
  {
    _single_path_visited_node_list = single_path_visited_node_list;
  }
  void set_path_head_node(IRNode* path_head_node) { _path_head_node = path_head_node; }
  void set_end_node_list_idx(const int32_t end_node_list_idx) { _end_node_list_idx = end_node_list_idx; }
#endif

 private:
  std::vector<IRNet> _ir_net_list;
  IRParameter _ir_parameter;
  std::vector<IRNet*> _ir_task_list;
  std::vector<GridMap<IRNode>> _layer_node_map;
#if 1  // astar
  // single topo
  IRTopo* _curr_ir_topo = nullptr;
  std::vector<std::vector<IRNode*>> _start_node_list_list;
  std::vector<std::vector<IRNode*>> _end_node_list_list;
  std::vector<IRNode*> _path_node_list;
  std::vector<IRNode*> _single_topo_visited_node_list;
  std::vector<Segment<LayerCoord>> _routing_segment_list;
  // single path
  PriorityQueue<IRNode*, std::vector<IRNode*>, CmpIRNodeCost> _open_queue;
  std::vector<IRNode*> _single_path_visited_node_list;
  IRNode* _path_head_node = nullptr;
  int32_t _end_node_list_idx = -1;
#endif
};

}  // namespace irt
