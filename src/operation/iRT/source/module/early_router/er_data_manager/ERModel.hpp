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

#include "ERComParam.hpp"
#include "ERNet.hpp"
#include "ERNode.hpp"
#include "ERTopo.hpp"
#include "PriorityQueue.hpp"

namespace irt {

class ERModel
{
 public:
  ERModel() = default;
  ~ERModel() = default;
  // getter
  std::vector<ERNet>& get_er_net_list() { return _er_net_list; }
  ERComParam& get_er_com_param() { return _er_com_param; }
  std::vector<ERNet*>& get_er_task_list() { return _er_task_list; }
  GridMap<ERNode>& get_planar_node_map() { return _planar_node_map; }
  std::vector<GridMap<ERNode>>& get_layer_node_map() { return _layer_node_map; }
  // setter
  void set_er_net_list(const std::vector<ERNet>& er_net_list) { _er_net_list = er_net_list; }
  void set_er_com_param(const ERComParam& er_com_param) { _er_com_param = er_com_param; }
  void set_er_task_list(const std::vector<ERNet*>& er_task_list) { _er_task_list = er_task_list; }
  void set_planar_node_map(const GridMap<ERNode>& planar_node_map) { _planar_node_map = planar_node_map; }
  void set_layer_node_map(const std::vector<GridMap<ERNode>>& layer_node_map) { _layer_node_map = layer_node_map; }
  // function
#if 1  // astar
  // single topo
  ERTopo* get_curr_er_topo() { return _curr_er_topo; }
  std::vector<std::vector<ERNode*>>& get_start_node_list_list() { return _start_node_list_list; }
  std::vector<std::vector<ERNode*>>& get_end_node_list_list() { return _end_node_list_list; }
  std::vector<ERNode*>& get_path_node_list() { return _path_node_list; }
  std::vector<ERNode*>& get_single_topo_visited_node_list() { return _single_topo_visited_node_list; }
  std::vector<Segment<LayerCoord>>& get_routing_segment_list() { return _routing_segment_list; }
  void set_curr_er_topo(ERTopo* curr_er_topo) { _curr_er_topo = curr_er_topo; }
  void set_start_node_list_list(const std::vector<std::vector<ERNode*>>& start_node_list_list) { _start_node_list_list = start_node_list_list; }
  void set_end_node_list_list(const std::vector<std::vector<ERNode*>>& end_node_list_list) { _end_node_list_list = end_node_list_list; }
  void set_path_node_list(const std::vector<ERNode*>& path_node_list) { _path_node_list = path_node_list; }
  void set_single_topo_visited_node_list(const std::vector<ERNode*>& single_topo_visited_node_list)
  {
    _single_topo_visited_node_list = single_topo_visited_node_list;
  }
  void set_routing_segment_list(const std::vector<Segment<LayerCoord>>& routing_segment_list) { _routing_segment_list = routing_segment_list; }
  // single path
  PriorityQueue<ERNode*, std::vector<ERNode*>, CmpERNodeCost>& get_open_queue() { return _open_queue; }
  std::vector<ERNode*>& get_single_path_visited_node_list() { return _single_path_visited_node_list; }
  ERNode* get_path_head_node() { return _path_head_node; }
  int32_t get_end_node_list_idx() const { return _end_node_list_idx; }
  void set_open_queue(const PriorityQueue<ERNode*, std::vector<ERNode*>, CmpERNodeCost>& open_queue) { _open_queue = open_queue; }
  void set_single_path_visited_node_list(const std::vector<ERNode*>& single_path_visited_node_list)
  {
    _single_path_visited_node_list = single_path_visited_node_list;
  }
  void set_path_head_node(ERNode* path_head_node) { _path_head_node = path_head_node; }
  void set_end_node_list_idx(const int32_t end_node_list_idx) { _end_node_list_idx = end_node_list_idx; }
#endif

 private:
  std::vector<ERNet> _er_net_list;
  ERComParam _er_com_param;
  std::vector<ERNet*> _er_task_list;
  GridMap<ERNode> _planar_node_map;
  std::vector<GridMap<ERNode>> _layer_node_map;
#if 1  // astar
  // single topo
  ERTopo* _curr_er_topo = nullptr;
  std::vector<std::vector<ERNode*>> _start_node_list_list;
  std::vector<std::vector<ERNode*>> _end_node_list_list;
  std::vector<ERNode*> _path_node_list;
  std::vector<ERNode*> _single_topo_visited_node_list;
  std::vector<Segment<LayerCoord>> _routing_segment_list;
  // single path
  PriorityQueue<ERNode*, std::vector<ERNode*>, CmpERNodeCost> _open_queue;
  std::vector<ERNode*> _single_path_visited_node_list;
  ERNode* _path_head_node = nullptr;
  int32_t _end_node_list_idx = -1;
#endif
};

}  // namespace irt
