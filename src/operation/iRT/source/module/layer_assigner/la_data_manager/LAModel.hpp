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

#include "LAComParam.hpp"
#include "LANet.hpp"
#include "LANode.hpp"
#include "LATopo.hpp"
#include "PriorityQueue.hpp"

namespace irt {

class LAModel
{
 public:
  LAModel() = default;
  ~LAModel() = default;
  // getter
  std::vector<LANet>& get_la_net_list() { return _la_net_list; }
  LAComParam& get_la_com_param() { return _la_com_param; }
  std::vector<LANet*>& get_la_task_list() { return _la_task_list; }
  std::vector<GridMap<LANode>>& get_layer_node_map() { return _layer_node_map; }
  // setter
  void set_la_net_list(const std::vector<LANet>& la_net_list) { _la_net_list = la_net_list; }
  void set_la_com_param(const LAComParam& la_com_param) { _la_com_param = la_com_param; }
  void set_la_task_list(const std::vector<LANet*>& la_task_list) { _la_task_list = la_task_list; }
  void set_layer_node_map(const std::vector<GridMap<LANode>>& layer_node_map) { _layer_node_map = layer_node_map; }
  // function
#if 1  // astar
  // single topo
  LATopo* get_curr_la_topo() { return _curr_la_topo; }
  std::vector<std::vector<LANode*>>& get_start_node_list_list() { return _start_node_list_list; }
  std::vector<std::vector<LANode*>>& get_end_node_list_list() { return _end_node_list_list; }
  std::vector<LANode*>& get_path_node_list() { return _path_node_list; }
  std::vector<LANode*>& get_single_topo_visited_node_list() { return _single_topo_visited_node_list; }
  std::vector<Segment<LayerCoord>>& get_routing_segment_list() { return _routing_segment_list; }
  void set_curr_la_topo(LATopo* curr_la_topo) { _curr_la_topo = curr_la_topo; }
  void set_start_node_list_list(const std::vector<std::vector<LANode*>>& start_node_list_list) { _start_node_list_list = start_node_list_list; }
  void set_end_node_list_list(const std::vector<std::vector<LANode*>>& end_node_list_list) { _end_node_list_list = end_node_list_list; }
  void set_path_node_list(const std::vector<LANode*>& path_node_list) { _path_node_list = path_node_list; }
  void set_single_topo_visited_node_list(const std::vector<LANode*>& single_topo_visited_node_list)
  {
    _single_topo_visited_node_list = single_topo_visited_node_list;
  }
  void set_routing_segment_list(const std::vector<Segment<LayerCoord>>& routing_segment_list) { _routing_segment_list = routing_segment_list; }
  // single path
  PriorityQueue<LANode*, std::vector<LANode*>, CmpLANodeCost>& get_open_queue() { return _open_queue; }
  std::vector<LANode*>& get_single_path_visited_node_list() { return _single_path_visited_node_list; }
  LANode* get_path_head_node() { return _path_head_node; }
  int32_t get_end_node_list_idx() const { return _end_node_list_idx; }
  void set_open_queue(const PriorityQueue<LANode*, std::vector<LANode*>, CmpLANodeCost>& open_queue) { _open_queue = open_queue; }
  void set_single_path_visited_node_list(const std::vector<LANode*>& single_path_visited_node_list)
  {
    _single_path_visited_node_list = single_path_visited_node_list;
  }
  void set_path_head_node(LANode* path_head_node) { _path_head_node = path_head_node; }
  void set_end_node_list_idx(const int32_t end_node_list_idx) { _end_node_list_idx = end_node_list_idx; }
#endif

 private:
  std::vector<LANet> _la_net_list;
  LAComParam _la_com_param;
  std::vector<LANet*> _la_task_list;
  std::vector<GridMap<LANode>> _layer_node_map;
#if 1  // astar
  // single topo
  LATopo* _curr_la_topo = nullptr;
  std::vector<std::vector<LANode*>> _start_node_list_list;
  std::vector<std::vector<LANode*>> _end_node_list_list;
  std::vector<LANode*> _path_node_list;
  std::vector<LANode*> _single_topo_visited_node_list;
  std::vector<Segment<LayerCoord>> _routing_segment_list;
  // single path
  PriorityQueue<LANode*, std::vector<LANode*>, CmpLANodeCost> _open_queue;
  std::vector<LANode*> _single_path_visited_node_list;
  LANode* _path_head_node = nullptr;
  int32_t _end_node_list_idx = -1;
#endif
};

}  // namespace irt
