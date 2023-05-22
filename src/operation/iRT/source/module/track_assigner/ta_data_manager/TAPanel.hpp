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

#include "EXTLayerRect.hpp"
#include "ScaleAxis.hpp"
#include "TANode.hpp"
#include "TAPanelStat.hpp"
#include "TATask.hpp"

namespace irt {

class TAPanel : public EXTLayerRect
{
 public:
  TAPanel() {}
  ~TAPanel() {}
  // getter
  irt_int get_panel_idx() const { return _panel_idx; }
  std::map<irt_int, std::vector<PlanarRect>>& get_net_blockage_map() { return _net_blockage_map; }
  std::map<irt_int, std::vector<PlanarRect>>& get_net_region_map() { return _net_region_map; }
  std::vector<TATask>& get_ta_task_list() { return _ta_task_list; }
  GridMap<TANode>& get_ta_node_map() { return _ta_node_map; }
  TAPanelStat& get_ta_panel_stat() { return _ta_panel_stat; }
  // setter
  void set_panel_idx(const irt_int panel_idx) { _panel_idx = panel_idx; }
  void set_net_blockage_map(const std::map<irt_int, std::vector<PlanarRect>>& net_blockage_map) { _net_blockage_map = net_blockage_map; }
  void set_net_region_map(const std::map<irt_int, std::vector<PlanarRect>>& net_region_map) { _net_region_map = net_region_map; }
  void set_ta_task_list(const std::vector<TATask>& ta_task_list) { _ta_task_list = ta_task_list; }
  void set_ta_node_map(const GridMap<TANode>& ta_node_map) { _ta_node_map = ta_node_map; }
  // function
  bool skipAssigning() { return _ta_task_list.empty(); }
  void freeNodeMap() { _ta_node_map.free(); }

#if 1  // astar
  double get_wire_unit() const { return _wire_unit; }
  double get_corner_unit() const { return _corner_unit; }
  double get_via_unit() const { return _via_unit; }
  const irt_int get_curr_task_idx() const { return _ta_task_ref->get_task_idx(); }
  const PlanarRect& get_curr_bounding_box() const { return _ta_task_ref->get_bounding_box(); }
  const std::map<LayerCoord, double, CmpLayerCoordByXASC>& get_curr_coord_cost_map() const { return _ta_task_ref->get_coord_cost_map(); }
  PlanarRect& get_routing_region() { return _routing_region; }
  std::vector<std::vector<TANode*>>& get_start_node_comb_list() { return _start_node_comb_list; }
  std::vector<std::vector<TANode*>>& get_end_node_comb_list() { return _end_node_comb_list; }
  std::vector<TANode*>& get_path_node_comb() { return _path_node_comb; }
  std::vector<Segment<TANode*>>& get_node_segment_list() { return _node_segment_list; }
  std::priority_queue<TANode*, std::vector<TANode*>, CmpTANodeCost>& get_open_queue() { return _open_queue; }
  std::vector<TANode*>& get_visited_node_list() { return _visited_node_list; }
  TANode* get_path_head_node() { return _path_head_node; }
  irt_int get_end_node_comb_idx() const { return _end_node_comb_idx; }
  void set_wire_unit(const double wire_unit) { _wire_unit = wire_unit; }
  void set_corner_unit(const double corner_unit) { _corner_unit = corner_unit; }
  void set_via_unit(const double via_unit) { _via_unit = via_unit; }
  void set_ta_task_ref(TATask* ta_task_ref) { _ta_task_ref = ta_task_ref; }
  void set_routing_region(const PlanarRect& routing_region) { _routing_region = routing_region; }
  void set_start_node_comb_list(const std::vector<std::vector<TANode*>>& start_node_comb_list)
  {
    _start_node_comb_list = start_node_comb_list;
  }
  void set_end_node_comb_list(const std::vector<std::vector<TANode*>>& end_node_comb_list) { _end_node_comb_list = end_node_comb_list; }
  void set_path_node_comb(const std::vector<TANode*>& path_node_comb) { _path_node_comb = path_node_comb; }
  void set_node_segment_list(const std::vector<Segment<TANode*>>& node_segment_list) { _node_segment_list = node_segment_list; }
  void set_forced_routing(const bool forced_routing) { _forced_routing = forced_routing; }
  void set_open_queue(const std::priority_queue<TANode*, std::vector<TANode*>, CmpTANodeCost>& open_queue) { _open_queue = open_queue; }
  void set_visited_node_list(const std::vector<TANode*>& visited_node_list) { _visited_node_list = visited_node_list; }
  void set_path_head_node(TANode* path_head_node) { _path_head_node = path_head_node; }
  void set_end_node_comb_idx(const irt_int end_node_comb_idx) { _end_node_comb_idx = end_node_comb_idx; }
  bool isForcedRouting() { return _forced_routing; }
#endif

 private:
  irt_int _panel_idx = -1;
  std::map<irt_int, std::vector<PlanarRect>> _net_blockage_map;
  std::map<irt_int, std::vector<PlanarRect>> _net_region_map;
  std::vector<TATask> _ta_task_list;
  GridMap<TANode> _ta_node_map;
  TAPanelStat _ta_panel_stat;
#if 1  // astar
  // config
  double _wire_unit = 1;
  double _corner_unit = 1;
  double _via_unit = 1;
  // single net
  TATask* _ta_task_ref = nullptr;
  PlanarRect _routing_region;
  std::vector<std::vector<TANode*>> _start_node_comb_list;
  std::vector<std::vector<TANode*>> _end_node_comb_list;
  std::vector<TANode*> _path_node_comb;
  std::vector<Segment<TANode*>> _node_segment_list;
  // single path
  bool _forced_routing = false;
  std::priority_queue<TANode*, std::vector<TANode*>, CmpTANodeCost> _open_queue;
  std::vector<TANode*> _visited_node_list;
  TANode* _path_head_node = nullptr;
  irt_int _end_node_comb_idx = -1;
#endif
};

}  // namespace irt