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

#include "ERConflictGroup.hpp"
#include "ERComParam.hpp"
#include "ERNet.hpp"
#include "ERNode.hpp"
#include "RTHeader.hpp"

namespace irt {

class ERModel
{
 public:
  ERModel() = default;
  ~ERModel() = default;
  // getter
  std::vector<ERNet>& get_er_net_list() { return _er_net_list; }
  ERComParam& get_er_com_param() { return _er_com_param; }
  std::vector<ERConflictGroup>& get_er_conflict_group_list() { return _er_conflict_group_list; }
  std::vector<std::vector<std::pair<LayerCoord, LayerCoord>>>& get_grid_pair_list_list() { return _grid_pair_list_list; }
  std::vector<ERNet*>& get_er_task_list() { return _er_task_list; }
  GridMap<ERNode>& get_planar_node_map() { return _planar_node_map; }
  std::vector<GridMap<ERNode>>& get_layer_node_map() { return _layer_node_map; }
  // setter
  void set_er_net_list(const std::vector<ERNet>& er_net_list) { _er_net_list = er_net_list; }
  void set_er_com_param(const ERComParam& er_com_param) { _er_com_param = er_com_param; }
  void set_er_conflict_group_list(const std::vector<ERConflictGroup>& er_conflict_group_list) { _er_conflict_group_list = er_conflict_group_list; }
  void set_grid_pair_list_list(const std::vector<std::vector<std::pair<LayerCoord, LayerCoord>>>& grid_pair_list_list)
  {
    _grid_pair_list_list = grid_pair_list_list;
  }
  void set_er_task_list(const std::vector<ERNet*>& er_task_list) { _er_task_list = er_task_list; }
  void set_planar_node_map(const GridMap<ERNode>& planar_node_map) { _planar_node_map = planar_node_map; }
  void set_layer_node_map(const std::vector<GridMap<ERNode>>& layer_node_map) { _layer_node_map = layer_node_map; }

#if 1
  // single task
  ERNet* get_curr_er_task() { return _curr_er_task; }
  void set_curr_er_task(ERNet* curr_er_task) { _curr_er_task = curr_er_task; }
#endif

 private:
  std::vector<ERNet> _er_net_list;
  ERComParam _er_com_param;
  std::vector<ERConflictGroup> _er_conflict_group_list;
  std::vector<std::vector<std::pair<LayerCoord, LayerCoord>>> _grid_pair_list_list;
  std::vector<ERNet*> _er_task_list;
  GridMap<ERNode> _planar_node_map;
  std::vector<GridMap<ERNode>> _layer_node_map;
#if 1
  // single task
  ERNet* _curr_er_task = nullptr;
#endif
};

}  // namespace irt
