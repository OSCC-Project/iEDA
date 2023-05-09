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

#include <vector>

#include "DRScaleOrient.hpp"

namespace irt {

class DRNodeGraph
{
 public:
  DRNodeGraph() = default;
  ~DRNodeGraph() = default;
  // getter
  irt_int get_layer_idx() const { return _layer_idx; }
  std::vector<irt_int>& get_x_scale_list() { return _x_scale_list; }
  std::vector<irt_int>& get_y_scale_list() { return _y_scale_list; }
  std::vector<DRScaleOrient>& get_x_scale_orient_list() { return _x_scale_orient_list; }
  std::vector<DRScaleOrient>& get_y_scale_orient_list() { return _y_scale_orient_list; }
  std::map<LayerCoord, std::map<Orientation, LayerCoord>, CmpLayerCoordByXASC>& get_coord_neighbor_map() { return _coord_neighbor_map; }
  std::map<irt_int, std::set<irt_int>>& get_x_y_map() { return _x_y_map; }
  std::map<irt_int, std::set<irt_int>>& get_y_x_map() { return _y_x_map; }
  std::vector<DRNode>& get_dr_node_list() { return _dr_node_list; }
  std::unordered_map<irt_int, std::unordered_map<irt_int, irt_int>>& get_x_y_idx_map() { return _x_y_idx_map; }
  // setter
  void set_layer_idx(const irt_int layer_idx) { _layer_idx = layer_idx; }
  void set_x_scale_list(const std::vector<irt_int>& x_scale_list) { _x_scale_list = x_scale_list; }
  void set_y_scale_list(const std::vector<irt_int>& y_scale_list) { _y_scale_list = y_scale_list; }
  void set_x_scale_orient_list(const std::vector<DRScaleOrient>& x_scale_orient_list) { _x_scale_orient_list = x_scale_orient_list; }
  void set_y_scale_orient_list(const std::vector<DRScaleOrient>& y_scale_orient_list) { _y_scale_orient_list = y_scale_orient_list; }
  void set_coord_neighbor_map(const std::map<LayerCoord, std::map<Orientation, LayerCoord>, CmpLayerCoordByXASC>& coord_neighbor_map)
  {
    _coord_neighbor_map = coord_neighbor_map;
  }
  void set_x_y_map(const std::map<irt_int, std::set<irt_int>>& x_y_map) { _x_y_map = x_y_map; }
  void set_y_x_map(const std::map<irt_int, std::set<irt_int>>& y_x_map) { _y_x_map = y_x_map; }
  void set_dr_node_list(const std::vector<DRNode>& dr_node_list) { _dr_node_list = dr_node_list; }
  void set_x_y_idx_map(const std::unordered_map<irt_int, std::unordered_map<irt_int, irt_int>>& x_y_idx_map) { _x_y_idx_map = x_y_idx_map; }
  // function
  void free()
  {
    _layer_idx = -1;
    _x_scale_list.clear();
    _y_scale_list.clear();
    _x_scale_orient_list.clear();
    _y_scale_orient_list.clear();
    _coord_neighbor_map.clear();
    _x_y_map.clear();
    _y_x_map.clear();
    _dr_node_list.clear();
  }

 private:
  irt_int _layer_idx = -1;
  std::vector<irt_int> _x_scale_list;
  std::vector<irt_int> _y_scale_list;
  std::vector<DRScaleOrient> _x_scale_orient_list;
  std::vector<DRScaleOrient> _y_scale_orient_list;
  std::map<LayerCoord, std::map<Orientation, LayerCoord>, CmpLayerCoordByXASC> _coord_neighbor_map;
  std::map<irt_int, std::set<irt_int>> _x_y_map;
  std::map<irt_int, std::set<irt_int>> _y_x_map;
  std::vector<DRNode> _dr_node_list;
  std::unordered_map<irt_int, std::unordered_map<irt_int, irt_int>> _x_y_idx_map;
};

}  // namespace irt
