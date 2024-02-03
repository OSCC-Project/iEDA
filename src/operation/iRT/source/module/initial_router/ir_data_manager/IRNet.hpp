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

#include "IRPin.hpp"
#include "Net.hpp"

namespace irt {

class IRNet
{
 public:
  IRNet() = default;
  ~IRNet() = default;
  // getter
  Net* get_origin_net() { return _origin_net; }
  irt_int get_net_idx() const { return _net_idx; }
  ConnectType get_connect_type() const { return _connect_type; }
  std::vector<IRPin>& get_ir_pin_list() { return _ir_pin_list; }
  BoundingBox& get_bounding_box() { return _bounding_box; }
  LayerCoord& get_driving_grid_coord() { return _driving_grid_coord; }
  std::vector<LayerCoord>& get_grid_coord_list() { return _grid_coord_list; }
  MTree<Guide>& get_ir_result_tree() { return _ir_result_tree; }
  // setter
  void set_origin_net(Net* origin_net) { _origin_net = origin_net; }
  void set_net_idx(const irt_int net_idx) { _net_idx = net_idx; }
  void set_connect_type(const ConnectType& connect_type) { _connect_type = connect_type; }
  void set_ir_pin_list(const std::vector<IRPin>& ir_pin_list) { _ir_pin_list = ir_pin_list; }
  void set_bounding_box(const BoundingBox& bounding_box) { _bounding_box = bounding_box; }
  void set_driving_grid_coord(const LayerCoord& driving_grid_coord) { _driving_grid_coord = driving_grid_coord; }
  void set_grid_coord_list(const std::vector<LayerCoord>& grid_coord_list) { _grid_coord_list = grid_coord_list; }
  void set_ir_result_tree(const MTree<Guide>& ir_result_tree) { _ir_result_tree = ir_result_tree; }
  // function

 private:
  Net* _origin_net = nullptr;
  irt_int _net_idx = -1;
  ConnectType _connect_type = ConnectType::kNone;
  std::vector<IRPin> _ir_pin_list;
  BoundingBox _bounding_box;
  LayerCoord _driving_grid_coord;
  std::vector<LayerCoord> _grid_coord_list;
  MTree<Guide> _ir_result_tree;
};

}  // namespace irt
