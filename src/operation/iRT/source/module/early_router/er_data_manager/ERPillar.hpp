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

#include "LALayerCost.hpp"
#include "PlanarCoord.hpp"

namespace irt {

class ERPillar
{
 public:
  ERPillar() = default;
  ~ERPillar() = default;
  // getter
  PlanarCoord& get_planar_coord() { return _planar_coord; }
  std::set<int32_t>& get_pin_layer_idx_set() { return _pin_layer_idx_set; }
  std::vector<LALayerCost>& get_layer_cost_list() { return _layer_cost_list; }
  int32_t get_layer_idx() const { return _layer_idx; }
  // setter
  void set_planar_coord(const PlanarCoord& planar_coord) { _planar_coord = planar_coord; }
  void set_pin_layer_idx_set(const std::set<int32_t>& pin_layer_idx_set) { _pin_layer_idx_set = pin_layer_idx_set; }
  void set_layer_cost_list(const std::vector<LALayerCost>& layer_cost_list) { _layer_cost_list = layer_cost_list; }
  void set_layer_idx(const int32_t layer_idx) { _layer_idx = layer_idx; }
  // function

 private:
  PlanarCoord _planar_coord;
  std::set<int32_t> _pin_layer_idx_set;
  std::vector<LALayerCost> _layer_cost_list;
  int32_t _layer_idx = -1;
};
}  // namespace irt
