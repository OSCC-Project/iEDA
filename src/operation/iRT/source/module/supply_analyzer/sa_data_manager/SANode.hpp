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

#include "AccessPoint.hpp"
#include "EXTLayerRect.hpp"
#include "LayerRect.hpp"

namespace irt {

class SANode
{
 public:
  SANode() = default;
  ~SANode() = default;
  // getter
  PlanarRect& get_shape() { return _shape; }
  std::map<int32_t, std::set<EXTLayerRect*>>& get_net_fixed_rect_map() { return _net_fixed_rect_map; }
  std::map<Orientation, int32_t>& get_orien_supply_map() { return _orien_supply_map; }
  // setter
  void set_shape(const PlanarRect& shape) { _shape = shape; }
  void set_net_fixed_rect_map(const std::map<int32_t, std::set<EXTLayerRect*>>& net_fixed_rect_map)
  {
    _net_fixed_rect_map = net_fixed_rect_map;
  }
  void set_orien_supply_map(const std::map<Orientation, int32_t>& orien_supply_map) { _orien_supply_map = orien_supply_map; }
  // function

 private:
  PlanarRect _shape;
  std::map<int32_t, std::set<EXTLayerRect*>> _net_fixed_rect_map;
  std::map<Orientation, int32_t> _orien_supply_map;
};

}  // namespace irt
