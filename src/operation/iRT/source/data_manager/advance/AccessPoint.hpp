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

#include "AccessPointType.hpp"
#include "EXTLayerCoord.hpp"
#include "LayerCoord.hpp"
#include "Orientation.hpp"
#include "RTHeader.hpp"

namespace irt {

class AccessPoint : public EXTLayerCoord
{
 public:
  AccessPoint() = default;
  AccessPoint(int32_t pin_idx, const LayerCoord& coord, const AccessPointType& type)
  {
    _pin_idx = pin_idx;
    set_real_coord(coord);
    set_layer_idx(coord.get_layer_idx());
    _type = type;
  }
  ~AccessPoint() = default;
  // getter
  int32_t get_pin_idx() const { return _pin_idx; }
  AccessPointType get_type() const { return _type; }
  // setter
  void set_pin_idx(const int32_t pin_idx) { _pin_idx = pin_idx; }
  void set_type(const AccessPointType& type) { _type = type; }
  // function
  LayerCoord getGridLayerCoord() { return LayerCoord(get_grid_coord(), get_layer_idx()); }
  LayerCoord getRealLayerCoord() { return LayerCoord(get_real_coord(), get_layer_idx()); }

 private:
  int32_t _pin_idx = -1;
  AccessPointType _type = AccessPointType::kNone;
};

}  // namespace irt
