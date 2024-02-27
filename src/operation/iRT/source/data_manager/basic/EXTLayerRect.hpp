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

#include "EXTPlanarRect.hpp"
#include "LayerRect.hpp"

namespace irt {

class EXTLayerRect : public EXTPlanarRect
{
 public:
  EXTLayerRect() = default;
  ~EXTLayerRect() = default;
  // getter
  int32_t get_layer_idx() const { return _layer_idx; }
  // setter
  void set_layer_idx(const int32_t layer_idx) { _layer_idx = layer_idx; }
  // function
  LayerRect getGridLayerRect() { return LayerRect(get_grid_rect(), get_layer_idx()); }
  LayerRect getRealLayerRect() { return LayerRect(get_real_rect(), get_layer_idx()); }

 private:
  int32_t _layer_idx = -1;
};

}  // namespace irt
