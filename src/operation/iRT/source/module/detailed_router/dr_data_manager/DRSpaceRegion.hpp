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

#include "DRNode.hpp"
#include "DRNodeGraph.hpp"
#include "DRTask.hpp"
#include "LayerCoord.hpp"
#include "LayerRect.hpp"

namespace irt {

class DRSpaceRegion
{
 public:
  DRSpaceRegion() = default;
  ~DRSpaceRegion() = default;
  // getter
  PlanarRect& get_base_region() { return _base_region; }
  irt_int get_top_layer_idx() const { return _top_layer_idx; }
  irt_int get_bottom_layer_idx() const { return _bottom_layer_idx; }
  // setter
  void set_base_region(const PlanarRect& base_region) { _base_region = base_region; }
  void set_top_layer_idx(const irt_int top_layer_idx) { _top_layer_idx = top_layer_idx; }
  void set_bottom_layer_idx(const irt_int bottom_layer_idx) { _bottom_layer_idx = bottom_layer_idx; }
  // function

 private:
  PlanarRect _base_region;
  irt_int _top_layer_idx = -1;
  irt_int _bottom_layer_idx = -1;
};

}  // namespace irt
