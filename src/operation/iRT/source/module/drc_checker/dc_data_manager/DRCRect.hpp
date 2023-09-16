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

#include "LayerRect.hpp"
#include "RTU.hpp"
#include "RTUtil.hpp"

namespace irt {

class DRCRect
{
 public:
  DRCRect() = default;
  DRCRect(irt_int net_idx, const LayerRect& layer_rect, bool is_routing)
  {
    _net_idx = net_idx;
    _layer_rect = layer_rect;
    _is_routing = is_routing;
  }
  ~DRCRect() = default;
  // getter
  irt_int get_net_idx() const { return _net_idx; }
  LayerRect& get_layer_rect() { return _layer_rect; }
  const LayerRect& get_layer_rect() const { return _layer_rect; }
  irt_int get_layer_idx() const { return _layer_rect.get_layer_idx(); }
  bool get_is_routing() const { return _is_routing; }
  // setter
  void set_net_idx(const irt_int net_idx) { _net_idx = net_idx; }
  void set_layer_rect(const LayerRect& layer_rect) { _layer_rect = layer_rect; }
  void set_is_routing(const bool is_routing) { _is_routing = is_routing; }
  // function

 private:
  irt_int _net_idx = -1;
  LayerRect _layer_rect;
  bool _is_routing = true;
};

}  // namespace irt
