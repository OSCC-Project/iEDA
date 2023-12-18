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

class DRCShape
{
 public:
  DRCShape() = default;
  DRCShape(irt_int net_idx, const LayerRect& layer_rect, bool is_routing)
  {
    _base_info.set_net_idx(net_idx);
    _layer_rect = layer_rect;
    _is_routing = is_routing;
  }
  DRCShape(const BaseInfo& base_info, const LayerRect& layer_rect, bool is_routing)
  {
    _base_info = base_info;
    _layer_rect = layer_rect;
    _is_routing = is_routing;
  }
  ~DRCShape() = default;
  // getter
  BaseInfo& get_base_info() { return _base_info; }
  LayerRect& get_layer_rect() { return _layer_rect; }
  bool get_is_routing() const { return _is_routing; }
  // const getter
  const BaseInfo& get_base_info() const { return _base_info; }
  const LayerRect& get_layer_rect() const { return _layer_rect; }
  // setter
  void set_base_info(const BaseInfo base_info) { _base_info = base_info; }
  void set_layer_rect(const LayerRect& layer_rect) { _layer_rect = layer_rect; }
  void set_is_routing(const bool is_routing) { _is_routing = is_routing; }
  // function

 private:
  BaseInfo _base_info;
  LayerRect _layer_rect;
  bool _is_routing = true;
};

}  // namespace irt
