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

#include "RTU.hpp"
#include "RoutingState.hpp"

namespace irt {

class TASchedule
{
 public:
  TASchedule() = default;
  TASchedule(const irt_int layer_idx, const irt_int panel_idx, const RoutingState& routing_state)
  {
    _layer_idx = layer_idx;
    _panel_idx = panel_idx;
    _routing_state = routing_state;
  }
  ~TASchedule() = default;
  // getter
  irt_int get_layer_idx() const { return _layer_idx; }
  irt_int get_panel_idx() const { return _panel_idx; }
  RoutingState get_routing_state() const { return _routing_state; }
  // setter
  void set_layer_idx(const irt_int layer_idx) { _layer_idx = layer_idx; }
  void set_panel_idx(const irt_int panel_idx) { _panel_idx = panel_idx; }
  void set_routing_state(const RoutingState& routing_state) { _routing_state = routing_state; }

  // function

 private:
  irt_int _layer_idx = -1;
  irt_int _panel_idx = -1;
  RoutingState _routing_state = RoutingState::kNone;
};

}  // namespace irt
