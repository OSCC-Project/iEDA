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

#include "DRCHeader.hpp"

namespace idrc {

class PropertyDefinition
{
 public:
  PropertyDefinition() = default;
  ~PropertyDefinition() = default;
  // getter
  int32_t get_max_via_stack_num() const { return _max_via_stack_num; }
  int32_t get_bottom_routing_layer_idx() const { return _bottom_routing_layer_idx; }
  int32_t get_top_routing_layer_idx() const { return _top_routing_layer_idx; }
  // setter
  void set_max_via_stack_num(const int32_t max_via_stack_num) { _max_via_stack_num = max_via_stack_num; }
  void set_bottom_routing_layer_idx(const int32_t bottom_routing_layer_idx) { _bottom_routing_layer_idx = bottom_routing_layer_idx; }
  void set_top_routing_layer_idx(const int32_t top_routing_layer_idx) { _top_routing_layer_idx = top_routing_layer_idx; }
  // function
 private:
  //  max via stack
  int32_t _max_via_stack_num = -1;
  int32_t _bottom_routing_layer_idx = -1;
  int32_t _top_routing_layer_idx = -1;
};

}  // namespace idrc
