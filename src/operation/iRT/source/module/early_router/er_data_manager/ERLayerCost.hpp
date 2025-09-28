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

#include "RTHeader.hpp"

namespace irt {

class ERLayerCost
{
 public:
  ERLayerCost() = default;
  ~ERLayerCost() = default;
  // getter
  int32_t get_parent_layer_idx() const { return _parent_layer_idx; }
  int32_t get_layer_idx() const { return _layer_idx; }
  double get_history_cost() const { return _history_cost; }
  // setter
  void set_parent_layer_idx(const int32_t parent_layer_idx) { _parent_layer_idx = parent_layer_idx; }
  void set_layer_idx(const int32_t layer_idx) { _layer_idx = layer_idx; }
  void set_history_cost(const double history_cost) { _history_cost = history_cost; }
  // function

 private:
  int32_t _parent_layer_idx = -1;
  int32_t _layer_idx = -1;
  double _history_cost = 0;
};

}  // namespace irt
