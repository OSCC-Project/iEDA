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

namespace irt {

class ViaMasterIdx
{
 public:
  ViaMasterIdx() = default;
  ~ViaMasterIdx() = default;
  // getter
  int32_t get_below_layer_idx() const { return _below_layer_idx; }
  int32_t get_via_idx() const { return _via_idx; }
  // setter
  void set_below_layer_idx(const int32_t below_layer_idx) { _below_layer_idx = below_layer_idx; }
  void set_via_idx(const int32_t via_idx) { _via_idx = via_idx; }
  // function

 private:
  int32_t _below_layer_idx = -1;
  int32_t _via_idx = -1;
};

}  // namespace irt
