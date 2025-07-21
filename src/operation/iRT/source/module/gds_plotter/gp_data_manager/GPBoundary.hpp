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

#include "PlanarCoord.hpp"
#include "PlanarRect.hpp"
#include "RTHeader.hpp"

namespace irt {

class GPBoundary : public LayerRect
{
 public:
  GPBoundary() = default;
  GPBoundary(const LayerRect& rect, const int32_t data_type) : LayerRect(rect) { _data_type = data_type; }
  GPBoundary(const PlanarRect& rect, const int32_t layer_idx, const int32_t data_type) : LayerRect(rect, layer_idx) { _data_type = data_type; }
  ~GPBoundary() = default;
  // getter
  int32_t get_data_type() const { return _data_type; }
  // setter
  void set_data_type(const int32_t data_type) { _data_type = data_type; }

  // function

 private:
  int32_t _data_type = 0;
};

}  // namespace irt
