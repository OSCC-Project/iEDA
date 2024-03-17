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

#include "PlanarRect.hpp"
#include "RTHeader.hpp"

namespace irt {

class CutLayer
{
 public:
  CutLayer() = default;
  ~CutLayer() = default;
  // getter
  int32_t get_layer_idx() const { return _layer_idx; }
  int32_t get_layer_order() const { return _layer_order; }
  std::string& get_layer_name() { return _layer_name; }
  int32_t get_spacing() const { return _spacing; }
  // setter
  void set_layer_idx(const int32_t layer_idx) { _layer_idx = layer_idx; }
  void set_layer_order(const int32_t layer_order) { _layer_order = layer_order; }
  void set_layer_name(const std::string& layer_name) { _layer_name = layer_name; }
  void set_spacing(const int32_t spacing) { _spacing = spacing; }
  // function
  int32_t getMinSpacing(const PlanarRect& rect) { return _spacing; }
  int32_t getMaxSpacing(const PlanarRect& rect) { return _spacing; }

 private:
  int32_t _layer_idx = -1;
  int32_t _layer_order = -1;
  std::string _layer_name;
  int32_t _spacing = -1;
};

}  // namespace irt
