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

#include "BaseInfo.hpp"
#include "Boost.hpp"

namespace irt {

class BaseShape
{
 public:
  BaseShape() = default;
  ~BaseShape() = default;
  // getter
  BaseInfo& get_base_info() { return _base_info; }
  BGRectInt& get_shape() { return _shape; }
  int32_t get_layer_idx() const { return _layer_idx; }
  bool get_is_routing() const { return _is_routing; }
  // setters
  void set_base_info(const BaseInfo base_info) { _base_info = base_info; }
  void set_shape(const BGRectInt& shape) { _shape = shape; }
  void set_layer_idx(const int32_t layer_idx) { _layer_idx = layer_idx; }
  void set_is_routing(const bool is_routing) { _is_routing = is_routing; }
  // function

 protected:
  BaseInfo _base_info;
  BGRectInt _shape;
  int32_t _layer_idx = -1;
  bool _is_routing = true;
};

}  // namespace irt
