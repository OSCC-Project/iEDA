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

namespace irt {

class RQShape
{
 public:
  RQShape() {}
  ~RQShape() {}
  // getter
  irt_int get_net_id() { return _net_id; }
  BoostBox& get_shape() { return _shape; }
  bool get_is_routing() const { return _is_routing; }
  irt_int get_routing_layer_idx() const { return _routing_layer_idx; }
  irt_int get_min_spacing() const { return _min_spacing; }
  BoostBox& get_enlarged_shape() { return _enlarged_shape; }
  // setters
  void set_net_id(const irt_int net_id) { _net_id = net_id; }
  void set_shape(const BoostBox& shape) { _shape = shape; }
  void set_is_routing(const bool is_routing) { _is_routing = is_routing; }
  void set_routing_layer_idx(const irt_int layer_idx) { _routing_layer_idx = layer_idx; }
  void set_min_spacing(const irt_int min_spacing) { _min_spacing = min_spacing; }
  void set_enlarged_shape(const BoostBox& enlarged_shape) { _enlarged_shape = enlarged_shape; }
  // function

 private:
  irt_int _net_id;
  BoostBox _shape;
  bool _is_routing = true;
  irt_int _routing_layer_idx = -1;
  irt_int _min_spacing = -1;
  BoostBox _enlarged_shape;
};

}  // namespace irt
