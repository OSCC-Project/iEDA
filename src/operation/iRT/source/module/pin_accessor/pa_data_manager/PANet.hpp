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

#include "Net.hpp"
#include "PAPin.hpp"

namespace irt {

class PANet
{
 public:
  PANet() = default;
  ~PANet() = default;
  // getter
  Net* get_origin_net() { return _origin_net; }
  int32_t get_net_idx() const { return _net_idx; }
  ConnectType get_connect_type() const { return _connect_type; }
  std::vector<PAPin>& get_pa_pin_list() { return _pa_pin_list; }
  BoundingBox& get_bounding_box() { return _bounding_box; }
  // setter
  void set_origin_net(Net* origin_net) { _origin_net = origin_net; }
  void set_net_idx(const int32_t net_idx) { _net_idx = net_idx; }
  void set_connect_type(const ConnectType& connect_type) { _connect_type = connect_type; }
  void set_pa_pin_list(const std::vector<PAPin>& pa_pin_list) { _pa_pin_list = pa_pin_list; }
  void set_bounding_box(const BoundingBox& bounding_box) { _bounding_box = bounding_box; }
  // function

 private:
  Net* _origin_net = nullptr;
  int32_t _net_idx = -1;
  ConnectType _connect_type = ConnectType::kNone;
  std::vector<PAPin> _pa_pin_list;
  BoundingBox _bounding_box;
};

}  // namespace irt
