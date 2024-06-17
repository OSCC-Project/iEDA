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

#include "BoundingBox.hpp"
#include "ConnectType.hpp"
#include "GridMap.hpp"
#include "Guide.hpp"
#include "MTree.hpp"
#include "Pin.hpp"

namespace irt {

class Net
{
 public:
  Net() = default;
  ~Net() = default;
  // getter
  int32_t get_net_idx() const { return _net_idx; }
  std::string& get_net_name() { return _net_name; }
  ConnectType get_connect_type() const { return _connect_type; }
  std::vector<Pin>& get_pin_list() { return _pin_list; }
  BoundingBox& get_bounding_box() { return _bounding_box; }

  // setter
  void set_net_idx(const int32_t net_idx) { _net_idx = net_idx; }
  void set_net_name(const std::string& net_name) { _net_name = net_name; }
  void set_connect_type(const ConnectType& connect_type) { _connect_type = connect_type; }
  void set_pin_list(const std::vector<Pin>& pin_list) { _pin_list = pin_list; }
  void set_bounding_box(const BoundingBox& bounding_box) { _bounding_box = bounding_box; }

 private:
  int32_t _net_idx = -1;
  std::string _net_name;
  ConnectType _connect_type = ConnectType::kNone;
  std::vector<Pin> _pin_list;
  BoundingBox _bounding_box;
};

}  // namespace irt
