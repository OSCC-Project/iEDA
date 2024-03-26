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

#include "PRPin.hpp"
#include "Net.hpp"

namespace irt {

class PRNet
{
 public:
  PRNet() = default;
  ~PRNet() = default;
  // getter
  Net* get_origin_net() { return _origin_net; }
  int32_t get_net_idx() const { return _net_idx; }
  ConnectType get_connect_type() const { return _connect_type; }
  std::vector<PRPin>& get_pr_pin_list() { return _pr_pin_list; }
  BoundingBox& get_bounding_box() { return _bounding_box; }
  MTree<Guide>& get_pr_result_tree() { return _pr_result_tree; }
  // setter
  void set_origin_net(Net* origin_net) { _origin_net = origin_net; }
  void set_net_idx(const int32_t net_idx) { _net_idx = net_idx; }
  void set_connect_type(const ConnectType& connect_type) { _connect_type = connect_type; }
  void set_pr_pin_list(const std::vector<PRPin>& pr_pin_list) { _pr_pin_list = pr_pin_list; }
  void set_bounding_box(const BoundingBox& bounding_box) { _bounding_box = bounding_box; }
  void set_pr_result_tree(const MTree<Guide>& pr_result_tree) { _pr_result_tree = pr_result_tree; }
  // function

 private:
  Net* _origin_net = nullptr;
  int32_t _net_idx = -1;
  ConnectType _connect_type = ConnectType::kNone;
  std::vector<PRPin> _pr_pin_list;
  BoundingBox _bounding_box;
  MTree<Guide> _pr_result_tree;
};

}  // namespace irt
