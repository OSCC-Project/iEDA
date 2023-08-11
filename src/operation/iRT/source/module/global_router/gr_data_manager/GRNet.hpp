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

#include "GRNetPriority.hpp"
#include "GRPin.hpp"
#include "LayerCoord.hpp"
#include "MTree.hpp"
#include "Net.hpp"
#include "RoutingState.hpp"

namespace irt {

class GRNet
{
 public:
  GRNet() = default;
  ~GRNet() = default;

  // getter
  Net* get_origin_net() { return _origin_net; }
  irt_int get_net_idx() const { return _net_idx; }
  ConnectType get_connect_type() const { return _connect_type; }
  std::vector<GRPin>& get_gr_pin_list() { return _gr_pin_list; }
  GRPin& get_gr_driving_pin() { return _gr_driving_pin; }
  BoundingBox& get_bounding_box() { return _bounding_box; }
  GridMap<double>& get_ra_cost_map() { return _ra_cost_map; }
  RoutingState get_routing_state() const { return _routing_state; }
  MTree<LayerCoord>& get_routing_tree() { return _routing_tree; }
  MTree<RTNode>& get_gr_result_tree() { return _gr_result_tree; }
  // setter
  void set_origin_net(Net* origin_net) { _origin_net = origin_net; }
  void set_net_idx(const irt_int net_idx) { _net_idx = net_idx; }
  void set_connect_type(const ConnectType& connect_type) { _connect_type = connect_type; }
  void set_gr_pin_list(std::vector<GRPin>& gr_pin_list) { _gr_pin_list = gr_pin_list; }
  void set_gr_driving_pin(const GRPin& gr_driving_pin) { _gr_driving_pin = gr_driving_pin; }
  void set_bounding_box(const BoundingBox& bounding_box) { _bounding_box = bounding_box; }
  void set_ra_cost_map(const GridMap<double>& ra_cost_map) { _ra_cost_map = ra_cost_map; }
  void set_routing_state(const RoutingState& routing_state) { _routing_state = routing_state; }
  void set_routing_tree(const MTree<LayerCoord>& routing_tree) { _routing_tree = routing_tree; }
  void set_gr_result_tree(const MTree<RTNode>& gr_result_tree) { _gr_result_tree = gr_result_tree; }

 private:
  Net* _origin_net = nullptr;
  irt_int _net_idx = -1;
  ConnectType _connect_type = ConnectType::kNone;
  std::vector<GRPin> _gr_pin_list;
  GRPin _gr_driving_pin;
  BoundingBox _bounding_box;
  GridMap<double> _ra_cost_map;
  RoutingState _routing_state = RoutingState::kNone;
  MTree<LayerCoord> _routing_tree;
  MTree<RTNode> _gr_result_tree;
};

}  // namespace irt
