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

#include "Die.hpp"
#include "EGRNet.hpp"
#include "EGRNode.hpp"

namespace irt {

class EGRDatabase
{
 public:
  EGRDatabase() = default;
  ~EGRDatabase() = default;
  // getter
  Die& get_die() { return _die; }
  std::vector<RoutingLayer>& get_routing_layer_list() { return _routing_layer_list; }
  std::vector<Blockage>& get_routing_blockage_list() { return _routing_blockage_list; }
  std::vector<EGRNet>& get_egr_net_list() { return _egr_net_list; }
  std::vector<GridMap<EGRNode>>& get_layer_resource_map() { return _layer_resource_map; }
  std::vector<irt_int>& get_h_layer_idx_list() { return _h_layer_idx_list; }
  std::vector<irt_int>& get_v_layer_idx_list() { return _v_layer_idx_list; }
  double get_total_wire_length() { return _total_wire_length; }
  double get_total_via_num() { return _total_via_num; }

  // setter
  void set_die(Die& die) { _die = die; }
  void set_routing_layer_list(const std::vector<RoutingLayer>& routing_layer_list) { _routing_layer_list = routing_layer_list; }
  void set_routing_blockage_list(const std::vector<Blockage>& routing_blockage_list) { _routing_blockage_list = routing_blockage_list; }
  void set_egr_net_list(const std::vector<EGRNet>& egr_net_list) { _egr_net_list = egr_net_list; }
  void set_layer_resource_map(const std::vector<GridMap<EGRNode>>& layer_resource_map) { _layer_resource_map = layer_resource_map; }
  void set_h_layer_idx_list(const std::vector<irt_int>& h_layer_idx_list) { _h_layer_idx_list = h_layer_idx_list; }
  void set_v_layer_idx_list(const std::vector<irt_int>& v_layer_idx_list) { _v_layer_idx_list = v_layer_idx_list; }
  void set_total_wire_length(const double total_wire_length) { _total_wire_length = total_wire_length; }
  void set_total_via_num(const double total_via_num) { _total_via_num = total_via_num; }

  // function
  void addWireLength(double wire_length) { _total_wire_length += wire_length; }
  void addViaNum(double via_num) { _total_via_num += via_num; }

 private:
  Die _die;
  std::vector<RoutingLayer> _routing_layer_list;
  std::vector<Blockage> _routing_blockage_list;
  std::vector<EGRNet> _egr_net_list;
  std::vector<GridMap<EGRNode>> _layer_resource_map;
  std::vector<irt_int> _h_layer_idx_list;
  std::vector<irt_int> _v_layer_idx_list;
  double _total_wire_length = 0;
  double _total_via_num = 0;
};

}  // namespace irt
