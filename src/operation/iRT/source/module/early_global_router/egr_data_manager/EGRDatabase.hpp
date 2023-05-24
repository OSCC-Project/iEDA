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
  irt_int get_micron_dbu() { return _micron_dbu; }
  Die& get_die() { return _die; }
  std::vector<RoutingLayer>& get_routing_layer_list() { return _routing_layer_list; }
  std::vector<CutLayer>& get_cut_layer_list() { return _cut_layer_list; }
  std::vector<std::vector<ViaMaster>>& get_layer_via_master_list() { return _layer_via_master_list; }
  std::vector<Blockage>& get_routing_blockage_list() { return _routing_blockage_list; }
  std::vector<EGRNet>& get_egr_net_list() { return _egr_net_list; }
  std::vector<GridMap<EGRNode>>& get_layer_resource_map() { return _layer_resource_map; }
  std::vector<irt_int>& get_h_layer_idx_list() { return _h_layer_idx_list; }
  std::vector<irt_int>& get_v_layer_idx_list() { return _v_layer_idx_list; }
  // getter congetion
  std::vector<std::map<irt_int, irt_int, std::greater<int>>>& get_overflow_map_list() { return _overflow_map_list; }
  std::map<irt_int, irt_int, std::greater<int>>& get_total_overflow_map() { return _total_overflow_map; }
  irt_int& get_total_track_overflow() { return _total_track_overflow; }
  std::vector<double>& get_wire_length_list() { return _wire_length_list; }
  std::vector<irt_int>& get_via_num_list() { return _via_num_list; }
  double get_total_wire_length() { return _total_wire_length; }
  double get_total_via_num() { return _total_via_num; }

  // setter
  void set_micron_dbu(irt_int micron_dbu) { _micron_dbu = micron_dbu; }
  void set_die(Die& die) { _die = die; }
  void set_routing_layer_list(const std::vector<RoutingLayer>& routing_layer_list) { _routing_layer_list = routing_layer_list; }
  void set_cut_layer_list(const std::vector<CutLayer>& cut_layer_list) { _cut_layer_list = cut_layer_list; }
  void set_layer_via_master_list(const std::vector<std::vector<ViaMaster>>& layer_via_master_list)
  {
    _layer_via_master_list = layer_via_master_list;
  }
  void set_routing_blockage_list(const std::vector<Blockage>& routing_blockage_list) { _routing_blockage_list = routing_blockage_list; }
  void set_egr_net_list(const std::vector<EGRNet>& egr_net_list) { _egr_net_list = egr_net_list; }
  void set_layer_resource_map(const std::vector<GridMap<EGRNode>>& layer_resource_map) { _layer_resource_map = layer_resource_map; }
  void set_h_layer_idx_list(const std::vector<irt_int>& h_layer_idx_list) { _h_layer_idx_list = h_layer_idx_list; }
  void set_v_layer_idx_list(const std::vector<irt_int>& v_layer_idx_list) { _v_layer_idx_list = v_layer_idx_list; }
  void set_overflow_map_list(std::vector<std::map<irt_int, irt_int, std::greater<int>>>& overflow_map_list)
  {
    _overflow_map_list = overflow_map_list;
  }
  void set_overflow_map_total(std::map<irt_int, irt_int, std::greater<int>>& total_overflow_map)
  {
    _total_overflow_map = total_overflow_map;
  }
  void set_total_track_overflow(irt_int& total_track_overflow) { _total_track_overflow = total_track_overflow; }
  void set_wire_length_list(std::vector<double>& wire_length_list) { _wire_length_list = wire_length_list; }
  void set_via_num_list(std::vector<irt_int>& via_num_list) { _via_num_list = via_num_list; }
  void set_total_wire_length(const double total_wire_length) { _total_wire_length = total_wire_length; }
  void set_total_via_num(const double total_via_num) { _total_via_num = total_via_num; }

  // function
  void addWireLength(double wire_length) { _total_wire_length += wire_length; }
  void addViaNum(double via_num) { _total_via_num += via_num; }

 private:
  irt_int _micron_dbu = -1;
  Die _die;
  std::vector<RoutingLayer> _routing_layer_list;
  std::vector<CutLayer> _cut_layer_list;
  std::vector<std::vector<ViaMaster>> _layer_via_master_list;
  std::vector<Blockage> _routing_blockage_list;
  std::vector<EGRNet> _egr_net_list;
  std::vector<GridMap<EGRNode>> _layer_resource_map;
  std::vector<irt_int> _h_layer_idx_list;
  std::vector<irt_int> _v_layer_idx_list;

  // statistics result
  std::vector<std::map<irt_int, irt_int, std::greater<int>>> _overflow_map_list;
  std::map<irt_int, irt_int, std::greater<int>> _total_overflow_map;
  irt_int _total_track_overflow = 0;
  std::vector<double> _wire_length_list;
  std::vector<irt_int> _via_num_list;
  double _total_wire_length = 0;
  double _total_via_num = 0;
};

}  // namespace irt
