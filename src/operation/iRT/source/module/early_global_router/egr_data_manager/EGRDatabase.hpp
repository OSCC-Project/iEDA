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
  std::string& get_design_name() { return _design_name; }
  int32_t get_micron_dbu() { return _micron_dbu; }
  Die& get_die() { return _die; }
  std::vector<RoutingLayer>& get_routing_layer_list() { return _routing_layer_list; }
  std::vector<CutLayer>& get_cut_layer_list() { return _cut_layer_list; }
  std::vector<std::vector<ViaMaster>>& get_layer_via_master_list() { return _layer_via_master_list; }
  std::vector<Blockage>& get_routing_blockage_list() { return _routing_blockage_list; }
  std::vector<EGRNet>& get_egr_net_list() { return _egr_net_list; }
  std::vector<GridMap<EGRNode>>& get_layer_resource_map() { return _layer_resource_map; }
  std::vector<int32_t>& get_h_layer_idx_list() { return _h_layer_idx_list; }
  std::vector<int32_t>& get_v_layer_idx_list() { return _v_layer_idx_list; }

  // setter
  void set_design_name(const std::string& design_name) { _design_name = design_name; }
  void set_micron_dbu(int32_t micron_dbu) { _micron_dbu = micron_dbu; }
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
  void set_h_layer_idx_list(const std::vector<int32_t>& h_layer_idx_list) { _h_layer_idx_list = h_layer_idx_list; }
  void set_v_layer_idx_list(const std::vector<int32_t>& v_layer_idx_list) { _v_layer_idx_list = v_layer_idx_list; }

 private:
  std::string _design_name;
  int32_t _micron_dbu = -1;
  Die _die;
  std::vector<RoutingLayer> _routing_layer_list;
  std::vector<CutLayer> _cut_layer_list;
  std::vector<std::vector<ViaMaster>> _layer_via_master_list;
  std::vector<Blockage> _routing_blockage_list;
  std::vector<EGRNet> _egr_net_list;
  std::vector<GridMap<EGRNode>> _layer_resource_map;
  std::vector<int32_t> _h_layer_idx_list;
  std::vector<int32_t> _v_layer_idx_list;
};

}  // namespace irt
