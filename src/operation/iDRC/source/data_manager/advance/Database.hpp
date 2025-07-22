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

#include "CutLayer.hpp"
#include "Die.hpp"
#include "MaxViaStackRule.hpp"
#include "OffGridOrWrongWayRule.hpp"
#include "RoutingLayer.hpp"
#include "ViolationType.hpp"

namespace idrc {

class Database
{
 public:
  Database() = default;
  ~Database() = default;
  // getter
  int32_t get_micron_dbu() const { return _micron_dbu; }
  Die& get_die() { return _die; }
  MaxViaStackRule& get_max_via_stack_rule() { return _max_via_stack_rule; }
  OffGridOrWrongWayRule& get_off_grid_or_wrong_way_rule() { return _off_grid_or_wrong_way_rule; }
  std::vector<RoutingLayer>& get_routing_layer_list() { return _routing_layer_list; }
  std::vector<CutLayer>& get_cut_layer_list() { return _cut_layer_list; }
  std::set<ViolationType>& get_exist_rule_set() { return _exist_rule_set; }
  std::map<int32_t, int32_t>& get_routing_idb_layer_id_to_idx_map() { return _routing_idb_layer_id_to_idx_map; }
  std::map<int32_t, int32_t>& get_cut_idb_layer_id_to_idx_map() { return _cut_idb_layer_id_to_idx_map; }
  std::map<std::string, int32_t>& get_routing_layer_name_to_idx_map() { return _routing_layer_name_to_idx_map; }
  std::map<int32_t, std::vector<int32_t>>& get_routing_to_adjacent_cut_map() { return _routing_to_adjacent_cut_map; }
  std::map<std::string, int32_t>& get_cut_layer_name_to_idx_map() { return _cut_layer_name_to_idx_map; }
  std::map<int32_t, std::vector<int32_t>>& get_cut_to_adjacent_routing_map() { return _cut_to_adjacent_routing_map; }
  // setter
  void set_micron_dbu(const int32_t micron_dbu) { _micron_dbu = micron_dbu; }
  // function

 private:
  int32_t _micron_dbu = -1;
  Die _die;
  MaxViaStackRule _max_via_stack_rule;
  OffGridOrWrongWayRule _off_grid_or_wrong_way_rule;
  std::vector<RoutingLayer> _routing_layer_list;
  std::vector<CutLayer> _cut_layer_list;
  std::set<ViolationType> _exist_rule_set;
  std::map<int32_t, int32_t> _routing_idb_layer_id_to_idx_map;
  std::map<int32_t, int32_t> _cut_idb_layer_id_to_idx_map;
  std::map<std::string, int32_t> _routing_layer_name_to_idx_map;
  std::map<int32_t, std::vector<int32_t>> _routing_to_adjacent_cut_map;
  std::map<std::string, int32_t> _cut_layer_name_to_idx_map;
  std::map<int32_t, std::vector<int32_t>> _cut_to_adjacent_routing_map;
};

}  // namespace idrc
