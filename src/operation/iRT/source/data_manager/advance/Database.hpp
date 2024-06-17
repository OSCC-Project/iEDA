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
#include "GCell.hpp"
#include "Logger.hpp"
#include "Net.hpp"
#include "Obstacle.hpp"
#include "RTHeader.hpp"
#include "RoutingLayer.hpp"
#include "Row.hpp"
#include "Utility.hpp"
#include "ViaMaster.hpp"
#include "builder.h"
#include "def_service.h"
#include "lef_service.h"

namespace irt {

class Database
{
 public:
  Database() = default;
  ~Database() = default;
  // getter
  idb::IdbBuilder* get_idb_builder() { return _idb_builder; }
  std::string& get_design_name() { return _design_name; }
  std::vector<std::string>& get_lef_file_path_list() { return _lef_file_path_list; }
  std::string& get_def_file_path() { return _def_file_path; }
  int32_t get_micron_dbu() const { return _micron_dbu; }
  ScaleAxis& get_gcell_axis() { return _gcell_axis; }
  Die& get_die() { return _die; }
  Row& get_row() { return _row; }
  std::vector<RoutingLayer>& get_routing_layer_list() { return _routing_layer_list; }
  std::vector<CutLayer>& get_cut_layer_list() { return _cut_layer_list; }
  std::map<int32_t, int32_t>& get_routing_idb_layer_id_to_idx_map() { return _routing_idb_layer_id_to_idx_map; }
  std::map<int32_t, int32_t>& get_cut_idb_layer_id_to_idx_map() { return _cut_idb_layer_id_to_idx_map; }
  std::map<std::string, int32_t>& get_routing_layer_name_to_idx_map() { return _routing_layer_name_to_idx_map; }
  std::map<std::string, int32_t>& get_cut_layer_name_to_idx_map() { return _cut_layer_name_to_idx_map; }
  std::map<int32_t, std::vector<int32_t>>& get_cut_to_adjacent_routing_map() { return _cut_to_adjacent_routing_map; }
  std::vector<std::vector<ViaMaster>>& get_layer_via_master_list() { return _layer_via_master_list; }
  std::map<int32_t, PlanarRect>& get_layer_enclosure_map() { return _layer_enclosure_map; }
  std::vector<Obstacle>& get_routing_obstacle_list() { return _routing_obstacle_list; }
  std::vector<Obstacle>& get_cut_obstacle_list() { return _cut_obstacle_list; }
  std::vector<Net>& get_net_list() { return _net_list; }
  GridMap<GCell>& get_gcell_map() { return _gcell_map; }
  // setter
  void set_idb_builder(idb::IdbBuilder* idb_builder) { _idb_builder = idb_builder; }
  void set_design_name(const std::string& design_name) { _design_name = design_name; }
  void set_lef_file_path_list(const std::vector<std::string>& lef_file_path_list) { _lef_file_path_list = lef_file_path_list; }
  void set_def_file_path(const std::string& def_file_path) { _def_file_path = def_file_path; }
  void set_micron_dbu(const int32_t micron_dbu) { _micron_dbu = micron_dbu; }
  // function

 private:
  idb::IdbBuilder* _idb_builder;
  std::string _design_name;
  std::vector<std::string> _lef_file_path_list;
  std::string _def_file_path;
  int32_t _micron_dbu = -1;
  ScaleAxis _gcell_axis;
  Die _die;
  Row _row;
  std::vector<RoutingLayer> _routing_layer_list;
  std::vector<CutLayer> _cut_layer_list;
  std::map<int32_t, int32_t> _routing_idb_layer_id_to_idx_map;
  std::map<int32_t, int32_t> _cut_idb_layer_id_to_idx_map;
  std::map<std::string, int32_t> _routing_layer_name_to_idx_map;
  std::map<std::string, int32_t> _cut_layer_name_to_idx_map;
  std::map<int32_t, std::vector<int32_t>> _cut_to_adjacent_routing_map;
  std::vector<std::vector<ViaMaster>> _layer_via_master_list;
  std::map<int32_t, PlanarRect> _layer_enclosure_map;
  std::vector<Obstacle> _routing_obstacle_list;
  std::vector<Obstacle> _cut_obstacle_list;
  std::vector<Net> _net_list;
  GridMap<GCell> _gcell_map;
};

}  // namespace irt
