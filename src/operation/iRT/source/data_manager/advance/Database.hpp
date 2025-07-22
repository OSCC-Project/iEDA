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
#include "Summary.hpp"
#include "Utility.hpp"
#include "ViaMaster.hpp"

namespace irt {

class Database
{
 public:
  Database() = default;
  ~Database() = default;
  // getter
  std::string& get_design_name() { return _design_name; }
  std::vector<std::string>& get_lef_file_path_list() { return _lef_file_path_list; }
  std::string& get_def_file_path() { return _def_file_path; }
  int32_t get_micron_dbu() const { return _micron_dbu; }
  int32_t get_manufacture_grid() const { return _manufacture_grid; }
  ScaleAxis& get_gcell_axis() { return _gcell_axis; }
  Die& get_die() { return _die; }
  Row& get_row() { return _row; }
  std::vector<RoutingLayer>& get_routing_layer_list() { return _routing_layer_list; }
  std::vector<CutLayer>& get_cut_layer_list() { return _cut_layer_list; }
  std::map<int32_t, int32_t>& get_routing_idb_layer_id_to_idx_map() { return _routing_idb_layer_id_to_idx_map; }
  std::map<int32_t, int32_t>& get_cut_idb_layer_id_to_idx_map() { return _cut_idb_layer_id_to_idx_map; }
  std::map<std::string, int32_t>& get_routing_layer_name_to_idx_map() { return _routing_layer_name_to_idx_map; }
  std::map<int32_t, std::vector<int32_t>>& get_routing_to_adjacent_cut_map() { return _routing_to_adjacent_cut_map; }
  std::map<std::string, int32_t>& get_cut_layer_name_to_idx_map() { return _cut_layer_name_to_idx_map; }
  std::map<int32_t, std::vector<int32_t>>& get_cut_to_adjacent_routing_map() { return _cut_to_adjacent_routing_map; }
  std::vector<std::vector<ViaMaster>>& get_layer_via_master_list() { return _layer_via_master_list; }
  std::map<int32_t, PlanarRect>& get_layer_enclosure_map() { return _layer_enclosure_map; }
  std::map<int32_t, PlanarRect>& get_layer_cut_shape_map() { return _layer_cut_shape_map; }
  std::vector<Obstacle>& get_routing_obstacle_list() { return _routing_obstacle_list; }
  std::vector<Obstacle>& get_cut_obstacle_list() { return _cut_obstacle_list; }
  std::map<std::string, PlanarRect>& get_block_shape_map() { return _block_shape_map; }
  std::vector<Net>& get_net_list() { return _net_list; }
  GridMap<GCell>& get_gcell_map() { return _gcell_map; }
  int32_t get_detection_distance() const { return _detection_distance; }
  Summary& get_summary() { return _summary; }
  // setter
  void set_design_name(const std::string& design_name) { _design_name = design_name; }
  void set_lef_file_path_list(const std::vector<std::string>& lef_file_path_list) { _lef_file_path_list = lef_file_path_list; }
  void set_def_file_path(const std::string& def_file_path) { _def_file_path = def_file_path; }
  void set_micron_dbu(const int32_t micron_dbu) { _micron_dbu = micron_dbu; }
  void set_manufacture_grid(const int32_t manufacture_grid) { _manufacture_grid = manufacture_grid; }
  void set_detection_distance(const int32_t detection_distance) { _detection_distance = detection_distance; }
  // function

 private:
  std::string _design_name;
  std::vector<std::string> _lef_file_path_list;
  std::string _def_file_path;
  int32_t _micron_dbu = -1;
  int32_t _manufacture_grid = -1;
  ScaleAxis _gcell_axis;
  Die _die;
  Row _row;
  std::vector<RoutingLayer> _routing_layer_list;
  std::vector<CutLayer> _cut_layer_list;
  std::map<int32_t, int32_t> _routing_idb_layer_id_to_idx_map;
  std::map<int32_t, int32_t> _cut_idb_layer_id_to_idx_map;
  std::map<std::string, int32_t> _routing_layer_name_to_idx_map;
  std::map<int32_t, std::vector<int32_t>> _routing_to_adjacent_cut_map;
  std::map<std::string, int32_t> _cut_layer_name_to_idx_map;
  std::map<int32_t, std::vector<int32_t>> _cut_to_adjacent_routing_map;
  std::vector<std::vector<ViaMaster>> _layer_via_master_list;
  std::map<int32_t, PlanarRect> _layer_enclosure_map;
  std::map<int32_t, PlanarRect> _layer_cut_shape_map;
  std::vector<Obstacle> _routing_obstacle_list;
  std::vector<Obstacle> _cut_obstacle_list;
  std::map<std::string, PlanarRect> _block_shape_map;
  std::vector<Net> _net_list;
  GridMap<GCell> _gcell_map;
  int32_t _detection_distance = -1;
  Summary _summary;
};

}  // namespace irt
