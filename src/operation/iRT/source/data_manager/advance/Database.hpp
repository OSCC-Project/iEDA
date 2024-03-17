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

#include "Obstacle.hpp"
#include "CutLayer.hpp"
#include "Die.hpp"
#include "GCell.hpp"
#include "Helper.hpp"
#include "Net.hpp"
#include "RoutingLayer.hpp"
#include "Row.hpp"
#include "ViaMaster.hpp"

namespace irt {

class Database
{
 public:
  Database() = default;
  ~Database() = default;
  // getter
  int32_t get_micron_dbu() const { return _micron_dbu; }
  ScaleAxis& get_gcell_axis() { return _gcell_axis; }
  Die& get_die() { return _die; }
  Row& get_row() { return _row; }
  std::vector<RoutingLayer>& get_routing_layer_list() { return _routing_layer_list; }
  std::vector<CutLayer>& get_cut_layer_list() { return _cut_layer_list; }
  std::vector<std::vector<ViaMaster>>& get_layer_via_master_list() { return _layer_via_master_list; }
  std::vector<Obstacle>& get_routing_obstacle_list() { return _routing_obstacle_list; }
  std::vector<Obstacle>& get_cut_obstacle_list() { return _cut_obstacle_list; }
  std::vector<Net>& get_net_list() { return _net_list; }
  GridMap<GCell>& get_gcell_map() { return _gcell_map; }
  // setter
  void set_micron_dbu(const int32_t micron_dbu) { _micron_dbu = micron_dbu; }
  // function

 private:
  int32_t _micron_dbu = -1;
  ScaleAxis _gcell_axis;
  Die _die;
  Row _row;
  std::vector<RoutingLayer> _routing_layer_list;
  std::vector<CutLayer> _cut_layer_list;
  std::vector<std::vector<ViaMaster>> _layer_via_master_list;
  std::vector<Obstacle> _routing_obstacle_list;
  std::vector<Obstacle> _cut_obstacle_list;
  std::vector<Net> _net_list;
  GridMap<GCell> _gcell_map;
};

}  // namespace irt
