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

#include "Blockage.hpp"
#include "CutLayer.hpp"
#include "Die.hpp"
#include "GCell.hpp"
#include "Helper.hpp"
#include "Net.hpp"
#include "RoutingLayer.hpp"
#include "ViaMaster.hpp"

namespace irt {

class Database
{
 public:
  Database() = default;
  ~Database() = default;
  // getter
  irt_int get_micron_dbu() const { return _micron_dbu; }
  ScaleAxis& get_gcell_axis() { return _gcell_axis; }
  Die& get_die() { return _die; }
  std::vector<RoutingLayer>& get_routing_layer_list() { return _routing_layer_list; }
  std::vector<CutLayer>& get_cut_layer_list() { return _cut_layer_list; }
  std::vector<std::vector<ViaMaster>>& get_layer_via_master_list() { return _layer_via_master_list; }
  std::vector<Blockage>& get_routing_blockage_list() { return _routing_blockage_list; }
  std::vector<Blockage>& get_cut_blockage_list() { return _cut_blockage_list; }
  std::vector<Net>& get_net_list() { return _net_list; }
  GridMap<GCell>& get_gcell_map() { return _gcell_map; }
  // setter
  void set_micron_dbu(const irt_int micron_dbu) { _micron_dbu = micron_dbu; }
  // function

 private:
  irt_int _micron_dbu = -1;
  ScaleAxis _gcell_axis;
  Die _die;
  std::vector<RoutingLayer> _routing_layer_list;
  std::vector<CutLayer> _cut_layer_list;
  std::vector<std::vector<ViaMaster>> _layer_via_master_list;
  std::vector<Blockage> _routing_blockage_list;
  std::vector<Blockage> _cut_blockage_list;
  std::vector<Net> _net_list;
  GridMap<GCell> _gcell_map;
};

}  // namespace irt
