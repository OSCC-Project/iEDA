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
#ifndef IDRC_SRC_DB_TECH_H_
#define IDRC_SRC_DB_TECH_H_

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "DrcLayer.h"
#include "DrcVia.h"

namespace idrc {
class Tech
{
 public:
  Tech() {}
  ~Tech()
  {
    clear_drc_routing_layer_list();
    clear_drc_cut_layer_list();
    clear_via_lib();
  }
  // setter
  // DrcRoutingLayer* add_routing_layer();
  // DrcCutLayer* add_cut_layer();
  // DrcVia* add_via();
  // getter
  std::vector<DrcRoutingLayer*>& get_drc_routing_layer_list() { return _drc_routing_layer_list; }
  std::vector<DrcCutLayer*>& get_drc_cut_layer_list() { return _drc_cut_layer_list; }
  std::vector<DrcVia*>& get_via_lib() { return _via_lib; }
  // function
  int getLayerIdByOrder(int layer_order) { return _layer_order_to_id_map[layer_order].second; }
  void insertOrderToIdMap(int order, bool is_cut, int layer_id) { _layer_order_to_id_map[order] = std::make_pair(is_cut, layer_id); }
  int getRoutingWidth(int routingLayerId);
  int getRoutingSpacing(int routingLayerId, int width);
  int getRoutingMinWidth(int routingLayerId);
  // int getRoutingMinArea(int routingLayerId);
  int getRoutingMinEnclosedArea(int routingLayerId);
  int getRoutingMaxRequireSpacing(int routingLayerId, DrcRect* target_rect);
  DrcVia* findViaByIdx(int idx) { return idx >= 0 && idx < (int) _via_lib.size() ? (_via_lib[idx]) : nullptr; }
  // LayerDirection getLayerDirection(int routingLayerId);
  int getCutSpacing(int cutLayerId);
  // DrcCutLayer* getCutLayerById(int layer_id);

  std::string getCutLayerNameById(int layer_id);
  std::string getRoutingLayerNameById(int layer_id);
  ///////////
  std::pair<bool, int> getLayerInfoByLayerName(const std::string& name);

  int getLayerIdByLayerName(const std::string& name);
  // clear
  void clear_drc_routing_layer_list();
  void clear_drc_cut_layer_list();
  void clear_via_lib();

 private:
  std::vector<DrcRoutingLayer*> _drc_routing_layer_list;
  std::vector<DrcCutLayer*> _drc_cut_layer_list;
  std::vector<DrcVia*> _via_lib;
  std::map<int, std::pair<int, int>> _layer_order_to_id_map;
};
}  // namespace idrc

#endif