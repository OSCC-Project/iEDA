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
#include "TADataManager.hpp"

#include "RTUtil.hpp"

namespace irt {

// public

void TADataManager::input(Config& config, Database& database)
{
  wrapConfig(config);
  wrapDatabase(database);
  buildConfig();
  buildDatabase();
}

std::vector<TANet> TADataManager::convertToTANetList(std::vector<Net>& net_list)
{
  std::vector<TANet> ta_net_list;
  ta_net_list.reserve(net_list.size());
  for (size_t i = 0; i < net_list.size(); i++) {
    ta_net_list.emplace_back(convertToTANet(net_list[i]));
  }
  return ta_net_list;
}

TANet TADataManager::convertToTANet(Net& net)
{
  TANet ta_net;
  ta_net.set_origin_net(&net);
  ta_net.set_net_idx(net.get_net_idx());
  for (Pin& pin : net.get_pin_list()) {
    ta_net.get_ta_pin_list().push_back(TAPin(pin));
  }
  ta_net.set_gr_result_tree(net.get_gr_result_tree());
  ta_net.set_ta_result_tree(net.get_gr_result_tree());
  return ta_net;
}

// private

void TADataManager::wrapConfig(Config& config)
{
  _ta_config.temp_directory_path = config.ta_temp_directory_path;
  _ta_config.bottom_routing_layer_idx = config.bottom_routing_layer_idx;
  _ta_config.top_routing_layer_idx = config.top_routing_layer_idx;
}

void TADataManager::wrapDatabase(Database& database)
{
  wrapMicronDBU(database);
  wrapGCellAxis(database);
  wrapDie(database);
  wrapRoutingLayerList(database);
  wrapLayerViaMasterList(database);
  wrapRoutingBlockageList(database);
}

void TADataManager::wrapMicronDBU(Database& database)
{
  _ta_database.set_micron_dbu(database.get_micron_dbu());
}

void TADataManager::wrapGCellAxis(Database& database)
{
  GCellAxis& gcell_axis = _ta_database.get_gcell_axis();
  gcell_axis = database.get_gcell_axis();
}

void TADataManager::wrapDie(Database& database)
{
  EXTPlanarRect& die = _ta_database.get_die();
  die = database.get_die();
}

void TADataManager::wrapRoutingLayerList(Database& database)
{
  std::vector<RoutingLayer>& routing_layer_list = _ta_database.get_routing_layer_list();
  routing_layer_list = database.get_routing_layer_list();
}

void TADataManager::wrapLayerViaMasterList(Database& database)
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _ta_database.get_layer_via_master_list();
  layer_via_master_list = database.get_layer_via_master_list();
}

void TADataManager::wrapRoutingBlockageList(Database& database)
{
  std::vector<Blockage>& routing_blockage_list = _ta_database.get_routing_blockage_list();
  routing_blockage_list = database.get_routing_blockage_list();
}

void TADataManager::buildConfig()
{
}

void TADataManager::buildDatabase()
{
}

}  // namespace irt
