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
#include "RADataManager.hpp"

namespace irt {

// public

void RADataManager::input(Config& config, Database& database)
{
  wrapConfig(config);
  wrapDatabase(database);
  buildConfig();
  buildDatabase();
}

std::vector<RANet> RADataManager::convertToRANetList(std::vector<Net>& net_list)
{
  std::vector<RANet> ra_net_list;
  ra_net_list.reserve(net_list.size());
  for (size_t i = 0; i < net_list.size(); i++) {
    ra_net_list.emplace_back(convertToRANet(net_list[i]));
  }
  return ra_net_list;
}

RANet RADataManager::convertToRANet(Net& net)
{
  RANet ra_net;
  ra_net.set_origin_net(&net);
  ra_net.set_net_idx(net.get_net_idx());
  for (Pin& pin : net.get_pin_list()) {
    ra_net.get_ra_pin_list().push_back(RAPin(pin));
  }
  ra_net.set_bounding_box(net.get_bounding_box());
  return ra_net;
}

// private

void RADataManager::wrapConfig(Config& config)
{
  _ra_config.temp_directory_path = config.ra_temp_directory_path;
  _ra_config.bottom_routing_layer_idx = config.bottom_routing_layer_idx;
  _ra_config.top_routing_layer_idx = config.top_routing_layer_idx;
  _ra_config.layer_idx_utilization_ratio = config.layer_idx_utilization_ratio;
  _ra_config.initial_penalty = config.resource_allocate_initial_penalty;
  _ra_config.penalty_drop_rate = config.resource_allocate_penalty_drop_rate;
  _ra_config.outer_iter_num = config.resource_allocate_outer_iter_num;
  _ra_config.inner_iter_num = config.resource_allocate_inner_iter_num;
}

void RADataManager::wrapDatabase(Database& database)
{
  wrapGCellAxis(database);
  wrapDie(database);
  wrapRoutingLayerList(database);
  wrapLayerViaMasterList(database);
  wrapRoutingBlockageList(database);
}

void RADataManager::wrapGCellAxis(Database& database)
{
  GCellAxis& gcell_axis = _ra_database.get_gcell_axis();
  gcell_axis = database.get_gcell_axis();
}

void RADataManager::wrapDie(Database& database)
{
  Die& die = _ra_database.get_die();
  die = database.get_die();
}

void RADataManager::wrapRoutingLayerList(Database& database)
{
  std::vector<RoutingLayer>& routing_layer_list = _ra_database.get_routing_layer_list();
  routing_layer_list = database.get_routing_layer_list();
}

void RADataManager::wrapLayerViaMasterList(Database& database)
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _ra_database.get_layer_via_master_list();
  layer_via_master_list = database.get_layer_via_master_list();
}

void RADataManager::wrapRoutingBlockageList(Database& database)
{
  std::vector<Blockage>& routing_blockage_list = _ra_database.get_routing_blockage_list();
  routing_blockage_list = database.get_routing_blockage_list();
}

void RADataManager::buildConfig()
{
}

void RADataManager::buildDatabase()
{
}

}  // namespace irt
