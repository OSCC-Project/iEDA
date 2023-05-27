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
#include "VRDataManager.hpp"

#include "RTUtil.hpp"
#include "VRNet.hpp"

namespace irt {

// public

void VRDataManager::input(Config& config, Database& database)
{
  wrapConfig(config);
  wrapDatabase(database);
  buildConfig();
  buildDatabase();
}

std::vector<VRNet> VRDataManager::convertToVRNetList(std::vector<Net>& net_list)
{
  std::vector<VRNet> vr_net_list;
  vr_net_list.reserve(net_list.size());
  for (Net& net : net_list) {
    vr_net_list.emplace_back(convertToVRNet(net));
  }
  return vr_net_list;
}

VRNet VRDataManager::convertToVRNet(Net& net)
{
  VRNet vr_net;
  vr_net.set_origin_net(&net);
  vr_net.set_net_idx(net.get_net_idx());
  for (Pin& pin : net.get_pin_list()) {
    vr_net.get_vr_pin_list().push_back(VRPin(pin));
  }
  vr_net.set_vr_driving_pin(VRPin(net.get_driving_pin()));
  vr_net.set_bounding_box(net.get_bounding_box());
  vr_net.set_dr_result_tree(net.get_dr_result_tree());
  return vr_net;
}

// private

void VRDataManager::wrapConfig(Config& config)
{
  _vr_config.temp_directory_path = config.vr_temp_directory_path;
}

void VRDataManager::wrapDatabase(Database& database)
{
  wrapMicronDBU(database);
  wrapGCellAxis(database);
  wrapDie(database);
  wrapRoutingLayerList(database);
  wrapCutLayerList(database);
  wrapLayerViaMasterList(database);
  wrapRoutingBlockageList(database);
}

void VRDataManager::wrapMicronDBU(Database& database)
{
  _vr_database.set_micron_dbu(database.get_micron_dbu());
}

void VRDataManager::wrapGCellAxis(Database& database)
{
  GCellAxis& gcell_axis = _vr_database.get_gcell_axis();
  gcell_axis = database.get_gcell_axis();
}

void VRDataManager::wrapDie(Database& database)
{
  Die& die = _vr_database.get_die();
  die = database.get_die();
}

void VRDataManager::wrapRoutingLayerList(Database& database)
{
  std::vector<RoutingLayer>& routing_layer_list = _vr_database.get_routing_layer_list();
  routing_layer_list = database.get_routing_layer_list();
}

void VRDataManager::wrapCutLayerList(Database& database)
{
  std::vector<CutLayer>& cut_layer_list = _vr_database.get_cut_layer_list();
  cut_layer_list = database.get_cut_layer_list();
}

void VRDataManager::wrapLayerViaMasterList(Database& database)
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _vr_database.get_layer_via_master_list();
  layer_via_master_list = database.get_layer_via_master_list();
}

void VRDataManager::wrapRoutingBlockageList(Database& database)
{
  std::vector<Blockage>& routing_blockage_list = _vr_database.get_routing_blockage_list();
  routing_blockage_list = database.get_routing_blockage_list();
}

void VRDataManager::buildConfig()
{
}

void VRDataManager::buildDatabase()
{
}

}  // namespace irt
