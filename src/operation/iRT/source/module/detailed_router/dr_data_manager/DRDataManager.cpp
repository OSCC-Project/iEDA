#include "DRDataManager.hpp"

#include "DRNet.hpp"
#include "RTUtil.hpp"

namespace irt {

// public

void DRDataManager::input(Config& config, Database& database)
{
  wrapConfig(config);
  wrapDatabase(database);
  buildConfig();
  buildDatabase();
}

std::vector<DRNet> DRDataManager::convertToDRNetList(std::vector<Net>& net_list)
{
  std::vector<DRNet> dr_net_list;
  dr_net_list.reserve(net_list.size());
  for (Net& net : net_list) {
    dr_net_list.emplace_back(convertToDRNet(net));
  }
  return dr_net_list;
}

DRNet DRDataManager::convertToDRNet(Net& net)
{
  DRNet dr_net;
  dr_net.set_origin_net(&net);
  dr_net.set_net_idx(net.get_net_idx());
  dr_net.set_connect_type(net.get_connect_type());
  for (Pin& pin : net.get_pin_list()) {
    dr_net.get_dr_pin_list().push_back(DRPin(pin));
  }
  dr_net.set_ta_result_tree(net.get_ta_result_tree());
  dr_net.set_dr_result_tree(net.get_ta_result_tree());
  return dr_net;
}

// private

void DRDataManager::wrapConfig(Config& config)
{
  _dr_config.temp_directory_path = config.dr_temp_directory_path;
  _dr_config.bottom_routing_layer_idx = config.bottom_routing_layer_idx;
  _dr_config.top_routing_layer_idx = config.top_routing_layer_idx;
}

void DRDataManager::wrapDatabase(Database& database)
{
  wrapMicronDBU(database);
  wrapGCellAxis(database);
  wrapDie(database);
  wrapRoutingLayerList(database);
  wrapCutLayerList(database);
  wrapLayerViaMasterList(database);
  wrapRoutingBlockageList(database);
}

void DRDataManager::wrapMicronDBU(Database& database)
{
  _dr_database.set_micron_dbu(database.get_micron_dbu());
}

void DRDataManager::wrapGCellAxis(Database& database)
{
  GCellAxis& gcell_axis = _dr_database.get_gcell_axis();
  gcell_axis = database.get_gcell_axis();
}

void DRDataManager::wrapDie(Database& database)
{
  EXTPlanarRect& die = _dr_database.get_die();
  die = database.get_die();
}

void DRDataManager::wrapRoutingLayerList(Database& database)
{
  std::vector<RoutingLayer>& routing_layer_list = _dr_database.get_routing_layer_list();
  routing_layer_list = database.get_routing_layer_list();
}

void DRDataManager::wrapCutLayerList(Database& database)
{
  std::vector<CutLayer>& cut_layer_list = _dr_database.get_cut_layer_list();
  cut_layer_list = database.get_cut_layer_list();
}

void DRDataManager::wrapLayerViaMasterList(Database& database)
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _dr_database.get_layer_via_master_list();
  layer_via_master_list = database.get_layer_via_master_list();
}

void DRDataManager::wrapRoutingBlockageList(Database& database)
{
  std::vector<Blockage>& routing_blockage_list = _dr_database.get_routing_blockage_list();
  routing_blockage_list = database.get_routing_blockage_list();
}

void DRDataManager::buildConfig()
{
}

void DRDataManager::buildDatabase()
{
}

}  // namespace irt
