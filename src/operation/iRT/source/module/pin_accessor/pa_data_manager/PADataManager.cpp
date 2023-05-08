#include "PADataManager.hpp"

#include "RTUtil.hpp"

namespace irt {

// public

void PADataManager::input(Config& config, Database& database)
{
  wrapConfig(config);
  wrapDatabase(database);
  buildConfig();
  buildDatabase();
}

std::vector<PANet> PADataManager::convertToPANetList(std::vector<Net>& net_list)
{
  std::vector<PANet> pa_net_list;
  pa_net_list.reserve(net_list.size());
  for (Net& net : net_list) {
    pa_net_list.emplace_back(convertToPANet(net));
  }
  return pa_net_list;
}

PANet PADataManager::convertToPANet(Net& net)
{
  PANet pa_net;
  pa_net.set_origin_net(&net);
  pa_net.set_net_idx(net.get_net_idx());
  pa_net.set_net_name(net.get_net_name());
  for (Pin& pin : net.get_pin_list()) {
    pa_net.get_pa_pin_list().push_back(PAPin(pin));
  }
  pa_net.set_pa_driving_pin(PAPin(net.get_driving_pin()));
  pa_net.set_bounding_box(net.get_bounding_box());
  return pa_net;
}

// private

void PADataManager::wrapConfig(Config& config)
{
  _pa_config.temp_directory_path = config.pa_temp_directory_path;
  _pa_config.bottom_routing_layer_idx = config.bottom_routing_layer_idx;
  _pa_config.top_routing_layer_idx = config.top_routing_layer_idx;
}

void PADataManager::wrapDatabase(Database& database)
{
  wrapCellAxis(database);
  wrapDie(database);
  wrapRoutingLayerList(database);
  wrapLayerViaMasterList(database);
  wrapRoutingBlockageList(database);
}

void PADataManager::wrapCellAxis(Database& database)
{
  GCellAxis& gcell_axis = _pa_database.get_gcell_axis();
  gcell_axis = database.get_gcell_axis();
}

void PADataManager::wrapDie(Database& database)
{
  Die& die = _pa_database.get_die();
  die = database.get_die();
}

void PADataManager::wrapRoutingLayerList(Database& database)
{
  std::vector<RoutingLayer>& routing_layer_list = _pa_database.get_routing_layer_list();
  routing_layer_list = database.get_routing_layer_list();
}

void PADataManager::wrapLayerViaMasterList(Database& database)
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _pa_database.get_layer_via_master_list();
  layer_via_master_list = database.get_layer_via_master_list();
}

void PADataManager::wrapRoutingBlockageList(Database& database)
{
  std::vector<Blockage>& routing_blockage_list = _pa_database.get_routing_blockage_list();
  routing_blockage_list = database.get_routing_blockage_list();
}

void PADataManager::buildConfig()
{
}

void PADataManager::buildDatabase()
{
}

}  // namespace irt
