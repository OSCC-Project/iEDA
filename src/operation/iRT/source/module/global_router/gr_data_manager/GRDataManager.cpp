#include "GRDataManager.hpp"

#include "RTUtil.hpp"

namespace irt {

// public

void GRDataManager::input(Config& config, Database& database)
{
  wrapConfig(config);
  wrapDatabase(database);
  buildConfig();
  buildDatabase();
}

std::vector<GRNet> GRDataManager::convertToGRNetList(std::vector<Net>& net_list)
{
  std::vector<GRNet> gr_net_list;
  gr_net_list.reserve(net_list.size());
  for (size_t i = 0; i < net_list.size(); i++) {
    gr_net_list.emplace_back(convertToGRNet(net_list[i]));
  }
  return gr_net_list;
}

GRNet GRDataManager::convertToGRNet(Net& net)
{
  GRNet gr_net;
  gr_net.set_origin_net(&net);
  gr_net.set_net_idx(net.get_net_idx());
  gr_net.set_connect_type(net.get_connect_type());
  for (Pin& pin : net.get_pin_list()) {
    gr_net.get_gr_pin_list().push_back(GRPin(pin));
  }
  gr_net.set_gr_driving_pin(GRPin(net.get_driving_pin()));
  gr_net.set_bounding_box(net.get_bounding_box());
  gr_net.set_ra_cost_map(net.get_ra_cost_map());
  return gr_net;
}

// private

void GRDataManager::wrapConfig(Config& config)
{
  _gr_config.temp_directory_path = config.gr_temp_directory_path;
  _gr_config.bottom_routing_layer_idx = config.bottom_routing_layer_idx;
  _gr_config.top_routing_layer_idx = config.top_routing_layer_idx;
  _gr_config.layer_idx_utilization_ratio = config.layer_idx_utilization_ratio;
}

void GRDataManager::wrapDatabase(Database& database)
{
  wrapMicronDBU(database);
  wrapGCellAxis(database);
  wrapDie(database);
  wrapRoutingLayerList(database);
  wrapCutLayerList(database);
  wrapLayerViaMasterList(database);
  wrapRoutingBlockageList(database);
}

void GRDataManager::wrapMicronDBU(Database& database)
{
  _gr_database.set_micron_dbu(database.get_micron_dbu());
}

void GRDataManager::wrapGCellAxis(Database& database)
{
  GCellAxis& gcell_axis = _gr_database.get_gcell_axis();
  gcell_axis = database.get_gcell_axis();
}

void GRDataManager::wrapDie(Database& database)
{
  Die& die = _gr_database.get_die();
  die = database.get_die();
}

void GRDataManager::wrapRoutingLayerList(Database& database)
{
  std::vector<RoutingLayer>& routing_layer_list = _gr_database.get_routing_layer_list();
  routing_layer_list = database.get_routing_layer_list();
}

void GRDataManager::wrapCutLayerList(Database& database)
{
  std::vector<CutLayer>& cut_layer_list = _gr_database.get_cut_layer_list();
  cut_layer_list = database.get_cut_layer_list();
}

void GRDataManager::wrapLayerViaMasterList(Database& database)
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _gr_database.get_layer_via_master_list();
  layer_via_master_list = database.get_layer_via_master_list();
}

void GRDataManager::wrapRoutingBlockageList(Database& database)
{
  std::vector<Blockage>& routing_blockage_list = _gr_database.get_routing_blockage_list();
  routing_blockage_list = database.get_routing_blockage_list();
}

void GRDataManager::buildConfig()
{
}

void GRDataManager::buildDatabase()
{
}

}  // namespace irt
