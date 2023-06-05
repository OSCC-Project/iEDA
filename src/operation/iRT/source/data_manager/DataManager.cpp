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
#include "DataManager.hpp"

#include "RTAPI.hpp"
#include "RTU.hpp"
#include "RTUtil.hpp"

using namespace std;

namespace irt {

// public

void DataManager::input(std::map<std::string, std::any>& config_map, idb::IdbBuilder* idb_builder)
{
  Monitor monitor;

  wrapConfig(config_map);
  wrapDatabase(idb_builder);
  buildConfig();
  buildDatabase();
  printConfig();
  printDatabase();

  LOG_INST.info(Loc::current(), "The data manager input completed!", monitor.getStatsInfo());
}

void DataManager::save(Stage stage)
{
  Monitor monitor;

  saveStageResult(stage);

  LOG_INST.info(Loc::current(), "The data manager save completed!", monitor.getStatsInfo());
}

void DataManager::load(Stage stage)
{
  Monitor monitor;

  loadStageResult(stage);

  LOG_INST.info(Loc::current(), "The data manager load completed!", monitor.getStatsInfo());
}

void DataManager::output(idb::IdbBuilder* idb_builder)
{
  Monitor monitor;

  outputGCellGrid(idb_builder);
  outputNetList(idb_builder);

  LOG_INST.info(Loc::current(), "The data manager output completed!", monitor.getStatsInfo());
}

// private

void DataManager::wrapConfig(std::map<std::string, std::any>& config_map)
{
  /////////////////////////////////////////////
  _config.temp_directory_path = RTUtil::getConfigValue<std::string>(config_map, "-temp_directory_path", "./rt_temp_directory");
  _config.log_level = RTUtil::getConfigValue<irt_int>(config_map, "-log_level", 0);
  _config.thread_number = RTUtil::getConfigValue<irt_int>(config_map, "-thread_number", 8);
  _config.bottom_routing_layer = RTUtil::getConfigValue<std::string>(config_map, "-bottom_routing_layer", "");
  _config.top_routing_layer = RTUtil::getConfigValue<std::string>(config_map, "-top_routing_layer", "");
  _config.layer_utilization_ratio = RTUtil::getConfigValue<std::map<std::string, double>>(config_map, "-layer_utilization_ratio", {});
  _config.enable_output_gds_files = RTUtil::getConfigValue<irt_int>(config_map, "-enable_output_gds_files", 0);
  _config.resource_allocate_initial_penalty = RTUtil::getConfigValue<double>(config_map, "-resource_allocate_initial_penalty", 100);
  _config.resource_allocate_penalty_drop_rate = RTUtil::getConfigValue<double>(config_map, "-resource_allocate_penalty_drop_rate", 0.8);
  _config.resource_allocate_outer_iter_num = RTUtil::getConfigValue<irt_int>(config_map, "-resource_allocate_outer_iter_num", 10);
  _config.resource_allocate_inner_iter_num = RTUtil::getConfigValue<irt_int>(config_map, "-resource_allocate_inner_iter_num", 10);
  /////////////////////////////////////////////
}

void DataManager::wrapDatabase(idb::IdbBuilder* idb_builder)
{
  wrapMicronDBU(idb_builder);
  wrapDie(idb_builder);
  wrapLayerList(idb_builder);
  wrapLayerViaMasterList(idb_builder);
  wrapBlockageList(idb_builder);
  wrapNetList(idb_builder);
  updateHelper(idb_builder);
}

void DataManager::wrapMicronDBU(idb::IdbBuilder* idb_builder)
{
  _database.set_micron_dbu(idb_builder->get_def_service()->get_design()->get_units()->get_micron_dbu());
}

void DataManager::wrapDie(idb::IdbBuilder* idb_builder)
{
  idb::IdbDie* die = idb_builder->get_lef_service()->get_layout()->get_die();

  EXTPlanarRect& die_box = _database.get_die();
  die_box.set_real_lb(die->get_llx(), die->get_lly());
  die_box.set_real_rt(die->get_urx(), die->get_ury());
}

void DataManager::wrapLayerList(idb::IdbBuilder* idb_builder)
{
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = _database.get_cut_layer_list();

  std::vector<idb::IdbLayer*>& idb_layers = idb_builder->get_lef_service()->get_layout()->get_layers()->get_layers();
  for (idb::IdbLayer* idb_layer : idb_layers) {
    if (idb_layer->is_routing()) {
      idb::IdbLayerRouting* idb_routing_layer = dynamic_cast<idb::IdbLayerRouting*>(idb_layer);
      RoutingLayer routing_layer;
      routing_layer.set_layer_idx(idb_routing_layer->get_id());
      routing_layer.set_layer_order(idb_routing_layer->get_order());
      routing_layer.set_layer_name(idb_routing_layer->get_name());
      routing_layer.set_min_width(idb_routing_layer->get_min_width());
      routing_layer.set_min_area(idb_routing_layer->get_area());
      routing_layer.set_direction(getRTDirectionByDB(idb_routing_layer->get_direction()));
      wrapTrackAxis(routing_layer, idb_routing_layer);
      wrapSpacingTable(routing_layer, idb_routing_layer);
      routing_layer_list.push_back(std::move(routing_layer));
    } else if (idb_layer->is_cut()) {
      idb::IdbLayerCut* idb_cut_layer = dynamic_cast<idb::IdbLayerCut*>(idb_layer);
      CutLayer cut_layer;
      cut_layer.set_layer_idx(idb_cut_layer->get_id());
      cut_layer.set_layer_order(idb_cut_layer->get_order());
      cut_layer.set_layer_name(idb_cut_layer->get_name());
      cut_layer_list.push_back(std::move(cut_layer));
    }
  }
}

void DataManager::wrapTrackAxis(RoutingLayer& routing_layer, idb::IdbLayerRouting* idb_layer)
{
  TrackAxis& track_axis = routing_layer.get_track_axis();

  for (idb::IdbTrackGrid* idb_track_grid : idb_layer->get_track_grid_list()) {
    idb::IdbTrack* idb_track = idb_track_grid->get_track();

    TrackGrid track_grid;
    track_grid.set_start_line(static_cast<irt_int>(idb_track->get_start()));
    track_grid.set_step_length(static_cast<irt_int>(idb_track->get_pitch()));
    track_grid.set_step_num(static_cast<irt_int>(idb_track_grid->get_track_num()));

    if (idb_track->get_direction() == idb::IdbTrackDirection::kDirectionX) {
      track_axis.set_x_track_grid(track_grid);
    } else if (idb_track->get_direction() == idb::IdbTrackDirection::kDirectionY) {
      track_axis.set_y_track_grid(track_grid);
    }
  }
}

void DataManager::wrapSpacingTable(RoutingLayer& routing_layer, idb::IdbLayerRouting* idb_layer)
{
  std::shared_ptr<idb::IdbParallelSpacingTable> idb_spacing_table;
  if (idb_layer->get_spacing_table().get()->get_parallel().get() != nullptr && idb_layer->get_spacing_table().get()->is_parallel()) {
    idb_spacing_table = idb_layer->get_spacing_table()->get_parallel();
  } else if (idb_layer->get_spacing_list() != nullptr && !idb_layer->get_spacing_table().get()->is_parallel()) {
    idb_spacing_table = idb_layer->get_spacing_table_from_spacing_list()->get_parallel();
  } else {
    LOG_INST.error(Loc::current(), "The idb spacing table is error!");
  }

  SpacingTable& spacing_table = routing_layer.get_spacing_table();
  std::vector<irt_int>& width_list = spacing_table.get_width_list();
  std::vector<irt_int>& parallel_length_list = spacing_table.get_parallel_length_list();
  GridMap<irt_int>& width_parallel_length_map = spacing_table.get_width_parallel_length_map();

  width_list = idb_spacing_table->get_width_list();
  parallel_length_list = idb_spacing_table->get_parallel_length_list();
  width_parallel_length_map.init(width_list.size(), parallel_length_list.size());
  for (irt_int x = 0; x < width_parallel_length_map.get_x_size(); x++) {
    for (irt_int y = 0; y < width_parallel_length_map.get_y_size(); y++) {
      width_parallel_length_map[x][y] = idb_spacing_table->get_spacing_table()[x][y];
    }
  }
}

void DataManager::wrapLayerViaMasterList(idb::IdbBuilder* idb_builder)
{
  idb::IdbVias* idb_via_list_lib = idb_builder->get_lef_service()->get_layout()->get_via_list();
  if (idb_via_list_lib == nullptr) {
    LOG_INST.error(Loc::current(), "Via list in tech lef is empty!");
  }

  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _database.get_layer_via_master_list();
  std::vector<idb::IdbLayer*> idb_routing_layers = idb_builder->get_lef_service()->get_layout()->get_layers()->get_routing_layers();
  layer_via_master_list.resize(idb_routing_layers.size());

  std::vector<idb::IdbVia*>& idb_via_list = idb_via_list_lib->get_via_list();
  for (size_t i = 0; i < idb_via_list.size(); i++) {
    idb::IdbVia* idb_via = idb_via_list[i];
    if (idb_via == nullptr) {
      LOG_INST.error(Loc::current(), "The via is empty!");
    }
    ViaMaster via_master;
    via_master.set_via_name(idb_via->get_name());
    idb::IdbViaMaster* idb_via_master = idb_via->get_instance();
    // top enclosure
    idb::IdbLayerShape* idb_shape_top = idb_via_master->get_top_layer_shape();
    idb::IdbLayerRouting* idb_layer_top = dynamic_cast<idb::IdbLayerRouting*>(idb_shape_top->get_layer());
    idb::IdbRect idb_box_top = idb_shape_top->get_bounding_box();
    LayerRect above_enclosure(idb_box_top.get_low_x(), idb_box_top.get_low_y(), idb_box_top.get_high_x(), idb_box_top.get_high_y(),
                              idb_layer_top->get_id());
    via_master.set_above_enclosure(above_enclosure);
    via_master.set_above_direction(getRTDirectionByDB(idb_layer_top->get_direction()));
    // bottom enclosure
    idb::IdbLayerShape* idb_shape_bottom = idb_via_master->get_bottom_layer_shape();
    idb::IdbLayerRouting* idb_layer_bottom = dynamic_cast<idb::IdbLayerRouting*>(idb_shape_bottom->get_layer());
    idb::IdbRect idb_box_bottom = idb_shape_bottom->get_bounding_box();
    LayerRect below_enclosure(idb_box_bottom.get_low_x(), idb_box_bottom.get_low_y(), idb_box_bottom.get_high_x(),
                              idb_box_bottom.get_high_y(), idb_layer_bottom->get_id());
    via_master.set_below_enclosure(below_enclosure);
    via_master.set_below_direction(getRTDirectionByDB(idb_layer_bottom->get_direction()));
    // cut shape
    idb::IdbLayerShape idb_shape_cut = idb_via->get_cut_layer_shape();
    std::vector<PlanarRect>& cut_shape_list = via_master.get_cut_shape_list();
    for (idb::IdbRect* idb_rect : idb_shape_cut.get_rect_list()) {
      PlanarRect cut_shape;
      cut_shape.set_lb(idb_rect->get_low_x(), idb_rect->get_low_y());
      cut_shape.set_rt(idb_rect->get_high_x(), idb_rect->get_high_y());
      cut_shape_list.push_back(std::move(cut_shape));
    }
    via_master.set_cut_layer_idx(idb_shape_cut.get_layer()->get_id());
    layer_via_master_list.front().push_back(std::move(via_master));
  }
}

void DataManager::wrapBlockageList(idb::IdbBuilder* idb_builder)
{
  wrapArtificialBlockage(idb_builder);
  wrapInstanceBlockage(idb_builder);
  wrapSpecialNetBlockage(idb_builder);
}

void DataManager::wrapArtificialBlockage(idb::IdbBuilder* idb_builder)
{
  std::vector<Blockage>& routing_blockage_list = _database.get_routing_blockage_list();

  // Artificial
  idb::IdbBlockageList* idb_blockage_list = idb_builder->get_def_service()->get_design()->get_blockage_list();
  for (idb::IdbBlockage* idb_blockage : idb_blockage_list->get_blockage_list()) {
    if (idb_blockage->is_routing_blockage()) {
      idb::IdbRoutingBlockage* idb_routing_blockage = dynamic_cast<idb::IdbRoutingBlockage*>(idb_blockage);
      for (idb::IdbRect* rect : idb_routing_blockage->get_rect_list()) {
        Blockage blockage;
        blockage.set_real_lb(rect->get_low_x(), rect->get_low_y());
        blockage.set_real_rt(rect->get_high_x(), rect->get_high_y());
        blockage.set_layer_idx(idb_routing_blockage->get_layer()->get_id());
        blockage.set_is_artificial(true);
        routing_blockage_list.push_back(std::move(blockage));
      }
    }
  }
}

void DataManager::wrapInstanceBlockage(idb::IdbBuilder* idb_builder)
{
  std::vector<Blockage>& routing_blockage_list = _database.get_routing_blockage_list();
  std::vector<Blockage>& cut_blockage_list = _database.get_cut_blockage_list();

  // instance
  std::vector<idb::IdbInstance*> instance_list = idb_builder->get_def_service()->get_design()->get_instance_list()->get_instance_list();
  std::vector<idb::IdbLayerShape*> layer_shape_list;
  for (idb::IdbInstance* instance : instance_list) {
    // instance obs
    std::vector<idb::IdbLayerShape*>& obs_box_list = instance->get_obs_box_list();
    layer_shape_list.insert(layer_shape_list.end(), obs_box_list.begin(), obs_box_list.end());
    // instance pin without net
    for (idb::IdbPin* idb_pin : instance->get_pin_list()->get_pin_list()) {
      if (idb_pin->get_net() != nullptr) {
        continue;
      }
      std::vector<idb::IdbLayerShape*>& port_box_list = idb_pin->get_port_box_list();
      layer_shape_list.insert(layer_shape_list.end(), port_box_list.begin(), port_box_list.end());
    }
  }
  for (idb::IdbLayerShape* layer_shape : layer_shape_list) {
    for (idb::IdbRect* rect : layer_shape->get_rect_list()) {
      Blockage blockage;
      blockage.set_real_lb(rect->get_low_x(), rect->get_low_y());
      blockage.set_real_rt(rect->get_high_x(), rect->get_high_y());
      blockage.set_layer_idx(layer_shape->get_layer()->get_id());
      blockage.set_is_artificial(false);
      if (layer_shape->get_layer()->is_routing()) {
        routing_blockage_list.push_back(std::move(blockage));
      } else if (layer_shape->get_layer()->is_cut()) {
        cut_blockage_list.push_back(std::move(blockage));
      }
    }
  }
}

void DataManager::wrapSpecialNetBlockage(idb::IdbBuilder* idb_builder)
{
  std::vector<Blockage>& routing_blockage_list = _database.get_routing_blockage_list();
  std::vector<Blockage>& cut_blockage_list = _database.get_cut_blockage_list();

  // special net
  idb::IdbSpecialNetList* idb_special_net_list = idb_builder->get_def_service()->get_design()->get_special_net_list();
  for (idb::IdbSpecialNet* idb_net : idb_special_net_list->get_net_list()) {
    for (idb::IdbSpecialWire* idb_wire : idb_net->get_wire_list()->get_wire_list()) {
      for (idb::IdbSpecialWireSegment* idb_segment : idb_wire->get_segment_list()) {
        if (idb_segment->is_via()) {
          std::vector<idb::IdbLayerShape> layer_shape_list;
          layer_shape_list.push_back(idb_segment->get_via()->get_top_layer_shape());
          layer_shape_list.push_back(idb_segment->get_via()->get_bottom_layer_shape());
          layer_shape_list.push_back(idb_segment->get_via()->get_cut_layer_shape());

          for (idb::IdbLayerShape& layer_shape : layer_shape_list) {
            for (idb::IdbRect* rect : layer_shape.get_rect_list()) {
              Blockage blockage;
              blockage.set_real_lb(rect->get_low_x(), rect->get_low_y());
              blockage.set_real_rt(rect->get_high_x(), rect->get_high_y());
              blockage.set_layer_idx(layer_shape.get_layer()->get_id());
              blockage.set_is_artificial(false);
              if (layer_shape.get_layer()->is_routing()) {
                routing_blockage_list.push_back(std::move(blockage));
              } else if (layer_shape.get_layer()->is_cut()) {
                cut_blockage_list.push_back(std::move(blockage));
              }
            }
          }
        } else {
          idb::IdbRect* idb_rect = idb_segment->get_bounding_box();
          // wire
          Blockage blockage;
          blockage.set_real_lb(idb_rect->get_low_x(), idb_rect->get_low_y());
          blockage.set_real_rt(idb_rect->get_high_x(), idb_rect->get_high_y());
          blockage.set_layer_idx(idb_segment->get_layer()->get_id());
          blockage.set_is_artificial(false);
          routing_blockage_list.push_back(std::move(blockage));
        }
      }
    }
  }
}

void DataManager::wrapNetList(idb::IdbBuilder* idb_builder)
{
  std::vector<Net>& net_list = _database.get_net_list();
  std::vector<idb::IdbNet*> idb_net_list = idb_builder->get_def_service()->get_design()->get_net_list()->get_net_list();

  // bound setting
  size_t lower_bound_value = 0;
  // size_t upper_bound_value = 500;
  size_t upper_bound_value = idb_net_list.size();
  size_t number = 0;
  for (idb::IdbNet* idb_net : idb_net_list) {
    if (lower_bound_value <= number && number <= upper_bound_value && !checkSkipping(idb_net)) {
      Net net;
      net.set_net_name(idb_net->get_net_name());
      net.set_connect_type(getRTConnectTypeByDB(idb_net->get_connect_type()));
      wrapPinList(net, idb_net);
      wrapDrivingPin(net, idb_net);
      net_list.push_back(std::move(net));
      number++;
    }
  }
}

bool DataManager::checkSkipping(idb::IdbNet* idb_net)
{
  std::string net_name = idb_net->get_net_name();

  // check pin number
  size_t pin_num = idb_net->get_instance_pin_list()->get_pin_num();
  if (pin_num <= 1) {
    LOG_INST.info(Loc::current(), "The net '", net_name, "' has ", pin_num, " pin! skipping...");
    return true;
  } else if (pin_num >= 500) {
    LOG_INST.warning(Loc::current(), "The ultra large net: ", net_name, " has ", pin_num, " pins!");
    sleep(2);
  }

  // check the connection form io_cell PAD to io_pin
  bool has_io_pin = false;
  if (idb_net->get_io_pin() != nullptr) {
    has_io_pin = true;
  }
  bool has_io_cell = false;
  for (idb::IdbInstance* instance : idb_net->get_instance_list()->get_instance_list()) {
    if (instance->get_cell_master()->is_pad()) {
      has_io_cell = true;
      break;
    }
  }
  if (has_io_pin && has_io_cell) {
    LOG_INST.info(Loc::current(), "The net '", net_name, "' connects PAD and io_pin! skipping...");
    return true;
  }

  if (idb_net->get_connect_type() == idb::IdbConnectType::kNone) {
    idb_net->set_connect_type(idb::IdbConnectType::kSignal);
  }
  if (idb_net->get_connect_type() == idb::IdbConnectType::kNone) {
    LOG_INST.warning(Loc::current(), "The connect type of net '", net_name, "' is none!");
  }

  return false;
}

void DataManager::wrapPinList(Net& net, idb::IdbNet* idb_net)
{
  std::vector<Pin>& pin_list = net.get_pin_list();

  // pin list in instance
  for (idb::IdbPin* idb_pin : idb_net->get_instance_pin_list()->get_pin_list()) {
    /// without term description in some cases
    if (idb_pin->get_term()->get_port_number() <= 0) {
      continue;
    }
    Pin pin;
    pin.set_pin_name(RTUtil::getString(idb_pin->get_instance()->get_name(), ": ", idb_pin->get_pin_name()));
    wrapPinShapeList(pin, idb_pin);
    pin_list.push_back(std::move(pin));
  }
  // io pin list
  if (idb_net->get_io_pin() != nullptr) {
    idb::IdbPin* io_pin = idb_net->get_io_pin();
    Pin pin;
    pin.set_pin_name(io_pin->get_pin_name());
    wrapPinShapeList(pin, io_pin);
    pin_list.push_back(std::move(pin));
  }
}

void DataManager::wrapPinShapeList(Pin& pin, idb::IdbPin* idb_pin)
{
  std::vector<EXTLayerRect>& routing_shape_list = pin.get_routing_shape_list();
  std::vector<EXTLayerRect>& cut_shape_list = pin.get_cut_shape_list();

  for (idb::IdbLayerShape* layer_shape : idb_pin->get_port_box_list()) {
    for (idb::IdbRect* rect : layer_shape->get_rect_list()) {
      EXTLayerRect pin_shape;
      pin_shape.set_real_lb(rect->get_low_x(), rect->get_low_y());
      pin_shape.set_real_rt(rect->get_high_x(), rect->get_high_y());
      pin_shape.set_layer_idx(layer_shape->get_layer()->get_id());
      if (layer_shape->get_layer()->is_routing()) {
        routing_shape_list.push_back(std::move(pin_shape));
      } else if (layer_shape->get_layer()->is_cut()) {
        cut_shape_list.push_back(std::move(pin_shape));
      }
    }
  }
}

void DataManager::wrapDrivingPin(Net& net, idb::IdbNet* idb_net)
{
  idb::IdbPin* idb_driving_pin = idb_net->get_driving_pin();
  if (idb_driving_pin == nullptr) {
    LOG_INST.warning(Loc::current(), "The net '", net.get_net_name(), "' without driving pin!");
    idb_driving_pin = idb_net->get_instance_pin_list()->get_pin_list().front();
  }
  std::string driving_pin_name = idb_driving_pin->get_pin_name();
  if (!idb_driving_pin->is_io_pin()) {
    driving_pin_name = RTUtil::getString(idb_driving_pin->get_instance()->get_name(), ": ", driving_pin_name);
  }
  net.get_driving_pin().set_pin_name(driving_pin_name);
}

void DataManager::updateHelper(idb::IdbBuilder* idb_builder)
{
  _helper.set_design_name(idb_builder->get_def_service()->get_design()->get_design_name());
  _helper.set_lef_file_path_list(idb_builder->get_lef_service()->get_lef_files());
  _helper.set_def_file_path(idb_builder->get_def_service()->get_def_file());

  std::map<irt_int, irt_int>& routing_layer_idx_db_to_rt_map = _helper.get_routing_layer_idx_db_to_rt_map();
  std::map<irt_int, irt_int>& cut_layer_idx_db_to_rt_map = _helper.get_cut_layer_idx_db_to_rt_map();
  std::map<std::string, irt_int>& routing_layer_name_to_idx_map = _helper.get_routing_layer_name_to_idx_map();
  std::map<std::string, irt_int>& cut_layer_name_to_idx_map = _helper.get_cut_layer_name_to_idx_map();

  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();
  for (size_t i = 0; i < routing_layer_list.size(); i++) {
    routing_layer_idx_db_to_rt_map[routing_layer_list[i].get_layer_idx()] = static_cast<irt_int>(i);
    routing_layer_name_to_idx_map[routing_layer_list[i].get_layer_name()] = static_cast<irt_int>(i);
  }
  std::vector<CutLayer>& cut_layer_list = _database.get_cut_layer_list();
  for (size_t i = 0; i < cut_layer_list.size(); i++) {
    cut_layer_idx_db_to_rt_map[cut_layer_list[i].get_layer_idx()] = static_cast<irt_int>(i);
    cut_layer_name_to_idx_map[cut_layer_list[i].get_layer_name()] = static_cast<irt_int>(i);
  }
}

void DataManager::buildConfig()
{
  /////////////////////////////////////////////
  // **********        RT         ********** //
  _config.temp_directory_path = std::filesystem::absolute(_config.temp_directory_path);
  _config.temp_directory_path += "/";
  _config.log_file_path = _config.temp_directory_path + "rt.log";
  if (_config.bottom_routing_layer.empty()) {
    _config.bottom_routing_layer = _database.get_routing_layer_list().front().get_layer_name();
  }
  if (_config.top_routing_layer.empty()) {
    _config.top_routing_layer = _database.get_routing_layer_list().back().get_layer_name();
  }
  _config.bottom_routing_layer_idx = _helper.getRoutingLayerIdxByName(_config.bottom_routing_layer);
  _config.top_routing_layer_idx = _helper.getRoutingLayerIdxByName(_config.top_routing_layer);
  if (_config.bottom_routing_layer_idx >= _config.top_routing_layer_idx) {
    LOG_INST.error(Loc::current(), "The routing layer should be at least two layers!");
  }
  for (auto [layer_name, utilization_ratio] : _config.layer_utilization_ratio) {
    _config.layer_idx_utilization_ratio[_helper.getRoutingLayerIdxByName(layer_name)] = utilization_ratio;
  }
  // **********    DataManager    ********** //
  _config.dm_temp_directory_path = _config.temp_directory_path + "data_manager/";
  // **********  DetailedRouter   ********** //
  _config.dr_temp_directory_path = _config.temp_directory_path + "detailed_router/";
  // **********    GDSPlotter     ********** //
  _config.gp_temp_directory_path = _config.temp_directory_path + "gds_plotter/";
  // **********   GlobalRouter    ********** //
  _config.gr_temp_directory_path = _config.temp_directory_path + "global_router/";
  // **********   PinAccessor     ********** //
  _config.pa_temp_directory_path = _config.temp_directory_path + "pin_accessor/";
  // ********   ResourceAllocator   ******** //
  _config.ra_temp_directory_path = _config.temp_directory_path + "resource_allocator/";
  // **********   TrackAssigner   ********** //
  _config.ta_temp_directory_path = _config.temp_directory_path + "track_assigner/";
  // **********  UniversalRouter  ********** //
  _config.ur_temp_directory_path = _config.temp_directory_path + "universal_router/";
  // ********** ViolationRepairer ********** //
  _config.vr_temp_directory_path = _config.temp_directory_path + "violation_repairer/";
  /////////////////////////////////////////////
  // **********        RT         ********** //
  RTUtil::createDir(_config.temp_directory_path);
  RTUtil::createDirByFile(_config.log_file_path);
  // **********    DataManager    ********** //
  RTUtil::createDir(_config.dm_temp_directory_path);
  // **********  DetailedRouter   ********** //
  RTUtil::createDir(_config.dr_temp_directory_path);
  // **********    GDSPlotter     ********** //
  RTUtil::createDir(_config.gp_temp_directory_path);
  // **********   GlobalRouter    ********** //
  RTUtil::createDir(_config.gr_temp_directory_path);
  // **********   PinAccessor     ********** //
  RTUtil::createDir(_config.pa_temp_directory_path);
  // **********   ResourceAllocator     ********** //
  RTUtil::createDir(_config.ra_temp_directory_path);
  // **********   TrackAssigner   ********** //
  RTUtil::createDir(_config.ta_temp_directory_path);
  // **********  UniversalRouter  ********** //
  RTUtil::createDir(_config.ur_temp_directory_path);
  // ********** ViolationRepairer ********** //
  RTUtil::createDir(_config.vr_temp_directory_path);
  /////////////////////////////////////////////
}

void DataManager::buildDatabase()
{
  buildGCellAxis();
  buildDie();
  buildLayerList();
  buildLayerViaMasterList();
  buildBlockageList();
  buildNetList();
  updateHelper();
}

void DataManager::buildGCellAxis()
{
  makeGCellAxis();
  checkGCellAxis();
}

void DataManager::makeGCellAxis()
{
  GCellAxis& gcell_axis = _database.get_gcell_axis();

  irt_int proposed_interval = getProposedInterval();
  std::vector<irt_int> x_gcell_scale_list = makeGCellScaleList(Direction::kVertical, proposed_interval);
  gcell_axis.set_x_grid_list(makeGCellGridList(x_gcell_scale_list));
  std::vector<irt_int> y_gcell_scale_list = makeGCellScaleList(Direction::kHorizontal, proposed_interval);
  gcell_axis.set_y_grid_list(makeGCellGridList(y_gcell_scale_list));
}

irt_int DataManager::getProposedInterval()
{
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();

  std::map<irt_int, irt_int> pitch_count_map;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    pitch_count_map[routing_layer.getPreferTrackGrid().get_step_length()]++;
  }
  irt_int ref_pitch = -1;
  irt_int max_count = INT32_MIN;
  for (auto [pitch, count] : pitch_count_map) {
    if (count > max_count) {
      max_count = count;
      ref_pitch = pitch;
    }
  }
  if (ref_pitch == -1) {
    LOG_INST.error(Loc::current(), "The ref_pitch is -1!");
  }
  return (15 * ref_pitch);
}

std::vector<irt_int> DataManager::makeGCellScaleList(Direction direction, irt_int proposed_gcell_interval)
{
  Die& die = _database.get_die();
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();

  irt_int start_gcell_scale = (direction == Direction::kVertical ? die.get_real_lb_x() : die.get_real_lb_y());
  irt_int end_gcell_scale = (direction == Direction::kVertical ? die.get_real_rt_x() : die.get_real_rt_y());

  std::set<irt_int> base_layer_idx_set;
  std::map<irt_int, std::set<irt_int>> scale_layer_map;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    base_layer_idx_set.insert(routing_layer.get_layer_idx());

    TrackGrid track_grid = (direction == Direction::kVertical ? routing_layer.getXTrackGrid() : routing_layer.getYTrackGrid());
    irt_int track_scale = track_grid.get_start_line();
    irt_int step_num = track_grid.get_step_num();
    while (step_num--) {
      scale_layer_map[track_scale].insert(routing_layer.get_layer_idx());
      track_scale += track_grid.get_step_length();
      if (track_scale > end_gcell_scale) {
        break;
      }
    }
  }
  std::vector<irt_int> gcell_scale_list = {start_gcell_scale};
  std::set<irt_int> curr_layer_idx_set;
  auto iter = scale_layer_map.begin();
  while (true) {
    irt_int track_scale = iter->first;
    curr_layer_idx_set.insert(iter->second.begin(), iter->second.end());
    iter++;
    if (iter == scale_layer_map.end()) {
      if (base_layer_idx_set != curr_layer_idx_set) {
        gcell_scale_list.pop_back();
      }
      gcell_scale_list.push_back(end_gcell_scale);
      break;
    }
    if (track_scale - gcell_scale_list.back() < proposed_gcell_interval) {
      continue;
    }
    if (base_layer_idx_set != curr_layer_idx_set) {
      continue;
    }
    curr_layer_idx_set.clear();
    gcell_scale_list.push_back((track_scale + iter->first) / 2);
  }
  return gcell_scale_list;
}

std::vector<GCellGrid> DataManager::makeGCellGridList(std::vector<irt_int>& gcell_scale_list)
{
  std::vector<GCellGrid> gcell_grid_list;

  for (size_t i = 1; i < gcell_scale_list.size(); i++) {
    irt_int pre_scale = gcell_scale_list[i - 1];
    irt_int curr_scale = gcell_scale_list[i];

    GCellGrid gcell_grid;
    gcell_grid.set_start_line(pre_scale);
    gcell_grid.set_step_length(curr_scale - pre_scale);
    gcell_grid.set_step_num(1);
    gcell_grid.set_end_line(curr_scale);
    gcell_grid_list.push_back(gcell_grid);
  }
  // merge
  RTUtil::merge(gcell_grid_list, [](GCellGrid& sentry, GCellGrid& soldier) {
    if (sentry.get_step_length() != soldier.get_step_length()) {
      return false;
    }
    sentry.set_start_line(std::min(sentry.get_start_line(), soldier.get_start_line()));
    sentry.set_step_num(sentry.get_step_num() + 1);
    sentry.set_end_line(std::max(sentry.get_end_line(), soldier.get_end_line()));
    return true;
  });

  return gcell_grid_list;
}

void DataManager::checkGCellAxis()
{
  GCellAxis& gcell_axis = _database.get_gcell_axis();
  std::vector<GCellGrid>& x_grid_list = gcell_axis.get_x_grid_list();
  std::vector<GCellGrid>& y_grid_list = gcell_axis.get_y_grid_list();

  if (x_grid_list.empty() || y_grid_list.empty()) {
    LOG_INST.error(Loc::current(), "The gcell grid list is empty!");
  }
  for (size_t i = 0; i < x_grid_list.size(); i++) {
    if (x_grid_list[i].get_step_length() <= 0) {
      LOG_INST.error(Loc::current(), "The step length of x grid '", x_grid_list[i].get_step_length(), "' is wrong!");
    }
  }
  for (size_t i = 0; i < y_grid_list.size(); i++) {
    if (y_grid_list[i].get_step_length() <= 0) {
      LOG_INST.error(Loc::current(), "The step length of y grid '", y_grid_list[i].get_step_length(), "' is wrong!");
    }
  }
  for (size_t i = 1; i < x_grid_list.size(); i++) {
    if (x_grid_list[i - 1].get_end_line() < x_grid_list[i].get_start_line()) {
      LOG_INST.error(Loc::current(), "The x grid with gap '", x_grid_list[i - 1].get_end_line(), " < ", x_grid_list[i].get_start_line(),
                     "'!");
    } else if (x_grid_list[i - 1].get_end_line() > x_grid_list[i].get_start_line()) {
      LOG_INST.error(Loc::current(), "The x grid with overlapping '", x_grid_list[i - 1].get_end_line(), " < ",
                     x_grid_list[i].get_start_line(), "'!");
    }
  }
  for (size_t i = 1; i < y_grid_list.size(); i++) {
    if (y_grid_list[i - 1].get_end_line() < y_grid_list[i].get_start_line()) {
      LOG_INST.error(Loc::current(), "The y grid with gap '", y_grid_list[i - 1].get_end_line(), " < ", y_grid_list[i].get_start_line(),
                     "'!");
    } else if (y_grid_list[i - 1].get_end_line() > y_grid_list[i].get_start_line()) {
      LOG_INST.error(Loc::current(), "The y grid with overlapping '", y_grid_list[i - 1].get_end_line(), " > ",
                     y_grid_list[i].get_start_line(), "'!");
    }
  }
}

void DataManager::buildDie()
{
  makeDie();
  checkDie();
}

void DataManager::makeDie()
{
  Die& die = _database.get_die();
  GCellAxis& gcell_axis = _database.get_gcell_axis();
  die.set_grid_rect(RTUtil::getOpenGridRect(die.get_real_rect(), gcell_axis));
}

void DataManager::checkDie()
{
  Die& die = _database.get_die();

  if (die.get_real_lb_x() < 0 || die.get_real_lb_y() < 0 || die.get_real_rt_x() < 0 || die.get_real_rt_y() < 0) {
    LOG_INST.error(Loc::current(), "The die '(", die.get_real_lb_x(), " , ", die.get_real_lb_y(), ") - (", die.get_real_rt_x(), " , ",
                   die.get_real_rt_y(), ")' is wrong!");
  }
  if ((die.get_real_rt_x() <= die.get_real_lb_x()) || (die.get_real_rt_y() <= die.get_real_lb_y())) {
    LOG_INST.error(Loc::current(), "The die '(", die.get_real_lb_x(), " , ", die.get_real_lb_y(), ") - (", die.get_real_rt_x(), " , ",
                   die.get_real_rt_y(), ")' is wrong!");
  }
}

void DataManager::buildLayerList()
{
  transLayerList();
  makeLayerList();
  checkLayerList();
}

void DataManager::transLayerList()
{
  for (RoutingLayer& routing_layer : _database.get_routing_layer_list()) {
    routing_layer.set_layer_idx(_helper.wrapIDBRoutingLayerIdxToRT(routing_layer.get_layer_idx()));
  }
  for (CutLayer& cut_layer_list : _database.get_cut_layer_list()) {
    cut_layer_list.set_layer_idx(_helper.wrapIDBCutLayerIdxToRT(cut_layer_list.get_layer_idx()));
  }
}

void DataManager::makeLayerList()
{
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();

  for (RoutingLayer& routing_layer : routing_layer_list) {
    TrackGrid& x_track_grid = routing_layer.get_track_axis().get_x_track_grid();
    x_track_grid.set_end_line(x_track_grid.get_start_line() + x_track_grid.get_step_length() * x_track_grid.get_step_num());
    TrackGrid& y_track_grid = routing_layer.get_track_axis().get_y_track_grid();
    y_track_grid.set_end_line(y_track_grid.get_start_line() + y_track_grid.get_step_length() * y_track_grid.get_step_num());
  }
}

void DataManager::checkLayerList()
{
  Die& die = _database.get_die();
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = _database.get_cut_layer_list();

  if (routing_layer_list.empty()) {
    LOG_INST.error(Loc::current(), "The routing_layer_list is empty!");
  }
  if (cut_layer_list.empty()) {
    LOG_INST.error(Loc::current(), "The cut_layer_list is empty!");
  }

  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::string& layer_name = routing_layer.get_layer_name();
    if (routing_layer.get_direction() == Direction::kNone) {
      LOG_INST.error(Loc::current(), "The layer '", layer_name, "' direction is none!");
    }
    TrackGrid& x_track_grid = routing_layer.get_track_axis().get_x_track_grid();
    if (x_track_grid.get_start_line() < die.get_real_lb_x() || die.get_real_rt_x() < x_track_grid.get_end_line()) {
      LOG_INST.warning(Loc::current(), "The layer ", routing_layer.get_layer_name(), " x_track_grid outside the die!");
    }
    if (x_track_grid.get_step_length() <= 0) {
      LOG_INST.error(Loc::current(), "The layer '", layer_name, "' x_track_grid step length '", x_track_grid.get_step_length(),
                     "' is wrong!");
    }
    TrackGrid& y_track_grid = routing_layer.get_track_axis().get_y_track_grid();
    if (y_track_grid.get_start_line() < die.get_real_lb_y() || die.get_real_rt_y() < y_track_grid.get_end_line()) {
      LOG_INST.warning(Loc::current(), "The layer ", routing_layer.get_layer_name(), " y_track_grid outside the die!");
    }
    if (y_track_grid.get_step_length() <= 0) {
      LOG_INST.error(Loc::current(), "The layer '", layer_name, "' y_track_grid step length '", y_track_grid.get_step_length(),
                     "' is wrong!");
    }
    SpacingTable& spacing_table = routing_layer.get_spacing_table();
    if (spacing_table.get_width_list().empty()) {
      LOG_INST.error(Loc::current(), "The layer '", layer_name, "' spacing width list is empty!");
    }
    for (irt_int width : spacing_table.get_width_list()) {
      if (width < 0) {
        LOG_INST.error(Loc::current(), "The layer '", layer_name, "' width < 0!");
      }
    }
    for (irt_int parallel_length : spacing_table.get_parallel_length_list()) {
      if (parallel_length < 0) {
        LOG_INST.error(Loc::current(), "The layer '", layer_name, "' parallel_length < 0!");
      }
    }
    GridMap<irt_int>& width_parallel_length_map = spacing_table.get_width_parallel_length_map();
    for (irt_int width_idx = 0; width_idx < width_parallel_length_map.get_x_size(); width_idx++) {
      for (irt_int parallel_length_idx = 0; parallel_length_idx < width_parallel_length_map.get_y_size(); parallel_length_idx++) {
        if (width_parallel_length_map[width_idx][parallel_length_idx] < 0) {
          LOG_INST.error(Loc::current(), "The layer '", layer_name, "' spacing < 0!");
        }
      }
    }
  }
}

void DataManager::buildLayerViaMasterList()
{
  transLayerViaMasterList();
  makeLayerViaMasterList();
}

void DataManager::transLayerViaMasterList()
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _database.get_layer_via_master_list();

  for (std::vector<ViaMaster>& via_master_list : layer_via_master_list) {
    for (ViaMaster& via_master : via_master_list) {
      // above
      LayerRect& above_enclosure = via_master.get_above_enclosure();
      above_enclosure.set_layer_idx(_helper.wrapIDBRoutingLayerIdxToRT(above_enclosure.get_layer_idx()));
      // below
      LayerRect& below_enclosure = via_master.get_below_enclosure();
      below_enclosure.set_layer_idx(_helper.wrapIDBRoutingLayerIdxToRT(below_enclosure.get_layer_idx()));
      // cut
      via_master.set_cut_layer_idx(_helper.wrapIDBCutLayerIdxToRT(via_master.get_cut_layer_idx()));
    }
  }
}

void DataManager::makeLayerViaMasterList()
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _database.get_layer_via_master_list();
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();

  std::vector<ViaMaster> first_via_master_list;
  for (ViaMaster& via_master : layer_via_master_list.front()) {
    irt_int below_layer_idx = via_master.get_below_enclosure().get_layer_idx();
    if (below_layer_idx == 0) {
      first_via_master_list.push_back(via_master);
    } else {
      layer_via_master_list[below_layer_idx].push_back(via_master);
    }
  }
  layer_via_master_list[0] = first_via_master_list;

  for (size_t layer_idx = 0; layer_idx < layer_via_master_list.size(); layer_idx++) {
    std::vector<ViaMaster>& via_master_list = layer_via_master_list[layer_idx];
    for (ViaMaster& via_master : via_master_list) {
      // above
      LayerRect& above_enclosure = via_master.get_above_enclosure();
      Direction above_layer_direction = routing_layer_list[above_enclosure.get_layer_idx()].get_direction();
      via_master.set_above_direction(above_enclosure.getRectDirection(above_layer_direction));
      // below
      LayerRect& below_enclosure = via_master.get_below_enclosure();
      Direction below_layer_direction = routing_layer_list[below_enclosure.get_layer_idx()].get_direction();
      via_master.set_below_direction(below_enclosure.getRectDirection(below_layer_direction));
    }
    std::sort(via_master_list.begin(), via_master_list.end(),
              [&](ViaMaster& via_master1, ViaMaster& via_master2) { return sortByMultiLevel(via_master1, via_master2); });

    std::sort(via_master_list.begin(), via_master_list.end(), [&](ViaMaster& a, ViaMaster& b) {
      std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();

      LayerRect& a_above = a.get_above_enclosure();
      LayerRect& a_below = a.get_below_enclosure();
      LayerRect& b_above = b.get_above_enclosure();
      LayerRect& b_below = b.get_below_enclosure();
      // 方向
      Direction a_above_layer_direction = routing_layer_list[a_above.get_layer_idx()].get_direction();
      Direction b_above_layer_direction = routing_layer_list[b_above.get_layer_idx()].get_direction();
      if (a.get_above_direction() == a_above_layer_direction && b.get_above_direction() != b_above_layer_direction) {
        return true;
      } else if (a.get_above_direction() != a_above_layer_direction && b.get_above_direction() == b_above_layer_direction) {
        return false;
      }
      Direction a_below_layer_direction = routing_layer_list[a_below.get_layer_idx()].get_direction();
      Direction b_below_layer_direction = routing_layer_list[b_below.get_layer_idx()].get_direction();
      if (a.get_below_direction() == a_below_layer_direction && b.get_below_direction() != b_below_layer_direction) {
        return true;
      } else if (a.get_below_direction() != a_below_layer_direction && b.get_below_direction() == b_below_layer_direction) {
        return false;
      }
      // 宽度
      if (a_above.getWidth() != b_above.getWidth()) {
        return a_above.getWidth() < b_above.getWidth();
      }
      if (a_below.getWidth() != b_below.getWidth()) {
        return a_below.getWidth() < b_below.getWidth();
      }
      // 对称
      irt_int a_above_center_diff = std::abs(a_above.get_lb_x() + a_above.get_rt_x());
      irt_int b_above_center_diff = std::abs(b_above.get_lb_x() + b_above.get_rt_x());
      if (a_above_center_diff != b_above_center_diff) {
        return a_above_center_diff < b_above_center_diff;
      }
      irt_int a_below_center_diff = std::abs(a_below.get_lb_x() + a_below.get_rt_x());
      irt_int b_below_center_diff = std::abs(b_below.get_lb_x() + b_below.get_rt_x());
      if (a_below_center_diff != b_below_center_diff) {
        return a_below_center_diff < b_below_center_diff;
      }
      // 长度
      if (a_above.getLength() != b_above.getLength()) {
        return a_above.getLength() < b_above.getLength();
      } else {
        return a_below.getLength() < b_below.getLength();
      }
    });
    for (size_t i = 0; i < via_master_list.size(); i++) {
      via_master_list[i].set_via_idx(std::make_pair(layer_idx, i));
    }
  }
}

bool DataManager::sortByMultiLevel(ViaMaster& via_master1, ViaMaster& via_master2)
{
  SortStatus sort_status = SortStatus::kNone;

  sort_status = sortByWidthASC(via_master1, via_master2);
  if (sort_status == SortStatus::kTrue) {
    return true;
  } else if (sort_status == SortStatus::kFalse) {
    return false;
  }
  sort_status = sortByLayerDirectionPriority(via_master1, via_master2);
  if (sort_status == SortStatus::kTrue) {
    return true;
  } else if (sort_status == SortStatus::kFalse) {
    return false;
  }
  sort_status = sortByLengthASC(via_master1, via_master2);
  if (sort_status == SortStatus::kTrue) {
    return true;
  } else if (sort_status == SortStatus::kFalse) {
    return false;
  }
  sort_status = sortBySymmetryPriority(via_master1, via_master2);
  if (sort_status == SortStatus::kTrue) {
    return true;
  } else if (sort_status == SortStatus::kFalse) {
    return false;
  }
  return false;
}

// 宽度升序
SortStatus DataManager::sortByWidthASC(ViaMaster& via_master1, ViaMaster& via_master2)
{
  LayerRect& via_master1_above = via_master1.get_above_enclosure();
  LayerRect& via_master1_below = via_master1.get_below_enclosure();
  LayerRect& via_master2_above = via_master2.get_above_enclosure();
  LayerRect& via_master2_below = via_master2.get_below_enclosure();

  if (via_master1_above.getWidth() < via_master2_above.getWidth()) {
    return SortStatus::kTrue;
  } else if (via_master1_above.getWidth() > via_master2_above.getWidth()) {
    return SortStatus::kFalse;
  } else {
    if (via_master1_below.getWidth() < via_master2_below.getWidth()) {
      return SortStatus::kTrue;
    } else if (via_master1_below.getWidth() > via_master2_below.getWidth()) {
      return SortStatus::kFalse;
    } else {
      return SortStatus::kEqual;
    }
  }
}

// 层方向优先
SortStatus DataManager::sortByLayerDirectionPriority(ViaMaster& via_master1, ViaMaster& via_master2)
{
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();

  Direction above_layer_direction = routing_layer_list[via_master1.get_above_enclosure().get_layer_idx()].get_direction();
  Direction below_layer_direction = routing_layer_list[via_master1.get_below_enclosure().get_layer_idx()].get_direction();

  if (via_master1.get_above_direction() == above_layer_direction && via_master2.get_above_direction() != above_layer_direction) {
    return SortStatus::kTrue;
  } else if (via_master1.get_above_direction() != above_layer_direction && via_master2.get_above_direction() == above_layer_direction) {
    return SortStatus::kFalse;
  } else {
    if (via_master1.get_below_direction() == below_layer_direction && via_master2.get_below_direction() != below_layer_direction) {
      return SortStatus::kTrue;
    } else if (via_master1.get_below_direction() != below_layer_direction && via_master2.get_below_direction() == below_layer_direction) {
      return SortStatus::kFalse;
    } else {
      return SortStatus::kEqual;
    }
  }
}

// 长度升序
SortStatus DataManager::sortByLengthASC(ViaMaster& via_master1, ViaMaster& via_master2)
{
  LayerRect& via_master1_above = via_master1.get_above_enclosure();
  LayerRect& via_master1_below = via_master1.get_below_enclosure();
  LayerRect& via_master2_above = via_master2.get_above_enclosure();
  LayerRect& via_master2_below = via_master2.get_below_enclosure();

  if (via_master1_above.getLength() < via_master2_above.getLength()) {
    return SortStatus::kTrue;
  } else if (via_master1_above.getLength() > via_master2_above.getLength()) {
    return SortStatus::kFalse;
  } else {
    if (via_master1_below.getLength() < via_master2_below.getLength()) {
      return SortStatus::kTrue;
    } else if (via_master1_below.getLength() > via_master2_below.getLength()) {
      return SortStatus::kFalse;
    } else {
      return SortStatus::kEqual;
    }
  }
}

// 对称优先
SortStatus DataManager::sortBySymmetryPriority(ViaMaster& via_master1, ViaMaster& via_master2)
{
  LayerRect& via_master1_above = via_master1.get_above_enclosure();
  LayerRect& via_master1_below = via_master1.get_below_enclosure();
  LayerRect& via_master2_above = via_master2.get_above_enclosure();
  LayerRect& via_master2_below = via_master2.get_below_enclosure();

  // via_master的lb为负数，rt为正数
  irt_int via_master1_above_center_diff = std::abs(via_master1_above.get_lb_x() + via_master1_above.get_rt_x());
  irt_int via_master2_above_center_diff = std::abs(via_master2_above.get_lb_x() + via_master2_above.get_rt_x());
  irt_int via_master1_below_center_diff = std::abs(via_master1_below.get_lb_x() + via_master1_below.get_rt_x());
  irt_int via_master2_below_center_diff = std::abs(via_master2_below.get_lb_x() + via_master2_below.get_rt_x());
  if (via_master1_above_center_diff < via_master2_above_center_diff) {
    return SortStatus::kTrue;
  } else if (via_master1_above_center_diff > via_master2_above_center_diff) {
    return SortStatus::kFalse;
  } else {
    if (via_master1_below_center_diff < via_master2_below_center_diff) {
      return SortStatus::kTrue;
    } else if (via_master1_below_center_diff > via_master2_below_center_diff) {
      return SortStatus::kFalse;
    } else {
      return SortStatus::kEqual;
    }
  }
}

void DataManager::buildBlockageList()
{
  transBlockageList();
  makeBlockageList();
  checkBlockageList();
}

void DataManager::transBlockageList()
{
  std::vector<Blockage>& routing_blockage_list = _database.get_routing_blockage_list();
  std::vector<Blockage>& cut_blockage_list = _database.get_cut_blockage_list();

  for (Blockage& blockage : routing_blockage_list) {
    blockage.set_layer_idx(_helper.wrapIDBRoutingLayerIdxToRT(blockage.get_layer_idx()));
  }
  for (Blockage& blockage : cut_blockage_list) {
    blockage.set_layer_idx(_helper.wrapIDBCutLayerIdxToRT(blockage.get_layer_idx()));
  }
}

void DataManager::makeBlockageList()
{
  std::vector<Blockage>& routing_blockage_list = _database.get_routing_blockage_list();
  std::vector<Blockage>& cut_blockage_list = _database.get_cut_blockage_list();
  GCellAxis& gcell_axis = _database.get_gcell_axis();

  for (Blockage& routing_blockage : routing_blockage_list) {
    routing_blockage.set_grid_rect(RTUtil::getClosedGridRect(routing_blockage.get_real_rect(), gcell_axis));
  }
  for (Blockage& cut_blockage : cut_blockage_list) {
    cut_blockage.set_grid_rect(RTUtil::getClosedGridRect(cut_blockage.get_real_rect(), gcell_axis));
  }
}

void DataManager::checkBlockageList()
{
  Die& die = _database.get_die();
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();
  std::vector<Blockage>& routing_blockage_list = _database.get_routing_blockage_list();
  std::vector<Blockage>& cut_blockage_list = _database.get_cut_blockage_list();

  for (Blockage& blockage : routing_blockage_list) {
    if (blockage.get_real_lb_x() < die.get_real_lb_x() || blockage.get_real_lb_y() < die.get_real_lb_y()
        || die.get_real_rt_x() < blockage.get_real_rt_x() || die.get_real_rt_y() < blockage.get_real_rt_y()) {
      // log
      LOG_INST.error(Loc::current(), "The blockage '(", blockage.get_real_lb_x(), " , ", blockage.get_real_lb_y(), ") - (",
                     blockage.get_real_rt_x(), " , ", blockage.get_real_rt_y(), ") ",
                     routing_layer_list[blockage.get_layer_idx()].get_layer_name(), "' is wrong! Die '(", die.get_real_lb_x(), " , ",
                     die.get_real_lb_y(), ") - (", die.get_real_rt_x(), " , ", die.get_real_rt_y(), ")'");
    }
  }
  for (Blockage& blockage : cut_blockage_list) {
    if (blockage.get_real_lb_x() < die.get_real_lb_x() || blockage.get_real_lb_y() < die.get_real_lb_y()
        || die.get_real_rt_x() < blockage.get_real_rt_x() || die.get_real_rt_y() < blockage.get_real_rt_y()) {
      // log
      LOG_INST.error(Loc::current(), "The blockage '(", blockage.get_real_lb_x(), " , ", blockage.get_real_lb_y(), ") - (",
                     blockage.get_real_rt_x(), " , ", blockage.get_real_rt_y(), ") ",
                     routing_layer_list[blockage.get_layer_idx()].get_layer_name(), "' is wrong! Die '(", die.get_real_lb_x(), " , ",
                     die.get_real_lb_y(), ") - (", die.get_real_rt_x(), " , ", die.get_real_rt_y(), ")'");
    }
  }
}

void DataManager::buildNetList()
{
  std::vector<Net>& net_list = _database.get_net_list();

  for (size_t net_idx = 0; net_idx < net_list.size(); net_idx++) {
    Net& net = net_list[net_idx];
    net.set_net_idx(static_cast<irt_int>(net_idx));
    buildPinList(net);
    buildDrivingPin(net);
  }
}

void DataManager::buildPinList(Net& net)
{
  transPinList(net);
  makePinList(net);
  checkPinList(net);
}

void DataManager::transPinList(Net& net)
{
  for (Pin& pin : net.get_pin_list()) {
    for (EXTLayerRect& routing_shape : pin.get_routing_shape_list()) {
      routing_shape.set_layer_idx(_helper.wrapIDBRoutingLayerIdxToRT(routing_shape.get_layer_idx()));
    }
    for (EXTLayerRect& cut_shape : pin.get_cut_shape_list()) {
      cut_shape.set_layer_idx(_helper.wrapIDBCutLayerIdxToRT(cut_shape.get_layer_idx()));
    }
  }
}

void DataManager::makePinList(Net& net)
{
  std::vector<Pin>& pin_list = net.get_pin_list();
  GCellAxis& gcell_axis = _database.get_gcell_axis();

  for (size_t pin_idx = 0; pin_idx < pin_list.size(); pin_idx++) {
    Pin& pin = pin_list[pin_idx];
    pin.set_pin_idx(static_cast<irt_int>(pin_idx));
    for (EXTLayerRect& routing_shape : pin.get_routing_shape_list()) {
      routing_shape.set_grid_rect(RTUtil::getClosedGridRect(routing_shape.get_real_rect(), gcell_axis));
    }
    for (EXTLayerRect& cut_shape : pin.get_cut_shape_list()) {
      cut_shape.set_grid_rect(RTUtil::getClosedGridRect(cut_shape.get_real_rect(), gcell_axis));
    }
  }
}

void DataManager::checkPinList(Net& net)
{
  Die& die = _database.get_die();
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = _database.get_cut_layer_list();

  for (Pin& pin : net.get_pin_list()) {
    for (EXTLayerRect& routing_shape : pin.get_routing_shape_list()) {
      if (routing_shape.get_real_lb_x() < die.get_real_lb_x() || routing_shape.get_real_lb_y() < die.get_real_lb_y()
          || die.get_real_rt_x() < routing_shape.get_real_rt_x() || die.get_real_rt_y() < routing_shape.get_real_rt_y()) {
        LOG_INST.error(Loc::current(), "The port '(", routing_shape.get_real_lb_x(), " , ", routing_shape.get_real_lb_y(), ") - (",
                       routing_shape.get_real_rt_x(), " , ", routing_shape.get_real_rt_y(), ") ",
                       routing_layer_list[routing_shape.get_layer_idx()].get_layer_name(), "' is wrong! Die '(", die.get_real_lb_x(), " , ",
                       die.get_real_lb_y(), ") - (", die.get_real_rt_x(), " , ", die.get_real_rt_y(), ")'");
      }
    }
    for (EXTLayerRect& cut_shape : pin.get_cut_shape_list()) {
      if (cut_shape.get_real_lb_x() < die.get_real_lb_x() || cut_shape.get_real_lb_y() < die.get_real_lb_y()
          || die.get_real_rt_x() < cut_shape.get_real_rt_x() || die.get_real_rt_y() < cut_shape.get_real_rt_y()) {
        LOG_INST.error(Loc::current(), "The port '(", cut_shape.get_real_lb_x(), " , ", cut_shape.get_real_lb_y(), ") - (",
                       cut_shape.get_real_rt_x(), " , ", cut_shape.get_real_rt_y(), ") ",
                       cut_layer_list[cut_shape.get_layer_idx()].get_layer_name(), "' is wrong! Die '(", die.get_real_lb_x(), " , ",
                       die.get_real_lb_y(), ") - (", die.get_real_rt_x(), " , ", die.get_real_rt_y(), ")'");
      }
    }
  }
}

void DataManager::buildDrivingPin(Net& net)
{
  std::vector<Pin>& pin_list = net.get_pin_list();
  for (size_t i = 0; i < pin_list.size(); i++) {
    Pin& pin = pin_list[i];
    if (net.get_driving_pin().get_pin_name() != pin.get_pin_name()) {
      continue;
    }
    net.set_driving_pin(pin);
    return;
  }
  LOG_INST.error(Loc::current(), "Unable to find a driving pin!");
}

void DataManager::updateHelper()
{
  std::map<std::string, std::pair<irt_int, irt_int>>& via_name_to_idx_map = _helper.get_via_name_to_idx_map();

  for (std::vector<ViaMaster>& via_master_list : _database.get_layer_via_master_list()) {
    for (ViaMaster& via_master : via_master_list) {
      via_name_to_idx_map[via_master.get_via_name()] = via_master.get_via_idx();
    }
  }
}

void DataManager::printConfig()
{
  omp_set_num_threads(std::max(_config.thread_number, 1));
  LOG_INST.setLogLevel(_config.log_level);
  LOG_INST.openLogFileStream(_config.log_file_path);
  /////////////////////////////////////////////
  // **********        RT         ********** //
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(0), "RT_CONFIG_INPUT");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "temp_directory_path");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _config.temp_directory_path);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "log_level");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _config.log_level);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "thread_number");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _config.thread_number);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "bottom_routing_layer");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _config.bottom_routing_layer);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "top_routing_layer");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _config.top_routing_layer);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "layer_utilization_ratio");
  for (auto [layer_name, utilization_ratio] : _config.layer_utilization_ratio) {
    LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), layer_name, ": ", utilization_ratio);
  }
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "enable_output_gds_files");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _config.enable_output_gds_files);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "resource_allocate_initial_penalty");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _config.resource_allocate_initial_penalty);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "resource_allocate_penalty_drop_rate");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _config.resource_allocate_penalty_drop_rate);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "resource_allocate_outer_iter_num");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _config.resource_allocate_outer_iter_num);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "resource_allocate_inner_iter_num");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _config.resource_allocate_inner_iter_num);
  // **********        RT         ********** //
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(0), "RT_CONFIG_BUILD");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "log_file_path");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _config.log_file_path);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "bottom_routing_layer_idx");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _config.bottom_routing_layer_idx);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "top_routing_layer_idx");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _config.top_routing_layer_idx);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "layer_idx_utilization_ratio");
  for (auto [layer_idx, utilization_ratio] : _config.layer_idx_utilization_ratio) {
    LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), layer_idx, ": ", utilization_ratio);
  }
  // **********    DataManager    ********** //
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "DataManager");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), "dm_temp_directory_path");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(3), _config.dm_temp_directory_path);
  // **********  DetailedRouter   ********** //
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "DetailedRouter");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), "dr_temp_directory_path");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(3), _config.dr_temp_directory_path);
  // **********    GDSPlotter     ********** //
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "GDSPlotter");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), "gp_temp_directory_path");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(3), _config.gp_temp_directory_path);
  // **********   GlobalRouter    ********** //
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "GlobalRouter");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), "gr_temp_directory_path");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(3), _config.gr_temp_directory_path);
  // **********   PinAccessor     ********** //
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "PinAccessor");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), "pa_temp_directory_path");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(3), _config.pa_temp_directory_path);
  // **********   ResourceAllocator   ********** //
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "ResourceAllocator");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), "ra_temp_directory_path");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(3), _config.ra_temp_directory_path);
  // **********   TrackAssigner   ********** //
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "TrackAssigner");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), "ta_temp_directory_path");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(3), _config.ta_temp_directory_path);
  // **********  UniversalRouter  ********** //
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "UniversalRouter");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), "ur_temp_directory_path");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(3), _config.ur_temp_directory_path);
  // ********** ViolationRepairer ********** //
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "ViolationRepairer");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), "vr_temp_directory_path");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(3), _config.vr_temp_directory_path);
  /////////////////////////////////////////////
  sleep(2);
}

void DataManager::printDatabase()
{
  ////////////////////////////////////////////////
  // ********** RT ********** //
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(0), "RT_DATABASE");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "design_name");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _helper.get_design_name());
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "lef_file_path_list");
  for (std::string& lef_file_path : _helper.get_lef_file_path_list()) {
    LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), lef_file_path);
  }
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "def_file_path");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _helper.get_def_file_path());
  // ********** MicronDBU ********** //
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "micron_dbu");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _database.get_micron_dbu());
  // ********** GCellAxis ********** //
  GCellAxis& gcell_axis = _database.get_gcell_axis();
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "gcell_axis");
  std::vector<GCellGrid>& x_grid_list = gcell_axis.get_x_grid_list();
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), "x_grid_list");
  for (GCellGrid& x_grid : x_grid_list) {
    LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(3), "start:", x_grid.get_start_line(), " step_length:", x_grid.get_step_length(),
                  " step_num:", x_grid.get_step_num(), " end:", x_grid.get_end_line());
  }
  std::vector<GCellGrid>& y_grid_list = gcell_axis.get_y_grid_list();
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), "y_grid_list");
  for (GCellGrid& y_grid : y_grid_list) {
    LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(3), "start:", y_grid.get_start_line(), " step_length:", y_grid.get_step_length(),
                  " step_num:", y_grid.get_step_num(), " end:", y_grid.get_end_line());
  }
  // ********** Die ********** //
  Die& die = _database.get_die();
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "die");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), "(", die.get_real_lb_x(), ",", die.get_real_lb_y(), ")-(", die.get_real_rt_x(),
                ",", die.get_real_rt_y(), ")");
  // ********** RoutingLayer ********** //
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "routing_layer_num");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), routing_layer_list.size());
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "routing_layer");
  for (RoutingLayer& routing_layer : routing_layer_list) {
    LOG_INST.info(
        Loc::current(), RTUtil::getSpaceByTabNum(2), "idx:", routing_layer.get_layer_idx(), " order:", routing_layer.get_layer_order(),
        " name:", routing_layer.get_layer_name(), " min_width:", routing_layer.get_min_width(), " min_area:", routing_layer.get_min_area(),
        " direction:", GetDirectionName()(routing_layer.get_direction()), " pitch:", routing_layer.getPreferTrackGrid().get_step_length());
  }
  // ********** CutLayer ********** //
  std::vector<CutLayer>& cut_layer_list = _database.get_cut_layer_list();
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "cut_layer_num");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), cut_layer_list.size());
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "cut_layer");
  for (CutLayer& cut_layer : cut_layer_list) {
    LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), "idx:", cut_layer.get_layer_idx(), " order:", cut_layer.get_layer_order(),
                  " name:", cut_layer.get_layer_name());
  }
  // ********** ViaMaster ********** //
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _database.get_layer_via_master_list();
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "layer_via_master_list");
  for (size_t below_layer_idx = 0; below_layer_idx < layer_via_master_list.size(); below_layer_idx++) {
    std::string via_master_name_string = (routing_layer_list[below_layer_idx].get_layer_name() + ": ");
    for (ViaMaster& via_master : layer_via_master_list[below_layer_idx]) {
      via_master_name_string += (via_master.get_via_name() + " ");
    }
    LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), via_master_name_string);
  }
  // ********** Blockage ********** //
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "routing_blockage_num");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _database.get_routing_blockage_list().size());
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "cut_blockage_num");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _database.get_cut_blockage_list().size());
  // ********** Net ********** //
  std::vector<Net>& net_list = _database.get_net_list();
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "net_num");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), net_list.size());
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "pin_num_ratio");

  size_t pin_num_upper_limit = 100;
  std::map<size_t, size_t> pin_net_map;
  for (Net& net : net_list) {
    pin_net_map[std::min(net.get_pin_list().size(), pin_num_upper_limit)]++;
  }
  for (auto [pin_num, net_num] : pin_net_map) {
    std::string head_info = "net with ";
    if (pin_num == pin_num_upper_limit) {
      head_info += ">=";
    }
    LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), head_info, pin_num, " pins: ", net_num, "(",
                  RTUtil::getPercentage(net_num, net_list.size()), "%)");
  }
  // ******************** //
  sleep(2);
  ////////////////////////////////////////////////
}

void DataManager::saveStageResult(Stage stage)
{
  Monitor monitor;

  nlohmann::json all_json;
  std::string current_stage = GetStageName()(stage);
  LOG_INST.info(Loc::current(), "The ", current_stage, " result is being saved...");
  saveHeadInfo(all_json, current_stage);

  for (Net net : _database.get_net_list()) {
    nlohmann::json& net_json = all_json["net_list"][net.get_net_name()];
    saveStageNetResult(net_json, net);
  }
  std::string json_file_path = _config.dm_temp_directory_path + GetStageName()(stage) + ".json";
  std::ofstream* json_stream = RTUtil::getOutputFileStream(json_file_path);
  (*json_stream) << all_json << std::endl;
  RTUtil::closeFileStream(json_stream);

  LOG_INST.info(Loc::current(), "The ", current_stage, " result has been saved in '", json_file_path, "'!", monitor.getStatsInfo());
}

void DataManager::saveHeadInfo(nlohmann::json& all_json, std::string& current_stage)
{
  std::string update_time = RTUtil::getTimestamp();
  all_json["stage"] = current_stage;
  all_json["update_time"] = update_time;
  all_json["design_name"] = _helper.get_design_name();
  all_json["lef_name_list"] = nlohmann::json::array();
  std::vector<std::string>& lef_file_path_list = _helper.get_lef_file_path_list();
  for (string& lef_file_path : lef_file_path_list) {
    all_json["lef_name_list"].push_back(RTUtil::getFileName(lef_file_path));
  }
  all_json["def_name"] = RTUtil::getFileName(_helper.get_def_file_path());
}

void DataManager::saveStageNetResult(nlohmann::json& net_json, Net& net)
{
  saveBasicInfo(net_json, net);
  savePinAccessorResult(net_json, net);
  saveResourceAllocatorResult(net_json, net);
  saveGlobalRouterResult(net_json, net);
  saveTrackAssignerResult(net_json, net);
  saveDetailedRouterResult(net_json, net);
  saveViolationRepairerResult(net_json, net);
}

void DataManager::saveBasicInfo(nlohmann::json& net_json, Net& net)
{
  net_json["net_idx"] = net.get_net_idx();
  net_json["connect_type"] = net.get_connect_type();
}

void DataManager::savePinAccessorResult(nlohmann::json& net_json, Net& net)
{
  std::vector<Pin>& pin_list = net.get_pin_list();
  Pin& driving_pin = net.get_driving_pin();
  BoundingBox& bounding_box = net.get_bounding_box();

  net_json["pin_list_size"] = pin_list.size();
  net_json["pin_list"] = nlohmann::json::array();
  for (Pin& pin : pin_list) {
    nlohmann::json pin_json;
    pin_json["pin_name"] = pin.get_pin_name();
    pin_json["pin_idx"] = pin.get_pin_idx();
    // routing_shape_list
    std::vector<EXTLayerRect>& routing_shape_list = pin.get_routing_shape_list();
    pin_json["routing_shape_list_size"] = routing_shape_list.size();
    pin_json["routing_shape_list"] = nlohmann::json::array();
    for (EXTLayerRect& routing_shape : routing_shape_list) {
      nlohmann::json routing_shape_json;
      routing_shape_json["layer_idx"] = routing_shape.get_layer_idx();
      routing_shape_json["grid_rect"] = RTUtil::getString(routing_shape.get_grid_lb_x(), " ", routing_shape.get_grid_lb_y(), " ",
                                                          routing_shape.get_grid_rt_x(), " ", routing_shape.get_grid_rt_y());
      routing_shape_json["real_rect"] = RTUtil::getString(routing_shape.get_real_lb_x(), " ", routing_shape.get_real_lb_y(), " ",
                                                          routing_shape.get_real_rt_x(), " ", routing_shape.get_real_rt_y());
      pin_json["routing_shape_list"].push_back(routing_shape_json);
    }
    // cut_shape_list
    std::vector<EXTLayerRect>& cut_shape_list = pin.get_cut_shape_list();
    pin_json["cut_shape_list_size"] = cut_shape_list.size();
    pin_json["cut_shape_list"] = nlohmann::json::array();
    for (EXTLayerRect& cut_shape : cut_shape_list) {
      nlohmann::json cut_shape_json;
      cut_shape_json["layer_idx"] = cut_shape.get_layer_idx();
      cut_shape_json["grid_rect"] = RTUtil::getString(cut_shape.get_grid_lb_x(), " ", cut_shape.get_grid_lb_y(), " ",
                                                      cut_shape.get_grid_rt_x(), " ", cut_shape.get_grid_rt_y());
      cut_shape_json["real_rect"] = RTUtil::getString(cut_shape.get_real_lb_x(), " ", cut_shape.get_real_lb_y(), " ",
                                                      cut_shape.get_real_rt_x(), " ", cut_shape.get_real_rt_y());
      pin_json["cut_shape_list"].push_back(cut_shape_json);
    }
    // access_point_list
    std::vector<AccessPoint>& access_point_list = pin.get_access_point_list();
    pin_json["access_point_list_size"] = access_point_list.size();
    pin_json["access_point_list"] = nlohmann::json::array();
    for (AccessPoint& access_point : access_point_list) {
      nlohmann::json access_point_json;
      access_point_json["type"] = access_point.get_type();
      access_point_json["layer_idx"] = access_point.get_layer_idx();
      access_point_json["grid"] = RTUtil::getString(access_point.get_grid_x(), " ", access_point.get_grid_y());
      access_point_json["real"] = RTUtil::getString(access_point.get_real_x(), " ", access_point.get_real_y());
      pin_json["access_point_list"].push_back(access_point_json);
    }
    net_json["pin_list"].push_back(pin_json);
  }

  net_json["driving_pin_idx"] = driving_pin.get_pin_idx();
  // rect "lb_x lb_y rt_x rt_y"
  net_json["bounding_box"]["grid_rect"] = RTUtil::getString(bounding_box.get_grid_lb_x(), " ", bounding_box.get_grid_lb_y(), " ",
                                                            bounding_box.get_grid_rt_x(), " ", bounding_box.get_grid_rt_y());
  net_json["bounding_box"]["real_rect"] = RTUtil::getString(bounding_box.get_real_lb_x(), " ", bounding_box.get_real_lb_y(), " ",
                                                            bounding_box.get_real_rt_x(), " ", bounding_box.get_real_rt_y());
}

void DataManager::saveResourceAllocatorResult(nlohmann::json& net_json, Net& net)
{
  GridMap<double>& ra_cost_map = net.get_ra_cost_map();
  irt_int x_size = ra_cost_map.get_x_size();
  irt_int y_size = ra_cost_map.get_y_size();
  net_json["ra_cost_map"]["x_size"] = x_size;
  net_json["ra_cost_map"]["y_size"] = y_size;
  net_json["ra_cost_map"]["_data_map"] = nlohmann::json::array();
  for (irt_int x = 0; x < x_size; ++x) {
    for (irt_int y = 0; y < y_size; ++y) {
      net_json["ra_cost_map"]["data_map"].push_back(ra_cost_map[x][y]);
    }
  }
}

void DataManager::saveGlobalRouterResult(nlohmann::json& net_json, Net& net)
{
  MTree<RTNode>& node_tree = net.get_gr_result_tree();
  nlohmann::json& gr_json = net_json["gr_result_tree"];
  saveResultTree(gr_json, net, node_tree);
}

void DataManager::saveResultTree(nlohmann::json& net_json, Net& net, MTree<RTNode>& node_tree)
{
  std::vector<TNode<RTNode>*> node_list = RTUtil::getNodeList(node_tree);
  net_json["node_list_size"] = node_list.size();
  std::map<TNode<RTNode>*, irt_int> node_ref_idx_map;
  for (size_t i = 0; i < node_list.size(); i++) {
    node_ref_idx_map.insert(std::make_pair(node_list[i], i));
  }
  net_json["root_node_idx"] = node_ref_idx_map[node_tree.get_root()];
  for (TNode<RTNode>* node : node_list) {
    nlohmann::json node_json;
    // node_idx
    node_json["node_idx"] = node_ref_idx_map[node];
    // child_node_idx_list
    node_json["child_node_idx_list"] = nlohmann::json::array();
    for (TNode<RTNode>* child_node : node->get_child_list()) {
      node_json["child_node_idx_list"].push_back(node_ref_idx_map[child_node]);
    }
    // guide "lb_x lb_y rt_x rt_y layer_idx grid_x grid_y"
    Guide& first_guide = node->value().get_first_guide();
    node_json["first_guide"] = RTUtil::getString(first_guide.get_lb_x(), " ", first_guide.get_lb_y(), " ", first_guide.get_rt_x(), " ",
                                                 first_guide.get_rt_y(), " ", first_guide.get_layer_idx(), " ",
                                                 first_guide.get_grid_coord().get_x(), " ", first_guide.get_grid_coord().get_y());
    Guide& second_guide = node->value().get_second_guide();
    node_json["second_guide"] = RTUtil::getString(second_guide.get_lb_x(), " ", second_guide.get_lb_y(), " ", second_guide.get_rt_x(), " ",
                                                  second_guide.get_rt_y(), " ", second_guide.get_layer_idx(), " ",
                                                  second_guide.get_grid_coord().get_x(), " ", second_guide.get_grid_coord().get_y());
    // pin_idx_list
    node_json["pin_list"] = nlohmann::json::array();
    for (irt_int pin_idx : node->value().get_pin_idx_set()) {
      node_json["pin_list"].push_back(net.get_pin_list()[pin_idx].get_pin_name());
    }
    // routing_tree
    MTree<LayerCoord>& routing_tree = node->value().get_routing_tree();
    std::vector<TNode<LayerCoord>*> coord_list = RTUtil::getNodeList(routing_tree);
    node_json["coord_list_size"] = coord_list.size();
    std::map<TNode<LayerCoord>*, irt_int> coord_ref_idx_map;
    for (size_t i = 0; i < coord_list.size(); i++) {
      coord_ref_idx_map.insert(std::make_pair(coord_list[i], i));
    }
    node_json["root_coord_idx"] = coord_ref_idx_map[routing_tree.get_root()];
    for (TNode<LayerCoord>* coord : coord_list) {
      nlohmann::json coord_json;
      // coord_idx
      coord_json["coord_idx"] = coord_ref_idx_map[coord];
      // child_coord_idx_list
      coord_json["child_coord_idx_list"] = nlohmann::json::array();
      for (TNode<LayerCoord>* child_coord : coord->get_child_list()) {
        coord_json["child_coord_idx_list"].push_back(coord_ref_idx_map[child_coord]);
      }
      // x y layer_idx
      coord_json["coord_value"]
          = RTUtil::getString(coord->value().get_x(), " ", coord->value().get_y(), " ", coord->value().get_layer_idx());
      node_json["coord_list"].push_back(coord_json);
    }
    net_json["node_list"].push_back(node_json);
  }
}

void DataManager::saveTrackAssignerResult(nlohmann::json& net_json, Net& net)
{
  MTree<RTNode>& node_tree = net.get_ta_result_tree();
  nlohmann::json& ta_json = net_json["ta_result_tree"];
  saveResultTree(ta_json, net, node_tree);
}

void DataManager::saveDetailedRouterResult(nlohmann::json& net_json, Net& net)
{
  MTree<RTNode>& node_tree = net.get_dr_result_tree();
  nlohmann::json& dr_json = net_json["dr_result_tree"];
  saveResultTree(dr_json, net, node_tree);
}

void DataManager::saveViolationRepairerResult(nlohmann::json& net_json, Net& net)
{
  MTree<PHYNode>& node_tree = net.get_vr_result_tree();
  std::vector<TNode<PHYNode>*> node_list = RTUtil::getNodeList(node_tree);
  nlohmann::json& vr_json = net_json["vr_result_tree"];
  vr_json["node_list_size"] = node_list.size();
  std::map<TNode<PHYNode>*, irt_int> node_ref_idx_map;
  for (size_t i = 0; i < node_list.size(); i++) {
    node_ref_idx_map.insert(std::make_pair(node_list[i], i));
  }
  vr_json["root_node_idx"] = node_ref_idx_map[node_tree.get_root()];
  for (TNode<PHYNode>* node : node_list) {
    nlohmann::json node_json;
    // node_idx
    node_json["node_idx"] = node_ref_idx_map[node];
    // child_node_idx_list
    node_json["child_node_idx_list"] = nlohmann::json::array();
    for (TNode<PHYNode>* child_node : node->get_child_list()) {
      node_json["child_node_idx_list"].push_back(node_ref_idx_map[child_node]);
    }
    // PinNode*, WireNode*, ViaNode*
    PHYNode& phy_node = node->value();
    if (phy_node.isEmpty()) {
      node_json["type"] = string("monostate");
    } else if (phy_node.isType<PinNode>()) {
      node_json["type"] = string("PinNode");
      PinNode& pin_node = phy_node.getNode<PinNode>();
      node_json["net_idx"] = pin_node.get_net_idx();
      node_json["pin_idx"] = pin_node.get_pin_idx();
      node_json["layer_idx"] = pin_node.get_layer_idx();
      node_json["coord"] = RTUtil::getString(pin_node.get_x(), " ", pin_node.get_y());
    } else if (phy_node.isType<WireNode>()) {
      node_json["type"] = string("WireNode");
      WireNode& wire_node = phy_node.getNode<WireNode>();
      node_json["net_idx"] = wire_node.get_net_idx();
      node_json["layer_idx"] = wire_node.get_layer_idx();
      node_json["wire_width"] = wire_node.get_wire_width();
      node_json["first"] = RTUtil::getString(wire_node.get_first().get_x(), " ", wire_node.get_first().get_y());
      node_json["second"] = RTUtil::getString(wire_node.get_second().get_x(), " ", wire_node.get_second().get_y());
    } else {
      node_json["type"] = string("ViaNode");
      ViaNode& via_node = phy_node.getNode<ViaNode>();
      node_json["net_idx"] = via_node.get_net_idx();
      node_json["via_idx"]["first"] = via_node.get_via_idx().first;
      node_json["via_idx"]["second"] = via_node.get_via_idx().second;
      node_json["coord"] = RTUtil::getString(via_node.get_x(), " ", via_node.get_y());
    }
    vr_json["node_list"].push_back(node_json);
  }
}

void DataManager::loadStageResult(Stage stage)
{
  Monitor monitor;

  std::string current_stage = GetStageName()(stage);
  LOG_INST.info(Loc::current(), "The ", current_stage, " result is loading...");

  std::string json_file_path = _config.dm_temp_directory_path + GetStageName()(stage) + ".json";
  std::ifstream* json_stream = RTUtil::getInputFileStream(json_file_path);
  nlohmann::json all_json = nlohmann::json::parse(*json_stream);
  RTUtil::closeFileStream(json_stream);

  checkHeadInfo(all_json, current_stage);
  for (Net& net : _database.get_net_list()) {
    nlohmann::json& net_json = all_json["net_list"][net.get_net_name()];
    loadStageNetResult(net_json, net);
  }
  LOG_INST.info(Loc::current(), "The ", current_stage, " result has been loaded from '", json_file_path, "'!", monitor.getStatsInfo());
}

void DataManager::checkHeadInfo(nlohmann::json& all_json, std::string current_stage)
{
  std::string file_stage = all_json["stage"];
  if (file_stage != current_stage) {
    LOG_INST.error(Loc::current(), "The file_stage '", file_stage, "' != current_stage '", current_stage, "'");
  }

  std::string file_design_name = all_json["design_name"];
  if (file_design_name != _helper.get_design_name()) {
    LOG_INST.error(Loc::current(), "The file_design_name '", file_design_name, "' != design_name '", _helper.get_design_name(), "'");
  }

  std::string file_def_name = all_json["def_name"];
  std::string def_name = RTUtil::getFileName(_helper.get_def_file_path());
  if (file_def_name != def_name) {
    LOG_INST.error(Loc::current(), "The file_def_name '", file_def_name, "' != def_name '", def_name, "'");
  }

  std::set<std::string> file_lef_name_set;
  for (std::string lef_name : all_json["lef_name_list"]) {
    file_lef_name_set.insert(lef_name);
  }
  for (std::string lef_file_path : _helper.get_lef_file_path_list()) {
    std::string lef_name = RTUtil::getFileName(lef_file_path);
    if (!RTUtil::exist(file_lef_name_set, lef_name)) {
      LOG_INST.error(Loc::current(), "The lef_file_path '", lef_file_path, "is inconsistent with the json file");
    }
    file_lef_name_set.erase(lef_name);
  }
  for (std::string file_name : file_lef_name_set) {
    LOG_INST.error(Loc::current(), "The lef_file '", file_name, "' is not found");
  }
}

void DataManager::loadStageNetResult(nlohmann::json& net_json, Net& net)
{
  loadBasicInfo(net_json, net);
  loadPinAccessorResult(net_json, net);
  loadResourceAllocatorResult(net_json, net);
  loadGlobalRouterResult(net_json, net);
  loadTrackAssignerResult(net_json, net);
  loadDetailedRouterResult(net_json, net);
  loadViolationRepairerResult(net_json, net);
}

void DataManager::loadBasicInfo(nlohmann::json& net_json, Net& net)
{
  net.set_net_idx(net_json["net_idx"]);
  net.set_connect_type(net_json["connect_type"]);
}

void DataManager::loadPinAccessorResult(nlohmann::json& net_json, Net& net)
{
  size_t pin_list_size = net_json["pin_list_size"];
  if (pin_list_size == 0) {
    return;
  }

  std::vector<Pin>& pin_list = net.get_pin_list();
  pin_list.clear();
  pin_list.resize(pin_list_size);
  for (nlohmann::json& pin_json : net_json["pin_list"]) {
    irt_int pin_idx = pin_json["pin_idx"];
    Pin& pin = pin_list[pin_idx];
    pin.set_pin_idx(pin_idx);
    pin.set_pin_name(pin_json["pin_name"]);

    // routing_shape_list
    size_t routing_shape_list_size = pin_json["routing_shape_list_size"];
    if (routing_shape_list_size != 0) {
      std::vector<EXTLayerRect>& routing_shape_list = pin.get_routing_shape_list();
      routing_shape_list.clear();
      routing_shape_list.reserve(routing_shape_list_size);
      for (nlohmann::json& routing_shape_json : pin_json["routing_shape_list"]) {
        EXTLayerRect routing_shape;
        routing_shape.set_layer_idx(routing_shape_json["layer_idx"]);
        // rect lb_x lb_y rt_x rt_y
        irt_int lb_x, lb_y, rt_x, rt_y;
        std::istringstream(std::string(routing_shape_json["grid_rect"])) >> lb_x >> lb_y >> rt_x >> rt_y;
        routing_shape.set_grid_rect(PlanarRect(lb_x, lb_y, rt_x, rt_y));
        std::istringstream(std::string(routing_shape_json["real_rect"])) >> lb_x >> lb_y >> rt_x >> rt_y;
        routing_shape.set_real_rect(PlanarRect(lb_x, lb_y, rt_x, rt_y));
        routing_shape_list.push_back(routing_shape);
      }
    }
    // cut_shape_list
    size_t cut_shape_list_size = pin_json["cut_shape_list_size"];
    if (cut_shape_list_size != 0) {
      std::vector<EXTLayerRect>& cut_shape_list = pin.get_cut_shape_list();
      cut_shape_list.clear();
      cut_shape_list.reserve(cut_shape_list_size);
      for (nlohmann::json& cut_shape_json : pin_json["cut_shape_list"]) {
        EXTLayerRect cut_shape;
        cut_shape.set_layer_idx(cut_shape_json["layer_idx"]);
        // rect lb_x lb_y rt_x rt_y
        irt_int lb_x, lb_y, rt_x, rt_y;
        std::istringstream(std::string(cut_shape_json["grid_rect"])) >> lb_x >> lb_y >> rt_x >> rt_y;
        cut_shape.set_grid_rect(PlanarRect(lb_x, lb_y, rt_x, rt_y));
        std::istringstream(std::string(cut_shape_json["real_rect"])) >> lb_x >> lb_y >> rt_x >> rt_y;
        cut_shape.set_real_rect(PlanarRect(lb_x, lb_y, rt_x, rt_y));
        cut_shape_list.push_back(cut_shape);
      }
    }

    // access_point_list
    size_t access_point_list_size = pin_json["access_point_list_size"];
    if (access_point_list_size != 0) {
      std::vector<AccessPoint>& access_point_list = pin.get_access_point_list();
      access_point_list.clear();
      access_point_list.reserve(access_point_list_size);
      for (nlohmann::json& access_point_json : pin_json["access_point_list"]) {
        AccessPoint access_point;
        access_point.set_type(access_point_json["type"]);
        access_point.set_layer_idx(access_point_json["layer_idx"]);
        irt_int x, y;
        istringstream(std::string(access_point_json["grid"])) >> x >> y;
        access_point.set_grid_coord(x, y);
        istringstream(std::string(access_point_json["real"])) >> x >> y;
        access_point.set_real_coord(x, y);
        access_point_list.push_back(access_point);
      }
    }
  }

  // driving_pin
  irt_int driving_pin_idx = net_json["driving_pin_idx"];
  net.set_driving_pin(net.get_pin_list()[driving_pin_idx]);

  // bounding_box
  BoundingBox& bounding_box = net.get_bounding_box();
  irt_int lb_x, lb_y, rt_x, rt_y;
  std::string grid_string = net_json["bounding_box"]["grid_rect"];
  istringstream(grid_string) >> lb_x >> lb_y >> rt_x >> rt_y;
  bounding_box.set_grid_rect(PlanarRect(lb_x, lb_y, rt_x, rt_y));

  std::string real_string = net_json["bounding_box"]["real_rect"];
  istringstream(real_string) >> lb_x >> lb_y >> rt_x >> rt_y;
  bounding_box.set_real_rect(PlanarRect(lb_x, lb_y, rt_x, rt_y));
}

void DataManager::loadResourceAllocatorResult(nlohmann::json& net_json, Net& net)
{
  irt_int x_size = net_json["ra_cost_map"]["x_size"];
  irt_int y_size = net_json["ra_cost_map"]["y_size"];
  if (x_size == 0 || y_size == 0) {
    return;
  }
  vector<double> costs;
  costs.reserve(x_size * y_size);
  for (double cost : net_json["ra_cost_map"]["data_map"]) {
    costs.push_back(cost);
  }
  if (costs.size() != static_cast<size_t>(x_size * y_size)) {
    LOG_INST.error(Loc::current(), "Error when load the size of ra_cost_map");
  }

  GridMap<double> cost_map(x_size, y_size);
  for (irt_int x = 0; x < x_size; ++x) {
    for (irt_int y = 0; y < y_size; ++y) {
      cost_map[x][y] = costs[x * y_size + y];
    }
  }
  net.set_ra_cost_map(cost_map);
}

void DataManager::loadGlobalRouterResult(nlohmann::json& net_json, Net& net)
{
  MTree<RTNode>& node_tree = net.get_gr_result_tree();
  nlohmann::json& gr_json = net_json["gr_result_tree"];
  loadResultTree(gr_json, net, node_tree);
}

void DataManager::loadResultTree(nlohmann::json& net_json, Net& net, MTree<RTNode>& node_tree)
{
  std::map<std::string, irt_int> pin_name_idx_map;
  for (Pin& pin : net.get_pin_list()) {
    pin_name_idx_map.insert(std::pair(pin.get_pin_name(), pin.get_pin_idx()));
  }
  size_t node_list_size = net_json["node_list_size"];
  if (node_list_size == 0) {
    return;
  }
  std::vector<TNode<RTNode>*> node_list;
  for (size_t i = 0; i < node_list_size; i++) {
    node_list.push_back(new TNode<RTNode>(RTNode()));
  }
  node_tree.set_root(node_list[net_json["root_node_idx"]]);
  for (nlohmann::json& node_json : net_json["node_list"]) {
    TNode<RTNode>* node = node_list[node_json["node_idx"]];
    // child_node_idx_list
    for (irt_int child_node_idx : node_json["child_node_idx_list"]) {
      node->addChild(node_list[child_node_idx]);
    }
    // guide "lb_x lb_y rt_x rt_y layer_idx grid_x grid_y"
    irt_int lb_x, lb_y, rt_x, rt_y, layer_idx, grid_x, grid_y;
    Guide& first_guide = node->value().get_first_guide();
    std::istringstream(std::string(node_json["first_guide"])) >> lb_x >> lb_y >> rt_x >> rt_y >> layer_idx >> grid_x >> grid_y;
    first_guide.set_rect(lb_x, lb_y, rt_x, rt_y);
    first_guide.set_layer_idx(layer_idx);
    first_guide.set_grid_coord(grid_x, grid_y);
    Guide& second_guide = node->value().get_second_guide();
    std::istringstream(std::string(node_json["second_guide"])) >> lb_x >> lb_y >> rt_x >> rt_y >> layer_idx >> grid_x >> grid_y;
    second_guide.set_rect(lb_x, lb_y, rt_x, rt_y);
    second_guide.set_layer_idx(layer_idx);
    second_guide.set_grid_coord(grid_x, grid_y);
    // pin_idx_list
    for (std::string pin_name : node_json["pin_list"]) {
      node->value().get_pin_idx_set().insert(pin_name_idx_map[pin_name]);
    }
    // routing_tree
    MTree<LayerCoord>& routing_tree = node->value().get_routing_tree();
    size_t coord_list_size = node_json["coord_list_size"];
    if (coord_list_size == 0) {
      continue;
    }
    std::vector<TNode<LayerCoord>*> coord_list;
    for (size_t i = 0; i < coord_list_size; i++) {
      coord_list.push_back(new TNode<LayerCoord>(LayerCoord()));
    }
    routing_tree.set_root(coord_list[node_json["root_coord_idx"]]);
    for (nlohmann::json& coord_json : node_json["coord_list"]) {
      TNode<LayerCoord>* coord = coord_list[coord_json["coord_idx"]];
      // child_coord_idx_list
      for (irt_int child_coord_idx : coord_json["child_coord_idx_list"]) {
        coord->addChild(coord_list[child_coord_idx]);
      }
      // x y layer_idx
      irt_int x, y, layer_idx;
      LayerCoord& coord_value = coord->value();
      std::istringstream(std::string(coord_json["coord_value"])) >> x >> y >> layer_idx;
      coord_value.set_coord(x, y);
      coord_value.set_layer_idx(layer_idx);
    }
  }
}

void DataManager::loadTrackAssignerResult(nlohmann::json& net_json, Net& net)
{
  MTree<RTNode>& node_tree = net.get_ta_result_tree();
  nlohmann::json& ta_json = net_json["ta_result_tree"];
  loadResultTree(ta_json, net, node_tree);
}

void DataManager::loadDetailedRouterResult(nlohmann::json& net_json, Net& net)
{
  MTree<RTNode>& node_tree = net.get_dr_result_tree();
  nlohmann::json& dr_json = net_json["dr_result_tree"];
  loadResultTree(dr_json, net, node_tree);
}

void DataManager::loadViolationRepairerResult(nlohmann::json& net_json, Net& net)
{
  nlohmann::json& vr_json = net_json["vr_result_tree"];
  std::map<std::string, irt_int> pin_name_idx_map;
  for (Pin& pin : net.get_pin_list()) {
    pin_name_idx_map.insert(std::pair(pin.get_pin_name(), pin.get_pin_idx()));
  }
  size_t node_list_size = vr_json["node_list_size"];
  if (node_list_size == 0) {
    return;
  }
  std::vector<TNode<PHYNode>*> node_list;
  for (size_t i = 0; i < node_list_size; i++) {
    node_list.push_back(new TNode<PHYNode>(PHYNode()));
  }
  MTree<PHYNode>& node_tree = net.get_vr_result_tree();
  node_tree.set_root(node_list[vr_json["root_node_idx"]]);
  for (nlohmann::json& node_json : vr_json["node_list"]) {
    TNode<PHYNode>* node = node_list[node_json["node_idx"]];
    // child_node_idx_list
    for (irt_int child_node_idx : node_json["child_node_idx_list"]) {
      node->addChild(node_list[child_node_idx]);
    }
    std::string node_type = node_json["type"];
    if (node_type == string("monostate")) {
    } else if (node_type == string("PinNode")) {
      PinNode& pin_node = node->value().getNode<PinNode>();
      pin_node.set_net_idx(node_json["net_idx"]);
      pin_node.set_pin_idx(node_json["pin_idx"]);
      pin_node.set_layer_idx(node_json["layer_idx"]);

      irt_int x, y;
      istringstream(std::string(node_json["coord"])) >> x >> y;
      pin_node.set_coord(x, y);

    } else if (node_type == string("WireNode")) {
      WireNode& wire_node = node->value().getNode<WireNode>();
      wire_node.set_net_idx(node_json["net_idx"]);
      wire_node.set_layer_idx(node_json["layer_idx"]);
      wire_node.set_wire_width(node_json["wire_width"]);

      irt_int x, y;
      istringstream(std::string(node_json["first"])) >> x >> y;
      wire_node.get_first().set_coord(x, y);
      istringstream(std::string(node_json["second"])) >> x >> y;
      wire_node.get_second().set_coord(x, y);
    } else if (node_type == string("ViaNode")) {
      ViaNode& via_node = node->value().getNode<ViaNode>();
      via_node.set_net_idx(node_json["net_idx"]);
      via_node.set_via_idx(make_pair(irt_int(node_json["via_idx"]["first"]), irt_int(node_json["via_idx"]["second"])));

      irt_int x, y;
      istringstream(std::string(node_json["coord"])) >> x >> y;
      via_node.set_coord(x, y);
    } else {
      LOG_INST.error(Loc::current(), "PHYNode type is ilegal");
    }
  }
}

void DataManager::outputGCellGrid(idb::IdbBuilder* idb_builder)
{
  GCellAxis& gcell_axis = _database.get_gcell_axis();

  idb::IdbGCellGridList* idb_gcell_grid_list = idb_builder->get_lef_service()->get_layout()->get_gcell_grid_list();
  idb_gcell_grid_list->clear();

  for (idb::IdbTrackDirection idb_track_direction : {idb::IdbTrackDirection::kDirectionX, idb::IdbTrackDirection::kDirectionY}) {
    std::vector<GCellGrid> gcell_grid_list;
    if (idb_track_direction == idb::IdbTrackDirection::kDirectionX) {
      gcell_grid_list = gcell_axis.get_x_grid_list();
    } else {
      gcell_grid_list = gcell_axis.get_y_grid_list();
    }
    for (GCellGrid& gcell_grid : gcell_grid_list) {
      idb::IdbGCellGrid* idb_gcell_grid = new idb::IdbGCellGrid();
      idb_gcell_grid->set_start(gcell_grid.get_start_line());
      idb_gcell_grid->set_space(gcell_grid.get_step_length());
      idb_gcell_grid->set_num(gcell_grid.get_step_num() + 1);
      idb_gcell_grid->set_direction(idb_track_direction);
      idb_gcell_grid_list->add_gcell_grid(idb_gcell_grid);
    }
  }
}

void DataManager::outputNetList(idb::IdbBuilder* idb_builder)
{
  std::vector<Net>& net_list = _database.get_net_list();

  idb::IdbNetList* idb_net_list = idb_builder->get_def_service()->get_design()->get_net_list();
  if (idb_net_list == nullptr) {
    LOG_INST.error(Loc::current(), "The idb net list is empty!");
  }

  for (size_t i = 0; i < net_list.size(); i++) {
    Net& net = net_list[i];
    std::string net_name = net.get_net_name();
    idb::IdbNet* idb_net = idb_net_list->find_net(net_name);
    if (idb_net == nullptr) {
      LOG_INST.info(Loc::current(), "The idb net named ", net_name, " cannot be found!");
      continue;
    }
    convertToIDBNet(idb_builder, net, idb_net);
  }
}

void DataManager::convertToIDBNet(idb::IdbBuilder* idb_builder, Net& net, idb::IdbNet* idb_net)
{
  idb::IdbLayers* idb_layer_list = idb_builder->get_def_service()->get_layout()->get_layers();

  idb::IdbVias* lef_via_list = idb_builder->get_lef_service()->get_layout()->get_via_list();
  idb::IdbVias* def_via_list = idb_builder->get_def_service()->get_design()->get_via_list();

  idb_net->clear_wire_list();
  idb::IdbRegularWireList* idb_wire_list = idb_net->get_wire_list();
  if (idb_wire_list == nullptr) {
    LOG_INST.error(Loc::current(), "The idb wire list is empty!");
  }
  idb::IdbRegularWire* idb_wire = idb_wire_list->add_wire();
  idb_wire->set_wire_state(idb::IdbWiringStatement::kRouted);

  irt_int print_new = false;
  for (TNode<PHYNode>* phy_node_node : RTUtil::getNodeList(net.get_vr_result_tree())) {
    PHYNode& phy_node = phy_node_node->value();
    if (phy_node.isType<PinNode>()) {
      continue;
    }
    idb::IdbRegularWireSegment* idb_segment = idb_wire->add_segment();
    if (phy_node.isType<WireNode>()) {
      convertToIDBWire(idb_layer_list, phy_node.getNode<WireNode>(), idb_segment);
    } else if (phy_node.isType<ViaNode>()) {
      convertToIDBVia(lef_via_list, def_via_list, phy_node.getNode<ViaNode>(), idb_segment);
    } else {
      LOG_INST.error(Loc::current(), "The phy node is incorrect type!");
    }
    if (print_new == false) {
      idb_segment->set_layer_as_new();
      print_new = true;
    }
  }
}

void DataManager::convertToIDBWire(idb::IdbLayers* idb_layer_list, WireNode& wire_node, idb::IdbRegularWireSegment* idb_segment)
{
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();

  std::string layer_name = routing_layer_list[wire_node.get_layer_idx()].get_layer_name();
  idb::IdbLayer* idb_layer = idb_layer_list->find_layer(layer_name);
  if (idb_layer == nullptr) {
    LOG_INST.error(Loc::current(), "Can not find idb layer ", layer_name);
  }
  PlanarCoord& first_coord = wire_node.get_first();
  PlanarCoord& second_coord = wire_node.get_second();
  if (RTUtil::isOblique(first_coord, second_coord)) {
    LOG_INST.error(Loc::current(), "The wire is oblique!");
  }
  idb_segment->set_layer(idb_layer);
  idb_segment->add_point(first_coord.get_x(), first_coord.get_y());
  idb_segment->add_point(second_coord.get_x(), second_coord.get_y());
}

void DataManager::convertToIDBVia(idb::IdbVias* lef_via_list, idb::IdbVias* def_via_list, ViaNode& via_node,
                                  idb::IdbRegularWireSegment* idb_segment)
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _database.get_layer_via_master_list();

  std::string via_name = layer_via_master_list[via_node.get_via_idx().first][via_node.get_via_idx().second].get_via_name();
  idb::IdbVia* idb_via = lef_via_list->find_via(via_name);
  if (idb_via == nullptr) {
    idb_via = def_via_list->find_via(via_name);
  }
  if (idb_via == nullptr) {
    LOG_INST.error(Loc::current(), "Can not find idb via ", via_name, "!");
  }
  idb::IdbLayer* idb_layer_top = idb_via->get_instance()->get_top_layer_shape()->get_layer();
  if (idb_layer_top == nullptr) {
    LOG_INST.error(Loc::current(), "Can not find layer from idb via ", via_name, "!");
  }
  idb_segment->set_layer(idb_layer_top);
  idb_segment->set_is_via(true);

  idb_segment->add_point(via_node.get_x(), via_node.get_y());
  idb::IdbVia* idb_via_new = idb_segment->copy_via(idb_via);
  idb_via_new->set_coordinate(via_node.get_x(), via_node.get_y());
}

Direction DataManager::getRTDirectionByDB(idb::IdbLayerDirection idb_direction)
{
  if (idb_direction == idb::IdbLayerDirection::kHorizontal) {
    return Direction::kHorizontal;
  } else if (idb_direction == idb::IdbLayerDirection::kVertical) {
    return Direction::kVertical;
  } else {
    return Direction::kOblique;
  }
}

ConnectType DataManager::getRTConnectTypeByDB(idb::IdbConnectType idb_connect_type)
{
  ConnectType connect_type;
  switch (idb_connect_type) {
    case idb::IdbConnectType::kSignal:
      connect_type = ConnectType::kSignal;
      break;
    case idb::IdbConnectType::kPower:
      connect_type = ConnectType::kPower;
      break;
    case idb::IdbConnectType::kGround:
      connect_type = ConnectType::kGround;
      break;
    case idb::IdbConnectType::kClock:
      connect_type = ConnectType::kClock;
      break;
    case idb::IdbConnectType::kAnalog:
      connect_type = ConnectType::kAnalog;
      break;
    case idb::IdbConnectType::kReset:
      connect_type = ConnectType::kReset;
      break;
    case idb::IdbConnectType::kScan:
      connect_type = ConnectType::kScan;
      break;
    case idb::IdbConnectType::kTieOff:
      connect_type = ConnectType::kTieoff;
      break;
    default:
      connect_type = ConnectType::kNone;
      break;
  }
  return connect_type;
}

}  // namespace irt
