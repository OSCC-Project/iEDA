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
#include "EGRDataManager.hpp"

#include "RTAPI.hpp"
#include "RTUtil.hpp"

namespace irt {

// public

void EGRDataManager::input(std::map<std::string, std::any>& config_map, idb::IdbBuilder* idb_builder)
{
  wrapConfig(config_map);
  wrapDatabase(idb_builder);
  buildConfig();
  buildDatabase();
  printConfig();
  printDatabase();
}

// private
void EGRDataManager::wrapConfig(std::map<std::string, std::any>& config_map)
{
  _egr_config.temp_directory_path
      = RTUtil::getConfigValue<std::string>(config_map, "-temp_directory_path", "./result/rt/egr_temp_directory/");
  _egr_config.thread_number = RTUtil::getConfigValue<int32_t>(config_map, "-thread_number", 8);
  _egr_config.congestion_cell_x_pitch = RTUtil::getConfigValue<int32_t>(config_map, "-congestion_cell_x_pitch", 15);
  _egr_config.congestion_cell_y_pitch = RTUtil::getConfigValue<int32_t>(config_map, "-congestion_cell_y_pitch", 15);
  _egr_config.bottom_routing_layer = RTUtil::getConfigValue<std::string>(config_map, "-bottom_routing_layer", "");
  _egr_config.top_routing_layer = RTUtil::getConfigValue<std::string>(config_map, "-top_routing_layer", "");
  _egr_config.accuracy = RTUtil::getConfigValue<int32_t>(config_map, "-accuracy", 2);
  _egr_config.skip_net_name_list
      = RTUtil::getConfigValue<std::vector<std::string>>(config_map, "-skip_net_name_list", std::vector<std::string>());
  _egr_config.strategy = RTUtil::getConfigValue<std::string>(config_map, "-strategy", "gradual");
  _egr_config.temp_directory_path = std::filesystem::absolute(_egr_config.temp_directory_path);
  _egr_config.log_file_path = _egr_config.temp_directory_path + "egr.log";
  RTUtil::createDir(_egr_config.temp_directory_path);
  RTUtil::createDirByFile(_egr_config.log_file_path);
  omp_set_num_threads(_egr_config.accuracy == 0 ? 1 : std::max(_egr_config.thread_number, 1));
}

void EGRDataManager::wrapDatabase(idb::IdbBuilder* idb_builder)
{
  wrapDesignName(idb_builder);
  wrapMicronDBU(idb_builder);
  wrapDie(idb_builder);
  wrapLayerList(idb_builder);
  wrapLayerViaMasterList(idb_builder);
  wrapObstacleList(idb_builder);
  wrapNetList(idb_builder);
  updateHelper(idb_builder);
}

void EGRDataManager::wrapDesignName(idb::IdbBuilder* idb_builder)
{
  _egr_database.set_design_name(idb_builder->get_def_service()->get_design()->get_design_name());
}
void EGRDataManager::wrapMicronDBU(idb::IdbBuilder* idb_builder)
{
  _egr_database.set_micron_dbu(idb_builder->get_def_service()->get_design()->get_units()->get_micron_dbu());
}

void EGRDataManager::wrapDie(idb::IdbBuilder* idb_builder)
{
  idb::IdbDie* die = idb_builder->get_lef_service()->get_layout()->get_die();

  EXTPlanarRect& die_box = _egr_database.get_die();
  die_box.set_real_ll(die->get_llx(), die->get_lly());
  die_box.set_real_ur(die->get_urx(), die->get_ury());
}

void EGRDataManager::wrapLayerList(idb::IdbBuilder* idb_builder)
{
  std::vector<RoutingLayer>& routing_layer_list = _egr_database.get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = _egr_database.get_cut_layer_list();
  std::vector<idb::IdbLayer*>& idb_layers = idb_builder->get_lef_service()->get_layout()->get_layers()->get_layers();

  for (idb::IdbLayer* idb_layer : idb_layers) {
    if (idb_layer->is_routing()) {
      idb::IdbLayerRouting* idb_routing_layer = dynamic_cast<idb::IdbLayerRouting*>(idb_layer);
      RoutingLayer routing_layer;
      routing_layer.set_layer_idx(idb_routing_layer->get_id());
      routing_layer.set_layer_order(idb_routing_layer->get_order());
      routing_layer.set_min_width(idb_routing_layer->get_min_width());
      routing_layer.set_layer_name(idb_routing_layer->get_name());
      routing_layer.set_prefer_direction(getRTDirectionByDB(idb_routing_layer->get_direction()));
      wrapTrackAxis(routing_layer, idb_routing_layer);
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
  std::cout << "routing_layer_size:" << routing_layer_list.size() << "   cut_layer_size:" << cut_layer_list.size() << std::endl;
}

void EGRDataManager::wrapTrackAxis(RoutingLayer& routing_layer, idb::IdbLayerRouting* idb_layer)
{
  ScaleAxis& track_axis = routing_layer.get_track_axis();

  for (idb::IdbTrackGrid* idb_track_grid : idb_layer->get_track_grid_list()) {
    idb::IdbTrack* idb_track = idb_track_grid->get_track();

    ScaleGrid track_grid;
    track_grid.set_start_line(static_cast<int32_t>(idb_track->get_start()));
    track_grid.set_step_length(static_cast<int32_t>(idb_track->get_pitch()));
    track_grid.set_step_num(static_cast<int32_t>(idb_track_grid->get_track_num()));

    if (idb_track->get_direction() == idb::IdbTrackDirection::kDirectionX) {
      track_axis.get_x_grid_list().push_back(track_grid);
    } else if (idb_track->get_direction() == idb::IdbTrackDirection::kDirectionY) {
      track_axis.get_y_grid_list().push_back(track_grid);
    }
  }
}

void EGRDataManager::wrapLayerViaMasterList(idb::IdbBuilder* idb_builder)
{
  idb::IdbVias* idb_via_list_lib = idb_builder->get_lef_service()->get_layout()->get_via_list();
  if (idb_via_list_lib == nullptr) {
    LOG_INST.error(Loc::current(), "Via list in tech lef is empty!");
  }

  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _egr_database.get_layer_via_master_list();
  std::vector<idb::IdbLayer*> idb_routing_layers = idb_builder->get_lef_service()->get_layout()->get_layers()->get_routing_layers();
  layer_via_master_list.resize(idb_routing_layers.size());

  std::vector<idb::IdbVia*>& idb_via_list = idb_via_list_lib->get_via_list();
  if (idb_via_list.size() == 0) {
    LOG_INST.error(Loc::current(), "no via error");
  }
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
      cut_shape.set_ll(idb_rect->get_low_x(), idb_rect->get_low_y());
      cut_shape.set_ur(idb_rect->get_high_x(), idb_rect->get_high_y());
      cut_shape_list.push_back(std::move(cut_shape));
    }
    via_master.set_cut_layer_idx(idb_shape_cut.get_layer()->get_id());
    layer_via_master_list.front().push_back(std::move(via_master));
  }
}

void EGRDataManager::wrapObstacleList(idb::IdbBuilder* idb_builder)
{
  wrapArtificialObstacle(idb_builder);
  wrapInstanceObstacle(idb_builder);
  wrapSpecialNetObstacle(idb_builder);
}

void EGRDataManager::wrapArtificialObstacle(idb::IdbBuilder* idb_builder)
{
  // Artificial
  // idb::IdbObstacleList* idb_obstacle_list = idb_builder->get_def_service()->get_design()->get_obstacle_list();
  // if (!idb_obstacle_list->get_obstacle_list().empty()) {
  //   LOG_INST.warn(Loc::current(), "The artificial obstacle will be ignored!");
  // }

  // std::vector<Obstacle>& routing_obstacle_list = _database.get_routing_obstacle_list();

  // LOG_INST.warn(Loc::current(), "The artificial obstacle will be ignored!");

  // // Artificial
  // idb::IdbObstacleList* idb_obstacle_list = idb_builder->get_def_service()->get_design()->get_obstacle_list();
  // for (idb::IdbObstacle* idb_obstacle : idb_obstacle_list->get_obstacle_list()) {
  //   if (idb_obstacle->is_routing_obstacle()) {
  //     idb::IdbRoutingObstacle* idb_routing_obstacle = dynamic_cast<idb::IdbRoutingObstacle*>(idb_obstacle);
  //     for (idb::IdbRect* rect : idb_routing_obstacle->get_rect_list()) {
  //       Obstacle obstacle;
  //       obstacle.set_real_ll(rect->get_low_x(), rect->get_low_y());
  //       obstacle.set_real_ur(rect->get_high_x(), rect->get_high_y());
  //       obstacle.set_layer_idx(idb_routing_obstacle->get_layer()->get_id());
  //       obstacle.set_is_artificial(true);
  //       routing_obstacle_list.push_back(std::move(obstacle));
  //     }
  //   }
  // }
}

void EGRDataManager::wrapInstanceObstacle(idb::IdbBuilder* idb_builder)
{
  std::vector<Obstacle>& routing_obstacle_list = _egr_database.get_routing_obstacle_list();

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
      Obstacle obstacle;
      obstacle.set_real_ll(rect->get_low_x(), rect->get_low_y());
      obstacle.set_real_ur(rect->get_high_x(), rect->get_high_y());
      obstacle.set_layer_idx(layer_shape->get_layer()->get_id());
      if (layer_shape->get_layer()->is_routing()) {
        routing_obstacle_list.push_back(std::move(obstacle));
      }
    }
  }
}

void EGRDataManager::wrapSpecialNetObstacle(idb::IdbBuilder* idb_builder)
{
  std::vector<Obstacle>& routing_obstacle_list = _egr_database.get_routing_obstacle_list();

  // special net
  idb::IdbSpecialNetList* idb_special_net_list = idb_builder->get_def_service()->get_design()->get_special_net_list();
  for (idb::IdbSpecialNet* idb_net : idb_special_net_list->get_net_list()) {
    for (idb::IdbSpecialWire* idb_wire : idb_net->get_wire_list()->get_wire_list()) {
      for (idb::IdbSpecialWireSegment* idb_segment : idb_wire->get_segment_list()) {
        if (idb_segment->is_via()) {
          std::vector<idb::IdbLayerShape> layer_shape_list;
          layer_shape_list.push_back(idb_segment->get_via()->get_top_layer_shape());
          layer_shape_list.push_back(idb_segment->get_via()->get_bottom_layer_shape());

          for (idb::IdbLayerShape& layer_shape : layer_shape_list) {
            for (idb::IdbRect* rect : layer_shape.get_rect_list()) {
              Obstacle obstacle;
              obstacle.set_real_ll(rect->get_low_x(), rect->get_low_y());
              obstacle.set_real_ur(rect->get_high_x(), rect->get_high_y());
              obstacle.set_layer_idx(layer_shape.get_layer()->get_id());
              if (layer_shape.get_layer()->is_routing()) {
                routing_obstacle_list.push_back(std::move(obstacle));
              }
            }
          }
        } else {
          idb::IdbRect* idb_rect = idb_segment->get_bounding_box();
          // wire
          Obstacle obstacle;
          obstacle.set_real_ll(idb_rect->get_low_x(), idb_rect->get_low_y());
          obstacle.set_real_ur(idb_rect->get_high_x(), idb_rect->get_high_y());
          obstacle.set_layer_idx(idb_segment->get_layer()->get_id());
          routing_obstacle_list.push_back(std::move(obstacle));
        }
      }
    }
  }
}

void EGRDataManager::wrapNetList(idb::IdbBuilder* idb_builder)
{
  std::vector<EGRNet>& egr_net_list = _egr_database.get_egr_net_list();
  std::vector<idb::IdbNet*> idb_net_list = idb_builder->get_def_service()->get_design()->get_net_list()->get_net_list();

  for (idb::IdbNet* idb_net : idb_net_list) {
    if (!checkSkipping(idb_net)) {
      EGRNet egr_net;
      egr_net.set_net_name(idb_net->get_net_name());
      wrapPinList(egr_net, idb_net);
      processEmptyShapePin(egr_net);
      wrapDrivingPin(egr_net, idb_net);
      egr_net_list.push_back(std::move(egr_net));
    }
  }
}

bool EGRDataManager::checkSkipping(idb::IdbNet* idb_net)
{
  std::string net_name = idb_net->get_net_name();

  // check pin number
  size_t pin_num = idb_net->get_instance_pin_list()->get_pin_num();
  if (pin_num <= 1) {
    return true;
  } else if (pin_num >= 500) {
    LOG_INST.info(Loc::current(), "The ultra large net: ", net_name, " has ", pin_num, " pins!");
  }
  // check the connection form io_cell PAD to io_pin
  bool has_io_pin = false;
  if (idb_net->has_io_pins()) {
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
    return true;
  }
  return false;
}

void EGRDataManager::wrapPinList(EGRNet& egr_net, idb::IdbNet* idb_net)
{
  std::vector<EGRPin>& egr_pin_list = egr_net.get_pin_list();

  // pin list in instance
  for (idb::IdbPin* idb_pin : idb_net->get_instance_pin_list()->get_pin_list()) {
    /// without term description in some cases
    if (idb_pin->get_term()->get_port_number() <= 0) {
      continue;
    }
    EGRPin egr_pin;
    egr_pin.set_pin_name(RTUtil::getString(idb_pin->get_instance()->get_name(), ":", idb_pin->get_pin_name()));
    wrapPinShapeList(egr_pin, idb_pin);
    egr_pin_list.push_back(std::move(egr_pin));
  }
  // io pin list
  for (auto* io_pin : idb_net->get_io_pins()->get_pin_list()) {
    EGRPin egr_pin;
    egr_pin.set_pin_name(io_pin->get_pin_name());
    wrapPinShapeList(egr_pin, io_pin);
    egr_pin_list.push_back(std::move(egr_pin));
  }
}

void EGRDataManager::wrapPinShapeList(EGRPin& egr_pin, idb::IdbPin* idb_pin)
{
  std::vector<EXTLayerRect>& routing_shape_list = egr_pin.get_routing_shape_list();

  for (idb::IdbLayerShape* layer_shape : idb_pin->get_port_box_list()) {
    for (idb::IdbRect* rect : layer_shape->get_rect_list()) {
      EXTLayerRect pin_shape;
      pin_shape.set_real_ll(rect->get_low_x(), rect->get_low_y());
      pin_shape.set_real_ur(rect->get_high_x(), rect->get_high_y());
      pin_shape.set_layer_idx(layer_shape->get_layer()->get_id());
      if (layer_shape->get_layer()->is_routing()) {
        routing_shape_list.push_back(std::move(pin_shape));
      }
    }
  }
}

void EGRDataManager::processEmptyShapePin(EGRNet& net)
{
  std::vector<EGRPin>& pin_list = net.get_pin_list();

  std::vector<int32_t> empty_pin_idx_list;
  for (size_t i = 0; i < pin_list.size(); i++) {
    EGRPin& pin = pin_list[i];
    if (pin.get_routing_shape_list().empty()) {
      empty_pin_idx_list.push_back(i);
    }
  }

  int32_t legal_pin_idx = -1;
  for (size_t i = 0; i < pin_list.size(); i++) {
    EGRPin& pin = pin_list[i];
    if (!pin.get_routing_shape_list().empty()) {
      legal_pin_idx = i;
      break;
    }
  }

  if (legal_pin_idx == -1) {
    LOG_INST.error(Loc::current(), "There is no legal pin for net ", net.get_net_name());
  }

  for (size_t i = 0; i < empty_pin_idx_list.size(); i++) {
    pin_list[empty_pin_idx_list[i]].set_routing_shape_list(pin_list[legal_pin_idx].get_routing_shape_list());
  }
}

void EGRDataManager::wrapDrivingPin(EGRNet& egr_net, idb::IdbNet* idb_net)
{
  idb::IdbPin* idb_driving_pin = idb_net->get_driving_pin();
  if (idb_driving_pin == nullptr) {
    idb_driving_pin = idb_net->get_instance_pin_list()->get_pin_list().front();
  }
  std::string driving_pin_name = idb_driving_pin->get_pin_name();
  if (!idb_driving_pin->is_io_pin()) {
    driving_pin_name = RTUtil::getString(idb_driving_pin->get_instance()->get_name(), ":", driving_pin_name);
  }
  egr_net.get_driving_pin().set_pin_name(driving_pin_name);
}

void EGRDataManager::updateHelper(idb::IdbBuilder* idb_builder)
{
  std::vector<RoutingLayer>& routing_layer_list = _egr_database.get_routing_layer_list();
  std::map<int32_t, int32_t>& db_to_egr_cut_layer_idx_map = _egr_helper.get_db_to_egr_cut_layer_idx_map();
  std::map<int32_t, int32_t>& db_to_egr_routing_layer_idx_map = _egr_helper.get_db_to_egr_routing_layer_idx_map();
  std::map<std::string, int32_t>& cut_layer_name_idx_map = _egr_helper.get_cut_layer_name_idx_map();
  std::map<std::string, int32_t>& routing_layer_name_idx_map = _egr_helper.get_routing_layer_name_idx_map();

  for (size_t i = 0; i < routing_layer_list.size(); ++i) {
    db_to_egr_routing_layer_idx_map[routing_layer_list[i].get_layer_idx()] = static_cast<int32_t>(i);
    routing_layer_name_idx_map[routing_layer_list[i].get_layer_name()] = static_cast<int32_t>(i);
  }

  std::vector<CutLayer>& cut_layer_list = _egr_database.get_cut_layer_list();
  for (size_t i = 0; i < cut_layer_list.size(); i++) {
    db_to_egr_cut_layer_idx_map[cut_layer_list[i].get_layer_idx()] = static_cast<int32_t>(i);
    cut_layer_name_idx_map[cut_layer_list[i].get_layer_name()] = static_cast<int32_t>(i);
  }
}

void EGRDataManager::buildConfig()
{
  buildSkipNetNameSet();
  buildCellSize();
  buildBottomTopLayerIdx();
  buildEGRStrategy();
}

void EGRDataManager::buildSkipNetNameSet()
{
  for (std::string& net_name : _egr_config.skip_net_name_list) {
    _egr_config.skip_net_name_set.insert(net_name);
  }
}

void EGRDataManager::buildCellSize()
{
  std::map<int32_t, int32_t> pitch_count_map;
  for (RoutingLayer& routing_layer : _egr_database.get_routing_layer_list()) {
    for (ScaleGrid& track_grid : routing_layer.getPreferTrackGridList()) {
      pitch_count_map[track_grid.get_step_length()]++;
    }
  }
  int32_t ref_pitch = -1;
  int32_t max_count = INT32_MIN;
  for (auto [pitch, count] : pitch_count_map) {
    if (count > max_count) {
      max_count = count;
      ref_pitch = pitch;
    }
  }
  _egr_config.cell_width = ref_pitch * _egr_config.congestion_cell_x_pitch;
  _egr_config.cell_height = ref_pitch * _egr_config.congestion_cell_y_pitch;
}

void EGRDataManager::buildBottomTopLayerIdx()
{
  std::vector<RoutingLayer>& routing_layer_list = _egr_database.get_routing_layer_list();
  std::map<std::string, int32_t>& routing_layer_name_idx_map = _egr_helper.get_routing_layer_name_idx_map();
  _egr_config.bottom_routing_layer_idx = 0;
  _egr_config.top_routing_layer_idx = routing_layer_list.back().get_layer_idx();

  if (_egr_config.bottom_routing_layer.empty() && _egr_config.top_routing_layer.empty()) {
    _egr_config.bottom_routing_layer = routing_layer_list.front().get_layer_name();
    _egr_config.top_routing_layer = routing_layer_list.back().get_layer_name();
    return;
  }
  if (RTUtil::exist(routing_layer_name_idx_map, _egr_config.bottom_routing_layer)) {
    _egr_config.bottom_routing_layer_idx = routing_layer_name_idx_map[_egr_config.bottom_routing_layer];
  }
  if (RTUtil::exist(routing_layer_name_idx_map, _egr_config.top_routing_layer)) {
    _egr_config.top_routing_layer_idx = routing_layer_name_idx_map[_egr_config.top_routing_layer];
  }
}

void EGRDataManager::buildEGRStrategy()
{
  std::string strategy = _egr_config.strategy;
  if (strategy == "gradual") {
    _egr_config.egr_strategy = EGRStrategy::kGradul;
  } else if (strategy == "topo") {
    _egr_config.egr_strategy = EGRStrategy::kTopo;
  } else {
    _egr_config.egr_strategy = EGRStrategy::kGradul;
    _egr_config.strategy = "gradual";
    LOG_INST.info(Loc::current(), "Optional strategys are 'topo' and 'gradual', use default strategy:gradual");
  }
}

void EGRDataManager::buildDatabase()
{
  buildLayerList();
  buildLayerViaMasterList();
  buildDie();
  buildObstacleList();
  buildNetList();
  buildLayerResourceMap();
  buildHVLayerIdxList();
}

void EGRDataManager::buildLayerList()
{
  std::vector<RoutingLayer>& routing_layer_list = _egr_database.get_routing_layer_list();
  for (RoutingLayer& routing_layer : routing_layer_list) {
    routing_layer.set_layer_idx(getEGRRoutingLayerIndexByDB(routing_layer.get_layer_idx()));
  }
  std::vector<CutLayer>& cut_layer_list = _egr_database.get_cut_layer_list();
  for (CutLayer& cut_layer : cut_layer_list) {
    cut_layer.set_layer_idx(getEGRCutLayerIndexByDB(cut_layer.get_layer_idx()));
  }
}

void EGRDataManager::buildLayerViaMasterList()
{
  transLayerViaMasterList();
  makeLayerViaMasterList();
}

void EGRDataManager::transLayerViaMasterList()
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _egr_database.get_layer_via_master_list();

  for (std::vector<ViaMaster>& via_master_list : layer_via_master_list) {
    for (ViaMaster& via_master : via_master_list) {
      // above
      LayerRect& above_enclosure = via_master.get_above_enclosure();
      above_enclosure.set_layer_idx(getEGRRoutingLayerIndexByDB(above_enclosure.get_layer_idx()));
      // below
      LayerRect& below_enclosure = via_master.get_below_enclosure();
      below_enclosure.set_layer_idx(getEGRRoutingLayerIndexByDB(below_enclosure.get_layer_idx()));
      // cut
      via_master.set_cut_layer_idx(getEGRCutLayerIndexByDB(via_master.get_cut_layer_idx()));
    }
  }
}

void EGRDataManager::makeLayerViaMasterList()
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _egr_database.get_layer_via_master_list();
  std::vector<RoutingLayer>& routing_layer_list = _egr_database.get_routing_layer_list();

  std::vector<ViaMaster> first_via_master_list;
  for (ViaMaster& via_master : layer_via_master_list.front()) {
    int32_t below_layer_idx = via_master.get_below_enclosure().get_layer_idx();
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
      Direction above_layer_direction = routing_layer_list[above_enclosure.get_layer_idx()].get_prefer_direction();
      via_master.set_above_direction(above_enclosure.getRectDirection(above_layer_direction));
      // below
      LayerRect& below_enclosure = via_master.get_below_enclosure();
      Direction below_layer_direction = routing_layer_list[below_enclosure.get_layer_idx()].get_prefer_direction();
      via_master.set_below_direction(below_enclosure.getRectDirection(below_layer_direction));
    }

    std::sort(via_master_list.begin(), via_master_list.end(), [&](ViaMaster& a, ViaMaster& b) {
      std::vector<RoutingLayer>& routing_layer_list = _egr_database.get_routing_layer_list();

      LayerRect& a_above = a.get_above_enclosure();
      LayerRect& a_below = a.get_below_enclosure();
      LayerRect& b_above = b.get_above_enclosure();
      LayerRect& b_below = b.get_below_enclosure();
      // 方向
      Direction a_above_layer_direction = routing_layer_list[a_above.get_layer_idx()].get_prefer_direction();
      Direction b_above_layer_direction = routing_layer_list[b_above.get_layer_idx()].get_prefer_direction();
      if (a.get_above_direction() == a_above_layer_direction && b.get_above_direction() != b_above_layer_direction) {
        return true;
      } else if (a.get_above_direction() != a_above_layer_direction && b.get_above_direction() == b_above_layer_direction) {
        return false;
      }
      Direction a_below_layer_direction = routing_layer_list[a_below.get_layer_idx()].get_prefer_direction();
      Direction b_below_layer_direction = routing_layer_list[b_below.get_layer_idx()].get_prefer_direction();
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
      int32_t a_above_center_diff = std::abs(a_above.get_ll_x() + a_above.get_ur_x());
      int32_t b_above_center_diff = std::abs(b_above.get_ll_x() + b_above.get_ur_x());
      if (a_above_center_diff != b_above_center_diff) {
        return a_above_center_diff < b_above_center_diff;
      }
      int32_t a_below_center_diff = std::abs(a_below.get_ll_x() + a_below.get_ur_x());
      int32_t b_below_center_diff = std::abs(b_below.get_ll_x() + b_below.get_ur_x());
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
      via_master_list[i].set_via_master_idx(layer_idx, i);
    }
  }
}

void EGRDataManager::buildDie()
{
  Die& die = _egr_database.get_die();
  die.set_grid_rect(getGridRect(die.get_real_rect()));
}

void EGRDataManager::buildObstacleList()
{
  std::vector<RoutingLayer>& routing_layer_list = _egr_database.get_routing_layer_list();
  int32_t die_real_ur_x = _egr_database.get_die().get_real_ur_x();
  int32_t die_real_ur_y = _egr_database.get_die().get_real_ur_y();

  for (Obstacle& obstacle : _egr_database.get_routing_obstacle_list()) {
    int32_t layer_idx = getEGRRoutingLayerIndexByDB(obstacle.get_layer_idx());
    int32_t half_wire_width = routing_layer_list[layer_idx].get_min_width() / 2;

    obstacle.set_real_rect(RTUtil::getEnlargedRect(obstacle.get_real_rect(), half_wire_width));
    if (obstacle.get_real_ur_x() > die_real_ur_x) {
      obstacle.get_real_rect().set_ur_x(die_real_ur_x);
    }
    if (obstacle.get_real_ur_y() > die_real_ur_y) {
      obstacle.get_real_rect().set_ur_y(die_real_ur_y);
    }
    obstacle.set_grid_rect(getGridRect(obstacle.get_real_rect()));
    obstacle.set_layer_idx(layer_idx);
  }
}

void EGRDataManager::buildNetList()
{
  std::vector<EGRNet>& egr_net_list = _egr_database.get_egr_net_list();
  for (EGRNet& egr_net : egr_net_list) {
    buildPinList(egr_net);
    buildDrivingPin(egr_net);
  }
}

void EGRDataManager::buildPinList(EGRNet& egr_net)
{
  std::vector<EGRPin>& pin_list = egr_net.get_pin_list();
  int32_t die_ur_x = _egr_database.get_die().get_real_ur_x();
  int32_t die_ur_y = _egr_database.get_die().get_real_ur_y();
  for (size_t i = 0; i < pin_list.size(); i++) {
    EGRPin& egr_pin = pin_list[i];

    egr_pin.set_pin_idx(static_cast<int32_t>(i));
    for (EXTLayerRect& routing_shape : egr_pin.get_routing_shape_list()) {
      routing_shape.set_layer_idx(getEGRRoutingLayerIndexByDB(routing_shape.get_layer_idx()));
      // checkPinShape
      PlanarRect& real_rect = routing_shape.get_real_rect();
      PlanarRect new_rect = real_rect;
      new_rect.set_ll_x(std::min(new_rect.get_ll_x(), die_ur_x));
      new_rect.set_ll_y(std::min(new_rect.get_ll_y(), die_ur_y));
      new_rect.set_ur_x(std::min(new_rect.get_ur_x(), die_ur_x));
      new_rect.set_ur_y(std::min(new_rect.get_ur_y(), die_ur_y));
      if (real_rect != new_rect) {
        LOG_INST.warn(Loc::current(), "Pin:", egr_pin.get_pin_name(), "(", real_rect.get_ll_x(), ",", real_rect.get_ll_y(), ")---(",
                      real_rect.get_ur_x(), ",", real_rect.get_ur_y(), ")", " is out of die");
        real_rect = new_rect;
      }

      routing_shape.set_grid_rect(getGridRect(routing_shape.get_real_rect()));
    }
    std::vector<AccessPoint>& access_point_list = egr_pin.get_access_point_list();
    for (EXTLayerRect& routing_shape : egr_pin.get_routing_shape_list()) {
      AccessPoint access_point;
      access_point.set_grid_coord(routing_shape.get_grid_rect().getMidPoint());
      access_point.set_layer_idx(routing_shape.get_layer_idx());
      access_point_list.push_back(access_point);
    }
  }
}

void EGRDataManager::buildDrivingPin(EGRNet& egr_net)
{
  std::vector<EGRPin>& pin_list = egr_net.get_pin_list();
  for (size_t i = 0; i < pin_list.size(); i++) {
    EGRPin& egr_pin = pin_list[i];
    if (egr_net.get_driving_pin().get_pin_name() != egr_pin.get_pin_name()) {
      continue;
    }
    egr_net.set_driving_pin(egr_pin);
    return;
  }
  LOG_INST.error(Loc::current(), "Unable to find a driving egr_pin!");
}

void EGRDataManager::buildLayerResourceMap()
{
  initLayerResourceMapSize();
  addResourceMapSupply();
  addResourceMapDemand();
  legalizeResourceMapDemand();
}

void EGRDataManager::initLayerResourceMapSize()
{
  std::vector<GridMap<EGRNode>>& layer_resource_map = _egr_database.get_layer_resource_map();
  EXTPlanarRect& die = _egr_database.get_die();

  layer_resource_map.reserve(_egr_database.get_routing_layer_list().size());
  for (size_t i = 0; i < _egr_database.get_routing_layer_list().size(); ++i) {
    layer_resource_map.emplace_back(die.getXSize(), die.getYSize());
  }
  for (GridMap<EGRNode>& resource_map : layer_resource_map) {
    for (int32_t x = 0; x < resource_map.get_x_size(); ++x) {
      for (int32_t y = 0; y < resource_map.get_y_size(); ++y) {
        resource_map[x][y].set_ll(_egr_config.cell_width * x, _egr_config.cell_height * y);
        resource_map[x][y].set_ur(_egr_config.cell_width * (x + 1), _egr_config.cell_height * (y + 1));
      }
    }
    for (int32_t x = 0; x < resource_map.get_x_size(); ++x) {
      resource_map[x][resource_map.get_y_size() - 1].set_ur_y(die.get_real_ur_y());
    }
    for (int32_t y = 0; y < resource_map.get_y_size(); ++y) {
      resource_map[resource_map.get_x_size() - 1][y].set_ur_x(die.get_real_ur_x());
    }
  }
}

void EGRDataManager::addResourceMapSupply()
{
  std::vector<GridMap<EGRNode>>& layer_resource_map = _egr_database.get_layer_resource_map();
  std::vector<RoutingLayer>& routing_layer_list = _egr_database.get_routing_layer_list();
  for (size_t i = 0; i < routing_layer_list.size(); ++i) {
    GridMap<EGRNode>& resource_map = layer_resource_map[i];

    RoutingLayer& routing_layer = routing_layer_list[i];
    int32_t track_start_line = routing_layer.getPreferTrackGridList().front().get_start_line();
    int32_t track_pitch = routing_layer.getPreferTrackGridList().front().get_step_length();

    for (int32_t x = 0; x < resource_map.get_x_size(); ++x) {
      for (int32_t y = 0; y < resource_map.get_y_size(); ++y) {
        EGRNode& resource_node = resource_map[x][y];
        if (routing_layer.isPreferH()) {
          double end_track = std::ceil(std::max(0, resource_node.get_ur_y() - track_start_line) / 1.0 / track_pitch);
          double start_track = std::ceil(std::max(0, resource_node.get_ll_y() - track_start_line) / 1.0 / track_pitch);
          int32_t track_num = static_cast<int32_t>(end_track - start_track);

          resource_node.addSupply(EGRResourceType::kWest, track_num);
          resource_node.addSupply(EGRResourceType::kEast, track_num);
          resource_node.addSupply(EGRResourceType::kTrack, track_num);
        } else {
          double end_track = std::ceil(std::max(0, resource_node.get_ur_x() - track_start_line) / 1.0 / track_pitch);
          double start_track = std::ceil(std::max(0, resource_node.get_ll_x() - track_start_line) / 1.0 / track_pitch);
          int32_t track_num = static_cast<int32_t>(end_track - start_track);

          resource_node.addSupply(EGRResourceType::kNorth, track_num);
          resource_node.addSupply(EGRResourceType::kSouth, track_num);
          resource_node.addSupply(EGRResourceType::kTrack, track_num);
        }
      }
    }
  }
}

void EGRDataManager::addResourceMapDemand()
{
  std::vector<GridMap<EGRNode>>& layer_resource_map = _egr_database.get_layer_resource_map();
  std::vector<RoutingLayer>& routing_layer_list = _egr_database.get_routing_layer_list();

  for (Obstacle& obstacle : _egr_database.get_routing_obstacle_list()) {
    RoutingLayer& routing_layer = routing_layer_list[obstacle.get_layer_idx()];
    GridMap<EGRNode>& resource_map = layer_resource_map[obstacle.get_layer_idx()];

    PlanarRect& obstacle_grid_rect = obstacle.get_grid_rect();
    for (int32_t x = obstacle_grid_rect.get_ll_x(); x <= obstacle_grid_rect.get_ur_x(); ++x) {
      for (int32_t y = obstacle_grid_rect.get_ll_y(); y <= obstacle_grid_rect.get_ur_y(); ++y) {
        EGRNode& resource_node = resource_map[x][y];
        int32_t real_ll_x = resource_node.get_ll_x();
        int32_t real_ll_y = resource_node.get_ll_y();
        int32_t real_ur_x = resource_node.get_ur_x();
        int32_t real_ur_y = resource_node.get_ur_y();

        if (routing_layer.isPreferH()) {
          PlanarRect east_rect((real_ll_x + real_ur_x) / 2, real_ll_y, real_ur_x, real_ur_y);
          double east_overlap_ratio = RTUtil::getOverlapRatio(east_rect, obstacle.get_real_rect());
          resource_node.addDemand(EGRResourceType::kEast, east_overlap_ratio * resource_node.get_east_supply());

          PlanarRect west_rect(real_ll_x, real_ll_y, (real_ll_x + real_ur_x) / 2, real_ur_y);
          double west_overlap_ratio = RTUtil::getOverlapRatio(west_rect, obstacle.get_real_rect());
          resource_node.addDemand(EGRResourceType::kWest, west_overlap_ratio * resource_node.get_west_supply());

          double track_overlap_ratio = (east_overlap_ratio + west_overlap_ratio) / 2;
          resource_node.addDemand(EGRResourceType::kTrack, track_overlap_ratio * resource_node.get_track_supply());
        } else {
          PlanarRect south_rect(real_ll_x, real_ll_y, real_ur_x, (real_ll_y + real_ur_y) / 2);
          double south_overlap_ratio = RTUtil::getOverlapRatio(south_rect, obstacle.get_real_rect());
          resource_node.addDemand(EGRResourceType::kSouth, south_overlap_ratio * resource_node.get_south_supply());

          PlanarRect north_rect(real_ll_x, (real_ll_y + real_ur_y) / 2, real_ur_x, real_ur_y);
          double north_overlap_ratio = RTUtil::getOverlapRatio(north_rect, obstacle.get_real_rect());
          resource_node.addDemand(EGRResourceType::kNorth, north_overlap_ratio * resource_node.get_north_supply());

          double track_overlap_ratio = (south_overlap_ratio + north_overlap_ratio) / 2;
          resource_node.addDemand(EGRResourceType::kTrack, track_overlap_ratio * resource_node.get_track_supply());
        }
      }
    }
  }
}

void EGRDataManager::legalizeResourceMapDemand()
{
  std::vector<GridMap<EGRNode>>& layer_resource_map = _egr_database.get_layer_resource_map();

  for (size_t layer_idx = 0; layer_idx < layer_resource_map.size(); ++layer_idx) {
    GridMap<EGRNode>& resource_map = layer_resource_map[layer_idx];
    for (int32_t x = 0; x < resource_map.get_x_size(); ++x) {
      for (int32_t y = 0; y < resource_map.get_y_size(); ++y) {
        EGRNode& resource_node = resource_map[x][y];
        for (EGRResourceType resource_type :
             {EGRResourceType::kTrack, EGRResourceType::kNorth, EGRResourceType::kSouth, EGRResourceType::kWest, EGRResourceType::kEast}) {
          double supply = resource_node.getSupply(resource_type);
          double demand = resource_node.getDemand(resource_type);
          resource_node.addSupply(resource_type, -1 * std::min(supply, demand));
          resource_node.addDemand(resource_type, -1 * demand);
        }
      }
    }
  }
}

void EGRDataManager::buildHVLayerIdxList()
{
  std::vector<RoutingLayer>& routing_layer_list = _egr_database.get_routing_layer_list();
  int32_t bottom_routing_layer_idx = _egr_config.bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = _egr_config.top_routing_layer_idx;
  std::vector<int32_t>& h_layer_idx_list = _egr_database.get_h_layer_idx_list();
  std::vector<int32_t>& v_layer_idx_list = _egr_database.get_v_layer_idx_list();

  for (RoutingLayer& routing_layer : routing_layer_list) {
    int32_t layer_idx = routing_layer.get_layer_idx();
    if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
      continue;
    }
    if (routing_layer.isPreferH()) {
      h_layer_idx_list.push_back(layer_idx);
    } else {
      v_layer_idx_list.push_back(layer_idx);
    }
  }
}

Direction EGRDataManager::getRTDirectionByDB(idb::IdbLayerDirection idb_direction)
{
  if (idb_direction == idb::IdbLayerDirection::kHorizontal) {
    return Direction::kHorizontal;
  } else if (idb_direction == idb::IdbLayerDirection::kVertical) {
    return Direction::kVertical;
  } else {
    return Direction::kOblique;
  }
}

int32_t EGRDataManager::getEGRRoutingLayerIndexByDB(int32_t db_layer_idx)
{
  std::map<int32_t, int32_t>& db_to_egr_routing_layer_idx_map = _egr_helper.get_db_to_egr_routing_layer_idx_map();
  if (!RTUtil::exist(db_to_egr_routing_layer_idx_map, db_layer_idx)) {
    LOG_INST.error(Loc::current(), "db_layer_idx not exist!");
  }
  return db_to_egr_routing_layer_idx_map[db_layer_idx];
}

int32_t EGRDataManager::getEGRCutLayerIndexByDB(int32_t db_layer_idx)
{
  std::map<int32_t, int32_t>& db_to_egr_cut_layer_idx_map = _egr_helper.get_db_to_egr_cut_layer_idx_map();
  if (!RTUtil::exist(db_to_egr_cut_layer_idx_map, db_layer_idx)) {
    LOG_INST.error(Loc::current(), "db_layer_idx not exist!");
  }
  return db_to_egr_cut_layer_idx_map[db_layer_idx];
}

PlanarRect EGRDataManager::getGridRect(PlanarRect& real_rect)
{
  PlanarRect grid_rect;
  grid_rect.set_ll_x(real_rect.get_ll_x() / _egr_config.cell_width);
  grid_rect.set_ll_y(real_rect.get_ll_y() / _egr_config.cell_height);

  int32_t ur_x = real_rect.get_ur_x() / _egr_config.cell_width;
  int32_t ur_y = real_rect.get_ur_y() / _egr_config.cell_height;
  if (real_rect.get_ur_x() % _egr_config.cell_width == 0) {
    ur_x = std::max(0, ur_x - 1);
  }
  if (real_rect.get_ur_y() % _egr_config.cell_height == 0) {
    ur_y = std::max(0, ur_y - 1);
  }
  grid_rect.set_ur_x(ur_x);
  grid_rect.set_ur_y(ur_y);
  return grid_rect;
}

void EGRDataManager::printConfig()
{  ////////////////////////////////////////////////
  LOG_INST.openLogFileStream(_egr_config.log_file_path);
  // ********** EGR ********** //
  LOG_INST.info(Loc::current(), "EGR_CONFIG");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "temp_directory_path");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _egr_config.temp_directory_path);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "thread_number");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _egr_config.thread_number);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "congestion_cell_x_pitch");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _egr_config.congestion_cell_x_pitch);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "cell_width");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _egr_config.cell_width);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "congestion_cell_y_pitch");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _egr_config.congestion_cell_y_pitch);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "cell_height");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _egr_config.cell_height);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "bottom_routing_layer");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _egr_config.bottom_routing_layer);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "bottom_routing_layer_idx");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _egr_config.bottom_routing_layer_idx);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "top_routing_layer");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _egr_config.top_routing_layer);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "top_routing_layer_idx");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _egr_config.top_routing_layer_idx);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "log_file_path");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _egr_config.log_file_path);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "accuracy");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _egr_config.accuracy);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "strategy");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _egr_config.strategy);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "skip_net_name_list");
  for (std::string net_name : _egr_config.skip_net_name_set) {
    LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), net_name);
  }

  ////////////////////////////////////////////////
}

void EGRDataManager::printDatabase()
{
  ////////////////////////////////////////////////
  // ********** EGR ********** //
  LOG_INST.info(Loc::current(), "EGR_DATABASE");
  // ********** Design Name ********** //
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "design_name");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _egr_database.get_design_name());
  // ********** Die ********** //
  Die& die = _egr_database.get_die();
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "die");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), "(", die.get_real_ll_x(), ",", die.get_real_ll_y(), ")-(", die.get_real_ur_x(),
                ",", die.get_real_ur_y(), ")");
  // ********** RoutingLayer ********** //
  std::vector<RoutingLayer>& routing_layer_list = _egr_database.get_routing_layer_list();
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "routing_layer_num");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), routing_layer_list.size());
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "routing_layer");
  std::string routing_layer_name_string;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    routing_layer_name_string += (routing_layer.get_layer_name() + " ");
  }
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), routing_layer_name_string);
  // ********** Routing Obstacle ********** //
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "routing_obstacle_num");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _egr_database.get_routing_obstacle_list().size());
  // ********** EGR Net ********** //
  std::vector<EGRNet>& egr_net_list = _egr_database.get_egr_net_list();
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "net_num");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), egr_net_list.size());
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "pin_num_ratio");

  size_t pin_num_upper_limit = 100;
  std::map<size_t, size_t> pin_net_map;
  for (EGRNet& egr_net : egr_net_list) {
    pin_net_map[std::min(egr_net.get_pin_list().size(), pin_num_upper_limit)]++;
  }
  for (auto [pin_num, net_num] : pin_net_map) {
    std::string head_info = "net with ";
    if (pin_num == pin_num_upper_limit) {
      head_info += ">=";
    }
    LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), head_info, pin_num, " pins: ", net_num, "(",
                  RTUtil::getPercentage(net_num, egr_net_list.size()), "%)");
  }
  // ********** Layer Resource Map ********** //
  std::vector<GridMap<EGRNode>>& layer_resource_map = _egr_database.get_layer_resource_map();
  size_t resource_map_num = layer_resource_map.size();
  int32_t x_size = layer_resource_map.front().get_x_size();
  int32_t y_size = layer_resource_map.front().get_y_size();
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "resource_map_num");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), resource_map_num);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "resource_map_x_size");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), x_size);
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "resource_map_y_size");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), y_size);
  // ******************** //
  ////////////////////////////////////////////////
}

}  // namespace irt