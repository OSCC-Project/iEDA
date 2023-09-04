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
  _egr_config.thread_number = RTUtil::getConfigValue<irt_int>(config_map, "-thread_number", 8);
  _egr_config.congestion_cell_x_pitch = RTUtil::getConfigValue<irt_int>(config_map, "-congestion_cell_x_pitch", 15);
  _egr_config.congestion_cell_y_pitch = RTUtil::getConfigValue<irt_int>(config_map, "-congestion_cell_y_pitch", 15);
  _egr_config.bottom_routing_layer = RTUtil::getConfigValue<std::string>(config_map, "-bottom_routing_layer", "");
  _egr_config.top_routing_layer = RTUtil::getConfigValue<std::string>(config_map, "-top_routing_layer", "");
  _egr_config.accuracy = RTUtil::getConfigValue<irt_int>(config_map, "-accuracy", 2);
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
  wrapBlockageList(idb_builder);
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
  die_box.set_real_lb(die->get_llx(), die->get_lly());
  die_box.set_real_rt(die->get_urx(), die->get_ury());
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
      routing_layer.set_direction(getRTDirectionByDB(idb_routing_layer->get_direction()));
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
    track_grid.set_start_line(static_cast<irt_int>(idb_track->get_start()));
    track_grid.set_step_length(static_cast<irt_int>(idb_track->get_pitch()));
    track_grid.set_step_num(static_cast<irt_int>(idb_track_grid->get_track_num()));

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
      cut_shape.set_lb(idb_rect->get_low_x(), idb_rect->get_low_y());
      cut_shape.set_rt(idb_rect->get_high_x(), idb_rect->get_high_y());
      cut_shape_list.push_back(std::move(cut_shape));
    }
    via_master.set_cut_layer_idx(idb_shape_cut.get_layer()->get_id());
    layer_via_master_list.front().push_back(std::move(via_master));
  }
}

void EGRDataManager::wrapBlockageList(idb::IdbBuilder* idb_builder)
{
  wrapArtificialBlockage(idb_builder);
  wrapInstanceBlockage(idb_builder);
  wrapSpecialNetBlockage(idb_builder);
}

void EGRDataManager::wrapArtificialBlockage(idb::IdbBuilder* idb_builder)
{
  // Artificial
  idb::IdbBlockageList* idb_blockage_list = idb_builder->get_def_service()->get_design()->get_blockage_list();
  if (!idb_blockage_list->get_blockage_list().empty()) {
    LOG_INST.warning(Loc::current(), "The artificial blockage will be ignored!");
  }

  // std::vector<Blockage>& routing_blockage_list = _database.get_routing_blockage_list();

  // LOG_INST.warning(Loc::current(), "The artificial blockage will be ignored!");

  // // Artificial
  // idb::IdbBlockageList* idb_blockage_list = idb_builder->get_def_service()->get_design()->get_blockage_list();
  // for (idb::IdbBlockage* idb_blockage : idb_blockage_list->get_blockage_list()) {
  //   if (idb_blockage->is_routing_blockage()) {
  //     idb::IdbRoutingBlockage* idb_routing_blockage = dynamic_cast<idb::IdbRoutingBlockage*>(idb_blockage);
  //     for (idb::IdbRect* rect : idb_routing_blockage->get_rect_list()) {
  //       Blockage blockage;
  //       blockage.set_real_lb(rect->get_low_x(), rect->get_low_y());
  //       blockage.set_real_rt(rect->get_high_x(), rect->get_high_y());
  //       blockage.set_layer_idx(idb_routing_blockage->get_layer()->get_id());
  //       blockage.set_is_artificial(true);
  //       routing_blockage_list.push_back(std::move(blockage));
  //     }
  //   }
  // }
}

void EGRDataManager::wrapInstanceBlockage(idb::IdbBuilder* idb_builder)
{
  std::vector<Blockage>& routing_blockage_list = _egr_database.get_routing_blockage_list();

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
      if (layer_shape->get_layer()->is_routing()) {
        routing_blockage_list.push_back(std::move(blockage));
      }
    }
  }
}

void EGRDataManager::wrapSpecialNetBlockage(idb::IdbBuilder* idb_builder)
{
  std::vector<Blockage>& routing_blockage_list = _egr_database.get_routing_blockage_list();

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
              Blockage blockage;
              blockage.set_real_lb(rect->get_low_x(), rect->get_low_y());
              blockage.set_real_rt(rect->get_high_x(), rect->get_high_y());
              blockage.set_layer_idx(layer_shape.get_layer()->get_id());
              if (layer_shape.get_layer()->is_routing()) {
                routing_blockage_list.push_back(std::move(blockage));
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
          routing_blockage_list.push_back(std::move(blockage));
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
  if (idb_net->get_io_pin() != nullptr) {
    idb::IdbPin* io_pin = idb_net->get_io_pin();
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
      pin_shape.set_real_lb(rect->get_low_x(), rect->get_low_y());
      pin_shape.set_real_rt(rect->get_high_x(), rect->get_high_y());
      pin_shape.set_layer_idx(layer_shape->get_layer()->get_id());
      if (layer_shape->get_layer()->is_routing()) {
        routing_shape_list.push_back(std::move(pin_shape));
      }
    }
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
  std::map<irt_int, irt_int>& db_to_egr_cut_layer_idx_map = _egr_helper.get_db_to_egr_cut_layer_idx_map();
  std::map<irt_int, irt_int>& db_to_egr_routing_layer_idx_map = _egr_helper.get_db_to_egr_routing_layer_idx_map();
  std::map<std::string, irt_int>& cut_layer_name_idx_map = _egr_helper.get_cut_layer_name_idx_map();
  std::map<std::string, irt_int>& routing_layer_name_idx_map = _egr_helper.get_routing_layer_name_idx_map();

  for (size_t i = 0; i < routing_layer_list.size(); ++i) {
    db_to_egr_routing_layer_idx_map[routing_layer_list[i].get_layer_idx()] = static_cast<irt_int>(i);
    routing_layer_name_idx_map[routing_layer_list[i].get_layer_name()] = static_cast<irt_int>(i);
  }

  std::vector<CutLayer>& cut_layer_list = _egr_database.get_cut_layer_list();
  for (size_t i = 0; i < cut_layer_list.size(); i++) {
    db_to_egr_cut_layer_idx_map[cut_layer_list[i].get_layer_idx()] = static_cast<irt_int>(i);
    cut_layer_name_idx_map[cut_layer_list[i].get_layer_name()] = static_cast<irt_int>(i);
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
  std::map<irt_int, irt_int> pitch_count_map;
  for (RoutingLayer& routing_layer : _egr_database.get_routing_layer_list()) {
    for (ScaleGrid& track_grid : routing_layer.getPreferTrackGridList()) {
      pitch_count_map[track_grid.get_step_length()]++;
    }
  }
  irt_int ref_pitch = -1;
  irt_int max_count = INT32_MIN;
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
  std::map<std::string, irt_int>& routing_layer_name_idx_map = _egr_helper.get_routing_layer_name_idx_map();
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
    LOG_INST.info(Loc::current(), "Optional strategys are 'topo' and 'gradual', use default strategy:gradual.");
  }
}

void EGRDataManager::buildDatabase()
{
  buildLayerList();
  buildLayerViaMasterList();
  buildDie();
  buildBlockageList();
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

    std::sort(via_master_list.begin(), via_master_list.end(), [&](ViaMaster& a, ViaMaster& b) {
      std::vector<RoutingLayer>& routing_layer_list = _egr_database.get_routing_layer_list();

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
      via_master_list[i].set_via_master_idx(layer_idx, i);
    }
  }
}

void EGRDataManager::buildDie()
{
  Die& die = _egr_database.get_die();
  die.set_grid_rect(getGridRect(die.get_real_rect()));
}

void EGRDataManager::buildBlockageList()
{
  std::vector<RoutingLayer>& routing_layer_list = _egr_database.get_routing_layer_list();
  irt_int die_real_rt_x = _egr_database.get_die().get_real_rt_x();
  irt_int die_real_rt_y = _egr_database.get_die().get_real_rt_y();

  for (Blockage& blockage : _egr_database.get_routing_blockage_list()) {
    irt_int layer_idx = getEGRRoutingLayerIndexByDB(blockage.get_layer_idx());
    irt_int half_wire_width = routing_layer_list[layer_idx].get_min_width() / 2;

    blockage.set_real_rect(RTUtil::getEnlargedRect(blockage.get_real_rect(), half_wire_width));
    if (blockage.get_real_rt_x() > die_real_rt_x) {
      blockage.get_real_rect().set_rt_x(die_real_rt_x);
    }
    if (blockage.get_real_rt_y() > die_real_rt_y) {
      blockage.get_real_rect().set_rt_y(die_real_rt_y);
    }
    blockage.set_grid_rect(getGridRect(blockage.get_real_rect()));
    blockage.set_layer_idx(layer_idx);
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
  irt_int die_rt_x = _egr_database.get_die().get_real_rt_x();
  irt_int die_rt_y = _egr_database.get_die().get_real_rt_y();
  for (size_t i = 0; i < pin_list.size(); i++) {
    EGRPin& egr_pin = pin_list[i];

    egr_pin.set_pin_idx(static_cast<irt_int>(i));
    for (EXTLayerRect& routing_shape : egr_pin.get_routing_shape_list()) {
      routing_shape.set_layer_idx(getEGRRoutingLayerIndexByDB(routing_shape.get_layer_idx()));
      // checkPinShape
      PlanarRect& real_rect = routing_shape.get_real_rect();
      PlanarRect new_rect = real_rect;
      new_rect.set_lb_x(std::min(new_rect.get_lb_x(), die_rt_x));
      new_rect.set_lb_y(std::min(new_rect.get_lb_y(), die_rt_y));
      new_rect.set_rt_x(std::min(new_rect.get_rt_x(), die_rt_x));
      new_rect.set_rt_y(std::min(new_rect.get_rt_y(), die_rt_y));
      if (real_rect != new_rect) {
        LOG_INST.warning(Loc::current(), "Pin:", egr_pin.get_pin_name(), "(", real_rect.get_lb_x(), ",", real_rect.get_lb_y(), ")---(",
                         real_rect.get_rt_x(), ",", real_rect.get_rt_y(), ")", " is out of die");
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
    for (irt_int x = 0; x < resource_map.get_x_size(); ++x) {
      for (irt_int y = 0; y < resource_map.get_y_size(); ++y) {
        resource_map[x][y].set_lb(_egr_config.cell_width * x, _egr_config.cell_height * y);
        resource_map[x][y].set_rt(_egr_config.cell_width * (x + 1), _egr_config.cell_height * (y + 1));
      }
    }
    for (irt_int x = 0; x < resource_map.get_x_size(); ++x) {
      resource_map[x][resource_map.get_y_size() - 1].set_rt_y(die.get_real_rt_y());
    }
    for (irt_int y = 0; y < resource_map.get_y_size(); ++y) {
      resource_map[resource_map.get_x_size() - 1][y].set_rt_x(die.get_real_rt_x());
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
    irt_int track_start_line = routing_layer.getPreferTrackGridList().front().get_start_line();
    irt_int track_pitch = routing_layer.getPreferTrackGridList().front().get_step_length();

    for (irt_int x = 0; x < resource_map.get_x_size(); ++x) {
      for (irt_int y = 0; y < resource_map.get_y_size(); ++y) {
        EGRNode& resource_node = resource_map[x][y];
        if (routing_layer.isPreferH()) {
          double end_track = std::ceil(std::max(0, resource_node.get_rt_y() - track_start_line) / 1.0 / track_pitch);
          double start_track = std::ceil(std::max(0, resource_node.get_lb_y() - track_start_line) / 1.0 / track_pitch);
          irt_int track_num = static_cast<irt_int>(end_track - start_track);

          resource_node.addSupply(EGRResourceType::kWest, track_num);
          resource_node.addSupply(EGRResourceType::kEast, track_num);
          resource_node.addSupply(EGRResourceType::kTrack, track_num);
        } else {
          double end_track = std::ceil(std::max(0, resource_node.get_rt_x() - track_start_line) / 1.0 / track_pitch);
          double start_track = std::ceil(std::max(0, resource_node.get_lb_x() - track_start_line) / 1.0 / track_pitch);
          irt_int track_num = static_cast<irt_int>(end_track - start_track);

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

  for (Blockage& blockage : _egr_database.get_routing_blockage_list()) {
    RoutingLayer& routing_layer = routing_layer_list[blockage.get_layer_idx()];
    GridMap<EGRNode>& resource_map = layer_resource_map[blockage.get_layer_idx()];

    PlanarRect& blockage_grid_rect = blockage.get_grid_rect();
    for (irt_int x = blockage_grid_rect.get_lb_x(); x <= blockage_grid_rect.get_rt_x(); ++x) {
      for (irt_int y = blockage_grid_rect.get_lb_y(); y <= blockage_grid_rect.get_rt_y(); ++y) {
        EGRNode& resource_node = resource_map[x][y];
        irt_int real_lb_x = resource_node.get_lb_x();
        irt_int real_lb_y = resource_node.get_lb_y();
        irt_int real_rt_x = resource_node.get_rt_x();
        irt_int real_rt_y = resource_node.get_rt_y();

        if (routing_layer.isPreferH()) {
          PlanarRect east_rect((real_lb_x + real_rt_x) / 2, real_lb_y, real_rt_x, real_rt_y);
          double east_overlap_ratio = RTUtil::getOverlapRatio(east_rect, blockage.get_real_rect());
          resource_node.addDemand(EGRResourceType::kEast, east_overlap_ratio * resource_node.get_east_supply());

          PlanarRect west_rect(real_lb_x, real_lb_y, (real_lb_x + real_rt_x) / 2, real_rt_y);
          double west_overlap_ratio = RTUtil::getOverlapRatio(west_rect, blockage.get_real_rect());
          resource_node.addDemand(EGRResourceType::kWest, west_overlap_ratio * resource_node.get_west_supply());

          double track_overlap_ratio = (east_overlap_ratio + west_overlap_ratio) / 2;
          resource_node.addDemand(EGRResourceType::kTrack, track_overlap_ratio * resource_node.get_track_supply());
        } else {
          PlanarRect south_rect(real_lb_x, real_lb_y, real_rt_x, (real_lb_y + real_rt_y) / 2);
          double south_overlap_ratio = RTUtil::getOverlapRatio(south_rect, blockage.get_real_rect());
          resource_node.addDemand(EGRResourceType::kSouth, south_overlap_ratio * resource_node.get_south_supply());

          PlanarRect north_rect(real_lb_x, (real_lb_y + real_rt_y) / 2, real_rt_x, real_rt_y);
          double north_overlap_ratio = RTUtil::getOverlapRatio(north_rect, blockage.get_real_rect());
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
    for (irt_int x = 0; x < resource_map.get_x_size(); ++x) {
      for (irt_int y = 0; y < resource_map.get_y_size(); ++y) {
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
  irt_int bottom_routing_layer_idx = _egr_config.bottom_routing_layer_idx;
  irt_int top_routing_layer_idx = _egr_config.top_routing_layer_idx;
  std::vector<irt_int>& h_layer_idx_list = _egr_database.get_h_layer_idx_list();
  std::vector<irt_int>& v_layer_idx_list = _egr_database.get_v_layer_idx_list();

  for (RoutingLayer& routing_layer : routing_layer_list) {
    irt_int layer_idx = routing_layer.get_layer_idx();
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

irt_int EGRDataManager::getEGRRoutingLayerIndexByDB(irt_int db_layer_idx)
{
  std::map<irt_int, irt_int>& db_to_egr_routing_layer_idx_map = _egr_helper.get_db_to_egr_routing_layer_idx_map();
  if (!RTUtil::exist(db_to_egr_routing_layer_idx_map, db_layer_idx)) {
    LOG_INST.error(Loc::current(), "db_layer_idx not exist!");
  }
  return db_to_egr_routing_layer_idx_map[db_layer_idx];
}

irt_int EGRDataManager::getEGRCutLayerIndexByDB(irt_int db_layer_idx)
{
  std::map<irt_int, irt_int>& db_to_egr_cut_layer_idx_map = _egr_helper.get_db_to_egr_cut_layer_idx_map();
  if (!RTUtil::exist(db_to_egr_cut_layer_idx_map, db_layer_idx)) {
    LOG_INST.error(Loc::current(), "db_layer_idx not exist!");
  }
  return db_to_egr_cut_layer_idx_map[db_layer_idx];
}

PlanarRect EGRDataManager::getGridRect(PlanarRect& real_rect)
{
  PlanarRect grid_rect;
  grid_rect.set_lb_x(real_rect.get_lb_x() / _egr_config.cell_width);
  grid_rect.set_lb_y(real_rect.get_lb_y() / _egr_config.cell_height);

  irt_int rt_x = real_rect.get_rt_x() / _egr_config.cell_width;
  irt_int rt_y = real_rect.get_rt_y() / _egr_config.cell_height;
  if (real_rect.get_rt_x() % _egr_config.cell_width == 0) {
    rt_x = std::max(0, rt_x - 1);
  }
  if (real_rect.get_rt_y() % _egr_config.cell_height == 0) {
    rt_y = std::max(0, rt_y - 1);
  }
  grid_rect.set_rt_x(rt_x);
  grid_rect.set_rt_y(rt_y);
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
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), "(", die.get_real_lb_x(), ",", die.get_real_lb_y(), ")-(", die.get_real_rt_x(),
                ",", die.get_real_rt_y(), ")");
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
  // ********** Routing Blockage ********** //
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(1), "routing_blockage_num");
  LOG_INST.info(Loc::current(), RTUtil::getSpaceByTabNum(2), _egr_database.get_routing_blockage_list().size());
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
  irt_int x_size = layer_resource_map.front().get_x_size();
  irt_int y_size = layer_resource_map.front().get_y_size();
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