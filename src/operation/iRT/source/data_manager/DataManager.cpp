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

#include "Monitor.hpp"
#include "RTHeader.hpp"
#include "RTInterface.hpp"
#include "Utility.hpp"

namespace irt {

// public

void DataManager::initInst()
{
  if (_dm_instance == nullptr) {
    _dm_instance = new DataManager();
  }
}

DataManager& DataManager::getInst()
{
  if (_dm_instance == nullptr) {
    RTLOG.error(Loc::current(), "The instance not initialized!");
  }
  return *_dm_instance;
}

void DataManager::destroyInst()
{
  if (_dm_instance != nullptr) {
    delete _dm_instance;
    _dm_instance = nullptr;
  }
}

// function

void DataManager::input(std::map<std::string, std::any>& config_map, idb::IdbBuilder* idb_builder)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");
  wrapConfig(config_map);
  wrapDatabase(idb_builder);
  buildConfig();
  buildDatabase();
  printConfig();
  printDatabase();
  writePYScript();
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void DataManager::output()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");
  outputTrackGrid();
  outputGCellGrid();
  outputNetList();
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

#if 1  // 更新GCellMap

void DataManager::updateFixedRectToGCellMap(ChangeType change_type, int32_t net_idx, EXTLayerRect* ext_layer_rect, bool is_routing)
{
  GridMap<GCell>& gcell_map = _database.get_gcell_map();

  for (int32_t x = ext_layer_rect->get_grid_ll_x(); x <= ext_layer_rect->get_grid_ur_x(); x++) {
    for (int32_t y = ext_layer_rect->get_grid_ll_y(); y <= ext_layer_rect->get_grid_ur_y(); y++) {
      auto& net_fixed_rect_map = gcell_map[x][y].get_type_layer_net_fixed_rect_map()[is_routing][ext_layer_rect->get_layer_idx()];
      if (change_type == ChangeType::kAdd) {
        net_fixed_rect_map[net_idx].insert(ext_layer_rect);
      } else {
        net_fixed_rect_map[net_idx].erase(ext_layer_rect);
        if (net_fixed_rect_map[net_idx].empty()) {
          net_fixed_rect_map.erase(net_idx);
        }
      }
    }
  }
  if (change_type == ChangeType::kDel) {
    // 由于在database内的obstacle_list引用过来，所以不需要delete，也不能delete
  }
}

void DataManager::updateAccessPointToGCellMap(ChangeType change_type, int32_t net_idx, AccessPoint* access_point)
{
  GridMap<GCell>& gcell_map = _database.get_gcell_map();

  auto& net_access_point_map = gcell_map[access_point->get_grid_x()][access_point->get_grid_y()].get_net_access_point_map();
  if (change_type == ChangeType::kAdd) {
    net_access_point_map[net_idx].insert(access_point);
  } else {
    net_access_point_map[net_idx].erase(access_point);
    if (net_access_point_map[net_idx].empty()) {
      net_access_point_map.erase(net_idx);
    }
  }
  if (change_type == ChangeType::kDel) {
    // 由于在pin内的access_point_list引用过来，所以不需要delete，也不能delete
  }
}

void DataManager::updateGlobalNetResultToGCellMap(ChangeType change_type, int32_t net_idx, Segment<LayerCoord>* segment)
{
  GridMap<GCell>& gcell_map = _database.get_gcell_map();

  LayerCoord& first_coord = segment->get_first();
  LayerCoord& second_coord = segment->get_second();

  int32_t first_x = first_coord.get_x();
  int32_t first_y = first_coord.get_y();
  int32_t second_x = second_coord.get_x();
  int32_t second_y = second_coord.get_y();
  RTUTIL.swapByASC(first_x, second_x);
  RTUTIL.swapByASC(first_y, second_y);

  for (int32_t x = first_x; x <= second_x; x++) {
    for (int32_t y = first_y; y <= second_y; y++) {
      auto& global_net_result_map = gcell_map[x][y].get_global_net_result_map();
      if (change_type == ChangeType::kAdd) {
        global_net_result_map[net_idx].insert(segment);
      } else {
        global_net_result_map[net_idx].erase(segment);
        if (global_net_result_map[net_idx].empty()) {
          global_net_result_map.erase(net_idx);
        }
      }
    }
  }
  if (change_type == ChangeType::kDel) {
    delete segment;
    segment = nullptr;
  }
}

void DataManager::updateDetailedNetResultToGCellMap(ChangeType change_type, int32_t net_idx, Segment<LayerCoord>* segment)
{
  ScaleAxis& gcell_axis = _database.get_gcell_axis();
  GridMap<GCell>& gcell_map = _database.get_gcell_map();

  PlanarRect grid_rect = RTUTIL.getClosedGCellGridRect(*segment, gcell_axis);

  for (int32_t x = grid_rect.get_ll_x(); x <= grid_rect.get_ur_x(); x++) {
    for (int32_t y = grid_rect.get_ll_y(); y <= grid_rect.get_ur_y(); y++) {
      auto& detailed_net_result_map = gcell_map[x][y].get_detailed_net_result_map();
      if (change_type == ChangeType::kAdd) {
        detailed_net_result_map[net_idx].insert(segment);
      } else {
        detailed_net_result_map[net_idx].erase(segment);
        if (detailed_net_result_map[net_idx].empty()) {
          detailed_net_result_map.erase(net_idx);
        }
      }
    }
  }
  if (change_type == ChangeType::kDel) {
    delete segment;
    segment = nullptr;
  }
}

void DataManager::updateNetPatchToGCellMap(ChangeType change_type, int32_t net_idx, EXTLayerRect* ext_layer_rect)
{
  GridMap<GCell>& gcell_map = _database.get_gcell_map();

  for (int32_t x = ext_layer_rect->get_grid_ll_x(); x <= ext_layer_rect->get_grid_ur_x(); x++) {
    for (int32_t y = ext_layer_rect->get_grid_ll_y(); y <= ext_layer_rect->get_grid_ur_y(); y++) {
      auto& net_patch_map = gcell_map[x][y].get_net_patch_map();
      if (change_type == ChangeType::kAdd) {
        net_patch_map[net_idx].insert(ext_layer_rect);
      } else {
        net_patch_map[net_idx].erase(ext_layer_rect);
        if (net_patch_map[net_idx].empty()) {
          net_patch_map.erase(net_idx);
        }
      }
    }
  }
  if (change_type == ChangeType::kDel) {
    delete ext_layer_rect;
    ext_layer_rect = nullptr;
  }
}

void DataManager::updateViolationToGCellMap(ChangeType change_type, Violation* violation)
{
  GridMap<GCell>& gcell_map = _database.get_gcell_map();

  PlanarRect& grid_rect = violation->get_violation_shape().get_grid_rect();

  for (int32_t x = grid_rect.get_ll_x(); x <= grid_rect.get_ur_x(); x++) {
    for (int32_t y = grid_rect.get_ll_y(); y <= grid_rect.get_ur_y(); y++) {
      GCell& gcell = gcell_map[x][y];
      if (change_type == ChangeType::kAdd) {
        gcell.get_violation_set().insert(violation);
      } else {
        gcell.get_violation_set().erase(violation);
      }
    }
  }
  if (change_type == ChangeType::kDel) {
    delete violation;
    violation = nullptr;
  }
}

std::map<bool, std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>>> DataManager::getTypeLayerNetFixedRectMap(
    EXTPlanarRect& region)
{
  GridMap<GCell>& gcell_map = _database.get_gcell_map();

  std::map<bool, std::map<int32_t, std::map<int32_t, std::set<EXTLayerRect*>>>> type_layer_net_fixed_rect_map;
  for (int32_t x = region.get_grid_ll_x(); x <= region.get_grid_ur_x(); x++) {
    for (int32_t y = region.get_grid_ll_y(); y <= region.get_grid_ur_y(); y++) {
      for (auto& [is_routing, layer_net_fixed_rect_map] : gcell_map[x][y].get_type_layer_net_fixed_rect_map()) {
        for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
          for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
            type_layer_net_fixed_rect_map[is_routing][layer_idx][net_idx].insert(fixed_rect_set.begin(), fixed_rect_set.end());
          }
        }
      }
    }
  }
  return type_layer_net_fixed_rect_map;
}

std::map<int32_t, std::set<AccessPoint*>> DataManager::getNetAccessPointMap(EXTPlanarRect& region)
{
  GridMap<GCell>& gcell_map = _database.get_gcell_map();

  std::map<int32_t, std::set<AccessPoint*>> net_access_point_map;
  for (int32_t x = region.get_grid_ll_x(); x <= region.get_grid_ur_x(); x++) {
    for (int32_t y = region.get_grid_ll_y(); y <= region.get_grid_ur_y(); y++) {
      for (auto& [net_idx, access_point_set] : gcell_map[x][y].get_net_access_point_map()) {
        net_access_point_map[net_idx].insert(access_point_set.begin(), access_point_set.end());
      }
    }
  }
  return net_access_point_map;
}

std::map<int32_t, std::set<Segment<LayerCoord>*>> DataManager::getGlobalNetResultMap(EXTPlanarRect& region)
{
  GridMap<GCell>& gcell_map = _database.get_gcell_map();

  std::map<int32_t, std::set<Segment<LayerCoord>*>> global_net_result_map;
  for (int32_t x = region.get_grid_ll_x(); x <= region.get_grid_ur_x(); x++) {
    for (int32_t y = region.get_grid_ll_y(); y <= region.get_grid_ur_y(); y++) {
      for (auto& [net_idx, result_set] : gcell_map[x][y].get_global_net_result_map()) {
        global_net_result_map[net_idx].insert(result_set.begin(), result_set.end());
      }
    }
  }
  return global_net_result_map;
}

std::map<int32_t, std::set<Segment<LayerCoord>*>> DataManager::getDetailedNetResultMap(EXTPlanarRect& region)
{
  GridMap<GCell>& gcell_map = _database.get_gcell_map();

  std::map<int32_t, std::set<Segment<LayerCoord>*>> detailed_net_result_map;
  for (int32_t x = region.get_grid_ll_x(); x <= region.get_grid_ur_x(); x++) {
    for (int32_t y = region.get_grid_ll_y(); y <= region.get_grid_ur_y(); y++) {
      for (auto& [net_idx, result_set] : gcell_map[x][y].get_detailed_net_result_map()) {
        detailed_net_result_map[net_idx].insert(result_set.begin(), result_set.end());
      }
    }
  }
  return detailed_net_result_map;
}

std::map<int32_t, std::set<EXTLayerRect*>> DataManager::getNetPatchMap(EXTPlanarRect& region)
{
  GridMap<GCell>& gcell_map = _database.get_gcell_map();

  std::map<int32_t, std::set<EXTLayerRect*>> net_patch_map;
  for (int32_t x = region.get_grid_ll_x(); x <= region.get_grid_ur_x(); x++) {
    for (int32_t y = region.get_grid_ll_y(); y <= region.get_grid_ur_y(); y++) {
      for (auto& [net_idx, patch_set] : gcell_map[x][y].get_net_patch_map()) {
        net_patch_map[net_idx].insert(patch_set.begin(), patch_set.end());
      }
    }
  }
  return net_patch_map;
}

std::set<Violation*> DataManager::getViolationSet(EXTPlanarRect& region)
{
  GridMap<GCell>& gcell_map = _database.get_gcell_map();

  std::set<Violation*> violation_set;
  for (int32_t x = region.get_grid_ll_x(); x <= region.get_grid_ur_x(); x++) {
    for (int32_t y = region.get_grid_ll_y(); y <= region.get_grid_ur_y(); y++) {
      violation_set.insert(gcell_map[x][y].get_violation_set().begin(), gcell_map[x][y].get_violation_set().end());
    }
  }
  return violation_set;
}

#endif

#if 1  // 获得NetShapeList

std::vector<NetShape> DataManager::getNetShapeList(int32_t net_idx, std::vector<Segment<LayerCoord>>& segment_list)
{
  std::vector<NetShape> drc_shape_list;
  for (Segment<LayerCoord>& segment : segment_list) {
    for (NetShape& drc_shape : getNetShapeList(net_idx, segment)) {
      drc_shape_list.push_back(drc_shape);
    }
  }
  return drc_shape_list;
}

std::vector<NetShape> DataManager::getNetShapeList(int32_t net_idx, Segment<LayerCoord>& segment)
{
  std::vector<NetShape> drc_shape_list;
  for (NetShape& drc_shape : getNetShapeList(net_idx, segment.get_first(), segment.get_second())) {
    drc_shape_list.push_back(drc_shape);
  }
  return drc_shape_list;
}

std::vector<NetShape> DataManager::getNetShapeList(int32_t net_idx, MTree<LayerCoord>& coord_tree)
{
  std::vector<NetShape> drc_shape_list;
  for (Segment<TNode<LayerCoord>*>& coord_segment : RTUTIL.getSegListByTree(coord_tree)) {
    for (NetShape& drc_shape : getNetShapeList(net_idx, coord_segment.get_first()->value(), coord_segment.get_second()->value())) {
      drc_shape_list.push_back(drc_shape);
    }
  }
  return drc_shape_list;
}

std::vector<NetShape> DataManager::getNetShapeList(int32_t net_idx, LayerCoord& first_coord, LayerCoord& second_coord)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();

  std::vector<NetShape> drc_shape_list;
  int32_t first_layer_idx = first_coord.get_layer_idx();
  int32_t second_layer_idx = second_coord.get_layer_idx();
  if (first_layer_idx != second_layer_idx) {
    RTUTIL.swapByASC(first_layer_idx, second_layer_idx);
    for (int32_t layer_idx = first_layer_idx; layer_idx < second_layer_idx; layer_idx++) {
      ViaMaster& via_master = layer_via_master_list[layer_idx].front();

      LayerRect& above_enclosure = via_master.get_above_enclosure();
      LayerRect offset_above_enclosure(RTUTIL.getOffsetRect(above_enclosure, first_coord), above_enclosure.get_layer_idx());
      drc_shape_list.emplace_back(net_idx, offset_above_enclosure, true);

      LayerRect& below_enclosure = via_master.get_below_enclosure();
      LayerRect offset_below_enclosure(RTUTIL.getOffsetRect(below_enclosure, first_coord), below_enclosure.get_layer_idx());
      drc_shape_list.emplace_back(net_idx, offset_below_enclosure, true);

      for (PlanarRect& cut_shape : via_master.get_cut_shape_list()) {
        LayerRect offset_cut_shape(RTUTIL.getOffsetRect(cut_shape, first_coord), via_master.get_cut_layer_idx());
        drc_shape_list.emplace_back(net_idx, offset_cut_shape, false);
      }
    }
  } else {
    int32_t half_width = routing_layer_list[first_layer_idx].get_min_width() / 2;
    LayerRect wire_rect(RTUTIL.getEnlargedRect(first_coord, second_coord, half_width), first_layer_idx);
    drc_shape_list.emplace_back(net_idx, wire_rect, true);
  }
  return drc_shape_list;
}

#endif

#if 1  // 获得IdbSegment

idb::IdbLayerShape* DataManager::getIDBLayerShapeByFixedRect(EXTLayerRect* fixed_rect, bool is_routing)
{
  idb::IdbLayers* idb_layer_list = _database.get_idb_builder()->get_def_service()->get_layout()->get_layers();
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = _database.get_cut_layer_list();

  std::string layer_name;
  if (is_routing) {
    layer_name = routing_layer_list[fixed_rect->get_layer_idx()].get_layer_name();
  } else {
    layer_name = cut_layer_list[fixed_rect->get_layer_idx()].get_layer_name();
  }
  idb::IdbLayer* idb_layer = idb_layer_list->find_layer(layer_name);
  if (idb_layer == nullptr) {
    RTLOG.error(Loc::current(), "Can not find idb layer ", layer_name);
  }
  PlanarRect& real_rect = fixed_rect->get_real_rect();

  idb::IdbLayerShape* idb_shape = new idb::IdbLayerShape();
  idb_shape->set_type_rect();
  idb_shape->add_rect(real_rect.get_ll_x(), real_rect.get_ll_y(), real_rect.get_ur_x(), real_rect.get_ur_y());
  idb_shape->set_layer(idb_layer);
  return idb_shape;
}

idb::IdbRegularWireSegment* DataManager::getIDBSegmentByNetResult(int32_t net_idx, Segment<LayerCoord>& segment)
{
  if (segment.get_first().get_layer_idx() == segment.get_second().get_layer_idx()) {
    return getIDBWire(net_idx, segment);
  } else {
    return getIDBVia(net_idx, segment);
  }
}

idb::IdbRegularWireSegment* DataManager::getIDBSegmentByNetPatch(int32_t net_idx, EXTLayerRect& ext_layer_rect)
{
  idb::IdbLayers* idb_layer_list = _database.get_idb_builder()->get_def_service()->get_layout()->get_layers();
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();

  std::string layer_name = routing_layer_list[ext_layer_rect.get_layer_idx()].get_layer_name();
  idb::IdbLayer* idb_layer = idb_layer_list->find_layer(layer_name);
  if (idb_layer == nullptr) {
    RTLOG.error(Loc::current(), "Can not find idb layer ", layer_name);
  }
  PlanarRect& real_rect = ext_layer_rect.get_real_rect();

  idb::IdbRegularWireSegment* idb_segment = new idb::IdbRegularWireSegment();
  idb_segment->set_layer(idb_layer);
  idb_segment->set_is_rect(true);
  idb_segment->add_point(real_rect.get_ll_x(), real_rect.get_ll_y());
  idb_segment->set_delta_rect(0, 0, real_rect.get_ur_x() - real_rect.get_ll_x(), real_rect.get_ur_y() - real_rect.get_ll_y());
  return idb_segment;
}

#endif

// private

DataManager* DataManager::_dm_instance = nullptr;

#if 1  // input

void DataManager::wrapConfig(std::map<std::string, std::any>& config_map)
{
  /////////////////////////////////////////////
  _config.temp_directory_path = RTUTIL.getConfigValue<std::string>(config_map, "-temp_directory_path", "./rt_temp_directory");
  _config.thread_number = RTUTIL.getConfigValue<int32_t>(config_map, "-thread_number", 128);
  omp_set_num_threads(std::max(_config.thread_number, 1));
  _config.bottom_routing_layer = RTUTIL.getConfigValue<std::string>(config_map, "-bottom_routing_layer", "");
  _config.top_routing_layer = RTUTIL.getConfigValue<std::string>(config_map, "-top_routing_layer", "");
  _config.output_csv = RTUTIL.getConfigValue<int32_t>(config_map, "-output_csv", 0);
  _config.enable_timing = RTUTIL.getConfigValue<int32_t>(config_map, "-enable_timing", 0);
  /////////////////////////////////////////////
}

void DataManager::wrapDatabase(idb::IdbBuilder* idb_builder)
{
  wrapDBInfo(idb_builder);
  wrapMicronDBU();
  wrapDie();
  wrapRow();
  wrapLayerList();
  wrapLayerInfo();
  wrapLayerViaMasterList();
  wrapObstacleList();
  wrapNetList();
}

void DataManager::wrapDBInfo(idb::IdbBuilder* idb_builder)
{
  _database.set_idb_builder(idb_builder);
  _database.set_design_name(idb_builder->get_def_service()->get_design()->get_design_name());
  _database.set_lef_file_path_list(idb_builder->get_lef_service()->get_lef_files());
  _database.set_def_file_path(idb_builder->get_def_service()->get_def_file());
}

void DataManager::wrapMicronDBU()
{
  _database.set_micron_dbu(_database.get_idb_builder()->get_def_service()->get_design()->get_units()->get_micron_dbu());
}

void DataManager::wrapDie()
{
  idb::IdbDie* die = _database.get_idb_builder()->get_lef_service()->get_layout()->get_die();

  EXTPlanarRect& die_box = _database.get_die();
  die_box.set_real_ll(die->get_llx(), die->get_lly());
  die_box.set_real_ur(die->get_urx(), die->get_ury());
}

void DataManager::wrapRow()
{
  int32_t start_x = INT32_MAX;
  int32_t start_y = INT32_MAX;
  for (idb::IdbRow* idb_row : _database.get_idb_builder()->get_def_service()->get_layout()->get_rows()->get_row_list()) {
    start_x = std::min(start_x, idb_row->get_original_coordinate()->get_x());
    start_y = std::min(start_y, idb_row->get_original_coordinate()->get_y());
  }
  Row& row = _database.get_row();
  row.set_start_x(start_x);
  row.set_start_y(start_y);
  row.set_height(_database.get_idb_builder()->get_def_service()->get_layout()->get_rows()->get_row_height());
}

void DataManager::wrapLayerList()
{
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = _database.get_cut_layer_list();

  std::vector<idb::IdbLayer*>& idb_layers = _database.get_idb_builder()->get_lef_service()->get_layout()->get_layers()->get_layers();
  for (idb::IdbLayer* idb_layer : idb_layers) {
    if (idb_layer->is_routing()) {
      idb::IdbLayerRouting* idb_routing_layer = dynamic_cast<idb::IdbLayerRouting*>(idb_layer);
      RoutingLayer routing_layer;
      routing_layer.set_layer_idx(idb_routing_layer->get_id());
      routing_layer.set_layer_order(idb_routing_layer->get_order());
      routing_layer.set_layer_name(idb_routing_layer->get_name());
      routing_layer.set_min_width(idb_routing_layer->get_min_width());
      routing_layer.set_min_area(idb_routing_layer->get_area());
      routing_layer.set_prefer_direction(getRTDirectionByDB(idb_routing_layer->get_direction()));
      wrapTrackAxis(routing_layer, idb_routing_layer);
      wrapSpacingTable(routing_layer, idb_routing_layer);
      routing_layer_list.push_back(std::move(routing_layer));
    } else if (idb_layer->is_cut()) {
      idb::IdbLayerCut* idb_cut_layer = dynamic_cast<idb::IdbLayerCut*>(idb_layer);
      CutLayer cut_layer;
      cut_layer.set_layer_idx(idb_cut_layer->get_id());
      cut_layer.set_layer_order(idb_cut_layer->get_order());
      cut_layer.set_layer_name(idb_cut_layer->get_name());
      cut_layer.set_spacing(0);
      cut_layer_list.push_back(std::move(cut_layer));
    }
  }
}

void DataManager::wrapTrackAxis(RoutingLayer& routing_layer, idb::IdbLayerRouting* idb_layer)
{
  ScaleAxis& track_axis = routing_layer.get_track_axis();

  for (idb::IdbTrackGrid* idb_track_grid : idb_layer->get_track_grid_list()) {
    idb::IdbTrack* idb_track = idb_track_grid->get_track();

    ScaleGrid track_grid;
    track_grid.set_start_line(static_cast<int32_t>(idb_track->get_start()));
    track_grid.set_step_length(static_cast<int32_t>(idb_track->get_pitch()));
    track_grid.set_step_num(static_cast<int32_t>(idb_track_grid->get_track_num()) - 1);

    if (idb_track->get_direction() == idb::IdbTrackDirection::kDirectionX) {
      track_axis.get_x_grid_list().push_back(track_grid);
    } else if (idb_track->get_direction() == idb::IdbTrackDirection::kDirectionY) {
      track_axis.get_y_grid_list().push_back(track_grid);
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
    RTLOG.error(Loc::current(), "The idb spacing table is error!");
  }

  SpacingTable& spacing_table = routing_layer.get_spacing_table();
  std::vector<int32_t>& width_list = spacing_table.get_width_list();
  std::vector<int32_t>& parallel_length_list = spacing_table.get_parallel_length_list();
  GridMap<int32_t>& width_parallel_length_map = spacing_table.get_width_parallel_length_map();

  width_list = idb_spacing_table->get_width_list();
  parallel_length_list = idb_spacing_table->get_parallel_length_list();
  width_parallel_length_map.init(width_list.size(), parallel_length_list.size());
  for (int32_t x = 0; x < width_parallel_length_map.get_x_size(); x++) {
    for (int32_t y = 0; y < width_parallel_length_map.get_y_size(); y++) {
      width_parallel_length_map[x][y] = idb_spacing_table->get_spacing_table()[x][y];
    }
  }
}

void DataManager::wrapLayerInfo()
{
  std::map<int32_t, int32_t>& routing_idb_layer_id_to_idx_map = _database.get_routing_idb_layer_id_to_idx_map();
  std::map<int32_t, int32_t>& cut_idb_layer_id_to_idx_map = _database.get_cut_idb_layer_id_to_idx_map();
  std::map<std::string, int32_t>& routing_layer_name_to_idx_map = _database.get_routing_layer_name_to_idx_map();
  std::map<std::string, int32_t>& cut_layer_name_to_idx_map = _database.get_cut_layer_name_to_idx_map();

  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();
  for (size_t i = 0; i < routing_layer_list.size(); i++) {
    routing_idb_layer_id_to_idx_map[routing_layer_list[i].get_layer_idx()] = static_cast<int32_t>(i);
    routing_layer_name_to_idx_map[routing_layer_list[i].get_layer_name()] = static_cast<int32_t>(i);
  }
  std::vector<CutLayer>& cut_layer_list = _database.get_cut_layer_list();
  for (size_t i = 0; i < cut_layer_list.size(); i++) {
    cut_idb_layer_id_to_idx_map[cut_layer_list[i].get_layer_idx()] = static_cast<int32_t>(i);
    cut_layer_name_to_idx_map[cut_layer_list[i].get_layer_name()] = static_cast<int32_t>(i);
  }
}

void DataManager::wrapLayerViaMasterList()
{
  idb::IdbVias* idb_via_list_lib = _database.get_idb_builder()->get_lef_service()->get_layout()->get_via_list();
  if (idb_via_list_lib == nullptr) {
    RTLOG.error(Loc::current(), "Via list in tech lef is empty!");
  }

  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _database.get_layer_via_master_list();
  std::vector<idb::IdbLayer*> idb_routing_layers
      = _database.get_idb_builder()->get_lef_service()->get_layout()->get_layers()->get_routing_layers();
  layer_via_master_list.resize(idb_routing_layers.size());

  std::vector<idb::IdbVia*>& idb_via_list = idb_via_list_lib->get_via_list();
  for (size_t i = 0; i < idb_via_list.size(); i++) {
    idb::IdbVia* idb_via = idb_via_list[i];
    if (idb_via == nullptr) {
      RTLOG.error(Loc::current(), "The via is empty!");
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

void DataManager::wrapObstacleList()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<Obstacle>& routing_obstacle_list = _database.get_routing_obstacle_list();
  std::vector<Obstacle>& cut_obstacle_list = _database.get_cut_obstacle_list();
  std::vector<idb::IdbInstance*>& instance_list
      = _database.get_idb_builder()->get_def_service()->get_design()->get_instance_list()->get_instance_list();
  idb::IdbSpecialNetList* idb_special_net_list = _database.get_idb_builder()->get_def_service()->get_design()->get_special_net_list();

  int32_t total_routing_obstacle_num = 0;
  int32_t total_cut_obstacle_num = 0;
  {
    // instance
    for (idb::IdbInstance* instance : instance_list) {
      for (idb::IdbLayerShape* obs_box : instance->get_obs_box_list()) {
        if (obs_box->get_layer()->is_routing()) {
          total_routing_obstacle_num += obs_box->get_rect_list().size();
        } else if (obs_box->get_layer()->is_cut()) {
          total_cut_obstacle_num += obs_box->get_rect_list().size();
        }
      }
      for (idb::IdbPin* idb_pin : instance->get_pin_list()->get_pin_list()) {
        if (idb_pin->get_net() != nullptr) {
          continue;
        }
        for (idb::IdbLayerShape* port_box : idb_pin->get_port_box_list()) {
          if (port_box->get_layer()->is_routing()) {
            total_routing_obstacle_num += port_box->get_rect_list().size();
          } else if (port_box->get_layer()->is_cut()) {
            total_cut_obstacle_num += port_box->get_rect_list().size();
          }
        }
      }
    }
    // special net
    for (idb::IdbSpecialNet* idb_net : idb_special_net_list->get_net_list()) {
      for (idb::IdbSpecialWire* idb_wire : idb_net->get_wire_list()->get_wire_list()) {
        for (idb::IdbSpecialWireSegment* idb_segment : idb_wire->get_segment_list()) {
          if (idb_segment->is_via()) {
            total_routing_obstacle_num += idb_segment->get_via()->get_top_layer_shape().get_rect_list().size();
            total_routing_obstacle_num += idb_segment->get_via()->get_bottom_layer_shape().get_rect_list().size();
            total_cut_obstacle_num += idb_segment->get_via()->get_cut_layer_shape().get_rect_list().size();
          } else {
            total_routing_obstacle_num += 1;
          }
        }
      }
    }
  }
  routing_obstacle_list.reserve(total_routing_obstacle_num);
  cut_obstacle_list.reserve(total_cut_obstacle_num);
  {
    // instance
    for (idb::IdbInstance* instance : instance_list) {
      // instance obs
      for (idb::IdbLayerShape* obs_box : instance->get_obs_box_list()) {
        for (idb::IdbRect* rect : obs_box->get_rect_list()) {
          Obstacle obstacle;
          obstacle.set_real_ll(rect->get_low_x(), rect->get_low_y());
          obstacle.set_real_ur(rect->get_high_x(), rect->get_high_y());
          obstacle.set_layer_idx(obs_box->get_layer()->get_id());
          if (obs_box->get_layer()->is_routing()) {
            routing_obstacle_list.push_back(std::move(obstacle));
          } else if (obs_box->get_layer()->is_cut()) {
            cut_obstacle_list.push_back(std::move(obstacle));
          }
        }
      }
      // instance pin without net
      for (idb::IdbPin* idb_pin : instance->get_pin_list()->get_pin_list()) {
        if (idb_pin->get_net() != nullptr) {
          continue;
        }
        for (idb::IdbLayerShape* port_box : idb_pin->get_port_box_list()) {
          for (idb::IdbRect* rect : port_box->get_rect_list()) {
            Obstacle obstacle;
            obstacle.set_real_ll(rect->get_low_x(), rect->get_low_y());
            obstacle.set_real_ur(rect->get_high_x(), rect->get_high_y());
            obstacle.set_layer_idx(port_box->get_layer()->get_id());
            if (port_box->get_layer()->is_routing()) {
              routing_obstacle_list.push_back(std::move(obstacle));
            } else if (port_box->get_layer()->is_cut()) {
              cut_obstacle_list.push_back(std::move(obstacle));
            }
          }
        }
      }
    }
    // special net
    for (idb::IdbSpecialNet* idb_net : idb_special_net_list->get_net_list()) {
      for (idb::IdbSpecialWire* idb_wire : idb_net->get_wire_list()->get_wire_list()) {
        for (idb::IdbSpecialWireSegment* idb_segment : idb_wire->get_segment_list()) {
          if (idb_segment->is_via()) {
            for (idb::IdbLayerShape layer_shape :
                 {idb_segment->get_via()->get_top_layer_shape(), idb_segment->get_via()->get_bottom_layer_shape()}) {
              for (idb::IdbRect* rect : layer_shape.get_rect_list()) {
                Obstacle obstacle;
                obstacle.set_real_ll(rect->get_low_x(), rect->get_low_y());
                obstacle.set_real_ur(rect->get_high_x(), rect->get_high_y());
                obstacle.set_layer_idx(layer_shape.get_layer()->get_id());
                routing_obstacle_list.push_back(std::move(obstacle));
              }
            }
            idb::IdbLayerShape cut_layer_shape = idb_segment->get_via()->get_cut_layer_shape();
            for (idb::IdbRect* rect : cut_layer_shape.get_rect_list()) {
              Obstacle obstacle;
              obstacle.set_real_ll(rect->get_low_x(), rect->get_low_y());
              obstacle.set_real_ur(rect->get_high_x(), rect->get_high_y());
              obstacle.set_layer_idx(cut_layer_shape.get_layer()->get_id());
              cut_obstacle_list.push_back(std::move(obstacle));
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
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void DataManager::wrapNetList()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<Net>& net_list = _database.get_net_list();
  std::vector<idb::IdbNet*>& idb_net_list = _database.get_idb_builder()->get_def_service()->get_design()->get_net_list()->get_net_list();

  std::vector<idb::IdbNet*> valid_idb_net_list;
  {
    valid_idb_net_list.reserve(idb_net_list.size());
    for (idb::IdbNet* idb_net : idb_net_list) {
      if (isSkipping(idb_net)) {
        continue;
      }
      valid_idb_net_list.push_back(idb_net);
    }
  }
  net_list.resize(valid_idb_net_list.size());
#pragma omp parallel for
  for (size_t i = 0; i < valid_idb_net_list.size(); i++) {
    idb::IdbNet* valid_idb_net = valid_idb_net_list[i];
    Net& net = net_list[i];
    net.set_net_name(valid_idb_net->get_net_name());
    net.set_connect_type(getRTConnectTypeByDB(valid_idb_net->get_connect_type()));
    wrapPinList(net, valid_idb_net);
    wrapDrivenPin(net, valid_idb_net);
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

bool DataManager::isSkipping(idb::IdbNet* idb_net)
{
  bool has_io_pin = false;
  if (idb_net->has_io_pins() && idb_net->get_io_pins()->get_pin_num() == 1) {
    has_io_pin = true;
  }
  bool has_io_cell = false;
  std::vector<idb::IdbInstance*>& instance_list = idb_net->get_instance_list()->get_instance_list();
  if (instance_list.size() == 1 && instance_list.front()->get_cell_master()->is_pad()) {
    has_io_cell = true;
  }
  if (has_io_pin && has_io_cell) {
    return true;
  }

  int32_t pin_num = 0;
  for (idb::IdbPin* idb_pin : idb_net->get_instance_pin_list()->get_pin_list()) {
    if (idb_pin->get_term()->get_port_number() <= 0) {
      continue;
    }
    pin_num++;
  }
  for (auto* io_pin : idb_net->get_io_pins()->get_pin_list()) {
    if (io_pin->get_term()->get_port_number() <= 0) {
      continue;
    }
    pin_num++;
  }
  if (pin_num <= 1) {
    return true;
  } else if (pin_num >= 500) {
    RTLOG.warn(Loc::current(), "The ultra large net: ", idb_net->get_net_name(), " has ", pin_num, " pins!");
  }
  return false;
}

void DataManager::wrapPinList(Net& net, idb::IdbNet* idb_net)
{
  std::vector<Pin>& pin_list = net.get_pin_list();

  for (idb::IdbPin* idb_pin : idb_net->get_instance_pin_list()->get_pin_list()) {
    if (idb_pin->get_term()->get_port_number() <= 0) {
      continue;
    }
    Pin pin;
    pin.set_pin_name(RTUTIL.getString(idb_pin->get_instance()->get_name(), ":", idb_pin->get_pin_name()));
    wrapPinShapeList(pin, idb_pin);
    pin_list.push_back(std::move(pin));
  }
  for (auto* io_pin : idb_net->get_io_pins()->get_pin_list()) {
    if (io_pin->get_term()->get_port_number() <= 0) {
      continue;
    }
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
      pin_shape.set_real_ll(rect->get_low_x(), rect->get_low_y());
      pin_shape.set_real_ur(rect->get_high_x(), rect->get_high_y());
      pin_shape.set_layer_idx(layer_shape->get_layer()->get_id());
      if (layer_shape->get_layer()->is_routing()) {
        routing_shape_list.push_back(std::move(pin_shape));
      } else if (layer_shape->get_layer()->is_cut()) {
        cut_shape_list.push_back(std::move(pin_shape));
      }
    }
  }
}

void DataManager::wrapDrivenPin(Net& net, idb::IdbNet* idb_net)
{
  idb::IdbPin* idb_driving_pin = idb_net->get_driving_pin();
  if (idb_driving_pin == nullptr) {
    return;
  }
  std::string driven_pin_name = idb_driving_pin->get_pin_name();
  if (!idb_driving_pin->is_io_pin()) {
    driven_pin_name = RTUTIL.getString(idb_driving_pin->get_instance()->get_name(), ":", driven_pin_name);
  }
  for (Pin& pin : net.get_pin_list()) {
    if (pin.get_pin_name() == driven_pin_name) {
      pin.set_is_driven(true);
    }
  }
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
    case idb::IdbConnectType::kClock:
      connect_type = ConnectType::kClock;
      break;
    default:
      connect_type = ConnectType::kSignal;
      break;
  }
  return connect_type;
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
  _config.bottom_routing_layer_idx = _database.get_routing_layer_name_to_idx_map()[_config.bottom_routing_layer];
  _config.top_routing_layer_idx = _database.get_routing_layer_name_to_idx_map()[_config.top_routing_layer];
  if (_config.bottom_routing_layer_idx >= _config.top_routing_layer_idx) {
    RTLOG.error(Loc::current(), "The routing layer should be at least two layers!");
  }
  // **********   PinAccessor     ********** //
  _config.pa_temp_directory_path = _config.temp_directory_path + "pin_accessor/";
  // ********     SupplyAnalyzer    ******** //
  _config.sa_temp_directory_path = _config.temp_directory_path + "supply_analyzer/";
  // ********   TopologyGenerator   ******** //
  _config.tg_temp_directory_path = _config.temp_directory_path + "topology_generator/";
  // **********   InitialRouter    ********** //
  _config.ir_temp_directory_path = _config.temp_directory_path + "initial_router/";
  // **********   GlobalRouter    ********** //
  _config.gr_temp_directory_path = _config.temp_directory_path + "global_router/";
  // **********   TrackAssigner   ********** //
  _config.ta_temp_directory_path = _config.temp_directory_path + "track_assigner/";
  // **********  DetailedRouter   ********** //
  _config.dr_temp_directory_path = _config.temp_directory_path + "detailed_router/";
  /////////////////////////////////////////////
  // **********        RT         ********** //
  RTUTIL.createDir(_config.temp_directory_path);
  RTUTIL.createDirByFile(_config.log_file_path);
  // **********   PinAccessor     ********** //
  RTUTIL.createDir(_config.pa_temp_directory_path);
  // **********   SupplyAnalyzer     ********** //
  RTUTIL.createDir(_config.sa_temp_directory_path);
  // *********  TopologyGenerator  ********* //
  RTUTIL.createDir(_config.tg_temp_directory_path);
  // **********   InitialRouter    ********** //
  RTUTIL.createDir(_config.ir_temp_directory_path);
  // **********   GlobalRouter    ********** //
  RTUTIL.createDir(_config.gr_temp_directory_path);
  // **********   TrackAssigner   ********** //
  RTUTIL.createDir(_config.ta_temp_directory_path);
  // **********  DetailedRouter   ********** //
  RTUTIL.createDir(_config.dr_temp_directory_path);
  /////////////////////////////////////////////
  RTLOG.openLogFileStream(_config.log_file_path);
}

void DataManager::buildDatabase()
{
  buildGCellAxis();
  buildDie();
  buildLayerList();
  buildLayerInfo();
  buildLayerViaMasterList();
  buildLayerViaMasterInfo();
  buildObstacleList();
  buildNetList();
  buildGCellMap();
}

void DataManager::buildGCellAxis()
{
  makeGCellAxis();
  checkGCellAxis();
}

void DataManager::makeGCellAxis()
{
  ScaleAxis& gcell_axis = _database.get_gcell_axis();

  int32_t recommended_pitch = getRecommendedPitch();
  gcell_axis.set_x_grid_list(makeGCellGridList(Direction::kVertical, recommended_pitch));
  gcell_axis.set_y_grid_list(makeGCellGridList(Direction::kHorizontal, recommended_pitch));
}

int32_t DataManager::getRecommendedPitch()
{
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();

  std::map<int32_t, int32_t> pitch_count_map;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    for (ScaleGrid& track_grid : routing_layer.getPreferTrackGridList()) {
      pitch_count_map[track_grid.get_step_length()]++;
    }
  }
  int32_t recommended_pitch = -1;
  int32_t max_count = INT32_MIN;
  for (auto [pitch, count] : pitch_count_map) {
    if (count > max_count) {
      max_count = count;
      recommended_pitch = pitch;
    }
  }
  if (max_count == 1) {
    int32_t min_pitch = INT32_MAX;
    for (auto [pitch, count] : pitch_count_map) {
      min_pitch = std::min(min_pitch, pitch);
    }
    recommended_pitch = min_pitch;
  }
  if (recommended_pitch == -1) {
    RTLOG.error(Loc::current(), "The recommended_pitch is -1!");
  }
  return recommended_pitch;
}

std::vector<ScaleGrid> DataManager::makeGCellGridList(Direction direction, int32_t recommended_pitch)
{
  Die& die = _database.get_die();
  Row& row = _database.get_row();

  int32_t die_start_scale = (direction == Direction::kVertical ? die.get_real_ll_x() : die.get_real_ll_y());
  int32_t die_end_scale = (direction == Direction::kVertical ? die.get_real_ur_x() : die.get_real_ur_y());
  int32_t row_mid_scale = (direction == Direction::kVertical ? row.get_start_x() : row.get_start_y());
  // 为了防止与track重合，减去一个recommended_pitch的一半
  row_mid_scale -= (recommended_pitch / 2);
  int32_t step_length = row.get_height();

  std::vector<int32_t> gcell_scale_list;
  gcell_scale_list.push_back(die_start_scale);
  for (int32_t gcell_scale = row_mid_scale; gcell_scale >= die_start_scale; gcell_scale -= step_length) {
    gcell_scale_list.push_back(gcell_scale);
  }
  for (int32_t gcell_scale = row_mid_scale; gcell_scale <= die_end_scale; gcell_scale += step_length) {
    gcell_scale_list.push_back(gcell_scale);
  }
  gcell_scale_list.push_back(die_end_scale);

  std::sort(gcell_scale_list.begin(), gcell_scale_list.end());
  // 删除小于step_length的
  for (int32_t i = 2; i < static_cast<int32_t>(gcell_scale_list.size()); i++) {
    if (std::abs(gcell_scale_list[i - 2] - gcell_scale_list[i - 1]) < step_length
        || std::abs(gcell_scale_list[i - 1] - gcell_scale_list[i]) < step_length) {
      gcell_scale_list[i - 1] = gcell_scale_list[i - 2];
    }
  }
  gcell_scale_list.erase(std::unique(gcell_scale_list.begin(), gcell_scale_list.end()), gcell_scale_list.end());
  return makeGCellGridList(gcell_scale_list);
}

std::vector<ScaleGrid> DataManager::makeGCellGridList(std::vector<int32_t>& gcell_scale_list)
{
  std::vector<ScaleGrid> gcell_grid_list;

  for (size_t i = 1; i < gcell_scale_list.size(); i++) {
    int32_t pre_scale = gcell_scale_list[i - 1];
    int32_t curr_scale = gcell_scale_list[i];

    ScaleGrid gcell_grid;
    gcell_grid.set_start_line(pre_scale);
    gcell_grid.set_step_length(curr_scale - pre_scale);
    gcell_grid.set_step_num(1);
    gcell_grid.set_end_line(curr_scale);
    gcell_grid_list.push_back(gcell_grid);
  }
  // merge
  RTUTIL.merge(gcell_grid_list, [](ScaleGrid& sentry, ScaleGrid& soldier) {
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
  ScaleAxis& gcell_axis = _database.get_gcell_axis();
  std::vector<ScaleGrid>& x_grid_list = gcell_axis.get_x_grid_list();
  std::vector<ScaleGrid>& y_grid_list = gcell_axis.get_y_grid_list();

  if (x_grid_list.empty() || y_grid_list.empty()) {
    RTLOG.error(Loc::current(), "The gcell grid list is empty!");
  }
  for (size_t i = 0; i < x_grid_list.size(); i++) {
    if (x_grid_list[i].get_step_length() <= 0) {
      RTLOG.error(Loc::current(), "The step length of x grid '", x_grid_list[i].get_step_length(), "' is wrong!");
    }
  }
  for (size_t i = 0; i < y_grid_list.size(); i++) {
    if (y_grid_list[i].get_step_length() <= 0) {
      RTLOG.error(Loc::current(), "The step length of y grid '", y_grid_list[i].get_step_length(), "' is wrong!");
    }
  }
  for (size_t i = 1; i < x_grid_list.size(); i++) {
    if (x_grid_list[i - 1].get_end_line() < x_grid_list[i].get_start_line()) {
      RTLOG.error(Loc::current(), "The x grid with gap '", x_grid_list[i - 1].get_end_line(), " < ", x_grid_list[i].get_start_line(), "'!");
    } else if (x_grid_list[i - 1].get_end_line() > x_grid_list[i].get_start_line()) {
      RTLOG.error(Loc::current(), "The x grid with overlapping '", x_grid_list[i - 1].get_end_line(), " < ",
                  x_grid_list[i].get_start_line(), "'!");
    }
  }
  for (size_t i = 1; i < y_grid_list.size(); i++) {
    if (y_grid_list[i - 1].get_end_line() < y_grid_list[i].get_start_line()) {
      RTLOG.error(Loc::current(), "The y grid with gap '", y_grid_list[i - 1].get_end_line(), " < ", y_grid_list[i].get_start_line(), "'!");
    } else if (y_grid_list[i - 1].get_end_line() > y_grid_list[i].get_start_line()) {
      RTLOG.error(Loc::current(), "The y grid with overlapping '", y_grid_list[i - 1].get_end_line(), " > ",
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
  ScaleAxis& gcell_axis = _database.get_gcell_axis();
  die.set_grid_rect(RTUTIL.getOpenGCellGridRect(die.get_real_rect(), gcell_axis));
}

void DataManager::checkDie()
{
  Die& die = _database.get_die();

  if (die.get_real_ll_x() < 0 || die.get_real_ll_y() < 0 || die.get_real_ur_x() < 0 || die.get_real_ur_y() < 0) {
    RTLOG.error(Loc::current(), "The die '(", die.get_real_ll_x(), " , ", die.get_real_ll_y(), ") - (", die.get_real_ur_x(), " , ",
                die.get_real_ur_y(), ")' is wrong!");
  }
  if ((die.get_real_ur_x() <= die.get_real_ll_x()) || (die.get_real_ur_y() <= die.get_real_ll_y())) {
    RTLOG.error(Loc::current(), "The die '(", die.get_real_ll_x(), " , ", die.get_real_ll_y(), ") - (", die.get_real_ur_x(), " , ",
                die.get_real_ur_y(), ")' is wrong!");
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
  std::map<int32_t, int32_t>& routing_idb_layer_id_to_idx_map = _database.get_routing_idb_layer_id_to_idx_map();
  std::map<int32_t, int32_t>& cut_idb_layer_id_to_idx_map = _database.get_cut_idb_layer_id_to_idx_map();

  for (RoutingLayer& routing_layer : _database.get_routing_layer_list()) {
    routing_layer.set_layer_idx(routing_idb_layer_id_to_idx_map[routing_layer.get_layer_idx()]);
  }
  for (CutLayer& cut_layer_list : _database.get_cut_layer_list()) {
    cut_layer_list.set_layer_idx(cut_idb_layer_id_to_idx_map[cut_layer_list.get_layer_idx()]);
  }
}

void DataManager::makeLayerList()
{
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();

  for (RoutingLayer& routing_layer : routing_layer_list) {
    for (ScaleGrid& x_track_grid : routing_layer.getXTrackGridList()) {
      x_track_grid.set_end_line(x_track_grid.get_start_line() + x_track_grid.get_step_length() * x_track_grid.get_step_num());
    }
    for (ScaleGrid& y_track_grid : routing_layer.getYTrackGridList()) {
      y_track_grid.set_end_line(y_track_grid.get_start_line() + y_track_grid.get_step_length() * y_track_grid.get_step_num());
    }
  }
  if (routing_layer_list.size() < 2) {
    RTLOG.error(Loc::current(), "The size of routing_layer < 2!");
  }
  ScaleAxis& standard_track_axis = routing_layer_list[1].get_track_axis();
  for (RoutingLayer& routing_layer : routing_layer_list) {
    routing_layer.set_track_axis(standard_track_axis);
  }
}

void DataManager::checkLayerList()
{
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = _database.get_cut_layer_list();

  if (routing_layer_list.empty()) {
    RTLOG.error(Loc::current(), "The routing_layer_list is empty!");
  }
  if (cut_layer_list.empty()) {
    RTLOG.error(Loc::current(), "The cut_layer_list is empty!");
  }

  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::string& layer_name = routing_layer.get_layer_name();
    if (routing_layer.get_prefer_direction() == Direction::kNone) {
      RTLOG.error(Loc::current(), "The layer '", layer_name, "' prefer_direction is none!");
    }
    for (ScaleGrid& x_track_grid : routing_layer.getXTrackGridList()) {
      if (x_track_grid.get_step_length() <= 0) {
        RTLOG.error(Loc::current(), "The layer '", layer_name, "' x_track_grid step length '", x_track_grid.get_step_length(),
                    "' is wrong!");
      }
    }
    for (ScaleGrid& y_track_grid : routing_layer.getYTrackGridList()) {
      if (y_track_grid.get_step_length() <= 0) {
        RTLOG.error(Loc::current(), "The layer '", layer_name, "' y_track_grid step length '", y_track_grid.get_step_length(),
                    "' is wrong!");
      }
    }
    SpacingTable& spacing_table = routing_layer.get_spacing_table();
    if (spacing_table.get_width_list().empty()) {
      RTLOG.error(Loc::current(), "The layer '", layer_name, "' spacing width list is empty!");
    }
    for (int32_t width : spacing_table.get_width_list()) {
      if (width < 0) {
        RTLOG.error(Loc::current(), "The layer '", layer_name, "' width < 0!");
      }
    }
    for (int32_t parallel_length : spacing_table.get_parallel_length_list()) {
      if (parallel_length < 0) {
        RTLOG.error(Loc::current(), "The layer '", layer_name, "' parallel_length < 0!");
      }
    }
    GridMap<int32_t>& width_parallel_length_map = spacing_table.get_width_parallel_length_map();
    for (int32_t width_idx = 0; width_idx < width_parallel_length_map.get_x_size(); width_idx++) {
      for (int32_t parallel_length_idx = 0; parallel_length_idx < width_parallel_length_map.get_y_size(); parallel_length_idx++) {
        if (width_parallel_length_map[width_idx][parallel_length_idx] < 0) {
          RTLOG.error(Loc::current(), "The layer '", layer_name, "' spacing < 0!");
        }
      }
    }
  }
}

void DataManager::buildLayerInfo()
{
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = _database.get_cut_to_adjacent_routing_map();

  std::map<int32_t, std::pair<bool, int32_t>> layer_order_to_idx_map;
  for (RoutingLayer& routing_layer : _database.get_routing_layer_list()) {
    layer_order_to_idx_map[routing_layer.get_layer_order()] = std::make_pair(false, routing_layer.get_layer_idx());
  }
  for (CutLayer& cut_layer : _database.get_cut_layer_list()) {
    layer_order_to_idx_map[cut_layer.get_layer_order()] = std::make_pair(true, cut_layer.get_layer_idx());
  }
  std::vector<std::pair<bool, int32_t>> layer_idx_pair_list;
  for (auto& [layer_order, type_layer_idx_pair] : layer_order_to_idx_map) {
    layer_idx_pair_list.push_back(type_layer_idx_pair);
  }

  for (int32_t i = 0; i < static_cast<int32_t>(layer_idx_pair_list.size()); i++) {
    bool is_cut = layer_idx_pair_list[i].first;
    int32_t cut_layer_idx = layer_idx_pair_list[i].second;
    if (!is_cut) {
      continue;
    }
    std::vector<int32_t> adj_routing_layer_idx_list;
    if (i - 1 >= 0 && layer_idx_pair_list[i - 1].first == false) {
      adj_routing_layer_idx_list.push_back(layer_idx_pair_list[i - 1].second);
    }
    if (i + 1 < static_cast<int32_t>(layer_idx_pair_list.size()) && layer_idx_pair_list[i + 1].first == false) {
      adj_routing_layer_idx_list.push_back(layer_idx_pair_list[i + 1].second);
    }
    if (!adj_routing_layer_idx_list.empty()) {
      cut_to_adjacent_routing_map[cut_layer_idx] = adj_routing_layer_idx_list;
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
  std::map<int32_t, int32_t>& routing_idb_layer_id_to_idx_map = _database.get_routing_idb_layer_id_to_idx_map();
  std::map<int32_t, int32_t>& cut_idb_layer_id_to_idx_map = _database.get_cut_idb_layer_id_to_idx_map();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _database.get_layer_via_master_list();

  for (std::vector<ViaMaster>& via_master_list : layer_via_master_list) {
    for (ViaMaster& via_master : via_master_list) {
      // above
      LayerRect& above_enclosure = via_master.get_above_enclosure();
      above_enclosure.set_layer_idx(routing_idb_layer_id_to_idx_map[above_enclosure.get_layer_idx()]);
      // below
      LayerRect& below_enclosure = via_master.get_below_enclosure();
      below_enclosure.set_layer_idx(routing_idb_layer_id_to_idx_map[below_enclosure.get_layer_idx()]);
      // cut
      via_master.set_cut_layer_idx(cut_idb_layer_id_to_idx_map[via_master.get_cut_layer_idx()]);
    }
  }
}

void DataManager::makeLayerViaMasterList()
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _database.get_layer_via_master_list();
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();

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
    std::sort(via_master_list.begin(), via_master_list.end(),
              [&](ViaMaster& via_master1, ViaMaster& via_master2) { return sortByMultiLevel(via_master1, via_master2); });
    for (size_t i = 0; i < via_master_list.size(); i++) {
      via_master_list[i].set_via_master_idx(layer_idx, i);
    }
  }
}

bool DataManager::sortByMultiLevel(ViaMaster& via_master1, ViaMaster& via_master2)
{
  SortStatus sort_status = SortStatus::kNone;

  sort_status = sortByLayerDirectionPriority(via_master1, via_master2);
  if (sort_status == SortStatus::kTrue) {
    return true;
  } else if (sort_status == SortStatus::kFalse) {
    return false;
  }
  sort_status = sortByWidthASC(via_master1, via_master2);
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

// 层方向优先
SortStatus DataManager::sortByLayerDirectionPriority(ViaMaster& via_master1, ViaMaster& via_master2)
{
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();

  Direction above_layer_direction = routing_layer_list[via_master1.get_above_enclosure().get_layer_idx()].get_prefer_direction();
  Direction below_layer_direction = routing_layer_list[via_master1.get_below_enclosure().get_layer_idx()].get_prefer_direction();

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

  // via_master的ll为负数，ur为正数
  int32_t via_master1_above_center_diff = std::abs(via_master1_above.get_ll_x() + via_master1_above.get_ur_x());
  int32_t via_master2_above_center_diff = std::abs(via_master2_above.get_ll_x() + via_master2_above.get_ur_x());
  int32_t via_master1_below_center_diff = std::abs(via_master1_below.get_ll_x() + via_master1_below.get_ur_x());
  int32_t via_master2_below_center_diff = std::abs(via_master2_below.get_ll_x() + via_master2_below.get_ur_x());
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

void DataManager::buildLayerViaMasterInfo()
{
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _database.get_layer_via_master_list();
  std::map<int32_t, PlanarRect>& layer_enclosure_map = _database.get_layer_enclosure_map();

  int32_t start_layer_idx = 0;
  int32_t end_layer_idx = static_cast<int32_t>(routing_layer_list.size()) - 1;

  layer_enclosure_map[start_layer_idx] = layer_via_master_list[start_layer_idx].front().get_below_enclosure();
  for (int32_t layer_idx = 1; layer_idx < end_layer_idx; layer_idx++) {
    std::vector<PlanarRect> rect_list;
    rect_list.push_back(layer_via_master_list[layer_idx - 1].front().get_above_enclosure());
    rect_list.push_back(layer_via_master_list[layer_idx].front().get_below_enclosure());
    layer_enclosure_map[layer_idx] = RTUTIL.getBoundingBox(rect_list);
  }
  layer_enclosure_map[end_layer_idx] = layer_via_master_list[end_layer_idx - 1].front().get_above_enclosure();
}

void DataManager::buildObstacleList()
{
  transObstacleList();
  makeObstacleList();
  checkObstacleList();
}

void DataManager::transObstacleList()
{
  std::map<int32_t, int32_t>& routing_idb_layer_id_to_idx_map = _database.get_routing_idb_layer_id_to_idx_map();
  std::map<int32_t, int32_t>& cut_idb_layer_id_to_idx_map = _database.get_cut_idb_layer_id_to_idx_map();
  std::vector<Obstacle>& routing_obstacle_list = _database.get_routing_obstacle_list();
  std::vector<Obstacle>& cut_obstacle_list = _database.get_cut_obstacle_list();

#pragma omp parallel for
  for (Obstacle& obstacle : routing_obstacle_list) {
    obstacle.set_layer_idx(routing_idb_layer_id_to_idx_map[obstacle.get_layer_idx()]);
  }
#pragma omp parallel for
  for (Obstacle& obstacle : cut_obstacle_list) {
    obstacle.set_layer_idx(cut_idb_layer_id_to_idx_map[obstacle.get_layer_idx()]);
  }
}

void DataManager::makeObstacleList()
{
  ScaleAxis& gcell_axis = _database.get_gcell_axis();
  Die& die = _database.get_die();
  std::vector<Obstacle>& routing_obstacle_list = _database.get_routing_obstacle_list();
  std::vector<Obstacle>& cut_obstacle_list = _database.get_cut_obstacle_list();

#pragma omp parallel for
  for (Obstacle& routing_obstacle : routing_obstacle_list) {
    routing_obstacle.set_real_rect(RTUTIL.getRegularRect(routing_obstacle.get_real_rect(), die.get_real_rect()));
    routing_obstacle.set_grid_rect(RTUTIL.getClosedGCellGridRect(routing_obstacle.get_real_rect(), gcell_axis));
  }
#pragma omp parallel for
  for (Obstacle& cut_obstacle : cut_obstacle_list) {
    cut_obstacle.set_real_rect(RTUTIL.getRegularRect(cut_obstacle.get_real_rect(), die.get_real_rect()));
    cut_obstacle.set_grid_rect(RTUTIL.getClosedGCellGridRect(cut_obstacle.get_real_rect(), gcell_axis));
  }
}

void DataManager::checkObstacleList()
{
  Die& die = _database.get_die();
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();
  std::vector<Obstacle>& routing_obstacle_list = _database.get_routing_obstacle_list();
  std::vector<Obstacle>& cut_obstacle_list = _database.get_cut_obstacle_list();

#pragma omp parallel for
  for (Obstacle& obstacle : routing_obstacle_list) {
    if (obstacle.get_real_ll_x() < die.get_real_ll_x() || obstacle.get_real_ll_y() < die.get_real_ll_y()
        || die.get_real_ur_x() < obstacle.get_real_ur_x() || die.get_real_ur_y() < obstacle.get_real_ur_y()) {
      // log
      RTLOG.error(Loc::current(), "The obstacle '(", obstacle.get_real_ll_x(), " , ", obstacle.get_real_ll_y(), ") - (",
                  obstacle.get_real_ur_x(), " , ", obstacle.get_real_ur_y(), ") ",
                  routing_layer_list[obstacle.get_layer_idx()].get_layer_name(), "' is wrong! Die '(", die.get_real_ll_x(), " , ",
                  die.get_real_ll_y(), ") - (", die.get_real_ur_x(), " , ", die.get_real_ur_y(), ")'");
    }
  }
#pragma omp parallel for
  for (Obstacle& obstacle : cut_obstacle_list) {
    if (obstacle.get_real_ll_x() < die.get_real_ll_x() || obstacle.get_real_ll_y() < die.get_real_ll_y()
        || die.get_real_ur_x() < obstacle.get_real_ur_x() || die.get_real_ur_y() < obstacle.get_real_ur_y()) {
      // log
      RTLOG.error(Loc::current(), "The obstacle '(", obstacle.get_real_ll_x(), " , ", obstacle.get_real_ll_y(), ") - (",
                  obstacle.get_real_ur_x(), " , ", obstacle.get_real_ur_y(), ") ",
                  routing_layer_list[obstacle.get_layer_idx()].get_layer_name(), "' is wrong! Die '(", die.get_real_ll_x(), " , ",
                  die.get_real_ll_y(), ") - (", die.get_real_ur_x(), " , ", die.get_real_ur_y(), ")'");
    }
  }
}

void DataManager::buildNetList()
{
  std::vector<Net>& net_list = _database.get_net_list();
#pragma omp parallel for
  for (size_t net_idx = 0; net_idx < net_list.size(); net_idx++) {
    Net& net = net_list[net_idx];
    net.set_net_idx(static_cast<int32_t>(net_idx));
    buildPinList(net);
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
  std::map<int32_t, int32_t>& routing_idb_layer_id_to_idx_map = _database.get_routing_idb_layer_id_to_idx_map();
  std::map<int32_t, int32_t>& cut_idb_layer_id_to_idx_map = _database.get_cut_idb_layer_id_to_idx_map();

  for (Pin& pin : net.get_pin_list()) {
    for (EXTLayerRect& routing_shape : pin.get_routing_shape_list()) {
      routing_shape.set_layer_idx(routing_idb_layer_id_to_idx_map[routing_shape.get_layer_idx()]);
    }
    for (EXTLayerRect& cut_shape : pin.get_cut_shape_list()) {
      cut_shape.set_layer_idx(cut_idb_layer_id_to_idx_map[cut_shape.get_layer_idx()]);
    }
  }
}

void DataManager::makePinList(Net& net)
{
  Die& die = _database.get_die();
  ScaleAxis& gcell_axis = _database.get_gcell_axis();
  std::vector<Pin>& pin_list = net.get_pin_list();

  for (size_t pin_idx = 0; pin_idx < pin_list.size(); pin_idx++) {
    Pin& pin = pin_list[pin_idx];
    pin.set_pin_idx(static_cast<int32_t>(pin_idx));
    for (EXTLayerRect& routing_shape : pin.get_routing_shape_list()) {
      routing_shape.set_real_rect(RTUTIL.getRegularRect(routing_shape.get_real_rect(), die.get_real_rect()));
      routing_shape.set_grid_rect(RTUTIL.getClosedGCellGridRect(routing_shape.get_real_rect(), gcell_axis));
    }
    for (EXTLayerRect& cut_shape : pin.get_cut_shape_list()) {
      cut_shape.set_real_rect(RTUTIL.getRegularRect(cut_shape.get_real_rect(), die.get_real_rect()));
      cut_shape.set_grid_rect(RTUTIL.getClosedGCellGridRect(cut_shape.get_real_rect(), gcell_axis));
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
      if (routing_shape.get_real_ll_x() < die.get_real_ll_x() || routing_shape.get_real_ll_y() < die.get_real_ll_y()
          || die.get_real_ur_x() < routing_shape.get_real_ur_x() || die.get_real_ur_y() < routing_shape.get_real_ur_y()) {
        RTLOG.error(Loc::current(), "The pin_shape '(", routing_shape.get_real_ll_x(), " , ", routing_shape.get_real_ll_y(), ") - (",
                    routing_shape.get_real_ur_x(), " , ", routing_shape.get_real_ur_y(), ") ",
                    routing_layer_list[routing_shape.get_layer_idx()].get_layer_name(), "' is wrong! Die '(", die.get_real_ll_x(), " , ",
                    die.get_real_ll_y(), ") - (", die.get_real_ur_x(), " , ", die.get_real_ur_y(), ")'");
      }
    }
    for (EXTLayerRect& cut_shape : pin.get_cut_shape_list()) {
      if (cut_shape.get_real_ll_x() < die.get_real_ll_x() || cut_shape.get_real_ll_y() < die.get_real_ll_y()
          || die.get_real_ur_x() < cut_shape.get_real_ur_x() || die.get_real_ur_y() < cut_shape.get_real_ur_y()) {
        RTLOG.error(Loc::current(), "The pin_shape '(", cut_shape.get_real_ll_x(), " , ", cut_shape.get_real_ll_y(), ") - (",
                    cut_shape.get_real_ur_x(), " , ", cut_shape.get_real_ur_y(), ") ",
                    cut_layer_list[cut_shape.get_layer_idx()].get_layer_name(), "' is wrong! Die '(", die.get_real_ll_x(), " , ",
                    die.get_real_ll_y(), ") - (", die.get_real_ur_x(), " , ", die.get_real_ur_y(), ")'");
      }
    }
  }
}

void DataManager::buildGCellMap()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  ScaleAxis& gcell_axis = _database.get_gcell_axis();
  Die& die = _database.get_die();
  std::vector<Obstacle>& routing_obstacle_list = _database.get_routing_obstacle_list();
  std::vector<Obstacle>& cut_obstacle_list = _database.get_cut_obstacle_list();
  std::vector<Net>& net_list = _database.get_net_list();

  GridMap<GCell>& gcell_map = _database.get_gcell_map();
  gcell_map.init(die.getXSize(), die.getYSize());

#pragma omp parallel for collapse(2)
  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      PlanarRect real_rect = RTUTIL.getRealRectByGCell(PlanarCoord(x, y), gcell_axis);
      gcell_map[x][y].set_ll(real_rect.get_ll());
      gcell_map[x][y].set_ur(real_rect.get_ur());
    }
  }
  struct Shape
  {
    int32_t net_idx = -1;
    EXTLayerRect* rect = nullptr;
    bool is_routing = true;
    bool is_save = false;
  };
  std::vector<Shape> shape_list;
  {
    size_t total_shape_num = 0;
    total_shape_num += routing_obstacle_list.size();
    total_shape_num += cut_obstacle_list.size();
    for (Net& net : net_list) {
      for (Pin& pin : net.get_pin_list()) {
        total_shape_num += pin.get_routing_shape_list().size();
        total_shape_num += pin.get_cut_shape_list().size();
      }
    }
    shape_list.reserve(total_shape_num);

    for (Obstacle& routing_obstacle : routing_obstacle_list) {
      shape_list.emplace_back(-1, &routing_obstacle, true);
    }
    for (Obstacle& cut_obstacle : cut_obstacle_list) {
      shape_list.emplace_back(-1, &cut_obstacle, false);
    }
    for (Net& net : net_list) {
      for (Pin& pin : net.get_pin_list()) {
        for (EXTLayerRect& routing_shape : pin.get_routing_shape_list()) {
          shape_list.emplace_back(net.get_net_idx(), &routing_shape, true);
        }
        for (EXTLayerRect& cut_shape : pin.get_cut_shape_list()) {
          shape_list.emplace_back(net.get_net_idx(), &cut_shape, false);
        }
      }
    }
  }
  int32_t interval_length = 6;
  if (interval_length <= (die.get_grid_ur_y() - die.get_grid_ll_y())) {
    std::map<int32_t, std::vector<std::vector<Shape>>> start_shape_list_list_map;
    for (int32_t interval_start : {0, interval_length / 2}) {
      std::vector<std::vector<Shape>>& shape_list_list = start_shape_list_list_map[interval_start];
      shape_list_list.resize((die.get_grid_ur_y() - interval_start) / interval_length + 1);
      for (Shape& shape : shape_list) {
        if (shape.is_save) {
          continue;
        }
        PlanarRect& grid_rect = shape.rect->get_grid_rect();
        int32_t y_interval_idx
            = getIntervalIdx(grid_rect.get_ll_y(), grid_rect.get_ur_y(), interval_start, die.get_grid_ur_y(), interval_length);
        if (y_interval_idx != -1) {
          shape_list_list[y_interval_idx].push_back(shape);
          shape.is_save = true;
        }
      }
    }
    for (auto& [interval_start, shape_list_list] : start_shape_list_list_map) {
#pragma omp parallel for
      for (std::vector<Shape>& shape_list : shape_list_list) {
        for (Shape& shape : shape_list) {
          int32_t net_idx = shape.net_idx;
          EXTLayerRect* ext_layer_rect = shape.rect;
          int32_t layer_idx = ext_layer_rect->get_layer_idx();
          bool is_routing = shape.is_routing;

          for (int32_t x = ext_layer_rect->get_grid_ll_x(); x <= ext_layer_rect->get_grid_ur_x(); x++) {
            for (int32_t y = ext_layer_rect->get_grid_ll_y(); y <= ext_layer_rect->get_grid_ur_y(); y++) {
              gcell_map[x][y].get_type_layer_net_fixed_rect_map()[is_routing][layer_idx][net_idx].insert(ext_layer_rect);
            }
          }
        }
      }
    }
  }
  for (Shape& shape : shape_list) {
    if (shape.is_save) {
      continue;
    }
    int32_t net_idx = shape.net_idx;
    EXTLayerRect* ext_layer_rect = shape.rect;
    int32_t layer_idx = ext_layer_rect->get_layer_idx();
    bool is_routing = shape.is_routing;
    for (int32_t x = ext_layer_rect->get_grid_ll_x(); x <= ext_layer_rect->get_grid_ur_x(); x++) {
      for (int32_t y = ext_layer_rect->get_grid_ll_y(); y <= ext_layer_rect->get_grid_ur_y(); y++) {
        gcell_map[x][y].get_type_layer_net_fixed_rect_map()[is_routing][layer_idx][net_idx].insert(ext_layer_rect);
      }
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

int32_t DataManager::getIntervalIdx(int32_t scale_start, int32_t scale_end, int32_t interval_start, int32_t interval_end,
                                    int32_t interval_length)
{
  if (scale_start < interval_start || scale_end > interval_end) {
    return -1;
  }
  int32_t start_idx = (scale_start - interval_start) / interval_length;
  int32_t end_idx = (scale_end - interval_start) / interval_length;
  if (start_idx != end_idx) {
    return -1;
  }
  return start_idx;
}

void DataManager::printConfig()
{
  /////////////////////////////////////////////
  // **********        RT         ********** //
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(0), "RT_CONFIG_INPUT");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "temp_directory_path");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), _config.temp_directory_path);
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "thread_number");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), _config.thread_number);
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "bottom_routing_layer");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), _config.bottom_routing_layer);
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "top_routing_layer");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), _config.top_routing_layer);
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "output_csv");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), _config.output_csv);
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "enable_timing");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), _config.enable_timing);
  // **********        RT         ********** //
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(0), "RT_CONFIG_BUILD");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "log_file_path");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), _config.log_file_path);
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "bottom_routing_layer_idx");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), _config.bottom_routing_layer_idx);
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "top_routing_layer_idx");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), _config.top_routing_layer_idx);
  // **********   PinAccessor     ********** //
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "PinAccessor");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), "pa_temp_directory_path");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(3), _config.pa_temp_directory_path);
  // **********   SupplyAnalyzer   ********** //
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "SupplyAnalyzer");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), "sa_temp_directory_path");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(3), _config.sa_temp_directory_path);
  // **********  TopologyGenerator  ********* //
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "TopologyGenerator");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), "tg_temp_directory_path");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(3), _config.tg_temp_directory_path);
  // **********   InitialRouter    ********** //
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "InitialRouter");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), "ir_temp_directory_path");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(3), _config.ir_temp_directory_path);
  // **********   GlobalRouter    ********** //
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "GlobalRouter");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), "gr_temp_directory_path");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(3), _config.gr_temp_directory_path);
  // **********   TrackAssigner   ********** //
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "TrackAssigner");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), "ta_temp_directory_path");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(3), _config.ta_temp_directory_path);
  // **********  DetailedRouter   ********** //
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "DetailedRouter");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), "dr_temp_directory_path");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(3), _config.dr_temp_directory_path);
  /////////////////////////////////////////////
}

void DataManager::printDatabase()
{
  ////////////////////////////////////////////////
  // ********** RT ********** //
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(0), "RT_DATABASE");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "design_name");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), _database.get_design_name());
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "lef_file_path_list");
  for (std::string& lef_file_path : _database.get_lef_file_path_list()) {
    RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), lef_file_path);
  }
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "def_file_path");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), _database.get_def_file_path());
  // ********** MicronDBU ********** //
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "micron_dbu");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), _database.get_micron_dbu());
  // ********** GCellAxis ********** //
  ScaleAxis& gcell_axis = _database.get_gcell_axis();
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "gcell_axis");
  std::vector<ScaleGrid>& x_grid_list = gcell_axis.get_x_grid_list();
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), "x_grid_list");
  for (ScaleGrid& x_grid : x_grid_list) {
    RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(3), "start:", x_grid.get_start_line(), " step_length:", x_grid.get_step_length(),
               " step_num:", x_grid.get_step_num(), " end:", x_grid.get_end_line());
  }
  std::vector<ScaleGrid>& y_grid_list = gcell_axis.get_y_grid_list();
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), "y_grid_list");
  for (ScaleGrid& y_grid : y_grid_list) {
    RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(3), "start:", y_grid.get_start_line(), " step_length:", y_grid.get_step_length(),
               " step_num:", y_grid.get_step_num(), " end:", y_grid.get_end_line());
  }
  // ********** Die ********** //
  Die& die = _database.get_die();
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "die");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), "(", die.get_real_ll_x(), ",", die.get_real_ll_y(), ")-(", die.get_real_ur_x(),
             ",", die.get_real_ur_y(), ")");
  // ********** RoutingLayer ********** //
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "routing_layer_num");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), routing_layer_list.size());
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "routing_layer");
  for (RoutingLayer& routing_layer : routing_layer_list) {
    RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), "idx:", routing_layer.get_layer_idx(),
               " order:", routing_layer.get_layer_order(), " name:", routing_layer.get_layer_name(),
               " min_width:", routing_layer.get_min_width(), " min_area:", routing_layer.get_min_area(),
               " prefer_direction:", GetDirectionName()(routing_layer.get_prefer_direction()));

    ScaleAxis& track_axis = routing_layer.get_track_axis();
    RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), "track_axis");
    std::vector<ScaleGrid>& x_grid_list = track_axis.get_x_grid_list();
    RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(3), "x_grid_list");
    for (ScaleGrid& x_grid : x_grid_list) {
      RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(4), "start:", x_grid.get_start_line(), " step_length:", x_grid.get_step_length(),
                 " step_num:", x_grid.get_step_num(), " end:", x_grid.get_end_line());
    }
    std::vector<ScaleGrid>& y_grid_list = track_axis.get_y_grid_list();
    RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(3), "y_grid_list");
    for (ScaleGrid& y_grid : y_grid_list) {
      RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(4), "start:", y_grid.get_start_line(), " step_length:", y_grid.get_step_length(),
                 " step_num:", y_grid.get_step_num(), " end:", y_grid.get_end_line());
    }
  }
  // ********** CutLayer ********** //
  std::vector<CutLayer>& cut_layer_list = _database.get_cut_layer_list();
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "cut_layer_num");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), cut_layer_list.size());
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "cut_layer");
  for (CutLayer& cut_layer : cut_layer_list) {
    RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), "idx:", cut_layer.get_layer_idx(), " order:", cut_layer.get_layer_order(),
               " name:", cut_layer.get_layer_name());
  }
  // ********** ViaMaster ********** //
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _database.get_layer_via_master_list();
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "layer_via_master_list");
  for (size_t below_layer_idx = 0; below_layer_idx < layer_via_master_list.size(); below_layer_idx++) {
    std::string via_master_name_string = (routing_layer_list[below_layer_idx].get_layer_name() + ": ");
    for (ViaMaster& via_master : layer_via_master_list[below_layer_idx]) {
      via_master_name_string += (via_master.get_via_name() + " ");
    }
    RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), via_master_name_string);
  }
  // ********** Obstacle ********** //
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "routing_obstacle_num");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), _database.get_routing_obstacle_list().size());
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "cut_obstacle_num");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), _database.get_cut_obstacle_list().size());
  // ********** Net ********** //
  std::vector<Net>& net_list = _database.get_net_list();
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "net_num");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), net_list.size());
  size_t total_pin_num = 0;
  for (Net& net : net_list) {
    total_pin_num += net.get_pin_list().size();
  }
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "pin_num");
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), total_pin_num);
  RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(1), "pin_num_ratio");

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
    RTLOG.info(Loc::current(), RTUTIL.getSpaceByTabNum(2), head_info, pin_num, " pins: ", net_num, "(",
               RTUTIL.getPercentage(net_num, net_list.size()), ")");
  }
  ////////////////////////////////////////////////
}

void DataManager::writePYScript()
{
  std::string& temp_directory_path = RTDM.getConfig().temp_directory_path;

  std::ofstream* python_file = RTUTIL.getOutputFileStream(RTUTIL.getString(temp_directory_path, "plot.py"));
  RTUTIL.pushStream(python_file, "import os", "\n");
  RTUTIL.pushStream(python_file, "import pandas as pd", "\n");
  RTUTIL.pushStream(python_file, "import matplotlib.pyplot as plt", "\n");
  RTUTIL.pushStream(python_file, "import seaborn as sns", "\n");
  RTUTIL.pushStream(python_file, "import numpy as np", "\n");
  RTUTIL.pushStream(python_file, "from multiprocessing import Pool", "\n");
  RTUTIL.pushStream(python_file, "directory = os.getcwd()", "\n");
  RTUTIL.pushStream(python_file, "output_dir = os.path.join(directory, 'output_png')", "\n");
  RTUTIL.pushStream(python_file, "os.makedirs(output_dir, exist_ok=True)", "\n");
  RTUTIL.pushStream(python_file, "def process_csv(filename):", "\n");
  RTUTIL.pushStream(
      python_file,
      "    output_name = os.path.basename(os.path.dirname(filename)) + \"_\" + os.path.splitext(os.path.basename(filename))[0]", "\n");
  RTUTIL.pushStream(python_file, "    plt.figure()  ", "\n");
  RTUTIL.pushStream(python_file, "    temp = sns.heatmap(np.array(pd.read_csv(filename)), cmap='hot_r')", "\n");
  RTUTIL.pushStream(python_file, "    temp.set_title(output_name)", "\n");
  RTUTIL.pushStream(python_file, "    plt.savefig(os.path.join(output_dir, output_name + \".png\"), dpi=300, bbox_inches='tight')", "\n");
  RTUTIL.pushStream(python_file, "    plt.close()  ", "\n");
  RTUTIL.pushStream(
      python_file,
      "csv_files = [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.endswith(\".csv\")]", "\n");
  RTUTIL.pushStream(python_file, "with Pool() as pool:", "\n");
  RTUTIL.pushStream(python_file, "    pool.map(process_csv, csv_files)", "\n");
  RTUTIL.closeFileStream(python_file);
}

#endif

#if 1  // output

void DataManager::outputTrackGrid()
{
  idb::IdbLayers* idb_layer_list = _database.get_idb_builder()->get_def_service()->get_layout()->get_layers();
  idb::IdbTrackGridList* idb_track_grid_list = _database.get_idb_builder()->get_def_service()->get_layout()->get_track_grid_list();
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();

  idb_track_grid_list->reset();

  for (int32_t i = static_cast<int32_t>(routing_layer_list.size()) - 1; i >= 0; --i) {
    RoutingLayer& routing_layer = routing_layer_list[i];

    std::string layer_name = routing_layer.get_layer_name();
    idb::IdbLayer* idb_layer = idb_layer_list->find_layer(layer_name);
    if (idb_layer == nullptr) {
      RTLOG.error(Loc::current(), "Can not find idb layer ", layer_name);
    }
    idb::IdbLayerRouting* idb_routing_layer = dynamic_cast<idb::IdbLayerRouting*>(idb_layer);
    idb_routing_layer->get_track_grid_list().clear();

    std::map<Direction, std::vector<ScaleGrid>> direction_scale_grid_list_map;
    for (ScaleGrid& x_grid : routing_layer.get_track_axis().get_x_grid_list()) {
      direction_scale_grid_list_map[Direction::kVertical].push_back(x_grid);
    }
    for (ScaleGrid& y_grid : routing_layer.get_track_axis().get_y_grid_list()) {
      direction_scale_grid_list_map[Direction::kHorizontal].push_back(y_grid);
    }
    for (auto& [direction, scale_grid_list] : direction_scale_grid_list_map) {
      for (ScaleGrid& scale_grid : scale_grid_list) {
        idb::IdbTrackGrid* idb_track_grid = idb_track_grid_list->add_track_grid();
        idb::IdbTrack* idb_track = idb_track_grid->get_track();
        if (direction == Direction::kVertical) {
          idb_track->set_direction(idb::IdbTrackDirection::kDirectionX);
        } else if (direction == Direction::kHorizontal) {
          idb_track->set_direction(idb::IdbTrackDirection::kDirectionY);
        }
        idb_track->set_start(scale_grid.get_start_line());
        idb_track->set_pitch(scale_grid.get_step_length());
        idb_track_grid->set_track_number(scale_grid.get_step_num() + 1);
        idb_track_grid->add_layer_list(idb_layer);
        idb_routing_layer->add_track_grid(idb_track_grid);
      }
    }
  }
}

void DataManager::outputGCellGrid()
{
  idb::IdbGCellGridList* idb_gcell_grid_list = _database.get_idb_builder()->get_lef_service()->get_layout()->get_gcell_grid_list();
  ScaleAxis& gcell_axis = _database.get_gcell_axis();

  idb_gcell_grid_list->clear();

  for (idb::IdbTrackDirection idb_track_direction : {idb::IdbTrackDirection::kDirectionX, idb::IdbTrackDirection::kDirectionY}) {
    std::vector<ScaleGrid> gcell_grid_list;
    if (idb_track_direction == idb::IdbTrackDirection::kDirectionX) {
      gcell_grid_list = gcell_axis.get_x_grid_list();
    } else {
      gcell_grid_list = gcell_axis.get_y_grid_list();
    }
    for (ScaleGrid& gcell_grid : gcell_grid_list) {
      idb::IdbGCellGrid* idb_gcell_grid = new idb::IdbGCellGrid();
      idb_gcell_grid->set_start(gcell_grid.get_start_line());
      idb_gcell_grid->set_space(gcell_grid.get_step_length());
      idb_gcell_grid->set_num(gcell_grid.get_step_num() + 1);
      idb_gcell_grid->set_direction(idb_track_direction);
      idb_gcell_grid_list->add_gcell_grid(idb_gcell_grid);
    }
  }
}

void DataManager::outputNetList()
{
  idb::IdbBuilder* idb_builder = _database.get_idb_builder();
  Die& die = _database.get_die();
  std::vector<Net>& net_list = _database.get_net_list();

  std::map<int32_t, std::vector<idb::IdbRegularWireSegment*>> net_idb_segment_map;
  for (auto& [net_idx, segment_set] : getDetailedNetResultMap(die)) {
    for (Segment<LayerCoord>* segment : segment_set) {
      net_idb_segment_map[net_idx].push_back(getIDBSegmentByNetResult(net_idx, *segment));
    }
  }
  for (auto& [net_idx, patch_set] : getNetPatchMap(die)) {
    for (EXTLayerRect* patch : patch_set) {
      net_idb_segment_map[net_idx].push_back(getIDBSegmentByNetPatch(net_idx, *patch));
    }
  }
  idb::IdbNetList* idb_net_list = idb_builder->get_def_service()->get_design()->get_net_list();
  if (idb_net_list == nullptr) {
    RTLOG.error(Loc::current(), "The idb net list is empty!");
  }
  for (auto& [net_idx, idb_segment_list] : net_idb_segment_map) {
    std::string net_name = net_list[net_idx].get_net_name();
    idb::IdbNet* idb_net = idb_net_list->find_net(net_name);
    if (idb_net == nullptr) {
      RTLOG.info(Loc::current(), "The idb net named ", net_name, " cannot be found!");
      continue;
    }
    idb_net->clear_wire_list();
    idb::IdbRegularWireList* idb_wire_list = idb_net->get_wire_list();
    if (idb_wire_list == nullptr) {
      RTLOG.error(Loc::current(), "The idb wire list is empty!");
    }
    idb::IdbRegularWire* idb_wire = idb_wire_list->add_wire();
    idb_wire->set_wire_state(idb::IdbWiringStatement::kRouted);

    int32_t print_new = false;
    for (idb::IdbRegularWireSegment* idb_segment : idb_segment_list) {
      idb_wire->add_segment(idb_segment);
      if (print_new == false) {
        idb_segment->set_layer_as_new();
        print_new = true;
      }
    }
  }
}

#endif

#if 1  // 获得IdbWireSegment

idb::IdbRegularWireSegment* DataManager::getIDBWire(int32_t net_idx, Segment<LayerCoord>& segment)
{
  idb::IdbLayers* idb_layer_list = _database.get_idb_builder()->get_def_service()->get_layout()->get_layers();
  std::vector<RoutingLayer>& routing_layer_list = _database.get_routing_layer_list();

  LayerCoord& first_coord = segment.get_first();
  LayerCoord& second_coord = segment.get_second();
  int32_t layer_idx = first_coord.get_layer_idx();

  if (RTUTIL.isOblique(first_coord, second_coord)) {
    RTLOG.error(Loc::current(), "The wire is oblique!");
  }
  std::string layer_name = routing_layer_list[layer_idx].get_layer_name();
  idb::IdbLayer* idb_layer = idb_layer_list->find_layer(layer_name);
  if (idb_layer == nullptr) {
    RTLOG.error(Loc::current(), "Can not find idb layer ", layer_name);
  }
  idb::IdbRegularWireSegment* idb_segment = new idb::IdbRegularWireSegment();
  idb_segment->set_layer(idb_layer);
  idb_segment->add_point(first_coord.get_x(), first_coord.get_y());
  idb_segment->add_point(second_coord.get_x(), second_coord.get_y());
  return idb_segment;
}

idb::IdbRegularWireSegment* DataManager::getIDBVia(int32_t net_idx, Segment<LayerCoord>& segment)
{
  idb::IdbVias* lef_via_list = _database.get_idb_builder()->get_lef_service()->get_layout()->get_via_list();
  idb::IdbVias* def_via_list = _database.get_idb_builder()->get_def_service()->get_design()->get_via_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _database.get_layer_via_master_list();

  LayerCoord& first_coord = segment.get_first();
  LayerCoord& second_coord = segment.get_second();
  int32_t below_layer_idx = std::min(first_coord.get_layer_idx(), second_coord.get_layer_idx());

  if (below_layer_idx < 0 || below_layer_idx >= static_cast<int32_t>(layer_via_master_list.size())) {
    RTLOG.error(Loc::current(), "The via below_layer_idx is illegal!");
  }
  std::string via_name = layer_via_master_list[below_layer_idx].front().get_via_name();
  idb::IdbVia* idb_via = lef_via_list->find_via(via_name);
  if (idb_via == nullptr) {
    idb_via = def_via_list->find_via(via_name);
  }
  if (idb_via == nullptr) {
    RTLOG.error(Loc::current(), "Can not find idb via ", via_name, "!");
  }
  idb::IdbLayer* idb_layer_top = idb_via->get_instance()->get_top_layer_shape()->get_layer();
  if (idb_layer_top == nullptr) {
    RTLOG.error(Loc::current(), "Can not find layer from idb via ", via_name, "!");
  }
  idb::IdbRegularWireSegment* idb_segment = new idb::IdbRegularWireSegment();
  idb_segment->set_layer(idb_layer_top);
  idb_segment->set_is_via(true);
  idb_segment->add_point(first_coord.get_x(), first_coord.get_y());
  idb::IdbVia* idb_via_new = idb_segment->copy_via(idb_via);
  idb_via_new->set_coordinate(first_coord.get_x(), first_coord.get_y());
  return idb_segment;
}

#endif

}  // namespace irt
