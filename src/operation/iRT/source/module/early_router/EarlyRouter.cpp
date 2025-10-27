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
#include "EarlyRouter.hpp"

#include "GDSPlotter.hpp"
#include "Monitor.hpp"
#include "RTInterface.hpp"
#include "Utility.hpp"

namespace irt {

// public

void EarlyRouter::initInst()
{
  if (_er_instance == nullptr) {
    _er_instance = new EarlyRouter();
  }
}

EarlyRouter& EarlyRouter::getInst()
{
  if (_er_instance == nullptr) {
    RTLOG.error(Loc::current(), "The instance not initialized!");
  }
  return *_er_instance;
}

void EarlyRouter::destroyInst()
{
  if (_er_instance != nullptr) {
    delete _er_instance;
    _er_instance = nullptr;
  }
}

// function

void EarlyRouter::route()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");
  ERModel er_model = initERModel();
  setERComParam(er_model);
  initAccessPointList(er_model);
  // debugPlotERModel(er_model, "before");
  buildConflictList(er_model);
  eliminateConflict(er_model);
  uploadAccessPoint(er_model);
  uploadAccessPatch(er_model);
  // debugPlotERModel(er_model, "middle");
  buildSupplySchedule(er_model);
  analyzeSupply(er_model);
  buildIgnoreNet(er_model);
  analyzeDemandUnit(er_model);
  initERTaskList(er_model);
  buildPlanarNodeMap(er_model);
  buildPlanarNodeNeighbor(er_model);
  buildPlanarOrientSupply(er_model);
  generateTopology(er_model);
  buildLayerNodeMap(er_model);
  buildLayerNodeNeighbor(er_model);
  buildLayerOrientSupply(er_model);
  assignLayer(er_model);
  outputResult(er_model);
  // debugPlotERModel(er_model, "after");
  cleanTempResult(er_model);
  updateSummary(er_model);
  printSummary(er_model);
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

// private

EarlyRouter* EarlyRouter::_er_instance = nullptr;

ERModel EarlyRouter::initERModel()
{
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();

  ERModel er_model;
  er_model.set_er_net_list(convertToERNetList(net_list));
  return er_model;
}

std::vector<ERNet> EarlyRouter::convertToERNetList(std::vector<Net>& net_list)
{
  std::vector<ERNet> er_net_list;
  er_net_list.reserve(net_list.size());
  for (size_t i = 0; i < net_list.size(); i++) {
    er_net_list.emplace_back(convertToERNet(net_list[i]));
  }
  return er_net_list;
}

ERNet EarlyRouter::convertToERNet(Net& net)
{
  ERNet er_net;
  er_net.set_origin_net(&net);
  er_net.set_net_idx(net.get_net_idx());
  er_net.set_connect_type(net.get_connect_type());
  for (Pin& pin : net.get_pin_list()) {
    er_net.get_er_pin_list().push_back(ERPin(pin));
  }
  er_net.set_bounding_box(net.get_bounding_box());
  return er_net;
}

void EarlyRouter::setERComParam(ERModel& er_model)
{
  int32_t max_candidate_point_num = 10;
  int32_t supply_reduction = 0;
  double boundary_wire_unit = 1;
  double internal_wire_unit = 1;
  double internal_via_unit = 1;
  int32_t topo_spilt_length = 10;
  int32_t expand_step_num = 5;
  int32_t expand_step_length = 2;
  double prefer_wire_unit = 1;
  double non_prefer_wire_unit = 2.5 * prefer_wire_unit;
  double via_unit = 2 * non_prefer_wire_unit;
  double overflow_unit = 4 * non_prefer_wire_unit;

  /**
   * max_candidate_point_num, supply_reduction, boundary_wire_unit, internal_wire_unit, internal_via_unit, topo_spilt_length, expand_step_num,
   * expand_step_length, via_unit, overflow_unit
   */
  ERComParam er_com_param(max_candidate_point_num, supply_reduction, boundary_wire_unit, internal_wire_unit, internal_via_unit, topo_spilt_length,
                          expand_step_num, expand_step_length, via_unit, overflow_unit);
  RTLOG.info(Loc::current(), "max_candidate_point_num: ", er_com_param.get_max_candidate_point_num());
  RTLOG.info(Loc::current(), "supply_reduction: ", er_com_param.get_supply_reduction());
  RTLOG.info(Loc::current(), "boundary_wire_unit: ", er_com_param.get_boundary_wire_unit());
  RTLOG.info(Loc::current(), "internal_wire_unit: ", er_com_param.get_internal_wire_unit());
  RTLOG.info(Loc::current(), "internal_via_unit: ", er_com_param.get_internal_via_unit());
  RTLOG.info(Loc::current(), "topo_spilt_length: ", er_com_param.get_topo_spilt_length());
  RTLOG.info(Loc::current(), "expand_step_num: ", er_com_param.get_expand_step_num());
  RTLOG.info(Loc::current(), "expand_step_length: ", er_com_param.get_expand_step_length());
  RTLOG.info(Loc::current(), "via_unit: ", er_com_param.get_via_unit());
  RTLOG.info(Loc::current(), "overflow_unit: ", er_com_param.get_overflow_unit());

  er_model.set_er_com_param(er_com_param);
}

void EarlyRouter::initAccessPointList(ERModel& er_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();

  std::vector<ERNet>& er_net_list = er_model.get_er_net_list();

  for (ERNet& er_net : er_net_list) {
    for (ERPin& er_pin : er_net.get_er_pin_list()) {
      std::vector<std::pair<int32_t, std::vector<EXTLayerRect>>> routing_pin_shape_list;
      {
        std::map<int32_t, std::vector<EXTLayerRect>> routing_pin_shape_map;
        for (EXTLayerRect& routing_shape : er_pin.get_routing_shape_list()) {
          routing_pin_shape_map[routing_shape.get_layer_idx()].emplace_back(routing_shape);
        }
        for (auto& [routing_layer_idx, pin_shape_list] : routing_pin_shape_map) {
          routing_pin_shape_list.emplace_back(routing_layer_idx, pin_shape_list);
        }
        if (er_pin.get_is_core()) {
          std::sort(
              routing_pin_shape_list.begin(), routing_pin_shape_list.end(),
              [](const std::pair<int32_t, std::vector<EXTLayerRect>>& a, const std::pair<int32_t, std::vector<EXTLayerRect>>& b) { return a.first > b.first; });
        } else {
          std::sort(routing_pin_shape_list.begin(), routing_pin_shape_list.end(),
                    [](const std::pair<int32_t, std::vector<EXTLayerRect>>& a, const std::pair<int32_t, std::vector<EXTLayerRect>>& b) {
                      return (a.first % 2 != 0 && b.first % 2 == 0) || (a.first % 2 == b.first % 2 && a.first > b.first);
                    });
        }
      }
      if (routing_pin_shape_list.empty()) {
        RTLOG.error(Loc::current(), "The routing_pin_shape_list is empty!");
      }
      for (LayerCoord access_coord : getAccessCoordList(er_model, routing_pin_shape_list.front().second)) {
        er_pin.get_access_point_list().emplace_back(er_pin.get_pin_idx(), access_coord);
      }
    }

    std::vector<PlanarCoord> coord_list;
    for (ERPin& er_pin : er_net.get_er_pin_list()) {
      for (AccessPoint& access_point : er_pin.get_access_point_list()) {
        coord_list.push_back(access_point.get_real_coord());
      }
    }
    BoundingBox& bounding_box = er_net.get_bounding_box();
    bounding_box.set_real_rect(RTUTIL.getBoundingBox(coord_list));
    bounding_box.set_grid_rect(RTUTIL.getOpenGCellGridRect(bounding_box.get_real_rect(), gcell_axis));
    for (ERPin& er_pin : er_net.get_er_pin_list()) {
      for (AccessPoint& access_point : er_pin.get_access_point_list()) {
        access_point.set_grid_coord(RTUTIL.getGCellGridCoordByBBox(access_point.get_real_coord(), gcell_axis, bounding_box));
        RTDM.updateNetAccessPointToGCellMap(ChangeType::kAdd, er_net.get_net_idx(), &access_point);
      }
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

std::vector<LayerCoord> EarlyRouter::getAccessCoordList(ERModel& er_model, std::vector<EXTLayerRect>& pin_shape_list)
{
  Die& die = RTDM.getDatabase().get_die();
  int32_t manufacture_grid = RTDM.getDatabase().get_manufacture_grid();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::map<int32_t, PlanarRect>& layer_enclosure_map = RTDM.getDatabase().get_layer_enclosure_map();

  int32_t curr_layer_idx;
  {
    for (EXTLayerRect& pin_shape : pin_shape_list) {
      if (pin_shape_list.front().get_layer_idx() != pin_shape.get_layer_idx()) {
        RTLOG.error(Loc::current(), "The pin_shape_list is not on the same layer!");
      }
    }
    curr_layer_idx = pin_shape_list.front().get_layer_idx();
  }
  std::vector<PlanarRect> legal_rect_list;
  {
    std::vector<PlanarRect> origin_pin_shape_list;
    for (EXTLayerRect& pin_shape : pin_shape_list) {
      origin_pin_shape_list.push_back(pin_shape.get_real_rect());
    }
    std::vector<PlanarRect> shrinked_rect_list;
    {
      PlanarRect& enclosure = layer_enclosure_map[curr_layer_idx];
      int32_t enclosure_half_x_span = enclosure.getXSpan() / 2;
      int32_t enclosure_half_y_span = enclosure.getYSpan() / 2;
      int32_t half_min_width = routing_layer_list[curr_layer_idx].get_min_width() / 2;
      int32_t shrinked_x_size = std::max(half_min_width, enclosure_half_x_span);
      int32_t shrinked_y_size = std::max(half_min_width, enclosure_half_y_span);
      for (PlanarRect& real_rect :
           RTUTIL.getClosedShrinkedRectListByBoost(origin_pin_shape_list, shrinked_x_size, shrinked_y_size, shrinked_x_size, shrinked_y_size)) {
        shrinked_rect_list.push_back(real_rect);
      }
    }
    if (shrinked_rect_list.empty()) {
      legal_rect_list = origin_pin_shape_list;
    } else {
      legal_rect_list = shrinked_rect_list;
    }
  }
  std::vector<LayerCoord> layer_coord_list;
  for (PlanarRect& legal_shape : legal_rect_list) {
    int32_t ll_x = legal_shape.get_ll_x();
    int32_t ll_y = legal_shape.get_ll_y();
    int32_t ur_x = legal_shape.get_ur_x();
    int32_t ur_y = legal_shape.get_ur_y();
    // 避免 off_grid
    while (ll_x % manufacture_grid != 0) {
      ll_x++;
    }
    while (ll_y % manufacture_grid != 0) {
      ll_y++;
    }
    while (ur_x % manufacture_grid != 0) {
      ur_x--;
    }
    while (ur_y % manufacture_grid != 0) {
      ur_y--;
    }
    RoutingLayer& curr_routing_layer = routing_layer_list[curr_layer_idx];
    std::vector<int32_t> x_track_list = RTUTIL.getScaleList(ll_x, ur_x, curr_routing_layer.getXTrackGridList());
    std::vector<int32_t> y_track_list = RTUTIL.getScaleList(ll_y, ur_y, curr_routing_layer.getYTrackGridList());
    std::vector<int32_t> x_shape_list;
    {
      x_shape_list.emplace_back(ll_x);
      if ((ur_x - ll_x) / manufacture_grid % 2 == 0) {
        x_shape_list.emplace_back((ll_x + ur_x) / 2);
      } else {
        x_shape_list.emplace_back((ll_x + ur_x - manufacture_grid) / 2);
        x_shape_list.emplace_back((ll_x + ur_x + manufacture_grid) / 2);
      }
      x_shape_list.emplace_back(ur_x);
    }
    std::vector<int32_t> y_shape_list;
    {
      y_shape_list.emplace_back(ll_y);
      if ((ur_y - ll_y) / manufacture_grid % 2 == 0) {
        y_shape_list.emplace_back((ll_y + ur_y) / 2);
      } else {
        y_shape_list.emplace_back((ll_y + ur_y - manufacture_grid) / 2);
        y_shape_list.emplace_back((ll_y + ur_y + manufacture_grid) / 2);
      }
      y_shape_list.emplace_back(ur_y);
    }
    // track grid
    for (int32_t x : x_track_list) {
      for (int32_t y : y_track_list) {
        layer_coord_list.emplace_back(x, y, curr_layer_idx);
      }
    }
    // on track
    {
      for (int32_t x : x_shape_list) {
        for (int32_t y : y_track_list) {
          layer_coord_list.emplace_back(x, y, curr_layer_idx);
        }
      }
      for (int32_t x : x_track_list) {
        for (int32_t y : y_shape_list) {
          layer_coord_list.emplace_back(x, y, curr_layer_idx);
        }
      }
    }
    // on shape
    for (int32_t x : x_shape_list) {
      for (int32_t y : y_shape_list) {
        layer_coord_list.emplace_back(x, y, curr_layer_idx);
      }
    }
  }
  {
    PlanarRect die_valid_rect = die.get_real_rect();
    int32_t shrinked_size = RTDM.getOnlyPitch();
    if (RTUTIL.hasShrinkedRect(die_valid_rect, shrinked_size)) {
      die_valid_rect = RTUTIL.getShrinkedRect(die_valid_rect, shrinked_size);
    }
    std::vector<LayerCoord> new_layer_coord_list;
    for (LayerCoord& layer_coord : layer_coord_list) {
      if (RTUTIL.isInside(die_valid_rect, layer_coord)) {
        new_layer_coord_list.push_back(layer_coord);
      }
    }
    layer_coord_list = new_layer_coord_list;
  }
  {
    for (LayerCoord& layer_coord : layer_coord_list) {
      if (layer_coord.get_x() % manufacture_grid != 0) {
        RTLOG.error(Loc::current(), "The coord is off_grid!");
      }
      if (layer_coord.get_y() % manufacture_grid != 0) {
        RTLOG.error(Loc::current(), "The coord is off_grid!");
      }
    }
  }
  std::sort(layer_coord_list.begin(), layer_coord_list.end(), CmpLayerCoordByXASC());
  layer_coord_list.erase(std::unique(layer_coord_list.begin(), layer_coord_list.end()), layer_coord_list.end());
  uniformSampleCoordList(er_model, layer_coord_list);
  if (layer_coord_list.empty()) {
    RTLOG.error(Loc::current(), "The layer_coord_list is empty!");
  }
  return layer_coord_list;
}

void EarlyRouter::uniformSampleCoordList(ERModel& er_model, std::vector<LayerCoord>& layer_coord_list)
{
  int32_t max_candidate_point_num = er_model.get_er_com_param().get_max_candidate_point_num();

  PlanarRect bounding_box = RTUTIL.getBoundingBox(layer_coord_list);
  int32_t grid_num = static_cast<int32_t>(std::sqrt(max_candidate_point_num));
  double grid_x_span = bounding_box.getXSpan() / grid_num;
  double grid_y_span = bounding_box.getYSpan() / grid_num;

  std::set<PlanarCoord, CmpPlanarCoordByXASC> visited_set;
  std::vector<LayerCoord> new_layer_coord_list;
  for (LayerCoord& layer_coord : layer_coord_list) {
    PlanarCoord grid_coord(static_cast<int32_t>((layer_coord.get_x() - bounding_box.get_ll_x()) / grid_x_span),
                           static_cast<int32_t>((layer_coord.get_y() - bounding_box.get_ll_y()) / grid_y_span));
    if (!RTUTIL.exist(visited_set, grid_coord)) {
      new_layer_coord_list.push_back(layer_coord);
      visited_set.insert(grid_coord);
      if (static_cast<int32_t>(new_layer_coord_list.size()) >= max_candidate_point_num) {
        break;
      }
    }
  }
  layer_coord_list = new_layer_coord_list;
}

void EarlyRouter::buildConflictList(ERModel& er_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<ERConflictGroup>& er_conflict_group_list = er_model.get_er_conflict_group_list();

  std::map<ERPin*, std::set<ERPin*>> pin_conflict_map;
  for (auto& [curr_pin, conflict_pin_set] : getPinConlictMap(er_model)) {
    pin_conflict_map[curr_pin] = conflict_pin_set;
  }
  for (auto& [curr_pin, conflict_pin_set] : pin_conflict_map) {
    if (conflict_pin_set.empty()) {
      continue;
    }
    std::vector<std::pair<ERPin*, ERPin*>> conflict_list;
    std::map<ERPin*, int32_t> pin_idx_map;
    std::queue<ERPin*> pin_queue = RTUTIL.initQueue(curr_pin);
    while (!pin_queue.empty()) {
      ERPin* er_pin = RTUTIL.getFrontAndPop(pin_queue);
      if (!RTUTIL.exist(pin_idx_map, er_pin)) {
        pin_idx_map[er_pin] = pin_idx_map.size();
      }
      if (!RTUTIL.exist(pin_conflict_map, er_pin)) {
        continue;
      }
      std::set<ERPin*>& conflict_pin_set = pin_conflict_map[er_pin];
      for (ERPin* conflict_pin : conflict_pin_set) {
        conflict_list.emplace_back(er_pin, conflict_pin);
        pin_queue.push(conflict_pin);
      }
      conflict_pin_set.clear();
    }
    ERConflictGroup er_conflict_group;
    std::vector<std::vector<ERConflictPoint>>& conflict_point_list_list = er_conflict_group.get_conflict_point_list_list();
    conflict_point_list_list.resize(pin_idx_map.size());
    for (auto& [er_pin, conflict_point_list_idx] : pin_idx_map) {
      std::vector<ERConflictPoint> conflict_point_list;
      for (AccessPoint& access_point : er_pin->get_access_point_list()) {
        ERConflictPoint conflict_point;
        conflict_point.set_er_pin(er_pin);
        conflict_point.set_access_point(&access_point);
        conflict_point.set_coord(access_point.get_real_coord());
        conflict_point.set_layer_idx(access_point.get_layer_idx());
        conflict_point_list.push_back(conflict_point);
      }
      conflict_point_list_list[conflict_point_list_idx] = conflict_point_list;
    }
    std::map<int32_t, std::vector<int32_t>>& conflict_map = er_conflict_group.get_conflict_map();
    for (std::pair<ERPin*, ERPin*>& conflict_pair : conflict_list) {
      conflict_map[pin_idx_map[conflict_pair.first]].push_back(pin_idx_map[conflict_pair.second]);
    }
    er_conflict_group_list.push_back(er_conflict_group);
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

std::vector<std::pair<ERPin*, std::set<ERPin*>>> EarlyRouter::getPinConlictMap(ERModel& er_model)
{
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  Die& die = RTDM.getDatabase().get_die();

  std::vector<ERNet>& er_net_list = er_model.get_er_net_list();

  std::vector<std::pair<ERPin*, std::set<ERPin*>>> pin_conflict_list;
  for (ERNet& er_net : er_net_list) {
    for (ERPin& er_pin : er_net.get_er_pin_list()) {
      pin_conflict_list.emplace_back(&er_pin, std::set<ERPin*>{});
    }
  }
#pragma omp parallel for
  for (std::pair<ERPin*, std::set<ERPin*>>& pin_conflict_pair : pin_conflict_list) {
    ERPin* er_pin = pin_conflict_pair.first;
    std::set<ERPin*>& conflict_pin_set = pin_conflict_pair.second;

    for (AccessPoint& access_point : er_pin->get_access_point_list()) {
      PlanarCoord& grid_coord = access_point.get_grid_coord();
      for (int32_t x : {grid_coord.get_x() - 1, grid_coord.get_x(), grid_coord.get_x() + 1}) {
        for (int32_t y : {grid_coord.get_y() - 1, grid_coord.get_y(), grid_coord.get_y() + 1}) {
          if (!RTUTIL.isInside(die.get_grid_rect(), PlanarCoord(x, y))) {
            continue;
          }
          for (auto& [net_idx, access_point_set] : gcell_map[x][y].get_net_access_point_map()) {
            for (AccessPoint* gcell_access_point : access_point_set) {
              ERPin* gcell_pin = &er_net_list[net_idx].get_er_pin_list()[gcell_access_point->get_pin_idx()];
              if (gcell_pin == er_pin) {
                continue;
              }
              if (hasConflict(access_point, *gcell_access_point)) {
                conflict_pin_set.insert(gcell_pin);
              }
            }
          }
        }
      }
    }
  }
  return pin_conflict_list;
}

bool EarlyRouter::hasConflict(AccessPoint& curr_access_point, AccessPoint& gcell_access_point)
{
  return hasConflict(curr_access_point.getRealLayerCoord(), gcell_access_point.getRealLayerCoord());
}

bool EarlyRouter::hasConflict(LayerCoord layer_coord1, LayerCoord layer_coord2)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  std::set<int32_t> conflict_layer_idx_set;
  {
    int32_t start_layer_idx = layer_coord1.get_layer_idx();
    int32_t end_layer_idx = layer_coord2.get_layer_idx();
    RTUTIL.swapByASC(start_layer_idx, end_layer_idx);
    for (int32_t layer_idx = start_layer_idx; layer_idx <= end_layer_idx; layer_idx++) {
      if (layer_idx < (static_cast<int32_t>(routing_layer_list.size()) - 1)) {
        conflict_layer_idx_set.insert(layer_idx);
        conflict_layer_idx_set.insert(layer_idx + 1);
      } else {
        conflict_layer_idx_set.insert(layer_idx);
        conflict_layer_idx_set.insert(layer_idx - 1);
      }
    }
  }
  PlanarCoord& planar_coord1 = layer_coord1.get_planar_coord();
  PlanarCoord& planar_coord2 = layer_coord2.get_planar_coord();
  for (int32_t conflict_layer_idx : conflict_layer_idx_set) {
    RoutingLayer& routing_layer = routing_layer_list[conflict_layer_idx];
    int32_t min_width = routing_layer.get_min_width();
    int32_t min_length = routing_layer.get_min_area() / min_width;

    int32_t x_distance = 0;
    int32_t y_distance = 0;
    if (routing_layer.isPreferH()) {
      x_distance = min_length;
      y_distance = min_width;
    } else {
      x_distance = min_width;
      y_distance = min_length;
    }
    PlanarRect searched_rect = RTUTIL.getEnlargedRect(planar_coord1, x_distance, y_distance, x_distance, y_distance);
    if (RTUTIL.isInside(searched_rect, planar_coord2, false)) {
      return true;
    }
  }
  return false;
}

void EarlyRouter::eliminateConflict(ERModel& er_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  // 给有conflict的pin设置
  for (ERConflictGroup& er_conflict_group : er_model.get_er_conflict_group_list()) {
    for (ERConflictPoint& best_point : getBestPointList(er_conflict_group)) {
      best_point.get_er_pin()->set_access_point(*best_point.get_access_point());
    }
  }
  // 给没有conflict的pin设置
  for (ERNet& er_net : er_model.get_er_net_list()) {
    for (ERPin& er_pin : er_net.get_er_pin_list()) {
      if (er_pin.get_access_point().get_layer_idx() < 0) {
        er_pin.set_access_point(er_pin.get_access_point_list().front());
      }
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

std::vector<ERConflictPoint> EarlyRouter::getBestPointList(ERConflictGroup& er_conflict_group)
{
  std::vector<std::vector<ERConflictPoint>>& conflict_point_list_list = er_conflict_group.get_conflict_point_list_list();
  std::map<int32_t, std::vector<int32_t>>& conflict_map = er_conflict_group.get_conflict_map();

  std::vector<ERConflictPoint> curr_ap_list;
  for (std::vector<ERConflictPoint>& conflict_point_list : conflict_point_list_list) {
    curr_ap_list.push_back(conflict_point_list.front());
  }
  bool improved = true;
  while (improved) {
    improved = false;
    for (int32_t i = 0; i < static_cast<int32_t>(conflict_point_list_list.size()); ++i) {
      std::vector<int32_t> conflict_j_list;
      if (RTUTIL.exist(conflict_map, i)) {
        conflict_j_list = conflict_map[i];
      } else {
        RTLOG.error(Loc::current(), "The conflict_map is not exist i!");
      }
      int32_t best_conflict_count = INT32_MAX;
      int32_t best_min_distance = INT32_MAX;
      ERConflictPoint best_ap = curr_ap_list[i];
      for (ERConflictPoint& conflict_point : conflict_point_list_list[i]) {
        int32_t conflict_count = 0;
        int32_t min_distance = INT32_MAX;
        for (int32_t j : conflict_j_list) {
          if (hasConflict(conflict_point, curr_ap_list[j])) {
            ++conflict_count;
          }
          min_distance = std::min(min_distance, RTUTIL.getManhattanDistance(conflict_point, curr_ap_list[j]));
        }
        // 优先比较冲突数，其次比较最小曼哈顿距离（越小越好）
        if (conflict_count < best_conflict_count || (conflict_count == best_conflict_count && min_distance < best_min_distance)) {
          best_conflict_count = conflict_count;
          best_min_distance = min_distance;
          best_ap = conflict_point;
        }
      }
      if (best_ap.get_access_point() != curr_ap_list[i].get_access_point()) {
        curr_ap_list[i] = best_ap;
        improved = true;
      }
    }
  }
  return curr_ap_list;
}

void EarlyRouter::uploadAccessPoint(ERModel& er_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();

  for (auto& [net_idx, access_point_set] : RTDM.getNetAccessPointMap(die)) {
    for (AccessPoint* access_point : access_point_set) {
      RTDM.updateNetAccessPointToGCellMap(ChangeType::kDel, net_idx, access_point);
    }
  }
  for (ERNet& er_net : er_model.get_er_net_list()) {
    std::vector<PlanarCoord> coord_list;
    for (ERPin& er_pin : er_net.get_er_pin_list()) {
      coord_list.push_back(er_pin.get_access_point().get_real_coord());
    }
    BoundingBox& bounding_box = er_net.get_bounding_box();
    bounding_box.set_real_rect(RTUTIL.getBoundingBox(coord_list));
    bounding_box.set_grid_rect(RTUTIL.getOpenGCellGridRect(bounding_box.get_real_rect(), gcell_axis));
    for (ERPin& er_pin : er_net.get_er_pin_list()) {
      AccessPoint& access_point = er_pin.get_access_point();
      access_point.set_grid_coord(RTUTIL.getGCellGridCoordByBBox(access_point.get_real_coord(), gcell_axis, bounding_box));
      RTDM.updateNetAccessPointToGCellMap(ChangeType::kAdd, er_net.get_net_idx(), &access_point);
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void EarlyRouter::uploadAccessPatch(ERModel& er_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::map<int32_t, PlanarRect>& layer_enclosure_map = RTDM.getDatabase().get_layer_enclosure_map();
  int32_t bottom_routing_layer_idx = RTDM.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = RTDM.getConfig().top_routing_layer_idx;
  int32_t only_pitch = RTDM.getOnlyPitch();

  for (ERNet& er_net : er_model.get_er_net_list()) {
    for (ERPin& er_pin : er_net.get_er_pin_list()) {
      PlanarCoord access_coord = er_pin.get_access_point().get_real_coord();
      int32_t curr_layer_idx = er_pin.get_access_point().get_layer_idx();
      {
        RoutingLayer& routing_layer = routing_layer_list[curr_layer_idx];
        std::vector<int32_t> x_track_list
            = RTUTIL.getScaleList(access_coord.get_x() - only_pitch, access_coord.get_x() + only_pitch, routing_layer.getXTrackGridList());
        std::vector<int32_t> y_track_list
            = RTUTIL.getScaleList(access_coord.get_y() - only_pitch, access_coord.get_y() + only_pitch, routing_layer.getYTrackGridList());

        int32_t min_distance = INT_MAX;
        PlanarCoord best_coord = access_coord;
        for (int32_t x : x_track_list) {
          for (int32_t y : y_track_list) {
            PlanarCoord track_coord(x, y);
            int32_t distance = RTUTIL.getManhattanDistance(access_coord, track_coord);
            if (distance < min_distance) {
              min_distance = distance;
              best_coord = track_coord;
            }
          }
        }
        access_coord = best_coord;
      }
      int32_t min_layer_idx = curr_layer_idx;
      int32_t max_layer_idx = curr_layer_idx;
      {
        if (er_pin.get_is_core()) {
          if (curr_layer_idx < bottom_routing_layer_idx) {
            max_layer_idx = bottom_routing_layer_idx + 1;
          } else if (top_routing_layer_idx < curr_layer_idx) {
            max_layer_idx = top_routing_layer_idx - 1;
          } else if (curr_layer_idx < top_routing_layer_idx) {
            max_layer_idx = curr_layer_idx + 1;
          } else {
            max_layer_idx = curr_layer_idx - 1;
          }
        } else {
          if (curr_layer_idx < bottom_routing_layer_idx) {
            max_layer_idx = bottom_routing_layer_idx;
          } else if (top_routing_layer_idx < curr_layer_idx) {
            max_layer_idx = top_routing_layer_idx;
          } else if (curr_layer_idx < top_routing_layer_idx) {
            max_layer_idx = curr_layer_idx;
          } else {
            max_layer_idx = curr_layer_idx;
          }
        }
        RTUTIL.swapByASC(min_layer_idx, max_layer_idx);
      }
      for (int32_t layer_idx = min_layer_idx; layer_idx <= max_layer_idx; layer_idx++) {
        if (layer_idx == curr_layer_idx) {
          continue;
        }
        RoutingLayer& routing_layer = routing_layer_list[layer_idx];
        int32_t half_width = routing_layer.get_min_width() / 2;
        int32_t min_length = routing_layer.get_min_area() / routing_layer.get_min_width();
        PlanarRect& enclosure = layer_enclosure_map[layer_idx];
        int32_t half_x_span = enclosure.getXSpan() / 2;
        int32_t half_y_span = enclosure.getYSpan() / 2;

        EXTLayerRect patch;
        if (routing_layer.isPreferH()) {
          patch.set_real_rect(RTUTIL.getEnlargedRect(access_coord, min_length - half_x_span, half_width, half_x_span, half_width));
        } else {
          patch.set_real_rect(RTUTIL.getEnlargedRect(access_coord, half_width, min_length - half_y_span, half_width, half_y_span));
        }
        patch.set_grid_rect(RTUTIL.getClosedGCellGridRect(patch.get_real_rect(), gcell_axis));
        patch.set_layer_idx(layer_idx);
        RTDM.updateNetDetailedPatchToGCellMap(ChangeType::kAdd, er_net.get_net_idx(), new EXTLayerRect(patch));
      }
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void EarlyRouter::buildSupplySchedule(ERModel& er_model)
{
  Die& die = RTDM.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  int32_t bottom_routing_layer_idx = RTDM.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = RTDM.getConfig().top_routing_layer_idx;

  for (RoutingLayer& routing_layer : routing_layer_list) {
    if (routing_layer.get_layer_idx() < bottom_routing_layer_idx || top_routing_layer_idx < routing_layer.get_layer_idx()) {
      continue;
    }
    if (routing_layer.isPreferH()) {
      for (int32_t begin_x = 1; begin_x <= 2; begin_x++) {
        std::vector<std::pair<LayerCoord, LayerCoord>> grid_pair_list;
        for (int32_t y = 0; y < die.getYSize(); y++) {
          for (int32_t x = begin_x; x < die.getXSize(); x += 2) {
            grid_pair_list.emplace_back(LayerCoord(x - 1, y, routing_layer.get_layer_idx()), LayerCoord(x, y, routing_layer.get_layer_idx()));
          }
        }
        er_model.get_grid_pair_list_list().push_back(grid_pair_list);
      }
    } else {
      for (int32_t begin_y = 1; begin_y <= 2; begin_y++) {
        std::vector<std::pair<LayerCoord, LayerCoord>> grid_pair_list;
        for (int32_t x = 0; x < die.getXSize(); x++) {
          for (int32_t y = begin_y; y < die.getYSize(); y += 2) {
            grid_pair_list.emplace_back(LayerCoord(x, y - 1, routing_layer.get_layer_idx()), LayerCoord(x, y, routing_layer.get_layer_idx()));
          }
        }
        er_model.get_grid_pair_list_list().push_back(grid_pair_list);
      }
    }
  }
}

void EarlyRouter::analyzeSupply(ERModel& er_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  int32_t supply_reduction = er_model.get_er_com_param().get_supply_reduction();

  size_t total_pair_num = 0;
  for (std::vector<std::pair<LayerCoord, LayerCoord>>& grid_pair_list : er_model.get_grid_pair_list_list()) {
    total_pair_num += grid_pair_list.size();
  }

  size_t analyzed_pair_num = 0;
  for (std::vector<std::pair<LayerCoord, LayerCoord>>& grid_pair_list : er_model.get_grid_pair_list_list()) {
    Monitor stage_monitor;
#pragma omp parallel for
    for (std::pair<LayerCoord, LayerCoord>& grid_pair : grid_pair_list) {
      LayerCoord first_coord = grid_pair.first;
      LayerCoord second_coord = grid_pair.second;
      EXTLayerRect search_rect = getSearchRect(first_coord, second_coord);

      std::map<Orientation, int32_t>& first_orient_supply_map
          = gcell_map[first_coord.get_x()][first_coord.get_y()].get_routing_orient_supply_map()[search_rect.get_layer_idx()];
      std::map<Orientation, int32_t>& second_orient_supply_map
          = gcell_map[second_coord.get_x()][second_coord.get_y()].get_routing_orient_supply_map()[search_rect.get_layer_idx()];

      Orientation first_orientation = RTUTIL.getOrientation(first_coord, second_coord);
      Orientation second_orientation = RTUTIL.getOppositeOrientation(first_orientation);

      std::vector<PlanarRect> obs_rect_list;
      {
        for (auto& [is_routing, layer_net_fixed_rect_map] : RTDM.getTypeLayerNetFixedRectMap(search_rect)) {
          if (!is_routing) {
            continue;
          }
          for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
            if (search_rect.get_layer_idx() != layer_idx) {
              continue;
            }
            for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
              for (EXTLayerRect* fixed_rect : fixed_rect_set) {
                obs_rect_list.push_back(fixed_rect->get_real_rect());
              }
            }
          }
        }
        for (auto& [net_idx, segment_set] : RTDM.getNetDetailedResultMap(search_rect)) {
          for (Segment<LayerCoord>* segment : segment_set) {
            for (NetShape& net_shape : RTDM.getNetDetailedShapeList(net_idx, *segment)) {
              if (!net_shape.get_is_routing()) {
                continue;
              }
              if (search_rect.get_layer_idx() != net_shape.get_layer_idx()) {
                continue;
              }
              obs_rect_list.push_back(net_shape);
            }
          }
        }
        for (auto& [net_idx, patch_set] : RTDM.getNetDetailedPatchMap(search_rect)) {
          for (EXTLayerRect* patch : patch_set) {
            if (search_rect.get_layer_idx() != patch->get_layer_idx()) {
              continue;
            }
            obs_rect_list.push_back(patch->get_real_rect());
          }
        }
      }
      std::vector<LayerRect> wire_list = getCrossingWireList(search_rect);
      int32_t max_supply = std::max(0, static_cast<int32_t>(wire_list.size()) - supply_reduction);

      int32_t supply = 0;
      for (LayerRect& wire : wire_list) {
        if (isAccess(wire, obs_rect_list)) {
          supply++;
        }
      }
      if (supply > 0) {
        first_orient_supply_map[first_orientation] = std::min(supply, max_supply);
        second_orient_supply_map[second_orientation] = std::min(supply, max_supply);
      }
    }
    analyzed_pair_num += grid_pair_list.size();
    RTLOG.info(Loc::current(), "Analyzed ", analyzed_pair_num, "/", total_pair_num, "(", RTUTIL.getPercentage(analyzed_pair_num, total_pair_num),
               ") grid pairs", stage_monitor.getStatsInfo());
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

EXTLayerRect EarlyRouter::getSearchRect(LayerCoord& first_coord, LayerCoord& second_coord)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();

  if (first_coord.get_layer_idx() != second_coord.get_layer_idx()) {
    RTLOG.error(Loc::current(), "The grid_pair layer_idx is not equal!");
  }
  PlanarRect search_real_rect;
  {
    PlanarRect first_real_rect = RTUTIL.getRealRectByGCell(first_coord, gcell_axis);
    PlanarCoord first_mid_coord = first_real_rect.getMidPoint();
    PlanarRect second_real_rect = RTUTIL.getRealRectByGCell(second_coord, gcell_axis);
    PlanarCoord second_mid_coord = second_real_rect.getMidPoint();
    if (RTUTIL.isHorizontal(first_coord, second_coord)) {
      std::vector<PlanarCoord> coord_list;
      coord_list.emplace_back(first_mid_coord.get_x(), first_real_rect.get_ll_y());
      coord_list.emplace_back(first_mid_coord.get_x(), first_real_rect.get_ur_y());
      coord_list.emplace_back(second_mid_coord.get_x(), second_real_rect.get_ll_y());
      coord_list.emplace_back(second_mid_coord.get_x(), second_real_rect.get_ur_y());
      search_real_rect = RTUTIL.getBoundingBox(coord_list);
    } else if (RTUTIL.isVertical(first_coord, second_coord)) {
      std::vector<PlanarCoord> coord_list;
      coord_list.emplace_back(first_real_rect.get_ll_x(), first_mid_coord.get_y());
      coord_list.emplace_back(first_real_rect.get_ur_x(), first_mid_coord.get_y());
      coord_list.emplace_back(second_real_rect.get_ll_x(), second_mid_coord.get_y());
      coord_list.emplace_back(second_real_rect.get_ur_x(), second_mid_coord.get_y());
      search_real_rect = RTUTIL.getBoundingBox(coord_list);
    }
  }
  EXTLayerRect search_rect;
  search_rect.set_real_rect(search_real_rect);
  search_rect.set_grid_rect(RTUTIL.getClosedGCellGridRect(search_rect.get_real_rect(), gcell_axis));
  search_rect.set_layer_idx(first_coord.get_layer_idx());
  return search_rect;
}

std::vector<LayerRect> EarlyRouter::getCrossingWireList(EXTLayerRect& search_rect)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  RoutingLayer& routing_layer = routing_layer_list[search_rect.get_layer_idx()];
  int32_t half_wire_width = routing_layer.get_min_width() / 2;

  int32_t real_ll_x = search_rect.get_real_ll_x();
  int32_t real_ll_y = search_rect.get_real_ll_y();
  int32_t real_ur_x = search_rect.get_real_ur_x();
  int32_t real_ur_y = search_rect.get_real_ur_y();

  std::vector<LayerRect> wire_list;
  if (routing_layer.isPreferH()) {
    for (int32_t y : RTUTIL.getScaleList(real_ll_y, real_ur_y, routing_layer.getYTrackGridList())) {
      wire_list.emplace_back(real_ll_x, y - half_wire_width, real_ur_x, y + half_wire_width, search_rect.get_layer_idx());
    }
  } else {
    for (int32_t x : RTUTIL.getScaleList(real_ll_x, real_ur_x, routing_layer.getXTrackGridList())) {
      wire_list.emplace_back(x - half_wire_width, real_ll_y, x + half_wire_width, real_ur_y, search_rect.get_layer_idx());
    }
  }
  return wire_list;
}

bool EarlyRouter::isAccess(LayerRect& wire, std::vector<PlanarRect>& obs_rect_list)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  RoutingLayer& routing_layer = routing_layer_list[wire.get_layer_idx()];

  for (PlanarRect& obs_rect : obs_rect_list) {
    int32_t enlarged_size = routing_layer.getPRLSpacing(obs_rect);
    PlanarRect enlarged_rect = RTUTIL.getEnlargedRect(obs_rect, enlarged_size);
    if (RTUTIL.isOpenOverlap(enlarged_rect, wire)) {
      // 阻塞
      return false;
    }
  }
  return true;
}

void EarlyRouter::buildIgnoreNet(ERModel& er_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  int32_t bottom_routing_layer_idx = RTDM.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = RTDM.getConfig().top_routing_layer_idx;

  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      std::map<int32_t, std::map<int32_t, std::set<Orientation>>> routing_ignore_net_orient_map;
      for (auto& [is_routing, layer_net_fixed_rect_map] : gcell_map[x][y].get_type_layer_net_fixed_rect_map()) {
        if (!is_routing) {
          continue;
        }
        for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
          for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
            if (net_idx == -1) {
              continue;
            }
            for (EXTLayerRect* fixed_rect : fixed_rect_set) {
              if (RTUTIL.isClosedOverlap(gcell_map[x][y], fixed_rect->get_real_rect())) {
                routing_ignore_net_orient_map[layer_idx][net_idx] = {};
              }
            }
          }
        }
      }
      for (auto& [net_idx, segment_set] : gcell_map[x][y].get_net_detailed_result_map()) {
        for (Segment<LayerCoord>* segment : segment_set) {
          for (NetShape& net_shape : RTDM.getNetDetailedShapeList(net_idx, *segment)) {
            if (!net_shape.get_is_routing()) {
              continue;
            }
            if (RTUTIL.isClosedOverlap(gcell_map[x][y], net_shape.get_rect())) {
              routing_ignore_net_orient_map[net_shape.get_layer_idx()][net_idx] = {};
            }
          }
        }
      }
      for (auto& [net_idx, patch_set] : gcell_map[x][y].get_net_detailed_patch_map()) {
        for (EXTLayerRect* patch : patch_set) {
          if (RTUTIL.isClosedOverlap(gcell_map[x][y], patch->get_real_rect())) {
            routing_ignore_net_orient_map[patch->get_layer_idx()][net_idx] = {};
          }
        }
      }
      for (auto& [routing_layer_idx, ignore_net_orient_map] : routing_ignore_net_orient_map) {
        std::set<Orientation> ignore_orient_set;
        ignore_orient_set.insert(Orientation::kAbove);
        ignore_orient_set.insert(Orientation::kBelow);
        if (bottom_routing_layer_idx <= routing_layer_idx && routing_layer_idx <= top_routing_layer_idx) {
          if (routing_layer_list[routing_layer_idx].isPreferH()) {
            ignore_orient_set.insert(Orientation::kWest);
            ignore_orient_set.insert(Orientation::kEast);
          } else {
            ignore_orient_set.insert(Orientation::kSouth);
            ignore_orient_set.insert(Orientation::kNorth);
          }
        }
        for (auto& [net_idx, orient_set] : ignore_net_orient_map) {
          orient_set = ignore_orient_set;
        }
      }
      gcell_map[x][y].set_routing_ignore_net_orient_map(routing_ignore_net_orient_map);
    }
  }
}

void EarlyRouter::analyzeDemandUnit(ERModel& er_model)
{
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  double boundary_wire_unit = er_model.get_er_com_param().get_boundary_wire_unit();
  double internal_wire_unit = er_model.get_er_com_param().get_internal_wire_unit();
  double internal_via_unit = er_model.get_er_com_param().get_internal_via_unit();

  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      GCell& gcell = gcell_map[x][y];
      gcell.set_boundary_wire_unit(boundary_wire_unit);
      gcell.set_internal_wire_unit(internal_wire_unit);
      gcell.set_internal_via_unit(internal_via_unit);
    }
  }
}

void EarlyRouter::initERTaskList(ERModel& er_model)
{
  std::vector<ERNet>& er_net_list = er_model.get_er_net_list();
  std::vector<ERNet*>& er_task_list = er_model.get_er_task_list();
  er_task_list.reserve(er_net_list.size());
  for (ERNet& er_net : er_net_list) {
    er_task_list.push_back(&er_net);
  }
  std::sort(er_task_list.begin(), er_task_list.end(), CmpERNet());
}

void EarlyRouter::buildPlanarNodeMap(ERModel& er_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  GridMap<ERNode>& planar_node_map = er_model.get_planar_node_map();
  planar_node_map.init(gcell_map.get_x_size(), gcell_map.get_y_size());
#pragma omp parallel for collapse(2)
  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      ERNode& er_node = planar_node_map[x][y];
      er_node.set_coord(x, y);
      er_node.set_boundary_wire_unit(gcell_map[x][y].get_boundary_wire_unit());
      er_node.set_internal_wire_unit(gcell_map[x][y].get_internal_wire_unit());
      er_node.set_internal_via_unit(gcell_map[x][y].get_internal_via_unit());
      for (auto& [routing_layer_idx, ignore_net_orient_map] : gcell_map[x][y].get_routing_ignore_net_orient_map()) {
        for (auto& [net_idx, orient_set] : ignore_net_orient_map) {
          er_node.get_ignore_net_orient_map()[net_idx].insert(orient_set.begin(), orient_set.end());
        }
      }
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void EarlyRouter::buildPlanarNodeNeighbor(ERModel& er_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();

  GridMap<ERNode>& planar_node_map = er_model.get_planar_node_map();
#pragma omp parallel for collapse(2)
  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      std::map<Orientation, ERNode*>& neighbor_node_map = planar_node_map[x][y].get_neighbor_node_map();
      if (x != 0) {
        neighbor_node_map[Orientation::kWest] = &planar_node_map[x - 1][y];
      }
      if (x != (planar_node_map.get_x_size() - 1)) {
        neighbor_node_map[Orientation::kEast] = &planar_node_map[x + 1][y];
      }
      if (y != 0) {
        neighbor_node_map[Orientation::kSouth] = &planar_node_map[x][y - 1];
      }
      if (y != (planar_node_map.get_y_size() - 1)) {
        neighbor_node_map[Orientation::kNorth] = &planar_node_map[x][y + 1];
      }
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void EarlyRouter::buildPlanarOrientSupply(ERModel& er_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  GridMap<ERNode>& planar_node_map = er_model.get_planar_node_map();

#pragma omp parallel for collapse(2)
  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      std::map<Orientation, int32_t> planar_orient_supply_map;
      for (auto& [layer_idx, orient_supply_map] : gcell_map[x][y].get_routing_orient_supply_map()) {
        for (auto& [orient, supply] : orient_supply_map) {
          planar_orient_supply_map[orient] += supply;
        }
      }
      planar_node_map[x][y].set_orient_supply_map(planar_orient_supply_map);
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void EarlyRouter::generateTopology(ERModel& er_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<ERNet*>& er_task_list = er_model.get_er_task_list();

  int32_t batch_size = RTUTIL.getBatchSize(er_task_list.size());

  Monitor stage_monitor;
  for (size_t i = 0; i < er_task_list.size(); i++) {
    generateERTask(er_model, er_task_list[i]);
    if ((i + 1) % batch_size == 0 || (i + 1) == er_task_list.size()) {
      RTLOG.info(Loc::current(), "Routed ", (i + 1), "/", er_task_list.size(), "(", RTUTIL.getPercentage(i + 1, er_task_list.size()), ") nets",
                 stage_monitor.getStatsInfo());
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void EarlyRouter::generateERTask(ERModel& er_model, ERNet* er_task)
{
  initSinglePlanarTask(er_model, er_task);
  std::vector<Segment<PlanarCoord>> routing_segment_list = getPlanarRoutingSegmentList(er_model);
  MTree<PlanarCoord> coord_tree = getCoordTree(er_model, routing_segment_list);
  updateDemandToGraph(er_model, ChangeType::kAdd, coord_tree);
  er_task->set_planar_tree(coord_tree);
  resetSinglePlanarTask(er_model);
}

void EarlyRouter::initSinglePlanarTask(ERModel& er_model, ERNet* er_task)
{
  er_model.set_curr_er_task(er_task);
}

std::vector<Segment<PlanarCoord>> EarlyRouter::getPlanarRoutingSegmentList(ERModel& er_model)
{
  std::vector<Segment<PlanarCoord>> planar_topo_list = getPlanarTopoList(er_model);

  std::vector<ERCandidate> er_candidate_list;
  for (size_t i = 0; i < planar_topo_list.size(); i++) {
    for (std::vector<Segment<PlanarCoord>> routing_segment_list : getRoutingSegmentListList(er_model, planar_topo_list[i])) {
      er_candidate_list.emplace_back(i, routing_segment_list, 0, false, 0);
    }
  }
#pragma omp parallel for
  for (ERCandidate& er_candidate : er_candidate_list) {
    updateERCandidate(er_model, er_candidate);
  }
  std::map<int32_t, ERCandidate*> topo_candidate_map;
  for (ERCandidate& er_candidate : er_candidate_list) {
    int32_t topo_idx = er_candidate.get_topo_idx();
    if (!RTUTIL.exist(topo_candidate_map, topo_idx)) {
      topo_candidate_map[topo_idx] = &er_candidate;
      continue;
    }
    ERCandidate* current_best = topo_candidate_map[topo_idx];
    if (!er_candidate.get_is_blocked() && current_best->get_is_blocked()) {
      topo_candidate_map[topo_idx] = &er_candidate;
    } else if (!er_candidate.get_is_blocked() && !current_best->get_is_blocked()) {
      if (er_candidate.get_total_length() < current_best->get_total_length()) {
        topo_candidate_map[topo_idx] = &er_candidate;
      }
    } else if (er_candidate.get_is_blocked() && current_best->get_is_blocked()) {
      if (er_candidate.get_total_cost() < current_best->get_total_cost()) {
        topo_candidate_map[topo_idx] = &er_candidate;
      }
    }
  }
  std::vector<Segment<PlanarCoord>> routing_segment_list;
  for (auto& [topo_idx, min_candidate] : topo_candidate_map) {
    for (Segment<PlanarCoord>& routing_segment : min_candidate->get_routing_segment_list()) {
      routing_segment_list.push_back(routing_segment);
    }
  }
  return routing_segment_list;
}

std::vector<Segment<PlanarCoord>> EarlyRouter::getPlanarTopoList(ERModel& er_model)
{
  int32_t topo_spilt_length = er_model.get_er_com_param().get_topo_spilt_length();

  std::vector<PlanarCoord> planar_coord_list;
  {
    for (ERPin& er_pin : er_model.get_curr_er_task()->get_er_pin_list()) {
      planar_coord_list.push_back(er_pin.get_access_point().get_grid_coord());
    }
    std::sort(planar_coord_list.begin(), planar_coord_list.end(), CmpPlanarCoordByXASC());
    planar_coord_list.erase(std::unique(planar_coord_list.begin(), planar_coord_list.end()), planar_coord_list.end());
  }
  std::vector<Segment<PlanarCoord>> planar_topo_list;
  for (Segment<PlanarCoord>& planar_topo : RTI.getPlanarTopoList(planar_coord_list)) {
    PlanarCoord& first_coord = planar_topo.get_first();
    PlanarCoord& second_coord = planar_topo.get_second();
    int32_t span_x = std::abs(first_coord.get_x() - second_coord.get_x());
    int32_t span_y = std::abs(first_coord.get_y() - second_coord.get_y());
    if (span_x > 1 && span_y > 1 && (span_x > topo_spilt_length || span_y > topo_spilt_length)) {
      int32_t stick_num_x;
      if (span_x % topo_spilt_length == 0) {
        stick_num_x = (span_x / topo_spilt_length - 1);
      } else {
        stick_num_x = (span_x < topo_spilt_length) ? (span_x - 1) : (span_x / topo_spilt_length);
      }
      int32_t stick_num_y;
      if (span_y % topo_spilt_length == 0) {
        stick_num_y = (span_y / topo_spilt_length - 1);
      } else {
        stick_num_y = (span_y < topo_spilt_length) ? (span_y - 1) : (span_y / topo_spilt_length);
      }
      int32_t stick_num = std::min(stick_num_x, stick_num_y);

      std::vector<PlanarCoord> coord_list;
      coord_list.push_back(first_coord);
      double delta_x = static_cast<double>(second_coord.get_x() - first_coord.get_x()) / (stick_num + 1);
      double delta_y = static_cast<double>(second_coord.get_y() - first_coord.get_y()) / (stick_num + 1);
      for (int32_t i = 1; i <= stick_num; i++) {
        coord_list.emplace_back(std::round(first_coord.get_x() + i * delta_x), std::round(first_coord.get_y() + i * delta_y));
      }
      coord_list.push_back(second_coord);
      for (size_t i = 1; i < coord_list.size(); i++) {
        planar_topo_list.emplace_back(coord_list[i - 1], coord_list[i]);
      }
    } else {
      planar_topo_list.emplace_back(first_coord, second_coord);
    }
  }
  return planar_topo_list;
}

std::vector<std::vector<Segment<PlanarCoord>>> EarlyRouter::getRoutingSegmentListList(ERModel& er_model, Segment<PlanarCoord>& planar_topo)
{
  std::vector<std::vector<Segment<PlanarCoord>>> routing_segment_list_list;
  for (auto getRoutingSegmentList : {std::bind(&EarlyRouter::getRoutingSegmentListByStraight, this, std::placeholders::_1, std::placeholders::_2),
                                     std::bind(&EarlyRouter::getRoutingSegmentListByLPattern, this, std::placeholders::_1, std::placeholders::_2),
                                     std::bind(&EarlyRouter::getRoutingSegmentListByZPattern, this, std::placeholders::_1, std::placeholders::_2),
                                     std::bind(&EarlyRouter::getRoutingSegmentListByUPattern, this, std::placeholders::_1, std::placeholders::_2),
                                     std::bind(&EarlyRouter::getRoutingSegmentListByInner3Bends, this, std::placeholders::_1, std::placeholders::_2),
                                     std::bind(&EarlyRouter::getRoutingSegmentListByOuter3Bends, this, std::placeholders::_1, std::placeholders::_2)}) {
    for (std::vector<Segment<PlanarCoord>> routing_segment_list : getRoutingSegmentList(er_model, planar_topo)) {
      routing_segment_list_list.push_back(routing_segment_list);
    }
  }
  return routing_segment_list_list;
}

std::vector<std::vector<Segment<PlanarCoord>>> EarlyRouter::getRoutingSegmentListByStraight(ERModel& er_model, Segment<PlanarCoord>& planar_topo)
{
  PlanarCoord& first_coord = planar_topo.get_first();
  PlanarCoord& second_coord = planar_topo.get_second();
  if (RTUTIL.isOblique(first_coord, second_coord)) {
    return {};
  }
  std::vector<std::vector<Segment<PlanarCoord>>> routing_segment_list_list;
  {
    std::vector<Segment<PlanarCoord>> routing_segment_list;
    routing_segment_list.emplace_back(first_coord, second_coord);
    routing_segment_list_list.push_back(routing_segment_list);
  }
  return routing_segment_list_list;
}

std::vector<std::vector<Segment<PlanarCoord>>> EarlyRouter::getRoutingSegmentListByLPattern(ERModel& er_model, Segment<PlanarCoord>& planar_topo)
{
  PlanarCoord& first_coord = planar_topo.get_first();
  PlanarCoord& second_coord = planar_topo.get_second();
  if (RTUTIL.isRightAngled(first_coord, second_coord)) {
    return {};
  }
  std::vector<std::vector<PlanarCoord>> inflection_list_list;
  PlanarCoord inflection_coord1(first_coord.get_x(), second_coord.get_y());
  inflection_list_list.push_back({inflection_coord1});
  PlanarCoord inflection_coord2(second_coord.get_x(), first_coord.get_y());
  inflection_list_list.push_back({inflection_coord2});

  std::vector<std::vector<Segment<PlanarCoord>>> routing_segment_list_list;
  for (std::vector<PlanarCoord>& inflection_list : inflection_list_list) {
    std::vector<Segment<PlanarCoord>> routing_segment_list;
    routing_segment_list.emplace_back(planar_topo.get_first(), inflection_list.front());
    for (size_t i = 1; i < inflection_list.size(); i++) {
      routing_segment_list.emplace_back(inflection_list[i - 1], inflection_list[i]);
    }
    routing_segment_list.emplace_back(inflection_list.back(), planar_topo.get_second());
    routing_segment_list_list.push_back(routing_segment_list);
  }
  return routing_segment_list_list;
}

std::vector<std::vector<Segment<PlanarCoord>>> EarlyRouter::getRoutingSegmentListByZPattern(ERModel& er_model, Segment<PlanarCoord>& planar_topo)
{
  PlanarCoord& first_coord = planar_topo.get_first();
  PlanarCoord& second_coord = planar_topo.get_second();
  if (RTUTIL.isRightAngled(first_coord, second_coord)) {
    return {};
  }
  std::vector<int32_t> x_mid_index_list = getMidIndexList(first_coord.get_x(), second_coord.get_x());
  std::vector<int32_t> y_mid_index_list = getMidIndexList(first_coord.get_y(), second_coord.get_y());
  if (x_mid_index_list.empty() && y_mid_index_list.empty()) {
    return {};
  }
  std::vector<std::vector<PlanarCoord>> inflection_list_list;
  for (size_t i = 0; i < x_mid_index_list.size(); i++) {
    PlanarCoord inflection_coord1(x_mid_index_list[i], first_coord.get_y());
    PlanarCoord inflection_coord2(x_mid_index_list[i], second_coord.get_y());
    inflection_list_list.push_back({inflection_coord1, inflection_coord2});
  }
  for (size_t i = 0; i < y_mid_index_list.size(); i++) {
    PlanarCoord inflection_coord1(first_coord.get_x(), y_mid_index_list[i]);
    PlanarCoord inflection_coord2(second_coord.get_x(), y_mid_index_list[i]);
    inflection_list_list.push_back({inflection_coord1, inflection_coord2});
  }
  std::vector<std::vector<Segment<PlanarCoord>>> routing_segment_list_list;
  for (std::vector<PlanarCoord>& inflection_list : inflection_list_list) {
    std::vector<Segment<PlanarCoord>> routing_segment_list;
    routing_segment_list.emplace_back(planar_topo.get_first(), inflection_list.front());
    for (size_t i = 1; i < inflection_list.size(); i++) {
      routing_segment_list.emplace_back(inflection_list[i - 1], inflection_list[i]);
    }
    routing_segment_list.emplace_back(inflection_list.back(), planar_topo.get_second());
    routing_segment_list_list.push_back(routing_segment_list);
  }
  return routing_segment_list_list;
}

std::vector<int32_t> EarlyRouter::getMidIndexList(int32_t first_idx, int32_t second_idx)
{
  std::vector<int32_t> mid_index_list;
  RTUTIL.swapByASC(first_idx, second_idx);
  mid_index_list.reserve(second_idx - first_idx - 1);
  for (int32_t i = (first_idx + 1); i <= (second_idx - 1); i++) {
    mid_index_list.push_back(i);
  }
  return mid_index_list;
}

std::vector<std::vector<Segment<PlanarCoord>>> EarlyRouter::getRoutingSegmentListByUPattern(ERModel& er_model, Segment<PlanarCoord>& planar_topo)
{
  Die& die = RTDM.getDatabase().get_die();
  int32_t expand_step_num = er_model.get_er_com_param().get_expand_step_num();
  int32_t expand_step_length = er_model.get_er_com_param().get_expand_step_length();

  PlanarCoord& first_coord = planar_topo.get_first();
  PlanarCoord& second_coord = planar_topo.get_second();
  if (RTUTIL.getManhattanDistance(first_coord, second_coord) <= 1) {
    return {};
  }
  int32_t first_x = first_coord.get_x();
  int32_t second_x = second_coord.get_x();
  int32_t first_y = first_coord.get_y();
  int32_t second_y = second_coord.get_y();
  RTUTIL.swapByASC(first_x, second_x);
  RTUTIL.swapByASC(first_y, second_y);

  std::vector<std::vector<PlanarCoord>> inflection_list_list;
  if (!RTUTIL.isHorizontal(first_coord, second_coord)) {
    for (int32_t i = 0; i < expand_step_num; i++) {
      first_x -= expand_step_length;
      if (first_x >= die.get_grid_ll_x()) {
        PlanarCoord inflection_coord1(first_x, first_coord.get_y());
        PlanarCoord inflection_coord2(first_x, second_coord.get_y());
        inflection_list_list.push_back({inflection_coord1, inflection_coord2});
      }
      second_x += expand_step_length;
      if (second_x <= die.get_grid_ur_x()) {
        PlanarCoord inflection_coord1(second_x, first_coord.get_y());
        PlanarCoord inflection_coord2(second_x, second_coord.get_y());
        inflection_list_list.push_back({inflection_coord1, inflection_coord2});
      }
    }
  }
  if (!RTUTIL.isVertical(first_coord, second_coord)) {
    for (int32_t i = 0; i < expand_step_num; i++) {
      first_y -= expand_step_length;
      if (first_y >= die.get_grid_ll_y()) {
        PlanarCoord inflection_coord1(first_coord.get_x(), first_y);
        PlanarCoord inflection_coord2(second_coord.get_x(), first_y);
        inflection_list_list.push_back({inflection_coord1, inflection_coord2});
      }
      second_y += expand_step_length;
      if (second_y <= die.get_grid_ur_y()) {
        PlanarCoord inflection_coord1(first_coord.get_x(), second_y);
        PlanarCoord inflection_coord2(second_coord.get_x(), second_y);
        inflection_list_list.push_back({inflection_coord1, inflection_coord2});
      }
    }
  }
  std::vector<std::vector<Segment<PlanarCoord>>> routing_segment_list_list;
  for (std::vector<PlanarCoord>& inflection_list : inflection_list_list) {
    std::vector<Segment<PlanarCoord>> routing_segment_list;
    routing_segment_list.emplace_back(planar_topo.get_first(), inflection_list.front());
    for (size_t i = 1; i < inflection_list.size(); i++) {
      routing_segment_list.emplace_back(inflection_list[i - 1], inflection_list[i]);
    }
    routing_segment_list.emplace_back(inflection_list.back(), planar_topo.get_second());
    routing_segment_list_list.push_back(routing_segment_list);
  }
  return routing_segment_list_list;
}

std::vector<std::vector<Segment<PlanarCoord>>> EarlyRouter::getRoutingSegmentListByInner3Bends(ERModel& er_model, Segment<PlanarCoord>& planar_topo)
{
  PlanarCoord& first_coord = planar_topo.get_first();
  PlanarCoord& second_coord = planar_topo.get_second();
  if (RTUTIL.isRightAngled(first_coord, second_coord)) {
    return {};
  }
  std::vector<int32_t> x_mid_index_list = getMidIndexList(first_coord.get_x(), second_coord.get_x());
  std::vector<int32_t> y_mid_index_list = getMidIndexList(first_coord.get_y(), second_coord.get_y());
  if (x_mid_index_list.empty() || y_mid_index_list.empty()) {
    return {};
  }
  std::vector<std::vector<PlanarCoord>> inflection_list_list;
  for (size_t i = 0; i < x_mid_index_list.size(); i++) {
    for (size_t j = 0; j < y_mid_index_list.size(); j++) {
      PlanarCoord inflection_coord1(x_mid_index_list[i], first_coord.get_y());
      PlanarCoord inflection_coord2(x_mid_index_list[i], y_mid_index_list[j]);
      PlanarCoord inflection_coord3(second_coord.get_x(), y_mid_index_list[j]);
      inflection_list_list.push_back({inflection_coord1, inflection_coord2, inflection_coord3});
    }
  }

  for (size_t i = 0; i < x_mid_index_list.size(); i++) {
    for (size_t j = 0; j < y_mid_index_list.size(); j++) {
      PlanarCoord inflection_coord1(first_coord.get_x(), y_mid_index_list[j]);
      PlanarCoord inflection_coord2(x_mid_index_list[i], y_mid_index_list[j]);
      PlanarCoord inflection_coord3(x_mid_index_list[i], second_coord.get_y());
      inflection_list_list.push_back({inflection_coord1, inflection_coord2, inflection_coord3});
    }
  }
  std::vector<std::vector<Segment<PlanarCoord>>> routing_segment_list_list;
  for (std::vector<PlanarCoord>& inflection_list : inflection_list_list) {
    std::vector<Segment<PlanarCoord>> routing_segment_list;
    routing_segment_list.emplace_back(planar_topo.get_first(), inflection_list.front());
    for (size_t i = 1; i < inflection_list.size(); i++) {
      routing_segment_list.emplace_back(inflection_list[i - 1], inflection_list[i]);
    }
    routing_segment_list.emplace_back(inflection_list.back(), planar_topo.get_second());
    routing_segment_list_list.push_back(routing_segment_list);
  }
  return routing_segment_list_list;
}

std::vector<std::vector<Segment<PlanarCoord>>> EarlyRouter::getRoutingSegmentListByOuter3Bends(ERModel& er_model, Segment<PlanarCoord>& planar_topo)
{
  Die& die = RTDM.getDatabase().get_die();
  int32_t expand_step_num = er_model.get_er_com_param().get_expand_step_num();
  int32_t expand_step_length = er_model.get_er_com_param().get_expand_step_length();

  PlanarCoord& first_coord = planar_topo.get_first();
  PlanarCoord& second_coord = planar_topo.get_second();
  if (RTUTIL.isRightAngled(first_coord, second_coord)) {
    return {};
  }
  int32_t start_x = first_coord.get_x();
  int32_t end_x = second_coord.get_x();
  int32_t start_y = first_coord.get_y();
  int32_t end_y = second_coord.get_y();

  int32_t box_lb_x = std::min(start_x, end_x);
  int32_t box_rt_x = std::max(start_x, end_x);
  int32_t box_lb_y = std::min(start_y, end_y);
  int32_t box_rt_y = std::max(start_y, end_y);

  std::vector<std::vector<PlanarCoord>> inflection_list_list;
  for (int32_t i = 0; i < expand_step_num; i++) {
    box_lb_x -= expand_step_length;
    box_rt_x += expand_step_length;
    box_lb_y -= expand_step_length;
    box_rt_y += expand_step_length;
    if (start_x < end_x) {
      if (start_y < end_y) {
        /**
         *    line style
         *
         *            x(e)
         *          x
         *        x
         *      x(s)
         *
         */
        if (die.get_grid_ll_y() <= box_lb_y && box_rt_x <= die.get_grid_ur_x()) {
          PlanarCoord inflection_coord1(start_x, box_lb_y);
          PlanarCoord inflection_coord2(box_rt_x, box_lb_y);
          PlanarCoord inflection_coord3(box_rt_x, end_y);
          inflection_list_list.push_back({inflection_coord1, inflection_coord2, inflection_coord3});
        }
        if (die.get_grid_ll_x() <= box_lb_x && box_rt_y <= die.get_grid_ur_y()) {
          PlanarCoord inflection_coord1(box_lb_x, start_y);
          PlanarCoord inflection_coord2(box_lb_x, box_rt_y);
          PlanarCoord inflection_coord3(end_x, box_rt_y);
          inflection_list_list.push_back({inflection_coord1, inflection_coord2, inflection_coord3});
        }
      } else {
        /**
         *    line style
         *
         *   x(s)
         *     x
         *       x
         *         x(e)
         *
         */
        if (box_rt_x <= die.get_grid_ur_x() && box_rt_y <= die.get_grid_ur_y()) {
          PlanarCoord inflection_coord1(start_x, box_rt_y);
          PlanarCoord inflection_coord2(box_rt_x, box_rt_y);
          PlanarCoord inflection_coord3(box_rt_x, end_y);
          inflection_list_list.push_back({inflection_coord1, inflection_coord2, inflection_coord3});
        }
        if (die.get_grid_ll_x() <= box_lb_x && die.get_grid_ll_y() <= box_lb_y) {
          PlanarCoord inflection_coord1(box_lb_x, start_y);
          PlanarCoord inflection_coord2(box_lb_x, box_lb_y);
          PlanarCoord inflection_coord3(end_x, box_lb_y);
          inflection_list_list.push_back({inflection_coord1, inflection_coord2, inflection_coord3});
        }
      }

    } else {
      if (start_y < end_y) {
        /**
         *    line style
         *
         *   x(e)
         *     x
         *       x
         *         x(s)
         *
         */
        if (box_rt_x <= die.get_grid_ur_x() && box_rt_y <= die.get_grid_ur_y()) {
          PlanarCoord inflection_coord1(box_rt_x, start_y);
          PlanarCoord inflection_coord2(box_rt_x, box_rt_y);
          PlanarCoord inflection_coord3(end_x, box_rt_y);
          inflection_list_list.push_back({inflection_coord1, inflection_coord2, inflection_coord3});
        }
        if (die.get_grid_ll_x() <= box_lb_x && die.get_grid_ll_y() <= box_lb_y) {
          PlanarCoord inflection_coord1(start_x, box_lb_y);
          PlanarCoord inflection_coord2(box_lb_x, box_lb_y);
          PlanarCoord inflection_coord3(box_lb_x, end_y);
          inflection_list_list.push_back({inflection_coord1, inflection_coord2, inflection_coord3});
        }
      } else {
        /**
         *    line style
         *
         *            x(s)
         *          x
         *        x
         *      x(e)
         *
         */
        if (die.get_grid_ll_y() <= box_lb_y && box_rt_x <= die.get_grid_ur_x()) {
          PlanarCoord inflection_coord1(box_rt_x, start_y);
          PlanarCoord inflection_coord2(box_rt_x, box_lb_y);
          PlanarCoord inflection_coord3(end_x, box_lb_y);
          inflection_list_list.push_back({inflection_coord1, inflection_coord2, inflection_coord3});
        }
        if (die.get_grid_ll_x() <= box_lb_x && box_rt_y <= die.get_grid_ur_y()) {
          PlanarCoord inflection_coord1(start_x, box_rt_y);
          PlanarCoord inflection_coord2(box_lb_x, box_rt_y);
          PlanarCoord inflection_coord3(box_lb_x, end_y);
          inflection_list_list.push_back({inflection_coord1, inflection_coord2, inflection_coord3});
        }
      }
    }
  }
  std::vector<std::vector<Segment<PlanarCoord>>> routing_segment_list_list;
  for (std::vector<PlanarCoord>& inflection_list : inflection_list_list) {
    std::vector<Segment<PlanarCoord>> routing_segment_list;
    routing_segment_list.emplace_back(planar_topo.get_first(), inflection_list.front());
    for (size_t i = 1; i < inflection_list.size(); i++) {
      routing_segment_list.emplace_back(inflection_list[i - 1], inflection_list[i]);
    }
    routing_segment_list.emplace_back(inflection_list.back(), planar_topo.get_second());
    routing_segment_list_list.push_back(routing_segment_list);
  }
  return routing_segment_list_list;
}

void EarlyRouter::updateERCandidate(ERModel& er_model, ERCandidate& er_candidate)
{
  double overflow_unit = er_model.get_er_com_param().get_overflow_unit();
  GridMap<ERNode>& planar_node_map = er_model.get_planar_node_map();
  int32_t curr_net_idx = er_model.get_curr_er_task()->get_net_idx();

  int32_t total_length = 0;
  for (Segment<PlanarCoord>& coord_segment : er_candidate.get_routing_segment_list()) {
    total_length += RTUTIL.getManhattanDistance(coord_segment.get_first(), coord_segment.get_second());
  }
  bool is_blocked = false;
  double total_cost = 0;
  for (Segment<PlanarCoord>& coord_segment : er_candidate.get_routing_segment_list()) {
    PlanarCoord& first_coord = coord_segment.get_first();
    PlanarCoord& second_coord = coord_segment.get_second();
    if (!RTUTIL.isRightAngled(first_coord, second_coord)) {
      RTLOG.error(Loc::current(), "The direction is error!");
    }
    int32_t first_x = first_coord.get_x();
    int32_t second_x = second_coord.get_x();
    int32_t first_y = first_coord.get_y();
    int32_t second_y = second_coord.get_y();
    RTUTIL.swapByASC(first_x, second_x);
    RTUTIL.swapByASC(first_y, second_y);
    Direction direction = RTUTIL.getDirection(first_coord, second_coord);
    for (int32_t x = first_x; x <= second_x; x++) {
      for (int32_t y = first_y; y <= second_y; y++) {
        double overflow_cost = planar_node_map[x][y].getOverflowCost(curr_net_idx, direction, overflow_unit);
        if (overflow_cost > 1) {
          is_blocked = true;
        }
        total_cost += overflow_cost;
      }
    }
  }
  er_candidate.set_total_length(total_length);
  er_candidate.set_is_blocked(is_blocked);
  er_candidate.set_total_cost(total_cost);
}

MTree<PlanarCoord> EarlyRouter::getCoordTree(ERModel& er_model, std::vector<Segment<PlanarCoord>>& routing_segment_list)
{
  std::vector<PlanarCoord> candidate_root_coord_list;
  std::map<PlanarCoord, std::set<int32_t>, CmpPlanarCoordByXASC> key_coord_pin_map;
  std::vector<ERPin>& er_pin_list = er_model.get_curr_er_task()->get_er_pin_list();
  for (size_t i = 0; i < er_pin_list.size(); i++) {
    PlanarCoord coord = er_pin_list[i].get_access_point().get_grid_coord();
    candidate_root_coord_list.push_back(coord);
    key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
  }
  return RTUTIL.getTreeByFullFlow(candidate_root_coord_list, routing_segment_list, key_coord_pin_map);
}

void EarlyRouter::resetSinglePlanarTask(ERModel& er_model)
{
  er_model.set_curr_er_task(nullptr);
}

void EarlyRouter::buildLayerNodeMap(ERModel& er_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  std::vector<GridMap<ERNode>>& layer_node_map = er_model.get_layer_node_map();
  layer_node_map.resize(routing_layer_list.size());
#pragma omp parallel for
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    GridMap<ERNode>& er_node_map = layer_node_map[layer_idx];
    er_node_map.init(gcell_map.get_x_size(), gcell_map.get_y_size());
    for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
      for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
        ERNode& er_node = er_node_map[x][y];
        er_node.set_coord(x, y);
        er_node.set_layer_idx(layer_idx);
        er_node.set_boundary_wire_unit(gcell_map[x][y].get_boundary_wire_unit());
        er_node.set_internal_wire_unit(gcell_map[x][y].get_internal_wire_unit());
        er_node.set_internal_via_unit(gcell_map[x][y].get_internal_via_unit());
        if (RTUTIL.exist(gcell_map[x][y].get_routing_ignore_net_orient_map(), layer_idx)) {
          er_node.set_ignore_net_orient_map(gcell_map[x][y].get_routing_ignore_net_orient_map()[layer_idx]);
        }
      }
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void EarlyRouter::buildLayerNodeNeighbor(ERModel& er_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  int32_t bottom_routing_layer_idx = RTDM.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = RTDM.getConfig().top_routing_layer_idx;

  std::vector<GridMap<ERNode>>& layer_node_map = er_model.get_layer_node_map();

#pragma omp parallel for
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    bool routing_h = routing_layer_list[layer_idx].isPreferH();
    bool routing_v = !routing_h;
    if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
      routing_h = false;
      routing_v = false;
    }
    GridMap<ERNode>& er_node_map = layer_node_map[layer_idx];
    for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
      for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
        std::map<Orientation, ERNode*>& neighbor_node_map = er_node_map[x][y].get_neighbor_node_map();
        if (routing_h) {
          if (x != 0) {
            neighbor_node_map[Orientation::kWest] = &er_node_map[x - 1][y];
          }
          if (x != (er_node_map.get_x_size() - 1)) {
            neighbor_node_map[Orientation::kEast] = &er_node_map[x + 1][y];
          }
        }
        if (routing_v) {
          if (y != 0) {
            neighbor_node_map[Orientation::kSouth] = &er_node_map[x][y - 1];
          }
          if (y != (er_node_map.get_y_size() - 1)) {
            neighbor_node_map[Orientation::kNorth] = &er_node_map[x][y + 1];
          }
        }
        if (layer_idx != 0) {
          neighbor_node_map[Orientation::kBelow] = &layer_node_map[layer_idx - 1][x][y];
        }
        if (layer_idx != static_cast<int32_t>(layer_node_map.size()) - 1) {
          neighbor_node_map[Orientation::kAbove] = &layer_node_map[layer_idx + 1][x][y];
        }
      }
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void EarlyRouter::buildLayerOrientSupply(ERModel& er_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();

  std::vector<GridMap<ERNode>>& layer_node_map = er_model.get_layer_node_map();

#pragma omp parallel for collapse(2)
  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
        layer_node_map[layer_idx][x][y].set_orient_supply_map(gcell_map[x][y].get_routing_orient_supply_map()[layer_idx]);
      }
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void EarlyRouter::assignLayer(ERModel& er_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<ERNet*>& er_task_list = er_model.get_er_task_list();

  int32_t batch_size = RTUTIL.getBatchSize(er_task_list.size());

  Monitor stage_monitor;
  for (size_t i = 0; i < er_task_list.size(); i++) {
    assignERTask(er_model, er_task_list[i]);
    if ((i + 1) % batch_size == 0 || (i + 1) == er_task_list.size()) {
      RTLOG.info(Loc::current(), "Routed ", (i + 1), "/", er_task_list.size(), "(", RTUTIL.getPercentage(i + 1, er_task_list.size()), ") nets",
                 stage_monitor.getStatsInfo());
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void EarlyRouter::assignERTask(ERModel& er_model, ERNet* er_task)
{
  initSingleTask(er_model, er_task);
  if (needRouting(er_model)) {
    spiltPlaneTree(er_model);
    buildPillarTree(er_model);
    assignPillarTree(er_model);
    buildLayerTree(er_model, er_task);
  }
  resetSingleLayerTask(er_model);
}

void EarlyRouter::initSingleTask(ERModel& er_model, ERNet* er_task)
{
  er_model.set_curr_er_task(er_task);
}

bool EarlyRouter::needRouting(ERModel& er_model)
{
  return (er_model.get_curr_er_task()->get_planar_tree().get_root() != nullptr);
}

void EarlyRouter::spiltPlaneTree(ERModel& er_model)
{
  int32_t topo_spilt_length = er_model.get_er_com_param().get_topo_spilt_length();

  TNode<PlanarCoord>* planar_tree_root = er_model.get_curr_er_task()->get_planar_tree().get_root();
  std::queue<TNode<PlanarCoord>*> planar_queue = RTUTIL.initQueue(planar_tree_root);
  while (!planar_queue.empty()) {
    TNode<PlanarCoord>* planar_node = RTUTIL.getFrontAndPop(planar_queue);
    std::vector<TNode<PlanarCoord>*> child_list = planar_node->get_child_list();
    for (size_t i = 0; i < child_list.size(); i++) {
      int32_t length = RTUTIL.getManhattanDistance(planar_node->value(), child_list[i]->value());
      if (length <= topo_spilt_length) {
        continue;
      }
      insertMidPoint(er_model, planar_node, child_list[i]);
    }
    RTUTIL.addListToQueue(planar_queue, child_list);
  }
}

void EarlyRouter::insertMidPoint(ERModel& er_model, TNode<PlanarCoord>* planar_node, TNode<PlanarCoord>* child_node)
{
  int32_t topo_spilt_length = er_model.get_er_com_param().get_topo_spilt_length();

  PlanarCoord& parent_coord = planar_node->value();
  PlanarCoord& child_coord = child_node->value();
  if (RTUTIL.isProximal(parent_coord, child_coord)) {
    return;
  }
  std::vector<PlanarCoord> mid_coord_list;
  int32_t x1 = parent_coord.get_x();
  int32_t x2 = child_coord.get_x();
  int32_t y1 = parent_coord.get_y();
  int32_t y2 = child_coord.get_y();
  if (RTUTIL.isHorizontal(parent_coord, child_coord)) {
    RTUTIL.swapByASC(x1, x2);
    for (int32_t x = x1 + topo_spilt_length; x < x2; x += topo_spilt_length) {
      mid_coord_list.emplace_back(x, y1);
    }
    if (parent_coord.get_x() > child_coord.get_x()) {
      for (size_t i = 0, j = mid_coord_list.size() - 1; i < j; i++, j--) {
        std::swap(mid_coord_list[i], mid_coord_list[j]);
      }
    }
  } else if (RTUTIL.isVertical(parent_coord, child_coord)) {
    RTUTIL.swapByASC(y1, y2);
    for (int32_t y = y1 + topo_spilt_length; y < y2; y += topo_spilt_length) {
      mid_coord_list.emplace_back(x1, y);
    }
    if (parent_coord.get_y() > child_coord.get_y()) {
      for (size_t i = 0, j = mid_coord_list.size() - 1; i < j; i++, j--) {
        std::swap(mid_coord_list[i], mid_coord_list[j]);
      }
    }
  } else {
    RTLOG.error(Loc::current(), "The segment is oblique!");
  }
  planar_node->delChild(child_node);
  TNode<PlanarCoord>* curr_node = planar_node;
  for (size_t i = 0; i < mid_coord_list.size(); i++) {
    TNode<PlanarCoord>* mid_node = new TNode<PlanarCoord>(mid_coord_list[i]);
    curr_node->addChild(mid_node);
    curr_node = mid_node;
  }
  curr_node->addChild(child_node);
}

void EarlyRouter::buildPillarTree(ERModel& er_model)
{
  ERNet* curr_er_task = er_model.get_curr_er_task();

  std::map<PlanarCoord, std::set<int32_t>, CmpPlanarCoordByXASC> coord_pin_layer_map;
  for (ERPin& er_pin : curr_er_task->get_er_pin_list()) {
    AccessPoint& access_point = er_pin.get_access_point();
    coord_pin_layer_map[access_point.get_grid_coord()].insert(access_point.get_layer_idx());
  }
  std::function<ERPillar(PlanarCoord&, std::map<PlanarCoord, std::set<int32_t>, CmpPlanarCoordByXASC>&)> convert;
  convert = std::bind(&EarlyRouter::convertERPillar, this, std::placeholders::_1, std::placeholders::_2);
  curr_er_task->set_pillar_tree(RTUTIL.convertTree(curr_er_task->get_planar_tree(), convert, coord_pin_layer_map));
}

ERPillar EarlyRouter::convertERPillar(PlanarCoord& planar_coord, std::map<PlanarCoord, std::set<int32_t>, CmpPlanarCoordByXASC>& coord_pin_layer_map)
{
  ERPillar er_pillar;
  er_pillar.set_planar_coord(planar_coord);
  er_pillar.set_pin_layer_idx_set(coord_pin_layer_map[planar_coord]);
  return er_pillar;
}

void EarlyRouter::assignPillarTree(ERModel& er_model)
{
  assignForward(er_model);
  assignBackward(er_model);
}

void EarlyRouter::assignForward(ERModel& er_model)
{
  TNode<ERPillar>* pillar_tree_root = er_model.get_curr_er_task()->get_pillar_tree().get_root();

  ERPackage er_package(pillar_tree_root, pillar_tree_root);
  for (int32_t candidate_layer_idx : getCandidateLayerList(er_model, er_package)) {
    std::set<int32_t>& pin_layer_idx_set = pillar_tree_root->value().get_pin_layer_idx_set();
    LALayerCost layer_cost;
    layer_cost.set_parent_layer_idx(candidate_layer_idx);
    layer_cost.set_layer_idx(candidate_layer_idx);
    layer_cost.set_history_cost(getFullViaCost(er_model, pin_layer_idx_set, candidate_layer_idx));
    pillar_tree_root->value().get_layer_cost_list().push_back(std::move(layer_cost));
  }
  std::queue<TNode<ERPillar>*> pillar_node_queue = RTUTIL.initQueue(pillar_tree_root);
  while (!pillar_node_queue.empty()) {
    TNode<ERPillar>* parent_pillar_node = RTUTIL.getFrontAndPop(pillar_node_queue);
    std::vector<TNode<ERPillar>*>& child_list = parent_pillar_node->get_child_list();
    for (size_t i = 0; i < child_list.size(); i++) {
      ERPackage er_package(parent_pillar_node, child_list[i]);
      buildLayerCost(er_model, er_package);
    }
    RTUTIL.addListToQueue(pillar_node_queue, child_list);
  }
}

std::vector<int32_t> EarlyRouter::getCandidateLayerList(ERModel& er_model, ERPackage& er_package)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  int32_t bottom_routing_layer_idx = RTDM.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = RTDM.getConfig().top_routing_layer_idx;

  Direction direction = RTUTIL.getDirection(er_package.getParentPillar().get_planar_coord(), er_package.getChildPillar().get_planar_coord());

  std::vector<int32_t> candidate_layer_idx_list;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    if (routing_layer.get_layer_idx() < bottom_routing_layer_idx || top_routing_layer_idx < routing_layer.get_layer_idx()) {
      continue;
    }
    if (direction == Direction::kProximal) {
      candidate_layer_idx_list.push_back(routing_layer.get_layer_idx());
    } else if (direction == routing_layer.get_prefer_direction()) {
      candidate_layer_idx_list.push_back(routing_layer.get_layer_idx());
    }
  }
  return candidate_layer_idx_list;
}

double EarlyRouter::getFullViaCost(ERModel& er_model, std::set<int32_t>& layer_idx_set, int32_t candidate_layer_idx)
{
  double via_unit = er_model.get_er_com_param().get_via_unit();

  int32_t via_num = 0;
  if (layer_idx_set.size() > 0) {
    std::set<int32_t> layer_idx_set_temp = layer_idx_set;
    layer_idx_set_temp.insert(candidate_layer_idx);
    via_num = std::abs(*layer_idx_set_temp.begin() - *layer_idx_set_temp.rbegin());
  }
  return (via_unit * via_num);
}

void EarlyRouter::buildLayerCost(ERModel& er_model, ERPackage& er_package)
{
  std::vector<LALayerCost>& layer_cost_list = er_package.getChildPillar().get_layer_cost_list();

  for (int32_t candidate_layer_idx : getCandidateLayerList(er_model, er_package)) {
    std::pair<int32_t, double> parent_pillar_cost_pair = getParentPillarCost(er_model, er_package, candidate_layer_idx);
    double segment_cost = getSegmentCost(er_model, er_package, candidate_layer_idx);
    double child_pillar_cost = getChildPillarCost(er_model, er_package, candidate_layer_idx);

    LALayerCost layer_cost;
    layer_cost.set_parent_layer_idx(parent_pillar_cost_pair.first);
    layer_cost.set_layer_idx(candidate_layer_idx);
    layer_cost.set_history_cost(parent_pillar_cost_pair.second + segment_cost + child_pillar_cost);
    layer_cost_list.push_back(std::move(layer_cost));
  }
}

std::pair<int32_t, double> EarlyRouter::getParentPillarCost(ERModel& er_model, ERPackage& er_package, int32_t candidate_layer_idx)
{
  ERPillar& parent_pillar = er_package.getParentPillar();

  std::pair<int32_t, double> layer_cost_pair;
  double min_cost = DBL_MAX;
  for (LALayerCost& layer_cost : parent_pillar.get_layer_cost_list()) {
    std::set<int32_t> layer_idx_set_temp = parent_pillar.get_pin_layer_idx_set();
    layer_idx_set_temp.insert(layer_cost.get_layer_idx());
    double curr_cost = layer_cost.get_history_cost() + getExtraViaCost(er_model, layer_idx_set_temp, candidate_layer_idx);

    if (curr_cost < min_cost) {
      min_cost = curr_cost;
      layer_cost_pair.first = layer_cost.get_layer_idx();
      layer_cost_pair.second = curr_cost;
    } else if (curr_cost == min_cost) {
      layer_cost_pair.first = std::min(layer_cost_pair.first, layer_cost.get_layer_idx());
    }
  }
  if (min_cost == DBL_MAX) {
    RTLOG.error(Loc::current(), "The min cost is wrong!");
  }
  return layer_cost_pair;
}

double EarlyRouter::getExtraViaCost(ERModel& er_model, std::set<int32_t>& layer_idx_set, int32_t candidate_layer_idx)
{
  double via_unit = er_model.get_er_com_param().get_via_unit();

  int32_t via_num = 0;
  if (layer_idx_set.size() > 0) {
    int32_t begin_layer_idx = *layer_idx_set.begin();
    int32_t end_layer_idx = *layer_idx_set.rbegin();
    if (candidate_layer_idx < begin_layer_idx) {
      via_num = std::abs(candidate_layer_idx - begin_layer_idx);
    } else if (end_layer_idx < candidate_layer_idx) {
      via_num = std::abs(candidate_layer_idx - end_layer_idx);
    } else {
      via_num = 0;
    }
  }
  return (via_unit * via_num);
}

double EarlyRouter::getSegmentCost(ERModel& er_model, ERPackage& er_package, int32_t candidate_layer_idx)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<GridMap<ERNode>>& layer_node_map = er_model.get_layer_node_map();
  double overflow_unit = er_model.get_er_com_param().get_overflow_unit();

  Direction prefer_direction = routing_layer_list[candidate_layer_idx].get_prefer_direction();

  PlanarCoord first_coord = er_package.getParentPillar().get_planar_coord();
  PlanarCoord second_coord = er_package.getChildPillar().get_planar_coord();
  int32_t first_x = first_coord.get_x();
  int32_t first_y = first_coord.get_y();
  int32_t second_x = second_coord.get_x();
  int32_t second_y = second_coord.get_y();
  RTUTIL.swapByASC(first_x, second_x);
  RTUTIL.swapByASC(first_y, second_y);

  double node_cost = 0;
  for (int32_t x = first_x; x <= second_x; x++) {
    for (int32_t y = first_y; y <= second_y; y++) {
      node_cost += layer_node_map[candidate_layer_idx][x][y].getOverflowCost(er_model.get_curr_er_task()->get_net_idx(), prefer_direction, overflow_unit);
    }
  }
  return node_cost;
}

double EarlyRouter::getChildPillarCost(ERModel& er_model, ERPackage& er_package, int32_t candidate_layer_idx)
{
  ERPillar& child_pillar = er_package.getChildPillar();
  return getFullViaCost(er_model, child_pillar.get_pin_layer_idx_set(), candidate_layer_idx);
}

void EarlyRouter::assignBackward(ERModel& er_model)
{
  std::vector<std::vector<TNode<ERPillar>*>> level_list = RTUTIL.getLevelOrder(er_model.get_curr_er_task()->get_pillar_tree());
  if (level_list.empty()) {
    return;
  }
  for (int32_t i = static_cast<int32_t>(level_list.size() - 1); i >= 0; i--) {
    for (size_t j = 0; j < level_list[i].size(); j++) {
      int32_t best_layer_idx;
      if (level_list[i][j]->isLeafNode()) {
        best_layer_idx = getBestLayerBySelf(level_list[i][j]);
      } else {
        best_layer_idx = getBestLayerByChild(level_list[i][j]);
      }
      level_list[i][j]->value().set_layer_idx(best_layer_idx);
    }
  }
}

int32_t EarlyRouter::getBestLayerBySelf(TNode<ERPillar>* pillar_node)
{
  std::vector<LALayerCost>& layer_cost_list = pillar_node->value().get_layer_cost_list();

  double min_cost = DBL_MAX;
  int32_t best_layer_idx = layer_cost_list.front().get_layer_idx();
  for (LALayerCost& layer_cost : layer_cost_list) {
    double cost = layer_cost.get_history_cost();
    if (cost < min_cost) {
      min_cost = cost;
      best_layer_idx = layer_cost.get_layer_idx();
    } else if (cost == min_cost) {
      best_layer_idx = std::min(best_layer_idx, layer_cost.get_layer_idx());
    }
  }
  if (min_cost == DBL_MAX) {
    RTLOG.error(Loc::current(), "The min cost is wrong!");
  }
  return best_layer_idx;
}

int32_t EarlyRouter::getBestLayerByChild(TNode<ERPillar>* parent_pillar_node)
{
  std::set<int32_t> candidate_layer_idx_set;
  for (TNode<ERPillar>* child_node : parent_pillar_node->get_child_list()) {
    for (LALayerCost& layer_cost : child_node->value().get_layer_cost_list()) {
      if (layer_cost.get_layer_idx() == child_node->value().get_layer_idx()) {
        candidate_layer_idx_set.insert(layer_cost.get_parent_layer_idx());
      }
    }
  }
  double min_cost = DBL_MAX;
  int32_t best_layer_idx = INT_MAX;
  for (int32_t candidate_layer_idx : candidate_layer_idx_set) {
    for (LALayerCost& layer_cost : parent_pillar_node->value().get_layer_cost_list()) {
      if (layer_cost.get_layer_idx() != candidate_layer_idx) {
        continue;
      }
      double curr_cost = layer_cost.get_history_cost();
      if (curr_cost < min_cost) {
        min_cost = curr_cost;
        best_layer_idx = candidate_layer_idx;
      } else if (curr_cost == min_cost) {
        best_layer_idx = std::min(best_layer_idx, candidate_layer_idx);
      }
      break;
    }
  }
  if (min_cost == DBL_MAX) {
    RTLOG.error(Loc::current(), "The min cost is wrong!");
  }
  return best_layer_idx;
}

void EarlyRouter::buildLayerTree(ERModel& er_model, ERNet* er_task)
{
  std::vector<Segment<LayerCoord>> routing_segment_list = getLayerRoutingSegmentList(er_model);
  MTree<LayerCoord> coord_tree = getCoordTree(er_model, routing_segment_list);
  updateDemandToGraph(er_model, ChangeType::kAdd, coord_tree);
  er_task->set_layer_tree(coord_tree);
}

std::vector<Segment<LayerCoord>> EarlyRouter::getLayerRoutingSegmentList(ERModel& er_model)
{
  std::vector<Segment<LayerCoord>> routing_segment_list;

  std::queue<TNode<ERPillar>*> pillar_node_queue = RTUTIL.initQueue(er_model.get_curr_er_task()->get_pillar_tree().get_root());
  while (!pillar_node_queue.empty()) {
    TNode<ERPillar>* parent_pillar_node = RTUTIL.getFrontAndPop(pillar_node_queue);
    std::vector<TNode<ERPillar>*>& child_list = parent_pillar_node->get_child_list();
    {
      std::set<int32_t> layer_idx_set = parent_pillar_node->value().get_pin_layer_idx_set();
      layer_idx_set.insert(parent_pillar_node->value().get_layer_idx());
      for (TNode<ERPillar>* child_node : child_list) {
        layer_idx_set.insert(child_node->value().get_layer_idx());
      }
      routing_segment_list.emplace_back(LayerCoord(parent_pillar_node->value().get_planar_coord(), *layer_idx_set.begin()),
                                        LayerCoord(parent_pillar_node->value().get_planar_coord(), *layer_idx_set.rbegin()));
    }
    for (TNode<ERPillar>* child_node : child_list) {
      routing_segment_list.emplace_back(LayerCoord(parent_pillar_node->value().get_planar_coord(), child_node->value().get_layer_idx()),
                                        LayerCoord(child_node->value().get_planar_coord(), child_node->value().get_layer_idx()));
    }
    RTUTIL.addListToQueue(pillar_node_queue, child_list);
  }
  return routing_segment_list;
}

MTree<LayerCoord> EarlyRouter::getCoordTree(ERModel& er_model, std::vector<Segment<LayerCoord>>& routing_segment_list)
{
  std::vector<LayerCoord> candidate_root_coord_list;
  std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
  std::vector<ERPin>& er_pin_list = er_model.get_curr_er_task()->get_er_pin_list();
  for (size_t i = 0; i < er_pin_list.size(); i++) {
    LayerCoord coord = er_pin_list[i].get_access_point().getGridLayerCoord();
    candidate_root_coord_list.push_back(coord);
    key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
  }
  return RTUTIL.getTreeByFullFlow(candidate_root_coord_list, routing_segment_list, key_coord_pin_map);
}

void EarlyRouter::resetSingleLayerTask(ERModel& er_model)
{
  er_model.set_curr_er_task(nullptr);
}

void EarlyRouter::outputResult(ERModel& er_model)
{
  outputGCellCSV(er_model);
  outputLayerSupplyCSV(er_model);
  outputLayerGuide(er_model);
  outputLayerNetCSV(er_model);
  outputLayerOverflowCSV(er_model);
}

void EarlyRouter::outputGCellCSV(ERModel& er_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::string& er_temp_directory_path = RTDM.getConfig().er_temp_directory_path;

  std::ofstream* guide_file_stream = RTUTIL.getOutputFileStream(RTUTIL.getString(er_temp_directory_path, "gcell.info"));
  if (guide_file_stream == nullptr) {
    return;
  }
  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      GCell& gcell = gcell_map[x][y];
      RTUTIL.pushStream(guide_file_stream, x, ",", y, ",", gcell.get_ll_x(), ",", gcell.get_ll_y(), ",", gcell.get_ur_x(), ",", gcell.get_ur_y(), "\n");
    }
  }
  RTUTIL.closeFileStream(guide_file_stream);

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void EarlyRouter::outputLayerSupplyCSV(ERModel& er_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::string& er_temp_directory_path = RTDM.getConfig().er_temp_directory_path;

  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::ofstream* supply_csv_file
        = RTUTIL.getOutputFileStream(RTUTIL.getString(er_temp_directory_path, "supply_map_", routing_layer.get_layer_name(), ".csv"));
    for (int32_t y = gcell_map.get_y_size() - 1; y >= 0; y--) {
      for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
        int32_t total_supply = 0;
        for (auto& [orient, supply] : gcell_map[x][y].get_routing_orient_supply_map()[routing_layer.get_layer_idx()]) {
          total_supply += supply;
        }
        RTUTIL.pushStream(supply_csv_file, total_supply, ",");
      }
      RTUTIL.pushStream(supply_csv_file, "\n");
    }
    RTUTIL.closeFileStream(supply_csv_file);
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void EarlyRouter::outputLayerGuide(ERModel& er_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  int32_t micron_dbu = RTDM.getDatabase().get_micron_dbu();
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& er_temp_directory_path = RTDM.getConfig().er_temp_directory_path;

  std::ofstream* guide_file_stream = RTUTIL.getOutputFileStream(RTUTIL.getString(er_temp_directory_path, "route.guide"));
  if (guide_file_stream == nullptr) {
    return;
  }
  RTUTIL.pushStream(guide_file_stream, "guide net_name\n");
  RTUTIL.pushStream(guide_file_stream, "pin grid_x grid_y real_x real_y layer energy name\n");
  RTUTIL.pushStream(guide_file_stream, "wire grid1_x grid1_y grid2_x grid2_y real1_x real1_y real2_x real2_y layer\n");
  RTUTIL.pushStream(guide_file_stream, "via grid_x grid_y real_x real_y layer1 layer2\n");

  for (ERNet& er_net : er_model.get_er_net_list()) {
    RTUTIL.pushStream(guide_file_stream, "guide ", er_net.get_origin_net()->get_net_name(), "\n");

    for (ERPin& er_pin : er_net.get_er_pin_list()) {
      AccessPoint& access_point = er_pin.get_access_point();
      double grid_x = access_point.get_grid_x();
      double grid_y = access_point.get_grid_y();
      double real_x = access_point.get_real_x() / 1.0 / micron_dbu;
      double real_y = access_point.get_real_y() / 1.0 / micron_dbu;
      std::string layer = routing_layer_list[access_point.get_layer_idx()].get_layer_name();
      std::string connnect;
      if (er_pin.get_is_driven()) {
        connnect = "driven";
      } else {
        connnect = "load";
      }
      RTUTIL.pushStream(guide_file_stream, "pin ", grid_x, " ", grid_y, " ", real_x, " ", real_y, " ", layer, " ", connnect, " ", er_pin.get_pin_name(), "\n");
    }

    for (Segment<TNode<LayerCoord>*>& segment : RTUTIL.getSegListByTree(er_net.get_layer_tree())) {
      LayerCoord first_layer_coord = segment.get_first()->value();
      double grid1_x = first_layer_coord.get_x();
      double grid1_y = first_layer_coord.get_y();
      int32_t first_layer_idx = first_layer_coord.get_layer_idx();

      PlanarCoord first_mid_coord = RTUTIL.getRealRectByGCell(first_layer_coord, gcell_axis).getMidPoint();
      double real1_x = first_mid_coord.get_x() / 1.0 / micron_dbu;
      double real1_y = first_mid_coord.get_y() / 1.0 / micron_dbu;

      LayerCoord second_layer_coord = segment.get_second()->value();
      double grid2_x = second_layer_coord.get_x();
      double grid2_y = second_layer_coord.get_y();
      int32_t second_layer_idx = second_layer_coord.get_layer_idx();

      PlanarCoord second_mid_coord = RTUTIL.getRealRectByGCell(second_layer_coord, gcell_axis).getMidPoint();
      double real2_x = second_mid_coord.get_x() / 1.0 / micron_dbu;
      double real2_y = second_mid_coord.get_y() / 1.0 / micron_dbu;

      if (first_layer_idx != second_layer_idx) {
        RTUTIL.swapByASC(first_layer_idx, second_layer_idx);
        std::string layer1 = routing_layer_list[first_layer_idx].get_layer_name();
        std::string layer2 = routing_layer_list[second_layer_idx].get_layer_name();
        RTUTIL.pushStream(guide_file_stream, "via ", grid1_x, " ", grid1_y, " ", real1_x, " ", real1_y, " ", layer1, " ", layer2, "\n");
      } else {
        std::string layer = routing_layer_list[first_layer_idx].get_layer_name();
        RTUTIL.pushStream(guide_file_stream, "wire ", grid1_x, " ", grid1_y, " ", grid2_x, " ", grid2_y, " ", real1_x, " ", real1_y, " ", real2_x, " ", real2_y,
                          " ", layer, "\n");
      }
    }
  }
  RTUTIL.closeFileStream(guide_file_stream);
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void EarlyRouter::outputLayerNetCSV(ERModel& er_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& er_temp_directory_path = RTDM.getConfig().er_temp_directory_path;

  std::vector<GridMap<ERNode>>& layer_node_map = er_model.get_layer_node_map();
  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::ofstream* net_csv_file = RTUTIL.getOutputFileStream(RTUTIL.getString(er_temp_directory_path, "net_map_", routing_layer.get_layer_name(), ".csv"));
    GridMap<ERNode>& er_node_map = layer_node_map[routing_layer.get_layer_idx()];
    for (int32_t y = er_node_map.get_y_size() - 1; y >= 0; y--) {
      for (int32_t x = 0; x < er_node_map.get_x_size(); x++) {
        RTUTIL.pushStream(net_csv_file, er_node_map[x][y].getDemand(), ",");
      }
      RTUTIL.pushStream(net_csv_file, "\n");
    }
    RTUTIL.closeFileStream(net_csv_file);
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void EarlyRouter::outputLayerOverflowCSV(ERModel& er_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& er_temp_directory_path = RTDM.getConfig().er_temp_directory_path;

  std::vector<GridMap<ERNode>>& layer_node_map = er_model.get_layer_node_map();
  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::ofstream* overflow_csv_file
        = RTUTIL.getOutputFileStream(RTUTIL.getString(er_temp_directory_path, "overflow_map_", routing_layer.get_layer_name(), ".csv"));

    GridMap<ERNode>& er_node_map = layer_node_map[routing_layer.get_layer_idx()];
    for (int32_t y = er_node_map.get_y_size() - 1; y >= 0; y--) {
      for (int32_t x = 0; x < er_node_map.get_x_size(); x++) {
        RTUTIL.pushStream(overflow_csv_file, er_node_map[x][y].getOverflow(), ",");
      }
      RTUTIL.pushStream(overflow_csv_file, "\n");
    }
    RTUTIL.closeFileStream(overflow_csv_file);
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void EarlyRouter::cleanTempResult(ERModel& er_model)
{
  Die& die = RTDM.getDatabase().get_die();

  for (auto& [net_idx, segment_set] : RTDM.getNetDetailedResultMap(die)) {
    for (Segment<LayerCoord>* segment : segment_set) {
      RTDM.updateNetDetailedResultToGCellMap(ChangeType::kDel, net_idx, segment);
    }
  }
  for (auto& [net_idx, patch_set] : RTDM.getNetDetailedPatchMap(die)) {
    for (EXTLayerRect* patch : patch_set) {
      RTDM.updateNetDetailedPatchToGCellMap(ChangeType::kDel, net_idx, patch);
    }
  }
}

#if 1  // update env

void EarlyRouter::updateDemandToGraph(ERModel& er_model, ChangeType change_type, MTree<PlanarCoord>& coord_tree)
{
  int32_t curr_net_idx = er_model.get_curr_er_task()->get_net_idx();

  std::vector<Segment<PlanarCoord>> routing_segment_list;
  for (Segment<TNode<PlanarCoord>*>& coord_segment : RTUTIL.getSegListByTree(coord_tree)) {
    routing_segment_list.emplace_back(coord_segment.get_first()->value(), coord_segment.get_second()->value());
  }
  std::map<PlanarCoord, std::set<Orientation>, CmpPlanarCoordByXASC> usage_map;
  for (Segment<PlanarCoord>& coord_segment : routing_segment_list) {
    PlanarCoord& first_coord = coord_segment.get_first();
    PlanarCoord& second_coord = coord_segment.get_second();

    Orientation orientation = RTUTIL.getOrientation(first_coord, second_coord);
    if (orientation == Orientation::kNone || orientation == Orientation::kOblique) {
      RTLOG.error(Loc::current(), "The orientation is error!");
    }
    Orientation opposite_orientation = RTUTIL.getOppositeOrientation(orientation);

    int32_t first_x = first_coord.get_x();
    int32_t first_y = first_coord.get_y();
    int32_t second_x = second_coord.get_x();
    int32_t second_y = second_coord.get_y();
    RTUTIL.swapByASC(first_x, second_x);
    RTUTIL.swapByASC(first_y, second_y);

    for (int32_t x = first_x; x <= second_x; x++) {
      for (int32_t y = first_y; y <= second_y; y++) {
        PlanarCoord coord(x, y);
        if (coord != first_coord) {
          usage_map[coord].insert(opposite_orientation);
        }
        if (coord != second_coord) {
          usage_map[coord].insert(orientation);
        }
      }
    }
  }
  GridMap<ERNode>& planar_node_map = er_model.get_planar_node_map();
  for (auto& [usage_coord, orientation_list] : usage_map) {
    ERNode& er_node = planar_node_map[usage_coord.get_x()][usage_coord.get_y()];
    er_node.updateDemand(curr_net_idx, orientation_list, change_type);
  }
}

void EarlyRouter::updateDemandToGraph(ERModel& er_model, ChangeType change_type, MTree<LayerCoord>& coord_tree)
{
  int32_t curr_net_idx = er_model.get_curr_er_task()->get_net_idx();

  std::vector<Segment<LayerCoord>> routing_segment_list;
  for (Segment<TNode<LayerCoord>*>& coord_segment : RTUTIL.getSegListByTree(coord_tree)) {
    routing_segment_list.emplace_back(coord_segment.get_first()->value(), coord_segment.get_second()->value());
  }
  std::map<LayerCoord, std::set<Orientation>, CmpLayerCoordByXASC> usage_map;
  for (Segment<LayerCoord>& coord_segment : routing_segment_list) {
    LayerCoord& first_coord = coord_segment.get_first();
    LayerCoord& second_coord = coord_segment.get_second();

    Orientation orientation = RTUTIL.getOrientation(first_coord, second_coord);
    if (orientation == Orientation::kNone || orientation == Orientation::kOblique) {
      RTLOG.error(Loc::current(), "The orientation is error!");
    }
    Orientation opposite_orientation = RTUTIL.getOppositeOrientation(orientation);

    int32_t first_x = first_coord.get_x();
    int32_t first_y = first_coord.get_y();
    int32_t first_layer_idx = first_coord.get_layer_idx();
    int32_t second_x = second_coord.get_x();
    int32_t second_y = second_coord.get_y();
    int32_t second_layer_idx = second_coord.get_layer_idx();
    RTUTIL.swapByASC(first_x, second_x);
    RTUTIL.swapByASC(first_y, second_y);
    RTUTIL.swapByASC(first_layer_idx, second_layer_idx);

    for (int32_t x = first_x; x <= second_x; x++) {
      for (int32_t y = first_y; y <= second_y; y++) {
        for (int32_t layer_idx = first_layer_idx; layer_idx <= second_layer_idx; layer_idx++) {
          LayerCoord coord(x, y, layer_idx);
          if (coord != first_coord) {
            usage_map[coord].insert(opposite_orientation);
          }
          if (coord != second_coord) {
            usage_map[coord].insert(orientation);
          }
        }
      }
    }
  }
  std::vector<GridMap<ERNode>>& layer_node_map = er_model.get_layer_node_map();
  for (auto& [usage_coord, orientation_list] : usage_map) {
    ERNode& er_node = layer_node_map[usage_coord.get_layer_idx()][usage_coord.get_x()][usage_coord.get_y()];
    er_node.updateDemand(curr_net_idx, orientation_list, change_type);
  }
}

#endif

#if 1  // exhibit

void EarlyRouter::updateSummary(ERModel& er_model)
{
  int32_t micron_dbu = RTDM.getDatabase().get_micron_dbu();
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();
  Summary& summary = RTDM.getDatabase().get_summary();
  int32_t enable_timing = RTDM.getConfig().enable_timing;

  std::map<int32_t, double>& routing_demand_map = summary.er_summary.routing_demand_map;
  double& total_demand = summary.er_summary.total_demand;
  std::map<int32_t, double>& routing_overflow_map = summary.er_summary.routing_overflow_map;
  double& total_overflow = summary.er_summary.total_overflow;
  std::map<int32_t, double>& routing_wire_length_map = summary.er_summary.routing_wire_length_map;
  double& total_wire_length = summary.er_summary.total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.er_summary.cut_via_num_map;
  int32_t& total_via_num = summary.er_summary.total_via_num;
  std::map<std::string, std::map<std::string, double>>& clock_timing_map = summary.er_summary.clock_timing_map;
  std::map<std::string, double>& type_power_map = summary.er_summary.type_power_map;

  std::vector<GridMap<ERNode>>& layer_node_map = er_model.get_layer_node_map();
  std::vector<ERNet>& er_net_list = er_model.get_er_net_list();

  routing_demand_map.clear();
  total_demand = 0;
  routing_overflow_map.clear();
  total_overflow = 0;
  routing_wire_length_map.clear();
  total_wire_length = 0;
  cut_via_num_map.clear();
  total_via_num = 0;
  clock_timing_map.clear();
  type_power_map.clear();

  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    GridMap<ERNode>& er_node_map = layer_node_map[layer_idx];
    for (int32_t x = 0; x < er_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < er_node_map.get_y_size(); y++) {
        double node_demand = er_node_map[x][y].getDemand();
        double node_overflow = er_node_map[x][y].getOverflow();
        routing_demand_map[layer_idx] += node_demand;
        total_demand += node_demand;
        routing_overflow_map[layer_idx] += node_overflow;
        total_overflow += node_overflow;
      }
    }
  }
  for (ERNet& er_net : er_net_list) {
    for (Segment<TNode<LayerCoord>*>& segment : RTUTIL.getSegListByTree(er_net.get_layer_tree())) {
      LayerCoord& first_coord = segment.get_first()->value();
      int32_t first_layer_idx = first_coord.get_layer_idx();
      LayerCoord& second_coord = segment.get_second()->value();
      int32_t second_layer_idx = second_coord.get_layer_idx();

      if (first_layer_idx == second_layer_idx) {
        GCell& first_gcell = gcell_map[first_coord.get_x()][first_coord.get_y()];
        GCell& second_gcell = gcell_map[second_coord.get_x()][second_coord.get_y()];
        double wire_length = RTUTIL.getManhattanDistance(first_gcell.getMidPoint(), second_gcell.getMidPoint()) / 1.0 / micron_dbu;
        routing_wire_length_map[first_layer_idx] += wire_length;
        total_wire_length += wire_length;
      } else {
        RTUTIL.swapByASC(first_layer_idx, second_layer_idx);
        for (int32_t layer_idx = first_layer_idx; layer_idx < second_layer_idx; layer_idx++) {
          cut_via_num_map[layer_via_master_list[layer_idx].front().get_cut_layer_idx()]++;
          total_via_num++;
        }
      }
    }
  }
  if (enable_timing) {
    std::vector<std::map<std::string, std::vector<LayerCoord>>> real_pin_coord_map_list;
    real_pin_coord_map_list.resize(er_net_list.size());
    std::vector<std::vector<Segment<LayerCoord>>> routing_segment_list_list;
    routing_segment_list_list.resize(er_net_list.size());
    for (ERNet& er_net : er_net_list) {
      for (ERPin& er_pin : er_net.get_er_pin_list()) {
        LayerCoord layer_coord = er_pin.get_access_point().getGridLayerCoord();
        real_pin_coord_map_list[er_net.get_net_idx()][er_pin.get_pin_name()].emplace_back(RTUTIL.getRealRectByGCell(layer_coord, gcell_axis).getMidPoint(),
                                                                                          layer_coord.get_layer_idx());
      }
    }
    for (ERNet& er_net : er_net_list) {
      for (Segment<TNode<LayerCoord>*>& segment : RTUTIL.getSegListByTree(er_net.get_layer_tree())) {
        LayerCoord first_layer_coord = segment.get_first()->value();
        LayerCoord first_real_coord(RTUTIL.getRealRectByGCell(first_layer_coord, gcell_axis).getMidPoint(), first_layer_coord.get_layer_idx());
        LayerCoord second_layer_coord = segment.get_second()->value();
        LayerCoord second_real_coord(RTUTIL.getRealRectByGCell(second_layer_coord, gcell_axis).getMidPoint(), second_layer_coord.get_layer_idx());

        routing_segment_list_list[er_net.get_net_idx()].emplace_back(first_real_coord, second_real_coord);
      }
    }
    RTI.updateTimingAndPower(real_pin_coord_map_list, routing_segment_list_list, clock_timing_map, type_power_map);
  }
}

void EarlyRouter::printSummary(ERModel& er_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  Summary& summary = RTDM.getDatabase().get_summary();
  int32_t enable_timing = RTDM.getConfig().enable_timing;

  std::map<int32_t, double>& routing_demand_map = summary.er_summary.routing_demand_map;
  double& total_demand = summary.er_summary.total_demand;
  std::map<int32_t, double>& routing_overflow_map = summary.er_summary.routing_overflow_map;
  double& total_overflow = summary.er_summary.total_overflow;
  std::map<int32_t, double>& routing_wire_length_map = summary.er_summary.routing_wire_length_map;
  double& total_wire_length = summary.er_summary.total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.er_summary.cut_via_num_map;
  int32_t& total_via_num = summary.er_summary.total_via_num;
  std::map<std::string, std::map<std::string, double>>& clock_timing_map = summary.er_summary.clock_timing_map;
  std::map<std::string, double>& type_power_map = summary.er_summary.type_power_map;

  fort::char_table routing_demand_map_table;
  {
    routing_demand_map_table.set_cell_text_align(fort::text_align::right);
    routing_demand_map_table << fort::header << "routing"
                             << "demand"
                             << "prop" << fort::endr;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      routing_demand_map_table << routing_layer.get_layer_name() << routing_demand_map[routing_layer.get_layer_idx()]
                               << RTUTIL.getPercentage(routing_demand_map[routing_layer.get_layer_idx()], total_demand) << fort::endr;
    }
    routing_demand_map_table << fort::header << "Total" << total_demand << RTUTIL.getPercentage(total_demand, total_demand) << fort::endr;
  }
  fort::char_table routing_overflow_map_table;
  {
    routing_overflow_map_table.set_cell_text_align(fort::text_align::right);
    routing_overflow_map_table << fort::header << "routing"
                               << "overflow"
                               << "prop" << fort::endr;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      routing_overflow_map_table << routing_layer.get_layer_name() << routing_overflow_map[routing_layer.get_layer_idx()]
                                 << RTUTIL.getPercentage(routing_overflow_map[routing_layer.get_layer_idx()], total_overflow) << fort::endr;
    }
    routing_overflow_map_table << fort::header << "Total" << total_overflow << RTUTIL.getPercentage(total_overflow, total_overflow) << fort::endr;
  }
  fort::char_table routing_wire_length_map_table;
  {
    routing_wire_length_map_table.set_cell_text_align(fort::text_align::right);
    routing_wire_length_map_table << fort::header << "routing"
                                  << "wire_length"
                                  << "prop" << fort::endr;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      routing_wire_length_map_table << routing_layer.get_layer_name() << routing_wire_length_map[routing_layer.get_layer_idx()]
                                    << RTUTIL.getPercentage(routing_wire_length_map[routing_layer.get_layer_idx()], total_wire_length) << fort::endr;
    }
    routing_wire_length_map_table << fort::header << "Total" << total_wire_length << RTUTIL.getPercentage(total_wire_length, total_wire_length) << fort::endr;
  }
  fort::char_table cut_via_num_map_table;
  {
    cut_via_num_map_table.set_cell_text_align(fort::text_align::right);
    cut_via_num_map_table << fort::header << "cut"
                          << "#via"
                          << "prop" << fort::endr;
    for (CutLayer& cut_layer : cut_layer_list) {
      cut_via_num_map_table << cut_layer.get_layer_name() << cut_via_num_map[cut_layer.get_layer_idx()]
                            << RTUTIL.getPercentage(cut_via_num_map[cut_layer.get_layer_idx()], total_via_num) << fort::endr;
    }
    cut_via_num_map_table << fort::header << "Total" << total_via_num << RTUTIL.getPercentage(total_via_num, total_via_num) << fort::endr;
  }
  fort::char_table timing_table;
  timing_table.set_cell_text_align(fort::text_align::right);
  fort::char_table power_table;
  power_table.set_cell_text_align(fort::text_align::right);
  if (enable_timing) {
    timing_table << fort::header << "clock_name"
                 << "tns"
                 << "wns"
                 << "freq" << fort::endr;
    for (auto& [clock_name, timing_map] : clock_timing_map) {
      timing_table << clock_name << timing_map["TNS"] << timing_map["WNS"] << timing_map["Freq(MHz)"] << fort::endr;
    }
    power_table << fort::header << "power_type";
    for (auto& [type, power] : type_power_map) {
      power_table << fort::header << type;
    }
    power_table << fort::endr;
    power_table << "power_value";
    for (auto& [type, power] : type_power_map) {
      power_table << power;
    }
    power_table << fort::endr;
  }
  RTUTIL.printTableList({routing_demand_map_table, routing_overflow_map_table, routing_wire_length_map_table, cut_via_num_map_table});
  RTUTIL.printTableList({timing_table, power_table});
}

#endif

#if 1  // debug

void EarlyRouter::debugPlotERModel(ERModel& er_model, std::string flag)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& er_temp_directory_path = RTDM.getConfig().er_temp_directory_path;

  int32_t point_size = 5;

  GPGDS gp_gds;

  // base_region
  {
    GPStruct base_region_struct("base_region");
    GPBoundary gp_boundary;
    gp_boundary.set_layer_idx(0);
    gp_boundary.set_data_type(0);
    gp_boundary.set_rect(die.get_real_rect());
    base_region_struct.push(gp_boundary);
    gp_gds.addStruct(base_region_struct);
  }

  // gcell_axis
  {
    GPStruct gcell_axis_struct("gcell_axis");
    std::vector<int32_t> gcell_x_list = RTUTIL.getScaleList(die.get_real_ll_x(), die.get_real_ur_x(), gcell_axis.get_x_grid_list());
    std::vector<int32_t> gcell_y_list = RTUTIL.getScaleList(die.get_real_ll_y(), die.get_real_ur_y(), gcell_axis.get_y_grid_list());
    for (int32_t x : gcell_x_list) {
      GPPath gp_path;
      gp_path.set_layer_idx(0);
      gp_path.set_data_type(1);
      gp_path.set_segment(x, die.get_real_ll_y(), x, die.get_real_ur_y());
      gcell_axis_struct.push(gp_path);
    }
    for (int32_t y : gcell_y_list) {
      GPPath gp_path;
      gp_path.set_layer_idx(0);
      gp_path.set_data_type(1);
      gp_path.set_segment(die.get_real_ll_x(), y, die.get_real_ur_x(), y);
      gcell_axis_struct.push(gp_path);
    }
    gp_gds.addStruct(gcell_axis_struct);
  }

  // track_axis_struct
  {
    GPStruct track_axis_struct("track_axis_struct");
    for (RoutingLayer& routing_layer : routing_layer_list) {
      std::vector<int32_t> x_list = RTUTIL.getScaleList(die.get_real_ll_x(), die.get_real_ur_x(), routing_layer.getXTrackGridList());
      std::vector<int32_t> y_list = RTUTIL.getScaleList(die.get_real_ll_y(), die.get_real_ur_y(), routing_layer.getYTrackGridList());
      for (int32_t x : x_list) {
        GPPath gp_path;
        gp_path.set_data_type(static_cast<int32_t>(GPDataType::kAxis));
        gp_path.set_segment(x, die.get_real_ll_y(), x, die.get_real_ur_y());
        gp_path.set_layer_idx(RTGP.getGDSIdxByRouting(routing_layer.get_layer_idx()));
        track_axis_struct.push(gp_path);
      }
      for (int32_t y : y_list) {
        GPPath gp_path;
        gp_path.set_data_type(static_cast<int32_t>(GPDataType::kAxis));
        gp_path.set_segment(die.get_real_ll_x(), y, die.get_real_ur_x(), y);
        gp_path.set_layer_idx(RTGP.getGDSIdxByRouting(routing_layer.get_layer_idx()));
        track_axis_struct.push(gp_path);
      }
    }
    gp_gds.addStruct(track_axis_struct);
  }

  // fixed_rect
  for (auto& [is_routing, layer_net_rect_map] : RTDM.getTypeLayerNetFixedRectMap(die)) {
    for (auto& [layer_idx, net_rect_map] : layer_net_rect_map) {
      for (auto& [net_idx, rect_set] : net_rect_map) {
        GPStruct fixed_rect_struct(RTUTIL.getString("fixed_rect(net_", net_idx, ")"));
        for (EXTLayerRect* rect : rect_set) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShape));
          gp_boundary.set_rect(rect->get_real_rect());
          if (is_routing) {
            gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(layer_idx));
          } else {
            gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(layer_idx));
          }
          fixed_rect_struct.push(gp_boundary);
        }
        gp_gds.addStruct(fixed_rect_struct);
      }
    }
  }

  // access_point
  for (auto& [net_idx, access_point_set] : RTDM.getNetAccessPointMap(die)) {
    GPStruct access_point_struct(RTUTIL.getString("access_point(net_", net_idx, ")"));
    for (AccessPoint* access_point : access_point_set) {
      int32_t x = access_point->get_real_x();
      int32_t y = access_point->get_real_y();

      GPBoundary access_point_boundary;
      access_point_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(access_point->get_layer_idx()));
      access_point_boundary.set_data_type(static_cast<int32_t>(GPDataType::kAccessPoint));
      access_point_boundary.set_rect(x - point_size, y - point_size, x + point_size, y + point_size);
      access_point_struct.push(access_point_boundary);
    }
    gp_gds.addStruct(access_point_struct);
  }

  // routing result
  for (ERNet& er_net : er_model.get_er_net_list()) {
    GPStruct global_result_struct(RTUTIL.getString("global_result(net_", er_net.get_net_idx(), ")"));
    for (Segment<TNode<LayerCoord>*>& segment : RTUTIL.getSegListByTree(er_net.get_layer_tree())) {
      for (NetShape& net_shape : RTDM.getNetGlobalShapeList(er_net.get_net_idx(), segment.get_first()->value(), segment.get_second()->value())) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kGlobalPath));
        gp_boundary.set_rect(net_shape.get_rect());
        if (net_shape.get_is_routing()) {
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(net_shape.get_layer_idx()));
        } else {
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(net_shape.get_layer_idx()));
        }
        global_result_struct.push(gp_boundary);
      }
    }
    gp_gds.addStruct(global_result_struct);
  }

  // routing result
  for (auto& [net_idx, segment_set] : RTDM.getNetDetailedResultMap(die)) {
    GPStruct detailed_result_struct(RTUTIL.getString("detailed_result(net_", net_idx, ")"));
    for (Segment<LayerCoord>* segment : segment_set) {
      for (NetShape& net_shape : RTDM.getNetDetailedShapeList(net_idx, *segment)) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShape));
        gp_boundary.set_rect(net_shape.get_rect());
        if (net_shape.get_is_routing()) {
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(net_shape.get_layer_idx()));
        } else {
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(net_shape.get_layer_idx()));
        }
        detailed_result_struct.push(gp_boundary);
      }
    }
    gp_gds.addStruct(detailed_result_struct);
  }

  // routing patch
  for (auto& [net_idx, patch_set] : RTDM.getNetDetailedPatchMap(die)) {
    GPStruct detailed_patch_struct(RTUTIL.getString("detailed_patch(net_", net_idx, ")"));
    for (EXTLayerRect* patch : patch_set) {
      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPatch));
      gp_boundary.set_rect(patch->get_real_rect());
      gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(patch->get_layer_idx()));
      detailed_patch_struct.push(gp_boundary);
    }
    gp_gds.addStruct(detailed_patch_struct);
  }

  // layer_node_map
  {
    std::vector<GridMap<ERNode>>& layer_node_map = er_model.get_layer_node_map();
    // er_node_map
    {
      GPStruct er_node_map_struct("er_node_map");
      for (GridMap<ERNode>& er_node_map : layer_node_map) {
        for (int32_t grid_x = 0; grid_x < er_node_map.get_x_size(); grid_x++) {
          for (int32_t grid_y = 0; grid_y < er_node_map.get_y_size(); grid_y++) {
            ERNode& er_node = er_node_map[grid_x][grid_y];
            PlanarRect real_rect = RTUTIL.getRealRectByGCell(er_node.get_planar_coord(), gcell_axis);
            int32_t y_reduced_span = std::max(1, real_rect.getYSpan() / 12);
            int32_t y = real_rect.get_ur_y();

            y -= y_reduced_span;
            GPText gp_text_node_real_coord;
            gp_text_node_real_coord.set_coord(real_rect.get_ll_x(), y);
            gp_text_node_real_coord.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            gp_text_node_real_coord.set_message(RTUTIL.getString("(", er_node.get_x(), " , ", er_node.get_y(), " , ", er_node.get_layer_idx(), ")"));
            gp_text_node_real_coord.set_layer_idx(RTGP.getGDSIdxByRouting(er_node.get_layer_idx()));
            gp_text_node_real_coord.set_presentation(GPTextPresentation::kLeftMiddle);
            er_node_map_struct.push(gp_text_node_real_coord);

            y -= y_reduced_span;
            GPText gp_text_node_grid_coord;
            gp_text_node_grid_coord.set_coord(real_rect.get_ll_x(), y);
            gp_text_node_grid_coord.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            gp_text_node_grid_coord.set_message(RTUTIL.getString("(", grid_x, " , ", grid_y, " , ", er_node.get_layer_idx(), ")"));
            gp_text_node_grid_coord.set_layer_idx(RTGP.getGDSIdxByRouting(er_node.get_layer_idx()));
            gp_text_node_grid_coord.set_presentation(GPTextPresentation::kLeftMiddle);
            er_node_map_struct.push(gp_text_node_grid_coord);

            y -= y_reduced_span;
            GPText gp_text_orient_supply_map;
            gp_text_orient_supply_map.set_coord(real_rect.get_ll_x(), y);
            gp_text_orient_supply_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            gp_text_orient_supply_map.set_message("orient_supply_map: ");
            gp_text_orient_supply_map.set_layer_idx(RTGP.getGDSIdxByRouting(er_node.get_layer_idx()));
            gp_text_orient_supply_map.set_presentation(GPTextPresentation::kLeftMiddle);
            er_node_map_struct.push(gp_text_orient_supply_map);

            if (!er_node.get_orient_supply_map().empty()) {
              y -= y_reduced_span;
              GPText gp_text_orient_supply_map_info;
              gp_text_orient_supply_map_info.set_coord(real_rect.get_ll_x(), y);
              gp_text_orient_supply_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
              std::string orient_supply_map_info_message = "--";
              for (auto& [orient, supply] : er_node.get_orient_supply_map()) {
                orient_supply_map_info_message += RTUTIL.getString("(", GetOrientationName()(orient), ",", supply, ")");
              }
              gp_text_orient_supply_map_info.set_message(orient_supply_map_info_message);
              gp_text_orient_supply_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(er_node.get_layer_idx()));
              gp_text_orient_supply_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
              er_node_map_struct.push(gp_text_orient_supply_map_info);
            }

            y -= y_reduced_span;
            GPText gp_text_ignore_net_orient_map;
            gp_text_ignore_net_orient_map.set_coord(real_rect.get_ll_x(), y);
            gp_text_ignore_net_orient_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            gp_text_ignore_net_orient_map.set_message("ignore_net_orient_map: ");
            gp_text_ignore_net_orient_map.set_layer_idx(RTGP.getGDSIdxByRouting(er_node.get_layer_idx()));
            gp_text_ignore_net_orient_map.set_presentation(GPTextPresentation::kLeftMiddle);
            er_node_map_struct.push(gp_text_ignore_net_orient_map);

            if (!er_node.get_ignore_net_orient_map().empty()) {
              y -= y_reduced_span;
              GPText gp_text_ignore_net_orient_map_info;
              gp_text_ignore_net_orient_map_info.set_coord(real_rect.get_ll_x(), y);
              gp_text_ignore_net_orient_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
              std::string ignore_net_orient_map_info_message = "--";
              for (auto& [net_idx, orient_set] : er_node.get_ignore_net_orient_map()) {
                ignore_net_orient_map_info_message += RTUTIL.getString("(", net_idx);
                for (Orientation orient : orient_set) {
                  ignore_net_orient_map_info_message += RTUTIL.getString(",", GetOrientationName()(orient));
                }
                ignore_net_orient_map_info_message += RTUTIL.getString(")");
              }
              gp_text_ignore_net_orient_map_info.set_message(ignore_net_orient_map_info_message);
              gp_text_ignore_net_orient_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(er_node.get_layer_idx()));
              gp_text_ignore_net_orient_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
              er_node_map_struct.push(gp_text_ignore_net_orient_map_info);
            }

            y -= y_reduced_span;
            GPText gp_text_orient_net_map;
            gp_text_orient_net_map.set_coord(real_rect.get_ll_x(), y);
            gp_text_orient_net_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            gp_text_orient_net_map.set_message("orient_net_map: ");
            gp_text_orient_net_map.set_layer_idx(RTGP.getGDSIdxByRouting(er_node.get_layer_idx()));
            gp_text_orient_net_map.set_presentation(GPTextPresentation::kLeftMiddle);
            er_node_map_struct.push(gp_text_orient_net_map);

            if (!er_node.get_orient_net_map().empty()) {
              y -= y_reduced_span;
              GPText gp_text_orient_net_map_info;
              gp_text_orient_net_map_info.set_coord(real_rect.get_ll_x(), y);
              gp_text_orient_net_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
              std::string orient_net_map_info_message = "--";
              for (auto& [orient, net_set] : er_node.get_orient_net_map()) {
                orient_net_map_info_message += RTUTIL.getString("(", GetOrientationName()(orient));
                for (int32_t net_idx : net_set) {
                  orient_net_map_info_message += RTUTIL.getString(",", net_idx);
                }
                orient_net_map_info_message += RTUTIL.getString(")");
              }
              gp_text_orient_net_map_info.set_message(orient_net_map_info_message);
              gp_text_orient_net_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(er_node.get_layer_idx()));
              gp_text_orient_net_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
              er_node_map_struct.push(gp_text_orient_net_map_info);
            }

            y -= y_reduced_span;
            GPText gp_text_net_orient_map;
            gp_text_net_orient_map.set_coord(real_rect.get_ll_x(), y);
            gp_text_net_orient_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            gp_text_net_orient_map.set_message("net_orient_map: ");
            gp_text_net_orient_map.set_layer_idx(RTGP.getGDSIdxByRouting(er_node.get_layer_idx()));
            gp_text_net_orient_map.set_presentation(GPTextPresentation::kLeftMiddle);
            er_node_map_struct.push(gp_text_net_orient_map);

            if (!er_node.get_net_orient_map().empty()) {
              y -= y_reduced_span;
              GPText gp_text_net_orient_map_info;
              gp_text_net_orient_map_info.set_coord(real_rect.get_ll_x(), y);
              gp_text_net_orient_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
              std::string net_orient_map_info_message = "--";
              for (auto& [net_idx, orient_set] : er_node.get_net_orient_map()) {
                net_orient_map_info_message += RTUTIL.getString("(", net_idx);
                for (Orientation orient : orient_set) {
                  net_orient_map_info_message += RTUTIL.getString(",", GetOrientationName()(orient));
                }
                net_orient_map_info_message += RTUTIL.getString(")");
              }
              gp_text_net_orient_map_info.set_message(net_orient_map_info_message);
              gp_text_net_orient_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(er_node.get_layer_idx()));
              gp_text_net_orient_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
              er_node_map_struct.push(gp_text_net_orient_map_info);
            }

            y -= y_reduced_span;
            GPText gp_text_overflow;
            gp_text_overflow.set_coord(real_rect.get_ll_x(), y);
            gp_text_overflow.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            gp_text_overflow.set_message(RTUTIL.getString("overflow: ", er_node.getOverflow()));
            gp_text_overflow.set_layer_idx(RTGP.getGDSIdxByRouting(er_node.get_layer_idx()));
            gp_text_overflow.set_presentation(GPTextPresentation::kLeftMiddle);
            er_node_map_struct.push(gp_text_overflow);
          }
        }
      }
      gp_gds.addStruct(er_node_map_struct);
    }
    // overflow
    {
      GPStruct overflow_struct("overflow");
      for (GridMap<ERNode>& er_node_map : layer_node_map) {
        for (int32_t grid_x = 0; grid_x < er_node_map.get_x_size(); grid_x++) {
          for (int32_t grid_y = 0; grid_y < er_node_map.get_y_size(); grid_y++) {
            ERNode& er_node = er_node_map[grid_x][grid_y];
            if (er_node.getOverflow() <= 0) {
              continue;
            }
            PlanarRect real_rect = RTUTIL.getRealRectByGCell(er_node.get_planar_coord(), gcell_axis);

            GPBoundary gp_boundary;
            gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kOverflow));
            gp_boundary.set_rect(real_rect);
            gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(er_node.get_layer_idx()));
            overflow_struct.push(gp_boundary);
          }
        }
      }
      gp_gds.addStruct(overflow_struct);
    }
  }

  std::string gds_file_path = RTUTIL.getString(er_temp_directory_path, flag, "_er_model.gds");
  RTGP.plot(gp_gds, gds_file_path);
}

#endif

}  // namespace irt
