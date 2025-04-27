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
  generateAccessPoint(er_model);
  initERTaskList(er_model);
  buildPlanarNodeMap(er_model);
  buildPlanarNodeNeighbor(er_model);
  buildPlanarOrientSupply(er_model);
  // debugCheckPlanarNodeMap(er_model);
  generateTopoTree(er_model);
  buildLayerNodeMap(er_model);
  buildLayerNodeNeighbor(er_model);
  buildLayerOrientSupply(er_model);
  // debugCheckLayerNodeMap(er_model);
  generateGlobalTree(er_model);
  updateSummary(er_model);
  printSummary(er_model);
  outputGuide(er_model);
  outputDemandCSV(er_model);
  outputOverflowCSV(er_model);
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
  int32_t topo_spilt_length = 10;
  double prefer_wire_unit = 1;
  double via_unit = 1;
  double overflow_unit = 2;
  /**
   * topo_spilt_length, prefer_wire_unit, via_unit, overflow_unit
   */
  // clang-format off
  ERComParam er_com_param(topo_spilt_length, prefer_wire_unit, via_unit, overflow_unit);
  // clang-format on
  RTLOG.info(Loc::current(), "topo_spilt_length: ", er_com_param.get_topo_spilt_length());
  RTLOG.info(Loc::current(), "prefer_wire_unit: ", er_com_param.get_prefer_wire_unit());
  RTLOG.info(Loc::current(), "via_unit: ", er_com_param.get_via_unit());
  RTLOG.info(Loc::current(), "overflow_unit: ", er_com_param.get_overflow_unit());
  er_model.set_er_com_param(er_com_param);
}

void EarlyRouter::generateAccessPoint(ERModel& er_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();

  std::vector<ERNet>& er_net_list = er_model.get_er_net_list();

  for (ERNet& er_net : er_net_list) {
    for (ERPin& er_pin : er_net.get_er_pin_list()) {
      std::vector<EXTLayerRect>& routing_shape_list = er_pin.get_routing_shape_list();

      LayerCoord layer_coord;
      {
        int32_t max_area = INT32_MIN;
        LayerRect max_area_shape = routing_shape_list.front().getRealLayerRect();
        for (EXTLayerRect& routing_shape : er_pin.get_routing_shape_list()) {
          PlanarRect& real_shape = routing_shape.get_real_rect();
          if (max_area < real_shape.getArea()) {
            max_area = static_cast<int32_t>(real_shape.getArea());
            max_area_shape.set_rect(real_shape);
          }
        }
        layer_coord.set_coord(max_area_shape.getMidPoint());
        layer_coord.set_layer_idx(max_area_shape.get_layer_idx());
      }
      er_pin.set_access_point(AccessPoint(er_pin.get_pin_idx(), layer_coord));
    }

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
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
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

void EarlyRouter::generateTopoTree(ERModel& er_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<ERNet*>& er_task_list = er_model.get_er_task_list();

  int32_t batch_size = RTUTIL.getBatchSize(er_task_list.size());

  Monitor stage_monitor;
  for (size_t i = 0; i < er_task_list.size(); i++) {
    routePlanarNet(er_model, er_task_list[i]);
    if ((i + 1) % batch_size == 0 || (i + 1) == er_task_list.size()) {
      RTLOG.info(Loc::current(), "Routed ", (i + 1), "/", er_task_list.size(), "(", RTUTIL.getPercentage(i + 1, er_task_list.size()), ") nets",
                 stage_monitor.getStatsInfo());
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void EarlyRouter::routePlanarNet(ERModel& er_model, ERNet* er_net)
{
  std::vector<Segment<PlanarCoord>> routing_segment_list;
  for (Segment<PlanarCoord>& planar_topo : getPlanarTopoList(er_model, er_net)) {
    for (Segment<PlanarCoord>& routing_segment : getRoutingSegmentList(er_model, planar_topo)) {
      routing_segment_list.push_back(routing_segment);
    }
  }
  MTree<LayerCoord> coord_tree = getPlanarCoordTree(er_net, routing_segment_list);
  updatePlanarDemandToGraph(er_model, ChangeType::kAdd, coord_tree);
  if (!RTUTIL.getSegListByTree(coord_tree).empty()) {
    er_net->set_topo_tree(coord_tree);
  }
}

std::vector<Segment<PlanarCoord>> EarlyRouter::getPlanarTopoList(ERModel& er_model, ERNet* er_net)
{
  int32_t topo_spilt_length = er_model.get_er_com_param().get_topo_spilt_length();

  std::vector<PlanarCoord> planar_coord_list;
  {
    for (ERPin& er_pin : er_net->get_er_pin_list()) {
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

std::vector<Segment<PlanarCoord>> EarlyRouter::getRoutingSegmentList(ERModel& er_model, Segment<PlanarCoord>& planar_topo)
{
  std::vector<Segment<PlanarCoord>> routing_segment_list;
  for (auto getRoutingSegmentList : {std::bind(&EarlyRouter::getRoutingSegmentListByStraight, this, std::placeholders::_1, std::placeholders::_2),
                                     std::bind(&EarlyRouter::getRoutingSegmentListByLPattern, this, std::placeholders::_1, std::placeholders::_2)}) {
    if (routing_segment_list.empty()) {
      routing_segment_list = getRoutingSegmentList(er_model, planar_topo);
    }
  }
  if (routing_segment_list.empty()) {
    RTLOG.error(Loc::current(), "The routing_segment_list is empty");
  }
  return routing_segment_list;
}

std::vector<Segment<PlanarCoord>> EarlyRouter::getRoutingSegmentListByStraight(ERModel& er_model, Segment<PlanarCoord>& planar_topo)
{
  PlanarCoord& first_coord = planar_topo.get_first();
  PlanarCoord& second_coord = planar_topo.get_second();
  if (RTUTIL.isOblique(first_coord, second_coord)) {
    return {};
  }
  std::vector<Segment<PlanarCoord>> routing_segment_list;
  routing_segment_list.emplace_back(first_coord, second_coord);
  return routing_segment_list;
}

std::vector<Segment<PlanarCoord>> EarlyRouter::getRoutingSegmentListByLPattern(ERModel& er_model, Segment<PlanarCoord>& planar_topo)
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
    if (inflection_list.empty()) {
      routing_segment_list.emplace_back(planar_topo.get_first(), planar_topo.get_second());
    } else {
      routing_segment_list.emplace_back(planar_topo.get_first(), inflection_list.front());
      for (size_t i = 1; i < inflection_list.size(); i++) {
        routing_segment_list.emplace_back(inflection_list[i - 1], inflection_list[i]);
      }
      routing_segment_list.emplace_back(inflection_list.back(), planar_topo.get_second());
    }
    routing_segment_list_list.push_back(routing_segment_list);
  }

  double min_cost = DBL_MAX;
  size_t min_i = 0;
  for (size_t i = 0; i < routing_segment_list_list.size(); i++) {
    double cost = getPlanarNodeCost(er_model, routing_segment_list_list[i]);
    if (cost < min_cost) {
      min_cost = cost;
      min_i = i;
    }
  }
  return routing_segment_list_list[min_i];
}

double EarlyRouter::getPlanarNodeCost(ERModel& er_model, std::vector<Segment<PlanarCoord>>& routing_segment_list)
{
  double overflow_unit = er_model.get_er_com_param().get_overflow_unit();
  GridMap<ERNode>& planar_node_map = er_model.get_planar_node_map();

  double node_cost = 0;
  for (Segment<PlanarCoord>& coord_segment : routing_segment_list) {
    PlanarCoord& first_coord = coord_segment.get_first();
    PlanarCoord& second_coord = coord_segment.get_second();

    Orientation orientation = RTUTIL.getOrientation(first_coord, second_coord);
    if (orientation == Orientation::kNone || orientation == Orientation::kOblique) {
      RTLOG.error(Loc::current(), "The orientation is error!");
    }
    Orientation opposite_orientation = RTUTIL.getOppositeOrientation(orientation);

    node_cost += planar_node_map[first_coord.get_x()][first_coord.get_y()].getOverflowCost(orientation, overflow_unit);
    node_cost += planar_node_map[second_coord.get_x()][second_coord.get_y()].getOverflowCost(opposite_orientation, overflow_unit);

    if (RTUTIL.isHorizontal(first_coord, second_coord)) {
      int32_t first_x = first_coord.get_x();
      int32_t second_x = second_coord.get_x();
      int32_t y = first_coord.get_y();
      RTUTIL.swapByASC(first_x, second_x);
      for (int32_t x = (first_x + 1); x <= (second_x - 1); x++) {
        node_cost += planar_node_map[x][y].getOverflowCost(orientation, overflow_unit);
        node_cost += planar_node_map[x][y].getOverflowCost(opposite_orientation, overflow_unit);
      }
    } else if (RTUTIL.isVertical(first_coord, second_coord)) {
      int32_t x = first_coord.get_x();
      int32_t first_y = first_coord.get_y();
      int32_t second_y = second_coord.get_y();
      RTUTIL.swapByASC(first_y, second_y);
      for (int32_t y = (first_y + 1); y <= (second_y - 1); y++) {
        node_cost += planar_node_map[x][y].getOverflowCost(orientation, overflow_unit);
        node_cost += planar_node_map[x][y].getOverflowCost(opposite_orientation, overflow_unit);
      }
    }
  }
  return node_cost;
}

MTree<LayerCoord> EarlyRouter::getPlanarCoordTree(ERNet* er_net, std::vector<Segment<PlanarCoord>>& planar_routing_segment_list)
{
  std::vector<Segment<LayerCoord>> routing_segment_list;
  for (Segment<PlanarCoord>& planar_routing_segment : planar_routing_segment_list) {
    routing_segment_list.emplace_back(LayerCoord(planar_routing_segment.get_first(), 0), LayerCoord(planar_routing_segment.get_second(), 0));
  }
  std::vector<LayerCoord> candidate_root_coord_list;
  std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
  std::vector<ERPin>& er_pin_list = er_net->get_er_pin_list();
  for (size_t i = 0; i < er_pin_list.size(); i++) {
    LayerCoord coord(er_pin_list[i].get_access_point().get_grid_coord(), 0);
    candidate_root_coord_list.push_back(coord);
    key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
  }
  return RTUTIL.getTreeByFullFlow(candidate_root_coord_list, routing_segment_list, key_coord_pin_map);
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

void EarlyRouter::generateGlobalTree(ERModel& er_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<ERNet*>& er_task_list = er_model.get_er_task_list();

  int32_t batch_size = RTUTIL.getBatchSize(er_task_list.size());

  Monitor stage_monitor;
  for (size_t i = 0; i < er_task_list.size(); i++) {
    routeLayerNet(er_model, er_task_list[i]);
    if ((i + 1) % batch_size == 0 || (i + 1) == er_task_list.size()) {
      RTLOG.info(Loc::current(), "Routed ", (i + 1), "/", er_task_list.size(), "(", RTUTIL.getPercentage(i + 1, er_task_list.size()), ") nets",
                 stage_monitor.getStatsInfo());
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void EarlyRouter::routeLayerNet(ERModel& er_model, ERNet* er_net)
{
  // 构建er_topo_list,并将通孔线段加入routing_segment_list
  std::vector<ERTopo> er_topo_list;
  std::vector<Segment<LayerCoord>> routing_segment_list;
  makeERTopoList(er_model, er_net, er_topo_list, routing_segment_list);
  for (ERTopo& er_topo : er_topo_list) {
    routeERTopo(er_model, &er_topo);
    for (Segment<LayerCoord>& routing_segment : er_topo.get_routing_segment_list()) {
      routing_segment_list.push_back(routing_segment);
    }
  }
  MTree<LayerCoord> coord_tree = getLayerCoordTree(er_net, routing_segment_list);
  updateLayerDemandToGraph(er_model, ChangeType::kAdd, coord_tree);
  uploadNetResult(er_net, coord_tree);
}

void EarlyRouter::makeERTopoList(ERModel& er_model, ERNet* er_net, std::vector<ERTopo>& er_topo_list, std::vector<Segment<LayerCoord>>& routing_segment_list)
{
  int32_t bottom_routing_layer_idx = RTDM.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = RTDM.getConfig().top_routing_layer_idx;
  int32_t topo_spilt_length = er_model.get_er_com_param().get_topo_spilt_length();

  if (er_net->get_topo_tree().get_root() == nullptr) {
    ERTopo er_topo;
    for (ERPin& er_pin : er_net->get_er_pin_list()) {
      ERGroup er_group;
      er_group.get_coord_list().push_back(er_pin.get_access_point().getGridLayerCoord());
      er_topo.get_er_group_list().push_back(er_group);
    }
    er_topo_list.push_back(er_topo);
    {
      std::set<PlanarCoord, CmpPlanarCoordByXASC> coord_set;
      for (ERTopo& er_topo : er_topo_list) {
        for (ERGroup& er_group : er_topo.get_er_group_list()) {
          for (LayerCoord& coord : er_group.get_coord_list()) {
            coord_set.insert(coord);
          }
        }
      }
      if (coord_set.size() > 1) {
        RTLOG.error(Loc::current(), "The topo_tree should not be empty!");
      }
    }
  } else {
    // planar_topo_list
    std::vector<Segment<PlanarCoord>> planar_topo_list;
    {
      for (Segment<TNode<LayerCoord>*>& coord_segment : RTUTIL.getSegListByTree(er_net->get_topo_tree())) {
        PlanarCoord& first_coord = coord_segment.get_first()->value();
        PlanarCoord& second_coord = coord_segment.get_second()->value();
        int32_t first_x = first_coord.get_x();
        int32_t first_y = first_coord.get_y();
        int32_t second_x = second_coord.get_x();
        int32_t second_y = second_coord.get_y();

        if (first_x == second_x) {
          int32_t y_diff = std::abs(second_y - first_y);
          int32_t segment_num = std::max(1, static_cast<int32_t>(std::ceil(y_diff / topo_spilt_length)));
          int32_t step = (second_y - first_y) / segment_num;
          for (int32_t i = 0; i < segment_num; ++i) {
            PlanarCoord start(first_x, first_y + i * step);
            PlanarCoord end(first_x, first_y + (i + 1) * step);
            planar_topo_list.emplace_back(start, end);
          }
          // Add the last segment to reach the exact second_coord
          if (planar_topo_list.back().get_second().get_y() != second_y) {
            planar_topo_list.emplace_back(planar_topo_list.back().get_second(), second_coord);
          }
        } else if (first_y == second_y) {
          int32_t x_diff = std::abs(second_x - first_x);
          int32_t segment_num = std::max(1, static_cast<int32_t>(std::ceil(x_diff / topo_spilt_length)));
          int32_t step = (second_x - first_x) / segment_num;
          for (int32_t i = 0; i < segment_num; ++i) {
            PlanarCoord start(first_x + i * step, first_y);
            PlanarCoord end(first_x + (i + 1) * step, first_y);
            planar_topo_list.emplace_back(start, end);
          }
          if (planar_topo_list.back().get_second().get_x() != second_x) {
            planar_topo_list.emplace_back(planar_topo_list.back().get_second(), second_coord);
          }
        } else {
          RTLOG.error(Loc::current(), "The segment is not horizontal or vertical!");
        }
      }
    }
    // planar_pin_group_map
    std::map<PlanarCoord, std::vector<ERGroup>, CmpPlanarCoordByXASC> planar_pin_group_map;
    {
      for (ERPin& er_pin : er_net->get_er_pin_list()) {
        LayerCoord grid_coord = er_pin.get_access_point().getGridLayerCoord();

        ERGroup er_group;
        er_group.get_coord_list().push_back(grid_coord);
        planar_pin_group_map[grid_coord.get_planar_coord()].push_back(er_group);
      }
    }
    // planar_steiner_group_map
    std::map<PlanarCoord, ERGroup, CmpPlanarCoordByXASC> planar_steiner_group_map;
    {
      for (Segment<PlanarCoord>& planar_topo : planar_topo_list) {
        for (PlanarCoord coord : {planar_topo.get_first(), planar_topo.get_second()}) {
          if (!RTUTIL.exist(planar_pin_group_map, coord) && !RTUTIL.exist(planar_steiner_group_map, coord)) {
            // 补充steiner point的垂直线段
            routing_segment_list.emplace_back(LayerCoord(coord, bottom_routing_layer_idx), LayerCoord(coord, top_routing_layer_idx));
            for (int32_t layer_idx = bottom_routing_layer_idx; layer_idx <= top_routing_layer_idx; layer_idx++) {
              planar_steiner_group_map[coord].get_coord_list().push_back(LayerCoord(coord, layer_idx));
            }
          }
        }
      }
    }
    // 生成topo group
    {
      for (Segment<PlanarCoord>& planar_topo : planar_topo_list) {
        ERTopo er_topo;
        for (PlanarCoord coord : {planar_topo.get_first(), planar_topo.get_second()}) {
          if (RTUTIL.exist(planar_pin_group_map, coord)) {
            for (ERGroup& er_group : planar_pin_group_map[coord]) {
              er_topo.get_er_group_list().push_back(er_group);
            }
          } else if (RTUTIL.exist(planar_steiner_group_map, coord)) {
            er_topo.get_er_group_list().push_back(planar_steiner_group_map[coord]);
          }
        }
        er_topo_list.push_back(er_topo);
      }
    }
  }
  // 构建topo的其他内容
  {
    for (ERTopo& er_topo : er_topo_list) {
      er_topo.set_net_idx(er_net->get_net_idx());
      std::vector<PlanarCoord> coord_list;
      for (ERGroup& er_group : er_topo.get_er_group_list()) {
        for (LayerCoord& coord : er_group.get_coord_list()) {
          coord_list.push_back(coord);
        }
      }
      er_topo.set_bounding_box(RTUTIL.getBoundingBox(coord_list));
    }
  }
}

void EarlyRouter::routeERTopo(ERModel& er_model, ERTopo* er_topo)
{
  initSingleTask(er_model, er_topo);
  while (!isConnectedAllEnd(er_model)) {
    routeSinglePath(er_model);
    updatePathResult(er_model);
    resetStartAndEnd(er_model);
    resetSinglePath(er_model);
  }
  updateTaskResult(er_model);
  resetSingleTask(er_model);
}

void EarlyRouter::initSingleTask(ERModel& er_model, ERTopo* er_topo)
{
  std::vector<GridMap<ERNode>>& layer_node_map = er_model.get_layer_node_map();

  // single topo
  er_model.set_curr_er_topo(er_topo);
  {
    std::vector<std::vector<ERNode*>> node_list_list;
    std::vector<ERGroup>& er_group_list = er_topo->get_er_group_list();
    for (ERGroup& er_group : er_group_list) {
      std::vector<ERNode*> node_list;
      for (LayerCoord& coord : er_group.get_coord_list()) {
        ERNode& er_node = layer_node_map[coord.get_layer_idx()][coord.get_x()][coord.get_y()];
        node_list.push_back(&er_node);
      }
      node_list_list.push_back(node_list);
    }
    for (size_t i = 0; i < node_list_list.size(); i++) {
      if (i == 0) {
        er_model.get_start_node_list_list().push_back(node_list_list[i]);
      } else {
        er_model.get_end_node_list_list().push_back(node_list_list[i]);
      }
    }
  }
  er_model.get_path_node_list().clear();
  er_model.get_single_topo_visited_node_list().clear();
  er_model.get_routing_segment_list().clear();
}

bool EarlyRouter::isConnectedAllEnd(ERModel& er_model)
{
  return er_model.get_end_node_list_list().empty();
}

void EarlyRouter::routeSinglePath(ERModel& er_model)
{
  initPathHead(er_model);
  while (!searchEnded(er_model)) {
    expandSearching(er_model);
    resetPathHead(er_model);
  }
}

void EarlyRouter::initPathHead(ERModel& er_model)
{
  std::vector<std::vector<ERNode*>>& start_node_list_list = er_model.get_start_node_list_list();
  std::vector<ERNode*>& path_node_list = er_model.get_path_node_list();

  for (std::vector<ERNode*>& start_node_list : start_node_list_list) {
    for (ERNode* start_node : start_node_list) {
      start_node->set_estimated_cost(getEstimateCostToEnd(er_model, start_node));
      pushToOpenList(er_model, start_node);
    }
  }
  for (ERNode* path_node : path_node_list) {
    path_node->set_estimated_cost(getEstimateCostToEnd(er_model, path_node));
    pushToOpenList(er_model, path_node);
  }
  resetPathHead(er_model);
}

bool EarlyRouter::searchEnded(ERModel& er_model)
{
  std::vector<std::vector<ERNode*>>& end_node_list_list = er_model.get_end_node_list_list();
  ERNode* path_head_node = er_model.get_path_head_node();

  if (path_head_node == nullptr) {
    er_model.set_end_node_list_idx(-1);
    return true;
  }
  for (size_t i = 0; i < end_node_list_list.size(); i++) {
    for (ERNode* end_node : end_node_list_list[i]) {
      if (path_head_node == end_node) {
        er_model.set_end_node_list_idx(static_cast<int32_t>(i));
        return true;
      }
    }
  }
  return false;
}

void EarlyRouter::expandSearching(ERModel& er_model)
{
  PriorityQueue<ERNode*, std::vector<ERNode*>, CmpERNodeCost>& open_queue = er_model.get_open_queue();
  ERNode* path_head_node = er_model.get_path_head_node();

  for (auto& [orientation, neighbor_node] : path_head_node->get_neighbor_node_map()) {
    if (neighbor_node == nullptr) {
      continue;
    }
    if (!RTUTIL.isInside(er_model.get_curr_er_topo()->get_bounding_box(), *neighbor_node)) {
      continue;
    }
    if (neighbor_node->isClose()) {
      continue;
    }
    double know_cost = getKnowCost(er_model, path_head_node, neighbor_node);
    if (neighbor_node->isOpen() && know_cost < neighbor_node->get_known_cost()) {
      neighbor_node->set_known_cost(know_cost);
      neighbor_node->set_parent_node(path_head_node);
      // 对优先队列中的值修改了,需要重新建堆
      std::make_heap(open_queue.begin(), open_queue.end(), CmpERNodeCost());
    } else if (neighbor_node->isNone()) {
      neighbor_node->set_known_cost(know_cost);
      neighbor_node->set_parent_node(path_head_node);
      neighbor_node->set_estimated_cost(getEstimateCostToEnd(er_model, neighbor_node));
      pushToOpenList(er_model, neighbor_node);
    }
  }
}

void EarlyRouter::resetPathHead(ERModel& er_model)
{
  er_model.set_path_head_node(popFromOpenList(er_model));
}

void EarlyRouter::updatePathResult(ERModel& er_model)
{
  for (Segment<LayerCoord>& routing_segment : getRoutingSegmentListByNode(er_model.get_path_head_node())) {
    er_model.get_routing_segment_list().push_back(routing_segment);
  }
}

std::vector<Segment<LayerCoord>> EarlyRouter::getRoutingSegmentListByNode(ERNode* node)
{
  std::vector<Segment<LayerCoord>> routing_segment_list;

  ERNode* curr_node = node;
  ERNode* pre_node = curr_node->get_parent_node();

  if (pre_node == nullptr) {
    // 起点和终点重合
    return routing_segment_list;
  }
  Orientation curr_orientation = RTUTIL.getOrientation(*curr_node, *pre_node);
  while (pre_node->get_parent_node() != nullptr) {
    Orientation pre_orientation = RTUTIL.getOrientation(*pre_node, *pre_node->get_parent_node());
    if (curr_orientation != pre_orientation) {
      routing_segment_list.emplace_back(*curr_node, *pre_node);
      curr_orientation = pre_orientation;
      curr_node = pre_node;
    }
    pre_node = pre_node->get_parent_node();
  }
  routing_segment_list.emplace_back(*curr_node, *pre_node);

  return routing_segment_list;
}

void EarlyRouter::resetStartAndEnd(ERModel& er_model)
{
  std::vector<std::vector<ERNode*>>& start_node_list_list = er_model.get_start_node_list_list();
  std::vector<std::vector<ERNode*>>& end_node_list_list = er_model.get_end_node_list_list();
  std::vector<ERNode*>& path_node_list = er_model.get_path_node_list();
  ERNode* path_head_node = er_model.get_path_head_node();
  int32_t end_node_list_idx = er_model.get_end_node_list_idx();

  // 对于抵达的终点pin,只保留到达的node
  end_node_list_list[end_node_list_idx].clear();
  end_node_list_list[end_node_list_idx].push_back(path_head_node);

  ERNode* path_node = path_head_node->get_parent_node();
  if (path_node == nullptr) {
    // 起点和终点重合
    path_node = path_head_node;
  } else {
    // 起点和终点不重合
    while (path_node->get_parent_node() != nullptr) {
      path_node_list.push_back(path_node);
      path_node = path_node->get_parent_node();
    }
  }
  if (start_node_list_list.size() == 1) {
    // 初始化时,要把start_node_list_list的pin只留一个ap点
    // 后续只要将end_node_list_list的pin保留一个ap点
    start_node_list_list.front().clear();
    start_node_list_list.front().push_back(path_node);
  }
  start_node_list_list.push_back(end_node_list_list[end_node_list_idx]);
  end_node_list_list.erase(end_node_list_list.begin() + end_node_list_idx);
}

void EarlyRouter::resetSinglePath(ERModel& er_model)
{
  PriorityQueue<ERNode*, std::vector<ERNode*>, CmpERNodeCost> empty_queue;
  er_model.set_open_queue(empty_queue);

  std::vector<ERNode*>& single_path_visited_node_list = er_model.get_single_path_visited_node_list();
  for (ERNode* visited_node : single_path_visited_node_list) {
    visited_node->set_state(ERNodeState::kNone);
    visited_node->set_parent_node(nullptr);
    visited_node->set_known_cost(0);
    visited_node->set_estimated_cost(0);
  }
  single_path_visited_node_list.clear();

  er_model.set_path_head_node(nullptr);
  er_model.set_end_node_list_idx(-1);
}

void EarlyRouter::updateTaskResult(ERModel& er_model)
{
  er_model.get_curr_er_topo()->set_routing_segment_list(getRoutingSegmentList(er_model));
}

std::vector<Segment<LayerCoord>> EarlyRouter::getRoutingSegmentList(ERModel& er_model)
{
  ERTopo* curr_er_topo = er_model.get_curr_er_topo();

  std::vector<LayerCoord> candidate_root_coord_list;
  std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
  std::vector<ERGroup>& er_group_list = curr_er_topo->get_er_group_list();
  for (size_t i = 0; i < er_group_list.size(); i++) {
    for (LayerCoord& coord : er_group_list[i].get_coord_list()) {
      candidate_root_coord_list.push_back(coord);
      key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
    }
  }
  MTree<LayerCoord> coord_tree = RTUTIL.getTreeByFullFlow(candidate_root_coord_list, er_model.get_routing_segment_list(), key_coord_pin_map);

  std::vector<Segment<LayerCoord>> routing_segment_list;
  for (Segment<TNode<LayerCoord>*>& coord_segment : RTUTIL.getSegListByTree(coord_tree)) {
    routing_segment_list.emplace_back(coord_segment.get_first()->value(), coord_segment.get_second()->value());
  }
  return routing_segment_list;
}

void EarlyRouter::resetSingleTask(ERModel& er_model)
{
  er_model.set_curr_er_topo(nullptr);
  er_model.get_start_node_list_list().clear();
  er_model.get_end_node_list_list().clear();
  er_model.get_path_node_list().clear();
  er_model.get_single_topo_visited_node_list().clear();
  er_model.get_routing_segment_list().clear();
}

// manager open list

void EarlyRouter::pushToOpenList(ERModel& er_model, ERNode* curr_node)
{
  PriorityQueue<ERNode*, std::vector<ERNode*>, CmpERNodeCost>& open_queue = er_model.get_open_queue();
  std::vector<ERNode*>& single_topo_visited_node_list = er_model.get_single_topo_visited_node_list();
  std::vector<ERNode*>& single_path_visited_node_list = er_model.get_single_path_visited_node_list();

  open_queue.push(curr_node);
  curr_node->set_state(ERNodeState::kOpen);
  single_topo_visited_node_list.push_back(curr_node);
  single_path_visited_node_list.push_back(curr_node);
}

ERNode* EarlyRouter::popFromOpenList(ERModel& er_model)
{
  PriorityQueue<ERNode*, std::vector<ERNode*>, CmpERNodeCost>& open_queue = er_model.get_open_queue();

  ERNode* node = nullptr;
  if (!open_queue.empty()) {
    node = open_queue.top();
    open_queue.pop();
    node->set_state(ERNodeState::kClose);
  }
  return node;
}

// calculate known cost

double EarlyRouter::getKnowCost(ERModel& er_model, ERNode* start_node, ERNode* end_node)
{
  bool exist_neighbor = false;
  for (auto& [orientation, neighbor_ptr] : start_node->get_neighbor_node_map()) {
    if (neighbor_ptr == end_node) {
      exist_neighbor = true;
      break;
    }
  }
  if (!exist_neighbor) {
    RTLOG.error(Loc::current(), "The neighbor not exist!");
  }

  double cost = 0;
  cost += start_node->get_known_cost();
  cost += getNodeCost(er_model, start_node, RTUTIL.getOrientation(*start_node, *end_node));
  cost += getNodeCost(er_model, end_node, RTUTIL.getOrientation(*end_node, *start_node));
  cost += getKnowWireCost(er_model, start_node, end_node);
  cost += getKnowViaCost(er_model, start_node, end_node);
  return cost;
}

double EarlyRouter::getNodeCost(ERModel& er_model, ERNode* curr_node, Orientation orientation)
{
  double overflow_unit = er_model.get_er_com_param().get_overflow_unit();

  double node_cost = 0;
  node_cost += curr_node->getOverflowCost(orientation, overflow_unit);
  return node_cost;
}

double EarlyRouter::getKnowWireCost(ERModel& er_model, ERNode* start_node, ERNode* end_node)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  double prefer_wire_unit = er_model.get_er_com_param().get_prefer_wire_unit();

  double wire_cost = 0;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    wire_cost += RTUTIL.getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());

    RoutingLayer& routing_layer = routing_layer_list[start_node->get_layer_idx()];
    if (routing_layer.get_prefer_direction() == RTUTIL.getDirection(*start_node, *end_node)) {
      wire_cost *= prefer_wire_unit;
    }
  }
  return wire_cost;
}

double EarlyRouter::getKnowViaCost(ERModel& er_model, ERNode* start_node, ERNode* end_node)
{
  double via_unit = er_model.get_er_com_param().get_via_unit();
  double via_cost = (via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
  return via_cost;
}

// calculate estimate cost

double EarlyRouter::getEstimateCostToEnd(ERModel& er_model, ERNode* curr_node)
{
  std::vector<std::vector<ERNode*>>& end_node_list_list = er_model.get_end_node_list_list();

  double estimate_cost = DBL_MAX;
  for (std::vector<ERNode*>& end_node_list : end_node_list_list) {
    for (ERNode* end_node : end_node_list) {
      if (end_node->isClose()) {
        continue;
      }
      estimate_cost = std::min(estimate_cost, getEstimateCost(er_model, curr_node, end_node));
    }
  }
  return estimate_cost;
}

double EarlyRouter::getEstimateCost(ERModel& er_model, ERNode* start_node, ERNode* end_node)
{
  double estimate_cost = 0;
  estimate_cost += getEstimateWireCost(er_model, start_node, end_node);
  estimate_cost += getEstimateViaCost(er_model, start_node, end_node);
  return estimate_cost;
}

double EarlyRouter::getEstimateWireCost(ERModel& er_model, ERNode* start_node, ERNode* end_node)
{
  double prefer_wire_unit = er_model.get_er_com_param().get_prefer_wire_unit();

  double wire_cost = 0;
  wire_cost += RTUTIL.getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());
  wire_cost *= prefer_wire_unit;
  return wire_cost;
}

double EarlyRouter::getEstimateViaCost(ERModel& er_model, ERNode* start_node, ERNode* end_node)
{
  double via_unit = er_model.get_er_com_param().get_via_unit();
  double via_cost = (via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
  return via_cost;
}

MTree<LayerCoord> EarlyRouter::getLayerCoordTree(ERNet* er_net, std::vector<Segment<LayerCoord>>& routing_segment_list)
{
  std::vector<LayerCoord> candidate_root_coord_list;
  std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
  std::vector<ERPin>& er_pin_list = er_net->get_er_pin_list();
  for (size_t i = 0; i < er_pin_list.size(); i++) {
    LayerCoord coord = er_pin_list[i].get_access_point().getGridLayerCoord();
    candidate_root_coord_list.push_back(coord);
    key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
  }
  return RTUTIL.getTreeByFullFlow(candidate_root_coord_list, routing_segment_list, key_coord_pin_map);
}

void EarlyRouter::uploadNetResult(ERNet* er_net, MTree<LayerCoord>& coord_tree)
{
  for (Segment<TNode<LayerCoord>*>& coord_segment : RTUTIL.getSegListByTree(coord_tree)) {
    Segment<LayerCoord>* segment = new Segment<LayerCoord>(coord_segment.get_first()->value(), coord_segment.get_second()->value());
    RTDM.updateNetGlobalResultToGCellMap(ChangeType::kAdd, er_net->get_net_idx(), segment);
  }
}

#if 1  // update env

void EarlyRouter::updatePlanarDemandToGraph(ERModel& er_model, ChangeType change_type, MTree<LayerCoord>& coord_tree)
{
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
  GridMap<ERNode>& planar_node_map = er_model.get_planar_node_map();
  for (auto& [usage_coord, orientation_list] : usage_map) {
    ERNode& er_node = planar_node_map[usage_coord.get_x()][usage_coord.get_y()];
    er_node.updateDemand(orientation_list, ChangeType::kAdd);
  }
}

void EarlyRouter::updateLayerDemandToGraph(ERModel& er_model, ChangeType change_type, MTree<LayerCoord>& coord_tree)
{
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
    er_node.updateDemand(orientation_list, ChangeType::kAdd);
  }
}

#endif

#if 1  // exhibit

void EarlyRouter::updateSummary(ERModel& er_model)
{
  int32_t micron_dbu = RTDM.getDatabase().get_micron_dbu();
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();
  Summary& summary = RTDM.getDatabase().get_summary();
  int32_t enable_timing = RTDM.getConfig().enable_timing;

  std::map<int32_t, int32_t>& routing_demand_map = summary.er_summary.routing_demand_map;
  int32_t& total_demand = summary.er_summary.total_demand;
  std::map<int32_t, int32_t>& routing_overflow_map = summary.er_summary.routing_overflow_map;
  int32_t& total_overflow = summary.er_summary.total_overflow;
  std::map<int32_t, double>& routing_wire_length_map = summary.er_summary.routing_wire_length_map;
  double& total_wire_length = summary.er_summary.total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.er_summary.cut_via_num_map;
  int32_t& total_via_num = summary.er_summary.total_via_num;
  std::map<std::string, std::map<std::string, double>>& clock_timing = summary.er_summary.clock_timing;
  std::map<std::string, double>& power_map = summary.er_summary.power_map;

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
  clock_timing.clear();
  power_map.clear();

  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    GridMap<ERNode>& er_node_map = layer_node_map[layer_idx];
    for (int32_t x = 0; x < er_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < er_node_map.get_y_size(); y++) {
        std::map<Orientation, int32_t>& orient_supply_map = er_node_map[x][y].get_orient_supply_map();
        std::map<Orientation, int32_t>& orient_demand_map = er_node_map[x][y].get_orient_demand_map();
        int32_t node_demand = 0;
        int32_t node_overflow = 0;
        if (routing_layer_list[layer_idx].isPreferH()) {
          node_demand = (orient_demand_map[Orientation::kEast] + orient_demand_map[Orientation::kWest]);
          node_overflow = std::max(0, orient_demand_map[Orientation::kEast] - orient_supply_map[Orientation::kEast])
                          + std::max(0, orient_demand_map[Orientation::kWest] - orient_supply_map[Orientation::kWest]);
        } else {
          node_demand = (orient_demand_map[Orientation::kSouth] + orient_demand_map[Orientation::kNorth]);
          node_overflow = std::max(0, orient_demand_map[Orientation::kSouth] - orient_supply_map[Orientation::kSouth])
                          + std::max(0, orient_demand_map[Orientation::kNorth] - orient_supply_map[Orientation::kNorth]);
        }
        routing_demand_map[layer_idx] += node_demand;
        total_demand += node_demand;
        routing_overflow_map[layer_idx] += node_overflow;
        total_overflow += node_overflow;
      }
    }
  }
  for (auto& [net_idx, segment_set] : RTDM.getNetGlobalResultMap(die)) {
    for (Segment<LayerCoord>* segment : segment_set) {
      LayerCoord& first_coord = segment->get_first();
      int32_t first_layer_idx = first_coord.get_layer_idx();
      LayerCoord& second_coord = segment->get_second();
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
    for (auto& [net_idx, segment_set] : RTDM.getNetGlobalResultMap(die)) {
      for (Segment<LayerCoord>* segment : segment_set) {
        LayerCoord first_layer_coord = segment->get_first();
        LayerCoord first_real_coord(RTUTIL.getRealRectByGCell(first_layer_coord, gcell_axis).getMidPoint(), first_layer_coord.get_layer_idx());
        LayerCoord second_layer_coord = segment->get_second();
        LayerCoord second_real_coord(RTUTIL.getRealRectByGCell(second_layer_coord, gcell_axis).getMidPoint(), second_layer_coord.get_layer_idx());

        routing_segment_list_list[net_idx].emplace_back(first_real_coord, second_real_coord);
      }
    }
    RTI.updateTimingAndPower(real_pin_coord_map_list, routing_segment_list_list, clock_timing, power_map);
  }
}

void EarlyRouter::printSummary(ERModel& er_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  Summary& summary = RTDM.getDatabase().get_summary();
  int32_t enable_timing = RTDM.getConfig().enable_timing;

  std::map<int32_t, int32_t>& routing_demand_map = summary.er_summary.routing_demand_map;
  int32_t& total_demand = summary.er_summary.total_demand;
  std::map<int32_t, int32_t>& routing_overflow_map = summary.er_summary.routing_overflow_map;
  int32_t& total_overflow = summary.er_summary.total_overflow;
  std::map<int32_t, double>& routing_wire_length_map = summary.er_summary.routing_wire_length_map;
  double& total_wire_length = summary.er_summary.total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.er_summary.cut_via_num_map;
  int32_t& total_via_num = summary.er_summary.total_via_num;
  std::map<std::string, std::map<std::string, double>>& clock_timing = summary.er_summary.clock_timing;
  std::map<std::string, double>& power_map = summary.er_summary.power_map;

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
    for (auto& [clock_name, timing_map] : clock_timing) {
      timing_table << clock_name << timing_map["TNS"] << timing_map["WNS"] << timing_map["Freq(MHz)"] << fort::endr;
    }
    power_table << fort::header << "power_type";
    for (auto& [type, power] : power_map) {
      power_table << fort::header << type;
    }
    power_table << fort::endr;
    power_table << "power_value";
    for (auto& [type, power] : power_map) {
      power_table << power;
    }
    power_table << fort::endr;
  }
  RTUTIL.printTableList({routing_demand_map_table, routing_overflow_map_table, routing_wire_length_map_table, cut_via_num_map_table});
  RTUTIL.printTableList({timing_table, power_table});
}

void EarlyRouter::outputGuide(ERModel& er_model)
{
  int32_t micron_dbu = RTDM.getDatabase().get_micron_dbu();
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& er_temp_directory_path = RTDM.getConfig().er_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  std::vector<ERNet>& er_net_list = er_model.get_er_net_list();

  std::ofstream* guide_file_stream = RTUTIL.getOutputFileStream(RTUTIL.getString(er_temp_directory_path, "route.guide"));
  if (guide_file_stream == nullptr) {
    return;
  }
  RTUTIL.pushStream(guide_file_stream, "guide net_name\n");
  RTUTIL.pushStream(guide_file_stream, "pin grid_x grid_y real_x real_y layer energy name\n");
  RTUTIL.pushStream(guide_file_stream, "wire grid1_x grid1_y grid2_x grid2_y real1_x real1_y real2_x real2_y layer\n");
  RTUTIL.pushStream(guide_file_stream, "via grid_x grid_y real_x real_y layer1 layer2\n");

  for (auto& [net_idx, segment_set] : RTDM.getNetGlobalResultMap(die)) {
    ERNet& er_net = er_net_list[net_idx];
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
    for (Segment<LayerCoord>* segment : segment_set) {
      LayerCoord first_layer_coord = segment->get_first();
      double grid1_x = first_layer_coord.get_x();
      double grid1_y = first_layer_coord.get_y();
      int32_t first_layer_idx = first_layer_coord.get_layer_idx();

      PlanarCoord first_mid_coord = RTUTIL.getRealRectByGCell(first_layer_coord, gcell_axis).getMidPoint();
      double real1_x = first_mid_coord.get_x() / 1.0 / micron_dbu;
      double real1_y = first_mid_coord.get_y() / 1.0 / micron_dbu;

      LayerCoord second_layer_coord = segment->get_second();
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
}

void EarlyRouter::outputDemandCSV(ERModel& er_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& er_temp_directory_path = RTDM.getConfig().er_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  std::vector<GridMap<ERNode>>& layer_node_map = er_model.get_layer_node_map();
  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::ofstream* demand_csv_file
        = RTUTIL.getOutputFileStream(RTUTIL.getString(er_temp_directory_path, "demand_map_", routing_layer.get_layer_name(), ".csv"));

    GridMap<ERNode>& er_node_map = layer_node_map[routing_layer.get_layer_idx()];
    for (int32_t y = er_node_map.get_y_size() - 1; y >= 0; y--) {
      for (int32_t x = 0; x < er_node_map.get_x_size(); x++) {
        std::map<Orientation, int32_t>& orient_demand_map = er_node_map[x][y].get_orient_demand_map();
        int32_t total_demand = 0;
        if (routing_layer.isPreferH()) {
          total_demand = (orient_demand_map[Orientation::kEast] + orient_demand_map[Orientation::kWest]);
        } else {
          total_demand = (orient_demand_map[Orientation::kSouth] + orient_demand_map[Orientation::kNorth]);
        }
        RTUTIL.pushStream(demand_csv_file, total_demand, ",");
      }
      RTUTIL.pushStream(demand_csv_file, "\n");
    }
    RTUTIL.closeFileStream(demand_csv_file);
  }
}

void EarlyRouter::outputOverflowCSV(ERModel& er_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& er_temp_directory_path = RTDM.getConfig().er_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  std::vector<GridMap<ERNode>>& layer_node_map = er_model.get_layer_node_map();
  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::ofstream* overflow_csv_file
        = RTUTIL.getOutputFileStream(RTUTIL.getString(er_temp_directory_path, "overflow_map_", routing_layer.get_layer_name(), ".csv"));

    GridMap<ERNode>& er_node_map = layer_node_map[routing_layer.get_layer_idx()];
    for (int32_t y = er_node_map.get_y_size() - 1; y >= 0; y--) {
      for (int32_t x = 0; x < er_node_map.get_x_size(); x++) {
        std::map<Orientation, int32_t>& orient_supply_map = er_node_map[x][y].get_orient_supply_map();
        std::map<Orientation, int32_t>& orient_demand_map = er_node_map[x][y].get_orient_demand_map();
        int32_t total_overflow = 0;
        if (routing_layer.isPreferH()) {
          total_overflow = std::max(0, orient_demand_map[Orientation::kEast] - orient_supply_map[Orientation::kEast])
                           + std::max(0, orient_demand_map[Orientation::kWest] - orient_supply_map[Orientation::kWest]);
        } else {
          total_overflow = std::max(0, orient_demand_map[Orientation::kSouth] - orient_supply_map[Orientation::kSouth])
                           + std::max(0, orient_demand_map[Orientation::kNorth] - orient_supply_map[Orientation::kNorth]);
        }
        RTUTIL.pushStream(overflow_csv_file, total_overflow, ",");
      }
      RTUTIL.pushStream(overflow_csv_file, "\n");
    }
    RTUTIL.closeFileStream(overflow_csv_file);
  }
}

#endif

#if 1  // debug

void EarlyRouter::debugCheckPlanarNodeMap(ERModel& er_model)
{
  GridMap<ERNode>& er_node_map = er_model.get_planar_node_map();
  for (int32_t x = 0; x < er_node_map.get_x_size(); x++) {
    for (int32_t y = 0; y < er_node_map.get_y_size(); y++) {
      ERNode& er_node = er_node_map[x][y];
      for (auto& [orient, neighbor] : er_node.get_neighbor_node_map()) {
        Orientation opposite_orient = RTUTIL.getOppositeOrientation(orient);
        if (!RTUTIL.exist(neighbor->get_neighbor_node_map(), opposite_orient)) {
          RTLOG.error(Loc::current(), "The er_node neighbor is not bidirectional!");
        }
        if (neighbor->get_neighbor_node_map()[opposite_orient] != &er_node) {
          RTLOG.error(Loc::current(), "The er_node neighbor is not bidirectional!");
        }
        if (RTUTIL.getOrientation(PlanarCoord(er_node), PlanarCoord(*neighbor)) == orient) {
          continue;
        }
        RTLOG.error(Loc::current(), "The neighbor orient is different with real region!");
      }
    }
  }
}

void EarlyRouter::debugCheckLayerNodeMap(ERModel& er_model)
{
  std::vector<GridMap<ERNode>>& layer_node_map = er_model.get_layer_node_map();
  for (GridMap<ERNode>& er_node_map : layer_node_map) {
    for (int32_t x = 0; x < er_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < er_node_map.get_y_size(); y++) {
        ERNode& er_node = er_node_map[x][y];
        for (auto& [orient, neighbor] : er_node.get_neighbor_node_map()) {
          Orientation opposite_orient = RTUTIL.getOppositeOrientation(orient);
          if (!RTUTIL.exist(neighbor->get_neighbor_node_map(), opposite_orient)) {
            RTLOG.error(Loc::current(), "The er_node neighbor is not bidirectional!");
          }
          if (neighbor->get_neighbor_node_map()[opposite_orient] != &er_node) {
            RTLOG.error(Loc::current(), "The er_node neighbor is not bidirectional!");
          }
          if (RTUTIL.getOrientation(LayerCoord(er_node), LayerCoord(*neighbor)) == orient) {
            continue;
          }
          RTLOG.error(Loc::current(), "The neighbor orient is different with real region!");
        }
      }
    }
  }
}

#endif

}  // namespace irt
