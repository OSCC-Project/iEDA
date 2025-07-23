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
#include "TopologyGenerator.hpp"

#include "GDSPlotter.hpp"
#include "RTInterface.hpp"
#include "Utility.hpp"

namespace irt {

// public

void TopologyGenerator::initInst()
{
  if (_tg_instance == nullptr) {
    _tg_instance = new TopologyGenerator();
  }
}

TopologyGenerator& TopologyGenerator::getInst()
{
  if (_tg_instance == nullptr) {
    RTLOG.error(Loc::current(), "The instance not initialized!");
  }
  return *_tg_instance;
}

void TopologyGenerator::destroyInst()
{
  if (_tg_instance != nullptr) {
    delete _tg_instance;
    _tg_instance = nullptr;
  }
}

// function

void TopologyGenerator::generate()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");
  TGModel tg_model = initTGModel();
  setTGComParam(tg_model);
  initTGTaskList(tg_model);
  buildTGNodeMap(tg_model);
  buildTGNodeNeighbor(tg_model);
  buildOrientSupply(tg_model);
  // debugCheckTGModel(tg_model);
  generateTGModel(tg_model);
  // debugPlotTGModel(tg_model, "after");
  updateSummary(tg_model);
  printSummary(tg_model);
  outputGuide(tg_model);
  outputNetCSV(tg_model);
  outputOverflowCSV(tg_model);
  outputJson(tg_model);
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

// private

TopologyGenerator* TopologyGenerator::_tg_instance = nullptr;

TGModel TopologyGenerator::initTGModel()
{
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();

  TGModel tg_model;
  tg_model.set_tg_net_list(convertToTGNetList(net_list));
  return tg_model;
}

std::vector<TGNet> TopologyGenerator::convertToTGNetList(std::vector<Net>& net_list)
{
  std::vector<TGNet> tg_net_list;
  tg_net_list.reserve(net_list.size());
  for (size_t i = 0; i < net_list.size(); i++) {
    tg_net_list.emplace_back(convertToTGNet(net_list[i]));
  }
  return tg_net_list;
}

TGNet TopologyGenerator::convertToTGNet(Net& net)
{
  TGNet tg_net;
  tg_net.set_origin_net(&net);
  tg_net.set_net_idx(net.get_net_idx());
  tg_net.set_connect_type(net.get_connect_type());
  for (Pin& pin : net.get_pin_list()) {
    tg_net.get_tg_pin_list().push_back(TGPin(pin));
  }
  tg_net.set_bounding_box(net.get_bounding_box());
  return tg_net;
}

void TopologyGenerator::setTGComParam(TGModel& tg_model)
{
  int32_t topo_spilt_length = 10;
  int32_t expand_step_num = 5;
  int32_t expand_step_length = 2;
  double prefer_wire_unit = 1;
  double non_prefer_wire_unit = 2.5 * prefer_wire_unit;
  double overflow_unit = 4 * non_prefer_wire_unit;
  /**
   * topo_spilt_length, expand_step_num, expand_step_length, overflow_unit
   */
  // clang-format off
  TGComParam tg_com_param(topo_spilt_length, expand_step_num, expand_step_length, overflow_unit);
  // clang-format on
  RTLOG.info(Loc::current(), "topo_spilt_length: ", tg_com_param.get_topo_spilt_length());
  RTLOG.info(Loc::current(), "expand_step_num: ", tg_com_param.get_expand_step_num());
  RTLOG.info(Loc::current(), "expand_step_length: ", tg_com_param.get_expand_step_length());
  RTLOG.info(Loc::current(), "overflow_unit: ", tg_com_param.get_overflow_unit());
  tg_model.set_tg_com_param(tg_com_param);
}

void TopologyGenerator::initTGTaskList(TGModel& tg_model)
{
  std::vector<TGNet>& tg_net_list = tg_model.get_tg_net_list();
  std::vector<TGNet*>& tg_task_list = tg_model.get_tg_task_list();
  tg_task_list.reserve(tg_net_list.size());
  for (TGNet& tg_net : tg_net_list) {
    tg_task_list.push_back(&tg_net);
  }
  std::sort(tg_task_list.begin(), tg_task_list.end(), CmpTGNet());
}

void TopologyGenerator::buildTGNodeMap(TGModel& tg_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  GridMap<TGNode>& tg_node_map = tg_model.get_tg_node_map();
  tg_node_map.init(gcell_map.get_x_size(), gcell_map.get_y_size());
#pragma omp parallel for collapse(2)
  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      TGNode& tg_node = tg_node_map[x][y];
      tg_node.set_coord(x, y);
      tg_node.set_boundary_wire_unit(gcell_map[x][y].get_boundary_wire_unit());
      tg_node.set_internal_wire_unit(gcell_map[x][y].get_internal_wire_unit());
      tg_node.set_internal_via_unit(gcell_map[x][y].get_internal_via_unit());
      for (auto& [routing_layer_idx, ignore_net_orient_map] : gcell_map[x][y].get_routing_ignore_net_orient_map()) {
        for (auto& [net_idx, orient_set] : ignore_net_orient_map) {
          tg_node.get_ignore_net_orient_map()[net_idx].insert(orient_set.begin(), orient_set.end());
        }
      }
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void TopologyGenerator::buildTGNodeNeighbor(TGModel& tg_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();

  GridMap<TGNode>& tg_node_map = tg_model.get_tg_node_map();
#pragma omp parallel for collapse(2)
  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      std::map<Orientation, TGNode*>& neighbor_node_map = tg_node_map[x][y].get_neighbor_node_map();
      if (x != 0) {
        neighbor_node_map[Orientation::kWest] = &tg_node_map[x - 1][y];
      }
      if (x != (tg_node_map.get_x_size() - 1)) {
        neighbor_node_map[Orientation::kEast] = &tg_node_map[x + 1][y];
      }
      if (y != 0) {
        neighbor_node_map[Orientation::kSouth] = &tg_node_map[x][y - 1];
      }
      if (y != (tg_node_map.get_y_size() - 1)) {
        neighbor_node_map[Orientation::kNorth] = &tg_node_map[x][y + 1];
      }
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void TopologyGenerator::buildOrientSupply(TGModel& tg_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  GridMap<TGNode>& tg_node_map = tg_model.get_tg_node_map();

#pragma omp parallel for collapse(2)
  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      std::map<Orientation, int32_t> planar_orient_supply_map;
      for (auto& [layer_idx, orient_supply_map] : gcell_map[x][y].get_routing_orient_supply_map()) {
        for (auto& [orient, supply] : orient_supply_map) {
          planar_orient_supply_map[orient] += supply;
        }
      }
      tg_node_map[x][y].set_orient_supply_map(planar_orient_supply_map);
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void TopologyGenerator::generateTGModel(TGModel& tg_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<TGNet*>& tg_task_list = tg_model.get_tg_task_list();

  int32_t batch_size = RTUTIL.getBatchSize(tg_task_list.size());

  Monitor stage_monitor;
  for (size_t i = 0; i < tg_task_list.size(); i++) {
    routeTGTask(tg_model, tg_task_list[i]);
    if ((i + 1) % batch_size == 0 || (i + 1) == tg_task_list.size()) {
      RTLOG.info(Loc::current(), "Routed ", (i + 1), "/", tg_task_list.size(), "(", RTUTIL.getPercentage(i + 1, tg_task_list.size()), ") nets",
                 stage_monitor.getStatsInfo());
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void TopologyGenerator::routeTGTask(TGModel& tg_model, TGNet* tg_task)
{
  initSingleTask(tg_model, tg_task);
  std::vector<Segment<PlanarCoord>> routing_segment_list;
  for (Segment<PlanarCoord>& planar_topo : getPlanarTopoList(tg_model)) {
    for (Segment<PlanarCoord>& routing_segment : getRoutingSegmentList(tg_model, planar_topo)) {
      routing_segment_list.push_back(routing_segment);
    }
  }
  MTree<PlanarCoord> coord_tree = getCoordTree(tg_model, routing_segment_list);
  updateDemandToGraph(tg_model, ChangeType::kAdd, coord_tree);
  uploadNetResult(tg_model, coord_tree);
  resetSingleTask(tg_model);
}

void TopologyGenerator::initSingleTask(TGModel& tg_model, TGNet* tg_task)
{
  tg_model.set_curr_tg_task(tg_task);
}

std::vector<Segment<PlanarCoord>> TopologyGenerator::getPlanarTopoList(TGModel& tg_model)
{
  int32_t topo_spilt_length = tg_model.get_tg_com_param().get_topo_spilt_length();

  std::vector<PlanarCoord> planar_coord_list;
  {
    for (TGPin& tg_pin : tg_model.get_curr_tg_task()->get_tg_pin_list()) {
      planar_coord_list.push_back(tg_pin.get_access_point().get_grid_coord());
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

std::vector<Segment<PlanarCoord>> TopologyGenerator::getRoutingSegmentList(TGModel& tg_model, Segment<PlanarCoord>& planar_topo)
{
  std::vector<std::vector<Segment<PlanarCoord>>> routing_segment_list_list;
  for (auto getRoutingSegmentList : {std::bind(&TopologyGenerator::getRoutingSegmentListByStraight, this, std::placeholders::_1, std::placeholders::_2),
                                     std::bind(&TopologyGenerator::getRoutingSegmentListByLPattern, this, std::placeholders::_1, std::placeholders::_2),
                                     std::bind(&TopologyGenerator::getRoutingSegmentListByZPattern, this, std::placeholders::_1, std::placeholders::_2),
                                     std::bind(&TopologyGenerator::getRoutingSegmentListByUPattern, this, std::placeholders::_1, std::placeholders::_2),
                                     std::bind(&TopologyGenerator::getRoutingSegmentListByInner3Bends, this, std::placeholders::_1, std::placeholders::_2),
                                     std::bind(&TopologyGenerator::getRoutingSegmentListByOuter3Bends, this, std::placeholders::_1, std::placeholders::_2)}) {
    for (std::vector<Segment<PlanarCoord>> routing_segment_list : getRoutingSegmentList(tg_model, planar_topo)) {
      routing_segment_list_list.push_back(routing_segment_list);
    }
  }

  double min_cost = DBL_MAX;
  size_t min_i = 0;
  for (size_t i = 0; i < routing_segment_list_list.size(); i++) {
    double cost = getNodeCost(tg_model, routing_segment_list_list[i]);
    if (cost < min_cost) {
      min_cost = cost;
      min_i = i;
    }
  }
  return routing_segment_list_list[min_i];
}

std::vector<std::vector<Segment<PlanarCoord>>> TopologyGenerator::getRoutingSegmentListByStraight(TGModel& tg_model, Segment<PlanarCoord>& planar_topo)
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

std::vector<std::vector<Segment<PlanarCoord>>> TopologyGenerator::getRoutingSegmentListByLPattern(TGModel& tg_model, Segment<PlanarCoord>& planar_topo)
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

std::vector<std::vector<Segment<PlanarCoord>>> TopologyGenerator::getRoutingSegmentListByZPattern(TGModel& tg_model, Segment<PlanarCoord>& planar_topo)
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

std::vector<int32_t> TopologyGenerator::getMidIndexList(int32_t first_idx, int32_t second_idx)
{
  std::vector<int32_t> mid_index_list;
  RTUTIL.swapByASC(first_idx, second_idx);
  mid_index_list.reserve(second_idx - first_idx - 1);
  for (int32_t i = (first_idx + 1); i <= (second_idx - 1); i++) {
    mid_index_list.push_back(i);
  }
  return mid_index_list;
}

std::vector<std::vector<Segment<PlanarCoord>>> TopologyGenerator::getRoutingSegmentListByUPattern(TGModel& tg_model, Segment<PlanarCoord>& planar_topo)
{
  Die& die = RTDM.getDatabase().get_die();
  int32_t expand_step_num = tg_model.get_tg_com_param().get_expand_step_num();
  int32_t expand_step_length = tg_model.get_tg_com_param().get_expand_step_length();

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

std::vector<std::vector<Segment<PlanarCoord>>> TopologyGenerator::getRoutingSegmentListByInner3Bends(TGModel& tg_model, Segment<PlanarCoord>& planar_topo)
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

std::vector<std::vector<Segment<PlanarCoord>>> TopologyGenerator::getRoutingSegmentListByOuter3Bends(TGModel& tg_model, Segment<PlanarCoord>& planar_topo)
{
  Die& die = RTDM.getDatabase().get_die();
  int32_t expand_step_num = tg_model.get_tg_com_param().get_expand_step_num();
  int32_t expand_step_length = tg_model.get_tg_com_param().get_expand_step_length();

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

double TopologyGenerator::getNodeCost(TGModel& tg_model, std::vector<Segment<PlanarCoord>>& routing_segment_list)
{
  double overflow_unit = tg_model.get_tg_com_param().get_overflow_unit();
  GridMap<TGNode>& tg_node_map = tg_model.get_tg_node_map();
  int32_t curr_net_idx = tg_model.get_curr_tg_task()->get_net_idx();

  std::set<PlanarCoord, CmpPlanarCoordByXASC> coord_set;
  for (Segment<PlanarCoord>& coord_segment : routing_segment_list) {
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
    for (int32_t x = first_x; x <= second_x; x++) {
      for (int32_t y = first_y; y <= second_y; y++) {
        coord_set.insert(PlanarCoord(x, y));
      }
    }
  }
  double node_cost = 0;
  for (const PlanarCoord& coord : coord_set) {
    node_cost += tg_node_map[coord.get_x()][coord.get_y()].getOverflowCost(curr_net_idx, overflow_unit);
  }
  return node_cost;
}

MTree<PlanarCoord> TopologyGenerator::getCoordTree(TGModel& tg_model, std::vector<Segment<PlanarCoord>>& routing_segment_list)
{
  std::vector<PlanarCoord> candidate_root_coord_list;
  std::map<PlanarCoord, std::set<int32_t>, CmpPlanarCoordByXASC> key_coord_pin_map;
  std::vector<TGPin>& tg_pin_list = tg_model.get_curr_tg_task()->get_tg_pin_list();
  for (size_t i = 0; i < tg_pin_list.size(); i++) {
    PlanarCoord coord = tg_pin_list[i].get_access_point().get_grid_coord();
    candidate_root_coord_list.push_back(coord);
    key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
  }
  return RTUTIL.getTreeByFullFlow(candidate_root_coord_list, routing_segment_list, key_coord_pin_map);
}

void TopologyGenerator::uploadNetResult(TGModel& tg_model, MTree<PlanarCoord>& coord_tree)
{
  for (Segment<TNode<PlanarCoord>*>& coord_segment : RTUTIL.getSegListByTree(coord_tree)) {
    Segment<LayerCoord>* segment = new Segment<LayerCoord>({coord_segment.get_first()->value(), 0}, {coord_segment.get_second()->value(), 0});
    RTDM.updateNetGlobalResultToGCellMap(ChangeType::kAdd, tg_model.get_curr_tg_task()->get_net_idx(), segment);
  }
}

void TopologyGenerator::resetSingleTask(TGModel& tg_model)
{
  tg_model.set_curr_tg_task(nullptr);
}

#if 1  // update env

void TopologyGenerator::updateDemandToGraph(TGModel& tg_model, ChangeType change_type, MTree<PlanarCoord>& coord_tree)
{
  int32_t curr_net_idx = tg_model.get_curr_tg_task()->get_net_idx();

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
  GridMap<TGNode>& tg_node_map = tg_model.get_tg_node_map();
  for (auto& [usage_coord, orientation_list] : usage_map) {
    TGNode& tg_node = tg_node_map[usage_coord.get_x()][usage_coord.get_y()];
    tg_node.updateDemand(curr_net_idx, orientation_list, change_type);
  }
}

#endif

#if 1  // exhibit

void TopologyGenerator::updateSummary(TGModel& tg_model)
{
  int32_t micron_dbu = RTDM.getDatabase().get_micron_dbu();
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  Summary& summary = RTDM.getDatabase().get_summary();
  int32_t enable_timing = RTDM.getConfig().enable_timing;

  double& total_demand = summary.tg_summary.total_demand;
  double& total_overflow = summary.tg_summary.total_overflow;
  double& total_wire_length = summary.tg_summary.total_wire_length;
  std::map<std::string, std::map<std::string, double>>& clock_timing_map = summary.tg_summary.clock_timing_map;
  std::map<std::string, double>& type_power_map = summary.tg_summary.type_power_map;

  std::vector<TGNet>& tg_net_list = tg_model.get_tg_net_list();
  GridMap<TGNode>& tg_node_map = tg_model.get_tg_node_map();

  total_demand = 0;
  total_overflow = 0;
  total_wire_length = 0;
  clock_timing_map.clear();
  type_power_map.clear();

  for (int32_t x = 0; x < tg_node_map.get_x_size(); x++) {
    for (int32_t y = 0; y < tg_node_map.get_y_size(); y++) {
      double node_demand = tg_node_map[x][y].getDemand();
      double node_overflow = tg_node_map[x][y].getOverflow();
      total_demand += node_demand;
      total_overflow += node_overflow;
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
        total_wire_length += wire_length;
      } else {
        RTLOG.error(Loc::current(), "first_layer_idx != second_layer_idx!");
      }
    }
  }
  if (enable_timing) {
    std::vector<std::map<std::string, std::vector<LayerCoord>>> real_pin_coord_map_list;
    real_pin_coord_map_list.resize(tg_net_list.size());
    std::vector<std::vector<Segment<LayerCoord>>> routing_segment_list_list;
    routing_segment_list_list.resize(tg_net_list.size());
    for (TGNet& tg_net : tg_net_list) {
      for (TGPin& tg_pin : tg_net.get_tg_pin_list()) {
        LayerCoord layer_coord = tg_pin.get_access_point().getGridLayerCoord();
        real_pin_coord_map_list[tg_net.get_net_idx()][tg_pin.get_pin_name()].emplace_back(RTUTIL.getRealRectByGCell(layer_coord, gcell_axis).getMidPoint(), 0);
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
    RTI.updateTimingAndPower(real_pin_coord_map_list, routing_segment_list_list, clock_timing_map, type_power_map);
  }
}

void TopologyGenerator::printSummary(TGModel& tg_model)
{
  Summary& summary = RTDM.getDatabase().get_summary();
  int32_t enable_timing = RTDM.getConfig().enable_timing;

  double& total_demand = summary.tg_summary.total_demand;
  double& total_overflow = summary.tg_summary.total_overflow;
  double& total_wire_length = summary.tg_summary.total_wire_length;
  std::map<std::string, std::map<std::string, double>>& clock_timing_map = summary.tg_summary.clock_timing_map;
  std::map<std::string, double>& type_power_map = summary.tg_summary.type_power_map;

  fort::char_table summary_table;
  {
    summary_table.set_cell_text_align(fort::text_align::right);
    summary_table << fort::header << "total_demand" << total_demand << fort::endr;
    summary_table << fort::header << "total_overflow" << total_overflow << fort::endr;
    summary_table << fort::header << "total_wire_length" << total_wire_length << fort::endr;
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
  RTUTIL.printTableList({summary_table});
  RTUTIL.printTableList({timing_table, power_table});
}

void TopologyGenerator::outputGuide(TGModel& tg_model)
{
  int32_t micron_dbu = RTDM.getDatabase().get_micron_dbu();
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& tg_temp_directory_path = RTDM.getConfig().tg_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  std::vector<TGNet>& tg_net_list = tg_model.get_tg_net_list();

  std::ofstream* guide_file_stream = RTUTIL.getOutputFileStream(RTUTIL.getString(tg_temp_directory_path, "route.guide"));
  if (guide_file_stream == nullptr) {
    return;
  }
  RTUTIL.pushStream(guide_file_stream, "guide net_name\n");
  RTUTIL.pushStream(guide_file_stream, "pin grid_x grid_y real_x real_y layer energy name\n");
  RTUTIL.pushStream(guide_file_stream, "wire grid1_x grid1_y grid2_x grid2_y real1_x real1_y real2_x real2_y layer\n");
  RTUTIL.pushStream(guide_file_stream, "via grid_x grid_y real_x real_y layer1 layer2\n");

  for (auto& [net_idx, segment_set] : RTDM.getNetGlobalResultMap(die)) {
    TGNet& tg_net = tg_net_list[net_idx];
    RTUTIL.pushStream(guide_file_stream, "guide ", tg_net.get_origin_net()->get_net_name(), "\n");

    for (TGPin& tg_pin : tg_net.get_tg_pin_list()) {
      AccessPoint& access_point = tg_pin.get_access_point();
      double grid_x = access_point.get_grid_x();
      double grid_y = access_point.get_grid_y();
      double real_x = access_point.get_real_x() / 1.0 / micron_dbu;
      double real_y = access_point.get_real_y() / 1.0 / micron_dbu;
      std::string layer = routing_layer_list[access_point.get_layer_idx()].get_layer_name();
      std::string connnect;
      if (tg_pin.get_is_driven()) {
        connnect = "driven";
      } else {
        connnect = "load";
      }
      RTUTIL.pushStream(guide_file_stream, "pin ", grid_x, " ", grid_y, " ", real_x, " ", real_y, " ", layer, " ", connnect, " ", tg_pin.get_pin_name(), "\n");
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

void TopologyGenerator::outputNetCSV(TGModel& tg_model)
{
  std::string& tg_temp_directory_path = RTDM.getConfig().tg_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  std::ofstream* net_csv_file = RTUTIL.getOutputFileStream(RTUTIL.getString(tg_temp_directory_path, "net_map.csv"));
  GridMap<TGNode>& tg_node_map = tg_model.get_tg_node_map();
  for (int32_t y = tg_node_map.get_y_size() - 1; y >= 0; y--) {
    for (int32_t x = 0; x < tg_node_map.get_x_size(); x++) {
      RTUTIL.pushStream(net_csv_file, tg_node_map[x][y].getDemand(), ",");
    }
    RTUTIL.pushStream(net_csv_file, "\n");
  }
  RTUTIL.closeFileStream(net_csv_file);
}

void TopologyGenerator::outputOverflowCSV(TGModel& tg_model)
{
  std::string& tg_temp_directory_path = RTDM.getConfig().tg_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  std::ofstream* overflow_csv_file = RTUTIL.getOutputFileStream(RTUTIL.getString(tg_temp_directory_path, "overflow_map.csv"));
  GridMap<TGNode>& tg_node_map = tg_model.get_tg_node_map();
  for (int32_t y = tg_node_map.get_y_size() - 1; y >= 0; y--) {
    for (int32_t x = 0; x < tg_node_map.get_x_size(); x++) {
      RTUTIL.pushStream(overflow_csv_file, tg_node_map[x][y].getOverflow(), ",");
    }
    RTUTIL.pushStream(overflow_csv_file, "\n");
  }
  RTUTIL.closeFileStream(overflow_csv_file);
}

void TopologyGenerator::outputJson(TGModel& tg_model)
{
  int32_t enable_notification = RTDM.getConfig().enable_notification;
  if (!enable_notification) {
    return;
  }
  std::map<std::string, std::string> json_path_map;
  json_path_map["net_map"] = outputNetJson(tg_model);
  json_path_map["overflow_map"] = outputOverflowJson(tg_model);
  json_path_map["summary"] = outputSummaryJson(tg_model);
  RTI.sendNotification("TG", 1, json_path_map);
}

std::string TopologyGenerator::outputNetJson(TGModel& tg_model)
{
  Die& die = RTDM.getDatabase().get_die();
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();
  std::string& tg_temp_directory_path = RTDM.getConfig().tg_temp_directory_path;

  std::vector<nlohmann::json> net_json_list;
  {
    nlohmann::json result_shape_json;
    for (auto& [net_idx, segment_set] : RTDM.getNetGlobalResultMap(die)) {
      std::string net_name = net_list[net_idx].get_net_name();
      for (Segment<LayerCoord>* segment : segment_set) {
        PlanarRect first_gcell = RTUTIL.getRealRectByGCell(segment->get_first(), gcell_axis);
        PlanarRect second_gcell = RTUTIL.getRealRectByGCell(segment->get_second(), gcell_axis);
        if (segment->get_first().get_layer_idx() != segment->get_second().get_layer_idx()) {
          result_shape_json["result_shape"][net_name]["path"].push_back({first_gcell.get_ll_x(), first_gcell.get_ll_y(), first_gcell.get_ur_x(),
                                                                         first_gcell.get_ur_y(),
                                                                         routing_layer_list[segment->get_first().get_layer_idx()].get_layer_name()});
          result_shape_json["result_shape"][net_name]["path"].push_back({second_gcell.get_ll_x(), second_gcell.get_ll_y(), second_gcell.get_ur_x(),
                                                                         second_gcell.get_ur_y(),
                                                                         routing_layer_list[segment->get_second().get_layer_idx()].get_layer_name()});
        } else {
          PlanarRect gcell = RTUTIL.getBoundingBox({first_gcell, second_gcell});
          result_shape_json["result_shape"][net_name]["path"].push_back({gcell.get_ll_x(), gcell.get_ll_y(), gcell.get_ur_x(), gcell.get_ur_y(),
                                                                         routing_layer_list[segment->get_first().get_layer_idx()].get_layer_name()});
        }
      }
    }
    net_json_list.push_back(result_shape_json);
  }
  std::string net_json_file_path = RTUTIL.getString(tg_temp_directory_path, "net_map.json");
  std::ofstream* net_json_file = RTUTIL.getOutputFileStream(net_json_file_path);
  (*net_json_file) << net_json_list;
  RTUTIL.closeFileStream(net_json_file);
  return net_json_file_path;
}

std::string TopologyGenerator::outputOverflowJson(TGModel& tg_model)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& tg_temp_directory_path = RTDM.getConfig().tg_temp_directory_path;

  GridMap<TGNode>& tg_node_map = tg_model.get_tg_node_map();
  std::vector<nlohmann::json> overflow_json_list;
  for (int32_t x = 0; x < tg_node_map.get_x_size(); x++) {
    for (int32_t y = 0; y < tg_node_map.get_y_size(); y++) {
      PlanarRect gcell = RTUTIL.getRealRectByGCell(PlanarCoord(x, y), gcell_axis);
      overflow_json_list.push_back(
          {gcell.get_ll_x(), gcell.get_ll_y(), gcell.get_ur_x(), gcell.get_ur_y(), routing_layer_list[0].get_layer_name(), tg_node_map[x][y].getOverflow()});
    }
  }
  std::string overflow_json_file_path = RTUTIL.getString(tg_temp_directory_path, "overflow_map.json");
  std::ofstream* overflow_json_file = RTUTIL.getOutputFileStream(overflow_json_file_path);
  (*overflow_json_file) << overflow_json_list;
  RTUTIL.closeFileStream(overflow_json_file);
  return overflow_json_file_path;
}

std::string TopologyGenerator::outputSummaryJson(TGModel& tg_model)
{
  Summary& summary = RTDM.getDatabase().get_summary();
  std::string& tg_temp_directory_path = RTDM.getConfig().tg_temp_directory_path;

  double& total_demand = summary.tg_summary.total_demand;
  double& total_overflow = summary.tg_summary.total_overflow;
  double& total_wire_length = summary.tg_summary.total_wire_length;
  std::map<std::string, std::map<std::string, double>>& clock_timing_map = summary.tg_summary.clock_timing_map;
  std::map<std::string, double>& type_power_map = summary.tg_summary.type_power_map;

  nlohmann::json summary_json;
  summary_json["total_demand"] = total_demand;
  summary_json["total_overflow"] = total_overflow;
  summary_json["total_wire_length"] = total_wire_length;
  for (auto& [clock_name, timing] : clock_timing_map) {
    summary_json["clock_timing_map"]["clock_name"] = clock_name;
    summary_json["clock_timing_map"]["timing"] = timing;
  }
  for (auto& [type, power] : type_power_map) {
    summary_json["type_power_map"]["type"] = type;
    summary_json["type_power_map"]["power"] = power;
  }
  std::string summary_json_file_path = RTUTIL.getString(tg_temp_directory_path, "summary.json");
  std::ofstream* summary_json_file = RTUTIL.getOutputFileStream(summary_json_file_path);
  (*summary_json_file) << summary_json;
  RTUTIL.closeFileStream(summary_json_file);
  return summary_json_file_path;
}

#endif

#if 1  // debug

void TopologyGenerator::debugPlotTGModel(TGModel& tg_model, std::string flag)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();
  std::string& tg_temp_directory_path = RTDM.getConfig().tg_temp_directory_path;

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
  for (auto& [net_idx, segment_set] : RTDM.getNetDetailedResultMap(die)) {
    GPStruct detailed_result_struct(RTUTIL.getString("detailed_result(net_", net_idx, ")"));
    for (Segment<LayerCoord>* segment : segment_set) {
      for (NetShape& net_shape : RTDM.getNetShapeList(net_idx, *segment)) {
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
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShape));
      gp_boundary.set_rect(patch->get_real_rect());
      gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(patch->get_layer_idx()));
      detailed_patch_struct.push(gp_boundary);
    }
    gp_gds.addStruct(detailed_patch_struct);
  }

  {
    GridMap<TGNode>& tg_node_map = tg_model.get_tg_node_map();
    // tg_node_map
    {
      GPStruct tg_node_map_struct("tg_node_map");
      for (int32_t grid_x = 0; grid_x < tg_node_map.get_x_size(); grid_x++) {
        for (int32_t grid_y = 0; grid_y < tg_node_map.get_y_size(); grid_y++) {
          TGNode& tg_node = tg_node_map[grid_x][grid_y];
          PlanarRect real_rect = RTUTIL.getRealRectByGCell(tg_node, gcell_axis);
          int32_t y_reduced_span = std::max(1, real_rect.getYSpan() / 12);
          int32_t y = real_rect.get_ur_y();

          y -= y_reduced_span;
          GPText gp_text_node_real_coord;
          gp_text_node_real_coord.set_coord(real_rect.get_ll_x(), y);
          gp_text_node_real_coord.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          gp_text_node_real_coord.set_message(RTUTIL.getString("(", tg_node.get_x(), " , ", tg_node.get_y(), " , ", 0, ")"));
          gp_text_node_real_coord.set_layer_idx(RTGP.getGDSIdxByRouting(0));
          gp_text_node_real_coord.set_presentation(GPTextPresentation::kLeftMiddle);
          tg_node_map_struct.push(gp_text_node_real_coord);

          y -= y_reduced_span;
          GPText gp_text_node_grid_coord;
          gp_text_node_grid_coord.set_coord(real_rect.get_ll_x(), y);
          gp_text_node_grid_coord.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          gp_text_node_grid_coord.set_message(RTUTIL.getString("(", grid_x, " , ", grid_y, " , ", 0, ")"));
          gp_text_node_grid_coord.set_layer_idx(RTGP.getGDSIdxByRouting(0));
          gp_text_node_grid_coord.set_presentation(GPTextPresentation::kLeftMiddle);
          tg_node_map_struct.push(gp_text_node_grid_coord);

          y -= y_reduced_span;
          GPText gp_text_orient_supply_map;
          gp_text_orient_supply_map.set_coord(real_rect.get_ll_x(), y);
          gp_text_orient_supply_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          gp_text_orient_supply_map.set_message("orient_supply_map: ");
          gp_text_orient_supply_map.set_layer_idx(RTGP.getGDSIdxByRouting(0));
          gp_text_orient_supply_map.set_presentation(GPTextPresentation::kLeftMiddle);
          tg_node_map_struct.push(gp_text_orient_supply_map);

          if (!tg_node.get_orient_supply_map().empty()) {
            y -= y_reduced_span;
            GPText gp_text_orient_supply_map_info;
            gp_text_orient_supply_map_info.set_coord(real_rect.get_ll_x(), y);
            gp_text_orient_supply_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            std::string orient_supply_map_info_message = "--";
            for (auto& [orient, supply] : tg_node.get_orient_supply_map()) {
              orient_supply_map_info_message += RTUTIL.getString("(", GetOrientationName()(orient), ",", supply, ")");
            }
            gp_text_orient_supply_map_info.set_message(orient_supply_map_info_message);
            gp_text_orient_supply_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(0));
            gp_text_orient_supply_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
            tg_node_map_struct.push(gp_text_orient_supply_map_info);
          }

          y -= y_reduced_span;
          GPText gp_text_ignore_net_orient_map;
          gp_text_ignore_net_orient_map.set_coord(real_rect.get_ll_x(), y);
          gp_text_ignore_net_orient_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          gp_text_ignore_net_orient_map.set_message("ignore_net_orient_map: ");
          gp_text_ignore_net_orient_map.set_layer_idx(RTGP.getGDSIdxByRouting(0));
          gp_text_ignore_net_orient_map.set_presentation(GPTextPresentation::kLeftMiddle);
          tg_node_map_struct.push(gp_text_ignore_net_orient_map);

          if (!tg_node.get_ignore_net_orient_map().empty()) {
            y -= y_reduced_span;
            GPText gp_text_ignore_net_orient_map_info;
            gp_text_ignore_net_orient_map_info.set_coord(real_rect.get_ll_x(), y);
            gp_text_ignore_net_orient_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            std::string ignore_net_orient_map_info_message = "--";
            for (auto& [net_idx, orient_set] : tg_node.get_ignore_net_orient_map()) {
              ignore_net_orient_map_info_message += RTUTIL.getString("(", net_idx);
              for (Orientation orient : orient_set) {
                ignore_net_orient_map_info_message += RTUTIL.getString(",", GetOrientationName()(orient));
              }
              ignore_net_orient_map_info_message += RTUTIL.getString(")");
            }
            gp_text_ignore_net_orient_map_info.set_message(ignore_net_orient_map_info_message);
            gp_text_ignore_net_orient_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(0));
            gp_text_ignore_net_orient_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
            tg_node_map_struct.push(gp_text_ignore_net_orient_map_info);
          }

          y -= y_reduced_span;
          GPText gp_text_orient_net_map;
          gp_text_orient_net_map.set_coord(real_rect.get_ll_x(), y);
          gp_text_orient_net_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          gp_text_orient_net_map.set_message("orient_net_map: ");
          gp_text_orient_net_map.set_layer_idx(RTGP.getGDSIdxByRouting(0));
          gp_text_orient_net_map.set_presentation(GPTextPresentation::kLeftMiddle);
          tg_node_map_struct.push(gp_text_orient_net_map);

          if (!tg_node.get_orient_net_map().empty()) {
            y -= y_reduced_span;
            GPText gp_text_orient_net_map_info;
            gp_text_orient_net_map_info.set_coord(real_rect.get_ll_x(), y);
            gp_text_orient_net_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            std::string orient_net_map_info_message = "--";
            for (auto& [orient, net_set] : tg_node.get_orient_net_map()) {
              orient_net_map_info_message += RTUTIL.getString("(", GetOrientationName()(orient));
              for (int32_t net_idx : net_set) {
                orient_net_map_info_message += RTUTIL.getString(",", net_idx);
              }
              orient_net_map_info_message += RTUTIL.getString(")");
            }
            gp_text_orient_net_map_info.set_message(orient_net_map_info_message);
            gp_text_orient_net_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(0));
            gp_text_orient_net_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
            tg_node_map_struct.push(gp_text_orient_net_map_info);
          }

          y -= y_reduced_span;
          GPText gp_text_net_orient_map;
          gp_text_net_orient_map.set_coord(real_rect.get_ll_x(), y);
          gp_text_net_orient_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          gp_text_net_orient_map.set_message("net_orient_map: ");
          gp_text_net_orient_map.set_layer_idx(RTGP.getGDSIdxByRouting(0));
          gp_text_net_orient_map.set_presentation(GPTextPresentation::kLeftMiddle);
          tg_node_map_struct.push(gp_text_net_orient_map);

          if (!tg_node.get_net_orient_map().empty()) {
            y -= y_reduced_span;
            GPText gp_text_net_orient_map_info;
            gp_text_net_orient_map_info.set_coord(real_rect.get_ll_x(), y);
            gp_text_net_orient_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            std::string net_orient_map_info_message = "--";
            for (auto& [net_idx, orient_set] : tg_node.get_net_orient_map()) {
              net_orient_map_info_message += RTUTIL.getString("(", net_idx);
              for (Orientation orient : orient_set) {
                net_orient_map_info_message += RTUTIL.getString(",", GetOrientationName()(orient));
              }
              net_orient_map_info_message += RTUTIL.getString(")");
            }
            gp_text_net_orient_map_info.set_message(net_orient_map_info_message);
            gp_text_net_orient_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(0));
            gp_text_net_orient_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
            tg_node_map_struct.push(gp_text_net_orient_map_info);
          }

          y -= y_reduced_span;
          GPText gp_text_overflow;
          gp_text_overflow.set_coord(real_rect.get_ll_x(), y);
          gp_text_overflow.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          gp_text_overflow.set_message(RTUTIL.getString("overflow: ", tg_node.getOverflow()));
          gp_text_overflow.set_layer_idx(RTGP.getGDSIdxByRouting(0));
          gp_text_overflow.set_presentation(GPTextPresentation::kLeftMiddle);
          tg_node_map_struct.push(gp_text_overflow);
        }
      }
      gp_gds.addStruct(tg_node_map_struct);
    }
    // overflow
    {
      GPStruct overflow_struct("overflow");
      for (int32_t grid_x = 0; grid_x < tg_node_map.get_x_size(); grid_x++) {
        for (int32_t grid_y = 0; grid_y < tg_node_map.get_y_size(); grid_y++) {
          TGNode& tg_node = tg_node_map[grid_x][grid_y];
          if (tg_node.getOverflow() <= 0) {
            continue;
          }
          PlanarRect real_rect = RTUTIL.getRealRectByGCell(tg_node, gcell_axis);

          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kOverflow));
          gp_boundary.set_rect(real_rect);
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(0));
          overflow_struct.push(gp_boundary);
        }
      }
      gp_gds.addStruct(overflow_struct);
    }
  }

  std::string gds_file_path = RTUTIL.getString(tg_temp_directory_path, flag, "_tg_model.gds");
  RTGP.plot(gp_gds, gds_file_path);
}

void TopologyGenerator::debugCheckTGModel(TGModel& tg_model)
{
  GridMap<TGNode>& tg_node_map = tg_model.get_tg_node_map();
  for (int32_t x = 0; x < tg_node_map.get_x_size(); x++) {
    for (int32_t y = 0; y < tg_node_map.get_y_size(); y++) {
      TGNode& tg_node = tg_node_map[x][y];
      for (auto& [orient, neighbor] : tg_node.get_neighbor_node_map()) {
        Orientation opposite_orient = RTUTIL.getOppositeOrientation(orient);
        if (!RTUTIL.exist(neighbor->get_neighbor_node_map(), opposite_orient)) {
          RTLOG.error(Loc::current(), "The tg_node neighbor is not bidirectional!");
        }
        if (neighbor->get_neighbor_node_map()[opposite_orient] != &tg_node) {
          RTLOG.error(Loc::current(), "The tg_node neighbor is not bidirectional!");
        }
        if (RTUTIL.getOrientation(PlanarCoord(tg_node), PlanarCoord(*neighbor)) == orient) {
          continue;
        }
        RTLOG.error(Loc::current(), "The neighbor orient is different with real region!");
      }
    }
  }
}

#endif

}  // namespace irt
