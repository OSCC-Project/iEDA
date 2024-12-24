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
  setTGParameter(tg_model);
  initTGTaskList(tg_model);
  buildTGNodeMap(tg_model);
  buildTGNodeNeighbor(tg_model);
  buildOrientSupply(tg_model);
  // debugCheckTGModel(tg_model);
  generateTGModel(tg_model);
  updateSummary(tg_model);
  printSummary(tg_model);
  outputGuide(tg_model);
  outputPlanarSupplyCSV(tg_model);
  outputPlanarDemandCSV(tg_model);
  outputPlanarOverflowCSV(tg_model);
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

void TopologyGenerator::setTGParameter(TGModel& tg_model)
{
  int32_t topo_spilt_length = 10;
  double congestion_unit = 2;
  /**
   * topo_spilt_length, congestion_unit
   */
  // clang-format off
  TGParameter tg_parameter(topo_spilt_length, congestion_unit);
  // clang-format on
  RTLOG.info(Loc::current(), "topo_spilt_length: ", tg_parameter.get_topo_spilt_length());
  RTLOG.info(Loc::current(), "congestion_unit: ", tg_parameter.get_congestion_unit());
  tg_model.set_tg_parameter(tg_parameter);
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
    routeTGNet(tg_model, tg_task_list[i]);
    if ((i + 1) % batch_size == 0 || (i + 1) == tg_task_list.size()) {
      RTLOG.info(Loc::current(), "Routed ", (i + 1), "/", tg_task_list.size(), "(", RTUTIL.getPercentage(i + 1, tg_task_list.size()),
                 ") nets", stage_monitor.getStatsInfo());
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void TopologyGenerator::routeTGNet(TGModel& tg_model, TGNet* tg_net)
{
  std::vector<Segment<PlanarCoord>> routing_segment_list;
  for (Segment<PlanarCoord>& planar_topo : getPlanarTopoList(tg_model, tg_net)) {
    for (Segment<PlanarCoord>& routing_segment : getRoutingSegmentList(tg_model, planar_topo)) {
      routing_segment_list.push_back(routing_segment);
    }
  }
  MTree<LayerCoord> coord_tree = getCoordTree(tg_net, routing_segment_list);
  updateDemand(tg_model, coord_tree);
  uploadNetResult(tg_net, coord_tree);
}

std::vector<Segment<PlanarCoord>> TopologyGenerator::getPlanarTopoList(TGModel& tg_model, TGNet* tg_net)
{
  int32_t topo_spilt_length = tg_model.get_tg_parameter().get_topo_spilt_length();

  std::vector<PlanarCoord> planar_coord_list;
  {
    for (TGPin& tg_pin : tg_net->get_tg_pin_list()) {
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
  std::vector<Segment<PlanarCoord>> routing_segment_list;
  for (auto getRoutingSegmentList :
       {std::bind(&TopologyGenerator::getRoutingSegmentListByStraight, this, std::placeholders::_1, std::placeholders::_2),
        std::bind(&TopologyGenerator::getRoutingSegmentListByLPattern, this, std::placeholders::_1, std::placeholders::_2)}) {
    if (routing_segment_list.empty()) {
      routing_segment_list = getRoutingSegmentList(tg_model, planar_topo);
    }
  }
  if (routing_segment_list.empty()) {
    RTLOG.error(Loc::current(), "The routing_segment_list is empty");
  }
  return routing_segment_list;
}

std::vector<Segment<PlanarCoord>> TopologyGenerator::getRoutingSegmentListByStraight(TGModel& tg_model, Segment<PlanarCoord>& planar_topo)
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

std::vector<Segment<PlanarCoord>> TopologyGenerator::getRoutingSegmentListByLPattern(TGModel& tg_model, Segment<PlanarCoord>& planar_topo)
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
    double cost = getNodeCost(tg_model, routing_segment_list_list[i]);
    if (cost < min_cost) {
      min_cost = cost;
      min_i = i;
    }
  }
  return routing_segment_list_list[min_i];
}

double TopologyGenerator::getNodeCost(TGModel& tg_model, std::vector<Segment<PlanarCoord>>& routing_segment_list)
{
  double congestion_unit = tg_model.get_tg_parameter().get_congestion_unit();
  GridMap<TGNode>& tg_node_map = tg_model.get_tg_node_map();

  double node_cost = 0;
  for (Segment<PlanarCoord>& coord_segment : routing_segment_list) {
    PlanarCoord& first_coord = coord_segment.get_first();
    PlanarCoord& second_coord = coord_segment.get_second();

    Orientation orientation = RTUTIL.getOrientation(first_coord, second_coord);
    if (orientation == Orientation::kNone || orientation == Orientation::kOblique) {
      RTLOG.error(Loc::current(), "The orientation is error!");
    }
    Orientation opposite_orientation = RTUTIL.getOppositeOrientation(orientation);

    node_cost += tg_node_map[first_coord.get_x()][first_coord.get_y()].getCongestionCost(orientation);
    node_cost += tg_node_map[second_coord.get_x()][second_coord.get_y()].getCongestionCost(opposite_orientation);

    if (RTUTIL.isHorizontal(first_coord, second_coord)) {
      int32_t first_x = first_coord.get_x();
      int32_t second_x = second_coord.get_x();
      int32_t y = first_coord.get_y();
      RTUTIL.swapByASC(first_x, second_x);
      for (int32_t x = (first_x + 1); x <= (second_x - 1); x++) {
        node_cost += tg_node_map[x][y].getCongestionCost(orientation);
        node_cost += tg_node_map[x][y].getCongestionCost(opposite_orientation);
      }
    } else if (RTUTIL.isVertical(first_coord, second_coord)) {
      int32_t x = first_coord.get_x();
      int32_t first_y = first_coord.get_y();
      int32_t second_y = second_coord.get_y();
      RTUTIL.swapByASC(first_y, second_y);
      for (int32_t y = (first_y + 1); y <= (second_y - 1); y++) {
        node_cost += tg_node_map[x][y].getCongestionCost(orientation);
        node_cost += tg_node_map[x][y].getCongestionCost(opposite_orientation);
      }
    }
  }
  node_cost *= congestion_unit;
  return node_cost;
}

MTree<LayerCoord> TopologyGenerator::getCoordTree(TGNet* tg_net, std::vector<Segment<PlanarCoord>>& planar_routing_segment_list)
{
  std::vector<Segment<LayerCoord>> routing_segment_list;
  for (Segment<PlanarCoord>& planar_routing_segment : planar_routing_segment_list) {
    routing_segment_list.emplace_back(LayerCoord(planar_routing_segment.get_first(), 0),
                                      LayerCoord(planar_routing_segment.get_second(), 0));
  }
  std::vector<LayerCoord> candidate_root_coord_list;
  std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
  std::vector<TGPin>& tg_pin_list = tg_net->get_tg_pin_list();
  for (size_t i = 0; i < tg_pin_list.size(); i++) {
    LayerCoord coord(tg_pin_list[i].get_access_point().get_grid_coord(), 0);
    candidate_root_coord_list.push_back(coord);
    key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
  }
  return RTUTIL.getTreeByFullFlow(candidate_root_coord_list, routing_segment_list, key_coord_pin_map);
}

void TopologyGenerator::updateDemand(TGModel& tg_model, MTree<LayerCoord>& coord_tree)
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
  GridMap<TGNode>& tg_node_map = tg_model.get_tg_node_map();
  for (auto& [usage_coord, orientation_list] : usage_map) {
    TGNode& tg_node = tg_node_map[usage_coord.get_x()][usage_coord.get_y()];
    tg_node.updateDemand(orientation_list, ChangeType::kAdd);
  }
}

void TopologyGenerator::uploadNetResult(TGNet* tg_net, MTree<LayerCoord>& coord_tree)
{
  for (Segment<TNode<LayerCoord>*>& coord_segment : RTUTIL.getSegListByTree(coord_tree)) {
    Segment<LayerCoord>* segment = new Segment<LayerCoord>(coord_segment.get_first()->value(), coord_segment.get_second()->value());
    RTDM.updateGlobalNetResultToGCellMap(ChangeType::kAdd, tg_net->get_net_idx(), segment);
  }
}

#if 1  // exhibit

void TopologyGenerator::updateSummary(TGModel& tg_model)
{
  int32_t micron_dbu = RTDM.getDatabase().get_micron_dbu();
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  Summary& summary = RTDM.getDatabase().get_summary();
  int32_t enable_timing = RTDM.getConfig().enable_timing;

  int32_t& total_demand = summary.tg_summary.total_demand;
  int32_t& total_overflow = summary.tg_summary.total_overflow;
  double& total_wire_length = summary.tg_summary.total_wire_length;
  std::map<std::string, std::map<std::string, double>>& clock_timing = summary.tg_summary.clock_timing;
  std::map<std::string, double>& power_map = summary.tg_summary.power_map;

  std::vector<TGNet>& tg_net_list = tg_model.get_tg_net_list();
  GridMap<TGNode>& tg_node_map = tg_model.get_tg_node_map();

  total_demand = 0;
  total_overflow = 0;
  total_wire_length = 0;
  clock_timing.clear();
  power_map.clear();

  for (int32_t x = 0; x < tg_node_map.get_x_size(); x++) {
    for (int32_t y = 0; y < tg_node_map.get_y_size(); y++) {
      std::map<Orientation, int32_t>& orient_supply_map = tg_node_map[x][y].get_orient_supply_map();
      std::map<Orientation, int32_t>& orient_demand_map = tg_node_map[x][y].get_orient_demand_map();
      int32_t node_demand = 0;
      int32_t node_overflow = 0;
      node_demand = (orient_demand_map[Orientation::kEast] + orient_demand_map[Orientation::kWest] + orient_demand_map[Orientation::kSouth]
                     + orient_demand_map[Orientation::kNorth]);
      node_overflow = std::max(0, orient_demand_map[Orientation::kEast] - orient_supply_map[Orientation::kEast])
                      + std::max(0, orient_demand_map[Orientation::kWest] - orient_supply_map[Orientation::kWest])
                      + std::max(0, orient_demand_map[Orientation::kSouth] - orient_supply_map[Orientation::kSouth])
                      + std::max(0, orient_demand_map[Orientation::kNorth] - orient_supply_map[Orientation::kNorth]);
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
        real_pin_coord_map_list[tg_net.get_net_idx()][tg_pin.get_pin_name()].emplace_back(
            RTUTIL.getRealRectByGCell(layer_coord, gcell_axis).getMidPoint(), 0);
      }
    }
    for (auto& [net_idx, segment_set] : RTDM.getNetGlobalResultMap(die)) {
      for (Segment<LayerCoord>* segment : segment_set) {
        LayerCoord first_layer_coord = segment->get_first();
        LayerCoord first_real_coord(RTUTIL.getRealRectByGCell(first_layer_coord, gcell_axis).getMidPoint(),
                                    first_layer_coord.get_layer_idx());
        LayerCoord second_layer_coord = segment->get_second();
        LayerCoord second_real_coord(RTUTIL.getRealRectByGCell(second_layer_coord, gcell_axis).getMidPoint(),
                                     second_layer_coord.get_layer_idx());

        routing_segment_list_list[net_idx].emplace_back(first_real_coord, second_real_coord);
      }
    }
    RTI.updateTimingAndPower(real_pin_coord_map_list, routing_segment_list_list, clock_timing, power_map);
  }
}

void TopologyGenerator::printSummary(TGModel& tg_model)
{
  Summary& summary = RTDM.getDatabase().get_summary();
  int32_t enable_timing = RTDM.getConfig().enable_timing;

  int32_t& total_demand = summary.tg_summary.total_demand;
  int32_t& total_overflow = summary.tg_summary.total_overflow;
  double& total_wire_length = summary.tg_summary.total_wire_length;
  std::map<std::string, std::map<std::string, double>>& clock_timing = summary.tg_summary.clock_timing;
  std::map<std::string, double>& power_map = summary.tg_summary.power_map;

  fort::char_table summary_table;
  {
    summary_table << fort::header << "total_demand" << total_demand << fort::endr;
    summary_table << fort::header << "total_overflow" << total_overflow << fort::endr;
    summary_table << fort::header << "total_wire_length" << total_wire_length << fort::endr;
  }
  fort::char_table timing_table;
  fort::char_table power_table;
  if (enable_timing) {
    timing_table << fort::header << "clock_name"
                 << "tns"
                 << "wns"
                 << "freq" << fort::endr;
    for (auto& [clock_name, timing_map] : clock_timing) {
      timing_table << clock_name << timing_map["TNS"] << timing_map["WNS"] << timing_map["Freq(MHz)"] << fort::endr;
    }
    power_table << fort::header << "power_type" << "power_value" << fort::endr;
    for (auto& [type, power] : power_map) {
      power_table << type << power << fort::endr;
    }
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

  std::ofstream* guide_file_stream = RTUTIL.getOutputFileStream(tg_temp_directory_path + "route.guide");
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
      RTUTIL.pushStream(guide_file_stream, "pin ", grid_x, " ", grid_y, " ", real_x, " ", real_y, " ", layer, " ", connnect, " ",
                        tg_pin.get_pin_name(), "\n");
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
        RTUTIL.pushStream(guide_file_stream, "wire ", grid1_x, " ", grid1_y, " ", grid2_x, " ", grid2_y, " ", real1_x, " ", real1_y, " ",
                          real2_x, " ", real2_y, " ", layer, "\n");
      }
    }
  }
  RTUTIL.closeFileStream(guide_file_stream);
}

void TopologyGenerator::outputPlanarSupplyCSV(TGModel& tg_model)
{
  std::string& tg_temp_directory_path = RTDM.getConfig().tg_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  std::ofstream* supply_csv_file = RTUTIL.getOutputFileStream(RTUTIL.getString(tg_temp_directory_path, "supply_map_planar.csv"));
  GridMap<TGNode>& tg_node_map = tg_model.get_tg_node_map();
  for (int32_t y = tg_node_map.get_y_size() - 1; y >= 0; y--) {
    for (int32_t x = 0; x < tg_node_map.get_x_size(); x++) {
      std::map<Orientation, int32_t>& orient_supply_map = tg_node_map[x][y].get_orient_supply_map();
      int32_t total_supply = 0;
      total_supply = (orient_supply_map[Orientation::kEast] + orient_supply_map[Orientation::kWest] + orient_supply_map[Orientation::kSouth]
                      + orient_supply_map[Orientation::kNorth]);
      RTUTIL.pushStream(supply_csv_file, total_supply, ",");
    }
    RTUTIL.pushStream(supply_csv_file, "\n");
  }
  RTUTIL.closeFileStream(supply_csv_file);
}

void TopologyGenerator::outputPlanarDemandCSV(TGModel& tg_model)
{
  std::string& tg_temp_directory_path = RTDM.getConfig().tg_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  std::ofstream* demand_csv_file = RTUTIL.getOutputFileStream(RTUTIL.getString(tg_temp_directory_path, "demand_map_planar.csv"));
  GridMap<TGNode>& tg_node_map = tg_model.get_tg_node_map();
  for (int32_t y = tg_node_map.get_y_size() - 1; y >= 0; y--) {
    for (int32_t x = 0; x < tg_node_map.get_x_size(); x++) {
      std::map<Orientation, int32_t>& orient_demand_map = tg_node_map[x][y].get_orient_demand_map();
      int32_t total_demand = 0;
      total_demand = (orient_demand_map[Orientation::kEast] + orient_demand_map[Orientation::kWest] + orient_demand_map[Orientation::kSouth]
                      + orient_demand_map[Orientation::kNorth]);
      RTUTIL.pushStream(demand_csv_file, total_demand, ",");
    }
    RTUTIL.pushStream(demand_csv_file, "\n");
  }
  RTUTIL.closeFileStream(demand_csv_file);
}

void TopologyGenerator::outputPlanarOverflowCSV(TGModel& tg_model)
{
  std::string& tg_temp_directory_path = RTDM.getConfig().tg_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  std::ofstream* overflow_csv_file = RTUTIL.getOutputFileStream(RTUTIL.getString(tg_temp_directory_path, "overflow_map_planar.csv"));
  GridMap<TGNode>& tg_node_map = tg_model.get_tg_node_map();
  for (int32_t y = tg_node_map.get_y_size() - 1; y >= 0; y--) {
    for (int32_t x = 0; x < tg_node_map.get_x_size(); x++) {
      std::map<Orientation, int32_t>& orient_supply_map = tg_node_map[x][y].get_orient_supply_map();
      std::map<Orientation, int32_t>& orient_demand_map = tg_node_map[x][y].get_orient_demand_map();
      int32_t total_overflow = 0;
      total_overflow = std::max(0, orient_demand_map[Orientation::kEast] - orient_supply_map[Orientation::kEast])
                       + std::max(0, orient_demand_map[Orientation::kWest] - orient_supply_map[Orientation::kWest])
                       + std::max(0, orient_demand_map[Orientation::kSouth] - orient_supply_map[Orientation::kSouth])
                       + std::max(0, orient_demand_map[Orientation::kNorth] - orient_supply_map[Orientation::kNorth]);
      RTUTIL.pushStream(overflow_csv_file, total_overflow, ",");
    }
    RTUTIL.pushStream(overflow_csv_file, "\n");
  }
  RTUTIL.closeFileStream(overflow_csv_file);
}

#endif

#if 1  // debug

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
