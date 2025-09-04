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
#include "LayerAssigner.hpp"

#include "GDSPlotter.hpp"
#include "Monitor.hpp"
#include "RTInterface.hpp"
#include "Utility.hpp"

namespace irt {

// public

void LayerAssigner::initInst()
{
  if (_la_instance == nullptr) {
    _la_instance = new LayerAssigner();
  }
}

LayerAssigner& LayerAssigner::getInst()
{
  if (_la_instance == nullptr) {
    RTLOG.error(Loc::current(), "The instance not initialized!");
  }
  return *_la_instance;
}

void LayerAssigner::destroyInst()
{
  if (_la_instance != nullptr) {
    delete _la_instance;
    _la_instance = nullptr;
  }
}

// function

void LayerAssigner::assign()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");
  LAModel la_model = initLAModel();
  setLAComParam(la_model);
  initLATaskList(la_model);
  buildLayerNodeMap(la_model);
  buildLANodeNeighbor(la_model);
  buildOrientSupply(la_model);
  // debugCheckLAModel(la_model);
  buildPlaneTree(la_model);
  routeLAModel(la_model);
  // debugPlotLAModel(la_model, "after");
  updateSummary(la_model);
  printSummary(la_model);
  outputGuide(la_model);
  outputNetCSV(la_model);
  outputOverflowCSV(la_model);
  outputJson(la_model);
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

// private

LayerAssigner* LayerAssigner::_la_instance = nullptr;

LAModel LayerAssigner::initLAModel()
{
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();

  LAModel la_model;
  la_model.set_la_net_list(convertToLANetList(net_list));
  return la_model;
}

std::vector<LANet> LayerAssigner::convertToLANetList(std::vector<Net>& net_list)
{
  std::vector<LANet> la_net_list;
  la_net_list.reserve(net_list.size());
  for (size_t i = 0; i < net_list.size(); i++) {
    la_net_list.emplace_back(convertToLANet(net_list[i]));
  }
  return la_net_list;
}

LANet LayerAssigner::convertToLANet(Net& net)
{
  LANet la_net;
  la_net.set_origin_net(&net);
  la_net.set_net_idx(net.get_net_idx());
  la_net.set_connect_type(net.get_connect_type());
  for (Pin& pin : net.get_pin_list()) {
    la_net.get_la_pin_list().push_back(LAPin(pin));
  }
  la_net.set_bounding_box(net.get_bounding_box());
  return la_net;
}

void LayerAssigner::setLAComParam(LAModel& la_model)
{
  int32_t topo_spilt_length = 1;
  double prefer_wire_unit = 1;
  double non_prefer_wire_unit = 2.5 * prefer_wire_unit;
  double via_unit = 2 * non_prefer_wire_unit;
  double overflow_unit = 4 * non_prefer_wire_unit;
  /**
   * topo_spilt_length, via_unit, overflow_unit
   */
  // clang-format off
  LAComParam la_com_param(topo_spilt_length, via_unit, overflow_unit);
  // clang-format on
  RTLOG.info(Loc::current(), "topo_spilt_length: ", la_com_param.get_topo_spilt_length());
  RTLOG.info(Loc::current(), "via_unit: ", la_com_param.get_via_unit());
  RTLOG.info(Loc::current(), "overflow_unit: ", la_com_param.get_overflow_unit());
  la_model.set_la_com_param(la_com_param);
}

void LayerAssigner::initLATaskList(LAModel& la_model)
{
  std::vector<LANet>& la_net_list = la_model.get_la_net_list();
  std::vector<LANet*>& la_task_list = la_model.get_la_task_list();
  la_task_list.reserve(la_net_list.size());
  for (LANet& la_net : la_net_list) {
    la_task_list.push_back(&la_net);
  }
  std::sort(la_task_list.begin(), la_task_list.end(), CmpLANet());
}

void LayerAssigner::buildLayerNodeMap(LAModel& la_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  std::vector<GridMap<LANode>>& layer_node_map = la_model.get_layer_node_map();
  layer_node_map.resize(routing_layer_list.size());
#pragma omp parallel for
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    GridMap<LANode>& la_node_map = layer_node_map[layer_idx];
    la_node_map.init(gcell_map.get_x_size(), gcell_map.get_y_size());
    for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
      for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
        LANode& la_node = la_node_map[x][y];
        la_node.set_coord(x, y);
        la_node.set_layer_idx(layer_idx);
        la_node.set_boundary_wire_unit(gcell_map[x][y].get_boundary_wire_unit());
        la_node.set_internal_wire_unit(gcell_map[x][y].get_internal_wire_unit());
        la_node.set_internal_via_unit(gcell_map[x][y].get_internal_via_unit());
        if (RTUTIL.exist(gcell_map[x][y].get_routing_ignore_net_orient_map(), layer_idx)) {
          la_node.set_ignore_net_orient_map(gcell_map[x][y].get_routing_ignore_net_orient_map()[layer_idx]);
        }
      }
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void LayerAssigner::buildLANodeNeighbor(LAModel& la_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  int32_t bottom_routing_layer_idx = RTDM.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = RTDM.getConfig().top_routing_layer_idx;

  std::vector<GridMap<LANode>>& layer_node_map = la_model.get_layer_node_map();

#pragma omp parallel for
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    bool routing_h = routing_layer_list[layer_idx].isPreferH();
    bool routing_v = !routing_h;
    if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
      routing_h = false;
      routing_v = false;
    }
    GridMap<LANode>& la_node_map = layer_node_map[layer_idx];
    for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
      for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
        std::map<Orientation, LANode*>& neighbor_node_map = la_node_map[x][y].get_neighbor_node_map();
        if (routing_h) {
          if (x != 0) {
            neighbor_node_map[Orientation::kWest] = &la_node_map[x - 1][y];
          }
          if (x != (la_node_map.get_x_size() - 1)) {
            neighbor_node_map[Orientation::kEast] = &la_node_map[x + 1][y];
          }
        }
        if (routing_v) {
          if (y != 0) {
            neighbor_node_map[Orientation::kSouth] = &la_node_map[x][y - 1];
          }
          if (y != (la_node_map.get_y_size() - 1)) {
            neighbor_node_map[Orientation::kNorth] = &la_node_map[x][y + 1];
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

void LayerAssigner::buildOrientSupply(LAModel& la_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();

  std::vector<GridMap<LANode>>& layer_node_map = la_model.get_layer_node_map();

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

void LayerAssigner::buildPlaneTree(LAModel& la_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();

  std::vector<LANet>& la_net_list = la_model.get_la_net_list();

  for (auto& [net_idx, segment_set] : RTDM.getNetGlobalResultMap(die)) {
    LANet& la_net = la_net_list[net_idx];

    std::vector<Segment<LayerCoord>> routing_segment_list;
    for (Segment<LayerCoord>* segment : segment_set) {
      routing_segment_list.push_back(*segment);
    }
    std::vector<LayerCoord> candidate_root_coord_list;
    std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
    std::vector<LAPin>& la_pin_list = la_net.get_la_pin_list();
    for (size_t i = 0; i < la_pin_list.size(); i++) {
      LayerCoord coord(la_pin_list[i].get_access_point().get_grid_coord(), 0);
      candidate_root_coord_list.push_back(coord);
      key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
    }
    la_net.set_planar_tree(RTUTIL.getTreeByFullFlow(candidate_root_coord_list, routing_segment_list, key_coord_pin_map));
  }
  for (auto& [net_idx, segment_set] : RTDM.getNetGlobalResultMap(die)) {
    for (Segment<LayerCoord>* segment : segment_set) {
      RTDM.updateNetGlobalResultToGCellMap(ChangeType::kDel, net_idx, segment);
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void LayerAssigner::routeLAModel(LAModel& la_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<LANet*>& la_task_list = la_model.get_la_task_list();

  int32_t batch_size = RTUTIL.getBatchSize(la_task_list.size());

  Monitor stage_monitor;
  for (size_t i = 0; i < la_task_list.size(); i++) {
    routeLATask(la_model, la_task_list[i]);
    if ((i + 1) % batch_size == 0 || (i + 1) == la_task_list.size()) {
      RTLOG.info(Loc::current(), "Routed ", (i + 1), "/", la_task_list.size(), "(", RTUTIL.getPercentage(i + 1, la_task_list.size()), ") nets",
                 stage_monitor.getStatsInfo());
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void LayerAssigner::routeLATask(LAModel& la_model, LANet* la_task)
{
  initSingleTask(la_model, la_task);
  if (needRouting(la_model)) {
    spiltPlaneTree(la_model);
    buildPillarTree(la_model);
    assignPillarTree(la_model);
    buildLayerTree(la_model);
  }
  resetSingleTask(la_model);
}

void LayerAssigner::initSingleTask(LAModel& la_model, LANet* la_task)
{
  la_model.set_curr_la_task(la_task);
}

bool LayerAssigner::needRouting(LAModel& la_model)
{
  return (la_model.get_curr_la_task()->get_planar_tree().get_root() != nullptr);
}

void LayerAssigner::spiltPlaneTree(LAModel& la_model)
{
  int32_t topo_spilt_length = la_model.get_la_com_param().get_topo_spilt_length();

  TNode<LayerCoord>* planar_tree_root = la_model.get_curr_la_task()->get_planar_tree().get_root();
  std::queue<TNode<LayerCoord>*> planar_queue = RTUTIL.initQueue(planar_tree_root);
  while (!planar_queue.empty()) {
    TNode<LayerCoord>* planar_node = RTUTIL.getFrontAndPop(planar_queue);
    std::vector<TNode<LayerCoord>*> child_list = planar_node->get_child_list();
    for (size_t i = 0; i < child_list.size(); i++) {
      int32_t length = RTUTIL.getManhattanDistance(planar_node->value().get_planar_coord(), child_list[i]->value().get_planar_coord());
      if (length <= topo_spilt_length) {
        continue;
      }
      insertMidPoint(la_model, planar_node, child_list[i]);
    }
    RTUTIL.addListToQueue(planar_queue, child_list);
  }
}

void LayerAssigner::insertMidPoint(LAModel& la_model, TNode<LayerCoord>* planar_node, TNode<LayerCoord>* child_node)
{
  int32_t topo_spilt_length = la_model.get_la_com_param().get_topo_spilt_length();

  PlanarCoord& parent_coord = planar_node->value().get_planar_coord();
  PlanarCoord& child_coord = child_node->value().get_planar_coord();
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
  TNode<LayerCoord>* curr_node = planar_node;
  for (size_t i = 0; i < mid_coord_list.size(); i++) {
    LayerCoord mid_coord(mid_coord_list[i], 0);
    TNode<LayerCoord>* mid_node = new TNode<LayerCoord>(mid_coord);
    curr_node->addChild(mid_node);
    curr_node = mid_node;
  }
  curr_node->addChild(child_node);
}

void LayerAssigner::buildPillarTree(LAModel& la_model)
{
  LANet* curr_la_task = la_model.get_curr_la_task();

  std::map<PlanarCoord, std::set<int32_t>, CmpPlanarCoordByXASC> coord_pin_layer_map;
  for (LAPin& la_pin : curr_la_task->get_la_pin_list()) {
    AccessPoint& access_point = la_pin.get_access_point();
    coord_pin_layer_map[access_point.get_grid_coord()].insert(access_point.get_layer_idx());
  }
  std::function<LAPillar(LayerCoord&, std::map<PlanarCoord, std::set<int32_t>, CmpPlanarCoordByXASC>&)> convert;
  convert = std::bind(&LayerAssigner::convertLAPillar, this, std::placeholders::_1, std::placeholders::_2);
  curr_la_task->set_pillar_tree(RTUTIL.convertTree(curr_la_task->get_planar_tree(), convert, coord_pin_layer_map));
}

LAPillar LayerAssigner::convertLAPillar(LayerCoord& layer_coord, std::map<PlanarCoord, std::set<int32_t>, CmpPlanarCoordByXASC>& coord_pin_layer_map)
{
  LAPillar la_pillar;
  la_pillar.set_planar_coord(layer_coord.get_planar_coord());
  la_pillar.set_pin_layer_idx_set(coord_pin_layer_map[layer_coord.get_planar_coord()]);
  return la_pillar;
}

void LayerAssigner::assignPillarTree(LAModel& la_model)
{
  assignForward(la_model);
  assignBackward(la_model);
}

void LayerAssigner::assignForward(LAModel& la_model)
{
  TNode<LAPillar>* pillar_tree_root = la_model.get_curr_la_task()->get_pillar_tree().get_root();

  LAPackage la_package(pillar_tree_root, pillar_tree_root);
  for (int32_t candidate_layer_idx : getCandidateLayerList(la_model, la_package)) {
    std::set<int32_t>& pin_layer_idx_set = pillar_tree_root->value().get_pin_layer_idx_set();
    LALayerCost layer_cost;
    layer_cost.set_parent_layer_idx(candidate_layer_idx);
    layer_cost.set_layer_idx(candidate_layer_idx);
    layer_cost.set_history_cost(getFullViaCost(la_model, pin_layer_idx_set, candidate_layer_idx));
    pillar_tree_root->value().get_layer_cost_list().push_back(std::move(layer_cost));
  }
  std::queue<TNode<LAPillar>*> pillar_node_queue = RTUTIL.initQueue(pillar_tree_root);
  while (!pillar_node_queue.empty()) {
    TNode<LAPillar>* parent_pillar_node = RTUTIL.getFrontAndPop(pillar_node_queue);
    std::vector<TNode<LAPillar>*>& child_list = parent_pillar_node->get_child_list();
    for (size_t i = 0; i < child_list.size(); i++) {
      LAPackage la_package(parent_pillar_node, child_list[i]);
      buildLayerCost(la_model, la_package);
    }
    RTUTIL.addListToQueue(pillar_node_queue, child_list);
  }
}

std::vector<int32_t> LayerAssigner::getCandidateLayerList(LAModel& la_model, LAPackage& la_package)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  int32_t bottom_routing_layer_idx = RTDM.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = RTDM.getConfig().top_routing_layer_idx;

  Direction direction = RTUTIL.getDirection(la_package.getParentPillar().get_planar_coord(), la_package.getChildPillar().get_planar_coord());

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

double LayerAssigner::getFullViaCost(LAModel& la_model, std::set<int32_t>& layer_idx_set, int32_t candidate_layer_idx)
{
  double via_unit = la_model.get_la_com_param().get_via_unit();

  int32_t via_num = 0;
  if (layer_idx_set.size() > 0) {
    std::set<int32_t> layer_idx_set_temp = layer_idx_set;
    layer_idx_set_temp.insert(candidate_layer_idx);
    via_num = std::abs(*layer_idx_set_temp.begin() - *layer_idx_set_temp.rbegin());
  }
  return (via_unit * via_num);
}

void LayerAssigner::buildLayerCost(LAModel& la_model, LAPackage& la_package)
{
  std::vector<LALayerCost>& layer_cost_list = la_package.getChildPillar().get_layer_cost_list();

  for (int32_t candidate_layer_idx : getCandidateLayerList(la_model, la_package)) {
    std::pair<int32_t, double> parent_pillar_cost_pair = getParentPillarCost(la_model, la_package, candidate_layer_idx);
    double segment_cost = getSegmentCost(la_model, la_package, candidate_layer_idx);
    double child_pillar_cost = getChildPillarCost(la_model, la_package, candidate_layer_idx);

    LALayerCost layer_cost;
    layer_cost.set_parent_layer_idx(parent_pillar_cost_pair.first);
    layer_cost.set_layer_idx(candidate_layer_idx);
    layer_cost.set_history_cost(parent_pillar_cost_pair.second + segment_cost + child_pillar_cost);
    layer_cost_list.push_back(std::move(layer_cost));
  }
}

std::pair<int32_t, double> LayerAssigner::getParentPillarCost(LAModel& la_model, LAPackage& la_package, int32_t candidate_layer_idx)
{
  LAPillar& parent_pillar = la_package.getParentPillar();

  std::pair<int32_t, double> layer_cost_pair;
  double min_cost = DBL_MAX;
  for (LALayerCost& layer_cost : parent_pillar.get_layer_cost_list()) {
    std::set<int32_t> layer_idx_set_temp = parent_pillar.get_pin_layer_idx_set();
    layer_idx_set_temp.insert(layer_cost.get_layer_idx());
    double curr_cost = layer_cost.get_history_cost() + getExtraViaCost(la_model, layer_idx_set_temp, candidate_layer_idx);

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

double LayerAssigner::getExtraViaCost(LAModel& la_model, std::set<int32_t>& layer_idx_set, int32_t candidate_layer_idx)
{
  double via_unit = la_model.get_la_com_param().get_via_unit();

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

double LayerAssigner::getSegmentCost(LAModel& la_model, LAPackage& la_package, int32_t candidate_layer_idx)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<GridMap<LANode>>& layer_node_map = la_model.get_layer_node_map();
  double overflow_unit = la_model.get_la_com_param().get_overflow_unit();

  Direction prefer_direction = routing_layer_list[candidate_layer_idx].get_prefer_direction();

  PlanarCoord first_coord = la_package.getParentPillar().get_planar_coord();
  PlanarCoord second_coord = la_package.getChildPillar().get_planar_coord();
  int32_t first_x = first_coord.get_x();
  int32_t first_y = first_coord.get_y();
  int32_t second_x = second_coord.get_x();
  int32_t second_y = second_coord.get_y();
  RTUTIL.swapByASC(first_x, second_x);
  RTUTIL.swapByASC(first_y, second_y);

  double node_cost = 0;
  for (int32_t x = first_x; x <= second_x; x++) {
    for (int32_t y = first_y; y <= second_y; y++) {
      node_cost += layer_node_map[candidate_layer_idx][x][y].getOverflowCost(la_model.get_curr_la_task()->get_net_idx(), prefer_direction, overflow_unit);
    }
  }
  return node_cost;
}

double LayerAssigner::getChildPillarCost(LAModel& la_model, LAPackage& la_package, int32_t candidate_layer_idx)
{
  LAPillar& child_pillar = la_package.getChildPillar();
  return getFullViaCost(la_model, child_pillar.get_pin_layer_idx_set(), candidate_layer_idx);
}

void LayerAssigner::assignBackward(LAModel& la_model)
{
  std::vector<std::vector<TNode<LAPillar>*>> level_list = RTUTIL.getLevelOrder(la_model.get_curr_la_task()->get_pillar_tree());
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

int32_t LayerAssigner::getBestLayerBySelf(TNode<LAPillar>* pillar_node)
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

int32_t LayerAssigner::getBestLayerByChild(TNode<LAPillar>* parent_pillar_node)
{
  std::set<int32_t> candidate_layer_idx_set;
  for (TNode<LAPillar>* child_node : parent_pillar_node->get_child_list()) {
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

void LayerAssigner::buildLayerTree(LAModel& la_model)
{
  std::vector<Segment<LayerCoord>> routing_segment_list = getRoutingSegmentList(la_model);
  MTree<LayerCoord> coord_tree = getCoordTree(la_model, routing_segment_list);
  updateDemandToGraph(la_model, ChangeType::kAdd, coord_tree);
  uploadNetResult(la_model, coord_tree);
}

std::vector<Segment<LayerCoord>> LayerAssigner::getRoutingSegmentList(LAModel& la_model)
{
  std::vector<Segment<LayerCoord>> routing_segment_list;

  std::queue<TNode<LAPillar>*> pillar_node_queue = RTUTIL.initQueue(la_model.get_curr_la_task()->get_pillar_tree().get_root());
  while (!pillar_node_queue.empty()) {
    TNode<LAPillar>* parent_pillar_node = RTUTIL.getFrontAndPop(pillar_node_queue);
    std::vector<TNode<LAPillar>*>& child_list = parent_pillar_node->get_child_list();
    {
      std::set<int32_t> layer_idx_set = parent_pillar_node->value().get_pin_layer_idx_set();
      layer_idx_set.insert(parent_pillar_node->value().get_layer_idx());
      for (TNode<LAPillar>* child_node : child_list) {
        layer_idx_set.insert(child_node->value().get_layer_idx());
      }
      routing_segment_list.emplace_back(LayerCoord(parent_pillar_node->value().get_planar_coord(), *layer_idx_set.begin()),
                                        LayerCoord(parent_pillar_node->value().get_planar_coord(), *layer_idx_set.rbegin()));
    }
    for (TNode<LAPillar>* child_node : child_list) {
      routing_segment_list.emplace_back(LayerCoord(parent_pillar_node->value().get_planar_coord(), child_node->value().get_layer_idx()),
                                        LayerCoord(child_node->value().get_planar_coord(), child_node->value().get_layer_idx()));
    }
    RTUTIL.addListToQueue(pillar_node_queue, child_list);
  }
  return routing_segment_list;
}

MTree<LayerCoord> LayerAssigner::getCoordTree(LAModel& la_model, std::vector<Segment<LayerCoord>>& routing_segment_list)
{
  std::vector<LayerCoord> candidate_root_coord_list;
  std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
  std::vector<LAPin>& la_pin_list = la_model.get_curr_la_task()->get_la_pin_list();
  for (size_t i = 0; i < la_pin_list.size(); i++) {
    LayerCoord coord = la_pin_list[i].get_access_point().getGridLayerCoord();
    candidate_root_coord_list.push_back(coord);
    key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
  }
  return RTUTIL.getTreeByFullFlow(candidate_root_coord_list, routing_segment_list, key_coord_pin_map);
}

void LayerAssigner::uploadNetResult(LAModel& la_model, MTree<LayerCoord>& coord_tree)
{
  for (Segment<TNode<LayerCoord>*>& coord_segment : RTUTIL.getSegListByTree(coord_tree)) {
    Segment<LayerCoord>* segment = new Segment<LayerCoord>(coord_segment.get_first()->value(), coord_segment.get_second()->value());
    RTDM.updateNetGlobalResultToGCellMap(ChangeType::kAdd, la_model.get_curr_la_task()->get_net_idx(), segment);
  }
}

void LayerAssigner::resetSingleTask(LAModel& la_model)
{
  la_model.set_curr_la_task(nullptr);
}

#if 1  // update env

void LayerAssigner::updateDemandToGraph(LAModel& la_model, ChangeType change_type, MTree<LayerCoord>& coord_tree)
{
  int32_t curr_net_idx = la_model.get_curr_la_task()->get_net_idx();

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
  std::vector<GridMap<LANode>>& layer_node_map = la_model.get_layer_node_map();
  for (auto& [usage_coord, orientation_list] : usage_map) {
    LANode& la_node = layer_node_map[usage_coord.get_layer_idx()][usage_coord.get_x()][usage_coord.get_y()];
    la_node.updateDemand(curr_net_idx, orientation_list, change_type);
  }
}

#endif

#if 1  // exhibit

void LayerAssigner::updateSummary(LAModel& la_model)
{
  int32_t micron_dbu = RTDM.getDatabase().get_micron_dbu();
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();
  Summary& summary = RTDM.getDatabase().get_summary();
  int32_t enable_timing = RTDM.getConfig().enable_timing;

  std::map<int32_t, double>& routing_demand_map = summary.la_summary.routing_demand_map;
  double& total_demand = summary.la_summary.total_demand;
  std::map<int32_t, double>& routing_overflow_map = summary.la_summary.routing_overflow_map;
  double& total_overflow = summary.la_summary.total_overflow;
  std::map<int32_t, double>& routing_wire_length_map = summary.la_summary.routing_wire_length_map;
  double& total_wire_length = summary.la_summary.total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.la_summary.cut_via_num_map;
  int32_t& total_via_num = summary.la_summary.total_via_num;
  std::map<std::string, std::map<std::string, double>>& clock_timing_map = summary.la_summary.clock_timing_map;
  std::map<std::string, double>& type_power_map = summary.la_summary.type_power_map;

  std::vector<GridMap<LANode>>& layer_node_map = la_model.get_layer_node_map();
  std::vector<LANet>& la_net_list = la_model.get_la_net_list();

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
    GridMap<LANode>& la_node_map = layer_node_map[layer_idx];
    for (int32_t x = 0; x < la_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < la_node_map.get_y_size(); y++) {
        double node_demand = la_node_map[x][y].getDemand();
        double node_overflow = la_node_map[x][y].getOverflow();
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
    real_pin_coord_map_list.resize(la_net_list.size());
    std::vector<std::vector<Segment<LayerCoord>>> routing_segment_list_list;
    routing_segment_list_list.resize(la_net_list.size());
    for (LANet& la_net : la_net_list) {
      for (LAPin& la_pin : la_net.get_la_pin_list()) {
        LayerCoord layer_coord = la_pin.get_access_point().getGridLayerCoord();
        real_pin_coord_map_list[la_net.get_net_idx()][la_pin.get_pin_name()].emplace_back(RTUTIL.getRealRectByGCell(layer_coord, gcell_axis).getMidPoint(),
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
    RTI.updateTimingAndPower(real_pin_coord_map_list, routing_segment_list_list, clock_timing_map, type_power_map);
  }
}

void LayerAssigner::printSummary(LAModel& la_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  Summary& summary = RTDM.getDatabase().get_summary();
  int32_t enable_timing = RTDM.getConfig().enable_timing;

  std::map<int32_t, double>& routing_demand_map = summary.la_summary.routing_demand_map;
  double& total_demand = summary.la_summary.total_demand;
  std::map<int32_t, double>& routing_overflow_map = summary.la_summary.routing_overflow_map;
  double& total_overflow = summary.la_summary.total_overflow;
  std::map<int32_t, double>& routing_wire_length_map = summary.la_summary.routing_wire_length_map;
  double& total_wire_length = summary.la_summary.total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.la_summary.cut_via_num_map;
  int32_t& total_via_num = summary.la_summary.total_via_num;
  std::map<std::string, std::map<std::string, double>>& clock_timing_map = summary.la_summary.clock_timing_map;
  std::map<std::string, double>& type_power_map = summary.la_summary.type_power_map;

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

void LayerAssigner::outputGuide(LAModel& la_model)
{
  int32_t micron_dbu = RTDM.getDatabase().get_micron_dbu();
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& la_temp_directory_path = RTDM.getConfig().la_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  std::vector<LANet>& la_net_list = la_model.get_la_net_list();

  std::ofstream* guide_file_stream = RTUTIL.getOutputFileStream(RTUTIL.getString(la_temp_directory_path, "route.guide"));
  if (guide_file_stream == nullptr) {
    return;
  }
  RTUTIL.pushStream(guide_file_stream, "guide net_name\n");
  RTUTIL.pushStream(guide_file_stream, "pin grid_x grid_y real_x real_y layer energy name\n");
  RTUTIL.pushStream(guide_file_stream, "wire grid1_x grid1_y grid2_x grid2_y real1_x real1_y real2_x real2_y layer\n");
  RTUTIL.pushStream(guide_file_stream, "via grid_x grid_y real_x real_y layer1 layer2\n");

  for (auto& [net_idx, segment_set] : RTDM.getNetGlobalResultMap(die)) {
    LANet& la_net = la_net_list[net_idx];
    RTUTIL.pushStream(guide_file_stream, "guide ", la_net.get_origin_net()->get_net_name(), "\n");

    for (LAPin& la_pin : la_net.get_la_pin_list()) {
      AccessPoint& access_point = la_pin.get_access_point();
      double grid_x = access_point.get_grid_x();
      double grid_y = access_point.get_grid_y();
      double real_x = access_point.get_real_x() / 1.0 / micron_dbu;
      double real_y = access_point.get_real_y() / 1.0 / micron_dbu;
      std::string layer = routing_layer_list[access_point.get_layer_idx()].get_layer_name();
      std::string connnect;
      if (la_pin.get_is_driven()) {
        connnect = "driven";
      } else {
        connnect = "load";
      }
      RTUTIL.pushStream(guide_file_stream, "pin ", grid_x, " ", grid_y, " ", real_x, " ", real_y, " ", layer, " ", connnect, " ", la_pin.get_pin_name(), "\n");
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
  RTLOG.info(Loc::current(), "The csv file has been saved");
}

void LayerAssigner::outputNetCSV(LAModel& la_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& la_temp_directory_path = RTDM.getConfig().la_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  std::vector<GridMap<LANode>>& layer_node_map = la_model.get_layer_node_map();
  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::ofstream* net_csv_file = RTUTIL.getOutputFileStream(RTUTIL.getString(la_temp_directory_path, "net_map_", routing_layer.get_layer_name(), ".csv"));
    GridMap<LANode>& la_node_map = layer_node_map[routing_layer.get_layer_idx()];
    for (int32_t y = la_node_map.get_y_size() - 1; y >= 0; y--) {
      for (int32_t x = 0; x < la_node_map.get_x_size(); x++) {
        RTUTIL.pushStream(net_csv_file, la_node_map[x][y].getDemand(), ",");
      }
      RTUTIL.pushStream(net_csv_file, "\n");
    }
    RTUTIL.closeFileStream(net_csv_file);
  }
  RTLOG.info(Loc::current(), "The csv file has been saved");
}

void LayerAssigner::outputOverflowCSV(LAModel& la_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& la_temp_directory_path = RTDM.getConfig().la_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  std::vector<GridMap<LANode>>& layer_node_map = la_model.get_layer_node_map();
  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::ofstream* overflow_csv_file
        = RTUTIL.getOutputFileStream(RTUTIL.getString(la_temp_directory_path, "overflow_map_", routing_layer.get_layer_name(), ".csv"));

    GridMap<LANode>& la_node_map = layer_node_map[routing_layer.get_layer_idx()];
    for (int32_t y = la_node_map.get_y_size() - 1; y >= 0; y--) {
      for (int32_t x = 0; x < la_node_map.get_x_size(); x++) {
        RTUTIL.pushStream(overflow_csv_file, la_node_map[x][y].getOverflow(), ",");
      }
      RTUTIL.pushStream(overflow_csv_file, "\n");
    }
    RTUTIL.closeFileStream(overflow_csv_file);
  }
  RTLOG.info(Loc::current(), "The csv file has been saved");
}

void LayerAssigner::outputJson(LAModel& la_model)
{
  int32_t enable_notification = RTDM.getConfig().enable_notification;
  if (!enable_notification) {
    return;
  }
  std::map<std::string, std::string> json_path_map;
  json_path_map["net_map"] = outputNetJson(la_model);
  json_path_map["overflow_map"] = outputOverflowJson(la_model);
  json_path_map["summary"] = outputSummaryJson(la_model);
  RTI.sendNotification("LA", 1, json_path_map);
}

std::string LayerAssigner::outputNetJson(LAModel& la_model)
{
  Die& die = RTDM.getDatabase().get_die();
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();
  std::string& la_temp_directory_path = RTDM.getConfig().la_temp_directory_path;

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
  std::string net_json_file_path = RTUTIL.getString(la_temp_directory_path, "net_map.json");
  std::ofstream* net_json_file = RTUTIL.getOutputFileStream(net_json_file_path);
  (*net_json_file) << net_json_list;
  RTUTIL.closeFileStream(net_json_file);
  return net_json_file_path;
}

std::string LayerAssigner::outputOverflowJson(LAModel& la_model)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& la_temp_directory_path = RTDM.getConfig().la_temp_directory_path;

  std::vector<GridMap<LANode>>& layer_node_map = la_model.get_layer_node_map();
  std::vector<nlohmann::json> overflow_json_list;
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    GridMap<LANode>& la_node_map = layer_node_map[layer_idx];
    for (int32_t x = 0; x < la_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < la_node_map.get_y_size(); y++) {
        PlanarRect gcell = RTUTIL.getRealRectByGCell(PlanarCoord(x, y), gcell_axis);
        overflow_json_list.push_back({gcell.get_ll_x(), gcell.get_ll_y(), gcell.get_ur_x(), gcell.get_ur_y(), routing_layer_list[layer_idx].get_layer_name(),
                                      la_node_map[x][y].getOverflow()});
      }
    }
  }
  std::string overflow_json_file_path = RTUTIL.getString(la_temp_directory_path, "overflow_map.json");
  std::ofstream* overflow_json_file = RTUTIL.getOutputFileStream(overflow_json_file_path);
  (*overflow_json_file) << overflow_json_list;
  RTUTIL.closeFileStream(overflow_json_file);
  return overflow_json_file_path;
}

std::string LayerAssigner::outputSummaryJson(LAModel& la_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  Summary& summary = RTDM.getDatabase().get_summary();
  std::string& la_temp_directory_path = RTDM.getConfig().la_temp_directory_path;

  std::map<int32_t, double>& routing_demand_map = summary.la_summary.routing_demand_map;
  double& total_demand = summary.la_summary.total_demand;
  std::map<int32_t, double>& routing_overflow_map = summary.la_summary.routing_overflow_map;
  double& total_overflow = summary.la_summary.total_overflow;
  std::map<int32_t, double>& routing_wire_length_map = summary.la_summary.routing_wire_length_map;
  double& total_wire_length = summary.la_summary.total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.la_summary.cut_via_num_map;
  int32_t& total_via_num = summary.la_summary.total_via_num;
  std::map<std::string, std::map<std::string, double>>& clock_timing_map = summary.la_summary.clock_timing_map;
  std::map<std::string, double>& type_power_map = summary.la_summary.type_power_map;

  nlohmann::json summary_json;
  for (auto& [routing_layer_idx, demand] : routing_demand_map) {
    summary_json["routing_demand_map"][routing_layer_list[routing_layer_idx].get_layer_name()] = demand;
  }
  summary_json["total_demand"] = total_demand;
  for (auto& [routing_layer_idx, overflow] : routing_overflow_map) {
    summary_json["routing_overflow_map"][routing_layer_list[routing_layer_idx].get_layer_name()] = overflow;
  }
  summary_json["total_overflow"] = total_overflow;
  for (auto& [routing_layer_idx, wire_length] : routing_wire_length_map) {
    summary_json["routing_wire_length_map"][routing_layer_list[routing_layer_idx].get_layer_name()] = wire_length;
  }
  summary_json["total_wire_length"] = total_wire_length;
  for (auto& [cut_layer_idx, via_num] : cut_via_num_map) {
    summary_json["cut_via_num_map"][cut_layer_list[cut_layer_idx].get_layer_name()] = via_num;
  }
  summary_json["total_via_num"] = total_via_num;
  for (auto& [clock_name, timing] : clock_timing_map) {
    summary_json["clock_timing_map"]["clock_name"] = clock_name;
    summary_json["clock_timing_map"]["timing"] = timing;
  }
  for (auto& [type, power] : type_power_map) {
    summary_json["type_power_map"]["type"] = type;
    summary_json["type_power_map"]["power"] = power;
  }
  std::string summary_json_file_path = RTUTIL.getString(la_temp_directory_path, "summary.json");
  std::ofstream* summary_json_file = RTUTIL.getOutputFileStream(summary_json_file_path);
  (*summary_json_file) << summary_json;
  RTUTIL.closeFileStream(summary_json_file);
  return summary_json_file_path;
}

#endif

#if 1  // debug

void LayerAssigner::debugPlotLAModel(LAModel& la_model, std::string flag)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();
  std::string& la_temp_directory_path = RTDM.getConfig().la_temp_directory_path;

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
  for (auto& [net_idx, segment_set] : RTDM.getNetGlobalResultMap(die)) {
    GPStruct global_result_struct(RTUTIL.getString("global_result(net_", net_idx, ")"));
    for (Segment<LayerCoord>* segment : segment_set) {
      for (NetShape& net_shape : RTDM.getNetGlobalShapeList(net_idx, *segment)) {
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
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShape));
      gp_boundary.set_rect(patch->get_real_rect());
      gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(patch->get_layer_idx()));
      detailed_patch_struct.push(gp_boundary);
    }
    gp_gds.addStruct(detailed_patch_struct);
  }

  // layer_node_map
  {
    std::vector<GridMap<LANode>>& layer_node_map = la_model.get_layer_node_map();
    // la_node_map
    {
      GPStruct la_node_map_struct("la_node_map");
      for (GridMap<LANode>& la_node_map : layer_node_map) {
        for (int32_t grid_x = 0; grid_x < la_node_map.get_x_size(); grid_x++) {
          for (int32_t grid_y = 0; grid_y < la_node_map.get_y_size(); grid_y++) {
            LANode& la_node = la_node_map[grid_x][grid_y];
            PlanarRect real_rect = RTUTIL.getRealRectByGCell(la_node.get_planar_coord(), gcell_axis);
            int32_t y_reduced_span = std::max(1, real_rect.getYSpan() / 12);
            int32_t y = real_rect.get_ur_y();

            y -= y_reduced_span;
            GPText gp_text_node_real_coord;
            gp_text_node_real_coord.set_coord(real_rect.get_ll_x(), y);
            gp_text_node_real_coord.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            gp_text_node_real_coord.set_message(RTUTIL.getString("(", la_node.get_x(), " , ", la_node.get_y(), " , ", la_node.get_layer_idx(), ")"));
            gp_text_node_real_coord.set_layer_idx(RTGP.getGDSIdxByRouting(la_node.get_layer_idx()));
            gp_text_node_real_coord.set_presentation(GPTextPresentation::kLeftMiddle);
            la_node_map_struct.push(gp_text_node_real_coord);

            y -= y_reduced_span;
            GPText gp_text_node_grid_coord;
            gp_text_node_grid_coord.set_coord(real_rect.get_ll_x(), y);
            gp_text_node_grid_coord.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            gp_text_node_grid_coord.set_message(RTUTIL.getString("(", grid_x, " , ", grid_y, " , ", la_node.get_layer_idx(), ")"));
            gp_text_node_grid_coord.set_layer_idx(RTGP.getGDSIdxByRouting(la_node.get_layer_idx()));
            gp_text_node_grid_coord.set_presentation(GPTextPresentation::kLeftMiddle);
            la_node_map_struct.push(gp_text_node_grid_coord);

            y -= y_reduced_span;
            GPText gp_text_orient_supply_map;
            gp_text_orient_supply_map.set_coord(real_rect.get_ll_x(), y);
            gp_text_orient_supply_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            gp_text_orient_supply_map.set_message("orient_supply_map: ");
            gp_text_orient_supply_map.set_layer_idx(RTGP.getGDSIdxByRouting(la_node.get_layer_idx()));
            gp_text_orient_supply_map.set_presentation(GPTextPresentation::kLeftMiddle);
            la_node_map_struct.push(gp_text_orient_supply_map);

            if (!la_node.get_orient_supply_map().empty()) {
              y -= y_reduced_span;
              GPText gp_text_orient_supply_map_info;
              gp_text_orient_supply_map_info.set_coord(real_rect.get_ll_x(), y);
              gp_text_orient_supply_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
              std::string orient_supply_map_info_message = "--";
              for (auto& [orient, supply] : la_node.get_orient_supply_map()) {
                orient_supply_map_info_message += RTUTIL.getString("(", GetOrientationName()(orient), ",", supply, ")");
              }
              gp_text_orient_supply_map_info.set_message(orient_supply_map_info_message);
              gp_text_orient_supply_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(la_node.get_layer_idx()));
              gp_text_orient_supply_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
              la_node_map_struct.push(gp_text_orient_supply_map_info);
            }

            y -= y_reduced_span;
            GPText gp_text_ignore_net_orient_map;
            gp_text_ignore_net_orient_map.set_coord(real_rect.get_ll_x(), y);
            gp_text_ignore_net_orient_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            gp_text_ignore_net_orient_map.set_message("ignore_net_orient_map: ");
            gp_text_ignore_net_orient_map.set_layer_idx(RTGP.getGDSIdxByRouting(la_node.get_layer_idx()));
            gp_text_ignore_net_orient_map.set_presentation(GPTextPresentation::kLeftMiddle);
            la_node_map_struct.push(gp_text_ignore_net_orient_map);

            if (!la_node.get_ignore_net_orient_map().empty()) {
              y -= y_reduced_span;
              GPText gp_text_ignore_net_orient_map_info;
              gp_text_ignore_net_orient_map_info.set_coord(real_rect.get_ll_x(), y);
              gp_text_ignore_net_orient_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
              std::string ignore_net_orient_map_info_message = "--";
              for (auto& [net_idx, orient_set] : la_node.get_ignore_net_orient_map()) {
                ignore_net_orient_map_info_message += RTUTIL.getString("(", net_idx);
                for (Orientation orient : orient_set) {
                  ignore_net_orient_map_info_message += RTUTIL.getString(",", GetOrientationName()(orient));
                }
                ignore_net_orient_map_info_message += RTUTIL.getString(")");
              }
              gp_text_ignore_net_orient_map_info.set_message(ignore_net_orient_map_info_message);
              gp_text_ignore_net_orient_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(la_node.get_layer_idx()));
              gp_text_ignore_net_orient_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
              la_node_map_struct.push(gp_text_ignore_net_orient_map_info);
            }

            y -= y_reduced_span;
            GPText gp_text_orient_net_map;
            gp_text_orient_net_map.set_coord(real_rect.get_ll_x(), y);
            gp_text_orient_net_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            gp_text_orient_net_map.set_message("orient_net_map: ");
            gp_text_orient_net_map.set_layer_idx(RTGP.getGDSIdxByRouting(la_node.get_layer_idx()));
            gp_text_orient_net_map.set_presentation(GPTextPresentation::kLeftMiddle);
            la_node_map_struct.push(gp_text_orient_net_map);

            if (!la_node.get_orient_net_map().empty()) {
              y -= y_reduced_span;
              GPText gp_text_orient_net_map_info;
              gp_text_orient_net_map_info.set_coord(real_rect.get_ll_x(), y);
              gp_text_orient_net_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
              std::string orient_net_map_info_message = "--";
              for (auto& [orient, net_set] : la_node.get_orient_net_map()) {
                orient_net_map_info_message += RTUTIL.getString("(", GetOrientationName()(orient));
                for (int32_t net_idx : net_set) {
                  orient_net_map_info_message += RTUTIL.getString(",", net_idx);
                }
                orient_net_map_info_message += RTUTIL.getString(")");
              }
              gp_text_orient_net_map_info.set_message(orient_net_map_info_message);
              gp_text_orient_net_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(la_node.get_layer_idx()));
              gp_text_orient_net_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
              la_node_map_struct.push(gp_text_orient_net_map_info);
            }

            y -= y_reduced_span;
            GPText gp_text_net_orient_map;
            gp_text_net_orient_map.set_coord(real_rect.get_ll_x(), y);
            gp_text_net_orient_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            gp_text_net_orient_map.set_message("net_orient_map: ");
            gp_text_net_orient_map.set_layer_idx(RTGP.getGDSIdxByRouting(la_node.get_layer_idx()));
            gp_text_net_orient_map.set_presentation(GPTextPresentation::kLeftMiddle);
            la_node_map_struct.push(gp_text_net_orient_map);

            if (!la_node.get_net_orient_map().empty()) {
              y -= y_reduced_span;
              GPText gp_text_net_orient_map_info;
              gp_text_net_orient_map_info.set_coord(real_rect.get_ll_x(), y);
              gp_text_net_orient_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
              std::string net_orient_map_info_message = "--";
              for (auto& [net_idx, orient_set] : la_node.get_net_orient_map()) {
                net_orient_map_info_message += RTUTIL.getString("(", net_idx);
                for (Orientation orient : orient_set) {
                  net_orient_map_info_message += RTUTIL.getString(",", GetOrientationName()(orient));
                }
                net_orient_map_info_message += RTUTIL.getString(")");
              }
              gp_text_net_orient_map_info.set_message(net_orient_map_info_message);
              gp_text_net_orient_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(la_node.get_layer_idx()));
              gp_text_net_orient_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
              la_node_map_struct.push(gp_text_net_orient_map_info);
            }

            y -= y_reduced_span;
            GPText gp_text_overflow;
            gp_text_overflow.set_coord(real_rect.get_ll_x(), y);
            gp_text_overflow.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            gp_text_overflow.set_message(RTUTIL.getString("overflow: ", la_node.getOverflow()));
            gp_text_overflow.set_layer_idx(RTGP.getGDSIdxByRouting(la_node.get_layer_idx()));
            gp_text_overflow.set_presentation(GPTextPresentation::kLeftMiddle);
            la_node_map_struct.push(gp_text_overflow);
          }
        }
      }
      gp_gds.addStruct(la_node_map_struct);
    }
    // overflow
    {
      GPStruct overflow_struct("overflow");
      for (GridMap<LANode>& la_node_map : layer_node_map) {
        for (int32_t grid_x = 0; grid_x < la_node_map.get_x_size(); grid_x++) {
          for (int32_t grid_y = 0; grid_y < la_node_map.get_y_size(); grid_y++) {
            LANode& la_node = la_node_map[grid_x][grid_y];
            if (la_node.getOverflow() <= 0) {
              continue;
            }
            PlanarRect real_rect = RTUTIL.getRealRectByGCell(la_node.get_planar_coord(), gcell_axis);

            GPBoundary gp_boundary;
            gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kOverflow));
            gp_boundary.set_rect(real_rect);
            gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(la_node.get_layer_idx()));
            overflow_struct.push(gp_boundary);
          }
        }
      }
      gp_gds.addStruct(overflow_struct);
    }
  }

  std::string gds_file_path = RTUTIL.getString(la_temp_directory_path, flag, "_la_model.gds");
  RTGP.plot(gp_gds, gds_file_path);
}

void LayerAssigner::debugCheckLAModel(LAModel& la_model)
{
  std::vector<GridMap<LANode>>& layer_node_map = la_model.get_layer_node_map();
  for (GridMap<LANode>& la_node_map : layer_node_map) {
    for (int32_t x = 0; x < la_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < la_node_map.get_y_size(); y++) {
        LANode& la_node = la_node_map[x][y];
        for (auto& [orient, neighbor] : la_node.get_neighbor_node_map()) {
          Orientation opposite_orient = RTUTIL.getOppositeOrientation(orient);
          if (!RTUTIL.exist(neighbor->get_neighbor_node_map(), opposite_orient)) {
            RTLOG.error(Loc::current(), "The la_node neighbor is not bidirectional!");
          }
          if (neighbor->get_neighbor_node_map()[opposite_orient] != &la_node) {
            RTLOG.error(Loc::current(), "The la_node neighbor is not bidirectional!");
          }
          if (RTUTIL.getOrientation(LayerCoord(la_node), LayerCoord(*neighbor)) == orient) {
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
