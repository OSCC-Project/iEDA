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

void LayerAssigner::route()
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
  buildTopoTree(la_model);
  routeLAModel(la_model);
  updateSummary(la_model);
  printSummary(la_model);
  outputGuide(la_model);
  outputDemandCSV(la_model);
  outputOverflowCSV(la_model);
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
  int32_t topo_spilt_length = 10;
  double prefer_wire_unit = 1;
  double via_unit = 1;
  double overflow_unit = 2;
  /**
   * topo_spilt_length, prefer_wire_unit, via_unit, overflow_unit
   */
  // clang-format off
  LAComParam la_com_param(topo_spilt_length, prefer_wire_unit, via_unit, overflow_unit);
  // clang-format on
  RTLOG.info(Loc::current(), "topo_spilt_length: ", la_com_param.get_topo_spilt_length());
  RTLOG.info(Loc::current(), "prefer_wire_unit: ", la_com_param.get_prefer_wire_unit());
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

void LayerAssigner::buildTopoTree(LAModel& la_model)
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
    la_net.set_topo_tree(RTUTIL.getTreeByFullFlow(candidate_root_coord_list, routing_segment_list, key_coord_pin_map));
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
    routeLANet(la_model, la_task_list[i]);
    if ((i + 1) % batch_size == 0 || (i + 1) == la_task_list.size()) {
      RTLOG.info(Loc::current(), "Routed ", (i + 1), "/", la_task_list.size(), "(", RTUTIL.getPercentage(i + 1, la_task_list.size()), ") nets",
                 stage_monitor.getStatsInfo());
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void LayerAssigner::routeLANet(LAModel& la_model, LANet* la_net)
{
  // 构建la_topo_list,并将通孔线段加入routing_segment_list
  std::vector<LATopo> la_topo_list;
  std::vector<Segment<LayerCoord>> routing_segment_list;
  makeLATopoList(la_model, la_net, la_topo_list, routing_segment_list);
  for (LATopo& la_topo : la_topo_list) {
    routeLATopo(la_model, &la_topo);
    for (Segment<LayerCoord>& routing_segment : la_topo.get_routing_segment_list()) {
      routing_segment_list.push_back(routing_segment);
    }
  }
  MTree<LayerCoord> coord_tree = getCoordTree(la_net, routing_segment_list);
  updateDemandToGraph(la_model, ChangeType::kAdd, coord_tree);
  uploadNetResult(la_net, coord_tree);
}

void LayerAssigner::makeLATopoList(LAModel& la_model, LANet* la_net, std::vector<LATopo>& la_topo_list, std::vector<Segment<LayerCoord>>& routing_segment_list)
{
  int32_t bottom_routing_layer_idx = RTDM.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = RTDM.getConfig().top_routing_layer_idx;
  int32_t topo_spilt_length = la_model.get_la_com_param().get_topo_spilt_length();

  if (la_net->get_topo_tree().get_root() == nullptr) {
    LATopo la_topo;
    for (LAPin& la_pin : la_net->get_la_pin_list()) {
      LAGroup la_group;
      la_group.get_coord_list().push_back(la_pin.get_access_point().getGridLayerCoord());
      la_topo.get_la_group_list().push_back(la_group);
    }
    la_topo_list.push_back(la_topo);
    {
      std::set<PlanarCoord, CmpPlanarCoordByXASC> coord_set;
      for (LATopo& la_topo : la_topo_list) {
        for (LAGroup& la_group : la_topo.get_la_group_list()) {
          for (LayerCoord& coord : la_group.get_coord_list()) {
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
      for (Segment<TNode<LayerCoord>*>& coord_segment : RTUTIL.getSegListByTree(la_net->get_topo_tree())) {
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
    std::map<PlanarCoord, std::vector<LAGroup>, CmpPlanarCoordByXASC> planar_pin_group_map;
    {
      for (LAPin& la_pin : la_net->get_la_pin_list()) {
        LayerCoord grid_coord = la_pin.get_access_point().getGridLayerCoord();

        LAGroup la_group;
        la_group.get_coord_list().push_back(grid_coord);
        planar_pin_group_map[grid_coord.get_planar_coord()].push_back(la_group);
      }
    }
    // planar_steiner_group_map
    std::map<PlanarCoord, LAGroup, CmpPlanarCoordByXASC> planar_steiner_group_map;
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
        LATopo la_topo;
        for (PlanarCoord coord : {planar_topo.get_first(), planar_topo.get_second()}) {
          if (RTUTIL.exist(planar_pin_group_map, coord)) {
            for (LAGroup& la_group : planar_pin_group_map[coord]) {
              la_topo.get_la_group_list().push_back(la_group);
            }
          } else if (RTUTIL.exist(planar_steiner_group_map, coord)) {
            la_topo.get_la_group_list().push_back(planar_steiner_group_map[coord]);
          }
        }
        la_topo_list.push_back(la_topo);
      }
    }
  }
  // 构建topo的其他内容
  {
    for (LATopo& la_topo : la_topo_list) {
      la_topo.set_net_idx(la_net->get_net_idx());
      std::vector<PlanarCoord> coord_list;
      for (LAGroup& la_group : la_topo.get_la_group_list()) {
        for (LayerCoord& coord : la_group.get_coord_list()) {
          coord_list.push_back(coord);
        }
      }
      la_topo.set_bounding_box(RTUTIL.getBoundingBox(coord_list));
    }
  }
}

void LayerAssigner::routeLATopo(LAModel& la_model, LATopo* la_topo)
{
  initSingleTask(la_model, la_topo);
  while (!isConnectedAllEnd(la_model)) {
    routeSinglePath(la_model);
    updatePathResult(la_model);
    resetStartAndEnd(la_model);
    resetSinglePath(la_model);
  }
  updateTaskResult(la_model);
  resetSingleTask(la_model);
}

void LayerAssigner::initSingleTask(LAModel& la_model, LATopo* la_topo)
{
  std::vector<GridMap<LANode>>& layer_node_map = la_model.get_layer_node_map();

  // single topo
  la_model.set_curr_la_topo(la_topo);
  {
    std::vector<std::vector<LANode*>> node_list_list;
    std::vector<LAGroup>& la_group_list = la_topo->get_la_group_list();
    for (LAGroup& la_group : la_group_list) {
      std::vector<LANode*> node_list;
      for (LayerCoord& coord : la_group.get_coord_list()) {
        LANode& la_node = layer_node_map[coord.get_layer_idx()][coord.get_x()][coord.get_y()];
        node_list.push_back(&la_node);
      }
      node_list_list.push_back(node_list);
    }
    for (size_t i = 0; i < node_list_list.size(); i++) {
      if (i == 0) {
        la_model.get_start_node_list_list().push_back(node_list_list[i]);
      } else {
        la_model.get_end_node_list_list().push_back(node_list_list[i]);
      }
    }
  }
  la_model.get_path_node_list().clear();
  la_model.get_single_topo_visited_node_list().clear();
  la_model.get_routing_segment_list().clear();
}

bool LayerAssigner::isConnectedAllEnd(LAModel& la_model)
{
  return la_model.get_end_node_list_list().empty();
}

void LayerAssigner::routeSinglePath(LAModel& la_model)
{
  initPathHead(la_model);
  while (!searchEnded(la_model)) {
    expandSearching(la_model);
    resetPathHead(la_model);
  }
}

void LayerAssigner::initPathHead(LAModel& la_model)
{
  std::vector<std::vector<LANode*>>& start_node_list_list = la_model.get_start_node_list_list();
  std::vector<LANode*>& path_node_list = la_model.get_path_node_list();

  for (std::vector<LANode*>& start_node_list : start_node_list_list) {
    for (LANode* start_node : start_node_list) {
      start_node->set_estimated_cost(getEstimateCostToEnd(la_model, start_node));
      pushToOpenList(la_model, start_node);
    }
  }
  for (LANode* path_node : path_node_list) {
    path_node->set_estimated_cost(getEstimateCostToEnd(la_model, path_node));
    pushToOpenList(la_model, path_node);
  }
  resetPathHead(la_model);
}

bool LayerAssigner::searchEnded(LAModel& la_model)
{
  std::vector<std::vector<LANode*>>& end_node_list_list = la_model.get_end_node_list_list();
  LANode* path_head_node = la_model.get_path_head_node();

  if (path_head_node == nullptr) {
    la_model.set_end_node_list_idx(-1);
    return true;
  }
  for (size_t i = 0; i < end_node_list_list.size(); i++) {
    for (LANode* end_node : end_node_list_list[i]) {
      if (path_head_node == end_node) {
        la_model.set_end_node_list_idx(static_cast<int32_t>(i));
        return true;
      }
    }
  }
  return false;
}

void LayerAssigner::expandSearching(LAModel& la_model)
{
  PriorityQueue<LANode*, std::vector<LANode*>, CmpLANodeCost>& open_queue = la_model.get_open_queue();
  LANode* path_head_node = la_model.get_path_head_node();

  for (auto& [orientation, neighbor_node] : path_head_node->get_neighbor_node_map()) {
    if (neighbor_node == nullptr) {
      continue;
    }
    if (!RTUTIL.isInside(la_model.get_curr_la_topo()->get_bounding_box(), *neighbor_node)) {
      continue;
    }
    if (neighbor_node->isClose()) {
      continue;
    }
    double know_cost = getKnowCost(la_model, path_head_node, neighbor_node);
    if (neighbor_node->isOpen() && know_cost < neighbor_node->get_known_cost()) {
      neighbor_node->set_known_cost(know_cost);
      neighbor_node->set_parent_node(path_head_node);
      // 对优先队列中的值修改了,需要重新建堆
      std::make_heap(open_queue.begin(), open_queue.end(), CmpLANodeCost());
    } else if (neighbor_node->isNone()) {
      neighbor_node->set_known_cost(know_cost);
      neighbor_node->set_parent_node(path_head_node);
      neighbor_node->set_estimated_cost(getEstimateCostToEnd(la_model, neighbor_node));
      pushToOpenList(la_model, neighbor_node);
    }
  }
}

void LayerAssigner::resetPathHead(LAModel& la_model)
{
  la_model.set_path_head_node(popFromOpenList(la_model));
}

void LayerAssigner::updatePathResult(LAModel& la_model)
{
  for (Segment<LayerCoord>& routing_segment : getRoutingSegmentListByNode(la_model.get_path_head_node())) {
    la_model.get_routing_segment_list().push_back(routing_segment);
  }
}

std::vector<Segment<LayerCoord>> LayerAssigner::getRoutingSegmentListByNode(LANode* node)
{
  std::vector<Segment<LayerCoord>> routing_segment_list;

  LANode* curr_node = node;
  LANode* pre_node = curr_node->get_parent_node();

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

void LayerAssigner::resetStartAndEnd(LAModel& la_model)
{
  std::vector<std::vector<LANode*>>& start_node_list_list = la_model.get_start_node_list_list();
  std::vector<std::vector<LANode*>>& end_node_list_list = la_model.get_end_node_list_list();
  std::vector<LANode*>& path_node_list = la_model.get_path_node_list();
  LANode* path_head_node = la_model.get_path_head_node();
  int32_t end_node_list_idx = la_model.get_end_node_list_idx();

  // 对于抵达的终点pin,只保留到达的node
  end_node_list_list[end_node_list_idx].clear();
  end_node_list_list[end_node_list_idx].push_back(path_head_node);

  LANode* path_node = path_head_node->get_parent_node();
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

void LayerAssigner::resetSinglePath(LAModel& la_model)
{
  PriorityQueue<LANode*, std::vector<LANode*>, CmpLANodeCost> empty_queue;
  la_model.set_open_queue(empty_queue);

  std::vector<LANode*>& single_path_visited_node_list = la_model.get_single_path_visited_node_list();
  for (LANode* visited_node : single_path_visited_node_list) {
    visited_node->set_state(LANodeState::kNone);
    visited_node->set_parent_node(nullptr);
    visited_node->set_known_cost(0);
    visited_node->set_estimated_cost(0);
  }
  single_path_visited_node_list.clear();

  la_model.set_path_head_node(nullptr);
  la_model.set_end_node_list_idx(-1);
}

void LayerAssigner::updateTaskResult(LAModel& la_model)
{
  la_model.get_curr_la_topo()->set_routing_segment_list(getRoutingSegmentList(la_model));
}

std::vector<Segment<LayerCoord>> LayerAssigner::getRoutingSegmentList(LAModel& la_model)
{
  LATopo* curr_la_topo = la_model.get_curr_la_topo();

  std::vector<LayerCoord> candidate_root_coord_list;
  std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
  std::vector<LAGroup>& la_group_list = curr_la_topo->get_la_group_list();
  for (size_t i = 0; i < la_group_list.size(); i++) {
    for (LayerCoord& coord : la_group_list[i].get_coord_list()) {
      candidate_root_coord_list.push_back(coord);
      key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
    }
  }
  MTree<LayerCoord> coord_tree = RTUTIL.getTreeByFullFlow(candidate_root_coord_list, la_model.get_routing_segment_list(), key_coord_pin_map);

  std::vector<Segment<LayerCoord>> routing_segment_list;
  for (Segment<TNode<LayerCoord>*>& coord_segment : RTUTIL.getSegListByTree(coord_tree)) {
    routing_segment_list.emplace_back(coord_segment.get_first()->value(), coord_segment.get_second()->value());
  }
  return routing_segment_list;
}

void LayerAssigner::resetSingleTask(LAModel& la_model)
{
  la_model.set_curr_la_topo(nullptr);
  la_model.get_start_node_list_list().clear();
  la_model.get_end_node_list_list().clear();
  la_model.get_path_node_list().clear();
  la_model.get_single_topo_visited_node_list().clear();
  la_model.get_routing_segment_list().clear();
}

// manager open list

void LayerAssigner::pushToOpenList(LAModel& la_model, LANode* curr_node)
{
  PriorityQueue<LANode*, std::vector<LANode*>, CmpLANodeCost>& open_queue = la_model.get_open_queue();
  std::vector<LANode*>& single_topo_visited_node_list = la_model.get_single_topo_visited_node_list();
  std::vector<LANode*>& single_path_visited_node_list = la_model.get_single_path_visited_node_list();

  open_queue.push(curr_node);
  curr_node->set_state(LANodeState::kOpen);
  single_topo_visited_node_list.push_back(curr_node);
  single_path_visited_node_list.push_back(curr_node);
}

LANode* LayerAssigner::popFromOpenList(LAModel& la_model)
{
  PriorityQueue<LANode*, std::vector<LANode*>, CmpLANodeCost>& open_queue = la_model.get_open_queue();

  LANode* node = nullptr;
  if (!open_queue.empty()) {
    node = open_queue.top();
    open_queue.pop();
    node->set_state(LANodeState::kClose);
  }
  return node;
}

// calculate known cost

double LayerAssigner::getKnowCost(LAModel& la_model, LANode* start_node, LANode* end_node)
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
  cost += getNodeCost(la_model, start_node, RTUTIL.getOrientation(*start_node, *end_node));
  cost += getNodeCost(la_model, end_node, RTUTIL.getOrientation(*end_node, *start_node));
  cost += getKnowWireCost(la_model, start_node, end_node);
  cost += getKnowViaCost(la_model, start_node, end_node);
  return cost;
}

double LayerAssigner::getNodeCost(LAModel& la_model, LANode* curr_node, Orientation orientation)
{
  double overflow_unit = la_model.get_la_com_param().get_overflow_unit();

  double node_cost = 0;
  node_cost += curr_node->getOverflowCost(orientation, overflow_unit);
  return node_cost;
}

double LayerAssigner::getKnowWireCost(LAModel& la_model, LANode* start_node, LANode* end_node)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  double prefer_wire_unit = la_model.get_la_com_param().get_prefer_wire_unit();

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

double LayerAssigner::getKnowViaCost(LAModel& la_model, LANode* start_node, LANode* end_node)
{
  double via_unit = la_model.get_la_com_param().get_via_unit();
  double via_cost = (via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
  return via_cost;
}

// calculate estimate cost

double LayerAssigner::getEstimateCostToEnd(LAModel& la_model, LANode* curr_node)
{
  std::vector<std::vector<LANode*>>& end_node_list_list = la_model.get_end_node_list_list();

  double estimate_cost = DBL_MAX;
  for (std::vector<LANode*>& end_node_list : end_node_list_list) {
    for (LANode* end_node : end_node_list) {
      if (end_node->isClose()) {
        continue;
      }
      estimate_cost = std::min(estimate_cost, getEstimateCost(la_model, curr_node, end_node));
    }
  }
  return estimate_cost;
}

double LayerAssigner::getEstimateCost(LAModel& la_model, LANode* start_node, LANode* end_node)
{
  double estimate_cost = 0;
  estimate_cost += getEstimateWireCost(la_model, start_node, end_node);
  estimate_cost += getEstimateViaCost(la_model, start_node, end_node);
  return estimate_cost;
}

double LayerAssigner::getEstimateWireCost(LAModel& la_model, LANode* start_node, LANode* end_node)
{
  double prefer_wire_unit = la_model.get_la_com_param().get_prefer_wire_unit();

  double wire_cost = 0;
  wire_cost += RTUTIL.getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());
  wire_cost *= prefer_wire_unit;
  return wire_cost;
}

double LayerAssigner::getEstimateViaCost(LAModel& la_model, LANode* start_node, LANode* end_node)
{
  double via_unit = la_model.get_la_com_param().get_via_unit();
  double via_cost = (via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
  return via_cost;
}

MTree<LayerCoord> LayerAssigner::getCoordTree(LANet* la_net, std::vector<Segment<LayerCoord>>& routing_segment_list)
{
  std::vector<LayerCoord> candidate_root_coord_list;
  std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
  std::vector<LAPin>& la_pin_list = la_net->get_la_pin_list();
  for (size_t i = 0; i < la_pin_list.size(); i++) {
    LayerCoord coord = la_pin_list[i].get_access_point().getGridLayerCoord();
    candidate_root_coord_list.push_back(coord);
    key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
  }
  return RTUTIL.getTreeByFullFlow(candidate_root_coord_list, routing_segment_list, key_coord_pin_map);
}

void LayerAssigner::uploadNetResult(LANet* la_net, MTree<LayerCoord>& coord_tree)
{
  for (Segment<TNode<LayerCoord>*>& coord_segment : RTUTIL.getSegListByTree(coord_tree)) {
    Segment<LayerCoord>* segment = new Segment<LayerCoord>(coord_segment.get_first()->value(), coord_segment.get_second()->value());
    RTDM.updateNetGlobalResultToGCellMap(ChangeType::kAdd, la_net->get_net_idx(), segment);
  }
}

#if 1  // update env

void LayerAssigner::updateDemandToGraph(LAModel& la_model, ChangeType change_type, MTree<LayerCoord>& coord_tree)
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
  std::vector<GridMap<LANode>>& layer_node_map = la_model.get_layer_node_map();
  for (auto& [usage_coord, orientation_list] : usage_map) {
    LANode& la_node = layer_node_map[usage_coord.get_layer_idx()][usage_coord.get_x()][usage_coord.get_y()];
    la_node.updateDemand(orientation_list, change_type);
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
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();
  Summary& summary = RTDM.getDatabase().get_summary();
  int32_t enable_timing = RTDM.getConfig().enable_timing;

  std::map<int32_t, int32_t>& routing_demand_map = summary.la_summary.routing_demand_map;
  int32_t& total_demand = summary.la_summary.total_demand;
  std::map<int32_t, int32_t>& routing_overflow_map = summary.la_summary.routing_overflow_map;
  int32_t& total_overflow = summary.la_summary.total_overflow;
  std::map<int32_t, double>& routing_wire_length_map = summary.la_summary.routing_wire_length_map;
  double& total_wire_length = summary.la_summary.total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.la_summary.cut_via_num_map;
  int32_t& total_via_num = summary.la_summary.total_via_num;
  std::map<std::string, std::map<std::string, double>>& clock_timing = summary.la_summary.clock_timing;
  std::map<std::string, double>& power_map = summary.la_summary.power_map;

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
  clock_timing.clear();
  power_map.clear();

  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    GridMap<LANode>& la_node_map = layer_node_map[layer_idx];
    for (int32_t x = 0; x < la_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < la_node_map.get_y_size(); y++) {
        std::map<Orientation, int32_t>& orient_supply_map = la_node_map[x][y].get_orient_supply_map();
        std::map<Orientation, int32_t>& orient_demand_map = la_node_map[x][y].get_orient_demand_map();
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
    RTI.updateTimingAndPower(real_pin_coord_map_list, routing_segment_list_list, clock_timing, power_map);
  }
}

void LayerAssigner::printSummary(LAModel& la_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  Summary& summary = RTDM.getDatabase().get_summary();
  int32_t enable_timing = RTDM.getConfig().enable_timing;

  std::map<int32_t, int32_t>& routing_demand_map = summary.la_summary.routing_demand_map;
  int32_t& total_demand = summary.la_summary.total_demand;
  std::map<int32_t, int32_t>& routing_overflow_map = summary.la_summary.routing_overflow_map;
  int32_t& total_overflow = summary.la_summary.total_overflow;
  std::map<int32_t, double>& routing_wire_length_map = summary.la_summary.routing_wire_length_map;
  double& total_wire_length = summary.la_summary.total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.la_summary.cut_via_num_map;
  int32_t& total_via_num = summary.la_summary.total_via_num;
  std::map<std::string, std::map<std::string, double>>& clock_timing = summary.la_summary.clock_timing;
  std::map<std::string, double>& power_map = summary.la_summary.power_map;

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
}

void LayerAssigner::outputDemandCSV(LAModel& la_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& la_temp_directory_path = RTDM.getConfig().la_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  std::vector<GridMap<LANode>>& layer_node_map = la_model.get_layer_node_map();
  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::ofstream* demand_csv_file
        = RTUTIL.getOutputFileStream(RTUTIL.getString(la_temp_directory_path, "demand_map_", routing_layer.get_layer_name(), ".csv"));

    GridMap<LANode>& la_node_map = layer_node_map[routing_layer.get_layer_idx()];
    for (int32_t y = la_node_map.get_y_size() - 1; y >= 0; y--) {
      for (int32_t x = 0; x < la_node_map.get_x_size(); x++) {
        std::map<Orientation, int32_t>& orient_demand_map = la_node_map[x][y].get_orient_demand_map();
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
        std::map<Orientation, int32_t>& orient_supply_map = la_node_map[x][y].get_orient_supply_map();
        std::map<Orientation, int32_t>& orient_demand_map = la_node_map[x][y].get_orient_demand_map();
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
