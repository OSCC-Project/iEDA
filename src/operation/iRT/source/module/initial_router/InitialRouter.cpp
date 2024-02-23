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
#include "InitialRouter.hpp"

#include "GDSPlotter.hpp"
#include "RTAPI.hpp"
#include "RTUtil.hpp"

namespace irt {

// public

void InitialRouter::initInst()
{
  if (_ir_instance == nullptr) {
    _ir_instance = new InitialRouter();
  }
}

InitialRouter& InitialRouter::getInst()
{
  if (_ir_instance == nullptr) {
    LOG_INST.error(Loc::current(), "The instance not initialized!");
  }
  return *_ir_instance;
}

void InitialRouter::destroyInst()
{
  if (_ir_instance != nullptr) {
    delete _ir_instance;
    _ir_instance = nullptr;
  }
}

// function

void InitialRouter::route(std::vector<Net>& net_list)
{
  Monitor monitor;
  LOG_INST.info(Loc::current(), "Begin routing...");
  IRModel ir_model = initIRModel(net_list);
  setIRParameter(ir_model);
  makeGridCoordList(ir_model);
  initLayerNodeMap(ir_model);
  buildIRNodeNeighbor(ir_model);
  buildOrienSupply(ir_model);
  checkIRModel(ir_model);
  sortIRModel(ir_model);
  routeIRModel(ir_model);
  updateIRModel(ir_model);
  outputGuide(ir_model);
  LOG_INST.info(Loc::current(), "End route", monitor.getStatsInfo());

  reportIRModel(ir_model);
}

// private

InitialRouter* InitialRouter::_ir_instance = nullptr;

IRModel InitialRouter::initIRModel(std::vector<Net>& net_list)
{
  IRModel ir_model;
  ir_model.set_ir_net_list(convertToIRNetList(net_list));
  return ir_model;
}

std::vector<IRNet> InitialRouter::convertToIRNetList(std::vector<Net>& net_list)
{
  std::vector<IRNet> ir_net_list;
  ir_net_list.reserve(net_list.size());
  for (size_t i = 0; i < net_list.size(); i++) {
    ir_net_list.emplace_back(convertToIRNet(net_list[i]));
  }
  return ir_net_list;
}

IRNet InitialRouter::convertToIRNet(Net& net)
{
  IRNet ir_net;
  ir_net.set_origin_net(&net);
  ir_net.set_net_idx(net.get_net_idx());
  ir_net.set_connect_type(net.get_connect_type());
  for (Pin& pin : net.get_pin_list()) {
    ir_net.get_ir_pin_list().push_back(IRPin(pin));
  }
  ir_net.set_bounding_box(net.get_bounding_box());
  return ir_net;
}

void InitialRouter::setIRParameter(IRModel& ir_model)
{
  IRParameter ir_parameter;
  LOG_INST.info(Loc::current(), "topo_spilt_length : ", ir_parameter.get_topo_spilt_length());
  LOG_INST.info(Loc::current(), "congestion_unit : ", ir_parameter.get_congestion_unit());
  LOG_INST.info(Loc::current(), "prefer_wire_unit : ", ir_parameter.get_prefer_wire_unit());
  LOG_INST.info(Loc::current(), "via_unit : ", ir_parameter.get_via_unit());
  LOG_INST.info(Loc::current(), "corner_unit : ", ir_parameter.get_corner_unit());
  ir_model.set_ir_parameter(ir_parameter);
}

void InitialRouter::makeGridCoordList(IRModel& ir_model)
{
  for (IRNet& ir_net : ir_model.get_ir_net_list()) {
    for (IRPin& ir_pin : ir_net.get_ir_pin_list()) {
      LayerCoord grid_coord = ir_pin.get_access_point_list().front().getGridLayerCoord();
      if (ir_pin.get_is_driving()) {
        ir_net.set_driving_grid_coord(grid_coord);
      }
      ir_net.get_grid_coord_list().push_back(grid_coord);
    }
  }
}

void InitialRouter::initLayerNodeMap(IRModel& ir_model)
{
  Die& die = DM_INST.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  std::vector<GridMap<IRNode>>& layer_node_map = ir_model.get_layer_node_map();
  layer_node_map.resize(routing_layer_list.size());
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    GridMap<IRNode>& ir_node_map = layer_node_map[layer_idx];
    ir_node_map.init(die.getXSize(), die.getYSize());
    for (int32_t x = 0; x < ir_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < ir_node_map.get_y_size(); y++) {
        IRNode& ir_node = ir_node_map[x][y];
        ir_node.set_coord(x, y);
        ir_node.set_layer_idx(layer_idx);
      }
    }
  }
}

void InitialRouter::buildIRNodeNeighbor(IRModel& ir_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  int32_t bottom_routing_layer_idx = DM_INST.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = DM_INST.getConfig().top_routing_layer_idx;

  std::vector<GridMap<IRNode>>& layer_node_map = ir_model.get_layer_node_map();
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    bool routing_h = routing_layer_list[layer_idx].isPreferH();
    bool routing_v = !routing_h;
    if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
      routing_h = false;
      routing_v = false;
    }
    GridMap<IRNode>& ir_node_map = layer_node_map[layer_idx];
    for (int32_t x = 0; x < ir_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < ir_node_map.get_y_size(); y++) {
        std::map<Orientation, IRNode*>& neighbor_ptr_map = ir_node_map[x][y].get_neighbor_node_map();
        if (routing_h) {
          if (x != 0) {
            neighbor_ptr_map[Orientation::kWest] = &ir_node_map[x - 1][y];
          }
          if (x != (ir_node_map.get_x_size() - 1)) {
            neighbor_ptr_map[Orientation::kEast] = &ir_node_map[x + 1][y];
          }
        }
        if (routing_v) {
          if (y != 0) {
            neighbor_ptr_map[Orientation::kSouth] = &ir_node_map[x][y - 1];
          }
          if (y != (ir_node_map.get_y_size() - 1)) {
            neighbor_ptr_map[Orientation::kNorth] = &ir_node_map[x][y + 1];
          }
        }
        if (layer_idx != 0) {
          neighbor_ptr_map[Orientation::kDown] = &layer_node_map[layer_idx - 1][x][y];
        }
        if (layer_idx != static_cast<int32_t>(layer_node_map.size()) - 1) {
          neighbor_ptr_map[Orientation::kUp] = &layer_node_map[layer_idx + 1][x][y];
        }
      }
    }
  }
}

void InitialRouter::buildOrienSupply(IRModel& ir_model)
{
  GridMap<GCell>& gcell_map = DM_INST.getDatabase().get_gcell_map();

  std::vector<GridMap<IRNode>>& layer_node_map = ir_model.get_layer_node_map();
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    GridMap<IRNode>& ir_node_map = layer_node_map[layer_idx];
    for (int32_t x = 0; x < ir_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < ir_node_map.get_y_size(); y++) {
        ir_node_map[x][y].set_orien_supply_map(gcell_map[x][y].get_routing_orien_supply_map()[layer_idx]);
      }
    }
  }
}

void InitialRouter::checkIRModel(IRModel& ir_model)
{
  std::vector<GridMap<IRNode>>& layer_node_map = ir_model.get_layer_node_map();
  for (GridMap<IRNode>& ir_node_map : layer_node_map) {
    for (int32_t x = 0; x < ir_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < ir_node_map.get_y_size(); y++) {
        IRNode& ir_node = ir_node_map[x][y];
        for (auto& [orien, neighbor] : ir_node.get_neighbor_node_map()) {
          Orientation opposite_orien = RTUtil::getOppositeOrientation(orien);
          if (!RTUtil::exist(neighbor->get_neighbor_node_map(), opposite_orien)) {
            LOG_INST.error(Loc::current(), "The ir_node neighbor is not bidirection!");
          }
          if (neighbor->get_neighbor_node_map()[opposite_orien] != &ir_node) {
            LOG_INST.error(Loc::current(), "The ir_node neighbor is not bidirection!");
          }
          LayerCoord node_coord(ir_node.get_planar_coord(), ir_node.get_layer_idx());
          LayerCoord neighbor_coord(neighbor->get_planar_coord(), neighbor->get_layer_idx());
          if (RTUtil::getOrientation(node_coord, neighbor_coord) == orien) {
            continue;
          }
          LOG_INST.error(Loc::current(), "The neighbor orien is different with real region!");
        }
      }
    }
  }
}

void InitialRouter::sortIRModel(IRModel& ir_model)
{
  Monitor monitor;
  LOG_INST.info(Loc::current(), "Begin sorting nets...");
  std::vector<int32_t>& ir_net_idx_list = ir_model.get_ir_net_idx_list();
  for (IRNet& ir_net : ir_model.get_ir_net_list()) {
    ir_net_idx_list.push_back(ir_net.get_net_idx());
  }
  std::sort(ir_net_idx_list.begin(), ir_net_idx_list.end(),
            [&](int32_t net_idx1, int32_t net_idx2) { return sortByMultiLevel(ir_model, net_idx1, net_idx2); });
  LOG_INST.info(Loc::current(), "End sort nets", monitor.getStatsInfo());
}

bool InitialRouter::sortByMultiLevel(IRModel& ir_model, int32_t net_idx1, int32_t net_idx2)
{
  IRNet& net1 = ir_model.get_ir_net_list()[net_idx1];
  IRNet& net2 = ir_model.get_ir_net_list()[net_idx2];

  SortStatus sort_status = SortStatus::kNone;

  sort_status = sortByClockPriority(net1, net2);
  if (sort_status == SortStatus::kTrue) {
    return true;
  } else if (sort_status == SortStatus::kFalse) {
    return false;
  }
  sort_status = sortByRoutingAreaASC(net1, net2);
  if (sort_status == SortStatus::kTrue) {
    return true;
  } else if (sort_status == SortStatus::kFalse) {
    return false;
  }
  sort_status = sortByLengthWidthRatioDESC(net1, net2);
  if (sort_status == SortStatus::kTrue) {
    return true;
  } else if (sort_status == SortStatus::kFalse) {
    return false;
  }
  sort_status = sortByPinNumDESC(net1, net2);
  if (sort_status == SortStatus::kTrue) {
    return true;
  } else if (sort_status == SortStatus::kFalse) {
    return false;
  }
  return false;
}

// 时钟线网优先
SortStatus InitialRouter::sortByClockPriority(IRNet& net1, IRNet& net2)
{
  ConnectType net1_connect_type = net1.get_connect_type();
  ConnectType net2_connect_type = net2.get_connect_type();

  if (net1_connect_type == ConnectType::kClock && net2_connect_type != ConnectType::kClock) {
    return SortStatus::kTrue;
  } else if (net1_connect_type != ConnectType::kClock && net2_connect_type == ConnectType::kClock) {
    return SortStatus::kFalse;
  } else {
    return SortStatus::kEqual;
  }
}

// RoutingArea 升序
SortStatus InitialRouter::sortByRoutingAreaASC(IRNet& net1, IRNet& net2)
{
  double net1_routing_area = net1.get_bounding_box().getTotalSize();
  double net2_routing_area = net2.get_bounding_box().getTotalSize();

  if (net1_routing_area < net2_routing_area) {
    return SortStatus::kTrue;
  } else if (net1_routing_area == net2_routing_area) {
    return SortStatus::kEqual;
  } else {
    return SortStatus::kFalse;
  }
}

// 长宽比 降序
SortStatus InitialRouter::sortByLengthWidthRatioDESC(IRNet& net1, IRNet& net2)
{
  BoundingBox& net1_bounding_box = net1.get_bounding_box();
  BoundingBox& net2_bounding_box = net2.get_bounding_box();

  double net1_length_width_ratio = net1_bounding_box.getXSize() / 1.0 / net1_bounding_box.getYSize();
  if (net1_length_width_ratio < 1) {
    net1_length_width_ratio = 1 / net1_length_width_ratio;
  }
  double net2_length_width_ratio = net2_bounding_box.getXSize() / 1.0 / net2_bounding_box.getYSize();
  if (net2_length_width_ratio < 1) {
    net2_length_width_ratio = 1 / net2_length_width_ratio;
  }
  if (net1_length_width_ratio > net2_length_width_ratio) {
    return SortStatus::kTrue;
  } else if (net1_length_width_ratio == net2_length_width_ratio) {
    return SortStatus::kEqual;
  } else {
    return SortStatus::kFalse;
  }
}

// PinNum 降序
SortStatus InitialRouter::sortByPinNumDESC(IRNet& net1, IRNet& net2)
{
  int32_t net1_pin_num = static_cast<int32_t>(net1.get_ir_pin_list().size());
  int32_t net2_pin_num = static_cast<int32_t>(net2.get_ir_pin_list().size());

  if (net1_pin_num > net2_pin_num) {
    return SortStatus::kTrue;
  } else if (net1_pin_num == net2_pin_num) {
    return SortStatus::kEqual;
  } else {
    return SortStatus::kFalse;
  }
}

void InitialRouter::routeIRModel(IRModel& ir_model)
{
  std::vector<IRNet>& ir_net_list = ir_model.get_ir_net_list();
  std::vector<int32_t>& ir_net_idx_list = ir_model.get_ir_net_idx_list();

  int32_t batch_size = RTUtil::getBatchSize(ir_net_idx_list.size());

  Monitor monitor;
  for (size_t i = 0; i < ir_net_idx_list.size(); i++) {
    routeIRNet(ir_model, ir_net_list[ir_net_idx_list[i]]);
    if ((i + 1) % batch_size == 0 || (i + 1) == ir_net_idx_list.size()) {
      LOG_INST.info(Loc::current(), "Routed ", (i + 1), "/", ir_net_idx_list.size(), " nets", monitor.getStatsInfo());
    }
  }
}

void InitialRouter::routeIRNet(IRModel& ir_model, IRNet& ir_net)
{
  // 构建ir_task_list，并将flute得到的通孔线段加入routing_segment_list
  std::vector<IRTask> ir_task_list;
  std::vector<Segment<LayerCoord>> routing_segment_list;
  makeIRTaskList(ir_model, ir_net, ir_task_list, routing_segment_list);
  for (IRTask& ir_task : ir_task_list) {
    routeIRTask(ir_model, &ir_task);
    for (Segment<LayerCoord>& routing_segment : ir_task.get_routing_segment_list()) {
      routing_segment_list.push_back(routing_segment);
    }
  }
  MTree<LayerCoord> coord_tree = getCoordTree(ir_net, routing_segment_list);
  updateDemand(ir_model, ir_net, coord_tree);

  std::function<Guide(LayerCoord&)> convertToGuide = [](LayerCoord& coord) {
    ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
    return Guide(LayerRect(RTUtil::getRealRectByGCell(coord, gcell_axis), coord.get_layer_idx()), coord);
  };
  ir_net.set_ir_result_tree(RTUtil::convertTree(coord_tree, convertToGuide));
}

void InitialRouter::makeIRTaskList(IRModel& ir_model, IRNet& ir_net, std::vector<IRTask>& ir_task_list,
                                   std::vector<Segment<LayerCoord>>& routing_segment_list)
{
  int32_t bottom_routing_layer_idx = DM_INST.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = DM_INST.getConfig().top_routing_layer_idx;
  int32_t topo_spilt_length = ir_model.get_ir_parameter().get_topo_spilt_length();
  // planar_coord_list
  std::vector<PlanarCoord> planar_coord_list;
  {
    for (LayerCoord& grid_coord : ir_net.get_grid_coord_list()) {
      planar_coord_list.push_back(grid_coord.get_planar_coord());
    }
    std::sort(planar_coord_list.begin(), planar_coord_list.end(), CmpPlanarCoordByXASC());
    planar_coord_list.erase(std::unique(planar_coord_list.begin(), planar_coord_list.end()), planar_coord_list.end());
  }
  if (planar_coord_list.size() == 1) {
    IRTask ir_task;
    for (LayerCoord& grid_coord : ir_net.get_grid_coord_list()) {
      IRGroup ir_group;
      ir_group.get_coord_list().push_back(grid_coord);
      ir_task.get_ir_group_list().push_back(ir_group);
    }
    ir_task_list.push_back(ir_task);
  } else {
    // key_planar_group_map
    std::map<PlanarCoord, std::vector<IRGroup>, CmpPlanarCoordByXASC> key_planar_group_map;
    {
      for (LayerCoord& grid_coord : ir_net.get_grid_coord_list()) {
        IRGroup ir_group;
        ir_group.get_coord_list().push_back(grid_coord);
        key_planar_group_map[grid_coord.get_planar_coord()].push_back(ir_group);
      }
    }
    // planar_topo_list
    std::vector<Segment<PlanarCoord>> planar_topo_list;
    {
      for (Segment<PlanarCoord>& planar_topo : getPlanarTopoListByFlute(planar_coord_list)) {
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
    }
    // add_planar_layer_map
    std::map<PlanarCoord, std::set<LayerCoord, CmpLayerCoordByLayerASC>, CmpPlanarCoordByXASC> add_planar_layer_map;
    {
      for (Segment<PlanarCoord>& planar_topo : planar_topo_list) {
        if (!RTUtil::exist(key_planar_group_map, planar_topo.get_first())) {
          for (int32_t layer_idx = bottom_routing_layer_idx; layer_idx <= top_routing_layer_idx; layer_idx++) {
            add_planar_layer_map[planar_topo.get_first()].insert(LayerCoord(planar_topo.get_first(), layer_idx));
          }
        }
        if (!RTUtil::exist(key_planar_group_map, planar_topo.get_second())) {
          for (int32_t layer_idx = bottom_routing_layer_idx; layer_idx <= top_routing_layer_idx; layer_idx++) {
            add_planar_layer_map[planar_topo.get_second()].insert(LayerCoord(planar_topo.get_second(), layer_idx));
          }
        }
      }
    }
    // 补充steiner point的垂直线段
    {
      for (auto& [add_planar_coord, layer_coord_set] : add_planar_layer_map) {
        LayerCoord first_coord = *layer_coord_set.begin();
        LayerCoord second_coord = *layer_coord_set.rbegin();
        if (first_coord == second_coord) {
          continue;
        }
        routing_segment_list.emplace_back(first_coord, second_coord);
      }
    }
    // 生成task group
    {
      for (Segment<PlanarCoord>& planar_topo : planar_topo_list) {
        IRTask ir_task;
        if (RTUtil::exist(key_planar_group_map, planar_topo.get_first())) {
          for (IRGroup& ir_group : key_planar_group_map[planar_topo.get_first()]) {
            ir_task.get_ir_group_list().push_back(ir_group);
          }
        } else if (RTUtil::exist(add_planar_layer_map, planar_topo.get_first())) {
          IRGroup ir_group;
          for (const LayerCoord& coord : add_planar_layer_map[planar_topo.get_first()]) {
            ir_group.get_coord_list().push_back(coord);
          }
          ir_task.get_ir_group_list().push_back(ir_group);
        }
        if (RTUtil::exist(key_planar_group_map, planar_topo.get_second())) {
          for (IRGroup& ir_group : key_planar_group_map[planar_topo.get_second()]) {
            ir_task.get_ir_group_list().push_back(ir_group);
          }
        } else if (RTUtil::exist(add_planar_layer_map, planar_topo.get_second())) {
          IRGroup ir_group;
          for (const LayerCoord& coord : add_planar_layer_map[planar_topo.get_second()]) {
            ir_group.get_coord_list().push_back(coord);
          }
          ir_task.get_ir_group_list().push_back(ir_group);
        }
        ir_task_list.push_back(ir_task);
      }
    }
  }
  // 构建task的其他内容
  {
    for (IRTask& ir_task : ir_task_list) {
      ir_task.set_net_idx(ir_net.get_net_idx());
      std::vector<PlanarCoord> coord_list;
      for (IRGroup& ir_group : ir_task.get_ir_group_list()) {
        for (LayerCoord& coord : ir_group.get_coord_list()) {
          coord_list.push_back(coord);
        }
      }
      ir_task.set_bounding_box(RTUtil::getBoundingBox(coord_list));
    }
  }
}

std::vector<Segment<PlanarCoord>> InitialRouter::getPlanarTopoListByFlute(std::vector<PlanarCoord>& planar_coord_list)
{
  size_t point_num = planar_coord_list.size();
  if (point_num == 1) {
    return {};
  }
  Flute::DTYPE* x_list = (Flute::DTYPE*) malloc(sizeof(Flute::DTYPE) * (point_num));
  Flute::DTYPE* y_list = (Flute::DTYPE*) malloc(sizeof(Flute::DTYPE) * (point_num));
  for (size_t i = 0; i < point_num; i++) {
    x_list[i] = planar_coord_list[i].get_x();
    y_list[i] = planar_coord_list[i].get_y();
  }
  Flute::Tree flute_tree = Flute::flute(point_num, x_list, y_list, FLUTE_ACCURACY);
  // Flute::printtree(flute_tree);
  free(x_list);
  free(y_list);

  std::vector<Segment<PlanarCoord>> planar_topo_list;
  for (int i = 0; i < 2 * flute_tree.deg - 2; i++) {
    int n_id = flute_tree.branch[i].n;
    PlanarCoord first_coord(flute_tree.branch[i].x, flute_tree.branch[i].y);
    PlanarCoord second_coord(flute_tree.branch[n_id].x, flute_tree.branch[n_id].y);
    planar_topo_list.emplace_back(first_coord, second_coord);
  }
  Flute::free_tree(flute_tree);
  return planar_topo_list;
}

void InitialRouter::routeIRTask(IRModel& ir_model, IRTask* ir_task)
{
  initSingleTask(ir_model, ir_task);
  while (!isConnectedAllEnd(ir_model)) {
    routeSinglePath(ir_model);
    updatePathResult(ir_model);
    updateDirectionSet(ir_model);
    resetStartAndEnd(ir_model);
    resetSinglePath(ir_model);
  }
  updateTaskResult(ir_model);
  resetSingleTask(ir_model);
}

void InitialRouter::initSingleTask(IRModel& ir_model, IRTask* ir_task)
{
  std::vector<GridMap<IRNode>>& layer_node_map = ir_model.get_layer_node_map();

  // single task
  ir_model.set_curr_ir_task(ir_task);
  {
    std::vector<std::vector<IRNode*>> node_list_list;
    std::vector<IRGroup>& ir_group_list = ir_task->get_ir_group_list();
    for (IRGroup& ir_group : ir_group_list) {
      std::vector<IRNode*> node_comb;
      for (LayerCoord& coord : ir_group.get_coord_list()) {
        IRNode& ir_node = layer_node_map[coord.get_layer_idx()][coord.get_x()][coord.get_y()];
        node_comb.push_back(&ir_node);
      }
      node_list_list.push_back(node_comb);
    }
    for (size_t i = 0; i < node_list_list.size(); i++) {
      if (i == 0) {
        ir_model.get_start_node_list_list().push_back(node_list_list[i]);
      } else {
        ir_model.get_end_node_list_list().push_back(node_list_list[i]);
      }
    }
  }
  ir_model.get_path_node_list().clear();
  ir_model.get_single_task_visited_node_list().clear();
  ir_model.get_routing_segment_list().clear();
}

bool InitialRouter::isConnectedAllEnd(IRModel& ir_model)
{
  return ir_model.get_end_node_list_list().empty();
}

void InitialRouter::routeSinglePath(IRModel& ir_model)
{
  initPathHead(ir_model);
  while (!searchEnded(ir_model)) {
    expandSearching(ir_model);
    resetPathHead(ir_model);
  }
}

void InitialRouter::initPathHead(IRModel& ir_model)
{
  std::vector<std::vector<IRNode*>>& start_node_list_list = ir_model.get_start_node_list_list();
  std::vector<IRNode*>& path_node_list = ir_model.get_path_node_list();

  for (std::vector<IRNode*>& start_node_comb : start_node_list_list) {
    for (IRNode* start_node : start_node_comb) {
      start_node->set_estimated_cost(getEstimateCostToEnd(ir_model, start_node));
      pushToOpenList(ir_model, start_node);
    }
  }
  for (IRNode* path_node : path_node_list) {
    path_node->set_estimated_cost(getEstimateCostToEnd(ir_model, path_node));
    pushToOpenList(ir_model, path_node);
  }
  resetPathHead(ir_model);
}

bool InitialRouter::searchEnded(IRModel& ir_model)
{
  std::vector<std::vector<IRNode*>>& end_node_list_list = ir_model.get_end_node_list_list();
  IRNode* path_head_node = ir_model.get_path_head_node();

  if (path_head_node == nullptr) {
    ir_model.set_end_node_comb_idx(-1);
    return true;
  }
  for (size_t i = 0; i < end_node_list_list.size(); i++) {
    for (IRNode* end_node : end_node_list_list[i]) {
      if (path_head_node == end_node) {
        ir_model.set_end_node_comb_idx(static_cast<int32_t>(i));
        return true;
      }
    }
  }
  return false;
}

void InitialRouter::expandSearching(IRModel& ir_model)
{
  PriorityQueue<IRNode*, std::vector<IRNode*>, CmpIRNodeCost>& open_queue = ir_model.get_open_queue();
  IRNode* path_head_node = ir_model.get_path_head_node();

  for (auto& [orientation, neighbor_node] : path_head_node->get_neighbor_node_map()) {
    if (neighbor_node == nullptr) {
      continue;
    }
    if (!RTUtil::isInside(ir_model.get_curr_ir_task()->get_bounding_box(), *neighbor_node)) {
      continue;
    }
    if (neighbor_node->isClose()) {
      continue;
    }
    double know_cost = getKnowCost(ir_model, path_head_node, neighbor_node);
    if (neighbor_node->isOpen() && know_cost < neighbor_node->get_known_cost()) {
      neighbor_node->set_known_cost(know_cost);
      neighbor_node->set_parent_node(path_head_node);
      // 对优先队列中的值修改了，需要重新建堆
      std::make_heap(open_queue.begin(), open_queue.end(), CmpIRNodeCost());
    } else if (neighbor_node->isNone()) {
      neighbor_node->set_known_cost(know_cost);
      neighbor_node->set_parent_node(path_head_node);
      neighbor_node->set_estimated_cost(getEstimateCostToEnd(ir_model, neighbor_node));
      pushToOpenList(ir_model, neighbor_node);
    }
  }
}

void InitialRouter::resetPathHead(IRModel& ir_model)
{
  ir_model.set_path_head_node(popFromOpenList(ir_model));
}

bool InitialRouter::isRoutingFailed(IRModel& ir_model)
{
  return ir_model.get_end_node_comb_idx() == -1;
}

void InitialRouter::resetSinglePath(IRModel& ir_model)
{
  PriorityQueue<IRNode*, std::vector<IRNode*>, CmpIRNodeCost> empty_queue;
  ir_model.set_open_queue(empty_queue);

  std::vector<IRNode*>& single_path_visited_node_list = ir_model.get_single_path_visited_node_list();
  for (IRNode* visited_node : single_path_visited_node_list) {
    visited_node->set_state(IRNodeState::kNone);
    visited_node->set_parent_node(nullptr);
    visited_node->set_known_cost(0);
    visited_node->set_estimated_cost(0);
  }
  single_path_visited_node_list.clear();

  ir_model.set_path_head_node(nullptr);
  ir_model.set_end_node_comb_idx(-1);
}

void InitialRouter::updatePathResult(IRModel& ir_model)
{
  for (Segment<LayerCoord>& routing_segment : getRoutingSegmentListByNode(ir_model.get_path_head_node())) {
    ir_model.get_routing_segment_list().push_back(routing_segment);
  }
}

std::vector<Segment<LayerCoord>> InitialRouter::getRoutingSegmentListByNode(IRNode* node)
{
  std::vector<Segment<LayerCoord>> routing_segment_list;

  IRNode* curr_node = node;
  IRNode* pre_node = curr_node->get_parent_node();

  if (pre_node == nullptr) {
    // 起点和终点重合
    return routing_segment_list;
  }
  Orientation curr_orientation = RTUtil::getOrientation(*curr_node, *pre_node);
  while (pre_node->get_parent_node() != nullptr) {
    Orientation pre_orientation = RTUtil::getOrientation(*pre_node, *pre_node->get_parent_node());
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

void InitialRouter::updateDirectionSet(IRModel& ir_model)
{
  IRNode* path_head_node = ir_model.get_path_head_node();

  IRNode* curr_node = path_head_node;
  IRNode* pre_node = curr_node->get_parent_node();
  while (pre_node != nullptr) {
    curr_node->get_direction_set().insert(RTUtil::getDirection(*curr_node, *pre_node));
    pre_node->get_direction_set().insert(RTUtil::getDirection(*pre_node, *curr_node));
    curr_node = pre_node;
    pre_node = curr_node->get_parent_node();
  }
}

void InitialRouter::resetStartAndEnd(IRModel& ir_model)
{
  std::vector<std::vector<IRNode*>>& start_node_list_list = ir_model.get_start_node_list_list();
  std::vector<std::vector<IRNode*>>& end_node_list_list = ir_model.get_end_node_list_list();
  std::vector<IRNode*>& path_node_list = ir_model.get_path_node_list();
  IRNode* path_head_node = ir_model.get_path_head_node();
  int32_t end_node_comb_idx = ir_model.get_end_node_comb_idx();

  // 对于抵达的终点pin，只保留到达的node
  end_node_list_list[end_node_comb_idx].clear();
  end_node_list_list[end_node_comb_idx].push_back(path_head_node);

  IRNode* path_node = path_head_node->get_parent_node();
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
    // 初始化时，要把start_node_list_list的pin只留一个ap点
    // 后续只要将end_node_list_list的pin保留一个ap点
    start_node_list_list.front().clear();
    start_node_list_list.front().push_back(path_node);
  }
  start_node_list_list.push_back(end_node_list_list[end_node_comb_idx]);
  end_node_list_list.erase(end_node_list_list.begin() + end_node_comb_idx);
}

void InitialRouter::updateTaskResult(IRModel& ir_model)
{
  ir_model.get_curr_ir_task()->set_routing_segment_list(getRoutingSegmentList(ir_model));
}

std::vector<Segment<LayerCoord>> InitialRouter::getRoutingSegmentList(IRModel& ir_model)
{
  IRTask* curr_ir_task = ir_model.get_curr_ir_task();

  std::vector<LayerCoord> driving_grid_coord_list;
  std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
  std::vector<IRGroup>& ir_group_list = curr_ir_task->get_ir_group_list();
  for (size_t i = 0; i < ir_group_list.size(); i++) {
    for (LayerCoord& coord : ir_group_list[i].get_coord_list()) {
      driving_grid_coord_list.push_back(coord);
      key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
    }
  }
  // 构建 优化 检查 routing_segment_list
  MTree<LayerCoord> coord_tree = RTUtil::getTreeByFullFlow(driving_grid_coord_list, ir_model.get_routing_segment_list(), key_coord_pin_map);

  std::vector<Segment<LayerCoord>> routing_segment_list;
  for (Segment<TNode<LayerCoord>*>& segment : RTUtil::getSegListByTree(coord_tree)) {
    routing_segment_list.emplace_back(segment.get_first()->value(), segment.get_second()->value());
  }
  return routing_segment_list;
}

void InitialRouter::resetSingleTask(IRModel& ir_model)
{
  ir_model.set_curr_ir_task(nullptr);
  ir_model.get_start_node_list_list().clear();
  ir_model.get_end_node_list_list().clear();
  ir_model.get_path_node_list().clear();

  std::vector<IRNode*>& single_task_visited_node_list = ir_model.get_single_task_visited_node_list();
  for (IRNode* single_task_visited_node : single_task_visited_node_list) {
    single_task_visited_node->get_direction_set().clear();
  }
  single_task_visited_node_list.clear();

  ir_model.get_routing_segment_list().clear();
}

// manager open list

void InitialRouter::pushToOpenList(IRModel& ir_model, IRNode* curr_node)
{
  PriorityQueue<IRNode*, std::vector<IRNode*>, CmpIRNodeCost>& open_queue = ir_model.get_open_queue();
  std::vector<IRNode*>& single_task_visited_node_list = ir_model.get_single_task_visited_node_list();
  std::vector<IRNode*>& single_path_visited_node_list = ir_model.get_single_path_visited_node_list();

  open_queue.push(curr_node);
  curr_node->set_state(IRNodeState::kOpen);
  single_task_visited_node_list.push_back(curr_node);
  single_path_visited_node_list.push_back(curr_node);
}

IRNode* InitialRouter::popFromOpenList(IRModel& ir_model)
{
  PriorityQueue<IRNode*, std::vector<IRNode*>, CmpIRNodeCost>& open_queue = ir_model.get_open_queue();

  IRNode* node = nullptr;
  if (!open_queue.empty()) {
    node = open_queue.top();
    open_queue.pop();
    node->set_state(IRNodeState::kClose);
  }
  return node;
}

// calculate known cost

double InitialRouter::getKnowCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node)
{
  bool exist_neighbor = false;
  for (auto& [orientation, neighbor_ptr] : start_node->get_neighbor_node_map()) {
    if (neighbor_ptr == end_node) {
      exist_neighbor = true;
      break;
    }
  }
  if (!exist_neighbor) {
    LOG_INST.error(Loc::current(), "The neighbor not exist!");
  }

  double cost = 0;
  cost += start_node->get_known_cost();
  cost += getNodeCost(ir_model, start_node, RTUtil::getOrientation(*start_node, *end_node));
  cost += getNodeCost(ir_model, end_node, RTUtil::getOrientation(*end_node, *start_node));
  cost += getKnowWireCost(ir_model, start_node, end_node);
  cost += getKnowCornerCost(ir_model, start_node, end_node);
  cost += getKnowViaCost(ir_model, start_node, end_node);
  return cost;
}

double InitialRouter::getNodeCost(IRModel& ir_model, IRNode* curr_node, Orientation orientation)
{
  double congestion_unit = ir_model.get_ir_parameter().get_congestion_unit();

  double node_cost = 0;
  node_cost += curr_node->getCongestionCost(orientation) * congestion_unit;
  return node_cost;
}

double InitialRouter::getKnowWireCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  double prefer_wire_unit = ir_model.get_ir_parameter().get_prefer_wire_unit();

  double wire_cost = 0;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    wire_cost += RTUtil::getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());

    RoutingLayer& routing_layer = routing_layer_list[start_node->get_layer_idx()];
    if (routing_layer.get_prefer_direction() == RTUtil::getDirection(*start_node, *end_node)) {
      wire_cost *= prefer_wire_unit;
    }
  }
  return wire_cost;
}

double InitialRouter::getKnowCornerCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node)
{
  double corner_unit = ir_model.get_ir_parameter().get_corner_unit();

  double corner_cost = 0;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    std::set<Direction> direction_set;
    // 添加start direction
    std::set<Direction>& start_direction_set = start_node->get_direction_set();
    direction_set.insert(start_direction_set.begin(), start_direction_set.end());
    // 添加start到parent的direction
    if (start_node->get_parent_node() != nullptr) {
      direction_set.insert(RTUtil::getDirection(*start_node->get_parent_node(), *start_node));
    }
    // 添加end direction
    std::set<Direction>& end_direction_set = end_node->get_direction_set();
    direction_set.insert(end_direction_set.begin(), end_direction_set.end());
    // 添加start到end的direction
    direction_set.insert(RTUtil::getDirection(*start_node, *end_node));

    if (direction_set.size() == 2) {
      corner_cost += corner_unit;
    } else if (direction_set.size() == 2) {
      LOG_INST.error(Loc::current(), "Direction set is error!");
    }
  }
  return corner_cost;
}

double InitialRouter::getKnowViaCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node)
{
  double via_unit = ir_model.get_ir_parameter().get_via_unit();
  double via_cost = (via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
  return via_cost;
}

// calculate estimate cost

double InitialRouter::getEstimateCostToEnd(IRModel& ir_model, IRNode* curr_node)
{
  std::vector<std::vector<IRNode*>>& end_node_list_list = ir_model.get_end_node_list_list();

  double estimate_cost = DBL_MAX;
  for (std::vector<IRNode*>& end_node_comb : end_node_list_list) {
    for (IRNode* end_node : end_node_comb) {
      if (end_node->isClose()) {
        continue;
      }
      estimate_cost = std::min(estimate_cost, getEstimateCost(ir_model, curr_node, end_node));
    }
  }
  return estimate_cost;
}

double InitialRouter::getEstimateCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node)
{
  double estimate_cost = 0;
  estimate_cost += getEstimateWireCost(ir_model, start_node, end_node);
  estimate_cost += getEstimateCornerCost(ir_model, start_node, end_node);
  estimate_cost += getEstimateViaCost(ir_model, start_node, end_node);
  return estimate_cost;
}

double InitialRouter::getEstimateWireCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node)
{
  double prefer_wire_unit = ir_model.get_ir_parameter().get_prefer_wire_unit();

  double wire_cost = 0;
  wire_cost += RTUtil::getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());
  wire_cost *= prefer_wire_unit;
  return wire_cost;
}

double InitialRouter::getEstimateCornerCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node)
{
  double corner_unit = ir_model.get_ir_parameter().get_corner_unit();

  double corner_cost = 0;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    if (RTUtil::isOblique(*start_node, *end_node)) {
      corner_cost += corner_unit;
    }
  }
  return corner_cost;
}

double InitialRouter::getEstimateViaCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node)
{
  double via_unit = ir_model.get_ir_parameter().get_via_unit();
  double via_cost = (via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
  return via_cost;
}

MTree<LayerCoord> InitialRouter::getCoordTree(IRNet& ir_net, std::vector<Segment<LayerCoord>>& routing_segment_list)
{
  LayerCoord& driving_grid_coord = ir_net.get_driving_grid_coord();
  std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;

  std::vector<LayerCoord>& grid_coord_list = ir_net.get_grid_coord_list();
  for (size_t i = 0; i < grid_coord_list.size(); i++) {
    key_coord_pin_map[grid_coord_list[i]].insert(static_cast<int32_t>(i));
  }
  return RTUtil::getTreeByFullFlow({driving_grid_coord}, routing_segment_list, key_coord_pin_map);
}

void InitialRouter::updateDemand(IRModel& ir_model, IRNet& ir_net, MTree<LayerCoord>& coord_tree)
{
  std::set<LayerCoord, CmpLayerCoordByXASC> key_coord_set;
  for (LayerCoord& grid_coord : ir_net.get_grid_coord_list()) {
    key_coord_set.insert(grid_coord);
  }
  std::vector<Segment<LayerCoord>> routing_segment_list;
  for (Segment<TNode<LayerCoord>*>& coord_segment : RTUtil::getSegListByTree(coord_tree)) {
    routing_segment_list.emplace_back(coord_segment.get_first()->value(), coord_segment.get_second()->value());
  }
  std::map<LayerCoord, std::set<Orientation>, CmpLayerCoordByXASC> usage_map;
  if (routing_segment_list.empty()) {
    // 单层的local net
    if (key_coord_set.size() > 1) {
      LOG_INST.error(Loc::current(), "The net is not local!");
    }
    for (Orientation orientation : {Orientation::kUp, Orientation::kDown}) {
      usage_map[*key_coord_set.begin()].insert(orientation);
    }
  } else {
    // 跨gcell线网和多层的local_net
    for (Segment<LayerCoord>& coord_segment : routing_segment_list) {
      LayerCoord& first_coord = coord_segment.get_first();
      LayerCoord& second_coord = coord_segment.get_second();

      Orientation orientation = RTUtil::getOrientation(first_coord, second_coord);
      if (orientation == Orientation::kNone || orientation == Orientation::kOblique) {
        LOG_INST.error(Loc::current(), "The orientation is error!");
      }
      Orientation oppo_orientation = RTUtil::getOppositeOrientation(orientation);

      int32_t first_x = first_coord.get_x();
      int32_t first_y = first_coord.get_y();
      int32_t first_layer_idx = first_coord.get_layer_idx();
      int32_t second_x = second_coord.get_x();
      int32_t second_y = second_coord.get_y();
      int32_t second_layer_idx = second_coord.get_layer_idx();
      RTUtil::swapByASC(first_x, second_x);
      RTUtil::swapByASC(first_y, second_y);
      RTUtil::swapByASC(first_layer_idx, second_layer_idx);

      for (int32_t x = first_x; x <= second_x; x++) {
        for (int32_t y = first_y; y <= second_y; y++) {
          for (int32_t layer_idx = first_layer_idx; layer_idx <= second_layer_idx; layer_idx++) {
            LayerCoord coord(x, y, layer_idx);
            if (coord != first_coord) {
              usage_map[coord].insert(oppo_orientation);
            }
            if (coord != second_coord) {
              usage_map[coord].insert(orientation);
            }
          }
        }
      }
    }
  }
  std::vector<GridMap<IRNode>>& layer_node_map = ir_model.get_layer_node_map();
  for (auto& [usage_coord, orientation_list] : usage_map) {
    IRNode& ir_node = layer_node_map[usage_coord.get_layer_idx()][usage_coord.get_x()][usage_coord.get_y()];
    ir_node.updateDemand(orientation_list, ChangeType::kAdd);
  }
}

void InitialRouter::updateIRModel(IRModel& ir_model)
{
  for (IRNet& ir_net : ir_model.get_ir_net_list()) {
    Net* origin_net = ir_net.get_origin_net();
    origin_net->set_ir_result_tree(ir_net.get_ir_result_tree());
  }
}

void InitialRouter::outputGuide(IRModel& ir_model)
{
  Monitor monitor;

  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::string ir_temp_directory_path = DM_INST.getConfig().ir_temp_directory_path;

  std::ofstream* guide_file_stream = RTUtil::getOutputFileStream(ir_temp_directory_path + "route.guide");
  if (guide_file_stream == nullptr) {
    return;
  }
  for (IRNet& ir_net : ir_model.get_ir_net_list()) {
    RTUtil::pushStream(guide_file_stream, ir_net.get_origin_net()->get_net_name(), "\n(\n");

    for (Segment<TNode<Guide>*> guide_node_seg : RTUtil::getSegListByTree(ir_net.get_ir_result_tree())) {
      Guide first_guide = guide_node_seg.get_first()->value();
      Guide second_guide = guide_node_seg.get_second()->value();

      int first_layer_idx = first_guide.get_layer_idx();
      int second_layer_idx = second_guide.get_layer_idx();

      if (first_layer_idx != second_layer_idx) {
        RTUtil::swapByASC(first_layer_idx, second_layer_idx);
        for (int layer_idx = first_layer_idx; layer_idx <= second_layer_idx; layer_idx++) {
          RTUtil::pushStream(guide_file_stream, first_guide.get_lb_x(), " ", first_guide.get_lb_y(), " ", first_guide.get_rt_x(), " ",
                             first_guide.get_rt_y(), " ", routing_layer_list[layer_idx].get_layer_name(), "\n");
        }
      } else {
        RTUtil::swapByCMP(first_guide, second_guide,
                          [](Guide& a, Guide& b) { return CmpPlanarCoordByXASC()(a.get_grid_coord(), b.get_grid_coord()); });
        RTUtil::pushStream(guide_file_stream, first_guide.get_lb_x(), " ", first_guide.get_lb_y(), " ", second_guide.get_rt_x(), " ",
                           second_guide.get_rt_y(), " ", routing_layer_list[first_guide.get_layer_idx()].get_layer_name(), "\n");
      }
    }
    RTUtil::pushStream(guide_file_stream, ")\n");
  }
  RTUtil::closeFileStream(guide_file_stream);
}

#if 1  // exhibit

void InitialRouter::reportIRModel(IRModel& ir_model)
{
  Monitor monitor;
  LOG_INST.info(Loc::current(), "Begin reporting...");
  reportSummary(ir_model);
  writeDemandCSV(ir_model);
  writeOverflowCSV(ir_model);
  LOG_INST.info(Loc::current(), "End report", monitor.getStatsInfo());
}

void InitialRouter::reportSummary(IRModel& ir_model)
{
  int32_t micron_dbu = DM_INST.getDatabase().get_micron_dbu();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DM_INST.getDatabase().get_cut_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = DM_INST.getDatabase().get_layer_via_master_list();
  std::map<int32_t, int32_t>& ir_routing_demand_map = DM_INST.getReporter().ir_routing_demand_map;
  int32_t& ir_total_demand_num = DM_INST.getReporter().ir_total_demand_num;
  std::map<int32_t, int32_t>& ir_routing_overflow_map = DM_INST.getReporter().ir_routing_overflow_map;
  int32_t& ir_total_overflow_num = DM_INST.getReporter().ir_total_overflow_num;
  std::map<int32_t, double>& ir_routing_wire_length_map = DM_INST.getReporter().ir_routing_wire_length_map;
  double& ir_total_wire_length = DM_INST.getReporter().ir_total_wire_length;
  std::map<int32_t, int32_t>& ir_cut_via_num_map = DM_INST.getReporter().ir_cut_via_num_map;
  int32_t& ir_total_via_num = DM_INST.getReporter().ir_total_via_num;
  std::map<std::string, std::vector<double>>& ir_timing = DM_INST.getReporter().ir_timing;

  for (RoutingLayer& routing_layer : routing_layer_list) {
    ir_routing_demand_map[routing_layer.get_layer_idx()] = 0;
  }
  ir_total_demand_num = 0;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    ir_routing_overflow_map[routing_layer.get_layer_idx()] = 0;
  }
  ir_total_overflow_num = 0;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    ir_routing_wire_length_map[routing_layer.get_layer_idx()] = 0;
  }
  ir_total_wire_length = 0;
  for (CutLayer& cut_layer : cut_layer_list) {
    ir_cut_via_num_map[cut_layer.get_layer_idx()] = 0;
  }
  ir_total_via_num = 0;

  std::vector<GridMap<IRNode>>& layer_node_map = ir_model.get_layer_node_map();
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    GridMap<IRNode>& ir_node_map = layer_node_map[layer_idx];
    for (int32_t x = 0; x < ir_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < ir_node_map.get_y_size(); y++) {
        std::map<Orientation, int32_t>& orien_supply_map = ir_node_map[x][y].get_orien_supply_map();
        std::map<Orientation, int32_t>& orien_demand_map = ir_node_map[x][y].get_orien_demand_map();
        int32_t total_demand = 0;
        int32_t total_overflow = 0;
        if (routing_layer_list[layer_idx].isPreferH()) {
          total_demand = (orien_demand_map[Orientation::kEast] + orien_demand_map[Orientation::kWest]);
          total_overflow = std::max(0, orien_demand_map[Orientation::kEast] - orien_supply_map[Orientation::kEast])
                           + std::max(0, orien_demand_map[Orientation::kWest] - orien_supply_map[Orientation::kWest]);
        } else {
          total_demand = (orien_demand_map[Orientation::kSouth] + orien_demand_map[Orientation::kNorth]);
          total_overflow = std::max(0, orien_demand_map[Orientation::kSouth] - orien_supply_map[Orientation::kSouth])
                           + std::max(0, orien_demand_map[Orientation::kNorth] - orien_supply_map[Orientation::kNorth]);
        }
        ir_routing_demand_map[layer_idx] += total_demand;
        ir_total_demand_num += total_demand;
        ir_routing_overflow_map[layer_idx] += total_overflow;
        ir_total_overflow_num += total_overflow;
      }
    }
  }

  for (IRNet& ir_net : ir_model.get_ir_net_list()) {
    for (Segment<TNode<Guide>*> guide_node_seg : RTUtil::getSegListByTree(ir_net.get_ir_result_tree())) {
      Guide& first_guide = guide_node_seg.get_first()->value();
      int32_t first_layer_idx = first_guide.get_layer_idx();
      Guide& second_guide = guide_node_seg.get_second()->value();
      int32_t second_layer_idx = second_guide.get_layer_idx();

      if (first_layer_idx == second_layer_idx) {
        double wire_length = RTUtil::getManhattanDistance(first_guide.getMidPoint(), second_guide.getMidPoint()) / 1.0 / micron_dbu;
        ir_routing_wire_length_map[first_layer_idx] += wire_length;
        ir_total_wire_length += wire_length;
      } else {
        RTUtil::swapByASC(first_layer_idx, second_layer_idx);
        for (int32_t layer_idx = first_layer_idx; layer_idx < second_layer_idx; layer_idx++) {
          ir_cut_via_num_map[layer_via_master_list[layer_idx].front().get_cut_layer_idx()]++;
          ir_total_via_num++;
        }
      }
    }
  }

  std::map<int32_t, std::map<std::string, LayerCoord>> net_pin_coord_map;
  std::map<int32_t, std::vector<Segment<LayerCoord>>> net_segment_map;

  ir_timing = RTAPI_INST.getTiming(net_pin_coord_map, net_segment_map);
}

void InitialRouter::writeDemandCSV(IRModel& ir_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::string ir_temp_directory_path = DM_INST.getConfig().ir_temp_directory_path;

  std::vector<GridMap<IRNode>>& layer_node_map = ir_model.get_layer_node_map();

  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::ofstream* demand_csv_file
        = RTUtil::getOutputFileStream(RTUtil::getString(ir_temp_directory_path, "demand_map_", routing_layer.get_layer_name(), ".csv"));

    GridMap<IRNode>& ir_node_map = layer_node_map[routing_layer.get_layer_idx()];
    for (int32_t y = ir_node_map.get_y_size() - 1; y >= 0; y--) {
      for (int32_t x = 0; x < ir_node_map.get_x_size(); x++) {
        std::map<Orientation, int32_t>& orien_demand_map = ir_node_map[x][y].get_orien_demand_map();
        int32_t total_demand = 0;
        if (routing_layer.isPreferH()) {
          total_demand = (orien_demand_map[Orientation::kEast] + orien_demand_map[Orientation::kWest]);
        } else {
          total_demand = (orien_demand_map[Orientation::kSouth] + orien_demand_map[Orientation::kNorth]);
        }
        RTUtil::pushStream(demand_csv_file, total_demand, ",");
      }
      RTUtil::pushStream(demand_csv_file, "\n");
    }
    RTUtil::closeFileStream(demand_csv_file);
  }
}

void InitialRouter::writeOverflowCSV(IRModel& ir_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::string ir_temp_directory_path = DM_INST.getConfig().ir_temp_directory_path;

  std::vector<GridMap<IRNode>>& layer_node_map = ir_model.get_layer_node_map();

  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::ofstream* overflow_csv_file
        = RTUtil::getOutputFileStream(RTUtil::getString(ir_temp_directory_path, "overflow_map_", routing_layer.get_layer_name(), ".csv"));

    GridMap<IRNode>& ir_node_map = layer_node_map[routing_layer.get_layer_idx()];
    for (int32_t y = ir_node_map.get_y_size() - 1; y >= 0; y--) {
      for (int32_t x = 0; x < ir_node_map.get_x_size(); x++) {
        std::map<Orientation, int32_t>& orien_supply_map = ir_node_map[x][y].get_orien_supply_map();
        std::map<Orientation, int32_t>& orien_demand_map = ir_node_map[x][y].get_orien_demand_map();
        int32_t total_overflow = 0;
        if (routing_layer.isPreferH()) {
          total_overflow = std::max(0, orien_demand_map[Orientation::kEast] - orien_supply_map[Orientation::kEast])
                           + std::max(0, orien_demand_map[Orientation::kWest] - orien_supply_map[Orientation::kWest]);
        } else {
          total_overflow = std::max(0, orien_demand_map[Orientation::kSouth] - orien_supply_map[Orientation::kSouth])
                           + std::max(0, orien_demand_map[Orientation::kNorth] - orien_supply_map[Orientation::kNorth]);
        }
        RTUtil::pushStream(overflow_csv_file, total_overflow, ",");
      }
      RTUtil::pushStream(overflow_csv_file, "\n");
    }
    RTUtil::closeFileStream(overflow_csv_file);
  }
}

#endif

}  // namespace irt
