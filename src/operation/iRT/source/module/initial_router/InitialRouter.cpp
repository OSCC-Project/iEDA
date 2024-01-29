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
  initLayerNodeMap(ir_model);
  buildIRNodeNeighbor(ir_model);
  buildAccessSupply(ir_model);
  checkIRModel(ir_model);
  routeIRModel(ir_model);
  updateIRModel(ir_model);
  LOG_INST.info(Loc::current(), "End route", monitor.getStatsInfo());
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
  IRParameter ir_parameter(1, 1, 1);
  LOG_INST.info(Loc::current(), "prefer_wire_unit : ", ir_parameter.get_prefer_wire_unit());
  LOG_INST.info(Loc::current(), "via_unit : ", ir_parameter.get_via_unit());
  LOG_INST.info(Loc::current(), "corner_unit : ", ir_parameter.get_corner_unit());
  ir_model.set_ir_parameter(ir_parameter);
}

void InitialRouter::initLayerNodeMap(IRModel& ir_model)
{
}

void InitialRouter::buildIRNodeNeighbor(IRModel& ir_model)
{
}

void InitialRouter::buildAccessSupply(IRModel& ir_model)
{
}

void InitialRouter::checkIRModel(IRModel& ir_model)
{
}

void InitialRouter::routeIRModel(IRModel& ir_model)
{
}

void InitialRouter::updateIRModel(IRModel& ir_model)
{
}

#if 0

std::vector<IRNet> InitialRouter::initLayerNodeMap(IRModel& ir_model){

}

void InitialRouter::buildIRNodeNeighborMap(IRModel& ir_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  irt_int bottom_routing_layer_idx = DM_INST.getConfig().bottom_routing_layer_idx;
  irt_int top_routing_layer_idx = DM_INST.getConfig().top_routing_layer_idx;

  std::vector<GridMap<IRNode>>& layer_node_map = ir_model.get_layer_node_map();
  for (irt_int layer_idx = 0; layer_idx < static_cast<irt_int>(layer_node_map.size()); layer_idx++) {
    bool routing_h = routing_layer_list[layer_idx].isPreferH();
    bool routing_v = !routing_h;
    if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
      routing_h = false;
      routing_v = false;
    }
    GridMap<IRNode>& ir_node_map = layer_node_map[layer_idx];
    for (irt_int x = 0; x < ir_node_map.get_x_size(); x++) {
      for (irt_int y = 0; y < ir_node_map.get_y_size(); y++) {
        std::map<Orientation, IRNode*>& neighbor_ptr_map = ir_node_map[x][y].get_neighbor_ptr_map();
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
        if (layer_idx != static_cast<irt_int>(layer_node_map.size()) - 1) {
          neighbor_ptr_map[Orientation::kUp] = &layer_node_map[layer_idx + 1][x][y];
        }
      }
    }
  }
}



void InitialRouter::buildAccessSupply(IRModel& ir_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  std::vector<GridMap<IRNode>>& layer_node_map = ir_model.get_layer_node_map();
  // access supply
  for (irt_int layer_idx = 0; layer_idx < static_cast<irt_int>(layer_node_map.size()); layer_idx++) {
    RoutingLayer& routing_layer = routing_layer_list[layer_idx];
    GridMap<IRNode>& node_map = layer_node_map[layer_idx];

    if (routing_layer.isPreferH()) {
#pragma omp parallel for collapse(2)
      for (irt_int y = 0; y < node_map.get_y_size(); y++) {
        for (irt_int x = 1; x < node_map.get_x_size(); x++) {
          IRNode& pre_node = node_map[x - 1][y];
          IRNode& curr_node = node_map[x][y];
          PlanarRect base_rect = RTUtil::getBoundingBox({pre_node.get_base_region(), curr_node.get_base_region()});
          for (PlanarRect& wire : getCrossingWireList(base_rect, routing_layer)) {
            if (isAccess(pre_node, curr_node, wire)) {
              pre_node.get_orien_access_supply_map()[Orientation::kEast]++;
              curr_node.get_orien_access_supply_map()[Orientation::kWest]++;
            }
          }
        }
      }
    } else {
#pragma omp parallel for collapse(2)
      for (irt_int x = 0; x < node_map.get_x_size(); x++) {
        for (irt_int y = 1; y < node_map.get_y_size(); y++) {
          IRNode& pre_node = node_map[x][y - 1];
          IRNode& curr_node = node_map[x][y];
          PlanarRect base_rect = RTUtil::getBoundingBox({pre_node.get_base_region(), curr_node.get_base_region()});
          for (PlanarRect& wire : getCrossingWireList(base_rect, routing_layer)) {
            if (isAccess(pre_node, curr_node, wire)) {
              pre_node.get_orien_access_supply_map()[Orientation::kNorth]++;
              curr_node.get_orien_access_supply_map()[Orientation::kSouth]++;
            }
          }
        }
      }
    }
  }
}



////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////

#if 1  // iterative

void InitialRouter::iterative(IRModel& ir_model)
{
  irt_int ir_max_iter_num = DM_INST.getConfig().ir_max_iter_num;

  for (irt_int iter = 1; iter <= ir_max_iter_num; iter++) {
    Monitor iter_monitor;
    LOG_INST.info(Loc::current(), "****** Start Model Iteration(", iter, "/", ir_max_iter_num, ") ******");
    ir_model.set_curr_iter(iter);
    resetIRModel(ir_model);
    routeIRModel(ir_model);
    processIRModel(ir_model);
    plotIRModel(ir_model);
    outputCongestionMap(ir_model);
    LOG_INST.info(Loc::current(), "****** End Model Iteration(", iter, "/", ir_max_iter_num, ")", iter_monitor.getStatsInfo(), " ******");
    if (stopIRModel(ir_model)) {
      if (iter < ir_max_iter_num) {
        LOG_INST.info(Loc::current(), "****** Terminate the iteration by reaching the condition in advance! ******");
      }
      ir_model.set_curr_iter(-1);
      break;
    }
  }
}

void InitialRouter::resetIRModel(IRModel& ir_model)
{
  if (ir_model.get_curr_iter() == 1) {
    sortIRModel(ir_model);
  } else {
    updateRipupGrid(ir_model);
    addHistoryCost(ir_model);
    ripupPassedNet(ir_model);
  }
}

void InitialRouter::sortIRModel(IRModel& ir_model)
{
  if (ir_model.get_curr_iter() != 1) {
    return;
  }
  Monitor monitor;
  if (omp_get_num_threads() == 1) {
    LOG_INST.info(Loc::current(), "Sorting all nets beginning...");
  }

  std::vector<irt_int> net_order_list;
  for (IRNet& ir_net : ir_model.get_ir_net_list()) {
    net_order_list.push_back(ir_net.get_net_idx());
  }
  std::sort(net_order_list.begin(), net_order_list.end(),
            [&](irt_int net_idx1, irt_int net_idx2) { return sortByMultiLevel(ir_model, net_idx1, net_idx2); });
  ir_model.get_net_order_list_list().push_back(net_order_list);

  if (omp_get_num_threads() == 1) {
    LOG_INST.info(Loc::current(), "Sorting all nets completed!", monitor.getStatsInfo());
  }
}

bool InitialRouter::sortByMultiLevel(IRModel& ir_model, irt_int net_idx1, irt_int net_idx2)
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
  irt_int net1_pin_num = static_cast<irt_int>(net1.get_ir_pin_list().size());
  irt_int net2_pin_num = static_cast<irt_int>(net2.get_ir_pin_list().size());

  if (net1_pin_num > net2_pin_num) {
    return SortStatus::kTrue;
  } else if (net1_pin_num == net2_pin_num) {
    return SortStatus::kEqual;
  } else {
    return SortStatus::kFalse;
  }
}

void InitialRouter::updateRipupGrid(IRModel& ir_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  std::vector<GridMap<IRNode>>& layer_node_map = ir_model.get_layer_node_map();

  /**
   * history_grid_access_orien_map不要清零
   * ripup_grid_set需要清零
   */
  ir_model.get_ripup_grid_set().clear();

  for (GridMap<IRNode>& node_map : layer_node_map) {
    for (irt_int grid_x = 0; grid_x < node_map.get_x_size(); grid_x++) {
      for (irt_int grid_y = 0; grid_y < node_map.get_y_size(); grid_y++) {
        IRNode& ir_node = node_map[grid_x][grid_y];
        std::map<Orientation, irt_int>& orien_access_supply_map = ir_node.get_orien_access_supply_map();
        std::map<Orientation, irt_int>& orien_access_demand_map = ir_node.get_orien_access_demand_map();

        if (routing_layer_list[ir_node.get_layer_idx()].isPreferH()) {
          for (Orientation orientation : {Orientation::kEast, Orientation::kWest}) {
            double access_overflow = (orien_access_demand_map[orientation] - orien_access_supply_map[orientation]);
            if (access_overflow > 1) {
              ir_model.get_history_grid_access_orien_map()[ir_node].insert(orientation);
              ir_model.get_ripup_grid_set().insert(ir_node);
            }
          }
        } else {
          for (Orientation orientation : {Orientation::kSouth, Orientation::kNorth}) {
            double access_overflow = (orien_access_demand_map[orientation] - orien_access_supply_map[orientation]);
            if (access_overflow > 1) {
              ir_model.get_history_grid_access_orien_map()[ir_node].insert(orientation);
              ir_model.get_ripup_grid_set().insert(ir_node);
            }
          }
        }
      }
    }
  }
}

void InitialRouter::addHistoryCost(IRModel& ir_model)
{
  double ir_history_cost_unit = DM_INST.getConfig().ir_history_cost_unit;

  std::vector<GridMap<IRNode>>& layer_node_map = ir_model.get_layer_node_map();

  // 由于history_grid_access_orien_map不清零，所以会添加新的history cost，并累加原有的
  for (auto& [coord, orient_set] : ir_model.get_history_grid_access_orien_map()) {
    IRNode& ir_node = layer_node_map[coord.get_layer_idx()][coord.get_x()][coord.get_y()];
    for (const auto& orientation : orient_set) {
      ir_node.get_history_orien_access_cost_map()[orientation] += ir_history_cost_unit;
    }
  }
}

void InitialRouter::ripupPassedNet(IRModel& ir_model)
{
  std::vector<GridMap<IRNode>>& layer_node_map = ir_model.get_layer_node_map();

  // 由于ripup_grid_set清零，所以只会重拆本次的
  std::set<irt_int> all_passed_net_set;
  for (const auto& coord : ir_model.get_ripup_grid_set()) {
    IRNode& ir_node = layer_node_map[coord.get_layer_idx()][coord.get_x()][coord.get_y()];
    std::set<irt_int>& passed_net_set = ir_node.get_passed_net_set();
    all_passed_net_set.insert(passed_net_set.begin(), passed_net_set.end());
  }

  for (irt_int net_idx : all_passed_net_set) {
    IRNet& ir_net = ir_model.get_ir_net_list()[net_idx];
    // 将env中的布线结果清空
    updateDemand(ir_model, ir_net, ChangeType::kDel);
    // 清空routing_tree
    ir_net.get_routing_tree().clear();
    ir_net.set_routing_state(RoutingState::kUnrouted);
  }
}

void InitialRouter::routeIRModel(IRModel& ir_model)
{
  Monitor monitor;

  std::vector<IRNet>& ir_net_list = ir_model.get_ir_net_list();
  if (ir_model.get_net_order_list_list().empty()) {
    LOG_INST.error(Loc::current(), "The net_order_list_list is empty!");
  }
  std::vector<irt_int>& net_order_list = ir_model.get_net_order_list_list().back();

  irt_int batch_size = RTUtil::getBatchSize(ir_net_list.size());

  Monitor stage_monitor;
  for (size_t i = 0; i < net_order_list.size(); i++) {
    routeIRNet(ir_model, ir_net_list[net_order_list[i]]);
    if (omp_get_num_threads() == 1 && (i + 1) % batch_size == 0) {
      LOG_INST.info(Loc::current(), "Routed ", (i + 1), " nets", stage_monitor.getStatsInfo());
    }
  }
  if (omp_get_num_threads() == 1) {
    LOG_INST.info(Loc::current(), "Routed ", ir_net_list.size(), " nets", monitor.getStatsInfo());
  }
}

void InitialRouter::routeIRNet(IRModel& ir_model, IRNet& ir_net)
{
  if (ir_net.get_routing_state() == RoutingState::kRouted) {
    return;
  }
  initSingleNet(ir_model, ir_net);
  // outputIRDataset(ir_model, ir_net);
  for (IRTask& ir_task : ir_model.get_ir_task_list()) {
    initSingleTask(ir_model, ir_task);
    while (!isConnectedAllEnd(ir_model)) {
      routeSinglePath(ir_model);
      updatePathResult(ir_model);
      updateDirectionSet(ir_model);
      resetStartAndEnd(ir_model);
      resetSinglePath(ir_model);
    }
    resetSingleTask(ir_model);
  }
  updateNetResult(ir_model, ir_net);
  resetSingleNet(ir_model);
}

void InitialRouter::outputIRDataset(IRModel& ir_model, IRNet& ir_net)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  static size_t written_net_num = 0;
  static std::string ir_dataset_path;
  static std::ofstream* ir_dataset;

  if (written_net_num == 0) {
    std::string def_file_path = DM_INST.getHelper().get_def_file_path();
    ir_dataset_path
        = RTUtil::getString(DM_INST.getConfig().ir_temp_directory_path, RTUtil::splitString(def_file_path, '/').back(), ".gr.txt");
    ir_dataset = RTUtil::getOutputFileStream(ir_dataset_path);
    RTUtil::pushStream(ir_dataset, "def_file_path", " ", def_file_path, "\n");
  }
  RTUtil::pushStream(ir_dataset, "net", " ", ir_net.get_net_idx(), "\n");
  RTUtil::pushStream(ir_dataset, "{", "\n");
  RTUtil::pushStream(ir_dataset, "pin_list", "\n");
  for (IRPin& ir_pin : ir_net.get_ir_pin_list()) {
    // RTUtil::pushStream(ir_dataset, "pin", " ", ir_pin.get_pin_idx(), "\n");
    // LayerCoord coord = ir_pin.get_protected_access_point().getGridLayerCoord();
    // RTUtil::pushStream(ir_dataset, coord.get_x(), " ", coord.get_y(), " ", coord.get_layer_idx(), "\n");
  }
  RTUtil::pushStream(ir_dataset, "cost_map", "\n");
  std::vector<GridMap<IRNode>>& layer_node_map = ir_model.get_layer_node_map();
  BoundingBox& bounding_box = ir_net.get_bounding_box();
  for (irt_int layer_idx = 0; layer_idx < static_cast<irt_int>(layer_node_map.size()); layer_idx++) {
    for (irt_int x = bounding_box.get_grid_lb_x(); x <= bounding_box.get_grid_rt_x(); x++) {
      for (irt_int y = bounding_box.get_grid_lb_y(); y <= bounding_box.get_grid_rt_y(); y++) {
        IRNode& ir_node = layer_node_map[layer_idx][x][y];
        double east_cost = -1;
        double west_cost = -1;
        double south_cost = -1;
        double north_cost = -1;
        double up_cost = -1;
        double down_cost = -1;
        if (routing_layer_list[layer_idx].isPreferH()) {
          east_cost = getNodeCost(ir_model, &ir_node, Orientation::kEast);
          west_cost = getNodeCost(ir_model, &ir_node, Orientation::kWest);
        } else {
          south_cost = getNodeCost(ir_model, &ir_node, Orientation::kSouth);
          north_cost = getNodeCost(ir_model, &ir_node, Orientation::kNorth);
        }
        if (layer_idx != 0) {
          down_cost = getNodeCost(ir_model, &ir_node, Orientation::kDown);
        }
        if (layer_idx != (static_cast<irt_int>(layer_node_map.size()) - 1)) {
          up_cost = getNodeCost(ir_model, &ir_node, Orientation::kUp);
        }
        RTUtil::pushStream(ir_dataset, x, " ", y, " ", layer_idx);
        RTUtil::pushStream(ir_dataset, " ", "E", " ", east_cost);
        RTUtil::pushStream(ir_dataset, " ", "W", " ", west_cost);
        RTUtil::pushStream(ir_dataset, " ", "S", " ", south_cost);
        RTUtil::pushStream(ir_dataset, " ", "N", " ", north_cost);
        RTUtil::pushStream(ir_dataset, " ", "U", " ", up_cost);
        RTUtil::pushStream(ir_dataset, " ", "D", " ", down_cost);
        RTUtil::pushStream(ir_dataset, "\n");
      }
    }
  }
  RTUtil::pushStream(ir_dataset, "}", "\n");

  written_net_num++;
  if (written_net_num % 10000 == 0) {
    LOG_INST.info(Loc::current(), "Written ", written_net_num, " nets");
  }
  if (written_net_num == ir_model.get_ir_net_list().size()) {
    LOG_INST.info(Loc::current(), "Written ", written_net_num, " nets");
    RTUtil::closeFileStream(ir_dataset);
    LOG_INST.info(Loc::current(), "The result has been written to '", ir_dataset_path, "'!");
    exit(0);
  }
}

void InitialRouter::initSingleNet(IRModel& ir_model, IRNet& ir_net)
{
  EXTPlanarRect& die = DM_INST.getDatabase().get_die();

  std::vector<GridMap<IRNode>>& layer_node_map = ir_model.get_layer_node_map();

  ir_model.set_ir_net_ref(&ir_net);
  if (ir_model.get_curr_iter() == 1) {
    ir_model.set_routing_region(ir_model.get_curr_bounding_box());
  } else {
    ir_model.set_routing_region(die.get_grid_rect());
  }
  ir_model.get_ir_task_list().clear();
  ir_model.get_node_segment_list().clear();

  if (ir_model.get_curr_iter() == 1) {
    irt_int bottom_routing_layer_idx = DM_INST.getConfig().bottom_routing_layer_idx;
    irt_int top_routing_layer_idx = DM_INST.getConfig().top_routing_layer_idx;

    std::vector<PlanarCoord> planar_coord_list;
    for (IRPin& ir_pin : ir_net.get_ir_pin_list()) {
      // planar_coord_list.push_back(ir_pin.get_protected_access_point().get_grid_coord());
    }
    std::sort(planar_coord_list.begin(), planar_coord_list.end(), CmpPlanarCoordByXASC());
    planar_coord_list.erase(std::unique(planar_coord_list.begin(), planar_coord_list.end()), planar_coord_list.end());

    if (planar_coord_list.size() == 1) {
      IRTask ir_task;
      for (IRPin& ir_pin : ir_net.get_ir_pin_list()) {
        // IRGroup ir_group;
        // LayerCoord coord = ir_pin.get_protected_access_point().getGridLayerCoord();
        // ir_group.get_ir_node_list().push_back(&layer_node_map[coord.get_layer_idx()][coord.get_x()][coord.get_y()]);
        // ir_task.get_ir_group_list().push_back(ir_group);
      }
      ir_model.get_ir_task_list().push_back(ir_task);
    } else {
      // pin的IRGroup
      std::map<PlanarCoord, std::vector<IRGroup>, CmpPlanarCoordByXASC> key_planar_group_map;
      for (IRPin& ir_pin : ir_net.get_ir_pin_list()) {
        // IRGroup ir_group;
        // LayerCoord coord = ir_pin.get_protected_access_point().getGridLayerCoord();
        // ir_group.get_ir_node_list().push_back(&layer_node_map[coord.get_layer_idx()][coord.get_x()][coord.get_y()]);
        // key_planar_group_map[coord].push_back(ir_group);
      }

      // steiner point的IRGroup
      std::vector<Segment<PlanarCoord>> planar_topo_list = getPlanarTopoListByFlute(planar_coord_list);
      std::map<PlanarCoord, std::set<LayerCoord, CmpLayerCoordByLayerASC>, CmpPlanarCoordByXASC> add_planar_layer_map;
      for (Segment<PlanarCoord>& planar_topo : planar_topo_list) {
        if (!RTUtil::exist(key_planar_group_map, planar_topo.get_first())) {
          for (irt_int layer_idx = 0; layer_idx < static_cast<irt_int>(layer_node_map.size()); layer_idx++) {
            if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
              continue;
            }
            add_planar_layer_map[planar_topo.get_first()].insert(LayerCoord(planar_topo.get_first(), layer_idx));
          }
        }
        if (!RTUtil::exist(key_planar_group_map, planar_topo.get_second())) {
          for (irt_int layer_idx = 0; layer_idx < static_cast<irt_int>(layer_node_map.size()); layer_idx++) {
            if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
              continue;
            }
            add_planar_layer_map[planar_topo.get_second()].insert(LayerCoord(planar_topo.get_second(), layer_idx));
          }
        }
      }
      // 补充steiner point的垂直线段
      for (auto& [add_planar_coord, layer_coord_set] : add_planar_layer_map) {
        LayerCoord first_coord = *layer_coord_set.begin();
        LayerCoord second_coord = *layer_coord_set.rbegin();
        if (first_coord == second_coord) {
          continue;
        }
        IRNode* first_node = &layer_node_map[first_coord.get_layer_idx()][first_coord.get_x()][first_coord.get_y()];
        IRNode* second_node = &layer_node_map[second_coord.get_layer_idx()][second_coord.get_x()][second_coord.get_y()];
        ir_model.get_node_segment_list().emplace_back(first_node, second_node);
      }
      // 生成task
      for (Segment<PlanarCoord>& planar_topo : planar_topo_list) {
        IRTask ir_task;

        if (RTUtil::exist(key_planar_group_map, planar_topo.get_first())) {
          for (IRGroup& ir_group : key_planar_group_map[planar_topo.get_first()]) {
            ir_task.get_ir_group_list().push_back(ir_group);
          }
        } else if (RTUtil::exist(add_planar_layer_map, planar_topo.get_first())) {
          IRGroup ir_group;
          for (LayerCoord coord : add_planar_layer_map[planar_topo.get_first()]) {
            ir_group.get_ir_node_list().push_back(&layer_node_map[coord.get_layer_idx()][coord.get_x()][coord.get_y()]);
          }
          ir_task.get_ir_group_list().push_back(ir_group);
        }

        if (RTUtil::exist(key_planar_group_map, planar_topo.get_second())) {
          for (IRGroup& ir_group : key_planar_group_map[planar_topo.get_second()]) {
            ir_task.get_ir_group_list().push_back(ir_group);
          }
        } else if (RTUtil::exist(add_planar_layer_map, planar_topo.get_second())) {
          IRGroup ir_group;
          for (LayerCoord coord : add_planar_layer_map[planar_topo.get_second()]) {
            ir_group.get_ir_node_list().push_back(&layer_node_map[coord.get_layer_idx()][coord.get_x()][coord.get_y()]);
          }
          ir_task.get_ir_group_list().push_back(ir_group);
        }
        ir_model.get_ir_task_list().push_back(ir_task);
      }
    }
  } else {
    IRTask ir_task;
    for (IRPin& ir_pin : ir_net.get_ir_pin_list()) {
      // IRGroup ir_group;
      // LayerCoord coord = ir_pin.get_protected_access_point().getGridLayerCoord();
      // ir_group.get_ir_node_list().push_back(&layer_node_map[coord.get_layer_idx()][coord.get_x()][coord.get_y()]);
      // ir_task.get_ir_group_list().push_back(ir_group);
    }
    ir_model.get_ir_task_list().push_back(ir_task);
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

void InitialRouter::initSingleTask(IRModel& ir_model, IRTask& ir_task)
{
  std::vector<IRGroup>& start_group_list = ir_model.get_start_group_list();
  std::vector<IRGroup>& end_group_list = ir_model.get_end_group_list();

  std::vector<IRGroup>& ir_group_list = ir_task.get_ir_group_list();
  start_group_list.push_back(ir_group_list[0]);
  for (size_t i = 1; i < ir_group_list.size(); i++) {
    end_group_list.push_back(ir_group_list[i]);
  }
}

bool InitialRouter::isConnectedAllEnd(IRModel& ir_model)
{
  return ir_model.get_end_group_list().empty();
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
  std::vector<IRGroup>& start_group_list = ir_model.get_start_group_list();
  IRGroup& path_group = ir_model.get_path_group();

  for (IRGroup& start_group : start_group_list) {
    for (IRNode* start_node : start_group.get_ir_node_list()) {
      start_node->set_estimated_cost(getEstimateCostToEnd(ir_model, start_node));
      pushToOpenList(ir_model, start_node);
    }
  }
  for (IRNode* path_node : path_group.get_ir_node_list()) {
    path_node->set_estimated_cost(getEstimateCostToEnd(ir_model, path_node));
    pushToOpenList(ir_model, path_node);
  }
  resetPathHead(ir_model);
}

bool InitialRouter::searchEnded(IRModel& ir_model)
{
  std::vector<IRGroup>& end_group_list = ir_model.get_end_group_list();
  IRNode* path_head_node = ir_model.get_path_head_node();

  if (path_head_node == nullptr) {
    ir_model.set_end_group_idx(-1);
    return true;
  }
  for (size_t i = 0; i < end_group_list.size(); i++) {
    for (IRNode* end_node : end_group_list[i].get_ir_node_list()) {
      if (path_head_node == end_node) {
        ir_model.set_end_group_idx(static_cast<irt_int>(i));
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

  for (auto& [orientation, neighbor_node] : path_head_node->get_neighbor_ptr_map()) {
    if (neighbor_node == nullptr) {
      continue;
    }
    if (!RTUtil::isInside(ir_model.get_routing_region(), *neighbor_node)) {
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
  return ir_model.get_end_group_idx() == -1;
}

void InitialRouter::resetSinglePath(IRModel& ir_model)
{
  PriorityQueue<IRNode*, std::vector<IRNode*>, CmpIRNodeCost> empty_queue;
  ir_model.set_open_queue(empty_queue);

  std::vector<IRNode*>& visited_node_list = ir_model.get_visited_node_list();
  for (IRNode* visited_node : visited_node_list) {
    visited_node->set_state(IRNodeState::kNone);
    visited_node->set_parent_node(nullptr);
    visited_node->set_known_cost(0);
    visited_node->set_estimated_cost(0);
  }
  visited_node_list.clear();

  ir_model.set_path_head_node(nullptr);
  ir_model.set_end_group_idx(-1);
}

void InitialRouter::updatePathResult(IRModel& ir_model)
{
  std::vector<Segment<IRNode*>>& node_segment_list = ir_model.get_node_segment_list();
  IRNode* path_head_node = ir_model.get_path_head_node();

  IRNode* curr_node = path_head_node;
  IRNode* pre_node = curr_node->get_parent_node();

  if (pre_node == nullptr) {
    // 起点和终点重合
    return;
  }
  Orientation curr_orientation = RTUtil::getOrientation(*curr_node, *pre_node);
  while (pre_node->get_parent_node() != nullptr) {
    Orientation pre_orientation = RTUtil::getOrientation(*pre_node, *pre_node->get_parent_node());
    if (curr_orientation != pre_orientation) {
      node_segment_list.emplace_back(curr_node, pre_node);
      curr_orientation = pre_orientation;
      curr_node = pre_node;
    }
    pre_node = pre_node->get_parent_node();
  }
  node_segment_list.emplace_back(curr_node, pre_node);
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
  std::vector<IRGroup>& start_group_list = ir_model.get_start_group_list();
  std::vector<IRGroup>& end_group_list = ir_model.get_end_group_list();
  IRGroup& path_group = ir_model.get_path_group();
  IRNode* path_head_node = ir_model.get_path_head_node();
  irt_int end_group_idx = ir_model.get_end_group_idx();

  end_group_list[end_group_idx].get_ir_node_list().clear();
  end_group_list[end_group_idx].get_ir_node_list().push_back(path_head_node);

  IRNode* path_node = path_head_node->get_parent_node();
  if (path_node == nullptr) {
    // 起点和终点重合
    path_node = path_head_node;
  } else {
    // 起点和终点不重合
    while (path_node->get_parent_node() != nullptr) {
      path_group.get_ir_node_list().push_back(path_node);
      path_node = path_node->get_parent_node();
    }
  }
  if (start_group_list.size() == 1) {
    // 初始化时，要把start_group_list的pin只留一个ap点
    // 后续只要将end_group_list的pin保留一个ap点
    start_group_list.front().get_ir_node_list().clear();
    start_group_list.front().get_ir_node_list().push_back(path_node);
  }
  start_group_list.push_back(end_group_list[end_group_idx]);
  end_group_list.erase(end_group_list.begin() + end_group_idx);
}

void InitialRouter::resetSingleTask(IRModel& ir_model)
{
  ir_model.get_start_group_list().clear();
  ir_model.get_end_group_list().clear();
  ir_model.set_path_group(IRGroup());
}

void InitialRouter::updateNetResult(IRModel& ir_model, IRNet& ir_net)
{
  updateRoutingTree(ir_model, ir_net);
  updateDemand(ir_model, ir_net, ChangeType::kAdd);
  ir_net.set_routing_state(RoutingState::kRouted);
}

void InitialRouter::updateRoutingTree(IRModel& ir_model, IRNet& ir_net)
{
  // std::vector<Segment<IRNode*>>& node_segment_list = ir_model.get_node_segment_list();

  // std::vector<Segment<LayerCoord>> routing_segment_list;
  // for (Segment<IRNode*>& node_segment : node_segment_list) {
  //   routing_segment_list.emplace_back(*node_segment.get_first(), *node_segment.get_second());
  // }
  // LayerCoord root_coord = ir_net.get_ir_driving_pin().get_protected_access_point().getGridLayerCoord();
  // std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC> key_coord_pin_map;
  // for (IRPin& ir_pin : ir_net.get_ir_pin_list()) {
  //   LayerCoord coord = ir_pin.get_protected_access_point().getGridLayerCoord();
  //   key_coord_pin_map[coord].insert(ir_pin.get_pin_idx());
  // }
  // ir_net.set_routing_tree(RTUtil::getTreeByFullFlow({root_coord}, routing_segment_list, key_coord_pin_map));
}

void InitialRouter::updateDemand(IRModel& ir_model, IRNet& ir_net, ChangeType change_type)
{
  // std::vector<GridMap<IRNode>>& layer_node_map = ir_model.get_layer_node_map();

  // std::set<IRNode*> key_node_set;
  // for (IRPin& ir_pin : ir_net.get_ir_pin_list()) {
  //   LayerCoord coord = ir_pin.get_protected_access_point().getGridLayerCoord();
  //   IRNode* ir_node = &layer_node_map[coord.get_layer_idx()][coord.get_x()][coord.get_y()];
  //   key_node_set.insert(ir_node);
  // }
  // std::vector<Segment<IRNode*>> node_segment_list;
  // for (Segment<TNode<LayerCoord>*>& coord_segment : RTUtil::getSegListByTree(ir_net.get_routing_tree())) {
  //   LayerCoord first_coord = coord_segment.get_first()->value();
  //   LayerCoord second_coord = coord_segment.get_second()->value();

  //   IRNode* first_node = &layer_node_map[first_coord.get_layer_idx()][first_coord.get_x()][first_coord.get_y()];
  //   IRNode* second_node = &layer_node_map[second_coord.get_layer_idx()][second_coord.get_x()][second_coord.get_y()];

  //   node_segment_list.emplace_back(first_node, second_node);
  // }

  // std::map<IRNode*, std::set<Orientation>> usage_map;

  // if (node_segment_list.empty()) {
  //   // 单层的local net
  //   if (key_node_set.size() > 1) {
  //     LOG_INST.error(Loc::current(), "The net is not local!");
  //   }
  //   IRNode* local_node = *key_node_set.begin();
  //   for (Orientation orientation : {Orientation::kUp, Orientation::kDown}) {
  //     usage_map[local_node].insert(orientation);
  //   }
  // } else {
  //   // 跨gcell线网和多层的local_net
  //   for (Segment<IRNode*>& node_segment : node_segment_list) {
  //     IRNode* first_node = node_segment.get_first();
  //     IRNode* second_node = node_segment.get_second();
  //     Orientation orientation = RTUtil::getOrientation(*first_node, *second_node);
  //     if (orientation == Orientation::kNone || orientation == Orientation::kOblique) {
  //       LOG_INST.error(Loc::current(), "The orientation is error!");
  //     }
  //     Orientation oppo_orientation = RTUtil::getOppositeOrientation(orientation);

  //     IRNode* node_i = first_node;
  //     while (true) {
  //       if (node_i != first_node) {
  //         usage_map[node_i].insert(oppo_orientation);
  //       }
  //       if (node_i != second_node) {
  //         usage_map[node_i].insert(orientation);
  //       }
  //       if (node_i == second_node) {
  //         break;
  //       }
  //       node_i = node_i->getNeighborNode(orientation);
  //     }
  //   }
  // }
  // for (auto& [usage_node, orientation_list] : usage_map) {
  //   usage_node->updateDemand(ir_net.get_net_idx(), orientation_list, change_type);
  // }
}

void InitialRouter::resetSingleNet(IRModel& ir_model)
{
  ir_model.set_ir_net_ref(nullptr);
  ir_model.set_routing_region(PlanarRect());
  ir_model.get_ir_task_list().clear();

  for (Segment<IRNode*>& node_segment : ir_model.get_node_segment_list()) {
    IRNode* first_node = node_segment.get_first();
    IRNode* second_node = node_segment.get_second();
    Orientation orientation = RTUtil::getOrientation(*first_node, *second_node);

    IRNode* node_i = first_node;
    while (true) {
      node_i->get_direction_set().clear();
      if (node_i == second_node) {
        break;
      }
      node_i = node_i->getNeighborNode(orientation);
    }
  }
  ir_model.get_node_segment_list().clear();
}

// manager open list

void InitialRouter::pushToOpenList(IRModel& ir_model, IRNode* curr_node)
{
  PriorityQueue<IRNode*, std::vector<IRNode*>, CmpIRNodeCost>& open_queue = ir_model.get_open_queue();
  std::vector<IRNode*>& visited_node_list = ir_model.get_visited_node_list();

  open_queue.push(curr_node);
  curr_node->set_state(IRNodeState::kOpen);
  visited_node_list.push_back(curr_node);
}

IRNode* InitialRouter::popFromOpenList(IRModel& ir_model)
{
  PriorityQueue<IRNode*, std::vector<IRNode*>, CmpIRNodeCost>& open_queue = ir_model.get_open_queue();

  IRNode* ir_node = nullptr;
  if (!open_queue.empty()) {
    ir_node = open_queue.top();
    open_queue.pop();
    ir_node->set_state(IRNodeState::kClose);
  }
  return ir_node;
}

// calculate known cost

double InitialRouter::getKnowCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node)
{
  bool exist_neighbor = false;
  for (auto& [orientation, neighbor_ptr] : start_node->get_neighbor_ptr_map()) {
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
#if 1
  double node_cost = 0;

  double env_cost = curr_node->getCost(ir_model.get_curr_net_idx(), orientation);
  node_cost += env_cost;

  return node_cost;
#else
  double env_cost = curr_node->getCost(ir_model.get_curr_net_idx(), orientation);

  double net_cost = 0;
  {
    const PlanarRect& curr_bounding_box = ir_model.get_curr_bounding_box();
    const GridMap<double>& curr_cost_map = ir_model.get_curr_cost_map();

    irt_int local_x = curr_node->get_x() - curr_bounding_box.get_lb_x();
    irt_int local_y = curr_node->get_y() - curr_bounding_box.get_lb_y();
    net_cost = (curr_cost_map.isInside(local_x, local_y) ? curr_cost_map[local_x][local_y] : 1);
  }

  double node_cost = env_cost + net_cost;
  return node_cost;
#endif
}

double InitialRouter::getKnowWireCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node)
{
  double ir_prefer_wire_unit = DM_INST.getConfig().ir_prefer_wire_unit;

  double wire_cost = 0;
  wire_cost += RTUtil::getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());
  wire_cost *= ir_prefer_wire_unit;
  return wire_cost;
}

double InitialRouter::getKnowCornerCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node)
{
  double ir_corner_unit = DM_INST.getConfig().ir_corner_unit;

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
      corner_cost += ir_corner_unit;
    } else if (direction_set.size() == 2) {
      LOG_INST.error(Loc::current(), "Direction set is error!");
    }
  }
  return corner_cost;
}

double InitialRouter::getKnowViaCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node)
{
  double ir_via_unit = DM_INST.getConfig().ir_via_unit;

  double via_cost = (ir_via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
  return via_cost;
}

// calculate estimate cost

double InitialRouter::getEstimateCostToEnd(IRModel& ir_model, IRNode* curr_node)
{
  std::vector<IRGroup>& end_group_list = ir_model.get_end_group_list();

  double estimate_cost = DBL_MAX;
  for (IRGroup& end_group : end_group_list) {
    for (IRNode* end_node : end_group.get_ir_node_list()) {
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
  double ir_prefer_wire_unit = DM_INST.getConfig().ir_prefer_wire_unit;

  double wire_cost = 0;
  wire_cost += RTUtil::getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());
  wire_cost *= ir_prefer_wire_unit;
  return wire_cost;
}

double InitialRouter::getEstimateCornerCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node)
{
  double ir_corner_unit = DM_INST.getConfig().ir_corner_unit;

  double corner_cost = 0;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    if (RTUtil::isOblique(*start_node, *end_node)) {
      corner_cost += ir_corner_unit;
    }
  }
  return corner_cost;
}

double InitialRouter::getEstimateViaCost(IRModel& ir_model, IRNode* start_node, IRNode* end_node)
{
  double ir_via_unit = DM_INST.getConfig().ir_via_unit;

  double via_cost = (ir_via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
  return via_cost;
}

void InitialRouter::processIRModel(IRModel& ir_model)
{
  // 检查布线状态
  for (IRNet& ir_net : ir_model.get_ir_net_list()) {
    if (ir_net.get_routing_state() == RoutingState::kUnrouted) {
      LOG_INST.error(Loc::current(), "The routing_state is ", GetRoutingStateName()(ir_net.get_routing_state()), "!");
    }
  }
#pragma omp parallel for
  for (IRNet& ir_net : ir_model.get_ir_net_list()) {
    buildRoutingResult(ir_net);
  }
}

void InitialRouter::buildRoutingResult(IRNet& ir_net)
{
  if (ir_net.get_routing_tree().get_root() == nullptr) {
    return;
  }
  std::function<Guide(LayerCoord&)> convertToGuide = [](LayerCoord& coord) {
    ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
    return Guide(LayerRect(RTUtil::getRealRectByGCell(coord, gcell_axis), coord.get_layer_idx()), coord);
  };
  ir_net.set_ir_result_tree(RTUtil::convertTree(ir_net.get_routing_tree(), convertToGuide));
}

bool InitialRouter::stopIRModel(IRModel& ir_model)
{
  return (ir_model.get_ir_model_stat().get_max_access_overflow() <= 1);
}

#endif

#if 1  // update

void InitialRouter::update(IRModel& ir_model)
{
  outputGuide(ir_model);
  for (IRNet& ir_net : ir_model.get_ir_net_list()) {
    Net* origin_net = ir_net.get_origin_net();
    origin_net->set_ir_result_tree(ir_net.get_ir_result_tree());
  }
}

void InitialRouter::outputGuide(IRModel& ir_model)
{
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

#endif

#if 1  // plot ir_model

void InitialRouter::outputCongestionMap(IRModel& ir_model)
{
  Die& die = DM_INST.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::string ir_temp_directory_path = DM_INST.getConfig().ir_temp_directory_path;

  GridMap<double> planar_overflow_map;
  planar_overflow_map.init(die.getXSize(), die.getYSize());

  for (RoutingLayer& routing_layer : routing_layer_list) {
    GridMap<IRNode>& node_map = ir_model.get_layer_node_map()[routing_layer.get_layer_idx()];
    for (irt_int grid_x = 0; grid_x < node_map.get_x_size(); grid_x++) {
      for (irt_int grid_y = 0; grid_y < node_map.get_y_size(); grid_y++) {
        IRNode& ir_node = node_map[grid_x][grid_y];
        std::map<Orientation, irt_int>& orien_access_supply_map = ir_node.get_orien_access_supply_map();
        std::map<Orientation, irt_int>& orien_access_demand_map = ir_node.get_orien_access_demand_map();

        double overflow = 0;
        if (routing_layer.isPreferH()) {
          for (Orientation orientation : {Orientation::kEast, Orientation::kWest}) {
            overflow += (orien_access_demand_map[orientation] - orien_access_supply_map[orientation]);
          }
        } else {
          for (Orientation orientation : {Orientation::kSouth, Orientation::kNorth}) {
            overflow += (orien_access_demand_map[orientation] - orien_access_supply_map[orientation]);
          }
        }
        overflow /= 2;
        planar_overflow_map[grid_x][grid_y] = std::max(overflow, planar_overflow_map[grid_x][grid_y]);
      }
    }
  }

  std::ofstream* csv_file
      = RTUtil::getOutputFileStream(RTUtil::getString(ir_temp_directory_path, "ir_model_", ir_model.get_curr_iter(), ".csv"));
  for (irt_int y = planar_overflow_map.get_y_size() - 1; y >= 0; y--) {
    for (irt_int x = 0; x < planar_overflow_map.get_x_size(); x++) {
      RTUtil::pushStream(csv_file, planar_overflow_map[x][y], ",");
    }
    RTUtil::pushStream(csv_file, "\n");
  }
  RTUtil::closeFileStream(csv_file);
}

void InitialRouter::plotIRModel(IRModel& ir_model, irt_int curr_net_idx)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  Die& die = DM_INST.getDatabase().get_die();
  std::string ir_temp_directory_path = DM_INST.getConfig().ir_temp_directory_path;

  GPGDS gp_gds;

  // base_region
  GPStruct base_region_struct("base_region");
  GPBoundary gp_boundary;
  gp_boundary.set_layer_idx(0);
  gp_boundary.set_data_type(0);
  gp_boundary.set_rect(die.get_real_rect());
  base_region_struct.push(gp_boundary);
  gp_gds.addStruct(base_region_struct);

  // gcell_axis
  GPStruct gcell_axis_struct("gcell_axis");
  std::vector<irt_int> gcell_x_list = RTUtil::getClosedScaleList(die.get_real_lb_x(), die.get_real_rt_x(), gcell_axis.get_x_grid_list());
  std::vector<irt_int> gcell_y_list = RTUtil::getClosedScaleList(die.get_real_lb_y(), die.get_real_rt_y(), gcell_axis.get_y_grid_list());
  for (irt_int x : gcell_x_list) {
    GPPath gp_path;
    gp_path.set_layer_idx(0);
    gp_path.set_data_type(1);
    gp_path.set_segment(x, die.get_real_lb_y(), x, die.get_real_rt_y());
    gcell_axis_struct.push(gp_path);
  }
  for (irt_int y : gcell_y_list) {
    GPPath gp_path;
    gp_path.set_layer_idx(0);
    gp_path.set_data_type(1);
    gp_path.set_segment(die.get_real_lb_x(), y, die.get_real_rt_x(), y);
    gcell_axis_struct.push(gp_path);
  }
  gp_gds.addStruct(gcell_axis_struct);

  // ir_node_map
  GPStruct ir_node_map_struct("ir_node_map");
  for (GridMap<IRNode>& ir_node_map : ir_model.get_layer_node_map()) {
    for (irt_int grid_x = 0; grid_x < ir_node_map.get_x_size(); grid_x++) {
      for (irt_int grid_y = 0; grid_y < ir_node_map.get_y_size(); grid_y++) {
        IRNode& ir_node = ir_node_map[grid_x][grid_y];
        PlanarRect real_rect = RTUtil::getRealRectByGCell(ir_node.get_planar_coord(), gcell_axis);
        irt_int y_reduced_span = real_rect.getYSpan() / 25;
        irt_int y = real_rect.get_rt_y();

        GPBoundary gp_boundary;
        switch (ir_node.get_state()) {
          case IRNodeState::kNone:
            gp_boundary.set_data_type(static_cast<irt_int>(GPDataType::kNone));
            break;
          case IRNodeState::kOpen:
            gp_boundary.set_data_type(static_cast<irt_int>(GPDataType::kOpen));
            break;
          case IRNodeState::kClose:
            gp_boundary.set_data_type(static_cast<irt_int>(GPDataType::kClose));
            break;
          default:
            LOG_INST.error(Loc::current(), "The type is error!");
            break;
        }
        gp_boundary.set_rect(real_rect);
        gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(ir_node.get_layer_idx()));
        ir_node_map_struct.push(gp_boundary);

        y -= y_reduced_span;
        GPText gp_text_node_coord;
        gp_text_node_coord.set_coord(real_rect.get_lb_x(), y);
        gp_text_node_coord.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_node_coord.set_message(RTUtil::getString("(", grid_x, " , ", grid_y, " , ", ir_node.get_layer_idx(), ")"));
        gp_text_node_coord.set_layer_idx(GP_INST.getGDSIdxByRouting(ir_node.get_layer_idx()));
        gp_text_node_coord.set_presentation(GPTextPresentation::kLeftMiddle);
        ir_node_map_struct.push(gp_text_node_coord);

        y -= y_reduced_span;
        GPText gp_text_cross_wire_demand;
        gp_text_cross_wire_demand.set_coord(real_rect.get_lb_x(), y);
        gp_text_cross_wire_demand.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_cross_wire_demand.set_message(RTUtil::getString("cross_wire_demand: ", ir_node.get_cross_wire_demand()));
        gp_text_cross_wire_demand.set_layer_idx(GP_INST.getGDSIdxByRouting(ir_node.get_layer_idx()));
        gp_text_cross_wire_demand.set_presentation(GPTextPresentation::kLeftMiddle);
        ir_node_map_struct.push(gp_text_cross_wire_demand);

        y -= y_reduced_span;
        GPText gp_text_local_wire_demand;
        gp_text_local_wire_demand.set_coord(real_rect.get_lb_x(), y);
        gp_text_local_wire_demand.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_local_wire_demand.set_message(RTUtil::getString("local_wire_demand: ", ir_node.get_local_wire_demand()));
        gp_text_local_wire_demand.set_layer_idx(GP_INST.getGDSIdxByRouting(ir_node.get_layer_idx()));
        gp_text_local_wire_demand.set_presentation(GPTextPresentation::kLeftMiddle);
        ir_node_map_struct.push(gp_text_local_wire_demand);

        y -= y_reduced_span;
        GPText gp_text_whole_via_demand;
        gp_text_whole_via_demand.set_coord(real_rect.get_lb_x(), y);
        gp_text_whole_via_demand.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_whole_via_demand.set_message(RTUtil::getString("whole_via_demand: ", ir_node.get_whole_via_demand()));
        gp_text_whole_via_demand.set_layer_idx(GP_INST.getGDSIdxByRouting(ir_node.get_layer_idx()));
        gp_text_whole_via_demand.set_presentation(GPTextPresentation::kLeftMiddle);
        ir_node_map_struct.push(gp_text_whole_via_demand);

        y -= y_reduced_span;
        GPText gp_text_net_via_demand_map;
        gp_text_net_via_demand_map.set_coord(real_rect.get_lb_x(), y);
        gp_text_net_via_demand_map.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_net_via_demand_map.set_message("net_via_demand_map: ");
        gp_text_net_via_demand_map.set_layer_idx(GP_INST.getGDSIdxByRouting(ir_node.get_layer_idx()));
        gp_text_net_via_demand_map.set_presentation(GPTextPresentation::kLeftMiddle);
        ir_node_map_struct.push(gp_text_net_via_demand_map);

        if (!ir_node.get_net_via_demand_map().empty()) {
          y -= y_reduced_span;
          GPText gp_text_net_via_demand_map_info;
          gp_text_net_via_demand_map_info.set_coord(real_rect.get_lb_x(), y);
          gp_text_net_via_demand_map_info.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
          std::string net_via_demand_map_message = "--";
          for (auto& [net_idx, via_demand] : ir_node.get_net_via_demand_map()) {
            net_via_demand_map_message += RTUtil::getString("(", net_idx, ")(", via_demand, ")");
          }
          gp_text_net_via_demand_map_info.set_message(net_via_demand_map_message);
          gp_text_net_via_demand_map_info.set_layer_idx(GP_INST.getGDSIdxByRouting(ir_node.get_layer_idx()));
          gp_text_net_via_demand_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
          ir_node_map_struct.push(gp_text_net_via_demand_map_info);
        }

        y -= y_reduced_span;
        GPText gp_text_whole_access_demand;
        gp_text_whole_access_demand.set_coord(real_rect.get_lb_x(), y);
        gp_text_whole_access_demand.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_whole_access_demand.set_message(RTUtil::getString("whole_access_demand: ", ir_node.get_whole_access_demand()));
        gp_text_whole_access_demand.set_layer_idx(GP_INST.getGDSIdxByRouting(ir_node.get_layer_idx()));
        gp_text_whole_access_demand.set_presentation(GPTextPresentation::kLeftMiddle);
        ir_node_map_struct.push(gp_text_whole_access_demand);

        y -= y_reduced_span;
        GPText gp_text_net_orien_access_demand_map;
        gp_text_net_orien_access_demand_map.set_coord(real_rect.get_lb_x(), y);
        gp_text_net_orien_access_demand_map.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_net_orien_access_demand_map.set_message("net_orien_access_demand_map: ");
        gp_text_net_orien_access_demand_map.set_layer_idx(GP_INST.getGDSIdxByRouting(ir_node.get_layer_idx()));
        gp_text_net_orien_access_demand_map.set_presentation(GPTextPresentation::kLeftMiddle);
        ir_node_map_struct.push(gp_text_net_orien_access_demand_map);

        if (!ir_node.get_net_orien_access_demand_map().empty()) {
          y -= y_reduced_span;
          GPText gp_text_net_orien_access_demand_map_info;
          gp_text_net_orien_access_demand_map_info.set_coord(real_rect.get_lb_x(), y);
          gp_text_net_orien_access_demand_map_info.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
          std::string net_orien_access_demand_map_message = "--";
          for (auto& [net_idx, orien_wire_demand_map] : ir_node.get_net_orien_access_demand_map()) {
            net_orien_access_demand_map_message += RTUtil::getString("(", net_idx, ")");
            for (auto& [orientation, wire_demand] : orien_wire_demand_map) {
              net_orien_access_demand_map_message += RTUtil::getString("(", GetOrientationName()(orientation), ":", wire_demand, ")");
            }
          }
          gp_text_net_orien_access_demand_map_info.set_message(net_orien_access_demand_map_message);
          gp_text_net_orien_access_demand_map_info.set_layer_idx(GP_INST.getGDSIdxByRouting(ir_node.get_layer_idx()));
          gp_text_net_orien_access_demand_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
          ir_node_map_struct.push(gp_text_net_orien_access_demand_map_info);
        }

        y -= y_reduced_span;
        GPText gp_text_orien_access_supply_map;
        gp_text_orien_access_supply_map.set_coord(real_rect.get_lb_x(), y);
        gp_text_orien_access_supply_map.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_orien_access_supply_map.set_message("orien_access_supply_map: ");
        gp_text_orien_access_supply_map.set_layer_idx(GP_INST.getGDSIdxByRouting(ir_node.get_layer_idx()));
        gp_text_orien_access_supply_map.set_presentation(GPTextPresentation::kLeftMiddle);
        ir_node_map_struct.push(gp_text_orien_access_supply_map);

        if (!ir_node.get_orien_access_supply_map().empty()) {
          y -= y_reduced_span;
          GPText gp_text_orien_access_supply_map_info;
          gp_text_orien_access_supply_map_info.set_coord(real_rect.get_lb_x(), y);
          gp_text_orien_access_supply_map_info.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
          std::string orien_access_supply_map_message = "--";
          for (auto& [orientation, access_supply] : ir_node.get_orien_access_supply_map()) {
            orien_access_supply_map_message += RTUtil::getString("(", GetOrientationName()(orientation), ":", access_supply, ")");
          }
          gp_text_orien_access_supply_map_info.set_message(orien_access_supply_map_message);
          gp_text_orien_access_supply_map_info.set_layer_idx(GP_INST.getGDSIdxByRouting(ir_node.get_layer_idx()));
          gp_text_orien_access_supply_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
          ir_node_map_struct.push(gp_text_orien_access_supply_map_info);
        }

        y -= y_reduced_span;
        GPText gp_text_orien_access_demand_map;
        gp_text_orien_access_demand_map.set_coord(real_rect.get_lb_x(), y);
        gp_text_orien_access_demand_map.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_orien_access_demand_map.set_message("orien_access_demand_map: ");
        gp_text_orien_access_demand_map.set_layer_idx(GP_INST.getGDSIdxByRouting(ir_node.get_layer_idx()));
        gp_text_orien_access_demand_map.set_presentation(GPTextPresentation::kLeftMiddle);
        ir_node_map_struct.push(gp_text_orien_access_demand_map);

        if (!ir_node.get_orien_access_demand_map().empty()) {
          y -= y_reduced_span;
          GPText gp_text_orien_access_demand_map_info;
          gp_text_orien_access_demand_map_info.set_coord(real_rect.get_lb_x(), y);
          gp_text_orien_access_demand_map_info.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
          std::string orien_access_demand_map_message = "--";
          for (auto& [orientation, access_demand] : ir_node.get_orien_access_demand_map()) {
            orien_access_demand_map_message += RTUtil::getString("(", GetOrientationName()(orientation), ":", access_demand, ")");
          }
          gp_text_orien_access_demand_map_info.set_message(orien_access_demand_map_message);
          gp_text_orien_access_demand_map_info.set_layer_idx(GP_INST.getGDSIdxByRouting(ir_node.get_layer_idx()));
          gp_text_orien_access_demand_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
          ir_node_map_struct.push(gp_text_orien_access_demand_map_info);
        }

        y -= y_reduced_span;
        GPText gp_text_resource_supply;
        gp_text_resource_supply.set_coord(real_rect.get_lb_x(), y);
        gp_text_resource_supply.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_resource_supply.set_message(RTUtil::getString("resource_supply: ", ir_node.get_resource_supply()));
        gp_text_resource_supply.set_layer_idx(GP_INST.getGDSIdxByRouting(ir_node.get_layer_idx()));
        gp_text_resource_supply.set_presentation(GPTextPresentation::kLeftMiddle);
        ir_node_map_struct.push(gp_text_resource_supply);

        y -= y_reduced_span;
        GPText gp_text_resource_demand;
        gp_text_resource_demand.set_coord(real_rect.get_lb_x(), y);
        gp_text_resource_demand.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_resource_demand.set_message(RTUtil::getString("resource_demand: ", ir_node.get_resource_demand()));
        gp_text_resource_demand.set_layer_idx(GP_INST.getGDSIdxByRouting(ir_node.get_layer_idx()));
        gp_text_resource_demand.set_presentation(GPTextPresentation::kLeftMiddle);
        ir_node_map_struct.push(gp_text_resource_demand);

        y -= y_reduced_span;
        GPText gp_text_passed_net_set;
        gp_text_passed_net_set.set_coord(real_rect.get_lb_x(), y);
        gp_text_passed_net_set.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_passed_net_set.set_message("passed_net_set: ");
        gp_text_passed_net_set.set_layer_idx(GP_INST.getGDSIdxByRouting(ir_node.get_layer_idx()));
        gp_text_passed_net_set.set_presentation(GPTextPresentation::kLeftMiddle);
        ir_node_map_struct.push(gp_text_passed_net_set);

        if (!ir_node.get_passed_net_set().empty()) {
          y -= y_reduced_span;
          GPText gp_text_passed_net_set_info;
          gp_text_passed_net_set_info.set_coord(real_rect.get_lb_x(), y);
          gp_text_passed_net_set_info.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
          std::string passed_net_set_info_message = "--";
          for (irt_int net_idx : ir_node.get_passed_net_set()) {
            passed_net_set_info_message += RTUtil::getString("(", net_idx, ")");
          }
          gp_text_passed_net_set_info.set_message(passed_net_set_info_message);
          gp_text_passed_net_set_info.set_layer_idx(GP_INST.getGDSIdxByRouting(ir_node.get_layer_idx()));
          gp_text_passed_net_set_info.set_presentation(GPTextPresentation::kLeftMiddle);
          ir_node_map_struct.push(gp_text_passed_net_set_info);
        }

        y -= y_reduced_span;
        GPText gp_text_direction_set;
        gp_text_direction_set.set_coord(real_rect.get_lb_x(), y);
        gp_text_direction_set.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_direction_set.set_message("direction_set: ");
        gp_text_direction_set.set_layer_idx(GP_INST.getGDSIdxByRouting(ir_node.get_layer_idx()));
        gp_text_direction_set.set_presentation(GPTextPresentation::kLeftMiddle);
        ir_node_map_struct.push(gp_text_direction_set);

        if (!ir_node.get_direction_set().empty()) {
          y -= y_reduced_span;
          GPText gp_text_direction_set_info;
          gp_text_direction_set_info.set_coord(real_rect.get_lb_x(), y);
          gp_text_direction_set_info.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
          std::string direction_set_info_message = "--";
          for (Direction direction : ir_node.get_direction_set()) {
            direction_set_info_message += RTUtil::getString("(", GetDirectionName()(direction), ")");
          }
          gp_text_direction_set_info.set_message(direction_set_info_message);
          gp_text_direction_set_info.set_layer_idx(GP_INST.getGDSIdxByRouting(ir_node.get_layer_idx()));
          gp_text_direction_set_info.set_presentation(GPTextPresentation::kLeftMiddle);
          ir_node_map_struct.push(gp_text_direction_set_info);
        }
      }
    }
  }
  gp_gds.addStruct(ir_node_map_struct);

  // neighbor_map
  GPStruct neighbor_map_struct("neighbor_map");
  for (GridMap<IRNode>& ir_node_map : ir_model.get_layer_node_map()) {
    for (irt_int grid_x = 0; grid_x < ir_node_map.get_x_size(); grid_x++) {
      for (irt_int grid_y = 0; grid_y < ir_node_map.get_y_size(); grid_y++) {
        IRNode& ir_node = ir_node_map[grid_x][grid_y];
        PlanarRect real_rect = RTUtil::getRealRectByGCell(ir_node.get_planar_coord(), gcell_axis);
        irt_int lb_x = real_rect.get_lb_x();
        irt_int lb_y = real_rect.get_lb_y();
        irt_int rt_x = real_rect.get_rt_x();
        irt_int rt_y = real_rect.get_rt_y();
        irt_int mid_x = (lb_x + rt_x) / 2;
        irt_int mid_y = (lb_y + rt_y) / 2;
        irt_int x_reduced_span = (rt_x - lb_x) / 4;
        irt_int y_reduced_span = (rt_y - lb_y) / 4;
        irt_int width = std::min(x_reduced_span, y_reduced_span) / 2;

        for (auto& [orientation, neighbor_node] : ir_node.get_neighbor_ptr_map()) {
          GPPath gp_path;
          switch (orientation) {
            case Orientation::kEast:
              gp_path.set_segment(rt_x - x_reduced_span, mid_y, rt_x, mid_y);
              break;
            case Orientation::kSouth:
              gp_path.set_segment(mid_x, lb_y, mid_x, lb_y + y_reduced_span);
              break;
            case Orientation::kWest:
              gp_path.set_segment(lb_x, mid_y, lb_x + x_reduced_span, mid_y);
              break;
            case Orientation::kNorth:
              gp_path.set_segment(mid_x, rt_y - y_reduced_span, mid_x, rt_y);
              break;
            case Orientation::kUp:
              gp_path.set_segment(rt_x - x_reduced_span, rt_y - y_reduced_span, rt_x, rt_y);
              break;
            case Orientation::kDown:
              gp_path.set_segment(lb_x, lb_y, lb_x + x_reduced_span, lb_y + y_reduced_span);
              break;
            default:
              LOG_INST.error(Loc::current(), "The orientation is oblique!");
              break;
          }
          gp_path.set_layer_idx(GP_INST.getGDSIdxByRouting(ir_node.get_layer_idx()));
          gp_path.set_width(width);
          gp_path.set_data_type(static_cast<irt_int>(GPDataType::kNeighbor));
          neighbor_map_struct.push(gp_path);
        }
      }
    }
  }
  gp_gds.addStruct(neighbor_map_struct);

  // source_region_query_map
  // std::vector<std::pair<IRSourceType, GPDataType>> source_graph_pair_list = {{IRSourceType::kBlockage, GPDataType::kBlockage},
  //                                                                             {IRSourceType::kNetShape, GPDataType::kNetShape},
  //                                                                             {IRSourceType::kReservedVia, GPDataType::kReservedVia}};
  // std::vector<GridMap<IRNode>>& layer_node_map = ir_model.get_layer_node_map();
  // for (irt_int layer_idx = 0; layer_idx < static_cast<irt_int>(layer_node_map.size()); layer_idx++) {
  //   GridMap<IRNode>& node_map = layer_node_map[layer_idx];
  //   for (irt_int grid_x = 0; grid_x < node_map.get_x_size(); grid_x++) {
  //     for (irt_int grid_y = 0; grid_y < node_map.get_y_size(); grid_y++) {
  //       IRNode& ir_node = node_map[grid_x][grid_y];
  //       for (auto& [ir_source_type, gp_graph_type] : source_graph_pair_list) {
  //         for (auto& [info, rect_set] : DC_INST.getLayerInfoRectMap(ir_node.getRegionQuery(ir_source_type), true)[layer_idx]) {
  //           GPStruct net_rect_struct(RTUtil::getString(GetIRSourceTypeName()(ir_source_type), "@", info.get_net_idx()));
  //           for (const LayerRect& rect : rect_set) {
  //             GPBoundary gp_boundary;
  //             gp_boundary.set_data_type(static_cast<irt_int>(gp_graph_type));
  //             gp_boundary.set_rect(rect);
  //             gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(rect.get_layer_idx()));
  //             net_rect_struct.push(gp_boundary);
  //           }
  //           gp_gds.addStruct(net_rect_struct);
  //         }
  //       }
  //     }
  //   }
  // }

  // net
  // for (IRNet& ir_net : ir_model.get_ir_net_list()) {
  //   GPStruct net_struct(RTUtil::getString("net_", ir_net.get_net_idx()));

  //   if (curr_net_idx == -1 || ir_net.get_net_idx() == curr_net_idx) {
  //     for (IRPin& ir_pin : ir_net.get_ir_pin_list()) {
  //       LayerCoord coord = ir_pin.get_protected_access_point().getGridLayerCoord();
  //       PlanarRect real_rect = RTUtil::getRealRectByGCell(coord.get_planar_coord(), gcell_axis);

  //       GPBoundary gp_boundary;
  //       gp_boundary.set_data_type(static_cast<irt_int>(GPDataType::kKey));
  //       gp_boundary.set_rect(real_rect);
  //       gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(coord.get_layer_idx()));
  //       net_struct.push(gp_boundary);
  //     }
  //   }
  //   {
  //     // bounding_box
  //     GPBoundary gp_boundary;
  //     gp_boundary.set_layer_idx(0);
  //     gp_boundary.set_data_type(2);
  //     gp_boundary.set_rect(ir_net.get_bounding_box().get_real_rect());
  //     net_struct.push(gp_boundary);
  //   }
  //   for (Segment<TNode<LayerCoord>*>& segment : RTUtil::getSegListByTree(ir_net.get_routing_tree())) {
  //     LayerCoord first_coord = segment.get_first()->value();
  //     LayerCoord second_coord = segment.get_second()->value();
  //     irt_int first_layer_idx = first_coord.get_layer_idx();
  //     irt_int second_layer_idx = second_coord.get_layer_idx();

  //     if (first_layer_idx == second_layer_idx) {
  //       GPBoundary gp_boundary;
  //       gp_boundary.set_data_type(static_cast<irt_int>(GPDataType::kPath));
  //       gp_boundary.set_rect(RTUtil::getRealRectByGCell(first_coord, second_coord, gcell_axis));
  //       gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(first_layer_idx));
  //       net_struct.push(gp_boundary);
  //     } else {
  //       RTUtil::swapByASC(first_layer_idx, second_layer_idx);
  //       for (irt_int layer_idx = first_layer_idx; layer_idx <= second_layer_idx; layer_idx++) {
  //         GPBoundary gp_boundary;
  //         gp_boundary.set_data_type(static_cast<irt_int>(GPDataType::kPath));
  //         gp_boundary.set_rect(RTUtil::getRealRectByGCell(first_coord, gcell_axis));
  //         gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(layer_idx));
  //         net_struct.push(gp_boundary);
  //       }
  //     }
  //   }
  //   gp_gds.addStruct(net_struct);
  // }
  GP_INST.plot(gp_gds, ir_temp_directory_path + "ir_model.gds");
}

#endif

#endif

}  // namespace irt
