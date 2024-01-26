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
#include "GlobalRouter.hpp"

#include "GDSPlotter.hpp"
#include "RTUtil.hpp"

namespace irt {

// public

void GlobalRouter::initInst()
{
  if (_gr_instance == nullptr) {
    _gr_instance = new GlobalRouter();
  }
}

GlobalRouter& GlobalRouter::getInst()
{
  if (_gr_instance == nullptr) {
    LOG_INST.error(Loc::current(), "The instance not initialized!");
  }
  return *_gr_instance;
}

void GlobalRouter::destroyInst()
{
  if (_gr_instance != nullptr) {
    delete _gr_instance;
    _gr_instance = nullptr;
  }
}

// function

void GlobalRouter::route(std::vector<Net>& net_list)
{
  Monitor monitor;

  routeNetList(net_list);

}

// private

GlobalRouter* GlobalRouter::_gr_instance = nullptr;

void GlobalRouter::routeNetList(std::vector<Net>& net_list)
{
  GRModel gr_model = init(net_list);
  iterative(gr_model);
  update(gr_model);
}

#if 1  // init

GRModel GlobalRouter::init(std::vector<Net>& net_list)
{
  GRModel gr_model = initGRModel(net_list);
  buildGRModel(gr_model);
  checkGRModel(gr_model);
  writePYScript();
  return gr_model;
}

GRModel GlobalRouter::initGRModel(std::vector<Net>& net_list)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  Die& die = DM_INST.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  GRModel gr_model;

  std::vector<GridMap<GRNode>>& layer_node_map = gr_model.get_layer_node_map();
  layer_node_map.resize(routing_layer_list.size());
  for (irt_int layer_idx = 0; layer_idx < static_cast<irt_int>(layer_node_map.size()); layer_idx++) {
    GridMap<GRNode>& node_map = layer_node_map[layer_idx];
    node_map.init(die.getXSize(), die.getYSize());
    for (irt_int x = 0; x < die.getXSize(); x++) {
      for (irt_int y = 0; y < die.getYSize(); y++) {
        GRNode& gr_node = node_map[x][y];
        gr_node.set_coord(x, y);
        gr_node.set_layer_idx(layer_idx);
        GRNodeId gr_node_id;
        gr_node_id.set_x(x);
        gr_node_id.set_y(y);
        gr_node_id.set_layer_idx(layer_idx);
        gr_node.set_gr_node_id(gr_node_id);
        gr_node.set_base_region(RTUtil::getRealRectByGCell(x, y, gcell_axis));
      }
    }
  }
  gr_model.set_gr_net_list(convertToGRNetList(net_list));

  return gr_model;
}

std::vector<GRNet> GlobalRouter::convertToGRNetList(std::vector<Net>& net_list)
{
  std::vector<GRNet> gr_net_list;
  gr_net_list.reserve(net_list.size());
  for (size_t i = 0; i < net_list.size(); i++) {
    gr_net_list.emplace_back(convertToGRNet(net_list[i]));
  }
  return gr_net_list;
}

GRNet GlobalRouter::convertToGRNet(Net& net)
{
  GRNet gr_net;
  gr_net.set_origin_net(&net);
  gr_net.set_net_idx(net.get_net_idx());
  gr_net.set_connect_type(net.get_connect_type());
  for (Pin& pin : net.get_pin_list()) {
    gr_net.get_gr_pin_list().push_back(GRPin(pin));
  }
  // gr_net.set_gr_driving_pin(GRPin(net.get_driving_pin()));
  gr_net.set_bounding_box(net.get_bounding_box());
  gr_net.set_ra_cost_map(net.get_ra_cost_map());
  return gr_net;
}

void GlobalRouter::buildGRModel(GRModel& gr_model)
{
  buildNeighborMap(gr_model);
  updateWholeDemand(gr_model);
  updateNetDemandMap(gr_model);
  updateNodeResourceSupply(gr_model);
  updateNodeAccessSupply(gr_model);
  makeRoutingState(gr_model);
}

void GlobalRouter::buildNeighborMap(GRModel& gr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  irt_int bottom_routing_layer_idx = DM_INST.getConfig().bottom_routing_layer_idx;
  irt_int top_routing_layer_idx = DM_INST.getConfig().top_routing_layer_idx;

  std::vector<GridMap<GRNode>>& layer_node_map = gr_model.get_layer_node_map();
  for (irt_int layer_idx = 0; layer_idx < static_cast<irt_int>(layer_node_map.size()); layer_idx++) {
    bool routing_h = routing_layer_list[layer_idx].isPreferH();
    bool routing_v = !routing_h;
    if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
      routing_h = false;
      routing_v = false;
    }
    GridMap<GRNode>& gr_node_map = layer_node_map[layer_idx];
    for (irt_int x = 0; x < gr_node_map.get_x_size(); x++) {
      for (irt_int y = 0; y < gr_node_map.get_y_size(); y++) {
        std::map<Orientation, GRNode*>& neighbor_ptr_map = gr_node_map[x][y].get_neighbor_ptr_map();
        if (routing_h) {
          if (x != 0) {
            neighbor_ptr_map[Orientation::kWest] = &gr_node_map[x - 1][y];
          }
          if (x != (gr_node_map.get_x_size() - 1)) {
            neighbor_ptr_map[Orientation::kEast] = &gr_node_map[x + 1][y];
          }
        }
        if (routing_v) {
          if (y != 0) {
            neighbor_ptr_map[Orientation::kSouth] = &gr_node_map[x][y - 1];
          }
          if (y != (gr_node_map.get_y_size() - 1)) {
            neighbor_ptr_map[Orientation::kNorth] = &gr_node_map[x][y + 1];
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

void GlobalRouter::updateWholeDemand(GRModel& gr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  std::vector<GridMap<GRNode>>& layer_node_map = gr_model.get_layer_node_map();
  // track supply
  for (irt_int layer_idx = 0; layer_idx < static_cast<irt_int>(layer_node_map.size()); layer_idx++) {
    RoutingLayer& routing_layer = routing_layer_list[layer_idx];

    GridMap<GRNode>& node_map = layer_node_map[layer_idx];
#pragma omp parallel for collapse(2)
    for (irt_int x = 0; x < node_map.get_x_size(); x++) {
      for (irt_int y = 0; y < node_map.get_y_size(); y++) {
        GRNode& gr_node = node_map[x][y];
        irt_int cross_wire_demand = 0;
        if (routing_layer.isPreferH()) {
          cross_wire_demand = gr_node.get_base_region().getXSpan();
        } else {
          cross_wire_demand = gr_node.get_base_region().getYSpan();
        }
        gr_node.set_cross_wire_demand(cross_wire_demand);

        irt_int local_wire_demand = routing_layer.get_min_area() / routing_layer.get_min_width();
        gr_node.set_local_wire_demand(local_wire_demand);

        irt_int whole_via_demand = routing_layer.get_min_area() / routing_layer.get_min_width();
        gr_node.set_whole_via_demand(whole_via_demand);
      }
    }
  }
}

void GlobalRouter::updateNetDemandMap(GRModel& gr_model)
{
  // std::vector<GridMap<GRNode>>& layer_node_map = gr_model.get_layer_node_map();

  // // 根据ap点的access进行资源补偿
  // for (GRNet& gr_net : gr_model.get_gr_net_list()) {
  //   irt_int net_idx = gr_net.get_net_idx();
  //   for (GRPin& gr_pin : gr_net.get_gr_pin_list()) {
  //     LayerCoord curr_coord = gr_pin.get_protected_access_point().getGridLayerCoord();
  //     std::set<Orientation>& access_orien_set = gr_pin.get_protected_access_point().get_access_orien_set();

  //     GridMap<GRNode>& gr_node_map = layer_node_map[curr_coord.get_layer_idx()];
  //     GRNode& curr_node = gr_node_map[curr_coord.get_x()][curr_coord.get_y()];

  //     // 补偿via资源
  //     if (RTUtil::exist(access_orien_set, Orientation::kUp) || RTUtil::exist(access_orien_set, Orientation::kDown)) {
  //       // 由于pin_shape阻塞，打GR的通孔不算消耗
  //       curr_node.get_net_via_demand_map()[net_idx] = 0;
  //     }
  //     // 补偿access资源
  //     for (const Orientation& access_orien : access_orien_set) {
  //       if (access_orien == Orientation::kUp || access_orien == Orientation::kDown) {
  //         continue;
  //       }
  //       curr_node.get_net_orien_access_demand_map()[net_idx][access_orien] = 0;
  //       LayerCoord neighbor_coord = curr_coord;
  //       if (access_orien == Orientation::kEast) {
  //         neighbor_coord.set_x(curr_coord.get_x() + 1);
  //       } else if (access_orien == Orientation::kWest) {
  //         neighbor_coord.set_x(curr_coord.get_x() - 1);
  //       } else if (access_orien == Orientation::kSouth) {
  //         neighbor_coord.set_y(curr_coord.get_y() - 1);
  //       } else if (access_orien == Orientation::kNorth) {
  //         neighbor_coord.set_y(curr_coord.get_y() + 1);
  //       }
  //       if (gr_node_map.isInside(neighbor_coord.get_x(), neighbor_coord.get_y())) {
  //         GRNode& neighbor_node = gr_node_map[neighbor_coord.get_x()][neighbor_coord.get_y()];
  //         neighbor_node.get_net_orien_access_demand_map()[net_idx][RTUtil::getOppositeOrientation(access_orien)] = 0;
  //       }
  //     }
  //   }
  // }
}

void GlobalRouter::updateNodeResourceSupply(GRModel& gr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  double supply_utilization_rate = DM_INST.getConfig().supply_utilization_rate;

  std::vector<GridMap<GRNode>>& layer_node_map = gr_model.get_layer_node_map();
  // resource supply
  for (irt_int layer_idx = 0; layer_idx < static_cast<irt_int>(layer_node_map.size()); layer_idx++) {
    RoutingLayer& routing_layer = routing_layer_list[layer_idx];
    GridMap<GRNode>& node_map = layer_node_map[layer_idx];
#pragma omp parallel for collapse(2)
    for (irt_int x = 0; x < node_map.get_x_size(); x++) {
      for (irt_int y = 0; y < node_map.get_y_size(); y++) {
        GRNode& gr_node = node_map[x][y];
        std::vector<PlanarRect> wire_list = getCrossingWireList(gr_node.get_base_region(), routing_layer);
        // check
        if (!wire_list.empty()) {
          irt_int real_cross_wire_demand = wire_list.front().getArea() / routing_layer.get_min_width();
          irt_int gcell_cross_wire_demand = 0;
          if (routing_layer.isPreferH()) {
            gcell_cross_wire_demand = gr_node.get_base_region().getXSpan();
          } else {
            gcell_cross_wire_demand = gr_node.get_base_region().getYSpan();
          }
          if (real_cross_wire_demand != gcell_cross_wire_demand) {
            LOG_INST.error(Loc::current(), "The real_cross_wire_demand and gcell_cross_wire_demand are not equal!");
          }
        }
        // for (GRSourceType gr_source_type : {GRSourceType::kBlockage, GRSourceType::kNetShape}) {
          // for (auto& [info, rect_set] : DC_INST.getLayerInfoRectMap(gr_node.getRegionQuery(gr_source_type), true)[layer_idx]) {
          //   for (const LayerRect& rect : rect_set) {
          //     for (const LayerRect& min_scope_real_rect : DC_INST.getMinScope(DRCShape(info, rect, true))) {
          //       std::vector<PlanarRect> new_wire_list;
          //       for (PlanarRect& wire : wire_list) {
          //         if (RTUtil::isOpenOverlap(min_scope_real_rect, wire)) {
          //           // 要切
          //           std::vector<PlanarRect> split_rect_list
          //               = RTUtil::getSplitRectList(wire, min_scope_real_rect, routing_layer.get_prefer_direction());
          //           new_wire_list.insert(new_wire_list.end(), split_rect_list.begin(), split_rect_list.end());
          //         } else {
          //           // 不切
          //           new_wire_list.push_back(wire);
          //         }
          //       }
          //       wire_list = new_wire_list;
          //     }
          //   }
          // }
        // }
        irt_int resource_supply = 0;
        for (PlanarRect& wire : wire_list) {
          irt_int supply = wire.getArea() / routing_layer.get_min_width();
          if (supply < gr_node.get_whole_via_demand()) {
            continue;
          }
          resource_supply += supply;
        }
        resource_supply *= supply_utilization_rate;
        gr_node.set_resource_supply(resource_supply);
      }
    }
  }
}

std::vector<PlanarRect> GlobalRouter::getCrossingWireList(PlanarRect& base_rect, RoutingLayer& routing_layer)
{
  irt_int real_lb_x = base_rect.get_lb_x();
  irt_int real_lb_y = base_rect.get_lb_y();
  irt_int real_rt_x = base_rect.get_rt_x();
  irt_int real_rt_y = base_rect.get_rt_y();
  std::vector<irt_int> x_list = RTUtil::getOpenScaleList(real_lb_x, real_rt_x, routing_layer.getXTrackGridList());
  std::vector<irt_int> y_list = RTUtil::getOpenScaleList(real_lb_y, real_rt_y, routing_layer.getYTrackGridList());
  irt_int half_width = routing_layer.get_min_width() / 2;

  std::vector<PlanarRect> wire_list;
  if (routing_layer.isPreferH()) {
    for (irt_int y : y_list) {
      wire_list.emplace_back(real_lb_x, y - half_width, real_rt_x, y + half_width);
    }
  } else {
    for (irt_int x : x_list) {
      wire_list.emplace_back(x - half_width, real_lb_y, x + half_width, real_rt_y);
    }
  }
  return wire_list;
}

void GlobalRouter::updateNodeAccessSupply(GRModel& gr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  std::vector<GridMap<GRNode>>& layer_node_map = gr_model.get_layer_node_map();
  // access supply
  for (irt_int layer_idx = 0; layer_idx < static_cast<irt_int>(layer_node_map.size()); layer_idx++) {
    RoutingLayer& routing_layer = routing_layer_list[layer_idx];
    GridMap<GRNode>& node_map = layer_node_map[layer_idx];

    if (routing_layer.isPreferH()) {
#pragma omp parallel for collapse(2)
      for (irt_int y = 0; y < node_map.get_y_size(); y++) {
        for (irt_int x = 1; x < node_map.get_x_size(); x++) {
          GRNode& pre_node = node_map[x - 1][y];
          GRNode& curr_node = node_map[x][y];
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
          GRNode& pre_node = node_map[x][y - 1];
          GRNode& curr_node = node_map[x][y];
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

bool GlobalRouter::isAccess(GRNode& pre_node, GRNode& curr_node, PlanarRect& wire)
{
  // if (pre_node.get_layer_idx() != curr_node.get_layer_idx()) {
  //   LOG_INST.error(Loc::current(), "The layer is not equal!");
  // }
  // irt_int layer_idx = pre_node.get_layer_idx();

  // for (GRSourceType gr_source_type : {GRSourceType::kBlockage, GRSourceType::kNetShape, GRSourceType::kReservedVia}) {
  //   for (auto& [info, rect_set] : DC_INST.getLayerInfoRectMap(pre_node.getRegionQuery(gr_source_type), true)[layer_idx]) {
  //     for (const LayerRect& rect : rect_set) {
  //       for (const LayerRect& min_scope_real_rect : DC_INST.getMinScope(DRCShape(info, rect, true))) {
  //         if (RTUtil::isOpenOverlap(min_scope_real_rect, wire)) {
  //           // 阻塞
  //           return false;
  //         }
  //       }
  //     }
  //   }
  //   for (auto& [info, rect_set] : DC_INST.getLayerInfoRectMap(curr_node.getRegionQuery(gr_source_type), true)[layer_idx]) {
  //     for (const LayerRect& rect : rect_set) {
  //       for (const LayerRect& min_scope_real_rect : DC_INST.getMinScope(DRCShape(info, rect, true))) {
  //         if (RTUtil::isOpenOverlap(min_scope_real_rect, wire)) {
  //           // 阻塞
  //           return false;
  //         }
  //       }
  //     }
  //   }
  // }
  return true;
}

void GlobalRouter::makeRoutingState(GRModel& gr_model)
{
  for (GRNet& gr_net : gr_model.get_gr_net_list()) {
    gr_net.set_routing_state(RoutingState::kUnrouted);
  }
}

void GlobalRouter::checkGRModel(GRModel& gr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  irt_int bottom_routing_layer_idx = DM_INST.getConfig().bottom_routing_layer_idx;
  irt_int top_routing_layer_idx = DM_INST.getConfig().top_routing_layer_idx;

  std::vector<GridMap<GRNode>>& layer_node_map = gr_model.get_layer_node_map();
  for (irt_int layer_idx = 0; layer_idx < static_cast<irt_int>(layer_node_map.size()); layer_idx++) {
    bool routing_h = routing_layer_list[layer_idx].isPreferH();
    bool routing_v = !routing_h;
    if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
      routing_h = false;
      routing_v = false;
    }
    GridMap<GRNode>& gr_node_map = layer_node_map[layer_idx];
    for (irt_int x = 0; x < gr_node_map.get_x_size(); x++) {
      for (irt_int y = 0; y < gr_node_map.get_y_size(); y++) {
        GRNode& gr_node = gr_node_map[x][y];
        std::map<Orientation, GRNode*>& neighbor_ptr_map = gr_node.get_neighbor_ptr_map();
        if (routing_h) {
          if (RTUtil::exist(neighbor_ptr_map, Orientation::kNorth) || RTUtil::exist(neighbor_ptr_map, Orientation::kSouth)) {
            LOG_INST.error(Loc::current(), "There is illegal vertical neighbor relations!");
          }
        }
        if (routing_v) {
          if (RTUtil::exist(neighbor_ptr_map, Orientation::kEast) || RTUtil::exist(neighbor_ptr_map, Orientation::kWest)) {
            LOG_INST.error(Loc::current(), "There is illegal horizontal neighbor relations!");
          }
        }
        for (auto& [orien, neighbor] : neighbor_ptr_map) {
          Orientation opposite_orien = RTUtil::getOppositeOrientation(orien);
          if (!RTUtil::exist(neighbor->get_neighbor_ptr_map(), opposite_orien)) {
            LOG_INST.error(Loc::current(), "The gr_node neighbor is not bidirection!");
          }
          if (neighbor->get_neighbor_ptr_map()[opposite_orien] != &gr_node) {
            LOG_INST.error(Loc::current(), "The gr_node neighbor is not bidirection!");
          }
          LayerCoord node_coord(gr_node.get_planar_coord(), gr_node.get_layer_idx());
          LayerCoord neighbor_coord(neighbor->get_planar_coord(), neighbor->get_layer_idx());
          if (RTUtil::getOrientation(node_coord, neighbor_coord) != orien) {
            LOG_INST.error(Loc::current(), "The neighbor orien is different with real region!");
          }
        }
        if (gr_node.get_cross_wire_demand() < 0) {
          LOG_INST.error(Loc::current(), "The cross_wire_demand < 0!");
        }
        if (gr_node.get_local_wire_demand() < 0) {
          LOG_INST.error(Loc::current(), "The local_wire_demand < 0!");
        }
        if (gr_node.get_whole_via_demand() < 0) {
          LOG_INST.error(Loc::current(), "The whole_via_demand < 0!");
        }
        for (auto& [net_idx, via_demand] : gr_node.get_net_via_demand_map()) {
          if (via_demand < 0) {
            LOG_INST.error(Loc::current(), "The via_demand < 0!");
          }
        }
        if (gr_node.get_whole_access_demand() < 0) {
          LOG_INST.error(Loc::current(), "The whole_access_demand < 0!");
        }
        for (auto& [net_idx, orien_access_demand_map] : gr_node.get_net_orien_access_demand_map()) {
          for (auto& [orien, access_demand] : orien_access_demand_map) {
            if (access_demand < 0) {
              LOG_INST.error(Loc::current(), "The access_demand < 0!");
            }
          }
        }
        std::map<Orientation, irt_int>& orien_access_supply_map = gr_node.get_orien_access_supply_map();
        if (routing_h) {
          if (RTUtil::exist(orien_access_supply_map, Orientation::kSouth) || RTUtil::exist(orien_access_supply_map, Orientation::kNorth)) {
            LOG_INST.error(Loc::current(), "The orientation is error!");
          }
        }
        if (routing_v) {
          if (RTUtil::exist(orien_access_supply_map, Orientation::kWest) || RTUtil::exist(orien_access_supply_map, Orientation::kEast)) {
            LOG_INST.error(Loc::current(), "The orientation is error!");
          }
        }
        for (auto& [orientation, access_supply] : orien_access_supply_map) {
          if (access_supply < 0) {
            LOG_INST.error(Loc::current(), "The access_supply < 0!");
          }
        }
        if (gr_node.get_resource_supply() < 0) {
          LOG_INST.error(Loc::current(), "The resource_supply < 0!");
        }
      }
    }
  }
  for (GRNet& gr_net : gr_model.get_gr_net_list()) {
    if (gr_net.get_routing_state() != RoutingState::kUnrouted) {
      LOG_INST.error(Loc::current(), "The routing_state is error!");
    }
  }
}

void GlobalRouter::writePYScript()
{
  std::string gr_temp_directory_path = DM_INST.getConfig().gr_temp_directory_path;
  irt_int gr_max_iter_num = DM_INST.getConfig().gr_max_iter_num;

  std::ofstream* python_file = RTUtil::getOutputFileStream(RTUtil::getString(gr_temp_directory_path, "plot.py"));

  RTUtil::pushStream(python_file, "from concurrent.futures import process", "\n");
  RTUtil::pushStream(python_file, "import numpy as np", "\n");
  RTUtil::pushStream(python_file, "import matplotlib.pyplot as plt", "\n");
  RTUtil::pushStream(python_file, "import seaborn as sns", "\n");
  RTUtil::pushStream(python_file, "import pandas as pd", "\n");
  RTUtil::pushStream(python_file, "from PIL import Image", "\n");
  RTUtil::pushStream(python_file, "import glob", "\n");
  RTUtil::pushStream(python_file, "", "\n");
  RTUtil::pushStream(python_file, "for i in range(1,", gr_max_iter_num + 1, "):", "\n");
  RTUtil::pushStream(python_file, "    csv_data = pd.read_csv('gr_model_'+ str(i) +'.csv')", "\n");
  RTUtil::pushStream(python_file, "    array_data = np.array(csv_data)", "\n");
  RTUtil::pushStream(python_file, "", "\n");
  RTUtil::pushStream(python_file, "    # 输出热力图", "\n");
  RTUtil::pushStream(python_file, "    plt.clf()", "\n");
  RTUtil::pushStream(python_file, "    hm=sns.heatmap(array_data, vmin=0, vmax=3, cmap='hot_r')", "\n");
  RTUtil::pushStream(python_file, "    hm.set_title('gr_model_'+ str(i))", "\n");
  RTUtil::pushStream(python_file, "    s1 = hm.get_figure()", "\n");
  RTUtil::pushStream(python_file, "    s1.savefig('gr_model_'+ str(i) +'.png',dpi=1000)", "\n");
  RTUtil::pushStream(python_file, "    # plt.show()", "\n");
  RTUtil::pushStream(python_file, "", "\n");
  RTUtil::pushStream(python_file, "", "\n");
  RTUtil::pushStream(python_file, "images = glob.glob('gr_model_*.png')", "\n");
  RTUtil::pushStream(python_file, "", "\n");
  RTUtil::pushStream(python_file, "# 提取文件名中的id数字部分,并转换为整数", "\n");
  RTUtil::pushStream(python_file, "sorted_images = sorted(images, key=lambda x: int(x.split('_')[-1].split('.')[0]))", "\n");
  RTUtil::pushStream(python_file, "", "\n");
  RTUtil::pushStream(python_file, "frames = []", "\n");
  RTUtil::pushStream(python_file, "for image in sorted_images:", "\n");
  RTUtil::pushStream(python_file, "    img = Image.open(image)", "\n");
  RTUtil::pushStream(python_file, "    img = img.resize((800, 600))", "\n");
  RTUtil::pushStream(python_file, "    frames.append(img)", "\n");
  RTUtil::pushStream(python_file, "", "\n");
  RTUtil::pushStream(python_file,
                     "frames[0].save('output.gif', format='GIF', append_images=frames[1:], save_all=True, duration=300, loop=0)", "\n");
  RTUtil::closeFileStream(python_file);
}

#endif

#if 1  // iterative

void GlobalRouter::iterative(GRModel& gr_model)
{
  irt_int gr_max_iter_num = DM_INST.getConfig().gr_max_iter_num;

  for (irt_int iter = 1; iter <= gr_max_iter_num; iter++) {
    Monitor iter_monitor;
    LOG_INST.info(Loc::current(), "****** Start Model Iteration(", iter, "/", gr_max_iter_num, ") ******");
    gr_model.set_curr_iter(iter);
    resetGRModel(gr_model);
    routeGRModel(gr_model);
    processGRModel(gr_model);
    plotGRModel(gr_model);
    outputCongestionMap(gr_model);
    LOG_INST.info(Loc::current(), "****** End Model Iteration(", iter, "/", gr_max_iter_num, ")", iter_monitor.getStatsInfo(), " ******");
    if (stopGRModel(gr_model)) {
      if (iter < gr_max_iter_num) {
        LOG_INST.info(Loc::current(), "****** Terminate the iteration by reaching the condition in advance! ******");
      }
      gr_model.set_curr_iter(-1);
      break;
    }
  }
}

void GlobalRouter::resetGRModel(GRModel& gr_model)
{
  if (gr_model.get_curr_iter() == 1) {
    sortGRModel(gr_model);
  } else {
    updateRipupGrid(gr_model);
    addHistoryCost(gr_model);
    ripupPassedNet(gr_model);
  }
}

void GlobalRouter::sortGRModel(GRModel& gr_model)
{
  if (gr_model.get_curr_iter() != 1) {
    return;
  }
  Monitor monitor;
  if (omp_get_num_threads() == 1) {
    LOG_INST.info(Loc::current(), "Sorting all nets beginning...");
  }

  std::vector<irt_int> net_order_list;
  for (GRNet& gr_net : gr_model.get_gr_net_list()) {
    net_order_list.push_back(gr_net.get_net_idx());
  }
  std::sort(net_order_list.begin(), net_order_list.end(),
            [&](irt_int net_idx1, irt_int net_idx2) { return sortByMultiLevel(gr_model, net_idx1, net_idx2); });
  gr_model.get_net_order_list_list().push_back(net_order_list);

  if (omp_get_num_threads() == 1) {
    LOG_INST.info(Loc::current(), "Sorting all nets completed!", monitor.getStatsInfo());
  }
}

bool GlobalRouter::sortByMultiLevel(GRModel& gr_model, irt_int net_idx1, irt_int net_idx2)
{
  GRNet& net1 = gr_model.get_gr_net_list()[net_idx1];
  GRNet& net2 = gr_model.get_gr_net_list()[net_idx2];

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
SortStatus GlobalRouter::sortByClockPriority(GRNet& net1, GRNet& net2)
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
SortStatus GlobalRouter::sortByRoutingAreaASC(GRNet& net1, GRNet& net2)
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
SortStatus GlobalRouter::sortByLengthWidthRatioDESC(GRNet& net1, GRNet& net2)
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
SortStatus GlobalRouter::sortByPinNumDESC(GRNet& net1, GRNet& net2)
{
  irt_int net1_pin_num = static_cast<irt_int>(net1.get_gr_pin_list().size());
  irt_int net2_pin_num = static_cast<irt_int>(net2.get_gr_pin_list().size());

  if (net1_pin_num > net2_pin_num) {
    return SortStatus::kTrue;
  } else if (net1_pin_num == net2_pin_num) {
    return SortStatus::kEqual;
  } else {
    return SortStatus::kFalse;
  }
}

void GlobalRouter::updateRipupGrid(GRModel& gr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  std::vector<GridMap<GRNode>>& layer_node_map = gr_model.get_layer_node_map();

  /**
   * history_grid_access_orien_map不要清零
   * ripup_grid_set需要清零
   */
  gr_model.get_ripup_grid_set().clear();

  for (GridMap<GRNode>& node_map : layer_node_map) {
    for (irt_int grid_x = 0; grid_x < node_map.get_x_size(); grid_x++) {
      for (irt_int grid_y = 0; grid_y < node_map.get_y_size(); grid_y++) {
        GRNode& gr_node = node_map[grid_x][grid_y];
        std::map<Orientation, irt_int>& orien_access_supply_map = gr_node.get_orien_access_supply_map();
        std::map<Orientation, irt_int>& orien_access_demand_map = gr_node.get_orien_access_demand_map();

        if (routing_layer_list[gr_node.get_layer_idx()].isPreferH()) {
          for (Orientation orientation : {Orientation::kEast, Orientation::kWest}) {
            double access_overflow = (orien_access_demand_map[orientation] - orien_access_supply_map[orientation]);
            if (access_overflow > 1) {
              gr_model.get_history_grid_access_orien_map()[gr_node].insert(orientation);
              gr_model.get_ripup_grid_set().insert(gr_node);
            }
          }
        } else {
          for (Orientation orientation : {Orientation::kSouth, Orientation::kNorth}) {
            double access_overflow = (orien_access_demand_map[orientation] - orien_access_supply_map[orientation]);
            if (access_overflow > 1) {
              gr_model.get_history_grid_access_orien_map()[gr_node].insert(orientation);
              gr_model.get_ripup_grid_set().insert(gr_node);
            }
          }
        }
      }
    }
  }
}

void GlobalRouter::addHistoryCost(GRModel& gr_model)
{
  double gr_history_cost_unit = DM_INST.getConfig().gr_history_cost_unit;

  std::vector<GridMap<GRNode>>& layer_node_map = gr_model.get_layer_node_map();

  // 由于history_grid_access_orien_map不清零，所以会添加新的history cost，并累加原有的
  for (auto& [coord, orient_set] : gr_model.get_history_grid_access_orien_map()) {
    GRNode& gr_node = layer_node_map[coord.get_layer_idx()][coord.get_x()][coord.get_y()];
    for (const auto& orientation : orient_set) {
      gr_node.get_history_orien_access_cost_map()[orientation] += gr_history_cost_unit;
    }
  }
}

void GlobalRouter::ripupPassedNet(GRModel& gr_model)
{
  std::vector<GridMap<GRNode>>& layer_node_map = gr_model.get_layer_node_map();

  // 由于ripup_grid_set清零，所以只会重拆本次的
  std::set<irt_int> all_passed_net_set;
  for (const auto& coord : gr_model.get_ripup_grid_set()) {
    GRNode& gr_node = layer_node_map[coord.get_layer_idx()][coord.get_x()][coord.get_y()];
    std::set<irt_int>& passed_net_set = gr_node.get_passed_net_set();
    all_passed_net_set.insert(passed_net_set.begin(), passed_net_set.end());
  }

  for (irt_int net_idx : all_passed_net_set) {
    GRNet& gr_net = gr_model.get_gr_net_list()[net_idx];
    // 将env中的布线结果清空
    updateDemand(gr_model, gr_net, ChangeType::kDel);
    // 清空routing_tree
    gr_net.get_routing_tree().clear();
    gr_net.set_routing_state(RoutingState::kUnrouted);
  }
}

void GlobalRouter::routeGRModel(GRModel& gr_model)
{
  Monitor monitor;

  std::vector<GRNet>& gr_net_list = gr_model.get_gr_net_list();
  if (gr_model.get_net_order_list_list().empty()) {
    LOG_INST.error(Loc::current(), "The net_order_list_list is empty!");
  }
  std::vector<irt_int>& net_order_list = gr_model.get_net_order_list_list().back();

  irt_int batch_size = RTUtil::getBatchSize(gr_net_list.size());

  Monitor stage_monitor;
  for (size_t i = 0; i < net_order_list.size(); i++) {
    routeGRNet(gr_model, gr_net_list[net_order_list[i]]);
    if (omp_get_num_threads() == 1 && (i + 1) % batch_size == 0) {
      LOG_INST.info(Loc::current(), "Routed ", (i + 1), " nets", stage_monitor.getStatsInfo());
    }
  }
  if (omp_get_num_threads() == 1) {
    LOG_INST.info(Loc::current(), "Routed ", gr_net_list.size(), " nets", monitor.getStatsInfo());
  }
}

void GlobalRouter::routeGRNet(GRModel& gr_model, GRNet& gr_net)
{
  if (gr_net.get_routing_state() == RoutingState::kRouted) {
    return;
  }
  initSingleNet(gr_model, gr_net);
  // outputGRDataset(gr_model, gr_net);
  for (GRTask& gr_task : gr_model.get_gr_task_list()) {
    initSingleTask(gr_model, gr_task);
    while (!isConnectedAllEnd(gr_model)) {
      routeSinglePath(gr_model);
      updatePathResult(gr_model);
      updateDirectionSet(gr_model);
      resetStartAndEnd(gr_model);
      resetSinglePath(gr_model);
    }
    resetSingleTask(gr_model);
  }
  updateNetResult(gr_model, gr_net);
  resetSingleNet(gr_model);
}

void GlobalRouter::outputGRDataset(GRModel& gr_model, GRNet& gr_net)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  static size_t written_net_num = 0;
  static std::string gr_dataset_path;
  static std::ofstream* gr_dataset;

  if (written_net_num == 0) {
    std::string def_file_path = DM_INST.getHelper().get_def_file_path();
    gr_dataset_path
        = RTUtil::getString(DM_INST.getConfig().gr_temp_directory_path, RTUtil::splitString(def_file_path, '/').back(), ".gr.txt");
    gr_dataset = RTUtil::getOutputFileStream(gr_dataset_path);
    RTUtil::pushStream(gr_dataset, "def_file_path", " ", def_file_path, "\n");
  }
  RTUtil::pushStream(gr_dataset, "net", " ", gr_net.get_net_idx(), "\n");
  RTUtil::pushStream(gr_dataset, "{", "\n");
  RTUtil::pushStream(gr_dataset, "pin_list", "\n");
  for (GRPin& gr_pin : gr_net.get_gr_pin_list()) {
    // RTUtil::pushStream(gr_dataset, "pin", " ", gr_pin.get_pin_idx(), "\n");
    // LayerCoord coord = gr_pin.get_protected_access_point().getGridLayerCoord();
    // RTUtil::pushStream(gr_dataset, coord.get_x(), " ", coord.get_y(), " ", coord.get_layer_idx(), "\n");
  }
  RTUtil::pushStream(gr_dataset, "cost_map", "\n");
  std::vector<GridMap<GRNode>>& layer_node_map = gr_model.get_layer_node_map();
  BoundingBox& bounding_box = gr_net.get_bounding_box();
  for (irt_int layer_idx = 0; layer_idx < static_cast<irt_int>(layer_node_map.size()); layer_idx++) {
    for (irt_int x = bounding_box.get_grid_lb_x(); x <= bounding_box.get_grid_rt_x(); x++) {
      for (irt_int y = bounding_box.get_grid_lb_y(); y <= bounding_box.get_grid_rt_y(); y++) {
        GRNode& gr_node = layer_node_map[layer_idx][x][y];
        double east_cost = -1;
        double west_cost = -1;
        double south_cost = -1;
        double north_cost = -1;
        double up_cost = -1;
        double down_cost = -1;
        if (routing_layer_list[layer_idx].isPreferH()) {
          east_cost = getNodeCost(gr_model, &gr_node, Orientation::kEast);
          west_cost = getNodeCost(gr_model, &gr_node, Orientation::kWest);
        } else {
          south_cost = getNodeCost(gr_model, &gr_node, Orientation::kSouth);
          north_cost = getNodeCost(gr_model, &gr_node, Orientation::kNorth);
        }
        if (layer_idx != 0) {
          down_cost = getNodeCost(gr_model, &gr_node, Orientation::kDown);
        }
        if (layer_idx != (static_cast<irt_int>(layer_node_map.size()) - 1)) {
          up_cost = getNodeCost(gr_model, &gr_node, Orientation::kUp);
        }
        RTUtil::pushStream(gr_dataset, x, " ", y, " ", layer_idx);
        RTUtil::pushStream(gr_dataset, " ", "E", " ", east_cost);
        RTUtil::pushStream(gr_dataset, " ", "W", " ", west_cost);
        RTUtil::pushStream(gr_dataset, " ", "S", " ", south_cost);
        RTUtil::pushStream(gr_dataset, " ", "N", " ", north_cost);
        RTUtil::pushStream(gr_dataset, " ", "U", " ", up_cost);
        RTUtil::pushStream(gr_dataset, " ", "D", " ", down_cost);
        RTUtil::pushStream(gr_dataset, "\n");
      }
    }
  }
  RTUtil::pushStream(gr_dataset, "}", "\n");

  written_net_num++;
  if (written_net_num % 10000 == 0) {
    LOG_INST.info(Loc::current(), "Written ", written_net_num, " nets");
  }
  if (written_net_num == gr_model.get_gr_net_list().size()) {
    LOG_INST.info(Loc::current(), "Written ", written_net_num, " nets");
    RTUtil::closeFileStream(gr_dataset);
    LOG_INST.info(Loc::current(), "The result has been written to '", gr_dataset_path, "'!");
    exit(0);
  }
}

void GlobalRouter::initSingleNet(GRModel& gr_model, GRNet& gr_net)
{
  EXTPlanarRect& die = DM_INST.getDatabase().get_die();

  std::vector<GridMap<GRNode>>& layer_node_map = gr_model.get_layer_node_map();

  gr_model.set_gr_net_ref(&gr_net);
  if (gr_model.get_curr_iter() == 1) {
    gr_model.set_routing_region(gr_model.get_curr_bounding_box());
  } else {
    gr_model.set_routing_region(die.get_grid_rect());
  }
  gr_model.get_gr_task_list().clear();
  gr_model.get_node_segment_list().clear();

  if (gr_model.get_curr_iter() == 1) {
    irt_int bottom_routing_layer_idx = DM_INST.getConfig().bottom_routing_layer_idx;
    irt_int top_routing_layer_idx = DM_INST.getConfig().top_routing_layer_idx;

    std::vector<PlanarCoord> planar_coord_list;
    for (GRPin& gr_pin : gr_net.get_gr_pin_list()) {
      // planar_coord_list.push_back(gr_pin.get_protected_access_point().get_grid_coord());
    }
    std::sort(planar_coord_list.begin(), planar_coord_list.end(), CmpPlanarCoordByXASC());
    planar_coord_list.erase(std::unique(planar_coord_list.begin(), planar_coord_list.end()), planar_coord_list.end());

    if (planar_coord_list.size() == 1) {
      GRTask gr_task;
      for (GRPin& gr_pin : gr_net.get_gr_pin_list()) {
        // GRGroup gr_group;
        // LayerCoord coord = gr_pin.get_protected_access_point().getGridLayerCoord();
        // gr_group.get_gr_node_list().push_back(&layer_node_map[coord.get_layer_idx()][coord.get_x()][coord.get_y()]);
        // gr_task.get_gr_group_list().push_back(gr_group);
      }
      gr_model.get_gr_task_list().push_back(gr_task);
    } else {
      // pin的GRGroup
      std::map<PlanarCoord, std::vector<GRGroup>, CmpPlanarCoordByXASC> key_planar_group_map;
      for (GRPin& gr_pin : gr_net.get_gr_pin_list()) {
        // GRGroup gr_group;
        // LayerCoord coord = gr_pin.get_protected_access_point().getGridLayerCoord();
        // gr_group.get_gr_node_list().push_back(&layer_node_map[coord.get_layer_idx()][coord.get_x()][coord.get_y()]);
        // key_planar_group_map[coord].push_back(gr_group);
      }

      // steiner point的GRGroup
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
        GRNode* first_node = &layer_node_map[first_coord.get_layer_idx()][first_coord.get_x()][first_coord.get_y()];
        GRNode* second_node = &layer_node_map[second_coord.get_layer_idx()][second_coord.get_x()][second_coord.get_y()];
        gr_model.get_node_segment_list().emplace_back(first_node, second_node);
      }
      // 生成task
      for (Segment<PlanarCoord>& planar_topo : planar_topo_list) {
        GRTask gr_task;

        if (RTUtil::exist(key_planar_group_map, planar_topo.get_first())) {
          for (GRGroup& gr_group : key_planar_group_map[planar_topo.get_first()]) {
            gr_task.get_gr_group_list().push_back(gr_group);
          }
        } else if (RTUtil::exist(add_planar_layer_map, planar_topo.get_first())) {
          GRGroup gr_group;
          for (LayerCoord coord : add_planar_layer_map[planar_topo.get_first()]) {
            gr_group.get_gr_node_list().push_back(&layer_node_map[coord.get_layer_idx()][coord.get_x()][coord.get_y()]);
          }
          gr_task.get_gr_group_list().push_back(gr_group);
        }

        if (RTUtil::exist(key_planar_group_map, planar_topo.get_second())) {
          for (GRGroup& gr_group : key_planar_group_map[planar_topo.get_second()]) {
            gr_task.get_gr_group_list().push_back(gr_group);
          }
        } else if (RTUtil::exist(add_planar_layer_map, planar_topo.get_second())) {
          GRGroup gr_group;
          for (LayerCoord coord : add_planar_layer_map[planar_topo.get_second()]) {
            gr_group.get_gr_node_list().push_back(&layer_node_map[coord.get_layer_idx()][coord.get_x()][coord.get_y()]);
          }
          gr_task.get_gr_group_list().push_back(gr_group);
        }
        gr_model.get_gr_task_list().push_back(gr_task);
      }
    }
  } else {
    GRTask gr_task;
    for (GRPin& gr_pin : gr_net.get_gr_pin_list()) {
      // GRGroup gr_group;
      // LayerCoord coord = gr_pin.get_protected_access_point().getGridLayerCoord();
      // gr_group.get_gr_node_list().push_back(&layer_node_map[coord.get_layer_idx()][coord.get_x()][coord.get_y()]);
      // gr_task.get_gr_group_list().push_back(gr_group);
    }
    gr_model.get_gr_task_list().push_back(gr_task);
  }
}

std::vector<Segment<PlanarCoord>> GlobalRouter::getPlanarTopoListByFlute(std::vector<PlanarCoord>& planar_coord_list)
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

void GlobalRouter::initSingleTask(GRModel& gr_model, GRTask& gr_task)
{
  std::vector<GRGroup>& start_group_list = gr_model.get_start_group_list();
  std::vector<GRGroup>& end_group_list = gr_model.get_end_group_list();

  std::vector<GRGroup>& gr_group_list = gr_task.get_gr_group_list();
  start_group_list.push_back(gr_group_list[0]);
  for (size_t i = 1; i < gr_group_list.size(); i++) {
    end_group_list.push_back(gr_group_list[i]);
  }
}

bool GlobalRouter::isConnectedAllEnd(GRModel& gr_model)
{
  return gr_model.get_end_group_list().empty();
}

void GlobalRouter::routeSinglePath(GRModel& gr_model)
{
  initPathHead(gr_model);
  while (!searchEnded(gr_model)) {
    expandSearching(gr_model);
    resetPathHead(gr_model);
  }
}

void GlobalRouter::initPathHead(GRModel& gr_model)
{
  std::vector<GRGroup>& start_group_list = gr_model.get_start_group_list();
  GRGroup& path_group = gr_model.get_path_group();

  for (GRGroup& start_group : start_group_list) {
    for (GRNode* start_node : start_group.get_gr_node_list()) {
      start_node->set_estimated_cost(getEstimateCostToEnd(gr_model, start_node));
      pushToOpenList(gr_model, start_node);
    }
  }
  for (GRNode* path_node : path_group.get_gr_node_list()) {
    path_node->set_estimated_cost(getEstimateCostToEnd(gr_model, path_node));
    pushToOpenList(gr_model, path_node);
  }
  resetPathHead(gr_model);
}

bool GlobalRouter::searchEnded(GRModel& gr_model)
{
  std::vector<GRGroup>& end_group_list = gr_model.get_end_group_list();
  GRNode* path_head_node = gr_model.get_path_head_node();

  if (path_head_node == nullptr) {
    gr_model.set_end_group_idx(-1);
    return true;
  }
  for (size_t i = 0; i < end_group_list.size(); i++) {
    for (GRNode* end_node : end_group_list[i].get_gr_node_list()) {
      if (path_head_node == end_node) {
        gr_model.set_end_group_idx(static_cast<irt_int>(i));
        return true;
      }
    }
  }
  return false;
}

void GlobalRouter::expandSearching(GRModel& gr_model)
{
  PriorityQueue<GRNode*, std::vector<GRNode*>, CmpGRNodeCost>& open_queue = gr_model.get_open_queue();
  GRNode* path_head_node = gr_model.get_path_head_node();

  for (auto& [orientation, neighbor_node] : path_head_node->get_neighbor_ptr_map()) {
    if (neighbor_node == nullptr) {
      continue;
    }
    if (!RTUtil::isInside(gr_model.get_routing_region(), *neighbor_node)) {
      continue;
    }
    if (neighbor_node->isClose()) {
      continue;
    }
    double know_cost = getKnowCost(gr_model, path_head_node, neighbor_node);
    if (neighbor_node->isOpen() && know_cost < neighbor_node->get_known_cost()) {
      neighbor_node->set_known_cost(know_cost);
      neighbor_node->set_parent_node(path_head_node);
      // 对优先队列中的值修改了，需要重新建堆
      std::make_heap(open_queue.begin(), open_queue.end(), CmpGRNodeCost());
    } else if (neighbor_node->isNone()) {
      neighbor_node->set_known_cost(know_cost);
      neighbor_node->set_parent_node(path_head_node);
      neighbor_node->set_estimated_cost(getEstimateCostToEnd(gr_model, neighbor_node));
      pushToOpenList(gr_model, neighbor_node);
    }
  }
}

void GlobalRouter::resetPathHead(GRModel& gr_model)
{
  gr_model.set_path_head_node(popFromOpenList(gr_model));
}

bool GlobalRouter::isRoutingFailed(GRModel& gr_model)
{
  return gr_model.get_end_group_idx() == -1;
}

void GlobalRouter::resetSinglePath(GRModel& gr_model)
{
  PriorityQueue<GRNode*, std::vector<GRNode*>, CmpGRNodeCost> empty_queue;
  gr_model.set_open_queue(empty_queue);

  std::vector<GRNode*>& visited_node_list = gr_model.get_visited_node_list();
  for (GRNode* visited_node : visited_node_list) {
    visited_node->set_state(GRNodeState::kNone);
    visited_node->set_parent_node(nullptr);
    visited_node->set_known_cost(0);
    visited_node->set_estimated_cost(0);
  }
  visited_node_list.clear();

  gr_model.set_path_head_node(nullptr);
  gr_model.set_end_group_idx(-1);
}

void GlobalRouter::updatePathResult(GRModel& gr_model)
{
  std::vector<Segment<GRNode*>>& node_segment_list = gr_model.get_node_segment_list();
  GRNode* path_head_node = gr_model.get_path_head_node();

  GRNode* curr_node = path_head_node;
  GRNode* pre_node = curr_node->get_parent_node();

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

void GlobalRouter::updateDirectionSet(GRModel& gr_model)
{
  GRNode* path_head_node = gr_model.get_path_head_node();

  GRNode* curr_node = path_head_node;
  GRNode* pre_node = curr_node->get_parent_node();
  while (pre_node != nullptr) {
    curr_node->get_direction_set().insert(RTUtil::getDirection(*curr_node, *pre_node));
    pre_node->get_direction_set().insert(RTUtil::getDirection(*pre_node, *curr_node));
    curr_node = pre_node;
    pre_node = curr_node->get_parent_node();
  }
}

void GlobalRouter::resetStartAndEnd(GRModel& gr_model)
{
  std::vector<GRGroup>& start_group_list = gr_model.get_start_group_list();
  std::vector<GRGroup>& end_group_list = gr_model.get_end_group_list();
  GRGroup& path_group = gr_model.get_path_group();
  GRNode* path_head_node = gr_model.get_path_head_node();
  irt_int end_group_idx = gr_model.get_end_group_idx();

  end_group_list[end_group_idx].get_gr_node_list().clear();
  end_group_list[end_group_idx].get_gr_node_list().push_back(path_head_node);

  GRNode* path_node = path_head_node->get_parent_node();
  if (path_node == nullptr) {
    // 起点和终点重合
    path_node = path_head_node;
  } else {
    // 起点和终点不重合
    while (path_node->get_parent_node() != nullptr) {
      path_group.get_gr_node_list().push_back(path_node);
      path_node = path_node->get_parent_node();
    }
  }
  if (start_group_list.size() == 1) {
    // 初始化时，要把start_group_list的pin只留一个ap点
    // 后续只要将end_group_list的pin保留一个ap点
    start_group_list.front().get_gr_node_list().clear();
    start_group_list.front().get_gr_node_list().push_back(path_node);
  }
  start_group_list.push_back(end_group_list[end_group_idx]);
  end_group_list.erase(end_group_list.begin() + end_group_idx);
}

void GlobalRouter::resetSingleTask(GRModel& gr_model)
{
  gr_model.get_start_group_list().clear();
  gr_model.get_end_group_list().clear();
  gr_model.set_path_group(GRGroup());
}

void GlobalRouter::updateNetResult(GRModel& gr_model, GRNet& gr_net)
{
  updateRoutingTree(gr_model, gr_net);
  updateDemand(gr_model, gr_net, ChangeType::kAdd);
  gr_net.set_routing_state(RoutingState::kRouted);
}

void GlobalRouter::updateRoutingTree(GRModel& gr_model, GRNet& gr_net)
{
  // std::vector<Segment<GRNode*>>& node_segment_list = gr_model.get_node_segment_list();

  // std::vector<Segment<LayerCoord>> routing_segment_list;
  // for (Segment<GRNode*>& node_segment : node_segment_list) {
  //   routing_segment_list.emplace_back(*node_segment.get_first(), *node_segment.get_second());
  // }
  // LayerCoord root_coord = gr_net.get_gr_driving_pin().get_protected_access_point().getGridLayerCoord();
  // std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC> key_coord_pin_map;
  // for (GRPin& gr_pin : gr_net.get_gr_pin_list()) {
  //   LayerCoord coord = gr_pin.get_protected_access_point().getGridLayerCoord();
  //   key_coord_pin_map[coord].insert(gr_pin.get_pin_idx());
  // }
  // gr_net.set_routing_tree(RTUtil::getTreeByFullFlow({root_coord}, routing_segment_list, key_coord_pin_map));
}

void GlobalRouter::updateDemand(GRModel& gr_model, GRNet& gr_net, ChangeType change_type)
{
  // std::vector<GridMap<GRNode>>& layer_node_map = gr_model.get_layer_node_map();

  // std::set<GRNode*> key_node_set;
  // for (GRPin& gr_pin : gr_net.get_gr_pin_list()) {
  //   LayerCoord coord = gr_pin.get_protected_access_point().getGridLayerCoord();
  //   GRNode* gr_node = &layer_node_map[coord.get_layer_idx()][coord.get_x()][coord.get_y()];
  //   key_node_set.insert(gr_node);
  // }
  // std::vector<Segment<GRNode*>> node_segment_list;
  // for (Segment<TNode<LayerCoord>*>& coord_segment : RTUtil::getSegListByTree(gr_net.get_routing_tree())) {
  //   LayerCoord first_coord = coord_segment.get_first()->value();
  //   LayerCoord second_coord = coord_segment.get_second()->value();

  //   GRNode* first_node = &layer_node_map[first_coord.get_layer_idx()][first_coord.get_x()][first_coord.get_y()];
  //   GRNode* second_node = &layer_node_map[second_coord.get_layer_idx()][second_coord.get_x()][second_coord.get_y()];

  //   node_segment_list.emplace_back(first_node, second_node);
  // }

  // std::map<GRNode*, std::set<Orientation>> usage_map;

  // if (node_segment_list.empty()) {
  //   // 单层的local net
  //   if (key_node_set.size() > 1) {
  //     LOG_INST.error(Loc::current(), "The net is not local!");
  //   }
  //   GRNode* local_node = *key_node_set.begin();
  //   for (Orientation orientation : {Orientation::kUp, Orientation::kDown}) {
  //     usage_map[local_node].insert(orientation);
  //   }
  // } else {
  //   // 跨gcell线网和多层的local_net
  //   for (Segment<GRNode*>& node_segment : node_segment_list) {
  //     GRNode* first_node = node_segment.get_first();
  //     GRNode* second_node = node_segment.get_second();
  //     Orientation orientation = RTUtil::getOrientation(*first_node, *second_node);
  //     if (orientation == Orientation::kNone || orientation == Orientation::kOblique) {
  //       LOG_INST.error(Loc::current(), "The orientation is error!");
  //     }
  //     Orientation oppo_orientation = RTUtil::getOppositeOrientation(orientation);

  //     GRNode* node_i = first_node;
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
  //   usage_node->updateDemand(gr_net.get_net_idx(), orientation_list, change_type);
  // }
}

void GlobalRouter::resetSingleNet(GRModel& gr_model)
{
  gr_model.set_gr_net_ref(nullptr);
  gr_model.set_routing_region(PlanarRect());
  gr_model.get_gr_task_list().clear();

  for (Segment<GRNode*>& node_segment : gr_model.get_node_segment_list()) {
    GRNode* first_node = node_segment.get_first();
    GRNode* second_node = node_segment.get_second();
    Orientation orientation = RTUtil::getOrientation(*first_node, *second_node);

    GRNode* node_i = first_node;
    while (true) {
      node_i->get_direction_set().clear();
      if (node_i == second_node) {
        break;
      }
      node_i = node_i->getNeighborNode(orientation);
    }
  }
  gr_model.get_node_segment_list().clear();
}

// manager open list

void GlobalRouter::pushToOpenList(GRModel& gr_model, GRNode* curr_node)
{
  PriorityQueue<GRNode*, std::vector<GRNode*>, CmpGRNodeCost>& open_queue = gr_model.get_open_queue();
  std::vector<GRNode*>& visited_node_list = gr_model.get_visited_node_list();

  open_queue.push(curr_node);
  curr_node->set_state(GRNodeState::kOpen);
  visited_node_list.push_back(curr_node);
}

GRNode* GlobalRouter::popFromOpenList(GRModel& gr_model)
{
  PriorityQueue<GRNode*, std::vector<GRNode*>, CmpGRNodeCost>& open_queue = gr_model.get_open_queue();

  GRNode* gr_node = nullptr;
  if (!open_queue.empty()) {
    gr_node = open_queue.top();
    open_queue.pop();
    gr_node->set_state(GRNodeState::kClose);
  }
  return gr_node;
}

// calculate known cost

double GlobalRouter::getKnowCost(GRModel& gr_model, GRNode* start_node, GRNode* end_node)
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
  cost += getNodeCost(gr_model, start_node, RTUtil::getOrientation(*start_node, *end_node));
  cost += getNodeCost(gr_model, end_node, RTUtil::getOrientation(*end_node, *start_node));
  cost += getKnowWireCost(gr_model, start_node, end_node);
  cost += getKnowCornerCost(gr_model, start_node, end_node);
  cost += getKnowViaCost(gr_model, start_node, end_node);
  return cost;
}

double GlobalRouter::getNodeCost(GRModel& gr_model, GRNode* curr_node, Orientation orientation)
{
#if 1
  double node_cost = 0;

  double env_cost = curr_node->getCost(gr_model.get_curr_net_idx(), orientation);
  node_cost += env_cost;

  return node_cost;
#else
  double env_cost = curr_node->getCost(gr_model.get_curr_net_idx(), orientation);

  double net_cost = 0;
  {
    const PlanarRect& curr_bounding_box = gr_model.get_curr_bounding_box();
    const GridMap<double>& curr_cost_map = gr_model.get_curr_cost_map();

    irt_int local_x = curr_node->get_x() - curr_bounding_box.get_lb_x();
    irt_int local_y = curr_node->get_y() - curr_bounding_box.get_lb_y();
    net_cost = (curr_cost_map.isInside(local_x, local_y) ? curr_cost_map[local_x][local_y] : 1);
  }

  double node_cost = env_cost + net_cost;
  return node_cost;
#endif
}

double GlobalRouter::getKnowWireCost(GRModel& gr_model, GRNode* start_node, GRNode* end_node)
{
  double gr_prefer_wire_unit = DM_INST.getConfig().gr_prefer_wire_unit;

  double wire_cost = 0;
  wire_cost += RTUtil::getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());
  wire_cost *= gr_prefer_wire_unit;
  return wire_cost;
}

double GlobalRouter::getKnowCornerCost(GRModel& gr_model, GRNode* start_node, GRNode* end_node)
{
  double gr_corner_unit = DM_INST.getConfig().gr_corner_unit;

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
      corner_cost += gr_corner_unit;
    } else if (direction_set.size() == 2) {
      LOG_INST.error(Loc::current(), "Direction set is error!");
    }
  }
  return corner_cost;
}

double GlobalRouter::getKnowViaCost(GRModel& gr_model, GRNode* start_node, GRNode* end_node)
{
  double gr_via_unit = DM_INST.getConfig().gr_via_unit;

  double via_cost = (gr_via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
  return via_cost;
}

// calculate estimate cost

double GlobalRouter::getEstimateCostToEnd(GRModel& gr_model, GRNode* curr_node)
{
  std::vector<GRGroup>& end_group_list = gr_model.get_end_group_list();

  double estimate_cost = DBL_MAX;
  for (GRGroup& end_group : end_group_list) {
    for (GRNode* end_node : end_group.get_gr_node_list()) {
      if (end_node->isClose()) {
        continue;
      }
      estimate_cost = std::min(estimate_cost, getEstimateCost(gr_model, curr_node, end_node));
    }
  }
  return estimate_cost;
}

double GlobalRouter::getEstimateCost(GRModel& gr_model, GRNode* start_node, GRNode* end_node)
{
  double estimate_cost = 0;
  estimate_cost += getEstimateWireCost(gr_model, start_node, end_node);
  estimate_cost += getEstimateCornerCost(gr_model, start_node, end_node);
  estimate_cost += getEstimateViaCost(gr_model, start_node, end_node);
  return estimate_cost;
}

double GlobalRouter::getEstimateWireCost(GRModel& gr_model, GRNode* start_node, GRNode* end_node)
{
  double gr_prefer_wire_unit = DM_INST.getConfig().gr_prefer_wire_unit;

  double wire_cost = 0;
  wire_cost += RTUtil::getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());
  wire_cost *= gr_prefer_wire_unit;
  return wire_cost;
}

double GlobalRouter::getEstimateCornerCost(GRModel& gr_model, GRNode* start_node, GRNode* end_node)
{
  double gr_corner_unit = DM_INST.getConfig().gr_corner_unit;

  double corner_cost = 0;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    if (RTUtil::isOblique(*start_node, *end_node)) {
      corner_cost += gr_corner_unit;
    }
  }
  return corner_cost;
}

double GlobalRouter::getEstimateViaCost(GRModel& gr_model, GRNode* start_node, GRNode* end_node)
{
  double gr_via_unit = DM_INST.getConfig().gr_via_unit;

  double via_cost = (gr_via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
  return via_cost;
}

void GlobalRouter::processGRModel(GRModel& gr_model)
{
  // 检查布线状态
  for (GRNet& gr_net : gr_model.get_gr_net_list()) {
    if (gr_net.get_routing_state() == RoutingState::kUnrouted) {
      LOG_INST.error(Loc::current(), "The routing_state is ", GetRoutingStateName()(gr_net.get_routing_state()), "!");
    }
  }
#pragma omp parallel for
  for (GRNet& gr_net : gr_model.get_gr_net_list()) {
    buildRoutingResult(gr_net);
  }
}

void GlobalRouter::buildRoutingResult(GRNet& gr_net)
{
  if (gr_net.get_routing_tree().get_root() == nullptr) {
    return;
  }
  std::function<Guide(LayerCoord&)> convertToGuide = [](LayerCoord& coord) {
    ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
    return Guide(LayerRect(RTUtil::getRealRectByGCell(coord, gcell_axis), coord.get_layer_idx()), coord);
  };
  gr_net.set_gr_result_tree(RTUtil::convertTree(gr_net.get_routing_tree(), convertToGuide));
}

bool GlobalRouter::stopGRModel(GRModel& gr_model)
{
  return (gr_model.get_gr_model_stat().get_max_access_overflow() <= 1);
}

#endif

#if 1  // update

void GlobalRouter::update(GRModel& gr_model)
{
  outputGuide(gr_model);
  for (GRNet& gr_net : gr_model.get_gr_net_list()) {
    Net* origin_net = gr_net.get_origin_net();
    origin_net->set_gr_result_tree(gr_net.get_gr_result_tree());
  }
}

void GlobalRouter::outputGuide(GRModel& gr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::string gr_temp_directory_path = DM_INST.getConfig().gr_temp_directory_path;

  std::ofstream* guide_file_stream = RTUtil::getOutputFileStream(gr_temp_directory_path + "route.guide");
  if (guide_file_stream == nullptr) {
    return;
  }
  for (GRNet& gr_net : gr_model.get_gr_net_list()) {
    RTUtil::pushStream(guide_file_stream, gr_net.get_origin_net()->get_net_name(), "\n(\n");

    for (Segment<TNode<Guide>*> guide_node_seg : RTUtil::getSegListByTree(gr_net.get_gr_result_tree())) {
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

#if 1  // plot gr_model

void GlobalRouter::outputCongestionMap(GRModel& gr_model)
{
  Die& die = DM_INST.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::string gr_temp_directory_path = DM_INST.getConfig().gr_temp_directory_path;

  GridMap<double> planar_overflow_map;
  planar_overflow_map.init(die.getXSize(), die.getYSize());

  for (RoutingLayer& routing_layer : routing_layer_list) {
    GridMap<GRNode>& node_map = gr_model.get_layer_node_map()[routing_layer.get_layer_idx()];
    for (irt_int grid_x = 0; grid_x < node_map.get_x_size(); grid_x++) {
      for (irt_int grid_y = 0; grid_y < node_map.get_y_size(); grid_y++) {
        GRNode& gr_node = node_map[grid_x][grid_y];
        std::map<Orientation, irt_int>& orien_access_supply_map = gr_node.get_orien_access_supply_map();
        std::map<Orientation, irt_int>& orien_access_demand_map = gr_node.get_orien_access_demand_map();

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
      = RTUtil::getOutputFileStream(RTUtil::getString(gr_temp_directory_path, "gr_model_", gr_model.get_curr_iter(), ".csv"));
  for (irt_int y = planar_overflow_map.get_y_size() - 1; y >= 0; y--) {
    for (irt_int x = 0; x < planar_overflow_map.get_x_size(); x++) {
      RTUtil::pushStream(csv_file, planar_overflow_map[x][y], ",");
    }
    RTUtil::pushStream(csv_file, "\n");
  }
  RTUtil::closeFileStream(csv_file);
}

void GlobalRouter::plotGRModel(GRModel& gr_model, irt_int curr_net_idx)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  Die& die = DM_INST.getDatabase().get_die();
  std::string gr_temp_directory_path = DM_INST.getConfig().gr_temp_directory_path;

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

  // gr_node_map
  GPStruct gr_node_map_struct("gr_node_map");
  for (GridMap<GRNode>& gr_node_map : gr_model.get_layer_node_map()) {
    for (irt_int grid_x = 0; grid_x < gr_node_map.get_x_size(); grid_x++) {
      for (irt_int grid_y = 0; grid_y < gr_node_map.get_y_size(); grid_y++) {
        GRNode& gr_node = gr_node_map[grid_x][grid_y];
        PlanarRect real_rect = RTUtil::getRealRectByGCell(gr_node.get_planar_coord(), gcell_axis);
        irt_int y_reduced_span = real_rect.getYSpan() / 25;
        irt_int y = real_rect.get_rt_y();

        GPBoundary gp_boundary;
        switch (gr_node.get_state()) {
          case GRNodeState::kNone:
            gp_boundary.set_data_type(static_cast<irt_int>(GPDataType::kNone));
            break;
          case GRNodeState::kOpen:
            gp_boundary.set_data_type(static_cast<irt_int>(GPDataType::kOpen));
            break;
          case GRNodeState::kClose:
            gp_boundary.set_data_type(static_cast<irt_int>(GPDataType::kClose));
            break;
          default:
            LOG_INST.error(Loc::current(), "The type is error!");
            break;
        }
        gp_boundary.set_rect(real_rect);
        gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        gr_node_map_struct.push(gp_boundary);

        y -= y_reduced_span;
        GPText gp_text_node_coord;
        gp_text_node_coord.set_coord(real_rect.get_lb_x(), y);
        gp_text_node_coord.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_node_coord.set_message(RTUtil::getString("(", grid_x, " , ", grid_y, " , ", gr_node.get_layer_idx(), ")"));
        gp_text_node_coord.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        gp_text_node_coord.set_presentation(GPTextPresentation::kLeftMiddle);
        gr_node_map_struct.push(gp_text_node_coord);

        y -= y_reduced_span;
        GPText gp_text_cross_wire_demand;
        gp_text_cross_wire_demand.set_coord(real_rect.get_lb_x(), y);
        gp_text_cross_wire_demand.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_cross_wire_demand.set_message(RTUtil::getString("cross_wire_demand: ", gr_node.get_cross_wire_demand()));
        gp_text_cross_wire_demand.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        gp_text_cross_wire_demand.set_presentation(GPTextPresentation::kLeftMiddle);
        gr_node_map_struct.push(gp_text_cross_wire_demand);

        y -= y_reduced_span;
        GPText gp_text_local_wire_demand;
        gp_text_local_wire_demand.set_coord(real_rect.get_lb_x(), y);
        gp_text_local_wire_demand.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_local_wire_demand.set_message(RTUtil::getString("local_wire_demand: ", gr_node.get_local_wire_demand()));
        gp_text_local_wire_demand.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        gp_text_local_wire_demand.set_presentation(GPTextPresentation::kLeftMiddle);
        gr_node_map_struct.push(gp_text_local_wire_demand);

        y -= y_reduced_span;
        GPText gp_text_whole_via_demand;
        gp_text_whole_via_demand.set_coord(real_rect.get_lb_x(), y);
        gp_text_whole_via_demand.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_whole_via_demand.set_message(RTUtil::getString("whole_via_demand: ", gr_node.get_whole_via_demand()));
        gp_text_whole_via_demand.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        gp_text_whole_via_demand.set_presentation(GPTextPresentation::kLeftMiddle);
        gr_node_map_struct.push(gp_text_whole_via_demand);

        y -= y_reduced_span;
        GPText gp_text_net_via_demand_map;
        gp_text_net_via_demand_map.set_coord(real_rect.get_lb_x(), y);
        gp_text_net_via_demand_map.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_net_via_demand_map.set_message("net_via_demand_map: ");
        gp_text_net_via_demand_map.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        gp_text_net_via_demand_map.set_presentation(GPTextPresentation::kLeftMiddle);
        gr_node_map_struct.push(gp_text_net_via_demand_map);

        if (!gr_node.get_net_via_demand_map().empty()) {
          y -= y_reduced_span;
          GPText gp_text_net_via_demand_map_info;
          gp_text_net_via_demand_map_info.set_coord(real_rect.get_lb_x(), y);
          gp_text_net_via_demand_map_info.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
          std::string net_via_demand_map_message = "--";
          for (auto& [net_idx, via_demand] : gr_node.get_net_via_demand_map()) {
            net_via_demand_map_message += RTUtil::getString("(", net_idx, ")(", via_demand, ")");
          }
          gp_text_net_via_demand_map_info.set_message(net_via_demand_map_message);
          gp_text_net_via_demand_map_info.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
          gp_text_net_via_demand_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
          gr_node_map_struct.push(gp_text_net_via_demand_map_info);
        }

        y -= y_reduced_span;
        GPText gp_text_whole_access_demand;
        gp_text_whole_access_demand.set_coord(real_rect.get_lb_x(), y);
        gp_text_whole_access_demand.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_whole_access_demand.set_message(RTUtil::getString("whole_access_demand: ", gr_node.get_whole_access_demand()));
        gp_text_whole_access_demand.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        gp_text_whole_access_demand.set_presentation(GPTextPresentation::kLeftMiddle);
        gr_node_map_struct.push(gp_text_whole_access_demand);

        y -= y_reduced_span;
        GPText gp_text_net_orien_access_demand_map;
        gp_text_net_orien_access_demand_map.set_coord(real_rect.get_lb_x(), y);
        gp_text_net_orien_access_demand_map.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_net_orien_access_demand_map.set_message("net_orien_access_demand_map: ");
        gp_text_net_orien_access_demand_map.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        gp_text_net_orien_access_demand_map.set_presentation(GPTextPresentation::kLeftMiddle);
        gr_node_map_struct.push(gp_text_net_orien_access_demand_map);

        if (!gr_node.get_net_orien_access_demand_map().empty()) {
          y -= y_reduced_span;
          GPText gp_text_net_orien_access_demand_map_info;
          gp_text_net_orien_access_demand_map_info.set_coord(real_rect.get_lb_x(), y);
          gp_text_net_orien_access_demand_map_info.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
          std::string net_orien_access_demand_map_message = "--";
          for (auto& [net_idx, orien_wire_demand_map] : gr_node.get_net_orien_access_demand_map()) {
            net_orien_access_demand_map_message += RTUtil::getString("(", net_idx, ")");
            for (auto& [orientation, wire_demand] : orien_wire_demand_map) {
              net_orien_access_demand_map_message += RTUtil::getString("(", GetOrientationName()(orientation), ":", wire_demand, ")");
            }
          }
          gp_text_net_orien_access_demand_map_info.set_message(net_orien_access_demand_map_message);
          gp_text_net_orien_access_demand_map_info.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
          gp_text_net_orien_access_demand_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
          gr_node_map_struct.push(gp_text_net_orien_access_demand_map_info);
        }

        y -= y_reduced_span;
        GPText gp_text_orien_access_supply_map;
        gp_text_orien_access_supply_map.set_coord(real_rect.get_lb_x(), y);
        gp_text_orien_access_supply_map.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_orien_access_supply_map.set_message("orien_access_supply_map: ");
        gp_text_orien_access_supply_map.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        gp_text_orien_access_supply_map.set_presentation(GPTextPresentation::kLeftMiddle);
        gr_node_map_struct.push(gp_text_orien_access_supply_map);

        if (!gr_node.get_orien_access_supply_map().empty()) {
          y -= y_reduced_span;
          GPText gp_text_orien_access_supply_map_info;
          gp_text_orien_access_supply_map_info.set_coord(real_rect.get_lb_x(), y);
          gp_text_orien_access_supply_map_info.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
          std::string orien_access_supply_map_message = "--";
          for (auto& [orientation, access_supply] : gr_node.get_orien_access_supply_map()) {
            orien_access_supply_map_message += RTUtil::getString("(", GetOrientationName()(orientation), ":", access_supply, ")");
          }
          gp_text_orien_access_supply_map_info.set_message(orien_access_supply_map_message);
          gp_text_orien_access_supply_map_info.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
          gp_text_orien_access_supply_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
          gr_node_map_struct.push(gp_text_orien_access_supply_map_info);
        }

        y -= y_reduced_span;
        GPText gp_text_orien_access_demand_map;
        gp_text_orien_access_demand_map.set_coord(real_rect.get_lb_x(), y);
        gp_text_orien_access_demand_map.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_orien_access_demand_map.set_message("orien_access_demand_map: ");
        gp_text_orien_access_demand_map.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        gp_text_orien_access_demand_map.set_presentation(GPTextPresentation::kLeftMiddle);
        gr_node_map_struct.push(gp_text_orien_access_demand_map);

        if (!gr_node.get_orien_access_demand_map().empty()) {
          y -= y_reduced_span;
          GPText gp_text_orien_access_demand_map_info;
          gp_text_orien_access_demand_map_info.set_coord(real_rect.get_lb_x(), y);
          gp_text_orien_access_demand_map_info.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
          std::string orien_access_demand_map_message = "--";
          for (auto& [orientation, access_demand] : gr_node.get_orien_access_demand_map()) {
            orien_access_demand_map_message += RTUtil::getString("(", GetOrientationName()(orientation), ":", access_demand, ")");
          }
          gp_text_orien_access_demand_map_info.set_message(orien_access_demand_map_message);
          gp_text_orien_access_demand_map_info.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
          gp_text_orien_access_demand_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
          gr_node_map_struct.push(gp_text_orien_access_demand_map_info);
        }

        y -= y_reduced_span;
        GPText gp_text_resource_supply;
        gp_text_resource_supply.set_coord(real_rect.get_lb_x(), y);
        gp_text_resource_supply.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_resource_supply.set_message(RTUtil::getString("resource_supply: ", gr_node.get_resource_supply()));
        gp_text_resource_supply.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        gp_text_resource_supply.set_presentation(GPTextPresentation::kLeftMiddle);
        gr_node_map_struct.push(gp_text_resource_supply);

        y -= y_reduced_span;
        GPText gp_text_resource_demand;
        gp_text_resource_demand.set_coord(real_rect.get_lb_x(), y);
        gp_text_resource_demand.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_resource_demand.set_message(RTUtil::getString("resource_demand: ", gr_node.get_resource_demand()));
        gp_text_resource_demand.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        gp_text_resource_demand.set_presentation(GPTextPresentation::kLeftMiddle);
        gr_node_map_struct.push(gp_text_resource_demand);

        y -= y_reduced_span;
        GPText gp_text_passed_net_set;
        gp_text_passed_net_set.set_coord(real_rect.get_lb_x(), y);
        gp_text_passed_net_set.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_passed_net_set.set_message("passed_net_set: ");
        gp_text_passed_net_set.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        gp_text_passed_net_set.set_presentation(GPTextPresentation::kLeftMiddle);
        gr_node_map_struct.push(gp_text_passed_net_set);

        if (!gr_node.get_passed_net_set().empty()) {
          y -= y_reduced_span;
          GPText gp_text_passed_net_set_info;
          gp_text_passed_net_set_info.set_coord(real_rect.get_lb_x(), y);
          gp_text_passed_net_set_info.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
          std::string passed_net_set_info_message = "--";
          for (irt_int net_idx : gr_node.get_passed_net_set()) {
            passed_net_set_info_message += RTUtil::getString("(", net_idx, ")");
          }
          gp_text_passed_net_set_info.set_message(passed_net_set_info_message);
          gp_text_passed_net_set_info.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
          gp_text_passed_net_set_info.set_presentation(GPTextPresentation::kLeftMiddle);
          gr_node_map_struct.push(gp_text_passed_net_set_info);
        }

        y -= y_reduced_span;
        GPText gp_text_direction_set;
        gp_text_direction_set.set_coord(real_rect.get_lb_x(), y);
        gp_text_direction_set.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
        gp_text_direction_set.set_message("direction_set: ");
        gp_text_direction_set.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        gp_text_direction_set.set_presentation(GPTextPresentation::kLeftMiddle);
        gr_node_map_struct.push(gp_text_direction_set);

        if (!gr_node.get_direction_set().empty()) {
          y -= y_reduced_span;
          GPText gp_text_direction_set_info;
          gp_text_direction_set_info.set_coord(real_rect.get_lb_x(), y);
          gp_text_direction_set_info.set_text_type(static_cast<irt_int>(GPDataType::kInfo));
          std::string direction_set_info_message = "--";
          for (Direction direction : gr_node.get_direction_set()) {
            direction_set_info_message += RTUtil::getString("(", GetDirectionName()(direction), ")");
          }
          gp_text_direction_set_info.set_message(direction_set_info_message);
          gp_text_direction_set_info.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
          gp_text_direction_set_info.set_presentation(GPTextPresentation::kLeftMiddle);
          gr_node_map_struct.push(gp_text_direction_set_info);
        }
      }
    }
  }
  gp_gds.addStruct(gr_node_map_struct);

  // neighbor_map
  GPStruct neighbor_map_struct("neighbor_map");
  for (GridMap<GRNode>& gr_node_map : gr_model.get_layer_node_map()) {
    for (irt_int grid_x = 0; grid_x < gr_node_map.get_x_size(); grid_x++) {
      for (irt_int grid_y = 0; grid_y < gr_node_map.get_y_size(); grid_y++) {
        GRNode& gr_node = gr_node_map[grid_x][grid_y];
        PlanarRect real_rect = RTUtil::getRealRectByGCell(gr_node.get_planar_coord(), gcell_axis);
        irt_int lb_x = real_rect.get_lb_x();
        irt_int lb_y = real_rect.get_lb_y();
        irt_int rt_x = real_rect.get_rt_x();
        irt_int rt_y = real_rect.get_rt_y();
        irt_int mid_x = (lb_x + rt_x) / 2;
        irt_int mid_y = (lb_y + rt_y) / 2;
        irt_int x_reduced_span = (rt_x - lb_x) / 4;
        irt_int y_reduced_span = (rt_y - lb_y) / 4;
        irt_int width = std::min(x_reduced_span, y_reduced_span) / 2;

        for (auto& [orientation, neighbor_node] : gr_node.get_neighbor_ptr_map()) {
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
          gp_path.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
          gp_path.set_width(width);
          gp_path.set_data_type(static_cast<irt_int>(GPDataType::kNeighbor));
          neighbor_map_struct.push(gp_path);
        }
      }
    }
  }
  gp_gds.addStruct(neighbor_map_struct);

  // source_region_query_map
  // std::vector<std::pair<GRSourceType, GPDataType>> source_graph_pair_list = {{GRSourceType::kBlockage, GPDataType::kBlockage},
  //                                                                             {GRSourceType::kNetShape, GPDataType::kNetShape},
  //                                                                             {GRSourceType::kReservedVia, GPDataType::kReservedVia}};
  // std::vector<GridMap<GRNode>>& layer_node_map = gr_model.get_layer_node_map();
  // for (irt_int layer_idx = 0; layer_idx < static_cast<irt_int>(layer_node_map.size()); layer_idx++) {
  //   GridMap<GRNode>& node_map = layer_node_map[layer_idx];
  //   for (irt_int grid_x = 0; grid_x < node_map.get_x_size(); grid_x++) {
  //     for (irt_int grid_y = 0; grid_y < node_map.get_y_size(); grid_y++) {
  //       GRNode& gr_node = node_map[grid_x][grid_y];
  //       for (auto& [gr_source_type, gp_graph_type] : source_graph_pair_list) {
  //         for (auto& [info, rect_set] : DC_INST.getLayerInfoRectMap(gr_node.getRegionQuery(gr_source_type), true)[layer_idx]) {
  //           GPStruct net_rect_struct(RTUtil::getString(GetGRSourceTypeName()(gr_source_type), "@", info.get_net_idx()));
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
  // for (GRNet& gr_net : gr_model.get_gr_net_list()) {
  //   GPStruct net_struct(RTUtil::getString("net_", gr_net.get_net_idx()));

  //   if (curr_net_idx == -1 || gr_net.get_net_idx() == curr_net_idx) {
  //     for (GRPin& gr_pin : gr_net.get_gr_pin_list()) {
  //       LayerCoord coord = gr_pin.get_protected_access_point().getGridLayerCoord();
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
  //     gp_boundary.set_rect(gr_net.get_bounding_box().get_real_rect());
  //     net_struct.push(gp_boundary);
  //   }
  //   for (Segment<TNode<LayerCoord>*>& segment : RTUtil::getSegListByTree(gr_net.get_routing_tree())) {
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
  GP_INST.plot(gp_gds, gr_temp_directory_path + "gr_model.gds");
}

#endif

}  // namespace irt
