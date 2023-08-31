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

#include "DRCChecker.hpp"
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

  LOG_INST.info(Loc::current(), "The ", GetStageName()(Stage::kGlobalRouter), " completed!", monitor.getStatsInfo());
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
  for (size_t layer_idx = 0; layer_idx < layer_node_map.size(); layer_idx++) {
    GridMap<GRNode>& node_map = layer_node_map[layer_idx];
    node_map.init(die.getXSize(), die.getYSize());
    for (irt_int x = 0; x < die.getXSize(); x++) {
      for (irt_int y = 0; y < die.getYSize(); y++) {
        GRNode& gr_node = node_map[x][y];
        gr_node.set_coord(x, y);
        gr_node.set_layer_idx(static_cast<irt_int>(layer_idx));
        gr_node.set_base_region(RTUtil::getRealRect(x, y, gcell_axis));
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
  gr_net.set_gr_driving_pin(GRPin(net.get_driving_pin()));
  gr_net.set_bounding_box(net.get_bounding_box());
  gr_net.set_ra_cost_map(net.get_ra_cost_map());
  return gr_net;
}

void GlobalRouter::buildGRModel(GRModel& gr_model)
{
  buildNeighborMap(gr_model);
  updateNetFixedRectMap(gr_model);
  updateNetReservedViaMap(gr_model);
  updateWholeDemand(gr_model);
  updateNetDemandMap(gr_model);
  updateNodeSupply(gr_model);
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

void GlobalRouter::updateNetFixedRectMap(GRModel& gr_model)
{
  std::vector<Blockage>& routing_blockage_list = DM_INST.getDatabase().get_routing_blockage_list();

  for (const Blockage& routing_blockage : routing_blockage_list) {
    LayerRect blockage_real_rect(routing_blockage.get_real_rect(), routing_blockage.get_layer_idx());
    addRectToEnv(gr_model, GRSourceType::kBlockAndPin, DRCRect(-1, blockage_real_rect, true));
  }
  for (GRNet& gr_net : gr_model.get_gr_net_list()) {
    for (GRPin& gr_pin : gr_net.get_gr_pin_list()) {
      for (const EXTLayerRect& routing_shape : gr_pin.get_routing_shape_list()) {
        LayerRect shape_real_rect(routing_shape.get_real_rect(), routing_shape.get_layer_idx());
        addRectToEnv(gr_model, GRSourceType::kBlockAndPin, DRCRect(gr_net.get_net_idx(), shape_real_rect, true));
      }
    }
  }
}

void GlobalRouter::addRectToEnv(GRModel& gr_model, GRSourceType gr_source_type, DRCRect drc_rect)
{
  if (drc_rect.get_is_routing() == false) {
    return;
  }
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  EXTPlanarRect& die = DM_INST.getDatabase().get_die();

  std::vector<GridMap<GRNode>>& layer_node_map = gr_model.get_layer_node_map();

  for (const LayerRect& max_scope_real_rect : DC_INST.getMaxScope(drc_rect)) {
    LayerRect max_scope_regular_rect = RTUtil::getRegularRect(max_scope_real_rect, die.get_real_rect());
    PlanarRect max_scope_grid_rect = RTUtil::getClosedGridRect(max_scope_regular_rect, gcell_axis);
    for (irt_int x = max_scope_grid_rect.get_lb_x(); x <= max_scope_grid_rect.get_rt_x(); x++) {
      for (irt_int y = max_scope_grid_rect.get_lb_y(); y <= max_scope_grid_rect.get_rt_y(); y++) {
        GRNode& gr_node = layer_node_map[drc_rect.get_layer_rect().get_layer_idx()][x][y];
        DC_INST.addEnvRectList(gr_node.getRegionQuery(gr_source_type), drc_rect);
      }
    }
  }
}

void GlobalRouter::updateNetReservedViaMap(GRModel& gr_model)
{
  irt_int bottom_routing_layer_idx = DM_INST.getConfig().bottom_routing_layer_idx;
  irt_int top_routing_layer_idx = DM_INST.getConfig().top_routing_layer_idx;

  for (GRNet& gr_net : gr_model.get_gr_net_list()) {
    std::set<LayerCoord, CmpLayerCoordByXASC> real_coord_set;
    for (GRPin& gr_pin : gr_net.get_gr_pin_list()) {
      for (LayerCoord& real_coord : gr_pin.getRealCoordList()) {
        real_coord_set.insert(real_coord);
      }
    }
    for (const LayerCoord& real_coord : real_coord_set) {
      irt_int layer_idx = real_coord.get_layer_idx();
      for (irt_int via_below_layer_idx :
           RTUtil::getReservedViaBelowLayerIdxList(layer_idx, bottom_routing_layer_idx, top_routing_layer_idx)) {
        std::vector<Segment<LayerCoord>> segment_list;
        segment_list.emplace_back(LayerCoord(real_coord.get_planar_coord(), via_below_layer_idx),
                                  LayerCoord(real_coord.get_planar_coord(), via_below_layer_idx + 1));
        for (DRCRect& drc_rect : DC_INST.getDRCRectList(gr_net.get_net_idx(), segment_list)) {
          addRectToEnv(gr_model, GRSourceType::kReservedVia, drc_rect);
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
        irt_int whole_wire_demand = 0;
        if (routing_layer.isPreferH()) {
          whole_wire_demand = gr_node.get_base_region().getXSpan();
        } else {
          whole_wire_demand = gr_node.get_base_region().getYSpan();
        }
        gr_node.set_whole_wire_demand(whole_wire_demand);

        irt_int whole_via_demand = routing_layer.get_min_area() / routing_layer.get_min_width();
        gr_node.set_whole_via_demand(whole_via_demand);
      }
    }
  }
}

void GlobalRouter::updateNetDemandMap(GRModel& gr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  std::vector<GridMap<GRNode>>& layer_node_map = gr_model.get_layer_node_map();

  for (GRNet& gr_net : gr_model.get_gr_net_list()) {
    std::map<PlanarCoord, std::vector<PlanarCoord>, CmpPlanarCoordByXASC> grid_pin_coord_map;
    for (GRPin& gr_pin : gr_net.get_gr_pin_list()) {
      for (AccessPoint& access_point : gr_pin.get_access_point_list()) {
        grid_pin_coord_map[access_point.get_grid_coord()].push_back(access_point.get_real_coord());
      }
    }
    /**
     * 取布线资源的下界(既min)，和gr_via取min_area一样，使得overflow向下调整
     */
    for (auto& [grid_coord, pin_coord_list] : grid_pin_coord_map) {
      for (irt_int layer_idx = 0; layer_idx < static_cast<irt_int>(layer_node_map.size()); layer_idx++) {
        RoutingLayer& routing_layer = routing_layer_list[layer_idx];

        GRNode& gr_node = layer_node_map[layer_idx][grid_coord.get_x()][grid_coord.get_y()];
        PlanarRect& base_region = gr_node.get_base_region();
        std::map<irt_int, std::map<Orientation, irt_int>>& net_orien_wire_demand_map = gr_node.get_net_orien_wire_demand_map();

        if (routing_layer.isPreferH()) {
          irt_int min_west_demand = INT_MAX;
          irt_int min_east_demand = INT_MAX;
          for (PlanarCoord& pin_coord : pin_coord_list) {
            min_west_demand = std::min(min_west_demand, std::abs(pin_coord.get_x() - base_region.get_lb_x()));
            min_east_demand = std::min(min_east_demand, std::abs(pin_coord.get_x() - base_region.get_rt_x()));
          }
          net_orien_wire_demand_map[gr_net.get_net_idx()][Orientation::kWest] = min_west_demand;
          net_orien_wire_demand_map[gr_net.get_net_idx()][Orientation::kEast] = min_east_demand;
        } else {
          irt_int min_south_demand = INT_MAX;
          irt_int min_north_demand = INT_MAX;
          for (PlanarCoord& pin_coord : pin_coord_list) {
            min_south_demand = std::min(min_south_demand, std::abs(pin_coord.get_x() - base_region.get_lb_x()));
            min_north_demand = std::min(min_north_demand, std::abs(pin_coord.get_x() - base_region.get_rt_x()));
          }
          net_orien_wire_demand_map[gr_net.get_net_idx()][Orientation::kSouth] = min_south_demand;
          net_orien_wire_demand_map[gr_net.get_net_idx()][Orientation::kNorth] = min_north_demand;
        }
      }
    }
  }
}

void GlobalRouter::updateNodeSupply(GRModel& gr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  double supply_utilization_rate = DM_INST.getConfig().supply_utilization_rate;

  std::vector<GridMap<GRNode>>& layer_node_map = gr_model.get_layer_node_map();
  // track supply
  for (irt_int layer_idx = 0; layer_idx < static_cast<irt_int>(layer_node_map.size()); layer_idx++) {
    RoutingLayer& routing_layer = routing_layer_list[layer_idx];
    GridMap<GRNode>& node_map = layer_node_map[layer_idx];
#pragma omp parallel for collapse(2)
    for (irt_int x = 0; x < node_map.get_x_size(); x++) {
      for (irt_int y = 0; y < node_map.get_y_size(); y++) {
        GRNode& gr_node = node_map[x][y];

        std::vector<PlanarRect> wire_list = getWireList(gr_node, routing_layer);
        if (!wire_list.empty()) {
          irt_int real_whole_wire_demand = wire_list.front().getArea() / routing_layer.get_min_width();
          irt_int gcell_whole_wire_demand = 0;
          if (routing_layer.isPreferH()) {
            gcell_whole_wire_demand = gr_node.get_base_region().getXSpan();
          } else {
            gcell_whole_wire_demand = gr_node.get_base_region().getYSpan();
          }
          if (real_whole_wire_demand != gcell_whole_wire_demand) {
            LOG_INST.error(Loc::current(), "The real_whole_wire_demand and gcell_whole_wire_demand are not equal!");
          }
        }
        for (GRSourceType gr_source_type : {GRSourceType::kBlockAndPin, GRSourceType::kReservedVia}) {
          for (const auto& [net_idx, rect_set] : DC_INST.getLayerNetRectMap(gr_node.getRegionQuery(gr_source_type), true)[layer_idx]) {
            for (const LayerRect& rect : rect_set) {
              for (const LayerRect& min_scope_real_rect : DC_INST.getMinScope(DRCRect(net_idx, rect, true))) {
                std::vector<PlanarRect> new_wire_list;
                for (PlanarRect& wire : wire_list) {
                  if (RTUtil::isOpenOverlap(min_scope_real_rect, wire)) {
                    // 要切
                    std::vector<PlanarRect> split_rect_list
                        = RTUtil::getSplitRectList(wire, min_scope_real_rect, routing_layer.get_direction());
                    new_wire_list.insert(new_wire_list.end(), split_rect_list.begin(), split_rect_list.end());
                  } else {
                    // 不切
                    new_wire_list.push_back(wire);
                  }
                }
                wire_list = new_wire_list;
              }
            }
          }
        }
        irt_int access_supply = 0;
        irt_int resource_supply = 0;
        for (PlanarRect& wire : wire_list) {
          irt_int supply = wire.getArea() / routing_layer.get_min_width();
          if (supply < gr_node.get_whole_via_demand()) {
            continue;
          }
          if (supply == gr_node.get_whole_wire_demand()) {
            access_supply++;
          }
          resource_supply += supply;
        }
        access_supply *= supply_utilization_rate;
        std::map<Orientation, irt_int>& orien_access_supply_map = gr_node.get_orien_access_supply_map();
        if (routing_layer_list[layer_idx].isPreferH()) {
          orien_access_supply_map.insert({Orientation::kEast, access_supply});
          orien_access_supply_map.insert({Orientation::kWest, access_supply});
        } else {
          orien_access_supply_map.insert({Orientation::kNorth, access_supply});
          orien_access_supply_map.insert({Orientation::kSouth, access_supply});
        }
        resource_supply *= supply_utilization_rate;
        gr_node.set_resource_supply(resource_supply);
      }
    }
  }
}

std::vector<PlanarRect> GlobalRouter::getWireList(GRNode& gr_node, RoutingLayer& routing_layer)
{
  irt_int real_lb_x = gr_node.get_base_region().get_lb_x();
  irt_int real_lb_y = gr_node.get_base_region().get_lb_y();
  irt_int real_rt_x = gr_node.get_base_region().get_rt_x();
  irt_int real_rt_y = gr_node.get_base_region().get_rt_y();
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
        if (gr_node.get_whole_wire_demand() < 0) {
          LOG_INST.error(Loc::current(), "The whole_wire_demand < 0!");
        }
        if (gr_node.get_whole_via_demand() < 0) {
          LOG_INST.error(Loc::current(), "The whole_via_demand < 0!");
        }
        for (auto& [net_idx, orien_wire_demand_map] : gr_node.get_net_orien_wire_demand_map()) {
          if (orien_wire_demand_map.empty()) {
            LOG_INST.error(Loc::current(), "The orien_wire_demand_map is empty!");
          }
          for (auto& [orientation, wire_demand] : orien_wire_demand_map) {
            if (wire_demand < 0) {
              LOG_INST.error(Loc::current(), "The wire_demand < 0!");
            }
          }
        }
        std::map<Orientation, irt_int>& orien_access_supply_map = gr_node.get_orien_access_supply_map();
        if (routing_h) {
          if (!RTUtil::exist(orien_access_supply_map, Orientation::kEast) || !RTUtil::exist(orien_access_supply_map, Orientation::kWest)) {
            LOG_INST.error(Loc::current(), "The orientation is error!");
          }
        }
        if (routing_v) {
          if (!RTUtil::exist(orien_access_supply_map, Orientation::kNorth)
              || !RTUtil::exist(orien_access_supply_map, Orientation::kSouth)) {
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

  RTUtil::pushStream(python_file, "## 导入绘图需要用到的python库", "\n");
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
  RTUtil::pushStream(python_file, "    hm=sns.heatmap(array_data,cmap='Greens')", "\n");
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
    LOG_INST.info(Loc::current(), "****** Start Iteration(", iter, "/", gr_max_iter_num, ") ******");
    gr_model.set_curr_iter(iter);
    resetGRModel(gr_model);
    routeGRModel(gr_model);
    processGRModel(gr_model);
    countGRModel(gr_model);
    reportGRModel(gr_model);
    // plotGRModel(gr_model);
    outputCongestionMap(gr_model);
    LOG_INST.info(Loc::current(), "****** End Iteration(", iter, "/", gr_max_iter_num, ")", iter_monitor.getStatsInfo(), " ******");
    if (stopGRModel(gr_model)) {
      LOG_INST.info(Loc::current(), "****** Reached the stopping condition, ending the iteration prematurely! ******");
      gr_model.set_curr_iter(-1);
      break;
    }
  }
}

void GlobalRouter::resetGRModel(GRModel& gr_model)
{
  if (gr_model.get_curr_iter() == 1) {
    // sortGRModel(gr_model);
  } else {
    std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
    double gr_resource_history_cost_unit = DM_INST.getConfig().gr_resource_history_cost_unit;
    double gr_access_history_cost_unit = DM_INST.getConfig().gr_access_history_cost_unit;

    std::vector<GridMap<GRNode>>& layer_node_map = gr_model.get_layer_node_map();

    std::set<irt_int> violation_net_set;

    for (RoutingLayer& routing_layer : routing_layer_list) {
      GridMap<GRNode>& node_map = layer_node_map[routing_layer.get_layer_idx()];
      for (irt_int grid_x = 0; grid_x < node_map.get_x_size(); grid_x++) {
        for (irt_int grid_y = 0; grid_y < node_map.get_y_size(); grid_y++) {
          GRNode& gr_node = node_map[grid_x][grid_y];
          std::set<irt_int>& contribution_net_set = gr_node.get_contribution_net_set();
          double resource_history_cost = gr_node.get_resource_history_cost();
          std::map<Orientation, double>& orien_access_history_cost_map = gr_node.get_orien_access_history_cost_map();

          double resource_overflow = gr_node.calcCost(gr_node.get_resource_demand(), gr_node.get_resource_supply());
          if (resource_overflow > 1) {
            gr_node.set_resource_history_cost(resource_history_cost + gr_resource_history_cost_unit);
            violation_net_set.insert(contribution_net_set.begin(), contribution_net_set.end());
          }

          if (routing_layer.isPreferH()) {
            for (Orientation orientation : {Orientation::kEast, Orientation::kWest}) {
              double access_overflow = gr_node.calcCost(gr_node.get_orien_access_demand_map()[orientation],
                                                        gr_node.get_orien_access_supply_map()[orientation]);
              if (access_overflow > 1) {
                orien_access_history_cost_map[orientation] += gr_access_history_cost_unit;
                violation_net_set.insert(contribution_net_set.begin(), contribution_net_set.end());
              }
            }
          } else {
            for (Orientation orientation : {Orientation::kSouth, Orientation::kNorth}) {
              double access_overflow = gr_node.calcCost(gr_node.get_orien_access_demand_map()[orientation],
                                                        gr_node.get_orien_access_supply_map()[orientation]);
              if (access_overflow > 1) {
                orien_access_history_cost_map[orientation] += gr_access_history_cost_unit;
                violation_net_set.insert(contribution_net_set.begin(), contribution_net_set.end());
              }
            }
          }
        }
      }
    }

    std::vector<GRNet>& gr_net_list = gr_model.get_gr_net_list();
    for (irt_int net_idx : violation_net_set) {
      GRNet& gr_net = gr_net_list[net_idx];
      // 将env中的布线结果清空
      updateDemand(gr_model, gr_net_list[net_idx], ChangeType::kDel);
      // 清空routing_tree
      gr_net.get_routing_tree().clear();
      gr_net.set_routing_state(RoutingState::kUnrouted);
    }
  }
}

void GlobalRouter::sortGRModel(GRModel& gr_model)
{
  if (gr_model.get_curr_iter() != 1) {
    return;
  }
  Monitor monitor;
  LOG_INST.info(Loc::current(), "Sorting all nets beginning...");

  std::vector<GRNet>& gr_net_list = gr_model.get_gr_net_list();
  std::sort(gr_net_list.begin(), gr_net_list.end(), [&](GRNet& net1, GRNet& net2) { return sortByMultiLevel(net1, net2); });

  LOG_INST.info(Loc::current(), "Sorting all nets completed!", monitor.getStatsInfo());
}

bool GlobalRouter::sortByMultiLevel(GRNet& net1, GRNet& net2)
{
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

void GlobalRouter::routeGRModel(GRModel& gr_model)
{
  Monitor monitor;

  std::vector<GRNet>& gr_net_list = gr_model.get_gr_net_list();

  irt_int batch_size = RTUtil::getBatchSize(gr_net_list.size());

  Monitor stage_monitor;
  for (size_t i = 0; i < gr_net_list.size(); i++) {
    routeGRNet(gr_model, gr_net_list[i]);
    if ((i + 1) % batch_size == 0) {
      LOG_INST.info(Loc::current(), "Routed ", (i + 1), " nets", stage_monitor.getStatsInfo());
    }
  }
  LOG_INST.info(Loc::current(), "Routed ", gr_net_list.size(), " nets", monitor.getStatsInfo());
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
    RTUtil::pushStream(gr_dataset, "pin", " ", gr_pin.get_pin_idx(), "\n");
    for (LayerCoord& coord : gr_pin.getGridCoordList()) {
      RTUtil::pushStream(gr_dataset, coord.get_x(), " ", coord.get_y(), " ", coord.get_layer_idx(), "\n");
    }
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
      for (LayerCoord& coord : gr_pin.getGridCoordList()) {
        planar_coord_list.push_back(coord.get_planar_coord());
      }
    }
    std::sort(planar_coord_list.begin(), planar_coord_list.end(), CmpPlanarCoordByXASC());
    planar_coord_list.erase(std::unique(planar_coord_list.begin(), planar_coord_list.end()), planar_coord_list.end());

    if (planar_coord_list.size() == 1) {
      GRTask gr_task;
      for (GRPin& gr_pin : gr_net.get_gr_pin_list()) {
        GRGroup gr_group;
        for (LayerCoord& coord : gr_pin.getGridCoordList()) {
          gr_group.get_gr_node_list().push_back(&layer_node_map[coord.get_layer_idx()][coord.get_x()][coord.get_y()]);
        }
        gr_task.get_gr_group_list().push_back(gr_group);
      }
      gr_model.get_gr_task_list().push_back(gr_task);
    } else {
      // pin的GRGroup
      std::map<PlanarCoord, std::vector<GRGroup>, CmpPlanarCoordByXASC> key_planar_group_map;
      for (GRPin& gr_pin : gr_net.get_gr_pin_list()) {
        GRGroup gr_group;
        for (LayerCoord& coord : gr_pin.getGridCoordList()) {
          gr_group.get_gr_node_list().push_back(&layer_node_map[coord.get_layer_idx()][coord.get_x()][coord.get_y()]);
        }
        key_planar_group_map[gr_pin.getGridCoordList().front().get_planar_coord()].push_back(gr_group);
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
      GRGroup gr_group;
      for (LayerCoord& coord : gr_pin.getGridCoordList()) {
        gr_group.get_gr_node_list().push_back(&layer_node_map[coord.get_layer_idx()][coord.get_x()][coord.get_y()]);
      }
      gr_task.get_gr_group_list().push_back(gr_group);
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
  std::priority_queue<GRNode*, std::vector<GRNode*>, CmpGRNodeCost> empty_queue;
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
  std::vector<Segment<GRNode*>>& node_segment_list = gr_model.get_node_segment_list();

  std::vector<Segment<LayerCoord>> routing_segment_list;
  for (Segment<GRNode*>& node_segment : node_segment_list) {
    routing_segment_list.emplace_back(*node_segment.get_first(), *node_segment.get_second());
  }
  std::vector<LayerCoord> driving_grid_coord_list = gr_net.get_gr_driving_pin().getGridCoordList();
  std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC> key_coord_pin_map;
  for (GRPin& gr_pin : gr_net.get_gr_pin_list()) {
    for (LayerCoord& grid_coord : gr_pin.getGridCoordList()) {
      key_coord_pin_map[grid_coord].insert(gr_pin.get_pin_idx());
    }
  }
  gr_net.set_routing_tree(RTUtil::getTreeByFullFlow(driving_grid_coord_list, routing_segment_list, key_coord_pin_map));
}

void GlobalRouter::updateDemand(GRModel& gr_model, GRNet& gr_net, ChangeType change_type)
{
  std::vector<GridMap<GRNode>>& layer_node_map = gr_model.get_layer_node_map();

  std::set<GRNode*> key_node_set;
  for (GRPin& gr_pin : gr_net.get_gr_pin_list()) {
    for (LayerCoord& coord : gr_pin.getGridCoordList()) {
      GRNode* gr_node = &layer_node_map[coord.get_layer_idx()][coord.get_x()][coord.get_y()];
      key_node_set.insert(gr_node);
    }
  }
  std::vector<Segment<GRNode*>> node_segment_list;
  for (Segment<TNode<LayerCoord>*>& coord_segment : RTUtil::getSegListByTree(gr_net.get_routing_tree())) {
    LayerCoord first_coord = coord_segment.get_first()->value();
    LayerCoord second_coord = coord_segment.get_second()->value();

    GRNode* first_node = &layer_node_map[first_coord.get_layer_idx()][first_coord.get_x()][first_coord.get_y()];
    GRNode* second_node = &layer_node_map[second_coord.get_layer_idx()][second_coord.get_x()][second_coord.get_y()];

    node_segment_list.emplace_back(first_node, second_node);
  }

  std::map<GRNode*, std::set<Orientation>> usage_map;

  if (node_segment_list.empty()) {
    // 单层的local net
    if (key_node_set.size() > 1) {
      LOG_INST.error(Loc::current(), "The net is not local!");
    }
    GRNode* local_node = *key_node_set.begin();
    for (Orientation orientation : {Orientation::kUp, Orientation::kDown}) {
      usage_map[local_node].insert(orientation);
    }
  } else {
    // 跨gcell线网和多层的local_net
    for (Segment<GRNode*>& node_segment : node_segment_list) {
      GRNode* first_node = node_segment.get_first();
      GRNode* second_node = node_segment.get_second();
      Orientation orientation = RTUtil::getOrientation(*first_node, *second_node);
      if (orientation == Orientation::kNone || orientation == Orientation::kOblique) {
        LOG_INST.error(Loc::current(), "The orientation is error!");
      }
      Orientation oppo_orientation = RTUtil::getOppositeOrientation(orientation);

      GRNode* node_i = first_node;
      while (true) {
        if (node_i != first_node) {
          usage_map[node_i].insert(oppo_orientation);
        }
        if (node_i != second_node) {
          usage_map[node_i].insert(orientation);
        }
        if (node_i == second_node) {
          break;
        }
        node_i = node_i->getNeighborNode(orientation);
      }
    }
  }
  for (auto& [usage_node, orientation_list] : usage_map) {
    usage_node->updateDemand(gr_net.get_net_idx(), orientation_list, change_type);
  }
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
  std::priority_queue<GRNode*, std::vector<GRNode*>, CmpGRNodeCost>& open_queue = gr_model.get_open_queue();
  std::vector<GRNode*>& visited_node_list = gr_model.get_visited_node_list();

  open_queue.push(curr_node);
  curr_node->set_state(GRNodeState::kOpen);
  visited_node_list.push_back(curr_node);
}

GRNode* GlobalRouter::popFromOpenList(GRModel& gr_model)
{
  std::priority_queue<GRNode*, std::vector<GRNode*>, CmpGRNodeCost>& open_queue = gr_model.get_open_queue();

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
#if 0
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
  std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC> key_coord_pin_map;
  for (GRPin& gr_pin : gr_net.get_gr_pin_list()) {
    for (LayerCoord& grid_coord : gr_pin.getGridCoordList()) {
      key_coord_pin_map[grid_coord].insert(gr_pin.get_pin_idx());
    }
  }
  std::function<RTNode(LayerCoord&, std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC>&)> convert
      = std::bind(&GlobalRouter::convertToRTNode, this, std::placeholders::_1, std::placeholders::_2);
  gr_net.set_gr_result_tree(RTUtil::convertTree(gr_net.get_routing_tree(), convert, key_coord_pin_map));

  std::vector<TNode<RTNode>*> delete_node_list;
  std::map<TNode<RTNode>*, TNode<RTNode>*> origin_to_merged_map;
  std::queue<TNode<RTNode>*> node_queue = RTUtil::initQueue(gr_net.get_gr_result_tree().get_root());
  while (!node_queue.empty()) {
    TNode<RTNode>* curr_node = RTUtil::getFrontAndPop(node_queue);
    RTUtil::addListToQueue(node_queue, curr_node->get_child_list());

    std::vector<TNode<RTNode>*> box_node_list;
    std::vector<TNode<RTNode>*> bridge_node_list;
    for (TNode<RTNode>* child_node : curr_node->get_child_list()) {
      PlanarCoord& curr_grid_coord = curr_node->value().get_first_guide().get_grid_coord();
      PlanarCoord& child_grid_coord = child_node->value().get_first_guide().get_grid_coord();
      if (curr_grid_coord == child_grid_coord) {
        box_node_list.push_back(child_node);
      } else {
        bridge_node_list.push_back(child_node);
      }
    }
    TNode<RTNode>* merged_node = curr_node;
    if (RTUtil::exist(origin_to_merged_map, curr_node)) {
      merged_node = origin_to_merged_map[curr_node];
    }
    for (TNode<RTNode>* child_node : box_node_list) {
      buildDRNode(merged_node, child_node);
      origin_to_merged_map[child_node] = merged_node;
      delete_node_list.push_back(child_node);
    }
    for (TNode<RTNode>* child_node : bridge_node_list) {
      buildTANode(merged_node, child_node);
    }
  }
  for (TNode<RTNode>* delete_node : delete_node_list) {
    delete delete_node;
  }
}

RTNode GlobalRouter::convertToRTNode(LayerCoord& coord, std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC>& key_coord_pin_map)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();

  Guide guide(RTUtil::getRealRect(coord, gcell_axis), coord.get_layer_idx(), coord.get_planar_coord());

  RTNode rt_node;
  rt_node.set_first_guide(guide);
  rt_node.set_second_guide(guide);
  if (RTUtil::exist(key_coord_pin_map, coord)) {
    rt_node.set_pin_idx_set(key_coord_pin_map[coord]);
  }
  return rt_node;
}

void GlobalRouter::buildDRNode(TNode<RTNode>* parent_node, TNode<RTNode>* child_node)
{
  irt_int child_layer_idx = child_node->value().get_first_guide().get_layer_idx();
  Guide& first_guide = parent_node->value().get_first_guide();
  Guide& second_guide = parent_node->value().get_second_guide();
  first_guide.set_layer_idx(std::min(first_guide.get_layer_idx(), child_layer_idx));
  second_guide.set_layer_idx(std::max(second_guide.get_layer_idx(), child_layer_idx));

  std::set<irt_int>& parent_pin_idx_set = parent_node->value().get_pin_idx_set();
  std::set<irt_int>& child_pin_idx_set = child_node->value().get_pin_idx_set();
  parent_pin_idx_set.insert(child_pin_idx_set.begin(), child_pin_idx_set.end());

  parent_node->addChildren(child_node->get_child_list());
  parent_node->delChild(child_node);
}

void GlobalRouter::buildTANode(TNode<RTNode>* parent_node, TNode<RTNode>* child_node)
{
  Guide first_guide = parent_node->value().get_first_guide();
  Guide second_guide = child_node->value().get_first_guide();
  first_guide.set_layer_idx(second_guide.get_layer_idx());
  if (!CmpPlanarCoordByXASC()(first_guide.get_grid_coord(), second_guide.get_grid_coord())) {
    std::swap(first_guide, second_guide);
  }
  RTNode rt_node;
  rt_node.set_first_guide(first_guide);
  rt_node.set_second_guide(second_guide);

  TNode<RTNode>* bridge_node = new TNode<RTNode>(rt_node);
  parent_node->addChild(bridge_node);
  bridge_node->addChild(child_node);
  parent_node->delChild(child_node);
}

void GlobalRouter::countGRModel(GRModel& gr_model)
{
  irt_int micron_dbu = DM_INST.getDatabase().get_micron_dbu();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = DM_INST.getDatabase().get_layer_via_master_list();

  GRModelStat gr_model_stat;

  std::map<irt_int, double>& routing_wire_length_map = gr_model_stat.get_routing_wire_length_map();
  std::map<irt_int, irt_int>& cut_via_number_map = gr_model_stat.get_cut_via_number_map();
  std::vector<double>& resource_overflow_list = gr_model_stat.get_resource_overflow_list();
  std::vector<double>& access_overflow_list = gr_model_stat.get_access_overflow_list();

  for (GRNet& gr_net : gr_model.get_gr_net_list()) {
    for (TNode<RTNode>* node : RTUtil::getNodeList(gr_net.get_gr_result_tree())) {
      Guide& first_guide = node->value().get_first_guide();
      irt_int first_layer_idx = first_guide.get_layer_idx();
      Guide& second_guide = node->value().get_second_guide();
      irt_int second_layer_idx = second_guide.get_layer_idx();

      if (first_layer_idx == second_layer_idx) {
        double wire_length = RTUtil::getManhattanDistance(first_guide.getMidPoint(), second_guide.getMidPoint()) / 1.0 / micron_dbu;
        routing_wire_length_map[first_layer_idx] += wire_length;
      } else {
        RTUtil::swapASC(first_layer_idx, second_layer_idx);
        for (irt_int layer_idx = first_layer_idx; layer_idx < second_layer_idx; layer_idx++) {
          cut_via_number_map[layer_via_master_list[layer_idx].front().get_cut_layer_idx()]++;
        }
      }
    }
  }
  for (RoutingLayer& routing_layer : routing_layer_list) {
    GridMap<GRNode>& node_map = gr_model.get_layer_node_map()[routing_layer.get_layer_idx()];
    for (irt_int grid_x = 0; grid_x < node_map.get_x_size(); grid_x++) {
      for (irt_int grid_y = 0; grid_y < node_map.get_y_size(); grid_y++) {
        GRNode& gr_node = node_map[grid_x][grid_y];

        double resource_overflow = gr_node.calcCost(gr_node.get_resource_demand(), gr_node.get_resource_supply());
        resource_overflow_list.push_back(resource_overflow);

        if (routing_layer.isPreferH()) {
          for (Orientation orientation : {Orientation::kEast, Orientation::kWest}) {
            double access_overflow
                = gr_node.calcCost(gr_node.get_orien_access_demand_map()[orientation], gr_node.get_orien_access_supply_map()[orientation]);
            access_overflow_list.push_back(access_overflow);
          }
        } else {
          for (Orientation orientation : {Orientation::kSouth, Orientation::kNorth}) {
            double access_overflow
                = gr_node.calcCost(gr_node.get_orien_access_demand_map()[orientation], gr_node.get_orien_access_supply_map()[orientation]);
            access_overflow_list.push_back(access_overflow);
          }
        }
      }
    }
  }
  double total_wire_length = 0;
  irt_int total_via_number = 0;
  double max_resource_overflow = -DBL_MAX;
  double max_access_overflow = -DBL_MAX;
  for (auto& [routing_layer_idx, wire_length] : routing_wire_length_map) {
    total_wire_length += wire_length;
  }
  for (auto& [cut_layer_idx, via_number] : cut_via_number_map) {
    total_via_number += via_number;
  }
  for (double resource_overflow : resource_overflow_list) {
    max_resource_overflow = std::max(max_resource_overflow, resource_overflow);
  }
  for (double access_overflow : access_overflow_list) {
    max_access_overflow = std::max(max_access_overflow, access_overflow);
  }
  gr_model_stat.set_total_wire_length(total_wire_length);
  gr_model_stat.set_total_via_number(total_via_number);
  gr_model_stat.set_max_resource_overflow(max_resource_overflow);
  gr_model_stat.set_max_access_overflow(max_access_overflow);

  gr_model.set_gr_model_stat(gr_model_stat);
}

void GlobalRouter::reportGRModel(GRModel& gr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DM_INST.getDatabase().get_cut_layer_list();

  GRModelStat& gr_model_stat = gr_model.get_gr_model_stat();
  std::map<irt_int, double>& routing_wire_length_map = gr_model_stat.get_routing_wire_length_map();
  std::map<irt_int, irt_int>& cut_via_number_map = gr_model_stat.get_cut_via_number_map();
  std::vector<double>& resource_overflow_list = gr_model_stat.get_resource_overflow_list();
  std::vector<double>& access_overflow_list = gr_model_stat.get_access_overflow_list();
  double total_wire_length = gr_model_stat.get_total_wire_length();
  irt_int total_via_number = gr_model_stat.get_total_via_number();

  // report wire info
  fort::char_table wire_table;
  wire_table.set_border_style(FT_SOLID_STYLE);
  wire_table << fort::header << "Routing Layer"
             << "Wire Length / um" << fort::endr;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    double wire_length = routing_wire_length_map[routing_layer.get_layer_idx()];
    wire_table << routing_layer.get_layer_name()
               << RTUtil::getString(wire_length, "(", RTUtil::getPercentage(wire_length, total_wire_length), "%)") << fort::endr;
  }
  wire_table << fort::header << "Total" << total_wire_length << fort::endr;

  // report via info
  fort::char_table via_table;
  via_table.set_border_style(FT_SOLID_STYLE);
  via_table << fort::header << "Cut Layer"
            << "Via Number" << fort::endr;
  for (CutLayer& cut_layer : cut_layer_list) {
    irt_int via_number = cut_via_number_map[cut_layer.get_layer_idx()];
    via_table << cut_layer.get_layer_name() << RTUtil::getString(via_number, "(", RTUtil::getPercentage(via_number, total_via_number), "%)")
              << fort::endr;
  }
  via_table << fort::header << "Total" << total_via_number << fort::endr;

  // report resource overflow info
  GridMap<std::string> resource_overflow_map = RTUtil::getRangeRatioMap(resource_overflow_list, {1.0});

  fort::char_table resource_overflow_table;
  resource_overflow_table.set_border_style(FT_SOLID_STYLE);
  resource_overflow_table << fort::header << "Resource Overflow"
                          << "GCell Number" << fort::endr;
  for (irt_int y = 0; y < resource_overflow_map.get_y_size(); y++) {
    resource_overflow_table << resource_overflow_map[0][y] << resource_overflow_map[1][y] << fort::endr;
  }
  resource_overflow_table << fort::header << "Total" << resource_overflow_list.size() << fort::endr;

  // report access overflow info
  GridMap<std::string> access_overflow_map = RTUtil::getRangeRatioMap(access_overflow_list, {1.0});

  fort::char_table access_overflow_table;
  access_overflow_table.set_border_style(FT_SOLID_STYLE);
  access_overflow_table << fort::header << "Access Overflow"
                        << "GCell Number" << fort::endr;
  for (irt_int y = 0; y < access_overflow_map.get_y_size(); y++) {
    access_overflow_table << access_overflow_map[0][y] << access_overflow_map[1][y] << fort::endr;
  }
  access_overflow_table << fort::header << "Total" << access_overflow_list.size() << fort::endr;

  std::vector<std::vector<std::string>> table_list;
  table_list.push_back(RTUtil::splitString(wire_table.to_string(), '\n'));
  table_list.push_back(RTUtil::splitString(via_table.to_string(), '\n'));
  table_list.push_back(RTUtil::splitString(resource_overflow_table.to_string(), '\n'));
  table_list.push_back(RTUtil::splitString(access_overflow_table.to_string(), '\n'));

  int max_size = INT_MIN;
  for (std::vector<std::string>& table : table_list) {
    max_size = std::max(max_size, static_cast<irt_int>(table.size()));
  }
  for (std::vector<std::string>& table : table_list) {
    for (irt_int i = table.size(); i < max_size; i++) {
      std::string table_str;
      table_str.append(table.front().length() / 3, ' ');
      table.push_back(table_str);
    }
  }

  for (irt_int i = 0; i < max_size; i++) {
    std::string table_str;
    for (std::vector<std::string>& table : table_list) {
      table_str += table[i];
      table_str += " ";
    }
    LOG_INST.info(Loc::current(), table_str);
  }
}

bool GlobalRouter::stopGRModel(GRModel& gr_model)
{
  return (gr_model.get_gr_model_stat().get_max_resource_overflow() <= 1 && gr_model.get_gr_model_stat().get_max_access_overflow() <= 1);
}

#endif

#if 1  // update

void GlobalRouter::update(GRModel& gr_model)
{
  for (GRNet& gr_net : gr_model.get_gr_net_list()) {
    Net* origin_net = gr_net.get_origin_net();
    origin_net->set_gr_result_tree(gr_net.get_gr_result_tree());
  }
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

        double overflow = 0;

        double resource_overflow = gr_node.calcCost(gr_node.get_resource_demand(), gr_node.get_resource_supply());
        overflow = std::max(overflow, resource_overflow);

        if (routing_layer.isPreferH()) {
          for (Orientation orientation : {Orientation::kEast, Orientation::kWest}) {
            double access_overflow
                = gr_node.calcCost(gr_node.get_orien_access_demand_map()[orientation], gr_node.get_orien_access_supply_map()[orientation]);
            overflow = std::max(overflow, access_overflow);
          }
        } else {
          for (Orientation orientation : {Orientation::kSouth, Orientation::kNorth}) {
            double access_overflow
                = gr_node.calcCost(gr_node.get_orien_access_demand_map()[orientation], gr_node.get_orien_access_supply_map()[orientation]);
            overflow = std::max(overflow, access_overflow);
          }
        }
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
        PlanarRect real_rect = RTUtil::getRealRect(gr_node.get_planar_coord(), gcell_axis);
        irt_int y_reduced_span = real_rect.getYSpan() / 25;
        irt_int y = real_rect.get_rt_y();

        GPBoundary gp_boundary;
        switch (gr_node.get_state()) {
          case GRNodeState::kNone:
            gp_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kNone));
            break;
          case GRNodeState::kOpen:
            gp_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kOpen));
            break;
          case GRNodeState::kClose:
            gp_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kClose));
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
        gp_text_node_coord.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
        gp_text_node_coord.set_message(RTUtil::getString("(", grid_x, " , ", grid_y, " , ", gr_node.get_layer_idx(), ")"));
        gp_text_node_coord.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        gp_text_node_coord.set_presentation(GPTextPresentation::kLeftMiddle);
        gr_node_map_struct.push(gp_text_node_coord);

        y -= y_reduced_span;
        GPText gp_text_whole_wire_demand;
        gp_text_whole_wire_demand.set_coord(real_rect.get_lb_x(), y);
        gp_text_whole_wire_demand.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
        gp_text_whole_wire_demand.set_message(RTUtil::getString("whole_wire_demand: ", gr_node.get_whole_wire_demand()));
        gp_text_whole_wire_demand.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        gp_text_whole_wire_demand.set_presentation(GPTextPresentation::kLeftMiddle);
        gr_node_map_struct.push(gp_text_whole_wire_demand);

        y -= y_reduced_span;
        GPText gp_text_whole_via_demand;
        gp_text_whole_via_demand.set_coord(real_rect.get_lb_x(), y);
        gp_text_whole_via_demand.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
        gp_text_whole_via_demand.set_message(RTUtil::getString("whole_via_demand: ", gr_node.get_whole_via_demand()));
        gp_text_whole_via_demand.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        gp_text_whole_via_demand.set_presentation(GPTextPresentation::kLeftMiddle);
        gr_node_map_struct.push(gp_text_whole_via_demand);

        y -= y_reduced_span;
        GPText gp_text_net_orien_wire_demand_map;
        gp_text_net_orien_wire_demand_map.set_coord(real_rect.get_lb_x(), y);
        gp_text_net_orien_wire_demand_map.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
        gp_text_net_orien_wire_demand_map.set_message("net_orien_wire_demand_map: ");
        gp_text_net_orien_wire_demand_map.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        gp_text_net_orien_wire_demand_map.set_presentation(GPTextPresentation::kLeftMiddle);
        gr_node_map_struct.push(gp_text_net_orien_wire_demand_map);

        if (!gr_node.get_net_orien_wire_demand_map().empty()) {
          y -= y_reduced_span;
          GPText gp_text_net_orien_wire_demand_map_info;
          gp_text_net_orien_wire_demand_map_info.set_coord(real_rect.get_lb_x(), y);
          gp_text_net_orien_wire_demand_map_info.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
          std::string net_orien_wire_demand_map_message = "--";
          for (auto& [net_idx, orien_wire_demand_map] : gr_node.get_net_orien_wire_demand_map()) {
            net_orien_wire_demand_map_message += RTUtil::getString("(", net_idx, ")");
            for (auto& [orientation, wire_demand] : orien_wire_demand_map) {
              net_orien_wire_demand_map_message += RTUtil::getString("(", GetOrientationName()(orientation), ":", wire_demand, ")");
            }
          }
          gp_text_net_orien_wire_demand_map_info.set_message(net_orien_wire_demand_map_message);
          gp_text_net_orien_wire_demand_map_info.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
          gp_text_net_orien_wire_demand_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
          gr_node_map_struct.push(gp_text_net_orien_wire_demand_map_info);
        }

        y -= y_reduced_span;
        GPText gp_text_orien_access_supply_map;
        gp_text_orien_access_supply_map.set_coord(real_rect.get_lb_x(), y);
        gp_text_orien_access_supply_map.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
        gp_text_orien_access_supply_map.set_message("orien_access_supply_map: ");
        gp_text_orien_access_supply_map.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        gp_text_orien_access_supply_map.set_presentation(GPTextPresentation::kLeftMiddle);
        gr_node_map_struct.push(gp_text_orien_access_supply_map);

        if (!gr_node.get_orien_access_supply_map().empty()) {
          y -= y_reduced_span;
          GPText gp_text_orien_access_supply_map_info;
          gp_text_orien_access_supply_map_info.set_coord(real_rect.get_lb_x(), y);
          gp_text_orien_access_supply_map_info.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
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
        gp_text_orien_access_demand_map.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
        gp_text_orien_access_demand_map.set_message("orien_access_demand_map: ");
        gp_text_orien_access_demand_map.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        gp_text_orien_access_demand_map.set_presentation(GPTextPresentation::kLeftMiddle);
        gr_node_map_struct.push(gp_text_orien_access_demand_map);

        if (!gr_node.get_orien_access_demand_map().empty()) {
          y -= y_reduced_span;
          GPText gp_text_orien_access_demand_map_info;
          gp_text_orien_access_demand_map_info.set_coord(real_rect.get_lb_x(), y);
          gp_text_orien_access_demand_map_info.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
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
        gp_text_resource_supply.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
        gp_text_resource_supply.set_message(RTUtil::getString("resource_supply: ", gr_node.get_resource_supply()));
        gp_text_resource_supply.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        gp_text_resource_supply.set_presentation(GPTextPresentation::kLeftMiddle);
        gr_node_map_struct.push(gp_text_resource_supply);

        y -= y_reduced_span;
        GPText gp_text_resource_demand;
        gp_text_resource_demand.set_coord(real_rect.get_lb_x(), y);
        gp_text_resource_demand.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
        gp_text_resource_demand.set_message(RTUtil::getString("resource_demand: ", gr_node.get_resource_demand()));
        gp_text_resource_demand.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        gp_text_resource_demand.set_presentation(GPTextPresentation::kLeftMiddle);
        gr_node_map_struct.push(gp_text_resource_demand);

        y -= y_reduced_span;
        GPText gp_text_contribution_net_set;
        gp_text_contribution_net_set.set_coord(real_rect.get_lb_x(), y);
        gp_text_contribution_net_set.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
        gp_text_contribution_net_set.set_message("contribution_net_set: ");
        gp_text_contribution_net_set.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        gp_text_contribution_net_set.set_presentation(GPTextPresentation::kLeftMiddle);
        gr_node_map_struct.push(gp_text_contribution_net_set);

        if (!gr_node.get_contribution_net_set().empty()) {
          y -= y_reduced_span;
          GPText gp_text_contribution_net_set_info;
          gp_text_contribution_net_set_info.set_coord(real_rect.get_lb_x(), y);
          gp_text_contribution_net_set_info.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
          std::string contribution_net_set_info_message = "--";
          for (irt_int net_idx : gr_node.get_contribution_net_set()) {
            contribution_net_set_info_message += RTUtil::getString("(", net_idx, ")");
          }
          gp_text_contribution_net_set_info.set_message(contribution_net_set_info_message);
          gp_text_contribution_net_set_info.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
          gp_text_contribution_net_set_info.set_presentation(GPTextPresentation::kLeftMiddle);
          gr_node_map_struct.push(gp_text_contribution_net_set_info);
        }

        // y -= y_reduced_span;
        // GPText gp_text_direction_set;
        // gp_text_direction_set.set_coord(real_rect.get_lb_x(), y);
        // gp_text_direction_set.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
        // gp_text_direction_set.set_message("direction_set: ");
        // gp_text_direction_set.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        // gp_text_direction_set.set_presentation(GPTextPresentation::kLeftMiddle);
        // gr_node_map_struct.push(gp_text_direction_set);

        // if (!gr_node.get_direction_set().empty()) {
        //   y -= y_reduced_span;
        //   GPText gp_text_direction_set_info;
        //   gp_text_direction_set_info.set_coord(real_rect.get_lb_x(), y);
        //   gp_text_direction_set_info.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
        //   std::string direction_set_info_message = "--";
        //   for (Direction direction : gr_node.get_direction_set()) {
        //     direction_set_info_message += RTUtil::getString("(", GetDirectionName()(direction), ")");
        //   }
        //   gp_text_direction_set_info.set_message(direction_set_info_message);
        //   gp_text_direction_set_info.set_layer_idx(GP_INST.getGDSIdxByRouting(gr_node.get_layer_idx()));
        //   gp_text_direction_set_info.set_presentation(GPTextPresentation::kLeftMiddle);
        //   gr_node_map_struct.push(gp_text_direction_set_info);
        // }
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
        PlanarRect real_rect = RTUtil::getRealRect(gr_node.get_planar_coord(), gcell_axis);
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
          gp_path.set_data_type(static_cast<irt_int>(GPGraphType::kNeighbor));
          neighbor_map_struct.push(gp_path);
        }
      }
    }
  }
  gp_gds.addStruct(neighbor_map_struct);

  // source_region_query_map
  std::vector<std::pair<GRSourceType, GPGraphType>> source_graph_pair_list = {{GRSourceType::kBlockAndPin, GPGraphType::kBlockAndPin}};
  std::vector<GridMap<GRNode>>& layer_node_map = gr_model.get_layer_node_map();
  for (irt_int layer_idx = 0; layer_idx < static_cast<irt_int>(layer_node_map.size()); layer_idx++) {
    GridMap<GRNode>& node_map = layer_node_map[layer_idx];
    for (irt_int grid_x = 0; grid_x < node_map.get_x_size(); grid_x++) {
      for (irt_int grid_y = 0; grid_y < node_map.get_y_size(); grid_y++) {
        GRNode& gr_node = node_map[grid_x][grid_y];
        for (auto& [gr_source_type, gp_graph_type] : source_graph_pair_list) {
          for (auto& [net_idx, rect_set] : DC_INST.getLayerNetRectMap(gr_node.getRegionQuery(gr_source_type), true)[layer_idx]) {
            GPStruct net_rect_struct(RTUtil::getString(GetGRSourceTypeName()(gr_source_type), "@", net_idx));
            for (const LayerRect& rect : rect_set) {
              GPBoundary gp_boundary;
              gp_boundary.set_data_type(static_cast<irt_int>(gp_graph_type));
              gp_boundary.set_rect(rect);
              gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(rect.get_layer_idx()));
              net_rect_struct.push(gp_boundary);
            }
            gp_gds.addStruct(net_rect_struct);
          }
        }
      }
    }
  }

  // net
  for (GRNet& gr_net : gr_model.get_gr_net_list()) {
    GPStruct net_struct(RTUtil::getString("net_", gr_net.get_net_idx()));

    if (curr_net_idx == -1 || gr_net.get_net_idx() == curr_net_idx) {
      for (GRPin& gr_pin : gr_net.get_gr_pin_list()) {
        for (LayerCoord& coord : gr_pin.getGridCoordList()) {
          PlanarRect real_rect = RTUtil::getRealRect(coord.get_planar_coord(), gcell_axis);

          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kKey));
          gp_boundary.set_rect(real_rect);
          gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(coord.get_layer_idx()));
          net_struct.push(gp_boundary);
        }
      }
    }
    {
      // bounding_box
      GPBoundary gp_boundary;
      gp_boundary.set_layer_idx(0);
      gp_boundary.set_data_type(2);
      gp_boundary.set_rect(gr_net.get_bounding_box().get_real_rect());
      net_struct.push(gp_boundary);
    }
    for (Segment<TNode<LayerCoord>*>& segment : RTUtil::getSegListByTree(gr_net.get_routing_tree())) {
      LayerCoord first_coord = segment.get_first()->value();
      LayerCoord second_coord = segment.get_second()->value();
      irt_int first_layer_idx = first_coord.get_layer_idx();
      irt_int second_layer_idx = second_coord.get_layer_idx();

      if (first_layer_idx == second_layer_idx) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kPath));
        gp_boundary.set_rect(RTUtil::getRealRect(first_coord, second_coord, gcell_axis));
        gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(first_layer_idx));
        net_struct.push(gp_boundary);
      } else {
        RTUtil::swapASC(first_layer_idx, second_layer_idx);
        for (irt_int layer_idx = first_layer_idx; layer_idx <= second_layer_idx; layer_idx++) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kPath));
          gp_boundary.set_rect(RTUtil::getRealRect(first_coord, gcell_axis));
          gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(layer_idx));
          net_struct.push(gp_boundary);
        }
      }
    }
    gp_gds.addStruct(net_struct);
  }
  GP_INST.plot(gp_gds, gr_temp_directory_path + "gr_model.gds", false, false);
}

#endif

}  // namespace irt
