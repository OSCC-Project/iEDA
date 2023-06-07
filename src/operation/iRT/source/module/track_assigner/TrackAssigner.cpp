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
#include "TrackAssigner.hpp"

#include "GDSPlotter.hpp"
#include "LayerCoord.hpp"
#include "TAPanel.hpp"
#include "TASchedule.hpp"

namespace irt {

// public

void TrackAssigner::initInst(Config& config, Database& database)
{
  if (_ta_instance == nullptr) {
    _ta_instance = new TrackAssigner(config, database);
  }
}

TrackAssigner& TrackAssigner::getInst()
{
  if (_ta_instance == nullptr) {
    LOG_INST.error(Loc::current(), "The instance not initialized!");
  }
  return *_ta_instance;
}

void TrackAssigner::destroyInst()
{
  if (_ta_instance != nullptr) {
    delete _ta_instance;
    _ta_instance = nullptr;
  }
}

void TrackAssigner::assign(std::vector<Net>& net_list)
{
  Monitor monitor;

  std::vector<TANet> ta_net_list = _ta_data_manager.convertToTANetList(net_list);
  assignTANetList(ta_net_list);

  LOG_INST.info(Loc::current(), "The ", GetStageName()(Stage::kTrackAssigner), " completed!", monitor.getStatsInfo());
}

// private

TrackAssigner* TrackAssigner::_ta_instance = nullptr;

void TrackAssigner::init(Config& config, Database& database)
{
  _ta_data_manager.input(config, database);
}

void TrackAssigner::assignTANetList(std::vector<TANet>& ta_net_list)
{
  TAModel ta_model = initTAModel(ta_net_list);
  buildTAModel(ta_model);
  assignTAModel(ta_model);
  updateTAModel(ta_model);
  reportTAModel(ta_model);
}

#if 1  // build ta_model

TAModel TrackAssigner::initTAModel(std::vector<TANet>& ta_net_list)
{
  GCellAxis& gcell_axis = _ta_data_manager.getDatabase().get_gcell_axis();
  Die& die = _ta_data_manager.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();

  TAModel ta_model;
  ta_model.set_ta_net_list(ta_net_list);

  std::vector<std::vector<TAPanel>>& layer_panel_list = ta_model.get_layer_panel_list();
  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::vector<TAPanel> ta_panel_list;
    if (routing_layer.isPreferH()) {
      for (GCellGrid& gcell_grid : gcell_axis.get_y_grid_list()) {
        for (irt_int line = gcell_grid.get_start_line(); line < gcell_grid.get_end_line(); line += gcell_grid.get_step_length()) {
          TAPanel ta_panel;
          ta_panel.set_real_rect(PlanarRect(die.get_real_lb_x(), line, die.get_real_rt_x(), line + gcell_grid.get_step_length()));
          ta_panel.set_panel_idx(static_cast<irt_int>(ta_panel_list.size()));
          ta_panel.set_layer_idx(routing_layer.get_layer_idx());
          ta_panel_list.push_back(ta_panel);
        }
      }
    } else {
      for (GCellGrid& gcell_grid : gcell_axis.get_x_grid_list()) {
        for (irt_int line = gcell_grid.get_start_line(); line < gcell_grid.get_end_line(); line += gcell_grid.get_step_length()) {
          TAPanel ta_panel;
          ta_panel.set_real_rect(PlanarRect(line, die.get_real_lb_y(), line + gcell_grid.get_step_length(), die.get_real_rt_y()));
          ta_panel.set_panel_idx(static_cast<irt_int>(ta_panel_list.size()));
          ta_panel.set_layer_idx(routing_layer.get_layer_idx());
          ta_panel_list.push_back(ta_panel);
        }
      }
    }
    layer_panel_list.push_back(ta_panel_list);
  }
  return ta_model;
}

void TrackAssigner::buildTAModel(TAModel& ta_model)
{
  buildTATaskList(ta_model);
  buildPanelRegion(ta_model);
  updateNetBlockageMap(ta_model);
  updateNetFenceRegionMap(ta_model);
  buildTATaskPriority(ta_model);
}

void TrackAssigner::buildTATaskList(TAModel& ta_model)
{
  irt_int bottom_routing_layer_idx = _ta_data_manager.getConfig().bottom_routing_layer_idx;
  irt_int top_routing_layer_idx = _ta_data_manager.getConfig().top_routing_layer_idx;

  std::vector<std::vector<TAPanel>>& layer_panel_list = ta_model.get_layer_panel_list();

  for (TANet& ta_net : ta_model.get_ta_net_list()) {
    for (auto& [ta_node_node, ta_task] : makeTANodeTaskMap(ta_net)) {
      RTNode& ta_node = ta_node_node->value();
      irt_int panel_idx = -1;
      PlanarCoord& first_grid_coord = ta_node.get_first_guide().get_grid_coord();
      PlanarCoord& second_grid_coord = ta_node.get_second_guide().get_grid_coord();
      if (RTUtil::isHorizontal(first_grid_coord, second_grid_coord)) {
        panel_idx = first_grid_coord.get_y();
      } else {
        panel_idx = first_grid_coord.get_x();
      }
      irt_int layer_idx = ta_node.get_first_guide().get_layer_idx();
      if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
        LOG_INST.error(Loc::current(), "The gr result contains non-routable layers!");
      }
      TAPanel& ta_panel = layer_panel_list[layer_idx][panel_idx];

      std::vector<TATask>& ta_task_list = ta_panel.get_ta_task_list();
      ta_task.set_origin_net_idx(ta_net.get_net_idx());
      ta_task.set_origin_node(ta_node_node);
      ta_task.set_task_idx(static_cast<irt_int>(ta_task_list.size()));
      ta_task.set_connect_type(ta_net.get_connect_type());
      std::vector<PlanarCoord> coord_list;
      for (TAGroup& ta_group : ta_task.get_ta_group_list()) {
        for (LayerCoord& coord : ta_group.get_coord_list()) {
          coord_list.push_back(coord);
        }
      }
      ta_task.set_bounding_box(RTUtil::getBoundingBox(coord_list));
      ta_task_list.push_back(ta_task);
    }
  }
}

std::map<TNode<RTNode>*, TATask> TrackAssigner::makeTANodeTaskMap(TANet& ta_net)
{
  std::map<TNode<RTNode>*, TATask> ta_node_task_map;
  makeGroupAndCost(ta_net, ta_node_task_map);
  expandCoordCostMap(ta_node_task_map);
  return ta_node_task_map;
}

void TrackAssigner::makeGroupAndCost(TANet& ta_net, std::map<TNode<RTNode>*, TATask>& ta_node_task_map)
{
  // dr_ta_list_map
  std::map<TNode<RTNode>*, std::vector<TNode<RTNode>*>> dr_ta_list_map;
  for (Segment<TNode<RTNode>*>& segment : RTUtil::getSegListByTree(ta_net.get_ta_result_tree())) {
    TNode<RTNode>* dr_node_node = segment.get_first();
    TNode<RTNode>* ta_node_node = segment.get_second();
    if (dr_node_node->value().isTANode()) {
      std::swap(dr_node_node, ta_node_node);
    }
    dr_ta_list_map[dr_node_node].push_back(ta_node_node);
  }
  for (auto& [dr_node_node, ta_node_node_list] : dr_ta_list_map) {
    // pin_coord_list
    std::vector<LayerCoord> pin_coord_list;
    for (irt_int pin_idx : dr_node_node->value().get_pin_idx_set()) {
      pin_coord_list.push_back(ta_net.get_ta_pin_list()[pin_idx].getRealCoordList().front());
    }
    std::map<TNode<RTNode>*, TAGroup> ta_group_map;
    for (TNode<RTNode>* ta_node_node : ta_node_node_list) {
      ta_group_map[ta_node_node] = makeTAGroup(dr_node_node, ta_node_node, pin_coord_list);
    }
    std::map<TNode<RTNode>*, std::map<LayerCoord, double, CmpLayerCoordByXASC>> ta_cost_map_map;
    for (TNode<RTNode>* ta_node_node : ta_node_node_list) {
      ta_cost_map_map[ta_node_node] = makeTACostMap(ta_node_node, ta_group_map, pin_coord_list);
    }
    for (TNode<RTNode>* ta_node_node : ta_node_node_list) {
      TATask& ta_task = ta_node_task_map[ta_node_node];
      ta_task.get_ta_group_list().push_back(ta_group_map[ta_node_node]);
      ta_task.get_coord_cost_map().insert(ta_cost_map_map[ta_node_node].begin(), ta_cost_map_map[ta_node_node].end());
    }
  }
}

TAGroup TrackAssigner::makeTAGroup(TNode<RTNode>* dr_node_node, TNode<RTNode>* ta_node_node, std::vector<LayerCoord>& pin_coord_list)
{
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();

  // dr info
  Guide& dr_guide = dr_node_node->value().get_first_guide();
  PlanarCoord& dr_grid_coord = dr_guide.get_grid_coord();

  // ta info
  PlanarCoord& first_grid_coord = ta_node_node->value().get_first_guide().get_grid_coord();
  PlanarCoord& second_grid_coord = ta_node_node->value().get_second_guide().get_grid_coord();
  irt_int ta_layer_idx = ta_node_node->value().get_first_guide().get_layer_idx();

  // layer
  RoutingLayer& routing_layer = routing_layer_list[ta_layer_idx];
  TrackGrid& x_track_grid = routing_layer.getXTrackGrid();
  TrackGrid& y_track_grid = routing_layer.getYTrackGrid();

  Orientation orientation = Orientation::kNone;
  if (dr_grid_coord == first_grid_coord) {
    orientation = RTUtil::getOrientation(dr_grid_coord, second_grid_coord);
  } else {
    orientation = RTUtil::getOrientation(dr_grid_coord, first_grid_coord);
  }
  PlanarRect routing_region = dr_guide;
  if (!pin_coord_list.empty()) {
    routing_region = RTUtil::getBoundingBox(pin_coord_list);
    if (!RTUtil::existGrid(routing_region, routing_layer.get_track_axis())) {
      routing_region = getTrackLineRect(routing_region, routing_layer.get_track_axis());
    }
    routing_region = RTUtil::getEnlargedRect(routing_region, 0, dr_guide);
  }
  std::vector<irt_int> x_list = RTUtil::getClosedScaleList(dr_guide.get_lb_x(), dr_guide.get_rt_x(), x_track_grid);
  std::vector<irt_int> y_list = RTUtil::getClosedScaleList(dr_guide.get_lb_y(), dr_guide.get_rt_y(), y_track_grid);
  if (orientation == Orientation::kEast || orientation == Orientation::kWest) {
    x_list = RTUtil::getClosedScaleList(routing_region.get_lb_x(), routing_region.get_rt_x(), x_track_grid);
    irt_int x = (orientation == Orientation::kEast ? x_list.back() : x_list.front());
    x_list.clear();
    x_list.push_back(x);
  } else if (orientation == Orientation::kNorth || orientation == Orientation::kSouth) {
    y_list = RTUtil::getClosedScaleList(routing_region.get_lb_y(), routing_region.get_rt_y(), y_track_grid);
    irt_int y = (orientation == Orientation::kNorth ? y_list.back() : y_list.front());
    y_list.clear();
    y_list.push_back(y);
  }
  TAGroup ta_group;
  std::vector<LayerCoord>& coord_list = ta_group.get_coord_list();
  for (irt_int x : x_list) {
    for (irt_int y : y_list) {
      coord_list.emplace_back(x, y, ta_layer_idx);
    }
  }
  return ta_group;
}

// 先将矩形按照x/y track pitch膨胀，膨胀后的矩形边界收缩到最近的track line上
PlanarRect TrackAssigner::getTrackLineRect(PlanarRect& rect, TrackAxis& track_axis)
{
  irt_int real_lb_x = rect.get_lb_x();
  irt_int real_rt_x = rect.get_rt_x();
  irt_int real_lb_y = rect.get_lb_y();
  irt_int real_rt_y = rect.get_rt_y();
  if (RTUtil::getClosedScaleList(real_lb_x, real_rt_x, track_axis.get_x_track_grid()).empty()) {
    real_lb_x = RTUtil::getFloorTrackLine(real_lb_x, track_axis.get_x_track_grid());
    real_rt_x = RTUtil::getCeilTrackLine(real_rt_x, track_axis.get_x_track_grid());
  }
  if (RTUtil::getClosedScaleList(real_lb_y, real_rt_y, track_axis.get_y_track_grid()).empty()) {
    real_lb_y = RTUtil::getFloorTrackLine(real_lb_y, track_axis.get_y_track_grid());
    real_rt_y = RTUtil::getCeilTrackLine(real_rt_y, track_axis.get_y_track_grid());
  }
  return PlanarRect(real_lb_x, real_lb_y, real_rt_x, real_rt_y);
}

std::map<LayerCoord, double, CmpLayerCoordByXASC> TrackAssigner::makeTACostMap(TNode<RTNode>* ta_node_node,
                                                                               std::map<TNode<RTNode>*, TAGroup>& ta_group_map,
                                                                               std::vector<LayerCoord>& pin_coord_list)
{
  std::map<LayerCoord, double, CmpLayerCoordByXASC> coord_distance_map;
  if (!pin_coord_list.empty()) {
    for (LayerCoord& coord : ta_group_map[ta_node_node].get_coord_list()) {
      for (LayerCoord& pin_coord : pin_coord_list) {
        coord_distance_map[coord] += RTUtil::getManhattanDistance(coord, pin_coord);
      }
    }
  } else {
    for (LayerCoord& coord : ta_group_map[ta_node_node].get_coord_list()) {
      for (auto& [other_ta_node_node, group] : ta_group_map) {
        if (other_ta_node_node == ta_node_node) {
          continue;
        }
        for (LayerCoord& group_coord : group.get_coord_list()) {
          coord_distance_map[coord] += RTUtil::getManhattanDistance(coord, group_coord);
        }
      }
    }
  }
  std::vector<std::pair<LayerCoord, double>> coord_distance_pair_list;
  for (auto& [coord, distance] : coord_distance_map) {
    coord_distance_pair_list.emplace_back(coord, distance);
  }
  std::sort(coord_distance_pair_list.begin(), coord_distance_pair_list.end(),
            [](std::pair<LayerCoord, double>& a, std::pair<LayerCoord, double>& b) { return a.second < b.second; });

  std::map<LayerCoord, double, CmpLayerCoordByXASC> coord_cost_map;
  for (size_t i = 0; i < coord_distance_pair_list.size(); i++) {
    coord_cost_map[coord_distance_pair_list[i].first] = static_cast<double>(i);
  }
  return coord_cost_map;
}

void TrackAssigner::expandCoordCostMap(std::map<TNode<RTNode>*, TATask>& ta_node_task_map)
{
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();

  // 扩充cost_map
  for (auto& [ta_node_node, ta_task] : ta_node_task_map) {
    RTNode& ta_node = ta_node_node->value();
    RoutingLayer& routing_layer = routing_layer_list[ta_node.get_first_guide().get_layer_idx()];

    std::map<LayerCoord, double, CmpLayerCoordByXASC> new_coosd_cost_map;
    if (RTUtil::isHorizontal(ta_node.get_first_guide().get_grid_coord(), ta_node.get_second_guide().get_grid_coord())) {
      irt_int min_x = INT_MAX;
      irt_int max_x = INT_MIN;
      for (TAGroup& ta_group : ta_task.get_ta_group_list()) {
        for (LayerCoord& coord : ta_group.get_coord_list()) {
          min_x = std::min(min_x, coord.get_x());
          max_x = std::max(max_x, coord.get_x());
        }
      }
      std::vector<irt_int> x_list = RTUtil::getClosedScaleList(min_x, max_x, routing_layer.getXTrackGrid());
      for (auto& [coord, cost] : ta_task.get_coord_cost_map()) {
        for (irt_int x : x_list) {
          LayerCoord new_coord(x, coord.get_y(), coord.get_layer_idx());
          if (RTUtil::exist(new_coosd_cost_map, new_coord)) {
            new_coosd_cost_map[new_coord] += cost;
          } else {
            new_coosd_cost_map[new_coord] = cost;
          }
        }
      }
    } else {
      irt_int min_y = INT_MAX;
      irt_int max_y = INT_MIN;
      for (TAGroup& ta_group : ta_task.get_ta_group_list()) {
        for (LayerCoord& coord : ta_group.get_coord_list()) {
          min_y = std::min(min_y, coord.get_y());
          max_y = std::max(max_y, coord.get_y());
        }
      }
      std::vector<irt_int> y_list = RTUtil::getClosedScaleList(min_y, max_y, routing_layer.getYTrackGrid());
      for (auto& [coord, cost] : ta_task.get_coord_cost_map()) {
        for (irt_int y : y_list) {
          LayerCoord new_coord(coord.get_x(), y, coord.get_layer_idx());
          if (RTUtil::exist(new_coosd_cost_map, new_coord)) {
            new_coosd_cost_map[new_coord] += cost;
          } else {
            new_coosd_cost_map[new_coord] = cost;
          }
        }
      }
    }
    ta_task.set_coord_cost_map(new_coosd_cost_map);
  }
}

void TrackAssigner::buildPanelRegion(TAModel& ta_model)
{
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();

  for (std::vector<TAPanel>& ta_panel_list : ta_model.get_layer_panel_list()) {
    for (TAPanel& ta_panel : ta_panel_list) {
      TrackAxis& track_axis = routing_layer_list[ta_panel.get_layer_idx()].get_track_axis();

      std::vector<PlanarCoord> coord_list;
      for (TATask& ta_task : ta_panel.get_ta_task_list()) {
        for (TAGroup& ta_group : ta_task.get_ta_group_list()) {
          coord_list.insert(coord_list.end(), ta_group.get_coord_list().begin(), ta_group.get_coord_list().end());
        }
      }
      if (coord_list.empty()) {
        continue;
      }
      PlanarRect panel_region = RTUtil::getBoundingBox(coord_list);
      if (routing_layer_list[ta_panel.get_layer_idx()].isPreferH()) {
        ta_panel.set_real_lb_x(panel_region.get_lb_x());
        ta_panel.set_real_rt_x(panel_region.get_rt_x());
      } else {
        ta_panel.set_real_lb_y(panel_region.get_lb_y());
        ta_panel.set_real_rt_y(panel_region.get_rt_y());
      }
      if (!RTUtil::existGrid(ta_panel.get_real_rect(), track_axis)) {
        LOG_INST.error(Loc::current(), "The panel not contain any grid!");
      }
      ta_panel.set_grid_rect(RTUtil::getGridRect(ta_panel.get_real_rect(), track_axis));
    }
  }
}

void TrackAssigner::updateNetBlockageMap(TAModel& ta_model)
{
  GCellAxis& gcell_axis = _ta_data_manager.getDatabase().get_gcell_axis();
  EXTPlanarRect& die = _ta_data_manager.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();
  std::vector<Blockage>& routing_blockage_list = _ta_data_manager.getDatabase().get_routing_blockage_list();

  std::vector<std::vector<TAPanel>>& layer_panel_list = ta_model.get_layer_panel_list();

  for (const Blockage& routing_blockage : routing_blockage_list) {
    irt_int layer_idx = routing_blockage.get_layer_idx();
    irt_int min_spacing = routing_layer_list[layer_idx].getMinSpacing(routing_blockage.get_real_rect());
    PlanarRect enlarged_real_rect = RTUtil::getEnlargedRect(routing_blockage.get_real_rect(), min_spacing, die.get_real_rect());
    PlanarRect enlarged_grid_rect = RTUtil::getClosedGridRect(enlarged_real_rect, gcell_axis);
    if (routing_layer_list[layer_idx].isPreferH()) {
      for (irt_int y = enlarged_grid_rect.get_lb_y(); y <= enlarged_grid_rect.get_rt_y(); y++) {
        TAPanel& ta_panel = layer_panel_list[layer_idx][y];
        if (!RTUtil::isClosedOverlap(ta_panel.get_real_rect(), enlarged_real_rect)) {
          continue;
        }
        ta_panel.get_net_blockage_map()[-1].push_back(enlarged_real_rect);
      }
    } else {
      for (irt_int x = enlarged_grid_rect.get_lb_x(); x <= enlarged_grid_rect.get_rt_x(); x++) {
        TAPanel& ta_panel = layer_panel_list[layer_idx][x];
        if (!RTUtil::isClosedOverlap(ta_panel.get_real_rect(), enlarged_real_rect)) {
          continue;
        }
        ta_panel.get_net_blockage_map()[-1].push_back(enlarged_real_rect);
      }
    }
  }
  for (TANet& ta_net : ta_model.get_ta_net_list()) {
    for (TAPin& ta_pin : ta_net.get_ta_pin_list()) {
      for (const EXTLayerRect& routing_shape : ta_pin.get_routing_shape_list()) {
        irt_int layer_idx = routing_shape.get_layer_idx();
        irt_int min_spacing = routing_layer_list[layer_idx].getMinSpacing(routing_shape.get_real_rect());
        PlanarRect enlarged_real_rect = RTUtil::getEnlargedRect(routing_shape.get_real_rect(), min_spacing, die.get_real_rect());
        PlanarRect enlarged_grid_rect = RTUtil::getClosedGridRect(enlarged_real_rect, gcell_axis);
        if (routing_layer_list[layer_idx].isPreferH()) {
          for (irt_int y = enlarged_grid_rect.get_lb_y(); y <= enlarged_grid_rect.get_rt_y(); y++) {
            TAPanel& ta_panel = layer_panel_list[layer_idx][y];
            if (!RTUtil::isClosedOverlap(ta_panel.get_real_rect(), enlarged_real_rect)) {
              continue;
            }
            ta_panel.get_net_blockage_map()[ta_net.get_net_idx()].push_back(enlarged_real_rect);
          }
        } else {
          for (irt_int x = enlarged_grid_rect.get_lb_x(); x <= enlarged_grid_rect.get_rt_x(); x++) {
            TAPanel& ta_panel = layer_panel_list[layer_idx][x];
            if (!RTUtil::isClosedOverlap(ta_panel.get_real_rect(), enlarged_real_rect)) {
              continue;
            }
            ta_panel.get_net_blockage_map()[ta_net.get_net_idx()].push_back(enlarged_real_rect);
          }
        }
      }
    }
  }
}

void TrackAssigner::updateNetFenceRegionMap(TAModel& ta_model)
{
  GCellAxis& gcell_axis = _ta_data_manager.getDatabase().get_gcell_axis();
  EXTPlanarRect& die = _ta_data_manager.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _ta_data_manager.getDatabase().get_layer_via_master_list();
  irt_int bottom_routing_layer_idx = _ta_data_manager.getConfig().bottom_routing_layer_idx;
  irt_int top_routing_layer_idx = _ta_data_manager.getConfig().top_routing_layer_idx;

  std::vector<std::vector<TAPanel>>& layer_panel_list = ta_model.get_layer_panel_list();

  for (TANet& ta_net : ta_model.get_ta_net_list()) {
    std::vector<LayerCoord> real_coord_list;
    for (TAPin& ta_pin : ta_net.get_ta_pin_list()) {
      real_coord_list.push_back(ta_pin.getRealCoordList().front());
    }
    std::vector<LayerRect> net_fence_region_list;
    for (LayerCoord& real_coord : real_coord_list) {
      irt_int layer_idx = real_coord.get_layer_idx();
      for (irt_int via_below_layer_idx : RTUtil::getViaBelowLayerIdxList(layer_idx, bottom_routing_layer_idx, top_routing_layer_idx)) {
        ViaMaster& via_master = layer_via_master_list[via_below_layer_idx].front();
        for (const LayerRect& enclosure : {via_master.get_below_enclosure(), via_master.get_above_enclosure()}) {
          LayerRect via_shape(RTUtil::getOffsetRect(enclosure, real_coord), enclosure.get_layer_idx());
          net_fence_region_list.push_back(via_shape);
        }
      }
    }
    for (const LayerRect& net_fence_region : net_fence_region_list) {
      irt_int layer_idx = net_fence_region.get_layer_idx();
      irt_int min_spacing = routing_layer_list[layer_idx].getMinSpacing(net_fence_region);
      PlanarRect enlarged_real_rect = RTUtil::getEnlargedRect(net_fence_region, min_spacing, die.get_real_rect());
      PlanarRect enlarged_grid_rect = RTUtil::getClosedGridRect(enlarged_real_rect, gcell_axis);
      if (routing_layer_list[layer_idx].isPreferH()) {
        for (irt_int y = enlarged_grid_rect.get_lb_y(); y <= enlarged_grid_rect.get_rt_y(); y++) {
          TAPanel& ta_panel = layer_panel_list[layer_idx][y];
          if (!RTUtil::isClosedOverlap(ta_panel.get_real_rect(), enlarged_real_rect)) {
            continue;
          }
          ta_panel.get_net_fence_region_map()[ta_net.get_net_idx()].push_back(enlarged_real_rect);
        }
      } else {
        for (irt_int x = enlarged_grid_rect.get_lb_x(); x <= enlarged_grid_rect.get_rt_x(); x++) {
          TAPanel& ta_panel = layer_panel_list[layer_idx][x];
          if (!RTUtil::isClosedOverlap(ta_panel.get_real_rect(), enlarged_real_rect)) {
            continue;
          }
          ta_panel.get_net_fence_region_map()[ta_net.get_net_idx()].push_back(enlarged_real_rect);
        }
      }
    }
  }
}

void TrackAssigner::buildTATaskPriority(TAModel& ta_model)
{
  for (std::vector<TAPanel>& ta_panel_list : ta_model.get_layer_panel_list()) {
    for (TAPanel& ta_panel : ta_panel_list) {
      for (TATask& ta_task : ta_panel.get_ta_task_list()) {
        TATaskPriority& ta_task_priority = ta_task.get_ta_task_priority();
        // connect_type
        ta_task_priority.set_connect_type(ta_task.get_connect_type());
        // length_width_ratio
        PlanarRect& bounding_box = ta_task.get_bounding_box();
        irt_int width = std::max(1, bounding_box.getWidth());
        ta_task_priority.set_length_width_ratio(bounding_box.getLength() / 1.0 / width);
      }
    }
  }
}

#endif

#if 1  // assign ta_model

void TrackAssigner::assignTAModel(TAModel& ta_model)
{
  Monitor monitor;

  std::vector<std::vector<TAPanel>>& layer_panel_list = ta_model.get_layer_panel_list();

  irt_int range = 2;

  std::vector<std::vector<TASchedule>> ta_schedule_comb_list;
  for (irt_int layer_idx = 0; layer_idx < static_cast<irt_int>(layer_panel_list.size()); layer_idx++) {
    for (irt_int start_i = 0; start_i < range; start_i++) {
      std::vector<TASchedule> ta_schedule_list;
      for (irt_int i = start_i; i < static_cast<irt_int>(layer_panel_list[layer_idx].size()); i += range) {
        ta_schedule_list.emplace_back(layer_idx, i);
      }
      ta_schedule_comb_list.push_back(ta_schedule_list);
    }
  }

  size_t total_panel_num = 0;
  for (std::vector<TASchedule>& ta_schedule_list : ta_schedule_comb_list) {
    Monitor stage_monitor;
#pragma omp parallel for
    for (TASchedule& ta_schedule : ta_schedule_list) {
      TAPanel& ta_panel = layer_panel_list[ta_schedule.get_layer_idx()][ta_schedule.get_panel_idx()];
      if (ta_panel.skipAssigning()) {
        continue;
      }
      buildTAPanel(ta_panel);
      checkTAPanel(ta_panel);
      sortTAPanel(ta_panel);
      assignTAPanel(ta_panel);
      updateTAPanel(ta_model, ta_panel);
      ta_panel.freeNodeMap();
    }
    total_panel_num += ta_schedule_list.size();
    LOG_INST.info(Loc::current(), "Processed ", ta_schedule_list.size(), " panels", stage_monitor.getStatsInfo());
  }
  LOG_INST.info(Loc::current(), "Processed ", total_panel_num, " panels", monitor.getStatsInfo());
}

#endif

#if 1  // build ta_panel

void TrackAssigner::buildTAPanel(TAPanel& ta_panel)
{
  initTANodeMap(ta_panel);
  buildNeighborMap(ta_panel);
  buildOBSTaskMap(ta_panel);
  buildFenceTaskMap(ta_panel);
}

void TrackAssigner::initTANodeMap(TAPanel& ta_panel)
{
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();

  irt_int layer_idx = ta_panel.get_layer_idx();
  TrackGrid& x_track_grid = routing_layer_list[layer_idx].getXTrackGrid();
  TrackGrid& y_track_grid = routing_layer_list[layer_idx].getYTrackGrid();

  GridMap<TANode>& ta_node_map = ta_panel.get_ta_node_map();
  ta_node_map.init(ta_panel.getXSize(), ta_panel.getYSize());

  PlanarCoord& real_lb = ta_panel.get_real_lb();
  PlanarCoord& real_rt = ta_panel.get_real_rt();
  std::vector<irt_int> x_list = RTUtil::getClosedScaleList(real_lb.get_x(), real_rt.get_x(), x_track_grid);
  std::vector<irt_int> y_list = RTUtil::getClosedScaleList(real_lb.get_y(), real_rt.get_y(), y_track_grid);
  if (static_cast<irt_int>(x_list.size()) != ta_node_map.get_x_size() || static_cast<irt_int>(y_list.size()) != ta_node_map.get_y_size()) {
    LOG_INST.error(Loc::current(), "The size of scale list is different with size of node map!");
  }
  for (irt_int x = 0; x < ta_node_map.get_x_size(); x++) {
    for (irt_int y = 0; y < ta_node_map.get_y_size(); y++) {
      TANode& ta_node = ta_node_map[x][y];
      ta_node.set_x(x_list[x]);
      ta_node.set_y(y_list[y]);
      ta_node.set_layer_idx(layer_idx);
      ta_node.set_fence_violation_cost(ta_panel.getRealWidth());
    }
  }
}

void TrackAssigner::buildNeighborMap(TAPanel& ta_panel)
{
  GridMap<TANode>& ta_node_map = ta_panel.get_ta_node_map();
  for (irt_int x = 0; x < ta_node_map.get_x_size(); x++) {
    for (irt_int y = 0; y < ta_node_map.get_y_size(); y++) {
      std::map<Orientation, TANode*>& neighbor_ptr_map = ta_node_map[x][y].get_neighbor_ptr_map();
      if (x != 0) {
        neighbor_ptr_map[Orientation::kWest] = &ta_node_map[x - 1][y];
      }
      if (x != (ta_node_map.get_x_size() - 1)) {
        neighbor_ptr_map[Orientation::kEast] = &ta_node_map[x + 1][y];
      }
      if (y != 0) {
        neighbor_ptr_map[Orientation::kSouth] = &ta_node_map[x][y - 1];
      }
      if (y != (ta_node_map.get_y_size() - 1)) {
        neighbor_ptr_map[Orientation::kNorth] = &ta_node_map[x][y + 1];
      }
    }
  }
}

void TrackAssigner::buildOBSTaskMap(TAPanel& ta_panel)
{
  GridMap<TANode>& ta_node_map = ta_panel.get_ta_node_map();

  std::map<irt_int, std::vector<irt_int>> net_task_map;
  for (TATask& ta_task : ta_panel.get_ta_task_list()) {
    net_task_map[ta_task.get_origin_net_idx()].push_back(ta_task.get_task_idx());
  }
  for (auto& [net_idx, blockage_list] : ta_panel.get_net_blockage_map()) {
    std::vector<irt_int>& task_idx_list = net_task_map[net_idx];
    for (PlanarRect& blockage : blockage_list) {
      for (auto& [grid_coord, orientation_set] : getGridOrientationMap(ta_panel, blockage)) {
        irt_int local_x = grid_coord.get_x() - ta_panel.get_grid_lb_x();
        irt_int local_y = grid_coord.get_y() - ta_panel.get_grid_lb_y();
        if (!ta_node_map.isInside(local_x, local_y)) {
          continue;
        }
        TANode& ta_node = ta_node_map[local_x][local_y];
        for (Orientation orientation : orientation_set) {
          if (task_idx_list.empty()) {
            ta_node.get_obs_task_map()[orientation].insert(-1);
          } else {
            ta_node.get_obs_task_map()[orientation].insert(task_idx_list.begin(), task_idx_list.end());
          }
        }
      }
    }
  }
}

std::map<PlanarCoord, std::set<Orientation>, CmpPlanarCoordByXASC> TrackAssigner::getGridOrientationMap(TAPanel& ta_panel,
                                                                                                        PlanarRect& enlarge_real_rect)
{
  // enlarge_real_rect为已经扩了spacing的矩形
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();
  RoutingLayer& routing_layer = routing_layer_list[ta_panel.get_layer_idx()];
  TrackAxis& track_axis = routing_layer.get_track_axis();

  std::map<PlanarCoord, std::set<Orientation>, CmpPlanarCoordByXASC> grid_orientation_map;
  for (Segment<LayerCoord>& real_segment : getRealSegmentList(ta_panel, enlarge_real_rect)) {
    LayerCoord& first_coord = real_segment.get_first();
    LayerCoord& second_coord = real_segment.get_second();

    if (RTUtil::isOpenOverlap(enlarge_real_rect, getRealRectList({real_segment}).front())) {
      if (!RTUtil::existGrid(first_coord, track_axis) || !RTUtil::existGrid(second_coord, track_axis)) {
        LOG_INST.error(Loc::current(), "The coord can not find grid!");
      }
      Orientation orientation = RTUtil::getOrientation(first_coord, second_coord);
      grid_orientation_map[RTUtil::getGridCoord(first_coord, track_axis)].insert(orientation);
      grid_orientation_map[RTUtil::getGridCoord(second_coord, track_axis)].insert(RTUtil::getOppositeOrientation(orientation));
    }
  }
  return grid_orientation_map;
}

std::vector<Segment<LayerCoord>> TrackAssigner::getRealSegmentList(TAPanel& ta_panel, PlanarRect& enlarge_real_rect)
{
  // enlarge_real_rect为已经扩了spacing的矩形
  // 获取enlarge_real_rect覆盖的线段
  std::vector<Segment<LayerCoord>> real_segment_list;

  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();
  RoutingLayer& routing_layer = routing_layer_list[ta_panel.get_layer_idx()];
  TrackAxis& track_axis = routing_layer.get_track_axis();

  // ta只需要膨胀half_width
  PlanarRect search_rect = RTUtil::getEnlargedRect(enlarge_real_rect, routing_layer.get_min_width() / 2);
  irt_int x_step_length = track_axis.get_x_track_grid().get_step_length();
  irt_int y_step_length = track_axis.get_y_track_grid().get_step_length();
  search_rect = RTUtil::getEnlargedRect(search_rect, x_step_length, y_step_length, x_step_length, y_step_length);

  std::vector<irt_int> x_list = RTUtil::getClosedScaleList(search_rect.get_lb_x(), search_rect.get_rt_x(), track_axis.get_x_track_grid());
  std::vector<irt_int> y_list = RTUtil::getClosedScaleList(search_rect.get_lb_y(), search_rect.get_rt_y(), track_axis.get_y_track_grid());
  for (size_t y_idx = 0; y_idx < y_list.size(); y_idx++) {
    irt_int y = y_list[y_idx];
    if (y == y_list.front() || y == y_list.back()) {
      continue;
    }
    for (irt_int x_idx = 0; x_idx < static_cast<irt_int>(x_list.size()) - 1; x_idx++) {
      real_segment_list.emplace_back(LayerCoord(x_list[x_idx], y, ta_panel.get_layer_idx()),
                                     LayerCoord(x_list[x_idx + 1], y, ta_panel.get_layer_idx()));
    }
  }
  for (size_t x_idx = 0; x_idx < x_list.size(); x_idx++) {
    irt_int x = x_list[x_idx];
    if (x == x_list.front() || x == x_list.back()) {
      continue;
    }
    for (irt_int y_idx = 0; y_idx < static_cast<irt_int>(y_list.size()) - 1; y_idx++) {
      real_segment_list.emplace_back(LayerCoord(x, y_list[y_idx], ta_panel.get_layer_idx()),
                                     LayerCoord(x, y_list[y_idx + 1], ta_panel.get_layer_idx()));
    }
  }
  return real_segment_list;
}

std::vector<LayerRect> TrackAssigner::getRealRectList(std::vector<Segment<LayerCoord>> segment_list)
{
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();

  std::map<irt_int, std::vector<PlanarRect>> layer_rect_map;
  for (Segment<LayerCoord>& segment : segment_list) {
    LayerCoord& first_coord = segment.get_first();
    LayerCoord& second_coord = segment.get_second();

    if (first_coord.get_layer_idx() == second_coord.get_layer_idx()) {
      irt_int half_width = routing_layer_list[first_coord.get_layer_idx()].get_min_width() / 2;
      PlanarRect wire_rect = RTUtil::getEnlargedRect(first_coord, second_coord, half_width);
      layer_rect_map[first_coord.get_layer_idx()].push_back(wire_rect);
    } else {
      LOG_INST.error(Loc::current(), "The segment is proximal!");
    }
  }
  std::vector<LayerRect> real_rect_list;
  for (auto& [layer_idx, rect_list] : layer_rect_map) {
    rect_list = RTUtil::getMergeRectList(rect_list);
    for (PlanarRect& rect : rect_list) {
      real_rect_list.emplace_back(rect, layer_idx);
    }
  }
  return real_rect_list;
}

void TrackAssigner::buildFenceTaskMap(TAPanel& ta_panel)
{
  GridMap<TANode>& ta_node_map = ta_panel.get_ta_node_map();

  std::map<irt_int, std::vector<irt_int>> net_task_map;
  for (TATask& ta_task : ta_panel.get_ta_task_list()) {
    net_task_map[ta_task.get_origin_net_idx()].push_back(ta_task.get_task_idx());
  }
  for (auto& [net_idx, region_list] : ta_panel.get_net_fence_region_map()) {
    std::vector<irt_int>& task_idx_list = net_task_map[net_idx];
    for (PlanarRect& region : region_list) {
      for (auto& [grid_coord, orientation_set] : getGridOrientationMap(ta_panel, region)) {
        irt_int local_x = grid_coord.get_x() - ta_panel.get_grid_lb_x();
        irt_int local_y = grid_coord.get_y() - ta_panel.get_grid_lb_y();
        if (!ta_node_map.isInside(local_x, local_y)) {
          continue;
        }
        TANode& ta_node = ta_node_map[local_x][local_y];
        for (Orientation orientation : orientation_set) {
          if (task_idx_list.empty()) {
            ta_node.get_fence_task_map()[orientation].insert(-1);
          } else {
            ta_node.get_fence_task_map()[orientation].insert(task_idx_list.begin(), task_idx_list.end());
          }
        }
      }
    }
  }
}

#endif

#if 1  // check ta_panel

void TrackAssigner::checkTAPanel(TAPanel& ta_panel)
{
  if (ta_panel.get_panel_idx() < 0) {
    LOG_INST.error(Loc::current(), "The panel_idx ", ta_panel.get_panel_idx(), " is error!");
  }
  for (auto& [net_idx, blockage_list] : ta_panel.get_net_blockage_map()) {
    for (PlanarRect& blockage : blockage_list) {
      if (RTUtil::isClosedOverlap(ta_panel.get_real_rect(), blockage)) {
        continue;
      }
      LOG_INST.error(Loc::current(), "The blockage(", blockage.get_lb_x(), ",", blockage.get_lb_y(), ")-(", blockage.get_rt_x(), ",",
                     blockage.get_rt_y(), ") is out of panel!");
    }
  }
  for (auto& [net_idx, region_list] : ta_panel.get_net_fence_region_map()) {
    for (PlanarRect& region : region_list) {
      if (RTUtil::isClosedOverlap(ta_panel.get_real_rect(), region)) {
        continue;
      }
      LOG_INST.error(Loc::current(), "The region(", region.get_lb_x(), ",", region.get_lb_y(), ")-(", region.get_rt_x(), ",",
                     region.get_rt_y(), ") is out of panel!");
    }
  }
  GridMap<TANode>& ta_node_map = ta_panel.get_ta_node_map();
  for (irt_int x_idx = 0; x_idx < ta_node_map.get_x_size(); x_idx++) {
    for (irt_int y_idx = 0; y_idx < ta_node_map.get_y_size(); y_idx++) {
      TANode& ta_node = ta_node_map[x_idx][y_idx];
      irt_int node_layer_idx = ta_node.get_layer_idx();
      if (node_layer_idx != ta_panel.get_layer_idx()) {
        LOG_INST.error(Loc::current(), "The node layer ", node_layer_idx, " is different with panel layer ", ta_panel.get_layer_idx());
      }
      for (auto& [orien, neighbor] : ta_node.get_neighbor_ptr_map()) {
        Orientation opposite_orien = RTUtil::getOppositeOrientation(orien);
        if (!RTUtil::exist(neighbor->get_neighbor_ptr_map(), opposite_orien)) {
          LOG_INST.error(Loc::current(), "The ta_node neighbor is not bidirection!");
        }
        if (neighbor->get_neighbor_ptr_map()[opposite_orien] != &ta_node) {
          LOG_INST.error(Loc::current(), "The ta_node neighbor is not bidirection!");
        }
        LayerCoord node_coord(ta_node.get_planar_coord(), ta_node.get_layer_idx());
        LayerCoord neighbor_coord(neighbor->get_planar_coord(), neighbor->get_layer_idx());
        if (RTUtil::getOrientation(node_coord, neighbor_coord) != orien) {
          LOG_INST.error(Loc::current(), "The neighbor orien is different with real orien!");
        }
      }
      for (auto& [orien, task_idx_list] : ta_node.get_obs_task_map()) {
        if (task_idx_list.empty()) {
          LOG_INST.error(Loc::current(), "The task_idx_list is empty!");
        }
      }
      for (auto& [orien, task_idx_list] : ta_node.get_fence_task_map()) {
        if (task_idx_list.empty()) {
          LOG_INST.error(Loc::current(), "The task_idx_list is empty!");
        }
      }
    }
  }
  for (TATask& ta_task : ta_panel.get_ta_task_list()) {
    std::vector<TAGroup>& ta_group_list = ta_task.get_ta_group_list();
    if (ta_group_list.size() != 2) {
      LOG_INST.error(Loc::current(), "The ta_group_list size is wrong!");
    }
    for (TAGroup& ta_group : ta_group_list) {
      for (LayerCoord& coord : ta_group.get_coord_list()) {
        irt_int group_coord_layer_idx = coord.get_layer_idx();
        if (group_coord_layer_idx != ta_panel.get_layer_idx()) {
          LOG_INST.error(Loc::current(), "The group coord layer ", group_coord_layer_idx, " is different with panel layer ",
                         ta_panel.get_layer_idx());
        }
        if (RTUtil::isInside(ta_panel.get_real_rect(), coord)) {
          continue;
        }
        LOG_INST.error(Loc::current(), "The coord(", coord.get_x(), ",", coord.get_y(), ",", coord.get_layer_idx(),
                       ") is outside the panel!");
      }
    }
  }
}

#endif

#if 1  // sort ta_panel

void TrackAssigner::sortTAPanel(TAPanel& ta_panel)
{
  Monitor monitor;
  if (omp_get_num_threads() == 1) {
    LOG_INST.info(Loc::current(), "Sorting all tasks beginning...");
  }

  std::vector<TATask>& ta_task_list = ta_panel.get_ta_task_list();
  std::sort(ta_task_list.begin(), ta_task_list.end(), [&](TATask& task1, TATask& task2) { return sortByMultiLevel(task1, task2); });

  if (omp_get_num_threads() == 1) {
    LOG_INST.info(Loc::current(), "Sorting all tasks completed!", monitor.getStatsInfo());
  }
}

bool TrackAssigner::sortByMultiLevel(TATask& task1, TATask& task2)
{
  SortStatus sort_status = SortStatus::kNone;

  sort_status = sortByClockPriority(task1, task2);
  if (sort_status == SortStatus::kTrue) {
    return true;
  } else if (sort_status == SortStatus::kFalse) {
    return false;
  }
  sort_status = sortByLengthWidthRatioDESC(task1, task2);
  if (sort_status == SortStatus::kTrue) {
    return true;
  } else if (sort_status == SortStatus::kFalse) {
    return false;
  }
  return false;
}

// 时钟线网优先
SortStatus TrackAssigner::sortByClockPriority(TATask& task1, TATask& task2)
{
  ConnectType task1_connect_type = task1.get_ta_task_priority().get_connect_type();
  ConnectType task2_connect_type = task2.get_ta_task_priority().get_connect_type();

  if (task1_connect_type == ConnectType::kClock && task2_connect_type != ConnectType::kClock) {
    return SortStatus::kTrue;
  } else if (task1_connect_type != ConnectType::kClock && task2_connect_type == ConnectType::kClock) {
    return SortStatus::kFalse;
  } else {
    return SortStatus::kEqual;
  }
}

// 长宽比 降序
SortStatus TrackAssigner::sortByLengthWidthRatioDESC(TATask& task1, TATask& task2)
{
  double task1_length_width_ratio = task1.get_ta_task_priority().get_length_width_ratio();
  double task2_length_width_ratio = task2.get_ta_task_priority().get_length_width_ratio();

  if (task1_length_width_ratio > task2_length_width_ratio) {
    return SortStatus::kTrue;
  } else if (task1_length_width_ratio == task2_length_width_ratio) {
    return SortStatus::kEqual;
  } else {
    return SortStatus::kFalse;
  }
}

#endif

#if 1  // assign ta_panel

void TrackAssigner::assignTAPanel(TAPanel& ta_panel)
{
  Monitor monitor;

  std::vector<TATask>& ta_task_list = ta_panel.get_ta_task_list();

  irt_int batch_size = RTUtil::getBatchSize(ta_task_list.size());

  Monitor stage_monitor;
  for (size_t i = 0; i < ta_task_list.size(); i++) {
    routeTATask(ta_panel, ta_task_list[i]);
    if (omp_get_num_threads() == 1 && (i + 1) % batch_size == 0) {
      LOG_INST.info(Loc::current(), "Processed ", (i + 1), " tasks", stage_monitor.getStatsInfo());
    }
  }
  if (omp_get_num_threads() == 1) {
    LOG_INST.info(Loc::current(), "Processed ", ta_task_list.size(), " tasks", monitor.getStatsInfo());
  }
}

void TrackAssigner::routeTATask(TAPanel& ta_panel, TATask& ta_task)
{
  initRoutingInfo(ta_panel, ta_task);
  while (!isConnectedAllEnd(ta_panel)) {
    routeSinglePath(ta_panel);
    for (TARouteStrategy ta_route_strategy :
         {TARouteStrategy::kIgnoringENV, TARouteStrategy::kIgnoringFence, TARouteStrategy::kIgnoringOBS}) {
      rerouteByIgnoring(ta_panel, ta_route_strategy);
    }
    updatePathResult(ta_panel);
    updateDirectionSet(ta_panel);
    resetStartAndEnd(ta_panel);
    resetSinglePath(ta_panel);
  }
  updateNetResult(ta_panel, ta_task);
  resetSingleNet(ta_panel);
}

void TrackAssigner::initRoutingInfo(TAPanel& ta_panel, TATask& ta_task)
{
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();
  TrackAxis& track_axis = routing_layer_list[ta_panel.get_layer_idx()].get_track_axis();

  GridMap<TANode>& ta_node_map = ta_panel.get_ta_node_map();
  std::vector<std::vector<TANode*>>& start_node_comb_list = ta_panel.get_start_node_comb_list();
  std::vector<std::vector<TANode*>>& end_node_comb_list = ta_panel.get_end_node_comb_list();

  ta_panel.set_wire_unit(1);
  ta_panel.set_via_unit(1);
  ta_panel.set_ta_task_ref(&ta_task);
  ta_panel.set_routing_region(ta_panel.get_curr_bounding_box());

  std::vector<std::vector<TANode*>> node_comb_list;
  std::vector<TAGroup>& ta_group_list = ta_task.get_ta_group_list();
  for (TAGroup& ta_group : ta_group_list) {
    std::vector<TANode*> node_comb;
    for (LayerCoord& coord : ta_group.get_coord_list()) {
      if (!RTUtil::existGrid(coord, track_axis)) {
        LOG_INST.error(Loc::current(), "The coord can not find grid!");
      }
      PlanarCoord grid_coord = RTUtil::getGridCoord(coord, track_axis);
      irt_int local_x = grid_coord.get_x() - ta_panel.get_grid_lb_x();
      irt_int local_y = grid_coord.get_y() - ta_panel.get_grid_lb_y();
      node_comb.push_back(&ta_node_map[local_x][local_y]);
    }
    node_comb_list.push_back(node_comb);
  }
  for (size_t i = 0; i < node_comb_list.size(); i++) {
    if (i == 0) {
      start_node_comb_list.push_back(node_comb_list[i]);
    } else {
      end_node_comb_list.push_back(node_comb_list[i]);
    }
  }
}

bool TrackAssigner::isConnectedAllEnd(TAPanel& ta_panel)
{
  return ta_panel.get_end_node_comb_list().empty();
}

void TrackAssigner::routeSinglePath(TAPanel& ta_panel)
{
  initPathHead(ta_panel);
  while (!searchEnded(ta_panel)) {
    expandSearching(ta_panel);
    resetPathHead(ta_panel);
  }
}

void TrackAssigner::initPathHead(TAPanel& ta_panel)
{
  std::vector<std::vector<TANode*>>& start_node_comb_list = ta_panel.get_start_node_comb_list();
  std::vector<TANode*>& path_node_comb = ta_panel.get_path_node_comb();

  for (std::vector<TANode*>& start_node_comb : start_node_comb_list) {
    for (TANode* start_node : start_node_comb) {
      start_node->set_estimated_cost(getEstimateCostToEnd(ta_panel, start_node));
      pushToOpenList(ta_panel, start_node);
    }
  }
  for (TANode* path_node : path_node_comb) {
    path_node->set_estimated_cost(getEstimateCostToEnd(ta_panel, path_node));
    pushToOpenList(ta_panel, path_node);
  }
  ta_panel.set_path_head_node(popFromOpenList(ta_panel));
}

bool TrackAssigner::searchEnded(TAPanel& ta_panel)
{
  std::vector<std::vector<TANode*>>& end_node_comb_list = ta_panel.get_end_node_comb_list();
  TANode* path_head_node = ta_panel.get_path_head_node();

  if (path_head_node == nullptr) {
    ta_panel.set_end_node_comb_idx(-1);
    return true;
  }
  for (size_t i = 0; i < end_node_comb_list.size(); i++) {
    for (TANode* end_node : end_node_comb_list[i]) {
      if (path_head_node == end_node) {
        ta_panel.set_end_node_comb_idx(static_cast<irt_int>(i));
        return true;
      }
    }
  }
  return false;
}

void TrackAssigner::expandSearching(TAPanel& ta_panel)
{
  TANode* path_head_node = ta_panel.get_path_head_node();

  for (auto& [orientation, neighbor_node] : path_head_node->get_neighbor_ptr_map()) {
    if (neighbor_node == nullptr) {
      continue;
    }
    if (!RTUtil::isInside(ta_panel.get_routing_region(), *neighbor_node)) {
      continue;
    }
    if (neighbor_node->isClose()) {
      continue;
    }
    if (!passCheckingSegment(ta_panel, path_head_node, neighbor_node)) {
      continue;
    }
    if (neighbor_node->isOpen() && replaceParentNode(ta_panel, path_head_node, neighbor_node)) {
      neighbor_node->set_known_cost(getKnowCost(ta_panel, path_head_node, neighbor_node));
      neighbor_node->set_parent_node(path_head_node);
    } else if (neighbor_node->isNone()) {
      neighbor_node->set_known_cost(getKnowCost(ta_panel, path_head_node, neighbor_node));
      neighbor_node->set_parent_node(path_head_node);
      neighbor_node->set_estimated_cost(getEstimateCostToEnd(ta_panel, neighbor_node));
      pushToOpenList(ta_panel, neighbor_node);
    }
  }
}

bool TrackAssigner::passCheckingSegment(TAPanel& ta_panel, TANode* start_node, TANode* end_node)
{
  Orientation orientation = RTUtil::getOrientation(*start_node, *end_node);
  if (orientation == Orientation::kNone) {
    return true;
  }
  Orientation opposite_orientation = RTUtil::getOppositeOrientation(orientation);

  TANode* pre_node = nullptr;
  TANode* curr_node = start_node;

  while (curr_node != end_node) {
    pre_node = curr_node;
    curr_node = pre_node->getNeighborNode(orientation);

    if (curr_node == nullptr) {
      return false;
    }
    if (pre_node->isOBS(ta_panel.get_curr_task_idx(), orientation, ta_panel.get_ta_route_strategy())) {
      return false;
    }
    if (curr_node->isOBS(ta_panel.get_curr_task_idx(), opposite_orientation, ta_panel.get_ta_route_strategy())) {
      return false;
    }
  }
  return true;
}

bool TrackAssigner::replaceParentNode(TAPanel& ta_panel, TANode* parent_node, TANode* child_node)
{
  return getKnowCost(ta_panel, parent_node, child_node) < child_node->get_known_cost();
}

void TrackAssigner::resetPathHead(TAPanel& ta_panel)
{
  ta_panel.set_path_head_node(popFromOpenList(ta_panel));
}

bool TrackAssigner::isRoutingFailed(TAPanel& ta_panel)
{
  return ta_panel.get_end_node_comb_idx() == -1;
}

void TrackAssigner::resetSinglePath(TAPanel& ta_panel)
{
  std::priority_queue<TANode*, std::vector<TANode*>, CmpTANodeCost> empty_queue;
  ta_panel.set_open_queue(empty_queue);

  std::vector<TANode*>& visited_node_list = ta_panel.get_visited_node_list();
  for (TANode* visited_node : visited_node_list) {
    visited_node->set_state(TANodeState::kNone);
    visited_node->set_parent_node(nullptr);
    visited_node->set_known_cost(0);
    visited_node->set_estimated_cost(0);
  }
  visited_node_list.clear();

  ta_panel.set_path_head_node(nullptr);
  ta_panel.set_end_node_comb_idx(-1);
}

void TrackAssigner::rerouteByIgnoring(TAPanel& ta_panel, TARouteStrategy ta_route_strategy)
{
  if (isRoutingFailed(ta_panel)) {
    resetSinglePath(ta_panel);
    ta_panel.set_ta_route_strategy(ta_route_strategy);
    routeSinglePath(ta_panel);
    ta_panel.set_ta_route_strategy(TARouteStrategy::kNone);
    if (!isRoutingFailed(ta_panel)) {
      if (omp_get_num_threads() == 1) {
        LOG_INST.info(Loc::current(), "The task ", ta_panel.get_curr_task_idx(), " reroute by ",
                      GetTARouteStrategyName()(ta_route_strategy), " successfully!");
      }
    } else if (ta_route_strategy == TARouteStrategy::kIgnoringOBS) {
      LOG_INST.error(Loc::current(), "The task ", ta_panel.get_curr_task_idx(), " reroute by ", GetTARouteStrategyName()(ta_route_strategy),
                     " failed!");
    }
  }
}

void TrackAssigner::updatePathResult(TAPanel& ta_panel)
{
  std::vector<Segment<TANode*>>& node_segment_list = ta_panel.get_node_segment_list();
  TANode* path_head_node = ta_panel.get_path_head_node();

  TANode* curr_node = path_head_node;
  TANode* pre_node = curr_node->get_parent_node();

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

void TrackAssigner::updateDirectionSet(TAPanel& ta_panel)
{
  TANode* path_head_node = ta_panel.get_path_head_node();

  TANode* curr_node = path_head_node;
  TANode* pre_node = curr_node->get_parent_node();
  while (pre_node != nullptr) {
    curr_node->get_direction_set().insert(RTUtil::getDirection(*curr_node, *pre_node));
    pre_node->get_direction_set().insert(RTUtil::getDirection(*pre_node, *curr_node));
    curr_node = pre_node;
    pre_node = curr_node->get_parent_node();
  }
}

void TrackAssigner::resetStartAndEnd(TAPanel& ta_panel)
{
  std::vector<std::vector<TANode*>>& start_node_comb_list = ta_panel.get_start_node_comb_list();
  std::vector<std::vector<TANode*>>& end_node_comb_list = ta_panel.get_end_node_comb_list();
  std::vector<TANode*>& path_node_comb = ta_panel.get_path_node_comb();
  TANode* path_head_node = ta_panel.get_path_head_node();
  irt_int end_node_comb_idx = ta_panel.get_end_node_comb_idx();

  end_node_comb_list[end_node_comb_idx].clear();
  end_node_comb_list[end_node_comb_idx].push_back(path_head_node);

  TANode* path_node = path_head_node->get_parent_node();
  if (path_node == nullptr) {
    // 起点和终点重合
    path_node = path_head_node;
  } else {
    // 起点和终点不重合
    while (path_node->get_parent_node() != nullptr) {
      path_node_comb.push_back(path_node);
      path_node = path_node->get_parent_node();
    }
  }
  if (start_node_comb_list.size() == 1) {
    start_node_comb_list.front().clear();
    start_node_comb_list.front().push_back(path_node);
  }
  start_node_comb_list.push_back(end_node_comb_list[end_node_comb_idx]);
  end_node_comb_list.erase(end_node_comb_list.begin() + end_node_comb_idx);
}

void TrackAssigner::updateNetResult(TAPanel& ta_panel, TATask& ta_task)
{
  updateENVTaskMap(ta_panel, ta_task);
  updateDemand(ta_panel, ta_task);
  updateResult(ta_panel, ta_task);
}

void TrackAssigner::updateENVTaskMap(TAPanel& ta_panel, TATask& ta_task)
{
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();
  GridMap<TANode>& ta_node_map = ta_panel.get_ta_node_map();

  std::vector<Segment<LayerCoord>> net_segment_list;
  for (Segment<TANode*>& node_segment : ta_panel.get_node_segment_list()) {
    net_segment_list.emplace_back(*node_segment.get_first(), *node_segment.get_second());
  }
  for (LayerRect& real_rect : getRealRectList(net_segment_list)) {
    irt_int min_spacing = routing_layer_list[real_rect.get_layer_idx()].getMinSpacing(real_rect);
    PlanarRect enlarge_real_rect = RTUtil::getEnlargedRect(real_rect, min_spacing);

    for (auto& [grid_coord, orientation_set] : getGridOrientationMap(ta_panel, enlarge_real_rect)) {
      irt_int local_x = grid_coord.get_x() - ta_panel.get_grid_lb_x();
      irt_int local_y = grid_coord.get_y() - ta_panel.get_grid_lb_y();
      if (!ta_node_map.isInside(local_x, local_y)) {
        continue;
      }
      TANode& ta_node = ta_node_map[local_x][local_y];
      for (Orientation orientation : orientation_set) {
        ta_node.addEnv(ta_task.get_task_idx(), orientation);
      }
    }
  }
}

void TrackAssigner::updateDemand(TAPanel& ta_panel, TATask& ta_task)
{
  std::set<TANode*> usage_set;
  for (Segment<TANode*>& node_segment : ta_panel.get_node_segment_list()) {
    TANode* first_node = node_segment.get_first();
    TANode* second_node = node_segment.get_second();
    Orientation orientation = RTUtil::getOrientation(*first_node, *second_node);

    TANode* node_i = first_node;
    while (true) {
      usage_set.insert(node_i);
      if (node_i == second_node) {
        break;
      }
      node_i = node_i->getNeighborNode(orientation);
    }
  }
  for (TANode* usage_node : usage_set) {
    usage_node->addDemand(ta_task.get_task_idx());
  }
}

void TrackAssigner::updateResult(TAPanel& ta_panel, TATask& ta_task)
{
  for (Segment<TANode*>& node_segment : ta_panel.get_node_segment_list()) {
    ta_task.get_routing_segment_list().emplace_back(*node_segment.get_first(), *node_segment.get_second());
  }
}

void TrackAssigner::resetSingleNet(TAPanel& ta_panel)
{
  ta_panel.set_ta_task_ref(nullptr);
  ta_panel.get_start_node_comb_list().clear();
  ta_panel.get_end_node_comb_list().clear();
  ta_panel.get_path_node_comb().clear();

  for (Segment<TANode*>& node_segment : ta_panel.get_node_segment_list()) {
    TANode* first_node = node_segment.get_first();
    TANode* second_node = node_segment.get_second();
    Orientation orientation = RTUtil::getOrientation(*first_node, *second_node);

    TANode* node_i = first_node;
    while (true) {
      node_i->get_direction_set().clear();
      if (node_i == second_node) {
        break;
      }
      node_i = node_i->getNeighborNode(orientation);
    }
  }
  ta_panel.get_node_segment_list().clear();
}

// manager open list

void TrackAssigner::pushToOpenList(TAPanel& ta_panel, TANode* curr_node)
{
  std::priority_queue<TANode*, std::vector<TANode*>, CmpTANodeCost>& open_queue = ta_panel.get_open_queue();
  std::vector<TANode*>& visited_node_list = ta_panel.get_visited_node_list();

  open_queue.push(curr_node);
  curr_node->set_state(TANodeState::kOpen);
  visited_node_list.push_back(curr_node);
}

TANode* TrackAssigner::popFromOpenList(TAPanel& ta_panel)
{
  std::priority_queue<TANode*, std::vector<TANode*>, CmpTANodeCost>& open_queue = ta_panel.get_open_queue();

  TANode* node = nullptr;
  if (!open_queue.empty()) {
    node = open_queue.top();
    open_queue.pop();
    node->set_state(TANodeState::kClose);
  }
  return node;
}

// calculate known cost

double TrackAssigner::getKnowCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node)
{
  double cost = 0;
  cost += start_node->get_known_cost();
  cost += getJointCost(ta_panel, end_node, RTUtil::getOrientation(*end_node, *start_node));
  cost += getKnowWireCost(ta_panel, start_node, end_node);
  cost += getKnowCornerCost(ta_panel, start_node, end_node);
  cost += getViaCost(ta_panel, start_node, end_node);
  return cost;
}

double TrackAssigner::getJointCost(TAPanel& ta_panel, TANode* curr_node, Orientation orientation)
{
  const std::map<LayerCoord, double, CmpLayerCoordByXASC>& curr_coord_cost_map = ta_panel.get_curr_coord_cost_map();

  double task_cost = 0;
  auto iter = curr_coord_cost_map.find(*curr_node);
  if (iter != curr_coord_cost_map.end()) {
    task_cost = iter->second;
  }
  double env_cost = curr_node->getCost(ta_panel.get_curr_task_idx(), orientation);

  double env_weight = 1;
  double task_weight = 1;
  double joint_cost = ((env_weight * env_cost + task_weight * task_cost)
                       * RTUtil::sigmoid((env_weight * env_cost + task_weight * task_cost), (env_weight + task_weight)));
  return joint_cost;
}

double TrackAssigner::getKnowWireCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node)
{
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();

  double wire_cost = 0;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    RoutingLayer& routing_layer = routing_layer_list[start_node->get_layer_idx()];

    irt_int x_distance = std::abs(start_node->get_x() - end_node->get_x());
    irt_int y_distance = std::abs(start_node->get_y() - end_node->get_y());

    if (routing_layer.isPreferH()) {
      wire_cost += (x_distance * ta_panel.get_wire_unit());
      wire_cost += (y_distance * 2 * ta_panel.get_wire_unit());
    } else {
      wire_cost += (y_distance * ta_panel.get_wire_unit());
      wire_cost += (x_distance * 2 * ta_panel.get_wire_unit());
    }
  } else {
    wire_cost += (ta_panel.get_wire_unit() * RTUtil::getManhattanDistance(*start_node, *end_node));
  }
  return wire_cost;
}

double TrackAssigner::getKnowCornerCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node)
{
  double corner_cost = 0;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    std::set<Direction> start_direction_set = start_node->get_direction_set();
    if (start_node->get_parent_node() != nullptr) {
      start_direction_set.insert(RTUtil::getDirection(*start_node->get_parent_node(), *start_node));
    }
    std::set<Direction> end_direction_set = end_node->get_direction_set();
    end_direction_set.insert(RTUtil::getDirection(*start_node, *end_node));

    if (start_direction_set.size() == 1 && end_direction_set.size() == 1) {
      if (*start_direction_set.begin() != *end_direction_set.begin()) {
        corner_cost += ta_panel.get_corner_unit();
      }
    }
  }
  return corner_cost;
}

double TrackAssigner::getViaCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node)
{
  return ta_panel.get_via_unit() * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx());
}

// calculate estimate cost

double TrackAssigner::getEstimateCostToEnd(TAPanel& ta_panel, TANode* curr_node)
{
  std::vector<std::vector<TANode*>>& end_node_comb_list = ta_panel.get_end_node_comb_list();

  double estimate_cost = DBL_MAX;
  for (std::vector<TANode*>& end_node_comb : end_node_comb_list) {
    for (TANode* end_node : end_node_comb) {
      if (end_node->isClose()) {
        continue;
      }
      estimate_cost = std::min(estimate_cost, getEstimateCost(ta_panel, curr_node, end_node));
    }
  }
  return estimate_cost;
}

double TrackAssigner::getEstimateCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node)
{
  double estimate_cost = 0;
  estimate_cost += getEstimateWireCost(ta_panel, start_node, end_node);
  estimate_cost += getEstimateCornerCost(ta_panel, start_node, end_node);
  estimate_cost += getViaCost(ta_panel, start_node, end_node);
  return estimate_cost;
}

double TrackAssigner::getEstimateWireCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node)
{
  return ta_panel.get_wire_unit() * RTUtil::getManhattanDistance(*start_node, *end_node);
}

double TrackAssigner::getEstimateCornerCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node)
{
  double corner_cost = 0;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    if (RTUtil::isOblique(*start_node, *end_node)) {
      corner_cost += ta_panel.get_corner_unit();
    }
  }
  return corner_cost;
}

#endif

#if 1  // plot ta_panel

void TrackAssigner::plotTAPanel(TAPanel& ta_panel, irt_int curr_task_idx)
{
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();
  std::map<irt_int, irt_int> layer_width_map;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    irt_int x_pitch = routing_layer.getXTrackGrid().get_step_length();
    irt_int y_pitch = routing_layer.getYTrackGrid().get_step_length();
    irt_int width = std::min(x_pitch, y_pitch) / 10;
    layer_width_map[routing_layer.get_layer_idx()] = width;
  }

  GPGDS gp_gds;

  // base_region
  GPStruct base_region_struct("base_region");
  GPBoundary gp_boundary;
  gp_boundary.set_layer_idx(0);
  gp_boundary.set_data_type(0);
  gp_boundary.set_rect(ta_panel.get_real_rect());
  base_region_struct.push(gp_boundary);
  gp_gds.addStruct(base_region_struct);

  GridMap<TANode>& node_map = ta_panel.get_ta_node_map();
  // node_graph
  GPStruct node_graph_struct("node_graph");
  for (irt_int grid_x = 0; grid_x < node_map.get_x_size(); grid_x++) {
    for (irt_int grid_y = 0; grid_y < node_map.get_y_size(); grid_y++) {
      TANode& ta_node = node_map[grid_x][grid_y];
      PlanarRect real_rect = RTUtil::getEnlargedRect(ta_node.get_planar_coord(), layer_width_map[ta_node.get_layer_idx()]);
      irt_int y_reduced_span = real_rect.getYSpan() / 25;
      irt_int y = real_rect.get_rt_y();

      GPBoundary gp_boundary;
      switch (ta_node.get_state()) {
        case TANodeState::kNone:
          gp_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kNone));
          break;
        case TANodeState::kOpen:
          gp_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kOpen));
          break;
        case TANodeState::kClose:
          gp_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kClose));
          break;
        default:
          LOG_INST.error(Loc::current(), "The type is error!");
          break;
      }
      gp_boundary.set_rect(real_rect);
      gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(ta_node.get_layer_idx()));
      node_graph_struct.push(gp_boundary);

      y -= y_reduced_span;
      GPText gp_text_node_coord;
      gp_text_node_coord.set_coord(real_rect.get_lb_x(), y);
      gp_text_node_coord.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
      gp_text_node_coord.set_message(RTUtil::getString("(", grid_x, " , ", grid_y, " , ", ta_node.get_layer_idx(), ")"));
      gp_text_node_coord.set_layer_idx(GP_INST.getGDSIdxByRouting(ta_node.get_layer_idx()));
      gp_text_node_coord.set_presentation(GPTextPresentation::kLeftMiddle);
      node_graph_struct.push(gp_text_node_coord);

      y -= y_reduced_span;
      GPText gp_text_obs_task_map;
      gp_text_obs_task_map.set_coord(real_rect.get_lb_x(), y);
      gp_text_obs_task_map.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
      gp_text_obs_task_map.set_message("obs_task_map: ");
      gp_text_obs_task_map.set_layer_idx(GP_INST.getGDSIdxByRouting(ta_node.get_layer_idx()));
      gp_text_obs_task_map.set_presentation(GPTextPresentation::kLeftMiddle);
      node_graph_struct.push(gp_text_obs_task_map);

      for (auto& [orientation, task_idx_set] : ta_node.get_obs_task_map()) {
        y -= y_reduced_span;
        GPText gp_text_obs_task_map_info;
        gp_text_obs_task_map_info.set_coord(real_rect.get_lb_x(), y);
        gp_text_obs_task_map_info.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
        std::string obs_task_map_info_message = RTUtil::getString("--", GetOrientationName()(orientation), ": ");
        for (irt_int task_idx : task_idx_set) {
          obs_task_map_info_message += RTUtil::getString("(", task_idx, ")");
        }
        gp_text_obs_task_map_info.set_message(obs_task_map_info_message);
        gp_text_obs_task_map_info.set_layer_idx(GP_INST.getGDSIdxByRouting(ta_node.get_layer_idx()));
        gp_text_obs_task_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
        node_graph_struct.push(gp_text_obs_task_map_info);
      }

      y -= y_reduced_span;
      GPText gp_text_fence_task_map;
      gp_text_fence_task_map.set_coord(real_rect.get_lb_x(), y);
      gp_text_fence_task_map.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
      gp_text_fence_task_map.set_message("fence_task_map: ");
      gp_text_fence_task_map.set_layer_idx(GP_INST.getGDSIdxByRouting(ta_node.get_layer_idx()));
      gp_text_fence_task_map.set_presentation(GPTextPresentation::kLeftMiddle);
      node_graph_struct.push(gp_text_fence_task_map);

      for (auto& [orientation, task_idx_set] : ta_node.get_fence_task_map()) {
        y -= y_reduced_span;
        GPText gp_text_fence_task_map_info;
        gp_text_fence_task_map_info.set_coord(real_rect.get_lb_x(), y);
        gp_text_fence_task_map_info.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
        std::string fence_task_map_info_message = RTUtil::getString("--", GetOrientationName()(orientation), ": ");
        for (irt_int task_idx : task_idx_set) {
          fence_task_map_info_message += RTUtil::getString("(", task_idx, ")");
        }
        gp_text_fence_task_map_info.set_message(fence_task_map_info_message);
        gp_text_fence_task_map_info.set_layer_idx(GP_INST.getGDSIdxByRouting(ta_node.get_layer_idx()));
        gp_text_fence_task_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
        node_graph_struct.push(gp_text_fence_task_map_info);
      }

      y -= y_reduced_span;
      GPText gp_text_env_task_map;
      gp_text_env_task_map.set_coord(real_rect.get_lb_x(), y);
      gp_text_env_task_map.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
      gp_text_env_task_map.set_message("env_task_map: ");
      gp_text_env_task_map.set_layer_idx(GP_INST.getGDSIdxByRouting(ta_node.get_layer_idx()));
      gp_text_env_task_map.set_presentation(GPTextPresentation::kLeftMiddle);
      node_graph_struct.push(gp_text_env_task_map);

      for (auto& [orientation, task_idx_set] : ta_node.get_env_task_map()) {
        y -= y_reduced_span;
        GPText gp_text_env_task_map_info;
        gp_text_env_task_map_info.set_coord(real_rect.get_lb_x(), y);
        gp_text_env_task_map_info.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
        std::string env_task_map_info_message = RTUtil::getString("--", GetOrientationName()(orientation), ": ");
        for (irt_int task_idx : task_idx_set) {
          env_task_map_info_message += RTUtil::getString("(", task_idx, ")");
        }
        gp_text_env_task_map_info.set_message(env_task_map_info_message);
        gp_text_env_task_map_info.set_layer_idx(GP_INST.getGDSIdxByRouting(ta_node.get_layer_idx()));
        gp_text_env_task_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
        node_graph_struct.push(gp_text_env_task_map_info);
      }

      y -= y_reduced_span;
      GPText gp_text_task_queue;
      gp_text_task_queue.set_coord(real_rect.get_lb_x(), y);
      gp_text_task_queue.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
      gp_text_task_queue.set_message("task_queue: ");
      gp_text_task_queue.set_layer_idx(GP_INST.getGDSIdxByRouting(ta_node.get_layer_idx()));
      gp_text_task_queue.set_presentation(GPTextPresentation::kLeftMiddle);
      node_graph_struct.push(gp_text_task_queue);

      if (!ta_node.get_task_queue().empty()) {
        y -= y_reduced_span;
        GPText gp_text_task_queue_info;
        gp_text_task_queue_info.set_coord(real_rect.get_lb_x(), y);
        gp_text_task_queue_info.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
        std::string task_queue_info_message = "--";
        for (irt_int task_idx : RTUtil::getListByQueue(ta_node.get_task_queue())) {
          task_queue_info_message += RTUtil::getString("(", task_idx, ")");
        }
        gp_text_task_queue_info.set_message(task_queue_info_message);
        gp_text_task_queue_info.set_layer_idx(GP_INST.getGDSIdxByRouting(ta_node.get_layer_idx()));
        gp_text_task_queue_info.set_presentation(GPTextPresentation::kLeftMiddle);
        node_graph_struct.push(gp_text_task_queue_info);
      }

      y -= y_reduced_span;
      GPText gp_text_direction_set;
      gp_text_direction_set.set_coord(real_rect.get_lb_x(), y);
      gp_text_direction_set.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
      gp_text_direction_set.set_message("direction_set: ");
      gp_text_direction_set.set_layer_idx(GP_INST.getGDSIdxByRouting(ta_node.get_layer_idx()));
      gp_text_direction_set.set_presentation(GPTextPresentation::kLeftMiddle);
      node_graph_struct.push(gp_text_direction_set);

      if (!ta_node.get_direction_set().empty()) {
        y -= y_reduced_span;
        GPText gp_text_direction_set_info;
        gp_text_direction_set_info.set_coord(real_rect.get_lb_x(), y);
        gp_text_direction_set_info.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
        std::string direction_set_info_message = "--";
        for (Direction direction : ta_node.get_direction_set()) {
          direction_set_info_message += RTUtil::getString("(", GetDirectionName()(direction), ")");
        }
        gp_text_direction_set_info.set_message(direction_set_info_message);
        gp_text_direction_set_info.set_layer_idx(GP_INST.getGDSIdxByRouting(ta_node.get_layer_idx()));
        gp_text_direction_set_info.set_presentation(GPTextPresentation::kLeftMiddle);
        node_graph_struct.push(gp_text_direction_set_info);
      }
    }
  }
  gp_gds.addStruct(node_graph_struct);

  // neighbor
  GPStruct neighbor_map_struct("neighbor_map");
  for (irt_int grid_x = 0; grid_x < node_map.get_x_size(); grid_x++) {
    for (irt_int grid_y = 0; grid_y < node_map.get_y_size(); grid_y++) {
      TANode& ta_node = node_map[grid_x][grid_y];
      PlanarRect real_rect = RTUtil::getEnlargedRect(ta_node.get_planar_coord(), layer_width_map[ta_node.get_layer_idx()]);

      irt_int lb_x = real_rect.get_lb_x();
      irt_int lb_y = real_rect.get_lb_y();
      irt_int rt_x = real_rect.get_rt_x();
      irt_int rt_y = real_rect.get_rt_y();
      irt_int mid_x = (lb_x + rt_x) / 2;
      irt_int mid_y = (lb_y + rt_y) / 2;
      irt_int x_reduced_span = (rt_x - lb_x) / 4;
      irt_int y_reduced_span = (rt_y - lb_y) / 4;
      irt_int width = std::min(x_reduced_span, y_reduced_span) / 2;

      for (auto& [orientation, neighbor_node] : ta_node.get_neighbor_ptr_map()) {
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
        gp_path.set_layer_idx(GP_INST.getGDSIdxByRouting(ta_node.get_layer_idx()));
        gp_path.set_width(width);
        gp_path.set_data_type(static_cast<irt_int>(GPGraphType::kNeighbor));
        neighbor_map_struct.push(gp_path);
      }
    }
  }
  gp_gds.addStruct(neighbor_map_struct);

  // net blockage
  for (auto& [net_idx, blockage_list] : ta_panel.get_net_blockage_map()) {
    GPStruct blockage_struct(RTUtil::getString("blockage@", net_idx));
    for (const PlanarRect& blockage : blockage_list) {
      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kBlockage));
      gp_boundary.set_rect(blockage);
      gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(ta_panel.get_layer_idx()));
      blockage_struct.push(gp_boundary);
    }
    gp_gds.addStruct(blockage_struct);
  }

  // net fence_region
  for (auto& [net_idx, fence_region_list] : ta_panel.get_net_fence_region_map()) {
    GPStruct fence_region_struct(RTUtil::getString("fence_region@", net_idx));
    for (const PlanarRect& fence_region : fence_region_list) {
      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kFenceRegion));
      gp_boundary.set_rect(fence_region);
      gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(ta_panel.get_layer_idx()));
      fence_region_struct.push(gp_boundary);
    }
    gp_gds.addStruct(fence_region_struct);
  }

  // task
  for (TATask& ta_task : ta_panel.get_ta_task_list()) {
    GPStruct task_struct(RTUtil::getString("task_", ta_task.get_task_idx(), "(net_", ta_task.get_origin_net_idx(), ")"));

    if (curr_task_idx == -1 || ta_task.get_task_idx() == curr_task_idx) {
      for (TAGroup& ta_group : ta_task.get_ta_group_list()) {
        for (LayerCoord& coord : ta_group.get_coord_list()) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kKey));
          gp_boundary.set_rect(RTUtil::getEnlargedRect(coord, layer_width_map[coord.get_layer_idx()]));
          gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(coord.get_layer_idx()));
          task_struct.push(gp_boundary);
        }
      }
    }
    {
      // bounding_box
      GPBoundary gp_boundary;
      gp_boundary.set_layer_idx(0);
      gp_boundary.set_data_type(1);
      gp_boundary.set_rect(ta_task.get_bounding_box());
      task_struct.push(gp_boundary);
    }
    for (Segment<LayerCoord>& segment : ta_task.get_routing_segment_list()) {
      LayerCoord first_coord = segment.get_first();
      irt_int first_layer_idx = first_coord.get_layer_idx();
      LayerCoord second_coord = segment.get_second();
      irt_int second_layer_idx = second_coord.get_layer_idx();

      if (first_layer_idx == second_layer_idx) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kPath));
        gp_boundary.set_rect(RTUtil::getEnlargedRect(first_coord, second_coord, layer_width_map[first_layer_idx]));
        gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(first_layer_idx));
        task_struct.push(gp_boundary);
      } else {
        RTUtil::sortASC(first_layer_idx, second_layer_idx);
        for (irt_int layer_idx = first_layer_idx; layer_idx <= second_layer_idx; layer_idx++) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kPath));
          gp_boundary.set_rect(RTUtil::getEnlargedRect(first_coord, layer_width_map[layer_idx]));
          gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(layer_idx));
          task_struct.push(gp_boundary);
        }
      }
    }
    gp_gds.addStruct(task_struct);
  }
  GP_INST.plot(gp_gds, _ta_data_manager.getConfig().temp_directory_path + "ta_model.gds", false, false);
}

#endif

#if 1  // update ta_panel

void TrackAssigner::updateTAPanel(TAModel& ta_model, TAPanel& ta_panel)
{
  GCellAxis& gcell_axis = _ta_data_manager.getDatabase().get_gcell_axis();
  EXTPlanarRect& die = _ta_data_manager.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();

  std::vector<std::vector<TAPanel>>& layer_panel_list = ta_model.get_layer_panel_list();

  for (TATask& ta_task : ta_panel.get_ta_task_list()) {
    for (LayerRect& real_rect : getRealRectList(ta_task.get_routing_segment_list())) {
      irt_int layer_idx = real_rect.get_layer_idx();
      irt_int min_spacing = routing_layer_list[layer_idx].getMinSpacing(real_rect);
      PlanarRect enlarged_real_rect = RTUtil::getEnlargedRect(real_rect, min_spacing, die.get_real_rect());
      PlanarRect enlarged_grid_rect = RTUtil::getClosedGridRect(enlarged_real_rect, gcell_axis);
      if (routing_layer_list[layer_idx].isPreferH()) {
        for (irt_int y = enlarged_grid_rect.get_lb_y(); y <= enlarged_grid_rect.get_rt_y(); y++) {
          TAPanel& ta_panel = layer_panel_list[layer_idx][y];
          if (!RTUtil::isClosedOverlap(ta_panel.get_real_rect(), enlarged_real_rect)) {
            continue;
          }
          ta_panel.get_net_blockage_map()[ta_task.get_origin_net_idx()].push_back(enlarged_real_rect);
        }
      } else {
        for (irt_int x = enlarged_grid_rect.get_lb_x(); x <= enlarged_grid_rect.get_rt_x(); x++) {
          TAPanel& ta_panel = layer_panel_list[layer_idx][x];
          if (!RTUtil::isClosedOverlap(ta_panel.get_real_rect(), enlarged_real_rect)) {
            continue;
          }
          ta_panel.get_net_blockage_map()[ta_task.get_origin_net_idx()].push_back(enlarged_real_rect);
        }
      }
    }
  }
}

#endif

#if 1  // update ta_model

void TrackAssigner::updateTAModel(TAModel& ta_model)
{
  for (std::vector<TAPanel>& ta_panel_list : ta_model.get_layer_panel_list()) {
#pragma omp parallel for
    for (TAPanel& ta_panel : ta_panel_list) {
      for (TATask& ta_task : ta_panel.get_ta_task_list()) {
        buildRoutingResult(ta_task);
      }
    }
  }
  updateOriginTAResultTree(ta_model);
}

void TrackAssigner::buildRoutingResult(TATask& ta_task)
{
  std::vector<LayerCoord> driving_grid_coord_list;
  std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC> key_coord_pin_map;
  std::vector<TAGroup> ta_group_list = getBoundaryTAGroupList(ta_task);
  for (size_t i = 0; i < ta_group_list.size(); i++) {
    for (LayerCoord& coord : ta_group_list[i].get_coord_list()) {
      driving_grid_coord_list.push_back(coord);
      key_coord_pin_map[coord].insert(static_cast<irt_int>(i));
    }
  }
  std::vector<Segment<LayerCoord>>& routing_segment_list = ta_task.get_routing_segment_list();
  RTNode& rt_node = ta_task.get_origin_node()->value();
  rt_node.set_routing_tree(RTUtil::getTreeByFullFlow(driving_grid_coord_list, routing_segment_list, key_coord_pin_map));
}

std::vector<TAGroup> TrackAssigner::getBoundaryTAGroupList(TATask& ta_task)
{
  std::vector<irt::RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();

  RTNode& rt_node = ta_task.get_origin_node()->value();
  Guide first_guide = rt_node.get_first_guide();
  Guide second_guide = rt_node.get_second_guide();
  if (!CmpPlanarCoordByXASC()(first_guide.get_grid_coord(), second_guide.get_grid_coord())) {
    std::swap(first_guide, second_guide);
  }
  PlanarCoord& first_grid_coord = first_guide.get_grid_coord();
  PlanarCoord& second_grid_coord = second_guide.get_grid_coord();
  RoutingLayer& routing_layer = routing_layer_list[first_guide.get_layer_idx()];

  std::vector<TAGroup> ta_group_list;
  for (Guide guide : {first_guide, second_guide}) {
    std::vector<irt_int> x_list = RTUtil::getClosedScaleList(guide.get_lb_x(), guide.get_rt_x(), routing_layer.getXTrackGrid());
    std::vector<irt_int> y_list = RTUtil::getClosedScaleList(guide.get_lb_y(), guide.get_rt_y(), routing_layer.getYTrackGrid());
    if (RTUtil::isHorizontal(first_grid_coord, second_grid_coord)) {
      irt_int x = (guide.get_grid_coord() == first_grid_coord) ? x_list.back() : x_list.front();
      x_list.clear();
      x_list.push_back(x);
    } else if (RTUtil::isVertical(first_grid_coord, second_grid_coord)) {
      irt_int y = (guide.get_grid_coord() == first_grid_coord) ? y_list.back() : y_list.front();
      y_list.clear();
      y_list.push_back(y);
    }
    TAGroup ta_group;
    for (irt_int x : x_list) {
      for (irt_int y : y_list) {
        ta_group.get_coord_list().emplace_back(x, y, guide.get_layer_idx());
      }
    }
    ta_group_list.push_back(ta_group);
  }
  return ta_group_list;
}

void TrackAssigner::updateOriginTAResultTree(TAModel& ta_model)
{
  for (TANet& ta_net : ta_model.get_ta_net_list()) {
    Net* origin_net = ta_net.get_origin_net();
    origin_net->set_ta_result_tree(ta_net.get_ta_result_tree());
  }
}

#endif

#if 1  // report ta_model

void TrackAssigner::reportTAModel(TAModel& ta_model)
{
  countTAModel(ta_model);
  reportTable(ta_model);
}

void TrackAssigner::countTAModel(TAModel& ta_model)
{
  irt_int micron_dbu = _ta_data_manager.getDatabase().get_micron_dbu();

  TAModelStat& ta_model_stat = ta_model.get_ta_model_stat();
  std::map<irt_int, double>& routing_wire_length = ta_model_stat.get_routing_wire_length_map();
  std::map<irt_int, std::set<PlanarRect, CmpPlanarRectByXASC>>& routing_net_and_obs_rect_map
      = ta_model_stat.get_routing_net_and_obs_rect_map();
  std::map<irt_int, std::set<PlanarRect, CmpPlanarRectByXASC>>& routing_net_and_fence_rect_map
      = ta_model_stat.get_routing_net_and_fence_rect_map();
  std::map<irt_int, std::set<PlanarRect, CmpPlanarRectByXASC>>& routing_net_and_net_rect_map
      = ta_model_stat.get_routing_net_and_net_rect_map();

  for (std::vector<TAPanel>& panel_list : ta_model.get_layer_panel_list()) {
    for (TAPanel& ta_panel : panel_list) {
      for (TATask& ta_task : ta_panel.get_ta_task_list()) {
        for (Segment<LayerCoord>& routing_segment : ta_task.get_routing_segment_list()) {
          double wire_length = RTUtil::getManhattanDistance(routing_segment.get_first(), routing_segment.get_second()) / 1.0 / micron_dbu;
          routing_wire_length[ta_panel.get_layer_idx()] += wire_length;
        }
        for (LayerRect& real_rect : getRealRectList(ta_task.get_routing_segment_list())) {
          for (auto& [net_idx, blockage_list] : ta_panel.get_net_blockage_map()) {
            if (ta_task.get_origin_net_idx() == net_idx) {
              continue;
            }
            for (PlanarRect& blockage : blockage_list) {
              if (RTUtil::isOpenOverlap(real_rect, blockage)) {
                PlanarRect violation_rect = RTUtil::getOverlap(real_rect, blockage);
                if (net_idx == -1) {
                  routing_net_and_obs_rect_map[ta_panel.get_layer_idx()].insert(violation_rect);
                } else {
                  routing_net_and_net_rect_map[ta_panel.get_layer_idx()].insert(violation_rect);
                }
              }
            }
          }
          for (auto& [net_idx, fence_region_list] : ta_panel.get_net_fence_region_map()) {
            if (ta_task.get_origin_net_idx() == net_idx) {
              continue;
            }
            for (PlanarRect& fence_region : fence_region_list) {
              if (RTUtil::isOpenOverlap(real_rect, fence_region)) {
                PlanarRect violation_rect = RTUtil::getOverlap(real_rect, fence_region);
                routing_net_and_fence_rect_map[ta_panel.get_layer_idx()].insert(violation_rect);
              }
            }
          }
        }
      }
    }
  }
  double total_wire_length = 0;
  irt_int total_net_and_obs_rect_number = 0;
  irt_int total_net_and_fence_rect_number = 0;
  irt_int total_net_and_net_rect_number = 0;
  for (auto& [routing_layer_idx, wire_length] : routing_wire_length) {
    total_wire_length += wire_length;
  }
  for (auto& [routing_layer_idx, rect_list] : routing_net_and_obs_rect_map) {
    total_net_and_obs_rect_number += static_cast<irt_int>(rect_list.size());
  }
  for (auto& [routing_layer_idx, rect_list] : routing_net_and_fence_rect_map) {
    total_net_and_fence_rect_number += static_cast<irt_int>(rect_list.size());
  }
  for (auto& [routing_layer_idx, rect_list] : routing_net_and_net_rect_map) {
    total_net_and_net_rect_number += static_cast<irt_int>(rect_list.size());
  }
  ta_model_stat.set_total_wire_length(total_wire_length);
  ta_model_stat.set_total_net_and_obs_rect_number(total_net_and_obs_rect_number);
  ta_model_stat.set_total_net_and_fence_rect_number(total_net_and_fence_rect_number);
  ta_model_stat.set_total_net_and_net_rect_number(total_net_and_net_rect_number);
}

void TrackAssigner::reportTable(TAModel& ta_model)
{
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();

  TAModelStat& ta_model_stat = ta_model.get_ta_model_stat();
  std::map<irt_int, double>& routing_wire_length = ta_model_stat.get_routing_wire_length_map();
  std::map<irt_int, std::set<PlanarRect, CmpPlanarRectByXASC>>& routing_net_and_obs_rect_map
      = ta_model_stat.get_routing_net_and_obs_rect_map();
  std::map<irt_int, std::set<PlanarRect, CmpPlanarRectByXASC>>& routing_net_and_fence_rect_map
      = ta_model_stat.get_routing_net_and_fence_rect_map();
  std::map<irt_int, std::set<PlanarRect, CmpPlanarRectByXASC>>& routing_net_and_net_rect_map
      = ta_model_stat.get_routing_net_and_net_rect_map();
  double total_wire_length = ta_model_stat.get_total_wire_length();
  irt_int total_net_and_obs_rect_number = ta_model_stat.get_total_net_and_obs_rect_number();
  irt_int total_net_and_fence_rect_number = ta_model_stat.get_total_net_and_fence_rect_number();
  irt_int total_net_and_net_rect_number = ta_model_stat.get_total_net_and_net_rect_number();

  fort::char_table wire_table;
  wire_table.set_border_style(FT_SOLID_STYLE);
  wire_table << fort::header << "Routing Layer"
             << "Wire Length / um" << fort::endr;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    double wire_length = routing_wire_length[routing_layer.get_layer_idx()];
    wire_table << routing_layer.get_layer_name()
               << RTUtil::getString(wire_length, "(", RTUtil::getPercentage(wire_length, total_wire_length), "%)") << fort::endr;
  }
  wire_table << fort::header << "Total" << total_wire_length << fort::endr;
  for (std::string table_str : RTUtil::splitString(wire_table.to_string(), '\n')) {
    LOG_INST.info(Loc::current(), table_str);
  }
  fort::char_table violation_table;
  violation_table.set_border_style(FT_SOLID_STYLE);
  violation_table << fort::header << "Routing Layer"
                  << "Net And Obs Rect Number"
                  << "Net And Fence Rect Number"
                  << "Net And Net Rect Number" << fort::endr;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    irt_int net_and_obs_rect_number = static_cast<irt_int>(routing_net_and_obs_rect_map[routing_layer.get_layer_idx()].size());
    irt_int net_and_fence_rect_number = static_cast<irt_int>(routing_net_and_fence_rect_map[routing_layer.get_layer_idx()].size());
    irt_int net_and_net_rect_number = static_cast<irt_int>(routing_net_and_net_rect_map[routing_layer.get_layer_idx()].size());
    violation_table << routing_layer.get_layer_name()
                    << RTUtil::getString(net_and_obs_rect_number, "(",
                                         RTUtil::getPercentage(net_and_obs_rect_number, total_net_and_obs_rect_number), "%)")
                    << RTUtil::getString(net_and_fence_rect_number, "(",
                                         RTUtil::getPercentage(net_and_fence_rect_number, total_net_and_fence_rect_number), "%)")
                    << RTUtil::getString(net_and_net_rect_number, "(",
                                         RTUtil::getPercentage(net_and_net_rect_number, total_net_and_net_rect_number), "%)")
                    << fort::endr;
  }
  violation_table << fort::header << "Total" << total_net_and_obs_rect_number << total_net_and_fence_rect_number
                  << total_net_and_net_rect_number << fort::endr;
  for (std::string table_str : RTUtil::splitString(violation_table.to_string(), '\n')) {
    LOG_INST.info(Loc::current(), table_str);
  }
}

#endif

}  // namespace irt
