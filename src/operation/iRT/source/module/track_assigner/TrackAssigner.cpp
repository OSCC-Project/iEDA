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
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();
  irt_int bottom_routing_layer_idx = _ta_data_manager.getConfig().bottom_routing_layer_idx;
  irt_int top_routing_layer_idx = _ta_data_manager.getConfig().top_routing_layer_idx;

  TAModel ta_model;
  ta_model.set_ta_net_list(ta_net_list);

  std::vector<std::vector<TAPanel>>& layer_panel_list = ta_model.get_layer_panel_list();
  for (RoutingLayer& routing_layer : routing_layer_list) {
    irt_int layer_idx = routing_layer.get_layer_idx();

    std::vector<TAPanel> ta_panel_list;
    if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
      layer_panel_list.push_back(ta_panel_list);
      continue;
    }
    std::vector<GCellGrid> gcell_grid_list;
    if (routing_layer.isPreferH()) {
      gcell_grid_list = gcell_axis.get_y_grid_list();
    } else {
      gcell_grid_list = gcell_axis.get_x_grid_list();
    }
    for (GCellGrid& gcell_grid : gcell_grid_list) {
      for (irt_int line = gcell_grid.get_start_line(); line < gcell_grid.get_end_line(); line += gcell_grid.get_step_length()) {
        TAPanel ta_panel;
        ta_panel.set_panel_idx(static_cast<irt_int>(ta_panel_list.size()));
        ta_panel.set_layer_idx(layer_idx);
        ta_panel_list.push_back(ta_panel);
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
  addBlockageList(ta_model);
  addNetRegionList(ta_model);
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
      ta_task.set_origin_node(ta_node_node);
      ta_task.set_task_idx(static_cast<irt_int>(ta_task_list.size()));
      ta_task.set_origin_net_idx(ta_net.get_net_idx());
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
  std::map<TNode<RTNode>*, TATask> ta_node_task_map = initGroupAndCost(ta_net);
  makeCoordCostMap(ta_node_task_map);
  return ta_node_task_map;
}

std::map<TNode<RTNode>*, TATask> TrackAssigner::initGroupAndCost(TANet& ta_net)
{
  std::map<TNode<RTNode>*, TATask> ta_node_task_map;
  for (auto& [dr_node_node, ta_node_node_list] : getDRTAListMap(ta_net)) {
    std::vector<LayerCoord> pin_coord_list;
    for (irt_int pin_idx : dr_node_node->value().get_pin_idx_set()) {
      std::vector<LayerCoord> real_coord_list = ta_net.get_ta_pin_list()[pin_idx].getRealCoordList();
      pin_coord_list.insert(pin_coord_list.end(), real_coord_list.begin(), real_coord_list.end());
    }

    std::map<TNode<RTNode>*, TAGroup> ta_node_group_map;
    for (TNode<RTNode>* ta_node_node : ta_node_node_list) {
      ta_node_group_map[ta_node_node] = makeTAGroup(dr_node_node, ta_node_node);
    }
    for (auto& [ta_node_node1, group1] : ta_node_group_map) {
      // 真实坐标
      std::vector<std::tuple<LayerCoord, irt_int, irt_int>> coord_distance_list;
      for (LayerCoord& coord : group1.get_coord_list()) {
        coord_distance_list.emplace_back(coord, 0, 0);
      }
      for (auto& [coord, pin_distance, group_distance] : coord_distance_list) {
        for (LayerCoord& pin_coord : pin_coord_list) {
          pin_distance += RTUtil::getManhattanDistance(coord, pin_coord);
        }
        for (auto& [ta_node_node2, group2] : ta_node_group_map) {
          for (LayerCoord& group_coord : group2.get_coord_list()) {
            group_distance += RTUtil::getManhattanDistance(coord, group_coord);
          }
        }
      }
      std::sort(coord_distance_list.begin(), coord_distance_list.end(),
                [](std::tuple<LayerCoord, irt_int, irt_int>& a, std::tuple<LayerCoord, irt_int, irt_int>& b) {
                  return std::get<1>(a) == std::get<1>(b) ? std::get<2>(a) < std::get<2>(b) : std::get<1>(a) < std::get<1>(b);
                });
      // 构建 ta_group_list 和 coord_cost_map
      TATask& ta_task = ta_node_task_map[ta_node_node1];
      ta_task.get_ta_group_list().push_back(group1);
      for (size_t i = 0; i < coord_distance_list.size(); i++) {
        ta_task.get_coord_cost_map()[std::get<0>(coord_distance_list[i])] = static_cast<double>(i);
      }
    }
  }
  return ta_node_task_map;
}

std::map<TNode<RTNode>*, std::vector<TNode<RTNode>*>> TrackAssigner::getDRTAListMap(TANet& ta_net)
{
  std::map<TNode<RTNode>*, std::vector<TNode<RTNode>*>> dr_ta_list_map;
  for (Segment<TNode<RTNode>*>& segment : RTUtil::getSegListByTree(ta_net.get_ta_result_tree())) {
    TNode<RTNode>* dr_node_node = segment.get_first();
    TNode<RTNode>* ta_node_node = segment.get_second();
    if (dr_node_node->value().isTANode()) {
      std::swap(dr_node_node, ta_node_node);
    }
    dr_ta_list_map[dr_node_node].push_back(ta_node_node);
  }
  return dr_ta_list_map;
}

TAGroup TrackAssigner::makeTAGroup(TNode<RTNode>* dr_node_node, TNode<RTNode>* ta_node_node)
{
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();

  Guide& dr_guide = dr_node_node->value().get_first_guide();
  PlanarCoord& dr_grid_coord = dr_guide.get_grid_coord();

  PlanarCoord& first_grid_coord = ta_node_node->value().get_first_guide().get_grid_coord();
  PlanarCoord& second_grid_coord = ta_node_node->value().get_second_guide().get_grid_coord();
  irt_int ta_layer_idx = ta_node_node->value().get_first_guide().get_layer_idx();

  RoutingLayer& routing_layer = routing_layer_list[ta_layer_idx];

  Orientation orientation = Orientation::kNone;
  if (dr_grid_coord == first_grid_coord) {
    orientation = RTUtil::getOrientation(dr_grid_coord, second_grid_coord);
  } else {
    orientation = RTUtil::getOrientation(dr_grid_coord, first_grid_coord);
  }
  std::vector<irt_int> x_list;
  std::vector<irt_int> y_list;
  if (orientation == Orientation::kEast || orientation == Orientation::kWest) {
    x_list = RTUtil::getOpenScaleList(dr_guide.get_lb_x(), dr_guide.get_rt_x(), routing_layer.getXTrackGrid());
    y_list = RTUtil::getOpenScaleList(dr_guide.get_lb_y(), dr_guide.get_rt_y(), routing_layer.getYTrackGrid());
    irt_int x = orientation == Orientation::kEast ? x_list.back() : x_list.front();
    x_list.clear();
    x_list.push_back(x);
  } else if (orientation == Orientation::kNorth || orientation == Orientation::kSouth) {
    x_list = RTUtil::getOpenScaleList(dr_guide.get_lb_x(), dr_guide.get_rt_x(), routing_layer.getXTrackGrid());
    y_list = RTUtil::getOpenScaleList(dr_guide.get_lb_y(), dr_guide.get_rt_y(), routing_layer.getYTrackGrid());
    irt_int y = orientation == Orientation::kNorth ? y_list.back() : y_list.front();
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

void TrackAssigner::makeCoordCostMap(std::map<TNode<RTNode>*, TATask>& ta_node_task_map)
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
      if (ta_panel.skipAssigning()) {
        continue;
      }
      TrackAxis& track_axis = routing_layer_list[ta_panel.get_layer_idx()].get_track_axis();

      std::vector<PlanarCoord> coord_list;
      for (TATask& ta_task : ta_panel.get_ta_task_list()) {
        for (TAGroup& ta_group : ta_task.get_ta_group_list()) {
          coord_list.insert(coord_list.end(), ta_group.get_coord_list().begin(), ta_group.get_coord_list().end());
        }
      }
      PlanarRect panel_region = RTUtil::getBoundingBox(coord_list);
      ta_panel.set_real_lb(panel_region.get_lb());
      ta_panel.set_real_rt(panel_region.get_rt());
      if (!RTUtil::existGrid(ta_panel.get_real_rect(), track_axis)) {
        LOG_INST.error(Loc::current(), "The panel not contain any grid!");
      }
      ta_panel.set_grid_rect(RTUtil::getGridRect(ta_panel.get_real_rect(), track_axis));
    }
  }
}

void TrackAssigner::addBlockageList(TAModel& ta_model)
{
  GCellAxis& gcell_axis = _ta_data_manager.getDatabase().get_gcell_axis();
  EXTPlanarRect& die = _ta_data_manager.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();
  std::vector<Blockage>& routing_blockage_list = _ta_data_manager.getDatabase().get_routing_blockage_list();
  irt_int bottom_routing_layer_idx = _ta_data_manager.getConfig().bottom_routing_layer_idx;
  irt_int top_routing_layer_idx = _ta_data_manager.getConfig().top_routing_layer_idx;

  std::vector<std::vector<TAPanel>>& layer_panel_list = ta_model.get_layer_panel_list();

  for (const Blockage& routing_blockage : routing_blockage_list) {
    irt_int layer_idx = routing_blockage.get_layer_idx();
    if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
      continue;
    }
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
        if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
          continue;
        }
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

void TrackAssigner::addNetRegionList(TAModel& ta_model)
{
  GCellAxis& gcell_axis = _ta_data_manager.getDatabase().get_gcell_axis();
  EXTPlanarRect& die = _ta_data_manager.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _ta_data_manager.getDatabase().get_layer_via_master_list();
  irt_int bottom_routing_layer_idx = _ta_data_manager.getConfig().bottom_routing_layer_idx;
  irt_int top_routing_layer_idx = _ta_data_manager.getConfig().top_routing_layer_idx;

  std::vector<std::vector<TAPanel>>& layer_panel_list = ta_model.get_layer_panel_list();

  for (TANet& ta_net : ta_model.get_ta_net_list()) {
    std::vector<EXTLayerRect> net_region_list;
    for (TAPin& ta_pin : ta_net.get_ta_pin_list()) {
      for (LayerCoord& real_coord : ta_pin.getRealCoordList()) {
        irt_int layer_idx = real_coord.get_layer_idx();
        for (irt_int via_below_layer_idx : RTUtil::getViaBelowLayerIdxList(layer_idx, bottom_routing_layer_idx, top_routing_layer_idx)) {
          ViaMaster& via_master = layer_via_master_list[via_below_layer_idx].front();

          const LayerRect& below_enclosure = via_master.get_below_enclosure();
          EXTLayerRect below_via_shape;
          below_via_shape.set_real_rect(RTUtil::getOffsetRect(below_enclosure, real_coord));
          below_via_shape.set_layer_idx(below_enclosure.get_layer_idx());
          net_region_list.push_back(below_via_shape);

          const LayerRect& above_enclosure = via_master.get_above_enclosure();
          EXTLayerRect above_via_shape;
          above_via_shape.set_real_rect(RTUtil::getOffsetRect(above_enclosure, real_coord));
          above_via_shape.set_layer_idx(above_enclosure.get_layer_idx());
          net_region_list.push_back(above_via_shape);
        }
      }
    }
    for (const EXTLayerRect& net_region : net_region_list) {
      irt_int layer_idx = net_region.get_layer_idx();
      if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
        continue;
      }
      irt_int min_spacing = routing_layer_list[layer_idx].getMinSpacing(net_region.get_real_rect());
      PlanarRect enlarged_real_rect = RTUtil::getEnlargedRect(net_region.get_real_rect(), min_spacing, die.get_real_rect());
      PlanarRect enlarged_grid_rect = RTUtil::getClosedGridRect(enlarged_real_rect, gcell_axis);

      if (routing_layer_list[layer_idx].isPreferH()) {
        for (irt_int y = enlarged_grid_rect.get_lb_y(); y <= enlarged_grid_rect.get_rt_y(); y++) {
          TAPanel& ta_panel = layer_panel_list[layer_idx][y];
          if (!RTUtil::isClosedOverlap(ta_panel.get_real_rect(), enlarged_real_rect)) {
            continue;
          }
          ta_panel.get_net_region_map()[ta_net.get_net_idx()].push_back(enlarged_real_rect);
        }
      } else {
        for (irt_int x = enlarged_grid_rect.get_lb_x(); x <= enlarged_grid_rect.get_rt_x(); x++) {
          TAPanel& ta_panel = layer_panel_list[layer_idx][x];
          if (!RTUtil::isClosedOverlap(ta_panel.get_real_rect(), enlarged_real_rect)) {
            continue;
          }
          ta_panel.get_net_region_map()[ta_net.get_net_idx()].push_back(enlarged_real_rect);
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
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();
  std::vector<std::vector<TAPanel>>& layer_panel_list = ta_model.get_layer_panel_list();

  Monitor monitor;

  size_t total_panel_num = 0;
  for (size_t layer_idx = 0; layer_idx < layer_panel_list.size(); layer_idx++) {
    std::vector<TAPanel>& ta_panel_list = layer_panel_list[layer_idx];
    Monitor stage_monitor;
#pragma omp parallel for
    for (TAPanel& ta_panel : ta_panel_list) {
      if (ta_panel.skipAssigning()) {
        continue;
      }
      buildTAPanel(ta_panel);
      checkTAPanel(ta_panel);
      sortTAPanel(ta_panel);
      assignTAPanel(ta_panel);
      countTAPanel(ta_panel);
      ta_panel.freeNodeMap();
    }
    total_panel_num += ta_panel_list.size();
    LOG_INST.info(Loc::current(), "Processed ", ta_panel_list.size(), " panels in ", routing_layer_list[layer_idx].get_layer_name(),
                  stage_monitor.getStatsInfo());
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
  buildCostTaskMap(ta_panel);
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
  for (irt_int x = 0; x < ta_node_map.get_x_size(); x++) {
    for (irt_int y = 0; y < ta_node_map.get_y_size(); y++) {
      TANode& ta_node = ta_node_map[x][y];
      ta_node.set_x(x_list[x]);
      ta_node.set_y(y_list[y]);
      ta_node.set_layer_idx(layer_idx);
    }
  }
}

void TrackAssigner::buildNeighborMap(TAPanel& ta_panel)
{
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();
  RoutingLayer& routing_layer = routing_layer_list[ta_panel.get_layer_idx()];

  GridMap<TANode>& ta_node_map = ta_panel.get_ta_node_map();
  for (irt_int x = 0; x < ta_node_map.get_x_size(); x++) {
    for (irt_int y = 0; y < ta_node_map.get_y_size(); y++) {
      std::map<Orientation, TANode*>& neighbor_ptr_map = ta_node_map[x][y].get_neighbor_ptr_map();

      if (routing_layer.isPreferH()) {
        if (x != 0) {
          neighbor_ptr_map[Orientation::kWest] = &ta_node_map[x - 1][y];
        }
        if (x != (ta_node_map.get_x_size() - 1)) {
          neighbor_ptr_map[Orientation::kEast] = &ta_node_map[x + 1][y];
        }
      } else {
        if (y != 0) {
          neighbor_ptr_map[Orientation::kSouth] = &ta_node_map[x][y - 1];
        }
        if (y != (ta_node_map.get_y_size() - 1)) {
          neighbor_ptr_map[Orientation::kNorth] = &ta_node_map[x][y + 1];
        }
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
      for (auto& [grid_coord, orientation_set] : getGridOrientationMap(ta_panel.get_layer_idx(), blockage)) {
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

std::map<PlanarCoord, std::set<Orientation>, CmpPlanarCoordByXASC> TrackAssigner::getGridOrientationMap(irt_int layer_idx,
                                                                                                        PlanarRect& blockage)
{
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();
  RoutingLayer& routing_layer = routing_layer_list[layer_idx];
  irt_int half_width = routing_layer.get_min_width() / 2;
  TrackAxis& track_axis = routing_layer.get_track_axis();

  // 先膨胀half_width
  PlanarRect search_rect = RTUtil::getEnlargedRect(blockage, half_width);
  irt_int x_step_length = track_axis.get_x_track_grid().get_step_length();
  irt_int y_step_length = track_axis.get_y_track_grid().get_step_length();
  search_rect = RTUtil::getEnlargedRect(search_rect, x_step_length, y_step_length, x_step_length, y_step_length);

  std::vector<Segment<PlanarCoord>> segment_list;
  std::vector<irt_int> x_list = RTUtil::getClosedScaleList(search_rect.get_lb_x(), search_rect.get_rt_x(), track_axis.get_x_track_grid());
  std::vector<irt_int> y_list = RTUtil::getClosedScaleList(search_rect.get_lb_y(), search_rect.get_rt_y(), track_axis.get_y_track_grid());
  for (size_t y_idx = 0; y_idx < y_list.size(); y_idx++) {
    irt_int y = y_list[y_idx];
    if (y == y_list.front() || y == y_list.back()) {
      continue;
    }
    for (irt_int x_idx = 0; x_idx < static_cast<irt_int>(x_list.size()) - 1; x_idx++) {
      segment_list.emplace_back(irt::PlanarCoord(x_list[x_idx], y), PlanarCoord(x_list[x_idx + 1], y));
    }
  }
  for (size_t x_idx = 0; x_idx < x_list.size(); x_idx++) {
    irt_int x = x_list[x_idx];
    if (x == x_list.front() || x == x_list.back()) {
      continue;
    }
    for (irt_int y_idx = 0; y_idx < static_cast<irt_int>(y_list.size()) - 1; y_idx++) {
      segment_list.emplace_back(irt::PlanarCoord(x, y_list[y_idx]), PlanarCoord(x, y_list[y_idx + 1]));
    }
  }

  std::map<PlanarCoord, std::set<Orientation>, CmpPlanarCoordByXASC> grid_obs_map;
  for (Segment<PlanarCoord>& segment : segment_list) {
    PlanarCoord& first_real = segment.get_first();
    PlanarCoord& second_real = segment.get_second();

    if (RTUtil::isOpenOverlap(blockage, RTUtil::getEnlargedRect(segment, half_width))) {
      if (!RTUtil::existGrid(first_real, track_axis) || !RTUtil::existGrid(second_real, track_axis)) {
        LOG_INST.error(Loc::current(), "The coord can not find grid!");
      }
      Orientation orientation = RTUtil::getOrientation(first_real, second_real);
      grid_obs_map[RTUtil::getGridCoord(first_real, track_axis)].insert(orientation);
      grid_obs_map[RTUtil::getGridCoord(second_real, track_axis)].insert(RTUtil::getOppositeOrientation(orientation));
    }
  }
  return grid_obs_map;
}

void TrackAssigner::buildCostTaskMap(TAPanel& ta_panel)
{
  GridMap<TANode>& ta_node_map = ta_panel.get_ta_node_map();

  std::map<irt_int, std::vector<irt_int>> net_task_map;
  for (TATask& ta_task : ta_panel.get_ta_task_list()) {
    net_task_map[ta_task.get_origin_net_idx()].push_back(ta_task.get_task_idx());
  }
  for (auto& [net_idx, region_list] : ta_panel.get_net_region_map()) {
    std::vector<irt_int>& task_idx_list = net_task_map[net_idx];
    for (PlanarRect& region : region_list) {
      for (auto& [grid_coord, orientation_set] : getGridOrientationMap(ta_panel.get_layer_idx(), region)) {
        irt_int local_x = grid_coord.get_x() - ta_panel.get_grid_lb_x();
        irt_int local_y = grid_coord.get_y() - ta_panel.get_grid_lb_y();
        if (!ta_node_map.isInside(local_x, local_y)) {
          continue;
        }
        TANode& ta_node = ta_node_map[local_x][local_y];
        for (Orientation orientation : orientation_set) {
          if (task_idx_list.empty()) {
            ta_node.get_cost_task_map()[orientation].insert(-1);
          } else {
            ta_node.get_cost_task_map()[orientation].insert(task_idx_list.begin(), task_idx_list.end());
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
  for (auto& [net_idx, region_list] : ta_panel.get_net_region_map()) {
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
          LOG_INST.error(Loc::current(), "The neighbor orien is different with real region!");
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
    rerouteByforcing(ta_panel);
    updatePathResult(ta_panel);
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
  if (ta_panel.isForcedRouting()) {
    return true;
  }
  Orientation orientation = getOrientation(start_node, end_node);
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
    if (pre_node->isOBS(ta_panel.get_curr_task_idx(), orientation)
        || curr_node->isOBS(ta_panel.get_curr_task_idx(), opposite_orientation)) {
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
  ta_panel.set_forced_routing(false);

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

void TrackAssigner::rerouteByforcing(TAPanel& ta_panel)
{
  if (isRoutingFailed(ta_panel)) {
    if (omp_get_num_threads() == 1) {
      LOG_INST.warning(Loc::current(), "The task ", ta_panel.get_curr_task_idx(), " forced routing!");
    }
    resetSinglePath(ta_panel);
    ta_panel.set_forced_routing(true);
    routeSinglePath(ta_panel);
    if (isRoutingFailed(ta_panel)) {
      LOG_INST.error(Loc::current(), "The task ", ta_panel.get_curr_task_idx(), " forced routing failed!");
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
    return;
  }
  Orientation curr_orientation = getOrientation(curr_node, pre_node);
  while (pre_node->get_parent_node() != nullptr) {
    Orientation pre_orientation = getOrientation(pre_node, pre_node->get_parent_node());
    if (curr_orientation != pre_orientation) {
      node_segment_list.emplace_back(curr_node, pre_node);
      curr_orientation = pre_orientation;
      curr_node = pre_node;
    }
    pre_node = pre_node->get_parent_node();
  }
  node_segment_list.emplace_back(curr_node, pre_node);
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
  std::vector<Segment<TANode*>>& node_segment_list = ta_panel.get_node_segment_list();

  std::set<TANode*> usage_set;

  for (Segment<TANode*>& node_segment : node_segment_list) {
    TANode* first_node = node_segment.get_first();
    TANode* second_node = node_segment.get_second();
    Orientation orientation = getOrientation(first_node, second_node);

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
    usage_node->addDemand(ta_panel.get_curr_task_idx());
  }
  std::vector<Segment<LayerCoord>>& routing_segment_list = ta_task.get_routing_segment_list();
  for (Segment<TANode*>& node_segment : node_segment_list) {
    routing_segment_list.emplace_back(*node_segment.get_first(), *node_segment.get_second());
  }
}

void TrackAssigner::resetSingleNet(TAPanel& ta_panel)
{
  ta_panel.set_ta_task_ref(nullptr);
  ta_panel.get_start_node_comb_list().clear();
  ta_panel.get_end_node_comb_list().clear();
  ta_panel.get_path_node_comb().clear();
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
  cost += getJointCost(ta_panel, end_node, getOrientation(end_node, start_node));
  cost += getWireCost(ta_panel, start_node, end_node);
  cost += getViaCost(ta_panel, start_node, end_node);
  return cost;
}

double TrackAssigner::getJointCost(TAPanel& ta_panel, TANode* curr_node, Orientation orientation)
{
  const std::map<LayerCoord, double, CmpLayerCoordByXASC>& curr_coord_cost_map = ta_panel.get_curr_coord_cost_map();

  auto iter = curr_coord_cost_map.find(*curr_node);
  double task_cost = 0;
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
  estimate_cost += getWireCost(ta_panel, start_node, end_node);
  estimate_cost += getViaCost(ta_panel, start_node, end_node);
  return estimate_cost;
}

// common

Orientation TrackAssigner::getOrientation(TANode* start_node, TANode* end_node)
{
  Orientation orientation = RTUtil::getOrientation(*start_node, *end_node);
  if (orientation == Orientation::kOblique) {
    LOG_INST.error(Loc::current(), "The segment (", (*start_node).get_x(), ",", (*start_node).get_y(), ",", (*start_node).get_layer_idx(),
                   ")-(", (*end_node).get_x(), ",", (*end_node).get_y(), ",", (*end_node).get_layer_idx(), ") is oblique!");
  }
  return orientation;
}

double TrackAssigner::getWireCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node)
{
  return ta_panel.get_wire_unit() * RTUtil::getManhattanDistance(*start_node, *end_node);
}

double TrackAssigner::getViaCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node)
{
  return ta_panel.get_via_unit() * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx());
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

  irt_int none_data_type = 0;
  irt_int open_data_type = 10;
  irt_int close_data_type = 20;
  irt_int info_data_type = 30;
  irt_int neighbor_data_type = 40;
  irt_int key_data_type = 50;
  irt_int path_data_type = 60;

  GPGDS gp_gds;
  GridMap<TANode>& node_map = ta_panel.get_ta_node_map();

  // node_graph
  GPStruct node_graph_struct("node_graph");
  for (irt_int grid_x = 0; grid_x < node_map.get_x_size(); grid_x++) {
    for (irt_int grid_y = 0; grid_y < node_map.get_y_size(); grid_y++) {
      TANode& ta_node = node_map[grid_x][grid_y];
      PlanarRect real_rect = RTUtil::getEnlargedRect(ta_node.get_planar_coord(), layer_width_map[ta_node.get_layer_idx()]);
      irt_int y_reduced_span = real_rect.getYSpan() / 15;
      irt_int y = real_rect.get_rt_y();

      GPBoundary gp_boundary;
      switch (ta_node.get_state()) {
        case TANodeState::kNone:
          gp_boundary.set_data_type(none_data_type);
          break;
        case TANodeState::kOpen:
          gp_boundary.set_data_type(open_data_type);
          break;
        case TANodeState::kClose:
          gp_boundary.set_data_type(close_data_type);
          break;
        default:
          LOG_INST.error(Loc::current(), "The type is error!");
          break;
      }
      gp_boundary.set_rect(real_rect);
      gp_boundary.set_layer_idx(ta_node.get_layer_idx());
      node_graph_struct.push(gp_boundary);

      y -= y_reduced_span;
      GPText gp_text_node_coord;
      gp_text_node_coord.set_coord(real_rect.get_lb_x(), y);
      gp_text_node_coord.set_text_type(info_data_type);
      gp_text_node_coord.set_message(RTUtil::getString("(", grid_x, " , ", grid_y, " , ", ta_node.get_layer_idx(), ")"));
      gp_text_node_coord.set_layer_idx(ta_node.get_layer_idx());
      gp_text_node_coord.set_presentation(GPTextPresentation::kLeftMiddle);
      node_graph_struct.push(gp_text_node_coord);

      y -= y_reduced_span;
      GPText gp_text_obs_task_map;
      gp_text_obs_task_map.set_coord(real_rect.get_lb_x(), y);
      gp_text_obs_task_map.set_text_type(info_data_type);
      gp_text_obs_task_map.set_message("obs_task_map: ");
      gp_text_obs_task_map.set_layer_idx(ta_node.get_layer_idx());
      gp_text_obs_task_map.set_presentation(GPTextPresentation::kLeftMiddle);
      node_graph_struct.push(gp_text_obs_task_map);

      for (auto& [orientation, task_idx_set] : ta_node.get_obs_task_map()) {
        y -= y_reduced_span;
        GPText gp_text_obs_task_map_info;
        gp_text_obs_task_map_info.set_coord(real_rect.get_lb_x(), y);
        gp_text_obs_task_map_info.set_text_type(info_data_type);
        std::string obs_task_map_info_message = RTUtil::getString("--", GetOrientationName()(orientation), ": ");
        for (irt_int task_idx : task_idx_set) {
          obs_task_map_info_message += RTUtil::getString("(", task_idx, ")");
        }
        gp_text_obs_task_map_info.set_message(obs_task_map_info_message);
        gp_text_obs_task_map_info.set_layer_idx(ta_node.get_layer_idx());
        gp_text_obs_task_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
        node_graph_struct.push(gp_text_obs_task_map_info);
      }

      y -= y_reduced_span;
      GPText gp_text_cost_task_map;
      gp_text_cost_task_map.set_coord(real_rect.get_lb_x(), y);
      gp_text_cost_task_map.set_text_type(info_data_type);
      gp_text_cost_task_map.set_message("cost_task_map: ");
      gp_text_cost_task_map.set_layer_idx(ta_node.get_layer_idx());
      gp_text_cost_task_map.set_presentation(GPTextPresentation::kLeftMiddle);
      node_graph_struct.push(gp_text_cost_task_map);

      for (auto& [orientation, task_idx_set] : ta_node.get_cost_task_map()) {
        y -= y_reduced_span;
        GPText gp_text_cost_task_map_info;
        gp_text_cost_task_map_info.set_coord(real_rect.get_lb_x(), y);
        gp_text_cost_task_map_info.set_text_type(info_data_type);
        std::string cost_task_map_info_message = RTUtil::getString("--", GetOrientationName()(orientation), ": ");
        for (irt_int task_idx : task_idx_set) {
          cost_task_map_info_message += RTUtil::getString("(", task_idx, ")");
        }
        gp_text_cost_task_map_info.set_message(cost_task_map_info_message);
        gp_text_cost_task_map_info.set_layer_idx(ta_node.get_layer_idx());
        gp_text_cost_task_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
        node_graph_struct.push(gp_text_cost_task_map_info);
      }

      y -= y_reduced_span;
      GPText gp_text_task_queue;
      gp_text_task_queue.set_coord(real_rect.get_lb_x(), y);
      gp_text_task_queue.set_text_type(info_data_type);
      gp_text_task_queue.set_message("task_queue: ");
      gp_text_task_queue.set_layer_idx(ta_node.get_layer_idx());
      gp_text_task_queue.set_presentation(GPTextPresentation::kLeftMiddle);
      node_graph_struct.push(gp_text_task_queue);

      if (!ta_node.get_task_queue().empty()) {
        y -= y_reduced_span;
        GPText gp_text_task_queue_info;
        gp_text_task_queue_info.set_coord(real_rect.get_lb_x(), y);
        gp_text_task_queue_info.set_text_type(info_data_type);
        std::string task_queue_info_message = "--";
        for (irt_int task_idx : RTUtil::getListByQueue(ta_node.get_task_queue())) {
          task_queue_info_message += RTUtil::getString("(", task_idx, ")");
        }
        gp_text_task_queue_info.set_message(task_queue_info_message);
        gp_text_task_queue_info.set_layer_idx(ta_node.get_layer_idx());
        gp_text_task_queue_info.set_presentation(GPTextPresentation::kLeftMiddle);
        node_graph_struct.push(gp_text_task_queue_info);
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
        gp_path.set_layer_idx(ta_node.get_layer_idx());
        gp_path.set_width(width);
        gp_path.set_data_type(neighbor_data_type);
        neighbor_map_struct.push(gp_path);
      }
    }
  }
  gp_gds.addStruct(neighbor_map_struct);

  // task
  for (TATask& ta_task : ta_panel.get_ta_task_list()) {
    GPStruct task_struct(RTUtil::getString("task_", ta_task.get_task_idx(), "(net_", ta_task.get_origin_net_idx(), ")"));

    if (curr_task_idx == -1 || ta_task.get_task_idx() == curr_task_idx) {
      for (TAGroup& ta_group : ta_task.get_ta_group_list()) {
        for (LayerCoord& coord : ta_group.get_coord_list()) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(key_data_type);
          gp_boundary.set_rect(RTUtil::getEnlargedRect(coord, layer_width_map[coord.get_layer_idx()]));
          gp_boundary.set_layer_idx(coord.get_layer_idx());
          task_struct.push(gp_boundary);
        }
      }
    }

    for (Segment<LayerCoord>& segment : ta_task.get_routing_segment_list()) {
      LayerCoord first_coord = segment.get_first();
      irt_int first_layer_idx = first_coord.get_layer_idx();
      LayerCoord second_coord = segment.get_second();
      irt_int second_layer_idx = second_coord.get_layer_idx();

      if (first_layer_idx == second_layer_idx) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(path_data_type);
        gp_boundary.set_rect(RTUtil::getEnlargedRect(first_coord, second_coord, layer_width_map[first_layer_idx]));
        gp_boundary.set_layer_idx(first_layer_idx);
        task_struct.push(gp_boundary);
      } else {
        RTUtil::sortASC(first_layer_idx, second_layer_idx);
        for (irt_int layer_idx = first_layer_idx; layer_idx <= second_layer_idx; layer_idx++) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(path_data_type);
          gp_boundary.set_rect(RTUtil::getEnlargedRect(first_coord, layer_width_map[layer_idx]));
          gp_boundary.set_layer_idx(layer_idx);
          task_struct.push(gp_boundary);
        }
      }
    }
    gp_gds.addStruct(task_struct);
  }
  GP_INST.plot(gp_gds, _ta_data_manager.getConfig().temp_directory_path + "ta_model.gds", false, false);
}

#endif

#if 1  // count ta_panel

void TrackAssigner::countTAPanel(TAPanel& ta_panel)
{
  irt_int micron_dbu = _ta_data_manager.getDatabase().get_micron_dbu();
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();

  TAPanelStat& ta_panel_stat = ta_panel.get_ta_panel_stat();

  for (TATask& ta_task : ta_panel.get_ta_task_list()) {
    for (Segment<LayerCoord>& routing_segment : ta_task.get_routing_segment_list()) {
      double wire_length = RTUtil::getManhattanDistance(routing_segment.get_first(), routing_segment.get_second()) / 1.0 / micron_dbu;
      ta_panel_stat.addTotalWireLength(wire_length);
    }
  }

  std::vector<TATask>& ta_task_list = ta_panel.get_ta_task_list();
  std::map<irt_int, std::vector<LayerRect>> task_rect_list_map;
  for (size_t i = 0; i < ta_task_list.size(); i++) {
    task_rect_list_map[i] = convertToRectList(ta_task_list[i].get_routing_segment_list());
  }

  for (size_t i = 0; i < ta_task_list.size(); i++) {
    for (LayerRect& curr_rect : task_rect_list_map[i]) {
      irt_int min_spacing = routing_layer_list[curr_rect.get_layer_idx()].getMinSpacing(curr_rect);
      PlanarRect enlarge_curr_rect = RTUtil::getEnlargedRect(curr_rect, min_spacing);

      for (size_t j = i + 1; j < ta_task_list.size(); j++) {
        for (LayerRect& next_rect : task_rect_list_map[j]) {
          if (curr_rect.get_layer_idx() != next_rect.get_layer_idx() || !RTUtil::isOpenOverlap(enlarge_curr_rect, next_rect)) {
            continue;
          }
          double violation_area = RTUtil::getOverlap(enlarge_curr_rect, next_rect).getArea();
          ta_panel_stat.addNetAndNetViolation(violation_area / (micron_dbu * micron_dbu));
        }
      }
    }
  }

  for (size_t i = 0; i < ta_task_list.size(); i++) {
    for (LayerRect& curr_rect : task_rect_list_map[i]) {
      if (curr_rect.get_layer_idx() != ta_panel.get_layer_idx()) {
        continue;
      }
      irt_int min_spacing = routing_layer_list[curr_rect.get_layer_idx()].getMinSpacing(curr_rect);
      PlanarRect enlarge_curr_rect = RTUtil::getEnlargedRect(curr_rect, min_spacing);

      for (auto& [origin_net_idx, blockage_list] : ta_panel.get_net_blockage_map()) {
        if (ta_task_list[i].get_origin_net_idx() == origin_net_idx) {
          continue;
        }
        for (PlanarRect& blockage : blockage_list) {
          double violation_area = RTUtil::getOverlap(enlarge_curr_rect, blockage).getArea();
          ta_panel_stat.addNetAndObsViolation(violation_area / (micron_dbu * micron_dbu));
        }
      }
    }
  }
}

std::vector<LayerRect> TrackAssigner::convertToRectList(std::vector<Segment<LayerCoord>>& segment_list)
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _ta_data_manager.getDatabase().get_layer_via_master_list();
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();

  std::vector<LayerRect> rect_list;
  for (Segment<LayerCoord>& segment : segment_list) {
    LayerCoord first_coord = segment.get_first();
    LayerCoord second_coord = segment.get_second();
    if (!CmpLayerCoordByLayerASC()(first_coord, second_coord)) {
      std::swap(first_coord, second_coord);
    }

    irt_int first_layer_idx = first_coord.get_layer_idx();
    irt_int second_layer_idx = second_coord.get_layer_idx();
    if (first_layer_idx != second_layer_idx) {
      for (irt_int layer_idx = first_layer_idx; layer_idx < second_layer_idx; layer_idx++) {
        ViaMaster& via_master = layer_via_master_list[layer_idx].front();

        LayerRect& above_enclosure = via_master.get_above_enclosure();
        PlanarRect offset_above_enclosure = RTUtil::getOffsetRect(above_enclosure, first_coord);
        rect_list.emplace_back(offset_above_enclosure, above_enclosure.get_layer_idx());

        LayerRect& below_enclosure = via_master.get_below_enclosure();
        PlanarRect offset_below_enclosure = RTUtil::getOffsetRect(below_enclosure, first_coord);
        rect_list.emplace_back(offset_below_enclosure, below_enclosure.get_layer_idx());
      }
    } else {
      irt_int half_width = routing_layer_list[first_layer_idx].get_min_width() / 2;
      PlanarRect wire_rect = RTUtil::getEnlargedRect(first_coord, second_coord, half_width);
      rect_list.emplace_back(wire_rect, first_layer_idx);
    }
  }
  return rect_list;
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
  std::vector<TAGroup>& ta_group_list = ta_task.get_ta_group_list();
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
  TAModelStat& ta_model_stat = ta_model.get_ta_model_stat();
  std::map<irt_int, double>& layer_wire_length = ta_model_stat.get_layer_wire_length_map();
  std::map<irt_int, double>& layer_net_and_net_violation_area = ta_model_stat.get_layer_net_and_net_violation_area_map();
  std::map<irt_int, double>& layer_net_and_obs_violation_area = ta_model_stat.get_layer_net_and_obs_violation_area_map();

  for (std::vector<TAPanel>& panel_list : ta_model.get_layer_panel_list()) {
    for (TAPanel& panel : panel_list) {
      TAPanelStat& ta_panel_stat = panel.get_ta_panel_stat();
      ta_model_stat.addTotalWireLength(ta_panel_stat.get_total_wire_length());
      ta_model_stat.addTotalNetAndNetViolation(ta_panel_stat.get_net_and_net_violation_area());
      ta_model_stat.addTotalNetAndObsViolation(ta_panel_stat.get_net_and_obs_violation_area());
      layer_wire_length[panel.get_layer_idx()] += ta_panel_stat.get_total_wire_length();
      layer_net_and_net_violation_area[panel.get_layer_idx()] += ta_panel_stat.get_net_and_net_violation_area();
      layer_net_and_obs_violation_area[panel.get_layer_idx()] += ta_panel_stat.get_net_and_obs_violation_area();
    }
  }
}

void TrackAssigner::reportTable(TAModel& ta_model)
{
  std::vector<RoutingLayer>& routing_layer_list = _ta_data_manager.getDatabase().get_routing_layer_list();

  TAModelStat& ta_model_stat = ta_model.get_ta_model_stat();
  double total_wire_length = ta_model_stat.get_total_wire_length();
  double total_net_and_net_violation_area = ta_model_stat.get_total_net_and_net_violation_area();
  double total_net_and_obs_violation_area = ta_model_stat.get_total_net_and_obs_violation_area();

  fort::char_table table;
  table.set_border_style(FT_SOLID_STYLE);

  table << fort::header << "Routing Layer"
        << "Wire Length / um"
        << "Net And Net Violation Area / um^2"
        << "Net And Obs Violation Area / um^2" << fort::endr;

  for (RoutingLayer& routing_layer : routing_layer_list) {
    double layer_wire_length = ta_model_stat.get_layer_wire_length_map()[routing_layer.get_layer_idx()];
    double layer_net_and_net_violation_area = ta_model_stat.get_layer_net_and_net_violation_area_map()[routing_layer.get_layer_idx()];
    double layer_net_and_obs_violation_area = ta_model_stat.get_layer_net_and_obs_violation_area_map()[routing_layer.get_layer_idx()];

    table << routing_layer.get_layer_name()
          << RTUtil::getString(layer_wire_length, "(", RTUtil::getPercentage(layer_wire_length, total_wire_length), "%)")
          << RTUtil::getString(layer_net_and_net_violation_area, "(",
                               RTUtil::getPercentage(layer_net_and_net_violation_area, total_net_and_net_violation_area), "%)")
          << RTUtil::getString(layer_net_and_obs_violation_area, "(",
                               RTUtil::getPercentage(layer_net_and_obs_violation_area, total_net_and_obs_violation_area), "%)")
          << fort::endr;
  }
  table << fort::header << "Total" << total_wire_length << total_net_and_net_violation_area << total_net_and_obs_violation_area
        << fort::endr;

  for (std::string table_str : RTUtil::splitString(table.to_string(), '\n')) {
    LOG_INST.info(Loc::current(), table_str);
  }
}

#endif

}  // namespace irt
