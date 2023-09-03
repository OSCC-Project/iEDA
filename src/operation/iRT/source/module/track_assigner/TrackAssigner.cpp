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

#include "DRCChecker.hpp"
#include "GDSPlotter.hpp"
#include "LayerCoord.hpp"
#include "TAPanel.hpp"

namespace irt {

// public

void TrackAssigner::initInst()
{
  if (_ta_instance == nullptr) {
    _ta_instance = new TrackAssigner();
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

// function

void TrackAssigner::assign(std::vector<Net>& net_list)
{
  Monitor monitor;

  assignNetList(net_list);

  LOG_INST.info(Loc::current(), "The ", GetStageName()(Stage::kTrackAssigner), " completed!", monitor.getStatsInfo());
}

// private

TrackAssigner* TrackAssigner::_ta_instance = nullptr;

void TrackAssigner::assignNetList(std::vector<Net>& net_list)
{
  TAModel ta_model = init(net_list);
  iterative(ta_model);
  update(ta_model);
}

#if 1  // init

TAModel TrackAssigner::init(std::vector<Net>& net_list)
{
  TAModel ta_model = initTAModel(net_list);
  buildTAModel(ta_model);
  return ta_model;
}

TAModel TrackAssigner::initTAModel(std::vector<Net>& net_list)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  Die& die = DM_INST.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  TAModel ta_model;

  std::vector<std::vector<TAPanel>>& layer_panel_list = ta_model.get_layer_panel_list();
  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::vector<TAPanel> ta_panel_list;
    if (routing_layer.isPreferH()) {
      for (ScaleGrid& gcell_grid : gcell_axis.get_y_grid_list()) {
        for (irt_int line = gcell_grid.get_start_line(); line < gcell_grid.get_end_line(); line += gcell_grid.get_step_length()) {
          TAPanel ta_panel;
          ta_panel.set_rect(PlanarRect(die.get_real_lb_x(), line, die.get_real_rt_x(), line + gcell_grid.get_step_length()));
          ta_panel.set_layer_idx(routing_layer.get_layer_idx());

          TAPanelId ta_panel_id;
          ta_panel_id.set_layer_idx(routing_layer.get_layer_idx());
          ta_panel_id.set_panel_idx(static_cast<irt_int>(ta_panel_list.size()));
          ta_panel.set_ta_panel_id(ta_panel_id);

          ta_panel_list.push_back(ta_panel);
        }
      }
    } else {
      for (ScaleGrid& gcell_grid : gcell_axis.get_x_grid_list()) {
        for (irt_int line = gcell_grid.get_start_line(); line < gcell_grid.get_end_line(); line += gcell_grid.get_step_length()) {
          TAPanel ta_panel;
          ta_panel.set_rect(PlanarRect(line, die.get_real_lb_y(), line + gcell_grid.get_step_length(), die.get_real_rt_y()));
          ta_panel.set_layer_idx(routing_layer.get_layer_idx());

          TAPanelId ta_panel_id;
          ta_panel_id.set_layer_idx(routing_layer.get_layer_idx());
          ta_panel_id.set_panel_idx(static_cast<irt_int>(ta_panel_list.size()));
          ta_panel.set_ta_panel_id(ta_panel_id);

          ta_panel_list.push_back(ta_panel);
        }
      }
    }
    layer_panel_list.push_back(ta_panel_list);
  }
  ta_model.set_ta_net_list(convertToTANetList(net_list));

  return ta_model;
}

std::vector<TANet> TrackAssigner::convertToTANetList(std::vector<Net>& net_list)
{
  std::vector<TANet> ta_net_list;
  ta_net_list.reserve(net_list.size());
  for (size_t i = 0; i < net_list.size(); i++) {
    ta_net_list.emplace_back(convertToTANet(net_list[i]));
  }
  return ta_net_list;
}

TANet TrackAssigner::convertToTANet(Net& net)
{
  TANet ta_net;
  ta_net.set_origin_net(&net);
  ta_net.set_net_idx(net.get_net_idx());
  ta_net.set_connect_type(net.get_connect_type());
  for (Pin& pin : net.get_pin_list()) {
    ta_net.get_ta_pin_list().push_back(TAPin(pin));
  }
  ta_net.set_ta_result_tree(net.get_gr_result_tree());
  return ta_net;
}

void TrackAssigner::buildTAModel(TAModel& ta_model)
{
  buildSchedule(ta_model);
  shrinkPanelRegion(ta_model);
  buildPanelTrackAxis(ta_model);
  updateNetFixedRectMap(ta_model);
  updateNetReservedViaMap(ta_model);
  buildTATaskList(ta_model);
  buildNetTaskMap(ta_model);
  // outputTADataset(ta_model);
}

void TrackAssigner::buildSchedule(TAModel& ta_model)
{
  std::vector<std::vector<TAPanel>>& layer_panel_list = ta_model.get_layer_panel_list();

  irt_int range = 2;

  std::vector<std::vector<TAPanelId>> ta_panel_id_list_list;
  for (irt_int layer_idx = 0; layer_idx < static_cast<irt_int>(layer_panel_list.size()); layer_idx++) {
    for (irt_int start_i = 0; start_i < range; start_i++) {
      std::vector<TAPanelId> ta_panel_id_list;
      for (irt_int i = start_i; i < static_cast<irt_int>(layer_panel_list[layer_idx].size()); i += range) {
        ta_panel_id_list.emplace_back(layer_idx, i);
      }
      ta_panel_id_list_list.push_back(ta_panel_id_list);
    }
  }
  ta_model.set_ta_panel_id_list_list(ta_panel_id_list_list);
}

void TrackAssigner::shrinkPanelRegion(TAModel& ta_model)
{
  std::vector<std::vector<TAPanel>>& layer_panel_list = ta_model.get_layer_panel_list();

  std::map<TAPanelId, std::vector<PlanarRect>, CmpTAPanelId> ta_panel_routing_region_map;
  for (TANet& ta_net : ta_model.get_ta_net_list()) {
    for (Segment<TNode<RTNode>*>& segment : RTUtil::getSegListByTree(ta_net.get_ta_result_tree())) {
      TNode<RTNode>* ta_node_node = segment.get_first();
      if (ta_node_node->value().isDRNode()) {
        ta_node_node = segment.get_second();
      }
      Guide& first_guide = ta_node_node->value().get_first_guide();
      Guide& second_guide = ta_node_node->value().get_second_guide();
      PlanarCoord& first_grid_coord = first_guide.get_grid_coord();
      PlanarCoord& second_grid_coord = second_guide.get_grid_coord();
      irt_int ta_layer_idx = first_guide.get_layer_idx();

      TAPanelId ta_panel_id;
      if (RTUtil::isHorizontal(first_grid_coord, second_grid_coord)) {
        ta_panel_id.set_panel_idx(first_grid_coord.get_y());
      } else {
        ta_panel_id.set_panel_idx(first_grid_coord.get_x());
      }
      ta_panel_id.set_layer_idx(ta_layer_idx);

      ta_panel_routing_region_map[ta_panel_id].push_back(RTUtil::getBoundingBox({first_guide.get_rect(), second_guide.get_rect()}));
    }
  }
  for (auto& [ta_panel_id, routing_region_list] : ta_panel_routing_region_map) {
    TAPanel& ta_panel = layer_panel_list[ta_panel_id.get_layer_idx()][ta_panel_id.get_panel_idx()];
    ta_panel.set_rect(RTUtil::getBoundingBox(routing_region_list));
  }
}

void TrackAssigner::buildPanelTrackAxis(TAModel& ta_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  for (std::vector<TAPanel>& ta_panel_list : ta_model.get_layer_panel_list()) {
    for (TAPanel& ta_panel : ta_panel_list) {
      RoutingLayer& routing_layer = routing_layer_list[ta_panel.get_layer_idx()];

      std::vector<irt_int> x_scale_list
          = RTUtil::getClosedScaleList(ta_panel.get_lb_x(), ta_panel.get_rt_x(), routing_layer.getXTrackGridList());
      std::vector<irt_int> y_scale_list
          = RTUtil::getClosedScaleList(ta_panel.get_lb_y(), ta_panel.get_rt_y(), routing_layer.getYTrackGridList());
      ScaleAxis& panel_track_axis = ta_panel.get_panel_track_axis();
      panel_track_axis.set_x_grid_list(RTUtil::makeScaleGridList(x_scale_list));
      panel_track_axis.set_y_grid_list(RTUtil::makeScaleGridList(y_scale_list));
    }
  }
}

void TrackAssigner::updateNetFixedRectMap(TAModel& ta_model)
{
  std::vector<Blockage>& routing_blockage_list = DM_INST.getDatabase().get_routing_blockage_list();

  for (const Blockage& routing_blockage : routing_blockage_list) {
    LayerRect blockage_real_rect(routing_blockage.get_real_rect(), routing_blockage.get_layer_idx());
    updateRectToEnv(ta_model, ChangeType::kAdd, TASourceType::kLayoutShape, TAPanelId(), DRCRect(-1, blockage_real_rect, true));
  }
  for (TANet& ta_net : ta_model.get_ta_net_list()) {
    for (TAPin& ta_pin : ta_net.get_ta_pin_list()) {
      for (const EXTLayerRect& routing_shape : ta_pin.get_routing_shape_list()) {
        LayerRect shape_real_rect(routing_shape.get_real_rect(), routing_shape.get_layer_idx());
        updateRectToEnv(ta_model, ChangeType::kAdd, TASourceType::kLayoutShape, TAPanelId(),
                        DRCRect(ta_net.get_net_idx(), shape_real_rect, true));
      }
    }
  }
}

/**
 * 当drc_rect是由于ta_panel布线产生时，ta_source_type必须设置为kUnknownPanel
 * 当drc_rect是由blockage或pin_shape或其他不由ta_panel布线产生时，ta_source_type可设置为对应值
 */
void TrackAssigner::updateRectToEnv(TAModel& ta_model, ChangeType change_type, TASourceType ta_source_type, TAPanelId ta_panel_id,
                                    DRCRect drc_rect)
{
  if (drc_rect.get_is_routing() == false) {
    return;
  }
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  EXTPlanarRect& die = DM_INST.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  std::vector<std::vector<TAPanel>>& layer_panel_list = ta_model.get_layer_panel_list();

  irt_int routing_layer_idx = drc_rect.get_layer_rect().get_layer_idx();
  for (const LayerRect& max_scope_real_rect : DC_INST.getMaxScope(drc_rect)) {
    LayerRect max_scope_regular_rect = RTUtil::getRegularRect(max_scope_real_rect, die.get_real_rect());
    PlanarRect max_scope_grid_rect = RTUtil::getClosedGridRect(max_scope_regular_rect, gcell_axis);
    if (routing_layer_list[routing_layer_idx].isPreferH()) {
      for (irt_int y = max_scope_grid_rect.get_lb_y(); y <= max_scope_grid_rect.get_rt_y(); y++) {
        TAPanel& curr_panel = layer_panel_list[routing_layer_idx][y];
        DC_INST.updateRectList(curr_panel.getRegionQuery(ta_source_type), change_type, drc_rect);
      }
    } else {
      for (irt_int x = max_scope_grid_rect.get_lb_x(); x <= max_scope_grid_rect.get_rt_x(); x++) {
        TAPanel& curr_panel = layer_panel_list[routing_layer_idx][x];
        DC_INST.updateRectList(curr_panel.getRegionQuery(ta_source_type), change_type, drc_rect);
      }
    }
  }
}

void TrackAssigner::updateNetReservedViaMap(TAModel& ta_model)
{
  irt_int bottom_routing_layer_idx = DM_INST.getConfig().bottom_routing_layer_idx;
  irt_int top_routing_layer_idx = DM_INST.getConfig().top_routing_layer_idx;

  for (TANet& ta_net : ta_model.get_ta_net_list()) {
    std::set<LayerCoord, CmpLayerCoordByXASC> real_coord_set;
    for (TAPin& ta_pin : ta_net.get_ta_pin_list()) {
      for (LayerCoord& real_coord : ta_pin.getRealCoordList()) {
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
        for (DRCRect& drc_rect : DC_INST.getDRCRectList(ta_net.get_net_idx(), segment_list)) {
          updateRectToEnv(ta_model, ChangeType::kAdd, TASourceType::kReservedVia, TAPanelId(), drc_rect);
        }
      }
    }
  }
}

void TrackAssigner::buildTATaskList(TAModel& ta_model)
{
  Monitor monitor;

  std::vector<TANet>& ta_net_list = ta_model.get_ta_net_list();

  irt_int batch_size = RTUtil::getBatchSize(ta_net_list.size());

  Monitor stage_monitor;
  for (size_t i = 0; i < ta_net_list.size(); i++) {
    buildTATask(ta_model, ta_net_list[i]);
    if ((i + 1) % batch_size == 0) {
      LOG_INST.info(Loc::current(), "Extracting task from ", (i + 1), " nets", stage_monitor.getStatsInfo());
    }
  }
  LOG_INST.info(Loc::current(), "Extracting task from ", ta_net_list.size(), " nets", monitor.getStatsInfo());
}

void TrackAssigner::buildTATask(TAModel& ta_model, TANet& ta_net)
{
  std::vector<std::vector<TAPanel>>& layer_panel_list = ta_model.get_layer_panel_list();

  for (auto& [ta_node_node, ta_task] : makeTANodeTaskMap(ta_model, ta_net)) {
    RTNode& ta_node = ta_node_node->value();
    PlanarCoord& first_grid_coord = ta_node.get_first_guide().get_grid_coord();
    PlanarCoord& second_grid_coord = ta_node.get_second_guide().get_grid_coord();
    irt_int layer_idx = ta_node.get_first_guide().get_layer_idx();

    irt_int panel_idx = -1;
    if (RTUtil::isHorizontal(first_grid_coord, second_grid_coord)) {
      panel_idx = first_grid_coord.get_y();
    } else {
      panel_idx = first_grid_coord.get_x();
    }
    TAPanel& ta_panel = layer_panel_list[layer_idx][panel_idx];

    std::vector<TATask>& ta_task_list = ta_panel.get_ta_task_list();
    ta_task.set_origin_net_idx(ta_net.get_net_idx());
    ta_task.set_origin_node(ta_node_node);
    ta_task.set_task_idx(static_cast<irt_int>(ta_task_list.size()));
    ta_task.set_connect_type(ta_net.get_connect_type());
    buildBoundingBox(ta_task);
    ta_task.set_routing_state(RoutingState::kUnrouted);
    ta_task_list.push_back(ta_task);
  }
}

std::map<TNode<RTNode>*, TATask> TrackAssigner::makeTANodeTaskMap(TAModel& ta_model, TANet& ta_net)
{
  /**
   * 构建
   *   std::vector<TAGroup> _ta_group_list;
   *   std::map<LayerCoord, double, CmpLayerCoordByXASC> _coord_cost_map;
   */
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
  // ta_node_task_map
  std::map<TNode<RTNode>*, TATask> ta_node_task_map;
  for (auto& [dr_node_node, ta_node_node_list] : dr_ta_list_map) {
    // pin_coord_list
    std::vector<LayerCoord> pin_coord_list;
    for (irt_int pin_idx : dr_node_node->value().get_pin_idx_set()) {
      pin_coord_list.push_back(ta_net.get_ta_pin_list()[pin_idx].getRealCoordList().front());
    }
    std::map<TNode<RTNode>*, TAGroup> ta_group_map;
    for (TNode<RTNode>* ta_node_node : ta_node_node_list) {
      ta_group_map[ta_node_node] = makeTAGroup(ta_model, dr_node_node, ta_node_node, pin_coord_list);
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
  return ta_node_task_map;
}

TAGroup TrackAssigner::makeTAGroup(TAModel& ta_model, TNode<RTNode>* dr_node_node, TNode<RTNode>* ta_node_node,
                                   std::vector<LayerCoord>& pin_coord_list)
{
  EXTPlanarRect& die = DM_INST.getDatabase().get_die();

  std::vector<std::vector<TAPanel>>& layer_panel_list = ta_model.get_layer_panel_list();

  // dr info
  Guide& dr_guide = dr_node_node->value().get_first_guide();
  PlanarCoord& dr_grid_coord = dr_guide.get_grid_coord();

  // ta info
  PlanarCoord& first_grid_coord = ta_node_node->value().get_first_guide().get_grid_coord();
  PlanarCoord& second_grid_coord = ta_node_node->value().get_second_guide().get_grid_coord();
  irt_int ta_layer_idx = ta_node_node->value().get_first_guide().get_layer_idx();

  // ta_panel
  irt_int panel_idx = -1;
  if (RTUtil::isHorizontal(first_grid_coord, second_grid_coord)) {
    panel_idx = first_grid_coord.get_y();
  } else {
    panel_idx = first_grid_coord.get_x();
  }
  TAPanel& ta_panel = layer_panel_list[ta_layer_idx][panel_idx];
  ScaleAxis& panel_track_axis = ta_panel.get_panel_track_axis();

  Orientation orientation = Orientation::kNone;
  if (dr_grid_coord == first_grid_coord) {
    orientation = RTUtil::getOrientation(dr_grid_coord, second_grid_coord);
  } else {
    orientation = RTUtil::getOrientation(dr_grid_coord, first_grid_coord);
  }
  PlanarRect routing_region = dr_guide;
  if (!pin_coord_list.empty()) {
    routing_region = RTUtil::getBoundingBox(pin_coord_list);
    if (!RTUtil::existGrid(routing_region, panel_track_axis)) {
      routing_region = RTUtil::getTrackRectByEnlarge(routing_region, panel_track_axis, die.get_real_rect());
    }
    routing_region = RTUtil::getEnlargedRect(routing_region, 0, dr_guide);
  }
  std::vector<irt_int> x_list;
  std::vector<irt_int> y_list;
  if (orientation == Orientation::kEast || orientation == Orientation::kWest) {
    x_list = RTUtil::getClosedScaleList(routing_region.get_lb_x(), routing_region.get_rt_x(), panel_track_axis.get_x_grid_list());
    y_list = RTUtil::getClosedScaleList(dr_guide.get_lb_y(), dr_guide.get_rt_y(), panel_track_axis.get_y_grid_list());
    irt_int x = (orientation == Orientation::kEast ? x_list.back() : x_list.front());
    x_list.clear();
    x_list.push_back(x);
  } else if (orientation == Orientation::kNorth || orientation == Orientation::kSouth) {
    x_list = RTUtil::getClosedScaleList(dr_guide.get_lb_x(), dr_guide.get_rt_x(), panel_track_axis.get_x_grid_list());
    y_list = RTUtil::getClosedScaleList(routing_region.get_lb_y(), routing_region.get_rt_y(), panel_track_axis.get_y_grid_list());
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

std::map<LayerCoord, double, CmpLayerCoordByXASC> TrackAssigner::makeTACostMap(TNode<RTNode>* ta_node_node,
                                                                               std::map<TNode<RTNode>*, TAGroup>& ta_group_map,
                                                                               std::vector<LayerCoord>& pin_coord_list)
{
  double ta_pin_distance_unit = DM_INST.getConfig().ta_pin_distance_unit;
  double ta_group_distance_unit = DM_INST.getConfig().ta_group_distance_unit;

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
  double distance_unit;
  if (!pin_coord_list.empty()) {
    distance_unit = ta_pin_distance_unit;
  } else {
    distance_unit = ta_group_distance_unit;
  }
  std::map<LayerCoord, double, CmpLayerCoordByXASC> coord_cost_map;
  for (size_t i = 0; i < coord_distance_pair_list.size(); i++) {
    coord_cost_map[coord_distance_pair_list[i].first] = (i * distance_unit);
  }
  return coord_cost_map;
}

void TrackAssigner::buildBoundingBox(TATask& ta_task)
{
  std::vector<PlanarCoord> coord_list;
  for (TAGroup& ta_group : ta_task.get_ta_group_list()) {
    for (LayerCoord& coord : ta_group.get_coord_list()) {
      coord_list.push_back(coord);
    }
  }
  ta_task.set_bounding_box(RTUtil::getBoundingBox(coord_list));
}

void TrackAssigner::buildNetTaskMap(TAModel& ta_model)
{
  for (std::vector<TAPanel>& ta_panel_list : ta_model.get_layer_panel_list()) {
    for (TAPanel& ta_panel : ta_panel_list) {
      if (ta_panel.get_ta_task_list().empty()) {
        continue;
      }
      std::map<irt_int, std::vector<irt_int>>& net_task_map = ta_panel.get_net_task_map();
      for (TATask& ta_task : ta_panel.get_ta_task_list()) {
        net_task_map[ta_task.get_origin_net_idx()].push_back(ta_task.get_task_idx());
      }
    }
  }
}

void TrackAssigner::outputTADataset(TAModel& ta_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  size_t written_panel_num = 0;
  std::string ta_dataset_path;
  std::ofstream* ta_dataset;

  std::string def_file_path = DM_INST.getHelper().get_def_file_path();
  ta_dataset_path
      = RTUtil::getString(DM_INST.getConfig().ta_temp_directory_path, RTUtil::splitString(def_file_path, '/').back(), ".ta.txt");
  ta_dataset = RTUtil::getOutputFileStream(ta_dataset_path);
  RTUtil::pushStream(ta_dataset, "def_file_path", " ", def_file_path, "\n");

  for (std::vector<TAPanel>& ta_panel_list : ta_model.get_layer_panel_list()) {
    for (TAPanel& ta_panel : ta_panel_list) {
      if (ta_panel.get_ta_task_list().empty()) {
        continue;
      }
      TAPanelId& ta_panel_id = ta_panel.get_ta_panel_id();
      RoutingLayer& routing_layer = routing_layer_list[ta_panel_id.get_layer_idx()];

      RTUtil::pushStream(ta_dataset, "panel", " ", ta_panel_id.get_layer_idx(), " ", ta_panel_id.get_panel_idx());
      RTUtil::pushStream(ta_dataset, " ", ta_panel.get_lb_x(), " ", ta_panel.get_lb_y());
      RTUtil::pushStream(ta_dataset, " ", ta_panel.get_rt_x(), " ", ta_panel.get_rt_y());
      if (routing_layer.isPreferH()) {
        RTUtil::pushStream(ta_dataset, " ", "H", "\n");
      } else {
        RTUtil::pushStream(ta_dataset, " ", "V", "\n");
      }
      RTUtil::pushStream(ta_dataset, "{", "\n");
      // track_list
      RTUtil::pushStream(ta_dataset, "track_list", "\n");
      for (ScaleGrid& x_grid : ta_panel.get_panel_track_axis().get_x_grid_list()) {
        RTUtil::pushStream(ta_dataset, "X", " ", x_grid.get_start_line(), " ", x_grid.get_step_length(), " ", x_grid.get_end_line(), "\n");
      }
      for (ScaleGrid& y_grid : ta_panel.get_panel_track_axis().get_y_grid_list()) {
        RTUtil::pushStream(ta_dataset, "Y", " ", y_grid.get_start_line(), " ", y_grid.get_step_length(), " ", y_grid.get_end_line(), "\n");
      }
      // wire_list
      RTUtil::pushStream(ta_dataset, "wire_list", "\n");
      for (TATask& ta_task : ta_panel.get_ta_task_list()) {
        std::vector<TAGroup>& ta_group_list = ta_task.get_ta_group_list();
        LayerCoord first_coord = ta_group_list.front().get_coord_list().front();
        LayerCoord second_coord = ta_group_list.back().get_coord_list().front();
        if (routing_layer.isPreferH()) {
          first_coord.set_y(0);
          second_coord.set_y(0);
        } else {
          first_coord.set_x(0);
          second_coord.set_x(0);
        }
        irt_int half_width = routing_layer.get_min_width() / 2;
        LayerRect rect(RTUtil::getEnlargedRect(first_coord, second_coord, half_width), ta_panel_id.get_layer_idx());
        RTUtil::pushStream(ta_dataset, ta_task.get_origin_net_idx(), " ", rect.get_lb_x(), " ", rect.get_lb_y(), " ", rect.get_rt_x(), " ",
                           rect.get_rt_y(), "\n");
      }
      // soft_shape_list
      RTUtil::pushStream(ta_dataset, "soft_shape_list", "\n");
      for (const auto& [net_idx, rect_set] :
           DC_INST.getLayerNetRectMap(ta_panel.getRegionQuery(TASourceType::kReservedVia), true)[ta_panel_id.get_layer_idx()]) {
        for (const LayerRect& rect : rect_set) {
          RTUtil::pushStream(ta_dataset, net_idx, " ", rect.get_lb_x(), " ", rect.get_lb_y(), " ", rect.get_rt_x(), " ", rect.get_rt_y(),
                             "\n");
        }
      }
      // hard_shape_list
      RTUtil::pushStream(ta_dataset, "hard_shape_list", "\n");
      for (const auto& [net_idx, rect_set] :
           DC_INST.getLayerNetRectMap(ta_panel.getRegionQuery(TASourceType::kLayoutShape), true)[ta_panel_id.get_layer_idx()]) {
        for (const LayerRect& rect : rect_set) {
          RTUtil::pushStream(ta_dataset, net_idx, " ", rect.get_lb_x(), " ", rect.get_lb_y(), " ", rect.get_rt_x(), " ", rect.get_rt_y(),
                             "\n");
        }
      }
      RTUtil::pushStream(ta_dataset, "}", "\n");
      written_panel_num++;
      if (written_panel_num % 10000 == 0) {
        LOG_INST.info(Loc::current(), "Written ", written_panel_num, " panels");
      }
    }
  }
  LOG_INST.info(Loc::current(), "Written ", written_panel_num, " panels");
  RTUtil::closeFileStream(ta_dataset);
  LOG_INST.info(Loc::current(), "The result has been written to '", ta_dataset_path, "'!");
  exit(0);
}

#endif

#if 1  // iterative

void TrackAssigner::iterative(TAModel& ta_model)
{
  irt_int ta_model_max_iter_num = DM_INST.getConfig().ta_model_max_iter_num;

  for (irt_int iter = 1; iter <= ta_model_max_iter_num; iter++) {
    Monitor iter_monitor;
    LOG_INST.info(Loc::current(), "****** Start Model Iteration(", iter, "/", ta_model_max_iter_num, ") ******");
    ta_model.set_curr_iter(iter);
    assignTAModel(ta_model);
    countTAModel(ta_model);
    reportTAModel(ta_model);
    LOG_INST.info(Loc::current(), "****** End Model Iteration(", iter, "/", ta_model_max_iter_num, ")", iter_monitor.getStatsInfo(),
                  " ******");
    if (stopTAModel(ta_model)) {
      LOG_INST.info(Loc::current(), "****** Reached the stopping condition, ending the iteration prematurely! ******");
      ta_model.set_curr_iter(-1);
      break;
    }
  }
}

void TrackAssigner::assignTAModel(TAModel& ta_model)
{
  Monitor monitor;

  size_t total_panel_num = 0;
  for (std::vector<TAPanelId>& ta_panel_id_list : ta_model.get_ta_panel_id_list_list()) {
    Monitor stage_monitor;
#pragma omp parallel for
    for (TAPanelId& ta_panel_id : ta_panel_id_list) {
      iterativeTAPanel(ta_model, ta_panel_id);
    }
    total_panel_num += ta_panel_id_list.size();
    LOG_INST.info(Loc::current(), "Assigned ", ta_panel_id_list.size(), " panels", stage_monitor.getStatsInfo());
  }
  LOG_INST.info(Loc::current(), "Assigned ", total_panel_num, " panels", monitor.getStatsInfo());
}

void TrackAssigner::iterativeTAPanel(TAModel& ta_model, TAPanelId& ta_panel_id)
{
  irt_int ta_panel_max_iter_num = DM_INST.getConfig().ta_panel_max_iter_num;

  std::vector<std::vector<TAPanel>>& layer_panel_list = ta_model.get_layer_panel_list();
  TAPanel& ta_panel = layer_panel_list[ta_panel_id.get_layer_idx()][ta_panel_id.get_panel_idx()];
  if (ta_panel.get_ta_task_list().empty()) {
    return;
  }
  buildTAPanel(ta_model, ta_panel);
  for (irt_int iter = 1; iter <= ta_panel_max_iter_num; iter++) {
    Monitor iter_monitor;
    if (omp_get_num_threads() == 1) {
      LOG_INST.info(Loc::current(), "****** Start Panel Iteration(", iter, "/", ta_panel_max_iter_num, ") ******");
    }
    ta_panel.set_curr_iter(iter);
    resetTAPanel(ta_model, ta_panel);
    assignTAPanel(ta_model, ta_panel);
    processTAPanel(ta_model, ta_panel);
    countTAPanel(ta_model, ta_panel);
    reportTAPanel(ta_model, ta_panel);
    // plotTAPanel(ta_panel);
    if (omp_get_num_threads() == 1) {
      LOG_INST.info(Loc::current(), "****** End Panel Iteration(", iter, "/", ta_panel_max_iter_num, ")", iter_monitor.getStatsInfo(),
                    " ******");
    }
    if (stopTAPanel(ta_model, ta_panel)) {
      if (omp_get_num_threads() == 1) {
        LOG_INST.info(Loc::current(), "****** Reached the stopping condition, ending the iteration prematurely! ******");
      }
      ta_panel.set_curr_iter(-1);
      break;
    }
  }
  freeTAPanel(ta_model, ta_panel);
}

void TrackAssigner::buildTAPanel(TAModel& ta_model, TAPanel& ta_panel)
{
  initTANodeMap(ta_panel);
  buildNeighborMap(ta_panel);
  buildSourceOrienTaskMap(ta_panel);
  checkTAPanel(ta_panel);
  saveTAPanel(ta_panel);
}

void TrackAssigner::initTANodeMap(TAPanel& ta_panel)
{
  ScaleAxis& panel_track_axis = ta_panel.get_panel_track_axis();

  std::vector<irt_int> x_list = RTUtil::getClosedScaleList(ta_panel.get_lb_x(), ta_panel.get_rt_x(), panel_track_axis.get_x_grid_list());
  std::vector<irt_int> y_list = RTUtil::getClosedScaleList(ta_panel.get_lb_y(), ta_panel.get_rt_y(), panel_track_axis.get_y_grid_list());

  GridMap<TANode>& ta_node_map = ta_panel.get_ta_node_map();
  ta_node_map.init(x_list.size(), y_list.size());
  for (irt_int x = 0; x < ta_node_map.get_x_size(); x++) {
    for (irt_int y = 0; y < ta_node_map.get_y_size(); y++) {
      TANode& ta_node = ta_node_map[x][y];
      ta_node.set_x(x_list[x]);
      ta_node.set_y(y_list[y]);
      ta_node.set_layer_idx(ta_panel.get_layer_idx());
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

void TrackAssigner::buildSourceOrienTaskMap(TAPanel& ta_panel)
{
  for (TASourceType ta_source_type : {TASourceType::kLayoutShape, TASourceType::kReservedVia}) {
    for (bool is_routing : {true, false}) {
      for (auto& [layer_idx, net_rect_map] : DC_INST.getLayerNetRectMap(ta_panel.getRegionQuery(ta_source_type), is_routing)) {
        for (auto& [net_idx, rect_set] : net_rect_map) {
          for (const auto& rect : rect_set) {
            updateRectCostToGraph(ta_panel, ChangeType::kAdd, ta_source_type, DRCRect(net_idx, rect, is_routing));
          }
        }
      }
    }
  }
}

/**
 * 当drc_rect是由于ta_panel布线产生时，ta_source_type必须设置为kSelfPanel
 * 当drc_rect是由blockage或pin_shape或其他不由ta_panel布线产生时，ta_source_type可设置为对应值
 */
void TrackAssigner::updateRectCostToGraph(TAPanel& ta_panel, ChangeType change_type, TASourceType ta_source_type, DRCRect drc_rect)
{
  if (drc_rect.get_is_routing() == false) {
    return;
  }
  GridMap<TANode>& ta_node_map = ta_panel.get_ta_node_map();

  for (auto& [grid_coord, orientation_set] : getGridOrientationMap(ta_panel, drc_rect)) {
    TANode& ta_node = ta_node_map[grid_coord.get_x()][grid_coord.get_y()];
    std::map<Orientation, std::set<irt_int>>& orien_net_map = ta_node.get_source_orien_net_map()[ta_source_type];
    for (Orientation orientation : orientation_set) {
      if (change_type == ChangeType::kAdd) {
        orien_net_map[orientation].insert(drc_rect.get_net_idx());
      } else if (change_type == ChangeType::kDel) {
        orien_net_map[orientation].erase(drc_rect.get_net_idx());
      }
    }
  }
}

std::map<LayerCoord, std::set<Orientation>, CmpLayerCoordByXASC> TrackAssigner::getGridOrientationMap(TAPanel& ta_panel,
                                                                                                      const DRCRect& drc_rect)
{
  // !传入的rect是原始形状
  std::map<LayerCoord, std::set<Orientation>, CmpLayerCoordByXASC> grid_orientation_map;

  ScaleAxis& panel_track_axis = ta_panel.get_panel_track_axis();

  for (LayerRect& min_scope_rect : DC_INST.getMinScope(drc_rect)) {
    for (Segment<LayerCoord>& segment : getSegmentList(ta_panel, min_scope_rect)) {
      LayerCoord first = segment.get_first();
      LayerCoord second = segment.get_second();
      irt_int first_layer_idx = first.get_layer_idx();
      irt_int second_layer_idx = second.get_layer_idx();

      Orientation orientation = RTUtil::getOrientation(first, second);
      if (orientation == Orientation::kOblique || std::abs(first_layer_idx - second_layer_idx) > 1) {
        LOG_INST.error(Loc::current(), "The node segment is illegal!");
      }
      for (LayerRect real_rect : getRealRectList({segment})) {
        if (real_rect.get_layer_idx() != min_scope_rect.get_layer_idx()) {
          continue;
        }
        if (RTUtil::isOpenOverlap(min_scope_rect, real_rect)) {
          LayerCoord first_grid(RTUtil::getGridCoord(first, panel_track_axis), first_layer_idx);
          LayerCoord second_grid(RTUtil::getGridCoord(second, panel_track_axis), second_layer_idx);
          grid_orientation_map[first_grid].insert(orientation);
          grid_orientation_map[second_grid].insert(RTUtil::getOppositeOrientation(orientation));
        }
      }
    }
  }
  return grid_orientation_map;
}

std::vector<Segment<LayerCoord>> TrackAssigner::getSegmentList(TAPanel& ta_panel, LayerRect min_scope_rect)
{
  std::vector<Segment<LayerCoord>> segment_list;

  // 对min_scope_rect使用max(1/2width, 1/2below_enclosure_length)膨胀rect
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  irt_int layer_idx = min_scope_rect.get_layer_idx();
  irt_int enlarge_size = routing_layer_list[layer_idx].get_min_width() / 2;
  LayerRect check_rect(RTUtil::getEnlargedRect(min_scope_rect, enlarge_size), layer_idx);

  ScaleAxis& panel_track_axis = ta_panel.get_panel_track_axis();
  GridMap<TANode>& ta_node_map = ta_panel.get_ta_node_map();

  if (RTUtil::existGrid(check_rect, panel_track_axis)) {
    PlanarRect grid_rect = RTUtil::getGridRect(check_rect, panel_track_axis);
    for (irt_int grid_x = grid_rect.get_lb_x(); grid_x <= grid_rect.get_rt_x(); grid_x++) {
      for (irt_int grid_y = grid_rect.get_lb_y(); grid_y <= grid_rect.get_rt_y(); grid_y++) {
        TANode& node = ta_node_map[grid_x][grid_y];
        for (auto& [orientation, neigbor_ptr] : node.get_neighbor_ptr_map()) {
          TANode node_a = node;
          TANode node_b = *neigbor_ptr;
          RTUtil::swapByCMP(node_a, node_b, CmpLayerCoordByLayerASC());
          segment_list.emplace_back(node_a, node_b);
        }
      }
    }
  }
  std::sort(segment_list.begin(), segment_list.end(), [](Segment<LayerCoord>& a, Segment<LayerCoord>& b) {
    if (a.get_first() != b.get_first()) {
      return CmpLayerCoordByLayerASC()(a.get_first(), b.get_first());
    } else {
      return CmpLayerCoordByLayerASC()(a.get_second(), b.get_second());
    }
  });
  RTUtil::merge(segment_list, [](Segment<LayerCoord>& sentry, Segment<LayerCoord>& soldier) {
    return (sentry.get_first() == soldier.get_first()) && (sentry.get_second() == soldier.get_second());
  });
  return segment_list;
}

std::vector<LayerRect> TrackAssigner::getRealRectList(std::vector<Segment<LayerCoord>> segment_list)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  std::vector<LayerRect> real_rect_list;
  for (Segment<LayerCoord>& segment : segment_list) {
    LayerCoord& first_coord = segment.get_first();
    LayerCoord& second_coord = segment.get_second();

    irt_int first_layer_idx = first_coord.get_layer_idx();
    irt_int second_layer_idx = second_coord.get_layer_idx();
    if (first_layer_idx == second_layer_idx) {
      irt_int half_width = routing_layer_list[first_layer_idx].get_min_width() / 2;
      PlanarRect wire_rect = RTUtil::getEnlargedRect(first_coord, second_coord, half_width);
      real_rect_list.emplace_back(wire_rect, first_layer_idx);
    } else {
      LOG_INST.error(Loc::current(), "The segment is proximal!");
    }
  }
  return real_rect_list;
}

void TrackAssigner::checkTAPanel(TAPanel& ta_panel)
{
  if (ta_panel.get_ta_panel_id().get_layer_idx() < 0 || ta_panel.get_ta_panel_id().get_panel_idx() < 0) {
    LOG_INST.error(Loc::current(), "The ta_panel_id is error!");
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
        if (RTUtil::isInside(ta_panel.get_rect(), coord)) {
          continue;
        }
        LOG_INST.error(Loc::current(), "The coord(", coord.get_x(), ",", coord.get_y(), ",", coord.get_layer_idx(),
                       ") is outside the panel!");
      }
    }
  }
}

void TrackAssigner::saveTAPanel(TAPanel& ta_panel)
{
}

void TrackAssigner::resetTAPanel(TAModel& ta_model, TAPanel& ta_panel)
{
  if (ta_panel.get_curr_iter() == 1) {
    sortTAPanel(ta_model, ta_panel);
  } else {
    resortTAPanel(ta_panel);
    addHistoryCost(ta_panel);
    ripupTAPanel(ta_model, ta_panel);
  }
}

void TrackAssigner::sortTAPanel(TAModel& ta_model, TAPanel& ta_panel)
{
  if (ta_panel.get_curr_iter() != 1) {
    return;
  }
  Monitor monitor;
  if (omp_get_num_threads() == 1) {
    LOG_INST.info(Loc::current(), "Sorting all tasks beginning...");
  }

  std::vector<irt_int> task_order_list;
  for (TATask& ta_task : ta_panel.get_ta_task_list()) {
    task_order_list.push_back(ta_task.get_task_idx());
  }
  std::sort(task_order_list.begin(), task_order_list.end(),
            [&](irt_int task_idx1, irt_int task_idx2) { return sortByMultiLevel(ta_panel, task_idx1, task_idx2); });
  ta_panel.get_task_order_list_list().push_back(task_order_list);

  if (omp_get_num_threads() == 1) {
    LOG_INST.info(Loc::current(), "Sorting all tasks completed!", monitor.getStatsInfo());
  }
}

bool TrackAssigner::sortByMultiLevel(TAPanel& ta_panel, irt_int task_idx1, irt_int task_idx2)
{
  TATask& task1 = ta_panel.get_ta_task_list()[task_idx1];
  TATask& task2 = ta_panel.get_ta_task_list()[task_idx2];

  SortStatus sort_status = SortStatus::kNone;

  sort_status = sortByClockPriority(task1, task2);
  if (sort_status == SortStatus::kTrue) {
    return true;
  } else if (sort_status == SortStatus::kFalse) {
    return false;
  }
  sort_status = sortByPreferLengthDESC(task1, task2);
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
  ConnectType task1_connect_type = task1.get_connect_type();
  ConnectType task2_connect_type = task2.get_connect_type();

  if (task1_connect_type == ConnectType::kClock && task2_connect_type != ConnectType::kClock) {
    return SortStatus::kTrue;
  } else if (task1_connect_type != ConnectType::kClock && task2_connect_type == ConnectType::kClock) {
    return SortStatus::kFalse;
  } else {
    return SortStatus::kEqual;
  }
}

// ta的宽都一样，prefer方向长度 降序 越长的越先布
SortStatus TrackAssigner::sortByPreferLengthDESC(TATask& task1, TATask& task2)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  irt_int task1_layer_idx = task1.get_origin_node()->value().get_first_guide().get_layer_idx();
  irt_int task2_layer_idx = task2.get_origin_node()->value().get_first_guide().get_layer_idx();
  if (task1_layer_idx != task2_layer_idx) {
    LOG_INST.error(Loc::current(), "The task1_layer_idx != task2_layer_idx!");
  }

  irt_int task1_prefer_length = 0;
  irt_int task2_prefer_length = 0;
  if (routing_layer_list[task1_layer_idx].isPreferH()) {
    task1_prefer_length = task1.get_bounding_box().getXSpan();
    task2_prefer_length = task2.get_bounding_box().getXSpan();
  } else {
    task1_prefer_length = task1.get_bounding_box().getYSpan();
    task2_prefer_length = task2.get_bounding_box().getYSpan();
  }
  if (task1_prefer_length > task2_prefer_length) {
    return SortStatus::kTrue;
  } else if (task1_prefer_length == task2_prefer_length) {
    return SortStatus::kEqual;
  } else {
    return SortStatus::kFalse;
  }
}

void TrackAssigner::resortTAPanel(TAPanel& ta_panel)
{
#if 1
  std::vector<std::vector<irt_int>>& task_order_list_list = ta_panel.get_task_order_list_list();
  std::vector<irt_int>& last_task_order_list = task_order_list_list.back();

  // 确定拆线重布任务并换序
  std::map<irt_int, irt_int> task_order_map;
  for (size_t i = 0; i < last_task_order_list.size(); i++) {
    task_order_map[last_task_order_list[i]] = i;
  }

  std::vector<irt_int> ripup_task_list;
  std::set<irt_int> ripup_task_set;
  for (std::vector<irt_int> violation_task_comb : getViolationTaskCombList(ta_panel)) {
    std::sort(violation_task_comb.begin(), violation_task_comb.end(),
              [&task_order_map](irt_int a, irt_int b) { return task_order_map[a] > task_order_map[b]; });
    for (irt_int violation_task : violation_task_comb) {
      if (!RTUtil::exist(ripup_task_set, violation_task)) {
        ripup_task_list.push_back(violation_task);
        ripup_task_set.insert(violation_task);
      }
    }
  }
  // 生成新的布线顺序
  std::vector<irt_int> new_task_order_list;
  for (irt_int task : last_task_order_list) {
    if (!RTUtil::exist(ripup_task_set, task)) {
      new_task_order_list.push_back(task);
    }
  }
  new_task_order_list.insert(new_task_order_list.end(), ripup_task_list.begin(), ripup_task_list.end());

  task_order_list_list.push_back(new_task_order_list);
#else
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<std::vector<irt_int>>& task_order_list_list = ta_panel.get_task_order_list_list();
  std::vector<irt_int>& last_task_order_list = task_order_list_list.back();
  std::vector<TATask>& ta_task_list = ta_panel.get_ta_task_list();

  std::map<irt_int, irt_int> task_order_map;
  for (size_t i = 0; i < last_task_order_list.size(); i++) {
    task_order_map[last_task_order_list[i]] = i;
  }

  irt_int pitch = routing_layer_list[ta_panel.get_layer_idx()].getPreferTrackGrid().get_step_length();
  irt_int max_iter_num = DM_INST.getConfig().ta_panel_max_iter_num;
  irt_int iter_num = 0;

  std::vector<irt_int> ripup_task_list;
  std::vector<irt_int> new_task_order_list;
  while (iter_num <= max_iter_num && ripup_task_list.size() != ta_task_list.size()) {
    // 扩大拆线重布任务规模
    std::vector<std::vector<irt_int>> violation_task_list_list;
    for (auto& [source, drc_violation_map] : ta_panel.get_ta_panel_stat().get_source_drc_violation_map()) {
      for (auto& [drc, violation_info_list] : drc_violation_map) {
        for (ViolationInfo& violation_info : violation_info_list) {
          plotTAPanel(ta_panel);
          LayerRect& violation_region = violation_info.get_violation_region();
          PlanarRect enlarge_rect = RTUtil::getEnlargedRect(violation_region, pitch * (iter_num++));
          std::vector<irt_int> violation_task_list;
          for (TATask& ta_task : ta_task_list) {
            if (RTUtil::isOpenOverlap(enlarge_rect, ta_task.get_bounding_box())) {
              violation_task_list.push_back(ta_task.get_task_idx());
            }
          }
          violation_task_list_list.push_back(violation_task_list);
        }
      }
    }
    // 确定拆线重布任务
    ripup_task_list.clear();
    std::set<irt_int> ripup_task_set;
    for (std::vector<irt_int>& violation_task_list : violation_task_list_list) {
      std::sort(violation_task_list.begin(), violation_task_list.end(),
                [&task_order_map](irt_int a, irt_int b) { return task_order_map[a] > task_order_map[b]; });
      for (irt_int violation_task : violation_task_list) {
        if (!RTUtil::exist(ripup_task_set, violation_task)) {
          ripup_task_list.push_back(violation_task);
          ripup_task_set.insert(violation_task);
        }
      }
    }
    // 生成新的布线顺序
    new_task_order_list.clear();
    for (irt_int task : last_task_order_list) {
      if (!RTUtil::exist(ripup_task_set, task)) {
        new_task_order_list.push_back(task);
      }
    }
    new_task_order_list.insert(new_task_order_list.end(), ripup_task_list.begin(), ripup_task_list.end());
    // check new_task_order_list
    std::set<irt_int> new_task_order_set(new_task_order_list.begin(), new_task_order_list.end());
    if (new_task_order_set.size() != last_task_order_list.size()) {
      LOG_INST.error(Loc::current(), "The size of new task order is error!");
    }
    // 是否为新的布线顺序
    if (std::find(task_order_list_list.begin(), task_order_list_list.end(), new_task_order_list) == task_order_list_list.end()) {
      break;
    }
  }

  for (irt_int task : ripup_task_list) {
    ta_task_list[task].set_routing_state(RoutingState::kUnrouted);
  }
  task_order_list_list.push_back(new_task_order_list);
#endif
}

std::vector<std::vector<irt_int>> TrackAssigner::getViolationTaskCombList(TAPanel& ta_panel)
{
  std::map<irt_int, std::vector<irt_int>>& net_task_map = ta_panel.get_net_task_map();

  std::vector<std::vector<irt_int>> violation_task_comb_list;
  for (auto& [source, drc_violation_map] : ta_panel.get_ta_panel_stat().get_source_drc_violation_map()) {
    for (auto& [drc, violation_info_list] : drc_violation_map) {
      for (ViolationInfo& violation_info : violation_info_list) {
        for (auto& [net_idx, rect_list] : violation_info.get_net_shape_map()) {
          if (!RTUtil::exist(net_task_map, net_idx)) {
            continue;
          }
          violation_task_comb_list.push_back(net_task_map[net_idx]);
        }
      }
    }
  }
  return violation_task_comb_list;
}

void TrackAssigner::addHistoryCost(TAPanel& ta_panel)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  for (auto& [source, drc_violation_map] : ta_panel.get_ta_panel_stat().get_source_drc_violation_map()) {
    for (auto& [drc, violation_info_list] : drc_violation_map) {
      for (ViolationInfo& violation_info : violation_info_list) {
        LayerRect& violation_region = violation_info.get_violation_region();
        irt_int layer_idx = violation_region.get_layer_idx();
        irt_int enlarge_size = routing_layer_list[layer_idx].getPreferTrackGrid().get_step_length();
        LayerRect enlarge_real_rect(RTUtil::getEnlargedRect(violation_region, enlarge_size), layer_idx);
        updateHistoryCostToGraph(ta_panel, ChangeType::kAdd, DRCRect(-1, enlarge_real_rect, violation_info.get_is_routing()));
      }
    }
  }
}

void TrackAssigner::updateHistoryCostToGraph(TAPanel& ta_panel, ChangeType change_type, DRCRect drc_rect)
{
  if (drc_rect.get_is_routing() == false) {
    return;
  }

  double ta_history_cost_unit = DM_INST.getConfig().ta_history_cost_unit;

  GridMap<TANode>& ta_node_map = ta_panel.get_ta_node_map();

  for (auto& [grid_coord, orientation_set] : getGridOrientationMap(ta_panel, drc_rect)) {
    TANode& ta_node = ta_node_map[grid_coord.get_x()][grid_coord.get_y()];
    std::map<Orientation, double>& orien_history_cost_map = ta_node.get_orien_history_cost_map();
    for (Orientation orientation : orientation_set) {
      if (!RTUtil::exist(orien_history_cost_map, orientation)) {
        orien_history_cost_map[orientation] = 0;
      }
      if (change_type == ChangeType::kAdd) {
        orien_history_cost_map[orientation] += ta_history_cost_unit;
      } else if (change_type == ChangeType::kDel) {
        orien_history_cost_map[orientation] -= ta_history_cost_unit;
      }
    }
  }
}

void TrackAssigner::ripupTAPanel(TAModel& ta_model, TAPanel& ta_panel)
{
  std::vector<TATask>& ta_task_list = ta_panel.get_ta_task_list();

  for (std::vector<irt_int> violation_task_comb : getViolationTaskCombList(ta_panel)) {
    for (irt_int violation_task : violation_task_comb) {
      ta_task_list[violation_task].set_routing_state(RoutingState::kUnrouted);
    }
  }

  for (TATask& ta_task : ta_task_list) {
    if (ta_task.get_routing_state() == RoutingState::kRouted) {
      continue;
    }
    // 将env中的布线结果清空
    for (DRCRect& drc_rect : DC_INST.getDRCRectList(ta_task.get_origin_net_idx(), ta_task.get_routing_tree())) {
      updateRectToEnv(ta_model, ChangeType::kDel, TASourceType::kLayoutShape, ta_panel.get_ta_panel_id(), drc_rect);
    }
    // 将graph中的布线结果清空
    for (DRCRect& drc_rect : DC_INST.getDRCRectList(ta_task.get_origin_net_idx(), ta_task.get_routing_tree())) {
      updateRectCostToGraph(ta_panel, ChangeType::kDel, TASourceType::kLayoutShape, drc_rect);
    }
    // 清空routing_tree
    ta_task.get_routing_tree().clear();
  }
}

void TrackAssigner::assignTAPanel(TAModel& ta_model, TAPanel& ta_panel)
{
  Monitor monitor;

  std::vector<TATask>& ta_task_list = ta_panel.get_ta_task_list();
  if (ta_panel.get_task_order_list_list().empty()) {
    LOG_INST.error(Loc::current(), "The task_order_list_list is empty!");
  }
  std::vector<irt_int>& task_order_list = ta_panel.get_task_order_list_list().back();

  irt_int batch_size = RTUtil::getBatchSize(ta_task_list.size());

  Monitor stage_monitor;
  for (size_t i = 0; i < task_order_list.size(); i++) {
    assignTATask(ta_model, ta_panel, ta_task_list[task_order_list[i]]);
    if (omp_get_num_threads() == 1 && (i + 1) % batch_size == 0) {
      LOG_INST.info(Loc::current(), "Assigned ", (i + 1), " tasks", stage_monitor.getStatsInfo());
    }
  }
  if (omp_get_num_threads() == 1) {
    LOG_INST.info(Loc::current(), "Assigned ", ta_task_list.size(), " tasks", monitor.getStatsInfo());
  }
}

void TrackAssigner::assignTATask(TAModel& ta_model, TAPanel& ta_panel, TATask& ta_task)
{
  if (ta_task.get_routing_state() == RoutingState::kRouted) {
    return;
  }
  initSingleTask(ta_panel, ta_task);
  while (!isConnectedAllEnd(ta_panel)) {
    routeSinglePath(ta_panel);
    updatePathResult(ta_panel);
    updateDirectionSet(ta_panel);
    resetStartAndEnd(ta_panel);
    resetSinglePath(ta_panel);
  }
  updateTaskResult(ta_model, ta_panel, ta_task);
  resetSingleTask(ta_panel);
}

void TrackAssigner::initSingleTask(TAPanel& ta_panel, TATask& ta_task)
{
  ScaleAxis& panel_track_axis = ta_panel.get_panel_track_axis();
  GridMap<TANode>& ta_node_map = ta_panel.get_ta_node_map();
  std::vector<std::vector<TANode*>>& start_node_list_list = ta_panel.get_start_node_list_list();
  std::vector<std::vector<TANode*>>& end_node_list_list = ta_panel.get_end_node_list_list();

  // single task
  ta_panel.set_ta_task_ref(&ta_task);
  ta_panel.set_routing_region(ta_panel.get_curr_bounding_box());
  {
    std::vector<std::vector<TANode*>> node_list_list;
    std::vector<TAGroup>& ta_group_list = ta_task.get_ta_group_list();
    for (TAGroup& ta_group : ta_group_list) {
      std::vector<TANode*> node_comb;
      for (LayerCoord& coord : ta_group.get_coord_list()) {
        if (!RTUtil::existGrid(coord, panel_track_axis)) {
          LOG_INST.error(Loc::current(), "The coord can not find grid!");
        }
        PlanarCoord grid_coord = RTUtil::getGridCoord(coord, panel_track_axis);
        node_comb.push_back(&ta_node_map[grid_coord.get_x()][grid_coord.get_y()]);
      }
      node_list_list.push_back(node_comb);
    }
    for (size_t i = 0; i < node_list_list.size(); i++) {
      if (i == 0) {
        start_node_list_list.push_back(node_list_list[i]);
      } else {
        end_node_list_list.push_back(node_list_list[i]);
      }
    }
  }
  {
    std::set<Orientation>& routing_offset_set = ta_panel.get_routing_offset_set();
    for (std::vector<TANode*>& start_node_comb : start_node_list_list) {
      for (TANode* start_node : start_node_comb) {
        for (std::vector<TANode*>& end_node_comb : end_node_list_list) {
          for (TANode* end_node : end_node_comb) {
            std::vector<Orientation> orientation_list = RTUtil::getOrientationList(*start_node, *end_node);
            routing_offset_set.insert(orientation_list.begin(), orientation_list.end());
          }
        }
      }
    }
    routing_offset_set.erase(Orientation::kNone);
  }
}

bool TrackAssigner::isConnectedAllEnd(TAPanel& ta_panel)
{
  return ta_panel.get_end_node_list_list().empty();
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
  std::vector<std::vector<TANode*>>& start_node_list_list = ta_panel.get_start_node_list_list();
  std::vector<TANode*>& path_node_list = ta_panel.get_path_node_list();

  for (std::vector<TANode*>& start_node_comb : start_node_list_list) {
    for (TANode* start_node : start_node_comb) {
      start_node->set_estimated_cost(getEstimateCostToEnd(ta_panel, start_node));
      pushToOpenList(ta_panel, start_node);
    }
  }
  for (TANode* path_node : path_node_list) {
    path_node->set_estimated_cost(getEstimateCostToEnd(ta_panel, path_node));
    pushToOpenList(ta_panel, path_node);
  }
  resetPathHead(ta_panel);
}

bool TrackAssigner::searchEnded(TAPanel& ta_panel)
{
  std::vector<std::vector<TANode*>>& end_node_list_list = ta_panel.get_end_node_list_list();
  TANode* path_head_node = ta_panel.get_path_head_node();

  if (path_head_node == nullptr) {
    ta_panel.set_end_node_comb_idx(-1);
    return true;
  }
  for (size_t i = 0; i < end_node_list_list.size(); i++) {
    for (TANode* end_node : end_node_list_list[i]) {
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
    if (!RTUtil::exist(ta_panel.get_routing_offset_set(), orientation)) {
      continue;
    }
    if (!RTUtil::isInside(ta_panel.get_routing_region(), *neighbor_node)) {
      continue;
    }
    if (neighbor_node->isClose()) {
      continue;
    }
    double know_cost = getKnowCost(ta_panel, path_head_node, neighbor_node);
    if (neighbor_node->isOpen() && know_cost < neighbor_node->get_known_cost()) {
      neighbor_node->set_known_cost(know_cost);
      neighbor_node->set_parent_node(path_head_node);
    } else if (neighbor_node->isNone()) {
      neighbor_node->set_known_cost(know_cost);
      neighbor_node->set_parent_node(path_head_node);
      neighbor_node->set_estimated_cost(getEstimateCostToEnd(ta_panel, neighbor_node));
      pushToOpenList(ta_panel, neighbor_node);
    }
  }
}

std::vector<Segment<LayerCoord>> TrackAssigner::getRoutingSegmentListByNode(TANode* node)
{
  std::vector<Segment<LayerCoord>> routing_segment_list;

  TANode* curr_node = node;
  TANode* pre_node = curr_node->get_parent_node();

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

  std::vector<TANode*>& single_path_visited_node_list = ta_panel.get_single_path_visited_node_list();
  for (TANode* single_path_visited_node : single_path_visited_node_list) {
    single_path_visited_node->set_state(TANodeState::kNone);
    single_path_visited_node->set_parent_node(nullptr);
    single_path_visited_node->set_known_cost(0);
    single_path_visited_node->set_estimated_cost(0);
  }
  single_path_visited_node_list.clear();

  ta_panel.set_path_head_node(nullptr);
  ta_panel.set_end_node_comb_idx(-1);
}

void TrackAssigner::updatePathResult(TAPanel& ta_panel)
{
  for (Segment<LayerCoord>& routing_segment : getRoutingSegmentListByNode(ta_panel.get_path_head_node())) {
    ta_panel.get_routing_segment_list().push_back(routing_segment);
  }
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
  std::vector<std::vector<TANode*>>& start_node_list_list = ta_panel.get_start_node_list_list();
  std::vector<std::vector<TANode*>>& end_node_list_list = ta_panel.get_end_node_list_list();
  std::vector<TANode*>& path_node_list = ta_panel.get_path_node_list();
  TANode* path_head_node = ta_panel.get_path_head_node();
  irt_int end_node_comb_idx = ta_panel.get_end_node_comb_idx();

  end_node_list_list[end_node_comb_idx].clear();
  end_node_list_list[end_node_comb_idx].push_back(path_head_node);

  TANode* path_node = path_head_node->get_parent_node();
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

void TrackAssigner::updateTaskResult(TAModel& ta_model, TAPanel& ta_panel, TATask& ta_task)
{
  // 添加routing_tree
  std::vector<LayerCoord> driving_grid_coord_list;
  std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC> key_coord_pin_map;
  std::vector<TAGroup>& ta_group_list = ta_task.get_ta_group_list();
  for (size_t i = 0; i < ta_group_list.size(); i++) {
    for (LayerCoord& coord : ta_group_list[i].get_coord_list()) {
      driving_grid_coord_list.push_back(coord);
      key_coord_pin_map[coord].insert(static_cast<irt_int>(i));
    }
  }
  std::vector<Segment<LayerCoord>> routing_segment_list = ta_panel.get_routing_segment_list();
  ta_task.set_routing_tree(RTUtil::getTreeByFullFlow(driving_grid_coord_list, routing_segment_list, key_coord_pin_map));
  // 将布线结果添加到env中
  for (DRCRect& drc_rect : DC_INST.getDRCRectList(ta_task.get_origin_net_idx(), ta_task.get_routing_tree())) {
    updateRectToEnv(ta_model, ChangeType::kAdd, TASourceType::kLayoutShape, ta_panel.get_ta_panel_id(), drc_rect);
  }
  // 将布线结果添加到graph中
  for (DRCRect& drc_rect : DC_INST.getDRCRectList(ta_task.get_origin_net_idx(), ta_task.get_routing_tree())) {
    updateRectCostToGraph(ta_panel, ChangeType::kAdd, TASourceType::kLayoutShape, drc_rect);
  }
  ta_task.set_routing_state(RoutingState::kRouted);
}

void TrackAssigner::resetSingleTask(TAPanel& ta_panel)
{
  ta_panel.set_ta_task_ref(nullptr);
  ta_panel.get_start_node_list_list().clear();
  ta_panel.get_end_node_list_list().clear();
  ta_panel.get_routing_offset_set().clear();
  ta_panel.get_path_node_list().clear();

  std::vector<TANode*>& single_task_visited_node_list = ta_panel.get_single_task_visited_node_list();
  for (TANode* single_task_visited_node : single_task_visited_node_list) {
    single_task_visited_node->get_direction_set().clear();
  }
  single_task_visited_node_list.clear();

  ta_panel.get_routing_segment_list().clear();
}

// manager open list

void TrackAssigner::pushToOpenList(TAPanel& ta_panel, TANode* curr_node)
{
  std::priority_queue<TANode*, std::vector<TANode*>, CmpTANodeCost>& open_queue = ta_panel.get_open_queue();
  std::vector<TANode*>& single_task_visited_node_list = ta_panel.get_single_task_visited_node_list();
  std::vector<TANode*>& single_path_visited_node_list = ta_panel.get_single_path_visited_node_list();

  open_queue.push(curr_node);
  curr_node->set_state(TANodeState::kOpen);
  single_task_visited_node_list.push_back(curr_node);
  single_path_visited_node_list.push_back(curr_node);
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
  cost += getNodeCost(ta_panel, start_node, RTUtil::getOrientation(*start_node, *end_node));
  cost += getNodeCost(ta_panel, end_node, RTUtil::getOrientation(*end_node, *start_node));
  cost += getKnowWireCost(ta_panel, start_node, end_node);
  cost += getKnowCornerCost(ta_panel, start_node, end_node);
  return cost;
}

double TrackAssigner::getNodeCost(TAPanel& ta_panel, TANode* curr_node, Orientation orientation)
{
  double env_cost = curr_node->getCost(ta_panel.get_curr_net_idx(), orientation);

  double task_cost = 0;
  LayerCoord node_coord = *curr_node;
  if (RTUtil::exist(ta_panel.get_curr_coord_cost_map(), node_coord)) {
    task_cost += ta_panel.get_curr_coord_cost_map().at(node_coord);
  }

  double node_cost = env_cost + task_cost;
  return node_cost;
}

double TrackAssigner::getKnowWireCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  double ta_prefer_wire_unit = DM_INST.getConfig().ta_prefer_wire_unit;
  double ta_nonprefer_wire_unit = DM_INST.getConfig().ta_nonprefer_wire_unit;

  double wire_cost = 0;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    wire_cost += RTUtil::getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());

    RoutingLayer& routing_layer = routing_layer_list[start_node->get_layer_idx()];
    if (routing_layer.get_direction() == RTUtil::getDirection(*start_node, *end_node)) {
      wire_cost *= ta_prefer_wire_unit;
    } else {
      wire_cost *= ta_nonprefer_wire_unit;
    }
  }
  return wire_cost;
}

double TrackAssigner::getKnowCornerCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node)
{
  double ta_corner_unit = DM_INST.getConfig().ta_corner_unit;

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
      corner_cost += ta_corner_unit;
    } else if (direction_set.size() == 2) {
      LOG_INST.error(Loc::current(), "Direction set is error!");
    }
  }
  return corner_cost;
}

// calculate estimate cost

double TrackAssigner::getEstimateCostToEnd(TAPanel& ta_panel, TANode* curr_node)
{
  std::vector<std::vector<TANode*>>& end_node_list_list = ta_panel.get_end_node_list_list();

  double estimate_cost = DBL_MAX;
  for (std::vector<TANode*>& end_node_comb : end_node_list_list) {
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
  return estimate_cost;
}

double TrackAssigner::getEstimateWireCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node)
{
  double ta_prefer_wire_unit = DM_INST.getConfig().ta_prefer_wire_unit;

  double wire_cost = 0;
  wire_cost += RTUtil::getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());
  wire_cost *= ta_prefer_wire_unit;
  return wire_cost;
}

double TrackAssigner::getEstimateCornerCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node)
{
  double ta_corner_unit = DM_INST.getConfig().ta_corner_unit;

  double corner_cost = 0;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    if (RTUtil::isOblique(*start_node, *end_node)) {
      corner_cost += ta_corner_unit;
    }
  }
  return corner_cost;
}

void TrackAssigner::processTAPanel(TAModel& ta_model, TAPanel& ta_panel)
{
  // 检查布线状态
  for (TATask& ta_task : ta_panel.get_ta_task_list()) {
    if (ta_task.get_routing_state() == RoutingState::kUnrouted) {
      LOG_INST.error(Loc::current(), "The routing_state is ", GetRoutingStateName()(ta_task.get_routing_state()), "!");
    }
  }
#pragma omp parallel for
  for (TATask& ta_task : ta_panel.get_ta_task_list()) {
    buildRoutingResult(ta_task);
  }
}

void TrackAssigner::buildRoutingResult(TATask& ta_task)
{
  RTNode& rt_node = ta_task.get_origin_node()->value();
  rt_node.set_routing_tree(ta_task.get_routing_tree());
}

void TrackAssigner::countTAPanel(TAModel& ta_model, TAPanel& ta_panel)
{
  irt_int micron_dbu = DM_INST.getDatabase().get_micron_dbu();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  TAPanelStat ta_panel_stat;

  double total_wire_length = 0;
  double total_prefer_wire_length = 0;
  double total_nonprefer_wire_length = 0;

  for (TATask& ta_task : ta_panel.get_ta_task_list()) {
    for (Segment<TNode<LayerCoord>*>& coord_segment : RTUtil::getSegListByTree(ta_task.get_origin_node()->value().get_routing_tree())) {
      LayerCoord& first = coord_segment.get_first()->value();
      LayerCoord& second = coord_segment.get_second()->value();
      if (first.get_layer_idx() != second.get_layer_idx()) {
        LOG_INST.error(Loc::current(), "The layer of ta segment is different!");
      }
      irt_int distance = RTUtil::getManhattanDistance(first, second) / 1.0 / micron_dbu;
      if (RTUtil::getDirection(first, second) == routing_layer_list[ta_panel.get_layer_idx()].get_direction()) {
        total_prefer_wire_length += distance;
      } else {
        total_nonprefer_wire_length += distance;
      }
      total_wire_length += distance;
    }
  }
  ta_panel_stat.set_total_wire_length(total_wire_length);
  ta_panel_stat.set_total_prefer_wire_length(total_prefer_wire_length);
  ta_panel_stat.set_total_nonprefer_wire_length(total_nonprefer_wire_length);

  std::map<TASourceType, std::map<std::string, std::vector<ViolationInfo>>>& source_drc_violation_map
      = ta_panel_stat.get_source_drc_violation_map();
  for (TASourceType ta_source_type : {TASourceType::kLayoutShape}) {
    for (auto& [drc, violation_info_list] : DC_INST.getViolationInfo(ta_panel.getRegionQuery(ta_source_type))) {
      source_drc_violation_map[ta_source_type][drc] = violation_info_list;
    }
  }

  std::map<std::string, irt_int>& rule_number_map = ta_panel_stat.get_drc_number_map();
  for (auto& [ta_source_type, drc_violation_map] : source_drc_violation_map) {
    for (auto& [drc, violation_list] : drc_violation_map) {
      rule_number_map[drc] += violation_list.size();
    }
  }
  std::map<std::string, irt_int>& source_number_map = ta_panel_stat.get_source_number_map();
  for (auto& [ta_source_type, drc_violation_map] : source_drc_violation_map) {
    irt_int total_number = 0;
    for (auto& [drc, violation_list] : drc_violation_map) {
      total_number += violation_list.size();
    }
    source_number_map[GetTASourceTypeName()(ta_source_type)] = total_number;
  }

  irt_int total_drc_number = 0;
  for (auto& [ta_source_type, drc_violation_map] : source_drc_violation_map) {
    for (auto& [drc, violation_list] : drc_violation_map) {
      total_drc_number += violation_list.size();
    }
  }
  ta_panel_stat.set_total_drc_number(total_drc_number);

  ta_panel.set_ta_panel_stat(ta_panel_stat);
}

void TrackAssigner::reportTAPanel(TAModel& ta_model, TAPanel& ta_panel)
{
  if (omp_get_num_threads() > 1) {
    return;
  }
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  TAPanelStat& ta_panel_stat = ta_panel.get_ta_panel_stat();
  double total_wire_length = ta_panel_stat.get_total_wire_length();
  double total_prefer_wire_length = ta_panel_stat.get_total_prefer_wire_length();
  double total_nonprefer_wire_length = ta_panel_stat.get_total_nonprefer_wire_length();
  std::map<TASourceType, std::map<std::string, std::vector<ViolationInfo>>>& source_drc_violation_map
      = ta_panel_stat.get_source_drc_violation_map();
  std::map<std::string, irt_int>& rule_number_map = ta_panel_stat.get_drc_number_map();
  std::map<std::string, irt_int>& source_number_map = ta_panel_stat.get_source_number_map();
  irt_int total_drc_number = ta_panel_stat.get_total_drc_number();

  // report wire info
  fort::char_table wire_table;
  wire_table.set_border_style(FT_SOLID_ROUND_STYLE);
  wire_table << fort::header << "Routing Layer"
             << "Prefer Wire Length"
             << "Nonprefer Wire Length"
             << "Wire Length / um" << fort::endr;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    wire_table << routing_layer.get_layer_name();
    if (routing_layer.get_layer_idx() == ta_panel.get_layer_idx()) {
      wire_table << total_prefer_wire_length << total_nonprefer_wire_length << total_wire_length;
    } else {
      wire_table << 0 << 0 << 0;
    }
    wire_table << fort::endr;
  }
  wire_table << fort::header << "Total" << total_prefer_wire_length << total_nonprefer_wire_length << total_wire_length << fort::endr;

  // init item column/row map
  irt_int row = 0;
  std::map<std::string, irt_int> item_row_map;
  for (auto& [drc_rule, drc_number] : rule_number_map) {
    item_row_map[drc_rule] = ++row;
  }
  item_row_map["Total"] = ++row;

  irt_int column = 0;
  std::map<std::string, irt_int> item_column_map;
  for (auto& [ta_source_type, drc_number_map] : source_number_map) {
    item_column_map[ta_source_type] = ++column;
  }
  item_column_map["Total"] = ++column;

  // build table
  fort::char_table drc_table;
  drc_table.set_border_style(FT_SOLID_ROUND_STYLE);
  drc_table << fort::header;
  drc_table[0][0] = "DRC\\Source";
  // first row item
  for (auto& [drc_rule, row] : item_row_map) {
    drc_table[row][0] = drc_rule;
  }
  // first column item
  drc_table << fort::header;
  for (auto& [source_name, column] : item_column_map) {
    drc_table[0][column] = source_name;
  }
  // element
  for (auto& [ta_source_type, drc_violation_map] : source_drc_violation_map) {
    irt_int column = item_column_map[GetTASourceTypeName()(ta_source_type)];
    for (auto& [drc_rule, row] : item_row_map) {
      if (RTUtil::exist(source_drc_violation_map[ta_source_type], drc_rule)) {
        drc_table[row][column] = RTUtil::getString(source_drc_violation_map[ta_source_type][drc_rule].size());
      } else {
        drc_table[row][column] = "0";
      }
    }
  }
  // last row
  for (auto& [ta_source_type, total_number] : source_number_map) {
    irt_int row = item_row_map["Total"];
    irt_int column = item_column_map[ta_source_type];
    drc_table[row][column] = RTUtil::getString(total_number);
  }
  // last column
  for (auto& [drc_rule, total_number] : rule_number_map) {
    irt_int row = item_row_map[drc_rule];
    irt_int column = item_column_map["Total"];
    drc_table[row][column] = RTUtil::getString(total_number);
  }
  drc_table[item_row_map["Total"]][item_column_map["Total"]] = RTUtil::getString(total_drc_number);

  // print
  std::vector<std::vector<std::string>> table_list;
  table_list.push_back(RTUtil::splitString(wire_table.to_string(), '\n'));
  table_list.push_back(RTUtil::splitString(drc_table.to_string(), '\n'));
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

bool TrackAssigner::stopTAPanel(TAModel& ta_model, TAPanel& ta_panel)
{
  return (ta_panel.get_ta_panel_stat().get_total_drc_number() == 0);
}

void TrackAssigner::freeTAPanel(TAModel& ta_model, TAPanel& ta_panel)
{
  GridMap<TANode>& ta_node_map = ta_panel.get_ta_node_map();
  ta_node_map.free();
}

void TrackAssigner::countTAModel(TAModel& ta_model)
{
  TAModelStat ta_model_stat;

  std::map<irt_int, double>& routing_wire_length_map = ta_model_stat.get_routing_wire_length_map();
  std::map<irt_int, double>& routing_prefer_wire_length_map = ta_model_stat.get_routing_prefer_wire_length_map();
  std::map<irt_int, double>& routing_nonprefer_wire_length_map = ta_model_stat.get_routing_nonprefer_wire_length_map();
  std::map<TASourceType, std::map<std::string, std::vector<ViolationInfo>>>& source_drc_violation_map
      = ta_model_stat.get_source_drc_violation_map();
  std::map<std::string, irt_int>& rule_number_map = ta_model_stat.get_drc_number_map();
  std::map<std::string, irt_int>& source_number_map = ta_model_stat.get_source_number_map();

  for (std::vector<TAPanel>& ta_panel_list : ta_model.get_layer_panel_list()) {
    for (TAPanel& ta_panel : ta_panel_list) {
      TAPanelStat& ta_panel_stat = ta_panel.get_ta_panel_stat();
      routing_wire_length_map[ta_panel.get_layer_idx()] += ta_panel_stat.get_total_wire_length();
      routing_prefer_wire_length_map[ta_panel.get_layer_idx()] += ta_panel_stat.get_total_prefer_wire_length();
      routing_nonprefer_wire_length_map[ta_panel.get_layer_idx()] += ta_panel_stat.get_total_nonprefer_wire_length();
    }
  }
  for (std::vector<TAPanel>& ta_panel_list : ta_model.get_layer_panel_list()) {
    for (TAPanel& ta_panel : ta_panel_list) {
      TAPanelStat& ta_panel_stat = ta_panel.get_ta_panel_stat();
      for (auto& [ta_source_type, drc_violation_map] : ta_panel_stat.get_source_drc_violation_map()) {
        for (auto& [drc, violation_list] : drc_violation_map) {
          for (ViolationInfo& violation : violation_list) {
            source_drc_violation_map[ta_source_type][drc].push_back(violation);
          }
        }
      }
    }
  }
  for (auto& [ta_source_type, drc_violation_map] : source_drc_violation_map) {
    for (auto& [drc, violation_list] : drc_violation_map) {
      rule_number_map[drc] += violation_list.size();
    }
  }
  for (auto& [ta_source_type, drc_violation_map] : source_drc_violation_map) {
    irt_int total_number = 0;
    for (auto& [drc, violation_list] : drc_violation_map) {
      total_number += violation_list.size();
    }
    source_number_map[GetTASourceTypeName()(ta_source_type)] = total_number;
  }

  double total_wire_length = 0;
  double total_prefer_wire_length = 0;
  double total_nonprefer_wire_length = 0;
  irt_int total_drc_number = 0;
  for (auto& [routing_layer_idx, wire_length] : routing_wire_length_map) {
    total_wire_length += wire_length;
  }
  for (auto& [routing_layer_idx, prefer_wire_length] : routing_prefer_wire_length_map) {
    total_prefer_wire_length += prefer_wire_length;
  }
  for (auto& [routing_layer_idx, nonprefer_wire_length] : routing_nonprefer_wire_length_map) {
    total_nonprefer_wire_length += nonprefer_wire_length;
  }
  for (auto& [ta_source_type, drc_violation_map] : source_drc_violation_map) {
    for (auto& [drc, violation_list] : drc_violation_map) {
      total_drc_number += violation_list.size();
    }
  }
  ta_model_stat.set_total_wire_length(total_wire_length);
  ta_model_stat.set_total_prefer_wire_length(total_prefer_wire_length);
  ta_model_stat.set_total_nonprefer_wire_length(total_nonprefer_wire_length);
  ta_model_stat.set_total_drc_number(total_drc_number);

  ta_model.set_ta_model_stat(ta_model_stat);
}

void TrackAssigner::reportTAModel(TAModel& ta_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  TAModelStat& ta_model_stat = ta_model.get_ta_model_stat();
  std::map<irt_int, double>& routing_wire_length_map = ta_model_stat.get_routing_wire_length_map();
  std::map<irt_int, double>& routing_prefer_wire_length_map = ta_model_stat.get_routing_prefer_wire_length_map();
  std::map<irt_int, double>& routing_nonprefer_wire_length_map = ta_model_stat.get_routing_nonprefer_wire_length_map();
  std::map<TASourceType, std::map<std::string, std::vector<ViolationInfo>>>& source_drc_violation_map
      = ta_model_stat.get_source_drc_violation_map();
  std::map<std::string, irt_int>& rule_number_map = ta_model_stat.get_drc_number_map();
  std::map<std::string, irt_int>& source_number_map = ta_model_stat.get_source_number_map();
  double total_wire_length = ta_model_stat.get_total_wire_length();
  double total_prefer_wire_length = ta_model_stat.get_total_prefer_wire_length();
  double total_nonprefer_wire_length = ta_model_stat.get_total_nonprefer_wire_length();
  irt_int total_drc_number = ta_model_stat.get_total_drc_number();

  // report wire info
  fort::char_table wire_table;
  wire_table.set_border_style(FT_SOLID_ROUND_STYLE);
  wire_table << fort::header << "Routing Layer"
             << "Prefer Wire Length"
             << "Nonprefer Wire Length"
             << "Wire Length / um" << fort::endr;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    double layer_idx = routing_layer.get_layer_idx();
    wire_table << routing_layer.get_layer_name() << routing_prefer_wire_length_map[layer_idx]
               << routing_nonprefer_wire_length_map[layer_idx] << routing_wire_length_map[layer_idx] << fort::endr;
  }
  wire_table << fort::header << "Total" << total_prefer_wire_length << total_nonprefer_wire_length << total_wire_length << fort::endr;

  // init item column/row map
  irt_int row = 0;
  std::map<std::string, irt_int> item_row_map;
  for (auto& [drc_rule, drc_number] : rule_number_map) {
    item_row_map[drc_rule] = ++row;
  }
  item_row_map["Total"] = ++row;

  irt_int column = 0;
  std::map<std::string, irt_int> item_column_map;
  for (auto& [ta_source_type, drc_number_map] : source_number_map) {
    item_column_map[ta_source_type] = ++column;
  }
  item_column_map["Total"] = ++column;

  // build table
  fort::char_table drc_table;
  drc_table.set_border_style(FT_SOLID_ROUND_STYLE);
  drc_table << fort::header;
  drc_table[0][0] = "DRC\\Source";
  // first row item
  for (auto& [drc_rule, row] : item_row_map) {
    drc_table[row][0] = drc_rule;
  }
  // first column item
  drc_table << fort::header;
  for (auto& [source_name, column] : item_column_map) {
    drc_table[0][column] = source_name;
  }
  // element
  for (auto& [ta_source_type, drc_violation_map] : source_drc_violation_map) {
    irt_int column = item_column_map[GetTASourceTypeName()(ta_source_type)];
    for (auto& [drc_rule, row] : item_row_map) {
      if (RTUtil::exist(source_drc_violation_map[ta_source_type], drc_rule)) {
        drc_table[row][column] = RTUtil::getString(source_drc_violation_map[ta_source_type][drc_rule].size());
      } else {
        drc_table[row][column] = "0";
      }
    }
  }
  // last row
  for (auto& [ta_source_type, total_number] : source_number_map) {
    irt_int row = item_row_map["Total"];
    irt_int column = item_column_map[ta_source_type];
    drc_table[row][column] = RTUtil::getString(total_number);
  }
  // last column
  for (auto& [drc_rule, total_number] : rule_number_map) {
    irt_int row = item_row_map[drc_rule];
    irt_int column = item_column_map["Total"];
    drc_table[row][column] = RTUtil::getString(total_number);
  }
  drc_table[item_row_map["Total"]][item_column_map["Total"]] = RTUtil::getString(total_drc_number);

  // print
  std::vector<std::vector<std::string>> table_list;
  table_list.push_back(RTUtil::splitString(wire_table.to_string(), '\n'));
  table_list.push_back(RTUtil::splitString(drc_table.to_string(), '\n'));
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

bool TrackAssigner::stopTAModel(TAModel& ta_model)
{
  return (ta_model.get_ta_model_stat().get_total_drc_number() == 0);
}

#endif

#if 1  // update

void TrackAssigner::update(TAModel& ta_model)
{
  for (TANet& ta_net : ta_model.get_ta_net_list()) {
    Net* origin_net = ta_net.get_origin_net();
    origin_net->set_ta_result_tree(ta_net.get_ta_result_tree());
  }
}

#endif

#if 1  // plot ta_panel

void TrackAssigner::plotTAPanel(TAPanel& ta_panel, irt_int curr_task_idx)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = DM_INST.getDatabase().get_layer_via_master_list();
  std::string ta_temp_directory_path = DM_INST.getConfig().ta_temp_directory_path;

  irt_int width = INT_MAX;
  for (ScaleGrid& x_grid : ta_panel.get_panel_track_axis().get_x_grid_list()) {
    width = std::min(width, x_grid.get_step_length());
  }
  for (ScaleGrid& y_grid : ta_panel.get_panel_track_axis().get_y_grid_list()) {
    width = std::min(width, y_grid.get_step_length());
  }
  width = std::max(1, width / 3);

  GPGDS gp_gds;

  // base_region
  GPStruct base_region_struct("base_region");
  GPBoundary gp_boundary;
  gp_boundary.set_layer_idx(0);
  gp_boundary.set_data_type(0);
  gp_boundary.set_rect(ta_panel.get_rect());
  base_region_struct.push(gp_boundary);
  gp_gds.addStruct(base_region_struct);

  // gcell_axis
  GPStruct gcell_axis_struct("gcell_axis");
  std::vector<irt_int> gcell_x_list = RTUtil::getClosedScaleList(ta_panel.get_lb_x(), ta_panel.get_rt_x(), gcell_axis.get_x_grid_list());
  std::vector<irt_int> gcell_y_list = RTUtil::getClosedScaleList(ta_panel.get_lb_y(), ta_panel.get_rt_y(), gcell_axis.get_y_grid_list());
  for (irt_int x : gcell_x_list) {
    GPPath gp_path;
    gp_path.set_layer_idx(0);
    gp_path.set_data_type(1);
    gp_path.set_segment(x, ta_panel.get_lb_y(), x, ta_panel.get_rt_y());
    gcell_axis_struct.push(gp_path);
  }
  for (irt_int y : gcell_y_list) {
    GPPath gp_path;
    gp_path.set_layer_idx(0);
    gp_path.set_data_type(1);
    gp_path.set_segment(ta_panel.get_lb_x(), y, ta_panel.get_rt_x(), y);
    gcell_axis_struct.push(gp_path);
  }
  gp_gds.addStruct(gcell_axis_struct);

  GridMap<TANode>& ta_node_map = ta_panel.get_ta_node_map();
  // ta_node_map
  GPStruct ta_node_map_struct("ta_node_map");
  for (irt_int grid_x = 0; grid_x < ta_node_map.get_x_size(); grid_x++) {
    for (irt_int grid_y = 0; grid_y < ta_node_map.get_y_size(); grid_y++) {
      TANode& ta_node = ta_node_map[grid_x][grid_y];
      PlanarRect real_rect = RTUtil::getEnlargedRect(ta_node.get_planar_coord(), width);
      irt_int y_reduced_span = std::max(1, real_rect.getYSpan() / 12);
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
      ta_node_map_struct.push(gp_boundary);

      y -= y_reduced_span;
      GPText gp_text_node_real_coord;
      gp_text_node_real_coord.set_coord(real_rect.get_lb_x(), y);
      gp_text_node_real_coord.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
      gp_text_node_real_coord.set_message(
          RTUtil::getString("(", ta_node.get_x(), " , ", ta_node.get_y(), " , ", ta_node.get_layer_idx(), ")"));
      gp_text_node_real_coord.set_layer_idx(GP_INST.getGDSIdxByRouting(ta_node.get_layer_idx()));
      gp_text_node_real_coord.set_presentation(GPTextPresentation::kLeftMiddle);
      ta_node_map_struct.push(gp_text_node_real_coord);

      y -= y_reduced_span;
      GPText gp_text_node_grid_coord;
      gp_text_node_grid_coord.set_coord(real_rect.get_lb_x(), y);
      gp_text_node_grid_coord.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
      gp_text_node_grid_coord.set_message(RTUtil::getString("(", grid_x, " , ", grid_y, " , ", ta_node.get_layer_idx(), ")"));
      gp_text_node_grid_coord.set_layer_idx(GP_INST.getGDSIdxByRouting(ta_node.get_layer_idx()));
      gp_text_node_grid_coord.set_presentation(GPTextPresentation::kLeftMiddle);
      ta_node_map_struct.push(gp_text_node_grid_coord);

      y -= y_reduced_span;
      GPText gp_text_direction_set;
      gp_text_direction_set.set_coord(real_rect.get_lb_x(), y);
      gp_text_direction_set.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
      gp_text_direction_set.set_message("direction_set: ");
      gp_text_direction_set.set_layer_idx(GP_INST.getGDSIdxByRouting(ta_node.get_layer_idx()));
      gp_text_direction_set.set_presentation(GPTextPresentation::kLeftMiddle);
      ta_node_map_struct.push(gp_text_direction_set);

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
        ta_node_map_struct.push(gp_text_direction_set_info);
      }
    }
  }
  gp_gds.addStruct(ta_node_map_struct);

  // neighbor_map
  GPStruct neighbor_map_struct("neighbor_map");
  for (irt_int grid_x = 0; grid_x < ta_node_map.get_x_size(); grid_x++) {
    for (irt_int grid_y = 0; grid_y < ta_node_map.get_y_size(); grid_y++) {
      TANode& ta_node = ta_node_map[grid_x][grid_y];
      PlanarRect real_rect = RTUtil::getEnlargedRect(ta_node.get_planar_coord(), width);

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

  // panel_track_axis
  GPStruct panel_track_axis_struct("panel_track_axis");
  ScaleAxis& panel_track_axis = ta_panel.get_panel_track_axis();
  std::vector<irt_int> track_x_list
      = RTUtil::getClosedScaleList(ta_panel.get_lb_x(), ta_panel.get_rt_x(), panel_track_axis.get_x_grid_list());
  std::vector<irt_int> track_y_list
      = RTUtil::getClosedScaleList(ta_panel.get_lb_y(), ta_panel.get_rt_y(), panel_track_axis.get_y_grid_list());
  for (irt_int x : track_x_list) {
    GPPath gp_path;
    gp_path.set_data_type(static_cast<irt_int>(GPGraphType::kTrackAxis));
    gp_path.set_segment(x, ta_panel.get_lb_y(), x, ta_panel.get_rt_y());
    gp_path.set_layer_idx(GP_INST.getGDSIdxByRouting(ta_panel.get_layer_idx()));
    panel_track_axis_struct.push(gp_path);
  }
  for (irt_int y : track_y_list) {
    GPPath gp_path;
    gp_path.set_data_type(static_cast<irt_int>(GPGraphType::kTrackAxis));
    gp_path.set_segment(ta_panel.get_lb_x(), y, ta_panel.get_rt_x(), y);
    gp_path.set_layer_idx(GP_INST.getGDSIdxByRouting(ta_panel.get_layer_idx()));
    panel_track_axis_struct.push(gp_path);
  }
  gp_gds.addStruct(panel_track_axis_struct);

  // source_region_query_map
  std::vector<std::pair<TASourceType, GPGraphType>> source_graph_pair_list
      = {{TASourceType::kLayoutShape, GPGraphType::kLayoutShape}, {TASourceType::kReservedVia, GPGraphType::kReservedVia}};
  for (auto& [ta_source_type, gp_graph_type] : source_graph_pair_list) {
    for (bool is_routing : {true, false}) {
      for (auto& [layer_idx, net_rect_map] : DC_INST.getLayerNetRectMap(ta_panel.getRegionQuery(ta_source_type), is_routing)) {
        for (auto& [net_idx, rect_set] : net_rect_map) {
          GPStruct net_rect_struct(RTUtil::getString(GetTASourceTypeName()(ta_source_type), "@", net_idx));
          for (const LayerRect& rect : rect_set) {
            GPBoundary gp_boundary;
            gp_boundary.set_data_type(static_cast<irt_int>(gp_graph_type));
            gp_boundary.set_rect(rect);
            if (is_routing) {
              gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(layer_idx));
            } else {
              gp_boundary.set_layer_idx(GP_INST.getGDSIdxByCut(layer_idx));
            }
            net_rect_struct.push(gp_boundary);
          }
          gp_gds.addStruct(net_rect_struct);
        }
      }
    }
  }

  // task
  for (TATask& ta_task : ta_panel.get_ta_task_list()) {
    GPStruct task_struct(RTUtil::getString("task_", ta_task.get_task_idx(), "(net_", ta_task.get_origin_net_idx(), ")"));

    if (curr_task_idx == -1 || ta_task.get_task_idx() == curr_task_idx) {
      for (TAGroup& ta_group : ta_task.get_ta_group_list()) {
        for (LayerCoord& coord : ta_group.get_coord_list()) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kKey));
          gp_boundary.set_rect(RTUtil::getEnlargedRect(coord, width));
          gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(coord.get_layer_idx()));
          task_struct.push(gp_boundary);
        }
      }
    }
    {
      // bounding_box
      GPBoundary gp_boundary;
      gp_boundary.set_layer_idx(0);
      gp_boundary.set_data_type(2);
      gp_boundary.set_rect(ta_task.get_bounding_box());
      task_struct.push(gp_boundary);
    }

    for (Segment<TNode<LayerCoord>*>& segment : RTUtil::getSegListByTree(ta_task.get_routing_tree())) {
      LayerCoord first_coord = segment.get_first()->value();
      LayerCoord second_coord = segment.get_second()->value();
      irt_int first_layer_idx = first_coord.get_layer_idx();
      irt_int second_layer_idx = second_coord.get_layer_idx();
      irt_int half_width = routing_layer_list[first_layer_idx].get_min_width() / 2;

      if (first_layer_idx == second_layer_idx) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kPath));
        gp_boundary.set_rect(RTUtil::getEnlargedRect(first_coord, second_coord, half_width));
        gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(first_layer_idx));
        task_struct.push(gp_boundary);
      } else {
        RTUtil::swapASC(first_layer_idx, second_layer_idx);
        for (irt_int layer_idx = first_layer_idx; layer_idx < second_layer_idx; layer_idx++) {
          ViaMaster& via_master = layer_via_master_list[layer_idx].front();

          LayerRect& above_enclosure = via_master.get_above_enclosure();
          LayerRect offset_above_enclosure(RTUtil::getOffsetRect(above_enclosure, first_coord), above_enclosure.get_layer_idx());
          GPBoundary above_boundary;
          above_boundary.set_rect(offset_above_enclosure);
          above_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(above_enclosure.get_layer_idx()));
          above_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kPath));
          task_struct.push(above_boundary);

          LayerRect& below_enclosure = via_master.get_below_enclosure();
          LayerRect offset_below_enclosure(RTUtil::getOffsetRect(below_enclosure, first_coord), below_enclosure.get_layer_idx());
          GPBoundary below_boundary;
          below_boundary.set_rect(offset_below_enclosure);
          below_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(below_enclosure.get_layer_idx()));
          below_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kPath));
          task_struct.push(below_boundary);

          for (PlanarRect& cut_shape : via_master.get_cut_shape_list()) {
            LayerRect offset_cut_shape(RTUtil::getOffsetRect(cut_shape, first_coord), via_master.get_cut_layer_idx());
            GPBoundary cut_boundary;
            cut_boundary.set_rect(offset_cut_shape);
            cut_boundary.set_layer_idx(GP_INST.getGDSIdxByCut(via_master.get_cut_layer_idx()));
            cut_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kPath));
            task_struct.push(cut_boundary);
          }
        }
      }
    }
    gp_gds.addStruct(task_struct);
  }
  std::string gds_file_path = RTUtil::getString(ta_temp_directory_path, "ta_panel_", ta_panel.get_ta_panel_id().get_layer_idx(), "_",
                                                ta_panel.get_ta_panel_id().get_panel_idx(), ".gds");
  GP_INST.plot(gp_gds, gds_file_path, false, false);
}

#endif

}  // namespace irt
