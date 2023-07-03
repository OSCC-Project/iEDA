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
#include "RTAPI.hpp"
#include "TAPanel.hpp"
#include "TASchedule.hpp"

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
  TAModel ta_model = initTAModel(net_list);
  buildTAModel(ta_model);
  assignTAModel(ta_model);
  updateTAModel(ta_model);
  reportTAModel(ta_model);
}

#if 1  // build ta_model

TAModel TrackAssigner::initTAModel(std::vector<Net>& net_list)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  Die& die = DM_INST.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  TAModel ta_model;
  ta_model.set_ta_net_list(convertToTANetList(net_list));

  std::vector<std::vector<TAPanel>>& layer_panel_list = ta_model.get_layer_panel_list();
  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::vector<TAPanel> ta_panel_list;
    if (routing_layer.isPreferH()) {
      for (ScaleGrid& gcell_grid : gcell_axis.get_y_grid_list()) {
        for (irt_int line = gcell_grid.get_start_line(); line < gcell_grid.get_end_line(); line += gcell_grid.get_step_length()) {
          TAPanel ta_panel;
          ta_panel.set_rect(PlanarRect(die.get_real_lb_x(), line, die.get_real_rt_x(), line + gcell_grid.get_step_length()));
          ta_panel.set_layer_idx(routing_layer.get_layer_idx());
          ta_panel.set_panel_idx(static_cast<irt_int>(ta_panel_list.size()));
          for (TASourceType ta_source_type : {TASourceType::kBlockage, TASourceType::kOtherPanelResult, TASourceType::kSelfPanelResult}) {
            ta_panel.get_source_region_query_map()[ta_source_type] = RTAPI_INST.initRegionQuery();
          }
          ta_panel_list.push_back(ta_panel);
        }
      }
    } else {
      for (ScaleGrid& gcell_grid : gcell_axis.get_x_grid_list()) {
        for (irt_int line = gcell_grid.get_start_line(); line < gcell_grid.get_end_line(); line += gcell_grid.get_step_length()) {
          TAPanel ta_panel;
          ta_panel.set_rect(PlanarRect(line, die.get_real_lb_y(), line + gcell_grid.get_step_length(), die.get_real_rt_y()));
          ta_panel.set_layer_idx(routing_layer.get_layer_idx());
          ta_panel.set_panel_idx(static_cast<irt_int>(ta_panel_list.size()));
          for (TASourceType ta_source_type : {TASourceType::kBlockage, TASourceType::kOtherPanelResult, TASourceType::kSelfPanelResult}) {
            ta_panel.get_source_region_query_map()[ta_source_type] = RTAPI_INST.initRegionQuery();
          }
          ta_panel_list.push_back(ta_panel);
        }
      }
    }
    layer_panel_list.push_back(ta_panel_list);
  }
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
  for (Pin& pin : net.get_pin_list()) {
    ta_net.get_ta_pin_list().push_back(TAPin(pin));
  }
  ta_net.set_gr_result_tree(net.get_gr_result_tree());
  ta_net.set_ta_result_tree(net.get_gr_result_tree());
  return ta_net;
}

void TrackAssigner::buildTAModel(TAModel& ta_model)
{
  updateNetBlockageMap(ta_model);
  buildPanelScaleAxis(ta_model);
  buildTATaskList(ta_model);
  buildLayerPanelList(ta_model);
}

void TrackAssigner::updateNetBlockageMap(TAModel& ta_model)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  EXTPlanarRect& die = DM_INST.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<Blockage>& routing_blockage_list = DM_INST.getDatabase().get_routing_blockage_list();

  std::vector<std::vector<TAPanel>>& layer_panel_list = ta_model.get_layer_panel_list();

  for (const Blockage& routing_blockage : routing_blockage_list) {
    irt_int blockage_layer_idx = routing_blockage.get_layer_idx();
    LayerRect blockage_real_rect(routing_blockage.get_real_rect(), blockage_layer_idx);
    for (const LayerRect& max_scope_real_rect : RTAPI_INST.getMaxScope(blockage_real_rect)) {
      LayerRect max_scope_regular_rect = RTUtil::getRegularRect(max_scope_real_rect, die.get_real_rect());
      PlanarRect max_scope_grid_rect = RTUtil::getClosedGridRect(max_scope_regular_rect, gcell_axis);
      if (routing_layer_list[blockage_layer_idx].isPreferH()) {
        for (irt_int y = max_scope_grid_rect.get_lb_y(); y <= max_scope_grid_rect.get_rt_y(); y++) {
          TAPanel& ta_panel = layer_panel_list[blockage_layer_idx][y];
          ta_panel.addRect(TASourceType::kBlockage, -1, blockage_real_rect);
        }
      } else {
        for (irt_int x = max_scope_grid_rect.get_lb_x(); x <= max_scope_grid_rect.get_rt_x(); x++) {
          TAPanel& ta_panel = layer_panel_list[blockage_layer_idx][x];
          ta_panel.addRect(TASourceType::kBlockage, -1, blockage_real_rect);
        }
      }
    }
  }
  for (TANet& ta_net : ta_model.get_ta_net_list()) {
    for (TAPin& ta_pin : ta_net.get_ta_pin_list()) {
      for (const EXTLayerRect& routing_shape : ta_pin.get_routing_shape_list()) {
        irt_int shape_layer_idx = routing_shape.get_layer_idx();
        LayerRect shape_real_rect(routing_shape.get_real_rect(), shape_layer_idx);
        for (const LayerRect& max_scope_real_rect : RTAPI_INST.getMaxScope(shape_real_rect)) {
          LayerRect max_scope_regular_rect = RTUtil::getRegularRect(max_scope_real_rect, die.get_real_rect());
          PlanarRect max_scope_grid_rect = RTUtil::getClosedGridRect(max_scope_regular_rect, gcell_axis);
          if (routing_layer_list[shape_layer_idx].isPreferH()) {
            for (irt_int y = max_scope_grid_rect.get_lb_y(); y <= max_scope_grid_rect.get_rt_y(); y++) {
              TAPanel& ta_panel = layer_panel_list[shape_layer_idx][y];
              ta_panel.addRect(TASourceType::kBlockage, ta_net.get_net_idx(), shape_real_rect);
            }
          } else {
            for (irt_int x = max_scope_grid_rect.get_lb_x(); x <= max_scope_grid_rect.get_rt_x(); x++) {
              TAPanel& ta_panel = layer_panel_list[shape_layer_idx][x];
              ta_panel.addRect(TASourceType::kBlockage, ta_net.get_net_idx(), shape_real_rect);
            }
          }
        }
      }
    }
  }
}

void TrackAssigner::buildPanelScaleAxis(TAModel& ta_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  for (std::vector<TAPanel>& ta_panel_list : ta_model.get_layer_panel_list()) {
    for (TAPanel& ta_panel : ta_panel_list) {
      RoutingLayer& routing_layer = routing_layer_list[ta_panel.get_layer_idx()];

      std::vector<irt_int> x_scale_list
          = RTUtil::getClosedScaleList(ta_panel.get_lb_x(), ta_panel.get_rt_x(), routing_layer.getXTrackGridList());
      std::vector<irt_int> y_scale_list
          = RTUtil::getClosedScaleList(ta_panel.get_lb_y(), ta_panel.get_rt_y(), routing_layer.getYTrackGridList());
      ScaleAxis& panel_scale_axis = ta_panel.get_panel_scale_axis();
      panel_scale_axis.set_x_grid_list(RTUtil::makeScaleGridList(x_scale_list));
      panel_scale_axis.set_y_grid_list(RTUtil::makeScaleGridList(y_scale_list));
    }
  }
}

void TrackAssigner::buildTATaskList(TAModel& ta_model)
{
  std::vector<std::vector<TAPanel>>& layer_panel_list = ta_model.get_layer_panel_list();

  for (TANet& ta_net : ta_model.get_ta_net_list()) {
    for (auto& [ta_node_node, ta_task] : makeTANodeTaskMap(layer_panel_list, ta_net)) {
      irt_int layer_idx = ta_node_node->value().get_first_guide().get_layer_idx();
      PlanarCoord& first_grid_coord = ta_node_node->value().get_first_guide().get_grid_coord();
      PlanarCoord& second_grid_coord = ta_node_node->value().get_second_guide().get_grid_coord();
      irt_int panel_idx = RTUtil::isHorizontal(first_grid_coord, second_grid_coord) ? first_grid_coord.get_y() : first_grid_coord.get_x();
      TAPanel& ta_panel = layer_panel_list[layer_idx][panel_idx];

      std::vector<TATask>& ta_task_list = ta_panel.get_ta_task_list();
      ta_task.set_origin_net_idx(ta_net.get_net_idx());
      ta_task.set_origin_node(ta_node_node);
      ta_task.set_task_idx(static_cast<irt_int>(ta_task_list.size()));
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

std::map<TNode<RTNode>*, TATask> TrackAssigner::makeTANodeTaskMap(std::vector<std::vector<TAPanel>>& layer_panel_list, TANet& ta_net)
{
  // ta_dr_list_map
  std::map<TNode<RTNode>*, std::vector<TNode<RTNode>*>> ta_dr_list_map;
  for (Segment<TNode<RTNode>*>& segment : RTUtil::getSegListByTree(ta_net.get_ta_result_tree())) {
    TNode<RTNode>* ta_node_node = segment.get_first();
    TNode<RTNode>* dr_node_node = segment.get_second();
    if (ta_node_node->value().isDRNode()) {
      std::swap(ta_node_node, dr_node_node);
    }
    ta_dr_list_map[ta_node_node].push_back(dr_node_node);
  }
  std::map<TNode<RTNode>*, TATask> ta_node_task_map;
  for (auto& [ta_node_node, dr_node_node_list] : ta_dr_list_map) {
    irt_int layer_idx = ta_node_node->value().get_first_guide().get_layer_idx();
    PlanarCoord& first_grid_coord = ta_node_node->value().get_first_guide().get_grid_coord();
    PlanarCoord& second_grid_coord = ta_node_node->value().get_second_guide().get_grid_coord();
    irt_int panel_idx = RTUtil::isHorizontal(first_grid_coord, second_grid_coord) ? first_grid_coord.get_y() : first_grid_coord.get_x();
    TAPanel& ta_panel = layer_panel_list[layer_idx][panel_idx];

    std::vector<TAGroup>& ta_group_list = ta_node_task_map[ta_node_node].get_ta_group_list();
    for (TNode<RTNode>* dr_node_node : dr_node_node_list) {
      ta_group_list.push_back(makeTAGroup(ta_panel, ta_node_node, dr_node_node));
    }
  }
  return ta_node_task_map;
}

TAGroup TrackAssigner::makeTAGroup(TAPanel& ta_panel, TNode<RTNode>* ta_node_node, TNode<RTNode>* dr_node_node)
{
  ScaleAxis& panel_scale_axis = ta_panel.get_panel_scale_axis();

  Guide& dr_guide = dr_node_node->value().get_first_guide();
  PlanarCoord& dr_grid_coord = dr_guide.get_grid_coord();

  PlanarCoord& first_grid_coord = ta_node_node->value().get_first_guide().get_grid_coord();
  PlanarCoord& second_grid_coord = ta_node_node->value().get_second_guide().get_grid_coord();

  Orientation orientation = Orientation::kNone;
  if (dr_grid_coord == first_grid_coord) {
    orientation = RTUtil::getOrientation(dr_grid_coord, second_grid_coord);
  } else {
    orientation = RTUtil::getOrientation(dr_grid_coord, first_grid_coord);
  }
  std::vector<irt_int> x_list = RTUtil::getClosedScaleList(dr_guide.get_lb_x(), dr_guide.get_rt_x(), panel_scale_axis.get_x_grid_list());
  std::vector<irt_int> y_list = RTUtil::getClosedScaleList(dr_guide.get_lb_y(), dr_guide.get_rt_y(), panel_scale_axis.get_y_grid_list());
  if (orientation == Orientation::kEast || orientation == Orientation::kWest) {
    irt_int x = (orientation == Orientation::kEast ? x_list.back() : x_list.front());
    x_list.clear();
    x_list.push_back(x);
  } else if (orientation == Orientation::kNorth || orientation == Orientation::kSouth) {
    irt_int y = (orientation == Orientation::kNorth ? y_list.back() : y_list.front());
    y_list.clear();
    y_list.push_back(y);
  }
  TAGroup ta_group;
  std::vector<LayerCoord>& coord_list = ta_group.get_coord_list();
  for (irt_int x : x_list) {
    for (irt_int y : y_list) {
      coord_list.emplace_back(x, y, ta_panel.get_layer_idx());
    }
  }
  return ta_group;
}

void TrackAssigner::buildLayerPanelList(TAModel& ta_model)
{
  for (std::vector<TAPanel>& ta_panel_list : ta_model.get_layer_panel_list()) {
#pragma omp parallel for
    for (TAPanel& ta_panel : ta_panel_list) {
      initTANodeMap(ta_panel);
      buildNeighborMap(ta_panel);
      buildOBSTaskMap(ta_panel);
      checkTAPanel(ta_panel);
      // saveTAPanel(ta_panel);
    }
  }
}

void TrackAssigner::initTANodeMap(TAPanel& ta_panel)
{
  ScaleAxis& panel_scale_axis = ta_panel.get_panel_scale_axis();

  std::vector<irt_int> x_list = RTUtil::getClosedScaleList(ta_panel.get_lb_x(), ta_panel.get_rt_x(), panel_scale_axis.get_x_grid_list());
  std::vector<irt_int> y_list = RTUtil::getClosedScaleList(ta_panel.get_lb_y(), ta_panel.get_rt_y(), panel_scale_axis.get_y_grid_list());

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

void TrackAssigner::buildOBSTaskMap(TAPanel& ta_panel)
{
  EXTPlanarRect& die = DM_INST.getDatabase().get_die();

  GridMap<TANode>& ta_node_map = ta_panel.get_ta_node_map();

  std::map<irt_int, std::vector<irt_int>> net_task_map;
  for (TATask& ta_task : ta_panel.get_ta_task_list()) {
    net_task_map[ta_task.get_origin_net_idx()].push_back(ta_task.get_task_idx());
  }
  for (auto& [net_idx, blockage_list] : ta_panel.get_source_net_rect_map()[TASourceType::kBlockage]) {
    std::vector<irt_int>& task_idx_list = net_task_map[net_idx];
    for (LayerRect& blockage : blockage_list) {
      for (const LayerRect& min_scope_real_rect : RTAPI_INST.getMinScope(blockage)) {
        LayerRect min_scope_regular_rect = RTUtil::getRegularRect(min_scope_real_rect, die.get_real_rect());
        for (auto& [grid_coord, orientation_set] : getGridOrientationMap(ta_panel, min_scope_regular_rect)) {
          TANode& ta_node = ta_node_map[grid_coord.get_x()][grid_coord.get_y()];
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
}

std::map<LayerCoord, std::set<Orientation>, CmpLayerCoordByLayerASC> TrackAssigner::getGridOrientationMap(TAPanel& ta_panel,
                                                                                                          LayerRect& min_scope_regular_rect)
{
  ScaleAxis& panel_scale_axis = ta_panel.get_panel_scale_axis();

  std::map<LayerCoord, std::set<Orientation>, CmpLayerCoordByLayerASC> grid_orientation_map;
  for (Segment<LayerCoord>& real_segment : getRealSegmentList(ta_panel, min_scope_regular_rect)) {
    std::vector<Segment<LayerCoord>> real_segment_list{real_segment};

    bool is_open_overlap = false;
    for (LayerRect& real_rect : DM_INST.getRealRectList(real_segment_list)) {
      if (RTUtil::isOpenOverlap(min_scope_regular_rect, real_rect)) {
        is_open_overlap = true;
        break;
      }
    }
    if (is_open_overlap) {
      LayerCoord& first_coord = real_segment.get_first();
      LayerCoord& second_coord = real_segment.get_second();
      if (!RTUtil::existGrid(first_coord, panel_scale_axis) || !RTUtil::existGrid(second_coord, panel_scale_axis)) {
        LOG_INST.error(Loc::current(), "The coord can not find grid!");
      }
      if (!CmpLayerCoordByLayerASC()(first_coord, second_coord)) {
        std::swap(first_coord, second_coord);
      }
      Orientation orientation = RTUtil::getOrientation(first_coord, second_coord);
      LayerCoord first_grid_coord(RTUtil::getGridCoord(first_coord, panel_scale_axis), first_coord.get_layer_idx());
      LayerCoord second_grid_coord(RTUtil::getGridCoord(second_coord, panel_scale_axis), second_coord.get_layer_idx());
      grid_orientation_map[first_grid_coord].insert(orientation);
      grid_orientation_map[second_grid_coord].insert(RTUtil::getOppositeOrientation(orientation));
    }
  }
  return grid_orientation_map;
}

std::vector<Segment<LayerCoord>> TrackAssigner::getRealSegmentList(TAPanel& ta_panel, LayerRect& min_scope_regular_rect)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  std::vector<Segment<LayerCoord>> real_segment_list;

  irt_int layer_idx = min_scope_regular_rect.get_layer_idx();
  ScaleAxis& panel_scale_axis = ta_panel.get_panel_scale_axis();

  // ta只需要膨胀half_width
  irt_int enlarge_size = routing_layer_list[layer_idx].get_min_width() / 2;
  PlanarRect search_rect = RTUtil::getEnlargedRect(min_scope_regular_rect, enlarge_size);

  std::vector<irt_int> x_list
      = RTUtil::getOpenEnlargedScaleList(search_rect.get_lb_x(), search_rect.get_rt_x(), panel_scale_axis.get_x_grid_list());
  std::vector<irt_int> y_list
      = RTUtil::getOpenEnlargedScaleList(search_rect.get_lb_y(), search_rect.get_rt_y(), panel_scale_axis.get_y_grid_list());
  for (size_t y_idx = 0; y_idx < y_list.size(); y_idx++) {
    irt_int y = y_list[y_idx];
    if (search_rect.get_rt_y() <= y || y <= search_rect.get_lb_y()) {
      continue;
    }
    for (size_t x_idx = 1; x_idx < x_list.size(); x_idx++) {
      real_segment_list.emplace_back(LayerCoord(x_list[x_idx - 1], y, layer_idx), LayerCoord(x_list[x_idx], y, layer_idx));
    }
  }
  for (size_t x_idx = 0; x_idx < x_list.size(); x_idx++) {
    irt_int x = x_list[x_idx];
    if (search_rect.get_rt_x() <= x || x <= search_rect.get_lb_x()) {
      continue;
    }
    for (size_t y_idx = 1; y_idx < y_list.size(); y_idx++) {
      real_segment_list.emplace_back(LayerCoord(x, y_list[y_idx - 1], layer_idx), LayerCoord(x, y_list[y_idx], layer_idx));
    }
  }
  return real_segment_list;
}

void TrackAssigner::checkTAPanel(TAPanel& ta_panel)
{
  if (ta_panel.get_panel_idx() < 0) {
    LOG_INST.error(Loc::current(), "The panel_idx ", ta_panel.get_panel_idx(), " is error!");
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
      int n = 1;
      while (n--) {
        assignTAPanel(ta_panel);
        countTAPanel(ta_panel);
      }
      updateTAPanel(ta_model, ta_panel);
    }
    total_panel_num += ta_schedule_list.size();
    LOG_INST.info(Loc::current(), "Processed ", ta_schedule_list.size(), " panels", stage_monitor.getStatsInfo());
  }
  LOG_INST.info(Loc::current(), "Processed ", total_panel_num, " panels", monitor.getStatsInfo());
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
  initSingleNet(ta_panel, ta_task);
  while (!isConnectedAllEnd(ta_panel)) {
    for (TARouteStrategy ta_route_strategy : {TARouteStrategy::kFullyConsider, TARouteStrategy::kIgnoringSelfPanelResult,
                                              TARouteStrategy::kIgnoringOtherPanelResult, TARouteStrategy::kIgnoringBlockage}) {
      routeByStrategy(ta_panel, ta_route_strategy);
    }
    updatePathResult(ta_panel);
    updateDirectionSet(ta_panel);
    resetStartAndEnd(ta_panel);
    resetSinglePath(ta_panel);
  }
  updateNetResult(ta_panel, ta_task);
  resetSingleNet(ta_panel);
}

void TrackAssigner::initSingleNet(TAPanel& ta_panel, TATask& ta_task)
{
  ScaleAxis& panel_scale_axis = ta_panel.get_panel_scale_axis();
  GridMap<TANode>& ta_node_map = ta_panel.get_ta_node_map();
  std::vector<std::vector<TANode*>>& start_node_comb_list = ta_panel.get_start_node_comb_list();
  std::vector<std::vector<TANode*>>& end_node_comb_list = ta_panel.get_end_node_comb_list();

  // config
  ta_panel.set_wire_unit(1);
  ta_panel.set_corner_unit(1);
  ta_panel.set_via_unit(1);
  // single task
  ta_panel.set_ta_task_ref(&ta_task);
  ta_panel.set_routing_region(ta_panel.get_curr_bounding_box());
  {
    std::vector<std::vector<TANode*>> node_comb_list;
    std::vector<TAGroup>& ta_group_list = ta_task.get_ta_group_list();
    for (TAGroup& ta_group : ta_group_list) {
      std::vector<TANode*> node_comb;
      for (LayerCoord& coord : ta_group.get_coord_list()) {
        if (!RTUtil::existGrid(coord, panel_scale_axis)) {
          LOG_INST.error(Loc::current(), "The coord can not find grid!");
        }
        PlanarCoord grid_coord = RTUtil::getGridCoord(coord, panel_scale_axis);
        node_comb.push_back(&ta_node_map[grid_coord.get_x()][grid_coord.get_y()]);
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
  {
    std::set<Orientation>& routing_offset_set = ta_panel.get_routing_offset_set();
    for (std::vector<TANode*>& start_node_comb : start_node_comb_list) {
      for (TANode* start_node : start_node_comb) {
        for (std::vector<TANode*>& end_node_comb : end_node_comb_list) {
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
  return ta_panel.get_end_node_comb_list().empty();
}

void TrackAssigner::routeByStrategy(TAPanel& ta_panel, TARouteStrategy ta_route_strategy)
{
  if (ta_route_strategy == TARouteStrategy::kFullyConsider) {
    routeSinglePath(ta_panel);
  } else if (isRoutingFailed(ta_panel)) {
    resetSinglePath(ta_panel);
    ta_panel.set_ta_route_strategy(ta_route_strategy);
    routeSinglePath(ta_panel);
    ta_panel.set_ta_route_strategy(TARouteStrategy::kNone);
    if (!isRoutingFailed(ta_panel)) {
      if (omp_get_num_threads() == 1) {
        LOG_INST.info(Loc::current(), "The task ", ta_panel.get_curr_task_idx(), " reroute by ",
                      GetTARouteStrategyName()(ta_route_strategy), " successfully!");
      }
    } else if (ta_route_strategy == TARouteStrategy::kIgnoringBlockage) {
      LOG_INST.error(Loc::current(), "The task ", ta_panel.get_curr_task_idx(), " reroute by ", GetTARouteStrategyName()(ta_route_strategy),
                     " failed!");
    }
  }
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
    if (!RTUtil::exist(ta_panel.get_routing_offset_set(), orientation)) {
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
  ta_panel.set_ta_route_strategy(TARouteStrategy::kNone);

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
    // 初始化时，要把start_node_comb_list的pin只留一个ap点
    // 后续只要将end_node_comb_list的pin保留一个ap点
    start_node_comb_list.front().clear();
    start_node_comb_list.front().push_back(path_node);
  }
  start_node_comb_list.push_back(end_node_comb_list[end_node_comb_idx]);
  end_node_comb_list.erase(end_node_comb_list.begin() + end_node_comb_idx);
}

void TrackAssigner::updateNetResult(TAPanel& ta_panel, TATask& ta_task)
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
  bool exist_neighbor = false;
  for (auto& [orientation, neighbor_ptr] : start_node->get_neighbor_ptr_map()) {
    if (neighbor_ptr == end_node) {
      exist_neighbor = true;
      break;
    }
  }
  if (!exist_neighbor) {
    LOG_INST.info(Loc::current(), "The neighbor not exist!");
  }

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
  return task_cost;
}

double TrackAssigner::getKnowWireCost(TAPanel& ta_panel, TANode* start_node, TANode* end_node)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  double wire_cost = 0;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    wire_cost += RTUtil::getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());

    RoutingLayer& routing_layer = routing_layer_list[start_node->get_layer_idx()];
    if (routing_layer.get_direction() != RTUtil::getDirection(*start_node, *end_node)) {
      wire_cost *= 2;
    }
  }
  wire_cost *= ta_panel.get_wire_unit();
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
  double wire_cost = 0;
  wire_cost += RTUtil::getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());
  wire_cost *= ta_panel.get_wire_unit();
  return wire_cost;
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

#if 1  // count ta_panel

void TrackAssigner::countTAPanel(TAPanel& ta_panel)
{
  irt_int micron_dbu = DM_INST.getDatabase().get_micron_dbu();

  TAPanelStat& ta_panel_stat = ta_panel.get_ta_panel_stat();

  double total_wire_length = 0;
  for (TATask& ta_task : ta_panel.get_ta_task_list()) {
    for (Segment<LayerCoord>& routing_segment : ta_task.get_routing_segment_list()) {
      irt_int first_layer_idx = routing_segment.get_first().get_layer_idx();
      irt_int second_layer_idx = routing_segment.get_second().get_layer_idx();
      if (first_layer_idx != second_layer_idx) {
        LOG_INST.error(Loc::current(), "The layer of TA Segment is different!");
      }
      total_wire_length += RTUtil::getManhattanDistance(routing_segment.get_first(), routing_segment.get_second()) / 1.0 / micron_dbu;
    }
  }
  ta_panel_stat.set_total_wire_length(total_wire_length);

  std::map<TASourceType, std::map<std::string, irt_int>>& source_drc_number_map = ta_panel_stat.get_source_drc_number_map();
  for (TATask& ta_task : ta_panel.get_ta_task_list()) {
    std::vector<LayerRect> real_rect_list = DM_INST.getRealRectList(ta_task.get_routing_segment_list());
    for (auto& [source, region_query] : ta_panel.get_source_region_query_map()) {
      std::map<std::string, irt_int> drc_number_map;
      if (source == TASourceType::kSelfPanelResult) {
        drc_number_map = RTAPI_INST.getViolation(region_query);
      } else {
        drc_number_map = RTAPI_INST.getViolation(region_query, real_rect_list);
      }
      for (auto& [drc, number] : drc_number_map) {
        source_drc_number_map[source][drc] += number;
      }
    }
  }

  irt_int total_drc_number = 0;
  for (auto& [source, drc_number_map] : source_drc_number_map) {
    for (auto& [drc, number] : drc_number_map) {
      total_drc_number += number;
    }
  }
  ta_panel_stat.set_total_drc_number(total_drc_number);
}

#endif

#if 1  // plot ta_panel

void TrackAssigner::plotTAPanel(TAPanel& ta_panel, irt_int curr_task_idx)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::string ta_temp_directory_path = DM_INST.getConfig().ta_temp_directory_path;

  irt_int width = INT_MAX;
  for (ScaleGrid& x_grid : ta_panel.get_panel_scale_axis().get_x_grid_list()) {
    width = std::min(width, x_grid.get_step_length());
  }
  for (ScaleGrid& y_grid : ta_panel.get_panel_scale_axis().get_y_grid_list()) {
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
      GPText gp_text_obs_task_map;
      gp_text_obs_task_map.set_coord(real_rect.get_lb_x(), y);
      gp_text_obs_task_map.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
      gp_text_obs_task_map.set_message("obs_task_map: ");
      gp_text_obs_task_map.set_layer_idx(GP_INST.getGDSIdxByRouting(ta_node.get_layer_idx()));
      gp_text_obs_task_map.set_presentation(GPTextPresentation::kLeftMiddle);
      ta_node_map_struct.push(gp_text_obs_task_map);

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
        ta_node_map_struct.push(gp_text_obs_task_map_info);
      }

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

  // net_blockage_map
  for (auto& [net_idx, blockage_list] : ta_panel.get_source_net_rect_map()[TASourceType::kBlockage]) {
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

  // net_other_panel_result_map
  for (auto& [net_idx, other_panel_result_list] : ta_panel.get_source_net_rect_map()[TASourceType::kOtherPanelResult]) {
    GPStruct other_panel_result_struct(RTUtil::getString("other_panel_result@", net_idx));
    for (const LayerRect& other_panel_result : other_panel_result_list) {
      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kOtherPanelResult));
      gp_boundary.set_rect(other_panel_result);
      gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(other_panel_result.get_layer_idx()));
      other_panel_result_struct.push(gp_boundary);
    }
    gp_gds.addStruct(other_panel_result_struct);
  }

  // net_self_panel_result_map
  for (auto& [net_idx, self_panel_result_list] : ta_panel.get_source_net_rect_map()[TASourceType::kSelfPanelResult]) {
    GPStruct self_panel_result_struct(RTUtil::getString("self_panel_result@", net_idx));
    for (const LayerRect& self_panel_result : self_panel_result_list) {
      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kSelfPanelResult));
      gp_boundary.set_rect(self_panel_result);
      gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(self_panel_result.get_layer_idx()));
      self_panel_result_struct.push(gp_boundary);
    }
    gp_gds.addStruct(self_panel_result_struct);
  }

  // panel_scale_axis
  GPStruct panel_scale_axis_struct("panel_scale_axis");
  ScaleAxis& panel_scale_axis = ta_panel.get_panel_scale_axis();
  std::vector<irt_int> x_list = RTUtil::getClosedScaleList(ta_panel.get_lb_x(), ta_panel.get_rt_x(), panel_scale_axis.get_x_grid_list());
  std::vector<irt_int> y_list = RTUtil::getClosedScaleList(ta_panel.get_lb_y(), ta_panel.get_rt_y(), panel_scale_axis.get_y_grid_list());
  for (irt_int x : x_list) {
    GPPath gp_path;
    gp_path.set_data_type(static_cast<irt_int>(GPGraphType::kScaleAxis));
    gp_path.set_segment(x, ta_panel.get_lb_y(), x, ta_panel.get_rt_y());
    gp_path.set_layer_idx(GP_INST.getGDSIdxByRouting(ta_panel.get_layer_idx()));
    panel_scale_axis_struct.push(gp_path);
  }
  for (irt_int y : y_list) {
    GPPath gp_path;
    gp_path.set_data_type(static_cast<irt_int>(GPGraphType::kScaleAxis));
    gp_path.set_segment(ta_panel.get_lb_x(), y, ta_panel.get_rt_x(), y);
    gp_path.set_layer_idx(GP_INST.getGDSIdxByRouting(ta_panel.get_layer_idx()));
    panel_scale_axis_struct.push(gp_path);
  }
  gp_gds.addStruct(panel_scale_axis_struct);

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
      gp_boundary.set_data_type(1);
      gp_boundary.set_rect(ta_task.get_bounding_box());
      task_struct.push(gp_boundary);
    }
    for (Segment<LayerCoord>& segment : ta_task.get_routing_segment_list()) {
      LayerCoord first_coord = segment.get_first();
      irt_int first_layer_idx = first_coord.get_layer_idx();
      LayerCoord second_coord = segment.get_second();
      irt_int second_layer_idx = second_coord.get_layer_idx();
      irt_int half_width = routing_layer_list[first_layer_idx].get_min_width() / 2;

      if (first_layer_idx == second_layer_idx) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kPath));
        gp_boundary.set_rect(RTUtil::getEnlargedRect(first_coord, second_coord, half_width));
        gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(first_layer_idx));
        task_struct.push(gp_boundary);
      } else {
        RTUtil::sortASC(first_layer_idx, second_layer_idx);
        for (irt_int layer_idx = first_layer_idx; layer_idx <= second_layer_idx; layer_idx++) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kPath));
          gp_boundary.set_rect(RTUtil::getEnlargedRect(first_coord, half_width));
          gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(layer_idx));
          task_struct.push(gp_boundary);
        }
      }
    }
    gp_gds.addStruct(task_struct);
  }
  std::string gds_file_path
      = RTUtil::getString(ta_temp_directory_path, "ta_panel_", ta_panel.get_layer_idx(), "_", ta_panel.get_panel_idx(), ".gds");
  GP_INST.plot(gp_gds, gds_file_path, false, false);
}

#endif

#if 1  // update ta_panel

void TrackAssigner::updateTAPanel(TAModel& ta_model, TAPanel& ta_panel)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  EXTPlanarRect& die = DM_INST.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  std::vector<std::vector<TAPanel>>& layer_panel_list = ta_model.get_layer_panel_list();

  for (TATask& ta_task : ta_panel.get_ta_task_list()) {
    for (LayerRect& real_rect : DM_INST.getRealRectList(ta_task.get_routing_segment_list())) {
      irt_int real_rect_layer_idx = real_rect.get_layer_idx();
      for (const LayerRect& max_scope_real_rect : RTAPI_INST.getMaxScope(real_rect)) {
        LayerRect max_scope_regular_rect = RTUtil::getRegularRect(max_scope_real_rect, die.get_real_rect());
        PlanarRect max_scope_grid_rect = RTUtil::getClosedGridRect(max_scope_regular_rect, gcell_axis);
        if (routing_layer_list[real_rect_layer_idx].isPreferH()) {
          for (irt_int y = max_scope_grid_rect.get_lb_y(); y <= max_scope_grid_rect.get_rt_y(); y++) {
            TAPanel& target_panel = layer_panel_list[real_rect_layer_idx][y];
            if (target_panel.get_layer_idx() == ta_panel.get_layer_idx() && target_panel.get_panel_idx() == ta_panel.get_panel_idx()) {
              target_panel.addRect(TASourceType::kSelfPanelResult, ta_task.get_origin_net_idx(), real_rect);
            } else {
              target_panel.addRect(TASourceType::kOtherPanelResult, ta_task.get_origin_net_idx(), real_rect);
            }
          }
        } else {
          for (irt_int x = max_scope_grid_rect.get_lb_x(); x <= max_scope_grid_rect.get_rt_x(); x++) {
            TAPanel& target_panel = layer_panel_list[real_rect_layer_idx][x];
            if (target_panel.get_layer_idx() == ta_panel.get_layer_idx() && target_panel.get_panel_idx() == ta_panel.get_panel_idx()) {
              target_panel.addRect(TASourceType::kSelfPanelResult, ta_task.get_origin_net_idx(), real_rect);
            } else {
              target_panel.addRect(TASourceType::kOtherPanelResult, ta_task.get_origin_net_idx(), real_rect);
            }
          }
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
  std::map<irt_int, double>& routing_wire_length_map = ta_model_stat.get_routing_wire_length_map();
  std::map<TASourceType, std::map<std::string, irt_int>>& source_drc_number_map = ta_model_stat.get_source_drc_number_map();

  for (std::vector<TAPanel>& ta_panel_list : ta_model.get_layer_panel_list()) {
    for (TAPanel& ta_panel : ta_panel_list) {
      routing_wire_length_map[ta_panel.get_layer_idx()] += ta_panel.get_ta_panel_stat().get_total_wire_length();
    }
  }
  for (std::vector<TAPanel>& ta_panel_list : ta_model.get_layer_panel_list()) {
    for (TAPanel& ta_panel : ta_panel_list) {
      TAPanelStat& ta_panel_stat = ta_panel.get_ta_panel_stat();
      for (auto& [source, drc_number_map] : ta_panel_stat.get_source_drc_number_map()) {
        for (auto& [drc, number] : drc_number_map) {
          source_drc_number_map[source][drc] += number;
        }
      }
    }
  }

  double total_wire_length = 0;
  irt_int total_drc_number = 0;
  for (auto& [routing_layer_idx, wire_length] : routing_wire_length_map) {
    total_wire_length += wire_length;
  }
  for (auto& [source, drc_number_map] : source_drc_number_map) {
    for (auto& [drc, number] : drc_number_map) {
      total_drc_number += number;
    }
  }
  ta_model_stat.set_total_wire_length(total_wire_length);
  ta_model_stat.set_total_drc_number(total_drc_number);
}

void TrackAssigner::reportTable(TAModel& ta_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  TAModelStat& ta_model_stat = ta_model.get_ta_model_stat();
  std::map<irt_int, double>& routing_wire_length_map = ta_model_stat.get_routing_wire_length_map();
  std::map<TASourceType, std::map<std::string, irt_int>>& source_drc_number_map = ta_model_stat.get_source_drc_number_map();
  double total_wire_length = ta_model_stat.get_total_wire_length();
  irt_int total_drc_number  = ta_model_stat.get_total_drc_number();

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
  for (std::string table_str : RTUtil::splitString(wire_table.to_string(), '\n')) {
    LOG_INST.info(Loc::current(), table_str);
  }
  // report drc info
  fort::char_table drc_table;
  drc_table.set_border_style(FT_SOLID_STYLE);
  drc_table << fort::header << "Source"
            << "DRC"
            << "Number" << fort::endr;
  for (auto& [source, drc_number_map] : source_drc_number_map) {
    for (auto& [drc, number] : drc_number_map) {
      drc_table << GetTASourceTypeName()(source) << drc
                << RTUtil::getString(number, "(", RTUtil::getPercentage(number, total_drc_number), "%") << fort::endr;
    }
  }
  drc_table << fort::header << "Total"
            << "Total" << total_drc_number << fort::endr;
  for (std::string table_str : RTUtil::splitString(drc_table.to_string(), '\n')) {
    LOG_INST.info(Loc::current(), table_str);
  }
}

#endif

}  // namespace irt
