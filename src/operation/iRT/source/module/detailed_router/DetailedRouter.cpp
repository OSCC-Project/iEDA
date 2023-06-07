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
#include "DetailedRouter.hpp"

#include "DRBox.hpp"
#include "DRNet.hpp"
#include "DRNode.hpp"
#include "DetailedRouter.hpp"
#include "GDSPlotter.hpp"

namespace irt {

// public

void DetailedRouter::initInst(Config& config, Database& database)
{
  if (_dr_instance == nullptr) {
    _dr_instance = new DetailedRouter(config, database);
  }
}

DetailedRouter& DetailedRouter::getInst()
{
  if (_dr_instance == nullptr) {
    LOG_INST.error(Loc::current(), "The instance not initialized!");
  }
  return *_dr_instance;
}

void DetailedRouter::destroyInst()
{
  if (_dr_instance != nullptr) {
    delete _dr_instance;
    _dr_instance = nullptr;
  }
}

void DetailedRouter::route(std::vector<Net>& net_list)
{
  Monitor monitor;

  std::vector<DRNet> dr_net_list = _dr_data_manager.convertToDRNetList(net_list);
  routeDRNetList(dr_net_list);

  LOG_INST.info(Loc::current(), "The ", GetStageName()(Stage::kDetailedRouter), " completed!", monitor.getStatsInfo());
}

// private

DetailedRouter* DetailedRouter::_dr_instance = nullptr;

void DetailedRouter::init(Config& config, Database& database)
{
  _dr_data_manager.input(config, database);
}

void DetailedRouter::routeDRNetList(std::vector<DRNet>& dr_net_list)
{
  DRModel dr_model = initDRModel(dr_net_list);
  buildDRModel(dr_model);
  routeDRModel(dr_model);
  updateDRModel(dr_model);
  reportDRModel(dr_model);
}

#if 1  // build dr_model

DRModel DetailedRouter::initDRModel(std::vector<DRNet>& dr_net_list)
{
  GCellAxis& gcell_axis = _dr_data_manager.getDatabase().get_gcell_axis();

  DRModel dr_model;
  dr_model.set_dr_net_list(dr_net_list);

  irt_int x_gcell_num = 0;
  for (GCellGrid& x_grid : gcell_axis.get_x_grid_list()) {
    x_gcell_num += x_grid.get_step_num();
  }
  irt_int y_gcell_num = 0;
  for (GCellGrid& y_grid : gcell_axis.get_y_grid_list()) {
    y_gcell_num += y_grid.get_step_num();
  }
  GridMap<DRBox>& dr_box_map = dr_model.get_dr_box_map();
  dr_box_map.init(x_gcell_num, y_gcell_num);
  for (irt_int x = 0; x < dr_box_map.get_x_size(); x++) {
    for (irt_int y = 0; y < dr_box_map.get_y_size(); y++) {
      dr_box_map[x][y].set_grid_coord(PlanarCoord(x, y));
    }
  }
  return dr_model;
}

void DetailedRouter::buildDRModel(DRModel& dr_model)
{
  addBlockageList(dr_model);
  addNetRegionList(dr_model);
  buildDRTaskList(dr_model);
  buildDRTaskPriority(dr_model);
}

void DetailedRouter::addBlockageList(DRModel& dr_model)
{
  GCellAxis& gcell_axis = _dr_data_manager.getDatabase().get_gcell_axis();
  EXTPlanarRect& die = _dr_data_manager.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = _dr_data_manager.getDatabase().get_routing_layer_list();
  std::vector<Blockage>& routing_blockage_list = _dr_data_manager.getDatabase().get_routing_blockage_list();

  GridMap<DRBox>& dr_box_map = dr_model.get_dr_box_map();

  for (const Blockage& routing_blockage : routing_blockage_list) {
    irt_int layer_idx = routing_blockage.get_layer_idx();
    irt_int min_spacing = routing_layer_list[layer_idx].getMinSpacing(routing_blockage.get_real_rect());
    PlanarRect enlarged_real_rect = RTUtil::getEnlargedRect(routing_blockage.get_real_rect(), min_spacing, die.get_real_rect());
    PlanarRect enlarged_grid_rect = RTUtil::getClosedGridRect(enlarged_real_rect, gcell_axis);
    for (irt_int x = enlarged_grid_rect.get_lb_x(); x <= enlarged_grid_rect.get_rt_x(); x++) {
      for (irt_int y = enlarged_grid_rect.get_lb_y(); y <= enlarged_grid_rect.get_rt_y(); y++) {
        dr_box_map[x][y].get_net_blockage_map()[-1].emplace_back(enlarged_real_rect, layer_idx);
      }
    }
  }
  for (DRNet& dr_net : dr_model.get_dr_net_list()) {
    for (DRPin& dr_pin : dr_net.get_dr_pin_list()) {
      for (const EXTLayerRect& routing_shape : dr_pin.get_routing_shape_list()) {
        irt_int layer_idx = routing_shape.get_layer_idx();
        irt_int min_spacing = routing_layer_list[layer_idx].getMinSpacing(routing_shape.get_real_rect());
        PlanarRect enlarged_real_rect = RTUtil::getEnlargedRect(routing_shape.get_real_rect(), min_spacing, die.get_real_rect());
        PlanarRect enlarged_grid_rect = RTUtil::getClosedGridRect(enlarged_real_rect, gcell_axis);
        for (irt_int x = enlarged_grid_rect.get_lb_x(); x <= enlarged_grid_rect.get_rt_x(); x++) {
          for (irt_int y = enlarged_grid_rect.get_lb_y(); y <= enlarged_grid_rect.get_rt_y(); y++) {
            dr_box_map[x][y].get_net_blockage_map()[dr_net.get_net_idx()].emplace_back(enlarged_real_rect, layer_idx);
          }
        }
      }
    }
    for (TNode<RTNode>* ta_node_node : RTUtil::getNodeList(dr_net.get_ta_result_tree())) {
      if (ta_node_node->value().isDRNode()) {
        continue;
      }
      irt_int layer_idx = ta_node_node->value().get_first_guide().get_layer_idx();
      irt_int half_width = routing_layer_list[layer_idx].get_min_width() / 2;
      for (Segment<TNode<LayerCoord>*>& routing_segment : RTUtil::getSegListByTree(ta_node_node->value().get_routing_tree())) {
        PlanarRect real_rect
            = RTUtil::getEnlargedRect(routing_segment.get_first()->value(), routing_segment.get_second()->value(), half_width);

        irt_int min_spacing = routing_layer_list[layer_idx].getMinSpacing(real_rect);
        PlanarRect enlarged_real_rect = RTUtil::getEnlargedRect(real_rect, min_spacing, die.get_real_rect());
        PlanarRect enlarged_grid_rect = RTUtil::getClosedGridRect(enlarged_real_rect, gcell_axis);
        for (irt_int x = enlarged_grid_rect.get_lb_x(); x <= enlarged_grid_rect.get_rt_x(); x++) {
          for (irt_int y = enlarged_grid_rect.get_lb_y(); y <= enlarged_grid_rect.get_rt_y(); y++) {
            dr_box_map[x][y].get_net_blockage_map()[dr_net.get_net_idx()].emplace_back(enlarged_real_rect, layer_idx);
          }
        }
      }
    }
  }
}

void DetailedRouter::addNetRegionList(DRModel& dr_model)
{
  GCellAxis& gcell_axis = _dr_data_manager.getDatabase().get_gcell_axis();
  EXTPlanarRect& die = _dr_data_manager.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = _dr_data_manager.getDatabase().get_routing_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _dr_data_manager.getDatabase().get_layer_via_master_list();
  irt_int bottom_routing_layer_idx = _dr_data_manager.getConfig().bottom_routing_layer_idx;
  irt_int top_routing_layer_idx = _dr_data_manager.getConfig().top_routing_layer_idx;

  GridMap<DRBox>& dr_box_map = dr_model.get_dr_box_map();

  for (DRNet& dr_net : dr_model.get_dr_net_list()) {
    std::vector<EXTLayerRect> net_region_list;
    for (DRPin& dr_pin : dr_net.get_dr_pin_list()) {
      std::vector<EXTLayerRect>& routing_shape_list = dr_pin.get_routing_shape_list();
      net_region_list.insert(net_region_list.end(), routing_shape_list.begin(), routing_shape_list.end());
      for (LayerCoord& real_coord : dr_pin.getRealCoordList()) {
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
      irt_int min_spacing = routing_layer_list[layer_idx].getMinSpacing(net_region.get_real_rect());
      PlanarRect enlarged_real_rect = RTUtil::getEnlargedRect(net_region.get_real_rect(), min_spacing, die.get_real_rect());
      PlanarRect enlarged_grid_rect = RTUtil::getClosedGridRect(enlarged_real_rect, gcell_axis);
      for (irt_int x = enlarged_grid_rect.get_lb_x(); x <= enlarged_grid_rect.get_rt_x(); x++) {
        for (irt_int y = enlarged_grid_rect.get_lb_y(); y <= enlarged_grid_rect.get_rt_y(); y++) {
          dr_box_map[x][y].get_net_region_map()[dr_net.get_net_idx()].emplace_back(enlarged_real_rect, layer_idx);
        }
      }
    }
  }
}

void DetailedRouter::buildDRTaskList(DRModel& dr_model)
{
  GridMap<DRBox>& dr_box_map = dr_model.get_dr_box_map();

  for (DRNet& dr_net : dr_model.get_dr_net_list()) {
    for (auto& [dr_node_node, dr_task] : makeDRNodeTaskMap(dr_net)) {
      PlanarCoord& grid_coord = dr_node_node->value().get_first_guide().get_grid_coord();
      DRBox& dr_box = dr_box_map[grid_coord.get_x()][grid_coord.get_y()];

      std::vector<DRTask>& dr_task_list = dr_box.get_dr_task_list();
      dr_task.set_origin_node(dr_node_node);
      dr_task.set_task_idx(static_cast<irt_int>(dr_task_list.size()));
      dr_task.set_origin_net_idx(dr_net.get_net_idx());
      dr_task.set_connect_type(dr_net.get_connect_type());
      buildBoundingBox(dr_box, dr_task);
      dr_task_list.push_back(dr_task);
    }
  }
}

std::map<TNode<RTNode>*, DRTask> DetailedRouter::makeDRNodeTaskMap(DRNet& dr_net)
{
  std::map<TNode<RTNode>*, DRTask> dr_node_task_map;
  for (auto& [dr_node_node, ta_node_node_list] : getDRTAListMap(dr_net)) {
    RTNode& dr_node = dr_node_node->value();
    PlanarRect dr_box_region = dr_node_node->value().get_first_guide().get_rect();

    std::vector<DRGroup> dr_group_list;
    for (irt_int pin_idx : dr_node.get_pin_idx_set()) {
      DRGroup dr_group;
      for (LayerCoord& real_coord : dr_net.get_dr_pin_list()[pin_idx].getRealCoordList()) {
        if (RTUtil::isInside(dr_box_region, real_coord)) {
          dr_group.get_coord_list().push_back(real_coord);
        }
      }
      dr_group_list.push_back(dr_group);
    }
    for (TNode<RTNode>* ta_node_node : ta_node_node_list) {
      dr_group_list.push_back(makeDRGroup(dr_node_node, ta_node_node));
    }
    dr_node_task_map[dr_node_node].set_dr_group_list(dr_group_list);
  }
  return dr_node_task_map;
}

std::map<TNode<RTNode>*, std::vector<TNode<RTNode>*>> DetailedRouter::getDRTAListMap(DRNet& dr_net)
{
  std::map<TNode<RTNode>*, std::vector<TNode<RTNode>*>> dr_ta_list_map;

  MTree<RTNode>& dr_result_tree = dr_net.get_dr_result_tree();
  std::vector<Segment<TNode<RTNode>*>> segment_list = RTUtil::getSegListByTree(dr_result_tree);
  if (dr_result_tree.get_root() != nullptr && segment_list.empty()) {
    // local net
    dr_ta_list_map[dr_result_tree.get_root()] = {};
  }
  for (Segment<TNode<RTNode>*>& segment : segment_list) {
    TNode<RTNode>* dr_node_node = segment.get_first();
    TNode<RTNode>* ta_node_node = segment.get_second();
    if (dr_node_node->value().isTANode()) {
      std::swap(dr_node_node, ta_node_node);
    }
    dr_ta_list_map[dr_node_node].push_back(ta_node_node);
  }
  return dr_ta_list_map;
}

DRGroup DetailedRouter::makeDRGroup(TNode<RTNode>* dr_node_node, TNode<RTNode>* ta_node_node)
{
  std::vector<RoutingLayer>& routing_layer_list = _dr_data_manager.getDatabase().get_routing_layer_list();

  PlanarRect dr_box_region = dr_node_node->value().get_first_guide().get_rect();
  RTNode& ta_node = ta_node_node->value();
  irt_int ta_layer_idx = ta_node.get_first_guide().get_layer_idx();
  RoutingLayer& routing_layer = routing_layer_list[ta_layer_idx];
  TrackGrid& x_track_grid = routing_layer.getXTrackGrid();
  TrackGrid& y_track_grid = routing_layer.getYTrackGrid();

  DRGroup dr_group;
  for (Segment<TNode<LayerCoord>*>& routing_segment : RTUtil::getSegListByTree(ta_node.get_routing_tree())) {
    Segment<PlanarCoord> cutting_segment(routing_segment.get_first()->value(), routing_segment.get_second()->value());
    if (!RTUtil::isOverlap(dr_box_region, cutting_segment)) {
      continue;
    }
    cutting_segment = RTUtil::getOverlap(dr_box_region, cutting_segment);
    PlanarCoord& first_coord = cutting_segment.get_first();
    irt_int first_x = first_coord.get_x();
    irt_int first_y = first_coord.get_y();
    PlanarCoord& second_coord = cutting_segment.get_second();
    if (RTUtil::isHorizontal(first_coord, second_coord)) {
      for (irt_int x : RTUtil::getClosedScaleList(first_x, second_coord.get_x(), x_track_grid)) {
        dr_group.get_coord_list().emplace_back(x, first_y, ta_layer_idx);
      }
    } else if (RTUtil::isVertical(first_coord, second_coord)) {
      for (irt_int y : RTUtil::getClosedScaleList(first_y, second_coord.get_y(), y_track_grid)) {
        dr_group.get_coord_list().emplace_back(first_x, y, ta_layer_idx);
      }
    } else if (RTUtil::isProximal(first_coord, second_coord)) {
      dr_group.get_coord_list().emplace_back(first_coord, ta_layer_idx);
    }
  }
  return dr_group;
}

void DetailedRouter::buildBoundingBox(DRBox& dr_box, DRTask& dr_task)
{
  std::vector<RoutingLayer>& routing_layer_list = _dr_data_manager.getDatabase().get_routing_layer_list();
  GCellAxis& gcell_axis = _dr_data_manager.getDatabase().get_gcell_axis();

  std::vector<PlanarCoord> coord_list;
  for (DRGroup& dr_group : dr_task.get_dr_group_list()) {
    for (LayerCoord& coord : dr_group.get_coord_list()) {
      coord_list.push_back(coord);
    }
  }
  PlanarRect real_bounding_box = RTUtil::getBoundingBox(coord_list);

  PlanarRect dr_box_region = RTUtil::getRealRect(dr_box.get_grid_coord(), gcell_axis);
  std::vector<PlanarRect> layer_bounding_box;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    irt_int lb_x = RTUtil::getFloorTrackLine(real_bounding_box.get_lb_x(), routing_layer.getXTrackGrid());
    irt_int lb_y = RTUtil::getFloorTrackLine(real_bounding_box.get_lb_y(), routing_layer.getYTrackGrid());
    irt_int rt_x = RTUtil::getCeilTrackLine(real_bounding_box.get_rt_x(), routing_layer.getXTrackGrid());
    irt_int rt_y = RTUtil::getCeilTrackLine(real_bounding_box.get_rt_y(), routing_layer.getYTrackGrid());
    layer_bounding_box.emplace_back(lb_x, lb_y, rt_x, rt_y);
  }
  dr_task.set_bounding_box(RTUtil::getBoundingBox(layer_bounding_box, dr_box_region));
}

void DetailedRouter::buildDRTaskPriority(DRModel& dr_model)
{
  GridMap<DRBox>& dr_box_map = dr_model.get_dr_box_map();
  for (irt_int x = 0; x < dr_box_map.get_x_size(); x++) {
    for (irt_int y = 0; y < dr_box_map.get_y_size(); y++) {
      for (DRTask& dr_task : dr_box_map[x][y].get_dr_task_list()) {
        DRTaskPriority& dr_task_priority = dr_task.get_dr_task_priority();
        // connect_type
        dr_task_priority.set_connect_type(dr_task.get_connect_type());

        PlanarRect& bounding_box = dr_task.get_bounding_box();
        irt_int width = std::max(1, bounding_box.getWidth());
        // routing_area
        dr_task_priority.set_routing_area(bounding_box.getLength() * width);
        // length_width_ratio
        dr_task_priority.set_length_width_ratio(bounding_box.getLength() / 1.0 / width);
        // pin_num
        dr_task_priority.set_pin_num(static_cast<irt_int>(dr_task.get_dr_group_list().size()));
      }
    }
  }
}

#endif

#if 1  // route dr_model

void DetailedRouter::routeDRModel(DRModel& dr_model)
{
  Monitor monitor;

  GridMap<DRBox>& dr_box_map = dr_model.get_dr_box_map();

  irt_int box_x_size = dr_box_map.get_x_size();
  irt_int box_y_size = dr_box_map.get_y_size();
  irt_int box_size = box_x_size * box_y_size;
  irt_int range = std::max(2, static_cast<irt_int>(std::sqrt(box_size / RTUtil::getBatchSize(box_size))));

  for (irt_int start_x = 0; start_x < range; start_x++) {
    for (irt_int start_y = 0; start_y < range; start_y++) {
      Monitor stage_monitor;
#pragma omp parallel for collapse(2)
      for (irt_int x = start_x; x < box_x_size; x += range) {
        for (irt_int y = start_y; y < box_y_size; y += range) {
          DRBox& dr_box = dr_box_map[x][y];
          if (dr_box.skipRouting()) {
            continue;
          }
          buildDRBox(dr_box);
          checkDRBox(dr_box);
          sortDRBox(dr_box);
          routeDRBox(dr_box);
          countDRBox(dr_box);
          dr_box.freeNodeGraph();
        }
      }
      irt_int stage_box_x = static_cast<irt_int>(std::ceil((box_x_size - start_x) / 1.0 / range));
      irt_int stage_box_y = static_cast<irt_int>(std::ceil((box_y_size - start_y) / 1.0 / range));
      LOG_INST.info(Loc::current(), "Processed ", stage_box_x * stage_box_y, " boxs", stage_monitor.getStatsInfo());
    }
  }
  LOG_INST.info(Loc::current(), "Processed ", box_size, " boxs", monitor.getStatsInfo());
}

#endif

#if 1  // build dr_box

void DetailedRouter::buildDRBox(DRBox& dr_box)
{
  initLayerGraphList(dr_box);
  buildScaleOrientList(dr_box);
  buildBasicLayerGraph(dr_box);
  buildCrossLayerGraph(dr_box);
  buildLayerNodeList(dr_box);
  buildOBSTaskMap(dr_box);
  buildCostTaskMap(dr_box);
}

void DetailedRouter::initLayerGraphList(DRBox& dr_box)
{
  std::vector<RoutingLayer>& routing_layer_list = _dr_data_manager.getDatabase().get_routing_layer_list();

  for (RoutingLayer& routing_layer : routing_layer_list) {
    DRNodeGraph node_graph;
    node_graph.set_layer_idx(routing_layer.get_layer_idx());
    dr_box.get_layer_graph_list().push_back(node_graph);
  }
}

void DetailedRouter::buildScaleOrientList(DRBox& dr_box)
{
  GCellAxis& gcell_axis = _dr_data_manager.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = _dr_data_manager.getDatabase().get_routing_layer_list();
  irt_int bottom_routing_layer_idx = _dr_data_manager.getConfig().bottom_routing_layer_idx;
  irt_int top_routing_layer_idx = _dr_data_manager.getConfig().top_routing_layer_idx;

  std::vector<DRNodeGraph>& layer_graph_list = dr_box.get_layer_graph_list();

  PlanarRect dr_box_region = RTUtil::getRealRect(dr_box.get_grid_coord(), gcell_axis);
  for (DRNodeGraph& node_graph : layer_graph_list) {
    irt_int layer_idx = node_graph.get_layer_idx();
    if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
      continue;
    }
    RoutingLayer& routing_layer = routing_layer_list[layer_idx];

#if 1
    if (routing_layer.isPreferH()) {
      node_graph.set_y_scale_list(
          RTUtil::getClosedScaleList(dr_box_region.get_lb_y(), dr_box_region.get_rt_y(), routing_layer.getYTrackGrid()));
    } else {
      node_graph.set_x_scale_list(
          RTUtil::getClosedScaleList(dr_box_region.get_lb_x(), dr_box_region.get_rt_x(), routing_layer.getXTrackGrid()));
    }
#elif
    node_graph.set_x_scale_list(
        RTUtil::getClosedScaleList(dr_box_region.get_lb_x(), dr_box_region.get_rt_x(), routing_layer.getXTrackGrid()));
    node_graph.set_y_scale_list(
        RTUtil::getClosedScaleList(dr_box_region.get_lb_y(), dr_box_region.get_rt_y(), routing_layer.getYTrackGrid()));
#endif
  }

  for (DRNodeGraph& node_graph : layer_graph_list) {
    irt_int curr_layer_idx = node_graph.get_layer_idx();
    if (curr_layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < curr_layer_idx) {
      continue;
    }
    std::vector<DRScaleOrient>& x_scale_orient_list = node_graph.get_x_scale_orient_list();
    std::vector<DRScaleOrient>& y_scale_orient_list = node_graph.get_y_scale_orient_list();
    // curr x
    for (irt_int x_scale : node_graph.get_x_scale_list()) {
      x_scale_orient_list.emplace_back(x_scale);
    }
    // curr y
    for (irt_int y_scale : node_graph.get_y_scale_list()) {
      y_scale_orient_list.emplace_back(y_scale);
    }
    // above xy
    irt_int above_layer_idx = curr_layer_idx + 1;
    if (above_layer_idx <= top_routing_layer_idx) {
      for (irt_int x : layer_graph_list[above_layer_idx].get_x_scale_list()) {
        x_scale_orient_list.emplace_back(x, Orientation::kUp);
      }
      for (irt_int y : layer_graph_list[above_layer_idx].get_y_scale_list()) {
        y_scale_orient_list.emplace_back(y, Orientation::kUp);
      }
    }
    // below xy
    irt_int below_layer_idx = curr_layer_idx - 1;
    if (bottom_routing_layer_idx <= below_layer_idx) {
      for (irt_int x : layer_graph_list[below_layer_idx].get_x_scale_list()) {
        x_scale_orient_list.emplace_back(x, Orientation::kDown);
      }
      for (irt_int y : layer_graph_list[below_layer_idx].get_y_scale_list()) {
        y_scale_orient_list.emplace_back(y, Orientation::kDown);
      }
    }
    // sort
    auto Cmp_pair = [](DRScaleOrient& a, DRScaleOrient& b) { return a.get_scale() < b.get_scale(); };
    std::sort(x_scale_orient_list.begin(), x_scale_orient_list.end(), Cmp_pair);
    std::sort(y_scale_orient_list.begin(), y_scale_orient_list.end(), Cmp_pair);
    // merge
    auto merge_pair = [](DRScaleOrient& sentry, DRScaleOrient& soldier) {
      if (sentry.get_scale() != soldier.get_scale()) {
        return false;
      }
      std::set<Orientation>& sentry_orientation_set = sentry.get_orientation_set();
      std::set<Orientation>& soldier_orientation_set = soldier.get_orientation_set();
      sentry_orientation_set.insert(soldier_orientation_set.begin(), soldier_orientation_set.end());
      return true;
    };
    RTUtil::merge(x_scale_orient_list, merge_pair);
    RTUtil::merge(y_scale_orient_list, merge_pair);
  }
}

void DetailedRouter::buildBasicLayerGraph(DRBox& dr_box)
{
  std::vector<DRNodeGraph>& layer_graph_list = dr_box.get_layer_graph_list();

  for (DRNodeGraph& node_graph : layer_graph_list) {
    irt_int curr_layer_idx = node_graph.get_layer_idx();
    std::vector<DRScaleOrient>& x_scale_orient_list = node_graph.get_x_scale_orient_list();
    std::vector<DRScaleOrient>& y_scale_orient_list = node_graph.get_y_scale_orient_list();

    for (irt_int x_scale : node_graph.get_x_scale_list()) {
      for (size_t i = 0; i < y_scale_orient_list.size(); i++) {
        LayerCoord curr_coord(x_scale, y_scale_orient_list[i].get_scale(), curr_layer_idx);
        std::map<Orientation, LayerCoord>& neighbor_map = node_graph.get_coord_neighbor_map()[curr_coord];

        for (Orientation orientation : y_scale_orient_list[i].get_orientation_set()) {
          irt_int layer_idx = -1;
          if (orientation == Orientation::kUp) {
            layer_idx = curr_layer_idx + 1;
          } else if (orientation == Orientation::kDown) {
            layer_idx = curr_layer_idx - 1;
          } else {
            LOG_INST.error(Loc::current(), "There are illegal orientation!");
          }
          neighbor_map[orientation] = LayerCoord(x_scale, y_scale_orient_list[i].get_scale(), layer_idx);
        }
        if (i != 0) {
          neighbor_map[Orientation::kSouth] = LayerCoord(x_scale, y_scale_orient_list[i - 1].get_scale(), curr_layer_idx);
        }
        if (i != y_scale_orient_list.size() - 1) {
          neighbor_map[Orientation::kNorth] = LayerCoord(x_scale, y_scale_orient_list[i + 1].get_scale(), curr_layer_idx);
        }
      }
    }

    for (irt_int y_scale : node_graph.get_y_scale_list()) {
      for (size_t i = 0; i < x_scale_orient_list.size(); i++) {
        LayerCoord curr_coord(x_scale_orient_list[i].get_scale(), y_scale, curr_layer_idx);
        std::map<Orientation, LayerCoord>& neighbor_map = node_graph.get_coord_neighbor_map()[curr_coord];

        for (Orientation orientation : x_scale_orient_list[i].get_orientation_set()) {
          irt_int layer_idx = -1;
          if (orientation == Orientation::kUp) {
            layer_idx = curr_layer_idx + 1;
          } else if (orientation == Orientation::kDown) {
            layer_idx = curr_layer_idx - 1;
          }
          neighbor_map[orientation] = LayerCoord(x_scale_orient_list[i].get_scale(), y_scale, layer_idx);
        }
        if (i != 0) {
          neighbor_map[Orientation::kWest] = LayerCoord(x_scale_orient_list[i - 1].get_scale(), y_scale, curr_layer_idx);
        }
        if (i != x_scale_orient_list.size() - 1) {
          neighbor_map[Orientation::kEast] = LayerCoord(x_scale_orient_list[i + 1].get_scale(), y_scale, curr_layer_idx);
        }
      }
    }
  }
  // 向 x_y_map y_x_map中记录
  for (DRNodeGraph& node_graph : layer_graph_list) {
    for (auto& [coord, orient_coord_map] : node_graph.get_coord_neighbor_map()) {
      node_graph.get_x_y_map()[coord.get_x()].insert(coord.get_y());
      node_graph.get_y_x_map()[coord.get_y()].insert(coord.get_x());
    }
  }
}

void DetailedRouter::buildCrossLayerGraph(DRBox& dr_box)
{
  for (DRTask& dr_task : dr_box.get_dr_task_list()) {
    for (DRGroup& dr_group : dr_task.get_dr_group_list()) {
      for (LayerCoord& coord : dr_group.get_coord_list()) {
        std::queue<LayerCoord> added_coord_queue = RTUtil::initQueue(coord);
        while (!added_coord_queue.empty()) {
          LayerCoord added_coord = RTUtil::getFrontAndPop(added_coord_queue);
          std::vector<LayerCoord> added_coord_list = addCoordToGraph(dr_box, added_coord);
          RTUtil::addListToQueue(added_coord_queue, added_coord_list);
        }
      }
    }
  }
}

std::vector<LayerCoord> DetailedRouter::addCoordToGraph(DRBox& dr_box, LayerCoord& added_coord)
{
  irt_int bottom_routing_layer_idx = _dr_data_manager.getConfig().bottom_routing_layer_idx;
  irt_int top_routing_layer_idx = _dr_data_manager.getConfig().top_routing_layer_idx;

  std::vector<LayerCoord> new_coord_list;

  DRNodeGraph& node_graph = dr_box.get_layer_graph_list()[added_coord.get_layer_idx()];
  // 已记录在x_y_map y_x_map中
  if (RTUtil::exist(node_graph.get_x_y_map(), added_coord.get_x())) {
    if (RTUtil::exist(node_graph.get_x_y_map()[added_coord.get_x()], added_coord.get_y())) {
      return new_coord_list;
    }
  }
  buildLayerNeighbor(new_coord_list, added_coord);
  if (bottom_routing_layer_idx <= added_coord.get_layer_idx() || added_coord.get_layer_idx() <= top_routing_layer_idx) {
    buildPlanarNeighbor(new_coord_list, node_graph, added_coord, Direction::kHorizontal);
    buildPlanarNeighbor(new_coord_list, node_graph, added_coord, Direction::kVertical);
  }
  for (LayerCoord& new_coord : new_coord_list) {
    addNeighborToGraph(dr_box, added_coord, new_coord);
  }
  // 向 x_y_map y_x_map中记录
  node_graph.get_x_y_map()[added_coord.get_x()].insert(added_coord.get_y());
  node_graph.get_y_x_map()[added_coord.get_y()].insert(added_coord.get_x());

  return new_coord_list;
}

void DetailedRouter::buildLayerNeighbor(std::vector<LayerCoord>& new_coord_list, LayerCoord& added_coord)
{
  std::vector<RoutingLayer>& routing_layer_list = _dr_data_manager.getDatabase().get_routing_layer_list();

  if (routing_layer_list.front().get_layer_idx() < added_coord.get_layer_idx()) {
    new_coord_list.emplace_back(added_coord, added_coord.get_layer_idx() - 1);
  }
  if (added_coord.get_layer_idx() < routing_layer_list.back().get_layer_idx()) {
    new_coord_list.emplace_back(added_coord, added_coord.get_layer_idx() + 1);
  }
}

void DetailedRouter::buildPlanarNeighbor(std::vector<LayerCoord>& new_coord_list, DRNodeGraph& node_graph, LayerCoord& added_coord,
                                         Direction direction)
{
  irt_int added_x = added_coord.get_x();
  irt_int added_y = added_coord.get_y();
  irt_int added_layer_idx = added_coord.get_layer_idx();

  std::vector<irt_int> scale_list;
  std::set<irt_int> scale_set;
  irt_int base_scale = -1;
  if (direction == Direction::kHorizontal) {
    scale_list = node_graph.get_x_scale_list();
    scale_set = node_graph.get_y_x_map()[added_y];
    base_scale = added_x;
  } else if (direction == Direction::kVertical) {
    scale_list = node_graph.get_y_scale_list();
    scale_set = node_graph.get_x_y_map()[added_x];
    base_scale = added_y;
  }
  if (!scale_list.empty()) {
    irt_int min_scale = -1;
    irt_int max_scale = INT32_MAX;
    for (irt_int scale : scale_list) {
      if (scale <= base_scale) {
        min_scale = scale;
      }
      if (base_scale <= scale) {
        max_scale = scale;
        break;
      }
    }
    if (min_scale == max_scale) {
      return;
    }
    irt_int lb_scale = -1;
    irt_int rt_scale = INT32_MAX;
    for (auto scale : scale_set) {
      if (scale <= base_scale) {
        lb_scale = scale;
      }
      if (base_scale <= scale) {
        rt_scale = scale;
        break;
      }
    }
    lb_scale = std::max(min_scale, lb_scale);
    if (lb_scale != -1) {
      if (direction == Direction::kHorizontal) {
        new_coord_list.emplace_back(lb_scale, added_y, added_layer_idx);
      } else {
        new_coord_list.emplace_back(added_x, lb_scale, added_layer_idx);
      }
    }
    rt_scale = std::min(max_scale, rt_scale);
    if (rt_scale != INT32_MAX) {
      if (direction == Direction::kHorizontal) {
        new_coord_list.emplace_back(rt_scale, added_y, added_layer_idx);
      } else {
        new_coord_list.emplace_back(added_x, rt_scale, added_layer_idx);
      }
    }
  }
}

void DetailedRouter::addNeighborToGraph(DRBox& dr_box, LayerCoord& first_coord, LayerCoord& second_coord)
{
  if (first_coord == second_coord) {
    return;
  }
  std::vector<DRNodeGraph>& layer_graph_list = dr_box.get_layer_graph_list();
  std::map<Orientation, LayerCoord>& first_neighbor_map
      = layer_graph_list[first_coord.get_layer_idx()].get_coord_neighbor_map()[first_coord];
  std::map<Orientation, LayerCoord>& second_neighbor_map
      = layer_graph_list[second_coord.get_layer_idx()].get_coord_neighbor_map()[second_coord];
  Orientation first_second_orientation = RTUtil::getOrientation(first_coord, second_coord);
  Orientation second_first_orientation = RTUtil::getOppositeOrientation(first_second_orientation);

  if (!RTUtil::exist(first_neighbor_map, first_second_orientation)) {
    first_neighbor_map[first_second_orientation] = second_coord;
  } else {
    LayerCoord middle_coord = first_neighbor_map[first_second_orientation];
    if (middle_coord != second_coord) {
      if (first_second_orientation == RTUtil::getOrientation(middle_coord, second_coord)) {
        // middle在first和second中间，不可能出现的情况
        LOG_INST.error(Loc::current(), "The middle coord between first and second!");
      }
      second_neighbor_map[first_second_orientation] = middle_coord;
      layer_graph_list[middle_coord.get_layer_idx()].get_coord_neighbor_map()[middle_coord][second_first_orientation] = second_coord;
      first_neighbor_map[first_second_orientation] = second_coord;
    }
  }

  if (!RTUtil::exist(second_neighbor_map, second_first_orientation)) {
    second_neighbor_map[second_first_orientation] = first_coord;
  } else {
    LayerCoord middle_coord = second_neighbor_map[second_first_orientation];
    if (middle_coord != first_coord) {
      if (second_first_orientation == RTUtil::getOrientation(middle_coord, first_coord)) {
        // middle在first和second中间，不可能出现的情况
        LOG_INST.error(Loc::current(), "The middle coord between first and second!");
      }
      first_neighbor_map[second_first_orientation] = middle_coord;
      layer_graph_list[middle_coord.get_layer_idx()].get_coord_neighbor_map()[middle_coord][first_second_orientation] = first_coord;
      second_neighbor_map[second_first_orientation] = first_coord;
    }
  }
}

void DetailedRouter::buildLayerNodeList(DRBox& dr_box)
{
  for (DRNodeGraph& node_graph : dr_box.get_layer_graph_list()) {
    std::map<LayerCoord, std::map<Orientation, LayerCoord>, CmpLayerCoordByXASC>& coord_neighbor_map = node_graph.get_coord_neighbor_map();
    std::vector<DRNode>& dr_node_list = node_graph.get_dr_node_list();

    dr_node_list.reserve(coord_neighbor_map.size());
    for (auto& [coord, orient_coord_map] : coord_neighbor_map) {
      node_graph.get_x_y_idx_map()[coord.get_x()][coord.get_y()] = static_cast<irt_int>(dr_node_list.size());
      DRNode dr_node;
      dr_node.set_coord(coord.get_planar_coord());
      dr_node.set_layer_idx(coord.get_layer_idx());
      dr_node_list.push_back(dr_node);
    }
  }
  for (DRNodeGraph& node_graph : dr_box.get_layer_graph_list()) {
    for (auto& [coord, orient_coord_map] : node_graph.get_coord_neighbor_map()) {
      DRNode* dr_node = dr_box.getNodeRef(coord);
      for (auto& [orient, neighbor_coord] : orient_coord_map) {
        dr_node->get_neighbor_ptr_map()[orient] = dr_box.getNodeRef(neighbor_coord);
      }
    }
  }
}

void DetailedRouter::buildOBSTaskMap(DRBox& dr_box)
{
  std::map<irt_int, std::vector<irt_int>> net_task_map;
  for (DRTask& dr_task : dr_box.get_dr_task_list()) {
    net_task_map[dr_task.get_origin_net_idx()].push_back(dr_task.get_task_idx());
  }
  for (auto& [net_idx, blockage_list] : dr_box.get_net_blockage_map()) {
    std::vector<irt_int>& task_idx_list = net_task_map[net_idx];
    for (LayerRect& blockage : blockage_list) {
      for (auto& [dr_node, orientation_set] : getNodeOrientationMap(dr_box, blockage)) {
        for (Orientation orientation : orientation_set) {
          if (task_idx_list.empty()) {
            dr_node->get_obs_task_map()[orientation].insert(-1);
          } else {
            dr_node->get_obs_task_map()[orientation].insert(task_idx_list.begin(), task_idx_list.end());
          }
        }
      }
    }
  }
}

std::map<DRNode*, std::set<Orientation>> DetailedRouter::getNodeOrientationMap(DRBox& dr_box, LayerRect& blockage)
{
  std::map<DRNode*, std::set<Orientation>> node_orientation_map;

  for (Segment<DRNode*>& node_segment : getNodeSegmentList(dr_box, blockage)) {
    DRNode* first = node_segment.get_first();
    DRNode* second = node_segment.get_second();
    irt_int first_layer_idx = first->get_layer_idx();
    irt_int second_layer_idx = second->get_layer_idx();

    Orientation orientation = RTUtil::getOrientation(*first, *second);
    if (orientation == Orientation::kOblique || std::abs(first_layer_idx - second_layer_idx) > 1) {
      LOG_INST.error(Loc::current(), "The node segment is illegal!");
    }
    for (LayerRect real_rect : getRealRectList({Segment<LayerCoord>(*first, *second)})) {
      if (real_rect.get_layer_idx() != blockage.get_layer_idx()) {
        continue;
      }
      if (RTUtil::isOpenOverlap(blockage, real_rect)) {
        node_orientation_map[first].insert(orientation);
        node_orientation_map[second].insert(RTUtil::getOppositeOrientation(orientation));
      }
    }
  }
  return node_orientation_map;
}

std::vector<Segment<DRNode*>> DetailedRouter::getNodeSegmentList(DRBox& dr_box, LayerRect& blockage)
{
  // 获取blockage覆盖的线段
  std::vector<Segment<DRNode*>> node_segment_list;
  for (DRNodeGraph& node_graph : dr_box.get_layer_graph_list()) {
    for (DRNode& dr_node : node_graph.get_dr_node_list()) {
      for (auto [orient, neighbor_ptr] : dr_node.get_neighbor_ptr_map()) {
        DRNode* node_a = &dr_node;
        DRNode* node_b = neighbor_ptr;
        RTUtil::sortASC(node_a, node_b);
        node_segment_list.emplace_back(node_a, node_b);
      }
    }
  }
  std::sort(node_segment_list.begin(), node_segment_list.end(), [](Segment<DRNode*>& a, Segment<DRNode*>& b) {
    if (a.get_first() != b.get_first()) {
      return a.get_first() < b.get_first();
    } else {
      return a.get_second() < b.get_second();
    }
  });
  RTUtil::merge(node_segment_list, [](Segment<DRNode*>& sentry, Segment<DRNode*>& soldier) {
    return (sentry.get_first() == soldier.get_first()) && (sentry.get_second() == soldier.get_second());
  });
  return node_segment_list;
}

std::vector<LayerRect> DetailedRouter::getRealRectList(std::vector<Segment<LayerCoord>> segment_list)
{
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _dr_data_manager.getDatabase().get_layer_via_master_list();
  std::vector<RoutingLayer>& routing_layer_list = _dr_data_manager.getDatabase().get_routing_layer_list();

  std::vector<LayerRect> rect_list;
  for (Segment<LayerCoord>& segment : segment_list) {
    LayerCoord& first_coord = segment.get_first();
    LayerCoord& second_coord = segment.get_second();

    irt_int first_layer_idx = first_coord.get_layer_idx();
    irt_int second_layer_idx = second_coord.get_layer_idx();
    if (first_layer_idx != second_layer_idx) {
      RTUtil::sortASC(first_layer_idx, second_layer_idx);
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

void DetailedRouter::buildCostTaskMap(DRBox& dr_box)
{
  std::map<irt_int, std::vector<irt_int>> net_task_map;
  for (DRTask& dr_task : dr_box.get_dr_task_list()) {
    net_task_map[dr_task.get_origin_net_idx()].push_back(dr_task.get_task_idx());
  }
  for (auto& [net_idx, region_list] : dr_box.get_net_region_map()) {
    std::vector<irt_int>& task_idx_list = net_task_map[net_idx];
    for (LayerRect& region : region_list) {
      for (auto& [dr_node, orientation_set] : getNodeOrientationMap(dr_box, region)) {
        for (Orientation orientation : orientation_set) {
          if (task_idx_list.empty()) {
            dr_node->get_cost_task_map()[orientation].insert(-1);
          } else {
            dr_node->get_cost_task_map()[orientation].insert(task_idx_list.begin(), task_idx_list.end());
          }
        }
      }
    }
  }
}

#endif

#if 1  // check dr_box

void DetailedRouter::checkDRBox(DRBox& dr_box)
{
  std::vector<RoutingLayer>& routing_layer_list = _dr_data_manager.getDatabase().get_routing_layer_list();
  PlanarCoord& grid_coord = dr_box.get_grid_coord();
  if (grid_coord.get_x() < 0 || grid_coord.get_y() < 0) {
    LOG_INST.error(Loc::current(), "The grid coord is illegal!");
  }

  PlanarRect box_region = RTUtil::getRealRect(grid_coord, _dr_data_manager.getDatabase().get_gcell_axis());
  for (auto& [net_idx, blockage_list] : dr_box.get_net_blockage_map()) {
    for (LayerRect& blockage : blockage_list) {
      if (RTUtil::isClosedOverlap(box_region, blockage)) {
        continue;
      }
      LOG_INST.error(Loc::current(), "The blockage(", blockage.get_lb_x(), ",", blockage.get_lb_y(), ")-(", blockage.get_rt_x(), ",",
                     blockage.get_rt_y(), ") is out of box!");
    }
  }
  for (auto& [net_idx, region_list] : dr_box.get_net_region_map()) {
    for (LayerRect& region : region_list) {
      if (RTUtil::isClosedOverlap(box_region, region)) {
        continue;
      }
      LOG_INST.error(Loc::current(), "The region(", region.get_lb_x(), ",", region.get_lb_y(), ")-(", region.get_rt_x(), ",",
                     region.get_rt_y(), ") is out of box!");
    }
  }
  std::vector<DRTask>& dr_task_list = dr_box.get_dr_task_list();
  for (DRTask& dr_task : dr_task_list) {
    if (dr_task.get_origin_node() == nullptr) {
      LOG_INST.error(Loc::current(), "The origin node of dr task is nullptr!");
    }
    if (dr_task.get_task_idx() < 0) {
      LOG_INST.error(Loc::current(), "The idx of dr task is illegal!");
    }
    if (dr_task.get_origin_net_idx() < 0) {
      LOG_INST.error(Loc::current(), "The idx of origin net is illegal!");
    }
    for (DRGroup& dr_group : dr_task.get_dr_group_list()) {
      for (LayerCoord& layer_coord : dr_group.get_coord_list()) {
        irt_int layer_idx = layer_coord.get_layer_idx();
        if (routing_layer_list.back().get_layer_idx() < layer_idx || layer_idx < routing_layer_list.front().get_layer_idx()) {
          LOG_INST.error(Loc::current(), "The layer idx of group coord is illegal!");
        }
        if (RTUtil::isInside(box_region, layer_coord)) {
          continue;
        }
        LOG_INST.error(Loc::current(), "The coord (", layer_coord.get_x(), ",", layer_coord.get_y(), ") is out of box!");
      }
    }
  }
  for (DRNodeGraph& node_graph : dr_box.get_layer_graph_list()) {
    irt_int layer_idx = node_graph.get_layer_idx();
    if (routing_layer_list.back().get_layer_idx() < layer_idx || layer_idx < routing_layer_list.front().get_layer_idx()) {
      LOG_INST.error(Loc::current(), "The idx of layer graph is illegal!");
    }
    for (DRNode& dr_node : node_graph.get_dr_node_list()) {
      if (dr_node.get_layer_idx() != layer_idx) {
        LOG_INST.error(Loc::current(), "The dr node layer idx is different with layer graph!");
      }
      if (!RTUtil::isInside(box_region, dr_node.get_planar_coord())) {
        LOG_INST.error(Loc::current(), "The dr node is out of box!");
      }
      for (auto& [orien, neighbor] : dr_node.get_neighbor_ptr_map()) {
        Orientation opposite_orien = RTUtil::getOppositeOrientation(orien);
        if (!RTUtil::exist(neighbor->get_neighbor_ptr_map(), opposite_orien)) {
          LOG_INST.error(Loc::current(), "The dr_node neighbor is not bidirection!");
        }
        if (neighbor->get_neighbor_ptr_map()[opposite_orien] != &dr_node) {
          LOG_INST.error(Loc::current(), "The dr_node neighbor is not bidirection!");
        }
        LayerCoord node_coord(dr_node.get_planar_coord(), dr_node.get_layer_idx());
        LayerCoord neighbor_coord(neighbor->get_planar_coord(), neighbor->get_layer_idx());
        if (RTUtil::getOrientation(node_coord, neighbor_coord) == orien) {
          continue;
        }
        LOG_INST.error(Loc::current(), "The neighbor orien is different with real region!");
      }
      for (auto& [orien, task_idx_list] : dr_node.get_obs_task_map()) {
        if (task_idx_list.empty()) {
          LOG_INST.error(Loc::current(), "The task_idx_list is empty!");
        }
      }
      for (auto& [orien, task_idx_list] : dr_node.get_cost_task_map()) {
        if (task_idx_list.empty()) {
          LOG_INST.error(Loc::current(), "The task_idx_list is empty!");
        }
      }
    }
  }
}

#endif

#if 1  // sort dr_box

void DetailedRouter::sortDRBox(DRBox& dr_box)
{
  Monitor monitor;
  if (omp_get_num_threads() == 1) {
    LOG_INST.info(Loc::current(), "Sorting all tasks beginning...");
  }

  std::vector<DRTask>& dr_task_list = dr_box.get_dr_task_list();
  std::sort(dr_task_list.begin(), dr_task_list.end(), [&](DRTask& task1, DRTask& task2) { return sortByMultiLevel(task1, task2); });

  if (omp_get_num_threads() == 1) {
    LOG_INST.info(Loc::current(), "Sorting all tasks completed!", monitor.getStatsInfo());
  }
}

bool DetailedRouter::sortByMultiLevel(DRTask& task1, DRTask& task2)
{
  SortStatus sort_status = SortStatus::kNone;

  sort_status = sortByClockPriority(task1, task2);
  if (sort_status == SortStatus::kTrue) {
    return true;
  } else if (sort_status == SortStatus::kFalse) {
    return false;
  }
  sort_status = sortByRoutingAreaASC(task1, task2);
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
  sort_status = sortByPinNumDESC(task1, task2);
  if (sort_status == SortStatus::kTrue) {
    return true;
  } else if (sort_status == SortStatus::kFalse) {
    return false;
  }
  return false;
}

// 时钟线网优先
SortStatus DetailedRouter::sortByClockPriority(DRTask& task1, DRTask& task2)
{
  ConnectType task1_connect_type = task1.get_dr_task_priority().get_connect_type();
  ConnectType task2_connect_type = task2.get_dr_task_priority().get_connect_type();

  if (task1_connect_type == ConnectType::kClock && task2_connect_type != ConnectType::kClock) {
    return SortStatus::kTrue;
  } else if (task1_connect_type != ConnectType::kClock && task2_connect_type == ConnectType::kClock) {
    return SortStatus::kFalse;
  } else {
    return SortStatus::kEqual;
  }
}

// RoutingArea 升序
SortStatus DetailedRouter::sortByRoutingAreaASC(DRTask& task1, DRTask& task2)
{
  double task1_routing_area = task1.get_dr_task_priority().get_routing_area();
  double task2_routing_area = task2.get_dr_task_priority().get_routing_area();

  if (task1_routing_area < task2_routing_area) {
    return SortStatus::kTrue;
  } else if (task1_routing_area == task2_routing_area) {
    return SortStatus::kEqual;
  } else {
    return SortStatus::kFalse;
  }
}

// 长宽比 降序
SortStatus DetailedRouter::sortByLengthWidthRatioDESC(DRTask& task1, DRTask& task2)
{
  double task1_length_width_ratio = task1.get_dr_task_priority().get_length_width_ratio();
  double task2_length_width_ratio = task2.get_dr_task_priority().get_length_width_ratio();

  if (task1_length_width_ratio > task2_length_width_ratio) {
    return SortStatus::kTrue;
  } else if (task1_length_width_ratio == task2_length_width_ratio) {
    return SortStatus::kEqual;
  } else {
    return SortStatus::kFalse;
  }
}

// PinNum 降序
SortStatus DetailedRouter::sortByPinNumDESC(DRTask& task1, DRTask& task2)
{
  double task1_pin_num = task1.get_dr_task_priority().get_pin_num();
  double task2_pin_num = task2.get_dr_task_priority().get_pin_num();

  if (task1_pin_num > task2_pin_num) {
    return SortStatus::kTrue;
  } else if (task1_pin_num == task2_pin_num) {
    return SortStatus::kEqual;
  } else {
    return SortStatus::kFalse;
  }
}

#endif

#if 1  // route dr_box

void DetailedRouter::routeDRBox(DRBox& dr_box)
{
  Monitor monitor;

  std::vector<DRTask>& dr_task_list = dr_box.get_dr_task_list();

  irt_int batch_size = RTUtil::getBatchSize(dr_task_list.size());

  Monitor stage_monitor;
  for (size_t i = 0; i < dr_task_list.size(); i++) {
    routeDRTask(dr_box, dr_task_list[i]);
    if (omp_get_num_threads() == 1 && (i + 1) % batch_size == 0) {
      LOG_INST.info(Loc::current(), "Processed ", (i + 1), " tasks", stage_monitor.getStatsInfo());
    }
  }
  if (omp_get_num_threads() == 1) {
    LOG_INST.info(Loc::current(), "Processed ", dr_task_list.size(), " tasks", monitor.getStatsInfo());
  }
}

void DetailedRouter::routeDRTask(DRBox& dr_box, DRTask& dr_task)
{
  initRoutingInfo(dr_box, dr_task);
  while (!isConnectedAllEnd(dr_box)) {
    routeSinglePath(dr_box);
    rerouteByEnlarging(dr_box);
    rerouteByforcing(dr_box);
    updatePathResult(dr_box);
    resetStartAndEnd(dr_box);
    resetSinglePath(dr_box);
  }
  updateNetResult(dr_box, dr_task);
  resetSingleNet(dr_box);
}

void DetailedRouter::initRoutingInfo(DRBox& dr_box, DRTask& dr_task)
{
  std::vector<std::vector<DRNode*>>& start_node_comb_list = dr_box.get_start_node_comb_list();
  std::vector<std::vector<DRNode*>>& end_node_comb_list = dr_box.get_end_node_comb_list();

  dr_box.set_wire_unit(1);
  dr_box.set_via_unit(1);
  dr_box.set_dr_task_ref(&dr_task);

  std::vector<std::vector<DRNode*>> node_comb_list;
  std::vector<DRGroup>& dr_group_list = dr_task.get_dr_group_list();
  for (DRGroup& dr_group : dr_group_list) {
    std::vector<DRNode*> node_comb;
    for (LayerCoord& coord : dr_group.get_coord_list()) {
      node_comb.push_back(dr_box.getNodeRef(coord));
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

bool DetailedRouter::isConnectedAllEnd(DRBox& dr_box)
{
  return dr_box.get_end_node_comb_list().empty();
}

void DetailedRouter::routeSinglePath(DRBox& dr_box)
{
  initPathHead(dr_box);
  while (!searchEnded(dr_box)) {
    expandSearching(dr_box);
    resetPathHead(dr_box);
  }
}

void DetailedRouter::initPathHead(DRBox& dr_box)
{
  std::vector<std::vector<DRNode*>>& start_node_comb_list = dr_box.get_start_node_comb_list();
  std::vector<DRNode*>& path_node_comb = dr_box.get_path_node_comb();

  for (std::vector<DRNode*>& start_node_comb : start_node_comb_list) {
    for (DRNode* start_node : start_node_comb) {
      start_node->set_estimated_cost(getEstimateCostToEnd(dr_box, start_node));
      pushToOpenList(dr_box, start_node);
    }
  }
  for (DRNode* path_node : path_node_comb) {
    path_node->set_estimated_cost(getEstimateCostToEnd(dr_box, path_node));
    pushToOpenList(dr_box, path_node);
  }
  dr_box.set_path_head_node(popFromOpenList(dr_box));
}

bool DetailedRouter::searchEnded(DRBox& dr_box)
{
  std::vector<std::vector<DRNode*>>& end_node_comb_list = dr_box.get_end_node_comb_list();
  DRNode* path_head_node = dr_box.get_path_head_node();

  if (path_head_node == nullptr) {
    dr_box.set_end_node_comb_idx(-1);
    return true;
  }
  for (size_t i = 0; i < end_node_comb_list.size(); i++) {
    for (DRNode* end_node : end_node_comb_list[i]) {
      if (path_head_node == end_node) {
        dr_box.set_end_node_comb_idx(static_cast<irt_int>(i));
        return true;
      }
    }
  }
  return false;
}

void DetailedRouter::expandSearching(DRBox& dr_box)
{
  DRNode* path_head_node = dr_box.get_path_head_node();

  for (auto& [orientation, neighbor_node] : path_head_node->get_neighbor_ptr_map()) {
    if (neighbor_node == nullptr) {
      continue;
    }
    if (!RTUtil::isInside(dr_box.get_routing_region(), *neighbor_node)) {
      continue;
    }
    if (neighbor_node->isClose()) {
      continue;
    }
    if (!passCheckingSegment(dr_box, path_head_node, neighbor_node)) {
      continue;
    }
    if (neighbor_node->isOpen() && replaceParentNode(dr_box, path_head_node, neighbor_node)) {
      neighbor_node->set_known_cost(getKnowCost(dr_box, path_head_node, neighbor_node));
      neighbor_node->set_parent_node(path_head_node);
    } else if (neighbor_node->isNone()) {
      neighbor_node->set_known_cost(getKnowCost(dr_box, path_head_node, neighbor_node));
      neighbor_node->set_parent_node(path_head_node);
      neighbor_node->set_estimated_cost(getEstimateCostToEnd(dr_box, neighbor_node));
      pushToOpenList(dr_box, neighbor_node);
    }
  }
}

bool DetailedRouter::passCheckingSegment(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
{
  if (dr_box.isForcedRouting()) {
    return true;
  }
  Orientation orientation = getOrientation(start_node, end_node);
  if (orientation == Orientation::kNone) {
    return true;
  }
  Orientation opposite_orientation = RTUtil::getOppositeOrientation(orientation);

  DRNode* pre_node = nullptr;
  DRNode* curr_node = start_node;

  while (curr_node != end_node) {
    pre_node = curr_node;
    curr_node = pre_node->getNeighborNode(orientation);

    if (curr_node == nullptr) {
      return false;
    }
    if (pre_node->isOBS(dr_box.get_curr_task_idx(), orientation) || curr_node->isOBS(dr_box.get_curr_task_idx(), opposite_orientation)) {
      return false;
    }
  }
  return true;
}

bool DetailedRouter::replaceParentNode(DRBox& dr_box, DRNode* parent_node, DRNode* child_node)
{
  return getKnowCost(dr_box, parent_node, child_node) < child_node->get_known_cost();
}

void DetailedRouter::resetPathHead(DRBox& dr_box)
{
  dr_box.set_path_head_node(popFromOpenList(dr_box));
}

void DetailedRouter::rerouteByEnlarging(DRBox& dr_box)
{
  GCellAxis& gcell_axis = _dr_data_manager.getDatabase().get_gcell_axis();

  PlanarRect dr_box_region = RTUtil::getRealRect(dr_box.get_grid_coord(), gcell_axis);
  if (isRoutingFailed(dr_box)) {
    resetSinglePath(dr_box);
    dr_box.set_routing_region(dr_box_region);
    routeSinglePath(dr_box);
    dr_box.set_routing_region(dr_box.get_curr_bounding_box());
    if (!isRoutingFailed(dr_box)) {
      if (omp_get_num_threads() == 1) {
        LOG_INST.info(Loc::current(), "The task ", dr_box.get_curr_task_idx(), " enlarged routing successfully!");
      }
    }
  }
}

bool DetailedRouter::isRoutingFailed(DRBox& dr_box)
{
  return dr_box.get_end_node_comb_idx() == -1;
}

void DetailedRouter::resetSinglePath(DRBox& dr_box)
{
  dr_box.set_forced_routing(false);

  std::priority_queue<DRNode*, std::vector<DRNode*>, CmpDRNodeCost> empty_queue;
  dr_box.set_open_queue(empty_queue);

  std::vector<DRNode*>& visited_node_list = dr_box.get_visited_node_list();
  for (DRNode* visited_node : visited_node_list) {
    visited_node->set_state(DRNodeState::kNone);
    visited_node->set_parent_node(nullptr);
    visited_node->set_known_cost(0);
    visited_node->set_estimated_cost(0);
  }
  visited_node_list.clear();

  dr_box.set_path_head_node(nullptr);
  dr_box.set_end_node_comb_idx(-1);
}

void DetailedRouter::rerouteByforcing(DRBox& dr_box)
{
  if (isRoutingFailed(dr_box)) {
    if (omp_get_num_threads() == 1) {
      LOG_INST.warning(Loc::current(), "The task ", dr_box.get_curr_task_idx(), " forced routing!");
    }
    resetSinglePath(dr_box);
    dr_box.set_forced_routing(true);
    routeSinglePath(dr_box);
    if (isRoutingFailed(dr_box)) {
      LOG_INST.error(Loc::current(), "The task ", dr_box.get_curr_task_idx(), " forced routing failed!");
    }
  }
}

void DetailedRouter::updatePathResult(DRBox& dr_box)
{
  std::vector<Segment<DRNode*>>& node_segment_list = dr_box.get_node_segment_list();
  DRNode* path_head_node = dr_box.get_path_head_node();

  DRNode* curr_node = path_head_node;
  DRNode* pre_node = curr_node->get_parent_node();

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

void DetailedRouter::resetStartAndEnd(DRBox& dr_box)
{
  std::vector<std::vector<DRNode*>>& start_node_comb_list = dr_box.get_start_node_comb_list();
  std::vector<std::vector<DRNode*>>& end_node_comb_list = dr_box.get_end_node_comb_list();
  std::vector<DRNode*>& path_node_comb = dr_box.get_path_node_comb();
  DRNode* path_head_node = dr_box.get_path_head_node();
  irt_int end_node_comb_idx = dr_box.get_end_node_comb_idx();

  end_node_comb_list[end_node_comb_idx].clear();
  end_node_comb_list[end_node_comb_idx].push_back(path_head_node);

  DRNode* path_node = path_head_node->get_parent_node();
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

void DetailedRouter::updateNetResult(DRBox& dr_box, DRTask& dr_task)
{
  std::vector<Segment<DRNode*>>& node_segment_list = dr_box.get_node_segment_list();

  std::set<DRNode*> usage_set;

  for (Segment<DRNode*>& node_segment : node_segment_list) {
    DRNode* first_node = node_segment.get_first();
    DRNode* second_node = node_segment.get_second();
    Orientation orientation = getOrientation(first_node, second_node);

    DRNode* node_i = first_node;
    while (true) {
      usage_set.insert(node_i);
      if (node_i == second_node) {
        break;
      }
      node_i = node_i->getNeighborNode(orientation);
    }
  }
  for (DRNode* usage_node : usage_set) {
    usage_node->addDemand(dr_box.get_curr_task_idx());
  }
  std::vector<Segment<LayerCoord>>& routing_segment_list = dr_task.get_routing_segment_list();
  for (Segment<DRNode*>& node_segment : node_segment_list) {
    routing_segment_list.emplace_back(*node_segment.get_first(), *node_segment.get_second());
  }
}

void DetailedRouter::resetSingleNet(DRBox& dr_box)
{
  dr_box.set_dr_task_ref(nullptr);
  dr_box.get_start_node_comb_list().clear();
  dr_box.get_end_node_comb_list().clear();
  dr_box.get_path_node_comb().clear();
  dr_box.get_node_segment_list().clear();
}

// manager open list

void DetailedRouter::pushToOpenList(DRBox& dr_box, DRNode* curr_node)
{
  std::priority_queue<DRNode*, std::vector<DRNode*>, CmpDRNodeCost>& open_queue = dr_box.get_open_queue();
  std::vector<DRNode*>& visited_node_list = dr_box.get_visited_node_list();

  open_queue.push(curr_node);
  curr_node->set_state(DRNodeState::kOpen);
  visited_node_list.push_back(curr_node);
}

DRNode* DetailedRouter::popFromOpenList(DRBox& dr_box)
{
  std::priority_queue<DRNode*, std::vector<DRNode*>, CmpDRNodeCost>& open_queue = dr_box.get_open_queue();

  DRNode* node = nullptr;
  if (!open_queue.empty()) {
    node = open_queue.top();
    open_queue.pop();
    node->set_state(DRNodeState::kClose);
  }
  return node;
}

// calculate known cost

double DetailedRouter::getKnowCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
{
  double cost = 0;
  cost += start_node->get_known_cost();
  cost += getJointCost(dr_box, end_node, getOrientation(end_node, start_node));
  cost += getWireCost(dr_box, start_node, end_node);
  cost += getViaCost(dr_box, start_node, end_node);
  return cost;
}

double DetailedRouter::getJointCost(DRBox& dr_box, DRNode* curr_node, Orientation orientation)
{
  const std::map<LayerCoord, double, CmpLayerCoordByXASC>& curr_coord_cost_map = dr_box.get_curr_coord_cost_map();

  auto iter = curr_coord_cost_map.find(*curr_node);
  double task_cost = 0;
  if (iter != curr_coord_cost_map.end()) {
    task_cost = iter->second;
  }
  double env_cost = curr_node->getCost(dr_box.get_curr_task_idx(), orientation);

  double env_weight = 1;
  double task_weight = 1;
  double joint_cost = ((env_weight * env_cost + task_weight * task_cost)
                       * RTUtil::sigmoid((env_weight * env_cost + task_weight * task_cost), (env_weight + task_weight)));
  return joint_cost;
}

// calculate estimate cost

double DetailedRouter::getEstimateCostToEnd(DRBox& dr_box, DRNode* curr_node)
{
  std::vector<std::vector<DRNode*>>& end_node_comb_list = dr_box.get_end_node_comb_list();

  double estimate_cost = DBL_MAX;
  for (std::vector<DRNode*>& end_node_comb : end_node_comb_list) {
    for (DRNode* end_node : end_node_comb) {
      if (end_node->isClose()) {
        continue;
      }
      estimate_cost = std::min(estimate_cost, getEstimateCost(dr_box, curr_node, end_node));
    }
  }
  return estimate_cost;
}

double DetailedRouter::getEstimateCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
{
  double estimate_cost = 0;
  estimate_cost += getWireCost(dr_box, start_node, end_node);
  estimate_cost += getViaCost(dr_box, start_node, end_node);
  return estimate_cost;
}

// common

Orientation DetailedRouter::getOrientation(DRNode* start_node, DRNode* end_node)
{
  Orientation orientation = RTUtil::getOrientation(*start_node, *end_node);
  if (orientation == Orientation::kOblique) {
    LOG_INST.error(Loc::current(), "The segment (", (*start_node).get_x(), ",", (*start_node).get_y(), ",", (*start_node).get_layer_idx(),
                   ")-(", (*end_node).get_x(), ",", (*end_node).get_y(), ",", (*end_node).get_layer_idx(), ") is oblique!");
  }
  return orientation;
}

double DetailedRouter::getWireCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
{
  return dr_box.get_wire_unit() * RTUtil::getManhattanDistance(*start_node, *end_node);
}

double DetailedRouter::getViaCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
{
  return dr_box.get_via_unit() * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx());
}

#endif

#if 1  // plot dr_box

void DetailedRouter::plotDRBox(DRBox& dr_box, irt_int curr_task_idx)
{
  std::vector<RoutingLayer>& routing_layer_list = _dr_data_manager.getDatabase().get_routing_layer_list();
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
  std::vector<DRNodeGraph>& layer_graph_list = dr_box.get_layer_graph_list();

  // node_graph
  GPStruct node_graph_struct("node_graph");
  for (DRNodeGraph& node_graph : layer_graph_list) {
    std::vector<DRNode>& dr_node_list = node_graph.get_dr_node_list();
    for (size_t node_idx = 0; node_idx < dr_node_list.size(); node_idx++) {
      DRNode& dr_node = dr_node_list[node_idx];
      PlanarRect real_rect = RTUtil::getEnlargedRect(dr_node.get_planar_coord(), layer_width_map[dr_node.get_layer_idx()]);
      irt_int y_reduced_span = real_rect.getYSpan() / 15;
      irt_int y = real_rect.get_rt_y();

      GPBoundary gp_boundary;
      switch (dr_node.get_state()) {
        case DRNodeState::kNone:
          gp_boundary.set_data_type(none_data_type);
          break;
        case DRNodeState::kOpen:
          gp_boundary.set_data_type(open_data_type);
          break;
        case DRNodeState::kClose:
          gp_boundary.set_data_type(close_data_type);
          break;
        default:
          LOG_INST.error(Loc::current(), "The type is error!");
          break;
      }
      gp_boundary.set_rect(real_rect);
      gp_boundary.set_layer_idx(dr_node.get_layer_idx());
      node_graph_struct.push(gp_boundary);

      y -= y_reduced_span;
      GPText gp_text_node_coord;
      gp_text_node_coord.set_coord(real_rect.get_lb_x(), y);
      gp_text_node_coord.set_text_type(info_data_type);
      gp_text_node_coord.set_message(RTUtil::getString("(", node_idx, " , ", dr_node.get_layer_idx(), ")"));
      gp_text_node_coord.set_layer_idx(dr_node.get_layer_idx());
      gp_text_node_coord.set_presentation(GPTextPresentation::kLeftMiddle);
      node_graph_struct.push(gp_text_node_coord);

      y -= y_reduced_span;
      GPText gp_text_obs_task_map;
      gp_text_obs_task_map.set_coord(real_rect.get_lb_x(), y);
      gp_text_obs_task_map.set_text_type(info_data_type);
      gp_text_obs_task_map.set_message("obs_task_map: ");
      gp_text_obs_task_map.set_layer_idx(dr_node.get_layer_idx());
      gp_text_obs_task_map.set_presentation(GPTextPresentation::kLeftMiddle);
      node_graph_struct.push(gp_text_obs_task_map);

      for (auto& [orientation, task_idx_set] : dr_node.get_obs_task_map()) {
        y -= y_reduced_span;
        GPText gp_text_obs_task_map_info;
        gp_text_obs_task_map_info.set_coord(real_rect.get_lb_x(), y);
        gp_text_obs_task_map_info.set_text_type(info_data_type);
        std::string obs_task_map_info_message = RTUtil::getString("--", GetOrientationName()(orientation), ": ");
        for (irt_int task_idx : task_idx_set) {
          obs_task_map_info_message += RTUtil::getString("(", task_idx, ")");
        }
        gp_text_obs_task_map_info.set_message(obs_task_map_info_message);
        gp_text_obs_task_map_info.set_layer_idx(dr_node.get_layer_idx());
        gp_text_obs_task_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
        node_graph_struct.push(gp_text_obs_task_map_info);
      }

      y -= y_reduced_span;
      GPText gp_text_cost_task_map;
      gp_text_cost_task_map.set_coord(real_rect.get_lb_x(), y);
      gp_text_cost_task_map.set_text_type(info_data_type);
      gp_text_cost_task_map.set_message("cost_task_map: ");
      gp_text_cost_task_map.set_layer_idx(dr_node.get_layer_idx());
      gp_text_cost_task_map.set_presentation(GPTextPresentation::kLeftMiddle);
      node_graph_struct.push(gp_text_cost_task_map);

      for (auto& [orientation, task_idx_set] : dr_node.get_cost_task_map()) {
        y -= y_reduced_span;
        GPText gp_text_cost_task_map_info;
        gp_text_cost_task_map_info.set_coord(real_rect.get_lb_x(), y);
        gp_text_cost_task_map_info.set_text_type(info_data_type);
        std::string cost_task_map_info_message = RTUtil::getString("--", GetOrientationName()(orientation), ": ");
        for (irt_int task_idx : task_idx_set) {
          cost_task_map_info_message += RTUtil::getString("(", task_idx, ")");
        }
        gp_text_cost_task_map_info.set_message(cost_task_map_info_message);
        gp_text_cost_task_map_info.set_layer_idx(dr_node.get_layer_idx());
        gp_text_cost_task_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
        node_graph_struct.push(gp_text_cost_task_map_info);
      }

      y -= y_reduced_span;
      GPText gp_text_task_queue;
      gp_text_task_queue.set_coord(real_rect.get_lb_x(), y);
      gp_text_task_queue.set_text_type(info_data_type);
      gp_text_task_queue.set_message("task_queue: ");
      gp_text_task_queue.set_layer_idx(dr_node.get_layer_idx());
      gp_text_task_queue.set_presentation(GPTextPresentation::kLeftMiddle);
      node_graph_struct.push(gp_text_task_queue);

      if (!dr_node.get_task_queue().empty()) {
        y -= y_reduced_span;
        GPText gp_text_task_queue_info;
        gp_text_task_queue_info.set_coord(real_rect.get_lb_x(), y);
        gp_text_task_queue_info.set_text_type(info_data_type);
        std::string task_queue_info_message = "--";
        for (irt_int task_idx : RTUtil::getListByQueue(dr_node.get_task_queue())) {
          task_queue_info_message += RTUtil::getString("(", task_idx, ")");
        }
        gp_text_task_queue_info.set_message(task_queue_info_message);
        gp_text_task_queue_info.set_layer_idx(dr_node.get_layer_idx());
        gp_text_task_queue_info.set_presentation(GPTextPresentation::kLeftMiddle);
        node_graph_struct.push(gp_text_task_queue_info);
      }
    }
  }
  gp_gds.addStruct(node_graph_struct);

  // neighbor
  GPStruct neighbor_map_struct("neighbor_map");
  for (DRNodeGraph& node_graph : layer_graph_list) {
    for (DRNode& dr_node : node_graph.get_dr_node_list()) {
      PlanarRect real_rect = RTUtil::getEnlargedRect(dr_node.get_planar_coord(), layer_width_map[dr_node.get_layer_idx()]);

      irt_int lb_x = real_rect.get_lb_x();
      irt_int lb_y = real_rect.get_lb_y();
      irt_int rt_x = real_rect.get_rt_x();
      irt_int rt_y = real_rect.get_rt_y();
      irt_int mid_x = (lb_x + rt_x) / 2;
      irt_int mid_y = (lb_y + rt_y) / 2;
      irt_int x_reduced_span = (rt_x - lb_x) / 4;
      irt_int y_reduced_span = (rt_y - lb_y) / 4;
      irt_int width = std::min(x_reduced_span, y_reduced_span) / 2;

      for (auto& [orientation, neighbor_node] : dr_node.get_neighbor_ptr_map()) {
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
        gp_path.set_layer_idx(dr_node.get_layer_idx());
        gp_path.set_width(width);
        gp_path.set_data_type(neighbor_data_type);
        neighbor_map_struct.push(gp_path);
      }
    }
  }
  gp_gds.addStruct(neighbor_map_struct);

  // task
  for (DRTask& dr_task : dr_box.get_dr_task_list()) {
    GPStruct task_struct(RTUtil::getString("task_", dr_task.get_task_idx(), "(net_", dr_task.get_origin_net_idx(), ")"));

    if (curr_task_idx == -1 || dr_task.get_task_idx() == curr_task_idx) {
      for (DRGroup& dr_group : dr_task.get_dr_group_list()) {
        for (LayerCoord& coord : dr_group.get_coord_list()) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(key_data_type);
          gp_boundary.set_rect(RTUtil::getEnlargedRect(coord, layer_width_map[coord.get_layer_idx()]));
          gp_boundary.set_layer_idx(coord.get_layer_idx());
          task_struct.push(gp_boundary);
        }
      }
    }

    for (Segment<LayerCoord>& segment : dr_task.get_routing_segment_list()) {
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
  GP_INST.plot(gp_gds, _dr_data_manager.getConfig().temp_directory_path + "dr_model.gds", false, false);
}

#endif

#if 1  // count dr_box

void DetailedRouter::countDRBox(DRBox& dr_box)
{
  irt_int micron_dbu = _dr_data_manager.getDatabase().get_micron_dbu();
  std::vector<RoutingLayer>& routing_layer_list = _dr_data_manager.getDatabase().get_routing_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = _dr_data_manager.getDatabase().get_layer_via_master_list();

  DRBoxStat& dr_box_stat = dr_box.get_dr_box_stat();

  std::vector<DRTask>& dr_task_list = dr_box.get_dr_task_list();
  for (DRTask& dr_task : dr_task_list) {
    for (Segment<LayerCoord>& routing_segment : dr_task.get_routing_segment_list()) {
      irt_int first_layer_idx = routing_segment.get_first().get_layer_idx();
      irt_int second_layer_idx = routing_segment.get_second().get_layer_idx();
      if (first_layer_idx == second_layer_idx) {
        double wire_length = RTUtil::getManhattanDistance(routing_segment.get_first(), routing_segment.get_second()) / 1.0 / micron_dbu;
        dr_box_stat.addTotalWireLength(wire_length);
        dr_box_stat.get_routing_wire_length_map()[first_layer_idx] += wire_length;
      } else {
        dr_box_stat.addTotalViaNumber(std::abs(first_layer_idx - second_layer_idx));
        for (irt_int i = std::min(first_layer_idx, second_layer_idx); i < std::max(first_layer_idx, second_layer_idx); i++) {
          dr_box_stat.get_cut_via_number_map()[layer_via_master_list[i].front().get_cut_layer_idx()]++;
        }
      }
    }
  }

  std::map<irt_int, std::vector<LayerRect>> task_rect_list_map;
  for (size_t i = 0; i < dr_task_list.size(); i++) {
    task_rect_list_map[i] = getRealRectList(dr_task_list[i].get_routing_segment_list());
  }

  for (size_t i = 0; i < dr_task_list.size(); i++) {
    for (LayerRect& curr_rect : task_rect_list_map[i]) {
      irt_int rect_layer_idx = curr_rect.get_layer_idx();
      irt_int min_spacing = routing_layer_list[curr_rect.get_layer_idx()].getMinSpacing(curr_rect);
      PlanarRect enlarge_curr_rect = RTUtil::getEnlargedRect(curr_rect, min_spacing);

      for (size_t j = i + 1; j < dr_task_list.size(); j++) {
        for (LayerRect& next_rect : task_rect_list_map[j]) {
          if (rect_layer_idx != next_rect.get_layer_idx() || !RTUtil::isOpenOverlap(enlarge_curr_rect, next_rect)) {
            continue;
          }
          double violation_area = RTUtil::getOverlap(enlarge_curr_rect, next_rect).getArea();
          dr_box_stat.get_routing_net_and_net_violation_area_map()[rect_layer_idx] += (violation_area / (micron_dbu * micron_dbu));
        }
      }
    }
  }

  for (size_t i = 0; i < dr_task_list.size(); i++) {
    for (LayerRect& curr_rect : task_rect_list_map[i]) {
      irt_int rect_layer_idx = curr_rect.get_layer_idx();
      for (auto& [origin_net_idx, blockage_list] : dr_box.get_net_blockage_map()) {
        if (dr_task_list[i].get_origin_net_idx() == origin_net_idx) {
          continue;
        }
        for (LayerRect& blockage : blockage_list) {
          if (rect_layer_idx != blockage.get_layer_idx() || !RTUtil::isOpenOverlap(curr_rect, blockage)) {
            continue;
          }
          double violation_area = RTUtil::getOverlap(curr_rect, blockage).getArea();
          dr_box_stat.get_routing_net_and_obs_violation_area_map()[rect_layer_idx] += (violation_area / (micron_dbu * micron_dbu));
        }
      }
    }
  }
}

#endif

#if 1  // update dr_model

void DetailedRouter::updateDRModel(DRModel& dr_model)
{
  GridMap<DRBox>& dr_box_map = dr_model.get_dr_box_map();
  for (irt_int x = 0; x < dr_box_map.get_x_size(); x++) {
    for (irt_int y = 0; y < dr_box_map.get_y_size(); y++) {
      for (DRTask& dr_task : dr_box_map[x][y].get_dr_task_list()) {
        buildRoutingResult(dr_task);
      }
    }
  }
  updateOriginTAResultTree(dr_model);
}

void DetailedRouter::buildRoutingResult(DRTask& dr_task)
{
  std::vector<LayerCoord> driving_grid_coord_list;
  std::map<LayerCoord, std::set<irt_int>, CmpLayerCoordByXASC> key_coord_pin_map;
  std::vector<DRGroup>& dr_group_list = dr_task.get_dr_group_list();
  for (size_t i = 0; i < dr_group_list.size(); i++) {
    for (LayerCoord& coord : dr_group_list[i].get_coord_list()) {
      driving_grid_coord_list.push_back(coord);
      key_coord_pin_map[coord].insert(static_cast<irt_int>(i));
    }
  }
  std::vector<Segment<LayerCoord>>& routing_segment_list = dr_task.get_routing_segment_list();
  RTNode& rt_node = dr_task.get_origin_node()->value();
  rt_node.set_routing_tree(RTUtil::getTreeByFullFlow(driving_grid_coord_list, routing_segment_list, key_coord_pin_map));
}

void DetailedRouter::updateOriginTAResultTree(DRModel& dr_model)
{
  for (DRNet& dr_net : dr_model.get_dr_net_list()) {
    Net* origin_net = dr_net.get_origin_net();
    origin_net->set_dr_result_tree(dr_net.get_dr_result_tree());
  }
}

#endif

#if 1  // report dr_model

void DetailedRouter::reportDRModel(DRModel& dr_model)
{
  countDRModel(dr_model);
  reportTable(dr_model);
}

void DetailedRouter::countDRModel(DRModel& dr_model)
{
  DRModelStat& dr_model_stat = dr_model.get_dr_model_stat();
  GridMap<DRBox>& dr_box_map = dr_model.get_dr_box_map();
  for (irt_int x = 0; x < dr_box_map.get_x_size(); x++) {
    for (irt_int y = 0; y < dr_box_map.get_y_size(); y++) {
      DRBoxStat& dr_box_stat = dr_box_map[x][y].get_dr_box_stat();
      dr_model_stat.addTotalWireLength(dr_box_stat.get_total_wire_length());
      for (auto& [layer_idx, wire_length] : dr_box_stat.get_routing_wire_length_map()) {
        dr_model_stat.get_routing_wire_length_map()[layer_idx] += wire_length;
      }
      dr_model_stat.addTotalViaNumber(dr_box_stat.get_total_via_number());
      for (auto& [layer_idx, via_num] : dr_box_stat.get_cut_via_number_map()) {
        dr_model_stat.get_cut_via_number_map()[layer_idx] += via_num;
      }
      for (auto& [layer_idx, violation_area] : dr_box_stat.get_routing_net_and_net_violation_area_map()) {
        dr_model_stat.addTotalNetAndNetViolation(violation_area);
        dr_model_stat.get_routing_net_and_net_violation_area_map()[layer_idx] += violation_area;
      }
      for (auto& [layer_idx, violation_area] : dr_box_stat.get_routing_net_and_obs_violation_area_map()) {
        dr_model_stat.addTotalNetAndObsViolation(violation_area);
        dr_model_stat.get_routing_net_and_obs_violation_area_map()[layer_idx] += violation_area;
      }
    }
  }
}

void DetailedRouter::reportTable(DRModel& dr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = _dr_data_manager.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = _dr_data_manager.getDatabase().get_cut_layer_list();

  // report overlap info
  DRModelStat& dr_model_stat = dr_model.get_dr_model_stat();
  double total_wire_length = dr_model_stat.get_total_wire_length();
  double total_net_and_net_violation_area = dr_model_stat.get_total_net_and_net_violation_area();
  double total_net_and_obs_violation_area = dr_model_stat.get_total_net_and_obs_violation_area();

  fort::char_table routing_table;
  routing_table.set_border_style(FT_SOLID_STYLE);
  routing_table << fort::header << "Routing Layer"
                << "Wire Length / um"
                << "Net And Net Violation Area / um^2"
                << "Net And Obs Violation Area / um^2" << fort::endr;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    double layer_wire_length = dr_model_stat.get_routing_wire_length_map()[routing_layer.get_layer_idx()];
    double routing_net_and_net_violation_area = dr_model_stat.get_routing_net_and_net_violation_area_map()[routing_layer.get_layer_idx()];
    double routing_net_and_obs_violation_area = dr_model_stat.get_routing_net_and_obs_violation_area_map()[routing_layer.get_layer_idx()];

    routing_table << routing_layer.get_layer_name()
                  << RTUtil::getString(layer_wire_length, "(", RTUtil::getPercentage(layer_wire_length, total_wire_length), "%)")
                  << RTUtil::getString(routing_net_and_net_violation_area, "(",
                                       RTUtil::getPercentage(routing_net_and_net_violation_area, total_net_and_net_violation_area), "%)")
                  << RTUtil::getString(routing_net_and_obs_violation_area, "(",
                                       RTUtil::getPercentage(routing_net_and_obs_violation_area, total_net_and_obs_violation_area), "%)")
                  << fort::endr;
  }
  routing_table << fort::header << "Total" << total_wire_length << total_net_and_net_violation_area << total_net_and_obs_violation_area
                << fort::endr;
  for (std::string table_str : RTUtil::splitString(routing_table.to_string(), '\n')) {
    LOG_INST.info(Loc::current(), table_str);
  }

  // report via info
  irt_int total_via_number = dr_model_stat.get_total_via_number();
  std::map<irt_int, irt_int>& cut_via_number_map = dr_model_stat.get_cut_via_number_map();

  fort::char_table cut_table;
  cut_table.set_border_style(FT_SOLID_STYLE);
  cut_table << fort::header << "Cut Layer"
            << "Via number" << fort::endr;
  for (CutLayer& cut_layer : cut_layer_list) {
    irt_int cut_via_number = cut_via_number_map[cut_layer.get_layer_idx()];
    cut_table << cut_layer.get_layer_name()
              << RTUtil::getString(cut_via_number, "(", RTUtil::getPercentage(cut_via_number, total_via_number), "%)") << fort::endr;
  }
  cut_table << fort::header << "Total" << total_via_number << fort::endr;
  for (std::string table_str : RTUtil::splitString(cut_table.to_string(), '\n')) {
    LOG_INST.info(Loc::current(), table_str);
  }
}

#endif

}  // namespace irt
