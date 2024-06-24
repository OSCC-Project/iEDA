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
#include "DRBoxId.hpp"
#include "DRNet.hpp"
#include "DRNode.hpp"
#include "DRParameter.hpp"
#include "DetailedRouter.hpp"
#include "GDSPlotter.hpp"
#include "Monitor.hpp"
#include "RTInterface.hpp"

namespace irt {

// public

void DetailedRouter::initInst()
{
  if (_dr_instance == nullptr) {
    _dr_instance = new DetailedRouter();
  }
}

DetailedRouter& DetailedRouter::getInst()
{
  if (_dr_instance == nullptr) {
    RTLOG.error(Loc::current(), "The instance not initialized!");
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

// function

void DetailedRouter::route()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");
  DRModel dr_model = initDRModel();
  iterativeDRModel(dr_model);
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

// private

DetailedRouter* DetailedRouter::_dr_instance = nullptr;

DRModel DetailedRouter::initDRModel()
{
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();

  DRModel dr_model;
  dr_model.set_dr_net_list(convertToDRNetList(net_list));
  return dr_model;
}

std::vector<DRNet> DetailedRouter::convertToDRNetList(std::vector<Net>& net_list)
{
  std::vector<DRNet> dr_net_list;
  dr_net_list.reserve(net_list.size());
  for (Net& net : net_list) {
    dr_net_list.emplace_back(convertToDRNet(net));
  }
  return dr_net_list;
}

DRNet DetailedRouter::convertToDRNet(Net& net)
{
  DRNet dr_net;
  dr_net.set_origin_net(&net);
  dr_net.set_net_idx(net.get_net_idx());
  dr_net.set_connect_type(net.get_connect_type());
  for (Pin& pin : net.get_pin_list()) {
    dr_net.get_dr_pin_list().push_back(DRPin(pin));
  }
  return dr_net;
}

void DetailedRouter::iterativeDRModel(DRModel& dr_model)
{
  int32_t cost_unit = 8;
  std::vector<DRParameter> dr_parameter_list = {
      /** format **/ {9, 0, 4 * cost_unit, 1 * cost_unit, 1 * cost_unit, true, 4},
      /** format **/ {9, -3, 8 * cost_unit, 2 * cost_unit, 2 * cost_unit, false, 4},
      /** format **/ {9, -6, 16 * cost_unit, 4 * cost_unit, 4 * cost_unit, false, 4},
      /** format **/ {9, 0, 32 * cost_unit, 8 * cost_unit, 8 * cost_unit, false, 4},
      /** format **/ {9, -3, 64 * cost_unit, 16 * cost_unit, 16 * cost_unit, false, 4},
      /** format **/ {9, -6, 128 * cost_unit, 32 * cost_unit, 32 * cost_unit, false, 4},
  };
  for (size_t i = 0, iter = 1; i < dr_parameter_list.size(); i++, iter++) {
    Monitor iter_monitor;
    RTLOG.info(Loc::current(), "***** Begin iteration ", iter, "/", dr_parameter_list.size(), "(",
               RTUTIL.getPercentage(iter, dr_parameter_list.size()), ") *****");
    setDRParameter(dr_model, iter, dr_parameter_list[i]);
    initDRBoxMap(dr_model);
    buildBoxSchedule(dr_model);
    routeDRBoxMap(dr_model);
    uploadNetResult(dr_model);
    updateSummary(dr_model);
    printSummary(dr_model);
    writeNetCSV(dr_model);
    writeViolationCSV(dr_model);
    RTLOG.info(Loc::current(), "***** End Iteration ", iter, "/", dr_parameter_list.size(), "(",
               RTUTIL.getPercentage(iter, dr_parameter_list.size()), ")", iter_monitor.getStatsInfo(), "*****");
    if (stopIteration(dr_model)) {
      break;
    }
  }
}

void DetailedRouter::setDRParameter(DRModel& dr_model, int32_t iter, DRParameter& dr_parameter)
{
  dr_model.set_iter(iter);
  RTLOG.info(Loc::current(), "prefer_wire_unit: ", dr_parameter.get_prefer_wire_unit());
  RTLOG.info(Loc::current(), "non_prefer_wire_unit: ", dr_parameter.get_non_prefer_wire_unit());
  RTLOG.info(Loc::current(), "via_unit: ", dr_parameter.get_via_unit());
  RTLOG.info(Loc::current(), "corner_unit: ", dr_parameter.get_corner_unit());
  RTLOG.info(Loc::current(), "size: ", dr_parameter.get_size());
  RTLOG.info(Loc::current(), "offset: ", dr_parameter.get_offset());
  RTLOG.info(Loc::current(), "fixed_rect_unit: ", dr_parameter.get_fixed_rect_unit());
  RTLOG.info(Loc::current(), "routed_rect_unit: ", dr_parameter.get_routed_rect_unit());
  RTLOG.info(Loc::current(), "violation_unit: ", dr_parameter.get_violation_unit());
  RTLOG.info(Loc::current(), "complete_rip_up: ", dr_parameter.get_complete_rip_up());
  RTLOG.info(Loc::current(), "max_routed_times: ", dr_parameter.get_max_routed_times());
  dr_model.set_dr_parameter(dr_parameter);
}

void DetailedRouter::initDRBoxMap(DRModel& dr_model)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();

  int32_t x_gcell_num = 0;
  for (ScaleGrid& x_grid : gcell_axis.get_x_grid_list()) {
    x_gcell_num += x_grid.get_step_num();
  }
  int32_t y_gcell_num = 0;
  for (ScaleGrid& y_grid : gcell_axis.get_y_grid_list()) {
    y_gcell_num += y_grid.get_step_num();
  }

  DRParameter& dr_parameter = dr_model.get_dr_parameter();
  int32_t size = dr_parameter.get_size();
  int32_t offset = dr_parameter.get_offset();
  int32_t x_box_num = std::ceil((x_gcell_num - offset) / 1.0 / size);
  int32_t y_box_num = std::ceil((y_gcell_num - offset) / 1.0 / size);

  GridMap<DRBox>& dr_box_map = dr_model.get_dr_box_map();
  dr_box_map.init(x_box_num, y_box_num);

  for (int32_t x = 0; x < dr_box_map.get_x_size(); x++) {
    for (int32_t y = 0; y < dr_box_map.get_y_size(); y++) {
      int32_t grid_ll_x = std::max(offset + x * size, 0);
      int32_t grid_ll_y = std::max(offset + y * size, 0);
      int32_t grid_ur_x = std::min(offset + (x + 1) * size - 1, x_gcell_num - 1);
      int32_t grid_ur_y = std::min(offset + (y + 1) * size - 1, y_gcell_num - 1);

      PlanarRect ll_gcell_rect = RTUTIL.getRealRectByGCell(PlanarCoord(grid_ll_x, grid_ll_y), gcell_axis);
      PlanarRect ur_gcell_rect = RTUTIL.getRealRectByGCell(PlanarCoord(grid_ur_x, grid_ur_y), gcell_axis);
      PlanarRect box_real_rect(ll_gcell_rect.get_ll(), ur_gcell_rect.get_ur());

      DRBox& dr_box = dr_box_map[x][y];

      EXTPlanarRect dr_box_rect;
      dr_box_rect.set_real_rect(box_real_rect);
      dr_box_rect.set_grid_rect(RTUTIL.getOpenGCellGridRect(box_real_rect, gcell_axis));
      dr_box.set_box_rect(dr_box_rect);
      DRBoxId dr_box_id;
      dr_box_id.set_x(x);
      dr_box_id.set_y(y);
      dr_box.set_dr_box_id(dr_box_id);
      dr_box.set_dr_parameter(&dr_parameter);
    }
  }
}

void DetailedRouter::buildBoxSchedule(DRModel& dr_model)
{
  GridMap<DRBox>& dr_box_map = dr_model.get_dr_box_map();

  int32_t range = 2;

  std::vector<std::vector<DRBoxId>> dr_box_id_list_list;
  for (int32_t start_x = 0; start_x < range; start_x++) {
    for (int32_t start_y = 0; start_y < range; start_y++) {
      std::vector<DRBoxId> dr_box_id_list;
      for (int32_t x = start_x; x < dr_box_map.get_x_size(); x += range) {
        for (int32_t y = start_y; y < dr_box_map.get_y_size(); y += range) {
          dr_box_id_list.emplace_back(x, y);
        }
      }
      dr_box_id_list_list.push_back(dr_box_id_list);
    }
  }
  dr_model.set_dr_box_id_list_list(dr_box_id_list_list);
}

void DetailedRouter::routeDRBoxMap(DRModel& dr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<DRBox>& dr_box_map = dr_model.get_dr_box_map();

  size_t total_box_num = 0;
  for (std::vector<DRBoxId>& dr_box_id_list : dr_model.get_dr_box_id_list_list()) {
    total_box_num += dr_box_id_list.size();
  }

  size_t routed_box_num = 0;
  for (std::vector<DRBoxId>& dr_box_id_list : dr_model.get_dr_box_id_list_list()) {
    Monitor stage_monitor;
#pragma omp parallel for
    for (DRBoxId& dr_box_id : dr_box_id_list) {
      DRBox& dr_box = dr_box_map[dr_box_id.get_x()][dr_box_id.get_y()];
      buildFixedRectList(dr_box);
      buildNetResultMap(dr_box);
      buildViolationList(dr_box);
      initDRTaskList(dr_model, dr_box);
      if (needRouting(dr_box)) {
        buildBoxTrackAxis(dr_box);
        buildLayerNodeMap(dr_box);
        buildDRNodeValid(dr_box);
        buildDRNodeNeighbor(dr_box);
        buildOrientNetMap(dr_box);
        // debugCheckDRBox(dr_box);
        // debugPlotDRBox(dr_box, -1, "before_routing");
        routeDRBox(dr_box);
        // debugPlotDRBox(dr_box, -1, "after_routing");
        uploadViolation(dr_box);
      }
      freeDRBox(dr_box);
    }
    routed_box_num += dr_box_id_list.size();
    RTLOG.info(Loc::current(), "Routed ", routed_box_num, "/", total_box_num, "(", RTUTIL.getPercentage(routed_box_num, total_box_num),
               ") boxes with ", getViolationNum(), " violations", stage_monitor.getStatsInfo());
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void DetailedRouter::buildFixedRectList(DRBox& dr_box)
{
  dr_box.set_type_layer_net_fixed_rect_map(RTDM.getTypeLayerNetFixedRectMap(dr_box.get_box_rect()));
}

void DetailedRouter::buildNetResultMap(DRBox& dr_box)
{
  for (auto& [net_idx, segment_set] : RTDM.getDetailedNetResultMap(dr_box.get_box_rect())) {
    for (Segment<LayerCoord>* segment : segment_set) {
      dr_box.get_net_result_map()[net_idx].push_back(*segment);
    }
  }
}

void DetailedRouter::buildViolationList(DRBox& dr_box)
{
  for (Violation* violation : RTDM.getViolationSet(dr_box.get_box_rect())) {
    dr_box.get_violation_list().push_back(*violation);
    RTDM.updateViolationToGCellMap(ChangeType::kDel, violation);
  }
}

void DetailedRouter::initDRTaskList(DRModel& dr_model, DRBox& dr_box)
{
  std::vector<DRNet>& dr_net_list = dr_model.get_dr_net_list();
  std::vector<DRTask*>& dr_task_list = dr_box.get_dr_task_list();

  EXTPlanarRect& box_rect = dr_box.get_box_rect();
  PlanarRect& real_rect = box_rect.get_real_rect();
  std::map<int32_t, std::set<AccessPoint*>> net_access_point_map = RTDM.getNetAccessPointMap(box_rect);
  std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_result_map = dr_box.get_net_result_map();

  std::map<int32_t, std::vector<DRGroup>> net_group_list_map;
  {
    for (auto& [net_idx, access_point_set] : net_access_point_map) {
      std::map<int32_t, DRGroup> pin_group_map;
      for (AccessPoint* access_point : access_point_set) {
        pin_group_map[access_point->get_pin_idx()].get_coord_direction_map()[access_point->getRealLayerCoord()];
      }
      for (auto& [pin_idx, group] : pin_group_map) {
        net_group_list_map[net_idx].push_back(group);
      }
    }
    for (auto& [net_idx, segment_list] : net_result_map) {
      std::map<LayerCoord, std::set<Direction>, CmpLayerCoordByXASC> coord_direction_map;
      for (const Segment<LayerCoord>& segment : segment_list) {
        const LayerCoord& first = segment.get_first();
        const LayerCoord& second = segment.get_second();
        if (first.get_layer_idx() != second.get_layer_idx()) {
          continue;
        }
        if (RTUTIL.isHorizontal(first, second)) {
          int32_t first_x = first.get_x();
          int32_t second_x = second.get_x();
          RTUTIL.swapByASC(first_x, second_x);
          if (first_x <= real_rect.get_ll_x() && real_rect.get_ll_x() <= second_x) {
            LayerCoord layer_coord(real_rect.get_ll_x(), first.get_y(), first.get_layer_idx());
            coord_direction_map[layer_coord].insert(Direction::kHorizontal);
          }
          if (first_x <= real_rect.get_ur_x() && real_rect.get_ur_x() <= second_x) {
            LayerCoord layer_coord(real_rect.get_ur_x(), first.get_y(), first.get_layer_idx());
            coord_direction_map[layer_coord].insert(Direction::kHorizontal);
          }
        } else if (RTUTIL.isVertical(first, second)) {
          int32_t first_y = first.get_y();
          int32_t second_y = second.get_y();
          RTUTIL.swapByASC(first_y, second_y);
          if (first_y <= real_rect.get_ll_y() && real_rect.get_ll_y() <= second_y) {
            LayerCoord layer_coord(first.get_x(), real_rect.get_ll_y(), first.get_layer_idx());
            coord_direction_map[layer_coord].insert(Direction::kVertical);
          }
          if (first_y <= real_rect.get_ur_y() && real_rect.get_ur_y() <= second_y) {
            LayerCoord layer_coord(first.get_x(), real_rect.get_ur_y(), first.get_layer_idx());
            coord_direction_map[layer_coord].insert(Direction::kVertical);
          }
        } else {
          RTLOG.error(Loc::current(), "The segment is oblique!");
        }
      }
      for (auto& [coord, direction_set] : coord_direction_map) {
        DRGroup dr_group;
        dr_group.get_coord_direction_map()[coord] = direction_set;
        net_group_list_map[net_idx].push_back(dr_group);
      }
    }
  }
  for (auto [net_idx, dr_group_list] : net_group_list_map) {
    if (dr_group_list.size() < 2) {
      continue;
    }
    DRTask* dr_task = new DRTask();
    dr_task->set_net_idx(net_idx);
    dr_task->set_connect_type(dr_net_list[net_idx].get_connect_type());
    dr_task->set_dr_group_list(dr_group_list);
    {
      std::vector<PlanarCoord> coord_list;
      for (DRGroup& dr_group : dr_task->get_dr_group_list()) {
        for (auto& [coord, direction_set] : dr_group.get_coord_direction_map()) {
          coord_list.push_back(coord);
        }
      }
      dr_task->set_bounding_box(RTUTIL.getBoundingBox(coord_list));
    }
    dr_task->set_routed_times(0);
    dr_task_list.push_back(dr_task);
  }
  std::sort(dr_task_list.begin(), dr_task_list.end(), CmpDRTask());
}

bool DetailedRouter::needRouting(DRBox& dr_box)
{
  if (dr_box.get_dr_task_list().empty()) {
    return false;
  }
  if (dr_box.get_dr_parameter()->get_complete_rip_up() == false && dr_box.get_violation_list().empty()) {
    return false;
  }
  return true;
}

void DetailedRouter::buildBoxTrackAxis(DRBox& dr_box)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  std::vector<int32_t> x_scale_list;
  std::vector<int32_t> y_scale_list;

  PlanarRect& box_region = dr_box.get_box_rect().get_real_rect();
  for (RoutingLayer& routing_layer : routing_layer_list) {
    for (int32_t x_scale : RTUTIL.getScaleList(box_region.get_ll_x(), box_region.get_ur_x(), routing_layer.getXTrackGridList())) {
      x_scale_list.push_back(x_scale);
    }
    for (int32_t y_scale : RTUTIL.getScaleList(box_region.get_ll_y(), box_region.get_ur_y(), routing_layer.getYTrackGridList())) {
      y_scale_list.push_back(y_scale);
    }
  }
  for (DRTask* dr_task : dr_box.get_dr_task_list()) {
    for (DRGroup& dr_group : dr_task->get_dr_group_list()) {
      for (auto& [coord, direction_set] : dr_group.get_coord_direction_map()) {
        x_scale_list.push_back(coord.get_x());
        y_scale_list.push_back(coord.get_y());
      }
    }
  }

  ScaleAxis& box_track_axis = dr_box.get_box_track_axis();
  std::sort(x_scale_list.begin(), x_scale_list.end());
  x_scale_list.erase(std::unique(x_scale_list.begin(), x_scale_list.end()), x_scale_list.end());
  box_track_axis.set_x_grid_list(RTUTIL.makeScaleGridList(x_scale_list));
  std::sort(y_scale_list.begin(), y_scale_list.end());
  y_scale_list.erase(std::unique(y_scale_list.begin(), y_scale_list.end()), y_scale_list.end());
  box_track_axis.set_y_grid_list(RTUTIL.makeScaleGridList(y_scale_list));
}

void DetailedRouter::buildLayerNodeMap(DRBox& dr_box)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  PlanarCoord& real_ll = dr_box.get_box_rect().get_real_ll();
  PlanarCoord& real_ur = dr_box.get_box_rect().get_real_ur();
  ScaleAxis& box_track_axis = dr_box.get_box_track_axis();
  std::vector<int32_t> x_list = RTUTIL.getScaleList(real_ll.get_x(), real_ur.get_x(), box_track_axis.get_x_grid_list());
  std::vector<int32_t> y_list = RTUTIL.getScaleList(real_ll.get_y(), real_ur.get_y(), box_track_axis.get_y_grid_list());

  std::vector<GridMap<DRNode>>& layer_node_map = dr_box.get_layer_node_map();
  layer_node_map.resize(routing_layer_list.size());
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    GridMap<DRNode>& dr_node_map = layer_node_map[layer_idx];
    dr_node_map.init(x_list.size(), y_list.size());
    for (size_t x = 0; x < x_list.size(); x++) {
      for (size_t y = 0; y < y_list.size(); y++) {
        DRNode& dr_node = dr_node_map[x][y];
        dr_node.set_x(x_list[x]);
        dr_node.set_y(y_list[y]);
        dr_node.set_layer_idx(layer_idx);
      }
    }
  }
}

void DetailedRouter::buildDRNodeValid(DRBox& dr_box)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  int32_t bottom_routing_layer_idx = RTDM.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = RTDM.getConfig().top_routing_layer_idx;

  int32_t max_neighbor_range = dr_box.get_dr_parameter()->get_max_neighbor_range();
  std::vector<GridMap<DRNode>>& layer_node_map = dr_box.get_layer_node_map();
  ScaleAxis& box_track_axis = dr_box.get_box_track_axis();

  std::map<int32_t, std::set<int32_t>> layer_x_scale_map;
  std::map<int32_t, std::set<int32_t>> layer_y_scale_map;
  // 本层track上的node设置点合法状态
  int32_t box_ll_x = dr_box.get_box_rect().get_real_ll_x();
  int32_t box_ll_y = dr_box.get_box_rect().get_real_ll_y();
  int32_t box_ur_x = dr_box.get_box_rect().get_real_ur_x();
  int32_t box_ur_y = dr_box.get_box_rect().get_real_ur_y();
  for (RoutingLayer& routing_layer : routing_layer_list) {
    int32_t layer_idx = routing_layer.get_layer_idx();
    for (int32_t curr_layer_idx : {layer_idx - 1, layer_idx, layer_idx + 1}) {
      if (curr_layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < curr_layer_idx) {
        continue;
      }
      RoutingLayer& curr_routing_layer = routing_layer_list[curr_layer_idx];
      std::vector<int32_t> x_scale_list = RTUTIL.getScaleList(box_ll_x, box_ur_x, curr_routing_layer.getXTrackGridList());
      std::vector<int32_t> y_scale_list = RTUTIL.getScaleList(box_ll_y, box_ur_y, curr_routing_layer.getYTrackGridList());
      if (curr_layer_idx == layer_idx) {
        layer_x_scale_map[layer_idx].insert(x_scale_list.begin(), x_scale_list.end());
        layer_y_scale_map[layer_idx].insert(y_scale_list.begin(), y_scale_list.end());
      } else if (curr_routing_layer.isPreferH()) {
        layer_y_scale_map[layer_idx].insert(y_scale_list.begin(), y_scale_list.end());
      } else {
        layer_x_scale_map[layer_idx].insert(x_scale_list.begin(), x_scale_list.end());
      }
    }
  }
  for (RoutingLayer& routing_layer : routing_layer_list) {
    int32_t layer_idx = routing_layer.get_layer_idx();
    for (int32_t x_scale : layer_x_scale_map[layer_idx]) {
      for (int32_t y_scale : layer_y_scale_map[layer_idx]) {
        PlanarCoord real_coord(x_scale, y_scale);
        if (!RTUTIL.existTrackGrid(real_coord, dr_box.get_box_track_axis())) {
          RTLOG.error(Loc::current(), "There is no grid coord for real coord(", x_scale, ",", y_scale, ")!");
        }
        PlanarCoord grid_coord = RTUTIL.getTrackGrid(real_coord, box_track_axis);
        layer_node_map[layer_idx][grid_coord.get_x()][grid_coord.get_y()].set_is_valid(true);
      }
    }
  }
  // ap点设置有效点
  for (DRTask* dr_task : dr_box.get_dr_task_list()) {
    for (DRGroup& dr_group : dr_task->get_dr_group_list()) {
      for (auto& [real_coord, direction_set] : dr_group.get_coord_direction_map()) {
        std::vector<int32_t> point_layer_idx_list
            = RTUTIL.getReservedViaBelowLayerIdxList(real_coord.get_layer_idx(), bottom_routing_layer_idx, top_routing_layer_idx);
        std::sort(point_layer_idx_list.begin(), point_layer_idx_list.end());
        point_layer_idx_list.push_back(point_layer_idx_list.back() + 1);
        PlanarCoord grid_coord = RTUTIL.getTrackGrid(real_coord, dr_box.get_box_track_axis());
        for (int32_t point_layer_idx : point_layer_idx_list) {
          // 设置ap点相关的空间有效点
          layer_node_map[point_layer_idx][grid_coord.get_x()][grid_coord.get_y()].set_is_valid(true);
          // 设置ap点相关的平面有效点
          if (point_layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < point_layer_idx) {
            continue;
          }
          bool on_v_track = RTUTIL.exist(layer_x_scale_map[point_layer_idx], real_coord.get_x());
          bool on_h_track = RTUTIL.exist(layer_y_scale_map[point_layer_idx], real_coord.get_y());
          if (on_v_track && on_h_track) {
            continue;
          }
          int32_t x_pitch = routing_layer_list[point_layer_idx].getXTrackGridList().front().get_step_length();
          int32_t x_neighbor_range = x_pitch * max_neighbor_range;
          std::vector<PlanarCoord> valid_h_coord_list;
          for (int32_t x_scale : layer_x_scale_map[point_layer_idx]) {
            if ((real_coord.get_x() - x_neighbor_range <= x_scale && x_scale <= real_coord.get_x() - x_pitch)
                || (real_coord.get_x() + x_pitch <= x_scale && x_scale <= real_coord.get_x() + x_neighbor_range)) {
              valid_h_coord_list.emplace_back(x_scale, real_coord.get_y());
              continue;
            }
            if (x_scale > real_coord.get_x() + x_neighbor_range) {
              break;
            }
          }
          int32_t y_pitch = routing_layer_list[point_layer_idx].getYTrackGridList().front().get_step_length();
          int32_t y_neighbor_range = y_pitch * max_neighbor_range;
          std::vector<PlanarCoord> valid_v_coord_list;
          for (int32_t y_scale : layer_y_scale_map[point_layer_idx]) {
            if ((real_coord.get_y() - y_neighbor_range <= y_scale && y_scale <= real_coord.get_y() - y_pitch)
                || (real_coord.get_y() + y_pitch <= y_scale && y_scale <= real_coord.get_y() + y_neighbor_range)) {
              valid_v_coord_list.emplace_back(real_coord.get_x(), y_scale);
              continue;
            }
            if (y_scale > real_coord.get_y() + y_neighbor_range) {
              break;
            }
          }

          if (on_v_track) {
            for (PlanarCoord& valid_h_coord : valid_h_coord_list) {
              PlanarCoord grid_coord = RTUTIL.getTrackGrid(valid_h_coord, dr_box.get_box_track_axis());
              layer_node_map[point_layer_idx][grid_coord.get_x()][grid_coord.get_y()].set_is_valid(true);
            }
          } else if (on_h_track) {
            for (PlanarCoord& valid_v_coord : valid_v_coord_list) {
              PlanarCoord grid_coord = RTUTIL.getTrackGrid(valid_v_coord, dr_box.get_box_track_axis());
              layer_node_map[point_layer_idx][grid_coord.get_x()][grid_coord.get_y()].set_is_valid(true);
            }
          } else {
            for (PlanarCoord& valid_h_coord : valid_h_coord_list) {
              PlanarCoord grid_coord = RTUTIL.getTrackGrid(valid_h_coord, dr_box.get_box_track_axis());
              layer_node_map[point_layer_idx][grid_coord.get_x()][grid_coord.get_y()].set_is_valid(true);
            }
            for (PlanarCoord& valid_v_coord : valid_v_coord_list) {
              PlanarCoord grid_coord = RTUTIL.getTrackGrid(valid_v_coord, dr_box.get_box_track_axis());
              layer_node_map[point_layer_idx][grid_coord.get_x()][grid_coord.get_y()].set_is_valid(true);
            }
          }
        }
      }
    }
  }
}

void DetailedRouter::buildDRNodeNeighbor(DRBox& dr_box)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  int32_t bottom_routing_layer_idx = RTDM.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = RTDM.getConfig().top_routing_layer_idx;

  int32_t max_neighbor_range = dr_box.get_dr_parameter()->get_max_neighbor_range();
  std::vector<GridMap<DRNode>>& layer_node_map = dr_box.get_layer_node_map();

  for (int32_t curr_layer_idx = 0; curr_layer_idx < static_cast<int32_t>(layer_node_map.size()); curr_layer_idx++) {
    for (int32_t curr_x = 0; curr_x < layer_node_map[curr_layer_idx].get_x_size(); curr_x++) {
      for (int32_t curr_y = 0; curr_y < layer_node_map[curr_layer_idx].get_y_size(); curr_y++) {
        DRNode& curr_node = layer_node_map[curr_layer_idx][curr_x][curr_y];
        if (!curr_node.get_is_valid()) {
          continue;
        }
        // 向上寻找，如果上邻居是有效结果，设置空间邻居关系
        if (curr_layer_idx + 1 < static_cast<int32_t>(layer_node_map.size())) {
          DRNode& above_node = layer_node_map[curr_layer_idx + 1][curr_x][curr_y];
          if (above_node.get_is_valid()) {
            curr_node.get_neighbor_node_map()[Orientation::kAbove] = &above_node;
            above_node.get_neighbor_node_map()[Orientation::kBelow] = &curr_node;
          }
        }
        if (curr_layer_idx < bottom_routing_layer_idx || curr_layer_idx > top_routing_layer_idx) {
          continue;
        }
        // 向东寻找，找到第一个有效结点即为东邻居，并将东邻居的西邻居设为自己
        int32_t x_pitch = routing_layer_list[curr_layer_idx].getXTrackGridList().front().get_step_length();
        int32_t x_neighbor_range = x_pitch * max_neighbor_range;
        for (int32_t east_x = curr_x + 1; east_x < layer_node_map[curr_layer_idx].get_x_size(); east_x++) {
          DRNode& east_node = layer_node_map[curr_layer_idx][east_x][curr_y];
          if (!east_node.get_is_valid()) {
            continue;
          }
          if (std::abs(curr_node.get_x() - east_node.get_x()) > x_neighbor_range) {
            break;
          }
          curr_node.get_neighbor_node_map()[Orientation::kEast] = &east_node;
          east_node.get_neighbor_node_map()[Orientation::kWest] = &curr_node;
          break;
        }
        // 向北寻找，找到第一个有效结点即为北邻居，并将北邻居的南邻居设为自己
        int32_t y_pitch = routing_layer_list[curr_layer_idx].getYTrackGridList().front().get_step_length();
        int32_t y_neighbor_range = y_pitch * max_neighbor_range;
        for (int32_t north_y = curr_y + 1; north_y < layer_node_map[curr_layer_idx].get_y_size(); north_y++) {
          DRNode& north_node = layer_node_map[curr_layer_idx][curr_x][north_y];
          if (!north_node.get_is_valid()) {
            continue;
          }
          if (std::abs(curr_node.get_y() - north_node.get_y()) > y_neighbor_range) {
            break;
          }
          curr_node.get_neighbor_node_map()[Orientation::kNorth] = &north_node;
          north_node.get_neighbor_node_map()[Orientation::kSouth] = &curr_node;
          break;
        }
      }
    }
  }
}

void DetailedRouter::buildOrientNetMap(DRBox& dr_box)
{
  for (auto& [is_routing, layer_net_fixed_rect_map] : dr_box.get_type_layer_net_fixed_rect_map()) {
    for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
      for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
        for (auto& fixed_rect : fixed_rect_set) {
          updateFixedRectToGraph(dr_box, ChangeType::kAdd, net_idx, fixed_rect, is_routing);
        }
      }
    }
  }
  for (auto& [net_idx, segment_list] : dr_box.get_net_result_map()) {
    for (Segment<LayerCoord>& segment : segment_list) {
      updateNetResultToGraph(dr_box, ChangeType::kAdd, net_idx, segment);
    }
  }
  for (Violation& violation : dr_box.get_violation_list()) {
    updateViolationToGraph(dr_box, ChangeType::kAdd, violation);
  }
}

void DetailedRouter::routeDRBox(DRBox& dr_box)
{
  std::vector<DRTask*> dr_task_list = initTaskSchedule(dr_box);
  while (!dr_task_list.empty()) {
    for (DRTask* dr_task : dr_task_list) {
      routeDRTask(dr_box, dr_task);
      dr_task->addRoutedTimes();
    }
    updateViolationList(dr_box);
    dr_task_list = getTaskScheduleByViolation(dr_box);
  }
}

std::vector<DRTask*> DetailedRouter::initTaskSchedule(DRBox& dr_box)
{
  bool complete_rip_up = dr_box.get_dr_parameter()->get_complete_rip_up();

  std::vector<DRTask*> dr_task_list;
  if (complete_rip_up) {
    for (DRTask* dr_task : dr_box.get_dr_task_list()) {
      dr_task_list.push_back(dr_task);
    }
  } else {
    dr_task_list = getTaskScheduleByViolation(dr_box);
  }
  return dr_task_list;
}

std::vector<DRTask*> DetailedRouter::getTaskScheduleByViolation(DRBox& dr_box)
{
  int32_t max_routed_times = dr_box.get_dr_parameter()->get_max_routed_times();

  std::set<int32_t> violation_net_set;
  for (Violation& violation : dr_box.get_violation_list()) {
    for (int32_t violation_net : violation.get_violation_net_set()) {
      violation_net_set.insert(violation_net);
    }
  }
  std::vector<DRTask*> dr_task_list;
  for (DRTask* dr_task : dr_box.get_dr_task_list()) {
    if (!RTUTIL.exist(violation_net_set, dr_task->get_net_idx())) {
      continue;
    }
    if (dr_task->get_routed_times() >= max_routed_times) {
      continue;
    }
    dr_task_list.push_back(dr_task);
  }
  return dr_task_list;
}

void DetailedRouter::routeDRTask(DRBox& dr_box, DRTask* dr_task)
{
  initSingleTask(dr_box, dr_task);
  while (!isConnectedAllEnd(dr_box)) {
    routeSinglePath(dr_box);
    updatePathResult(dr_box);
    updateDirectionSet(dr_box);
    resetStartAndEnd(dr_box);
    resetSinglePath(dr_box);
  }
  updateTaskResult(dr_box);
  resetSingleTask(dr_box);
}

void DetailedRouter::initSingleTask(DRBox& dr_box, DRTask* dr_task)
{
  ScaleAxis& box_track_axis = dr_box.get_box_track_axis();
  std::vector<GridMap<DRNode>>& layer_node_map = dr_box.get_layer_node_map();

  // single task
  dr_box.set_curr_dr_task(dr_task);
  {
    std::vector<std::vector<DRNode*>> node_list_list;
    std::vector<DRGroup>& dr_group_list = dr_task->get_dr_group_list();
    for (DRGroup& dr_group : dr_group_list) {
      std::vector<DRNode*> node_list;
      for (auto& [coord, direction_set] : dr_group.get_coord_direction_map()) {
        if (!RTUTIL.existTrackGrid(coord, box_track_axis)) {
          RTLOG.error(Loc::current(), "The coord can not find grid!");
        }
        PlanarCoord grid_coord = RTUTIL.getTrackGrid(coord, box_track_axis);
        DRNode& dr_node = layer_node_map[coord.get_layer_idx()][grid_coord.get_x()][grid_coord.get_y()];
        dr_node.set_direction_set(direction_set);
        node_list.push_back(&dr_node);
      }
      node_list_list.push_back(node_list);
    }
    for (size_t i = 0; i < node_list_list.size(); i++) {
      if (i == 0) {
        dr_box.get_start_node_list_list().push_back(node_list_list[i]);
      } else {
        dr_box.get_end_node_list_list().push_back(node_list_list[i]);
      }
    }
  }
  dr_box.get_path_node_list().clear();
  dr_box.get_single_task_visited_node_list().clear();
  dr_box.get_routing_segment_list().clear();
}

bool DetailedRouter::isConnectedAllEnd(DRBox& dr_box)
{
  return dr_box.get_end_node_list_list().empty();
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
  std::vector<std::vector<DRNode*>>& start_node_list_list = dr_box.get_start_node_list_list();
  std::vector<DRNode*>& path_node_list = dr_box.get_path_node_list();

  for (std::vector<DRNode*>& start_node_list : start_node_list_list) {
    for (DRNode* start_node : start_node_list) {
      start_node->set_estimated_cost(getEstimateCostToEnd(dr_box, start_node));
      pushToOpenList(dr_box, start_node);
    }
  }
  for (DRNode* path_node : path_node_list) {
    path_node->set_estimated_cost(getEstimateCostToEnd(dr_box, path_node));
    pushToOpenList(dr_box, path_node);
  }
  resetPathHead(dr_box);
}

bool DetailedRouter::searchEnded(DRBox& dr_box)
{
  std::vector<std::vector<DRNode*>>& end_node_list_list = dr_box.get_end_node_list_list();
  DRNode* path_head_node = dr_box.get_path_head_node();

  if (path_head_node == nullptr) {
    dr_box.set_end_node_list_idx(-1);
    return true;
  }
  for (size_t i = 0; i < end_node_list_list.size(); i++) {
    for (DRNode* end_node : end_node_list_list[i]) {
      if (path_head_node == end_node) {
        dr_box.set_end_node_list_idx(static_cast<int32_t>(i));
        return true;
      }
    }
  }
  return false;
}

void DetailedRouter::expandSearching(DRBox& dr_box)
{
  PriorityQueue<DRNode*, std::vector<DRNode*>, CmpDRNodeCost>& open_queue = dr_box.get_open_queue();
  DRNode* path_head_node = dr_box.get_path_head_node();

  for (auto& [orientation, neighbor_node] : path_head_node->get_neighbor_node_map()) {
    if (neighbor_node == nullptr) {
      continue;
    }
    if (neighbor_node->isClose()) {
      continue;
    }
    double know_cost = getKnowCost(dr_box, path_head_node, neighbor_node);
    if (neighbor_node->isOpen() && know_cost < neighbor_node->get_known_cost()) {
      neighbor_node->set_known_cost(know_cost);
      neighbor_node->set_parent_node(path_head_node);
      // 对优先队列中的值修改了，需要重新建堆
      std::make_heap(open_queue.begin(), open_queue.end(), CmpDRNodeCost());
    } else if (neighbor_node->isNone()) {
      neighbor_node->set_known_cost(know_cost);
      neighbor_node->set_parent_node(path_head_node);
      neighbor_node->set_estimated_cost(getEstimateCostToEnd(dr_box, neighbor_node));
      pushToOpenList(dr_box, neighbor_node);
    }
  }
}

void DetailedRouter::resetPathHead(DRBox& dr_box)
{
  dr_box.set_path_head_node(popFromOpenList(dr_box));
}

bool DetailedRouter::isRoutingFailed(DRBox& dr_box)
{
  return dr_box.get_end_node_list_idx() == -1;
}

void DetailedRouter::resetSinglePath(DRBox& dr_box)
{
  PriorityQueue<DRNode*, std::vector<DRNode*>, CmpDRNodeCost> empty_queue;
  dr_box.set_open_queue(empty_queue);

  std::vector<DRNode*>& single_path_visited_node_list = dr_box.get_single_path_visited_node_list();
  for (DRNode* visited_node : single_path_visited_node_list) {
    visited_node->set_state(DRNodeState::kNone);
    visited_node->set_parent_node(nullptr);
    visited_node->set_known_cost(0);
    visited_node->set_estimated_cost(0);
  }
  single_path_visited_node_list.clear();

  dr_box.set_path_head_node(nullptr);
  dr_box.set_end_node_list_idx(-1);
}

void DetailedRouter::updatePathResult(DRBox& dr_box)
{
  for (Segment<LayerCoord>& routing_segment : getRoutingSegmentListByNode(dr_box.get_path_head_node())) {
    dr_box.get_routing_segment_list().push_back(routing_segment);
  }
}

std::vector<Segment<LayerCoord>> DetailedRouter::getRoutingSegmentListByNode(DRNode* node)
{
  std::vector<Segment<LayerCoord>> routing_segment_list;

  DRNode* curr_node = node;
  DRNode* pre_node = curr_node->get_parent_node();

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

void DetailedRouter::updateDirectionSet(DRBox& dr_box)
{
  DRNode* path_head_node = dr_box.get_path_head_node();

  DRNode* curr_node = path_head_node;
  DRNode* pre_node = curr_node->get_parent_node();
  while (pre_node != nullptr) {
    curr_node->get_direction_set().insert(RTUTIL.getDirection(*curr_node, *pre_node));
    pre_node->get_direction_set().insert(RTUTIL.getDirection(*pre_node, *curr_node));
    curr_node = pre_node;
    pre_node = curr_node->get_parent_node();
  }
}

void DetailedRouter::resetStartAndEnd(DRBox& dr_box)
{
  std::vector<std::vector<DRNode*>>& start_node_list_list = dr_box.get_start_node_list_list();
  std::vector<std::vector<DRNode*>>& end_node_list_list = dr_box.get_end_node_list_list();
  std::vector<DRNode*>& path_node_list = dr_box.get_path_node_list();
  DRNode* path_head_node = dr_box.get_path_head_node();
  int32_t end_node_list_idx = dr_box.get_end_node_list_idx();

  // 对于抵达的终点pin，只保留到达的node
  end_node_list_list[end_node_list_idx].clear();
  end_node_list_list[end_node_list_idx].push_back(path_head_node);

  DRNode* path_node = path_head_node->get_parent_node();
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
  start_node_list_list.push_back(end_node_list_list[end_node_list_idx]);
  end_node_list_list.erase(end_node_list_list.begin() + end_node_list_idx);
}

void DetailedRouter::updateTaskResult(DRBox& dr_box)
{
  std::vector<Segment<LayerCoord>> new_routing_segment_list = getRoutingSegmentList(dr_box);

  int32_t curr_net_idx = dr_box.get_curr_dr_task()->get_net_idx();
  std::vector<Segment<LayerCoord>>& routing_segment_list = dr_box.get_net_result_map()[curr_net_idx];
  // 原结果从graph删除
  for (Segment<LayerCoord>& routing_segment : routing_segment_list) {
    updateNetResultToGraph(dr_box, ChangeType::kDel, curr_net_idx, routing_segment);
  }
  routing_segment_list = new_routing_segment_list;
  // 新结果添加到graph
  for (Segment<LayerCoord>& routing_segment : routing_segment_list) {
    updateNetResultToGraph(dr_box, ChangeType::kAdd, curr_net_idx, routing_segment);
  }
}

std::vector<Segment<LayerCoord>> DetailedRouter::getRoutingSegmentList(DRBox& dr_box)
{
  DRTask* curr_dr_task = dr_box.get_curr_dr_task();

  std::vector<LayerCoord> candidate_root_coord_list;
  std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
  std::vector<DRGroup>& dr_group_list = curr_dr_task->get_dr_group_list();
  for (size_t i = 0; i < dr_group_list.size(); i++) {
    for (auto& [coord, direction_set] : dr_group_list[i].get_coord_direction_map()) {
      candidate_root_coord_list.push_back(coord);
      key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
    }
  }
  MTree<LayerCoord> coord_tree = RTUTIL.getTreeByFullFlow(candidate_root_coord_list, dr_box.get_routing_segment_list(), key_coord_pin_map);

  std::vector<Segment<LayerCoord>> routing_segment_list;
  for (Segment<TNode<LayerCoord>*>& coord_segment : RTUTIL.getSegListByTree(coord_tree)) {
    routing_segment_list.emplace_back(coord_segment.get_first()->value(), coord_segment.get_second()->value());
  }
  return routing_segment_list;
}

void DetailedRouter::resetSingleTask(DRBox& dr_box)
{
  dr_box.set_curr_dr_task(nullptr);
  dr_box.get_start_node_list_list().clear();
  dr_box.get_end_node_list_list().clear();
  dr_box.get_path_node_list().clear();

  std::vector<DRNode*>& single_task_visited_node_list = dr_box.get_single_task_visited_node_list();
  for (DRNode* single_task_visited_node : single_task_visited_node_list) {
    single_task_visited_node->get_direction_set().clear();
  }
  single_task_visited_node_list.clear();

  dr_box.get_routing_segment_list().clear();
}

// manager open list

void DetailedRouter::pushToOpenList(DRBox& dr_box, DRNode* curr_node)
{
  PriorityQueue<DRNode*, std::vector<DRNode*>, CmpDRNodeCost>& open_queue = dr_box.get_open_queue();
  std::vector<DRNode*>& single_task_visited_node_list = dr_box.get_single_task_visited_node_list();
  std::vector<DRNode*>& single_path_visited_node_list = dr_box.get_single_path_visited_node_list();

  open_queue.push(curr_node);
  curr_node->set_state(DRNodeState::kOpen);
  single_task_visited_node_list.push_back(curr_node);
  single_path_visited_node_list.push_back(curr_node);
}

DRNode* DetailedRouter::popFromOpenList(DRBox& dr_box)
{
  PriorityQueue<DRNode*, std::vector<DRNode*>, CmpDRNodeCost>& open_queue = dr_box.get_open_queue();

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
  cost += getNodeCost(dr_box, start_node, RTUTIL.getOrientation(*start_node, *end_node));
  cost += getNodeCost(dr_box, end_node, RTUTIL.getOrientation(*end_node, *start_node));
  cost += getKnowWireCost(dr_box, start_node, end_node);
  cost += getKnowCornerCost(dr_box, start_node, end_node);
  cost += getKnowViaCost(dr_box, start_node, end_node);
  return cost;
}

double DetailedRouter::getNodeCost(DRBox& dr_box, DRNode* curr_node, Orientation orientation)
{
  double fixed_rect_unit = dr_box.get_dr_parameter()->get_fixed_rect_unit();
  double routed_rect_unit = dr_box.get_dr_parameter()->get_routed_rect_unit();
  double violation_unit = dr_box.get_dr_parameter()->get_violation_unit();

  int32_t net_idx = dr_box.get_curr_dr_task()->get_net_idx();

  double cost = 0;
  cost += curr_node->getFixedRectCost(net_idx, orientation, fixed_rect_unit);
  cost += curr_node->getRoutedRectCost(net_idx, orientation, routed_rect_unit);
  cost += curr_node->getViolationCost(orientation, violation_unit);
  return cost;
}

double DetailedRouter::getKnowWireCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  double prefer_wire_unit = dr_box.get_dr_parameter()->get_prefer_wire_unit();
  double non_prefer_wire_unit = dr_box.get_dr_parameter()->get_non_prefer_wire_unit();

  double wire_cost = 0;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    wire_cost += RTUTIL.getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());

    RoutingLayer& routing_layer = routing_layer_list[start_node->get_layer_idx()];
    if (routing_layer.get_prefer_direction() == RTUTIL.getDirection(*start_node, *end_node)) {
      wire_cost *= prefer_wire_unit;
    } else {
      wire_cost *= non_prefer_wire_unit;
    }
  }
  return wire_cost;
}

double DetailedRouter::getKnowCornerCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
{
  double corner_unit = dr_box.get_dr_parameter()->get_corner_unit();

  double corner_cost = 0;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    std::set<Direction> direction_set;
    // 添加start direction
    std::set<Direction>& start_direction_set = start_node->get_direction_set();
    direction_set.insert(start_direction_set.begin(), start_direction_set.end());
    // 添加start到parent的direction
    if (start_node->get_parent_node() != nullptr) {
      direction_set.insert(RTUTIL.getDirection(*start_node->get_parent_node(), *start_node));
    }
    // 添加end direction
    std::set<Direction>& end_direction_set = end_node->get_direction_set();
    direction_set.insert(end_direction_set.begin(), end_direction_set.end());
    // 添加start到end的direction
    direction_set.insert(RTUTIL.getDirection(*start_node, *end_node));

    if (direction_set.size() == 2) {
      corner_cost += corner_unit;
    } else if (direction_set.size() == 2) {
      RTLOG.error(Loc::current(), "Direction set is error!");
    }
  }
  return corner_cost;
}

double DetailedRouter::getKnowViaCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
{
  double via_unit = dr_box.get_dr_parameter()->get_via_unit();
  double via_cost = (via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
  return via_cost;
}

// calculate estimate cost

double DetailedRouter::getEstimateCostToEnd(DRBox& dr_box, DRNode* curr_node)
{
  std::vector<std::vector<DRNode*>>& end_node_list_list = dr_box.get_end_node_list_list();

  double estimate_cost = DBL_MAX;
  for (std::vector<DRNode*>& end_node_list : end_node_list_list) {
    for (DRNode* end_node : end_node_list) {
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
  estimate_cost += getEstimateWireCost(dr_box, start_node, end_node);
  estimate_cost += getEstimateCornerCost(dr_box, start_node, end_node);
  estimate_cost += getEstimateViaCost(dr_box, start_node, end_node);
  return estimate_cost;
}

double DetailedRouter::getEstimateWireCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
{
  double prefer_wire_unit = dr_box.get_dr_parameter()->get_prefer_wire_unit();

  double wire_cost = 0;
  wire_cost += RTUTIL.getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());
  wire_cost *= prefer_wire_unit;
  return wire_cost;
}

double DetailedRouter::getEstimateCornerCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
{
  double corner_unit = dr_box.get_dr_parameter()->get_corner_unit();

  double corner_cost = 0;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    if (RTUTIL.isOblique(*start_node, *end_node)) {
      corner_cost += corner_unit;
    }
  }
  return corner_cost;
}

double DetailedRouter::getEstimateViaCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
{
  double via_unit = dr_box.get_dr_parameter()->get_via_unit();
  double via_cost = (via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
  return via_cost;
}

void DetailedRouter::updateViolationList(DRBox& dr_box)
{
  std::vector<Violation> new_violation_list = getViolationList(dr_box);

  std::vector<Violation>& violation_list = dr_box.get_violation_list();
  // 原结果从graph删除
  for (Violation& violation : violation_list) {
    updateViolationToGraph(dr_box, ChangeType::kDel, violation);
  }
  violation_list = new_violation_list;
  // 新结果添加到graph
  for (Violation& violation : violation_list) {
    updateViolationToGraph(dr_box, ChangeType::kAdd, violation);
  }
}

std::vector<Violation> DetailedRouter::getViolationList(DRBox& dr_box)
{
  std::vector<idb::IdbLayerShape*> env_shape_list;
  std::map<int32_t, std::vector<idb::IdbLayerShape*>> net_pin_shape_map;
  for (auto& [is_routing, layer_net_fixed_rect_map] : dr_box.get_type_layer_net_fixed_rect_map()) {
    for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
      for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
        if (net_idx == -1) {
          for (auto& fixed_rect : fixed_rect_set) {
            env_shape_list.push_back(RTDM.getIDBLayerShapeByFixedRect(fixed_rect, is_routing));
          }
        } else {
          for (auto& fixed_rect : fixed_rect_set) {
            net_pin_shape_map[net_idx].push_back(RTDM.getIDBLayerShapeByFixedRect(fixed_rect, is_routing));
          }
        }
      }
    }
  }
  std::map<int32_t, std::vector<idb::IdbRegularWireSegment*>> net_wire_via_map;
  for (auto& [net_idx, segment_list] : dr_box.get_net_result_map()) {
    for (Segment<LayerCoord>& segment : segment_list) {
      net_wire_via_map[net_idx].push_back(RTDM.getIDBSegmentByNetResult(net_idx, segment));
    }
  }
  std::vector<Violation> violation_list = RTI.getViolationList(env_shape_list, net_pin_shape_map, net_wire_via_map, "DR");
  // free memory
  {
    for (idb::IdbLayerShape* env_shape : env_shape_list) {
      delete env_shape;
      env_shape = nullptr;
    }
    for (auto& [net_idx, pin_shape_list] : net_pin_shape_map) {
      for (idb::IdbLayerShape* pin_shape : pin_shape_list) {
        delete pin_shape;
        pin_shape = nullptr;
      }
    }
    for (auto& [net_idx, wire_via_list] : net_wire_via_map) {
      for (idb::IdbRegularWireSegment* wire_via : wire_via_list) {
        delete wire_via;
        wire_via = nullptr;
      }
    }
  }
  return violation_list;
}

void DetailedRouter::uploadViolation(DRBox& dr_box)
{
  for (Violation& violation : dr_box.get_violation_list()) {
    RTDM.updateViolationToGCellMap(ChangeType::kAdd, new Violation(violation));
  }
}

void DetailedRouter::freeDRBox(DRBox& dr_box)
{
  for (DRTask* dr_task : dr_box.get_dr_task_list()) {
    delete dr_task;
    dr_task = nullptr;
  }
  dr_box.get_dr_task_list().clear();
  dr_box.get_layer_node_map().clear();
}

int32_t DetailedRouter::getViolationNum()
{
  Die& die = RTDM.getDatabase().get_die();

  return static_cast<int32_t>(RTDM.getViolationSet(die).size());
}

void DetailedRouter::uploadNetResult(DRModel& dr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();
  for (auto& [net_idx, segment_set] : RTDM.getDetailedNetResultMap(die)) {
    for (Segment<LayerCoord>* segment : segment_set) {
      RTDM.updateDetailedNetResultToGCellMap(ChangeType::kDel, net_idx, segment);
    }
  }
  std::map<int32_t, std::vector<Segment<LayerCoord>>> net_result_map;
  GridMap<DRBox>& dr_box_map = dr_model.get_dr_box_map();
  for (int32_t x = 0; x < dr_box_map.get_x_size(); x++) {
    for (int32_t y = 0; y < dr_box_map.get_y_size(); y++) {
      DRBox& dr_box = dr_box_map[x][y];
      for (auto& [net_idx, segment_list] : dr_box.get_net_result_map()) {
        for (Segment<LayerCoord>& segment : segment_list) {
          net_result_map[net_idx].push_back(segment);
        }
      }
    }
  }
  std::vector<DRNet>& dr_net_list = dr_model.get_dr_net_list();
  for (auto& [net_idx, routing_segment_list] : net_result_map) {
    DRNet& dr_net = dr_net_list[net_idx];

    std::vector<LayerCoord> candidate_root_coord_list;
    std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
    std::vector<DRPin>& dr_pin_list = dr_net.get_dr_pin_list();
    for (size_t i = 0; i < dr_pin_list.size(); i++) {
      LayerCoord coord = dr_pin_list[i].get_key_access_point().getRealLayerCoord();
      candidate_root_coord_list.push_back(coord);
      key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
    }
    MTree<LayerCoord> coord_tree = RTUTIL.getTreeByFullFlow(candidate_root_coord_list, routing_segment_list, key_coord_pin_map);
    for (Segment<TNode<LayerCoord>*>& coord_segment : RTUTIL.getSegListByTree(coord_tree)) {
      Segment<LayerCoord>* segment = new Segment<LayerCoord>(coord_segment.get_first()->value(), coord_segment.get_second()->value());
      RTDM.updateDetailedNetResultToGCellMap(ChangeType::kAdd, net_idx, segment);
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

bool DetailedRouter::stopIteration(DRModel& dr_model)
{
  if (getViolationNum() == 0) {
    RTLOG.info(Loc::current(), "***** Iteration stopped early *****");
    return true;
  }
  return false;
}

#if 1  // update env

void DetailedRouter::updateFixedRectToGraph(DRBox& dr_box, ChangeType change_type, int32_t net_idx, EXTLayerRect* fixed_rect,
                                            bool is_routing)
{
  NetShape net_shape(net_idx, fixed_rect->getRealLayerRect(), is_routing);
  for (auto& [dr_node, orientation_set] : getNodeOrientationMap(dr_box, net_shape)) {
    if (!dr_node->get_is_valid()) {
      continue;
    }
    for (Orientation orientation : orientation_set) {
      if (change_type == ChangeType::kAdd) {
        dr_node->get_orient_fixed_rect_map()[orientation].insert(net_shape.get_net_idx());
      } else if (change_type == ChangeType::kDel) {
        dr_node->get_orient_fixed_rect_map()[orientation].erase(net_shape.get_net_idx());
      }
    }
  }
}

void DetailedRouter::updateNetResultToGraph(DRBox& dr_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment)
{
  for (NetShape& net_shape : RTDM.getNetShapeList(net_idx, segment)) {
    for (auto& [dr_node, orientation_set] : getNodeOrientationMap(dr_box, net_shape)) {
      if (!dr_node->get_is_valid()) {
        continue;
      }
      for (Orientation orientation : orientation_set) {
        if (change_type == ChangeType::kAdd) {
          dr_node->get_orient_routed_rect_map()[orientation].insert(net_shape.get_net_idx());
        } else if (change_type == ChangeType::kDel) {
          dr_node->get_orient_routed_rect_map()[orientation].erase(net_shape.get_net_idx());
        }
      }
    }
  }
}

void DetailedRouter::updateViolationToGraph(DRBox& dr_box, ChangeType change_type, Violation& violation)
{
  NetShape net_shape(-1, violation.get_violation_shape().getRealLayerRect(), violation.get_is_routing());
  for (auto& [dr_node, orientation_set] : getNodeOrientationMap(dr_box, net_shape)) {
    if (!dr_node->get_is_valid()) {
      continue;
    }
    for (Orientation orientation : orientation_set) {
      if (change_type == ChangeType::kAdd) {
        dr_node->get_orient_violation_number_map()[orientation]++;
      } else if (change_type == ChangeType::kDel) {
        dr_node->get_orient_violation_number_map()[orientation]--;
      }
    }
  }
}

std::map<DRNode*, std::set<Orientation>> DetailedRouter::getNodeOrientationMap(DRBox& dr_box, NetShape& net_shape)
{
  std::map<DRNode*, std::set<Orientation>> node_orientation_map;
  if (net_shape.get_is_routing()) {
    node_orientation_map = getRoutingNodeOrientationMap(dr_box, net_shape);
  } else {
    node_orientation_map = getCutNodeOrientationMap(dr_box, net_shape);
  }
  return node_orientation_map;
}

std::map<DRNode*, std::set<Orientation>> DetailedRouter::getRoutingNodeOrientationMap(DRBox& dr_box, NetShape& net_shape)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::map<int32_t, PlanarRect>& layer_enclosure_map = RTDM.getDatabase().get_layer_enclosure_map();
  if (!net_shape.get_is_routing()) {
    RTLOG.error(Loc::current(), "The type of net_shape is cut!");
  }
  int32_t layer_idx = net_shape.get_layer_idx();
  RoutingLayer& routing_layer = routing_layer_list[layer_idx];
  int32_t min_spacing = routing_layer.getMinSpacing(net_shape.get_rect());
  int32_t half_wire_width = routing_layer.get_min_width() / 2;
  PlanarRect& enclosure = layer_enclosure_map[layer_idx];
  int32_t enclosure_half_x_span = enclosure.getXSpan() / 2;
  int32_t enclosure_half_y_span = enclosure.getYSpan() / 2;

  GridMap<DRNode>& dr_node_map = dr_box.get_layer_node_map()[layer_idx];
  std::map<DRNode*, std::set<Orientation>> node_orientation_map;
  // wire
  {
    // 膨胀size为 min_spacing + half_wire_width
    int32_t enlarged_size = min_spacing + half_wire_width;
    // 贴合的也不算违例
    enlarged_size -= 1;
    PlanarRect planar_enlarged_rect = RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarged_size);
    for (auto& [grid_coord, orientation_set] : RTUTIL.getTrackGridOrientationMap(planar_enlarged_rect, dr_box.get_box_track_axis())) {
      DRNode& node = dr_node_map[grid_coord.get_x()][grid_coord.get_y()];
      for (const Orientation& orientation : orientation_set) {
        if (orientation == Orientation::kAbove || orientation == Orientation::kBelow) {
          continue;
        }
        if (!RTUTIL.exist(node.get_neighbor_node_map(), orientation)) {
          continue;
        }
        node_orientation_map[&node].insert(orientation);
        node_orientation_map[node.get_neighbor_node_map()[orientation]].insert(RTUTIL.getOppositeOrientation(orientation));
      }
    }
  }
  // via
  {
    // 膨胀size为 min_spacing + enclosure_half_span
    int32_t enlarged_x_size = min_spacing + enclosure_half_x_span;
    int32_t enlarged_y_size = min_spacing + enclosure_half_y_span;
    // 贴合的也不算违例
    enlarged_x_size -= 1;
    enlarged_y_size -= 1;
    PlanarRect space_enlarged_rect
        = RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size);
    for (auto& [grid_coord, orientation_set] : RTUTIL.getTrackGridOrientationMap(space_enlarged_rect, dr_box.get_box_track_axis())) {
      DRNode& node = dr_node_map[grid_coord.get_x()][grid_coord.get_y()];
      for (const Orientation& orientation : orientation_set) {
        if (orientation == Orientation::kEast || orientation == Orientation::kWest || orientation == Orientation::kSouth
            || orientation == Orientation::kNorth) {
          continue;
        }
        if (!RTUTIL.exist(node.get_neighbor_node_map(), orientation)) {
          continue;
        }
        node_orientation_map[&node].insert(orientation);
        node_orientation_map[node.get_neighbor_node_map()[orientation]].insert(RTUTIL.getOppositeOrientation(orientation));
      }
    }
  }
  return node_orientation_map;
}

std::map<DRNode*, std::set<Orientation>> DetailedRouter::getCutNodeOrientationMap(DRBox& dr_box, NetShape& net_shape)
{
  return {};
#if 0
  // 暂时关闭
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = RTDM.getDatabase().get_cut_to_adjacent_routing_map();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();
  if (net_shape.get_is_routing()) {
    RTLOG.error(Loc::current(), "The type of net_shape is routing!");
  }
  std::vector<int32_t> adjacent_routing_layer_idx_list = cut_to_adjacent_routing_map[net_shape.get_layer_idx()];
  if (adjacent_routing_layer_idx_list.size() != 2) {
    // 如果相邻层只有一个，那么将不会在此构建graph
    return {};
  }
  int32_t below_routing_layer_idx = adjacent_routing_layer_idx_list.front();
  int32_t above_routing_layer_idx = adjacent_routing_layer_idx_list.back();
  RTUTIL.swapByASC(below_routing_layer_idx, above_routing_layer_idx);

  // 膨胀size为 min_spacing + 当前cut层的half_length和half_width
  int32_t cut_spacing = cut_layer_list[net_shape.get_layer_idx()].getMinSpacing(net_shape.get_rect());
  PlanarRect& cut_shape = layer_via_master_list[below_routing_layer_idx].front().get_cut_shape_list().front();
  int32_t enlarge_x_size = cut_spacing + cut_shape.getXSpan() / 2;
  int32_t enlarge_y_size = cut_spacing + cut_shape.getYSpan() / 2;
  PlanarRect enlarged_rect = RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarge_x_size, enlarge_y_size, enlarge_x_size, enlarge_y_size);

  std::vector<GridMap<DRNode>>& layer_node_map = dr_box.get_layer_node_map();

  std::map<DRNode*, std::set<Orientation>> node_orientation_map;
  if (RTUTIL.existTrackGrid(enlarged_rect, dr_box.get_box_track_axis())) {
    PlanarRect grid_rect = RTUTIL.getTrackGridRect(enlarged_rect, dr_box.get_box_track_axis());
    for (int32_t grid_x = grid_rect.get_ll_x(); grid_x <= grid_rect.get_ur_x(); grid_x++) {
      for (int32_t grid_y = grid_rect.get_ll_y(); grid_y <= grid_rect.get_ur_y(); grid_y++) {
        node_orientation_map[&layer_node_map[below_routing_layer_idx][grid_x][grid_y]].insert(Orientation::kAbove);
        node_orientation_map[&layer_node_map[above_routing_layer_idx][grid_x][grid_y]].insert(Orientation::kBelow);
      }
    }
  }
  return node_orientation_map;
#endif
}

#endif

#if 1  // exhibit

void DetailedRouter::updateSummary(DRModel& dr_model)
{
  int32_t micron_dbu = RTDM.getDatabase().get_micron_dbu();
  Die& die = RTDM.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();
  int32_t enable_timing = RTDM.getConfig().enable_timing;
  std::map<int32_t, double>& routing_wire_length_map = RTDM.getSummary().iter_dr_summary_map[dr_model.get_iter()].routing_wire_length_map;
  double& total_wire_length = RTDM.getSummary().iter_dr_summary_map[dr_model.get_iter()].total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = RTDM.getSummary().iter_dr_summary_map[dr_model.get_iter()].cut_via_num_map;
  int32_t& total_via_num = RTDM.getSummary().iter_dr_summary_map[dr_model.get_iter()].total_via_num;
  std::map<int32_t, int32_t>& routing_violation_num_map
      = RTDM.getSummary().iter_dr_summary_map[dr_model.get_iter()].routing_violation_num_map;
  int32_t& total_violation_num = RTDM.getSummary().iter_dr_summary_map[dr_model.get_iter()].total_violation_num;
  std::map<std::string, std::vector<double>>& timing = RTDM.getSummary().iter_dr_summary_map[dr_model.get_iter()].timing;

  std::vector<DRNet>& dr_net_list = dr_model.get_dr_net_list();

  for (RoutingLayer& routing_layer : routing_layer_list) {
    routing_wire_length_map[routing_layer.get_layer_idx()] = 0;
    routing_violation_num_map[routing_layer.get_layer_idx()] = 0;
  }
  total_wire_length = 0;
  total_violation_num = 0;
  for (CutLayer& cut_layer : cut_layer_list) {
    cut_via_num_map[cut_layer.get_layer_idx()] = 0;
  }
  total_via_num = 0;

  for (auto& [net_idx, segment_set] : RTDM.getDetailedNetResultMap(die)) {
    for (Segment<LayerCoord>* segment : segment_set) {
      LayerCoord& first_coord = segment->get_first();
      int32_t first_layer_idx = first_coord.get_layer_idx();
      LayerCoord& second_coord = segment->get_second();
      int32_t second_layer_idx = second_coord.get_layer_idx();

      if (first_layer_idx == second_layer_idx) {
        double wire_length = RTUTIL.getManhattanDistance(first_coord, second_coord) / 1.0 / micron_dbu;
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
  for (Violation* violation : RTDM.getViolationSet(die)) {
    routing_violation_num_map[violation->get_violation_shape().get_layer_idx()]++;
    total_violation_num++;
  }
  if (enable_timing) {
    std::vector<std::map<std::string, std::vector<LayerCoord>>> real_pin_coord_map_list;
    real_pin_coord_map_list.resize(dr_net_list.size());
    std::vector<std::vector<Segment<LayerCoord>>> routing_segment_list_list;
    routing_segment_list_list.resize(dr_net_list.size());
    for (DRNet& dr_net : dr_net_list) {
      for (DRPin& dr_pin : dr_net.get_dr_pin_list()) {
        real_pin_coord_map_list[dr_net.get_net_idx()][dr_pin.get_pin_name()].push_back(dr_pin.get_key_access_point().getRealLayerCoord());
      }
    }
    for (auto& [net_idx, segment_set] : RTDM.getDetailedNetResultMap(die)) {
      for (Segment<LayerCoord>* segment : segment_set) {
        routing_segment_list_list[net_idx].emplace_back(segment->get_first(), segment->get_second());
      }
    }
    timing = RTI.getTiming(real_pin_coord_map_list, routing_segment_list_list);
  }
}

void DetailedRouter::printSummary(DRModel& dr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  std::map<int32_t, double>& routing_wire_length_map = RTDM.getSummary().iter_dr_summary_map[dr_model.get_iter()].routing_wire_length_map;
  double& total_wire_length = RTDM.getSummary().iter_dr_summary_map[dr_model.get_iter()].total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = RTDM.getSummary().iter_dr_summary_map[dr_model.get_iter()].cut_via_num_map;
  int32_t& total_via_num = RTDM.getSummary().iter_dr_summary_map[dr_model.get_iter()].total_via_num;
  std::map<int32_t, int32_t>& routing_violation_num_map
      = RTDM.getSummary().iter_dr_summary_map[dr_model.get_iter()].routing_violation_num_map;
  int32_t& total_violation_num = RTDM.getSummary().iter_dr_summary_map[dr_model.get_iter()].total_violation_num;

  fort::char_table routing_wire_length_map_table;
  {
    routing_wire_length_map_table << fort::header << "routing_layer" << "wire_length" << "proportion" << fort::endr;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      routing_wire_length_map_table << routing_layer.get_layer_name() << routing_wire_length_map[routing_layer.get_layer_idx()]
                                    << RTUTIL.getPercentage(routing_wire_length_map[routing_layer.get_layer_idx()], total_wire_length)
                                    << fort::endr;
    }
    routing_wire_length_map_table << fort::header << "Total" << total_wire_length
                                  << RTUTIL.getPercentage(total_wire_length, total_wire_length) << fort::endr;
  }
  fort::char_table cut_via_num_map_table;
  {
    cut_via_num_map_table << fort::header << "cut_layer" << "via_num" << "proportion" << fort::endr;
    for (CutLayer& cut_layer : cut_layer_list) {
      cut_via_num_map_table << cut_layer.get_layer_name() << cut_via_num_map[cut_layer.get_layer_idx()]
                            << RTUTIL.getPercentage(cut_via_num_map[cut_layer.get_layer_idx()], total_via_num) << fort::endr;
    }
    cut_via_num_map_table << fort::header << "Total" << total_via_num << RTUTIL.getPercentage(total_via_num, total_via_num) << fort::endr;
  }
  fort::char_table routing_violation_num_map_table;
  {
    routing_violation_num_map_table << fort::header << "routing_layer" << "violation_num" << "proportion" << fort::endr;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      routing_violation_num_map_table << routing_layer.get_layer_name() << routing_violation_num_map[routing_layer.get_layer_idx()]
                                      << RTUTIL.getPercentage(routing_violation_num_map[routing_layer.get_layer_idx()], total_violation_num)
                                      << fort::endr;
    }
    routing_violation_num_map_table << fort::header << "Total" << total_violation_num
                                    << RTUTIL.getPercentage(total_violation_num, total_violation_num) << fort::endr;
  }
  RTUTIL.printTableList({routing_wire_length_map_table, cut_via_num_map_table, routing_violation_num_map_table});
}

void DetailedRouter::writeNetCSV(DRModel& dr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::string& dr_temp_directory_path = RTDM.getConfig().dr_temp_directory_path;
  int32_t output_csv = RTDM.getConfig().output_csv;
  if (!output_csv) {
    return;
  }
  std::vector<GridMap<int32_t>> layer_net_map;
  layer_net_map.resize(routing_layer_list.size());
  for (GridMap<int32_t>& net_map : layer_net_map) {
    net_map.init(gcell_map.get_x_size(), gcell_map.get_y_size());
  }
  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      std::map<int32_t, std::set<int32_t>> net_layer_map;
      for (auto& [net_idx, segment_set] : gcell_map[x][y].get_detailed_net_result_map()) {
        for (Segment<LayerCoord>* segment : segment_set) {
          int32_t first_layer_idx = segment->get_first().get_layer_idx();
          int32_t second_layer_idx = segment->get_second().get_layer_idx();
          RTUTIL.swapByASC(first_layer_idx, second_layer_idx);
          for (int32_t layer_idx = first_layer_idx; layer_idx <= second_layer_idx; layer_idx++) {
            net_layer_map[net_idx].insert(layer_idx);
          }
        }
      }
      for (auto& [net_idx, layer_set] : net_layer_map) {
        for (int32_t layer_idx : layer_set) {
          layer_net_map[layer_idx][x][y]++;
        }
      }
    }
  }
  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::ofstream* net_csv_file = RTUTIL.getOutputFileStream(
        RTUTIL.getString(dr_temp_directory_path, "net_map_", routing_layer.get_layer_name(), "_", dr_model.get_iter(), ".csv"));
    GridMap<int32_t>& net_map = layer_net_map[routing_layer.get_layer_idx()];
    for (int32_t y = net_map.get_y_size() - 1; y >= 0; y--) {
      for (int32_t x = 0; x < net_map.get_x_size(); x++) {
        RTUTIL.pushStream(net_csv_file, net_map[x][y], ",");
      }
      RTUTIL.pushStream(net_csv_file, "\n");
    }
    RTUTIL.closeFileStream(net_csv_file);
  }
}

void DetailedRouter::writeViolationCSV(DRModel& dr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::string& dr_temp_directory_path = RTDM.getConfig().dr_temp_directory_path;
  int32_t output_csv = RTDM.getConfig().output_csv;
  if (!output_csv) {
    return;
  }
  std::vector<GridMap<int32_t>> layer_violation_map;
  layer_violation_map.resize(routing_layer_list.size());
  for (GridMap<int32_t>& violation_map : layer_violation_map) {
    violation_map.init(gcell_map.get_x_size(), gcell_map.get_y_size());
  }
  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      for (Violation* violation : gcell_map[x][y].get_violation_set()) {
        layer_violation_map[violation->get_violation_shape().get_layer_idx()][x][y]++;
      }
    }
  }
  for (RoutingLayer& routing_layer : routing_layer_list) {
    std::ofstream* violation_csv_file = RTUTIL.getOutputFileStream(
        RTUTIL.getString(dr_temp_directory_path, "violation_map_", routing_layer.get_layer_name(), "_", dr_model.get_iter(), ".csv"));
    GridMap<int32_t>& violation_map = layer_violation_map[routing_layer.get_layer_idx()];
    for (int32_t y = violation_map.get_y_size() - 1; y >= 0; y--) {
      for (int32_t x = 0; x < violation_map.get_x_size(); x++) {
        RTUTIL.pushStream(violation_csv_file, violation_map[x][y], ",");
      }
      RTUTIL.pushStream(violation_csv_file, "\n");
    }
    RTUTIL.closeFileStream(violation_csv_file);
  }
}

#endif

#if 1  // debug

void DetailedRouter::debugCheckDRBox(DRBox& dr_box)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  DRBoxId& dr_box_id = dr_box.get_dr_box_id();
  if (dr_box_id.get_x() < 0 || dr_box_id.get_y() < 0) {
    RTLOG.error(Loc::current(), "The grid coord is illegal!");
  }

  std::vector<GridMap<DRNode>>& layer_node_map = dr_box.get_layer_node_map();
  for (GridMap<DRNode>& dr_node_map : layer_node_map) {
    for (int32_t x = 0; x < dr_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < dr_node_map.get_y_size(); y++) {
        DRNode& dr_node = dr_node_map[x][y];
        if (!RTUTIL.isInside(dr_box.get_box_rect().get_real_rect(), dr_node.get_planar_coord())) {
          RTLOG.error(Loc::current(), "The dr_node is out of box!");
        }
        for (auto& [orient, neighbor] : dr_node.get_neighbor_node_map()) {
          Orientation opposite_orient = RTUTIL.getOppositeOrientation(orient);
          if (!RTUTIL.exist(neighbor->get_neighbor_node_map(), opposite_orient)) {
            RTLOG.error(Loc::current(), "The dr_node neighbor is not bidirectional!");
          }
          if (neighbor->get_neighbor_node_map()[opposite_orient] != &dr_node) {
            RTLOG.error(Loc::current(), "The dr_node neighbor is not bidirectional!");
          }
          LayerCoord node_coord(dr_node.get_planar_coord(), dr_node.get_layer_idx());
          LayerCoord neighbor_coord(neighbor->get_planar_coord(), neighbor->get_layer_idx());
          if (RTUTIL.getOrientation(node_coord, neighbor_coord) == orient) {
            continue;
          }
          RTLOG.error(Loc::current(), "The neighbor orient is different with real region!");
        }
      }
    }
  }

  for (DRTask* dr_task : dr_box.get_dr_task_list()) {
    if (dr_task->get_net_idx() < 0) {
      RTLOG.error(Loc::current(), "The idx of origin net is illegal!");
    }
    for (DRGroup& dr_group : dr_task->get_dr_group_list()) {
      if (dr_group.get_coord_direction_map().empty()) {
        RTLOG.error(Loc::current(), "The coord_direction_map is empty!");
      }
      for (auto& [coord, direction_set] : dr_group.get_coord_direction_map()) {
        int32_t layer_idx = coord.get_layer_idx();
        if (routing_layer_list.back().get_layer_idx() < layer_idx || layer_idx < routing_layer_list.front().get_layer_idx()) {
          RTLOG.error(Loc::current(), "The layer idx of group coord is illegal!");
        }
        if (!RTUTIL.existTrackGrid(coord, dr_box.get_box_track_axis())) {
          RTLOG.error(Loc::current(), "There is no grid coord for real coord(", coord.get_x(), ",", coord.get_y(), ")!");
        }
        PlanarCoord grid_coord = RTUTIL.getTrackGrid(coord, dr_box.get_box_track_axis());
        DRNode& dr_node = layer_node_map[layer_idx][grid_coord.get_x()][grid_coord.get_y()];
        if (dr_node.get_neighbor_node_map().empty()) {
          RTLOG.error(Loc::current(), "The neighbor of group coord (", coord.get_x(), ",", coord.get_y(), ",", layer_idx,
                      ") is empty in box(", dr_box_id.get_x(), ",", dr_box_id.get_y(), ")");
        }
        if (RTUTIL.isInside(dr_box.get_box_rect().get_real_rect(), coord)) {
          continue;
        }
        RTLOG.error(Loc::current(), "The coord (", coord.get_x(), ",", coord.get_y(), ") is out of box!");
      }
    }
  }
}

void DetailedRouter::debugPlotDRBox(DRBox& dr_box, int32_t curr_task_idx, std::string flag)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();
  std::string& dr_temp_directory_path = RTDM.getConfig().dr_temp_directory_path;

  PlanarRect box_rect = dr_box.get_box_rect().get_real_rect();

  int32_t width = INT32_MAX;
  for (ScaleGrid& x_grid : dr_box.get_box_track_axis().get_x_grid_list()) {
    width = std::min(width, x_grid.get_step_length());
  }
  for (ScaleGrid& y_grid : dr_box.get_box_track_axis().get_y_grid_list()) {
    width = std::min(width, y_grid.get_step_length());
  }
  width = std::max(1, width / 3);

  GPGDS gp_gds;

  // base_region
  GPStruct base_region_struct("base_region");
  GPBoundary gp_boundary;
  gp_boundary.set_layer_idx(0);
  gp_boundary.set_data_type(0);
  gp_boundary.set_rect(box_rect);
  base_region_struct.push(gp_boundary);
  gp_gds.addStruct(base_region_struct);

  // gcell_axis
  GPStruct gcell_axis_struct("gcell_axis");
  for (int32_t x : RTUTIL.getScaleList(box_rect.get_ll_x(), box_rect.get_ur_x(), gcell_axis.get_x_grid_list())) {
    GPPath gp_path;
    gp_path.set_layer_idx(0);
    gp_path.set_data_type(1);
    gp_path.set_segment(x, box_rect.get_ll_y(), x, box_rect.get_ur_y());
    gcell_axis_struct.push(gp_path);
  }
  for (int32_t y : RTUTIL.getScaleList(box_rect.get_ll_y(), box_rect.get_ur_y(), gcell_axis.get_y_grid_list())) {
    GPPath gp_path;
    gp_path.set_layer_idx(0);
    gp_path.set_data_type(1);
    gp_path.set_segment(box_rect.get_ll_x(), y, box_rect.get_ur_x(), y);
    gcell_axis_struct.push(gp_path);
  }
  gp_gds.addStruct(gcell_axis_struct);

  std::vector<GridMap<DRNode>>& layer_node_map = dr_box.get_layer_node_map();
  // dr_node_map
  GPStruct dr_node_map_struct("dr_node_map");
  for (GridMap<DRNode>& dr_node_map : layer_node_map) {
    for (int32_t grid_x = 0; grid_x < dr_node_map.get_x_size(); grid_x++) {
      for (int32_t grid_y = 0; grid_y < dr_node_map.get_y_size(); grid_y++) {
        DRNode& dr_node = dr_node_map[grid_x][grid_y];
        PlanarRect real_rect = RTUTIL.getEnlargedRect(dr_node.get_planar_coord(), width);
        int32_t y_reduced_span = std::max(1, real_rect.getYSpan() / 12);
        int32_t y = real_rect.get_ur_y();

        GPBoundary gp_boundary;
        switch (dr_node.get_state()) {
          case DRNodeState::kNone:
            gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kNone));
            break;
          case DRNodeState::kOpen:
            gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kOpen));
            break;
          case DRNodeState::kClose:
            gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kClose));
            break;
          default:
            RTLOG.error(Loc::current(), "The type is error!");
            break;
        }
        gp_boundary.set_rect(real_rect);
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(dr_node.get_layer_idx()));
        dr_node_map_struct.push(gp_boundary);

        y -= y_reduced_span;
        GPText gp_text_node_real_coord;
        gp_text_node_real_coord.set_coord(real_rect.get_ll_x(), y);
        gp_text_node_real_coord.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
        gp_text_node_real_coord.set_message(
            RTUTIL.getString("(", dr_node.get_x(), " , ", dr_node.get_y(), " , ", dr_node.get_layer_idx(), ")"));
        gp_text_node_real_coord.set_layer_idx(RTGP.getGDSIdxByRouting(dr_node.get_layer_idx()));
        gp_text_node_real_coord.set_presentation(GPTextPresentation::kLeftMiddle);
        dr_node_map_struct.push(gp_text_node_real_coord);

        y -= y_reduced_span;
        GPText gp_text_node_grid_coord;
        gp_text_node_grid_coord.set_coord(real_rect.get_ll_x(), y);
        gp_text_node_grid_coord.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
        gp_text_node_grid_coord.set_message(RTUTIL.getString("(", grid_x, " , ", grid_y, " , ", dr_node.get_layer_idx(), ")"));
        gp_text_node_grid_coord.set_layer_idx(RTGP.getGDSIdxByRouting(dr_node.get_layer_idx()));
        gp_text_node_grid_coord.set_presentation(GPTextPresentation::kLeftMiddle);
        dr_node_map_struct.push(gp_text_node_grid_coord);

        y -= y_reduced_span;
        GPText gp_text_orient_fixed_rect_map;
        gp_text_orient_fixed_rect_map.set_coord(real_rect.get_ll_x(), y);
        gp_text_orient_fixed_rect_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
        gp_text_orient_fixed_rect_map.set_message("orient_fixed_rect_map: ");
        gp_text_orient_fixed_rect_map.set_layer_idx(RTGP.getGDSIdxByRouting(dr_node.get_layer_idx()));
        gp_text_orient_fixed_rect_map.set_presentation(GPTextPresentation::kLeftMiddle);
        dr_node_map_struct.push(gp_text_orient_fixed_rect_map);

        if (!dr_node.get_orient_fixed_rect_map().empty()) {
          y -= y_reduced_span;
          GPText gp_text_orient_fixed_rect_map_info;
          gp_text_orient_fixed_rect_map_info.set_coord(real_rect.get_ll_x(), y);
          gp_text_orient_fixed_rect_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          std::string orient_fixed_rect_map_info_message = "--";
          for (auto& [orient, net_set] : dr_node.get_orient_fixed_rect_map()) {
            orient_fixed_rect_map_info_message += RTUTIL.getString("(", GetOrientationName()(orient));
            for (int32_t net_idx : net_set) {
              orient_fixed_rect_map_info_message += RTUTIL.getString(",", net_idx);
            }
            orient_fixed_rect_map_info_message += RTUTIL.getString(")");
          }
          gp_text_orient_fixed_rect_map_info.set_message(orient_fixed_rect_map_info_message);
          gp_text_orient_fixed_rect_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(dr_node.get_layer_idx()));
          gp_text_orient_fixed_rect_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
          dr_node_map_struct.push(gp_text_orient_fixed_rect_map_info);
        }

        y -= y_reduced_span;
        GPText gp_text_orient_routed_rect_map;
        gp_text_orient_routed_rect_map.set_coord(real_rect.get_ll_x(), y);
        gp_text_orient_routed_rect_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
        gp_text_orient_routed_rect_map.set_message("orient_routed_rect_map: ");
        gp_text_orient_routed_rect_map.set_layer_idx(RTGP.getGDSIdxByRouting(dr_node.get_layer_idx()));
        gp_text_orient_routed_rect_map.set_presentation(GPTextPresentation::kLeftMiddle);
        dr_node_map_struct.push(gp_text_orient_routed_rect_map);

        if (!dr_node.get_orient_routed_rect_map().empty()) {
          y -= y_reduced_span;
          GPText gp_text_orient_routed_rect_map_info;
          gp_text_orient_routed_rect_map_info.set_coord(real_rect.get_ll_x(), y);
          gp_text_orient_routed_rect_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          std::string orient_routed_rect_map_info_message = "--";
          for (auto& [orient, net_set] : dr_node.get_orient_routed_rect_map()) {
            orient_routed_rect_map_info_message += RTUTIL.getString("(", GetOrientationName()(orient));
            for (int32_t net_idx : net_set) {
              orient_routed_rect_map_info_message += RTUTIL.getString(",", net_idx);
            }
            orient_routed_rect_map_info_message += RTUTIL.getString(")");
          }
          gp_text_orient_routed_rect_map_info.set_message(orient_routed_rect_map_info_message);
          gp_text_orient_routed_rect_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(dr_node.get_layer_idx()));
          gp_text_orient_routed_rect_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
          dr_node_map_struct.push(gp_text_orient_routed_rect_map_info);
        }

        y -= y_reduced_span;
        GPText gp_text_orient_violation_number_map;
        gp_text_orient_violation_number_map.set_coord(real_rect.get_ll_x(), y);
        gp_text_orient_violation_number_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
        gp_text_orient_violation_number_map.set_message("orient_violation_number_map: ");
        gp_text_orient_violation_number_map.set_layer_idx(RTGP.getGDSIdxByRouting(dr_node.get_layer_idx()));
        gp_text_orient_violation_number_map.set_presentation(GPTextPresentation::kLeftMiddle);
        dr_node_map_struct.push(gp_text_orient_violation_number_map);

        if (!dr_node.get_orient_violation_number_map().empty()) {
          y -= y_reduced_span;
          GPText gp_text_orient_violation_number_map_info;
          gp_text_orient_violation_number_map_info.set_coord(real_rect.get_ll_x(), y);
          gp_text_orient_violation_number_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          std::string orient_violation_number_map_info_message = "--";
          for (auto& [orient, violation_number] : dr_node.get_orient_violation_number_map()) {
            orient_violation_number_map_info_message
                += RTUTIL.getString("(", GetOrientationName()(orient), ",", violation_number != 0, ")");
          }
          gp_text_orient_violation_number_map_info.set_message(orient_violation_number_map_info_message);
          gp_text_orient_violation_number_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(dr_node.get_layer_idx()));
          gp_text_orient_violation_number_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
          dr_node_map_struct.push(gp_text_orient_violation_number_map_info);
        }

        y -= y_reduced_span;
        GPText gp_text_direction_set;
        gp_text_direction_set.set_coord(real_rect.get_ll_x(), y);
        gp_text_direction_set.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
        gp_text_direction_set.set_message("direction_set: ");
        gp_text_direction_set.set_layer_idx(RTGP.getGDSIdxByRouting(dr_node.get_layer_idx()));
        gp_text_direction_set.set_presentation(GPTextPresentation::kLeftMiddle);
        dr_node_map_struct.push(gp_text_direction_set);

        if (!dr_node.get_direction_set().empty()) {
          y -= y_reduced_span;
          GPText gp_text_direction_set_info;
          gp_text_direction_set_info.set_coord(real_rect.get_ll_x(), y);
          gp_text_direction_set_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          std::string direction_set_info_message = "--";
          for (Direction direction : dr_node.get_direction_set()) {
            direction_set_info_message += RTUTIL.getString("(", GetDirectionName()(direction), ")");
          }
          gp_text_direction_set_info.set_message(direction_set_info_message);
          gp_text_direction_set_info.set_layer_idx(RTGP.getGDSIdxByRouting(dr_node.get_layer_idx()));
          gp_text_direction_set_info.set_presentation(GPTextPresentation::kLeftMiddle);
          dr_node_map_struct.push(gp_text_direction_set_info);
        }
      }
    }
  }
  gp_gds.addStruct(dr_node_map_struct);

  // neighbor_map
  GPStruct neighbor_map_struct("neighbor_map");
  for (GridMap<DRNode>& dr_node_map : layer_node_map) {
    for (int32_t grid_x = 0; grid_x < dr_node_map.get_x_size(); grid_x++) {
      for (int32_t grid_y = 0; grid_y < dr_node_map.get_y_size(); grid_y++) {
        DRNode& dr_node = dr_node_map[grid_x][grid_y];
        PlanarRect real_rect = RTUTIL.getEnlargedRect(dr_node.get_planar_coord(), width);

        int32_t ll_x = real_rect.get_ll_x();
        int32_t ll_y = real_rect.get_ll_y();
        int32_t ur_x = real_rect.get_ur_x();
        int32_t ur_y = real_rect.get_ur_y();
        int32_t mid_x = (ll_x + ur_x) / 2;
        int32_t mid_y = (ll_y + ur_y) / 2;
        int32_t x_reduced_span = (ur_x - ll_x) / 4;
        int32_t y_reduced_span = (ur_y - ll_y) / 4;
        int32_t width = std::min(x_reduced_span, y_reduced_span) / 2;

        for (auto& [orientation, neighbor_node] : dr_node.get_neighbor_node_map()) {
          GPPath gp_path;
          switch (orientation) {
            case Orientation::kEast:
              gp_path.set_segment(ur_x - x_reduced_span, mid_y, ur_x, mid_y);
              break;
            case Orientation::kSouth:
              gp_path.set_segment(mid_x, ll_y, mid_x, ll_y + y_reduced_span);
              break;
            case Orientation::kWest:
              gp_path.set_segment(ll_x, mid_y, ll_x + x_reduced_span, mid_y);
              break;
            case Orientation::kNorth:
              gp_path.set_segment(mid_x, ur_y - y_reduced_span, mid_x, ur_y);
              break;
            case Orientation::kAbove:
              gp_path.set_segment(ur_x - x_reduced_span, ur_y - y_reduced_span, ur_x, ur_y);
              break;
            case Orientation::kBelow:
              gp_path.set_segment(ll_x, ll_y, ll_x + x_reduced_span, ll_y + y_reduced_span);
              break;
            default:
              RTLOG.error(Loc::current(), "The orientation is oblique!");
              break;
          }
          gp_path.set_layer_idx(RTGP.getGDSIdxByRouting(dr_node.get_layer_idx()));
          gp_path.set_width(width);
          gp_path.set_data_type(static_cast<int32_t>(GPDataType::kNeighbor));
          neighbor_map_struct.push(gp_path);
        }
      }
    }
  }
  gp_gds.addStruct(neighbor_map_struct);

  // box_track_axis
  GPStruct box_track_axis_struct("box_track_axis");
  PlanarCoord& real_ll = box_rect.get_ll();
  PlanarCoord& real_ur = box_rect.get_ur();
  ScaleAxis& box_track_axis = dr_box.get_box_track_axis();
  std::vector<int32_t> x_list = RTUTIL.getScaleList(real_ll.get_x(), real_ur.get_x(), box_track_axis.get_x_grid_list());
  std::vector<int32_t> y_list = RTUTIL.getScaleList(real_ll.get_y(), real_ur.get_y(), box_track_axis.get_y_grid_list());
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
#if 1
    RoutingLayer& routing_layer = routing_layer_list[layer_idx];
    x_list = RTUTIL.getScaleList(real_ll.get_x(), real_ur.get_x(), routing_layer.getXTrackGridList());
    y_list = RTUTIL.getScaleList(real_ll.get_y(), real_ur.get_y(), routing_layer.getYTrackGridList());
#endif
    for (int32_t x : x_list) {
      GPPath gp_path;
      gp_path.set_data_type(static_cast<int32_t>(GPDataType::kAxis));
      gp_path.set_segment(x, real_ll.get_y(), x, real_ur.get_y());
      gp_path.set_layer_idx(RTGP.getGDSIdxByRouting(layer_idx));
      box_track_axis_struct.push(gp_path);
    }
    for (int32_t y : y_list) {
      GPPath gp_path;
      gp_path.set_data_type(static_cast<int32_t>(GPDataType::kAxis));
      gp_path.set_segment(real_ll.get_x(), y, real_ur.get_x(), y);
      gp_path.set_layer_idx(RTGP.getGDSIdxByRouting(layer_idx));
      box_track_axis_struct.push(gp_path);
    }
  }
  gp_gds.addStruct(box_track_axis_struct);

  for (auto& [is_routing, layer_net_rect_map] : dr_box.get_type_layer_net_fixed_rect_map()) {
    for (auto& [layer_idx, net_rect_map] : layer_net_rect_map) {
      for (auto& [net_idx, rect_set] : net_rect_map) {
        GPStruct fixed_rect_struct(RTUTIL.getString("fixed_rect(net_", net_idx, ")"));
        for (EXTLayerRect* rect : rect_set) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShape));
          gp_boundary.set_rect(rect->get_real_rect());
          if (is_routing) {
            gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(layer_idx));
          } else {
            gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(layer_idx));
          }
          fixed_rect_struct.push(gp_boundary);
        }
        gp_gds.addStruct(fixed_rect_struct);
      }
    }
  }

  GPStruct violation_struct("violation");
  for (Violation& violation : dr_box.get_violation_list()) {
    EXTLayerRect& violation_shape = violation.get_violation_shape();

    GPBoundary gp_boundary;
    gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kViolation));
    gp_boundary.set_rect(violation_shape.get_real_rect());
    if (violation.get_is_routing()) {
      gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(violation_shape.get_layer_idx()));
    } else {
      gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(violation_shape.get_layer_idx()));
    }
    violation_struct.push(gp_boundary);
  }
  gp_gds.addStruct(violation_struct);

  // task
  for (DRTask* dr_task : dr_box.get_dr_task_list()) {
    GPStruct task_struct(RTUTIL.getString("task(net_", dr_task->get_net_idx(), ")"));

    if (curr_task_idx == -1 || dr_task->get_net_idx() == curr_task_idx) {
      for (DRGroup& dr_group : dr_task->get_dr_group_list()) {
        for (auto& [coord, direction_set] : dr_group.get_coord_direction_map()) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kKey));
          gp_boundary.set_rect(RTUTIL.getEnlargedRect(coord, width));
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(coord.get_layer_idx()));
          task_struct.push(gp_boundary);
        }
      }
    }
    // {
    //   // bounding_box
    //   GPBoundary gp_boundary;
    //   gp_boundary.set_layer_idx(0);
    //   gp_boundary.set_data_type(2);
    //   gp_boundary.set_rect(dr_task.get_bounding_box().get_base_region());
    //   task_struct.push(gp_boundary);
    // }
    for (Segment<LayerCoord>& segment : dr_box.get_net_result_map()[dr_task->get_net_idx()]) {
      LayerCoord first_coord = segment.get_first();
      LayerCoord second_coord = segment.get_second();
      int32_t first_layer_idx = first_coord.get_layer_idx();
      int32_t second_layer_idx = second_coord.get_layer_idx();
      int32_t half_width = routing_layer_list[first_layer_idx].get_min_width() / 2;

      if (first_layer_idx == second_layer_idx) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPath));
        gp_boundary.set_rect(RTUTIL.getEnlargedRect(first_coord, second_coord, half_width));
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(first_layer_idx));
        task_struct.push(gp_boundary);
      } else {
        RTUTIL.swapByASC(first_layer_idx, second_layer_idx);
        for (int32_t layer_idx = first_layer_idx; layer_idx < second_layer_idx; layer_idx++) {
          ViaMaster& via_master = layer_via_master_list[layer_idx].front();

          LayerRect& above_enclosure = via_master.get_above_enclosure();
          LayerRect offset_above_enclosure(RTUTIL.getOffsetRect(above_enclosure, first_coord), above_enclosure.get_layer_idx());
          GPBoundary above_boundary;
          above_boundary.set_rect(offset_above_enclosure);
          above_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(above_enclosure.get_layer_idx()));
          above_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPath));
          task_struct.push(above_boundary);

          LayerRect& below_enclosure = via_master.get_below_enclosure();
          LayerRect offset_below_enclosure(RTUTIL.getOffsetRect(below_enclosure, first_coord), below_enclosure.get_layer_idx());
          GPBoundary below_boundary;
          below_boundary.set_rect(offset_below_enclosure);
          below_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(below_enclosure.get_layer_idx()));
          below_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPath));
          task_struct.push(below_boundary);

          for (PlanarRect& cut_shape : via_master.get_cut_shape_list()) {
            LayerRect offset_cut_shape(RTUTIL.getOffsetRect(cut_shape, first_coord), via_master.get_cut_layer_idx());
            GPBoundary cut_boundary;
            cut_boundary.set_rect(offset_cut_shape);
            cut_boundary.set_layer_idx(RTGP.getGDSIdxByCut(via_master.get_cut_layer_idx()));
            cut_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPath));
            task_struct.push(cut_boundary);
          }
        }
      }
    }
    gp_gds.addStruct(task_struct);
  }
  std::string gds_file_path = RTUTIL.getString(dr_temp_directory_path, flag, "_dr_box_", dr_box.get_dr_box_id().get_x(), "_",
                                               dr_box.get_dr_box_id().get_y(), ".gds");
  RTGP.plot(gp_gds, gds_file_path);
}

#endif

}  // namespace irt
