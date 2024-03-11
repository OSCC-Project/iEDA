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
#include "RTAPI.hpp"

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

// function

void DetailedRouter::route()
{
  Monitor monitor;
  LOG_INST.info(Loc::current(), "Begin routing...");
  DRModel dr_model = initDRModel();
  iterativeDRModel(dr_model);
  LOG_INST.info(Loc::current(), "End route", monitor.getStatsInfo());
}

// private

DetailedRouter* DetailedRouter::_dr_instance = nullptr;

DRModel DetailedRouter::initDRModel()
{
  std::vector<Net>& net_list = DM_INST.getDatabase().get_net_list();

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
      {8, 0, cost_unit, cost_unit, 4 * cost_unit, true},
      {8, -2, cost_unit, cost_unit, 4 * cost_unit, true},
      {8, -4, cost_unit, cost_unit, 4 * cost_unit, true},
      {8, -6, cost_unit, cost_unit, 4 * cost_unit, true},
      {8, 0, 8 * cost_unit, 4 * cost_unit, 4 * cost_unit, false},
      {8, -1, 8 * cost_unit, 4 * cost_unit, 4 * cost_unit, false},
      {8, -2, 8 * cost_unit, 4 * cost_unit, 4 * cost_unit, false},
      {8, -3, 8 * cost_unit, 4 * cost_unit, 4 * cost_unit, false},
      {8, -4, 8 * cost_unit, 4 * cost_unit, 4 * cost_unit, false},
      {8, -5, 8 * cost_unit, 4 * cost_unit, 4 * cost_unit, false},
      {8, -6, 8 * cost_unit, 4 * cost_unit, 4 * cost_unit, false},
      {8, -7, 8 * cost_unit, 4 * cost_unit, 4 * cost_unit, false},
      {8, 0, 16 * cost_unit, 8 * cost_unit, 8 * cost_unit, false},
      {8, -1, 16 * cost_unit, 8 * cost_unit, 8 * cost_unit, false},
      {8, -2, 16 * cost_unit, 8 * cost_unit, 8 * cost_unit, false},
      {8, -3, 16 * cost_unit, 8 * cost_unit, 8 * cost_unit, false},
      {8, -4, 16 * cost_unit, 8 * cost_unit, 8 * cost_unit, false},
      {8, -5, 16 * cost_unit, 8 * cost_unit, 8 * cost_unit, false},
      {8, -6, 16 * cost_unit, 8 * cost_unit, 8 * cost_unit, false},
      {8, -7, 16 * cost_unit, 8 * cost_unit, 8 * cost_unit, false},
  };
  for (size_t i = 0, iter = 1; i < dr_parameter_list.size(); i++, iter++) {
    Monitor iter_monitor;
    LOG_INST.info(Loc::current(), "***** Begin iteration ", iter, "/", dr_parameter_list.size(), "(",
                  RTUtil::getPercentage(iter, dr_parameter_list.size()), ") *****");
    setDRParameter(dr_model, iter, dr_parameter_list[i]);
    initDRBoxMap(dr_model);
    splitNetResult(dr_model);
    buildBoxSchedule(dr_model);
    routeDRBoxMap(dr_model);
    updateSummary(dr_model);
    printSummary(dr_model);
    writeNetCSV(dr_model);
    writeViolationCSV(dr_model);
    // debugOutputDef(dr_model);
    LOG_INST.info(Loc::current(), "***** End Iteration ", iter, "/", dr_parameter_list.size(), "(",
                  RTUtil::getPercentage(iter, dr_parameter_list.size()), ")", iter_monitor.getStatsInfo(), "*****");
  }
}

void DetailedRouter::setDRParameter(DRModel& dr_model, int32_t iter, DRParameter& dr_parameter)
{
  dr_model.set_iter(iter);
  LOG_INST.info(Loc::current(), "prefer_wire_unit : ", dr_parameter.get_prefer_wire_unit());
  LOG_INST.info(Loc::current(), "nonprefer_wire_unit : ", dr_parameter.get_nonprefer_wire_unit());
  LOG_INST.info(Loc::current(), "via_unit : ", dr_parameter.get_via_unit());
  LOG_INST.info(Loc::current(), "corner_unit : ", dr_parameter.get_corner_unit());
  LOG_INST.info(Loc::current(), "size : ", dr_parameter.get_size());
  LOG_INST.info(Loc::current(), "offset : ", dr_parameter.get_offset());
  LOG_INST.info(Loc::current(), "fixed_rect_unit : ", dr_parameter.get_fixed_rect_unit());
  LOG_INST.info(Loc::current(), "routed_rect_unit : ", dr_parameter.get_routed_rect_unit());
  LOG_INST.info(Loc::current(), "violation_unit : ", dr_parameter.get_violation_unit());
  LOG_INST.info(Loc::current(), "complete_ripup : ", dr_parameter.get_complete_ripup());
  dr_model.set_dr_parameter(dr_parameter);
}

void DetailedRouter::initDRBoxMap(DRModel& dr_model)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();

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
      int32_t grid_lb_x = std::max(offset + x * size, 0);
      int32_t grid_lb_y = std::max(offset + y * size, 0);
      int32_t grid_rt_x = std::min(offset + (x + 1) * size - 1, x_gcell_num - 1);
      int32_t grid_rt_y = std::min(offset + (y + 1) * size - 1, y_gcell_num - 1);

      PlanarRect lb_gcell_rect = RTUtil::getRealRectByGCell(PlanarCoord(grid_lb_x, grid_lb_y), gcell_axis);
      PlanarRect rt_gcell_rect = RTUtil::getRealRectByGCell(PlanarCoord(grid_rt_x, grid_rt_y), gcell_axis);
      PlanarRect box_real_rect(lb_gcell_rect.get_lb(), rt_gcell_rect.get_rt());

      DRBox& dr_box = dr_box_map[x][y];

      EXTPlanarRect dr_box_rect;
      dr_box_rect.set_real_rect(box_real_rect);
      dr_box_rect.set_grid_rect(RTUtil::getOpenGCellGridRect(box_real_rect, gcell_axis));
      dr_box.set_box_rect(dr_box_rect);
      DRBoxId dr_box_id;
      dr_box_id.set_x(x);
      dr_box_id.set_y(y);
      dr_box.set_dr_box_id(dr_box_id);
      dr_box.set_dr_parameter(&dr_parameter);
    }
  }
}

void DetailedRouter::splitNetResult(DRModel& dr_model)
{
  GridMap<DRBox>& dr_box_map = dr_model.get_dr_box_map();
  for (int32_t x = 0; x < dr_box_map.get_x_size(); x++) {
    for (int32_t y = 0; y < dr_box_map.get_y_size(); y++) {
      DRBox& dr_box = dr_box_map[x][y];

      PlanarRect& real_rect = dr_box.get_box_rect().get_real_rect();
      int32_t box_lb_x = real_rect.get_lb_x();
      int32_t box_lb_y = real_rect.get_lb_y();
      int32_t box_rt_x = real_rect.get_rt_x();
      int32_t box_rt_y = real_rect.get_rt_y();

      for (auto& [net_idx, segment_set] : DM_INST.getNetResultMap(dr_box.get_box_rect())) {
        for (Segment<LayerCoord>* segment : segment_set) {
          LayerCoord first = segment->get_first();
          int32_t first_x = first.get_x();
          int32_t first_y = first.get_y();
          int32_t first_layer_idx = first.get_layer_idx();

          LayerCoord second = segment->get_second();
          int32_t second_x = second.get_x();
          int32_t second_y = second.get_y();
          int32_t second_layer_idx = second.get_layer_idx();
          RTUtil::swapByASC(first_x, second_x);
          RTUtil::swapByASC(first_y, second_y);

          if (first_layer_idx != second_layer_idx) {
            continue;
          }
          if (!CmpLayerCoordByLayerASC()(first, second)) {
            std::swap(first, second);
          }
          std::vector<LayerCoord> end_point_list = {first, second};
          if (RTUtil::isHorizontal(first, second)) {
            if (first_x <= box_lb_x && box_lb_x <= second_x) {
              end_point_list.emplace_back(box_lb_x, first_y, first_layer_idx);
            }
            if (first_x <= box_rt_x && box_rt_x <= second_x) {
              end_point_list.emplace_back(box_rt_x, first_y, first_layer_idx);
            }
          } else if (RTUtil::isVertical(first, second)) {
            if (first_y <= box_lb_y && box_lb_y <= second_y) {
              end_point_list.emplace_back(first_x, box_lb_y, first_layer_idx);
            }
            if (first_y <= box_rt_y && box_rt_y <= second_y) {
              end_point_list.emplace_back(first_x, box_rt_y, first_layer_idx);
            }
          } else {
            LOG_INST.error(Loc::current(), "Routing Segmet is oblique!");
          }
          std::sort(end_point_list.begin(), end_point_list.end(), CmpLayerCoordByLayerASC());
          end_point_list.erase(std::unique(end_point_list.begin(), end_point_list.end()), end_point_list.end());
          if (end_point_list.size() == 2) {
            continue;
          }

          // update split result
          DM_INST.updateNetResultToGCellMap(ChangeType::kDel, net_idx, segment);
          for (size_t i = 1; i < end_point_list.size(); i++) {
            DM_INST.updateNetResultToGCellMap(ChangeType::kAdd, net_idx, new Segment<LayerCoord>(end_point_list[i - 1], end_point_list[i]));
          }
        }
      }
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
      initDRTaskList(dr_model, dr_box);
      buildViolationList(dr_box);
      if (needRouting(dr_box)) {
        buildFixedRectList(dr_box);
        buildBoxTrackAxis(dr_box);
        initLayerNodeMap(dr_box);
        initDRNodeValid(dr_box);
        buildDRNodeNeighbor(dr_box);
        buildOrienNetMap(dr_box);
        // debugCheckDRBox(dr_box);
        routeDRBox(dr_box);
      }
      updateDRTaskToGcellMap(dr_box);
      updateViolationToGcellMap(dr_box);
      // debugPlotDRBox(dr_box, -1, "routed");
      freeDRBox(dr_box);
    }
    routed_box_num += dr_box_id_list.size();

    LOG_INST.info(Loc::current(), "Routed ", routed_box_num, "/", total_box_num, "(", RTUtil::getPercentage(routed_box_num, total_box_num),
                  ") boxes with ", getViolationNum(), " violations", stage_monitor.getStatsInfo());
  }
}

void DetailedRouter::initDRTaskList(DRModel& dr_model, DRBox& dr_box)
{
  std::vector<DRNet>& dr_net_list = dr_model.get_dr_net_list();
  std::vector<DRTask*>& dr_task_list = dr_box.get_dr_task_list();

  EXTPlanarRect& box_rect = dr_box.get_box_rect();
  PlanarRect& real_rect = box_rect.get_real_rect();
  std::map<int32_t, std::set<AccessPoint*>> net_access_point_map = DM_INST.getNetAccessPointMap(box_rect);
  std::map<int32_t, std::set<Segment<LayerCoord>*>> net_result_map = DM_INST.getNetResultMap(box_rect);
  std::map<int32_t, std::set<EXTLayerRect*>> net_patch_map = DM_INST.getNetPatchMap(box_rect);

  std::map<int32_t, std::set<LayerCoord, CmpLayerCoordByLayerASC>> net_connect_point_map;
  {
    for (auto& [net_idx, access_point_set] : net_access_point_map) {
      for (AccessPoint* access_point : access_point_set) {
        net_connect_point_map[net_idx].insert(LayerCoord(access_point->get_real_coord(), access_point->get_layer_idx()));
      }
    }
    for (auto& [net_idx, segment_set] : net_result_map) {
      for (Segment<LayerCoord>* segment : segment_set) {
        LayerCoord& first = segment->get_first();
        LayerCoord& second = segment->get_second();
        if (first.get_layer_idx() != second.get_layer_idx()) {
          continue;
        }
        if (RTUtil::isHorizontal(first, second)) {
          int32_t first_x = first.get_x();
          int32_t second_x = second.get_x();
          RTUtil::swapByASC(first_x, second_x);
          if (first_x <= real_rect.get_lb_x() && real_rect.get_lb_x() <= second_x) {
            net_connect_point_map[net_idx].insert(LayerCoord(real_rect.get_lb_x(), first.get_y(), first.get_layer_idx()));
          }
          if (first_x <= real_rect.get_rt_x() && real_rect.get_rt_x() <= second_x) {
            net_connect_point_map[net_idx].insert(LayerCoord(real_rect.get_rt_x(), first.get_y(), first.get_layer_idx()));
          }
        } else if (RTUtil::isVertical(first, second)) {
          int32_t first_y = first.get_y();
          int32_t second_y = second.get_y();
          RTUtil::swapByASC(first_y, second_y);
          if (first_y <= real_rect.get_lb_y() && real_rect.get_lb_y() <= second_y) {
            net_connect_point_map[net_idx].insert(LayerCoord(first.get_x(), real_rect.get_lb_y(), first.get_layer_idx()));
          }
          if (first_y <= real_rect.get_rt_y() && real_rect.get_rt_y() <= second_y) {
            net_connect_point_map[net_idx].insert(LayerCoord(first.get_x(), real_rect.get_rt_y(), first.get_layer_idx()));
          }
        } else {
          LOG_INST.error(Loc::current(), "The segment is oblique!");
        }
      }
    }
  }
  for (auto [net_idx, connect_point_set] : net_connect_point_map) {
    if (connect_point_set.size() < 2) {
      continue;
    }
    std::vector<DRGroup> dr_group_list;
    for (const LayerCoord& connect_point : connect_point_set) {
      DRGroup dr_group;
      dr_group.get_coord_direction_map()[connect_point].insert({});
      dr_group_list.push_back(dr_group);
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
      dr_task->set_bounding_box(RTUtil::getBoundingBox(coord_list));
    }
    dr_task->set_routed_times(0);
    {
      for (Segment<LayerCoord>* segment : net_result_map[net_idx]) {
        if (RTUtil::isInside(real_rect, Segment<PlanarCoord>(segment->get_first(), segment->get_second()))) {
          dr_task->get_routing_segment_list().push_back(*segment);
          DM_INST.updateNetResultToGCellMap(ChangeType::kDel, net_idx, segment);
        }
      }
    }
    {
      for (EXTLayerRect* patch : net_patch_map[net_idx]) {
        if (RTUtil::isInside(real_rect, patch->get_real_rect())) {
          dr_task->get_patch_list().push_back(*patch);
          DM_INST.updatePatchToGCellMap(ChangeType::kDel, net_idx, patch);
        }
      }
    }
    dr_task_list.push_back(dr_task);
  }
  std::sort(dr_task_list.begin(), dr_task_list.end(), CmpDRTask());
}

void DetailedRouter::buildViolationList(DRBox& dr_box)
{
  for (Violation* violation : DM_INST.getViolationSet(dr_box.get_box_rect())) {
    if (RTUtil::isInside(dr_box.get_box_rect().get_real_rect(), violation->get_violation_shape().get_real_rect())) {
      dr_box.get_violation_list().push_back(*violation);
      DM_INST.updateViolationToGCellMap(ChangeType::kDel, violation);
    }
  }
}

bool DetailedRouter::needRouting(DRBox& dr_box)
{
  if (dr_box.get_dr_task_list().empty()) {
    return false;
  }
  if (dr_box.get_dr_parameter()->get_complete_ripup() == false && dr_box.get_violation_list().empty()) {
    return false;
  }
  return true;
}

void DetailedRouter::buildFixedRectList(DRBox& dr_box)
{
  dr_box.set_type_layer_net_fixed_rect_map(DM_INST.getTypeLayerNetFixedRectMap(dr_box.get_box_rect()));
}

void DetailedRouter::buildBoxTrackAxis(DRBox& dr_box)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  std::vector<int32_t> x_scale_list;
  std::vector<int32_t> y_scale_list;

  PlanarRect& box_region = dr_box.get_box_rect().get_real_rect();
  for (RoutingLayer& routing_layer : routing_layer_list) {
    for (int32_t x_scale : RTUtil::getScaleList(box_region.get_lb_x(), box_region.get_rt_x(), routing_layer.getXTrackGridList())) {
      x_scale_list.push_back(x_scale);
    }
    for (int32_t y_scale : RTUtil::getScaleList(box_region.get_lb_y(), box_region.get_rt_y(), routing_layer.getYTrackGridList())) {
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
  box_track_axis.set_x_grid_list(RTUtil::makeScaleGridList(x_scale_list));
  std::sort(y_scale_list.begin(), y_scale_list.end());
  y_scale_list.erase(std::unique(y_scale_list.begin(), y_scale_list.end()), y_scale_list.end());
  box_track_axis.set_y_grid_list(RTUtil::makeScaleGridList(y_scale_list));
}

void DetailedRouter::initLayerNodeMap(DRBox& dr_box)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  PlanarCoord& real_lb = dr_box.get_box_rect().get_real_lb();
  PlanarCoord& real_rt = dr_box.get_box_rect().get_real_rt();
  ScaleAxis& box_track_axis = dr_box.get_box_track_axis();
  std::vector<int32_t> x_list = RTUtil::getScaleList(real_lb.get_x(), real_rt.get_x(), box_track_axis.get_x_grid_list());
  std::vector<int32_t> y_list = RTUtil::getScaleList(real_lb.get_y(), real_rt.get_y(), box_track_axis.get_y_grid_list());

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

void DetailedRouter::initDRNodeValid(DRBox& dr_box)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  int32_t bottom_routing_layer_idx = DM_INST.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = DM_INST.getConfig().top_routing_layer_idx;

  std::vector<GridMap<DRNode>>& layer_node_map = dr_box.get_layer_node_map();
  ScaleAxis& box_track_axis = dr_box.get_box_track_axis();

  std::map<int32_t, std::set<int32_t>> layer_x_scale_map;
  std::map<int32_t, std::set<int32_t>> layer_y_scale_map;
  // point映射
  for (DRTask* dr_task : dr_box.get_dr_task_list()) {
    for (DRGroup& dr_group : dr_task->get_dr_group_list()) {
      for (auto& [real_coord, direction_set] : dr_group.get_coord_direction_map()) {
        std::vector<int32_t> point_layer_idx_list
            = RTUtil::getReservedViaBelowLayerIdxList(real_coord.get_layer_idx(), bottom_routing_layer_idx, top_routing_layer_idx);
        std::sort(point_layer_idx_list.begin(), point_layer_idx_list.end());
        point_layer_idx_list.push_back(point_layer_idx_list.back() + 1);
        for (int32_t point_layer_idx : point_layer_idx_list) {
          layer_x_scale_map[point_layer_idx].insert(real_coord.get_x());
          layer_y_scale_map[point_layer_idx].insert(real_coord.get_y());
        }
      }
    }
  }
  // 本层track上的node设置点合法状态
  int32_t box_lb_x = dr_box.get_box_rect().get_real_lb_x();
  int32_t box_lb_y = dr_box.get_box_rect().get_real_lb_y();
  int32_t box_rt_x = dr_box.get_box_rect().get_real_rt_x();
  int32_t box_rt_y = dr_box.get_box_rect().get_real_rt_y();
  for (RoutingLayer& curr_routing_layer : routing_layer_list) {
    int32_t curr_layer_idx = curr_routing_layer.get_layer_idx();
    if (curr_layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < curr_layer_idx) {
      continue;
    }
    for (int32_t x_scale : RTUtil::getScaleList(box_lb_x, box_rt_x, curr_routing_layer.getXTrackGridList())) {
      layer_x_scale_map[curr_layer_idx].insert(x_scale);
    }
    for (int32_t y_scale : RTUtil::getScaleList(box_lb_y, box_rt_y, curr_routing_layer.getYTrackGridList())) {
      layer_y_scale_map[curr_layer_idx].insert(y_scale);
    }
    int32_t below_layer_idx = curr_layer_idx - 1;
    if (bottom_routing_layer_idx <= below_layer_idx && below_layer_idx <= top_routing_layer_idx) {
      RoutingLayer& below_routing_layer = routing_layer_list[below_layer_idx];
      if (below_routing_layer.isPreferH()) {
        for (int32_t y_scale : RTUtil::getScaleList(box_lb_y, box_rt_y, below_routing_layer.getYTrackGridList())) {
          layer_y_scale_map[curr_layer_idx].insert(y_scale);
        }
      } else {
        for (int32_t x_scale : RTUtil::getScaleList(box_lb_x, box_rt_x, below_routing_layer.getXTrackGridList())) {
          layer_x_scale_map[curr_layer_idx].insert(x_scale);
        }
      }
    }
    int32_t above_layer_idx = curr_layer_idx + 1;
    if (bottom_routing_layer_idx <= above_layer_idx && above_layer_idx <= top_routing_layer_idx) {
      RoutingLayer& above_routing_layer = routing_layer_list[above_layer_idx];
      if (above_routing_layer.isPreferH()) {
        for (int32_t y_scale : RTUtil::getScaleList(box_lb_y, box_rt_y, above_routing_layer.getYTrackGridList())) {
          layer_y_scale_map[curr_layer_idx].insert(y_scale);
        }
      } else {
        for (int32_t x_scale : RTUtil::getScaleList(box_lb_x, box_rt_x, above_routing_layer.getXTrackGridList())) {
          layer_x_scale_map[curr_layer_idx].insert(x_scale);
        }
      }
    }
  }

  for (RoutingLayer& routing_layer : routing_layer_list) {
    int32_t layer_idx = routing_layer.get_layer_idx();
    if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
      continue;
    }
    for (int32_t x_scale : layer_x_scale_map[layer_idx]) {
      for (int32_t y_scale : layer_y_scale_map[layer_idx]) {
        PlanarCoord real_coord(x_scale, y_scale);
        if (!RTUtil::existTrackGrid(real_coord, dr_box.get_box_track_axis())) {
          LOG_INST.error(Loc::current(), "There is no grid coord for real coord(", x_scale, ",", y_scale, ")!");
        }
        PlanarCoord grid_coord = RTUtil::getTrackGrid(real_coord, box_track_axis);
        layer_node_map[layer_idx][grid_coord.get_x()][grid_coord.get_y()].set_is_valid(true);
      }
    }
  }
}

void DetailedRouter::buildDRNodeNeighbor(DRBox& dr_box)
{
  std::vector<GridMap<DRNode>>& layer_node_map = dr_box.get_layer_node_map();

  for (int32_t curr_layer_idx = 0; curr_layer_idx < static_cast<int32_t>(layer_node_map.size()); curr_layer_idx++) {
    for (int32_t curr_x = 0; curr_x < layer_node_map[curr_layer_idx].get_x_size(); curr_x++) {
      for (int32_t curr_y = 0; curr_y < layer_node_map[curr_layer_idx].get_y_size(); curr_y++) {
        DRNode& curr_node = layer_node_map[curr_layer_idx][curr_x][curr_y];
        if (!curr_node.get_is_valid()) {
          continue;
        }
        // 向东寻找，找到第一个有效结点即为东邻居，并将东邻居的西邻居设为自己
        for (int32_t east_x = curr_x + 1; east_x < layer_node_map[curr_layer_idx].get_x_size(); east_x++) {
          DRNode& east_node = layer_node_map[curr_layer_idx][east_x][curr_y];
          if (!east_node.get_is_valid()) {
            continue;
          }
          curr_node.get_neighbor_node_map()[Orientation::kEast] = &east_node;
          east_node.get_neighbor_node_map()[Orientation::kWest] = &curr_node;
          break;
        }
        // 向北寻找，找到第一个有效结点即为北邻居，并将北邻居的南邻居设为自己
        for (int32_t north_y = curr_y + 1; north_y < layer_node_map[curr_layer_idx].get_y_size(); north_y++) {
          DRNode& north_node = layer_node_map[curr_layer_idx][curr_x][north_y];
          if (!north_node.get_is_valid()) {
            continue;
          }
          curr_node.get_neighbor_node_map()[Orientation::kNorth] = &north_node;
          north_node.get_neighbor_node_map()[Orientation::kSouth] = &curr_node;
          break;
        }
        // 向上寻找，找到第一个有效结点即为上邻居，并将上邻居的下邻居设为自己
        for (int32_t above_layer_idx = curr_layer_idx + 1; above_layer_idx < static_cast<int32_t>(layer_node_map.size());
             above_layer_idx++) {
          DRNode& above_node = layer_node_map[above_layer_idx][curr_x][curr_y];
          if (!above_node.get_is_valid()) {
            continue;
          }
          curr_node.get_neighbor_node_map()[Orientation::kAbove] = &above_node;
          above_node.get_neighbor_node_map()[Orientation::kBelow] = &curr_node;
          break;
        }
      }
    }
  }
}

void DetailedRouter::buildOrienNetMap(DRBox& dr_box)
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
  for (DRTask* dr_task : dr_box.get_dr_task_list()) {
    for (Segment<LayerCoord>& routing_segment : dr_task->get_routing_segment_list()) {
      updateNetResultToGraph(dr_box, ChangeType::kAdd, dr_task->get_net_idx(), routing_segment);
    }
    for (EXTLayerRect& patch : dr_task->get_patch_list()) {
      updatePatchToGraph(dr_box, ChangeType::kAdd, dr_task->get_net_idx(), patch);
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
      applyPatch(dr_box, dr_task);
      dr_task->addRoutedTimes();
    }
    updateViolationList(dr_box);
    dr_task_list = getTaskScheduleByViolation(dr_box);
  }
}

std::vector<DRTask*> DetailedRouter::initTaskSchedule(DRBox& dr_box)
{
  bool complete_ripup = dr_box.get_dr_parameter()->get_complete_ripup();

  std::vector<DRTask*> dr_task_list;
  if (complete_ripup) {
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
  std::set<int32_t> violation_net_set;
  for (Violation& violation : dr_box.get_violation_list()) {
    for (int32_t violation_net : violation.get_violation_net_set()) {
      violation_net_set.insert(violation_net);
    }
  }
  std::vector<DRTask*> dr_task_list;
  for (DRTask* dr_task : dr_box.get_dr_task_list()) {
    if (!RTUtil::exist(violation_net_set, dr_task->get_net_idx())) {
      continue;
    }
    if (dr_task->get_routed_times() > 1) {
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
      std::vector<DRNode*> node_comb;
      for (auto& [coord, direction_set] : dr_group.get_coord_direction_map()) {
        if (!RTUtil::existTrackGrid(coord, box_track_axis)) {
          LOG_INST.error(Loc::current(), "The coord can not find grid!");
        }
        PlanarCoord grid_coord = RTUtil::getTrackGrid(coord, box_track_axis);
        DRNode& dr_node = layer_node_map[coord.get_layer_idx()][grid_coord.get_x()][grid_coord.get_y()];
        dr_node.set_direction_set(direction_set);
        node_comb.push_back(&dr_node);
      }
      node_list_list.push_back(node_comb);
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

  for (std::vector<DRNode*>& start_node_comb : start_node_list_list) {
    for (DRNode* start_node : start_node_comb) {
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
    dr_box.set_end_node_comb_idx(-1);
    return true;
  }
  for (size_t i = 0; i < end_node_list_list.size(); i++) {
    for (DRNode* end_node : end_node_list_list[i]) {
      if (path_head_node == end_node) {
        dr_box.set_end_node_comb_idx(static_cast<int32_t>(i));
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
  return dr_box.get_end_node_comb_idx() == -1;
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
  dr_box.set_end_node_comb_idx(-1);
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

void DetailedRouter::updateDirectionSet(DRBox& dr_box)
{
  DRNode* path_head_node = dr_box.get_path_head_node();

  DRNode* curr_node = path_head_node;
  DRNode* pre_node = curr_node->get_parent_node();
  while (pre_node != nullptr) {
    curr_node->get_direction_set().insert(RTUtil::getDirection(*curr_node, *pre_node));
    pre_node->get_direction_set().insert(RTUtil::getDirection(*pre_node, *curr_node));
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
  int32_t end_node_comb_idx = dr_box.get_end_node_comb_idx();

  // 对于抵达的终点pin，只保留到达的node
  end_node_list_list[end_node_comb_idx].clear();
  end_node_list_list[end_node_comb_idx].push_back(path_head_node);

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
  start_node_list_list.push_back(end_node_list_list[end_node_comb_idx]);
  end_node_list_list.erase(end_node_list_list.begin() + end_node_comb_idx);
}

void DetailedRouter::updateTaskResult(DRBox& dr_box)
{
  std::vector<Segment<LayerCoord>> new_routing_segment_list = getRoutingSegmentList(dr_box);

  DRTask* curr_dr_task = dr_box.get_curr_dr_task();
  // 原结果从graph删除
  for (Segment<LayerCoord>& routing_segment : curr_dr_task->get_routing_segment_list()) {
    updateNetResultToGraph(dr_box, ChangeType::kDel, curr_dr_task->get_net_idx(), routing_segment);
  }
  curr_dr_task->set_routing_segment_list(new_routing_segment_list);
  // 新结果添加到graph
  for (Segment<LayerCoord>& routing_segment : curr_dr_task->get_routing_segment_list()) {
    updateNetResultToGraph(dr_box, ChangeType::kAdd, curr_dr_task->get_net_idx(), routing_segment);
  }
}

std::vector<Segment<LayerCoord>> DetailedRouter::getRoutingSegmentList(DRBox& dr_box)
{
  DRTask* curr_dr_task = dr_box.get_curr_dr_task();

  std::vector<LayerCoord> driving_grid_coord_list;
  std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
  std::vector<DRGroup>& dr_group_list = curr_dr_task->get_dr_group_list();
  for (size_t i = 0; i < dr_group_list.size(); i++) {
    for (auto& [coord, direction_set] : dr_group_list[i].get_coord_direction_map()) {
      driving_grid_coord_list.push_back(coord);
      key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
    }
  }
  // 构建 优化 检查 routing_segment_list
  MTree<LayerCoord> coord_tree = RTUtil::getTreeByFullFlow(driving_grid_coord_list, dr_box.get_routing_segment_list(), key_coord_pin_map);

  std::vector<Segment<LayerCoord>> routing_segment_list;
  for (Segment<TNode<LayerCoord>*>& segment : RTUtil::getSegListByTree(coord_tree)) {
    routing_segment_list.emplace_back(segment.get_first()->value(), segment.get_second()->value());
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
    LOG_INST.error(Loc::current(), "The neighbor not exist!");
  }

  double cost = 0;
  cost += start_node->get_known_cost();
  cost += getNodeCost(dr_box, start_node, RTUtil::getOrientation(*start_node, *end_node));
  cost += getNodeCost(dr_box, end_node, RTUtil::getOrientation(*end_node, *start_node));
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
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  double prefer_wire_unit = dr_box.get_dr_parameter()->get_prefer_wire_unit();
  double nonprefer_wire_unit = dr_box.get_dr_parameter()->get_nonprefer_wire_unit();

  double wire_cost = 0;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    wire_cost += RTUtil::getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());

    RoutingLayer& routing_layer = routing_layer_list[start_node->get_layer_idx()];
    if (routing_layer.get_prefer_direction() == RTUtil::getDirection(*start_node, *end_node)) {
      wire_cost *= prefer_wire_unit;
    } else {
      wire_cost *= nonprefer_wire_unit;
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
  for (std::vector<DRNode*>& end_node_comb : end_node_list_list) {
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
  estimate_cost += getEstimateWireCost(dr_box, start_node, end_node);
  estimate_cost += getEstimateCornerCost(dr_box, start_node, end_node);
  estimate_cost += getEstimateViaCost(dr_box, start_node, end_node);
  return estimate_cost;
}

double DetailedRouter::getEstimateWireCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
{
  double prefer_wire_unit = dr_box.get_dr_parameter()->get_prefer_wire_unit();

  double wire_cost = 0;
  wire_cost += RTUtil::getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());
  wire_cost *= prefer_wire_unit;
  return wire_cost;
}

double DetailedRouter::getEstimateCornerCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
{
  double corner_unit = dr_box.get_dr_parameter()->get_corner_unit();

  double corner_cost = 0;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    if (RTUtil::isOblique(*start_node, *end_node)) {
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

void DetailedRouter::applyPatch(DRBox& dr_box, DRTask* dr_task)
{
  std::vector<EXTLayerRect> new_patch_list = getPatchList(dr_box, dr_task);

  // 原结果从graph删除
  for (EXTLayerRect& patch : dr_task->get_patch_list()) {
    updatePatchToGraph(dr_box, ChangeType::kDel, dr_task->get_net_idx(), patch);
  }
  dr_task->set_patch_list(new_patch_list);
  // 新结果添加到graph
  for (EXTLayerRect& patch : dr_task->get_patch_list()) {
    updatePatchToGraph(dr_box, ChangeType::kAdd, dr_task->get_net_idx(), patch);
  }
}

std::vector<EXTLayerRect> DetailedRouter::getPatchList(DRBox& dr_box, DRTask* dr_task)
{
  std::vector<EXTLayerRect> patch_list;
  return patch_list;
}

void DetailedRouter::updateViolationList(DRBox& dr_box)
{
  std::vector<Violation> new_violation_list = getViolationList(dr_box);

  // 原结果从graph删除
  for (Violation& violation : dr_box.get_violation_list()) {
    updateViolationToGraph(dr_box, ChangeType::kDel, violation);
  }
  dr_box.set_violation_list(new_violation_list);
  // 新结果添加到graph
  for (Violation& violation : dr_box.get_violation_list()) {
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
            env_shape_list.push_back(DM_INST.getIDBLayerShapeByFixedRect(fixed_rect, is_routing));
          }
        } else {
          for (auto& fixed_rect : fixed_rect_set) {
            net_pin_shape_map[net_idx].push_back(DM_INST.getIDBLayerShapeByFixedRect(fixed_rect, is_routing));
          }
        }
      }
    }
  }
  std::map<int32_t, std::vector<idb::IdbRegularWireSegment*>> net_wire_via_map;
  for (DRTask* dr_task : dr_box.get_dr_task_list()) {
    for (Segment<LayerCoord>& routing_segment : dr_task->get_routing_segment_list()) {
      net_wire_via_map[dr_task->get_net_idx()].push_back(DM_INST.getIDBSegmentByNetResult(dr_task->get_net_idx(), routing_segment));
    }
    for (EXTLayerRect& patch : dr_task->get_patch_list()) {
      net_wire_via_map[dr_task->get_net_idx()].push_back(DM_INST.getIDBSegmentByNetPatch(dr_task->get_net_idx(), patch));
    }
  }
  std::vector<Violation> violation_list = RTAPI_INST.getViolationList(env_shape_list, net_pin_shape_map, net_wire_via_map);
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

void DetailedRouter::updateDRTaskToGcellMap(DRBox& dr_box)
{
  for (DRTask* dr_task : dr_box.get_dr_task_list()) {
    for (Segment<LayerCoord>& routing_segment : dr_task->get_routing_segment_list()) {
      DM_INST.updateNetResultToGCellMap(ChangeType::kAdd, dr_task->get_net_idx(), new Segment<LayerCoord>(routing_segment));
    }
    for (EXTLayerRect& patch : dr_task->get_patch_list()) {
      DM_INST.updatePatchToGCellMap(ChangeType::kAdd, dr_task->get_net_idx(), new EXTLayerRect(patch));
    }
  }
}

void DetailedRouter::updateViolationToGcellMap(DRBox& dr_box)
{
  for (Violation& violation : dr_box.get_violation_list()) {
    DM_INST.updateViolationToGCellMap(ChangeType::kAdd, new Violation(violation));
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
  Die& die = DM_INST.getDatabase().get_die();

  return static_cast<int32_t>(DM_INST.getViolationSet(die).size());
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
        dr_node->get_orien_fixed_rect_map()[orientation].insert(net_shape.get_net_idx());
      } else if (change_type == ChangeType::kDel) {
        dr_node->get_orien_fixed_rect_map()[orientation].erase(net_shape.get_net_idx());
      }
    }
  }
}

void DetailedRouter::updateNetResultToGraph(DRBox& dr_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment)
{
  for (NetShape& net_shape : DM_INST.getNetShapeList(net_idx, segment)) {
    for (auto& [dr_node, orientation_set] : getNodeOrientationMap(dr_box, net_shape)) {
      if (!dr_node->get_is_valid()) {
        continue;
      }
      for (Orientation orientation : orientation_set) {
        if (change_type == ChangeType::kAdd) {
          dr_node->get_orien_routed_rect_map()[orientation].insert(net_shape.get_net_idx());
        } else if (change_type == ChangeType::kDel) {
          dr_node->get_orien_routed_rect_map()[orientation].erase(net_shape.get_net_idx());
        }
      }
    }
  }
}

void DetailedRouter::updatePatchToGraph(DRBox& dr_box, ChangeType change_type, int32_t net_idx, EXTLayerRect& patch)
{
  NetShape net_shape(net_idx, patch.getRealLayerRect(), true);
  for (auto& [dr_node, orientation_set] : getNodeOrientationMap(dr_box, net_shape)) {
    if (!dr_node->get_is_valid()) {
      continue;
    }
    for (Orientation orientation : orientation_set) {
      if (change_type == ChangeType::kAdd) {
        dr_node->get_orien_routed_rect_map()[orientation].insert(net_shape.get_net_idx());
      } else if (change_type == ChangeType::kDel) {
        dr_node->get_orien_routed_rect_map()[orientation].erase(net_shape.get_net_idx());
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
        dr_node->get_orien_violation_number_map()[orientation]++;
      } else if (change_type == ChangeType::kDel) {
        dr_node->get_orien_violation_number_map()[orientation]--;
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
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  if (!net_shape.get_is_routing()) {
    LOG_INST.error(Loc::current(), "The type of net_shape is cut!");
  }
  // 膨胀size为 min_spacing + half_width
  RoutingLayer& routing_layer = routing_layer_list[net_shape.get_layer_idx()];
  int32_t enlarged_size = routing_layer.getMinSpacing(net_shape.get_rect()) + (routing_layer.get_min_width() / 2);
  PlanarRect enlarged_rect = RTUtil::getEnlargedRect(net_shape.get_rect(), enlarged_size);

  GridMap<DRNode>& dr_node_map = dr_box.get_layer_node_map()[net_shape.get_layer_idx()];

  std::map<DRNode*, std::set<Orientation>> node_orientation_map;
  if (RTUtil::existTrackGrid(enlarged_rect, dr_box.get_box_track_axis())) {
    PlanarRect grid_rect = RTUtil::getTrackGridRect(enlarged_rect, dr_box.get_box_track_axis());
    for (int32_t grid_x = grid_rect.get_lb_x(); grid_x <= grid_rect.get_rt_x(); grid_x++) {
      for (int32_t grid_y = grid_rect.get_lb_y(); grid_y <= grid_rect.get_rt_y(); grid_y++) {
        DRNode& node = dr_node_map[grid_x][grid_y];
        for (auto& [orientation, neigbor_ptr] : node.get_neighbor_node_map()) {
          node_orientation_map[&node].insert(orientation);
          node_orientation_map[neigbor_ptr].insert(RTUtil::getOppositeOrientation(orientation));
        }
      }
    }
  }
  return node_orientation_map;
}

std::map<DRNode*, std::set<Orientation>> DetailedRouter::getCutNodeOrientationMap(DRBox& dr_box, NetShape& net_shape)
{
  std::vector<CutLayer>& cut_layer_list = DM_INST.getDatabase().get_cut_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = DM_INST.getDatabase().get_layer_via_master_list();
  Helper& helper = DM_INST.getHelper();
  if (net_shape.get_is_routing()) {
    LOG_INST.error(Loc::current(), "The type of net_shape is routing!");
  }
  std::vector<int32_t> adjacent_routing_layer_idx_list = helper.getAdjacentRoutingLayerIdxList(net_shape.get_layer_idx());
  if (adjacent_routing_layer_idx_list.size() != 2) {
    // 如果相邻层只有一个，那么将不会在此构建graph
    return {};
  }
  int32_t below_routing_layer_idx = adjacent_routing_layer_idx_list.front();
  int32_t above_routing_layer_idx = adjacent_routing_layer_idx_list.back();
  RTUtil::swapByASC(below_routing_layer_idx, above_routing_layer_idx);

  // 膨胀size为 min_spacing + 当前cut层的half_length和half_width
  int32_t cut_spacing = cut_layer_list[net_shape.get_layer_idx()].getMinSpacing(net_shape.get_rect());
  PlanarRect& cut_shape = layer_via_master_list[below_routing_layer_idx].front().get_cut_shape_list().front();
  int32_t enlarge_x_size = cut_spacing + cut_shape.getXSpan() / 2;
  int32_t enlarge_y_size = cut_spacing + cut_shape.getYSpan() / 2;
  PlanarRect enlarged_rect = RTUtil::getEnlargedRect(net_shape.get_rect(), enlarge_x_size, enlarge_y_size, enlarge_x_size, enlarge_y_size);

  std::vector<GridMap<DRNode>>& layer_node_map = dr_box.get_layer_node_map();

  std::map<DRNode*, std::set<Orientation>> node_orientation_map;
  if (RTUtil::existTrackGrid(enlarged_rect, dr_box.get_box_track_axis())) {
    PlanarRect grid_rect = RTUtil::getTrackGridRect(enlarged_rect, dr_box.get_box_track_axis());
    for (int32_t grid_x = grid_rect.get_lb_x(); grid_x <= grid_rect.get_rt_x(); grid_x++) {
      for (int32_t grid_y = grid_rect.get_lb_y(); grid_y <= grid_rect.get_rt_y(); grid_y++) {
        node_orientation_map[&layer_node_map[below_routing_layer_idx][grid_x][grid_y]].insert(Orientation::kAbove);
        node_orientation_map[&layer_node_map[above_routing_layer_idx][grid_x][grid_y]].insert(Orientation::kBelow);
      }
    }
  }
  return node_orientation_map;
}

#endif

#if 1  // debug

void DetailedRouter::debugCheckDRBox(DRBox& dr_box)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  DRBoxId& dr_box_id = dr_box.get_dr_box_id();
  if (dr_box_id.get_x() < 0 || dr_box_id.get_y() < 0) {
    LOG_INST.error(Loc::current(), "The grid coord is illegal!");
  }

  std::vector<GridMap<DRNode>>& layer_node_map = dr_box.get_layer_node_map();
  for (GridMap<DRNode>& dr_node_map : layer_node_map) {
    for (int32_t x = 0; x < dr_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < dr_node_map.get_y_size(); y++) {
        DRNode& dr_node = dr_node_map[x][y];
        if (!RTUtil::isInside(dr_box.get_box_rect().get_real_rect(), dr_node.get_planar_coord())) {
          LOG_INST.error(Loc::current(), "The dr_node is out of box!");
        }
        for (auto& [orien, neighbor] : dr_node.get_neighbor_node_map()) {
          Orientation opposite_orien = RTUtil::getOppositeOrientation(orien);
          if (!RTUtil::exist(neighbor->get_neighbor_node_map(), opposite_orien)) {
            LOG_INST.error(Loc::current(), "The dr_node neighbor is not bidirection!");
          }
          if (neighbor->get_neighbor_node_map()[opposite_orien] != &dr_node) {
            LOG_INST.error(Loc::current(), "The dr_node neighbor is not bidirection!");
          }
          LayerCoord node_coord(dr_node.get_planar_coord(), dr_node.get_layer_idx());
          LayerCoord neighbor_coord(neighbor->get_planar_coord(), neighbor->get_layer_idx());
          if (RTUtil::getOrientation(node_coord, neighbor_coord) == orien) {
            continue;
          }
          LOG_INST.error(Loc::current(), "The neighbor orien is different with real region!");
        }
      }
    }
  }

  for (DRTask* dr_task : dr_box.get_dr_task_list()) {
    if (dr_task->get_net_idx() < 0) {
      LOG_INST.error(Loc::current(), "The idx of origin net is illegal!");
    }
    for (DRGroup& dr_group : dr_task->get_dr_group_list()) {
      if (dr_group.get_coord_direction_map().empty()) {
        LOG_INST.error(Loc::current(), "The coord_direction_map is empty!");
      }
      for (auto& [coord, direction_set] : dr_group.get_coord_direction_map()) {
        int32_t layer_idx = coord.get_layer_idx();
        if (routing_layer_list.back().get_layer_idx() < layer_idx || layer_idx < routing_layer_list.front().get_layer_idx()) {
          LOG_INST.error(Loc::current(), "The layer idx of group coord is illegal!");
        }
        if (!RTUtil::existTrackGrid(coord, dr_box.get_box_track_axis())) {
          LOG_INST.error(Loc::current(), "There is no grid coord for real coord(", coord.get_x(), ",", coord.get_y(), ")!");
        }
        PlanarCoord grid_coord = RTUtil::getTrackGrid(coord, dr_box.get_box_track_axis());
        DRNode& dr_node = layer_node_map[layer_idx][grid_coord.get_x()][grid_coord.get_y()];
        if (dr_node.get_neighbor_node_map().empty()) {
          LOG_INST.error(Loc::current(), "The neighbor of group coord (", coord.get_x(), ",", coord.get_y(), ",", layer_idx,
                         ") is empty in box(", dr_box_id.get_x(), ",", dr_box_id.get_y(), ")");
        }
        if (RTUtil::isInside(dr_box.get_box_rect().get_real_rect(), coord)) {
          continue;
        }
        LOG_INST.error(Loc::current(), "The coord (", coord.get_x(), ",", coord.get_y(), ") is out of box!");
      }
    }
  }
}

void DetailedRouter::debugPlotDRBox(DRBox& dr_box, int32_t curr_task_idx, std::string flag)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = DM_INST.getDatabase().get_layer_via_master_list();
  std::string& dr_temp_directory_path = DM_INST.getConfig().dr_temp_directory_path;

  PlanarRect box_rect = dr_box.get_box_rect().get_real_rect();

  int32_t width = INT_MAX;
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
  for (int32_t x : RTUtil::getScaleList(box_rect.get_lb_x(), box_rect.get_rt_x(), gcell_axis.get_x_grid_list())) {
    GPPath gp_path;
    gp_path.set_layer_idx(0);
    gp_path.set_data_type(1);
    gp_path.set_segment(x, box_rect.get_lb_y(), x, box_rect.get_rt_y());
    gcell_axis_struct.push(gp_path);
  }
  for (int32_t y : RTUtil::getScaleList(box_rect.get_lb_y(), box_rect.get_rt_y(), gcell_axis.get_y_grid_list())) {
    GPPath gp_path;
    gp_path.set_layer_idx(0);
    gp_path.set_data_type(1);
    gp_path.set_segment(box_rect.get_lb_x(), y, box_rect.get_rt_x(), y);
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
        PlanarRect real_rect = RTUtil::getEnlargedRect(dr_node.get_planar_coord(), width);
        int32_t y_reduced_span = std::max(1, real_rect.getYSpan() / 12);
        int32_t y = real_rect.get_rt_y();

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
            LOG_INST.error(Loc::current(), "The type is error!");
            break;
        }
        gp_boundary.set_rect(real_rect);
        gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(dr_node.get_layer_idx()));
        dr_node_map_struct.push(gp_boundary);

        y -= y_reduced_span;
        GPText gp_text_node_real_coord;
        gp_text_node_real_coord.set_coord(real_rect.get_lb_x(), y);
        gp_text_node_real_coord.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
        gp_text_node_real_coord.set_message(
            RTUtil::getString("(", dr_node.get_x(), " , ", dr_node.get_y(), " , ", dr_node.get_layer_idx(), ")"));
        gp_text_node_real_coord.set_layer_idx(GP_INST.getGDSIdxByRouting(dr_node.get_layer_idx()));
        gp_text_node_real_coord.set_presentation(GPTextPresentation::kLeftMiddle);
        dr_node_map_struct.push(gp_text_node_real_coord);

        y -= y_reduced_span;
        GPText gp_text_node_grid_coord;
        gp_text_node_grid_coord.set_coord(real_rect.get_lb_x(), y);
        gp_text_node_grid_coord.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
        gp_text_node_grid_coord.set_message(RTUtil::getString("(", grid_x, " , ", grid_y, " , ", dr_node.get_layer_idx(), ")"));
        gp_text_node_grid_coord.set_layer_idx(GP_INST.getGDSIdxByRouting(dr_node.get_layer_idx()));
        gp_text_node_grid_coord.set_presentation(GPTextPresentation::kLeftMiddle);
        dr_node_map_struct.push(gp_text_node_grid_coord);

        y -= y_reduced_span;
        GPText gp_text_direction_set;
        gp_text_direction_set.set_coord(real_rect.get_lb_x(), y);
        gp_text_direction_set.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
        gp_text_direction_set.set_message("direction_set: ");
        gp_text_direction_set.set_layer_idx(GP_INST.getGDSIdxByRouting(dr_node.get_layer_idx()));
        gp_text_direction_set.set_presentation(GPTextPresentation::kLeftMiddle);
        dr_node_map_struct.push(gp_text_direction_set);

        if (!dr_node.get_direction_set().empty()) {
          y -= y_reduced_span;
          GPText gp_text_direction_set_info;
          gp_text_direction_set_info.set_coord(real_rect.get_lb_x(), y);
          gp_text_direction_set_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          std::string direction_set_info_message = "--";
          for (Direction direction : dr_node.get_direction_set()) {
            direction_set_info_message += RTUtil::getString("(", GetDirectionName()(direction), ")");
          }
          gp_text_direction_set_info.set_message(direction_set_info_message);
          gp_text_direction_set_info.set_layer_idx(GP_INST.getGDSIdxByRouting(dr_node.get_layer_idx()));
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
        PlanarRect real_rect = RTUtil::getEnlargedRect(dr_node.get_planar_coord(), width);

        int32_t lb_x = real_rect.get_lb_x();
        int32_t lb_y = real_rect.get_lb_y();
        int32_t rt_x = real_rect.get_rt_x();
        int32_t rt_y = real_rect.get_rt_y();
        int32_t mid_x = (lb_x + rt_x) / 2;
        int32_t mid_y = (lb_y + rt_y) / 2;
        int32_t x_reduced_span = (rt_x - lb_x) / 4;
        int32_t y_reduced_span = (rt_y - lb_y) / 4;
        int32_t width = std::min(x_reduced_span, y_reduced_span) / 2;

        for (auto& [orientation, neighbor_node] : dr_node.get_neighbor_node_map()) {
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
            case Orientation::kAbove:
              gp_path.set_segment(rt_x - x_reduced_span, rt_y - y_reduced_span, rt_x, rt_y);
              break;
            case Orientation::kBelow:
              gp_path.set_segment(lb_x, lb_y, lb_x + x_reduced_span, lb_y + y_reduced_span);
              break;
            default:
              LOG_INST.error(Loc::current(), "The orientation is oblique!");
              break;
          }
          gp_path.set_layer_idx(GP_INST.getGDSIdxByRouting(dr_node.get_layer_idx()));
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
  PlanarCoord& real_lb = box_rect.get_lb();
  PlanarCoord& real_rt = box_rect.get_rt();
  ScaleAxis& box_track_axis = dr_box.get_box_track_axis();
  std::vector<int32_t> x_list = RTUtil::getScaleList(real_lb.get_x(), real_rt.get_x(), box_track_axis.get_x_grid_list());
  std::vector<int32_t> y_list = RTUtil::getScaleList(real_lb.get_y(), real_rt.get_y(), box_track_axis.get_y_grid_list());
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
#if 1
    RoutingLayer& routing_layer = routing_layer_list[layer_idx];
    x_list = RTUtil::getScaleList(real_lb.get_x(), real_rt.get_x(), routing_layer.getXTrackGridList());
    y_list = RTUtil::getScaleList(real_lb.get_y(), real_rt.get_y(), routing_layer.getYTrackGridList());
#endif
    for (int32_t x : x_list) {
      GPPath gp_path;
      gp_path.set_data_type(static_cast<int32_t>(GPDataType::kAxis));
      gp_path.set_segment(x, real_lb.get_y(), x, real_rt.get_y());
      gp_path.set_layer_idx(GP_INST.getGDSIdxByRouting(layer_idx));
      box_track_axis_struct.push(gp_path);
    }
    for (int32_t y : y_list) {
      GPPath gp_path;
      gp_path.set_data_type(static_cast<int32_t>(GPDataType::kAxis));
      gp_path.set_segment(real_lb.get_x(), y, real_rt.get_x(), y);
      gp_path.set_layer_idx(GP_INST.getGDSIdxByRouting(layer_idx));
      box_track_axis_struct.push(gp_path);
    }
  }
  gp_gds.addStruct(box_track_axis_struct);

  for (auto& [is_routing, layer_net_rect_map] : dr_box.get_type_layer_net_fixed_rect_map()) {
    for (auto& [layer_idx, net_rect_map] : layer_net_rect_map) {
      for (auto& [net_idx, rect_set] : net_rect_map) {
        GPStruct fixed_rect_struct(RTUtil::getString("fixed_rect(net_", net_idx, ")"));
        for (EXTLayerRect* rect : rect_set) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShape));
          gp_boundary.set_rect(rect->get_real_rect());
          if (is_routing) {
            gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(layer_idx));
          } else {
            gp_boundary.set_layer_idx(GP_INST.getGDSIdxByCut(layer_idx));
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
      gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(violation_shape.get_layer_idx()));
    } else {
      gp_boundary.set_layer_idx(GP_INST.getGDSIdxByCut(violation_shape.get_layer_idx()));
    }
    violation_struct.push(gp_boundary);
  }
  gp_gds.addStruct(violation_struct);

  // task
  for (DRTask* dr_task : dr_box.get_dr_task_list()) {
    GPStruct task_struct(RTUtil::getString("task(net_", dr_task->get_net_idx(), ")"));

    if (curr_task_idx == -1 || dr_task->get_net_idx() == curr_task_idx) {
      for (DRGroup& dr_group : dr_task->get_dr_group_list()) {
        for (auto& [coord, direction_set] : dr_group.get_coord_direction_map()) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kKey));
          gp_boundary.set_rect(RTUtil::getEnlargedRect(coord, width));
          gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(coord.get_layer_idx()));
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
    for (Segment<LayerCoord>& segment : dr_task->get_routing_segment_list()) {
      LayerCoord first_coord = segment.get_first();
      LayerCoord second_coord = segment.get_second();
      int32_t first_layer_idx = first_coord.get_layer_idx();
      int32_t second_layer_idx = second_coord.get_layer_idx();
      int32_t half_width = routing_layer_list[first_layer_idx].get_min_width() / 2;

      if (first_layer_idx == second_layer_idx) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPath));
        gp_boundary.set_rect(RTUtil::getEnlargedRect(first_coord, second_coord, half_width));
        gp_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(first_layer_idx));
        task_struct.push(gp_boundary);
      } else {
        RTUtil::swapByASC(first_layer_idx, second_layer_idx);
        for (int32_t layer_idx = first_layer_idx; layer_idx < second_layer_idx; layer_idx++) {
          ViaMaster& via_master = layer_via_master_list[layer_idx].front();

          LayerRect& above_enclosure = via_master.get_above_enclosure();
          LayerRect offset_above_enclosure(RTUtil::getOffsetRect(above_enclosure, first_coord), above_enclosure.get_layer_idx());
          GPBoundary above_boundary;
          above_boundary.set_rect(offset_above_enclosure);
          above_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(above_enclosure.get_layer_idx()));
          above_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPath));
          task_struct.push(above_boundary);

          LayerRect& below_enclosure = via_master.get_below_enclosure();
          LayerRect offset_below_enclosure(RTUtil::getOffsetRect(below_enclosure, first_coord), below_enclosure.get_layer_idx());
          GPBoundary below_boundary;
          below_boundary.set_rect(offset_below_enclosure);
          below_boundary.set_layer_idx(GP_INST.getGDSIdxByRouting(below_enclosure.get_layer_idx()));
          below_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPath));
          task_struct.push(below_boundary);

          for (PlanarRect& cut_shape : via_master.get_cut_shape_list()) {
            LayerRect offset_cut_shape(RTUtil::getOffsetRect(cut_shape, first_coord), via_master.get_cut_layer_idx());
            GPBoundary cut_boundary;
            cut_boundary.set_rect(offset_cut_shape);
            cut_boundary.set_layer_idx(GP_INST.getGDSIdxByCut(via_master.get_cut_layer_idx()));
            cut_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPath));
            task_struct.push(cut_boundary);
          }
        }
      }
    }
    gp_gds.addStruct(task_struct);
  }
  std::string gds_file_path = RTUtil::getString(dr_temp_directory_path, flag, "_dr_box_", dr_box.get_dr_box_id().get_x(), "_",
                                                dr_box.get_dr_box_id().get_y(), ".gds");
  GP_INST.plot(gp_gds, gds_file_path);
}

void DetailedRouter::debugOutputDef(DRModel& dr_model)
{
  std::string& dr_temp_directory_path = DM_INST.getConfig().dr_temp_directory_path;

  DM_INST.output();
  RTAPI_INST.outputDef(RTUtil::getString(dr_temp_directory_path, "dr.def.temp"));
}

#endif

#if 1  // exhibit

void DetailedRouter::updateSummary(DRModel& dr_model)
{
  int32_t micron_dbu = DM_INST.getDatabase().get_micron_dbu();
  Die& die = DM_INST.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DM_INST.getDatabase().get_cut_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = DM_INST.getDatabase().get_layer_via_master_list();
  std::map<int32_t, double>& routing_wire_length_map
      = DM_INST.getSummary().iter_dr_summary_map[dr_model.get_iter()].routing_wire_length_map;
  double& total_wire_length = DM_INST.getSummary().iter_dr_summary_map[dr_model.get_iter()].total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = DM_INST.getSummary().iter_dr_summary_map[dr_model.get_iter()].cut_via_num_map;
  int32_t& total_via_num = DM_INST.getSummary().iter_dr_summary_map[dr_model.get_iter()].total_via_num;
  std::map<int32_t, int32_t>& routing_patch_num_map = DM_INST.getSummary().iter_dr_summary_map[dr_model.get_iter()].routing_patch_num_map;
  int32_t& total_patch_num = DM_INST.getSummary().iter_dr_summary_map[dr_model.get_iter()].total_patch_num;
  std::map<int32_t, int32_t>& routing_violation_num_map
      = DM_INST.getSummary().iter_dr_summary_map[dr_model.get_iter()].routing_violation_num_map;
  int32_t& total_violation_num = DM_INST.getSummary().iter_dr_summary_map[dr_model.get_iter()].total_violation_num;

  for (RoutingLayer& routing_layer : routing_layer_list) {
    routing_wire_length_map[routing_layer.get_layer_idx()] = 0;
    routing_patch_num_map[routing_layer.get_layer_idx()] = 0;
    routing_violation_num_map[routing_layer.get_layer_idx()] = 0;
  }
  total_wire_length = 0;
  total_patch_num = 0;
  total_violation_num = 0;
  for (CutLayer& cut_layer : cut_layer_list) {
    cut_via_num_map[cut_layer.get_layer_idx()] = 0;
  }
  total_via_num = 0;

  for (auto& [net_idx, segment_set] : DM_INST.getNetResultMap(die)) {
    for (Segment<LayerCoord>* segment : segment_set) {
      LayerCoord& first_coord = segment->get_first();
      int32_t first_layer_idx = first_coord.get_layer_idx();
      LayerCoord& second_coord = segment->get_second();
      int32_t second_layer_idx = second_coord.get_layer_idx();

      if (first_layer_idx == second_layer_idx) {
        double wire_length = RTUtil::getManhattanDistance(first_coord, second_coord) / 1.0 / micron_dbu;
        routing_wire_length_map[first_layer_idx] += wire_length;
        total_wire_length += wire_length;
      } else {
        RTUtil::swapByASC(first_layer_idx, second_layer_idx);
        for (int32_t layer_idx = first_layer_idx; layer_idx < second_layer_idx; layer_idx++) {
          cut_via_num_map[layer_via_master_list[layer_idx].front().get_cut_layer_idx()]++;
          total_via_num++;
        }
      }
    }
  }
  for (auto& [net_idx, patch_set] : DM_INST.getNetPatchMap(die)) {
    for (EXTLayerRect* patch : patch_set) {
      routing_patch_num_map[patch->get_layer_idx()]++;
      total_patch_num++;
    }
  }
  for (Violation* violation : DM_INST.getViolationSet(die)) {
    routing_violation_num_map[violation->get_violation_shape().get_layer_idx()]++;
    total_violation_num++;
  }
#if 0
  std::map<std::string, std::vector<double>>& timing = DM_INST.getSummary().iter_dr_summary_map[dr_model.get_iter()].timing;
  std::map<int32_t, std::map<LayerCoord, std::vector<std::string>, CmpLayerCoordByXASC>> net_coord_real_pin_map;
  std::map<int32_t, std::vector<Segment<LayerCoord>>> net_routing_segment_map;
  for (DRNet& dr_net : dr_model.get_dr_net_list()) {
    for (DRPin& dr_pin : dr_net.get_dr_pin_list()) {
      for (AccessPoint& access_point : dr_pin.get_access_point_list()) {
        net_coord_real_pin_map[dr_net.get_net_idx()][access_point.getRealLayerCoord()].push_back(dr_pin.get_pin_name());
      }
    }
  }
  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      for (auto& [net_idx, segment_set] : gcell_map[x][y].get_net_result_map()) {
        for (Segment<LayerCoord>* segment : segment_set) {
          net_routing_segment_map[net_idx].emplace_back(segment->get_first(), segment->get_second());
        }
      }
    }
  }
  timing = RTAPI_INST.getTiming(net_coord_real_pin_map, net_routing_segment_map);
#endif
}

void DetailedRouter::printSummary(DRModel& dr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DM_INST.getDatabase().get_cut_layer_list();
  std::map<int32_t, double>& routing_wire_length_map
      = DM_INST.getSummary().iter_dr_summary_map[dr_model.get_iter()].routing_wire_length_map;
  double& total_wire_length = DM_INST.getSummary().iter_dr_summary_map[dr_model.get_iter()].total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = DM_INST.getSummary().iter_dr_summary_map[dr_model.get_iter()].cut_via_num_map;
  int32_t& total_via_num = DM_INST.getSummary().iter_dr_summary_map[dr_model.get_iter()].total_via_num;
  std::map<int32_t, int32_t>& routing_patch_num_map = DM_INST.getSummary().iter_dr_summary_map[dr_model.get_iter()].routing_patch_num_map;
  int32_t& total_patch_num = DM_INST.getSummary().iter_dr_summary_map[dr_model.get_iter()].total_patch_num;
  std::map<int32_t, int32_t>& routing_violation_num_map
      = DM_INST.getSummary().iter_dr_summary_map[dr_model.get_iter()].routing_violation_num_map;
  int32_t& total_violation_num = DM_INST.getSummary().iter_dr_summary_map[dr_model.get_iter()].total_violation_num;

  fort::char_table routing_wire_length_map_table;
  {
    routing_wire_length_map_table << fort::header << "routing_layer"
                                  << "wire_length"
                                  << "proportion" << fort::endr;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      routing_wire_length_map_table << routing_layer.get_layer_name() << routing_wire_length_map[routing_layer.get_layer_idx()]
                                    << RTUtil::getPercentage(routing_wire_length_map[routing_layer.get_layer_idx()], total_wire_length)
                                    << fort::endr;
    }
    routing_wire_length_map_table << fort::header << "Total" << total_wire_length
                                  << RTUtil::getPercentage(total_wire_length, total_wire_length) << fort::endr;
  }
  fort::char_table cut_via_num_map_table;
  {
    cut_via_num_map_table << fort::header << "cut_layer"
                          << "via_num"
                          << "proportion" << fort::endr;
    for (CutLayer& cut_layer : cut_layer_list) {
      cut_via_num_map_table << cut_layer.get_layer_name() << cut_via_num_map[cut_layer.get_layer_idx()]
                            << RTUtil::getPercentage(cut_via_num_map[cut_layer.get_layer_idx()], total_via_num) << fort::endr;
    }
    cut_via_num_map_table << fort::header << "Total" << total_via_num << RTUtil::getPercentage(total_via_num, total_via_num) << fort::endr;
  }
  fort::char_table routing_patch_num_map_table;
  {
    routing_patch_num_map_table << fort::header << "routing_layer"
                                << "patch_num"
                                << "proportion" << fort::endr;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      routing_patch_num_map_table << routing_layer.get_layer_name() << routing_patch_num_map[routing_layer.get_layer_idx()]
                                  << RTUtil::getPercentage(routing_patch_num_map[routing_layer.get_layer_idx()], total_patch_num)
                                  << fort::endr;
    }
    routing_patch_num_map_table << fort::header << "Total" << total_patch_num << RTUtil::getPercentage(total_patch_num, total_patch_num)
                                << fort::endr;
  }
  fort::char_table routing_violation_num_map_table;
  {
    routing_violation_num_map_table << fort::header << "routing_layer"
                                    << "violation_num"
                                    << "proportion" << fort::endr;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      routing_violation_num_map_table << routing_layer.get_layer_name() << routing_violation_num_map[routing_layer.get_layer_idx()]
                                      << RTUtil::getPercentage(routing_violation_num_map[routing_layer.get_layer_idx()],
                                                               total_violation_num)
                                      << fort::endr;
    }
    routing_violation_num_map_table << fort::header << "Total" << total_violation_num
                                    << RTUtil::getPercentage(total_violation_num, total_violation_num) << fort::endr;
  }
  RTUtil::printTableList(
      {routing_wire_length_map_table, cut_via_num_map_table, routing_patch_num_map_table, routing_violation_num_map_table});
}

void DetailedRouter::writeNetCSV(DRModel& dr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = DM_INST.getDatabase().get_gcell_map();
  std::string& dr_temp_directory_path = DM_INST.getConfig().dr_temp_directory_path;
  int32_t output_csv = DM_INST.getConfig().output_csv;
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
      for (auto& [net_idx, segment_set] : gcell_map[x][y].get_net_result_map()) {
        for (Segment<LayerCoord>* segment : segment_set) {
          int32_t first_layer_idx = segment->get_first().get_layer_idx();
          int32_t second_layer_idx = segment->get_second().get_layer_idx();
          RTUtil::swapByASC(first_layer_idx, second_layer_idx);
          for (int32_t layer_idx = first_layer_idx; layer_idx <= second_layer_idx; layer_idx++) {
            net_layer_map[net_idx].insert(layer_idx);
          }
        }
      }
      for (auto& [net_idx, patch_set] : gcell_map[x][y].get_net_patch_map()) {
        for (EXTLayerRect* patch : patch_set) {
          net_layer_map[net_idx].insert(patch->get_layer_idx());
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
    std::ofstream* net_csv_file = RTUtil::getOutputFileStream(
        RTUtil::getString(dr_temp_directory_path, "net_map_", routing_layer.get_layer_name(), "_", dr_model.get_iter(), ".csv"));
    GridMap<int32_t>& net_map = layer_net_map[routing_layer.get_layer_idx()];
    for (int32_t y = net_map.get_y_size() - 1; y >= 0; y--) {
      for (int32_t x = 0; x < net_map.get_x_size(); x++) {
        RTUtil::pushStream(net_csv_file, net_map[x][y], ",");
      }
      RTUtil::pushStream(net_csv_file, "\n");
    }
    RTUtil::closeFileStream(net_csv_file);
  }
}

void DetailedRouter::writeViolationCSV(DRModel& dr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = DM_INST.getDatabase().get_gcell_map();
  std::string& dr_temp_directory_path = DM_INST.getConfig().dr_temp_directory_path;
  int32_t output_csv = DM_INST.getConfig().output_csv;
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
    std::ofstream* violation_csv_file = RTUtil::getOutputFileStream(
        RTUtil::getString(dr_temp_directory_path, "violation_map_", routing_layer.get_layer_name(), "_", dr_model.get_iter(), ".csv"));
    GridMap<int32_t>& violation_map = layer_violation_map[routing_layer.get_layer_idx()];
    for (int32_t y = violation_map.get_y_size() - 1; y >= 0; y--) {
      for (int32_t x = 0; x < violation_map.get_x_size(); x++) {
        RTUtil::pushStream(violation_csv_file, violation_map[x][y], ",");
      }
      RTUtil::pushStream(violation_csv_file, "\n");
    }
    RTUtil::closeFileStream(violation_csv_file);
  }
}

#endif

}  // namespace irt
