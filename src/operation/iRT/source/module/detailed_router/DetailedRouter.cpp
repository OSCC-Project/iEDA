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
#include "DRCChecker.hpp"
#include "DRNet.hpp"
#include "DRNode.hpp"
#include "DRParameter.hpp"
#include "DetailedRouter.hpp"
#include "GDSPlotter.hpp"

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

void DetailedRouter::route(std::vector<Net>& net_list)
{
  Monitor monitor;

  routeNetList(net_list);

  LOG_INST.info(Loc::current(), "The ", GetStageName()(Stage::kDetailedRouter), " completed!", monitor.getStatsInfo());
}

// private

DetailedRouter* DetailedRouter::_dr_instance = nullptr;

void DetailedRouter::routeNetList(std::vector<Net>& net_list)
{
  DRModel dr_model = initDRModel(net_list);
  addAccessPointToGCellMap(dr_model);
  addTAResultToGCellMap(dr_model);
  iterativeDRModel(dr_model);
}

DRModel DetailedRouter::initDRModel(std::vector<Net>& net_list)
{
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
  dr_net.set_ta_result_list(net.get_ta_result_list());
  return dr_net;
}

void DetailedRouter::addAccessPointToGCellMap(DRModel& dr_model)
{
  std::vector<DRNet>& dr_net_list = dr_model.get_dr_net_list();

  for (DRNet& dr_net : dr_net_list) {
    for (DRPin& dr_pin : dr_net.get_dr_pin_list()) {
      DM_INST.updateAccessPointToGCellMap(ChangeType::kAdd, dr_net.get_net_idx(), &dr_pin.get_protected_access_point());
    }
  }
}

void DetailedRouter::addTAResultToGCellMap(DRModel& dr_model)
{
  std::vector<DRNet>& dr_net_list = dr_model.get_dr_net_list();

  for (DRNet& dr_net : dr_net_list) {
    for (Segment<LayerCoord>& segment : dr_net.get_ta_result_list()) {
      DM_INST.updateNetResultToGCellMap(ChangeType::kAdd, dr_net.get_net_idx(), new Segment<LayerCoord>(segment));
    }
  }
}

void DetailedRouter::iterativeDRModel(DRModel& dr_model)
{
  std::vector<DRParameter> dr_parameter_list = {{1, 7, 0, 8, 0, true}};
  // std::vector<DRParameter> dr_parameter_list = {{1, 7, 0, 8, 0, true}, {2, 7, -3, 8, 8, true}, {3, 7, -5, 8, 8, true}};
  for (DRParameter& dr_parameter : dr_parameter_list) {
    Monitor iter_monitor;
    LOG_INST.info(Loc::current(), "****** Start Model Iteration(", dr_parameter.get_curr_iter(), "/", dr_parameter_list.size(), ") ******");
    dr_model.set_curr_dr_parameter(dr_parameter);
    initDRBoxMap(dr_model);
    buildDRBoxMap(dr_model);
    buildBoxSchedule(dr_model);
    routeDRBoxMap(dr_model);
    LOG_INST.info(Loc::current(), "****** End Model Iteration(", dr_parameter.get_curr_iter(), "/", dr_parameter_list.size(), ")",
                  iter_monitor.getStatsInfo(), " ******");
  }
}

void DetailedRouter::initDRBoxMap(DRModel& dr_model)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();

  irt_int x_gcell_num = 0;
  for (ScaleGrid& x_grid : gcell_axis.get_x_grid_list()) {
    x_gcell_num += x_grid.get_step_num();
  }
  irt_int y_gcell_num = 0;
  for (ScaleGrid& y_grid : gcell_axis.get_y_grid_list()) {
    y_gcell_num += y_grid.get_step_num();
  }

  DRParameter& curr_dr_parameter = dr_model.get_curr_dr_parameter();
  irt_int size = curr_dr_parameter.get_size();
  irt_int offset = curr_dr_parameter.get_offset();
  irt_int x_box_num = std::ceil((x_gcell_num - offset) / 1.0 / size);
  irt_int y_box_num = std::ceil((y_gcell_num - offset) / 1.0 / size);

  GridMap<DRBox>& dr_box_map = dr_model.get_dr_box_map();
  dr_box_map.init(x_box_num, y_box_num);

  for (irt_int x = 0; x < dr_box_map.get_x_size(); x++) {
    for (irt_int y = 0; y < dr_box_map.get_y_size(); y++) {
      irt_int grid_lb_x = std::max(offset + x * size, 0);
      irt_int grid_lb_y = std::max(offset + y * size, 0);
      irt_int grid_rt_x = std::min(offset + (x + 1) * size - 1, x_gcell_num - 1);
      irt_int grid_rt_y = std::min(offset + (y + 1) * size - 1, y_gcell_num - 1);

      PlanarRect lb_gcell_rect = RTUtil::getRealRect(PlanarCoord(grid_lb_x, grid_lb_y), gcell_axis);
      PlanarRect rt_gcell_rect = RTUtil::getRealRect(PlanarCoord(grid_rt_x, grid_rt_y), gcell_axis);
      PlanarRect box_real_rect(lb_gcell_rect.get_lb(), rt_gcell_rect.get_rt());
      PlanarRect box_grid_rect(grid_lb_x, grid_lb_y, grid_rt_x, grid_rt_y);

      DRBox& dr_box = dr_box_map[x][y];

      EXTPlanarRect dr_box_rect;
      dr_box_rect.set_real_rect(box_real_rect);
      dr_box_rect.set_grid_rect(box_grid_rect);
      dr_box.set_box_rect(dr_box_rect);
      DRBoxId dr_box_id;
      dr_box_id.set_x(x);
      dr_box_id.set_y(y);
      dr_box.set_dr_box_id(dr_box_id);
      dr_box.set_curr_dr_parameter(&curr_dr_parameter);
    }
  }
}

void DetailedRouter::buildDRBoxMap(DRModel& dr_model)
{
  GridMap<DRBox>& dr_box_map = dr_model.get_dr_box_map();
  for (irt_int x = 0; x < dr_box_map.get_x_size(); x++) {
    for (irt_int y = 0; y < dr_box_map.get_y_size(); y++) {
      DRBox& dr_box = dr_box_map[x][y];
      buildBoxTrackAxis(dr_box);
      buildGraphRect(dr_box);
      splitNetResult(dr_box);
    }
  }
}

void DetailedRouter::buildBoxTrackAxis(DRBox& dr_box)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  irt_int bottom_routing_layer_idx = DM_INST.getConfig().bottom_routing_layer_idx;
  irt_int top_routing_layer_idx = DM_INST.getConfig().top_routing_layer_idx;

  std::vector<irt_int> x_scale_list;
  std::vector<irt_int> y_scale_list;

  PlanarRect& box_region = dr_box.get_box_rect().get_real_rect();
  for (RoutingLayer& routing_layer : routing_layer_list) {
    if (routing_layer.get_layer_idx() < bottom_routing_layer_idx || top_routing_layer_idx < routing_layer.get_layer_idx()) {
      continue;
    }
    std::vector<irt_int> x_list
        = RTUtil::getClosedScaleList(box_region.get_lb_x(), box_region.get_rt_x(), routing_layer.getXTrackGridList());
    x_scale_list.insert(x_scale_list.end(), x_list.begin(), x_list.end());
    std::vector<irt_int> y_list
        = RTUtil::getClosedScaleList(box_region.get_lb_y(), box_region.get_rt_y(), routing_layer.getYTrackGridList());
    y_scale_list.insert(y_scale_list.end(), y_list.begin(), y_list.end());
  }
  for (auto [net_idx, access_point_set] : DM_INST.getNetAccessPointMap(dr_box.get_box_rect())) {
    for (const auto access_point : access_point_set) {
      x_scale_list.push_back(access_point->get_real_x());
      y_scale_list.push_back(access_point->get_real_y());
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

void DetailedRouter::buildGraphRect(DRBox& dr_box)
{
}

void DetailedRouter::splitNetResult(DRBox& dr_box)
{
  EXTPlanarRect& box_rect = dr_box.get_box_rect();
  PlanarCoord& real_lb = box_rect.get_real_lb();
  PlanarCoord& real_rt = box_rect.get_real_rt();
  ScaleAxis& box_track_axis = dr_box.get_box_track_axis();
  std::vector<irt_int> x_list = RTUtil::getClosedScaleList(real_lb.get_x(), real_rt.get_x(), box_track_axis.get_x_grid_list());
  std::vector<irt_int> y_list = RTUtil::getClosedScaleList(real_lb.get_y(), real_rt.get_y(), box_track_axis.get_y_grid_list());
  if (x_list.empty() || y_list.empty()) {
    LOG_INST.error(Loc::current(), "The track scale list is empty in box : (", dr_box.get_dr_box_id().get_x(), ",",
                   dr_box.get_dr_box_id().get_y(), ")!");
  }
  irt_int x_begin_scale = x_list.front();
  irt_int x_end_scale = x_list.back();
  irt_int y_begin_scale = y_list.front();
  irt_int y_end_scale = y_list.back();

  for (auto [net_idx, segment_list] : DM_INST.getNetResultMap(box_rect)) {
    for (const auto segment : segment_list) {
      LayerCoord first = segment->get_first();
      irt_int first_x = first.get_x();
      irt_int first_y = first.get_y();
      irt_int first_layer_idx = first.get_layer_idx();

      LayerCoord second = segment->get_second();
      irt_int second_x = second.get_x();
      irt_int second_y = second.get_y();
      irt_int second_layer_idx = second.get_layer_idx();
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
        if (first_x <= x_begin_scale && x_begin_scale <= second_x) {
          end_point_list.emplace_back(x_begin_scale, first_y, first_layer_idx);
        }
        if (first_x <= x_end_scale && x_end_scale <= second_x) {
          end_point_list.emplace_back(x_end_scale, first_y, first_layer_idx);
        }
      } else if (RTUtil::isVertical(first, second)) {
        if (first_y <= y_begin_scale && y_begin_scale <= second_y) {
          end_point_list.emplace_back(first_x, y_begin_scale, first_layer_idx);
        }
        if (first_y <= y_end_scale && y_end_scale <= second_y) {
          end_point_list.emplace_back(first_x, y_end_scale, first_layer_idx);
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
        Segment<LayerCoord>* split_segment = new Segment<LayerCoord>(end_point_list[i - 1], end_point_list[i]);
        DM_INST.updateNetResultToGCellMap(ChangeType::kAdd, net_idx, split_segment);
      }
    }
  }
}

void DetailedRouter::buildBoxSchedule(DRModel& dr_model)
{
  GridMap<DRBox>& dr_box_map = dr_model.get_dr_box_map();

  irt_int box_size = dr_box_map.get_x_size() * dr_box_map.get_y_size();
  irt_int range = std::max(3, static_cast<irt_int>(std::sqrt(box_size / RTUtil::getBatchSize(box_size))));

  std::vector<std::vector<DRBoxId>> dr_box_id_list_list;
  for (irt_int start_x = 0; start_x < range; start_x++) {
    for (irt_int start_y = 0; start_y < range; start_y++) {
      std::vector<DRBoxId> dr_box_id_list;
      for (irt_int x = start_x; x < dr_box_map.get_x_size(); x += range) {
        for (irt_int y = start_y; y < dr_box_map.get_y_size(); y += range) {
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

  GridMap<DRBox>& dr_box_map = dr_model.get_dr_box_map();

  size_t total_box_num = 0;
  for (std::vector<DRBoxId>& dr_box_id_list : dr_model.get_dr_box_id_list_list()) {
    Monitor stage_monitor;
#pragma omp parallel for
    for (DRBoxId& dr_box_id : dr_box_id_list) {
      DRBox& dr_box = dr_box_map[dr_box_id.get_x()][dr_box_id.get_y()];
      initDRTaskList(dr_model, dr_box);
      buildDRTaskList(dr_box);
      buildViolationList(dr_box);
      initLayerNodeMap(dr_box);
      buildNeighborMap(dr_box);
      buildOrienNetMap(dr_box);
      checkDRBox(dr_box);
      routeDRBox(dr_box);
      updateDRTaskToGcellMap(dr_box);
      updateViolationToGcellMap(dr_box);
      freeDRBox(dr_box);
    }
    total_box_num += dr_box_id_list.size();
    LOG_INST.info(Loc::current(), "Routed ", dr_box_id_list.size(), " boxes", stage_monitor.getStatsInfo());
  }
  LOG_INST.info(Loc::current(), "Routed ", total_box_num, " boxes", monitor.getStatsInfo());
}

void DetailedRouter::initDRTaskList(DRModel& dr_model, DRBox& dr_box)
{
  std::vector<DRNet>& dr_net_list = dr_model.get_dr_net_list();
  std::vector<DRTask*>& dr_task_list = dr_box.get_dr_task_list();
  for (auto [net_idx, connect_point_list] : getNetConnectPointMap(dr_box)) {
    if (connect_point_list.size() < 2) {
      LOG_INST.error(Loc::current(), "The size of connect points is illegal for net : ", net_idx, "!");
    }
    std::vector<DRGroup> dr_group_list;
    for (const LayerCoord& connect_point : connect_point_list) {
      DRGroup dr_group;
      dr_group.get_coord_direction_map()[connect_point].insert({});
      dr_group_list.push_back(dr_group);
    }
    DRTask* dr_task = new DRTask();
    dr_task->set_net_idx(net_idx);
    dr_task->set_connect_type(dr_net_list[net_idx].get_connect_type());
    dr_task->set_dr_group_list(dr_group_list);
    buildBoundingBox(dr_box, dr_task);
    dr_task->set_routed_times(0);
    dr_task_list.push_back(dr_task);
  }
  std::sort(dr_task_list.begin(), dr_task_list.end(), CmpDRTask());
}

std::map<irt_int, std::vector<LayerCoord>> DetailedRouter::getNetConnectPointMap(DRBox& dr_box)
{
  std::map<irt_int, std::vector<LayerCoord>> net_connect_point_map;
  for (auto [net_idx, access_point_list] : DM_INST.getNetAccessPointMap(dr_box.get_box_rect())) {
    for (AccessPoint* access_point : access_point_list) {
      net_connect_point_map[net_idx].push_back(LayerCoord(access_point->get_real_coord(), access_point->get_layer_idx()));
    }
  }
  for (auto [net_idx, boundary_point_list] : getBoundaryPointMap(dr_box)) {
    std::sort(boundary_point_list.begin(), boundary_point_list.end(), CmpLayerCoordByLayerASC());
    boundary_point_list.erase(std::unique(boundary_point_list.begin(), boundary_point_list.end()), boundary_point_list.end());
    for (LayerCoord& boundary_point : boundary_point_list) {
      net_connect_point_map[net_idx].push_back(boundary_point);
    }
  }
  for (auto& [net_idx, connect_point_list] : net_connect_point_map) {
    if (connect_point_list.size() < 2) {
      LOG_INST.error(Loc::current(), "The connect point size is illegal for net:", net_idx, " in DRBox (", dr_box.get_dr_box_id().get_x(),
                     ",", dr_box.get_dr_box_id().get_y(), ")!");
    }
  }
  return net_connect_point_map;
}

std::map<irt_int, std::vector<LayerCoord>> DetailedRouter::getBoundaryPointMap(DRBox& dr_box)
{
  std::map<irt_int, std::vector<LayerCoord>> boundary_point_map;

  EXTPlanarRect& box_rect = dr_box.get_box_rect();
  PlanarCoord& real_lb = box_rect.get_real_lb();
  PlanarCoord& real_rt = box_rect.get_real_rt();
  ScaleAxis& box_track_axis = dr_box.get_box_track_axis();
  std::vector<irt_int> x_list = RTUtil::getClosedScaleList(real_lb.get_x(), real_rt.get_x(), box_track_axis.get_x_grid_list());
  std::vector<irt_int> y_list = RTUtil::getClosedScaleList(real_lb.get_y(), real_rt.get_y(), box_track_axis.get_y_grid_list());
  if (x_list.empty() || y_list.empty()) {
    LOG_INST.error(Loc::current(), "The track scale list is empty in box : (", dr_box.get_dr_box_id().get_x(), ",",
                   dr_box.get_dr_box_id().get_y(), ")!");
  }
  irt_int x_begin_scale = x_list.front();
  irt_int x_end_scale = x_list.back();
  irt_int y_begin_scale = y_list.front();
  irt_int y_end_scale = y_list.back();

  for (auto [net_idx, segment_list] : DM_INST.getNetResultMap(box_rect)) {
    std::vector<LayerCoord>& boundary_point_list = boundary_point_map[net_idx];
    for (const auto segment : segment_list) {
      LayerCoord first = segment->get_first();
      irt_int first_x = first.get_x();
      irt_int first_y = first.get_y();
      irt_int first_layer_idx = first.get_layer_idx();

      LayerCoord second = segment->get_second();
      irt_int second_x = second.get_x();
      irt_int second_y = second.get_y();
      irt_int second_layer_idx = second.get_layer_idx();
      RTUtil::swapByASC(first_x, second_x);
      RTUtil::swapByASC(first_y, second_y);

      if (first_layer_idx != second_layer_idx) {
        continue;
      }

      if (RTUtil::isHorizontal(first, second)) {
        if (first_x <= x_begin_scale && x_begin_scale <= second_x) {
          boundary_point_list.emplace_back(x_begin_scale, first_y, first_layer_idx);
        }
        if (first_x <= x_end_scale && x_end_scale <= second_x) {
          boundary_point_list.emplace_back(x_end_scale, first_y, first_layer_idx);
        }
      } else if (RTUtil::isVertical(first, second)) {
        if (first_y <= y_begin_scale && y_begin_scale <= second_y) {
          boundary_point_list.emplace_back(first_x, y_begin_scale, first_layer_idx);
        }
        if (first_y <= y_end_scale && y_end_scale <= second_y) {
          boundary_point_list.emplace_back(first_x, y_end_scale, first_layer_idx);
        }
      } else {
        LOG_INST.error(Loc::current(), "Routing Segmet is oblique!");
      }
    }
  }
  return boundary_point_map;
}

void DetailedRouter::buildBoundingBox(DRBox& dr_box, DRTask* dr_task)
{
  std::vector<PlanarCoord> coord_list;
  for (DRGroup& dr_group : dr_task->get_dr_group_list()) {
    for (auto& [coord, direction_set] : dr_group.get_coord_direction_map()) {
      coord_list.push_back(coord);
    }
  }
  dr_task->set_bounding_box(RTUtil::getBoundingBox(coord_list));
}

void DetailedRouter::buildDRTaskList(DRBox& dr_box)
{
  // 将GCellMap中的segment+patch（segment和patch完全在Box内）拷贝到DRTask下，并且在GCellMap中移除
  EXTPlanarRect& box_rect = dr_box.get_box_rect();
  PlanarCoord& real_lb = box_rect.get_real_lb();
  PlanarCoord& real_rt = box_rect.get_real_rt();
  ScaleAxis& box_track_axis = dr_box.get_box_track_axis();
  std::vector<irt_int> x_list = RTUtil::getClosedScaleList(real_lb.get_x(), real_rt.get_x(), box_track_axis.get_x_grid_list());
  std::vector<irt_int> y_list = RTUtil::getClosedScaleList(real_lb.get_y(), real_rt.get_y(), box_track_axis.get_y_grid_list());
  if (x_list.empty() || y_list.empty()) {
    LOG_INST.error(Loc::current(), "The track scale list is empty in box : (", dr_box.get_dr_box_id().get_x(), ",",
                   dr_box.get_dr_box_id().get_y(), ")!");
  }
  PlanarRect track_bounding_box(x_list.front(), y_list.front(), x_list.back(), y_list.back());

  std::map<irt_int, DRTask*> net_task_map;
  for (DRTask* dr_task : dr_box.get_dr_task_list()) {
    net_task_map[dr_task->get_net_idx()] = dr_task;
  }
  for (auto [net_idx, segment_set] : DM_INST.getNetResultMap(box_rect)) {
    if (!RTUtil::exist(net_task_map, net_idx)) {
      LOG_INST.error(Loc::current(), "Can not find DRTask by net : ", net_idx, "!");
    }
    for (Segment<LayerCoord>* segment : segment_set) {
      if (RTUtil::isInside(track_bounding_box, segment->get_first()) && RTUtil::isInside(track_bounding_box, segment->get_second())) {
        net_task_map[net_idx]->get_routing_segment_list().push_back(*segment);
        DM_INST.updateNetResultToGCellMap(ChangeType::kDel, net_idx, segment);
      }
    }
  }
  for (auto [net_idx, patch_set] : DM_INST.getNetPatchMap(box_rect)) {
    if (!RTUtil::exist(net_task_map, net_idx)) {
      LOG_INST.error(Loc::current(), "Can not find DRTask by net : ", net_idx, "!");
    }
    for (EXTLayerRect* patch : patch_set) {
      if (RTUtil::isInside(track_bounding_box, patch->get_real_rect())) {
        net_task_map[net_idx]->get_patch_list().push_back(*patch);
        DM_INST.updatePatchToGCellMap(ChangeType::kDel, net_idx, patch);
      }
    }
  }
}

void DetailedRouter::buildViolationList(DRBox& dr_box)
{
  for (Violation* violation : DM_INST.getViolationSet(dr_box.get_box_rect())) {
    if (RTUtil::isInside(dr_box.get_graph_rect(), violation->get_violation_shape().get_real_rect())) {
      dr_box.get_violation_list().push_back(*violation);
      DM_INST.updateViolationToGCellMap(ChangeType::kDel, violation);
    }
  }
}

void DetailedRouter::initLayerNodeMap(DRBox& dr_box)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  PlanarCoord& real_lb = dr_box.get_box_rect().get_real_lb();
  PlanarCoord& real_rt = dr_box.get_box_rect().get_real_rt();
  ScaleAxis& box_track_axis = dr_box.get_box_track_axis();
  std::vector<irt_int> x_list = RTUtil::getClosedScaleList(real_lb.get_x(), real_rt.get_x(), box_track_axis.get_x_grid_list());
  std::vector<irt_int> y_list = RTUtil::getClosedScaleList(real_lb.get_y(), real_rt.get_y(), box_track_axis.get_y_grid_list());

  std::vector<GridMap<DRNode>>& layer_node_map = dr_box.get_layer_node_map();
  layer_node_map.resize(routing_layer_list.size());
  for (irt_int layer_idx = 0; layer_idx < static_cast<irt_int>(layer_node_map.size()); layer_idx++) {
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

void DetailedRouter::buildNeighborMap(DRBox& dr_box)
{
#if 1  // all connect
  irt_int bottom_routing_layer_idx = DM_INST.getConfig().bottom_routing_layer_idx;
  irt_int top_routing_layer_idx = DM_INST.getConfig().top_routing_layer_idx;

  std::vector<GridMap<DRNode>>& layer_node_map = dr_box.get_layer_node_map();
  for (irt_int layer_idx = 0; layer_idx < static_cast<irt_int>(layer_node_map.size()); layer_idx++) {
    bool routing_hv = true;
    if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
      routing_hv = false;
    }
    GridMap<DRNode>& dr_node_map = layer_node_map[layer_idx];
    for (irt_int x = 0; x < dr_node_map.get_x_size(); x++) {
      for (irt_int y = 0; y < dr_node_map.get_y_size(); y++) {
        std::map<Orientation, DRNode*>& neighbor_node_map = dr_node_map[x][y].get_neighbor_node_map();
        if (routing_hv) {
          if (x != 0) {
            neighbor_node_map[Orientation::kWest] = &dr_node_map[x - 1][y];
          }
          if (x != (dr_node_map.get_x_size() - 1)) {
            neighbor_node_map[Orientation::kEast] = &dr_node_map[x + 1][y];
          }
          if (y != 0) {
            neighbor_node_map[Orientation::kSouth] = &dr_node_map[x][y - 1];
          }
          if (y != (dr_node_map.get_y_size() - 1)) {
            neighbor_node_map[Orientation::kNorth] = &dr_node_map[x][y + 1];
          }
        }
        if (layer_idx != 0) {
          neighbor_node_map[Orientation::kDown] = &layer_node_map[layer_idx - 1][x][y];
        }
        if (layer_idx != static_cast<irt_int>(layer_node_map.size()) - 1) {
          neighbor_node_map[Orientation::kUp] = &layer_node_map[layer_idx + 1][x][y];
        }
      }
    }
  }
#else
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  irt_int bottom_routing_layer_idx = DM_INST.getConfig().bottom_routing_layer_idx;
  irt_int top_routing_layer_idx = DM_INST.getConfig().top_routing_layer_idx;

  PlanarCoord& real_lb = dr_box.get_base_region().get_lb();
  PlanarCoord& real_rt = dr_box.get_base_region().get_rt();
  ScaleAxis& box_track_axis = dr_box.get_box_track_axis();
  std::vector<irt_int> x_list = RTUtil::getClosedScaleList(real_lb.get_x(), real_rt.get_x(), box_track_axis.get_x_grid_list());
  std::vector<irt_int> y_list = RTUtil::getClosedScaleList(real_lb.get_y(), real_rt.get_y(), box_track_axis.get_y_grid_list());

  std::map<irt_int, std::vector<irt_int>> layer_grid_x_map;
  std::map<irt_int, std::vector<irt_int>> layer_grid_y_map;
  for (RoutingLayer& routing_layer : routing_layer_list) {
    irt_int layer_idx = routing_layer.get_layer_idx();
    if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
      continue;
    }
    std::vector<irt_int> x_scale_list = RTUtil::getClosedScaleList(real_lb.get_x(), real_rt.get_x(), routing_layer.getXTrackGridList());
    std::set<irt_int> x_scale_set(x_scale_list.begin(), x_scale_list.end());
    for (irt_int x = 0; x < static_cast<irt_int>(x_list.size()); x++) {
      if (!RTUtil::exist(x_scale_set, x_list[x])) {
        continue;
      }
      layer_grid_x_map[layer_idx].push_back(x);
    }
    std::vector<irt_int> y_scale_list = RTUtil::getClosedScaleList(real_lb.get_y(), real_rt.get_y(), routing_layer.getYTrackGridList());
    std::set<irt_int> y_scale_set(y_scale_list.begin(), y_scale_list.end());
    for (irt_int y = 0; y < static_cast<irt_int>(y_list.size()); y++) {
      if (!RTUtil::exist(y_scale_set, y_list[y])) {
        continue;
      }
      layer_grid_y_map[layer_idx].push_back(y);
    }
  }

  std::vector<GridMap<DRNode>>& layer_node_map = dr_box.get_layer_node_map();

  // 在可布线层，为本层track建立平面连接
  for (RoutingLayer& routing_layer : routing_layer_list) {
    irt_int layer_idx = routing_layer.get_layer_idx();
    if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
      continue;
    }
    GridMap<DRNode>& dr_node_map = layer_node_map[layer_idx];
    for (irt_int x : layer_grid_x_map[layer_idx]) {
      for (irt_int y = 0; y < dr_node_map.get_y_size(); y++) {
        std::map<Orientation, DRNode*>& neighbor_node_map = dr_node_map[x][y].get_neighbor_node_map();
        if (y != 0) {
          neighbor_node_map[Orientation::kSouth] = &dr_node_map[x][y - 1];
        }
        if (y != (dr_node_map.get_y_size() - 1)) {
          neighbor_node_map[Orientation::kNorth] = &dr_node_map[x][y + 1];
        }
      }
    }
    for (irt_int y : layer_grid_y_map[layer_idx]) {
      for (irt_int x = 0; x < dr_node_map.get_x_size(); x++) {
        std::map<Orientation, DRNode*>& neighbor_node_map = dr_node_map[x][y].get_neighbor_node_map();
        if (x != 0) {
          neighbor_node_map[Orientation::kWest] = &dr_node_map[x - 1][y];
        }
        if (x != (dr_node_map.get_x_size() - 1)) {
          neighbor_node_map[Orientation::kEast] = &dr_node_map[x + 1][y];
        }
      }
    }
  }
  // 在可布线层，为相邻层track的交点建立空间连接
  for (RoutingLayer& routing_layer : routing_layer_list) {
    irt_int layer_idx = routing_layer.get_layer_idx();
    if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
      continue;
    }
    GridMap<DRNode>& dr_node_map = layer_node_map[layer_idx];
    if (layer_idx != bottom_routing_layer_idx) {
      for (irt_int x : layer_grid_x_map[layer_idx]) {
        for (irt_int y : layer_grid_y_map[layer_idx - 1]) {
          std::map<Orientation, DRNode*>& neighbor_node_map = dr_node_map[x][y].get_neighbor_node_map();
          neighbor_node_map[Orientation::kDown] = &layer_node_map[layer_idx - 1][x][y];
        }
      }
      for (irt_int x : layer_grid_x_map[layer_idx - 1]) {
        for (irt_int y : layer_grid_y_map[layer_idx]) {
          std::map<Orientation, DRNode*>& neighbor_node_map = dr_node_map[x][y].get_neighbor_node_map();
          neighbor_node_map[Orientation::kDown] = &layer_node_map[layer_idx - 1][x][y];
        }
      }
    }
    if (layer_idx != top_routing_layer_idx) {
      for (irt_int x : layer_grid_x_map[layer_idx]) {
        for (irt_int y : layer_grid_y_map[layer_idx + 1]) {
          std::map<Orientation, DRNode*>& neighbor_node_map = dr_node_map[x][y].get_neighbor_node_map();
          neighbor_node_map[Orientation::kUp] = &layer_node_map[layer_idx + 1][x][y];
        }
      }
      for (irt_int x : layer_grid_x_map[layer_idx + 1]) {
        for (irt_int y : layer_grid_y_map[layer_idx]) {
          std::map<Orientation, DRNode*>& neighbor_node_map = dr_node_map[x][y].get_neighbor_node_map();
          neighbor_node_map[Orientation::kUp] = &layer_node_map[layer_idx + 1][x][y];
        }
      }
    }
  }
  // ap点映射到全层，平面上，在可布线层内连接到最近的当前层track上；空间上，贯穿所有层的通孔。
  std::vector<PlanarCoord> access_grid_coord_list;
  irt_int x_size = static_cast<irt_int>(x_list.size());
  irt_int y_size = static_cast<irt_int>(y_list.size());
  for (DRTask& dr_task : dr_box.get_dr_task_list()) {
    for (DRGroup& dr_group : dr_task.get_dr_group_list()) {
      for (auto& [coord, direction_set] : dr_group.get_coord_direction_map()) {
        access_grid_coord_list.push_back(RTUtil::getGridCoord(coord, box_track_axis));
      }
    }
  }
  for (irt_int x_idx = 0; x_idx < x_size; x_idx++) {
    if (x_idx == 0 || x_idx == x_size - 1) {
      for (irt_int y_idx = 0; y_idx < y_size; y_idx++) {
        access_grid_coord_list.emplace_back(x_idx, y_idx);
      }
    } else {
      access_grid_coord_list.emplace_back(x_idx, 0);
      access_grid_coord_list.emplace_back(x_idx, y_size - 1);
    }
  }

  for (PlanarCoord& access_grid_coord : access_grid_coord_list) {
    irt_int grid_x = access_grid_coord.get_x();
    irt_int grid_y = access_grid_coord.get_y();
    for (RoutingLayer& routing_layer : routing_layer_list) {
      irt_int layer_idx = routing_layer.get_layer_idx();
      if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
        continue;
      }
      GridMap<DRNode>& dr_node_map = layer_node_map[layer_idx];
      {
        std::vector<irt_int>& grid_x_list = layer_grid_x_map[layer_idx];
        std::pair<irt_int, irt_int> curr_layer_adj_grid_x = RTUtil::getAdjacentScale(grid_x, grid_x_list);
        irt_int begin_grid_x = curr_layer_adj_grid_x.first;
        irt_int end_grid_x = curr_layer_adj_grid_x.second;
        for (irt_int x = begin_grid_x + 1; x <= end_grid_x; x++) {
          DRNode& west = dr_node_map[x - 1][grid_y];
          DRNode& east = dr_node_map[x][grid_y];
          west.get_neighbor_node_map()[Orientation::kEast] = &east;
          east.get_neighbor_node_map()[Orientation::kWest] = &west;
        }
      }
      {
        std::vector<irt_int>& grid_y_list = layer_grid_y_map[layer_idx];
        std::pair<irt_int, irt_int> curr_layer_adj_grid_y = RTUtil::getAdjacentScale(grid_y, grid_y_list);
        irt_int begin_grid_y = curr_layer_adj_grid_y.first;
        irt_int end_grid_y = curr_layer_adj_grid_y.second;
        for (irt_int y = begin_grid_y + 1; y <= end_grid_y; y++) {
          DRNode& south = dr_node_map[grid_x][y - 1];
          DRNode& north = dr_node_map[grid_x][y];
          south.get_neighbor_node_map()[Orientation::kNorth] = &north;
          north.get_neighbor_node_map()[Orientation::kSouth] = &south;
        }
      }
    }
  }

  for (PlanarCoord& access_grid_coord : access_grid_coord_list) {
    irt_int grid_x = access_grid_coord.get_x();
    irt_int grid_y = access_grid_coord.get_y();
    for (irt_int via_below_layer_idx = routing_layer_list.front().get_layer_idx();
         via_below_layer_idx < routing_layer_list.back().get_layer_idx(); via_below_layer_idx++) {
      DRNode& down = layer_node_map[via_below_layer_idx][grid_x][grid_y];
      DRNode& up = layer_node_map[via_below_layer_idx + 1][grid_x][grid_y];
      down.get_neighbor_node_map()[Orientation::kUp] = &up;
      up.get_neighbor_node_map()[Orientation::kDown] = &down;
    }
  }
#endif
}

void DetailedRouter::buildOrienNetMap(DRBox& dr_box)
{
  for (bool is_routing : {true, false}) {
    for (auto& [layer_idx, net_fixed_rect_map] : DM_INST.getLayerNetFixedRectMap(dr_box.get_box_rect(), is_routing)) {
      for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
        for (EXTLayerRect* fixed_rect : fixed_rect_set) {
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

void DetailedRouter::checkDRBox(DRBox& dr_box)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  DRBoxId& dr_box_id = dr_box.get_dr_box_id();
  if (dr_box_id.get_x() < 0 || dr_box_id.get_y() < 0) {
    LOG_INST.error(Loc::current(), "The grid coord is illegal!");
  }

  PlanarCoord& real_lb = dr_box.get_box_rect().get_real_lb();
  PlanarCoord& real_rt = dr_box.get_box_rect().get_real_rt();
  ScaleAxis& box_track_axis = dr_box.get_box_track_axis();
  std::vector<irt_int> x_scale_list = RTUtil::getClosedScaleList(real_lb.get_x(), real_rt.get_x(), box_track_axis.get_x_grid_list());
  std::vector<irt_int> y_scale_list = RTUtil::getClosedScaleList(real_lb.get_y(), real_rt.get_y(), box_track_axis.get_y_grid_list());

  for (GridMap<DRNode>& dr_node_map : dr_box.get_layer_node_map()) {
    for (irt_int x_idx = 0; x_idx < dr_node_map.get_x_size(); x_idx++) {
      for (irt_int y_idx = 0; y_idx < dr_node_map.get_y_size(); y_idx++) {
        DRNode& dr_node = dr_node_map[x_idx][y_idx];
        if (!RTUtil::isInside(dr_box.get_box_rect().get_real_rect(), dr_node.get_planar_coord())) {
          LOG_INST.error(Loc::current(), "The dr node is out of box!");
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
        irt_int node_x = dr_node.get_planar_coord().get_x();
        irt_int node_y = dr_node.get_planar_coord().get_y();
        for (auto& [orien, neighbor] : dr_node.get_neighbor_node_map()) {
          if (orien == Orientation::kUp || orien == Orientation::kDown) {
            continue;
          }
          PlanarCoord neighbor_coord(node_x, node_y);
          switch (orien) {
            case Orientation::kEast:
              if (x_scale_list[x_idx] != node_x || (x_idx + 1) >= static_cast<irt_int>(x_scale_list.size())) {
                LOG_INST.error(Loc::current(), "The adjacent scale does not exist!");
              }
              neighbor_coord.set_x(x_scale_list[x_idx + 1]);
              break;
            case Orientation::kWest:
              if (x_scale_list[x_idx] != node_x || (x_idx - 1) < 0) {
                LOG_INST.error(Loc::current(), "The adjacent scale does not exist!");
              }
              neighbor_coord.set_x(x_scale_list[x_idx - 1]);
              break;
            case Orientation::kNorth:
              if (y_scale_list[y_idx] != node_y || (y_idx + 1) >= static_cast<irt_int>(y_scale_list.size())) {
                LOG_INST.error(Loc::current(), "The adjacent scale does not exist!");
              }
              neighbor_coord.set_y(y_scale_list[y_idx + 1]);
              break;
            case Orientation::kSouth:
              if (y_scale_list[y_idx] != node_y || (y_idx - 1) < 0) {
                LOG_INST.error(Loc::current(), "The adjacent scale does not exist!");
              }
              neighbor_coord.set_y(y_scale_list[y_idx - 1]);
              break;
            default:
              break;
          }
          if (neighbor_coord == neighbor->get_planar_coord()) {
            continue;
          }
          LOG_INST.error(Loc::current(), "The neighbor coord is different with real coord!");
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
        irt_int layer_idx = coord.get_layer_idx();
        if (routing_layer_list.back().get_layer_idx() < layer_idx || layer_idx < routing_layer_list.front().get_layer_idx()) {
          LOG_INST.error(Loc::current(), "The layer idx of group coord is illegal!");
        }
        if (RTUtil::isInside(dr_box.get_box_rect().get_real_rect(), coord)) {
          continue;
        }
        LOG_INST.error(Loc::current(), "The coord (", coord.get_x(), ",", coord.get_y(), ") is out of box!");
      }
    }
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
  bool complete_ripup = dr_box.get_curr_dr_parameter()->get_complete_ripup();

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
  std::set<irt_int> violation_net_set;
  for (Violation* violation : DM_INST.getViolationSet(dr_box.get_box_rect())) {
    for (irt_int violation_net : violation->get_violation_net_set()) {
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
  updateTaskResult(dr_box, dr_task);
  resetSingleTask(dr_box);
}

void DetailedRouter::initSingleTask(DRBox& dr_box, DRTask* dr_task)
{
  ScaleAxis& box_track_axis = dr_box.get_box_track_axis();
  std::vector<GridMap<DRNode>>& layer_node_map = dr_box.get_layer_node_map();

  // single task
  dr_box.set_curr_net_idx(dr_task->get_net_idx());
  {
    std::vector<std::vector<DRNode*>> node_list_list;
    std::vector<DRGroup>& dr_group_list = dr_task->get_dr_group_list();
    for (DRGroup& dr_group : dr_group_list) {
      std::vector<DRNode*> node_comb;
      for (auto& [coord, direction_set] : dr_group.get_coord_direction_map()) {
        if (!RTUtil::existTrackGrid(coord, box_track_axis)) {
          LOG_INST.error(Loc::current(), "The coord can not find grid!");
        }
        PlanarCoord grid_coord = RTUtil::getTrackGridCoord(coord, box_track_axis);
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
        dr_box.set_end_node_comb_idx(static_cast<irt_int>(i));
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
  irt_int end_node_comb_idx = dr_box.get_end_node_comb_idx();

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

void DetailedRouter::updateTaskResult(DRBox& dr_box, DRTask* dr_task)
{
  std::vector<Segment<LayerCoord>>& routing_segment_list = dr_task->get_routing_segment_list();

  // 原结果从graph删除
  for (Segment<LayerCoord>& routing_segment : routing_segment_list) {
    updateNetResultToGraph(dr_box, ChangeType::kDel, dr_task->get_net_idx(), routing_segment);
  }
  routing_segment_list = dr_box.get_routing_segment_list();
  // 新结果添加到graph
  for (Segment<LayerCoord>& routing_segment : routing_segment_list) {
    updateNetResultToGraph(dr_box, ChangeType::kAdd, dr_task->get_net_idx(), routing_segment);
  }
}

void DetailedRouter::resetSingleTask(DRBox& dr_box)
{
  dr_box.set_curr_net_idx(-1);
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
  irt_int shape_cost = dr_box.get_curr_dr_parameter()->get_shape_cost();

  double cost = 0;
  cost += (shape_cost * curr_node->getOverlapShapeNum(dr_box.get_curr_net_idx(), orientation));
  cost += curr_node->getViolationCost(orientation);
  return cost;
}

double DetailedRouter::getKnowWireCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
{
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  double dr_prefer_wire_unit = DM_INST.getConfig().dr_prefer_wire_unit;
  double dr_nonprefer_wire_unit = DM_INST.getConfig().dr_nonprefer_wire_unit;

  double wire_cost = 0;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    wire_cost += RTUtil::getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());

    RoutingLayer& routing_layer = routing_layer_list[start_node->get_layer_idx()];
    if (routing_layer.get_prefer_direction() == RTUtil::getDirection(*start_node, *end_node)) {
      wire_cost *= dr_prefer_wire_unit;
    } else {
      wire_cost *= dr_nonprefer_wire_unit;
    }
  }
  return wire_cost;
}

double DetailedRouter::getKnowCornerCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
{
  double dr_corner_unit = DM_INST.getConfig().dr_corner_unit;

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
      corner_cost += dr_corner_unit;
    } else if (direction_set.size() == 2) {
      LOG_INST.error(Loc::current(), "Direction set is error!");
    }
  }
  return corner_cost;
}

double DetailedRouter::getKnowViaCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
{
  double dr_via_unit = DM_INST.getConfig().dr_via_unit;

  double via_cost = (dr_via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
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
  double dr_prefer_wire_unit = DM_INST.getConfig().dr_prefer_wire_unit;

  double wire_cost = 0;
  wire_cost += RTUtil::getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());
  wire_cost *= dr_prefer_wire_unit;
  return wire_cost;
}

double DetailedRouter::getEstimateCornerCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
{
  double dr_corner_unit = DM_INST.getConfig().dr_corner_unit;

  double corner_cost = 0;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    if (RTUtil::isOblique(*start_node, *end_node)) {
      corner_cost += dr_corner_unit;
    }
  }
  return corner_cost;
}

double DetailedRouter::getEstimateViaCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
{
  double dr_via_unit = DM_INST.getConfig().dr_via_unit;

  double via_cost = (dr_via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
  return via_cost;
}

void DetailedRouter::applyPatch(DRBox& dr_box, DRTask* dr_task)
{
  std::vector<EXTLayerRect> new_patch_list;
  // std::vector<EXTLayerRect> new_patch_list = getPatch();

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

#if 0 

void ViolationRepairer::repairNotch(VRModel& vr_model, VRNet& vr_net)
{
  std::map<irt_int, GTLPolySetInt> layer_polygon_set_map;
  {
    // pin_shape
    for (VRPin& vr_pin : vr_net.get_vr_pin_list()) {
      for (const EXTLayerRect& routing_shape : vr_pin.get_routing_shape_list()) {
        LayerRect shape_real_rect(routing_shape.get_real_rect(), routing_shape.get_layer_idx());
        layer_polygon_set_map[shape_real_rect.get_layer_idx()] += RTUtil::convertToGTLRectInt(shape_real_rect);
      }
    }
    // vr_result_tree
    for (DRCShape& drc_shape : getDRCShapeList(vr_net.get_net_idx(), vr_net.get_vr_result_tree())) {
      if (!drc_shape.get_is_routing()) {
        continue;
      }
      LayerRect& layer_rect = drc_shape.get_layer_rect();
      layer_polygon_set_map[layer_rect.get_layer_idx()] += RTUtil::convertToGTLRectInt(layer_rect);
    }
  }
  std::vector<LayerRect> candidate_patch_list;
  for (auto& [layer_idx, polygon_set] : layer_polygon_set_map) {
    std::vector<GTLPolyInt> polygon_list;
    polygon_set.get_polygons(polygon_list);
    for (GTLPolyInt& polygon : polygon_list) {
      std::vector<GTLPointInt> gtl_point_list(polygon.begin(), polygon.end());
      // 构建点集
      std::vector<PlanarCoord> origin_point_list;
      for (GTLPointInt& gtl_point : gtl_point_list) {
        origin_point_list.emplace_back(gtl_point.x(), gtl_point.y());
      }
      // 构建任务
      std::vector<std::vector<PlanarCoord>> task_point_list_list;
      for (size_t i = 0; i < origin_point_list.size(); i++) {
        std::vector<PlanarCoord> task_point_list;
        for (size_t j = 0; j < 4; j++) {
          task_point_list.push_back(origin_point_list[(i + j) % origin_point_list.size()]);
        }
        task_point_list_list.push_back(task_point_list);
      }
      for (std::vector<PlanarCoord>& task_point_list : task_point_list_list) {
        for (LayerRect& patch : getNotchPatchList(layer_idx, task_point_list)) {
          candidate_patch_list.push_back(patch);
        }
      }
    }
  }
  std::vector<LayerRect> patch_list;
  for (LayerRect& candidate_patch : candidate_patch_list) {
    DRCShape drc_shape(vr_net.get_net_idx(), candidate_patch, true);
    if (!hasVREnvViolation(vr_model, VRSourceType::kBlockage, {DRCCheckType::kSpacing}, drc_shape)
        && !hasVREnvViolation(vr_model, VRSourceType::kNetShape, {DRCCheckType::kSpacing}, drc_shape)) {
      patch_list.push_back(candidate_patch);
    }
  }
  for (LayerRect& patch : patch_list) {
    TNode<PhysicalNode>* root_node = vr_net.get_vr_result_tree().get_root();

    PhysicalNode physical_node;
    PatchNode& patch_node = physical_node.getNode<PatchNode>();
    patch_node.set_net_idx(vr_net.get_net_idx());
    patch_node.set_rect(patch);
    patch_node.set_layer_idx(patch.get_layer_idx());

    root_node->addChild(new TNode<PhysicalNode>(physical_node));

    for (DRCShape& drc_shape : DC_INST.getDRCShapeList(vr_net.get_net_idx(), physical_node)) {
      updateRectToUnit(vr_model, ChangeType::kAdd, VRSourceType::kNetShape, drc_shape);
    }
  }
}

std::vector<LayerRect> ViolationRepairer::getNotchPatchList(irt_int layer_idx, std::vector<PlanarCoord>& task_point_list)
{
  /**
   * task_point_list 顺时针
   * notch spacing数据是从这里来 _layer_notch_spacing_length_map
   */

  if (task_point_list.size() < 4) {
    LOG_INST.error(Loc::current(), "insufficient points to detect a notch.");
    return {};
  }

  std::vector<LayerRect> patch_list;

  double notch_length = _layer_notch_spacing_length_map[layer_idx].first;
  double notch_space = _layer_notch_spacing_length_map[layer_idx].second;

  std::array<irt_int, 3> length_list;
  for (irt_int i = 0; i < 3; i++) {
    length_list[i] = RTUtil::getManhattanDistance(task_point_list[i], task_point_list[i + 1]);
  }

  if (length_list[0] < notch_length || length_list[2] < notch_length) {
    if (layer_idx >= 2 || (layer_idx < 2 && (length_list[0] >= notch_length || length_list[2] >= notch_length))) {
      if (length_list[1] <= notch_space) {
        if (RTUtil::isConcaveCorner(task_point_list[0], task_point_list[1], task_point_list[2])
            && RTUtil::isConcaveCorner(task_point_list[1], task_point_list[2], task_point_list[3])) {
          irt_int offset = 0;
          if (length_list[0] > length_list[2]) {
            // use point 123
            offset = 1;
          }
          irt_int lb_x
              = std::min({task_point_list[0 + offset].get_x(), task_point_list[1 + offset].get_x(), task_point_list[2 + offset].get_x()});
          irt_int lb_y
              = std::min({task_point_list[0 + offset].get_y(), task_point_list[1 + offset].get_y(), task_point_list[2 + offset].get_y()});
          irt_int rt_x
              = std::max({task_point_list[0 + offset].get_x(), task_point_list[1 + offset].get_x(), task_point_list[2 + offset].get_x()});
          irt_int rt_y
              = std::max({task_point_list[0 + offset].get_y(), task_point_list[1 + offset].get_y(), task_point_list[2 + offset].get_y()});

          PlanarCoord lb_coord(lb_x, lb_y);
          PlanarCoord rt_coord(rt_x, rt_y);

          patch_list.emplace_back(lb_coord, rt_coord, layer_idx);
        }
      }
    }
  }

  return patch_list;
}

void ViolationRepairer::repairNotch(VRModel& vr_model, VRNet& vr_net)
{
  std::map<irt_int, GTLPolySetInt> layer_polygon_set_map;
  {
    // pin_shape
    for (VRPin& vr_pin : vr_net.get_vr_pin_list()) {
      for (const EXTLayerRect& routing_shape : vr_pin.get_routing_shape_list()) {
        LayerRect shape_real_rect(routing_shape.get_real_rect(), routing_shape.get_layer_idx());
        layer_polygon_set_map[shape_real_rect.get_layer_idx()] += RTUtil::convertToGTLRectInt(shape_real_rect);
      }
    }
    // vr_result_tree
    for (DRCShape& drc_shape : getDRCShapeList(vr_net.get_net_idx(), vr_net.get_vr_result_tree())) {
      if (!drc_shape.get_is_routing()) {
        continue;
      }
      LayerRect& layer_rect = drc_shape.get_layer_rect();
      layer_polygon_set_map[layer_rect.get_layer_idx()] += RTUtil::convertToGTLRectInt(layer_rect);
    }
  }
  std::vector<LayerRect> candidate_patch_list;
  for (auto& [layer_idx, polygon_set] : layer_polygon_set_map) {
    std::vector<GTLPolyInt> polygon_list;
    polygon_set.get_polygons(polygon_list);
    for (GTLPolyInt& polygon : polygon_list) {
      std::vector<GTLPointInt> gtl_point_list(polygon.begin(), polygon.end());
      // 构建点集
      std::vector<PlanarCoord> origin_point_list;
      for (GTLPointInt& gtl_point : gtl_point_list) {
        origin_point_list.emplace_back(gtl_point.x(), gtl_point.y());
      }
      // 构建任务
      std::vector<std::vector<PlanarCoord>> task_point_list_list;
      for (size_t i = 0; i < origin_point_list.size(); i++) {
        std::vector<PlanarCoord> task_point_list;
        for (size_t j = 0; j < 4; j++) {
          task_point_list.push_back(origin_point_list[(i + j) % origin_point_list.size()]);
        }
        task_point_list_list.push_back(task_point_list);
      }
      for (std::vector<PlanarCoord>& task_point_list : task_point_list_list) {
        for (LayerRect& patch : getNotchPatchList(layer_idx, task_point_list)) {
          candidate_patch_list.push_back(patch);
        }
      }
    }
  }
  std::vector<LayerRect> patch_list;
  for (LayerRect& candidate_patch : candidate_patch_list) {
    DRCShape drc_shape(vr_net.get_net_idx(), candidate_patch, true);
    if (!hasVREnvViolation(vr_model, VRSourceType::kBlockage, {DRCCheckType::kSpacing}, drc_shape)
        && !hasVREnvViolation(vr_model, VRSourceType::kNetShape, {DRCCheckType::kSpacing}, drc_shape)) {
      patch_list.push_back(candidate_patch);
    }
  }
  for (LayerRect& patch : patch_list) {
    TNode<PhysicalNode>* root_node = vr_net.get_vr_result_tree().get_root();

    PhysicalNode physical_node;
    PatchNode& patch_node = physical_node.getNode<PatchNode>();
    patch_node.set_net_idx(vr_net.get_net_idx());
    patch_node.set_rect(patch);
    patch_node.set_layer_idx(patch.get_layer_idx());

    root_node->addChild(new TNode<PhysicalNode>(physical_node));

    for (DRCShape& drc_shape : DC_INST.getDRCShapeList(vr_net.get_net_idx(), physical_node)) {
      updateRectToUnit(vr_model, ChangeType::kAdd, VRSourceType::kNetShape, drc_shape);
    }
  }
}

std::vector<LayerRect> ViolationRepairer::getNotchPatchList(irt_int layer_idx, std::vector<PlanarCoord>& task_point_list)
{
  /**
   * task_point_list 顺时针
   * notch spacing数据是从这里来 _layer_notch_spacing_length_map
   */

  if (task_point_list.size() < 4) {
    LOG_INST.error(Loc::current(), "insufficient points to detect a notch.");
    return {};
  }

  std::vector<LayerRect> patch_list;

  double notch_length = _layer_notch_spacing_length_map[layer_idx].first;
  double notch_space = _layer_notch_spacing_length_map[layer_idx].second;

  std::array<irt_int, 3> length_list;
  for (irt_int i = 0; i < 3; i++) {
    length_list[i] = RTUtil::getManhattanDistance(task_point_list[i], task_point_list[i + 1]);
  }

  if (length_list[0] < notch_length || length_list[2] < notch_length) {
    if (layer_idx >= 2 || (layer_idx < 2 && (length_list[0] >= notch_length || length_list[2] >= notch_length))) {
      if (length_list[1] <= notch_space) {
        if (RTUtil::isConcaveCorner(task_point_list[0], task_point_list[1], task_point_list[2])
            && RTUtil::isConcaveCorner(task_point_list[1], task_point_list[2], task_point_list[3])) {
          irt_int offset = 0;
          if (length_list[0] > length_list[2]) {
            // use point 123
            offset = 1;
          }
          irt_int lb_x
              = std::min({task_point_list[0 + offset].get_x(), task_point_list[1 + offset].get_x(), task_point_list[2 + offset].get_x()});
          irt_int lb_y
              = std::min({task_point_list[0 + offset].get_y(), task_point_list[1 + offset].get_y(), task_point_list[2 + offset].get_y()});
          irt_int rt_x
              = std::max({task_point_list[0 + offset].get_x(), task_point_list[1 + offset].get_x(), task_point_list[2 + offset].get_x()});
          irt_int rt_y
              = std::max({task_point_list[0 + offset].get_y(), task_point_list[1 + offset].get_y(), task_point_list[2 + offset].get_y()});

          PlanarCoord lb_coord(lb_x, lb_y);
          PlanarCoord rt_coord(rt_x, rt_y);

          patch_list.emplace_back(lb_coord, rt_coord, layer_idx);
        }
      }
    }
  }

  return patch_list;
}

void ViolationRepairer::repairMinArea(VRModel& vr_model, VRNet& vr_net)
{
  Die& die = DM_INST.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();

  std::map<irt_int, GTLPolySetInt> layer_polygon_set_map;
  {
    // pin_shape
    for (VRPin& vr_pin : vr_net.get_vr_pin_list()) {
      for (EXTLayerRect& routing_shape : vr_pin.get_routing_shape_list()) {
        LayerRect shape_real_rect = routing_shape.getRealLayerRect();
        layer_polygon_set_map[shape_real_rect.get_layer_idx()] += RTUtil::convertToGTLRectInt(shape_real_rect);
      }
    }
    // vr_result_tree
    for (DRCShape& drc_shape : getDRCShapeList(vr_net.get_net_idx(), vr_net.get_vr_result_tree())) {
      if (!drc_shape.get_is_routing()) {
        continue;
      }
      LayerRect& layer_rect = drc_shape.get_layer_rect();
      layer_polygon_set_map[layer_rect.get_layer_idx()] += RTUtil::convertToGTLRectInt(layer_rect);
    }
  }
  std::map<LayerRect, irt_int, CmpLayerRectByXASC> violation_rect_added_area_map;
  for (auto& [layer_idx, polygon_set] : layer_polygon_set_map) {
    irt_int layer_min_area = routing_layer_list[layer_idx].get_min_area();
    std::vector<GTLPolyInt> polygon_list;
    polygon_set.get_polygons(polygon_list);
    for (GTLPolyInt& polygon : polygon_list) {
      if (gtl::area(polygon) >= layer_min_area) {
        continue;
      }
      // 取polygon中最大的矩形进行膨胀
      PlanarRect max_violation_rect;
      std::vector<GTLRectInt> gtl_rect_list;
      gtl::get_max_rectangles(gtl_rect_list, polygon);
      for (GTLRectInt& gtl_rect : gtl_rect_list) {
        if (max_violation_rect.getArea() < gtl::area(gtl_rect)) {
          max_violation_rect = RTUtil::convertToPlanarRect(gtl_rect);
        }
      }
      irt_int added_area = layer_min_area - gtl::area(polygon);
      violation_rect_added_area_map[LayerRect(max_violation_rect, layer_idx)] = added_area;
    }
  }
  std::vector<LayerRect> patch_list;
  for (auto& [violation_rect, added_area] : violation_rect_added_area_map) {
    irt_int layer_idx = violation_rect.get_layer_idx();
    std::vector<LayerRect> h_candidate_patch_list;
    {
      irt_int h_enlarged_offset = static_cast<irt_int>(std::ceil(added_area / 1.0 / violation_rect.getYSpan()));
      for (irt_int lb_offset = 0; lb_offset <= h_enlarged_offset; lb_offset++) {
        PlanarRect enlarged_rect
            = RTUtil::getEnlargedRect(violation_rect, lb_offset, 0, h_enlarged_offset - lb_offset, 0, die.get_real_rect());
        if (lb_offset == 0 || lb_offset == h_enlarged_offset) {
          std::vector<PlanarRect> split_rect_list = RTUtil::getSplitRectList(enlarged_rect, violation_rect, Direction::kHorizontal);
          if (split_rect_list.size() != 1) {
            LOG_INST.error(Loc::current(), "The size of split_rect_list is not equal 1!");
          }
          enlarged_rect = split_rect_list.front();
        }
        h_candidate_patch_list.emplace_back(enlarged_rect, layer_idx);
      }
    }
    std::vector<LayerRect> v_candidate_patch_list;
    {
      irt_int v_enlarged_offset = static_cast<irt_int>(std::ceil(added_area / 1.0 / violation_rect.getXSpan()));
      for (irt_int lb_offset = 0; lb_offset <= v_enlarged_offset; lb_offset++) {
        PlanarRect enlarged_rect
            = RTUtil::getEnlargedRect(violation_rect, 0, lb_offset, 0, v_enlarged_offset - lb_offset, die.get_real_rect());
        if (lb_offset == 0 || lb_offset == v_enlarged_offset) {
          std::vector<PlanarRect> split_rect_list = RTUtil::getSplitRectList(enlarged_rect, violation_rect, Direction::kVertical);
          if (split_rect_list.size() != 1) {
            LOG_INST.error(Loc::current(), "The size of split_rect_list is not equal 1!");
          }
          enlarged_rect = split_rect_list.front();
        }
        v_candidate_patch_list.emplace_back(enlarged_rect, layer_idx);
      }
    }
    std::vector<LayerRect> candidate_patch_list;
    if (routing_layer_list[layer_idx].isPreferH()) {
      candidate_patch_list.insert(candidate_patch_list.end(), h_candidate_patch_list.begin(), h_candidate_patch_list.end());
      candidate_patch_list.insert(candidate_patch_list.end(), v_candidate_patch_list.begin(), v_candidate_patch_list.end());
    } else {
      candidate_patch_list.insert(candidate_patch_list.end(), v_candidate_patch_list.begin(), v_candidate_patch_list.end());
      candidate_patch_list.insert(candidate_patch_list.end(), h_candidate_patch_list.begin(), h_candidate_patch_list.end());
    }
    bool has_patch = false;
    for (LayerRect& candidate_patch : candidate_patch_list) {
      DRCShape drc_shape(vr_net.get_net_idx(), candidate_patch, true);
      if (!hasVREnvViolation(vr_model, VRSourceType::kBlockage, {DRCCheckType::kSpacing}, drc_shape)
          && !hasVREnvViolation(vr_model, VRSourceType::kNetShape, {DRCCheckType::kSpacing}, drc_shape)) {
        patch_list.push_back(candidate_patch);
        has_patch = true;
        break;
      }
    }
    if (!has_patch) {
      LOG_INST.warn(Loc::current(), "There is no legal patch for min area violation!");
    }
  }
  for (LayerRect& patch : patch_list) {
    TNode<PhysicalNode>* root_node = vr_net.get_vr_result_tree().get_root();

    PhysicalNode physical_node;
    PatchNode& patch_node = physical_node.getNode<PatchNode>();
    patch_node.set_net_idx(vr_net.get_net_idx());
    patch_node.set_rect(patch);
    patch_node.set_layer_idx(patch.get_layer_idx());

    root_node->addChild(new TNode<PhysicalNode>(physical_node));
  }
}

#endif

void DetailedRouter::updateViolationList(DRBox& dr_box)
{
  std::vector<Violation> new_violation_list;
  // std::vector<Violation> new_violation_list = getViolationByDRC();

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

std::vector<Violation> DetailedRouter::getViolationListByIDRC(DRBox& dr_box)
{
  std::map<irt_int, std::vector<idb::IdbRegularWireSegment*>> net_idb_segment_map;

  for (bool is_routing : {true, false}) {
    for (auto& [layer_idx, net_fixed_rect_map] : DM_INST.getLayerNetFixedRectMap(dr_box.get_box_rect(), is_routing)) {
      for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
        for (EXTLayerRect* fixed_rect : fixed_rect_set) {
          net_idb_segment_map[net_idx].push_back(DM_INST.getIDBSegment(net_idx, fixed_rect));
        }
      }
    }
  }
  for (DRTask* dr_task : dr_box.get_dr_task_list()) {
    for (Segment<LayerCoord>& routing_segment : dr_task->get_routing_segment_list()) {
      net_idb_segment_map[dr_task->get_net_idx()].push_back(DM_INST.getIDBSegment(dr_task->get_net_idx(), &routing_segment));
    }
    for (EXTLayerRect& patch : dr_task->get_patch_list()) {
      net_idb_segment_map[dr_task->get_net_idx()].push_back(DM_INST.getIDBSegment(dr_task->get_net_idx(), &patch));
    }
  }
  return RTAPI_INST.getViolationList(net_idb_segment_map);
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

#if 1  // update env

void DetailedRouter::updateFixedRectToGraph(DRBox& dr_box, ChangeType change_type, irt_int net_idx, EXTLayerRect* fixed_rect,
                                            bool is_routing)
{
  NetShape net_shape(net_idx, fixed_rect->getRealLayerRect(), is_routing);
  updateNetShapeToGraph(dr_box, change_type, net_shape);
}

void DetailedRouter::updateNetResultToGraph(DRBox& dr_box, ChangeType change_type, irt_int net_idx, Segment<LayerCoord>& segment)
{
  for (NetShape& net_shape : DM_INST.getNetShapeList(net_idx, segment)) {
    updateNetShapeToGraph(dr_box, change_type, net_shape);
  }
}

void DetailedRouter::updatePatchToGraph(DRBox& dr_box, ChangeType change_type, irt_int net_idx, EXTLayerRect& patch)
{
  NetShape net_shape(net_idx, patch.getRealLayerRect(), true);
  updateNetShapeToGraph(dr_box, change_type, net_shape);
}

void DetailedRouter::updateNetShapeToGraph(DRBox& dr_box, ChangeType change_type, NetShape& net_shape)
{
  for (auto& [dr_node, orientation_set] : getNodeOrientationMap(dr_box, net_shape)) {
    if (!dr_node->get_is_valid()) {
      continue;
    }
    for (Orientation orientation : orientation_set) {
      if (change_type == ChangeType::kAdd) {
        dr_node->get_orien_net_map()[orientation].insert(net_shape.get_net_idx());
      } else if (change_type == ChangeType::kDel) {
        dr_node->get_orien_net_map()[orientation].erase(net_shape.get_net_idx());
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
  if (!net_shape.get_is_routing()) {
    LOG_INST.error(Loc::current(), "The type of net_shape is cut!");
  }
  RoutingLayer& routing_layer = DM_INST.getDatabase().get_routing_layer_list()[net_shape.get_layer_idx()];
  ViaMaster& via_master = DM_INST.getDatabase().get_layer_via_master_list()[net_shape.get_layer_idx()].front();

  // 膨胀size为 min_spacing + std::max(1/2*width, 1/2*below_enclosure_length)
  irt_int enlarged_size = routing_layer.getMinSpacing(net_shape.get_rect())
                          + std::max(routing_layer.get_min_width() / 2, via_master.get_below_enclosure().getLength() / 2);
  PlanarRect searched_rect = RTUtil::getEnlargedRect(net_shape.get_rect(), enlarged_size);

  std::map<DRNode*, std::set<Orientation>> node_orientation_map;
  if (RTUtil::existTrackGrid(searched_rect, dr_box.get_box_track_axis())) {
    PlanarRect grid_rect = RTUtil::getTrackGridRect(searched_rect, dr_box.get_box_track_axis());
    for (irt_int grid_x = grid_rect.get_lb_x(); grid_x <= grid_rect.get_rt_x(); grid_x++) {
      for (irt_int grid_y = grid_rect.get_lb_y(); grid_y <= grid_rect.get_rt_y(); grid_y++) {
        DRNode& node = dr_box.get_layer_node_map()[net_shape.get_layer_idx()][grid_x][grid_y];
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
  if (net_shape.get_is_routing()) {
    LOG_INST.error(Loc::current(), "The type of net_shape is routing!");
  }

  std::vector<irt_int> adjacent_routing_layer_idx_list = DM_INST.getHelper().getAdjacentRoutingLayerIdxList(net_shape.get_layer_idx());
  if (adjacent_routing_layer_idx_list.size() != 2) {
    // 如果相邻层只有一个，那么将不会在这构建graph
    return {};
  }
  irt_int bottom_routing_layer_idx = adjacent_routing_layer_idx_list.front();
  irt_int top_routing_layer_idx = adjacent_routing_layer_idx_list.back();
  RTUtil::swapByASC(bottom_routing_layer_idx, top_routing_layer_idx);

  irt_int cut_spacing = DM_INST.getDatabase().get_cut_layer_list()[net_shape.get_layer_idx()].getMinSpacing(net_shape.get_rect());
  PlanarRect& cut_shape = DM_INST.getDatabase().get_layer_via_master_list()[bottom_routing_layer_idx].front().get_cut_shape_list().front();

  // 膨胀size为 min_spacing + 当前cut层的half_length和half_width
  irt_int enlarge_x_size = cut_spacing + cut_shape.getXSpan() / 2;
  irt_int enlarge_y_size = cut_spacing + cut_shape.getYSpan() / 2;

  PlanarRect searched_rect = RTUtil::getEnlargedRect(net_shape.get_rect(), enlarge_x_size, enlarge_y_size, enlarge_x_size, enlarge_y_size);

  std::map<DRNode*, std::set<Orientation>> node_orientation_map;
  if (RTUtil::existTrackGrid(searched_rect, dr_box.get_box_track_axis())) {
    PlanarRect grid_rect = RTUtil::getTrackGridRect(searched_rect, dr_box.get_box_track_axis());
    for (irt_int grid_x = grid_rect.get_lb_x(); grid_x <= grid_rect.get_rt_x(); grid_x++) {
      for (irt_int grid_y = grid_rect.get_lb_y(); grid_y <= grid_rect.get_rt_y(); grid_y++) {
        node_orientation_map[&dr_box.get_layer_node_map()[bottom_routing_layer_idx][grid_x][grid_y]].insert(Orientation::kUp);
        node_orientation_map[&dr_box.get_layer_node_map()[top_routing_layer_idx][grid_x][grid_y]].insert(Orientation::kDown);
      }
    }
  }
  return node_orientation_map;
}

void DetailedRouter::updateViolationToGraph(DRBox& dr_box, ChangeType change_type, Violation& violation)
{
  irt_int violation_cost = dr_box.get_curr_dr_parameter()->get_violation_cost();
  NetShape net_shape(-1, violation.get_violation_shape().getRealLayerRect(), violation.get_is_routing());

  for (auto& [dr_node, orientation_set] : getNodeOrientationMap(dr_box, net_shape)) {
    if (!dr_node->get_is_valid()) {
      continue;
    }
    for (Orientation orientation : orientation_set) {
      if (change_type == ChangeType::kAdd) {
        dr_node->get_orien_violation_cost_map()[orientation] += violation_cost;
      } else if (change_type == ChangeType::kDel) {
        dr_node->get_orien_violation_cost_map()[orientation] -= violation_cost;
      }
    }
  }
}

#endif

#if 1  // plot dr_box

void DetailedRouter::plotDRBox(DRBox& dr_box, irt_int curr_task_idx)
{
#if 0
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = DM_INST.getDatabase().get_routing_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = DM_INST.getDatabase().get_layer_via_master_list();
  std::string dr_temp_directory_path = DM_INST.getConfig().dr_temp_directory_path;

  irt_int width = INT_MAX;
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
  gp_boundary.set_rect(dr_box.get_base_region());
  base_region_struct.push(gp_boundary);
  gp_gds.addStruct(base_region_struct);

  // gcell_axis
  GPStruct gcell_axis_struct("gcell_axis");
  std::vector<irt_int> gcell_x_list
      = RTUtil::getClosedScaleList(dr_box.get_base_region().get_lb_x(), dr_box.get_base_region().get_rt_x(), gcell_axis.get_x_grid_list());
  std::vector<irt_int> gcell_y_list
      = RTUtil::getClosedScaleList(dr_box.get_base_region().get_lb_y(), dr_box.get_base_region().get_rt_y(), gcell_axis.get_y_grid_list());
  for (irt_int x : gcell_x_list) {
    GPPath gp_path;
    gp_path.set_layer_idx(0);
    gp_path.set_data_type(1);
    gp_path.set_segment(x, dr_box.get_base_region().get_lb_y(), x, dr_box.get_base_region().get_rt_y());
    gcell_axis_struct.push(gp_path);
  }
  for (irt_int y : gcell_y_list) {
    GPPath gp_path;
    gp_path.set_layer_idx(0);
    gp_path.set_data_type(1);
    gp_path.set_segment(dr_box.get_base_region().get_lb_x(), y, dr_box.get_base_region().get_rt_x(), y);
    gcell_axis_struct.push(gp_path);
  }
  gp_gds.addStruct(gcell_axis_struct);

  std::vector<GridMap<DRNode>>& layer_node_map = dr_box.get_layer_node_map();
  // dr_node_map
  GPStruct dr_node_map_struct("dr_node_map");
  for (GridMap<DRNode>& dr_node_map : layer_node_map) {
    for (irt_int grid_x = 0; grid_x < dr_node_map.get_x_size(); grid_x++) {
      for (irt_int grid_y = 0; grid_y < dr_node_map.get_y_size(); grid_y++) {
        DRNode& dr_node = dr_node_map[grid_x][grid_y];
        PlanarRect real_rect = RTUtil::getEnlargedRect(dr_node.get_planar_coord(), width);
        irt_int y_reduced_span = std::max(1, real_rect.getYSpan() / 12);
        irt_int y = real_rect.get_rt_y();

        GPBoundary gp_boundary;
        switch (dr_node.get_state()) {
          case DRNodeState::kNone:
            gp_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kNone));
            break;
          case DRNodeState::kOpen:
            gp_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kOpen));
            break;
          case DRNodeState::kClose:
            gp_boundary.set_data_type(static_cast<irt_int>(GPGraphType::kClose));
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
        gp_text_node_real_coord.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
        gp_text_node_real_coord.set_message(
            RTUtil::getString("(", dr_node.get_x(), " , ", dr_node.get_y(), " , ", dr_node.get_layer_idx(), ")"));
        gp_text_node_real_coord.set_layer_idx(GP_INST.getGDSIdxByRouting(dr_node.get_layer_idx()));
        gp_text_node_real_coord.set_presentation(GPTextPresentation::kLeftMiddle);
        dr_node_map_struct.push(gp_text_node_real_coord);

        y -= y_reduced_span;
        GPText gp_text_node_grid_coord;
        gp_text_node_grid_coord.set_coord(real_rect.get_lb_x(), y);
        gp_text_node_grid_coord.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
        gp_text_node_grid_coord.set_message(RTUtil::getString("(", grid_x, " , ", grid_y, " , ", dr_node.get_layer_idx(), ")"));
        gp_text_node_grid_coord.set_layer_idx(GP_INST.getGDSIdxByRouting(dr_node.get_layer_idx()));
        gp_text_node_grid_coord.set_presentation(GPTextPresentation::kLeftMiddle);
        dr_node_map_struct.push(gp_text_node_grid_coord);

        y -= y_reduced_span;
        GPText gp_text_direction_set;
        gp_text_direction_set.set_coord(real_rect.get_lb_x(), y);
        gp_text_direction_set.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
        gp_text_direction_set.set_message("direction_set: ");
        gp_text_direction_set.set_layer_idx(GP_INST.getGDSIdxByRouting(dr_node.get_layer_idx()));
        gp_text_direction_set.set_presentation(GPTextPresentation::kLeftMiddle);
        dr_node_map_struct.push(gp_text_direction_set);

        if (!dr_node.get_direction_set().empty()) {
          y -= y_reduced_span;
          GPText gp_text_direction_set_info;
          gp_text_direction_set_info.set_coord(real_rect.get_lb_x(), y);
          gp_text_direction_set_info.set_text_type(static_cast<irt_int>(GPGraphType::kInfo));
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
    for (irt_int grid_x = 0; grid_x < dr_node_map.get_x_size(); grid_x++) {
      for (irt_int grid_y = 0; grid_y < dr_node_map.get_y_size(); grid_y++) {
        DRNode& dr_node = dr_node_map[grid_x][grid_y];
        PlanarRect real_rect = RTUtil::getEnlargedRect(dr_node.get_planar_coord(), width);

        irt_int lb_x = real_rect.get_lb_x();
        irt_int lb_y = real_rect.get_lb_y();
        irt_int rt_x = real_rect.get_rt_x();
        irt_int rt_y = real_rect.get_rt_y();
        irt_int mid_x = (lb_x + rt_x) / 2;
        irt_int mid_y = (lb_y + rt_y) / 2;
        irt_int x_reduced_span = (rt_x - lb_x) / 4;
        irt_int y_reduced_span = (rt_y - lb_y) / 4;
        irt_int width = std::min(x_reduced_span, y_reduced_span) / 2;

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
          gp_path.set_layer_idx(GP_INST.getGDSIdxByRouting(dr_node.get_layer_idx()));
          gp_path.set_width(width);
          gp_path.set_data_type(static_cast<irt_int>(GPGraphType::kNeighbor));
          neighbor_map_struct.push(gp_path);
        }
      }
    }
  }
  gp_gds.addStruct(neighbor_map_struct);

  // box_track_axis
  GPStruct box_track_axis_struct("box_track_axis");
  PlanarCoord& real_lb = dr_box.get_base_region().get_lb();
  PlanarCoord& real_rt = dr_box.get_base_region().get_rt();
  ScaleAxis& box_track_axis = dr_box.get_box_track_axis();
  std::vector<irt_int> x_list = RTUtil::getClosedScaleList(real_lb.get_x(), real_rt.get_x(), box_track_axis.get_x_grid_list());
  std::vector<irt_int> y_list = RTUtil::getClosedScaleList(real_lb.get_y(), real_rt.get_y(), box_track_axis.get_y_grid_list());
  for (irt_int layer_idx = 0; layer_idx < static_cast<irt_int>(layer_node_map.size()); layer_idx++) {
#if 1
    RoutingLayer& routing_layer = routing_layer_list[layer_idx];
    x_list = RTUtil::getClosedScaleList(real_lb.get_x(), real_rt.get_x(), routing_layer.getXTrackGridList());
    y_list = RTUtil::getClosedScaleList(real_lb.get_y(), real_rt.get_y(), routing_layer.getYTrackGridList());
#endif
    for (irt_int x : x_list) {
      GPPath gp_path;
      gp_path.set_data_type(static_cast<irt_int>(GPGraphType::kTrackAxis));
      gp_path.set_segment(x, real_lb.get_y(), x, real_rt.get_y());
      gp_path.set_layer_idx(GP_INST.getGDSIdxByRouting(layer_idx));
      box_track_axis_struct.push(gp_path);
    }
    for (irt_int y : y_list) {
      GPPath gp_path;
      gp_path.set_data_type(static_cast<irt_int>(GPGraphType::kTrackAxis));
      gp_path.set_segment(real_lb.get_x(), y, real_rt.get_x(), y);
      gp_path.set_layer_idx(GP_INST.getGDSIdxByRouting(layer_idx));
      box_track_axis_struct.push(gp_path);
    }
  }
  gp_gds.addStruct(box_track_axis_struct);

  // source_region_query_map
  std::vector<std::pair<DRSourceType, GPGraphType>> source_graph_pair_list = {{DRSourceType::kBlockage, GPGraphType::kBlockage},
                                                                              {DRSourceType::kNetShape, GPGraphType::kNetShape},
                                                                              {DRSourceType::kReservedVia, GPGraphType::kReservedVia}};
  for (auto& [dr_source_type, gp_graph_type] : source_graph_pair_list) {
    for (bool is_routing : {true, false}) {
      for (auto& [layer_idx, info_rect_map] : DC_INST.getLayerInfoRectMap(dr_box.getRegionQuery(dr_source_type), is_routing)) {
        for (auto& [info, rect_set] : info_rect_map) {
          GPStruct net_rect_struct(RTUtil::getString(GetDRSourceTypeName()(dr_source_type), "@", info.get_net_idx()));
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
  for (DRTask& dr_task : dr_box.get_dr_task_list()) {
    GPStruct task_struct(RTUtil::getString("task_", dr_task.get_task_idx(), "(net_", dr_task.get_net_idx(), ")"));

    if (curr_task_idx == -1 || dr_task.get_task_idx() == curr_task_idx) {
      for (DRGroup& dr_group : dr_task.get_dr_group_list()) {
        for (auto& [coord, direction_set] : dr_group.get_coord_direction_map()) {
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
      gp_boundary.set_rect(dr_task.get_bounding_box().get_base_region());
      task_struct.push(gp_boundary);
    }
    for (Segment<TNode<LayerCoord>*>& segment : RTUtil::getSegListByTree(dr_task.get_routing_tree())) {
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
        RTUtil::swapByASC(first_layer_idx, second_layer_idx);
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
  std::string gds_file_path
      = RTUtil::getString(dr_temp_directory_path, "dr_box_", dr_box.get_dr_box_id().get_x(), "_", dr_box.get_dr_box_id().get_y(), ".gds");
  GP_INST.plot(gp_gds, gds_file_path, false, false);
#endif
}

#endif

#if 1  // valid drc

bool DetailedRouter::hasDREnvViolation(DRModel& dr_model, DRSourceType dr_source_type, const std::vector<DRCCheckType>& check_type_list,
                                       const DRCShape& drc_shape)
{
  return !getDREnvViolation(dr_model, dr_source_type, check_type_list, drc_shape).empty();
}

bool DetailedRouter::hasDREnvViolation(DRModel& dr_model, DRSourceType dr_source_type, const std::vector<DRCCheckType>& check_type_list,
                                       const std::vector<DRCShape>& drc_shape_list)
{
  return !getDREnvViolation(dr_model, dr_source_type, check_type_list, drc_shape_list).empty();
}

std::map<std::string, std::vector<ViolationInfo>> DetailedRouter::getDREnvViolation(DRModel& dr_model, DRSourceType dr_source_type,
                                                                                    const std::vector<DRCCheckType>& check_type_list,
                                                                                    const DRCShape& drc_shape)
{
  std::vector<DRCShape> drc_shape_list = {drc_shape};
  return getDREnvViolation(dr_model, dr_source_type, check_type_list, drc_shape_list);
}

std::map<std::string, std::vector<ViolationInfo>> DetailedRouter::getDREnvViolation(DRModel& dr_model, DRSourceType dr_source_type,
                                                                                    const std::vector<DRCCheckType>& check_type_list,
                                                                                    const std::vector<DRCShape>& drc_shape_list)
{
  ScaleAxis& gcell_axis = DM_INST.getDatabase().get_gcell_axis();
  EXTPlanarRect& die = DM_INST.getDatabase().get_die();

  GridMap<DRBox>& dr_box_map = dr_model.get_dr_box_map();

  std::map<DRBoxId, std::vector<DRCShape>, CmpDRBoxId> box_rect_map;
  for (const DRCShape& drc_shape : drc_shape_list) {
    for (const LayerRect& max_scope_real_rect : DC_INST.getMaxScope(drc_shape)) {
      PlanarRect max_scope_regular_rect = RTUtil::getRegularRect(max_scope_real_rect, die.get_real_rect());
      PlanarRect max_scope_grid_rect = RTUtil::getClosedGridRect(max_scope_regular_rect, gcell_axis);
      for (irt_int x = max_scope_grid_rect.get_lb_x(); x <= max_scope_grid_rect.get_rt_x(); x++) {
        for (irt_int y = max_scope_grid_rect.get_lb_y(); y <= max_scope_grid_rect.get_rt_y(); y++) {
          box_rect_map[DRBoxId(x, y)].push_back(drc_shape);
        }
      }
    }
  }
  std::map<std::string, std::vector<ViolationInfo>> drc_violation_map;
  for (const auto& [dr_box_id, drc_shape_list] : box_rect_map) {
    DRBox& dr_box = dr_box_map[dr_box_id.get_x()][dr_box_id.get_y()];
    for (auto& [drc, violation_list] : getDREnvViolationBySingle(dr_box, dr_source_type, check_type_list, drc_shape_list)) {
      for (auto& violation : violation_list) {
        drc_violation_map[drc].push_back(violation);
      }
    }
  }
  return drc_violation_map;
}

std::map<std::string, std::vector<ViolationInfo>> DetailedRouter::getDREnvViolationBySingle(
    DRBox& dr_box, DRSourceType dr_source_type, const std::vector<DRCCheckType>& check_type_list,
    const std::vector<DRCShape>& drc_shape_list)
{
  std::map<std::string, std::vector<ViolationInfo>> drc_violation_map;
  // drc_violation_map = DC_INST.getEnvViolationInfo(dr_box.getRegionQuery(dr_source_type), check_type_list, drc_shape_list);
  // removeInvalidDREnvViolationBySingle(dr_box, drc_violation_map);
  return drc_violation_map;
}

void DetailedRouter::removeInvalidDREnvViolationBySingle(DRBox& dr_box,
                                                         std::map<std::string, std::vector<ViolationInfo>>& drc_violation_map)
{
  // 移除dr阶段无法解决的drc
  for (auto& [drc, violation_list] : drc_violation_map) {
    std::vector<ViolationInfo> valid_violation_list;
    for (ViolationInfo& violation_info : violation_list) {
      bool is_valid = false;
      for (const BaseInfo& base_info : violation_info.get_base_info_set()) {
        if (base_info.get_dr_task_idx() != -1) {
          is_valid = true;
          break;
        }
      }
      if (is_valid) {
        valid_violation_list.push_back(violation_info);
      }
    }
    drc_violation_map[drc] = valid_violation_list;
  }
  for (auto iter = drc_violation_map.begin(); iter != drc_violation_map.end();) {
    if (iter->second.empty()) {
      iter = drc_violation_map.erase(iter);
    } else {
      iter++;
    }
  }
}

#endif

}  // namespace irt
