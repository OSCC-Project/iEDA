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
#include "DRCEngine.hpp"
#include "DRIterParam.hpp"
#include "DRNet.hpp"
#include "DRNode.hpp"
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
  routeDRModel(dr_model);
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

void DetailedRouter::routeDRModel(DRModel& dr_model)
{
  int32_t cost_unit = RTDM.getOnlyPitch();
  double prefer_wire_unit = 1;
  double non_prefer_wire_unit = 2.5 * prefer_wire_unit;
  double bend_unit = 2 * prefer_wire_unit * cost_unit;
  double via_unit = 2 * non_prefer_wire_unit * cost_unit;
  double fixed_rect_unit = 4 * non_prefer_wire_unit * cost_unit;
  double routed_rect_unit = 2 * non_prefer_wire_unit * cost_unit;
  double violation_unit = 4 * non_prefer_wire_unit * cost_unit;
  /**
   * prefer_wire_unit, non_prefer_wire_unit, bend_unit, via_unit, size, offset, schedule_interval, fixed_rect_unit, routed_rect_unit, violation_unit,
   * max_routed_times, max_candidate_patch_num
   */
  std::vector<DRIterParam> dr_iter_param_list;
  // clang-format off
  dr_iter_param_list.emplace_back(prefer_wire_unit, non_prefer_wire_unit, bend_unit, via_unit, 12, 0, 3, fixed_rect_unit, routed_rect_unit, violation_unit, 3, 10);
  dr_iter_param_list.emplace_back(prefer_wire_unit, non_prefer_wire_unit, bend_unit, via_unit, 12, 4, 3, fixed_rect_unit, routed_rect_unit, violation_unit, 3, 10);
  dr_iter_param_list.emplace_back(prefer_wire_unit, non_prefer_wire_unit, bend_unit, via_unit, 12, 8, 3, fixed_rect_unit, routed_rect_unit, violation_unit, 3, 10);
  dr_iter_param_list.emplace_back(prefer_wire_unit, non_prefer_wire_unit, bend_unit, via_unit, 12, 0, 3, fixed_rect_unit, routed_rect_unit, violation_unit, 3, 10);
  dr_iter_param_list.emplace_back(prefer_wire_unit, non_prefer_wire_unit, bend_unit, via_unit, 12, 4, 3, fixed_rect_unit, routed_rect_unit, violation_unit, 3, 10);
  dr_iter_param_list.emplace_back(prefer_wire_unit, non_prefer_wire_unit, bend_unit, via_unit, 12, 8, 3, fixed_rect_unit, routed_rect_unit, violation_unit, 3, 10);
  // clang-format on
  initRoutingState(dr_model);
  for (int32_t i = 0, iter = 1; i < static_cast<int32_t>(dr_iter_param_list.size()); i++, iter++) {
    Monitor iter_monitor;
    RTLOG.info(Loc::current(), "***** Begin iteration ", iter, "/", dr_iter_param_list.size(), "(", RTUTIL.getPercentage(iter, dr_iter_param_list.size()),
               ") *****");
    // debugPlotDRModel(dr_model, "before");
    setDRIterParam(dr_model, iter, dr_iter_param_list[i]);
    initDRBoxMap(dr_model);
    resetRoutingState(dr_model);
    buildBoxSchedule(dr_model);
    splitNetResult(dr_model);
    // debugPlotDRModel(dr_model, "middle");
    routeDRBoxMap(dr_model);
    uploadNetResult(dr_model);
    uploadNetPatch(dr_model);
    uploadViolation(dr_model);
    updateBestResult(dr_model);
    // debugPlotDRModel(dr_model, "after");
    updateSummary(dr_model);
    printSummary(dr_model);
    outputNetCSV(dr_model);
    outputViolationCSV(dr_model);
    outputJson(dr_model);
    RTLOG.info(Loc::current(), "***** End Iteration ", iter, "/", dr_iter_param_list.size(), "(", RTUTIL.getPercentage(iter, dr_iter_param_list.size()), ")",
               iter_monitor.getStatsInfo(), "*****");
    if (stopIteration(dr_model)) {
      break;
    }
  }
  selectBestResult(dr_model);
}

void DetailedRouter::initRoutingState(DRModel& dr_model)
{
  dr_model.set_initial_routing(true);
}

void DetailedRouter::setDRIterParam(DRModel& dr_model, int32_t iter, DRIterParam& dr_iter_param)
{
  dr_model.set_iter(iter);
  RTLOG.info(Loc::current(), "prefer_wire_unit: ", dr_iter_param.get_prefer_wire_unit());
  RTLOG.info(Loc::current(), "non_prefer_wire_unit: ", dr_iter_param.get_non_prefer_wire_unit());
  RTLOG.info(Loc::current(), "bend_unit: ", dr_iter_param.get_bend_unit());
  RTLOG.info(Loc::current(), "via_unit: ", dr_iter_param.get_via_unit());
  RTLOG.info(Loc::current(), "size: ", dr_iter_param.get_size());
  RTLOG.info(Loc::current(), "offset: ", dr_iter_param.get_offset());
  RTLOG.info(Loc::current(), "schedule_interval: ", dr_iter_param.get_schedule_interval());
  RTLOG.info(Loc::current(), "fixed_rect_unit: ", dr_iter_param.get_fixed_rect_unit());
  RTLOG.info(Loc::current(), "routed_rect_unit: ", dr_iter_param.get_routed_rect_unit());
  RTLOG.info(Loc::current(), "violation_unit: ", dr_iter_param.get_violation_unit());
  RTLOG.info(Loc::current(), "max_routed_times: ", dr_iter_param.get_max_routed_times());
  RTLOG.info(Loc::current(), "max_candidate_patch_num: ", dr_iter_param.get_max_candidate_patch_num());
  dr_model.set_dr_iter_param(dr_iter_param);
}

void DetailedRouter::initDRBoxMap(DRModel& dr_model)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  DRIterParam& dr_iter_param = dr_model.get_dr_iter_param();

  int32_t size = dr_iter_param.get_size();
  int32_t offset = dr_iter_param.get_offset();
  while (offset >= size) {
    offset -= size;
  }
  std::vector<int32_t> x_scale_list;
  {
    int32_t x_gcell_num = 0;
    for (ScaleGrid& x_grid : gcell_axis.get_x_grid_list()) {
      x_gcell_num += x_grid.get_step_num();
    }
    x_scale_list.push_back(0);
    for (int32_t x_scale = offset; x_scale <= x_gcell_num; x_scale += size) {
      x_scale_list.push_back(x_scale);
    }
    x_scale_list.push_back(x_gcell_num);
    std::sort(x_scale_list.begin(), x_scale_list.end());
    x_scale_list.erase(std::unique(x_scale_list.begin(), x_scale_list.end()), x_scale_list.end());
  }
  std::vector<int32_t> y_scale_list;
  {
    int32_t y_gcell_num = 0;
    for (ScaleGrid& y_grid : gcell_axis.get_y_grid_list()) {
      y_gcell_num += y_grid.get_step_num();
    }
    y_scale_list.push_back(0);
    for (int32_t y_scale = offset; y_scale <= y_gcell_num; y_scale += size) {
      y_scale_list.push_back(y_scale);
    }
    y_scale_list.push_back(y_gcell_num);
    std::sort(y_scale_list.begin(), y_scale_list.end());
    y_scale_list.erase(std::unique(y_scale_list.begin(), y_scale_list.end()), y_scale_list.end());
  }
  GridMap<DRBox>& dr_box_map = dr_model.get_dr_box_map();
  {
    int32_t x_box_num = static_cast<int32_t>(x_scale_list.size()) - 1;
    int32_t y_box_num = static_cast<int32_t>(y_scale_list.size()) - 1;
    dr_box_map.init(x_box_num, y_box_num);
  }
  for (int32_t x = 0; x < dr_box_map.get_x_size(); x++) {
    for (int32_t y = 0; y < dr_box_map.get_y_size(); y++) {
      int32_t grid_ll_x = x_scale_list[x];
      int32_t grid_ll_y = y_scale_list[y];
      int32_t grid_ur_x = x_scale_list[x + 1] - 1;
      int32_t grid_ur_y = y_scale_list[y + 1] - 1;

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
      dr_box.set_dr_iter_param(&dr_iter_param);
      dr_box.set_initial_routing(dr_model.get_initial_routing());
    }
  }
}

void DetailedRouter::resetRoutingState(DRModel& dr_model)
{
  dr_model.set_initial_routing(false);
}

void DetailedRouter::buildBoxSchedule(DRModel& dr_model)
{
  GridMap<DRBox>& dr_box_map = dr_model.get_dr_box_map();
  int32_t schedule_interval = dr_model.get_dr_iter_param().get_schedule_interval();

  std::vector<std::vector<DRBoxId>> dr_box_id_list_list;
  for (int32_t start_x = 0; start_x < schedule_interval; start_x++) {
    for (int32_t start_y = 0; start_y < schedule_interval; start_y++) {
      std::vector<DRBoxId> dr_box_id_list;
      for (int32_t x = start_x; x < dr_box_map.get_x_size(); x += schedule_interval) {
        for (int32_t y = start_y; y < dr_box_map.get_y_size(); y += schedule_interval) {
          dr_box_id_list.emplace_back(x, y);
        }
      }
      if (!dr_box_id_list.empty()) {
        dr_box_id_list_list.push_back(dr_box_id_list);
      }
    }
  }
  dr_model.set_dr_box_id_list_list(dr_box_id_list_list);
}

void DetailedRouter::splitNetResult(DRModel& dr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();

  for (auto& [net_idx, segment_set] : RTDM.getNetDetailedResultMap(die)) {
    std::set<Segment<LayerCoord>*> del_segment_set;
    std::vector<Segment<LayerCoord>> new_segment_list;
    for (Segment<LayerCoord>* segment : segment_set) {
      LayerCoord& first_coord = segment->get_first();
      LayerCoord& second_coord = segment->get_second();
      if (first_coord.get_layer_idx() != second_coord.get_layer_idx()) {
        continue;
      }
      if (RTUTIL.isHorizontal(first_coord, second_coord)) {
        int32_t first_x = first_coord.get_x();
        int32_t second_x = second_coord.get_x();
        RTUTIL.swapByASC(first_x, second_x);
        std::set<int32_t> x_pre_set;
        std::set<int32_t> x_mid_set;
        std::set<int32_t> x_post_set;
        RTUTIL.getTrackScaleSet(gcell_axis.get_x_grid_list(), first_x, second_x, x_pre_set, x_mid_set, x_post_set);
        x_mid_set.erase(first_x);
        x_mid_set.erase(second_x);
        if (x_mid_set.empty()) {
          continue;
        }
        std::vector<int32_t> x_scale_list;
        x_scale_list.push_back(first_x);
        for (int32_t x_scale : x_mid_set) {
          x_scale_list.push_back(x_scale);
        }
        x_scale_list.push_back(second_x);
        del_segment_set.insert(segment);
        for (size_t i = 1; i < x_scale_list.size(); i++) {
          new_segment_list.emplace_back(LayerCoord(x_scale_list[i - 1], first_coord.get_y(), first_coord.get_layer_idx()),
                                        LayerCoord(x_scale_list[i], first_coord.get_y(), first_coord.get_layer_idx()));
        }
      } else if (RTUTIL.isVertical(first_coord, second_coord)) {
        int32_t first_y = first_coord.get_y();
        int32_t second_y = second_coord.get_y();
        RTUTIL.swapByASC(first_y, second_y);
        std::set<int32_t> y_pre_set;
        std::set<int32_t> y_mid_set;
        std::set<int32_t> y_post_set;
        RTUTIL.getTrackScaleSet(gcell_axis.get_y_grid_list(), first_y, second_y, y_pre_set, y_mid_set, y_post_set);
        y_mid_set.erase(first_y);
        y_mid_set.erase(second_y);
        if (y_mid_set.empty()) {
          continue;
        }
        std::vector<int32_t> y_scale_list;
        y_scale_list.push_back(first_y);
        for (int32_t y_scale : y_mid_set) {
          y_scale_list.push_back(y_scale);
        }
        y_scale_list.push_back(second_y);
        del_segment_set.insert(segment);
        for (size_t i = 1; i < y_scale_list.size(); i++) {
          new_segment_list.emplace_back(LayerCoord(first_coord.get_x(), y_scale_list[i - 1], first_coord.get_layer_idx()),
                                        LayerCoord(first_coord.get_x(), y_scale_list[i], first_coord.get_layer_idx()));
        }
      }
    }
    for (Segment<LayerCoord>* del_segment : del_segment_set) {
      RTDM.updateNetDetailedResultToGCellMap(ChangeType::kDel, net_idx, del_segment);
    }
    for (Segment<LayerCoord>& new_segment : new_segment_list) {
      RTDM.updateNetDetailedResultToGCellMap(ChangeType::kAdd, net_idx, new Segment<LayerCoord>(new_segment));
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
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
      buildFixedRect(dr_box);
      buildAccessPoint(dr_box);
      buildNetResult(dr_box);
      buildNetPatch(dr_box);
      initDRTaskList(dr_model, dr_box);
      buildRouteViolation(dr_box);
      if (needRouting(dr_box)) {
        buildBoxTrackAxis(dr_box);
        buildLayerNodeMap(dr_box);
        buildLayerShadowMap(dr_box);
        buildDRNodeNeighbor(dr_box);
        buildOrientNetMap(dr_box);
        buildNetShadowMap(dr_box);
        exemptPinShape(dr_model, dr_box);
        // debugCheckDRBox(dr_box);
        // debugPlotDRBox(dr_box, "before");
        routeDRBox(dr_box);
        // debugPlotDRBox(dr_box, "after");
      }
      selectBestResult(dr_box);
      freeDRBox(dr_box);
    }
    routed_box_num += dr_box_id_list.size();
    RTLOG.info(Loc::current(), "Routed ", routed_box_num, "/", total_box_num, "(", RTUTIL.getPercentage(routed_box_num, total_box_num), ") boxes with ",
               getRouteViolationNum(dr_model), " violations", stage_monitor.getStatsInfo());
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void DetailedRouter::buildFixedRect(DRBox& dr_box)
{
  dr_box.set_type_layer_net_fixed_rect_map(RTDM.getTypeLayerNetFixedRectMap(dr_box.get_box_rect()));
}

void DetailedRouter::buildAccessPoint(DRBox& dr_box)
{
  dr_box.set_net_access_point_map(RTDM.getNetAccessPointMap(dr_box.get_box_rect()));
}

void DetailedRouter::buildNetResult(DRBox& dr_box)
{
  PlanarRect& box_real_rect = dr_box.get_box_rect().get_real_rect();

  for (auto& [net_idx, segment_set] : RTDM.getNetDetailedResultMap(dr_box.get_box_rect())) {
    for (Segment<LayerCoord>* segment : segment_set) {
      bool least_one_coord_in_box = false;
      if (RTUTIL.isInside(box_real_rect, segment->get_first()) && RTUTIL.isInside(box_real_rect, segment->get_second())) {
        if (RTUTIL.isInside(box_real_rect, segment->get_first(), false) || RTUTIL.isInside(box_real_rect, segment->get_second(), false)) {
          // 线段在box_real_rect内,但不贴边的
          least_one_coord_in_box = true;
        }
      }
      if (least_one_coord_in_box) {
        dr_box.get_net_task_detailed_result_map()[net_idx].push_back(*segment);
        RTDM.updateNetDetailedResultToGCellMap(ChangeType::kDel, net_idx, segment);
      } else {
        dr_box.get_net_detailed_result_map()[net_idx].insert(segment);
      }
    }
  }
}

void DetailedRouter::buildNetPatch(DRBox& dr_box)
{
  PlanarRect& box_real_rect = dr_box.get_box_rect().get_real_rect();

  for (auto& [net_idx, patch_set] : RTDM.getNetDetailedPatchMap(dr_box.get_box_rect())) {
    for (EXTLayerRect* patch : patch_set) {
      if (RTUTIL.isOpenOverlap(box_real_rect, patch->get_real_rect())) {
        dr_box.get_net_task_detailed_patch_map()[net_idx].push_back(*patch);
        RTDM.updateNetDetailedPatchToGCellMap(ChangeType::kDel, net_idx, patch);
      } else {
        dr_box.get_net_detailed_patch_map()[net_idx].insert(patch);
      }
    }
  }
}

void DetailedRouter::initDRTaskList(DRModel& dr_model, DRBox& dr_box)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  std::vector<DRNet>& dr_net_list = dr_model.get_dr_net_list();
  std::vector<DRTask*>& dr_task_list = dr_box.get_dr_task_list();

  EXTPlanarRect& box_rect = dr_box.get_box_rect();
  PlanarRect& box_real_rect = box_rect.get_real_rect();
  std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_task_detailed_result_map = dr_box.get_net_task_detailed_result_map();

  std::map<int32_t, std::vector<DRGroup>> net_group_list_map;
  {
    for (auto& [net_idx, access_point_set] : dr_box.get_net_access_point_map()) {
      std::map<int32_t, DRGroup> pin_group_map;
      for (AccessPoint* access_point : access_point_set) {
        if (!RTUTIL.isInside(box_real_rect, access_point->get_real_coord())) {
          continue;
        }
        pin_group_map[access_point->get_pin_idx()].get_coord_direction_map()[access_point->getRealLayerCoord()].insert(
            routing_layer_list[access_point->get_layer_idx()].get_prefer_direction());
      }
      for (auto& [pin_idx, group] : pin_group_map) {
        net_group_list_map[net_idx].push_back(group);
      }
    }
    for (auto& [net_idx, segment_list] : net_task_detailed_result_map) {
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
          if (first.get_y() < box_real_rect.get_ll_y() || box_real_rect.get_ur_y() < first.get_y()) {
            continue;
          }
          RTUTIL.swapByASC(first_x, second_x);
          if (first_x <= box_real_rect.get_ll_x() && box_real_rect.get_ll_x() <= second_x) {
            LayerCoord layer_coord(box_real_rect.get_ll_x(), first.get_y(), first.get_layer_idx());
            coord_direction_map[layer_coord].insert(Direction::kHorizontal);
          }
          if (first_x <= box_real_rect.get_ur_x() && box_real_rect.get_ur_x() <= second_x) {
            LayerCoord layer_coord(box_real_rect.get_ur_x(), first.get_y(), first.get_layer_idx());
            coord_direction_map[layer_coord].insert(Direction::kHorizontal);
          }
        } else if (RTUTIL.isVertical(first, second)) {
          int32_t first_y = first.get_y();
          int32_t second_y = second.get_y();
          if (first.get_x() < box_real_rect.get_ll_x() || box_real_rect.get_ur_x() < first.get_x()) {
            continue;
          }
          RTUTIL.swapByASC(first_y, second_y);
          if (first_y <= box_real_rect.get_ll_y() && box_real_rect.get_ll_y() <= second_y) {
            LayerCoord layer_coord(first.get_x(), box_real_rect.get_ll_y(), first.get_layer_idx());
            coord_direction_map[layer_coord].insert(Direction::kVertical);
          }
          if (first_y <= box_real_rect.get_ur_y() && box_real_rect.get_ur_y() <= second_y) {
            LayerCoord layer_coord(first.get_x(), box_real_rect.get_ur_y(), first.get_layer_idx());
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
  for (auto& [net_idx, dr_group_list] : net_group_list_map) {
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
        for (auto& [coord, _] : dr_group.get_coord_direction_map()) {
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

void DetailedRouter::buildRouteViolation(DRBox& dr_box)
{
  std::set<int32_t> need_checked_net_set;
  for (DRTask* dr_task : dr_box.get_dr_task_list()) {
    need_checked_net_set.insert(dr_task->get_net_idx());
  }
  for (Violation* violation : RTDM.getViolationSet(dr_box.get_box_rect())) {
    bool exist_checked_net = false;
    for (int32_t violation_net_idx : violation->get_violation_net_set()) {
      if (RTUTIL.exist(need_checked_net_set, violation_net_idx)) {
        exist_checked_net = true;
        break;
      }
    }
    if (exist_checked_net) {
      dr_box.get_route_violation_list().push_back(*violation);
      RTDM.updateViolationToGCellMap(ChangeType::kDel, violation);
    }
  }
}

bool DetailedRouter::needRouting(DRBox& dr_box)
{
  if (dr_box.get_dr_task_list().empty()) {
    return false;
  }
  if (dr_box.get_initial_routing() == false && dr_box.get_route_violation_list().empty()) {
    return false;
  }
  return true;
}

void DetailedRouter::buildBoxTrackAxis(DRBox& dr_box)
{
  int32_t manufacture_grid = RTDM.getDatabase().get_manufacture_grid();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  std::vector<int32_t> x_scale_list;
  std::vector<int32_t> y_scale_list;

  PlanarRect& box_real_rect = dr_box.get_box_rect().get_real_rect();
  int32_t ll_x = box_real_rect.get_ll_x();
  int32_t ll_y = box_real_rect.get_ll_y();
  int32_t ur_x = box_real_rect.get_ur_x();
  int32_t ur_y = box_real_rect.get_ur_y();
  // 避免 off_grid
  while (ll_x % manufacture_grid != 0) {
    ll_x++;
  }
  while (ll_y % manufacture_grid != 0) {
    ll_y++;
  }
  while (ur_x % manufacture_grid != 0) {
    ur_x--;
  }
  while (ur_y % manufacture_grid != 0) {
    ur_y--;
  }
  std::map<int32_t, std::pair<std::set<int32_t>, std::set<int32_t>>>& layer_axis_map = dr_box.get_layer_axis_map();
  for (RoutingLayer& routing_layer : routing_layer_list) {
    for (int32_t x_scale : RTUTIL.getScaleList(ll_x, ur_x, routing_layer.getXTrackGridList())) {
      if (routing_layer.isPreferH())
        layer_axis_map[routing_layer.get_layer_idx()].first.insert(x_scale);
    }
    for (int32_t y_scale : RTUTIL.getScaleList(ll_y, ur_y, routing_layer.getYTrackGridList())) {
      if (!routing_layer.isPreferH())
        layer_axis_map[routing_layer.get_layer_idx()].second.insert(y_scale);
    }
  }
  for (DRTask* dr_task : dr_box.get_dr_task_list()) {
    for (DRGroup& dr_group : dr_task->get_dr_group_list()) {
      for (auto& [coord, _] : dr_group.get_coord_direction_map()) {
        int32_t layer_idx = coord.get_layer_idx();
        if (routing_layer_list[layer_idx].isPreferH()) {
          layer_axis_map[layer_idx].first.insert(coord.get_x());
        } else {
          layer_axis_map[layer_idx].second.insert(coord.get_y());
        }
      }
    }
  }
  for (RoutingLayer& routing_layer : routing_layer_list) {
    for (int32_t x_scale : RTUTIL.getScaleList(ll_x, ur_x, routing_layer.getXTrackGridList())) {
      x_scale_list.push_back(x_scale);
    }
    for (int32_t y_scale : RTUTIL.getScaleList(ll_y, ur_y, routing_layer.getYTrackGridList())) {
      y_scale_list.push_back(y_scale);
    }
  }
  for (DRTask* dr_task : dr_box.get_dr_task_list()) {
    for (DRGroup& dr_group : dr_task->get_dr_group_list()) {
      for (auto& [coord, _] : dr_group.get_coord_direction_map()) {
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

void DetailedRouter::buildLayerShadowMap(DRBox& dr_box)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  dr_box.get_layer_shadow_map().resize(routing_layer_list.size());
}

void DetailedRouter::buildDRNodeNeighbor(DRBox& dr_box)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  int32_t bottom_routing_layer_idx = RTDM.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = RTDM.getConfig().top_routing_layer_idx;

  std::vector<GridMap<DRNode>>& layer_node_map = dr_box.get_layer_node_map();
  std::map<int32_t, std::pair<std::set<int32_t>, std::set<int32_t>>>& layer_axis_map = dr_box.get_layer_axis_map();
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    bool routing_hv = true;
    if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
      routing_hv = false;
    }
    GridMap<DRNode>& dr_node_map = layer_node_map[layer_idx];
    std::set<int32_t> neighbor_layer_x_axis_set;
    std::set<int32_t> neighbor_layer_y_axis_set;
    if (layer_idx != 0) {
      neighbor_layer_x_axis_set.insert(layer_axis_map[layer_idx - 1].first.begin(), layer_axis_map[layer_idx - 1].first.end());
      neighbor_layer_y_axis_set.insert(layer_axis_map[layer_idx - 1].second.begin(), layer_axis_map[layer_idx - 1].second.end());
    }
    if (layer_idx != static_cast<int32_t>(layer_node_map.size()) - 1) {
      neighbor_layer_x_axis_set.insert(layer_axis_map[layer_idx + 1].first.begin(), layer_axis_map[layer_idx + 1].first.end());
      neighbor_layer_y_axis_set.insert(layer_axis_map[layer_idx + 1].second.begin(), layer_axis_map[layer_idx + 1].second.end());
    }
    std::set<int32_t>& curr_axis = (routing_layer_list[layer_idx].isPreferH()) ? layer_axis_map[layer_idx].first : layer_axis_map[layer_idx].second;
    for (int32_t x = 0; x < dr_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < dr_node_map.get_y_size(); y++) {
        std::map<Orientation, DRNode*>& neighbor_node_map = dr_node_map[x][y].get_neighbor_node_map();
        if (routing_hv) {
          if (!routing_layer_list[layer_idx].isPreferH()) {
            if (RTUTIL.exist(curr_axis, dr_node_map[x][y].get_y())) {
              if (x != 0) {
                neighbor_node_map[Orientation::kWest] = &dr_node_map[x - 1][y];
              }
              if (x != (dr_node_map.get_x_size() - 1)) {
                neighbor_node_map[Orientation::kEast] = &dr_node_map[x + 1][y];
              }
            }
            if (RTUTIL.exist(neighbor_layer_x_axis_set, dr_node_map[x][y].get_x())) {
              if (y != 0) {
                neighbor_node_map[Orientation::kSouth] = &dr_node_map[x][y - 1];
              }
              if (y != (dr_node_map.get_y_size() - 1)) {
                neighbor_node_map[Orientation::kNorth] = &dr_node_map[x][y + 1];
              }
            }
          } else if (routing_layer_list[layer_idx].isPreferH()) {
            if (RTUTIL.exist(curr_axis, dr_node_map[x][y].get_x())) {
              if (y != 0) {
                neighbor_node_map[Orientation::kSouth] = &dr_node_map[x][y - 1];
              }
              if (y != (dr_node_map.get_y_size() - 1)) {
                neighbor_node_map[Orientation::kNorth] = &dr_node_map[x][y + 1];
              }
            }
            if (RTUTIL.exist(neighbor_layer_y_axis_set, dr_node_map[x][y].get_y())) {
              if (x != 0) {
                neighbor_node_map[Orientation::kWest] = &dr_node_map[x - 1][y];
              }
              if (x != (dr_node_map.get_x_size() - 1)) {
                neighbor_node_map[Orientation::kEast] = &dr_node_map[x + 1][y];
              }
            }
          }
        }
        if (layer_idx != 0) {
          neighbor_node_map[Orientation::kBelow] = &layer_node_map[layer_idx - 1][x][y];
        }
        if (layer_idx != static_cast<int32_t>(layer_node_map.size()) - 1) {
          neighbor_node_map[Orientation::kAbove] = &layer_node_map[layer_idx + 1][x][y];
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
  for (auto& [net_idx, segment_set] : dr_box.get_net_detailed_result_map()) {
    for (Segment<LayerCoord>* segment : segment_set) {
      updateFixedRectToGraph(dr_box, ChangeType::kAdd, net_idx, segment);
    }
  }
  for (auto& [net_idx, segment_list] : dr_box.get_net_task_detailed_result_map()) {
    for (Segment<LayerCoord>& segment : segment_list) {
      updateRoutedRectToGraph(dr_box, ChangeType::kAdd, net_idx, segment);
    }
  }
  for (auto& [net_idx, patch_set] : dr_box.get_net_detailed_patch_map()) {
    for (EXTLayerRect* patch : patch_set) {
      updateFixedRectToGraph(dr_box, ChangeType::kAdd, net_idx, patch, true);
    }
  }
  for (auto& [net_idx, patch_list] : dr_box.get_net_task_detailed_patch_map()) {
    for (EXTLayerRect& patch : patch_list) {
      updateRoutedRectToGraph(dr_box, ChangeType::kAdd, net_idx, patch, true);
    }
  }
  for (Violation& violation : dr_box.get_route_violation_list()) {
    addRouteViolationToGraph(dr_box, violation);
  }
}

void DetailedRouter::buildNetShadowMap(DRBox& dr_box)
{
  for (auto& [is_routing, layer_net_fixed_rect_map] : dr_box.get_type_layer_net_fixed_rect_map()) {
    for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
      for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
        for (auto& fixed_rect : fixed_rect_set) {
          updateFixedRectToShadow(dr_box, ChangeType::kAdd, net_idx, fixed_rect, is_routing);
        }
      }
    }
  }
  for (auto& [net_idx, segment_set] : dr_box.get_net_detailed_result_map()) {
    for (Segment<LayerCoord>* segment : segment_set) {
      updateFixedRectToShadow(dr_box, ChangeType::kAdd, net_idx, segment);
    }
  }
  for (auto& [net_idx, segment_list] : dr_box.get_net_task_detailed_result_map()) {
    for (Segment<LayerCoord>& segment : segment_list) {
      updateRoutedRectToShadow(dr_box, ChangeType::kAdd, net_idx, segment);
    }
  }
  for (auto& [net_idx, patch_set] : dr_box.get_net_detailed_patch_map()) {
    for (EXTLayerRect* patch : patch_set) {
      updateFixedRectToShadow(dr_box, ChangeType::kAdd, net_idx, patch, true);
    }
  }
  for (auto& [net_idx, patch_list] : dr_box.get_net_task_detailed_patch_map()) {
    for (EXTLayerRect& patch : patch_list) {
      updateRoutedRectToShadow(dr_box, ChangeType::kAdd, net_idx, patch, true);
    }
  }
}

void DetailedRouter::exemptPinShape(DRModel& dr_model, DRBox& dr_box)
{
  int32_t detection_distance = RTDM.getDatabase().get_detection_distance();
  std::vector<DRNet>& dr_net_list = dr_model.get_dr_net_list();
  ScaleAxis& box_track_axis = dr_box.get_box_track_axis();
  std::vector<GridMap<DRNode>>& layer_node_map = dr_box.get_layer_node_map();

  for (auto& [dr_net_idx, access_point_set] : dr_box.get_net_access_point_map()) {
    std::map<int32_t, std::vector<EXTLayerRect*>> routing_obs_rect_map;
    for (auto& [routing_layer_idx, net_fixed_rect_map] : dr_box.get_type_layer_net_fixed_rect_map()[true]) {
      for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
        if (dr_net_idx == net_idx) {
          continue;
        }
        for (auto& fixed_rect : fixed_rect_set) {
          routing_obs_rect_map[routing_layer_idx].push_back(fixed_rect);
        }
      }
    }
    std::vector<DRPin>& dr_pin_list = dr_net_list[dr_net_idx].get_dr_pin_list();
    for (AccessPoint* access_point : access_point_set) {
      if (dr_pin_list[access_point->get_pin_idx()].get_is_core()) {
        if (!RTUTIL.existTrackGrid(access_point->get_real_coord(), box_track_axis)) {
          continue;
        }
        PlanarCoord grid_coord = RTUTIL.getTrackGrid(access_point->get_real_coord(), box_track_axis);
        DRNode& dr_node = layer_node_map[access_point->get_layer_idx()][grid_coord.get_x()][grid_coord.get_y()];
        for (auto& [orient, net_set] : dr_node.get_orient_fixed_rect_map()) {
          if (orient == Orientation::kAbove || orient == Orientation::kBelow) {
            net_set.erase(-1);
            DRNode* neighbor_node = dr_node.getNeighborNode(orient);
            if (neighbor_node == nullptr) {
              continue;
            }
            Orientation oppo_orientation = RTUTIL.getOppositeOrientation(orient);
            if (RTUTIL.exist(neighbor_node->get_orient_fixed_rect_map(), oppo_orientation)) {
              neighbor_node->get_orient_fixed_rect_map()[oppo_orientation].erase(-1);
            }
          }
        }
      } else {
        PlanarRect real_rect = RTUTIL.getEnlargedRect(access_point->get_real_coord(), detection_distance);
        if (!RTUTIL.existTrackGrid(real_rect, box_track_axis)) {
          continue;
        }
        PlanarRect grid_rect = RTUTIL.getTrackGrid(real_rect, box_track_axis);
        for (int32_t x = grid_rect.get_ll_x(); x <= grid_rect.get_ur_x(); x++) {
          for (int32_t y = grid_rect.get_ll_y(); y <= grid_rect.get_ur_y(); y++) {
            DRNode& dr_node = layer_node_map[access_point->get_layer_idx()][x][y];

            bool within_shape = false;
            for (EXTLayerRect* obs_rect : routing_obs_rect_map[dr_node.get_layer_idx()]) {
              if (RTUTIL.isInside(obs_rect->get_real_rect(), dr_node.get_planar_coord())) {
                within_shape = true;
                break;
              }
            }
            if (within_shape) {
              continue;
            }
            for (auto& [orient, net_set] : dr_node.get_orient_fixed_rect_map()) {
              if (orient == Orientation::kEast || orient == Orientation::kWest || orient == Orientation::kSouth || orient == Orientation::kNorth) {
                net_set.erase(-1);
              }
            }
          }
        }
      }
    }
  }
}

void DetailedRouter::routeDRBox(DRBox& dr_box)
{
  std::vector<DRTask*> routing_task_list = initTaskSchedule(dr_box);
  while (!routing_task_list.empty()) {
    for (DRTask* routing_task : routing_task_list) {
      updateGraph(dr_box, routing_task);
      routeDRTask(dr_box, routing_task);
      patchDRTask(dr_box, routing_task);
      routing_task->addRoutedTimes();
    }
    updateRouteViolationList(dr_box);
    updateBestResult(dr_box);
    updateTaskSchedule(dr_box, routing_task_list);
  }
}

std::vector<DRTask*> DetailedRouter::initTaskSchedule(DRBox& dr_box)
{
  bool initial_routing = dr_box.get_initial_routing();

  std::vector<DRTask*> routing_task_list;
  if (initial_routing) {
    for (DRTask* dr_task : dr_box.get_dr_task_list()) {
      routing_task_list.push_back(dr_task);
    }
  } else {
    updateTaskSchedule(dr_box, routing_task_list);
  }
  return routing_task_list;
}

void DetailedRouter::updateGraph(DRBox& dr_box, DRTask* dr_task)
{
  int32_t curr_net_idx = dr_task->get_net_idx();
  std::vector<Segment<LayerCoord>>& routing_segment_list = dr_box.get_net_task_detailed_result_map()[curr_net_idx];
  std::vector<EXTLayerRect>& routing_patch_list = dr_box.get_net_task_detailed_patch_map()[curr_net_idx];

  for (Segment<LayerCoord>& routing_segment : routing_segment_list) {
    updateRoutedRectToGraph(dr_box, ChangeType::kDel, curr_net_idx, routing_segment);
    updateRoutedRectToShadow(dr_box, ChangeType::kDel, curr_net_idx, routing_segment);
  }
  for (EXTLayerRect& routing_patch : routing_patch_list) {
    updateRoutedRectToGraph(dr_box, ChangeType::kDel, curr_net_idx, routing_patch, true);
    updateRoutedRectToShadow(dr_box, ChangeType::kDel, curr_net_idx, routing_patch, true);
  }
}

void DetailedRouter::routeDRTask(DRBox& dr_box, DRTask* dr_task)
{
  initSingleRouteTask(dr_box, dr_task);
  while (!isConnectedAllEnd(dr_box)) {
    routeSinglePath(dr_box);
    updatePathResult(dr_box);
    updateDirectionSet(dr_box);
    resetStartAndEnd(dr_box);
    resetSinglePath(dr_box);
  }
  updateTaskResult(dr_box);
  resetSingleRouteTask(dr_box);
}

void DetailedRouter::initSingleRouteTask(DRBox& dr_box, DRTask* dr_task)
{
  ScaleAxis& box_track_axis = dr_box.get_box_track_axis();
  std::vector<GridMap<DRNode>>& layer_node_map = dr_box.get_layer_node_map();

  // single task
  dr_box.set_curr_route_task(dr_task);
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
  OpenQueue<DRNode>& open_queue = dr_box.get_open_queue();
  DRNode* path_head_node = dr_box.get_path_head_node();

  for (auto& [orientation, neighbor_node] : path_head_node->get_neighbor_node_map()) {
    if (neighbor_node == nullptr) {
      continue;
    }
    if (neighbor_node->isClose()) {
      continue;
    }
    double known_cost = getKnownCost(dr_box, path_head_node, neighbor_node);
    if (neighbor_node->isOpen() && known_cost < neighbor_node->get_known_cost()) {
      neighbor_node->set_known_cost(known_cost);
      neighbor_node->set_parent_node(path_head_node);
      open_queue.push(neighbor_node);
    } else if (neighbor_node->isNone()) {
      neighbor_node->set_known_cost(known_cost);
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

  // 对于抵达的终点pin,只保留到达的node
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
    start_node_list_list.front().clear();
    start_node_list_list.front().push_back(path_node);
  }
  start_node_list_list.push_back(end_node_list_list[end_node_list_idx]);
  end_node_list_list.erase(end_node_list_list.begin() + end_node_list_idx);
}

void DetailedRouter::resetSinglePath(DRBox& dr_box)
{
  dr_box.get_open_queue().clear();
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

void DetailedRouter::updateTaskResult(DRBox& dr_box)
{
  int32_t curr_net_idx = dr_box.get_curr_route_task()->get_net_idx();
  std::vector<Segment<LayerCoord>>& routing_segment_list = dr_box.get_net_task_detailed_result_map()[curr_net_idx];
  routing_segment_list = getRoutingSegmentList(dr_box);
  // 新结果添加到graph
  for (Segment<LayerCoord>& routing_segment : routing_segment_list) {
    updateRoutedRectToGraph(dr_box, ChangeType::kAdd, curr_net_idx, routing_segment);
    updateRoutedRectToShadow(dr_box, ChangeType::kAdd, curr_net_idx, routing_segment);
  }
}

std::vector<Segment<LayerCoord>> DetailedRouter::getRoutingSegmentList(DRBox& dr_box)
{
  DRTask* curr_route_task = dr_box.get_curr_route_task();

  std::vector<LayerCoord> candidate_root_coord_list;
  std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
  std::vector<DRGroup>& dr_group_list = curr_route_task->get_dr_group_list();
  for (size_t i = 0; i < dr_group_list.size(); i++) {
    for (auto& [coord, _] : dr_group_list[i].get_coord_direction_map()) {
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

void DetailedRouter::resetSingleRouteTask(DRBox& dr_box)
{
  dr_box.set_curr_route_task(nullptr);
  dr_box.get_start_node_list_list().clear();
  dr_box.get_end_node_list_list().clear();
  dr_box.get_path_node_list().clear();
  for (DRNode* single_task_visited_node : dr_box.get_single_task_visited_node_list()) {
    single_task_visited_node->get_direction_set().clear();
  }
  dr_box.get_single_task_visited_node_list().clear();
  dr_box.get_routing_segment_list().clear();
}

// manager open list

void DetailedRouter::pushToOpenList(DRBox& dr_box, DRNode* curr_node)
{
  OpenQueue<DRNode>& open_queue = dr_box.get_open_queue();
  std::vector<DRNode*>& single_task_visited_node_list = dr_box.get_single_task_visited_node_list();
  std::vector<DRNode*>& single_path_visited_node_list = dr_box.get_single_path_visited_node_list();

  open_queue.push(curr_node);
  curr_node->set_state(DRNodeState::kOpen);
  single_task_visited_node_list.push_back(curr_node);
  single_path_visited_node_list.push_back(curr_node);
}

DRNode* DetailedRouter::popFromOpenList(DRBox& dr_box)
{
  DRNode* node = dr_box.get_open_queue().pop();
  if (node != nullptr) {
    node->set_state(DRNodeState::kClose);
  }
  return node;
}

// calculate known

double DetailedRouter::getKnownCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
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
  cost += getKnownWireCost(dr_box, start_node, end_node);
  cost += getKnownViaCost(dr_box, start_node, end_node);
  cost += getKnownBendCost(dr_box, start_node, end_node);
  cost += getKnownSelfCost(dr_box, start_node, end_node);
  return cost;
}

double DetailedRouter::getNodeCost(DRBox& dr_box, DRNode* curr_node, Orientation orientation)
{
  double fixed_rect_unit = dr_box.get_dr_iter_param()->get_fixed_rect_unit();
  double routed_rect_unit = dr_box.get_dr_iter_param()->get_routed_rect_unit();
  double violation_unit = dr_box.get_dr_iter_param()->get_violation_unit();

  int32_t net_idx = dr_box.get_curr_route_task()->get_net_idx();

  double cost = 0;
  cost += curr_node->getFixedRectCost(net_idx, orientation, fixed_rect_unit);
  cost += curr_node->getRoutedRectCost(net_idx, orientation, routed_rect_unit);
  cost += curr_node->getViolationCost(orientation, violation_unit);
  return cost;
}

double DetailedRouter::getKnownWireCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  double prefer_wire_unit = dr_box.get_dr_iter_param()->get_prefer_wire_unit();
  double non_prefer_wire_unit = dr_box.get_dr_iter_param()->get_non_prefer_wire_unit();

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

double DetailedRouter::getKnownViaCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
{
  double via_unit = dr_box.get_dr_iter_param()->get_via_unit();
  double via_cost = (via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
  return via_cost;
}

double DetailedRouter::getKnownBendCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
{
  double bend_unit = dr_box.get_dr_iter_param()->get_bend_unit();

  double bend_cost = 0;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    std::set<Direction> direction_set;
    direction_set.insert(start_node->get_direction_set().begin(), start_node->get_direction_set().end());
    if (start_node->get_parent_node() != nullptr) {
      direction_set.insert(RTUTIL.getDirection(*start_node->get_parent_node(), *start_node));
    }
    direction_set.insert(end_node->get_direction_set().begin(), end_node->get_direction_set().end());
    direction_set.insert(RTUTIL.getDirection(*start_node, *end_node));
    if (direction_set.size() == 2) {
      bend_cost += bend_unit;
    }
  }
  return bend_cost;
}

double DetailedRouter::getKnownSelfCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  double routed_rect_unit = dr_box.get_dr_iter_param()->get_routed_rect_unit();

  bool nonprefer_and_segment_end = false;
  if (start_node->get_layer_idx() == end_node->get_layer_idx()) {
    RoutingLayer& routing_layer = routing_layer_list[start_node->get_layer_idx()];
    if (routing_layer.get_prefer_direction() != RTUTIL.getDirection(*start_node, *end_node)) {
      for (std::vector<DRNode*>& end_node_list : dr_box.get_end_node_list_list()) {
        if (RTUTIL.exist(end_node_list, end_node)) {
          nonprefer_and_segment_end = true;
          break;
        }
      }
      if (!nonprefer_and_segment_end) {
        return 0;
      }
    }
  }
  RoutingLayer& routing_layer = routing_layer_list[start_node->get_layer_idx()];
  int32_t wire_width = routing_layer.get_min_width();
  int32_t target_wire_length = std::max(routing_layer.getPRLSpacing(wire_width), routing_layer.get_notch_spacing()) + wire_width;

  int32_t non_prefer_wire_length = 0;
  {
    DRNode* curr_node = start_node;
    DRNode* pre_node = curr_node->get_parent_node();
    while (pre_node != nullptr) {
      if (pre_node->get_layer_idx() != curr_node->get_layer_idx()) {
        break;
      }
      if (routing_layer.get_prefer_direction() == RTUTIL.getDirection(*pre_node, *curr_node)) {
        break;
      }
      non_prefer_wire_length += RTUTIL.getManhattanDistance(pre_node->get_planar_coord(), curr_node->get_planar_coord());
      curr_node = pre_node;
      pre_node = curr_node->get_parent_node();
    }
    if (nonprefer_and_segment_end) {
      non_prefer_wire_length += RTUTIL.getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());
    }
  }
  double self_cost = 0;
  if (0 < non_prefer_wire_length && non_prefer_wire_length < target_wire_length) {
    self_cost += routed_rect_unit;
  }
  return self_cost;
}

// calculate estimate

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
  estimate_cost += getEstimateViaCost(dr_box, start_node, end_node);
  return estimate_cost;
}

double DetailedRouter::getEstimateWireCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
{
  double prefer_wire_unit = dr_box.get_dr_iter_param()->get_prefer_wire_unit();
  double non_prefer_wire_unit = dr_box.get_dr_iter_param()->get_non_prefer_wire_unit();

  double wire_cost = 0;
  wire_cost += RTUTIL.getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());
  wire_cost *= std::min(prefer_wire_unit, non_prefer_wire_unit);
  return wire_cost;
}

double DetailedRouter::getEstimateViaCost(DRBox& dr_box, DRNode* start_node, DRNode* end_node)
{
  double via_unit = dr_box.get_dr_iter_param()->get_via_unit();
  double via_cost = (via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
  return via_cost;
}

void DetailedRouter::patchDRTask(DRBox& dr_box, DRTask* dr_task)
{
  initSinglePatchTask(dr_box, dr_task);
  while (searchViolation(dr_box)) {
    addViolationToShadow(dr_box);
    patchSingleViolation(dr_box);
    resetSingleViolation(dr_box);
    clearViolationShadow(dr_box);
  }
  updateTaskPatch(dr_box);
  resetSinglePatchTask(dr_box);
}

void DetailedRouter::initSinglePatchTask(DRBox& dr_box, DRTask* dr_task)
{
  // single task
  dr_box.set_curr_patch_task(dr_task);
  dr_box.get_routing_patch_list().clear();
  dr_box.set_patch_violation_list(getPatchViolationList(dr_box, {ViolationType::kMinimumArea}, {}));
  dr_box.get_tried_fix_violation_set().clear();
}

std::vector<Violation> DetailedRouter::getPatchViolationList(DRBox& dr_box, const std::set<ViolationType>& check_type_set,
                                                             const std::vector<LayerRect>& check_region_list)
{
  std::string top_name = RTUTIL.getString("dr_box_", dr_box.get_dr_box_id().get_x(), "_", dr_box.get_dr_box_id().get_y());
  std::vector<std::pair<EXTLayerRect*, bool>> env_shape_list;
  std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>> net_pin_shape_map;
  for (auto& [is_routing, layer_net_fixed_rect_map] : dr_box.get_type_layer_net_fixed_rect_map()) {
    for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
      for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
        if (net_idx == -1) {
          for (auto& fixed_rect : fixed_rect_set) {
            env_shape_list.emplace_back(fixed_rect, is_routing);
          }
        } else {
          for (auto& fixed_rect : fixed_rect_set) {
            net_pin_shape_map[net_idx].emplace_back(fixed_rect, is_routing);
          }
        }
      }
    }
  }
  std::map<int32_t, std::vector<Segment<LayerCoord>*>> net_result_map;
  for (auto& [net_idx, segment_set] : dr_box.get_net_detailed_result_map()) {
    for (Segment<LayerCoord>* segment : segment_set) {
      net_result_map[net_idx].push_back(segment);
    }
  }
  for (auto& [net_idx, segment_list] : dr_box.get_net_task_detailed_result_map()) {
    for (Segment<LayerCoord>& segment : segment_list) {
      net_result_map[net_idx].emplace_back(&segment);
    }
  }
  std::map<int32_t, std::vector<EXTLayerRect*>> net_patch_map;
  for (auto& [net_idx, patch_set] : dr_box.get_net_detailed_patch_map()) {
    for (EXTLayerRect* patch : patch_set) {
      net_patch_map[net_idx].push_back(patch);
    }
  }
  for (auto& [net_idx, patch_list] : dr_box.get_net_task_detailed_patch_map()) {
    if (net_idx == dr_box.get_curr_patch_task()->get_net_idx()) {
      for (EXTLayerRect& patch : dr_box.get_routing_patch_list()) {
        net_patch_map[net_idx].emplace_back(&patch);
      }
    } else {
      for (EXTLayerRect& patch : patch_list) {
        net_patch_map[net_idx].emplace_back(&patch);
      }
    }
  }
  std::set<int32_t> need_checked_net_set;
  for (DRTask* dr_task : dr_box.get_dr_task_list()) {
    need_checked_net_set.insert(dr_task->get_net_idx());
  }

  DETask de_task;
  de_task.set_proc_type(DEProcType::kGet);
  de_task.set_net_type(DENetType::kPatchHybrid);
  de_task.set_top_name(top_name);
  de_task.set_env_shape_list(env_shape_list);
  de_task.set_net_pin_shape_map(net_pin_shape_map);
  de_task.set_net_result_map(net_result_map);
  de_task.set_net_patch_map(net_patch_map);
  de_task.set_need_checked_net_set(need_checked_net_set);
  de_task.set_check_type_set(check_type_set);
  de_task.set_check_region_list(check_region_list);
  return RTDE.getViolationList(de_task);
}

bool DetailedRouter::searchViolation(DRBox& dr_box)
{
  for (Violation& violation : dr_box.get_patch_violation_list()) {
    if (!isValidPatchViolation(dr_box, violation)) {
      continue;
    }
    if (RTUTIL.exist(dr_box.get_tried_fix_violation_set(), violation)) {
      continue;
    }
    int32_t net_idx = *violation.get_violation_net_set().begin();
    if (dr_box.get_curr_patch_task()->get_net_idx() != net_idx) {
      continue;
    }
    if (getViolationOverlapRect(dr_box, violation).empty()) {
      continue;
    }
    dr_box.set_curr_patch_violation(violation);
    return true;
  }
  return false;
}

bool DetailedRouter::isValidPatchViolation(DRBox& dr_box, Violation& violation)
{
  PlanarRect& box_real_rect = dr_box.get_box_rect().get_real_rect();

  bool is_valid = true;
  if (!RTUTIL.isOpenOverlap(box_real_rect, violation.get_violation_shape().get_real_rect())) {
    is_valid = false;
  }
  if (violation.get_violation_type() != ViolationType::kMinimumArea) {
    is_valid = false;
  }
  return is_valid;
}

std::vector<PlanarRect> DetailedRouter::getViolationOverlapRect(DRBox& dr_box, Violation& violation)
{
  int32_t curr_net_idx = dr_box.get_curr_patch_task()->get_net_idx();
  EXTLayerRect& violation_shape = violation.get_violation_shape();
  PlanarRect violation_real_rect = violation_shape.get_real_rect();
  int32_t violation_layer_idx = violation_shape.get_layer_idx();

  GTLPolySetInt gtl_poly_set;
  {
    for (EXTLayerRect* fixed_rect : dr_box.get_type_layer_net_fixed_rect_map()[true][violation_layer_idx][curr_net_idx]) {
      if (RTUTIL.isClosedOverlap(violation_real_rect, fixed_rect->get_real_rect())) {
        gtl_poly_set += RTUTIL.convertToGTLRectInt(fixed_rect->get_real_rect());
      }
    }
    for (Segment<LayerCoord>* segment : dr_box.get_net_detailed_result_map()[curr_net_idx]) {
      for (NetShape& net_shape : RTDM.getNetDetailedShapeList(curr_net_idx, *segment)) {
        if (!net_shape.get_is_routing()) {
          continue;
        }
        if (violation_layer_idx == net_shape.get_layer_idx() && RTUTIL.isClosedOverlap(violation_real_rect, net_shape.get_rect())) {
          gtl_poly_set += RTUTIL.convertToGTLRectInt(net_shape.get_rect());
        }
      }
    }
    for (Segment<LayerCoord>& segment : dr_box.get_net_task_detailed_result_map()[curr_net_idx]) {
      for (NetShape& net_shape : RTDM.getNetDetailedShapeList(curr_net_idx, segment)) {
        if (!net_shape.get_is_routing()) {
          continue;
        }
        if (violation_layer_idx == net_shape.get_layer_idx() && RTUTIL.isClosedOverlap(violation_real_rect, net_shape.get_rect())) {
          gtl_poly_set += RTUTIL.convertToGTLRectInt(net_shape.get_rect());
        }
      }
    }
    for (EXTLayerRect* patch : dr_box.get_net_detailed_patch_map()[curr_net_idx]) {
      if (violation_layer_idx == patch->get_layer_idx() && RTUTIL.isClosedOverlap(violation_real_rect, patch->get_real_rect())) {
        gtl_poly_set += RTUTIL.convertToGTLRectInt(patch->get_real_rect());
      }
    }
  }
  std::vector<GTLPolyInt> gtl_poly_list;
  gtl_poly_set.get_polygons(gtl_poly_list);
  if (gtl_poly_list.empty()) {
    return {};
  }
  GTLPolyInt best_gtl_poly = gtl_poly_list.front();
  {
    int32_t max_overlap_area = INT32_MIN;
    for (GTLPolyInt& gtl_poly : gtl_poly_list) {
      int32_t overlap_area = static_cast<int32_t>(gtl::area(gtl_poly & RTUTIL.convertToGTLRectInt(violation_real_rect)));
      if (max_overlap_area < overlap_area) {
        best_gtl_poly = gtl_poly;
      }
    }
  }
  std::vector<GTLRectInt> gtl_rect_list;
  gtl::get_max_rectangles(gtl_rect_list, best_gtl_poly);
  std::vector<PlanarRect> overlap_rect_list;
  for (GTLRectInt& gtl_rect : gtl_rect_list) {
    overlap_rect_list.push_back(RTUTIL.convertToPlanarRect(gtl_rect));
  }
  return overlap_rect_list;
}

void DetailedRouter::addViolationToShadow(DRBox& dr_box)
{
  for (Violation& patch_violation : dr_box.get_patch_violation_list()) {
    if (patch_violation.get_violation_type() == ViolationType::kMinimumArea) {
      continue;
    }
    addPatchViolationToShadow(dr_box, patch_violation);
  }
}

void DetailedRouter::patchSingleViolation(DRBox& dr_box)
{
  std::vector<EXTLayerRect>& routing_patch_list = dr_box.get_routing_patch_list();
  std::set<Violation, CmpViolation>& tried_fix_violation_set = dr_box.get_tried_fix_violation_set();
  LayerRect violation_rect = dr_box.get_curr_patch_violation().get_violation_shape().getRealLayerRect();

  std::vector<DRPatch> dr_patch_list = getCandidatePatchList(dr_box);
  if (dr_patch_list.size() == 1) {
    routing_patch_list.push_back(dr_patch_list.front().get_patch());
  } else if (dr_patch_list.size() >= 2) {
    std::vector<Violation> origin_patch_violation_list = getPatchViolationList(dr_box, {}, {violation_rect});

    bool curr_is_solved = false;
    for (DRPatch& dr_patch : dr_patch_list) {
      std::vector<Violation> curr_patch_violation_list;
      {
        routing_patch_list.push_back(dr_patch.get_patch());
        curr_patch_violation_list = getPatchViolationList(dr_box, {}, {violation_rect});
        routing_patch_list.pop_back();
      }
      curr_is_solved = getSolvedStatus(dr_box, origin_patch_violation_list, curr_patch_violation_list);
      if (curr_is_solved) {
        routing_patch_list.push_back(dr_patch.get_patch());
        break;
      }
    }
    if (!curr_is_solved) {
      routing_patch_list.push_back(dr_patch_list.front().get_patch());
    }
  }
  tried_fix_violation_set.insert(dr_box.get_curr_patch_violation());
}

std::vector<DRPatch> DetailedRouter::getCandidatePatchList(DRBox& dr_box)
{
  int32_t manufacture_grid = RTDM.getDatabase().get_manufacture_grid();
  Die& die = RTDM.getDatabase().get_die();
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  int32_t max_candidate_patch_num = dr_box.get_dr_iter_param()->get_max_candidate_patch_num();

  int32_t curr_net_idx = dr_box.get_curr_patch_task()->get_net_idx();
  Violation& curr_patch_violation = dr_box.get_curr_patch_violation();
  int32_t violation_layer_idx = curr_patch_violation.get_violation_shape().get_layer_idx();

  RoutingLayer& routing_layer = routing_layer_list[violation_layer_idx];
  Direction layer_direction = routing_layer.get_prefer_direction();
  int32_t min_area = routing_layer.get_min_area();
  int32_t wire_width = routing_layer.get_min_width();

  GTLPolyInt gtl_poly;
  {
    GTLPolySetInt gtl_poly_set;
    for (PlanarRect& overlap_rect : getViolationOverlapRect(dr_box, curr_patch_violation)) {
      gtl_poly_set += RTUTIL.convertToGTLRectInt(overlap_rect);
    }
    std::vector<GTLPolyInt> gtl_poly_list;
    gtl_poly_set.get_polygons(gtl_poly_list);
    gtl_poly = gtl_poly_list.front();
    if (min_area <= static_cast<int32_t>(gtl::area(gtl_poly))) {
      return {};
    }
  }
  PlanarRect h_cutting_rect;
  {
    std::vector<GTLRectInt> gtl_rect_list;
    gtl::get_rectangles(gtl_rect_list, gtl_poly, gtl::HORIZONTAL);
    GTLRectInt best_gtl_rect;
    int32_t max_x_span = 0;
    for (GTLRectInt& gtl_rect : gtl_rect_list) {
      int32_t curr_x_span = std::abs(gtl::xl(gtl_rect) - gtl::xh(gtl_rect));
      if (max_x_span <= curr_x_span) {
        max_x_span = curr_x_span;
        best_gtl_rect = gtl_rect;
      }
    }
    h_cutting_rect = RTUTIL.convertToPlanarRect(best_gtl_rect);
  }
  PlanarRect v_cutting_rect;
  {
    std::vector<GTLRectInt> gtl_rect_list;
    gtl::get_rectangles(gtl_rect_list, gtl_poly, gtl::VERTICAL);
    GTLRectInt best_gtl_rect;
    int32_t max_y_span = 0;
    for (GTLRectInt& gtl_rect : gtl_rect_list) {
      int32_t curr_y_span = std::abs(gtl::yl(gtl_rect) - gtl::yh(gtl_rect));
      if (max_y_span <= curr_y_span) {
        max_y_span = curr_y_span;
        best_gtl_rect = gtl_rect;
      }
    }
    v_cutting_rect = RTUTIL.convertToPlanarRect(best_gtl_rect);
  }
  std::vector<DRPatch> dr_patch_list;
  {
    int32_t h_wire_length = static_cast<int32_t>(std::ceil((min_area - v_cutting_rect.getArea()) / wire_width) + v_cutting_rect.getXSpan());
    while (h_wire_length % manufacture_grid != 0) {
      h_wire_length++;
    }
    for (int32_t y : {h_cutting_rect.get_ll_y(), v_cutting_rect.get_ll_y(), v_cutting_rect.get_ur_y() - wire_width}) {
      for (int32_t x = v_cutting_rect.get_ur_x() - h_wire_length; x <= v_cutting_rect.get_ll_x(); x += manufacture_grid) {
        PlanarRect h_real_rect = RTUTIL.getEnlargedRect(PlanarCoord(x, y), 0, 0, h_wire_length, wire_width);
        if (!RTUTIL.isInside(die.get_real_rect(), h_real_rect)) {
          continue;
        }
        dr_patch_list.emplace_back(h_real_rect, violation_layer_idx);
      }
    }
    int32_t v_wire_length = static_cast<int32_t>(std::ceil((min_area - h_cutting_rect.getArea()) / wire_width) + h_cutting_rect.getYSpan());
    while (v_wire_length % manufacture_grid != 0) {
      v_wire_length++;
    }
    for (int32_t x : {v_cutting_rect.get_ll_x(), h_cutting_rect.get_ll_x(), h_cutting_rect.get_ur_x() - wire_width}) {
      for (int32_t y = h_cutting_rect.get_ur_y() - v_wire_length; y <= h_cutting_rect.get_ll_y(); y += manufacture_grid) {
        PlanarRect v_real_rect = RTUTIL.getEnlargedRect(PlanarCoord(x, y), 0, 0, wire_width, v_wire_length);
        if (!RTUTIL.isInside(die.get_real_rect(), v_real_rect)) {
          continue;
        }
        dr_patch_list.emplace_back(v_real_rect, violation_layer_idx);
      }
    }
    for (DRPatch& dr_patch : dr_patch_list) {
      EXTLayerRect& patch = dr_patch.get_patch();
      patch.set_grid_rect(RTUTIL.getClosedGCellGridRect(patch.get_real_rect(), gcell_axis));
      dr_patch.set_fixed_rect_cost(getFixedRectCost(dr_box, curr_net_idx, patch));
      dr_patch.set_routed_rect_cost(getRoutedRectCost(dr_box, curr_net_idx, patch));
      dr_patch.set_violation_cost(getViolationCost(dr_box, curr_net_idx, patch));
      dr_patch.set_direction(patch.get_real_rect().getRectDirection(layer_direction));
      dr_patch.set_overlap_area(static_cast<int32_t>(gtl::area(gtl_poly & RTUTIL.convertToGTLRectInt(patch.get_real_rect()))));
    }
    std::sort(dr_patch_list.begin(), dr_patch_list.end(), [&layer_direction](DRPatch& a, DRPatch& b) { return CmpDRPatch()(a, b, layer_direction); });
    if (dr_patch_list.empty()) {
      RTLOG.error(Loc::current(), "The dr_patch_list is empty!");
    }
  }
  std::vector<DRPatch> candidate_patch_list;
  {
    std::vector<DRPatch> dr_patch_list_temp;
    for (DRPatch& dr_patch : dr_patch_list) {
      if (dr_patch.getTotalCost() > 0) {
        continue;
      }
      dr_patch_list_temp.push_back(dr_patch);
    }
    if (dr_patch_list_temp.empty()) {
      dr_patch_list_temp.push_back(dr_patch_list.front());
    }
    int32_t patch_size = static_cast<int32_t>(dr_patch_list_temp.size());
    if (patch_size <= max_candidate_patch_num) {
      candidate_patch_list = dr_patch_list_temp;
    } else {
      int32_t candidate_step = (patch_size - 2) / (max_candidate_patch_num - 2);
      candidate_patch_list.push_back(dr_patch_list_temp.front());
      for (int32_t i = candidate_step; i < (patch_size - candidate_step); i += candidate_step) {
        candidate_patch_list.push_back(dr_patch_list_temp[i]);
      }
      candidate_patch_list.push_back(dr_patch_list_temp.back());
    }
  }
  return candidate_patch_list;
}

bool DetailedRouter::getSolvedStatus(DRBox& dr_box, std::vector<Violation>& origin_patch_violation_list, std::vector<Violation>& curr_patch_violation_list)
{
  std::map<ViolationType, std::pair<int32_t, int32_t>> env_type_origin_curr_map;
  std::map<ViolationType, std::pair<int32_t, int32_t>> valid_type_origin_curr_map;
  std::map<ViolationType, std::pair<int32_t, int32_t>> within_net_map;
  for (Violation& origin_violation : origin_patch_violation_list) {
    if (!isValidPatchViolation(dr_box, origin_violation)) {
      env_type_origin_curr_map[origin_violation.get_violation_type()].first++;
    } else {
      valid_type_origin_curr_map[origin_violation.get_violation_type()].first++;
    }
    if (origin_violation.get_violation_net_set().size() > 1) {
      within_net_map[origin_violation.get_violation_type()].first++;
    }
  }
  for (Violation& curr_violation : curr_patch_violation_list) {
    if (!isValidPatchViolation(dr_box, curr_violation)) {
      env_type_origin_curr_map[curr_violation.get_violation_type()].second++;
    } else {
      valid_type_origin_curr_map[curr_violation.get_violation_type()].second++;
    }
    if (curr_violation.get_violation_net_set().size() > 1) {
      within_net_map[curr_violation.get_violation_type()].second++;
    }
  }
  bool curr_is_solved = true;
  for (auto& [violation_type, origin_curr] : env_type_origin_curr_map) {
    if (!curr_is_solved) {
      break;
    }
    curr_is_solved = origin_curr.second <= origin_curr.first;
  }
  for (auto& [violation_type, origin_curr] : valid_type_origin_curr_map) {
    if (!curr_is_solved) {
      break;
    }
    curr_is_solved = origin_curr.second < origin_curr.first;
  }
  for (auto& [violation_type, origin_curr] : within_net_map) {
    if (!curr_is_solved) {
      break;
    }
    curr_is_solved = origin_curr.second <= origin_curr.first;
  }
  return curr_is_solved;
}

void DetailedRouter::resetSingleViolation(DRBox& dr_box)
{
  dr_box.set_curr_patch_violation(Violation());
}

void DetailedRouter::clearViolationShadow(DRBox& dr_box)
{
  for (DRShadow& dr_shadow : dr_box.get_layer_shadow_map()) {
    dr_shadow.get_violation_set().clear();
  }
}

void DetailedRouter::updateTaskPatch(DRBox& dr_box)
{
  int32_t curr_net_idx = dr_box.get_curr_patch_task()->get_net_idx();
  std::vector<EXTLayerRect>& routing_patch_list = dr_box.get_net_task_detailed_patch_map()[curr_net_idx];
  routing_patch_list = dr_box.get_routing_patch_list();
  // 新结果添加到graph
  for (EXTLayerRect& routing_patch : routing_patch_list) {
    updateRoutedRectToGraph(dr_box, ChangeType::kAdd, curr_net_idx, routing_patch, true);
    updateRoutedRectToShadow(dr_box, ChangeType::kAdd, curr_net_idx, routing_patch, true);
  }
}

void DetailedRouter::resetSinglePatchTask(DRBox& dr_box)
{
  dr_box.set_curr_patch_task(nullptr);
  dr_box.get_routing_patch_list().clear();
  dr_box.get_patch_violation_list().clear();
  dr_box.get_tried_fix_violation_set().clear();
}

void DetailedRouter::updateRouteViolationList(DRBox& dr_box)
{
  dr_box.get_route_violation_list().clear();
  for (Violation new_violation : getRouteViolationList(dr_box)) {
    dr_box.get_route_violation_list().push_back(new_violation);
  }
  // 新结果添加到graph
  for (Violation& violation : dr_box.get_route_violation_list()) {
    addRouteViolationToGraph(dr_box, violation);
  }
}

std::vector<Violation> DetailedRouter::getRouteViolationList(DRBox& dr_box)
{
  std::string top_name = RTUTIL.getString("dr_box_", dr_box.get_dr_box_id().get_x(), "_", dr_box.get_dr_box_id().get_y());
  std::vector<std::pair<EXTLayerRect*, bool>> env_shape_list;
  std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>> net_pin_shape_map;
  for (auto& [is_routing, layer_net_fixed_rect_map] : dr_box.get_type_layer_net_fixed_rect_map()) {
    for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
      for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
        if (net_idx == -1) {
          for (auto& fixed_rect : fixed_rect_set) {
            env_shape_list.emplace_back(fixed_rect, is_routing);
          }
        } else {
          for (auto& fixed_rect : fixed_rect_set) {
            net_pin_shape_map[net_idx].emplace_back(fixed_rect, is_routing);
          }
        }
      }
    }
  }
  std::map<int32_t, std::vector<Segment<LayerCoord>*>> net_result_map;
  for (auto& [net_idx, segment_set] : dr_box.get_net_detailed_result_map()) {
    for (Segment<LayerCoord>* segment : segment_set) {
      net_result_map[net_idx].push_back(segment);
    }
  }
  for (auto& [net_idx, segment_list] : dr_box.get_net_task_detailed_result_map()) {
    for (Segment<LayerCoord>& segment : segment_list) {
      net_result_map[net_idx].emplace_back(&segment);
    }
  }
  std::map<int32_t, std::vector<EXTLayerRect*>> net_patch_map;
  for (auto& [net_idx, patch_set] : dr_box.get_net_detailed_patch_map()) {
    for (EXTLayerRect* patch : patch_set) {
      net_patch_map[net_idx].push_back(patch);
    }
  }
  for (auto& [net_idx, patch_list] : dr_box.get_net_task_detailed_patch_map()) {
    for (EXTLayerRect& patch : patch_list) {
      net_patch_map[net_idx].emplace_back(&patch);
    }
  }
  std::set<int32_t> need_checked_net_set;
  for (DRTask* dr_task : dr_box.get_dr_task_list()) {
    need_checked_net_set.insert(dr_task->get_net_idx());
  }

  DETask de_task;
  de_task.set_proc_type(DEProcType::kGet);
  de_task.set_net_type(DENetType::kRouteHybrid);
  de_task.set_top_name(top_name);
  de_task.set_env_shape_list(env_shape_list);
  de_task.set_net_pin_shape_map(net_pin_shape_map);
  de_task.set_net_result_map(net_result_map);
  de_task.set_net_patch_map(net_patch_map);
  de_task.set_need_checked_net_set(need_checked_net_set);
  return RTDE.getViolationList(de_task);
}

void DetailedRouter::updateBestResult(DRBox& dr_box)
{
  std::map<int32_t, std::vector<Segment<LayerCoord>>>& best_net_task_detailed_result_map = dr_box.get_best_net_task_detailed_result_map();
  std::map<int32_t, std::vector<EXTLayerRect>>& best_net_task_detailed_patch_map = dr_box.get_best_net_task_detailed_patch_map();
  std::vector<Violation>& best_route_violation_list = dr_box.get_best_route_violation_list();

  int32_t curr_violation_num = static_cast<int32_t>(dr_box.get_route_violation_list().size());
  if (!best_net_task_detailed_result_map.empty()) {
    if (static_cast<int32_t>(best_route_violation_list.size()) < curr_violation_num) {
      return;
    }
  }
  best_net_task_detailed_result_map = dr_box.get_net_task_detailed_result_map();
  best_net_task_detailed_patch_map = dr_box.get_net_task_detailed_patch_map();
  best_route_violation_list = dr_box.get_route_violation_list();
}

void DetailedRouter::updateTaskSchedule(DRBox& dr_box, std::vector<DRTask*>& routing_task_list)
{
  int32_t max_routed_times = dr_box.get_dr_iter_param()->get_max_routed_times();

  std::set<DRTask*> visited_routing_task_set;
  std::vector<DRTask*> new_routing_task_list;
  for (Violation& violation : dr_box.get_route_violation_list()) {
    EXTLayerRect& violation_shape = violation.get_violation_shape();
    if (!RTUTIL.isInside(dr_box.get_box_rect().get_real_rect(), violation_shape.get_real_rect())) {
      continue;
    }
    for (DRTask* dr_task : dr_box.get_dr_task_list()) {
      if (!RTUTIL.exist(violation.get_violation_net_set(), dr_task->get_net_idx())) {
        continue;
      }
      if (dr_task->get_routed_times() < max_routed_times && !RTUTIL.exist(visited_routing_task_set, dr_task)) {
        visited_routing_task_set.insert(dr_task);
        new_routing_task_list.push_back(dr_task);
      }
      break;
    }
  }
  routing_task_list = new_routing_task_list;

  std::vector<DRTask*> new_dr_task_list;
  for (DRTask* dr_task : dr_box.get_dr_task_list()) {
    if (!RTUTIL.exist(visited_routing_task_set, dr_task)) {
      new_dr_task_list.push_back(dr_task);
    }
  }
  for (DRTask* routing_task : routing_task_list) {
    new_dr_task_list.push_back(routing_task);
  }
  dr_box.set_dr_task_list(new_dr_task_list);
}

void DetailedRouter::selectBestResult(DRBox& dr_box)
{
  updateBestResult(dr_box);
  uploadBestResult(dr_box);
}

void DetailedRouter::uploadBestResult(DRBox& dr_box)
{
  for (auto& [net_idx, segment_list] : dr_box.get_best_net_task_detailed_result_map()) {
    for (Segment<LayerCoord>& segment : segment_list) {
      RTDM.updateNetDetailedResultToGCellMap(ChangeType::kAdd, net_idx, new Segment<LayerCoord>(segment));
    }
  }
  for (auto& [net_idx, patch_list] : dr_box.get_best_net_task_detailed_patch_map()) {
    for (EXTLayerRect& patch : patch_list) {
      RTDM.updateNetDetailedPatchToGCellMap(ChangeType::kAdd, net_idx, new EXTLayerRect(patch));
    }
  }
  for (Violation& violation : dr_box.get_best_route_violation_list()) {
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

int32_t DetailedRouter::getRouteViolationNum(DRModel& dr_model)
{
  Die& die = RTDM.getDatabase().get_die();

  return static_cast<int32_t>(RTDM.getViolationSet(die).size());
}

void DetailedRouter::uploadNetResult(DRModel& dr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();
  std::vector<DRNet>& dr_net_list = dr_model.get_dr_net_list();

  // detailed result
  {
    std::vector<std::set<Segment<LayerCoord>*>> detailed_result_list;
    detailed_result_list.resize(dr_net_list.size());
    for (auto& [net_idx, segment_set] : RTDM.getNetDetailedResultMap(die)) {
      detailed_result_list[net_idx] = segment_set;
    }
    std::vector<std::set<Segment<LayerCoord>*>> new_detailed_result_list;
    new_detailed_result_list.resize(dr_net_list.size());
#pragma omp parallel for
    for (int32_t net_idx = 0; net_idx < static_cast<int32_t>(detailed_result_list.size()); net_idx++) {
      std::vector<Segment<LayerCoord>> routing_segment_list;
      for (Segment<LayerCoord>* segment : detailed_result_list[net_idx]) {
        routing_segment_list.emplace_back(segment->get_first(), segment->get_second());
      }
      std::vector<LayerCoord> candidate_root_coord_list;
      std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
      std::vector<DRPin>& dr_pin_list = dr_net_list[net_idx].get_dr_pin_list();
      for (size_t i = 0; i < dr_pin_list.size(); i++) {
        LayerCoord coord = dr_pin_list[i].get_access_point().getRealLayerCoord();
        candidate_root_coord_list.push_back(coord);
        key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
      }
      MTree<LayerCoord> coord_tree = RTUTIL.getTreeByFullFlow(candidate_root_coord_list, routing_segment_list, key_coord_pin_map);
      for (Segment<TNode<LayerCoord>*>& coord_segment : RTUTIL.getSegListByTree(coord_tree)) {
        new_detailed_result_list[net_idx].insert(new Segment<LayerCoord>(coord_segment.get_first()->value(), coord_segment.get_second()->value()));
      }
    }
    for (int32_t net_idx = 0; net_idx < static_cast<int32_t>(detailed_result_list.size()); net_idx++) {
      for (Segment<LayerCoord>* segment : detailed_result_list[net_idx]) {
        RTDM.updateNetDetailedResultToGCellMap(ChangeType::kDel, net_idx, segment);
      }
    }
    for (int32_t net_idx = 0; net_idx < static_cast<int32_t>(new_detailed_result_list.size()); net_idx++) {
      for (Segment<LayerCoord>* segment : new_detailed_result_list[net_idx]) {
        RTDM.updateNetDetailedResultToGCellMap(ChangeType::kAdd, net_idx, segment);
      }
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void DetailedRouter::uploadNetPatch(DRModel& dr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();

  std::map<int32_t, std::set<EXTLayerRect*>> net_detailed_patch_map = RTDM.getNetDetailedPatchMap(die);
  std::map<int32_t, std::set<Segment<LayerCoord>*>> net_detailed_result_map = RTDM.getNetDetailedResultMap(die);
  for (auto& [net_idx, patch_set] : net_detailed_patch_map) {
    std::map<int32_t, std::vector<PlanarRect>> layer_rect_map;
    for (Segment<LayerCoord>* segment : net_detailed_result_map[net_idx]) {
      for (NetShape& net_shape : RTDM.getNetDetailedShapeList(net_idx, *segment)) {
        if (!net_shape.get_is_routing()) {
          continue;
        }
        layer_rect_map[net_shape.get_layer_idx()].push_back(net_shape.get_rect());
      }
    }
    std::vector<EXTLayerRect*> del_patch_list;
    for (EXTLayerRect* patch : patch_set) {
      bool is_used = false;
      for (PlanarRect& rect : layer_rect_map[patch->get_layer_idx()]) {
        if (RTUTIL.isClosedOverlap(patch->get_real_rect(), rect)) {
          is_used = true;
          break;
        }
      }
      if (!is_used) {
        del_patch_list.push_back(patch);
      }
    }
    for (EXTLayerRect* del_patch : del_patch_list) {
      RTDM.updateNetDetailedPatchToGCellMap(ChangeType::kDel, net_idx, del_patch);
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void DetailedRouter::uploadViolation(DRModel& dr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();

  for (Violation* violation : RTDM.getViolationSet(die)) {
    RTDM.updateViolationToGCellMap(ChangeType::kDel, violation);
  }
  for (Violation violation : getRouteViolationList(dr_model)) {
    RTDM.updateViolationToGCellMap(ChangeType::kAdd, new Violation(violation));
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

std::vector<Violation> DetailedRouter::getRouteViolationList(DRModel& dr_model)
{
  Die& die = RTDM.getDatabase().get_die();

  DETask de_task;
  {
    std::string top_name = RTUTIL.getString("dr_model");
    std::vector<std::pair<EXTLayerRect*, bool>> env_shape_list;
    std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>> net_pin_shape_map;
    for (auto& [is_routing, layer_net_fixed_rect_map] : RTDM.getTypeLayerNetFixedRectMap(die)) {
      for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
        for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
          if (net_idx == -1) {
            for (auto& fixed_rect : fixed_rect_set) {
              env_shape_list.emplace_back(fixed_rect, is_routing);
            }
          } else {
            for (auto& fixed_rect : fixed_rect_set) {
              net_pin_shape_map[net_idx].emplace_back(fixed_rect, is_routing);
            }
          }
        }
      }
    }
    std::map<int32_t, std::vector<Segment<LayerCoord>*>> net_result_map;
    for (auto& [net_idx, segment_set] : RTDM.getNetDetailedResultMap(die)) {
      for (Segment<LayerCoord>* segment : segment_set) {
        net_result_map[net_idx].push_back(segment);
      }
    }
    std::map<int32_t, std::vector<EXTLayerRect*>> net_patch_map;
    for (auto& [net_idx, patch_set] : RTDM.getNetDetailedPatchMap(die)) {
      for (EXTLayerRect* patch : patch_set) {
        net_patch_map[net_idx].emplace_back(patch);
      }
    }
    std::set<int32_t> need_checked_net_set;
    for (DRNet& dr_net : dr_model.get_dr_net_list()) {
      need_checked_net_set.insert(dr_net.get_net_idx());
    }

    de_task.set_proc_type(DEProcType::kGet);
    de_task.set_net_type(DENetType::kRouteHybrid);
    de_task.set_top_name(top_name);
    de_task.set_env_shape_list(env_shape_list);
    de_task.set_net_pin_shape_map(net_pin_shape_map);
    de_task.set_net_result_map(net_result_map);
    de_task.set_net_patch_map(net_patch_map);
    de_task.set_need_checked_net_set(need_checked_net_set);
  }
  return RTDE.getViolationList(de_task);
}

void DetailedRouter::updateBestResult(DRModel& dr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();

  std::map<int32_t, std::vector<Segment<LayerCoord>>>& best_net_detailed_result_map = dr_model.get_best_net_detailed_result_map();
  std::map<int32_t, std::vector<EXTLayerRect>>& best_net_detailed_patch_map = dr_model.get_best_net_detailed_patch_map();
  std::vector<Violation>& best_route_violation_list = dr_model.get_best_route_violation_list();

  int32_t curr_violation_num = getRouteViolationNum(dr_model);
  if (!best_net_detailed_result_map.empty()) {
    if (static_cast<int32_t>(best_route_violation_list.size()) < curr_violation_num) {
      return;
    }
  }
  best_net_detailed_result_map.clear();
  for (auto& [net_idx, segment_set] : RTDM.getNetDetailedResultMap(die)) {
    for (Segment<LayerCoord>* segment : segment_set) {
      best_net_detailed_result_map[net_idx].push_back(*segment);
    }
  }
  best_net_detailed_patch_map.clear();
  for (auto& [net_idx, patch_set] : RTDM.getNetDetailedPatchMap(die)) {
    for (EXTLayerRect* patch : patch_set) {
      best_net_detailed_patch_map[net_idx].push_back(*patch);
    }
  }
  best_route_violation_list.clear();
  for (Violation* violation : RTDM.getViolationSet(die)) {
    best_route_violation_list.push_back(*violation);
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

bool DetailedRouter::stopIteration(DRModel& dr_model)
{
  if (getRouteViolationNum(dr_model) == 0) {
    RTLOG.info(Loc::current(), "***** Iteration stopped early *****");
    return true;
  }
  return false;
}

void DetailedRouter::selectBestResult(DRModel& dr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  dr_model.set_iter(dr_model.get_iter() + 1);
  uploadBestResult(dr_model);
  updateSummary(dr_model);
  printSummary(dr_model);
  outputNetCSV(dr_model);
  outputViolationCSV(dr_model);
  outputJson(dr_model);

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void DetailedRouter::uploadBestResult(DRModel& dr_model)
{
  Die& die = RTDM.getDatabase().get_die();

  for (auto& [net_idx, segment_set] : RTDM.getNetDetailedResultMap(die)) {
    for (Segment<LayerCoord>* segment : segment_set) {
      RTDM.updateNetDetailedResultToGCellMap(ChangeType::kDel, net_idx, segment);
    }
  }
  for (auto& [net_idx, patch_set] : RTDM.getNetDetailedPatchMap(die)) {
    for (EXTLayerRect* patch : patch_set) {
      RTDM.updateNetDetailedPatchToGCellMap(ChangeType::kDel, net_idx, patch);
    }
  }
  for (Violation* violation : RTDM.getViolationSet(die)) {
    RTDM.updateViolationToGCellMap(ChangeType::kDel, violation);
  }

  for (auto& [net_idx, segment_list] : dr_model.get_best_net_detailed_result_map()) {
    for (Segment<LayerCoord>& segment : segment_list) {
      RTDM.updateNetDetailedResultToGCellMap(ChangeType::kAdd, net_idx, new Segment<LayerCoord>(segment));
    }
  }
  for (auto& [net_idx, patch_list] : dr_model.get_best_net_detailed_patch_map()) {
    for (EXTLayerRect& patch : patch_list) {
      RTDM.updateNetDetailedPatchToGCellMap(ChangeType::kAdd, net_idx, new EXTLayerRect(patch));
    }
  }
  for (Violation violation : dr_model.get_best_route_violation_list()) {
    RTDM.updateViolationToGCellMap(ChangeType::kAdd, new Violation(violation));
  }
}

#if 1  // update env

void DetailedRouter::updateFixedRectToGraph(DRBox& dr_box, ChangeType change_type, int32_t net_idx, EXTLayerRect* fixed_rect, bool is_routing)
{
  NetShape net_shape(net_idx, fixed_rect->getRealLayerRect(), is_routing);
  for (auto& [dr_node, orientation_set] : getNodeOrientationMap(dr_box, net_shape)) {
    for (Orientation orientation : orientation_set) {
      if (change_type == ChangeType::kAdd) {
        dr_node->get_orient_fixed_rect_map()[orientation].insert(net_shape.get_net_idx());
      } else if (change_type == ChangeType::kDel) {
        dr_node->get_orient_fixed_rect_map()[orientation].erase(net_shape.get_net_idx());
      }
    }
  }
}

void DetailedRouter::updateFixedRectToGraph(DRBox& dr_box, ChangeType change_type, int32_t net_idx, LayerRect& real_rect, bool is_routing)
{
  NetShape net_shape(net_idx, real_rect, is_routing);
  for (auto& [dr_node, orientation_set] : getNodeOrientationMap(dr_box, net_shape)) {
    for (Orientation orientation : orientation_set) {
      if (change_type == ChangeType::kAdd) {
        dr_node->get_orient_fixed_rect_map()[orientation].insert(net_shape.get_net_idx());
      } else if (change_type == ChangeType::kDel) {
        dr_node->get_orient_fixed_rect_map()[orientation].erase(net_shape.get_net_idx());
      }
    }
  }
}

void DetailedRouter::updateFixedRectToGraph(DRBox& dr_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>* segment)
{
  for (NetShape& net_shape : RTDM.getNetDetailedShapeList(net_idx, *segment)) {
    for (auto& [dr_node, orientation_set] : getNodeOrientationMap(dr_box, net_shape)) {
      for (Orientation orientation : orientation_set) {
        if (change_type == ChangeType::kAdd) {
          dr_node->get_orient_fixed_rect_map()[orientation].insert(net_shape.get_net_idx());
        } else if (change_type == ChangeType::kDel) {
          dr_node->get_orient_fixed_rect_map()[orientation].erase(net_shape.get_net_idx());
        }
      }
    }
  }
}

void DetailedRouter::updateRoutedRectToGraph(DRBox& dr_box, ChangeType change_type, int32_t net_idx, LayerRect& real_rect, bool is_routing)
{
  NetShape net_shape(net_idx, real_rect, is_routing);
  for (auto& [dr_node, orientation_set] : getNodeOrientationMap(dr_box, net_shape)) {
    for (Orientation orientation : orientation_set) {
      if (change_type == ChangeType::kAdd) {
        dr_node->get_orient_routed_rect_map()[orientation].insert(net_shape.get_net_idx());
      } else if (change_type == ChangeType::kDel) {
        dr_node->get_orient_routed_rect_map()[orientation].erase(net_shape.get_net_idx());
      }
    }
  }
}

void DetailedRouter::updateRoutedRectToGraph(DRBox& dr_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment)
{
  for (NetShape& net_shape : RTDM.getNetDetailedShapeList(net_idx, segment)) {
    for (auto& [dr_node, orientation_set] : getNodeOrientationMap(dr_box, net_shape)) {
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

void DetailedRouter::updateRoutedRectToGraph(DRBox& dr_box, ChangeType change_type, int32_t net_idx, EXTLayerRect& routed_rect, bool is_routing)
{
  NetShape net_shape(net_idx, routed_rect.getRealLayerRect(), is_routing);
  for (auto& [dr_node, orientation_set] : getNodeOrientationMap(dr_box, net_shape)) {
    for (Orientation orientation : orientation_set) {
      if (change_type == ChangeType::kAdd) {
        dr_node->get_orient_routed_rect_map()[orientation].insert(net_shape.get_net_idx());
      } else if (change_type == ChangeType::kDel) {
        dr_node->get_orient_routed_rect_map()[orientation].erase(net_shape.get_net_idx());
      }
    }
  }
}

void DetailedRouter::addRouteViolationToGraph(DRBox& dr_box, Violation& violation)
{
  LayerRect searched_rect = violation.get_violation_shape().get_real_rect();
  std::vector<Segment<LayerCoord>> overlap_segment_list;
  while (true) {
    searched_rect.set_rect(RTUTIL.getEnlargedRect(searched_rect, RTDM.getOnlyPitch()));
    if (violation.get_is_routing()) {
      searched_rect.set_layer_idx(violation.get_violation_shape().get_layer_idx());
    } else {
      RTLOG.error(Loc::current(), "The violation layer is cut!");
    }
    for (auto& [net_idx, segment_list] : dr_box.get_net_task_detailed_result_map()) {
      if (!RTUTIL.exist(violation.get_violation_net_set(), net_idx)) {
        continue;
      }
      for (Segment<LayerCoord>& segment : segment_list) {
        if (!RTUTIL.isOverlap(searched_rect, segment)) {
          continue;
        }
        overlap_segment_list.push_back(segment);
      }
    }
    if (!overlap_segment_list.empty()) {
      break;
    }
    if (!RTUTIL.isInside(dr_box.get_box_rect().get_real_rect(), searched_rect)) {
      break;
    }
  }
  addRouteViolationToGraph(dr_box, searched_rect, overlap_segment_list);
}

void DetailedRouter::addRouteViolationToGraph(DRBox& dr_box, LayerRect& searched_rect, std::vector<Segment<LayerCoord>>& overlap_segment_list)
{
  ScaleAxis& box_track_axis = dr_box.get_box_track_axis();
  std::vector<GridMap<DRNode>>& layer_node_map = dr_box.get_layer_node_map();

  for (Segment<LayerCoord>& overlap_segment : overlap_segment_list) {
    LayerCoord& first_coord = overlap_segment.get_first();
    LayerCoord& second_coord = overlap_segment.get_second();
    if (first_coord == second_coord) {
      continue;
    }
    PlanarRect real_rect = RTUTIL.getRect(first_coord, second_coord);
    if (!RTUTIL.existTrackGrid(real_rect, box_track_axis)) {
      continue;
    }
    PlanarRect grid_rect = RTUTIL.getTrackGrid(real_rect, box_track_axis);
    std::map<int32_t, std::set<DRNode*>> distance_node_map;
    {
      int32_t first_layer_idx = first_coord.get_layer_idx();
      int32_t second_layer_idx = second_coord.get_layer_idx();
      RTUTIL.swapByASC(first_layer_idx, second_layer_idx);
      for (int32_t layer_idx = first_layer_idx; layer_idx <= second_layer_idx; layer_idx++) {
        for (int32_t x = grid_rect.get_ll_x(); x <= grid_rect.get_ur_x(); x++) {
          for (int32_t y = grid_rect.get_ll_y(); y <= grid_rect.get_ur_y(); y++) {
            DRNode* dr_node = &layer_node_map[layer_idx][x][y];
            if (searched_rect.get_layer_idx() != dr_node->get_layer_idx()) {
              continue;
            }
            int32_t distance = 0;
            if (!RTUTIL.isInside(searched_rect.get_rect(), dr_node->get_planar_coord())) {
              distance = RTUTIL.getManhattanDistance(searched_rect.getMidPoint(), dr_node->get_planar_coord());
            }
            distance_node_map[distance].insert(dr_node);
          }
        }
      }
    }
    std::set<DRNode*> valid_node_set;
    if (!distance_node_map[0].empty()) {
      valid_node_set = distance_node_map[0];
    } else {
      for (auto& [distance, node_set] : distance_node_map) {
        valid_node_set.insert(node_set.begin(), node_set.end());
        if (valid_node_set.size() >= 2) {
          break;
        }
      }
    }
    Orientation orientation = RTUTIL.getOrientation(first_coord, second_coord);
    Orientation oppo_orientation = RTUTIL.getOppositeOrientation(orientation);
    for (DRNode* valid_node : valid_node_set) {
      if (LayerCoord(*valid_node) != first_coord) {
        valid_node->get_orient_violation_number_map()[oppo_orientation]++;
        if (RTUTIL.exist(valid_node->get_neighbor_node_map(), oppo_orientation)) {
          valid_node->get_neighbor_node_map()[oppo_orientation]->get_orient_violation_number_map()[orientation]++;
        }
      }
      if (LayerCoord(*valid_node) != second_coord) {
        valid_node->get_orient_violation_number_map()[orientation]++;
        if (RTUTIL.exist(valid_node->get_neighbor_node_map(), orientation)) {
          valid_node->get_neighbor_node_map()[orientation]->get_orient_violation_number_map()[oppo_orientation]++;
        }
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
  // x_spacing y_spacing
  std::vector<std::pair<int32_t, int32_t>> spacing_pair_list;
  {
    // prl
    int32_t prl_spacing = routing_layer.getPRLSpacing(net_shape.get_rect());
    spacing_pair_list.emplace_back(prl_spacing, prl_spacing);
    // eol
    int32_t max_eol_spacing = std::max(routing_layer.get_eol_spacing(), routing_layer.get_eol_ete());
    if (routing_layer.isPreferH()) {
      spacing_pair_list.emplace_back(max_eol_spacing, routing_layer.get_eol_within());
    } else {
      spacing_pair_list.emplace_back(routing_layer.get_eol_within(), max_eol_spacing);
    }
  }
  int32_t half_wire_width = routing_layer.get_min_width() / 2;
  PlanarRect& enclosure = layer_enclosure_map[layer_idx];
  int32_t enclosure_half_x_span = enclosure.getXSpan() / 2;
  int32_t enclosure_half_y_span = enclosure.getYSpan() / 2;

  GridMap<DRNode>& dr_node_map = dr_box.get_layer_node_map()[layer_idx];
  std::map<DRNode*, std::set<Orientation>> node_orientation_map;
  // wire 与 net_shape
  for (auto& [x_spacing, y_spacing] : spacing_pair_list) {
    // 膨胀size为 half_wire_width + spacing
    int32_t enlarged_x_size = half_wire_width + x_spacing;
    int32_t enlarged_y_size = half_wire_width + y_spacing;
    // 贴合的也不算违例
    enlarged_x_size -= 1;
    enlarged_y_size -= 1;
    PlanarRect planar_enlarged_rect = RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size);
    for (auto& [set_pair, orientation_set] : RTUTIL.getTrackGridOrientationMap(planar_enlarged_rect, dr_box.get_box_track_axis())) {
      auto& x_set = set_pair.first;
      auto& y_set = set_pair.second;
      for(auto x: *x_set){
        for(auto y: *y_set){
          DRNode& node = dr_node_map[x][y];
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
    }
  }
  // enclosure 与 net_shape
  for (auto& [x_spacing, y_spacing] : spacing_pair_list) {
    // 膨胀size为 enclosure_half_span + spacing
    int32_t enlarged_x_size = enclosure_half_x_span + x_spacing;
    int32_t enlarged_y_size = enclosure_half_y_span + y_spacing;
    // 贴合的也不算违例
    enlarged_x_size -= 1;
    enlarged_y_size -= 1;
    PlanarRect space_enlarged_rect = RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size);
    for (auto& [set_pair, orientation_set] : RTUTIL.getTrackGridOrientationMap(space_enlarged_rect, dr_box.get_box_track_axis())) {
      auto& x_set = set_pair.first;
      auto& y_set = set_pair.second;
      for(auto x: *x_set){
        for(auto y: *y_set){
          DRNode& node = dr_node_map[x][y];
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
    }
  }
  return node_orientation_map;
}

std::map<DRNode*, std::set<Orientation>> DetailedRouter::getCutNodeOrientationMap(DRBox& dr_box, NetShape& net_shape)
{
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = RTDM.getDatabase().get_cut_to_adjacent_routing_map();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();
  if (net_shape.get_is_routing()) {
    RTLOG.error(Loc::current(), "The type of net_shape is routing!");
  }
  CutLayer& cut_layer = cut_layer_list[net_shape.get_layer_idx()];
  std::map<int32_t, std::vector<std::pair<int32_t, int32_t>>> cut_spacing_map;
  {
    int32_t curr_cut_layer_idx = net_shape.get_layer_idx();
    if (0 <= curr_cut_layer_idx && curr_cut_layer_idx < static_cast<int32_t>(cut_layer_list.size())) {
      std::vector<int32_t> adjacent_routing_layer_idx_list = cut_to_adjacent_routing_map[curr_cut_layer_idx];
      if (adjacent_routing_layer_idx_list.size() == 2) {
        std::vector<std::pair<int32_t, int32_t>> spacing_pair_list;
        // prl
        spacing_pair_list.emplace_back(0, cut_layer.get_curr_spacing());
        spacing_pair_list.emplace_back(cut_layer.get_curr_spacing(), 0);
        spacing_pair_list.emplace_back(cut_layer.get_curr_spacing() / RT_SQRT_2, cut_layer.get_curr_spacing() / RT_SQRT_2);
        spacing_pair_list.emplace_back(cut_layer.get_curr_prl(), cut_layer.get_curr_prl_spacing());
        spacing_pair_list.emplace_back(cut_layer.get_curr_prl_spacing(), cut_layer.get_curr_prl());
        // eol
        spacing_pair_list.emplace_back(0, cut_layer.get_curr_eol_spacing());
        spacing_pair_list.emplace_back(cut_layer.get_curr_eol_spacing(), 0);
        spacing_pair_list.emplace_back(cut_layer.get_curr_eol_spacing() / RT_SQRT_2, cut_layer.get_curr_eol_spacing() / RT_SQRT_2);
        spacing_pair_list.emplace_back(cut_layer.get_curr_eol_prl(), cut_layer.get_curr_eol_prl_spacing());
        spacing_pair_list.emplace_back(cut_layer.get_curr_eol_prl_spacing(), cut_layer.get_curr_eol_prl());
        cut_spacing_map[curr_cut_layer_idx] = spacing_pair_list;
      }
    }
    int32_t below_cut_layer_idx = net_shape.get_layer_idx() - 1;
    if (0 <= below_cut_layer_idx && below_cut_layer_idx < static_cast<int32_t>(cut_layer_list.size())) {
      std::vector<int32_t> adjacent_routing_layer_idx_list = cut_to_adjacent_routing_map[below_cut_layer_idx];
      if (adjacent_routing_layer_idx_list.size() == 2) {
        std::vector<std::pair<int32_t, int32_t>> spacing_pair_list;
        // prl
        spacing_pair_list.emplace_back(0, cut_layer.get_below_spacing());
        spacing_pair_list.emplace_back(cut_layer.get_below_spacing(), 0);
        spacing_pair_list.emplace_back(cut_layer.get_below_spacing() / RT_SQRT_2, cut_layer.get_below_spacing() / RT_SQRT_2);
        spacing_pair_list.emplace_back(cut_layer.get_below_prl(), cut_layer.get_below_prl_spacing());
        spacing_pair_list.emplace_back(cut_layer.get_below_prl_spacing(), cut_layer.get_below_prl());
        cut_spacing_map[below_cut_layer_idx] = spacing_pair_list;
      }
    }
    int32_t above_cut_layer_idx = net_shape.get_layer_idx() + 1;
    if (0 <= above_cut_layer_idx && above_cut_layer_idx < static_cast<int32_t>(cut_layer_list.size())) {
      std::vector<int32_t> adjacent_routing_layer_idx_list = cut_to_adjacent_routing_map[above_cut_layer_idx];
      if (adjacent_routing_layer_idx_list.size() == 2) {
        std::vector<std::pair<int32_t, int32_t>> spacing_pair_list;
        // prl
        spacing_pair_list.emplace_back(0, cut_layer.get_above_spacing());
        spacing_pair_list.emplace_back(cut_layer.get_above_spacing(), 0);
        spacing_pair_list.emplace_back(cut_layer.get_above_spacing() / RT_SQRT_2, cut_layer.get_above_spacing() / RT_SQRT_2);
        spacing_pair_list.emplace_back(cut_layer.get_above_prl(), cut_layer.get_above_prl_spacing());
        spacing_pair_list.emplace_back(cut_layer.get_above_prl_spacing(), cut_layer.get_above_prl());
        cut_spacing_map[above_cut_layer_idx] = spacing_pair_list;
      }
    }
  }
  std::map<DRNode*, std::set<Orientation>> node_orientation_map;
  for (auto& [cut_layer_idx, spacing_pair_list] : cut_spacing_map) {
    std::vector<int32_t> adjacent_routing_layer_idx_list = cut_to_adjacent_routing_map[cut_layer_idx];
    int32_t below_routing_layer_idx = adjacent_routing_layer_idx_list.front();
    int32_t above_routing_layer_idx = adjacent_routing_layer_idx_list.back();
    RTUTIL.swapByASC(below_routing_layer_idx, above_routing_layer_idx);
    PlanarRect& cut_shape = layer_via_master_list[below_routing_layer_idx].front().get_cut_shape_list().front();
    int32_t cut_shape_half_x_span = cut_shape.getXSpan() / 2;
    int32_t cut_shape_half_y_span = cut_shape.getYSpan() / 2;
    std::vector<GridMap<DRNode>>& layer_node_map = dr_box.get_layer_node_map();
    for (auto& [x_spacing, y_spacing] : spacing_pair_list) {
      // 膨胀size为 cut_shape_half_span + spacing
      int32_t enlarged_x_size = cut_shape_half_x_span + x_spacing;
      int32_t enlarged_y_size = cut_shape_half_y_span + y_spacing;
      // 贴合的也不算违例
      enlarged_x_size -= 1;
      enlarged_y_size -= 1;
      PlanarRect space_enlarged_rect = RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size);
      for (auto& [set_pair, orientation_set] : RTUTIL.getTrackGridOrientationMap(space_enlarged_rect, dr_box.get_box_track_axis())) {
        auto& x_set = set_pair.first;
        auto& y_set = set_pair.second;
        for(auto x: *x_set){
          for(auto y: *y_set){
            if (!RTUTIL.exist(orientation_set, Orientation::kAbove) && !RTUTIL.exist(orientation_set, Orientation::kBelow)) {
              continue;
            }
            DRNode& below_node = layer_node_map[below_routing_layer_idx][x][y];
            if (RTUTIL.exist(below_node.get_neighbor_node_map(), Orientation::kAbove)) {
              node_orientation_map[&below_node].insert(Orientation::kAbove);
            }
            DRNode& above_node = layer_node_map[above_routing_layer_idx][x][y];
            if (RTUTIL.exist(above_node.get_neighbor_node_map(), Orientation::kBelow)) {
              node_orientation_map[&above_node].insert(Orientation::kBelow);
            }
          }
        }
      }
    }
  }
  return node_orientation_map;
}

void DetailedRouter::updateFixedRectToShadow(DRBox& dr_box, ChangeType change_type, int32_t net_idx, EXTLayerRect* fixed_rect, bool is_routing)
{
  NetShape net_shape(net_idx, fixed_rect->getRealLayerRect(), is_routing);
  if (!net_shape.get_is_routing()) {
    return;
  }
  for (PlanarRect& shadow_shape : getShadowShape(dr_box, net_shape)) {
    DRShadow& dr_shadow = dr_box.get_layer_shadow_map()[net_shape.get_layer_idx()];
    if (change_type == ChangeType::kAdd) {
      dr_shadow.get_net_fixed_rect_map()[net_idx].insert(shadow_shape);
    } else if (change_type == ChangeType::kDel) {
      dr_shadow.get_net_fixed_rect_map()[net_idx].erase(shadow_shape);
    }
  }
}

void DetailedRouter::updateFixedRectToShadow(DRBox& dr_box, ChangeType change_type, int32_t net_idx, LayerRect& real_rect, bool is_routing)
{
  NetShape net_shape(net_idx, real_rect, is_routing);
  if (!net_shape.get_is_routing()) {
    return;
  }
  for (PlanarRect& shadow_shape : getShadowShape(dr_box, net_shape)) {
    DRShadow& dr_shadow = dr_box.get_layer_shadow_map()[net_shape.get_layer_idx()];
    if (change_type == ChangeType::kAdd) {
      dr_shadow.get_net_fixed_rect_map()[net_idx].insert(shadow_shape);
    } else if (change_type == ChangeType::kDel) {
      dr_shadow.get_net_fixed_rect_map()[net_idx].erase(shadow_shape);
    }
  }
}

void DetailedRouter::updateFixedRectToShadow(DRBox& dr_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>* segment)
{
  for (NetShape& net_shape : RTDM.getNetDetailedShapeList(net_idx, *segment)) {
    if (!net_shape.get_is_routing()) {
      continue;
    }
    for (PlanarRect& shadow_shape : getShadowShape(dr_box, net_shape)) {
      DRShadow& dr_shadow = dr_box.get_layer_shadow_map()[net_shape.get_layer_idx()];
      if (change_type == ChangeType::kAdd) {
        dr_shadow.get_net_fixed_rect_map()[net_idx].insert(shadow_shape);
      } else if (change_type == ChangeType::kDel) {
        dr_shadow.get_net_fixed_rect_map()[net_idx].erase(shadow_shape);
      }
    }
  }
}

void DetailedRouter::updateRoutedRectToShadow(DRBox& dr_box, ChangeType change_type, int32_t net_idx, LayerRect& real_rect, bool is_routing)
{
  NetShape net_shape(net_idx, real_rect, is_routing);
  if (!net_shape.get_is_routing()) {
    return;
  }
  for (PlanarRect& shadow_shape : getShadowShape(dr_box, net_shape)) {
    DRShadow& dr_shadow = dr_box.get_layer_shadow_map()[net_shape.get_layer_idx()];
    if (change_type == ChangeType::kAdd) {
      dr_shadow.get_net_routed_rect_map()[net_idx].insert(shadow_shape);
    } else if (change_type == ChangeType::kDel) {
      dr_shadow.get_net_routed_rect_map()[net_idx].erase(shadow_shape);
    }
  }
}

void DetailedRouter::updateRoutedRectToShadow(DRBox& dr_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment)
{
  for (NetShape& net_shape : RTDM.getNetDetailedShapeList(net_idx, segment)) {
    if (!net_shape.get_is_routing()) {
      continue;
    }
    for (PlanarRect& shadow_shape : getShadowShape(dr_box, net_shape)) {
      DRShadow& dr_shadow = dr_box.get_layer_shadow_map()[net_shape.get_layer_idx()];
      if (change_type == ChangeType::kAdd) {
        dr_shadow.get_net_routed_rect_map()[net_idx].insert(shadow_shape);
      } else if (change_type == ChangeType::kDel) {
        dr_shadow.get_net_routed_rect_map()[net_idx].erase(shadow_shape);
      }
    }
  }
}

void DetailedRouter::updateRoutedRectToShadow(DRBox& dr_box, ChangeType change_type, int32_t net_idx, EXTLayerRect& routed_rect, bool is_routing)
{
  NetShape net_shape(net_idx, routed_rect.getRealLayerRect(), is_routing);
  if (!net_shape.get_is_routing()) {
    return;
  }
  for (PlanarRect& shadow_shape : getShadowShape(dr_box, net_shape)) {
    DRShadow& dr_shadow = dr_box.get_layer_shadow_map()[net_shape.get_layer_idx()];
    if (change_type == ChangeType::kAdd) {
      dr_shadow.get_net_routed_rect_map()[net_idx].insert(shadow_shape);
    } else if (change_type == ChangeType::kDel) {
      dr_shadow.get_net_routed_rect_map()[net_idx].erase(shadow_shape);
    }
  }
}

void DetailedRouter::addPatchViolationToShadow(DRBox& dr_box, Violation& violation)
{
  EXTLayerRect& violation_shape = violation.get_violation_shape();

  DRShadow& dr_shadow = dr_box.get_layer_shadow_map()[violation_shape.get_layer_idx()];
  dr_shadow.get_violation_set().insert(violation_shape.get_real_rect());
}

std::vector<PlanarRect> DetailedRouter::getShadowShape(DRBox& dr_box, NetShape& net_shape)
{
  std::vector<PlanarRect> shadow_shape_list;
  if (net_shape.get_is_routing()) {
    shadow_shape_list = getRoutingShadowShapeList(dr_box, net_shape);
  } else {
    RTLOG.error(Loc::current(), "The type of net_shape is cut!");
  }
  return shadow_shape_list;
}

std::vector<PlanarRect> DetailedRouter::getRoutingShadowShapeList(DRBox& dr_box, NetShape& net_shape)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  if (!net_shape.get_is_routing()) {
    RTLOG.error(Loc::current(), "The type of net_shape is cut!");
  }
  int32_t layer_idx = net_shape.get_layer_idx();
  RoutingLayer& routing_layer = routing_layer_list[layer_idx];
  // x_spacing y_spacing
  std::vector<std::pair<int32_t, int32_t>> spacing_pair_list;
  {
    // prl
    int32_t prl_spacing = routing_layer.getPRLSpacing(net_shape.get_rect());
    spacing_pair_list.emplace_back(prl_spacing, prl_spacing);
    // eol
    int32_t max_eol_spacing = std::max(routing_layer.get_eol_spacing(), routing_layer.get_eol_ete());
    if (routing_layer.isPreferH()) {
      spacing_pair_list.emplace_back(max_eol_spacing, routing_layer.get_eol_within());
    } else {
      spacing_pair_list.emplace_back(routing_layer.get_eol_within(), max_eol_spacing);
    }
  }
  std::vector<PlanarRect> shadow_shape_list;
  // wire 与 net_shape
  for (auto& [x_spacing, y_spacing] : spacing_pair_list) {
    // 膨胀size为 spacing
    int32_t enlarged_x_size = x_spacing;
    int32_t enlarged_y_size = y_spacing;
    // 贴合的也不算违例
    enlarged_x_size -= 1;
    enlarged_y_size -= 1;
    shadow_shape_list.push_back(RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size));
  }
  // enclosure 与 net_shape
  for (auto& [x_spacing, y_spacing] : spacing_pair_list) {
    // 膨胀size为 spacing
    int32_t enlarged_x_size = x_spacing;
    int32_t enlarged_y_size = y_spacing;
    // 贴合的也不算违例
    enlarged_x_size -= 1;
    enlarged_y_size -= 1;
    shadow_shape_list.push_back(RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size));
  }
  return shadow_shape_list;
}

#endif

#if 1  // get env

double DetailedRouter::getFixedRectCost(DRBox& dr_box, int32_t net_idx, EXTLayerRect& patch)
{
  double fixed_rect_unit = dr_box.get_dr_iter_param()->get_fixed_rect_unit();
  std::vector<DRShadow>& layer_shadow_map = dr_box.get_layer_shadow_map();

  double fixed_rect_cost = 0;
  for (auto& [graph_net_idx, fixed_rect_set] : layer_shadow_map[patch.get_layer_idx()].get_net_fixed_rect_map()) {
    if (net_idx == graph_net_idx) {
      continue;
    }
    for (const PlanarRect& fixed_rect : fixed_rect_set) {
      if (RTUTIL.isOpenOverlap(patch.get_real_rect(), fixed_rect)) {
        fixed_rect_cost += fixed_rect_unit;
      }
    }
  }
  return fixed_rect_cost;
}

double DetailedRouter::getRoutedRectCost(DRBox& dr_box, int32_t net_idx, EXTLayerRect& patch)
{
  double routed_rect_unit = dr_box.get_dr_iter_param()->get_routed_rect_unit();
  std::vector<DRShadow>& layer_shadow_map = dr_box.get_layer_shadow_map();

  double routed_rect_cost = 0;
  for (auto& [graph_net_idx, routed_rect_set] : layer_shadow_map[patch.get_layer_idx()].get_net_routed_rect_map()) {
    if (net_idx == graph_net_idx) {
      continue;
    }
    for (const PlanarRect& routed_rect : routed_rect_set) {
      if (RTUTIL.isOpenOverlap(patch.get_real_rect(), routed_rect)) {
        routed_rect_cost += routed_rect_unit;
      }
    }
  }
  return routed_rect_cost;
}

double DetailedRouter::getViolationCost(DRBox& dr_box, int32_t net_idx, EXTLayerRect& patch)
{
  double violation_unit = dr_box.get_dr_iter_param()->get_violation_unit();
  std::vector<DRShadow>& layer_shadow_map = dr_box.get_layer_shadow_map();

  double violation_cost = 0;
  for (const PlanarRect& violation : layer_shadow_map[patch.get_layer_idx()].get_violation_set()) {
    if (RTUTIL.isOpenOverlap(patch.get_real_rect(), violation)) {
      violation_cost += violation_unit;
    }
  }
  return violation_cost;
}

#endif

#if 1  // exhibit

void DetailedRouter::updateSummary(DRModel& dr_model)
{
  int32_t micron_dbu = RTDM.getDatabase().get_micron_dbu();
  Die& die = RTDM.getDatabase().get_die();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();
  Summary& summary = RTDM.getDatabase().get_summary();
  int32_t enable_timing = RTDM.getConfig().enable_timing;

  std::map<int32_t, double>& routing_wire_length_map = summary.iter_dr_summary_map[dr_model.get_iter()].routing_wire_length_map;
  double& total_wire_length = summary.iter_dr_summary_map[dr_model.get_iter()].total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.iter_dr_summary_map[dr_model.get_iter()].cut_via_num_map;
  int32_t& total_via_num = summary.iter_dr_summary_map[dr_model.get_iter()].total_via_num;
  std::map<int32_t, int32_t>& routing_patch_num_map = summary.iter_dr_summary_map[dr_model.get_iter()].routing_patch_num_map;
  int32_t& total_patch_num = summary.iter_dr_summary_map[dr_model.get_iter()].total_patch_num;
  std::map<int32_t, int32_t>& routing_violation_num_map = summary.iter_dr_summary_map[dr_model.get_iter()].routing_violation_num_map;
  int32_t& total_violation_num = summary.iter_dr_summary_map[dr_model.get_iter()].total_violation_num;
  std::map<std::string, std::map<std::string, double>>& clock_timing_map = summary.iter_dr_summary_map[dr_model.get_iter()].clock_timing_map;
  std::map<std::string, double>& type_power_map = summary.iter_dr_summary_map[dr_model.get_iter()].type_power_map;

  std::vector<DRNet>& dr_net_list = dr_model.get_dr_net_list();

  routing_wire_length_map.clear();
  total_wire_length = 0;
  cut_via_num_map.clear();
  total_via_num = 0;
  routing_patch_num_map.clear();
  total_patch_num = 0;
  routing_violation_num_map.clear();
  total_violation_num = 0;
  clock_timing_map.clear();
  type_power_map.clear();

  for (auto& [net_idx, segment_set] : RTDM.getNetDetailedResultMap(die)) {
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
  for (auto& [net_idx, patch_set] : RTDM.getNetDetailedPatchMap(die)) {
    for (EXTLayerRect* patch : patch_set) {
      routing_patch_num_map[patch->get_layer_idx()]++;
      total_patch_num++;
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
        real_pin_coord_map_list[dr_net.get_net_idx()][dr_pin.get_pin_name()].push_back(dr_pin.get_access_point().getRealLayerCoord());
      }
    }
    for (auto& [net_idx, segment_set] : RTDM.getNetDetailedResultMap(die)) {
      for (Segment<LayerCoord>* segment : segment_set) {
        routing_segment_list_list[net_idx].emplace_back(segment->get_first(), segment->get_second());
      }
    }
    RTI.updateTimingAndPower(real_pin_coord_map_list, routing_segment_list_list, clock_timing_map, type_power_map);
  }
}

void DetailedRouter::printSummary(DRModel& dr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  Summary& summary = RTDM.getDatabase().get_summary();
  int32_t enable_timing = RTDM.getConfig().enable_timing;

  std::map<int32_t, double>& routing_wire_length_map = summary.iter_dr_summary_map[dr_model.get_iter()].routing_wire_length_map;
  double& total_wire_length = summary.iter_dr_summary_map[dr_model.get_iter()].total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.iter_dr_summary_map[dr_model.get_iter()].cut_via_num_map;
  int32_t& total_via_num = summary.iter_dr_summary_map[dr_model.get_iter()].total_via_num;
  std::map<int32_t, int32_t>& routing_patch_num_map = summary.iter_dr_summary_map[dr_model.get_iter()].routing_patch_num_map;
  int32_t& total_patch_num = summary.iter_dr_summary_map[dr_model.get_iter()].total_patch_num;
  std::map<int32_t, int32_t>& routing_violation_num_map = summary.iter_dr_summary_map[dr_model.get_iter()].routing_violation_num_map;
  int32_t& total_violation_num = summary.iter_dr_summary_map[dr_model.get_iter()].total_violation_num;
  std::map<std::string, std::map<std::string, double>>& clock_timing_map = summary.iter_dr_summary_map[dr_model.get_iter()].clock_timing_map;
  std::map<std::string, double>& type_power_map = summary.iter_dr_summary_map[dr_model.get_iter()].type_power_map;

  fort::char_table routing_wire_length_map_table;
  {
    routing_wire_length_map_table.set_cell_text_align(fort::text_align::right);
    routing_wire_length_map_table << fort::header << "routing"
                                  << "wire_length"
                                  << "prop" << fort::endr;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      routing_wire_length_map_table << routing_layer.get_layer_name() << routing_wire_length_map[routing_layer.get_layer_idx()]
                                    << RTUTIL.getPercentage(routing_wire_length_map[routing_layer.get_layer_idx()], total_wire_length) << fort::endr;
    }
    routing_wire_length_map_table << fort::header << "Total" << total_wire_length << RTUTIL.getPercentage(total_wire_length, total_wire_length) << fort::endr;
  }
  fort::char_table cut_via_num_map_table;
  {
    cut_via_num_map_table.set_cell_text_align(fort::text_align::right);
    cut_via_num_map_table << fort::header << "cut"
                          << "#via"
                          << "prop" << fort::endr;
    for (CutLayer& cut_layer : cut_layer_list) {
      cut_via_num_map_table << cut_layer.get_layer_name() << cut_via_num_map[cut_layer.get_layer_idx()]
                            << RTUTIL.getPercentage(cut_via_num_map[cut_layer.get_layer_idx()], total_via_num) << fort::endr;
    }
    cut_via_num_map_table << fort::header << "Total" << total_via_num << RTUTIL.getPercentage(total_via_num, total_via_num) << fort::endr;
  }
  fort::char_table routing_patch_num_map_table;
  {
    routing_patch_num_map_table.set_cell_text_align(fort::text_align::right);
    routing_patch_num_map_table << fort::header << "routing"
                                << "#patch"
                                << "prop" << fort::endr;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      routing_patch_num_map_table << routing_layer.get_layer_name() << routing_patch_num_map[routing_layer.get_layer_idx()]
                                  << RTUTIL.getPercentage(routing_patch_num_map[routing_layer.get_layer_idx()], total_patch_num) << fort::endr;
    }
    routing_patch_num_map_table << fort::header << "Total" << total_patch_num << RTUTIL.getPercentage(total_patch_num, total_patch_num) << fort::endr;
  }
  fort::char_table routing_violation_num_map_table;
  {
    routing_violation_num_map_table.set_cell_text_align(fort::text_align::right);
    routing_violation_num_map_table << fort::header << "routing"
                                    << "#violation"
                                    << "prop" << fort::endr;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      routing_violation_num_map_table << routing_layer.get_layer_name() << routing_violation_num_map[routing_layer.get_layer_idx()]
                                      << RTUTIL.getPercentage(routing_violation_num_map[routing_layer.get_layer_idx()], total_violation_num) << fort::endr;
    }
    routing_violation_num_map_table << fort::header << "Total" << total_violation_num << RTUTIL.getPercentage(total_violation_num, total_violation_num)
                                    << fort::endr;
  }
  fort::char_table timing_table;
  timing_table.set_cell_text_align(fort::text_align::right);
  fort::char_table power_table;
  power_table.set_cell_text_align(fort::text_align::right);
  if (enable_timing) {
    timing_table << fort::header << "clock_name"
                 << "tns"
                 << "wns"
                 << "freq" << fort::endr;
    for (auto& [clock_name, timing_map] : clock_timing_map) {
      timing_table << clock_name << timing_map["TNS"] << timing_map["WNS"] << timing_map["Freq(MHz)"] << fort::endr;
    }
    power_table << fort::header << "power_type";
    for (auto& [type, power] : type_power_map) {
      power_table << fort::header << type;
    }
    power_table << fort::endr;
    power_table << "power_value";
    for (auto& [type, power] : type_power_map) {
      power_table << power;
    }
    power_table << fort::endr;
  }
  RTUTIL.printTableList({routing_wire_length_map_table, cut_via_num_map_table, routing_patch_num_map_table});
  RTUTIL.printTableList({routing_violation_num_map_table});
  RTUTIL.printTableList({timing_table, power_table});
}

void DetailedRouter::outputNetCSV(DRModel& dr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::string& dr_temp_directory_path = RTDM.getConfig().dr_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  std::vector<GridMap<int32_t>> layer_net_map;
  layer_net_map.resize(routing_layer_list.size());
  for (GridMap<int32_t>& net_map : layer_net_map) {
    net_map.init(gcell_map.get_x_size(), gcell_map.get_y_size());
  }
  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      std::map<int32_t, std::set<int32_t>> net_layer_map;
      for (auto& [net_idx, segment_set] : gcell_map[x][y].get_net_detailed_result_map()) {
        for (Segment<LayerCoord>* segment : segment_set) {
          int32_t first_layer_idx = segment->get_first().get_layer_idx();
          int32_t second_layer_idx = segment->get_second().get_layer_idx();
          RTUTIL.swapByASC(first_layer_idx, second_layer_idx);
          for (int32_t layer_idx = first_layer_idx; layer_idx <= second_layer_idx; layer_idx++) {
            net_layer_map[net_idx].insert(layer_idx);
          }
        }
      }
      for (auto& [net_idx, patch_set] : gcell_map[x][y].get_net_detailed_patch_map()) {
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
    std::ofstream* net_csv_file
        = RTUTIL.getOutputFileStream(RTUTIL.getString(dr_temp_directory_path, "net_map_", routing_layer.get_layer_name(), "_", dr_model.get_iter(), ".csv"));
    GridMap<int32_t>& net_map = layer_net_map[routing_layer.get_layer_idx()];
    for (int32_t y = net_map.get_y_size() - 1; y >= 0; y--) {
      for (int32_t x = 0; x < net_map.get_x_size(); x++) {
        RTUTIL.pushStream(net_csv_file, net_map[x][y], ",");
      }
      RTUTIL.pushStream(net_csv_file, "\n");
    }
    RTUTIL.closeFileStream(net_csv_file);
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void DetailedRouter::outputViolationCSV(DRModel& dr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::string& dr_temp_directory_path = RTDM.getConfig().dr_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

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
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void DetailedRouter::outputJson(DRModel& dr_model)
{
  int32_t enable_notification = RTDM.getConfig().enable_notification;
  if (!enable_notification) {
    return;
  }
  std::map<std::string, std::string> json_path_map;
  json_path_map["net_map"] = outputNetJson(dr_model);
  json_path_map["violation_map"] = outputViolationJson(dr_model);
  json_path_map["summary"] = outputSummaryJson(dr_model);
  RTI.sendNotification("DR", dr_model.get_iter(), json_path_map);
}

std::string DetailedRouter::outputNetJson(DRModel& dr_model)
{
  Die& die = RTDM.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();
  std::string& dr_temp_directory_path = RTDM.getConfig().dr_temp_directory_path;

  std::vector<nlohmann::json> net_json_list;
  {
    nlohmann::json result_shape_json;
    for (auto& [net_idx, segment_set] : RTDM.getNetDetailedResultMap(die)) {
      std::string net_name = net_list[net_idx].get_net_name();
      for (Segment<LayerCoord>* segment : segment_set) {
        for (NetShape& net_shape : RTDM.getNetDetailedShapeList(net_idx, *segment)) {
          std::string layer_name;
          if (net_shape.get_is_routing()) {
            layer_name = routing_layer_list[net_shape.get_layer_idx()].get_layer_name();
          } else {
            layer_name = cut_layer_list[net_shape.get_layer_idx()].get_layer_name();
          }
          result_shape_json["result_shape"][net_name]["path"].push_back(
              {net_shape.get_ll_x(), net_shape.get_ll_y(), net_shape.get_ur_x(), net_shape.get_ur_y(), layer_name});
        }
      }
    }
    for (auto& [net_idx, patch_set] : RTDM.getNetDetailedPatchMap(die)) {
      std::string net_name = net_list[net_idx].get_net_name();
      for (EXTLayerRect* patch : patch_set) {
        result_shape_json["result_shape"][net_name]["patch"].push_back({patch->get_real_ll_x(), patch->get_real_ll_y(), patch->get_real_ur_x(),
                                                                        patch->get_real_ur_y(), routing_layer_list[patch->get_layer_idx()].get_layer_name()});
      }
    }
    net_json_list.push_back(result_shape_json);
  }
  std::string net_json_file_path = RTUTIL.getString(dr_temp_directory_path, "net_map_", dr_model.get_iter(), ".json");
  std::ofstream* net_json_file = RTUTIL.getOutputFileStream(net_json_file_path);
  (*net_json_file) << net_json_list;
  RTUTIL.closeFileStream(net_json_file);
  return net_json_file_path;
}

std::string DetailedRouter::outputViolationJson(DRModel& dr_model)
{
  Die& die = RTDM.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();
  std::string& dr_temp_directory_path = RTDM.getConfig().dr_temp_directory_path;

  std::vector<nlohmann::json> violation_json_list;
  for (Violation* violation : RTDM.getViolationSet(die)) {
    EXTLayerRect& violation_shape = violation->get_violation_shape();

    nlohmann::json violation_json;
    violation_json["type"] = GetViolationTypeName()(violation->get_violation_type());
    violation_json["shape"]
        = {violation_shape.get_real_rect().get_ll_x(), violation_shape.get_real_rect().get_ll_y(), violation_shape.get_real_rect().get_ur_x(),
           violation_shape.get_real_rect().get_ur_y(), routing_layer_list[violation_shape.get_layer_idx()].get_layer_name()};
    for (int32_t net_idx : violation->get_violation_net_set()) {
      if (net_idx != -1) {
        violation_json["net"].push_back(net_list[net_idx].get_net_name());
      } else {
        violation_json["net"].push_back("obs");
      }
    }
    violation_json_list.push_back(violation_json);
  }
  std::string violation_json_file_path = RTUTIL.getString(dr_temp_directory_path, "violation_map_", dr_model.get_iter(), ".json");
  std::ofstream* violation_json_file = RTUTIL.getOutputFileStream(violation_json_file_path);
  (*violation_json_file) << violation_json_list;
  RTUTIL.closeFileStream(violation_json_file);
  return violation_json_file_path;
}

std::string DetailedRouter::outputSummaryJson(DRModel& dr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  Summary& summary = RTDM.getDatabase().get_summary();
  std::string& dr_temp_directory_path = RTDM.getConfig().dr_temp_directory_path;

  std::map<int32_t, double>& routing_wire_length_map = summary.iter_dr_summary_map[dr_model.get_iter()].routing_wire_length_map;
  double& total_wire_length = summary.iter_dr_summary_map[dr_model.get_iter()].total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.iter_dr_summary_map[dr_model.get_iter()].cut_via_num_map;
  int32_t& total_via_num = summary.iter_dr_summary_map[dr_model.get_iter()].total_via_num;
  std::map<int32_t, int32_t>& routing_patch_num_map = summary.iter_dr_summary_map[dr_model.get_iter()].routing_patch_num_map;
  int32_t& total_patch_num = summary.iter_dr_summary_map[dr_model.get_iter()].total_patch_num;
  std::map<int32_t, int32_t>& routing_violation_num_map = summary.iter_dr_summary_map[dr_model.get_iter()].routing_violation_num_map;
  int32_t& total_violation_num = summary.iter_dr_summary_map[dr_model.get_iter()].total_violation_num;
  std::map<std::string, std::map<std::string, double>>& clock_timing_map = summary.iter_dr_summary_map[dr_model.get_iter()].clock_timing_map;
  std::map<std::string, double>& type_power_map = summary.iter_dr_summary_map[dr_model.get_iter()].type_power_map;

  nlohmann::json summary_json;
  summary_json["iter"] = dr_model.get_iter();
  for (auto& [routing_layer_idx, wire_length] : routing_wire_length_map) {
    summary_json["routing_wire_length_map"][routing_layer_list[routing_layer_idx].get_layer_name()] = wire_length;
  }
  summary_json["total_wire_length"] = total_wire_length;
  for (auto& [cut_layer_idx, via_num] : cut_via_num_map) {
    summary_json["cut_via_num_map"][cut_layer_list[cut_layer_idx].get_layer_name()] = via_num;
  }
  summary_json["total_via_num"] = total_via_num;
  for (auto& [routing_layer_idx, patch_num] : routing_patch_num_map) {
    summary_json["routing_patch_num_map"][routing_layer_list[routing_layer_idx].get_layer_name()] = patch_num;
  }
  summary_json["total_patch_num"] = total_patch_num;
  for (auto& [routing_layer_idx, violation_num] : routing_violation_num_map) {
    summary_json["routing_violation_num_map"][routing_layer_list[routing_layer_idx].get_layer_name()] = violation_num;
  }
  summary_json["total_violation_num"] = total_violation_num;
  for (auto& [clock_name, timing] : clock_timing_map) {
    summary_json["clock_timing_map"]["clock_name"] = clock_name;
    summary_json["clock_timing_map"]["timing"] = timing;
  }
  for (auto& [type, power] : type_power_map) {
    summary_json["type_power_map"]["type"] = type;
    summary_json["type_power_map"]["power"] = power;
  }

  std::string summary_json_file_path = RTUTIL.getString(dr_temp_directory_path, "summary_", dr_model.get_iter(), ".json");
  std::ofstream* summary_json_file = RTUTIL.getOutputFileStream(summary_json_file_path);
  (*summary_json_file) << summary_json;
  RTUTIL.closeFileStream(summary_json_file);
  return summary_json_file_path;
}

#endif

#if 1  // debug

void DetailedRouter::debugPlotDRModel(DRModel& dr_model, std::string flag)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& dr_temp_directory_path = RTDM.getConfig().dr_temp_directory_path;

  int32_t point_size = 5;

  GPGDS gp_gds;

  // base_region
  {
    GPStruct base_region_struct("base_region");
    GPBoundary gp_boundary;
    gp_boundary.set_layer_idx(0);
    gp_boundary.set_data_type(0);
    gp_boundary.set_rect(die.get_real_rect());
    base_region_struct.push(gp_boundary);
    gp_gds.addStruct(base_region_struct);
  }

  // gcell_axis
  {
    GPStruct gcell_axis_struct("gcell_axis");
    std::vector<int32_t> gcell_x_list = RTUTIL.getScaleList(die.get_real_ll_x(), die.get_real_ur_x(), gcell_axis.get_x_grid_list());
    std::vector<int32_t> gcell_y_list = RTUTIL.getScaleList(die.get_real_ll_y(), die.get_real_ur_y(), gcell_axis.get_y_grid_list());
    for (int32_t x : gcell_x_list) {
      GPPath gp_path;
      gp_path.set_layer_idx(0);
      gp_path.set_data_type(1);
      gp_path.set_segment(x, die.get_real_ll_y(), x, die.get_real_ur_y());
      gcell_axis_struct.push(gp_path);
    }
    for (int32_t y : gcell_y_list) {
      GPPath gp_path;
      gp_path.set_layer_idx(0);
      gp_path.set_data_type(1);
      gp_path.set_segment(die.get_real_ll_x(), y, die.get_real_ur_x(), y);
      gcell_axis_struct.push(gp_path);
    }
    gp_gds.addStruct(gcell_axis_struct);
  }

  // track_axis_struct
  {
    GPStruct track_axis_struct("track_axis_struct");
    for (RoutingLayer& routing_layer : routing_layer_list) {
      std::vector<int32_t> x_list = RTUTIL.getScaleList(die.get_real_ll_x(), die.get_real_ur_x(), routing_layer.getXTrackGridList());
      std::vector<int32_t> y_list = RTUTIL.getScaleList(die.get_real_ll_y(), die.get_real_ur_y(), routing_layer.getYTrackGridList());
      for (int32_t x : x_list) {
        GPPath gp_path;
        gp_path.set_data_type(static_cast<int32_t>(GPDataType::kAxis));
        gp_path.set_segment(x, die.get_real_ll_y(), x, die.get_real_ur_y());
        gp_path.set_layer_idx(RTGP.getGDSIdxByRouting(routing_layer.get_layer_idx()));
        track_axis_struct.push(gp_path);
      }
      for (int32_t y : y_list) {
        GPPath gp_path;
        gp_path.set_data_type(static_cast<int32_t>(GPDataType::kAxis));
        gp_path.set_segment(die.get_real_ll_x(), y, die.get_real_ur_x(), y);
        gp_path.set_layer_idx(RTGP.getGDSIdxByRouting(routing_layer.get_layer_idx()));
        track_axis_struct.push(gp_path);
      }
    }
    gp_gds.addStruct(track_axis_struct);
  }

  // fixed_rect
  for (auto& [is_routing, layer_net_fixed_rect_map] : RTDM.getTypeLayerNetFixedRectMap(die)) {
    for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
      for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
        GPStruct fixed_rect_struct(RTUTIL.getString("fixed_rect(net_", net_idx, ")"));
        for (auto& fixed_rect : fixed_rect_set) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShape));
          gp_boundary.set_rect(fixed_rect->get_real_rect());
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

  // access_point
  for (auto& [net_idx, access_point_set] : RTDM.getNetAccessPointMap(die)) {
    GPStruct access_point_struct(RTUTIL.getString("access_point(net_", net_idx, ")"));
    for (AccessPoint* access_point : access_point_set) {
      int32_t x = access_point->get_real_x();
      int32_t y = access_point->get_real_y();

      GPBoundary access_point_boundary;
      access_point_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(access_point->get_layer_idx()));
      access_point_boundary.set_data_type(static_cast<int32_t>(GPDataType::kAccessPoint));
      access_point_boundary.set_rect(x - point_size, y - point_size, x + point_size, y + point_size);
      access_point_struct.push(access_point_boundary);
    }
    gp_gds.addStruct(access_point_struct);
  }

  // routing result
  for (auto& [net_idx, segment_set] : RTDM.getNetGlobalResultMap(die)) {
    GPStruct global_result_struct(RTUTIL.getString("global_result(net_", net_idx, ")"));
    for (Segment<LayerCoord>* segment : segment_set) {
      for (NetShape& net_shape : RTDM.getNetGlobalShapeList(net_idx, *segment)) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kGlobalPath));
        gp_boundary.set_rect(net_shape.get_rect());
        if (net_shape.get_is_routing()) {
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(net_shape.get_layer_idx()));
        } else {
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(net_shape.get_layer_idx()));
        }
        global_result_struct.push(gp_boundary);
      }
    }
    gp_gds.addStruct(global_result_struct);
  }

  // routing result
  for (auto& [net_idx, segment_set] : RTDM.getNetDetailedResultMap(die)) {
    GPStruct detailed_result_struct(RTUTIL.getString("detailed_result(net_", net_idx, ")"));
    for (Segment<LayerCoord>* segment : segment_set) {
      for (NetShape& net_shape : RTDM.getNetDetailedShapeList(net_idx, *segment)) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kDetailedPath));
        gp_boundary.set_rect(net_shape.get_rect());
        if (net_shape.get_is_routing()) {
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(net_shape.get_layer_idx()));
        } else {
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(net_shape.get_layer_idx()));
        }
        detailed_result_struct.push(gp_boundary);
      }
    }
    gp_gds.addStruct(detailed_result_struct);
  }

  // routing patch
  for (auto& [net_idx, patch_set] : RTDM.getNetDetailedPatchMap(die)) {
    GPStruct detailed_patch_struct(RTUTIL.getString("detailed_patch(net_", net_idx, ")"));
    for (EXTLayerRect* patch : patch_set) {
      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPatch));
      gp_boundary.set_rect(patch->get_real_rect());
      gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(patch->get_layer_idx()));
      detailed_patch_struct.push(gp_boundary);
    }
    gp_gds.addStruct(detailed_patch_struct);
  }

  // violation
  {
    for (Violation* violation : RTDM.getViolationSet(die)) {
      GPStruct violation_struct(RTUTIL.getString("violation_", GetViolationTypeName()(violation->get_violation_type())));
      EXTLayerRect& violation_shape = violation->get_violation_shape();

      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kRouteViolation));
      gp_boundary.set_rect(violation_shape.get_real_rect());
      if (violation->get_is_routing()) {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(violation_shape.get_layer_idx()));
      } else {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(violation_shape.get_layer_idx()));
      }
      violation_struct.push(gp_boundary);
      gp_gds.addStruct(violation_struct);
    }
  }

  std::string gds_file_path = RTUTIL.getString(dr_temp_directory_path, flag, "_dr_model.gds");
  RTGP.plot(gp_gds, gds_file_path);
}

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
          if (RTUTIL.getOrientation(LayerCoord(dr_node), LayerCoord(*neighbor)) == orient) {
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
      for (auto& [coord, _] : dr_group.get_coord_direction_map()) {
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
          RTLOG.error(Loc::current(), "The neighbor of group coord (", coord.get_x(), ",", coord.get_y(), ",", layer_idx, ") is empty in box(",
                      dr_box_id.get_x(), ",", dr_box_id.get_y(), ")");
        }
        if (RTUTIL.isInside(dr_box.get_box_rect().get_real_rect(), coord)) {
          continue;
        }
        RTLOG.error(Loc::current(), "The coord (", coord.get_x(), ",", coord.get_y(), ") is out of box!");
      }
    }
  }
}

void DetailedRouter::debugPlotDRBox(DRBox& dr_box, std::string flag)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& dr_temp_directory_path = RTDM.getConfig().dr_temp_directory_path;

  PlanarRect box_real_rect = dr_box.get_box_rect().get_real_rect();

  int32_t point_size = 5;

  GPGDS gp_gds;

  // base_region
  {
    GPStruct base_region_struct("base_region");
    GPBoundary gp_boundary;
    gp_boundary.set_layer_idx(0);
    gp_boundary.set_data_type(0);
    gp_boundary.set_rect(box_real_rect);
    base_region_struct.push(gp_boundary);
    gp_gds.addStruct(base_region_struct);
  }

  // gcell_axis
  {
    GPStruct gcell_axis_struct("gcell_axis");
    for (int32_t x : RTUTIL.getScaleList(box_real_rect.get_ll_x(), box_real_rect.get_ur_x(), gcell_axis.get_x_grid_list())) {
      GPPath gp_path;
      gp_path.set_layer_idx(0);
      gp_path.set_data_type(1);
      gp_path.set_segment(x, box_real_rect.get_ll_y(), x, box_real_rect.get_ur_y());
      gcell_axis_struct.push(gp_path);
    }
    for (int32_t y : RTUTIL.getScaleList(box_real_rect.get_ll_y(), box_real_rect.get_ur_y(), gcell_axis.get_y_grid_list())) {
      GPPath gp_path;
      gp_path.set_layer_idx(0);
      gp_path.set_data_type(1);
      gp_path.set_segment(box_real_rect.get_ll_x(), y, box_real_rect.get_ur_x(), y);
      gcell_axis_struct.push(gp_path);
    }
    gp_gds.addStruct(gcell_axis_struct);
  }

  // box_track_axis
  {
    GPStruct box_track_axis_struct("box_track_axis");
    PlanarCoord& real_ll = box_real_rect.get_ll();
    PlanarCoord& real_ur = box_real_rect.get_ur();
    ScaleAxis& box_track_axis = dr_box.get_box_track_axis();
    std::vector<int32_t> x_list = RTUTIL.getScaleList(real_ll.get_x(), real_ur.get_x(), box_track_axis.get_x_grid_list());
    std::vector<int32_t> y_list = RTUTIL.getScaleList(real_ll.get_y(), real_ur.get_y(), box_track_axis.get_y_grid_list());
    for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(routing_layer_list.size()); layer_idx++) {
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
  }

  // fixed_rect
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

  // access_point
  for (auto& [net_idx, access_point_set] : dr_box.get_net_access_point_map()) {
    GPStruct access_point_struct(RTUTIL.getString("access_point(net_", net_idx, ")"));
    for (AccessPoint* access_point : access_point_set) {
      int32_t x = access_point->get_real_x();
      int32_t y = access_point->get_real_y();

      GPBoundary access_point_boundary;
      access_point_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(access_point->get_layer_idx()));
      access_point_boundary.set_data_type(static_cast<int32_t>(GPDataType::kAccessPoint));
      access_point_boundary.set_rect(x - point_size, y - point_size, x + point_size, y + point_size);
      access_point_struct.push(access_point_boundary);
    }
    gp_gds.addStruct(access_point_struct);
  }

  // net_detailed_result
  for (auto& [net_idx, segment_set] : dr_box.get_net_detailed_result_map()) {
    GPStruct detailed_result_struct(RTUTIL.getString("detailed_result(net_", net_idx, ")"));
    for (Segment<LayerCoord>* segment : segment_set) {
      for (NetShape& net_shape : RTDM.getNetDetailedShapeList(net_idx, *segment)) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShape));
        gp_boundary.set_rect(net_shape.get_rect());
        if (net_shape.get_is_routing()) {
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(net_shape.get_layer_idx()));
        } else {
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(net_shape.get_layer_idx()));
        }
        detailed_result_struct.push(gp_boundary);
      }
    }
    gp_gds.addStruct(detailed_result_struct);
  }

  // net_detailed_patch
  for (auto& [net_idx, patch_set] : dr_box.get_net_detailed_patch_map()) {
    GPStruct detailed_patch_struct(RTUTIL.getString("detailed_patch(net_", net_idx, ")"));
    for (EXTLayerRect* patch : patch_set) {
      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShape));
      gp_boundary.set_rect(patch->get_real_rect());
      gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(patch->get_layer_idx()));
      detailed_patch_struct.push(gp_boundary);
    }
    gp_gds.addStruct(detailed_patch_struct);
  }

  // layer_node_map
  {
    std::vector<GridMap<DRNode>>& layer_node_map = dr_box.get_layer_node_map();
    // dr_node_map
    {
      GPStruct dr_node_map_struct("dr_node_map");
      for (GridMap<DRNode>& dr_node_map : layer_node_map) {
        for (int32_t grid_x = 0; grid_x < dr_node_map.get_x_size(); grid_x++) {
          for (int32_t grid_y = 0; grid_y < dr_node_map.get_y_size(); grid_y++) {
            DRNode& dr_node = dr_node_map[grid_x][grid_y];
            PlanarRect real_rect = RTUTIL.getEnlargedRect(dr_node.get_planar_coord(), point_size);
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
            gp_text_node_real_coord.set_message(RTUTIL.getString("(", dr_node.get_x(), " , ", dr_node.get_y(), " , ", dr_node.get_layer_idx(), ")"));
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
                orient_violation_number_map_info_message += RTUTIL.getString("(", GetOrientationName()(orient), ",", violation_number != 0, ")");
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
    }

    // neighbor_map
    {
      GPStruct neighbor_map_struct("neighbor_map");
      for (GridMap<DRNode>& dr_node_map : layer_node_map) {
        for (int32_t grid_x = 0; grid_x < dr_node_map.get_x_size(); grid_x++) {
          for (int32_t grid_y = 0; grid_y < dr_node_map.get_y_size(); grid_y++) {
            DRNode& dr_node = dr_node_map[grid_x][grid_y];
            PlanarRect real_rect = RTUTIL.getEnlargedRect(dr_node.get_planar_coord(), point_size);

            int32_t ll_x = real_rect.get_ll_x();
            int32_t ll_y = real_rect.get_ll_y();
            int32_t ur_x = real_rect.get_ur_x();
            int32_t ur_y = real_rect.get_ur_y();
            int32_t mid_x = (ll_x + ur_x) / 2;
            int32_t mid_y = (ll_y + ur_y) / 2;
            int32_t x_reduced_span = (ur_x - ll_x) / 4;
            int32_t y_reduced_span = (ur_y - ll_y) / 4;

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
              gp_path.set_width(std::min(x_reduced_span, y_reduced_span) / 2);
              gp_path.set_data_type(static_cast<int32_t>(GPDataType::kNeighbor));
              neighbor_map_struct.push(gp_path);
            }
          }
        }
      }
      gp_gds.addStruct(neighbor_map_struct);
    }
  }

  // layer_shadow_map
  {
    std::vector<DRShadow>& layer_shadow_map = dr_box.get_layer_shadow_map();
    for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(routing_layer_list.size()); layer_idx++) {
      DRShadow& dr_shadow = layer_shadow_map[layer_idx];

      for (auto& [net_idx, rect_set] : dr_shadow.get_net_fixed_rect_map()) {
        GPStruct fixed_rect_struct(RTUTIL.getString("shadow_fixed_rect(net_", net_idx, ")"));
        for (const PlanarRect& rect : rect_set) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShadow));
          gp_boundary.set_rect(rect);
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(layer_idx));
          fixed_rect_struct.push(gp_boundary);
        }
        gp_gds.addStruct(fixed_rect_struct);
      }

      for (auto& [net_idx, rect_set] : dr_shadow.get_net_routed_rect_map()) {
        GPStruct routed_rect_struct(RTUTIL.getString("shadow_routed_rect(net_", net_idx, ")"));
        for (const PlanarRect& rect : rect_set) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShadow));
          gp_boundary.set_rect(rect);
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(layer_idx));
          routed_rect_struct.push(gp_boundary);
        }
        gp_gds.addStruct(routed_rect_struct);
      }

      GPStruct violation_struct(RTUTIL.getString("shadow_violation"));
      for (const PlanarRect& rect : dr_shadow.get_violation_set()) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShadow));
        gp_boundary.set_rect(rect);
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(layer_idx));
        violation_struct.push(gp_boundary);
      }
      gp_gds.addStruct(violation_struct);
    }
  }

  // task
  for (DRTask* dr_task : dr_box.get_dr_task_list()) {
    GPStruct task_struct(RTUTIL.getString("task(net_", dr_task->get_net_idx(), ")"));

    for (DRGroup& dr_group : dr_task->get_dr_group_list()) {
      for (auto& [coord, _] : dr_group.get_coord_direction_map()) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kKey));
        gp_boundary.set_rect(RTUTIL.getEnlargedRect(coord, point_size));
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(coord.get_layer_idx()));
        task_struct.push(gp_boundary);
      }
    }
    {
      // bounding_box
      GPBoundary gp_boundary;
      gp_boundary.set_layer_idx(0);
      gp_boundary.set_data_type(2);
      gp_boundary.set_rect(dr_task->get_bounding_box());
      task_struct.push(gp_boundary);
    }
    for (Segment<LayerCoord>& segment : dr_box.get_net_task_detailed_result_map()[dr_task->get_net_idx()]) {
      for (NetShape& net_shape : RTDM.getNetDetailedShapeList(dr_task->get_net_idx(), segment)) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kDetailedPath));
        gp_boundary.set_rect(net_shape.get_rect());
        if (net_shape.get_is_routing()) {
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(net_shape.get_layer_idx()));
        } else {
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(net_shape.get_layer_idx()));
        }
        task_struct.push(gp_boundary);
      }
    }
    for (EXTLayerRect& patch : dr_box.get_net_task_detailed_patch_map()[dr_task->get_net_idx()]) {
      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPatch));
      gp_boundary.set_rect(patch.get_real_rect());
      gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(patch.get_layer_idx()));
      task_struct.push(gp_boundary);
    }
    gp_gds.addStruct(task_struct);
  }

  // violation
  {
    for (Violation& violation : dr_box.get_route_violation_list()) {
      GPStruct violation_struct(RTUTIL.getString("violation_", GetViolationTypeName()(violation.get_violation_type())));
      EXTLayerRect& violation_shape = violation.get_violation_shape();

      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kRouteViolation));
      gp_boundary.set_rect(violation_shape.get_real_rect());
      if (violation.get_is_routing()) {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(violation_shape.get_layer_idx()));
      } else {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(violation_shape.get_layer_idx()));
      }
      violation_struct.push(gp_boundary);
      gp_gds.addStruct(violation_struct);
    }
    for (Violation& violation : dr_box.get_patch_violation_list()) {
      GPStruct violation_struct(RTUTIL.getString("violation_", GetViolationTypeName()(violation.get_violation_type())));
      EXTLayerRect& violation_shape = violation.get_violation_shape();

      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPatchViolation));
      gp_boundary.set_rect(violation_shape.get_real_rect());
      if (violation.get_is_routing()) {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(violation_shape.get_layer_idx()));
      } else {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(violation_shape.get_layer_idx()));
      }
      violation_struct.push(gp_boundary);
      gp_gds.addStruct(violation_struct);
    }
  }

  std::string gds_file_path
      = RTUTIL.getString(dr_temp_directory_path, flag, "_dr_box_", dr_box.get_dr_box_id().get_x(), "_", dr_box.get_dr_box_id().get_y(), ".gds");
  RTGP.plot(gp_gds, gds_file_path);
}

#endif

}  // namespace irt
