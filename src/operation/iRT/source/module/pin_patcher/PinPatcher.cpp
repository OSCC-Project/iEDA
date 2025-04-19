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
#include "PinPatcher.hpp"

#include "DRCEngine.hpp"
#include "GDSPlotter.hpp"
#include "PPPatch.hpp"
#include "RTInterface.hpp"
#include "Utility.hpp"

namespace irt {

// public

void PinPatcher::initInst()
{
  if (_pp_instance == nullptr) {
    _pp_instance = new PinPatcher();
  }
}

PinPatcher& PinPatcher::getInst()
{
  if (_pp_instance == nullptr) {
    RTLOG.error(Loc::current(), "The instance not initialized!");
  }
  return *_pp_instance;
}

void PinPatcher::destroyInst()
{
  if (_pp_instance != nullptr) {
    delete _pp_instance;
    _pp_instance = nullptr;
  }
}

// function

void PinPatcher::patch()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");
  PPModel pp_model = initPPModel();
  uploadViolation(pp_model);
  updateSummary(pp_model);
  printSummary(pp_model);
  iterativePPModel(pp_model);
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

// private

PinPatcher* PinPatcher::_pp_instance = nullptr;

PPModel PinPatcher::initPPModel()
{
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();

  PPModel pp_model;
  pp_model.set_pp_net_list(convertToPPNetList(net_list));
  return pp_model;
}

std::vector<PPNet> PinPatcher::convertToPPNetList(std::vector<Net>& net_list)
{
  std::vector<PPNet> pp_net_list;
  pp_net_list.reserve(net_list.size());
  for (Net& net : net_list) {
    pp_net_list.emplace_back(convertToPPNet(net));
  }
  return pp_net_list;
}

PPNet PinPatcher::convertToPPNet(Net& net)
{
  PPNet pp_net;
  pp_net.set_origin_net(&net);
  pp_net.set_net_idx(net.get_net_idx());
  pp_net.set_connect_type(net.get_connect_type());
  for (Pin& pin : net.get_pin_list()) {
    pp_net.get_pp_pin_list().push_back(PPPin(pin));
  }
  return pp_net;
}

void PinPatcher::uploadViolation(PPModel& pp_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();

  for (Violation* violation : RTDM.getViolationSet(die)) {
    RTDM.updateViolationToGCellMap(ChangeType::kDel, violation);
  }
  for (Violation violation : getViolationList(pp_model)) {
    RTDM.updateViolationToGCellMap(ChangeType::kAdd, new Violation(violation));
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

std::vector<Violation> PinPatcher::getViolationList(PPModel& pp_model)
{
  Die& die = RTDM.getDatabase().get_die();

  DETask de_task;
  {
    std::string top_name = RTUTIL.getString("pp_model");
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
    for (auto& [net_idx, pin_access_result_map] : RTDM.getNetPinAccessResultMap(die)) {
      for (auto& [pin_idx, segment_set] : pin_access_result_map) {
        for (Segment<LayerCoord>* segment : segment_set) {
          net_result_map[net_idx].push_back(segment);
        }
      }
    }
    std::map<int32_t, std::vector<EXTLayerRect*>> net_patch_map;
    for (auto& [net_idx, patch_set] : RTDM.getNetAccessPatchMap(die)) {
      for (EXTLayerRect* patch : patch_set) {
        net_patch_map[net_idx].emplace_back(patch);
      }
    }
    std::set<int32_t> need_checked_net_set;
    for (PPNet& pp_net : pp_model.get_pp_net_list()) {
      need_checked_net_set.insert(pp_net.get_net_idx());
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

void PinPatcher::iterativePPModel(PPModel& pp_model)
{
  int32_t cost_unit = RTDM.getOnlyPitch();
  double prefer_wire_unit = 1;
  double non_prefer_wire_unit = 2.5 * prefer_wire_unit;
  double via_unit = cost_unit;
  double fixed_rect_unit = 4 * non_prefer_wire_unit * cost_unit;
  double routed_rect_unit = 2 * via_unit;
  /**
   * size, offset, schedule_interval, fixed_rect_unit, routed_rect_unit
   */
  std::vector<PPIterParam> pp_iter_param_list;
  // clang-format off
  pp_iter_param_list.emplace_back(3, 0, 3, fixed_rect_unit, routed_rect_unit);
  pp_iter_param_list.emplace_back(3, 1, 3, fixed_rect_unit, routed_rect_unit);
  pp_iter_param_list.emplace_back(3, 2, 3, fixed_rect_unit, routed_rect_unit);
  pp_iter_param_list.emplace_back(3, 0, 3, fixed_rect_unit, routed_rect_unit);
  pp_iter_param_list.emplace_back(3, 1, 3, fixed_rect_unit, routed_rect_unit);
  pp_iter_param_list.emplace_back(3, 2, 3, fixed_rect_unit, routed_rect_unit);
  // clang-format on
  for (int32_t i = 0, iter = 1; i < static_cast<int32_t>(pp_iter_param_list.size()); i++, iter++) {
    Monitor iter_monitor;
    RTLOG.info(Loc::current(), "***** Begin iteration ", iter, "/", pp_iter_param_list.size(), "(", RTUTIL.getPercentage(iter, pp_iter_param_list.size()),
               ") *****");
    // debugPlotPPModel(pp_model, "before");
    setPPIterParam(pp_model, iter, pp_iter_param_list[i]);
    initPPBoxMap(pp_model);
    buildBoxSchedule(pp_model);
    // debugPlotPPModel(pp_model, "middle");
    routePPBoxMap(pp_model);
    uploadAccessPatch(pp_model);
    uploadViolation(pp_model);
    // debugPlotPPModel(pp_model, "after");
    updateSummary(pp_model);
    printSummary(pp_model);
    outputNetCSV(pp_model);
    outputViolationCSV(pp_model);
    RTLOG.info(Loc::current(), "***** End Iteration ", iter, "/", pp_iter_param_list.size(), "(", RTUTIL.getPercentage(iter, pp_iter_param_list.size()), ")",
               iter_monitor.getStatsInfo(), "*****");
    if (stopIteration(pp_model)) {
      break;
    }
  }
}

void PinPatcher::setPPIterParam(PPModel& pp_model, int32_t iter, PPIterParam& pp_iter_param)
{
  pp_model.set_iter(iter);
  RTLOG.info(Loc::current(), "size: ", pp_iter_param.get_size());
  RTLOG.info(Loc::current(), "offset: ", pp_iter_param.get_offset());
  RTLOG.info(Loc::current(), "schedule_interval: ", pp_iter_param.get_schedule_interval());
  pp_model.set_pp_iter_param(pp_iter_param);
}

void PinPatcher::initPPBoxMap(PPModel& pp_model)
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

  PPIterParam& pp_iter_param = pp_model.get_pp_iter_param();
  int32_t size = pp_iter_param.get_size();
  int32_t offset = pp_iter_param.get_offset();
  int32_t x_box_num = static_cast<int32_t>(std::ceil((x_gcell_num - offset) / 1.0 / size));
  int32_t y_box_num = static_cast<int32_t>(std::ceil((y_gcell_num - offset) / 1.0 / size));

  GridMap<PPBox>& pp_box_map = pp_model.get_pp_box_map();
  pp_box_map.init(x_box_num, y_box_num);

  for (int32_t x = 0; x < pp_box_map.get_x_size(); x++) {
    for (int32_t y = 0; y < pp_box_map.get_y_size(); y++) {
      int32_t grid_ll_x = std::max(offset + x * size, 0);
      int32_t grid_ll_y = std::max(offset + y * size, 0);
      int32_t grid_ur_x = std::min(offset + (x + 1) * size - 1, x_gcell_num - 1);
      int32_t grid_ur_y = std::min(offset + (y + 1) * size - 1, y_gcell_num - 1);

      PlanarRect ll_gcell_rect = RTUTIL.getRealRectByGCell(PlanarCoord(grid_ll_x, grid_ll_y), gcell_axis);
      PlanarRect ur_gcell_rect = RTUTIL.getRealRectByGCell(PlanarCoord(grid_ur_x, grid_ur_y), gcell_axis);
      PlanarRect box_real_rect(ll_gcell_rect.get_ll(), ur_gcell_rect.get_ur());

      PPBox& pp_box = pp_box_map[x][y];

      EXTPlanarRect pp_box_rect;
      pp_box_rect.set_real_rect(box_real_rect);
      pp_box_rect.set_grid_rect(RTUTIL.getOpenGCellGridRect(box_real_rect, gcell_axis));
      pp_box.set_box_rect(pp_box_rect);
      PPBoxId pp_box_id;
      pp_box_id.set_x(x);
      pp_box_id.set_y(y);
      pp_box.set_pp_box_id(pp_box_id);
      pp_box.set_pp_iter_param(&pp_iter_param);
    }
  }
}

void PinPatcher::buildBoxSchedule(PPModel& pp_model)
{
  GridMap<PPBox>& pp_box_map = pp_model.get_pp_box_map();
  int32_t schedule_interval = pp_model.get_pp_iter_param().get_schedule_interval();

  std::vector<std::vector<PPBoxId>> pp_box_id_list_list;
  for (int32_t start_x = 0; start_x < schedule_interval; start_x++) {
    for (int32_t start_y = 0; start_y < schedule_interval; start_y++) {
      std::vector<PPBoxId> pp_box_id_list;
      for (int32_t x = start_x; x < pp_box_map.get_x_size(); x += schedule_interval) {
        for (int32_t y = start_y; y < pp_box_map.get_y_size(); y += schedule_interval) {
          pp_box_id_list.emplace_back(x, y);
        }
      }
      pp_box_id_list_list.push_back(pp_box_id_list);
    }
  }
  pp_model.set_pp_box_id_list_list(pp_box_id_list_list);
}

void PinPatcher::routePPBoxMap(PPModel& pp_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<PPBox>& pp_box_map = pp_model.get_pp_box_map();

  size_t total_box_num = 0;
  for (std::vector<PPBoxId>& pp_box_id_list : pp_model.get_pp_box_id_list_list()) {
    total_box_num += pp_box_id_list.size();
  }

  size_t routed_box_num = 0;
  for (std::vector<PPBoxId>& pp_box_id_list : pp_model.get_pp_box_id_list_list()) {
    Monitor stage_monitor;
#pragma omp parallel for
    for (PPBoxId& pp_box_id : pp_box_id_list) {
      PPBox& pp_box = pp_box_map[pp_box_id.get_x()][pp_box_id.get_y()];
      buildFixedRect(pp_box);
      buildAccessResult(pp_box);
      buildAccessPatch(pp_box);
      buildViolation(pp_box);
      if (needRouting(pp_box)) {
        buildGraphShapeMap(pp_box);
        // debugCheckPPBox(pp_box);
        routePPBox(pp_box);
      }
      uploadAccessPatch(pp_box);
      freePPBox(pp_box);
    }
    routed_box_num += pp_box_id_list.size();
    RTLOG.info(Loc::current(), "Routed ", routed_box_num, "/", total_box_num, "(", RTUTIL.getPercentage(routed_box_num, total_box_num), ")",
               stage_monitor.getStatsInfo());
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void PinPatcher::buildFixedRect(PPBox& pp_box)
{
  pp_box.set_type_layer_net_fixed_rect_map(RTDM.getTypeLayerNetFixedRectMap(pp_box.get_box_rect()));
}

void PinPatcher::buildAccessResult(PPBox& pp_box)
{
  pp_box.set_net_pin_access_result_map(RTDM.getNetPinAccessResultMap(pp_box.get_box_rect()));
}

void PinPatcher::buildAccessPatch(PPBox& pp_box)
{
  pp_box.set_net_access_patch_map(RTDM.getNetAccessPatchMap(pp_box.get_box_rect()));
}

void PinPatcher::buildViolation(PPBox& pp_box)
{
  pp_box.set_violation_list(getViolationList(pp_box));
}

bool PinPatcher::needRouting(PPBox& pp_box)
{
  int32_t valid_violation_num = 0;
  for (Violation& violation : pp_box.get_violation_list()) {
    if (!isValid(pp_box, violation)) {
      continue;
    }
    valid_violation_num++;
  }
  if (valid_violation_num == 0) {
    return false;
  }
  return true;
}

bool PinPatcher::isValid(PPBox& pp_box, Violation& violation)
{
  PlanarRect& box_real_rect = pp_box.get_box_rect().get_real_rect();

  bool is_valid = true;
  if (!RTUTIL.isInside(box_real_rect, violation.get_violation_shape().get_real_rect())) {
    is_valid = false;
  }
  if (violation.get_violation_type() != ViolationType::kMinimumArea) {
    is_valid = false;
  }
  return is_valid;
}

void PinPatcher::buildGraphShapeMap(PPBox& pp_box)
{
  for (auto& [is_routing, layer_net_fixed_rect_map] : pp_box.get_type_layer_net_fixed_rect_map()) {
    for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
      for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
        for (auto& fixed_rect : fixed_rect_set) {
          updateFixedRectToGraph(pp_box, ChangeType::kAdd, net_idx, fixed_rect, is_routing);
        }
      }
    }
  }
  for (auto& [net_idx, pin_access_result_map] : pp_box.get_net_pin_access_result_map()) {
    for (auto& [pin_idx, segment_set] : pin_access_result_map) {
      for (Segment<LayerCoord>* segment : segment_set) {
        updateFixedRectToGraph(pp_box, ChangeType::kAdd, net_idx, *segment);
      }
    }
  }
  for (auto& [net_idx, patch_set] : pp_box.get_net_access_patch_map()) {
    for (EXTLayerRect* patch : patch_set) {
      updateFixedRectToGraph(pp_box, ChangeType::kAdd, net_idx, patch, true);
    }
  }
}

void PinPatcher::routePPBox(PPBox& pp_box)
{
  while (true) {
    initSingleTask(pp_box);
    if (pp_box.get_curr_net_idx() == -1) {
      break;
    }
    // debugPlotPPBox(pp_box, "before");
    for (PPSolution& pp_solution : getSolution(pp_box)) {
      updateCurrViolationList(pp_box, pp_solution);
      updateCurrSolvedStatus(pp_box);
      // debugPlotPPBox(pp_box, "after");
      if (pp_box.get_curr_is_solved()) {
        updateTaskPatch(pp_box);
        updateViolationList(pp_box);
        break;
      }
    }
    pp_box.get_tried_fix_violation_set().insert(pp_box.get_curr_violation());
    resetSingleTask(pp_box);
  }
}

void PinPatcher::initSingleTask(PPBox& pp_box)
{
  for (Violation& violation : pp_box.get_violation_list()) {
    if (!isValid(pp_box, violation)) {
      continue;
    }
    if (RTUTIL.exist(pp_box.get_tried_fix_violation_set(), violation)) {
      continue;
    }
    int32_t net_idx = *violation.get_violation_net_set().begin();
    pp_box.set_curr_net_idx(net_idx);
    pp_box.set_curr_violation(violation);
    pp_box.set_curr_routing_patch_list(pp_box.get_net_task_access_patch_map()[net_idx]);
    break;
  }
}

std::vector<PPSolution> PinPatcher::getSolution(PPBox& pp_box)
{
  int32_t manufacture_grid = RTDM.getDatabase().get_manufacture_grid();
  Die& die = RTDM.getDatabase().get_die();
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  int32_t curr_net_idx = pp_box.get_curr_net_idx();
  EXTLayerRect& violation_shape = pp_box.get_curr_violation().get_violation_shape();
  PlanarRect violation_real_rect = violation_shape.get_real_rect();
  int32_t violation_layer_idx = violation_shape.get_layer_idx();

  RoutingLayer& routing_layer = routing_layer_list[violation_layer_idx];
  Direction layer_direction = routing_layer.get_prefer_direction();
  int32_t min_area = routing_layer.get_min_area();
  int32_t wire_width = routing_layer.get_min_width();

  GTLPolyInt gtl_poly;
  {
    GTLPolySetInt gtl_poly_set;
    for (EXTLayerRect* fixed_rect : pp_box.get_type_layer_net_fixed_rect_map()[true][violation_layer_idx][curr_net_idx]) {
      if (RTUTIL.isClosedOverlap(violation_real_rect, fixed_rect->get_real_rect())) {
        gtl_poly_set += RTUTIL.convertToGTLRectInt(fixed_rect->get_real_rect());
      }
    }
    for (auto& [pin_idx, segment_set] : pp_box.get_net_pin_access_result_map()[curr_net_idx]) {
      for (Segment<LayerCoord>* segment : segment_set) {
        for (NetShape& net_shape : RTDM.getNetShapeList(curr_net_idx, *segment)) {
          if (!net_shape.get_is_routing()) {
            continue;
          }
          if (violation_layer_idx == net_shape.get_layer_idx() && RTUTIL.isClosedOverlap(violation_real_rect, net_shape.get_rect())) {
            gtl_poly_set += RTUTIL.convertToGTLRectInt(net_shape.get_rect());
          }
        }
      }
    }
    for (auto& [net_idx, patch_set] : pp_box.get_net_access_patch_map()) {
      for (EXTLayerRect* patch : patch_set) {
        if (violation_layer_idx == patch->get_layer_idx() && RTUTIL.isClosedOverlap(violation_real_rect, patch->get_real_rect())) {
          gtl_poly_set += RTUTIL.convertToGTLRectInt(patch->get_real_rect());
        }
      }
    }
    for (EXTLayerRect& patch : pp_box.get_curr_routing_patch_list()) {
      if (violation_layer_idx == patch.get_layer_idx() && RTUTIL.isClosedOverlap(violation_real_rect, patch.get_real_rect())) {
        gtl_poly_set += RTUTIL.convertToGTLRectInt(patch.get_real_rect());
      }
    }
    std::vector<GTLPolyInt> gtl_poly_list;
    gtl_poly_set.get_polygons(gtl_poly_list);
    if (gtl_poly_list.size() != 1) {
      RTLOG.error(Loc::current(), "The violation poly size != 1!");
    }
    gtl_poly = gtl_poly_list.front();
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
  std::vector<PPPatch> pp_patch_list;
  {
    int32_t h_wire_length = (min_area - v_cutting_rect.getArea()) / wire_width + v_cutting_rect.getXSpan();
    while (h_wire_length % manufacture_grid != 0) {
      h_wire_length++;
    }
    for (int32_t y : {h_cutting_rect.get_ll_y(), v_cutting_rect.get_ll_y(), v_cutting_rect.get_ur_y() - wire_width}) {
      for (int32_t x = v_cutting_rect.get_ur_x() - h_wire_length; x <= v_cutting_rect.get_ll_x(); x += manufacture_grid) {
        PlanarRect h_real_rect = RTUTIL.getEnlargedRect(PlanarCoord(x, y), 0, 0, h_wire_length, wire_width);
        if (!RTUTIL.isInside(die.get_real_rect(), h_real_rect)) {
          continue;
        }
        pp_patch_list.emplace_back(h_real_rect, violation_layer_idx);
      }
    }
    int32_t v_wire_length = (min_area - h_cutting_rect.getArea()) / wire_width + h_cutting_rect.getYSpan();
    while (v_wire_length % manufacture_grid != 0) {
      v_wire_length++;
    }
    for (int32_t x : {v_cutting_rect.get_ll_x(), h_cutting_rect.get_ll_x(), h_cutting_rect.get_ur_x() - wire_width}) {
      for (int32_t y = h_cutting_rect.get_ur_y() - v_wire_length; y <= h_cutting_rect.get_ll_y(); y += manufacture_grid) {
        PlanarRect v_real_rect = RTUTIL.getEnlargedRect(PlanarCoord(x, y), 0, 0, wire_width, v_wire_length);
        if (!RTUTIL.isInside(die.get_real_rect(), v_real_rect)) {
          continue;
        }
        pp_patch_list.emplace_back(v_real_rect, violation_layer_idx);
      }
    }
    for (PPPatch& pp_patch : pp_patch_list) {
      EXTLayerRect& patch = pp_patch.get_patch();
      patch.set_grid_rect(RTUTIL.getClosedGCellGridRect(patch.get_real_rect(), gcell_axis));
      pp_patch.set_direction(patch.get_real_rect().getRectDirection(layer_direction));
      pp_patch.set_overlap_area(gtl::area(gtl_poly & RTUTIL.convertToGTLRectInt(patch.get_real_rect())));
      pp_patch.set_env_cost(getEnvCost(pp_box, curr_net_idx, patch));
    }
    std::sort(pp_patch_list.begin(), pp_patch_list.end(), [&layer_direction](PPPatch& a, PPPatch& b) { return CmpPPPatch()(a, b, layer_direction); });
  }
  std::vector<PPSolution> pp_solution_list;
  for (PPPatch& pp_patch : pp_patch_list) {
    if (pp_patch.get_env_cost() > 0) {
      continue;
    }
    PPSolution pp_solution = getNewSolution(pp_box);
    pp_solution.get_routing_patch_list().push_back(pp_patch.get_patch());
    pp_solution_list.push_back(pp_solution);
  }
  return pp_solution_list;
}

PPSolution PinPatcher::getNewSolution(PPBox& pp_box)
{
  PPSolution pp_solution;
  pp_solution.set_routing_patch_list(pp_box.get_curr_routing_patch_list());
  return pp_solution;
}

void PinPatcher::updateCurrViolationList(PPBox& pp_box, PPSolution& pp_solution)
{
  pp_box.set_curr_routing_patch_list(pp_solution.get_routing_patch_list());
  pp_box.set_curr_violation_list(getViolationList(pp_box));
}

std::vector<Violation> PinPatcher::getViolationList(PPBox& pp_box)
{
  std::string top_name = RTUTIL.getString("pp_box_", pp_box.get_pp_box_id().get_x(), "_", pp_box.get_pp_box_id().get_y());
  std::vector<std::pair<EXTLayerRect*, bool>> env_shape_list;
  std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>> net_pin_shape_map;
  for (auto& [is_routing, layer_net_fixed_rect_map] : pp_box.get_type_layer_net_fixed_rect_map()) {
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
  for (auto& [net_idx, pin_access_result_map] : pp_box.get_net_pin_access_result_map()) {
    for (auto& [pin_idx, segment_set] : pin_access_result_map) {
      for (Segment<LayerCoord>* segment : segment_set) {
        net_result_map[net_idx].push_back(segment);
      }
    }
  }
  std::map<int32_t, std::vector<EXTLayerRect*>> net_patch_map;
  for (auto& [net_idx, patch_set] : pp_box.get_net_access_patch_map()) {
    for (EXTLayerRect* patch : patch_set) {
      net_patch_map[net_idx].push_back(patch);
    }
  }
  for (auto& [net_idx, patch_list] : pp_box.get_net_task_access_patch_map()) {
    if (net_idx == pp_box.get_curr_net_idx()) {
      for (EXTLayerRect& patch : pp_box.get_curr_routing_patch_list()) {
        net_patch_map[net_idx].emplace_back(&patch);
      }
    } else {
      for (EXTLayerRect& patch : patch_list) {
        net_patch_map[net_idx].emplace_back(&patch);
      }
    }
  }
  std::set<int32_t> need_checked_net_set;
  for (auto& [net_idx, pin_shape_list] : net_pin_shape_map) {
    need_checked_net_set.insert(net_idx);
  }
  for (auto& [net_idx, segment_list] : net_result_map) {
    need_checked_net_set.insert(net_idx);
  }
  for (auto& [net_idx, patch_set] : net_patch_map) {
    need_checked_net_set.insert(net_idx);
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

void PinPatcher::updateCurrSolvedStatus(PPBox& pp_box)
{
  std::map<ViolationType, std::pair<int32_t, int32_t>> env_type_origin_curr_map;
  std::map<ViolationType, std::pair<int32_t, int32_t>> valid_type_origin_curr_map;
  for (Violation& violation : pp_box.get_violation_list()) {
    if (!isValid(pp_box, violation)) {
      env_type_origin_curr_map[violation.get_violation_type()].first++;
    } else {
      valid_type_origin_curr_map[violation.get_violation_type()].first++;
    }
  }
  for (Violation& curr_violation : pp_box.get_curr_violation_list()) {
    if (!isValid(pp_box, curr_violation)) {
      env_type_origin_curr_map[curr_violation.get_violation_type()].second++;
    } else {
      valid_type_origin_curr_map[curr_violation.get_violation_type()].second++;
    }
  }
  bool is_solved = true;
  for (auto& [violation_type, origin_curr] : env_type_origin_curr_map) {
    if (!is_solved) {
      break;
    }
    is_solved = origin_curr.second <= origin_curr.first;
  }
  for (auto& [violation_type, origin_curr] : valid_type_origin_curr_map) {
    if (!is_solved) {
      break;
    }
    is_solved = origin_curr.second < origin_curr.first;
  }
  pp_box.set_curr_is_solved(is_solved);
}

void PinPatcher::updateTaskPatch(PPBox& pp_box)
{
  std::vector<EXTLayerRect> new_routing_patch_list = pp_box.get_curr_routing_patch_list();

  int32_t curr_net_idx = pp_box.get_curr_net_idx();
  std::vector<EXTLayerRect>& routing_patch_list = pp_box.get_net_task_access_patch_map()[curr_net_idx];

  // 原结果从graph删除,由于task有对应net_idx,所以不需要在布线前进行删除也不会影响结果
  for (EXTLayerRect& routing_patch : routing_patch_list) {
    updateRoutedRectToGraph(pp_box, ChangeType::kDel, curr_net_idx, routing_patch);
  }
  routing_patch_list = new_routing_patch_list;
  // 新结果添加到graph
  for (EXTLayerRect& routing_patch : routing_patch_list) {
    updateRoutedRectToGraph(pp_box, ChangeType::kAdd, curr_net_idx, routing_patch);
  }
}

void PinPatcher::updateViolationList(PPBox& pp_box)
{
  pp_box.set_violation_list(pp_box.get_curr_violation_list());
}

void PinPatcher::resetSingleTask(PPBox& pp_box)
{
  pp_box.set_curr_net_idx(-1);
  pp_box.set_curr_violation(Violation());
  pp_box.get_curr_routing_patch_list().clear();
  pp_box.get_curr_violation_list().clear();
  pp_box.set_curr_is_solved(false);
}

void PinPatcher::uploadAccessPatch(PPBox& pp_box)
{
  for (auto& [net_idx, patch_list] : pp_box.get_net_task_access_patch_map()) {
    for (EXTLayerRect& patch : patch_list) {
      RTDM.updateNetAccessPatchToGCellMap(ChangeType::kAdd, net_idx, new EXTLayerRect(patch));
    }
  }
}

void PinPatcher::freePPBox(PPBox& pp_box)
{
  pp_box.get_graph_routing_net_fixed_rect_map().clear();
  pp_box.get_graph_routing_net_routed_rect_map().clear();
}

int32_t PinPatcher::getViolationNum(PPModel& pp_model)
{
  Die& die = RTDM.getDatabase().get_die();

  return static_cast<int32_t>(RTDM.getViolationSet(die).size());
}

void PinPatcher::uploadAccessPatch(PPModel& pp_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();

  std::map<int32_t, std::set<EXTLayerRect*>> net_access_patch_map = RTDM.getNetAccessPatchMap(die);
  std::map<int32_t, std::map<int32_t, std::set<Segment<LayerCoord>*>>> net_pin_access_result_map = RTDM.getNetPinAccessResultMap(die);
  for (auto& [net_idx, patch_set] : net_access_patch_map) {
    std::map<int32_t, std::vector<PlanarRect>> layer_rect_map;
    for (auto& [pin_idx, segment_set] : net_pin_access_result_map[net_idx]) {
      for (Segment<LayerCoord>* segment : segment_set) {
        for (NetShape& net_shape : RTDM.getNetShapeList(net_idx, *segment)) {
          if (!net_shape.get_is_routing()) {
            continue;
          }
          layer_rect_map[net_shape.get_layer_idx()].push_back(net_shape.get_rect());
        }
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
      RTDM.updateNetAccessPatchToGCellMap(ChangeType::kDel, net_idx, del_patch);
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

bool PinPatcher::stopIteration(PPModel& pp_model)
{
  if (getViolationNum(pp_model) == 0) {
    RTLOG.info(Loc::current(), "***** Iteration stopped early *****");
    return true;
  }
  return false;
}

#if 1  // get env

double PinPatcher::getEnvCost(PPBox& pp_box, int32_t net_idx, EXTLayerRect& patch)
{
  double fixed_rect_unit = pp_box.get_pp_iter_param()->get_fixed_rect_unit();
  double routed_rect_unit = pp_box.get_pp_iter_param()->get_routed_rect_unit();

  double env_cost = 0;
  for (auto& [graph_net_idx, fixed_rect_set] : pp_box.get_graph_routing_net_fixed_rect_map()[patch.get_layer_idx()]) {
    if (net_idx == graph_net_idx) {
      continue;
    }
    for (const PlanarRect& fixed_rect : fixed_rect_set) {
      if (RTUTIL.isOpenOverlap(patch.get_real_rect(), fixed_rect)) {
        env_cost += fixed_rect_unit;
      }
    }
  }
  for (auto& [graph_net_idx, routed_rect_set] : pp_box.get_graph_routing_net_routed_rect_map()[patch.get_layer_idx()]) {
    if (net_idx == graph_net_idx) {
      continue;
    }
    for (const PlanarRect& routed_rect : routed_rect_set) {
      if (RTUTIL.isOpenOverlap(patch.get_real_rect(), routed_rect)) {
        env_cost += routed_rect_unit;
      }
    }
  }
  return env_cost;
}

#endif

#if 1  // update env

void PinPatcher::updateFixedRectToGraph(PPBox& pp_box, ChangeType change_type, int32_t net_idx, EXTLayerRect* fixed_rect, bool is_routing)
{
  NetShape net_shape(net_idx, fixed_rect->getRealLayerRect(), is_routing);
  if (!net_shape.get_is_routing()) {
    return;
  }
  for (PlanarRect& graph_shape : getGraphShape(pp_box, net_shape)) {
    if (change_type == ChangeType::kAdd) {
      pp_box.get_graph_routing_net_fixed_rect_map()[net_shape.get_layer_idx()][net_idx].insert(graph_shape);
    } else if (change_type == ChangeType::kDel) {
      pp_box.get_graph_routing_net_fixed_rect_map()[net_shape.get_layer_idx()][net_idx].erase(graph_shape);
    }
  }
}

void PinPatcher::updateFixedRectToGraph(PPBox& pp_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment)
{
  for (NetShape& net_shape : RTDM.getNetShapeList(net_idx, segment)) {
    if (!net_shape.get_is_routing()) {
      continue;
      ;
    }
    for (PlanarRect& graph_shape : getGraphShape(pp_box, net_shape)) {
      if (change_type == ChangeType::kAdd) {
        pp_box.get_graph_routing_net_fixed_rect_map()[net_shape.get_layer_idx()][net_idx].insert(graph_shape);
      } else if (change_type == ChangeType::kDel) {
        pp_box.get_graph_routing_net_fixed_rect_map()[net_shape.get_layer_idx()][net_idx].erase(graph_shape);
      }
    }
  }
}

void PinPatcher::updateRoutedRectToGraph(PPBox& pp_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment)
{
  for (NetShape& net_shape : RTDM.getNetShapeList(net_idx, segment)) {
    if (!net_shape.get_is_routing()) {
      continue;
      ;
    }
    for (PlanarRect& graph_shape : getGraphShape(pp_box, net_shape)) {
      if (change_type == ChangeType::kAdd) {
        pp_box.get_graph_routing_net_routed_rect_map()[net_shape.get_layer_idx()][net_idx].insert(graph_shape);
      } else if (change_type == ChangeType::kDel) {
        pp_box.get_graph_routing_net_routed_rect_map()[net_shape.get_layer_idx()][net_idx].erase(graph_shape);
      }
    }
  }
}

void PinPatcher::updateRoutedRectToGraph(PPBox& pp_box, ChangeType change_type, int32_t net_idx, EXTLayerRect& patch)
{
  NetShape net_shape(net_idx, patch.getRealLayerRect(), true);
  if (!net_shape.get_is_routing()) {
    return;
  }
  for (PlanarRect& graph_shape : getGraphShape(pp_box, net_shape)) {
    if (change_type == ChangeType::kAdd) {
      pp_box.get_graph_routing_net_routed_rect_map()[net_shape.get_layer_idx()][net_idx].insert(graph_shape);
    } else if (change_type == ChangeType::kDel) {
      pp_box.get_graph_routing_net_routed_rect_map()[net_shape.get_layer_idx()][net_idx].erase(graph_shape);
    }
  }
}

std::vector<PlanarRect> PinPatcher::getGraphShape(PPBox& pp_box, NetShape& net_shape)
{
  std::vector<PlanarRect> graph_shape_list;
  if (net_shape.get_is_routing()) {
    graph_shape_list = getRoutingGraphShapeList(pp_box, net_shape);
  } else {
    RTLOG.error(Loc::current(), "The type of net_shape is cut!");
  }
  return graph_shape_list;
}

std::vector<PlanarRect> PinPatcher::getRoutingGraphShapeList(PPBox& pp_box, NetShape& net_shape)
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
  std::vector<PlanarRect> graph_shape_list;
  // wire 与 net_shape
  for (auto& [x_spacing, y_spacing] : spacing_pair_list) {
    // 膨胀size为 spacing
    int32_t enlarged_x_size = x_spacing;
    int32_t enlarged_y_size = y_spacing;
    // 贴合的也不算违例
    enlarged_x_size -= 1;
    enlarged_y_size -= 1;
    graph_shape_list.push_back(RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size));
  }
  // enclosure 与 net_shape
  for (auto& [x_spacing, y_spacing] : spacing_pair_list) {
    // 膨胀size为 spacing
    int32_t enlarged_x_size = x_spacing;
    int32_t enlarged_y_size = y_spacing;
    // 贴合的也不算违例
    enlarged_x_size -= 1;
    enlarged_y_size -= 1;
    graph_shape_list.push_back(RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size));
  }
  return graph_shape_list;
}

#endif

#if 1  // exhibit

void PinPatcher::updateSummary(PPModel& pp_model)
{
  int32_t micron_dbu = RTDM.getDatabase().get_micron_dbu();
  Die& die = RTDM.getDatabase().get_die();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();
  Summary& summary = RTDM.getDatabase().get_summary();

  std::map<int32_t, double>& routing_wire_length_map = summary.iter_pp_summary_map[pp_model.get_iter()].routing_wire_length_map;
  double& total_wire_length = summary.iter_pp_summary_map[pp_model.get_iter()].total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.iter_pp_summary_map[pp_model.get_iter()].cut_via_num_map;
  int32_t& total_via_num = summary.iter_pp_summary_map[pp_model.get_iter()].total_via_num;
  std::map<int32_t, int32_t>& routing_patch_num_map = summary.iter_pp_summary_map[pp_model.get_iter()].routing_patch_num_map;
  int32_t& total_patch_num = summary.iter_pp_summary_map[pp_model.get_iter()].total_patch_num;
  std::map<int32_t, std::map<std::string, int32_t>>& within_net_routing_violation_type_num_map
      = summary.iter_pp_summary_map[pp_model.get_iter()].within_net_routing_violation_type_num_map;
  std::map<std::string, int32_t>& within_net_violation_type_num_map = summary.iter_pp_summary_map[pp_model.get_iter()].within_net_violation_type_num_map;
  std::map<int32_t, int32_t>& within_net_routing_violation_num_map = summary.iter_pp_summary_map[pp_model.get_iter()].within_net_routing_violation_num_map;
  int32_t& within_net_total_violation_num = summary.iter_pp_summary_map[pp_model.get_iter()].within_net_total_violation_num;
  std::map<int32_t, std::map<std::string, int32_t>>& among_net_routing_violation_type_num_map
      = summary.iter_pp_summary_map[pp_model.get_iter()].among_net_routing_violation_type_num_map;
  std::map<std::string, int32_t>& among_net_violation_type_num_map = summary.iter_pp_summary_map[pp_model.get_iter()].among_net_violation_type_num_map;
  std::map<int32_t, int32_t>& among_net_routing_violation_num_map = summary.iter_pp_summary_map[pp_model.get_iter()].among_net_routing_violation_num_map;
  int32_t& among_net_total_violation_num = summary.iter_pp_summary_map[pp_model.get_iter()].among_net_total_violation_num;

  routing_wire_length_map.clear();
  total_wire_length = 0;
  cut_via_num_map.clear();
  total_via_num = 0;
  routing_patch_num_map.clear();
  total_patch_num = 0;
  within_net_routing_violation_type_num_map.clear();
  within_net_violation_type_num_map.clear();
  within_net_routing_violation_num_map.clear();
  within_net_total_violation_num = 0;
  among_net_routing_violation_type_num_map.clear();
  among_net_violation_type_num_map.clear();
  among_net_routing_violation_num_map.clear();
  among_net_total_violation_num = 0;

  for (auto& [net_idx, pin_access_result_map] : RTDM.getNetPinAccessResultMap(die)) {
    for (auto& [pin_idx, segment_set] : pin_access_result_map) {
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
  }
  for (auto& [net_idx, patch_set] : RTDM.getNetAccessPatchMap(die)) {
    for (EXTLayerRect* patch : patch_set) {
      routing_patch_num_map[patch->get_layer_idx()]++;
      total_patch_num++;
    }
  }
  for (Violation* violation : RTDM.getViolationSet(die)) {
    if (violation->get_violation_net_set().size() >= 2) {
      continue;
    }
    within_net_routing_violation_type_num_map[violation->get_violation_shape().get_layer_idx()][GetViolationTypeName()(violation->get_violation_type())]++;
    within_net_violation_type_num_map[GetViolationTypeName()(violation->get_violation_type())]++;
    within_net_routing_violation_num_map[violation->get_violation_shape().get_layer_idx()]++;
    within_net_total_violation_num++;
  }
  for (Violation* violation : RTDM.getViolationSet(die)) {
    if (violation->get_violation_net_set().size() < 2) {
      continue;
    }
    among_net_routing_violation_type_num_map[violation->get_violation_shape().get_layer_idx()][GetViolationTypeName()(violation->get_violation_type())]++;
    among_net_violation_type_num_map[GetViolationTypeName()(violation->get_violation_type())]++;
    among_net_routing_violation_num_map[violation->get_violation_shape().get_layer_idx()]++;
    among_net_total_violation_num++;
  }
}

void PinPatcher::printSummary(PPModel& pp_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  Summary& summary = RTDM.getDatabase().get_summary();

  std::map<int32_t, double>& routing_wire_length_map = summary.iter_pp_summary_map[pp_model.get_iter()].routing_wire_length_map;
  double& total_wire_length = summary.iter_pp_summary_map[pp_model.get_iter()].total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.iter_pp_summary_map[pp_model.get_iter()].cut_via_num_map;
  int32_t& total_via_num = summary.iter_pp_summary_map[pp_model.get_iter()].total_via_num;
  std::map<int32_t, int32_t>& routing_patch_num_map = summary.iter_pp_summary_map[pp_model.get_iter()].routing_patch_num_map;
  int32_t& total_patch_num = summary.iter_pp_summary_map[pp_model.get_iter()].total_patch_num;
  std::map<int32_t, std::map<std::string, int32_t>>& within_net_routing_violation_type_num_map
      = summary.iter_pp_summary_map[pp_model.get_iter()].within_net_routing_violation_type_num_map;
  std::map<std::string, int32_t>& within_net_violation_type_num_map = summary.iter_pp_summary_map[pp_model.get_iter()].within_net_violation_type_num_map;
  std::map<int32_t, int32_t>& within_net_routing_violation_num_map = summary.iter_pp_summary_map[pp_model.get_iter()].within_net_routing_violation_num_map;
  int32_t& within_net_total_violation_num = summary.iter_pp_summary_map[pp_model.get_iter()].within_net_total_violation_num;
  std::map<int32_t, std::map<std::string, int32_t>>& among_net_routing_violation_type_num_map
      = summary.iter_pp_summary_map[pp_model.get_iter()].among_net_routing_violation_type_num_map;
  std::map<std::string, int32_t>& among_net_violation_type_num_map = summary.iter_pp_summary_map[pp_model.get_iter()].among_net_violation_type_num_map;
  std::map<int32_t, int32_t>& among_net_routing_violation_num_map = summary.iter_pp_summary_map[pp_model.get_iter()].among_net_routing_violation_num_map;
  int32_t& among_net_total_violation_num = summary.iter_pp_summary_map[pp_model.get_iter()].among_net_total_violation_num;

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
  fort::char_table within_net_routing_violation_map_table;
  {
    within_net_routing_violation_map_table.set_cell_text_align(fort::text_align::right);
    within_net_routing_violation_map_table << fort::header << "within_net";
    for (size_t i = 0; i < within_net_violation_type_num_map.size(); ++i) {
      within_net_routing_violation_map_table << fort::header << " ";
    }
    within_net_routing_violation_map_table << fort::header << " " << fort::endr;
    within_net_routing_violation_map_table << fort::header << "routing";
    for (auto& [type, num] : within_net_violation_type_num_map) {
      within_net_routing_violation_map_table << fort::header << type;
    }
    within_net_routing_violation_map_table << fort::header << "Total" << fort::endr;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      within_net_routing_violation_map_table << routing_layer.get_layer_name();
      for (auto& [type, num] : within_net_violation_type_num_map) {
        within_net_routing_violation_map_table << within_net_routing_violation_type_num_map[routing_layer.get_layer_idx()][type];
      }
      within_net_routing_violation_map_table << within_net_routing_violation_num_map[routing_layer.get_layer_idx()] << fort::endr;
    }
    within_net_routing_violation_map_table << fort::header << "Total";
    for (auto& [type, num] : within_net_violation_type_num_map) {
      within_net_routing_violation_map_table << fort::header << num;
    }
    within_net_routing_violation_map_table << fort::header << within_net_total_violation_num << fort::endr;
  }
  fort::char_table among_net_routing_violation_map_table;
  {
    among_net_routing_violation_map_table.set_cell_text_align(fort::text_align::right);
    among_net_routing_violation_map_table << fort::header << "among_net";
    for (size_t i = 0; i < among_net_violation_type_num_map.size(); ++i) {
      among_net_routing_violation_map_table << fort::header << " ";
    }
    among_net_routing_violation_map_table << fort::header << " " << fort::endr;
    among_net_routing_violation_map_table << fort::header << "routing";
    for (auto& [type, num] : among_net_violation_type_num_map) {
      among_net_routing_violation_map_table << fort::header << type;
    }
    among_net_routing_violation_map_table << fort::header << "Total" << fort::endr;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      among_net_routing_violation_map_table << routing_layer.get_layer_name();
      for (auto& [type, num] : among_net_violation_type_num_map) {
        among_net_routing_violation_map_table << among_net_routing_violation_type_num_map[routing_layer.get_layer_idx()][type];
      }
      among_net_routing_violation_map_table << among_net_routing_violation_num_map[routing_layer.get_layer_idx()] << fort::endr;
    }
    among_net_routing_violation_map_table << fort::header << "Total";
    for (auto& [type, num] : among_net_violation_type_num_map) {
      among_net_routing_violation_map_table << fort::header << num;
    }
    among_net_routing_violation_map_table << fort::header << among_net_total_violation_num << fort::endr;
  }
  RTUTIL.printTableList({routing_wire_length_map_table, cut_via_num_map_table, routing_patch_num_map_table});
  RTUTIL.printTableList({within_net_routing_violation_map_table, among_net_routing_violation_map_table});
}

void PinPatcher::outputNetCSV(PPModel& pp_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::string& pp_temp_directory_path = RTDM.getConfig().pp_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
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
      for (auto& [net_idx, pin_access_result_map] : gcell_map[x][y].get_net_pin_access_result_map()) {
        for (auto& [pin_idx, segment_set] : pin_access_result_map) {
          for (Segment<LayerCoord>* segment : segment_set) {
            int32_t first_layer_idx = segment->get_first().get_layer_idx();
            int32_t second_layer_idx = segment->get_second().get_layer_idx();
            RTUTIL.swapByASC(first_layer_idx, second_layer_idx);
            for (int32_t layer_idx = first_layer_idx; layer_idx <= second_layer_idx; layer_idx++) {
              net_layer_map[net_idx].insert(layer_idx);
            }
          }
        }
      }
      for (auto& [net_idx, patch_set] : gcell_map[x][y].get_net_access_patch_map()) {
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
        = RTUTIL.getOutputFileStream(RTUTIL.getString(pp_temp_directory_path, "net_map_", routing_layer.get_layer_name(), "_", pp_model.get_iter(), ".csv"));
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

void PinPatcher::outputViolationCSV(PPModel& pp_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::string& pp_temp_directory_path = RTDM.getConfig().pp_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
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
        RTUTIL.getString(pp_temp_directory_path, "violation_map_", routing_layer.get_layer_name(), "_", pp_model.get_iter(), ".csv"));
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

void PinPatcher::debugPlotPPModel(PPModel& pp_model, std::string flag)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& pp_temp_directory_path = RTDM.getConfig().pp_temp_directory_path;

  int32_t point_size = 5;

  GPGDS gp_gds;

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

  // access result
  for (auto& [net_idx, pin_access_result_map] : RTDM.getNetPinAccessResultMap(die)) {
    GPStruct access_result_struct(RTUTIL.getString("access_result(net_", net_idx, ")"));
    for (auto& [pin_idx, segment_set] : pin_access_result_map) {
      for (Segment<LayerCoord>* segment : segment_set) {
        for (NetShape& net_shape : RTDM.getNetShapeList(net_idx, *segment)) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPath));
          gp_boundary.set_rect(net_shape.get_rect());
          if (net_shape.get_is_routing()) {
            gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(net_shape.get_layer_idx()));
          } else {
            gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(net_shape.get_layer_idx()));
          }
          access_result_struct.push(gp_boundary);
        }
      }
    }
    gp_gds.addStruct(access_result_struct);
  }

  // access patch
  for (auto& [net_idx, patch_set] : RTDM.getNetAccessPatchMap(die)) {
    GPStruct access_patch_struct(RTUTIL.getString("access_patch(net_", net_idx, ")"));
    for (EXTLayerRect* patch : patch_set) {
      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPath));
      gp_boundary.set_rect(patch->get_real_rect());
      gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(patch->get_layer_idx()));
      access_patch_struct.push(gp_boundary);
    }
    gp_gds.addStruct(access_patch_struct);
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

  // violation
  {
    for (Violation* violation : RTDM.getViolationSet(die)) {
      if (violation->get_violation_net_set().size() >= 2) {
        continue;
      }
      GPStruct within_net_violation_struct(RTUTIL.getString("within_net_violation_", GetViolationTypeName()(violation->get_violation_type())));
      EXTLayerRect& violation_shape = violation->get_violation_shape();

      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kViolation));
      gp_boundary.set_rect(violation_shape.get_real_rect());
      if (violation->get_is_routing()) {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(violation_shape.get_layer_idx()));
      } else {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(violation_shape.get_layer_idx()));
      }
      within_net_violation_struct.push(gp_boundary);
      gp_gds.addStruct(within_net_violation_struct);
    }
    for (Violation* violation : RTDM.getViolationSet(die)) {
      if (violation->get_violation_net_set().size() < 2) {
        continue;
      }
      GPStruct among_net_violation_struct(RTUTIL.getString("among_net_violation_", GetViolationTypeName()(violation->get_violation_type())));
      EXTLayerRect& violation_shape = violation->get_violation_shape();

      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kViolation));
      gp_boundary.set_rect(violation_shape.get_real_rect());
      if (violation->get_is_routing()) {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(violation_shape.get_layer_idx()));
      } else {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(violation_shape.get_layer_idx()));
      }
      among_net_violation_struct.push(gp_boundary);
      gp_gds.addStruct(among_net_violation_struct);
    }
  }

  std::string gds_file_path = RTUTIL.getString(pp_temp_directory_path, flag, "_pp_model.gds");
  RTGP.plot(gp_gds, gds_file_path);
}

void PinPatcher::debugCheckPPBox(PPBox& pp_box)
{
  PPBoxId& pp_box_id = pp_box.get_pp_box_id();
  if (pp_box_id.get_x() < 0 || pp_box_id.get_y() < 0) {
    RTLOG.error(Loc::current(), "The grid coord is illegal!");
  }
}

void PinPatcher::debugPlotPPBox(PPBox& pp_box, std::string flag)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& pp_temp_directory_path = RTDM.getConfig().pp_temp_directory_path;

  PlanarRect box_real_rect = pp_box.get_box_rect().get_real_rect();

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

  // graph shape
  {
    for (auto& [routing_layer_idx, net_rect_map] : pp_box.get_graph_routing_net_fixed_rect_map()) {
      for (auto& [net_idx, rect_set] : net_rect_map) {
        GPStruct fixed_rect_struct(RTUTIL.getString("graph_fixed_rect(net_", net_idx, ")"));
        for (const PlanarRect& rect : rect_set) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kGraphShape));
          gp_boundary.set_rect(rect);
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(routing_layer_idx));
          fixed_rect_struct.push(gp_boundary);
        }
        gp_gds.addStruct(fixed_rect_struct);
      }
    }
    for (auto& [routing_layer_idx, net_rect_map] : pp_box.get_graph_routing_net_routed_rect_map()) {
      for (auto& [net_idx, rect_set] : net_rect_map) {
        GPStruct routed_rect_struct(RTUTIL.getString("graph_routed_rect(net_", net_idx, ")"));
        for (const PlanarRect& rect : rect_set) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kGraphShape));
          gp_boundary.set_rect(rect);
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(routing_layer_idx));
          routed_rect_struct.push(gp_boundary);
        }
        gp_gds.addStruct(routed_rect_struct);
      }
    }
  }

  // box_track_axis
  {
    GPStruct box_track_axis_struct("box_track_axis");
    PlanarCoord& real_ll = box_real_rect.get_ll();
    PlanarCoord& real_ur = box_real_rect.get_ur();
    ScaleAxis& box_track_axis = pp_box.get_box_track_axis();
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
  for (auto& [is_routing, layer_net_rect_map] : pp_box.get_type_layer_net_fixed_rect_map()) {
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

  // net_access_result
  for (auto& [net_idx, pin_access_result_map] : pp_box.get_net_pin_access_result_map()) {
    GPStruct access_result_struct(RTUTIL.getString("access_result(net_", net_idx, ")"));
    for (auto& [pin_idx, segment_set] : pin_access_result_map) {
      for (Segment<LayerCoord>* segment : segment_set) {
        for (NetShape& net_shape : RTDM.getNetShapeList(net_idx, *segment)) {
          GPBoundary gp_boundary;
          gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShape));
          gp_boundary.set_rect(net_shape.get_rect());
          if (net_shape.get_is_routing()) {
            gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(net_shape.get_layer_idx()));
          } else {
            gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(net_shape.get_layer_idx()));
          }
          access_result_struct.push(gp_boundary);
        }
      }
    }
    gp_gds.addStruct(access_result_struct);
  }

  // net_access_patch
  for (auto& [net_idx, patch_set] : pp_box.get_net_access_patch_map()) {
    GPStruct access_patch_struct(RTUTIL.getString("access_patch(net_", net_idx, ")"));
    for (EXTLayerRect* patch : patch_set) {
      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShape));
      gp_boundary.set_rect(patch->get_real_rect());
      gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(patch->get_layer_idx()));
      access_patch_struct.push(gp_boundary);
    }
    gp_gds.addStruct(access_patch_struct);
  }

  // task
  for (auto& [net_idx, access_patch_list] : pp_box.get_net_task_access_patch_map()) {
    GPStruct task_struct(RTUTIL.getString("task(net_", net_idx, ")"));
    if (net_idx == pp_box.get_curr_net_idx()) {
      for (EXTLayerRect& patch : pp_box.get_curr_routing_patch_list()) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPath));
        gp_boundary.set_rect(patch.get_real_rect());
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(patch.get_layer_idx()));
        task_struct.push(gp_boundary);
      }
    } else {
      for (EXTLayerRect& patch : access_patch_list) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPath));
        gp_boundary.set_rect(patch.get_real_rect());
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(patch.get_layer_idx()));
        task_struct.push(gp_boundary);
      }
    }
    gp_gds.addStruct(task_struct);
  }

  // violation
  {
    for (Violation& violation : pp_box.get_curr_violation_list()) {
      if (violation.get_violation_net_set().size() >= 2) {
        continue;
      }
      GPStruct within_net_violation_struct(RTUTIL.getString("within_net_violation_", GetViolationTypeName()(violation.get_violation_type())));
      EXTLayerRect& violation_shape = violation.get_violation_shape();

      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kViolation));
      gp_boundary.set_rect(violation_shape.get_real_rect());
      if (violation.get_is_routing()) {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(violation_shape.get_layer_idx()));
      } else {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(violation_shape.get_layer_idx()));
      }
      within_net_violation_struct.push(gp_boundary);
      gp_gds.addStruct(within_net_violation_struct);
    }
    for (Violation& violation : pp_box.get_curr_violation_list()) {
      if (violation.get_violation_net_set().size() < 2) {
        continue;
      }
      GPStruct among_net_violation_struct(RTUTIL.getString("among_net_violation_", GetViolationTypeName()(violation.get_violation_type())));
      EXTLayerRect& violation_shape = violation.get_violation_shape();

      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kViolation));
      gp_boundary.set_rect(violation_shape.get_real_rect());
      if (violation.get_is_routing()) {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(violation_shape.get_layer_idx()));
      } else {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(violation_shape.get_layer_idx()));
      }
      among_net_violation_struct.push(gp_boundary);
      gp_gds.addStruct(among_net_violation_struct);
    }
  }

  std::string gds_file_path
      = RTUTIL.getString(pp_temp_directory_path, flag, "_pp_box_", pp_box.get_pp_box_id().get_x(), "_", pp_box.get_pp_box_id().get_y(), ".gds");
  RTGP.plot(gp_gds, gds_file_path);
}

#endif

}  // namespace irt
