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
#include "HybridRouter.hpp"

#include "DRCEngine.hpp"
#include "GDSPlotter.hpp"
#include "HRBox.hpp"
#include "HRBoxId.hpp"
#include "HRIterParam.hpp"
#include "HRNet.hpp"
#include "HRNode.hpp"
#include "HybridRouter.hpp"
#include "Monitor.hpp"
#include "RTInterface.hpp"

namespace irt {

// public

void HybridRouter::initInst()
{
  if (_hr_instance == nullptr) {
    _hr_instance = new HybridRouter();
  }
}

HybridRouter& HybridRouter::getInst()
{
  if (_hr_instance == nullptr) {
    RTLOG.error(Loc::current(), "The instance not initialized!");
  }
  return *_hr_instance;
}

void HybridRouter::destroyInst()
{
  if (_hr_instance != nullptr) {
    delete _hr_instance;
    _hr_instance = nullptr;
  }
}

// function

void HybridRouter::route()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");
  HRModel hr_model = initHRModel();
  updateAccessPoint(hr_model);
  initNetFinalResultMap(hr_model);
  buildNetFinalResultMap(hr_model);
  clearIgnoredViolation(hr_model);
  uploadViolation(hr_model);
  iterativeHRModel(hr_model);
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

// private

HybridRouter* HybridRouter::_hr_instance = nullptr;

HRModel HybridRouter::initHRModel()
{
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();

  HRModel hr_model;
  hr_model.set_hr_net_list(convertToHRNetList(net_list));
  return hr_model;
}

std::vector<HRNet> HybridRouter::convertToHRNetList(std::vector<Net>& net_list)
{
  std::vector<HRNet> hr_net_list;
  hr_net_list.reserve(net_list.size());
  for (Net& net : net_list) {
    hr_net_list.emplace_back(convertToHRNet(net));
  }
  return hr_net_list;
}

HRNet HybridRouter::convertToHRNet(Net& net)
{
  HRNet hr_net;
  hr_net.set_origin_net(&net);
  hr_net.set_net_idx(net.get_net_idx());
  hr_net.set_connect_type(net.get_connect_type());
  for (Pin& pin : net.get_pin_list()) {
    hr_net.get_hr_pin_list().push_back(HRPin(pin));
  }
  return hr_net;
}

void HybridRouter::updateAccessPoint(HRModel& hr_model)
{
  Die& die = RTDM.getDatabase().get_die();

  for (auto& [net_idx, access_point_set] : RTDM.getNetAccessPointMap(die)) {
    for (AccessPoint* access_point : access_point_set) {
      RTDM.updateAccessNetPointToGCellMap(ChangeType::kDel, net_idx, access_point);
    }
  }
  for (HRNet& hr_net : hr_model.get_hr_net_list()) {
    Net* origin_net = hr_net.get_origin_net();
    if (origin_net->get_net_idx() != hr_net.get_net_idx()) {
      RTLOG.error(Loc::current(), "The net idx is not equal!");
    }
    for (HRPin& hr_pin : hr_net.get_hr_pin_list()) {
      Pin& origin_pin = origin_net->get_pin_list()[hr_pin.get_pin_idx()];
      if (origin_pin.get_pin_idx() != hr_pin.get_pin_idx()) {
        RTLOG.error(Loc::current(), "The pin idx is not equal!");
      }
      hr_pin.set_access_point(hr_pin.get_origin_access_point());
      // 之后流程将暂时使用origin_access_point作为主要access point
      origin_pin.set_access_point(origin_pin.get_origin_access_point());
      RTDM.updateAccessNetPointToGCellMap(ChangeType::kAdd, hr_net.get_net_idx(), &origin_pin.get_access_point());
    }
  }
}

void HybridRouter::initNetFinalResultMap(HRModel& hr_model)
{
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();

  for (int32_t x = 0; x < gcell_map.get_x_size(); x++) {
    for (int32_t y = 0; y < gcell_map.get_y_size(); y++) {
      std::map<int32_t, std::set<Segment<LayerCoord>*>>& net_final_result_map = gcell_map[x][y].get_net_final_result_map();
      for (auto& [net_idx, pin_access_result_map] : gcell_map[x][y].get_net_pin_access_result_map()) {
        for (auto& [pin_idx, segment_set] : pin_access_result_map) {
          for (Segment<LayerCoord>* segment : segment_set) {
            net_final_result_map[net_idx].insert(segment);
          }
        }
      }
      gcell_map[x][y].get_net_pin_access_result_map().clear();
      for (auto& [net_idx, segment_set] : gcell_map[x][y].get_net_detailed_result_map()) {
        for (Segment<LayerCoord>* segment : segment_set) {
          net_final_result_map[net_idx].insert(segment);
        }
      }
      gcell_map[x][y].get_net_detailed_result_map().clear();
    }
  }
}

void HybridRouter::buildNetFinalResultMap(HRModel& hr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();
  std::vector<HRNet>& hr_net_list = hr_model.get_hr_net_list();

  std::map<int32_t, std::set<Segment<LayerCoord>*>> net_final_result_map = RTDM.getNetFinalResultMap(die);
  for (auto& [net_idx, segment_set] : net_final_result_map) {
    std::vector<Segment<LayerCoord>> routing_segment_list;
    for (Segment<LayerCoord>* segment : segment_set) {
      routing_segment_list.emplace_back(segment->get_first(), segment->get_second());
    }
    std::vector<LayerCoord> candidate_root_coord_list;
    std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
    std::vector<HRPin>& hr_pin_list = hr_net_list[net_idx].get_hr_pin_list();
    for (size_t i = 0; i < hr_pin_list.size(); i++) {
      LayerCoord coord = hr_pin_list[i].get_access_point().getRealLayerCoord();
      candidate_root_coord_list.push_back(coord);
      key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
    }
    MTree<LayerCoord> coord_tree = RTUTIL.getTreeByFullFlow(candidate_root_coord_list, routing_segment_list, key_coord_pin_map);
    for (Segment<TNode<LayerCoord>*>& coord_segment : RTUTIL.getSegListByTree(coord_tree)) {
      RTDM.updateNetFinalResultToGCellMap(ChangeType::kAdd, net_idx,
                                          new Segment<LayerCoord>(coord_segment.get_first()->value(), coord_segment.get_second()->value()));
    }
  }
  for (auto& [net_idx, segment_set] : net_final_result_map) {
    for (Segment<LayerCoord>* segment : segment_set) {
      RTDM.updateNetFinalResultToGCellMap(ChangeType::kDel, net_idx, segment);
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void HybridRouter::clearIgnoredViolation(HRModel& hr_model)
{
  RTDE.clearTempIgnoredViolationSet();
}

void HybridRouter::uploadViolation(HRModel& hr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();

  for (Violation* violation : RTDM.getViolationSet(die)) {
    RTDM.updateViolationToGCellMap(ChangeType::kDel, violation);
  }
  for (Violation violation : getViolationList(hr_model)) {
    RTDM.updateViolationToGCellMap(ChangeType::kAdd, new Violation(violation));
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

std::vector<Violation> HybridRouter::getViolationList(HRModel& hr_model)
{
  Die& die = RTDM.getDatabase().get_die();

  DETask de_task;
  {
    std::string top_name = RTUTIL.getString("hr_model");
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
    for (auto& [net_idx, segment_set] : RTDM.getNetFinalResultMap(die)) {
      for (Segment<LayerCoord>* segment : segment_set) {
        net_result_map[net_idx].push_back(segment);
      }
    }
    std::map<int32_t, std::vector<EXTLayerRect*>> net_patch_map;
    for (auto& [net_idx, patch_set] : RTDM.getNetFinalPatchMap(die)) {
      for (EXTLayerRect* patch : patch_set) {
        net_patch_map[net_idx].emplace_back(patch);
      }
    }
    std::set<int32_t> need_checked_net_set;
    for (HRNet& hr_net : hr_model.get_hr_net_list()) {
      need_checked_net_set.insert(hr_net.get_net_idx());
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

void HybridRouter::iterativeHRModel(HRModel& hr_model)
{
  int32_t cost_unit = RTDM.getOnlyPitch();
  double prefer_wire_unit = 1;
  double non_prefer_wire_unit = 2.5 * prefer_wire_unit;
  double via_unit = cost_unit;
  double fixed_rect_unit = 4 * non_prefer_wire_unit * cost_unit;
  double routed_rect_unit = 2 * via_unit;
  double violation_unit = 4 * non_prefer_wire_unit * cost_unit;
  /**
   * prefer_wire_unit, non_prefer_wire_unit, via_unit, size, offset, schedule_interval, fixed_rect_unit, routed_rect_unit, violation_unit, max_routed_times
   */
  std::vector<HRIterParam> hr_iter_param_list;
  // clang-format off
  hr_iter_param_list.emplace_back(prefer_wire_unit, non_prefer_wire_unit, via_unit, 5, 0, 3, fixed_rect_unit, routed_rect_unit, violation_unit, 3);
  hr_iter_param_list.emplace_back(prefer_wire_unit, non_prefer_wire_unit, via_unit, 5, 1, 3, fixed_rect_unit, routed_rect_unit, violation_unit, 3);
  hr_iter_param_list.emplace_back(prefer_wire_unit, non_prefer_wire_unit, via_unit, 5, 2, 3, fixed_rect_unit, routed_rect_unit, violation_unit, 3);
  hr_iter_param_list.emplace_back(prefer_wire_unit, non_prefer_wire_unit, via_unit, 5, 3, 3, fixed_rect_unit, routed_rect_unit, violation_unit, 3);
  hr_iter_param_list.emplace_back(prefer_wire_unit, non_prefer_wire_unit, via_unit, 5, 4, 3, fixed_rect_unit, routed_rect_unit, violation_unit, 3);
  hr_iter_param_list.emplace_back(prefer_wire_unit, non_prefer_wire_unit, via_unit, 5, 0, 3, fixed_rect_unit, routed_rect_unit, violation_unit, 3);
  hr_iter_param_list.emplace_back(prefer_wire_unit, non_prefer_wire_unit, via_unit, 5, 1, 3, fixed_rect_unit, routed_rect_unit, violation_unit, 3);
  hr_iter_param_list.emplace_back(prefer_wire_unit, non_prefer_wire_unit, via_unit, 5, 2, 3, fixed_rect_unit, routed_rect_unit, violation_unit, 3);
  hr_iter_param_list.emplace_back(prefer_wire_unit, non_prefer_wire_unit, via_unit, 5, 3, 3, fixed_rect_unit, routed_rect_unit, violation_unit, 3);
  hr_iter_param_list.emplace_back(prefer_wire_unit, non_prefer_wire_unit, via_unit, 5, 4, 3, fixed_rect_unit, routed_rect_unit, violation_unit, 3);
  // clang-format on
  initRoutingState(hr_model);
  for (int32_t i = 0, iter = 1; i < static_cast<int32_t>(hr_iter_param_list.size()); i++, iter++) {
    Monitor iter_monitor;
    RTLOG.info(Loc::current(), "***** Begin iteration ", iter, "/", hr_iter_param_list.size(), "(", RTUTIL.getPercentage(iter, hr_iter_param_list.size()),
               ") *****");
    // debugPlotHRModel(hr_model, "before");
    setHRIterParam(hr_model, iter, hr_iter_param_list[i]);
    initHRBoxMap(hr_model);
    resetRoutingState(hr_model);
    buildBoxSchedule(hr_model);
    splitNetResult(hr_model);
    // debugPlotHRModel(hr_model, "middle");
    routeHRBoxMap(hr_model);
    uploadNetResult(hr_model);
    uploadViolation(hr_model);
    updateBestResult(hr_model);
    // debugPlotHRModel(hr_model, "after");
    updateSummary(hr_model);
    printSummary(hr_model);
    outputNetCSV(hr_model);
    outputViolationCSV(hr_model);
    RTLOG.info(Loc::current(), "***** End Iteration ", iter, "/", hr_iter_param_list.size(), "(", RTUTIL.getPercentage(iter, hr_iter_param_list.size()), ")",
               iter_monitor.getStatsInfo(), "*****");
    if (stopIteration(hr_model)) {
      break;
    }
  }
  selectBestResult(hr_model);
}

void HybridRouter::initRoutingState(HRModel& hr_model)
{
  hr_model.set_initial_routing(true);
}

void HybridRouter::setHRIterParam(HRModel& hr_model, int32_t iter, HRIterParam& hr_iter_param)
{
  hr_model.set_iter(iter);
  RTLOG.info(Loc::current(), "prefer_wire_unit: ", hr_iter_param.get_prefer_wire_unit());
  RTLOG.info(Loc::current(), "non_prefer_wire_unit: ", hr_iter_param.get_non_prefer_wire_unit());
  RTLOG.info(Loc::current(), "via_unit: ", hr_iter_param.get_via_unit());
  RTLOG.info(Loc::current(), "size: ", hr_iter_param.get_size());
  RTLOG.info(Loc::current(), "offset: ", hr_iter_param.get_offset());
  RTLOG.info(Loc::current(), "schedule_interval: ", hr_iter_param.get_schedule_interval());
  RTLOG.info(Loc::current(), "fixed_rect_unit: ", hr_iter_param.get_fixed_rect_unit());
  RTLOG.info(Loc::current(), "routed_rect_unit: ", hr_iter_param.get_routed_rect_unit());
  RTLOG.info(Loc::current(), "violation_unit: ", hr_iter_param.get_violation_unit());
  RTLOG.info(Loc::current(), "max_routed_times: ", hr_iter_param.get_max_routed_times());
  hr_model.set_hr_iter_param(hr_iter_param);
}

void HybridRouter::initHRBoxMap(HRModel& hr_model)
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

  HRIterParam& hr_iter_param = hr_model.get_hr_iter_param();
  int32_t size = hr_iter_param.get_size();
  int32_t offset = hr_iter_param.get_offset();
  int32_t x_box_num = static_cast<int32_t>(std::ceil((x_gcell_num - offset) / 1.0 / size));
  int32_t y_box_num = static_cast<int32_t>(std::ceil((y_gcell_num - offset) / 1.0 / size));

  GridMap<HRBox>& hr_box_map = hr_model.get_hr_box_map();
  hr_box_map.init(x_box_num, y_box_num);

  for (int32_t x = 0; x < hr_box_map.get_x_size(); x++) {
    for (int32_t y = 0; y < hr_box_map.get_y_size(); y++) {
      int32_t grid_ll_x = std::max(offset + x * size, 0);
      int32_t grid_ll_y = std::max(offset + y * size, 0);
      int32_t grid_ur_x = std::min(offset + (x + 1) * size - 1, x_gcell_num - 1);
      int32_t grid_ur_y = std::min(offset + (y + 1) * size - 1, y_gcell_num - 1);

      PlanarRect ll_gcell_rect = RTUTIL.getRealRectByGCell(PlanarCoord(grid_ll_x, grid_ll_y), gcell_axis);
      PlanarRect ur_gcell_rect = RTUTIL.getRealRectByGCell(PlanarCoord(grid_ur_x, grid_ur_y), gcell_axis);
      PlanarRect box_real_rect(ll_gcell_rect.get_ll(), ur_gcell_rect.get_ur());

      HRBox& hr_box = hr_box_map[x][y];

      EXTPlanarRect hr_box_rect;
      hr_box_rect.set_real_rect(box_real_rect);
      hr_box_rect.set_grid_rect(RTUTIL.getOpenGCellGridRect(box_real_rect, gcell_axis));
      hr_box.set_box_rect(hr_box_rect);
      HRBoxId hr_box_id;
      hr_box_id.set_x(x);
      hr_box_id.set_y(y);
      hr_box.set_hr_box_id(hr_box_id);
      hr_box.set_hr_iter_param(&hr_iter_param);
      hr_box.set_initial_routing(hr_model.get_initial_routing());
    }
  }
}

void HybridRouter::resetRoutingState(HRModel& hr_model)
{
  hr_model.set_initial_routing(false);
}

void HybridRouter::buildBoxSchedule(HRModel& hr_model)
{
  GridMap<HRBox>& hr_box_map = hr_model.get_hr_box_map();
  int32_t schedule_interval = hr_model.get_hr_iter_param().get_schedule_interval();

  std::vector<std::vector<HRBoxId>> hr_box_id_list_list;
  for (int32_t start_x = 0; start_x < schedule_interval; start_x++) {
    for (int32_t start_y = 0; start_y < schedule_interval; start_y++) {
      std::vector<HRBoxId> hr_box_id_list;
      for (int32_t x = start_x; x < hr_box_map.get_x_size(); x += schedule_interval) {
        for (int32_t y = start_y; y < hr_box_map.get_y_size(); y += schedule_interval) {
          hr_box_id_list.emplace_back(x, y);
        }
      }
      hr_box_id_list_list.push_back(hr_box_id_list);
    }
  }
  hr_model.set_hr_box_id_list_list(hr_box_id_list_list);
}

void HybridRouter::splitNetResult(HRModel& hr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();

  for (auto& [net_idx, segment_set] : RTDM.getNetFinalResultMap(die)) {
    std::vector<Segment<LayerCoord>*> del_segment_list;
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
        del_segment_list.push_back(segment);
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
        del_segment_list.push_back(segment);
        for (size_t i = 1; i < y_scale_list.size(); i++) {
          new_segment_list.emplace_back(LayerCoord(first_coord.get_x(), y_scale_list[i - 1], first_coord.get_layer_idx()),
                                        LayerCoord(first_coord.get_x(), y_scale_list[i], first_coord.get_layer_idx()));
        }
      }
    }
    for (Segment<LayerCoord>* del_segment : del_segment_list) {
      RTDM.updateNetFinalResultToGCellMap(ChangeType::kDel, net_idx, del_segment);
    }
    for (Segment<LayerCoord>& new_segment : new_segment_list) {
      RTDM.updateNetFinalResultToGCellMap(ChangeType::kAdd, net_idx, new Segment<LayerCoord>(new_segment));
    }
  }
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void HybridRouter::routeHRBoxMap(HRModel& hr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<HRBox>& hr_box_map = hr_model.get_hr_box_map();

  size_t total_box_num = 0;
  for (std::vector<HRBoxId>& hr_box_id_list : hr_model.get_hr_box_id_list_list()) {
    total_box_num += hr_box_id_list.size();
  }

  size_t routed_box_num = 0;
  for (std::vector<HRBoxId>& hr_box_id_list : hr_model.get_hr_box_id_list_list()) {
    Monitor stage_monitor;
#pragma omp parallel for
    for (HRBoxId& hr_box_id : hr_box_id_list) {
      HRBox& hr_box = hr_box_map[hr_box_id.get_x()][hr_box_id.get_y()];
      buildFixedRect(hr_box);
      buildNetResult(hr_box);
      initHRTaskList(hr_model, hr_box);
      buildViolation(hr_box);
      if (needRouting(hr_box)) {
        buildBoxTrackAxis(hr_box);
        buildLayerNodeMap(hr_box);
        buildHRNodeNeighbor(hr_box);
        buildOrientNetMap(hr_box);
        exemptPinShape(hr_box);
        // debugCheckHRBox(hr_box);
        // debugPlotHRBox(hr_box, "before");
        routeHRBox(hr_box);
        // debugPlotHRBox(hr_box, "after");
      }
      selectBestResult(hr_box);
      freeHRBox(hr_box);
    }
    routed_box_num += hr_box_id_list.size();
    RTLOG.info(Loc::current(), "Routed ", routed_box_num, "/", total_box_num, "(", RTUTIL.getPercentage(routed_box_num, total_box_num), ") boxes with ",
               getViolationNum(hr_model), " violations", stage_monitor.getStatsInfo());
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void HybridRouter::buildFixedRect(HRBox& hr_box)
{
  hr_box.set_type_layer_net_fixed_rect_map(RTDM.getTypeLayerNetFixedRectMap(hr_box.get_box_rect()));
}

void HybridRouter::buildNetResult(HRBox& hr_box)
{
  PlanarRect& box_real_rect = hr_box.get_box_rect().get_real_rect();

  for (auto& [net_idx, segment_set] : RTDM.getNetFinalResultMap(hr_box.get_box_rect())) {
    for (Segment<LayerCoord>* segment : segment_set) {
      bool least_one_coord_in_box = false;
      if (RTUTIL.isInside(box_real_rect, segment->get_first()) && RTUTIL.isInside(box_real_rect, segment->get_second())) {
        if (RTUTIL.isInside(box_real_rect, segment->get_first(), false) || RTUTIL.isInside(box_real_rect, segment->get_second(), false)) {
          // 线段在box_real_rect内,但不贴边的
          least_one_coord_in_box = true;
        }
      }
      if (least_one_coord_in_box) {
        hr_box.get_net_task_final_result_map()[net_idx].push_back(*segment);
        RTDM.updateNetFinalResultToGCellMap(ChangeType::kDel, net_idx, segment);
      } else {
        hr_box.get_net_final_result_map()[net_idx].insert(segment);
      }
    }
  }
}

void HybridRouter::initHRTaskList(HRModel& hr_model, HRBox& hr_box)
{
  std::vector<HRNet>& hr_net_list = hr_model.get_hr_net_list();
  std::vector<HRTask*>& hr_task_list = hr_box.get_hr_task_list();

  EXTPlanarRect& box_rect = hr_box.get_box_rect();
  PlanarRect& box_real_rect = box_rect.get_real_rect();
  std::map<int32_t, std::set<AccessPoint*>> net_access_point_map = RTDM.getNetAccessPointMap(box_rect);
  std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_task_final_result_map = hr_box.get_net_task_final_result_map();

  std::map<int32_t, std::vector<HRGroup>> net_group_list_map;
  {
    for (auto& [net_idx, access_point_set] : net_access_point_map) {
      std::map<int32_t, HRGroup> pin_group_map;
      for (AccessPoint* access_point : access_point_set) {
        pin_group_map[access_point->get_pin_idx()].get_coord_list().push_back(access_point->getRealLayerCoord());
      }
      for (auto& [pin_idx, group] : pin_group_map) {
        net_group_list_map[net_idx].push_back(group);
      }
    }
    for (auto& [net_idx, segment_list] : net_task_final_result_map) {
      std::vector<LayerCoord> coord_list;
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
            coord_list.emplace_back(box_real_rect.get_ll_x(), first.get_y(), first.get_layer_idx());
          }
          if (first_x <= box_real_rect.get_ur_x() && box_real_rect.get_ur_x() <= second_x) {
            coord_list.emplace_back(box_real_rect.get_ur_x(), first.get_y(), first.get_layer_idx());
          }
        } else if (RTUTIL.isVertical(first, second)) {
          int32_t first_y = first.get_y();
          int32_t second_y = second.get_y();
          if (first.get_x() < box_real_rect.get_ll_x() || box_real_rect.get_ur_x() < first.get_x()) {
            continue;
          }
          RTUTIL.swapByASC(first_y, second_y);
          if (first_y <= box_real_rect.get_ll_y() && box_real_rect.get_ll_y() <= second_y) {
            coord_list.emplace_back(first.get_x(), box_real_rect.get_ll_y(), first.get_layer_idx());
          }
          if (first_y <= box_real_rect.get_ur_y() && box_real_rect.get_ur_y() <= second_y) {
            coord_list.emplace_back(first.get_x(), box_real_rect.get_ur_y(), first.get_layer_idx());
          }
        } else {
          RTLOG.error(Loc::current(), "The segment is oblique!");
        }
      }
      for (LayerCoord& coord : coord_list) {
        HRGroup hr_group;
        hr_group.get_coord_list().push_back(coord);
        net_group_list_map[net_idx].push_back(hr_group);
      }
    }
  }
  for (auto& [net_idx, hr_group_list] : net_group_list_map) {
    if (hr_group_list.size() < 2) {
      continue;
    }
    HRTask* hr_task = new HRTask();
    hr_task->set_net_idx(net_idx);
    hr_task->set_connect_type(hr_net_list[net_idx].get_connect_type());
    hr_task->set_hr_group_list(hr_group_list);
    {
      std::vector<PlanarCoord> coord_list;
      for (HRGroup& hr_group : hr_task->get_hr_group_list()) {
        for (LayerCoord& coord : hr_group.get_coord_list()) {
          coord_list.push_back(coord);
        }
      }
      hr_task->set_bounding_box(RTUTIL.getBoundingBox(coord_list));
    }
    hr_task->set_routed_times(0);
    hr_task_list.push_back(hr_task);
  }
  std::sort(hr_task_list.begin(), hr_task_list.end(), CmpHRTask());
}

void HybridRouter::buildViolation(HRBox& hr_box)
{
  std::set<int32_t> need_checked_net_set;
  for (HRTask* hr_task : hr_box.get_hr_task_list()) {
    need_checked_net_set.insert(hr_task->get_net_idx());
  }
  for (Violation* violation : RTDM.getViolationSet(hr_box.get_box_rect())) {
    bool exist_checked_net = false;
    for (int32_t violation_net_idx : violation->get_violation_net_set()) {
      if (RTUTIL.exist(need_checked_net_set, violation_net_idx)) {
        exist_checked_net = true;
        break;
      }
    }
    if (exist_checked_net) {
      hr_box.get_violation_list().push_back(*violation);
      RTDM.updateViolationToGCellMap(ChangeType::kDel, violation);
    }
  }
}

bool HybridRouter::needRouting(HRBox& hr_box)
{
  if (hr_box.get_hr_task_list().empty()) {
    return false;
  }
  if (hr_box.get_initial_routing() == false && hr_box.get_violation_list().empty()) {
    return false;
  }
  return true;
}

void HybridRouter::buildBoxTrackAxis(HRBox& hr_box)
{
  int32_t manufacture_grid = RTDM.getDatabase().get_manufacture_grid();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  std::vector<int32_t> x_scale_list;
  std::vector<int32_t> y_scale_list;

  PlanarRect& box_real_rect = hr_box.get_box_rect().get_real_rect();
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
  for (RoutingLayer& routing_layer : routing_layer_list) {
    for (int32_t x_scale : RTUTIL.getScaleList(ll_x, ur_x, routing_layer.getXTrackGridList())) {
      x_scale_list.push_back(x_scale);
    }
    for (int32_t y_scale : RTUTIL.getScaleList(ll_y, ur_y, routing_layer.getYTrackGridList())) {
      y_scale_list.push_back(y_scale);
    }
  }
  for (HRTask* hr_task : hr_box.get_hr_task_list()) {
    for (HRGroup& hr_group : hr_task->get_hr_group_list()) {
      for (LayerCoord& coord : hr_group.get_coord_list()) {
        x_scale_list.push_back(coord.get_x());
        y_scale_list.push_back(coord.get_y());
      }
    }
  }

  ScaleAxis& box_track_axis = hr_box.get_box_track_axis();
  std::sort(x_scale_list.begin(), x_scale_list.end());
  x_scale_list.erase(std::unique(x_scale_list.begin(), x_scale_list.end()), x_scale_list.end());
  box_track_axis.set_x_grid_list(RTUTIL.makeScaleGridList(x_scale_list));
  std::sort(y_scale_list.begin(), y_scale_list.end());
  y_scale_list.erase(std::unique(y_scale_list.begin(), y_scale_list.end()), y_scale_list.end());
  box_track_axis.set_y_grid_list(RTUTIL.makeScaleGridList(y_scale_list));
}

void HybridRouter::buildLayerNodeMap(HRBox& hr_box)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  PlanarCoord& real_ll = hr_box.get_box_rect().get_real_ll();
  PlanarCoord& real_ur = hr_box.get_box_rect().get_real_ur();
  ScaleAxis& box_track_axis = hr_box.get_box_track_axis();
  std::vector<int32_t> x_list = RTUTIL.getScaleList(real_ll.get_x(), real_ur.get_x(), box_track_axis.get_x_grid_list());
  std::vector<int32_t> y_list = RTUTIL.getScaleList(real_ll.get_y(), real_ur.get_y(), box_track_axis.get_y_grid_list());

  std::vector<GridMap<HRNode>>& layer_node_map = hr_box.get_layer_node_map();
  layer_node_map.resize(routing_layer_list.size());
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    GridMap<HRNode>& hr_node_map = layer_node_map[layer_idx];
    hr_node_map.init(x_list.size(), y_list.size());
    for (size_t x = 0; x < x_list.size(); x++) {
      for (size_t y = 0; y < y_list.size(); y++) {
        HRNode& hr_node = hr_node_map[x][y];
        hr_node.set_x(x_list[x]);
        hr_node.set_y(y_list[y]);
        hr_node.set_layer_idx(layer_idx);
      }
    }
  }
}

void HybridRouter::buildHRNodeNeighbor(HRBox& hr_box)
{
  int32_t bottom_routing_layer_idx = RTDM.getConfig().bottom_routing_layer_idx;
  int32_t top_routing_layer_idx = RTDM.getConfig().top_routing_layer_idx;

  std::vector<GridMap<HRNode>>& layer_node_map = hr_box.get_layer_node_map();
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    bool routing_hv = true;
    if (layer_idx < bottom_routing_layer_idx || top_routing_layer_idx < layer_idx) {
      routing_hv = false;
    }
    GridMap<HRNode>& hr_node_map = layer_node_map[layer_idx];
    for (int32_t x = 0; x < hr_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < hr_node_map.get_y_size(); y++) {
        std::map<Orientation, HRNode*>& neighbor_node_map = hr_node_map[x][y].get_neighbor_node_map();
        if (routing_hv) {
          if (x != 0) {
            neighbor_node_map[Orientation::kWest] = &hr_node_map[x - 1][y];
          }
          if (x != (hr_node_map.get_x_size() - 1)) {
            neighbor_node_map[Orientation::kEast] = &hr_node_map[x + 1][y];
          }
          if (y != 0) {
            neighbor_node_map[Orientation::kSouth] = &hr_node_map[x][y - 1];
          }
          if (y != (hr_node_map.get_y_size() - 1)) {
            neighbor_node_map[Orientation::kNorth] = &hr_node_map[x][y + 1];
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

void HybridRouter::buildOrientNetMap(HRBox& hr_box)
{
  for (auto& [is_routing, layer_net_fixed_rect_map] : hr_box.get_type_layer_net_fixed_rect_map()) {
    for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
      for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
        for (auto& fixed_rect : fixed_rect_set) {
          updateFixedRectToGraph(hr_box, ChangeType::kAdd, net_idx, fixed_rect, is_routing);
        }
      }
    }
  }
  for (auto& [net_idx, segment_set] : hr_box.get_net_final_result_map()) {
    for (Segment<LayerCoord>* segment : segment_set) {
      updateFixedRectToGraph(hr_box, ChangeType::kAdd, net_idx, *segment);
    }
  }
  for (auto& [net_idx, segment_list] : hr_box.get_net_task_final_result_map()) {
    for (Segment<LayerCoord>& segment : segment_list) {
      updateRoutedRectToGraph(hr_box, ChangeType::kAdd, net_idx, segment);
    }
  }
  for (Violation& violation : hr_box.get_violation_list()) {
    addViolationToGraph(hr_box, violation);
  }
}

void HybridRouter::exemptPinShape(HRBox& hr_box)
{
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = RTDM.getDatabase().get_cut_to_adjacent_routing_map();

  std::map<int32_t, std::map<EXTLayerRect*, std::set<Orientation>>> routing_layer_pin_shape_orient_map;
  for (auto& [is_routing, layer_net_fixed_rect_map] : hr_box.get_type_layer_net_fixed_rect_map()) {
    for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
      std::map<int32_t, std::set<Orientation>> routing_layer_orient_map;
      if (is_routing) {
        routing_layer_orient_map[layer_idx].insert(Orientation::kEast);
        routing_layer_orient_map[layer_idx].insert(Orientation::kWest);
        routing_layer_orient_map[layer_idx].insert(Orientation::kSouth);
        routing_layer_orient_map[layer_idx].insert(Orientation::kNorth);
      } else {
        if (cut_to_adjacent_routing_map[layer_idx].size() < 2) {
          continue;
        }
        int32_t below_routing_layer_idx = cut_to_adjacent_routing_map[layer_idx].front();
        int32_t above_routing_layer_idx = cut_to_adjacent_routing_map[layer_idx].back();
        RTUTIL.swapByASC(below_routing_layer_idx, above_routing_layer_idx);
        routing_layer_orient_map[below_routing_layer_idx].insert(Orientation::kAbove);
        routing_layer_orient_map[above_routing_layer_idx].insert(Orientation::kBelow);
      }
      for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
        if (net_idx == -1) {
          continue;
        }
        for (auto& fixed_rect : fixed_rect_set) {
          for (auto& [routing_layer_idx, orient_set] : routing_layer_orient_map) {
            routing_layer_pin_shape_orient_map[routing_layer_idx][fixed_rect] = orient_set;
          }
        }
      }
    }
  }
  std::vector<GridMap<HRNode>>& layer_node_map = hr_box.get_layer_node_map();
  for (GridMap<HRNode>& hr_node_map : layer_node_map) {
    for (int32_t x = 0; x < hr_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < hr_node_map.get_y_size(); y++) {
        HRNode& hr_node = hr_node_map[x][y];
        for (auto& [pin_shape, orient_set] : routing_layer_pin_shape_orient_map[hr_node.get_layer_idx()]) {
          if (!RTUTIL.isInside(pin_shape->get_real_rect(), hr_node.get_planar_coord())) {
            continue;
          }
          for (auto& [orient, net_set] : hr_node.get_orient_fixed_rect_map()) {
            if (RTUTIL.exist(orient_set, orient)) {
              net_set.erase(-1);
            }
          }
        }
      }
    }
  }
}

void HybridRouter::routeHRBox(HRBox& hr_box)
{
  std::vector<HRTask*> routing_task_list = initTaskSchedule(hr_box);
  while (!routing_task_list.empty()) {
    for (HRTask* routing_task : routing_task_list) {
      routeHRTask(hr_box, routing_task);
      routing_task->addRoutedTimes();
    }
    updateViolationList(hr_box);
    updateBestResult(hr_box);
    updateTaskSchedule(hr_box, routing_task_list);
  }
}

std::vector<HRTask*> HybridRouter::initTaskSchedule(HRBox& hr_box)
{
  bool initial_routing = hr_box.get_initial_routing();

  std::vector<HRTask*> routing_task_list;
  if (initial_routing) {
    for (HRTask* hr_task : hr_box.get_hr_task_list()) {
      routing_task_list.push_back(hr_task);
    }
  } else {
    updateTaskSchedule(hr_box, routing_task_list);
  }
  return routing_task_list;
}

void HybridRouter::routeHRTask(HRBox& hr_box, HRTask* hr_task)
{
  initSingleTask(hr_box, hr_task);
  while (!isConnectedAllEnd(hr_box)) {
    routeSinglePath(hr_box);
    updatePathResult(hr_box);
    resetStartAndEnd(hr_box);
    resetSinglePath(hr_box);
  }
  updateTaskResult(hr_box);
  resetSingleTask(hr_box);
}

void HybridRouter::initSingleTask(HRBox& hr_box, HRTask* hr_task)
{
  ScaleAxis& box_track_axis = hr_box.get_box_track_axis();
  std::vector<GridMap<HRNode>>& layer_node_map = hr_box.get_layer_node_map();

  // single task
  hr_box.set_curr_hr_task(hr_task);
  {
    std::vector<std::vector<HRNode*>> node_list_list;
    std::vector<HRGroup>& hr_group_list = hr_task->get_hr_group_list();
    for (HRGroup& hr_group : hr_group_list) {
      std::vector<HRNode*> node_list;
      for (LayerCoord& coord : hr_group.get_coord_list()) {
        if (!RTUTIL.existTrackGrid(coord, box_track_axis)) {
          RTLOG.error(Loc::current(), "The coord can not find grid!");
        }
        PlanarCoord grid_coord = RTUTIL.getTrackGrid(coord, box_track_axis);
        HRNode& hr_node = layer_node_map[coord.get_layer_idx()][grid_coord.get_x()][grid_coord.get_y()];
        node_list.push_back(&hr_node);
      }
      node_list_list.push_back(node_list);
    }
    for (size_t i = 0; i < node_list_list.size(); i++) {
      if (i == 0) {
        hr_box.get_start_node_list_list().push_back(node_list_list[i]);
      } else {
        hr_box.get_end_node_list_list().push_back(node_list_list[i]);
      }
    }
  }
  hr_box.get_path_node_list().clear();
  hr_box.get_single_task_visited_node_list().clear();
  hr_box.get_routing_segment_list().clear();
}

bool HybridRouter::isConnectedAllEnd(HRBox& hr_box)
{
  return hr_box.get_end_node_list_list().empty();
}

void HybridRouter::routeSinglePath(HRBox& hr_box)
{
  initPathHead(hr_box);
  while (!searchEnded(hr_box)) {
    expandSearching(hr_box);
    resetPathHead(hr_box);
  }
}

void HybridRouter::initPathHead(HRBox& hr_box)
{
  std::vector<std::vector<HRNode*>>& start_node_list_list = hr_box.get_start_node_list_list();
  std::vector<HRNode*>& path_node_list = hr_box.get_path_node_list();

  for (std::vector<HRNode*>& start_node_list : start_node_list_list) {
    for (HRNode* start_node : start_node_list) {
      start_node->set_estimated_cost(getEstimateCostToEnd(hr_box, start_node));
      pushToOpenList(hr_box, start_node);
    }
  }
  for (HRNode* path_node : path_node_list) {
    path_node->set_estimated_cost(getEstimateCostToEnd(hr_box, path_node));
    pushToOpenList(hr_box, path_node);
  }
  resetPathHead(hr_box);
}

bool HybridRouter::searchEnded(HRBox& hr_box)
{
  std::vector<std::vector<HRNode*>>& end_node_list_list = hr_box.get_end_node_list_list();
  HRNode* path_head_node = hr_box.get_path_head_node();

  if (path_head_node == nullptr) {
    hr_box.set_end_node_list_idx(-1);
    return true;
  }
  for (size_t i = 0; i < end_node_list_list.size(); i++) {
    for (HRNode* end_node : end_node_list_list[i]) {
      if (path_head_node == end_node) {
        hr_box.set_end_node_list_idx(static_cast<int32_t>(i));
        return true;
      }
    }
  }
  return false;
}

void HybridRouter::expandSearching(HRBox& hr_box)
{
  PriorityQueue<HRNode*, std::vector<HRNode*>, CmpHRNodeCost>& open_queue = hr_box.get_open_queue();
  HRNode* path_head_node = hr_box.get_path_head_node();

  for (auto& [orientation, neighbor_node] : path_head_node->get_neighbor_node_map()) {
    if (neighbor_node == nullptr) {
      continue;
    }
    if (neighbor_node->isClose()) {
      continue;
    }
    double known_cost = getKnownCost(hr_box, path_head_node, neighbor_node);
    if (neighbor_node->isOpen() && known_cost < neighbor_node->get_known_cost()) {
      neighbor_node->set_known_cost(known_cost);
      neighbor_node->set_parent_node(path_head_node);
      // 对优先队列中的值修改了,需要重新建堆
      std::make_heap(open_queue.begin(), open_queue.end(), CmpHRNodeCost());
    } else if (neighbor_node->isNone()) {
      neighbor_node->set_known_cost(known_cost);
      neighbor_node->set_parent_node(path_head_node);
      neighbor_node->set_estimated_cost(getEstimateCostToEnd(hr_box, neighbor_node));
      pushToOpenList(hr_box, neighbor_node);
    }
  }
}

void HybridRouter::resetPathHead(HRBox& hr_box)
{
  hr_box.set_path_head_node(popFromOpenList(hr_box));
}

void HybridRouter::updatePathResult(HRBox& hr_box)
{
  for (Segment<LayerCoord>& routing_segment : getRoutingSegmentListByNode(hr_box.get_path_head_node())) {
    hr_box.get_routing_segment_list().push_back(routing_segment);
  }
}

std::vector<Segment<LayerCoord>> HybridRouter::getRoutingSegmentListByNode(HRNode* node)
{
  std::vector<Segment<LayerCoord>> routing_segment_list;

  HRNode* curr_node = node;
  HRNode* pre_node = curr_node->get_parent_node();

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

void HybridRouter::resetStartAndEnd(HRBox& hr_box)
{
  std::vector<std::vector<HRNode*>>& start_node_list_list = hr_box.get_start_node_list_list();
  std::vector<std::vector<HRNode*>>& end_node_list_list = hr_box.get_end_node_list_list();
  std::vector<HRNode*>& path_node_list = hr_box.get_path_node_list();
  HRNode* path_head_node = hr_box.get_path_head_node();
  int32_t end_node_list_idx = hr_box.get_end_node_list_idx();

  // 对于抵达的终点pin,只保留到达的node
  end_node_list_list[end_node_list_idx].clear();
  end_node_list_list[end_node_list_idx].push_back(path_head_node);

  HRNode* path_node = path_head_node->get_parent_node();
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
    // 初始化时,要把start_node_list_list的pin只留一个ap点
    // 后续只要将end_node_list_list的pin保留一个ap点
    start_node_list_list.front().clear();
    start_node_list_list.front().push_back(path_node);
  }
  start_node_list_list.push_back(end_node_list_list[end_node_list_idx]);
  end_node_list_list.erase(end_node_list_list.begin() + end_node_list_idx);
}

void HybridRouter::resetSinglePath(HRBox& hr_box)
{
  PriorityQueue<HRNode*, std::vector<HRNode*>, CmpHRNodeCost> empty_queue;
  hr_box.set_open_queue(empty_queue);

  std::vector<HRNode*>& single_path_visited_node_list = hr_box.get_single_path_visited_node_list();
  for (HRNode* visited_node : single_path_visited_node_list) {
    visited_node->set_state(HRNodeState::kNone);
    visited_node->set_parent_node(nullptr);
    visited_node->set_known_cost(0);
    visited_node->set_estimated_cost(0);
  }
  single_path_visited_node_list.clear();

  hr_box.set_path_head_node(nullptr);
  hr_box.set_end_node_list_idx(-1);
}

void HybridRouter::updateTaskResult(HRBox& hr_box)
{
  std::vector<Segment<LayerCoord>> new_routing_segment_list = getRoutingSegmentList(hr_box);

  int32_t curr_net_idx = hr_box.get_curr_hr_task()->get_net_idx();
  std::vector<Segment<LayerCoord>>& routing_segment_list = hr_box.get_net_task_final_result_map()[curr_net_idx];

  // 原结果从graph删除,由于task有对应net_idx,所以不需要在布线前进行删除也不会影响结果
  for (Segment<LayerCoord>& routing_segment : routing_segment_list) {
    updateRoutedRectToGraph(hr_box, ChangeType::kDel, curr_net_idx, routing_segment);
  }
  routing_segment_list = new_routing_segment_list;
  // 新结果添加到graph
  for (Segment<LayerCoord>& routing_segment : routing_segment_list) {
    updateRoutedRectToGraph(hr_box, ChangeType::kAdd, curr_net_idx, routing_segment);
  }
}

std::vector<Segment<LayerCoord>> HybridRouter::getRoutingSegmentList(HRBox& hr_box)
{
  HRTask* curr_hr_task = hr_box.get_curr_hr_task();

  std::vector<LayerCoord> candidate_root_coord_list;
  std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
  std::vector<HRGroup>& hr_group_list = curr_hr_task->get_hr_group_list();
  for (size_t i = 0; i < hr_group_list.size(); i++) {
    for (LayerCoord& coord : hr_group_list[i].get_coord_list()) {
      candidate_root_coord_list.push_back(coord);
      key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
    }
  }
  MTree<LayerCoord> coord_tree = RTUTIL.getTreeByFullFlow(candidate_root_coord_list, hr_box.get_routing_segment_list(), key_coord_pin_map);

  std::vector<Segment<LayerCoord>> routing_segment_list;
  for (Segment<TNode<LayerCoord>*>& coord_segment : RTUTIL.getSegListByTree(coord_tree)) {
    routing_segment_list.emplace_back(coord_segment.get_first()->value(), coord_segment.get_second()->value());
  }
  return routing_segment_list;
}

void HybridRouter::resetSingleTask(HRBox& hr_box)
{
  hr_box.set_curr_hr_task(nullptr);
  hr_box.get_start_node_list_list().clear();
  hr_box.get_end_node_list_list().clear();
  hr_box.get_path_node_list().clear();
  hr_box.get_single_task_visited_node_list().clear();
  hr_box.get_routing_segment_list().clear();
}

// manager open list

void HybridRouter::pushToOpenList(HRBox& hr_box, HRNode* curr_node)
{
  PriorityQueue<HRNode*, std::vector<HRNode*>, CmpHRNodeCost>& open_queue = hr_box.get_open_queue();
  std::vector<HRNode*>& single_task_visited_node_list = hr_box.get_single_task_visited_node_list();
  std::vector<HRNode*>& single_path_visited_node_list = hr_box.get_single_path_visited_node_list();

  open_queue.push(curr_node);
  curr_node->set_state(HRNodeState::kOpen);
  single_task_visited_node_list.push_back(curr_node);
  single_path_visited_node_list.push_back(curr_node);
}

HRNode* HybridRouter::popFromOpenList(HRBox& hr_box)
{
  PriorityQueue<HRNode*, std::vector<HRNode*>, CmpHRNodeCost>& open_queue = hr_box.get_open_queue();

  HRNode* node = nullptr;
  if (!open_queue.empty()) {
    node = open_queue.top();
    open_queue.pop();
    node->set_state(HRNodeState::kClose);
  }
  return node;
}

// calculate known

double HybridRouter::getKnownCost(HRBox& hr_box, HRNode* start_node, HRNode* end_node)
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
  cost += getNodeCost(hr_box, start_node, RTUTIL.getOrientation(*start_node, *end_node));
  cost += getNodeCost(hr_box, end_node, RTUTIL.getOrientation(*end_node, *start_node));
  cost += getKnownWireCost(hr_box, start_node, end_node);
  cost += getKnownViaCost(hr_box, start_node, end_node);
  return cost;
}

double HybridRouter::getNodeCost(HRBox& hr_box, HRNode* curr_node, Orientation orientation)
{
  double fixed_rect_unit = hr_box.get_hr_iter_param()->get_fixed_rect_unit();
  double routed_rect_unit = hr_box.get_hr_iter_param()->get_routed_rect_unit();
  double violation_unit = hr_box.get_hr_iter_param()->get_violation_unit();

  int32_t net_idx = hr_box.get_curr_hr_task()->get_net_idx();

  double cost = 0;
  cost += curr_node->getFixedRectCost(net_idx, orientation, fixed_rect_unit);
  cost += curr_node->getRoutedRectCost(net_idx, orientation, routed_rect_unit);
  cost += curr_node->getViolationCost(orientation, violation_unit);
  return cost;
}

double HybridRouter::getKnownWireCost(HRBox& hr_box, HRNode* start_node, HRNode* end_node)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  double prefer_wire_unit = hr_box.get_hr_iter_param()->get_prefer_wire_unit();
  double non_prefer_wire_unit = hr_box.get_hr_iter_param()->get_non_prefer_wire_unit();

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

double HybridRouter::getKnownViaCost(HRBox& hr_box, HRNode* start_node, HRNode* end_node)
{
  double via_unit = hr_box.get_hr_iter_param()->get_via_unit();
  double via_cost = (via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
  return via_cost;
}

// calculate estimate

double HybridRouter::getEstimateCostToEnd(HRBox& hr_box, HRNode* curr_node)
{
  std::vector<std::vector<HRNode*>>& end_node_list_list = hr_box.get_end_node_list_list();

  double estimate_cost = DBL_MAX;
  for (std::vector<HRNode*>& end_node_list : end_node_list_list) {
    for (HRNode* end_node : end_node_list) {
      if (end_node->isClose()) {
        continue;
      }
      estimate_cost = std::min(estimate_cost, getEstimateCost(hr_box, curr_node, end_node));
    }
  }
  return estimate_cost;
}

double HybridRouter::getEstimateCost(HRBox& hr_box, HRNode* start_node, HRNode* end_node)
{
  double estimate_cost = 0;
  estimate_cost += getEstimateWireCost(hr_box, start_node, end_node);
  estimate_cost += getEstimateViaCost(hr_box, start_node, end_node);
  return estimate_cost;
}

double HybridRouter::getEstimateWireCost(HRBox& hr_box, HRNode* start_node, HRNode* end_node)
{
  double prefer_wire_unit = hr_box.get_hr_iter_param()->get_prefer_wire_unit();
  double non_prefer_wire_unit = hr_box.get_hr_iter_param()->get_non_prefer_wire_unit();

  double wire_cost = 0;
  wire_cost += RTUTIL.getManhattanDistance(start_node->get_planar_coord(), end_node->get_planar_coord());
  wire_cost *= std::min(prefer_wire_unit, non_prefer_wire_unit);
  return wire_cost;
}

double HybridRouter::getEstimateViaCost(HRBox& hr_box, HRNode* start_node, HRNode* end_node)
{
  double via_unit = hr_box.get_hr_iter_param()->get_via_unit();
  double via_cost = (via_unit * std::abs(start_node->get_layer_idx() - end_node->get_layer_idx()));
  return via_cost;
}

void HybridRouter::updateViolationList(HRBox& hr_box)
{
  hr_box.get_violation_list().clear();
  for (Violation new_violation : getViolationList(hr_box)) {
    hr_box.get_violation_list().push_back(new_violation);
  }
  // 新结果添加到graph
  for (Violation& violation : hr_box.get_violation_list()) {
    addViolationToGraph(hr_box, violation);
  }
}

std::vector<Violation> HybridRouter::getViolationList(HRBox& hr_box)
{
  std::string top_name = RTUTIL.getString("hr_box_", hr_box.get_hr_box_id().get_x(), "_", hr_box.get_hr_box_id().get_y());
  std::vector<std::pair<EXTLayerRect*, bool>> env_shape_list;
  std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>> net_pin_shape_map;
  for (auto& [is_routing, layer_net_fixed_rect_map] : hr_box.get_type_layer_net_fixed_rect_map()) {
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
  for (auto& [net_idx, segment_set] : hr_box.get_net_final_result_map()) {
    for (Segment<LayerCoord>* segment : segment_set) {
      net_result_map[net_idx].push_back(segment);
    }
  }
  for (auto& [net_idx, segment_list] : hr_box.get_net_task_final_result_map()) {
    for (Segment<LayerCoord>& segment : segment_list) {
      net_result_map[net_idx].emplace_back(&segment);
    }
  }
  std::set<int32_t> need_checked_net_set;
  for (HRTask* hr_task : hr_box.get_hr_task_list()) {
    need_checked_net_set.insert(hr_task->get_net_idx());
  }

  DETask de_task;
  de_task.set_proc_type(DEProcType::kGet);
  de_task.set_net_type(DENetType::kRouteHybrid);
  de_task.set_top_name(top_name);
  de_task.set_env_shape_list(env_shape_list);
  de_task.set_net_pin_shape_map(net_pin_shape_map);
  de_task.set_net_result_map(net_result_map);
  de_task.set_need_checked_net_set(need_checked_net_set);
  return RTDE.getViolationList(de_task);
}

void HybridRouter::updateBestResult(HRBox& hr_box)
{
  std::map<int32_t, std::vector<Segment<LayerCoord>>>& best_net_task_final_result_map = hr_box.get_best_net_task_final_result_map();
  std::vector<Violation>& best_violation_list = hr_box.get_best_violation_list();

  int32_t curr_violation_num = static_cast<int32_t>(hr_box.get_violation_list().size());
  if (!best_net_task_final_result_map.empty()) {
    if (static_cast<int32_t>(best_violation_list.size()) < curr_violation_num) {
      return;
    }
  }
  best_net_task_final_result_map = hr_box.get_net_task_final_result_map();
  best_violation_list = hr_box.get_violation_list();
}

void HybridRouter::updateTaskSchedule(HRBox& hr_box, std::vector<HRTask*>& routing_task_list)
{
  int32_t max_routed_times = hr_box.get_hr_iter_param()->get_max_routed_times();

  std::set<HRTask*> visited_routing_task_set;
  std::vector<HRTask*> new_routing_task_list;
  for (Violation& violation : hr_box.get_violation_list()) {
    for (HRTask* hr_task : hr_box.get_hr_task_list()) {
      if (!RTUTIL.exist(violation.get_violation_net_set(), hr_task->get_net_idx())) {
        continue;
      }
      if (hr_task->get_routed_times() < max_routed_times && !RTUTIL.exist(visited_routing_task_set, hr_task)) {
        visited_routing_task_set.insert(hr_task);
        new_routing_task_list.push_back(hr_task);
      }
      break;
    }
  }
  routing_task_list = new_routing_task_list;

  std::vector<HRTask*> new_hr_task_list;
  for (HRTask* hr_task : hr_box.get_hr_task_list()) {
    if (!RTUTIL.exist(visited_routing_task_set, hr_task)) {
      new_hr_task_list.push_back(hr_task);
    }
  }
  for (HRTask* routing_task : routing_task_list) {
    new_hr_task_list.push_back(routing_task);
  }
  hr_box.set_hr_task_list(new_hr_task_list);
}

void HybridRouter::selectBestResult(HRBox& hr_box)
{
  updateBestResult(hr_box);
  uploadBestResult(hr_box);
}

void HybridRouter::uploadBestResult(HRBox& hr_box)
{
  for (auto& [net_idx, segment_list] : hr_box.get_best_net_task_final_result_map()) {
    for (Segment<LayerCoord>& segment : segment_list) {
      RTDM.updateNetFinalResultToGCellMap(ChangeType::kAdd, net_idx, new Segment<LayerCoord>(segment));
    }
  }
  for (Violation& violation : hr_box.get_best_violation_list()) {
    RTDM.updateViolationToGCellMap(ChangeType::kAdd, new Violation(violation));
  }
}

void HybridRouter::freeHRBox(HRBox& hr_box)
{
  for (HRTask* hr_task : hr_box.get_hr_task_list()) {
    delete hr_task;
    hr_task = nullptr;
  }
  hr_box.get_hr_task_list().clear();
  hr_box.get_layer_node_map().clear();
}

int32_t HybridRouter::getViolationNum(HRModel& hr_model)
{
  Die& die = RTDM.getDatabase().get_die();

  return static_cast<int32_t>(RTDM.getViolationSet(die).size());
}

void HybridRouter::uploadNetResult(HRModel& hr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();
  std::vector<HRNet>& hr_net_list = hr_model.get_hr_net_list();

  // final result
  {
    std::map<int32_t, std::set<Segment<LayerCoord>*>> net_final_result_map = RTDM.getNetFinalResultMap(die);
    for (auto& [net_idx, segment_set] : net_final_result_map) {
      std::vector<Segment<LayerCoord>> routing_segment_list;
      for (Segment<LayerCoord>* segment : segment_set) {
        routing_segment_list.emplace_back(segment->get_first(), segment->get_second());
      }
      std::vector<LayerCoord> candidate_root_coord_list;
      std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
      std::vector<HRPin>& hr_pin_list = hr_net_list[net_idx].get_hr_pin_list();
      for (size_t i = 0; i < hr_pin_list.size(); i++) {
        LayerCoord coord = hr_pin_list[i].get_access_point().getRealLayerCoord();
        candidate_root_coord_list.push_back(coord);
        key_coord_pin_map[coord].insert(static_cast<int32_t>(i));
      }
      MTree<LayerCoord> coord_tree = RTUTIL.getTreeByFullFlow(candidate_root_coord_list, routing_segment_list, key_coord_pin_map);
      for (Segment<TNode<LayerCoord>*>& coord_segment : RTUTIL.getSegListByTree(coord_tree)) {
        RTDM.updateNetFinalResultToGCellMap(ChangeType::kAdd, net_idx,
                                            new Segment<LayerCoord>(coord_segment.get_first()->value(), coord_segment.get_second()->value()));
      }
    }
    for (auto& [net_idx, segment_set] : net_final_result_map) {
      for (Segment<LayerCoord>* segment : segment_set) {
        RTDM.updateNetFinalResultToGCellMap(ChangeType::kDel, net_idx, segment);
      }
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void HybridRouter::updateBestResult(HRModel& hr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();

  std::map<int32_t, std::vector<Segment<LayerCoord>>>& best_net_final_result_map = hr_model.get_best_net_final_result_map();
  std::vector<Violation>& best_violation_list = hr_model.get_best_violation_list();

  int32_t curr_violation_num = getViolationNum(hr_model);
  if (!best_net_final_result_map.empty()) {
    if (static_cast<int32_t>(best_violation_list.size()) < curr_violation_num) {
      return;
    }
  }
  best_net_final_result_map.clear();
  for (auto& [net_idx, segment_set] : RTDM.getNetFinalResultMap(die)) {
    for (Segment<LayerCoord>* segment : segment_set) {
      best_net_final_result_map[net_idx].push_back(*segment);
    }
  }
  best_violation_list.clear();
  for (Violation* violation : RTDM.getViolationSet(die)) {
    best_violation_list.push_back(*violation);
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

bool HybridRouter::stopIteration(HRModel& hr_model)
{
  if (getViolationNum(hr_model) == 0) {
    RTLOG.info(Loc::current(), "***** Iteration stopped early *****");
    return true;
  }
  return false;
}

void HybridRouter::selectBestResult(HRModel& hr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  hr_model.set_iter(hr_model.get_iter() + 1);
  uploadBestResult(hr_model);
  updateSummary(hr_model);
  printSummary(hr_model);
  outputNetCSV(hr_model);
  outputViolationCSV(hr_model);

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void HybridRouter::uploadBestResult(HRModel& hr_model)
{
  Die& die = RTDM.getDatabase().get_die();

  for (auto& [net_idx, segment_set] : RTDM.getNetFinalResultMap(die)) {
    for (Segment<LayerCoord>* segment : segment_set) {
      RTDM.updateNetFinalResultToGCellMap(ChangeType::kDel, net_idx, segment);
    }
  }
  for (Violation* violation : RTDM.getViolationSet(die)) {
    RTDM.updateViolationToGCellMap(ChangeType::kDel, violation);
  }

  for (auto& [net_idx, segment_list] : hr_model.get_best_net_final_result_map()) {
    for (Segment<LayerCoord>& segment : segment_list) {
      RTDM.updateNetFinalResultToGCellMap(ChangeType::kAdd, net_idx, new Segment<LayerCoord>(segment));
    }
  }
  for (Violation violation : hr_model.get_best_violation_list()) {
    RTDM.updateViolationToGCellMap(ChangeType::kAdd, new Violation(violation));
  }
}

#if 1  // update env

void HybridRouter::updateFixedRectToGraph(HRBox& hr_box, ChangeType change_type, int32_t net_idx, EXTLayerRect* fixed_rect, bool is_routing)
{
  NetShape net_shape(net_idx, fixed_rect->getRealLayerRect(), is_routing);
  for (auto& [hr_node, orientation_set] : getNodeOrientationMap(hr_box, net_shape)) {
    for (Orientation orientation : orientation_set) {
      if (change_type == ChangeType::kAdd) {
        hr_node->get_orient_fixed_rect_map()[orientation].insert(net_shape.get_net_idx());
      } else if (change_type == ChangeType::kDel) {
        hr_node->get_orient_fixed_rect_map()[orientation].erase(net_shape.get_net_idx());
      }
    }
  }
}

void HybridRouter::updateFixedRectToGraph(HRBox& hr_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment)
{
  for (NetShape& net_shape : RTDM.getNetShapeList(net_idx, segment)) {
    for (auto& [hr_node, orientation_set] : getNodeOrientationMap(hr_box, net_shape)) {
      for (Orientation orientation : orientation_set) {
        if (change_type == ChangeType::kAdd) {
          hr_node->get_orient_fixed_rect_map()[orientation].insert(net_shape.get_net_idx());
        } else if (change_type == ChangeType::kDel) {
          hr_node->get_orient_fixed_rect_map()[orientation].erase(net_shape.get_net_idx());
        }
      }
    }
  }
}

void HybridRouter::updateRoutedRectToGraph(HRBox& hr_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment)
{
  for (NetShape& net_shape : RTDM.getNetShapeList(net_idx, segment)) {
    for (auto& [hr_node, orientation_set] : getNodeOrientationMap(hr_box, net_shape)) {
      for (Orientation orientation : orientation_set) {
        if (change_type == ChangeType::kAdd) {
          hr_node->get_orient_routed_rect_map()[orientation].insert(net_shape.get_net_idx());
        } else if (change_type == ChangeType::kDel) {
          hr_node->get_orient_routed_rect_map()[orientation].erase(net_shape.get_net_idx());
        }
      }
    }
  }
}

void HybridRouter::addViolationToGraph(HRBox& hr_box, Violation& violation)
{
  LayerRect searched_rect;
  {
    EXTLayerRect& violation_shape = violation.get_violation_shape();
    searched_rect.set_rect(RTUTIL.getEnlargedRect(violation_shape.get_real_rect(), RTDM.getOnlyPitch()));
    if (violation.get_is_routing()) {
      searched_rect.set_layer_idx(violation_shape.get_layer_idx());
    } else {
      RTLOG.error(Loc::current(), "The violation layer is cut!");
    }
  }
  std::vector<Segment<LayerCoord>> overlap_segment_list;
  for (auto& [net_idx, segment_list] : hr_box.get_net_task_final_result_map()) {
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
  addViolationToGraph(hr_box, searched_rect, overlap_segment_list);
}

void HybridRouter::addViolationToGraph(HRBox& hr_box, LayerRect& searched_rect, std::vector<Segment<LayerCoord>>& overlap_segment_list)
{
  ScaleAxis& box_track_axis = hr_box.get_box_track_axis();
  std::vector<GridMap<HRNode>>& layer_node_map = hr_box.get_layer_node_map();

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
    std::map<int32_t, std::set<HRNode*>> distance_node_map;
    {
      int32_t first_layer_idx = first_coord.get_layer_idx();
      int32_t second_layer_idx = second_coord.get_layer_idx();
      RTUTIL.swapByASC(first_layer_idx, second_layer_idx);
      for (int32_t layer_idx = first_layer_idx; layer_idx <= second_layer_idx; layer_idx++) {
        for (int32_t x = grid_rect.get_ll_x(); x <= grid_rect.get_ur_x(); x++) {
          for (int32_t y = grid_rect.get_ll_y(); y <= grid_rect.get_ur_y(); y++) {
            HRNode* hr_node = &layer_node_map[layer_idx][x][y];
            if (searched_rect.get_layer_idx() != hr_node->get_layer_idx()) {
              continue;
            }
            int32_t distance = 0;
            if (!RTUTIL.isInside(searched_rect.get_rect(), hr_node->get_planar_coord())) {
              distance = RTUTIL.getManhattanDistance(searched_rect.getMidPoint(), hr_node->get_planar_coord());
            }
            distance_node_map[distance].insert(hr_node);
          }
        }
      }
    }
    std::set<HRNode*> valid_node_set;
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
    for (HRNode* valid_node : valid_node_set) {
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

std::map<HRNode*, std::set<Orientation>> HybridRouter::getNodeOrientationMap(HRBox& hr_box, NetShape& net_shape)
{
  std::map<HRNode*, std::set<Orientation>> node_orientation_map;
  if (net_shape.get_is_routing()) {
    node_orientation_map = getRoutingNodeOrientationMap(hr_box, net_shape);
  } else {
    node_orientation_map = getCutNodeOrientationMap(hr_box, net_shape);
  }
  return node_orientation_map;
}

std::map<HRNode*, std::set<Orientation>> HybridRouter::getRoutingNodeOrientationMap(HRBox& hr_box, NetShape& net_shape)
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

  GridMap<HRNode>& hr_node_map = hr_box.get_layer_node_map()[layer_idx];
  std::map<HRNode*, std::set<Orientation>> node_orientation_map;
  // wire 与 net_shape
  for (auto& [x_spacing, y_spacing] : spacing_pair_list) {
    // 膨胀size为 half_wire_width + spacing
    int32_t enlarged_x_size = half_wire_width + x_spacing;
    int32_t enlarged_y_size = half_wire_width + y_spacing;
    // 贴合的也不算违例
    enlarged_x_size -= 1;
    enlarged_y_size -= 1;
    PlanarRect planar_enlarged_rect = RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size);
    for (auto& [grid_coord, orientation_set] : RTUTIL.getTrackGridOrientationMap(planar_enlarged_rect, hr_box.get_box_track_axis())) {
      HRNode& node = hr_node_map[grid_coord.get_x()][grid_coord.get_y()];
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
  // enclosure 与 net_shape
  for (auto& [x_spacing, y_spacing] : spacing_pair_list) {
    // 膨胀size为 enclosure_half_span + spacing
    int32_t enlarged_x_size = enclosure_half_x_span + x_spacing;
    int32_t enlarged_y_size = enclosure_half_y_span + y_spacing;
    // 贴合的也不算违例
    enlarged_x_size -= 1;
    enlarged_y_size -= 1;
    PlanarRect space_enlarged_rect = RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size);
    for (auto& [grid_coord, orientation_set] : RTUTIL.getTrackGridOrientationMap(space_enlarged_rect, hr_box.get_box_track_axis())) {
      HRNode& node = hr_node_map[grid_coord.get_x()][grid_coord.get_y()];
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

std::map<HRNode*, std::set<Orientation>> HybridRouter::getCutNodeOrientationMap(HRBox& hr_box, NetShape& net_shape)
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
  std::map<HRNode*, std::set<Orientation>> node_orientation_map;
  for (auto& [cut_layer_idx, spacing_pair_list] : cut_spacing_map) {
    std::vector<int32_t> adjacent_routing_layer_idx_list = cut_to_adjacent_routing_map[cut_layer_idx];
    int32_t below_routing_layer_idx = adjacent_routing_layer_idx_list.front();
    int32_t above_routing_layer_idx = adjacent_routing_layer_idx_list.back();
    RTUTIL.swapByASC(below_routing_layer_idx, above_routing_layer_idx);
    PlanarRect& cut_shape = layer_via_master_list[below_routing_layer_idx].front().get_cut_shape_list().front();
    int32_t cut_shape_half_x_span = cut_shape.getXSpan() / 2;
    int32_t cut_shape_half_y_span = cut_shape.getYSpan() / 2;
    std::vector<GridMap<HRNode>>& layer_node_map = hr_box.get_layer_node_map();
    for (auto& [x_spacing, y_spacing] : spacing_pair_list) {
      // 膨胀size为 cut_shape_half_span + spacing
      int32_t enlarged_x_size = cut_shape_half_x_span + x_spacing;
      int32_t enlarged_y_size = cut_shape_half_y_span + y_spacing;
      // 贴合的也不算违例
      enlarged_x_size -= 1;
      enlarged_y_size -= 1;
      PlanarRect space_enlarged_rect = RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size);
      for (auto& [grid_coord, orientation_set] : RTUTIL.getTrackGridOrientationMap(space_enlarged_rect, hr_box.get_box_track_axis())) {
        if (!RTUTIL.exist(orientation_set, Orientation::kAbove) && !RTUTIL.exist(orientation_set, Orientation::kBelow)) {
          continue;
        }
        HRNode& below_node = layer_node_map[below_routing_layer_idx][grid_coord.get_x()][grid_coord.get_y()];
        if (RTUTIL.exist(below_node.get_neighbor_node_map(), Orientation::kAbove)) {
          node_orientation_map[&below_node].insert(Orientation::kAbove);
        }
        HRNode& above_node = layer_node_map[above_routing_layer_idx][grid_coord.get_x()][grid_coord.get_y()];
        if (RTUTIL.exist(above_node.get_neighbor_node_map(), Orientation::kBelow)) {
          node_orientation_map[&above_node].insert(Orientation::kBelow);
        }
      }
    }
  }
  return node_orientation_map;
}

#endif

#if 1  // exhibit

void HybridRouter::updateSummary(HRModel& hr_model)
{
  int32_t micron_dbu = RTDM.getDatabase().get_micron_dbu();
  Die& die = RTDM.getDatabase().get_die();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();
  Summary& summary = RTDM.getDatabase().get_summary();
  int32_t enable_timing = RTDM.getConfig().enable_timing;

  std::map<int32_t, double>& routing_wire_length_map = summary.iter_hr_summary_map[hr_model.get_iter()].routing_wire_length_map;
  double& total_wire_length = summary.iter_hr_summary_map[hr_model.get_iter()].total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.iter_hr_summary_map[hr_model.get_iter()].cut_via_num_map;
  int32_t& total_via_num = summary.iter_hr_summary_map[hr_model.get_iter()].total_via_num;
  std::map<int32_t, int32_t>& routing_violation_num_map = summary.iter_hr_summary_map[hr_model.get_iter()].routing_violation_num_map;
  int32_t& total_violation_num = summary.iter_hr_summary_map[hr_model.get_iter()].total_violation_num;
  std::map<std::string, std::map<std::string, double>>& clock_timing = summary.iter_hr_summary_map[hr_model.get_iter()].clock_timing;
  std::map<std::string, double>& power_map = summary.iter_hr_summary_map[hr_model.get_iter()].power_map;

  std::vector<HRNet>& hr_net_list = hr_model.get_hr_net_list();

  routing_wire_length_map.clear();
  total_wire_length = 0;
  cut_via_num_map.clear();
  total_via_num = 0;
  routing_violation_num_map.clear();
  total_violation_num = 0;
  clock_timing.clear();
  power_map.clear();

  for (auto& [net_idx, segment_set] : RTDM.getNetFinalResultMap(die)) {
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
    real_pin_coord_map_list.resize(hr_net_list.size());
    std::vector<std::vector<Segment<LayerCoord>>> routing_segment_list_list;
    routing_segment_list_list.resize(hr_net_list.size());
    for (HRNet& hr_net : hr_net_list) {
      for (HRPin& hr_pin : hr_net.get_hr_pin_list()) {
        real_pin_coord_map_list[hr_net.get_net_idx()][hr_pin.get_pin_name()].push_back(hr_pin.get_access_point().getRealLayerCoord());
      }
    }
    for (auto& [net_idx, segment_set] : RTDM.getNetFinalResultMap(die)) {
      for (Segment<LayerCoord>* segment : segment_set) {
        routing_segment_list_list[net_idx].emplace_back(segment->get_first(), segment->get_second());
      }
    }
    RTI.updateTimingAndPower(real_pin_coord_map_list, routing_segment_list_list, clock_timing, power_map);
  }
}

void HybridRouter::printSummary(HRModel& hr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  Summary& summary = RTDM.getDatabase().get_summary();
  int32_t enable_timing = RTDM.getConfig().enable_timing;

  std::map<int32_t, double>& routing_wire_length_map = summary.iter_hr_summary_map[hr_model.get_iter()].routing_wire_length_map;
  double& total_wire_length = summary.iter_hr_summary_map[hr_model.get_iter()].total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.iter_hr_summary_map[hr_model.get_iter()].cut_via_num_map;
  int32_t& total_via_num = summary.iter_hr_summary_map[hr_model.get_iter()].total_via_num;
  std::map<int32_t, int32_t>& routing_violation_num_map = summary.iter_hr_summary_map[hr_model.get_iter()].routing_violation_num_map;
  int32_t& total_violation_num = summary.iter_hr_summary_map[hr_model.get_iter()].total_violation_num;
  std::map<std::string, std::map<std::string, double>>& clock_timing = summary.iter_hr_summary_map[hr_model.get_iter()].clock_timing;
  std::map<std::string, double>& power_map = summary.iter_hr_summary_map[hr_model.get_iter()].power_map;

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
    for (auto& [clock_name, timing_map] : clock_timing) {
      timing_table << clock_name << timing_map["TNS"] << timing_map["WNS"] << timing_map["Freq(MHz)"] << fort::endr;
    }
    power_table << fort::header << "power_type";
    for (auto& [type, power] : power_map) {
      power_table << fort::header << type;
    }
    power_table << fort::endr;
    power_table << "power_value";
    for (auto& [type, power] : power_map) {
      power_table << power;
    }
    power_table << fort::endr;
  }
  RTUTIL.printTableList({routing_wire_length_map_table, cut_via_num_map_table, routing_violation_num_map_table});
  RTUTIL.printTableList({timing_table, power_table});
}

void HybridRouter::outputNetCSV(HRModel& hr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::string& hr_temp_directory_path = RTDM.getConfig().hr_temp_directory_path;
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
      for (auto& [net_idx, segment_set] : gcell_map[x][y].get_net_final_result_map()) {
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
    std::ofstream* net_csv_file
        = RTUTIL.getOutputFileStream(RTUTIL.getString(hr_temp_directory_path, "net_map_", routing_layer.get_layer_name(), "_", hr_model.get_iter(), ".csv"));
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

void HybridRouter::outputViolationCSV(HRModel& hr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::string& hr_temp_directory_path = RTDM.getConfig().hr_temp_directory_path;
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
        RTUTIL.getString(hr_temp_directory_path, "violation_map_", routing_layer.get_layer_name(), "_", hr_model.get_iter(), ".csv"));
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

void HybridRouter::debugPlotHRModel(HRModel& hr_model, std::string flag)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& hr_temp_directory_path = RTDM.getConfig().hr_temp_directory_path;

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
  for (auto& [net_idx, segment_set] : RTDM.getNetFinalResultMap(die)) {
    GPStruct final_result_struct(RTUTIL.getString("final_result(net_", net_idx, ")"));
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
        final_result_struct.push(gp_boundary);
      }
    }
    gp_gds.addStruct(final_result_struct);
  }

  // violation
  {
    for (Violation* violation : RTDM.getViolationSet(die)) {
      GPStruct violation_struct(RTUTIL.getString("violation_", GetViolationTypeName()(violation->get_violation_type())));
      EXTLayerRect& violation_shape = violation->get_violation_shape();

      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kViolation));
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

  std::string gds_file_path = RTUTIL.getString(hr_temp_directory_path, flag, "_hr_model.gds");
  RTGP.plot(gp_gds, gds_file_path);
}

void HybridRouter::debugCheckHRBox(HRBox& hr_box)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  HRBoxId& hr_box_id = hr_box.get_hr_box_id();
  if (hr_box_id.get_x() < 0 || hr_box_id.get_y() < 0) {
    RTLOG.error(Loc::current(), "The grid coord is illegal!");
  }

  std::vector<GridMap<HRNode>>& layer_node_map = hr_box.get_layer_node_map();
  for (GridMap<HRNode>& hr_node_map : layer_node_map) {
    for (int32_t x = 0; x < hr_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < hr_node_map.get_y_size(); y++) {
        HRNode& hr_node = hr_node_map[x][y];
        if (!RTUTIL.isInside(hr_box.get_box_rect().get_real_rect(), hr_node.get_planar_coord())) {
          RTLOG.error(Loc::current(), "The hr_node is out of box!");
        }
        for (auto& [orient, neighbor] : hr_node.get_neighbor_node_map()) {
          Orientation opposite_orient = RTUTIL.getOppositeOrientation(orient);
          if (!RTUTIL.exist(neighbor->get_neighbor_node_map(), opposite_orient)) {
            RTLOG.error(Loc::current(), "The hr_node neighbor is not bidirectional!");
          }
          if (neighbor->get_neighbor_node_map()[opposite_orient] != &hr_node) {
            RTLOG.error(Loc::current(), "The hr_node neighbor is not bidirectional!");
          }
          if (RTUTIL.getOrientation(LayerCoord(hr_node), LayerCoord(*neighbor)) == orient) {
            continue;
          }
          RTLOG.error(Loc::current(), "The neighbor orient is different with real region!");
        }
      }
    }
  }

  for (HRTask* hr_task : hr_box.get_hr_task_list()) {
    if (hr_task->get_net_idx() < 0) {
      RTLOG.error(Loc::current(), "The idx of origin net is illegal!");
    }
    for (HRGroup& hr_group : hr_task->get_hr_group_list()) {
      if (hr_group.get_coord_list().empty()) {
        RTLOG.error(Loc::current(), "The coord_list is empty!");
      }
      for (LayerCoord& coord : hr_group.get_coord_list()) {
        int32_t layer_idx = coord.get_layer_idx();
        if (routing_layer_list.back().get_layer_idx() < layer_idx || layer_idx < routing_layer_list.front().get_layer_idx()) {
          RTLOG.error(Loc::current(), "The layer idx of group coord is illegal!");
        }
        if (!RTUTIL.existTrackGrid(coord, hr_box.get_box_track_axis())) {
          RTLOG.error(Loc::current(), "There is no grid coord for real coord(", coord.get_x(), ",", coord.get_y(), ")!");
        }
        PlanarCoord grid_coord = RTUTIL.getTrackGrid(coord, hr_box.get_box_track_axis());
        HRNode& hr_node = layer_node_map[layer_idx][grid_coord.get_x()][grid_coord.get_y()];
        if (hr_node.get_neighbor_node_map().empty()) {
          RTLOG.error(Loc::current(), "The neighbor of group coord (", coord.get_x(), ",", coord.get_y(), ",", layer_idx, ") is empty in box(",
                      hr_box_id.get_x(), ",", hr_box_id.get_y(), ")");
        }
        if (RTUTIL.isInside(hr_box.get_box_rect().get_real_rect(), coord)) {
          continue;
        }
        RTLOG.error(Loc::current(), "The coord (", coord.get_x(), ",", coord.get_y(), ") is out of box!");
      }
    }
  }
}

void HybridRouter::debugPlotHRBox(HRBox& hr_box, std::string flag)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  std::string& hr_temp_directory_path = RTDM.getConfig().hr_temp_directory_path;

  PlanarRect box_real_rect = hr_box.get_box_rect().get_real_rect();

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

  std::vector<GridMap<HRNode>>& layer_node_map = hr_box.get_layer_node_map();
  // hr_node_map
  {
    GPStruct hr_node_map_struct("hr_node_map");
    for (GridMap<HRNode>& hr_node_map : layer_node_map) {
      for (int32_t grid_x = 0; grid_x < hr_node_map.get_x_size(); grid_x++) {
        for (int32_t grid_y = 0; grid_y < hr_node_map.get_y_size(); grid_y++) {
          HRNode& hr_node = hr_node_map[grid_x][grid_y];
          PlanarRect real_rect = RTUTIL.getEnlargedRect(hr_node.get_planar_coord(), point_size);
          int32_t y_reduced_span = std::max(1, real_rect.getYSpan() / 12);
          int32_t y = real_rect.get_ur_y();

          GPBoundary gp_boundary;
          switch (hr_node.get_state()) {
            case HRNodeState::kNone:
              gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kNone));
              break;
            case HRNodeState::kOpen:
              gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kOpen));
              break;
            case HRNodeState::kClose:
              gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kClose));
              break;
            default:
              RTLOG.error(Loc::current(), "The type is error!");
              break;
          }
          gp_boundary.set_rect(real_rect);
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(hr_node.get_layer_idx()));
          hr_node_map_struct.push(gp_boundary);

          y -= y_reduced_span;
          GPText gp_text_node_real_coord;
          gp_text_node_real_coord.set_coord(real_rect.get_ll_x(), y);
          gp_text_node_real_coord.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          gp_text_node_real_coord.set_message(RTUTIL.getString("(", hr_node.get_x(), " , ", hr_node.get_y(), " , ", hr_node.get_layer_idx(), ")"));
          gp_text_node_real_coord.set_layer_idx(RTGP.getGDSIdxByRouting(hr_node.get_layer_idx()));
          gp_text_node_real_coord.set_presentation(GPTextPresentation::kLeftMiddle);
          hr_node_map_struct.push(gp_text_node_real_coord);

          y -= y_reduced_span;
          GPText gp_text_node_grid_coord;
          gp_text_node_grid_coord.set_coord(real_rect.get_ll_x(), y);
          gp_text_node_grid_coord.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          gp_text_node_grid_coord.set_message(RTUTIL.getString("(", grid_x, " , ", grid_y, " , ", hr_node.get_layer_idx(), ")"));
          gp_text_node_grid_coord.set_layer_idx(RTGP.getGDSIdxByRouting(hr_node.get_layer_idx()));
          gp_text_node_grid_coord.set_presentation(GPTextPresentation::kLeftMiddle);
          hr_node_map_struct.push(gp_text_node_grid_coord);

          y -= y_reduced_span;
          GPText gp_text_orient_fixed_rect_map;
          gp_text_orient_fixed_rect_map.set_coord(real_rect.get_ll_x(), y);
          gp_text_orient_fixed_rect_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          gp_text_orient_fixed_rect_map.set_message("orient_fixed_rect_map: ");
          gp_text_orient_fixed_rect_map.set_layer_idx(RTGP.getGDSIdxByRouting(hr_node.get_layer_idx()));
          gp_text_orient_fixed_rect_map.set_presentation(GPTextPresentation::kLeftMiddle);
          hr_node_map_struct.push(gp_text_orient_fixed_rect_map);

          if (!hr_node.get_orient_fixed_rect_map().empty()) {
            y -= y_reduced_span;
            GPText gp_text_orient_fixed_rect_map_info;
            gp_text_orient_fixed_rect_map_info.set_coord(real_rect.get_ll_x(), y);
            gp_text_orient_fixed_rect_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            std::string orient_fixed_rect_map_info_message = "--";
            for (auto& [orient, net_set] : hr_node.get_orient_fixed_rect_map()) {
              orient_fixed_rect_map_info_message += RTUTIL.getString("(", GetOrientationName()(orient));
              for (int32_t net_idx : net_set) {
                orient_fixed_rect_map_info_message += RTUTIL.getString(",", net_idx);
              }
              orient_fixed_rect_map_info_message += RTUTIL.getString(")");
            }
            gp_text_orient_fixed_rect_map_info.set_message(orient_fixed_rect_map_info_message);
            gp_text_orient_fixed_rect_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(hr_node.get_layer_idx()));
            gp_text_orient_fixed_rect_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
            hr_node_map_struct.push(gp_text_orient_fixed_rect_map_info);
          }

          y -= y_reduced_span;
          GPText gp_text_orient_routed_rect_map;
          gp_text_orient_routed_rect_map.set_coord(real_rect.get_ll_x(), y);
          gp_text_orient_routed_rect_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          gp_text_orient_routed_rect_map.set_message("orient_routed_rect_map: ");
          gp_text_orient_routed_rect_map.set_layer_idx(RTGP.getGDSIdxByRouting(hr_node.get_layer_idx()));
          gp_text_orient_routed_rect_map.set_presentation(GPTextPresentation::kLeftMiddle);
          hr_node_map_struct.push(gp_text_orient_routed_rect_map);

          if (!hr_node.get_orient_routed_rect_map().empty()) {
            y -= y_reduced_span;
            GPText gp_text_orient_routed_rect_map_info;
            gp_text_orient_routed_rect_map_info.set_coord(real_rect.get_ll_x(), y);
            gp_text_orient_routed_rect_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            std::string orient_routed_rect_map_info_message = "--";
            for (auto& [orient, net_set] : hr_node.get_orient_routed_rect_map()) {
              orient_routed_rect_map_info_message += RTUTIL.getString("(", GetOrientationName()(orient));
              for (int32_t net_idx : net_set) {
                orient_routed_rect_map_info_message += RTUTIL.getString(",", net_idx);
              }
              orient_routed_rect_map_info_message += RTUTIL.getString(")");
            }
            gp_text_orient_routed_rect_map_info.set_message(orient_routed_rect_map_info_message);
            gp_text_orient_routed_rect_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(hr_node.get_layer_idx()));
            gp_text_orient_routed_rect_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
            hr_node_map_struct.push(gp_text_orient_routed_rect_map_info);
          }

          y -= y_reduced_span;
          GPText gp_text_orient_violation_number_map;
          gp_text_orient_violation_number_map.set_coord(real_rect.get_ll_x(), y);
          gp_text_orient_violation_number_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          gp_text_orient_violation_number_map.set_message("orient_violation_number_map: ");
          gp_text_orient_violation_number_map.set_layer_idx(RTGP.getGDSIdxByRouting(hr_node.get_layer_idx()));
          gp_text_orient_violation_number_map.set_presentation(GPTextPresentation::kLeftMiddle);
          hr_node_map_struct.push(gp_text_orient_violation_number_map);

          if (!hr_node.get_orient_violation_number_map().empty()) {
            y -= y_reduced_span;
            GPText gp_text_orient_violation_number_map_info;
            gp_text_orient_violation_number_map_info.set_coord(real_rect.get_ll_x(), y);
            gp_text_orient_violation_number_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            std::string orient_violation_number_map_info_message = "--";
            for (auto& [orient, violation_number] : hr_node.get_orient_violation_number_map()) {
              orient_violation_number_map_info_message += RTUTIL.getString("(", GetOrientationName()(orient), ",", violation_number != 0, ")");
            }
            gp_text_orient_violation_number_map_info.set_message(orient_violation_number_map_info_message);
            gp_text_orient_violation_number_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(hr_node.get_layer_idx()));
            gp_text_orient_violation_number_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
            hr_node_map_struct.push(gp_text_orient_violation_number_map_info);
          }
        }
      }
    }
    gp_gds.addStruct(hr_node_map_struct);
  }

  // neighbor_map
  {
    GPStruct neighbor_map_struct("neighbor_map");
    for (GridMap<HRNode>& hr_node_map : layer_node_map) {
      for (int32_t grid_x = 0; grid_x < hr_node_map.get_x_size(); grid_x++) {
        for (int32_t grid_y = 0; grid_y < hr_node_map.get_y_size(); grid_y++) {
          HRNode& hr_node = hr_node_map[grid_x][grid_y];
          PlanarRect real_rect = RTUTIL.getEnlargedRect(hr_node.get_planar_coord(), point_size);

          int32_t ll_x = real_rect.get_ll_x();
          int32_t ll_y = real_rect.get_ll_y();
          int32_t ur_x = real_rect.get_ur_x();
          int32_t ur_y = real_rect.get_ur_y();
          int32_t mid_x = (ll_x + ur_x) / 2;
          int32_t mid_y = (ll_y + ur_y) / 2;
          int32_t x_reduced_span = (ur_x - ll_x) / 4;
          int32_t y_reduced_span = (ur_y - ll_y) / 4;

          for (auto& [orientation, neighbor_node] : hr_node.get_neighbor_node_map()) {
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
            gp_path.set_layer_idx(RTGP.getGDSIdxByRouting(hr_node.get_layer_idx()));
            gp_path.set_width(std::min(x_reduced_span, y_reduced_span) / 2);
            gp_path.set_data_type(static_cast<int32_t>(GPDataType::kNeighbor));
            neighbor_map_struct.push(gp_path);
          }
        }
      }
    }
    gp_gds.addStruct(neighbor_map_struct);
  }
  // box_track_axis
  {
    GPStruct box_track_axis_struct("box_track_axis");
    PlanarCoord& real_ll = box_real_rect.get_ll();
    PlanarCoord& real_ur = box_real_rect.get_ur();
    ScaleAxis& box_track_axis = hr_box.get_box_track_axis();
    std::vector<int32_t> x_list = RTUTIL.getScaleList(real_ll.get_x(), real_ur.get_x(), box_track_axis.get_x_grid_list());
    std::vector<int32_t> y_list = RTUTIL.getScaleList(real_ll.get_y(), real_ur.get_y(), box_track_axis.get_y_grid_list());
    for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
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
  for (auto& [is_routing, layer_net_rect_map] : hr_box.get_type_layer_net_fixed_rect_map()) {
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

  // net_final_result
  for (auto& [net_idx, segment_set] : hr_box.get_net_final_result_map()) {
    GPStruct final_result_struct(RTUTIL.getString("final_result(net_", net_idx, ")"));
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
        final_result_struct.push(gp_boundary);
      }
    }
    gp_gds.addStruct(final_result_struct);
  }

  // task
  for (HRTask* hr_task : hr_box.get_hr_task_list()) {
    GPStruct task_struct(RTUTIL.getString("task(net_", hr_task->get_net_idx(), ")"));

    for (HRGroup& hr_group : hr_task->get_hr_group_list()) {
      for (LayerCoord& coord : hr_group.get_coord_list()) {
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
      gp_boundary.set_rect(hr_task->get_bounding_box());
      task_struct.push(gp_boundary);
    }
    for (Segment<LayerCoord>& segment : hr_box.get_net_task_final_result_map()[hr_task->get_net_idx()]) {
      for (NetShape& net_shape : RTDM.getNetShapeList(hr_task->get_net_idx(), segment)) {
        GPBoundary gp_boundary;
        gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPath));
        gp_boundary.set_rect(net_shape.get_rect());
        if (net_shape.get_is_routing()) {
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(net_shape.get_layer_idx()));
        } else {
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(net_shape.get_layer_idx()));
        }
        task_struct.push(gp_boundary);
      }
    }
    gp_gds.addStruct(task_struct);
  }

  // violation
  {
    for (Violation& violation : hr_box.get_violation_list()) {
      GPStruct violation_struct(RTUTIL.getString("violation_", GetViolationTypeName()(violation.get_violation_type())));
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
      gp_gds.addStruct(violation_struct);
    }
  }

  std::string gds_file_path
      = RTUTIL.getString(hr_temp_directory_path, flag, "_hr_box_", hr_box.get_hr_box_id().get_x(), "_", hr_box.get_hr_box_id().get_y(), ".gds");
  RTGP.plot(gp_gds, gds_file_path);
}

#endif

}  // namespace irt
