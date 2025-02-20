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
#include "ViolationRepairer.hpp"

#include "DRCEngine.hpp"
#include "GDSPlotter.hpp"
#include "RTInterface.hpp"
#include "Utility.hpp"

namespace irt {

// public

void ViolationRepairer::initInst()
{
  if (_vr_instance == nullptr) {
    _vr_instance = new ViolationRepairer();
  }
}

ViolationRepairer& ViolationRepairer::getInst()
{
  if (_vr_instance == nullptr) {
    RTLOG.error(Loc::current(), "The instance not initialized!");
  }
  return *_vr_instance;
}

void ViolationRepairer::destroyInst()
{
  if (_vr_instance != nullptr) {
    delete _vr_instance;
    _vr_instance = nullptr;
  }
}

// function

void ViolationRepairer::repair()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");
  VRModel vr_model = initVRModel();
  updateAccessPoint(vr_model);
  initNetFinalResultMap(vr_model);
  buildNetFinalResultMap(vr_model);
  clearIgnoredViolation(vr_model);
  uploadViolation(vr_model);
  updateSummary(vr_model);
  printSummary(vr_model);
  iterativeVRModel(vr_model);
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

// private

ViolationRepairer* ViolationRepairer::_vr_instance = nullptr;

VRModel ViolationRepairer::initVRModel()
{
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();

  VRModel vr_model;
  vr_model.set_vr_net_list(convertToVRNetList(net_list));
  return vr_model;
}

std::vector<VRNet> ViolationRepairer::convertToVRNetList(std::vector<Net>& net_list)
{
  std::vector<VRNet> vr_net_list;
  vr_net_list.reserve(net_list.size());
  for (Net& net : net_list) {
    vr_net_list.emplace_back(convertToVRNet(net));
  }
  return vr_net_list;
}

VRNet ViolationRepairer::convertToVRNet(Net& net)
{
  VRNet vr_net;
  vr_net.set_origin_net(&net);
  vr_net.set_net_idx(net.get_net_idx());
  vr_net.set_connect_type(net.get_connect_type());
  for (Pin& pin : net.get_pin_list()) {
    vr_net.get_vr_pin_list().push_back(VRPin(pin));
  }
  return vr_net;
}

void ViolationRepairer::updateAccessPoint(VRModel& vr_model)
{
  Die& die = RTDM.getDatabase().get_die();

  for (auto& [net_idx, access_point_set] : RTDM.getNetAccessPointMap(die)) {
    for (AccessPoint* access_point : access_point_set) {
      RTDM.updateAccessNetPointToGCellMap(ChangeType::kDel, net_idx, access_point);
    }
  }
  for (VRNet& vr_net : vr_model.get_vr_net_list()) {
    Net* origin_net = vr_net.get_origin_net();
    if (origin_net->get_net_idx() != vr_net.get_net_idx()) {
      RTLOG.error(Loc::current(), "The net idx is not equal!");
    }
    for (VRPin& vr_pin : vr_net.get_vr_pin_list()) {
      Pin& origin_pin = origin_net->get_pin_list()[vr_pin.get_pin_idx()];
      if (origin_pin.get_pin_idx() != vr_pin.get_pin_idx()) {
        RTLOG.error(Loc::current(), "The pin idx is not equal!");
      }
      vr_pin.set_access_point(vr_pin.get_origin_access_point());
      // 之后流程将暂时使用origin_access_point作为主要access point
      origin_pin.set_access_point(origin_pin.get_origin_access_point());
      RTDM.updateAccessNetPointToGCellMap(ChangeType::kAdd, vr_net.get_net_idx(), &origin_pin.get_access_point());
    }
  }
}

void ViolationRepairer::initNetFinalResultMap(VRModel& vr_model)
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

void ViolationRepairer::buildNetFinalResultMap(VRModel& vr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();
  std::vector<VRNet>& vr_net_list = vr_model.get_vr_net_list();

  std::map<int32_t, std::set<Segment<LayerCoord>*>> net_final_result_map = RTDM.getNetFinalResultMap(die);
  for (auto& [net_idx, segment_set] : net_final_result_map) {
    std::vector<Segment<LayerCoord>> routing_segment_list;
    for (Segment<LayerCoord>* segment : segment_set) {
      routing_segment_list.emplace_back(segment->get_first(), segment->get_second());
    }
    std::vector<LayerCoord> candidate_root_coord_list;
    std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
    std::vector<VRPin>& vr_pin_list = vr_net_list[net_idx].get_vr_pin_list();
    for (size_t i = 0; i < vr_pin_list.size(); i++) {
      LayerCoord coord = vr_pin_list[i].get_access_point().getRealLayerCoord();
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

void ViolationRepairer::clearIgnoredViolation(VRModel& vr_model)
{
  RTDE.clearTempIgnoredViolationSet();
}

void ViolationRepairer::uploadViolation(VRModel& vr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();

  for (Violation* violation : RTDM.getViolationSet(die)) {
    RTDM.updateViolationToGCellMap(ChangeType::kDel, violation);
  }
  for (Violation violation : getMultiNetViolationList(vr_model)) {
    RTDM.updateViolationToGCellMap(ChangeType::kAdd, new Violation(violation));
  }
  for (Violation violation : getSingleNetViolationList(vr_model)) {
    RTDM.updateViolationToGCellMap(ChangeType::kAdd, new Violation(violation));
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

std::vector<Violation> ViolationRepairer::getMultiNetViolationList(VRModel& vr_model)
{
  Die& die = RTDM.getDatabase().get_die();

  DETask de_task;
  {
    std::string top_name = RTUTIL.getString("vr_model");
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
    for (VRNet& vr_net : vr_model.get_vr_net_list()) {
      need_checked_net_set.insert(vr_net.get_net_idx());
    }

    de_task.set_proc_type(DEProcType::kGet);
    de_task.set_net_type(DENetType::kMultiNet);
    de_task.set_top_name(top_name);
    de_task.set_env_shape_list(env_shape_list);
    de_task.set_net_pin_shape_map(net_pin_shape_map);
    de_task.set_net_result_map(net_result_map);
    de_task.set_net_patch_map(net_patch_map);
    de_task.set_need_checked_net_set(need_checked_net_set);
  }
  // return RTDE.getViolationListByTemp(de_task);
  return RTDE.getViolationList(de_task);
}

std::vector<Violation> ViolationRepairer::getSingleNetViolationList(VRModel& vr_model)
{
  Die& die = RTDM.getDatabase().get_die();

  DETask de_task;
  {
    std::string top_name = RTUTIL.getString("vr_model");
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
    for (VRNet& vr_net : vr_model.get_vr_net_list()) {
      need_checked_net_set.insert(vr_net.get_net_idx());
    }

    de_task.set_proc_type(DEProcType::kGet);
    de_task.set_net_type(DENetType::kSingleNet);
    de_task.set_top_name(top_name);
    de_task.set_env_shape_list(env_shape_list);
    de_task.set_net_pin_shape_map(net_pin_shape_map);
    de_task.set_net_result_map(net_result_map);
    de_task.set_net_patch_map(net_patch_map);
    de_task.set_need_checked_net_set(need_checked_net_set);
  }
  // return RTDE.getViolationListByTemp(de_task);
  return RTDE.getViolationList(de_task);
}

void ViolationRepairer::iterativeVRModel(VRModel& vr_model)
{
  int32_t cost_unit = RTDM.getOnlyPitch();
  double prefer_wire_unit = 1;
  double non_prefer_wire_unit = 2.5 * prefer_wire_unit;
  double via_unit = cost_unit;
  double fixed_rect_unit = 4 * non_prefer_wire_unit * cost_unit;
  double routed_rect_unit = 2 * via_unit;
  double violation_unit = 4 * non_prefer_wire_unit * cost_unit;
  /**
   * size, offset, schedule_interval, fixed_rect_unit, routed_rect_unit, violation_unit, max_routed_times
   */
  std::vector<VRIterParam> vr_iter_param_list;
  // clang-format off
  vr_iter_param_list.emplace_back(3, 0, 3, fixed_rect_unit, routed_rect_unit, violation_unit, 10);
  vr_iter_param_list.emplace_back(3, 1, 3, fixed_rect_unit, routed_rect_unit, violation_unit, 10);
  vr_iter_param_list.emplace_back(3, 2, 3, fixed_rect_unit, routed_rect_unit, violation_unit, 10);
  // clang-format on
  for (size_t i = 0, iter = 1; i < vr_iter_param_list.size(); i++, iter++) {
    Monitor iter_monitor;
    RTLOG.info(Loc::current(), "***** Begin iteration ", iter, "/", vr_iter_param_list.size(), "(", RTUTIL.getPercentage(iter, vr_iter_param_list.size()),
               ") *****");
    // debugPlotVRModel(vr_model, "before");
    setVRIterParam(vr_model, iter, vr_iter_param_list[i]);
    initVRBoxMap(vr_model);
    buildBoxSchedule(vr_model);
    splitNetResult(vr_model);
    // debugPlotVRModel(vr_model, "middle");
    routeVRBoxMap(vr_model);
    uploadNetResult(vr_model);
    uploadNetPatch(vr_model);
    uploadViolation(vr_model);
    // debugPlotVRModel(vr_model, "after");
    updateSummary(vr_model);
    printSummary(vr_model);
    outputNetCSV(vr_model);
    outputViolationCSV(vr_model);
    RTLOG.info(Loc::current(), "***** End Iteration ", iter, "/", vr_iter_param_list.size(), "(", RTUTIL.getPercentage(iter, vr_iter_param_list.size()), ")",
               iter_monitor.getStatsInfo(), "*****");
    if (stopIteration(vr_model)) {
      break;
    }
  }
}

void ViolationRepairer::setVRIterParam(VRModel& vr_model, int32_t iter, VRIterParam& vr_iter_param)
{
  vr_model.set_iter(iter);
  RTLOG.info(Loc::current(), "size: ", vr_iter_param.get_size());
  RTLOG.info(Loc::current(), "offset: ", vr_iter_param.get_offset());
  RTLOG.info(Loc::current(), "schedule_interval: ", vr_iter_param.get_schedule_interval());
  RTLOG.info(Loc::current(), "fixed_rect_unit: ", vr_iter_param.get_fixed_rect_unit());
  RTLOG.info(Loc::current(), "routed_rect_unit: ", vr_iter_param.get_routed_rect_unit());
  RTLOG.info(Loc::current(), "violation_unit: ", vr_iter_param.get_violation_unit());
  RTLOG.info(Loc::current(), "max_routed_times: ", vr_iter_param.get_max_routed_times());
  vr_model.set_vr_iter_param(vr_iter_param);
}

void ViolationRepairer::initVRBoxMap(VRModel& vr_model)
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

  VRIterParam& vr_iter_param = vr_model.get_vr_iter_param();
  int32_t size = vr_iter_param.get_size();
  int32_t offset = vr_iter_param.get_offset();
  int32_t x_box_num = static_cast<int32_t>(std::ceil((x_gcell_num - offset) / 1.0 / size));
  int32_t y_box_num = static_cast<int32_t>(std::ceil((y_gcell_num - offset) / 1.0 / size));

  GridMap<VRBox>& vr_box_map = vr_model.get_vr_box_map();
  vr_box_map.init(x_box_num, y_box_num);

  for (int32_t x = 0; x < vr_box_map.get_x_size(); x++) {
    for (int32_t y = 0; y < vr_box_map.get_y_size(); y++) {
      int32_t grid_ll_x = std::max(offset + x * size, 0);
      int32_t grid_ll_y = std::max(offset + y * size, 0);
      int32_t grid_ur_x = std::min(offset + (x + 1) * size - 1, x_gcell_num - 1);
      int32_t grid_ur_y = std::min(offset + (y + 1) * size - 1, y_gcell_num - 1);

      PlanarRect ll_gcell_rect = RTUTIL.getRealRectByGCell(PlanarCoord(grid_ll_x, grid_ll_y), gcell_axis);
      PlanarRect ur_gcell_rect = RTUTIL.getRealRectByGCell(PlanarCoord(grid_ur_x, grid_ur_y), gcell_axis);
      PlanarRect box_real_rect(ll_gcell_rect.get_ll(), ur_gcell_rect.get_ur());

      VRBox& vr_box = vr_box_map[x][y];

      EXTPlanarRect vr_box_rect;
      vr_box_rect.set_real_rect(box_real_rect);
      vr_box_rect.set_grid_rect(RTUTIL.getOpenGCellGridRect(box_real_rect, gcell_axis));
      vr_box.set_box_rect(vr_box_rect);
      VRBoxId vr_box_id;
      vr_box_id.set_x(x);
      vr_box_id.set_y(y);
      vr_box.set_vr_box_id(vr_box_id);
      vr_box.set_vr_iter_param(&vr_iter_param);
    }
  }
}

void ViolationRepairer::buildBoxSchedule(VRModel& vr_model)
{
  GridMap<VRBox>& vr_box_map = vr_model.get_vr_box_map();
  int32_t schedule_interval = vr_model.get_vr_iter_param().get_schedule_interval();

  std::vector<std::vector<VRBoxId>> vr_box_id_list_list;
  for (int32_t start_x = 0; start_x < schedule_interval; start_x++) {
    for (int32_t start_y = 0; start_y < schedule_interval; start_y++) {
      std::vector<VRBoxId> vr_box_id_list;
      for (int32_t x = start_x; x < vr_box_map.get_x_size(); x += schedule_interval) {
        for (int32_t y = start_y; y < vr_box_map.get_y_size(); y += schedule_interval) {
          vr_box_id_list.emplace_back(x, y);
        }
      }
      vr_box_id_list_list.push_back(vr_box_id_list);
    }
  }
  vr_model.set_vr_box_id_list_list(vr_box_id_list_list);
}

void ViolationRepairer::splitNetResult(VRModel& vr_model)
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

void ViolationRepairer::routeVRBoxMap(VRModel& vr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  GridMap<VRBox>& vr_box_map = vr_model.get_vr_box_map();

  size_t total_box_num = 0;
  for (std::vector<VRBoxId>& vr_box_id_list : vr_model.get_vr_box_id_list_list()) {
    total_box_num += vr_box_id_list.size();
  }

  size_t routed_box_num = 0;
  for (std::vector<VRBoxId>& vr_box_id_list : vr_model.get_vr_box_id_list_list()) {
    Monitor stage_monitor;
#pragma omp parallel for
    for (VRBoxId& vr_box_id : vr_box_id_list) {
      VRBox& vr_box = vr_box_map[vr_box_id.get_x()][vr_box_id.get_y()];
      buildFixedRect(vr_box);
      buildViolation(vr_box);
      initVRTaskList(vr_model, vr_box);
      buildNetResult(vr_box);
      buildNetPatch(vr_box);
      if (needRouting(vr_box)) {
        buildBoxTrackAxis(vr_box);
        buildLayerNodeMap(vr_box);
        buildObsTypeNetMap(vr_box);
        exemptPinShape(vr_box);
        // debugCheckVRBox(vr_box);
        // debugPlotVRBox(vr_box, -1, "before");
        routeVRBox(vr_box);
        // debugPlotVRBox(vr_box, -1, "after");
      }
      uploadNetResult(vr_box);
      uploadNetPatch(vr_box);
      uploadViolation(vr_box);
      freeVRBox(vr_box);
    }
    routed_box_num += vr_box_id_list.size();
    RTLOG.info(Loc::current(), "Routed ", routed_box_num, "/", total_box_num, "(", RTUTIL.getPercentage(routed_box_num, total_box_num), ") boxes with ",
               getViolationNum(vr_model), " violations", stage_monitor.getStatsInfo());
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void ViolationRepairer::buildFixedRect(VRBox& vr_box)
{
  vr_box.set_type_layer_net_fixed_rect_map(RTDM.getTypeLayerNetFixedRectMap(vr_box.get_box_rect()));
}

void ViolationRepairer::buildViolation(VRBox& vr_box)
{
  PlanarRect& box_real_rect = vr_box.get_box_rect().get_real_rect();

  for (Violation* violation : RTDM.getViolationSet(vr_box.get_box_rect())) {
    bool shape_in_box = false;
    if (RTUTIL.isInside(box_real_rect, violation->get_violation_shape().get_real_rect())) {
      shape_in_box = true;
    }
    if (shape_in_box) {
      vr_box.get_violation_list().push_back(*violation);
      RTDM.updateViolationToGCellMap(ChangeType::kDel, violation);
    }
  }
}

void ViolationRepairer::initVRTaskList(VRModel& vr_model, VRBox& vr_box)
{
  std::vector<VRNet>& vr_net_list = vr_model.get_vr_net_list();
  std::vector<VRTask*>& vr_task_list = vr_box.get_vr_task_list();

  std::set<int32_t> net_idx_set;
  for (Violation& violation : vr_box.get_violation_list()) {
    if (violation.get_violation_net_set().size() == 1) {
      net_idx_set.insert(*violation.get_violation_net_set().begin());
    }
  }
  for (int32_t net_idx : net_idx_set) {
    VRTask* vr_task = new VRTask();
    vr_task->set_net_idx(net_idx);
    vr_task->set_connect_type(vr_net_list[net_idx].get_connect_type());
    vr_task->set_routed_times(0);
    vr_task_list.push_back(vr_task);
  }
  std::sort(vr_task_list.begin(), vr_task_list.end(), CmpVRTask());
}

void ViolationRepairer::buildNetResult(VRBox& vr_box)
{
  PlanarRect& box_real_rect = vr_box.get_box_rect().get_real_rect();

  std::set<int32_t> net_idx_set;
  for (VRTask* vr_task : vr_box.get_vr_task_list()) {
    net_idx_set.insert(vr_task->get_net_idx());
  }
  for (auto& [net_idx, segment_set] : RTDM.getNetFinalResultMap(vr_box.get_box_rect())) {
    if (RTUTIL.exist(net_idx_set, net_idx)) {
      for (Segment<LayerCoord>* segment : segment_set) {
        bool least_one_coord_in_box = false;
        if (RTUTIL.isInside(box_real_rect, segment->get_first()) && RTUTIL.isInside(box_real_rect, segment->get_second())) {
          if (RTUTIL.isInside(box_real_rect, segment->get_first(), false) || RTUTIL.isInside(box_real_rect, segment->get_second(), false)) {
            // 线段在box_real_rect内,但不贴边的
            least_one_coord_in_box = true;
          }
        }
        if (least_one_coord_in_box) {
          vr_box.get_net_task_final_result_map()[net_idx].push_back(*segment);
          RTDM.updateNetFinalResultToGCellMap(ChangeType::kDel, net_idx, segment);
        } else {
          vr_box.get_net_final_result_map()[net_idx].insert(segment);
        }
      }
    } else {
      for (Segment<LayerCoord>* segment : segment_set) {
        vr_box.get_net_final_result_map()[net_idx].insert(segment);
      }
    }
  }
}

void ViolationRepairer::buildNetPatch(VRBox& vr_box)
{
  PlanarRect& box_real_rect = vr_box.get_box_rect().get_real_rect();

  std::set<int32_t> net_idx_set;
  for (VRTask* vr_task : vr_box.get_vr_task_list()) {
    net_idx_set.insert(vr_task->get_net_idx());
  }
  for (auto& [net_idx, patch_set] : RTDM.getNetFinalPatchMap(vr_box.get_box_rect())) {
    if (RTUTIL.exist(net_idx_set, net_idx)) {
      for (EXTLayerRect* patch : patch_set) {
        bool shape_in_box = false;
        if (RTUTIL.isInside(box_real_rect, patch->get_real_rect())) {
          shape_in_box = true;
        }
        if (shape_in_box) {
          vr_box.get_net_task_final_patch_map()[net_idx].push_back(*patch);
          RTDM.updateNetFinalPatchToGCellMap(ChangeType::kDel, net_idx, patch);
        } else {
          vr_box.get_net_final_patch_map()[net_idx].insert(patch);
        }
      }
    } else {
      for (EXTLayerRect* patch : patch_set) {
        vr_box.get_net_final_patch_map()[net_idx].insert(patch);
      }
    }
  }
}

bool ViolationRepairer::needRouting(VRBox& vr_box)
{
  if (vr_box.get_vr_task_list().empty()) {
    return false;
  }
  if (vr_box.get_violation_list().empty()) {
    return false;
  }
  return true;
}

void ViolationRepairer::buildBoxTrackAxis(VRBox& vr_box)
{
  int32_t manufacture_grid = RTDM.getDatabase().get_manufacture_grid();

  int32_t track_step_length = manufacture_grid * 10;

  PlanarRect& box_real_rect = vr_box.get_box_rect().get_real_rect();
  ScaleAxis& box_track_axis = vr_box.get_box_track_axis();

  int32_t ll_x = box_real_rect.get_ll_x();
  int32_t ll_y = box_real_rect.get_ll_y();
  int32_t ur_x = box_real_rect.get_ur_x();
  int32_t ur_y = box_real_rect.get_ur_y();
  // 对齐manufacture_grid
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
  ScaleGrid x_grid;
  x_grid.set_start_line(ll_x);
  x_grid.set_step_length(track_step_length);
  x_grid.set_step_num((ur_x - ll_x) / track_step_length);
  x_grid.set_end_line(ur_x);
  box_track_axis.get_x_grid_list().emplace_back(x_grid);
  ScaleGrid y_grid;
  y_grid.set_start_line(ll_y);
  y_grid.set_step_length(track_step_length);
  y_grid.set_step_num((ur_y - ll_y) / track_step_length);
  y_grid.set_end_line(ur_y);
  box_track_axis.get_y_grid_list().emplace_back(y_grid);
}

void ViolationRepairer::buildLayerNodeMap(VRBox& vr_box)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  PlanarCoord& real_ll = vr_box.get_box_rect().get_real_ll();
  PlanarCoord& real_ur = vr_box.get_box_rect().get_real_ur();
  ScaleAxis& box_track_axis = vr_box.get_box_track_axis();
  std::vector<int32_t> x_list = RTUTIL.getScaleList(real_ll.get_x(), real_ur.get_x(), box_track_axis.get_x_grid_list());
  std::vector<int32_t> y_list = RTUTIL.getScaleList(real_ll.get_y(), real_ur.get_y(), box_track_axis.get_y_grid_list());

  std::vector<GridMap<VRNode>>& layer_node_map = vr_box.get_layer_node_map();
  layer_node_map.resize(routing_layer_list.size());
  for (int32_t layer_idx = 0; layer_idx < static_cast<int32_t>(layer_node_map.size()); layer_idx++) {
    GridMap<VRNode>& vr_node_map = layer_node_map[layer_idx];
    vr_node_map.init(x_list.size(), y_list.size());
    for (size_t x = 0; x < x_list.size(); x++) {
      for (size_t y = 0; y < y_list.size(); y++) {
        VRNode& vr_node = vr_node_map[x][y];
        vr_node.set_x(x_list[x]);
        vr_node.set_y(y_list[y]);
        vr_node.set_layer_idx(layer_idx);
      }
    }
  }
}

void ViolationRepairer::buildObsTypeNetMap(VRBox& vr_box)
{
  for (auto& [is_routing, layer_net_fixed_rect_map] : vr_box.get_type_layer_net_fixed_rect_map()) {
    for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
      for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
        for (auto& fixed_rect : fixed_rect_set) {
          updateFixedRectToGraph(vr_box, ChangeType::kAdd, net_idx, fixed_rect, is_routing);
        }
      }
    }
  }
  for (auto& [net_idx, segment_set] : vr_box.get_net_final_result_map()) {
    for (Segment<LayerCoord>* segment : segment_set) {
      updateFixedRectToGraph(vr_box, ChangeType::kAdd, net_idx, *segment);
    }
  }
  for (auto& [net_idx, patch_set] : vr_box.get_net_final_patch_map()) {
    for (EXTLayerRect* patch : patch_set) {
      updateFixedRectToGraph(vr_box, ChangeType::kAdd, net_idx, patch, true);
    }
  }
  for (auto& [net_idx, segment_list] : vr_box.get_net_task_final_result_map()) {
    for (Segment<LayerCoord>& segment : segment_list) {
      updateRoutedRectToGraph(vr_box, ChangeType::kAdd, net_idx, segment);
    }
  }
  for (auto& [net_idx, patch_list] : vr_box.get_net_task_final_patch_map()) {
    for (EXTLayerRect& patch : patch_list) {
      updateRoutedRectToGraph(vr_box, ChangeType::kAdd, net_idx, patch);
    }
  }
  for (Violation& violation : vr_box.get_violation_list()) {
    addViolationToGraph(vr_box, violation);
  }
}

void ViolationRepairer::exemptPinShape(VRBox& vr_box)
{
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = RTDM.getDatabase().get_cut_to_adjacent_routing_map();

  std::map<int32_t, std::map<EXTLayerRect*, std::set<VRObsType>>> routing_layer_pin_shape_obs_type_map;
  for (auto& [is_routing, layer_net_fixed_rect_map] : vr_box.get_type_layer_net_fixed_rect_map()) {
    for (auto& [layer_idx, net_fixed_rect_map] : layer_net_fixed_rect_map) {
      std::map<int32_t, std::set<VRObsType>> routing_layer_obs_type_map;
      if (is_routing) {
        routing_layer_obs_type_map[layer_idx].insert(VRObsType::kPlanar);
      } else {
        if (cut_to_adjacent_routing_map[layer_idx].size() < 2) {
          continue;
        }
        routing_layer_obs_type_map[layer_idx].insert(VRObsType::kSpace);
      }
      for (auto& [net_idx, fixed_rect_set] : net_fixed_rect_map) {
        if (net_idx == -1) {
          continue;
        }
        for (auto& fixed_rect : fixed_rect_set) {
          for (auto& [routing_layer_idx, obs_type_set] : routing_layer_obs_type_map) {
            routing_layer_pin_shape_obs_type_map[routing_layer_idx][fixed_rect] = obs_type_set;
          }
        }
      }
    }
  }
  std::vector<GridMap<VRNode>>& layer_node_map = vr_box.get_layer_node_map();
  for (GridMap<VRNode>& vr_node_map : layer_node_map) {
    for (int32_t x = 0; x < vr_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < vr_node_map.get_y_size(); y++) {
        VRNode& vr_node = vr_node_map[x][y];
        for (auto& [pin_shape, obs_type_set] : routing_layer_pin_shape_obs_type_map[vr_node.get_layer_idx()]) {
          if (!RTUTIL.isInside(pin_shape->get_real_rect(), vr_node.get_planar_coord())) {
            continue;
          }
          for (auto& [obs_type, net_set] : vr_node.get_obs_type_fixed_rect_map()) {
            if (RTUTIL.exist(obs_type_set, obs_type)) {
              net_set.erase(-1);
            }
          }
        }
      }
    }
  }
}

void ViolationRepairer::routeVRBox(VRBox& vr_box)
{
  std::vector<VRTask*> routing_task_list = initTaskSchedule(vr_box);
  while (!routing_task_list.empty()) {
    for (VRTask* routing_task : routing_task_list) {
      routeVRTask(vr_box, routing_task);
      routing_task->addRoutedTimes();
    }
    updateViolationList(vr_box);
    updateTaskSchedule(vr_box, routing_task_list);
  }
}

std::vector<VRTask*> ViolationRepairer::initTaskSchedule(VRBox& vr_box)
{
  std::vector<VRTask*> routing_task_list;
  updateTaskSchedule(vr_box, routing_task_list);
  return routing_task_list;
}

void ViolationRepairer::routeVRTask(VRBox& vr_box, VRTask* vr_task)
{
  initSingleTask(vr_box, vr_task);
  {
    vr_box.set_routing_segment_list(vr_box.get_net_task_final_result_map()[vr_task->get_net_idx()]);
    vr_box.set_routing_patch_list(vr_box.get_net_task_final_patch_map()[vr_task->get_net_idx()]);
  }
  updateTaskResult(vr_box);
  updateTaskPatch(vr_box);
  resetSingleTask(vr_box);
}

void ViolationRepairer::initSingleTask(VRBox& vr_box, VRTask* vr_task)
{
  // single task
  vr_box.set_curr_vr_task(vr_task);
  vr_box.get_routing_segment_list().clear();
  vr_box.get_routing_patch_list().clear();
}

void ViolationRepairer::updateTaskResult(VRBox& vr_box)
{
  std::vector<Segment<LayerCoord>> new_routing_segment_list = vr_box.get_routing_segment_list();

  int32_t curr_net_idx = vr_box.get_curr_vr_task()->get_net_idx();
  std::vector<Segment<LayerCoord>>& routing_segment_list = vr_box.get_net_task_final_result_map()[curr_net_idx];

  // 原结果从graph删除,由于task有对应net_idx,所以不需要在布线前进行删除也不会影响结果
  for (Segment<LayerCoord>& routing_segment : routing_segment_list) {
    updateRoutedRectToGraph(vr_box, ChangeType::kDel, curr_net_idx, routing_segment);
  }
  routing_segment_list = new_routing_segment_list;
  // 新结果添加到graph
  for (Segment<LayerCoord>& routing_segment : routing_segment_list) {
    updateRoutedRectToGraph(vr_box, ChangeType::kAdd, curr_net_idx, routing_segment);
  }
}

void ViolationRepairer::updateTaskPatch(VRBox& vr_box)
{
  std::vector<EXTLayerRect> new_routing_patch_list = vr_box.get_routing_patch_list();

  int32_t curr_net_idx = vr_box.get_curr_vr_task()->get_net_idx();
  std::vector<EXTLayerRect>& routing_patch_list = vr_box.get_net_task_final_patch_map()[curr_net_idx];

  // 原结果从graph删除,由于task有对应net_idx,所以不需要在布线前进行删除也不会影响结果
  for (EXTLayerRect& routing_patch : routing_patch_list) {
    updateRoutedRectToGraph(vr_box, ChangeType::kDel, curr_net_idx, routing_patch);
  }
  routing_patch_list = new_routing_patch_list;
  // 新结果添加到graph
  for (EXTLayerRect& routing_patch : routing_patch_list) {
    updateRoutedRectToGraph(vr_box, ChangeType::kAdd, curr_net_idx, routing_patch);
  }
}

void ViolationRepairer::resetSingleTask(VRBox& vr_box)
{
  vr_box.set_curr_vr_task(nullptr);
  vr_box.get_routing_segment_list().clear();
  vr_box.get_routing_patch_list().clear();
}

void ViolationRepairer::updateViolationList(VRBox& vr_box)
{
  vr_box.get_violation_list().clear();
  for (Violation new_violation : getMultiNetViolationList(vr_box)) {
    vr_box.get_violation_list().push_back(new_violation);
  }
  for (Violation new_violation : getSingleNetViolationList(vr_box)) {
    vr_box.get_violation_list().push_back(new_violation);
  }
  // 新结果添加到graph
  for (Violation& violation : vr_box.get_violation_list()) {
    addViolationToGraph(vr_box, violation);
  }
}

std::vector<Violation> ViolationRepairer::getMultiNetViolationList(VRBox& vr_box)
{
  std::string top_name = RTUTIL.getString("vr_box_", vr_box.get_vr_box_id().get_x(), "_", vr_box.get_vr_box_id().get_y());
  std::vector<std::pair<EXTLayerRect*, bool>> env_shape_list;
  std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>> net_pin_shape_map;
  for (auto& [is_routing, layer_net_fixed_rect_map] : vr_box.get_type_layer_net_fixed_rect_map()) {
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
  for (auto& [net_idx, segment_set] : vr_box.get_net_final_result_map()) {
    for (Segment<LayerCoord>* segment : segment_set) {
      net_result_map[net_idx].push_back(segment);
    }
  }
  for (auto& [net_idx, segment_list] : vr_box.get_net_task_final_result_map()) {
    for (Segment<LayerCoord>& segment : segment_list) {
      net_result_map[net_idx].emplace_back(&segment);
    }
  }
  std::map<int32_t, std::vector<EXTLayerRect*>> net_patch_map;
  for (auto& [net_idx, patch_set] : vr_box.get_net_final_patch_map()) {
    for (EXTLayerRect* patch : patch_set) {
      net_patch_map[net_idx].push_back(patch);
    }
  }
  for (auto& [net_idx, patch_list] : vr_box.get_net_task_final_patch_map()) {
    for (EXTLayerRect& patch : patch_list) {
      net_patch_map[net_idx].emplace_back(&patch);
    }
  }
  std::set<int32_t> need_checked_net_set;
  for (VRTask* vr_task : vr_box.get_vr_task_list()) {
    need_checked_net_set.insert(vr_task->get_net_idx());
  }

  DETask de_task;
  de_task.set_proc_type(DEProcType::kGet);
  de_task.set_net_type(DENetType::kMultiNet);
  de_task.set_top_name(top_name);
  de_task.set_env_shape_list(env_shape_list);
  de_task.set_net_pin_shape_map(net_pin_shape_map);
  de_task.set_net_result_map(net_result_map);
  de_task.set_net_patch_map(net_patch_map);
  de_task.set_need_checked_net_set(need_checked_net_set);
  // return RTDE.getViolationListByTemp(de_task);
  return RTDE.getViolationList(de_task);
}

std::vector<Violation> ViolationRepairer::getSingleNetViolationList(VRBox& vr_box)
{
  std::string top_name = RTUTIL.getString("vr_box_", vr_box.get_vr_box_id().get_x(), "_", vr_box.get_vr_box_id().get_y());
  std::vector<std::pair<EXTLayerRect*, bool>> env_shape_list;
  std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>> net_pin_shape_map;
  for (auto& [is_routing, layer_net_fixed_rect_map] : vr_box.get_type_layer_net_fixed_rect_map()) {
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
  for (auto& [net_idx, segment_set] : vr_box.get_net_final_result_map()) {
    for (Segment<LayerCoord>* segment : segment_set) {
      net_result_map[net_idx].push_back(segment);
    }
  }
  for (auto& [net_idx, segment_list] : vr_box.get_net_task_final_result_map()) {
    for (Segment<LayerCoord>& segment : segment_list) {
      net_result_map[net_idx].emplace_back(&segment);
    }
  }
  std::map<int32_t, std::vector<EXTLayerRect*>> net_patch_map;
  for (auto& [net_idx, patch_set] : vr_box.get_net_final_patch_map()) {
    for (EXTLayerRect* patch : patch_set) {
      net_patch_map[net_idx].push_back(patch);
    }
  }
  for (auto& [net_idx, patch_list] : vr_box.get_net_task_final_patch_map()) {
    for (EXTLayerRect& patch : patch_list) {
      net_patch_map[net_idx].emplace_back(&patch);
    }
  }
  std::set<int32_t> need_checked_net_set;
  for (VRTask* vr_task : vr_box.get_vr_task_list()) {
    need_checked_net_set.insert(vr_task->get_net_idx());
  }

  DETask de_task;
  de_task.set_proc_type(DEProcType::kGet);
  de_task.set_net_type(DENetType::kSingleNet);
  de_task.set_top_name(top_name);
  de_task.set_env_shape_list(env_shape_list);
  de_task.set_net_pin_shape_map(net_pin_shape_map);
  de_task.set_net_result_map(net_result_map);
  de_task.set_net_patch_map(net_patch_map);
  de_task.set_need_checked_net_set(need_checked_net_set);
  // return RTDE.getViolationListByTemp(de_task);
  return RTDE.getViolationList(de_task);
}

void ViolationRepairer::updateTaskSchedule(VRBox& vr_box, std::vector<VRTask*>& routing_task_list)
{
  int32_t max_routed_times = vr_box.get_vr_iter_param()->get_max_routed_times();

  std::set<VRTask*> visited_routing_task_set;
  std::vector<VRTask*> new_routing_task_list;
  for (Violation& violation : vr_box.get_violation_list()) {
    for (VRTask* vr_task : vr_box.get_vr_task_list()) {
      if (!RTUTIL.exist(violation.get_violation_net_set(), vr_task->get_net_idx())) {
        continue;
      }
      if (vr_task->get_routed_times() < max_routed_times && !RTUTIL.exist(visited_routing_task_set, vr_task)) {
        visited_routing_task_set.insert(vr_task);
        new_routing_task_list.push_back(vr_task);
      }
      break;
    }
  }
  routing_task_list = new_routing_task_list;

  std::vector<VRTask*> new_vr_task_list;
  for (VRTask* vr_task : vr_box.get_vr_task_list()) {
    if (!RTUTIL.exist(visited_routing_task_set, vr_task)) {
      new_vr_task_list.push_back(vr_task);
    }
  }
  for (VRTask* routing_task : routing_task_list) {
    new_vr_task_list.push_back(routing_task);
  }
  vr_box.set_vr_task_list(new_vr_task_list);
}

void ViolationRepairer::uploadNetResult(VRBox& vr_box)
{
  for (auto& [net_idx, segment_list] : vr_box.get_net_task_final_result_map()) {
    for (Segment<LayerCoord>& segment : segment_list) {
      RTDM.updateNetFinalResultToGCellMap(ChangeType::kAdd, net_idx, new Segment<LayerCoord>(segment));
    }
  }
}

void ViolationRepairer::uploadNetPatch(VRBox& vr_box)
{
  for (auto& [net_idx, patch_list] : vr_box.get_net_task_final_patch_map()) {
    for (EXTLayerRect& patch : patch_list) {
      RTDM.updateNetFinalPatchToGCellMap(ChangeType::kAdd, net_idx, new EXTLayerRect(patch));
    }
  }
}

void ViolationRepairer::uploadViolation(VRBox& vr_box)
{
  for (Violation& violation : vr_box.get_violation_list()) {
    RTDM.updateViolationToGCellMap(ChangeType::kAdd, new Violation(violation));
  }
}

void ViolationRepairer::freeVRBox(VRBox& vr_box)
{
  for (VRTask* vr_task : vr_box.get_vr_task_list()) {
    delete vr_task;
    vr_task = nullptr;
  }
  vr_box.get_vr_task_list().clear();
  vr_box.get_layer_node_map().clear();
}

int32_t ViolationRepairer::getViolationNum(VRModel& vr_model)
{
  Die& die = RTDM.getDatabase().get_die();

  return static_cast<int32_t>(RTDM.getViolationSet(die).size());
}

void ViolationRepairer::uploadNetResult(VRModel& vr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();
  std::vector<VRNet>& vr_net_list = vr_model.get_vr_net_list();

  std::map<int32_t, std::set<Segment<LayerCoord>*>> net_final_result_map = RTDM.getNetFinalResultMap(die);
  for (auto& [net_idx, segment_set] : net_final_result_map) {
    std::vector<Segment<LayerCoord>> routing_segment_list;
    for (Segment<LayerCoord>* segment : segment_set) {
      routing_segment_list.emplace_back(segment->get_first(), segment->get_second());
    }
    std::vector<LayerCoord> candidate_root_coord_list;
    std::map<LayerCoord, std::set<int32_t>, CmpLayerCoordByXASC> key_coord_pin_map;
    std::vector<VRPin>& vr_pin_list = vr_net_list[net_idx].get_vr_pin_list();
    for (size_t i = 0; i < vr_pin_list.size(); i++) {
      LayerCoord coord = vr_pin_list[i].get_access_point().getRealLayerCoord();
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

void ViolationRepairer::uploadNetPatch(VRModel& vr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();

  std::map<int32_t, std::set<EXTLayerRect*>> net_final_patch_map = RTDM.getNetFinalPatchMap(die);
  std::map<int32_t, std::set<Segment<LayerCoord>*>> net_final_result_map = RTDM.getNetFinalResultMap(die);
  for (auto& [net_idx, patch_set] : net_final_patch_map) {
    std::map<int32_t, std::vector<PlanarRect>> layer_rect_map;
    for (Segment<LayerCoord>* segment : net_final_result_map[net_idx]) {
      for (NetShape& net_shape : RTDM.getNetShapeList(net_idx, *segment)) {
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
      RTDM.updateNetFinalPatchToGCellMap(ChangeType::kDel, net_idx, del_patch);
    }
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

bool ViolationRepairer::stopIteration(VRModel& vr_model)
{
  if (getViolationNum(vr_model) == 0) {
    RTLOG.info(Loc::current(), "***** Iteration stopped early *****");
    return true;
  }
  return false;
}

#if 1  // update env

void ViolationRepairer::updateFixedRectToGraph(VRBox& vr_box, ChangeType change_type, int32_t net_idx, EXTLayerRect* fixed_rect, bool is_routing)
{
  NetShape net_shape(net_idx, fixed_rect->getRealLayerRect(), is_routing);
  for (auto& [vr_node, obs_type_set] : getNodeObsTypeMap(vr_box, net_shape, true)) {
    for (VRObsType obs_type : obs_type_set) {
      if (change_type == ChangeType::kAdd) {
        vr_node->get_obs_type_fixed_rect_map()[obs_type].insert(net_shape.get_net_idx());
      } else if (change_type == ChangeType::kDel) {
        vr_node->get_obs_type_fixed_rect_map()[obs_type].erase(net_shape.get_net_idx());
      }
    }
  }
}

void ViolationRepairer::updateFixedRectToGraph(VRBox& vr_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment)
{
  for (NetShape& net_shape : RTDM.getNetShapeList(net_idx, segment)) {
    for (auto& [vr_node, obs_type_set] : getNodeObsTypeMap(vr_box, net_shape, true)) {
      for (VRObsType obs_type : obs_type_set) {
        if (change_type == ChangeType::kAdd) {
          vr_node->get_obs_type_fixed_rect_map()[obs_type].insert(net_shape.get_net_idx());
        } else if (change_type == ChangeType::kDel) {
          vr_node->get_obs_type_fixed_rect_map()[obs_type].erase(net_shape.get_net_idx());
        }
      }
    }
  }
}

void ViolationRepairer::updateRoutedRectToGraph(VRBox& vr_box, ChangeType change_type, int32_t net_idx, Segment<LayerCoord>& segment)
{
  for (NetShape& net_shape : RTDM.getNetShapeList(net_idx, segment)) {
    for (auto& [vr_node, obs_type_set] : getNodeObsTypeMap(vr_box, net_shape, true)) {
      for (VRObsType obs_type : obs_type_set) {
        if (change_type == ChangeType::kAdd) {
          vr_node->get_obs_type_routed_rect_map()[obs_type].insert(net_shape.get_net_idx());
        } else if (change_type == ChangeType::kDel) {
          vr_node->get_obs_type_routed_rect_map()[obs_type].erase(net_shape.get_net_idx());
        }
      }
    }
  }
}

void ViolationRepairer::updateRoutedRectToGraph(VRBox& vr_box, ChangeType change_type, int32_t net_idx, EXTLayerRect& patch)
{
  NetShape net_shape(net_idx, patch.getRealLayerRect(), true);
  for (auto& [vr_node, obs_type_set] : getNodeObsTypeMap(vr_box, net_shape, true)) {
    for (VRObsType obs_type : obs_type_set) {
      if (change_type == ChangeType::kAdd) {
        vr_node->get_obs_type_routed_rect_map()[obs_type].insert(net_shape.get_net_idx());
      } else if (change_type == ChangeType::kDel) {
        vr_node->get_obs_type_routed_rect_map()[obs_type].erase(net_shape.get_net_idx());
      }
    }
  }
}

void ViolationRepairer::addViolationToGraph(VRBox& vr_box, Violation& violation)
{
  NetShape net_shape(-1, violation.get_violation_shape().getRealLayerRect(), violation.get_is_routing());
  for (auto& [vr_node, obs_type_set] : getNodeObsTypeMap(vr_box, net_shape, true)) {
    for (VRObsType obs_type : obs_type_set) {
      vr_node->get_obs_type_violation_number_map()[obs_type]++;
    }
  }
}

std::map<VRNode*, std::set<VRObsType>> ViolationRepairer::getNodeObsTypeMap(VRBox& vr_box, NetShape& net_shape, bool need_enlarged)
{
  std::map<VRNode*, std::set<VRObsType>> node_obs_type_map;
  if (net_shape.get_is_routing()) {
    node_obs_type_map = getRoutingNodeObsTypeMap(vr_box, net_shape, need_enlarged);
  } else {
    node_obs_type_map = getCutNodeObsTypeMap(vr_box, net_shape, need_enlarged);
  }
  return node_obs_type_map;
}

std::map<VRNode*, std::set<VRObsType>> ViolationRepairer::getRoutingNodeObsTypeMap(VRBox& vr_box, NetShape& net_shape, bool need_enlarged)
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

  GridMap<VRNode>& vr_node_map = vr_box.get_layer_node_map()[layer_idx];
  std::map<VRNode*, std::set<VRObsType>> node_obs_type_map;
  // wire 与 net_shape
  for (auto& [x_spacing, y_spacing] : spacing_pair_list) {
    int32_t enlarged_x_size = half_wire_width;
    int32_t enlarged_y_size = half_wire_width;
    if (need_enlarged) {
      // 膨胀size为 half_wire_width + spacing
      enlarged_x_size += x_spacing;
      enlarged_y_size += y_spacing;
    }
    // 贴合的也不算违例
    enlarged_x_size -= 1;
    enlarged_y_size -= 1;
    PlanarRect planar_enlarged_rect = RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size);
    if (!RTUTIL.existTrackGrid(planar_enlarged_rect, vr_box.get_box_track_axis())) {
      continue;
    }
    PlanarRect grid_rect = RTUTIL.getTrackGrid(planar_enlarged_rect, vr_box.get_box_track_axis());
    for (int32_t x = grid_rect.get_ll_x(); x <= grid_rect.get_ur_x(); x++) {
      for (int32_t y = grid_rect.get_ll_y(); y <= grid_rect.get_ur_y(); y++) {
        VRNode& node = vr_node_map[x][y];
        node_obs_type_map[&node].insert(VRObsType::kPlanar);
      }
    }
  }
  // enclosure 与 net_shape
  for (auto& [x_spacing, y_spacing] : spacing_pair_list) {
    int32_t enlarged_x_size = enclosure_half_x_span;
    int32_t enlarged_y_size = enclosure_half_y_span;
    if (need_enlarged) {
      // 膨胀size为 enclosure_half_span + spacing
      enlarged_x_size += x_spacing;
      enlarged_y_size += y_spacing;
    }
    // 贴合的也不算违例
    enlarged_x_size -= 1;
    enlarged_y_size -= 1;
    PlanarRect space_enlarged_rect = RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size);
    if (!RTUTIL.existTrackGrid(space_enlarged_rect, vr_box.get_box_track_axis())) {
      continue;
    }
    PlanarRect grid_rect = RTUTIL.getTrackGrid(space_enlarged_rect, vr_box.get_box_track_axis());
    for (int32_t x = grid_rect.get_ll_x(); x <= grid_rect.get_ur_x(); x++) {
      for (int32_t y = grid_rect.get_ll_y(); y <= grid_rect.get_ur_y(); y++) {
        VRNode& node = vr_node_map[x][y];
        node_obs_type_map[&node].insert(VRObsType::kSpace);
      }
    }
  }
  return node_obs_type_map;
}

std::map<VRNode*, std::set<VRObsType>> ViolationRepairer::getCutNodeObsTypeMap(VRBox& vr_box, NetShape& net_shape, bool need_enlarged)
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
  std::map<VRNode*, std::set<VRObsType>> node_obs_type_map;
  for (auto& [cut_layer_idx, spacing_pair_list] : cut_spacing_map) {
    std::vector<int32_t> adjacent_routing_layer_idx_list = cut_to_adjacent_routing_map[cut_layer_idx];
    int32_t below_routing_layer_idx = adjacent_routing_layer_idx_list.front();
    int32_t above_routing_layer_idx = adjacent_routing_layer_idx_list.back();
    RTUTIL.swapByASC(below_routing_layer_idx, above_routing_layer_idx);
    PlanarRect& cut_shape = layer_via_master_list[below_routing_layer_idx].front().get_cut_shape_list().front();
    int32_t cut_shape_half_x_span = cut_shape.getXSpan() / 2;
    int32_t cut_shape_half_y_span = cut_shape.getYSpan() / 2;
    std::vector<GridMap<VRNode>>& layer_node_map = vr_box.get_layer_node_map();
    for (auto& [x_spacing, y_spacing] : spacing_pair_list) {
      int32_t enlarged_x_size = cut_shape_half_x_span;
      int32_t enlarged_y_size = cut_shape_half_y_span;
      if (need_enlarged) {
        // 膨胀size为 cut_shape_half_span + spacing
        enlarged_x_size += x_spacing;
        enlarged_y_size += y_spacing;
      }
      // 贴合的也不算违例
      enlarged_x_size -= 1;
      enlarged_y_size -= 1;
      PlanarRect space_enlarged_rect = RTUTIL.getEnlargedRect(net_shape.get_rect(), enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size);
      if (!RTUTIL.existTrackGrid(space_enlarged_rect, vr_box.get_box_track_axis())) {
        continue;
      }
      PlanarRect grid_rect = RTUTIL.getTrackGrid(space_enlarged_rect, vr_box.get_box_track_axis());
      for (int32_t x = grid_rect.get_ll_x(); x <= grid_rect.get_ur_x(); x++) {
        for (int32_t y = grid_rect.get_ll_y(); y <= grid_rect.get_ur_y(); y++) {
          VRNode& below_node = layer_node_map[below_routing_layer_idx][x][y];
          VRNode& above_node = layer_node_map[above_routing_layer_idx][x][y];
          node_obs_type_map[&below_node].insert(VRObsType::kSpace);
          node_obs_type_map[&above_node].insert(VRObsType::kSpace);
        }
      }
    }
  }
  return node_obs_type_map;
}

#endif

#if 1  // exhibit

void ViolationRepairer::updateSummary(VRModel& vr_model)
{
  int32_t micron_dbu = RTDM.getDatabase().get_micron_dbu();
  Die& die = RTDM.getDatabase().get_die();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();
  Summary& summary = RTDM.getDatabase().get_summary();
  int32_t enable_timing = RTDM.getConfig().enable_timing;

  std::map<int32_t, double>& routing_wire_length_map = summary.iter_vr_summary_map[vr_model.get_iter()].routing_wire_length_map;
  double& total_wire_length = summary.iter_vr_summary_map[vr_model.get_iter()].total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.iter_vr_summary_map[vr_model.get_iter()].cut_via_num_map;
  int32_t& total_via_num = summary.iter_vr_summary_map[vr_model.get_iter()].total_via_num;
  std::map<int32_t, int32_t>& routing_patch_num_map = summary.iter_vr_summary_map[vr_model.get_iter()].routing_patch_num_map;
  int32_t& total_patch_num = summary.iter_vr_summary_map[vr_model.get_iter()].total_patch_num;
  std::map<int32_t, std::map<std::string, int32_t>>& single_net_routing_violation_type_num_map
      = summary.iter_vr_summary_map[vr_model.get_iter()].single_net_routing_violation_type_num_map;
  std::map<std::string, int32_t>& single_net_violation_type_num_map = summary.iter_vr_summary_map[vr_model.get_iter()].single_net_violation_type_num_map;
  std::map<int32_t, int32_t>& single_net_routing_violation_num_map = summary.iter_vr_summary_map[vr_model.get_iter()].single_net_routing_violation_num_map;
  int32_t& single_net_total_violation_num = summary.iter_vr_summary_map[vr_model.get_iter()].single_net_total_violation_num;
  std::map<int32_t, std::map<std::string, int32_t>>& multi_net_routing_violation_type_num_map
      = summary.iter_vr_summary_map[vr_model.get_iter()].multi_net_routing_violation_type_num_map;
  std::map<std::string, int32_t>& multi_net_violation_type_num_map = summary.iter_vr_summary_map[vr_model.get_iter()].multi_net_violation_type_num_map;
  std::map<int32_t, int32_t>& multi_net_routing_violation_num_map = summary.iter_vr_summary_map[vr_model.get_iter()].multi_net_routing_violation_num_map;
  int32_t& multi_net_total_violation_num = summary.iter_vr_summary_map[vr_model.get_iter()].multi_net_total_violation_num;
  std::map<std::string, std::map<std::string, double>>& clock_timing = summary.iter_vr_summary_map[vr_model.get_iter()].clock_timing;
  std::map<std::string, double>& power_map = summary.iter_vr_summary_map[vr_model.get_iter()].power_map;

  std::vector<VRNet>& vr_net_list = vr_model.get_vr_net_list();

  routing_wire_length_map.clear();
  total_wire_length = 0;
  cut_via_num_map.clear();
  total_via_num = 0;
  routing_patch_num_map.clear();
  total_patch_num = 0;
  single_net_routing_violation_type_num_map.clear();
  single_net_violation_type_num_map.clear();
  single_net_routing_violation_num_map.clear();
  single_net_total_violation_num = 0;
  multi_net_routing_violation_type_num_map.clear();
  multi_net_violation_type_num_map.clear();
  multi_net_routing_violation_num_map.clear();
  multi_net_total_violation_num = 0;
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
  for (auto& [net_idx, patch_set] : RTDM.getNetFinalPatchMap(die)) {
    for (EXTLayerRect* patch : patch_set) {
      routing_patch_num_map[patch->get_layer_idx()]++;
      total_patch_num++;
    }
  }
  for (Violation* violation : RTDM.getViolationSet(die)) {
    if (violation->get_violation_net_set().size() >= 2) {
      continue;
    }
    single_net_routing_violation_type_num_map[violation->get_violation_shape().get_layer_idx()][GetViolationTypeName()(violation->get_violation_type())]++;
    single_net_violation_type_num_map[GetViolationTypeName()(violation->get_violation_type())]++;
    single_net_routing_violation_num_map[violation->get_violation_shape().get_layer_idx()]++;
    single_net_total_violation_num++;
  }
  for (Violation* violation : RTDM.getViolationSet(die)) {
    if (violation->get_violation_net_set().size() < 2) {
      continue;
    }
    multi_net_routing_violation_type_num_map[violation->get_violation_shape().get_layer_idx()][GetViolationTypeName()(violation->get_violation_type())]++;
    multi_net_violation_type_num_map[GetViolationTypeName()(violation->get_violation_type())]++;
    multi_net_routing_violation_num_map[violation->get_violation_shape().get_layer_idx()]++;
    multi_net_total_violation_num++;
  }
  if (enable_timing) {
    std::vector<std::map<std::string, std::vector<LayerCoord>>> real_pin_coord_map_list;
    real_pin_coord_map_list.resize(vr_net_list.size());
    std::vector<std::vector<Segment<LayerCoord>>> routing_segment_list_list;
    routing_segment_list_list.resize(vr_net_list.size());
    for (VRNet& vr_net : vr_net_list) {
      for (VRPin& vr_pin : vr_net.get_vr_pin_list()) {
        real_pin_coord_map_list[vr_net.get_net_idx()][vr_pin.get_pin_name()].push_back(vr_pin.get_access_point().getRealLayerCoord());
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

void ViolationRepairer::printSummary(VRModel& vr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  Summary& summary = RTDM.getDatabase().get_summary();
  int32_t enable_timing = RTDM.getConfig().enable_timing;

  std::map<int32_t, double>& routing_wire_length_map = summary.iter_vr_summary_map[vr_model.get_iter()].routing_wire_length_map;
  double& total_wire_length = summary.iter_vr_summary_map[vr_model.get_iter()].total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.iter_vr_summary_map[vr_model.get_iter()].cut_via_num_map;
  int32_t& total_via_num = summary.iter_vr_summary_map[vr_model.get_iter()].total_via_num;
  std::map<int32_t, int32_t>& routing_patch_num_map = summary.iter_vr_summary_map[vr_model.get_iter()].routing_patch_num_map;
  int32_t& total_patch_num = summary.iter_vr_summary_map[vr_model.get_iter()].total_patch_num;
  std::map<int32_t, std::map<std::string, int32_t>>& single_net_routing_violation_type_num_map
      = summary.iter_vr_summary_map[vr_model.get_iter()].single_net_routing_violation_type_num_map;
  std::map<std::string, int32_t>& single_net_violation_type_num_map = summary.iter_vr_summary_map[vr_model.get_iter()].single_net_violation_type_num_map;
  std::map<int32_t, int32_t>& single_net_routing_violation_num_map = summary.iter_vr_summary_map[vr_model.get_iter()].single_net_routing_violation_num_map;
  int32_t& single_net_total_violation_num = summary.iter_vr_summary_map[vr_model.get_iter()].single_net_total_violation_num;
  std::map<int32_t, std::map<std::string, int32_t>>& multi_net_routing_violation_type_num_map
      = summary.iter_vr_summary_map[vr_model.get_iter()].multi_net_routing_violation_type_num_map;
  std::map<std::string, int32_t>& multi_net_violation_type_num_map = summary.iter_vr_summary_map[vr_model.get_iter()].multi_net_violation_type_num_map;
  std::map<int32_t, int32_t>& multi_net_routing_violation_num_map = summary.iter_vr_summary_map[vr_model.get_iter()].multi_net_routing_violation_num_map;
  int32_t& multi_net_total_violation_num = summary.iter_vr_summary_map[vr_model.get_iter()].multi_net_total_violation_num;
  std::map<std::string, std::map<std::string, double>>& clock_timing = summary.iter_vr_summary_map[vr_model.get_iter()].clock_timing;
  std::map<std::string, double>& power_map = summary.iter_vr_summary_map[vr_model.get_iter()].power_map;

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
  fort::char_table single_net_routing_violation_map_table;
  {
    single_net_routing_violation_map_table.set_cell_text_align(fort::text_align::right);
    single_net_routing_violation_map_table << fort::header << "single_net";
    for (size_t i = 0; i < single_net_violation_type_num_map.size(); ++i) {
      single_net_routing_violation_map_table << fort::header << " ";
    }
    single_net_routing_violation_map_table << fort::header << " " << fort::endr;
    single_net_routing_violation_map_table << fort::header << "routing";
    for (auto& [type, num] : single_net_violation_type_num_map) {
      single_net_routing_violation_map_table << fort::header << type;
    }
    single_net_routing_violation_map_table << fort::header << "Total" << fort::endr;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      single_net_routing_violation_map_table << routing_layer.get_layer_name();
      for (auto& [type, num] : single_net_violation_type_num_map) {
        single_net_routing_violation_map_table << single_net_routing_violation_type_num_map[routing_layer.get_layer_idx()][type];
      }
      single_net_routing_violation_map_table << single_net_routing_violation_num_map[routing_layer.get_layer_idx()] << fort::endr;
    }
    single_net_routing_violation_map_table << fort::header << "Total";
    for (auto& [type, num] : single_net_violation_type_num_map) {
      single_net_routing_violation_map_table << fort::header << num;
    }
    single_net_routing_violation_map_table << fort::header << single_net_total_violation_num << fort::endr;
  }
  fort::char_table multi_net_routing_violation_map_table;
  {
    multi_net_routing_violation_map_table.set_cell_text_align(fort::text_align::right);
    multi_net_routing_violation_map_table << fort::header << "multi_net";
    for (size_t i = 0; i < multi_net_violation_type_num_map.size(); ++i) {
      multi_net_routing_violation_map_table << fort::header << " ";
    }
    multi_net_routing_violation_map_table << fort::header << " " << fort::endr;
    multi_net_routing_violation_map_table << fort::header << "routing";
    for (auto& [type, num] : multi_net_violation_type_num_map) {
      multi_net_routing_violation_map_table << fort::header << type;
    }
    multi_net_routing_violation_map_table << fort::header << "Total" << fort::endr;
    for (RoutingLayer& routing_layer : routing_layer_list) {
      multi_net_routing_violation_map_table << routing_layer.get_layer_name();
      for (auto& [type, num] : multi_net_violation_type_num_map) {
        multi_net_routing_violation_map_table << multi_net_routing_violation_type_num_map[routing_layer.get_layer_idx()][type];
      }
      multi_net_routing_violation_map_table << multi_net_routing_violation_num_map[routing_layer.get_layer_idx()] << fort::endr;
    }
    multi_net_routing_violation_map_table << fort::header << "Total";
    for (auto& [type, num] : multi_net_violation_type_num_map) {
      multi_net_routing_violation_map_table << fort::header << num;
    }
    multi_net_routing_violation_map_table << fort::header << multi_net_total_violation_num << fort::endr;
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
  RTUTIL.printTableList({routing_wire_length_map_table, cut_via_num_map_table, routing_patch_num_map_table});
  RTUTIL.printTableList({single_net_routing_violation_map_table, multi_net_routing_violation_map_table});
  RTUTIL.printTableList({timing_table, power_table});
}

void ViolationRepairer::outputNetCSV(VRModel& vr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::string& vr_temp_directory_path = RTDM.getConfig().vr_temp_directory_path;
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
      for (auto& [net_idx, patch_set] : gcell_map[x][y].get_net_final_patch_map()) {
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
        = RTUTIL.getOutputFileStream(RTUTIL.getString(vr_temp_directory_path, "net_map_", routing_layer.get_layer_name(), "_", vr_model.get_iter(), ".csv"));
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

void ViolationRepairer::outputViolationCSV(VRModel& vr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  GridMap<GCell>& gcell_map = RTDM.getDatabase().get_gcell_map();
  std::string& vr_temp_directory_path = RTDM.getConfig().vr_temp_directory_path;
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
        RTUTIL.getString(vr_temp_directory_path, "violation_map_", routing_layer.get_layer_name(), "_", vr_model.get_iter(), ".csv"));
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

void ViolationRepairer::debugPlotVRModel(VRModel& vr_model, std::string flag)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::string& vr_temp_directory_path = RTDM.getConfig().vr_temp_directory_path;

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

  // routing patch
  for (auto& [net_idx, patch_set] : RTDM.getNetFinalPatchMap(die)) {
    GPStruct final_patch_struct(RTUTIL.getString("final_patch(net_", net_idx, ")"));
    for (EXTLayerRect* patch : patch_set) {
      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShape));
      gp_boundary.set_rect(patch->get_real_rect());
      gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(patch->get_layer_idx()));
      final_patch_struct.push(gp_boundary);
    }
    gp_gds.addStruct(final_patch_struct);
  }

  // violation
  {
    for (Violation* violation : RTDM.getViolationSet(die)) {
      if (violation->get_violation_net_set().size() >= 2) {
        continue;
      }
      GPStruct single_net_violation_struct(RTUTIL.getString("single_net_violation_", GetViolationTypeName()(violation->get_violation_type())));
      EXTLayerRect& violation_shape = violation->get_violation_shape();

      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kViolation));
      gp_boundary.set_rect(violation_shape.get_real_rect());
      if (violation->get_is_routing()) {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(violation_shape.get_layer_idx()));
      } else {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(violation_shape.get_layer_idx()));
      }
      single_net_violation_struct.push(gp_boundary);
      gp_gds.addStruct(single_net_violation_struct);
    }
    for (Violation* violation : RTDM.getViolationSet(die)) {
      if (violation->get_violation_net_set().size() < 2) {
        continue;
      }
      GPStruct multi_net_violation_struct(RTUTIL.getString("multi_net_violation_", GetViolationTypeName()(violation->get_violation_type())));
      EXTLayerRect& violation_shape = violation->get_violation_shape();

      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kViolation));
      gp_boundary.set_rect(violation_shape.get_real_rect());
      if (violation->get_is_routing()) {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(violation_shape.get_layer_idx()));
      } else {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(violation_shape.get_layer_idx()));
      }
      multi_net_violation_struct.push(gp_boundary);
      gp_gds.addStruct(multi_net_violation_struct);
    }
  }

  std::string gds_file_path = RTUTIL.getString(vr_temp_directory_path, flag, "_vr_model.gds");
  RTGP.plot(gp_gds, gds_file_path);
}

void ViolationRepairer::debugCheckVRBox(VRBox& vr_box)
{
  VRBoxId& vr_box_id = vr_box.get_vr_box_id();
  if (vr_box_id.get_x() < 0 || vr_box_id.get_y() < 0) {
    RTLOG.error(Loc::current(), "The grid coord is illegal!");
  }

  std::vector<GridMap<VRNode>>& layer_node_map = vr_box.get_layer_node_map();
  for (GridMap<VRNode>& vr_node_map : layer_node_map) {
    for (int32_t x = 0; x < vr_node_map.get_x_size(); x++) {
      for (int32_t y = 0; y < vr_node_map.get_y_size(); y++) {
        VRNode& vr_node = vr_node_map[x][y];
        if (!RTUTIL.isInside(vr_box.get_box_rect().get_real_rect(), vr_node.get_planar_coord())) {
          RTLOG.error(Loc::current(), "The vr_node is out of box!");
        }
      }
    }
  }

  for (VRTask* vr_task : vr_box.get_vr_task_list()) {
    if (vr_task->get_net_idx() < 0) {
      RTLOG.error(Loc::current(), "The idx of origin net is illegal!");
    }
  }
}

void ViolationRepairer::debugPlotVRBox(VRBox& vr_box, int32_t curr_task_idx, std::string flag)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  std::string& vr_temp_directory_path = RTDM.getConfig().vr_temp_directory_path;

  PlanarRect box_real_rect = vr_box.get_box_rect().get_real_rect();

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

  std::vector<GridMap<VRNode>>& layer_node_map = vr_box.get_layer_node_map();
  // vr_node_map
  {
    GPStruct vr_node_map_struct("vr_node_map");
    for (GridMap<VRNode>& vr_node_map : layer_node_map) {
      for (int32_t grid_x = 0; grid_x < vr_node_map.get_x_size(); grid_x++) {
        for (int32_t grid_y = 0; grid_y < vr_node_map.get_y_size(); grid_y++) {
          VRNode& vr_node = vr_node_map[grid_x][grid_y];
          PlanarRect real_rect = RTUTIL.getEnlargedRect(vr_node.get_planar_coord(), point_size);
          int32_t y_reduced_span = std::max(1, real_rect.getYSpan() / 12);
          int32_t y = real_rect.get_ur_y();

          GPBoundary gp_boundary;
          gp_boundary.set_rect(real_rect);
          gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(vr_node.get_layer_idx()));
          vr_node_map_struct.push(gp_boundary);

          y -= y_reduced_span;
          GPText gp_text_node_real_coord;
          gp_text_node_real_coord.set_coord(real_rect.get_ll_x(), y);
          gp_text_node_real_coord.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          gp_text_node_real_coord.set_message(RTUTIL.getString("(", vr_node.get_x(), " , ", vr_node.get_y(), " , ", vr_node.get_layer_idx(), ")"));
          gp_text_node_real_coord.set_layer_idx(RTGP.getGDSIdxByRouting(vr_node.get_layer_idx()));
          gp_text_node_real_coord.set_presentation(GPTextPresentation::kLeftMiddle);
          vr_node_map_struct.push(gp_text_node_real_coord);

          y -= y_reduced_span;
          GPText gp_text_node_grid_coord;
          gp_text_node_grid_coord.set_coord(real_rect.get_ll_x(), y);
          gp_text_node_grid_coord.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          gp_text_node_grid_coord.set_message(RTUTIL.getString("(", grid_x, " , ", grid_y, " , ", vr_node.get_layer_idx(), ")"));
          gp_text_node_grid_coord.set_layer_idx(RTGP.getGDSIdxByRouting(vr_node.get_layer_idx()));
          gp_text_node_grid_coord.set_presentation(GPTextPresentation::kLeftMiddle);
          vr_node_map_struct.push(gp_text_node_grid_coord);

          y -= y_reduced_span;
          GPText gp_text_obs_type_fixed_rect_map;
          gp_text_obs_type_fixed_rect_map.set_coord(real_rect.get_ll_x(), y);
          gp_text_obs_type_fixed_rect_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          gp_text_obs_type_fixed_rect_map.set_message("obs_type_fixed_rect_map: ");
          gp_text_obs_type_fixed_rect_map.set_layer_idx(RTGP.getGDSIdxByRouting(vr_node.get_layer_idx()));
          gp_text_obs_type_fixed_rect_map.set_presentation(GPTextPresentation::kLeftMiddle);
          vr_node_map_struct.push(gp_text_obs_type_fixed_rect_map);

          if (!vr_node.get_obs_type_fixed_rect_map().empty()) {
            y -= y_reduced_span;
            GPText gp_text_obs_type_fixed_rect_map_info;
            gp_text_obs_type_fixed_rect_map_info.set_coord(real_rect.get_ll_x(), y);
            gp_text_obs_type_fixed_rect_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            std::string obs_type_fixed_rect_map_info_message = "--";
            for (auto& [obs_type, net_set] : vr_node.get_obs_type_fixed_rect_map()) {
              obs_type_fixed_rect_map_info_message += RTUTIL.getString("(", GetVRObsTypeName()(obs_type));
              for (int32_t net_idx : net_set) {
                obs_type_fixed_rect_map_info_message += RTUTIL.getString(",", net_idx);
              }
              obs_type_fixed_rect_map_info_message += RTUTIL.getString(")");
            }
            gp_text_obs_type_fixed_rect_map_info.set_message(obs_type_fixed_rect_map_info_message);
            gp_text_obs_type_fixed_rect_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(vr_node.get_layer_idx()));
            gp_text_obs_type_fixed_rect_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
            vr_node_map_struct.push(gp_text_obs_type_fixed_rect_map_info);
          }

          y -= y_reduced_span;
          GPText gp_text_obs_type_routed_rect_map;
          gp_text_obs_type_routed_rect_map.set_coord(real_rect.get_ll_x(), y);
          gp_text_obs_type_routed_rect_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          gp_text_obs_type_routed_rect_map.set_message("obs_type_routed_rect_map: ");
          gp_text_obs_type_routed_rect_map.set_layer_idx(RTGP.getGDSIdxByRouting(vr_node.get_layer_idx()));
          gp_text_obs_type_routed_rect_map.set_presentation(GPTextPresentation::kLeftMiddle);
          vr_node_map_struct.push(gp_text_obs_type_routed_rect_map);

          if (!vr_node.get_obs_type_routed_rect_map().empty()) {
            y -= y_reduced_span;
            GPText gp_text_obs_type_routed_rect_map_info;
            gp_text_obs_type_routed_rect_map_info.set_coord(real_rect.get_ll_x(), y);
            gp_text_obs_type_routed_rect_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            std::string obs_type_routed_rect_map_info_message = "--";
            for (auto& [obs_type, net_set] : vr_node.get_obs_type_routed_rect_map()) {
              obs_type_routed_rect_map_info_message += RTUTIL.getString("(", GetVRObsTypeName()(obs_type));
              for (int32_t net_idx : net_set) {
                obs_type_routed_rect_map_info_message += RTUTIL.getString(",", net_idx);
              }
              obs_type_routed_rect_map_info_message += RTUTIL.getString(")");
            }
            gp_text_obs_type_routed_rect_map_info.set_message(obs_type_routed_rect_map_info_message);
            gp_text_obs_type_routed_rect_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(vr_node.get_layer_idx()));
            gp_text_obs_type_routed_rect_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
            vr_node_map_struct.push(gp_text_obs_type_routed_rect_map_info);
          }

          y -= y_reduced_span;
          GPText gp_text_obs_type_violation_number_map;
          gp_text_obs_type_violation_number_map.set_coord(real_rect.get_ll_x(), y);
          gp_text_obs_type_violation_number_map.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
          gp_text_obs_type_violation_number_map.set_message("obs_type_violation_number_map: ");
          gp_text_obs_type_violation_number_map.set_layer_idx(RTGP.getGDSIdxByRouting(vr_node.get_layer_idx()));
          gp_text_obs_type_violation_number_map.set_presentation(GPTextPresentation::kLeftMiddle);
          vr_node_map_struct.push(gp_text_obs_type_violation_number_map);

          if (!vr_node.get_obs_type_violation_number_map().empty()) {
            y -= y_reduced_span;
            GPText gp_text_obs_type_violation_number_map_info;
            gp_text_obs_type_violation_number_map_info.set_coord(real_rect.get_ll_x(), y);
            gp_text_obs_type_violation_number_map_info.set_text_type(static_cast<int32_t>(GPDataType::kInfo));
            std::string obs_type_violation_number_map_info_message = "--";
            for (auto& [obs_type, violation_number] : vr_node.get_obs_type_violation_number_map()) {
              obs_type_violation_number_map_info_message += RTUTIL.getString("(", GetVRObsTypeName()(obs_type), ",", violation_number != 0, ")");
            }
            gp_text_obs_type_violation_number_map_info.set_message(obs_type_violation_number_map_info_message);
            gp_text_obs_type_violation_number_map_info.set_layer_idx(RTGP.getGDSIdxByRouting(vr_node.get_layer_idx()));
            gp_text_obs_type_violation_number_map_info.set_presentation(GPTextPresentation::kLeftMiddle);
            vr_node_map_struct.push(gp_text_obs_type_violation_number_map_info);
          }
        }
      }
    }
    gp_gds.addStruct(vr_node_map_struct);
  }

  // box_track_axis
  {
    GPStruct box_track_axis_struct("box_track_axis");
    PlanarCoord& real_ll = box_real_rect.get_ll();
    PlanarCoord& real_ur = box_real_rect.get_ur();
    ScaleAxis& box_track_axis = vr_box.get_box_track_axis();
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
  for (auto& [is_routing, layer_net_rect_map] : vr_box.get_type_layer_net_fixed_rect_map()) {
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
  for (auto& [net_idx, segment_set] : vr_box.get_net_final_result_map()) {
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

  // net_final_patch
  for (auto& [net_idx, patch_set] : vr_box.get_net_final_patch_map()) {
    GPStruct final_patch_struct(RTUTIL.getString("final_patch(net_", net_idx, ")"));
    for (EXTLayerRect* patch : patch_set) {
      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kShape));
      gp_boundary.set_rect(patch->get_real_rect());
      gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(patch->get_layer_idx()));
      final_patch_struct.push(gp_boundary);
    }
    gp_gds.addStruct(final_patch_struct);
  }

  // task
  for (VRTask* vr_task : vr_box.get_vr_task_list()) {
    GPStruct task_struct(RTUTIL.getString("task(net_", vr_task->get_net_idx(), ")"));

    for (Segment<LayerCoord>& segment : vr_box.get_net_task_final_result_map()[vr_task->get_net_idx()]) {
      for (NetShape& net_shape : RTDM.getNetShapeList(vr_task->get_net_idx(), segment)) {
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
    for (EXTLayerRect& patch : vr_box.get_net_task_final_patch_map()[vr_task->get_net_idx()]) {
      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPath));
      gp_boundary.set_rect(patch.get_real_rect());
      gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(patch.get_layer_idx()));
      task_struct.push(gp_boundary);
    }
    gp_gds.addStruct(task_struct);
  }

  // violation
  {
    for (Violation& violation : vr_box.get_violation_list()) {
      if (violation.get_violation_net_set().size() >= 2) {
        continue;
      }
      GPStruct single_net_violation_struct(RTUTIL.getString("single_net_violation_", GetViolationTypeName()(violation.get_violation_type())));
      EXTLayerRect& violation_shape = violation.get_violation_shape();

      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kViolation));
      gp_boundary.set_rect(violation_shape.get_real_rect());
      if (violation.get_is_routing()) {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(violation_shape.get_layer_idx()));
      } else {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(violation_shape.get_layer_idx()));
      }
      single_net_violation_struct.push(gp_boundary);
      gp_gds.addStruct(single_net_violation_struct);
    }
    for (Violation& violation : vr_box.get_violation_list()) {
      if (violation.get_violation_net_set().size() < 2) {
        continue;
      }
      GPStruct multi_net_violation_struct(RTUTIL.getString("multi_net_violation_", GetViolationTypeName()(violation.get_violation_type())));
      EXTLayerRect& violation_shape = violation.get_violation_shape();

      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kViolation));
      gp_boundary.set_rect(violation_shape.get_real_rect());
      if (violation.get_is_routing()) {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByRouting(violation_shape.get_layer_idx()));
      } else {
        gp_boundary.set_layer_idx(RTGP.getGDSIdxByCut(violation_shape.get_layer_idx()));
      }
      multi_net_violation_struct.push(gp_boundary);
      gp_gds.addStruct(multi_net_violation_struct);
    }
  }

  std::string gds_file_path
      = RTUTIL.getString(vr_temp_directory_path, flag, "_vr_box_", vr_box.get_vr_box_id().get_x(), "_", vr_box.get_vr_box_id().get_y(), ".gds");
  RTGP.plot(gp_gds, gds_file_path);
}

#endif

}  // namespace irt
