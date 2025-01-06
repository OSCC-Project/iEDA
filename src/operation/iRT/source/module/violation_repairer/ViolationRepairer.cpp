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
  // debugPlotVRModel(vr_model, "before");
  repairViolation(vr_model);
  // debugPlotVRModel(vr_model, "after");
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
  return RTDE.getViolationListByTemp(de_task);
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
  return RTDE.getViolationListByTemp(de_task);
}

void ViolationRepairer::repairViolation(VRModel& vr_model)
{
  for (size_t i = 0; i < 1; i++) {
    initVRBoxList(vr_model);
    buildBoxSchedule(vr_model);
    repairVRBoxList(vr_model);
    uploadViolation(vr_model);
    updateSummary(vr_model);
    printSummary(vr_model);
  }
}

void ViolationRepairer::initVRBoxList(VRModel& vr_model)
{
}

void ViolationRepairer::buildBoxSchedule(VRModel& vr_model)
{
}

void ViolationRepairer::repairVRBoxList(VRModel& vr_model)
{
}

#if 1  // exhibit

void ViolationRepairer::updateSummary(VRModel& vr_model)
{
  Die& die = RTDM.getDatabase().get_die();
  Summary& summary = RTDM.getDatabase().get_summary();

  std::map<int32_t, std::map<std::string, int32_t>>& routing_violation_type_num_map = summary.vr_summary.routing_violation_type_num_map;
  std::map<std::string, int32_t>& violation_type_num_map = summary.vr_summary.violation_type_num_map;
  std::map<int32_t, int32_t>& routing_violation_num_map = summary.vr_summary.routing_violation_num_map;
  int32_t& total_violation_num = summary.vr_summary.total_violation_num;

  routing_violation_type_num_map.clear();
  violation_type_num_map.clear();
  routing_violation_num_map.clear();
  total_violation_num = 0;

  for (Violation* violation : RTDM.getViolationSet(die)) {
    routing_violation_type_num_map[violation->get_violation_shape().get_layer_idx()]
                                  [GetViolationTypeName()(violation->get_violation_type())]++;
    violation_type_num_map[GetViolationTypeName()(violation->get_violation_type())]++;
    routing_violation_num_map[violation->get_violation_shape().get_layer_idx()]++;
    total_violation_num++;
  }
}

void ViolationRepairer::printSummary(VRModel& vr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  Summary& summary = RTDM.getDatabase().get_summary();

  std::map<int32_t, std::map<std::string, int32_t>>& routing_violation_type_num_map = summary.vr_summary.routing_violation_type_num_map;
  std::map<std::string, int32_t>& violation_type_num_map = summary.vr_summary.violation_type_num_map;
  std::map<int32_t, int32_t>& routing_violation_num_map = summary.vr_summary.routing_violation_num_map;
  int32_t& total_violation_num = summary.vr_summary.total_violation_num;

  fort::char_table violation_map_table;
  {
    violation_map_table << fort::header << "routing_layer";
    for (auto& [type, num] : violation_type_num_map) {
      violation_map_table << fort::header << type;
    }
    violation_map_table << fort::header << "Total" << fort::endr;

    for (RoutingLayer& routing_layer : routing_layer_list) {
      violation_map_table << routing_layer.get_layer_name();
      for (auto& [type, num] : violation_type_num_map) {
        violation_map_table << routing_violation_type_num_map[routing_layer.get_layer_idx()][type];
      }
      violation_map_table << routing_violation_num_map[routing_layer.get_layer_idx()] << fort::endr;
    }

    violation_map_table << fort::header << "Total";
    for (auto& [type, num] : violation_type_num_map) {
      violation_map_table << fort::header << num;
    }
    violation_map_table << fort::header << total_violation_num << fort::endr;
  }
  RTUTIL.printTableList({violation_map_table});
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

  std::string gds_file_path = RTUTIL.getString(vr_temp_directory_path, flag, "_vr_model.gds");
  RTGP.plot(gp_gds, gds_file_path);
}

#endif

}  // namespace irt
