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
#include "ViolationReporter.hpp"

#include "DRCEngine.hpp"
#include "GDSPlotter.hpp"
#include "RTInterface.hpp"
#include "Utility.hpp"

namespace irt {

// public

void ViolationReporter::initInst()
{
  if (_vr_instance == nullptr) {
    _vr_instance = new ViolationReporter();
  }
}

ViolationReporter& ViolationReporter::getInst()
{
  if (_vr_instance == nullptr) {
    RTLOG.error(Loc::current(), "The instance not initialized!");
  }
  return *_vr_instance;
}

void ViolationReporter::destroyInst()
{
  if (_vr_instance != nullptr) {
    delete _vr_instance;
    _vr_instance = nullptr;
  }
}

// function

void ViolationReporter::report()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");
  VRModel vr_model = initVRModel();
  uploadViolation(vr_model);
  // debugPlotVRModel(vr_model, "best");
  updateSummary(vr_model);
  printSummary(vr_model);
  outputNetCSV(vr_model);
  outputViolationCSV(vr_model);
  outputNetJson(vr_model);
  outputViolationJson(vr_model);
  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

// private

ViolationReporter* ViolationReporter::_vr_instance = nullptr;

VRModel ViolationReporter::initVRModel()
{
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();

  VRModel vr_model;
  vr_model.set_vr_net_list(convertToVRNetList(net_list));
  return vr_model;
}

std::vector<VRNet> ViolationReporter::convertToVRNetList(std::vector<Net>& net_list)
{
  std::vector<VRNet> vr_net_list;
  vr_net_list.reserve(net_list.size());
  for (Net& net : net_list) {
    vr_net_list.emplace_back(convertToVRNet(net));
  }
  return vr_net_list;
}

VRNet ViolationReporter::convertToVRNet(Net& net)
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

void ViolationReporter::uploadViolation(VRModel& vr_model)
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  Die& die = RTDM.getDatabase().get_die();

  for (Violation* violation : RTDM.getViolationSet(die)) {
    RTDM.updateViolationToGCellMap(ChangeType::kDel, violation);
  }
  for (Violation violation : getViolationList(vr_model)) {
    RTDM.updateViolationToGCellMap(ChangeType::kAdd, new Violation(violation));
  }

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

std::vector<Violation> ViolationReporter::getViolationList(VRModel& vr_model)
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
    for (VRNet& vr_net : vr_model.get_vr_net_list()) {
      need_checked_net_set.insert(vr_net.get_net_idx());
    }

    de_task.set_proc_type(DEProcType::kGet);
    de_task.set_net_type(DENetType::kPatchHybrid);
    de_task.set_top_name(top_name);
    de_task.set_env_shape_list(env_shape_list);
    de_task.set_net_pin_shape_map(net_pin_shape_map);
    de_task.set_net_result_map(net_result_map);
    de_task.set_net_patch_map(net_patch_map);
    de_task.set_need_checked_net_set(need_checked_net_set);
  }
  return RTDE.getViolationList(de_task);
}

#if 1  // exhibit

void ViolationReporter::updateSummary(VRModel& vr_model)
{
  int32_t micron_dbu = RTDM.getDatabase().get_micron_dbu();
  Die& die = RTDM.getDatabase().get_die();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();
  Summary& summary = RTDM.getDatabase().get_summary();
  int32_t enable_timing = RTDM.getConfig().enable_timing;

  std::map<int32_t, double>& routing_wire_length_map = summary.vr_summary.routing_wire_length_map;
  double& total_wire_length = summary.vr_summary.total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.vr_summary.cut_via_num_map;
  int32_t& total_via_num = summary.vr_summary.total_via_num;
  std::map<int32_t, int32_t>& routing_patch_num_map = summary.vr_summary.routing_patch_num_map;
  int32_t& total_patch_num = summary.vr_summary.total_patch_num;
  std::map<int32_t, std::map<std::string, int32_t>>& within_net_routing_violation_type_num_map = summary.vr_summary.within_net_routing_violation_type_num_map;
  std::map<std::string, int32_t>& within_net_violation_type_num_map = summary.vr_summary.within_net_violation_type_num_map;
  std::map<int32_t, int32_t>& within_net_routing_violation_num_map = summary.vr_summary.within_net_routing_violation_num_map;
  int32_t& within_net_total_violation_num = summary.vr_summary.within_net_total_violation_num;
  std::map<int32_t, std::map<std::string, int32_t>>& among_net_routing_violation_type_num_map = summary.vr_summary.among_net_routing_violation_type_num_map;
  std::map<std::string, int32_t>& among_net_violation_type_num_map = summary.vr_summary.among_net_violation_type_num_map;
  std::map<int32_t, int32_t>& among_net_routing_violation_num_map = summary.vr_summary.among_net_routing_violation_num_map;
  int32_t& among_net_total_violation_num = summary.vr_summary.among_net_total_violation_num;
  std::map<std::string, std::map<std::string, double>>& clock_timing_map = summary.vr_summary.clock_timing_map;
  std::map<std::string, double>& type_power_map = summary.vr_summary.type_power_map;

  std::vector<VRNet>& vr_net_list = vr_model.get_vr_net_list();

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
    for (auto& [net_idx, segment_set] : RTDM.getNetDetailedResultMap(die)) {
      for (Segment<LayerCoord>* segment : segment_set) {
        routing_segment_list_list[net_idx].emplace_back(segment->get_first(), segment->get_second());
      }
    }
    RTI.updateTimingAndPower(real_pin_coord_map_list, routing_segment_list_list, clock_timing_map, type_power_map);
  }
}

void ViolationReporter::printSummary(VRModel& vr_model)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  Summary& summary = RTDM.getDatabase().get_summary();
  int32_t enable_timing = RTDM.getConfig().enable_timing;

  std::map<int32_t, double>& routing_wire_length_map = summary.vr_summary.routing_wire_length_map;
  double& total_wire_length = summary.vr_summary.total_wire_length;
  std::map<int32_t, int32_t>& cut_via_num_map = summary.vr_summary.cut_via_num_map;
  int32_t& total_via_num = summary.vr_summary.total_via_num;
  std::map<int32_t, int32_t>& routing_patch_num_map = summary.vr_summary.routing_patch_num_map;
  int32_t& total_patch_num = summary.vr_summary.total_patch_num;
  std::map<int32_t, std::map<std::string, int32_t>>& within_net_routing_violation_type_num_map = summary.vr_summary.within_net_routing_violation_type_num_map;
  std::map<std::string, int32_t>& within_net_violation_type_num_map = summary.vr_summary.within_net_violation_type_num_map;
  std::map<int32_t, int32_t>& within_net_routing_violation_num_map = summary.vr_summary.within_net_routing_violation_num_map;
  int32_t& within_net_total_violation_num = summary.vr_summary.within_net_total_violation_num;
  std::map<int32_t, std::map<std::string, int32_t>>& among_net_routing_violation_type_num_map = summary.vr_summary.among_net_routing_violation_type_num_map;
  std::map<std::string, int32_t>& among_net_violation_type_num_map = summary.vr_summary.among_net_violation_type_num_map;
  std::map<int32_t, int32_t>& among_net_routing_violation_num_map = summary.vr_summary.among_net_routing_violation_num_map;
  int32_t& among_net_total_violation_num = summary.vr_summary.among_net_total_violation_num;
  std::map<std::string, std::map<std::string, double>>& clock_timing_map = summary.vr_summary.clock_timing_map;
  std::map<std::string, double>& type_power_map = summary.vr_summary.type_power_map;

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
  RTUTIL.printTableList({within_net_routing_violation_map_table});
  RTUTIL.printTableList({among_net_routing_violation_map_table});
  RTUTIL.printTableList({timing_table, power_table});
}

void ViolationReporter::outputNetCSV(VRModel& vr_model)
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
    std::ofstream* net_csv_file = RTUTIL.getOutputFileStream(RTUTIL.getString(vr_temp_directory_path, "net_map_", routing_layer.get_layer_name(), ".csv"));
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

void ViolationReporter::outputViolationCSV(VRModel& vr_model)
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
    std::ofstream* violation_csv_file
        = RTUTIL.getOutputFileStream(RTUTIL.getString(vr_temp_directory_path, "violation_map_", routing_layer.get_layer_name(), ".csv"));
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

void ViolationReporter::outputNetJson(VRModel& vr_model)
{
  Die& die = RTDM.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();
  std::string& vr_temp_directory_path = RTDM.getConfig().vr_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
  std::vector<nlohmann::json> net_json_list;
  net_json_list.resize(net_list.size());
  for (Net& net : net_list) {
    net_json_list[net.get_net_idx()]["net_name"] = net.get_net_name();
  }
  for (auto& [net_idx, segment_set] : RTDM.getNetDetailedResultMap(die)) {
    for (Segment<LayerCoord>* segment : segment_set) {
      for (NetShape& net_shape : RTDM.getNetShapeList(net_idx, *segment)) {
        std::string layer_name;
        if (net_shape.get_is_routing()) {
          layer_name = routing_layer_list[net_shape.get_layer_idx()].get_layer_name();
        } else {
          layer_name = cut_layer_list[net_shape.get_layer_idx()].get_layer_name();
        }
        net_json_list[net_idx]["result"].push_back({net_shape.get_ll_x(), net_shape.get_ll_y(), net_shape.get_ur_x(), net_shape.get_ur_y(), layer_name});
      }
    }
  }
  for (auto& [net_idx, patch_set] : RTDM.getNetDetailedPatchMap(die)) {
    for (EXTLayerRect* patch : patch_set) {
      net_json_list[net_idx]["patch"].push_back({patch->get_real_ll_x(), patch->get_real_ll_y(), patch->get_real_ur_x(), patch->get_real_ur_y(),
                                                 routing_layer_list[patch->get_layer_idx()].get_layer_name()});
    }
  }
  std::string net_json_file_path = RTUTIL.getString(RTUTIL.getString(vr_temp_directory_path, "net_map.json"));
  std::ofstream* net_json_file = RTUTIL.getOutputFileStream(net_json_file_path);
  (*net_json_file) << net_json_list;
  RTUTIL.closeFileStream(net_json_file);
  RTI.sendNotification(RTUTIL.getString("RT_VR_net_map"), net_json_file_path);
}

void ViolationReporter::outputViolationJson(VRModel& vr_model)
{
  Die& die = RTDM.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();
  std::string& vr_temp_directory_path = RTDM.getConfig().vr_temp_directory_path;
  int32_t output_inter_result = RTDM.getConfig().output_inter_result;
  if (!output_inter_result) {
    return;
  }
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
  std::string violation_json_file_path = RTUTIL.getString(vr_temp_directory_path, "violation_map.json");
  std::ofstream* violation_json_file = RTUTIL.getOutputFileStream(violation_json_file_path);
  (*violation_json_file) << violation_json_list;
  RTUTIL.closeFileStream(violation_json_file);
  RTI.sendNotification(RTUTIL.getString("RT_VR_violation_map"), violation_json_file_path);
}

#endif

#if 1  // debug

void ViolationReporter::debugPlotVRModel(VRModel& vr_model, std::string flag)
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
  for (auto& [net_idx, segment_set] : RTDM.getNetDetailedResultMap(die)) {
    GPStruct detailed_result_struct(RTUTIL.getString("detailed_result(net_", net_idx, ")"));
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
      if (violation->get_violation_net_set().size() >= 2) {
        continue;
      }
      GPStruct within_net_violation_struct(RTUTIL.getString("within_net_violation_", GetViolationTypeName()(violation->get_violation_type())));
      EXTLayerRect& violation_shape = violation->get_violation_shape();

      GPBoundary gp_boundary;
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPatchViolation));
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
      gp_boundary.set_data_type(static_cast<int32_t>(GPDataType::kPatchViolation));
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

  std::string gds_file_path = RTUTIL.getString(vr_temp_directory_path, flag, "_vr_model.gds");
  RTGP.plot(gp_gds, gds_file_path);
}

#endif

}  // namespace irt
