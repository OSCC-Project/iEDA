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
#include "DRCEngine.hpp"

#include "GDSPlotter.hpp"
#include "RTInterface.hpp"
#include "Utility.hpp"

namespace irt {

// public

void DRCEngine::initInst()
{
  if (_de_instance == nullptr) {
    _de_instance = new DRCEngine();
  }
}

DRCEngine& DRCEngine::getInst()
{
  if (_de_instance == nullptr) {
    RTLOG.error(Loc::current(), "The instance not initialized!");
  }
  return *_de_instance;
}

void DRCEngine::destroyInst()
{
  if (_de_instance != nullptr) {
    delete _de_instance;
    _de_instance = nullptr;
  }
}

// function

void DRCEngine::init()
{
  Monitor monitor;
  RTLOG.info(Loc::current(), "Starting...");

  updateIgnoreViolationSet();

  RTLOG.info(Loc::current(), "Completed", monitor.getStatsInfo());
}

void DRCEngine::updateIgnoreViolationSet()
{
  Die& die = RTDM.getDatabase().get_die();
  std::vector<Net>& net_list = RTDM.getDatabase().get_net_list();

  std::string top_name = RTUTIL.getString("ignore_violation");
  PlanarRect check_region = die.get_real_rect();
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
  for (auto& [net_idx, segment_set] : RTDM.getNetAccessResultMap(die)) {
    for (Segment<LayerCoord>* segment : segment_set) {
      net_result_map[net_idx].push_back(segment);
    }
  }
  for (auto& [net_idx, segment_set] : RTDM.getNetDetailedResultMap(die)) {
    for (Segment<LayerCoord>* segment : segment_set) {
      net_result_map[net_idx].push_back(segment);
    }
  }
  std::map<int32_t, std::vector<EXTLayerRect*>> net_patch_map;
  for (auto& [net_idx, patch_set] : RTDM.getNetAccessPatchMap(die)) {
    for (EXTLayerRect* patch : patch_set) {
      net_patch_map[net_idx].push_back(patch);
    }
  }
  for (auto& [net_idx, patch_set] : RTDM.getNetDetailedPatchMap(die)) {
    for (EXTLayerRect* patch : patch_set) {
      net_patch_map[net_idx].push_back(patch);
    }
  }
  std::set<int32_t> need_checked_net_set;
  for (Net& net : net_list) {
    need_checked_net_set.insert(net.get_net_idx());
  }

  DETask de_task;
  de_task.set_process_type_set({DEProcessType::kCost});
  de_task.set_top_name(top_name);
  de_task.set_check_region(check_region);
  de_task.set_env_shape_list(env_shape_list);
  de_task.set_net_pin_shape_map(net_pin_shape_map);
  de_task.set_net_result_map(net_result_map);
  de_task.set_net_patch_map(net_patch_map);
  de_task.set_need_checked_net_set(need_checked_net_set);
  for (Violation violation : getViolationList(de_task, false)) {
    _ignore_violation_set.insert(violation);
  }
}

std::vector<Violation> DRCEngine::getViolationList(DETask& de_task)
{
  return getViolationList(de_task, true);
}

// private

DRCEngine* DRCEngine::_de_instance = nullptr;

std::vector<Violation> DRCEngine::getViolationList(DETask& de_task, bool post_process)
{
  getViolationListByInterface(de_task);

  // getViolationListBySelf(de_task);
  filterViolationList(de_task);
  if (post_process) {
    fixViolationNetSet(de_task);
    explandViolationList(de_task);
    buildViolationList(de_task);
  }
  return de_task.get_violation_list();
}

void DRCEngine::getViolationListBySelf(DETask& de_task)
{
  buildTask(de_task);
  writeTask(de_task);
  readTask(de_task);
}

void DRCEngine::buildTask(DETask& de_task)
{
  std::string& de_temp_directory_path = RTDM.getConfig().de_temp_directory_path;

  de_task.set_top_dir_path(RTUTIL.getString(de_temp_directory_path, de_task.get_top_name()));
  de_task.set_def_file_path(RTUTIL.getString(de_task.get_top_dir_path(), "/clean.def"));
  de_task.set_netlist_file_path(RTUTIL.getString(de_task.get_top_dir_path(), "/clean.v"));
  de_task.set_prepared_file_path(RTUTIL.getString(de_task.get_top_dir_path(), "/prepared"));
  de_task.set_finished_file_path(RTUTIL.getString(de_task.get_top_dir_path(), "/finished"));
  de_task.set_violation_file_path(RTUTIL.getString(de_task.get_top_dir_path(), "/drc.irt"));
  // 删除再构建top文件夹
  RTUTIL.removeDir(de_task.get_top_dir_path());
  RTUTIL.createDir(de_task.get_top_dir_path());
  // 修改文件夹权限
  std::filesystem::perms permissions
      = std::filesystem::perms::owner_all | std::filesystem::perms::group_all | std::filesystem::perms::others_all;
  RTUTIL.changePermissions(de_task.get_top_dir_path(), permissions);
}

void DRCEngine::writeTask(DETask& de_task)
{
  int32_t micron_dbu = RTDM.getDatabase().get_micron_dbu();
  Die& die = RTDM.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = RTDM.getDatabase().get_cut_to_adjacent_routing_map();

  std::string& top_name = de_task.get_top_name();
  std::vector<std::pair<EXTLayerRect*, bool>>& env_shape_list = de_task.get_env_shape_list();
  std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>>& net_pin_shape_map = de_task.get_net_pin_shape_map();
  std::map<int32_t, std::vector<Segment<LayerCoord>*>>& net_result_map = de_task.get_net_result_map();
  std::map<int32_t, std::vector<EXTLayerRect*>>& net_patch_map = de_task.get_net_patch_map();
  std::string& def_file_path = de_task.get_def_file_path();
  std::string& netlist_file_path = de_task.get_netlist_file_path();
  std::string& prepared_file_path = de_task.get_prepared_file_path();

  std::set<int32_t> net_idx_set;
  // 获取所有net
  {
    net_idx_set.insert(-1);
    for (auto& [net_idx, pin_shape_list] : net_pin_shape_map) {
      net_idx_set.insert(net_idx);
    }
    for (auto& [net_idx, segment_list] : net_result_map) {
      net_idx_set.insert(net_idx);
    }
    for (auto& [net_idx, patch_list] : net_patch_map) {
      net_idx_set.insert(net_idx);
    }
  }
  // 构建def
  {
    std::ofstream* def_file = RTUTIL.getOutputFileStream(def_file_path);
    // 构建header
    RTUTIL.pushStream(def_file, "VERSION 5.8 ;", "\n");
    RTUTIL.pushStream(def_file, "DESIGN ", top_name, " ;", "\n");
    RTUTIL.pushStream(def_file, "UNITS DISTANCE MICRONS ", micron_dbu, " ;", "\n");
    RTUTIL.pushStream(def_file, "DIEAREA ( ", die.get_real_ll_x(), " ", die.get_real_ll_y(), " ) ( ", die.get_real_ur_x(), " ",
                      die.get_real_ur_y(), " ) ;", "\n");
    RTUTIL.pushStream(def_file, "\n");
    // 构建track
    for (auto it = routing_layer_list.rbegin(); it != routing_layer_list.rend(); ++it) {
      RoutingLayer& routing_layer = *it;
      ScaleGrid& y_track_grid = routing_layer.getYTrackGridList().front();
      ScaleGrid& x_track_grid = routing_layer.getXTrackGridList().front();
      RTUTIL.pushStream(def_file, "TRACKS Y ", y_track_grid.get_start_line(), " DO ", y_track_grid.get_step_num() + 1, " STEP ",
                        y_track_grid.get_step_length(), " LAYER ", routing_layer.get_layer_name(), " ;", "\n");
      RTUTIL.pushStream(def_file, "TRACKS X ", x_track_grid.get_start_line(), " DO ", x_track_grid.get_step_num() + 1, " STEP ",
                        x_track_grid.get_step_length(), " LAYER ", routing_layer.get_layer_name(), " ;", "\n");
    }
    RTUTIL.pushStream(def_file, "\n");
    // 构建via
    std::map<LayerRect, std::string, CmpLayerRectByXASC> cut_shape_via_map;
    for (int32_t net_idx : net_idx_set) {
      if (net_idx == -1) {
        for (auto& [env_shape, is_routing] : env_shape_list) {
          PlanarRect& real_rect = env_shape->get_real_rect();
          int32_t layer_idx = env_shape->get_layer_idx();
          PlanarCoord mid_point = real_rect.getMidPoint();
          if (!is_routing) {
            if (cut_to_adjacent_routing_map[layer_idx].size() < 2) {
              continue;
            }
            LayerRect cut_shape(real_rect.get_ll_x() - mid_point.get_x(), real_rect.get_ll_y() - mid_point.get_y(),
                                real_rect.get_ur_x() - mid_point.get_x(), real_rect.get_ur_y() - mid_point.get_y(), layer_idx);
            cut_shape_via_map[cut_shape] = "";
          }
        }
      } else {
        if (RTUTIL.exist(net_pin_shape_map, net_idx)) {
          for (auto& [pin_shape, is_routing] : net_pin_shape_map[net_idx]) {
            PlanarRect& real_rect = pin_shape->get_real_rect();
            int32_t layer_idx = pin_shape->get_layer_idx();
            PlanarCoord mid_point = real_rect.getMidPoint();
            if (!is_routing) {
              if (cut_to_adjacent_routing_map[layer_idx].size() < 2) {
                continue;
              }
              LayerRect cut_shape(real_rect.get_ll_x() - mid_point.get_x(), real_rect.get_ll_y() - mid_point.get_y(),
                                  real_rect.get_ur_x() - mid_point.get_x(), real_rect.get_ur_y() - mid_point.get_y(), layer_idx);
              cut_shape_via_map[cut_shape] = "";
            }
          }
        }
      }
    }
    RTUTIL.pushStream(def_file, "VIAS ", cut_shape_via_map.size(), " ;", "\n");
    for (auto& [cut_shape, via_name] : cut_shape_via_map) {
      std::vector<std::string> layer_name_list;
      for (int32_t routing_layer_idx : cut_to_adjacent_routing_map[cut_shape.get_layer_idx()]) {
        layer_name_list.push_back(routing_layer_list[routing_layer_idx].get_layer_name());
      }
      layer_name_list.push_back(cut_layer_list[cut_shape.get_layer_idx()].get_layer_name());
      via_name = RTUTIL.getString("VIA_CUT_", cut_shape.get_layer_idx(), "_", cut_shape.get_ll_x(), "_", cut_shape.get_ll_y(), "_",
                                  cut_shape.get_ur_x(), "_", cut_shape.get_ur_y());
      RTUTIL.pushStream(def_file, "- ", via_name, "\n");
      for (std::string& layer_name : layer_name_list) {
        RTUTIL.pushStream(def_file, " + RECT ", layer_name, " ( ", cut_shape.get_ll_x(), " ", cut_shape.get_ll_y(), " ) ( ",
                          cut_shape.get_ur_x(), " ", cut_shape.get_ur_y(), " )", "\n");
      }
      RTUTIL.pushStream(def_file, " ;", "\n");
    }
    RTUTIL.pushStream(def_file, "END VIAS", "\n");
    RTUTIL.pushStream(def_file, "\n");
    // 构建net
    RTUTIL.pushStream(def_file, "NETS ", net_idx_set.size(), " ;", "\n");
    for (int32_t net_idx : net_idx_set) {
      std::string flag = "  + ROUTED";

      if (net_idx == -1) {
        RTUTIL.pushStream(def_file, "- net_blockage", "\n");
        for (auto& [env_shape, is_routing] : env_shape_list) {
          PlanarRect& real_rect = env_shape->get_real_rect();
          int32_t layer_idx = env_shape->get_layer_idx();
          PlanarCoord mid_point = real_rect.getMidPoint();
          if (is_routing) {
            RTUTIL.pushStream(def_file, flag, " ", routing_layer_list[layer_idx].get_layer_name(), " ( ", mid_point.get_x(), " ",
                              mid_point.get_y(), " ) RECT ( ", real_rect.get_ll_x() - mid_point.get_x(), " ",
                              real_rect.get_ll_y() - mid_point.get_y(), " ", real_rect.get_ur_x() - mid_point.get_x(), " ",
                              real_rect.get_ur_y() - mid_point.get_y(), " )", "\n");
            flag = "    NEW";
          } else {
            LayerRect cut_shape(real_rect.get_ll_x() - mid_point.get_x(), real_rect.get_ll_y() - mid_point.get_y(),
                                real_rect.get_ur_x() - mid_point.get_x(), real_rect.get_ur_y() - mid_point.get_y(), layer_idx);
            if (RTUTIL.exist(cut_shape_via_map, cut_shape)) {
              std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[layer_idx];
              int32_t routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
              RTUTIL.pushStream(def_file, flag, " ", routing_layer_list[routing_layer_idx].get_layer_name(), " ( ", mid_point.get_x(), " ",
                                mid_point.get_y(), " ) ", cut_shape_via_map[cut_shape], "\n");
              flag = "    NEW";
            }
          }
        }
      } else {
        RTUTIL.pushStream(def_file, "- net_", net_idx, "\n");
        if (RTUTIL.exist(net_pin_shape_map, net_idx)) {
          for (auto& [pin_shape, is_routing] : net_pin_shape_map[net_idx]) {
            PlanarRect& real_rect = pin_shape->get_real_rect();
            int32_t layer_idx = pin_shape->get_layer_idx();
            PlanarCoord mid_point = real_rect.getMidPoint();
            if (is_routing) {
              RTUTIL.pushStream(def_file, flag, " ", routing_layer_list[layer_idx].get_layer_name(), " ( ", mid_point.get_x(), " ",
                                mid_point.get_y(), " ) RECT ( ", real_rect.get_ll_x() - mid_point.get_x(), " ",
                                real_rect.get_ll_y() - mid_point.get_y(), " ", real_rect.get_ur_x() - mid_point.get_x(), " ",
                                real_rect.get_ur_y() - mid_point.get_y(), " )", "\n");
              flag = "    NEW";
            } else {
              LayerRect cut_shape(real_rect.get_ll_x() - mid_point.get_x(), real_rect.get_ll_y() - mid_point.get_y(),
                                  real_rect.get_ur_x() - mid_point.get_x(), real_rect.get_ur_y() - mid_point.get_y(), layer_idx);
              if (RTUTIL.exist(cut_shape_via_map, cut_shape)) {
                std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[layer_idx];
                int32_t routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
                RTUTIL.pushStream(def_file, flag, " ", routing_layer_list[routing_layer_idx].get_layer_name(), " ( ", mid_point.get_x(),
                                  " ", mid_point.get_y(), " ) ", cut_shape_via_map[cut_shape], "\n");
                flag = "    NEW";
              }
            }
          }
        }
        if (RTUTIL.exist(net_result_map, net_idx)) {
          for (Segment<LayerCoord>* segment : net_result_map[net_idx]) {
            LayerCoord& first_coord = segment->get_first();
            LayerCoord& second_coord = segment->get_second();
            int32_t first_layer_idx = first_coord.get_layer_idx();
            int32_t second_layer_idx = second_coord.get_layer_idx();
            if (first_layer_idx != second_layer_idx) {
              RTUTIL.swapByASC(first_layer_idx, second_layer_idx);
              for (int32_t layer_idx = first_layer_idx; layer_idx < second_layer_idx; layer_idx++) {
                ViaMaster& via_master = layer_via_master_list[layer_idx].front();
                std::string layer_name = routing_layer_list[via_master.get_above_enclosure().get_layer_idx()].get_layer_name();
                RTUTIL.pushStream(def_file, flag, " ", layer_name, " ( ", first_coord.get_x(), " ", first_coord.get_y(), " ) ",
                                  via_master.get_via_name(), "\n");
                flag = "    NEW";
              }
            } else {
              for (NetShape& net_shape : RTDM.getNetShapeList(net_idx, *segment)) {
                if (!net_shape.get_is_routing()) {
                  RTLOG.error(Loc::current(), "The net_shape is not routing!");
                }
                std::string layer_name = routing_layer_list[net_shape.get_layer_idx()].get_layer_name();
                PlanarCoord mid_point = net_shape.getMidPoint();
                RTUTIL.pushStream(def_file, flag, " ", layer_name, " ( ", mid_point.get_x(), " ", mid_point.get_y(), " ) RECT ( ",
                                  net_shape.get_ll_x() - mid_point.get_x(), " ", net_shape.get_ll_y() - mid_point.get_y(), " ",
                                  net_shape.get_ur_x() - mid_point.get_x(), " ", net_shape.get_ur_y() - mid_point.get_y(), " )", "\n");
                flag = "    NEW";
              }
            }
          }
        }
        if (RTUTIL.exist(net_patch_map, net_idx)) {
          for (EXTLayerRect* patch : net_patch_map[net_idx]) {
            std::string layer_name = routing_layer_list[patch->get_layer_idx()].get_layer_name();
            PlanarRect& real_rect = patch->get_real_rect();
            PlanarCoord mid_point = real_rect.getMidPoint();
            RTUTIL.pushStream(def_file, flag, " ", layer_name, " ( ", mid_point.get_x(), " ", mid_point.get_y(), " ) RECT ( ",
                              real_rect.get_ll_x() - mid_point.get_x(), " ", real_rect.get_ll_y() - mid_point.get_y(), " ",
                              real_rect.get_ur_x() - mid_point.get_x(), " ", real_rect.get_ur_y() - mid_point.get_y(), " )", "\n");
            flag = "    NEW";
          }
        }
      }
      RTUTIL.pushStream(def_file, " ;\n");
    }
    RTUTIL.pushStream(def_file, "END NETS", "\n");
    RTUTIL.pushStream(def_file, "\n");
    // 构建footer
    RTUTIL.pushStream(def_file, "END DESIGN", "\n");
    RTUTIL.pushStream(def_file, "\n");
    RTUTIL.closeFileStream(def_file);
  }
  // 构建netlist
  {
    std::ofstream* netlist_file = RTUTIL.getOutputFileStream(netlist_file_path);
    // 构建header
    RTUTIL.pushStream(netlist_file, "module ", top_name, " ();", "\n");
    RTUTIL.pushStream(netlist_file, "\n");
    // 构建net
    for (int32_t net_idx : net_idx_set) {
      if (net_idx == -1) {
        RTUTIL.pushStream(netlist_file, "wire net_blockage ;", "\n");
      } else {
        RTUTIL.pushStream(netlist_file, "wire net_", net_idx, " ;", "\n");
      }
    }
    RTUTIL.pushStream(netlist_file, "\n");
    // 构建footer
    RTUTIL.pushStream(netlist_file, "endmodule", "\n");
    RTUTIL.pushStream(netlist_file, "\n");
    RTUTIL.closeFileStream(netlist_file);
  }
  // 构建任务状态文件
  {
    std::ofstream* prepared_file = RTUTIL.getOutputFileStream(prepared_file_path);
    RTUTIL.pushStream(prepared_file, " ");
    RTUTIL.closeFileStream(prepared_file);
  }
}

void DRCEngine::readTask(DETask& de_task)
{
  int32_t micron_dbu = RTDM.getDatabase().get_micron_dbu();
  std::map<std::string, int32_t>& routing_layer_name_to_idx_map = RTDM.getDatabase().get_routing_layer_name_to_idx_map();

  std::string& top_name = de_task.get_top_name();
  std::string& finished_file_path = de_task.get_finished_file_path();
  std::string& violation_file_path = de_task.get_violation_file_path();

  // 等待直到任务结束
  {
    int32_t waiting_time = 0;
    while (!RTUTIL.existFile(finished_file_path)) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      waiting_time++;
      if (waiting_time % 500 == 0) {
        RTLOG.warn(Loc::current(), "The task ", top_name, " waited for ", waiting_time, " seconds");
      }
    }
  }
  // 从中得到违例信息
  if (RTUTIL.existFile(violation_file_path)) {
    std::ifstream* violation_file = RTUTIL.getInputFileStream(violation_file_path);

    std::string new_line;
    while (std::getline(*violation_file, new_line)) {
      if (new_line.empty()) {
        continue;
      }
      std::set<std::string> net_name_set;
      std::string required;
      std::string drc_rule_name;
      std::string layer_name;
      std::string ll_x_string;
      std::string ll_y_string;
      std::string ur_x_string;
      std::string ur_y_string;
      // 读取
      std::istringstream net_name_set_stream(new_line);
      std::string net_name;
      while (std::getline(net_name_set_stream, net_name, ',')) {
        if (!net_name.empty()) {
          net_name_set.insert(net_name);
        }
      }
      std::getline(*violation_file, new_line);
      std::istringstream drc_info_stream(new_line);
      drc_info_stream >> required >> drc_rule_name;
      std::getline(*violation_file, new_line);
      std::istringstream shape_stream(new_line);
      shape_stream >> layer_name >> ll_x_string >> ll_y_string >> ur_x_string >> ur_y_string;
      // 解析
      ViolationType violation_type = GetViolationTypeByName()(drc_rule_name);
      if (violation_type == ViolationType::kNone) {
        RTLOG.warn(Loc::current(), "Unknow rule! '", drc_rule_name, "'");
      }
      EXTLayerRect ext_layer_rect;
      ext_layer_rect.set_real_ll(static_cast<int32_t>(std::round(std::stod(ll_x_string) * micron_dbu)),
                                 static_cast<int32_t>(std::round(std::stod(ll_y_string) * micron_dbu)));
      ext_layer_rect.set_real_ur(static_cast<int32_t>(std::round(std::stod(ur_x_string) * micron_dbu)),
                                 static_cast<int32_t>(std::round(std::stod(ur_y_string) * micron_dbu)));
      ext_layer_rect.set_layer_idx(routing_layer_name_to_idx_map[layer_name]);
      std::set<int32_t> violation_net_set;
      for (const std::string& net_name : net_name_set) {
        if (net_name == "net_blockage") {
          violation_net_set.insert(-1);
        } else {
          violation_net_set.insert(std::stoi(RTUTIL.splitString(net_name, '_').back()));
        }
      }
      if (violation_net_set.size() > 2) {
        RTLOG.error(Loc::current(), "The violation_net_set size > 2!");
      }
      int32_t required_size;
      if (required == "null") {
        required_size = 0;
      } else {
        required_size = static_cast<int32_t>(std::round(std::stod(required) * micron_dbu));
      }
      Violation violation;
      violation.set_violation_type(violation_type);
      violation.set_violation_shape(ext_layer_rect);
      violation.set_is_routing(true);
      violation.set_violation_net_set(violation_net_set);
      violation.set_required_size(required_size);
      de_task.get_violation_list().push_back(violation);
    }
    RTUTIL.closeFileStream(violation_file);
  } else {
    RTLOG.warn(Loc::current(), "The task ", top_name, " violation_file_path is not exist!");
  }
  // 删除文件夹
  RTUTIL.removeDir(de_task.get_top_dir_path());
}

void DRCEngine::getViolationListByInterface(DETask& de_task)
{
  std::map<int32_t, std::vector<Segment<LayerCoord>>> net_result_map;
  for (auto& [net_idx, segment_list] : de_task.get_net_result_map()) {
    for (Segment<LayerCoord>* segment : segment_list) {
      net_result_map[net_idx].push_back(*segment);
    }
  }
  for (auto& [net_idx, segment_list] : de_task.get_net_result_map()) {
    for (Segment<LayerCoord>* segment : segment_list) {
      net_result_map[net_idx].push_back(*segment);
    }
  }
  de_task.set_violation_list(RTI.getViolationList(de_task.get_env_shape_list(), de_task.get_net_pin_shape_map(), net_result_map));
  for (Violation& violation : de_task.get_violation_list()) {
    violation.set_violation_type(ViolationType::kParallelRunLengthSpacing);
  }
}

void DRCEngine::filterViolationList(DETask& de_task)
{
  std::vector<Violation> new_violation_list;
  for (Violation& violation : de_task.get_violation_list()) {
    if (violation.get_violation_type() == ViolationType::kNone) {
      // 未知规则舍弃
      continue;
    }
    DEProcessType process_type = getDEProcessType(violation);
    if (!RTUTIL.exist(de_task.get_process_type_set(), process_type)) {
      // 非处理类型舍弃
      continue;
    }
    PlanarRect& real_rect = violation.get_violation_shape().get_real_rect();
    if (!RTUTIL.isOverlap(de_task.get_check_region(), real_rect)) {
      // 不在检查区域内的舍弃
      continue;
    }
    std::set<int32_t>& violation_net_set = violation.get_violation_net_set();
    {
      // 构建违例net
      std::set<int32_t> new_violation_net_set;
      for (int32_t violation_net_idx : violation_net_set) {
        if (violation_net_idx == -1 || RTUTIL.exist(de_task.get_need_checked_net_set(), violation_net_idx)) {
          new_violation_net_set.insert(violation_net_idx);
        }
      }
      violation_net_set = new_violation_net_set;
    }
    if (violation_net_set.empty() || (violation_net_set.size() == 1 && *violation_net_set.begin() == -1)) {
      // net不是布线net的舍弃
      continue;
    }
    if (RTUTIL.exist(_ignore_violation_set, violation)) {
      // 自带的违例舍弃
      continue;
    }
    new_violation_list.push_back(violation);
  }
  de_task.set_violation_list(new_violation_list);
}

void DRCEngine::fixViolationNetSet(DETask& de_task)
{
  // {
  //   for (Violation& violation : pa_box.get_violation_list()) {
  //     std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = RTDM.getDatabase().get_cut_to_adjacent_routing_map();

  //     std::vector<LayerRect> searched_rect_list;
  //     {
  //       EXTLayerRect& violation_shape = violation.get_violation_shape();
  //       PlanarRect enlarged_rect = RTUTIL.getEnlargedRect(violation_shape.get_real_rect(), RTDM.getOnlyPitch());
  //       if (violation.get_is_routing()) {
  //         searched_rect_list.emplace_back(enlarged_rect, violation_shape.get_layer_idx());
  //       } else {
  //         for (int32_t layer_idx : cut_to_adjacent_routing_map[violation_shape.get_layer_idx()]) {
  //           searched_rect_list.emplace_back(enlarged_rect, layer_idx);
  //         }
  //       }
  //     }
  //     std::set<int32_t> violation_net_set;
  //     for (auto& [net_idx, task_result_map] : pa_box.get_net_task_result_map()) {
  //       for (auto& [task_idx, segment_list] : task_result_map) {
  //         for (Segment<LayerCoord>& segment : segment_list) {
  //           for (LayerRect& searched_rect : searched_rect_list) {
  //             if (!RTUTIL.isOverlap(searched_rect, segment)) {
  //               continue;
  //             }
  //             violation_net_set.insert(net_idx);
  //             if (violation_net_set.size() >= 2) {
  //               break;
  //             }
  //           }
  //         }
  //       }
  //     }
  //     violation.set_violation_net_set(violation_net_set);
  //   }
  // }
}

void DRCEngine::explandViolationList(DETask& de_task)
{
  std::vector<Violation> new_violation_list;
  for (Violation& violation : de_task.get_violation_list()) {
    for (Violation new_violation : expandViolation(violation)) {
      new_violation_list.push_back(new_violation);
    }
  }
  de_task.set_violation_list(new_violation_list);
}

void DRCEngine::buildViolationList(DETask& de_task)
{
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();

  for (Violation& violation : de_task.get_violation_list()) {
    EXTLayerRect& violation_shape = violation.get_violation_shape();
    violation_shape.set_grid_rect(RTUTIL.getClosedGCellGridRect(violation_shape.get_real_rect(), gcell_axis));
  }
}

#if 1  // aux

DEProcessType DRCEngine::getDEProcessType(Violation& violation)
{
  DEProcessType de_process_type;
  std::vector<Violation> expanded_violation_list;
  buildByFunc(violation, DEFuncType::kGetDEProcessType, de_process_type, expanded_violation_list);
  return de_process_type;
}

std::vector<Violation> DRCEngine::expandViolation(Violation& violation)
{
  DEProcessType de_process_type;
  std::vector<Violation> expanded_violation_list;
  buildByFunc(violation, DEFuncType::kExpandViolation, de_process_type, expanded_violation_list);
  return expanded_violation_list;
}

void DRCEngine::buildByFunc(Violation& violation, const DEFuncType& de_func_type, DEProcessType& de_process_type,
                            std::vector<Violation>& expanded_violation_list)
{
  if (de_func_type == DEFuncType::kNone) {
    RTLOG.error(Loc::current(), "The de_func_type is none!");
  }
  PlanarRect real_rect = violation.get_violation_shape().get_real_rect();
  std::vector<std::pair<int32_t, bool>> layer_routing_list;
  switch (violation.get_violation_type()) {
    case ViolationType::kAdjacentCutSpacing:
      if (de_func_type == DEFuncType::kGetDEProcessType) {
        de_process_type = DEProcessType::kSkip;
      } else if (de_func_type == DEFuncType::kExpandViolation) {
        RTLOG.error(Loc::current(), "This is skipping!");
      }
      break;
    case ViolationType::kCornerFillSpacing:
      if (de_func_type == DEFuncType::kGetDEProcessType) {
        de_process_type = DEProcessType::kSkip;
      } else if (de_func_type == DEFuncType::kExpandViolation) {
        RTLOG.error(Loc::current(), "This is skipping!");
      }
      break;
    case ViolationType::kCutEOLSpacing:
      if (de_func_type == DEFuncType::kGetDEProcessType) {
        de_process_type = DEProcessType::kCost;
      } else if (de_func_type == DEFuncType::kExpandViolation) {
        real_rect = enlargeRect(real_rect, violation.get_required_size());
        layer_routing_list = expandUpOneLayer(violation);
      }
      break;
    case ViolationType::kCutShort:
      if (de_func_type == DEFuncType::kGetDEProcessType) {
        de_process_type = DEProcessType::kCost;
      } else if (de_func_type == DEFuncType::kExpandViolation) {
        real_rect = enlargeRect(real_rect, violation.get_required_size());
        layer_routing_list = expandUpOneLayer(violation);
      }
      break;
    case ViolationType::kDifferentLayerCutSpacing:
      if (de_func_type == DEFuncType::kGetDEProcessType) {
        de_process_type = DEProcessType::kCost;
      } else if (de_func_type == DEFuncType::kExpandViolation) {
        real_rect = enlargeRect(real_rect, violation.get_required_size());
        layer_routing_list = expandUpTwoLayer(violation);
      }
      break;
    case ViolationType::kEndOfLineSpacing:
      if (de_func_type == DEFuncType::kGetDEProcessType) {
        de_process_type = DEProcessType::kCost;
      } else if (de_func_type == DEFuncType::kExpandViolation) {
        real_rect = enlargeRect(real_rect, violation.get_required_size());
        layer_routing_list = expandAdjacentOneLayer(violation);
      }
      break;
    case ViolationType::kEnclosure:
      if (de_func_type == DEFuncType::kGetDEProcessType) {
        de_process_type = DEProcessType::kSkip;
      } else if (de_func_type == DEFuncType::kExpandViolation) {
        RTLOG.error(Loc::current(), "This is skipping!");
      }
      break;
    case ViolationType::kEnclosureEdge:
      if (de_func_type == DEFuncType::kGetDEProcessType) {
        de_process_type = DEProcessType::kSkip;
      } else if (de_func_type == DEFuncType::kExpandViolation) {
        RTLOG.error(Loc::current(), "This is skipping!");
      }
      break;
    case ViolationType::kEnclosureParallel:
      if (de_func_type == DEFuncType::kGetDEProcessType) {
        de_process_type = DEProcessType::kSkip;
      } else if (de_func_type == DEFuncType::kExpandViolation) {
        RTLOG.error(Loc::current(), "This is skipping!");
      }
      break;
    case ViolationType::kFloatingPatch:
      if (de_func_type == DEFuncType::kGetDEProcessType) {
        de_process_type = DEProcessType::kSkip;
      } else if (de_func_type == DEFuncType::kExpandViolation) {
        RTLOG.error(Loc::current(), "This is skipping!");
      }
      break;
    case ViolationType::kJogToJogSpacing:
      if (de_func_type == DEFuncType::kGetDEProcessType) {
        de_process_type = DEProcessType::kCost;
      } else if (de_func_type == DEFuncType::kExpandViolation) {
        real_rect = enlargeRect(real_rect, violation.get_required_size());
        layer_routing_list = expandAdjacentOneLayer(violation);
      }
      break;
    case ViolationType::kMaxViaStack:
      if (de_func_type == DEFuncType::kGetDEProcessType) {
        de_process_type = DEProcessType::kCost;
      } else if (de_func_type == DEFuncType::kExpandViolation) {
        real_rect = enlargeRect(real_rect, 0);
        layer_routing_list = expandUpOneLayer(violation);
      }
      break;
    case ViolationType::kMetalShort:
      if (de_func_type == DEFuncType::kGetDEProcessType) {
        de_process_type = DEProcessType::kCost;
      } else if (de_func_type == DEFuncType::kExpandViolation) {
        real_rect = enlargeRect(real_rect, violation.get_required_size());
        layer_routing_list = expandAdjacentOneLayer(violation);
      }
      break;
    case ViolationType::kMinHole:
      if (de_func_type == DEFuncType::kGetDEProcessType) {
        de_process_type = DEProcessType::kCost;
      } else if (de_func_type == DEFuncType::kExpandViolation) {
        real_rect = enlargeRect(real_rect, std::ceil(std::sqrt(violation.get_required_size())));
        layer_routing_list = expandAdjacentOneLayer(violation);
      }
      break;
    case ViolationType::kMinimumArea:
      if (de_func_type == DEFuncType::kGetDEProcessType) {
        de_process_type = DEProcessType::kSkip;
      } else if (de_func_type == DEFuncType::kExpandViolation) {
        RTLOG.error(Loc::current(), "This is skipping!");
      }
      break;
    case ViolationType::kMinimumCut:
      if (de_func_type == DEFuncType::kGetDEProcessType) {
        de_process_type = DEProcessType::kSkip;
      } else if (de_func_type == DEFuncType::kExpandViolation) {
        RTLOG.error(Loc::current(), "This is skipping!");
      }
      break;
    case ViolationType::kMinimumWidth:
      if (de_func_type == DEFuncType::kGetDEProcessType) {
        de_process_type = DEProcessType::kSkip;
      } else if (de_func_type == DEFuncType::kExpandViolation) {
        RTLOG.error(Loc::current(), "This is skipping!");
      }
      break;
    case ViolationType::kMinStep:
      if (de_func_type == DEFuncType::kGetDEProcessType) {
        de_process_type = DEProcessType::kSkip;
      } else if (de_func_type == DEFuncType::kExpandViolation) {
        RTLOG.error(Loc::current(), "This is skipping!");
      }
      break;
    case ViolationType::kNonsufficientMetalOverlap:
      if (de_func_type == DEFuncType::kGetDEProcessType) {
        de_process_type = DEProcessType::kSkip;
      } else if (de_func_type == DEFuncType::kExpandViolation) {
        RTLOG.error(Loc::current(), "This is skipping!");
      }
      break;
    case ViolationType::kNotchSpacing:
      if (de_func_type == DEFuncType::kGetDEProcessType) {
        de_process_type = DEProcessType::kCost;
      } else if (de_func_type == DEFuncType::kExpandViolation) {
        real_rect = enlargeRect(real_rect, violation.get_required_size());
        layer_routing_list = expandAdjacentOneLayer(violation);
      }
      break;
    case ViolationType::kOffGridOrWrongWay:
      if (de_func_type == DEFuncType::kGetDEProcessType) {
        de_process_type = DEProcessType::kSkip;
      } else if (de_func_type == DEFuncType::kExpandViolation) {
        RTLOG.error(Loc::current(), "This is skipping!");
      }
      break;
    case ViolationType::kOutOfDie:
      if (de_func_type == DEFuncType::kGetDEProcessType) {
        de_process_type = DEProcessType::kSkip;
      } else if (de_func_type == DEFuncType::kExpandViolation) {
        RTLOG.error(Loc::current(), "This is skipping!");
      }
      break;
    case ViolationType::kParallelRunLengthSpacing:
      if (de_func_type == DEFuncType::kGetDEProcessType) {
        de_process_type = DEProcessType::kCost;
      } else if (de_func_type == DEFuncType::kExpandViolation) {
        real_rect = enlargeRect(real_rect, violation.get_required_size());
        layer_routing_list = expandAdjacentOneLayer(violation);
      }
      break;
    case ViolationType::kSameLayerCutSpacing:
      if (de_func_type == DEFuncType::kGetDEProcessType) {
        de_process_type = DEProcessType::kCost;
      } else if (de_func_type == DEFuncType::kExpandViolation) {
        real_rect = enlargeRect(real_rect, violation.get_required_size());
        layer_routing_list = expandUpOneLayer(violation);
      }
      break;
    default:
      RTLOG.error(Loc::current(), "No violation type!");
      break;
  }
  for (std::pair<int32_t, bool>& layer_routing : layer_routing_list) {
    Violation expanded_violation = violation;
    expanded_violation.get_violation_shape().set_real_rect(real_rect);
    expanded_violation.get_violation_shape().set_layer_idx(layer_routing.first);
    expanded_violation.set_is_routing(layer_routing.second);
    expanded_violation_list.push_back(expanded_violation);
  }
}

PlanarRect DRCEngine::enlargeRect(PlanarRect& real_rect, int32_t required_size)
{
  int32_t enlarged_x_size = 0;
  if (real_rect.getXSpan() < required_size) {
    enlarged_x_size = required_size - real_rect.getXSpan();
  }
  int32_t enlarged_y_size = 0;
  if (real_rect.getYSpan() < required_size) {
    enlarged_y_size = required_size - real_rect.getYSpan();
  }
  return RTUTIL.getEnlargedRect(real_rect, enlarged_x_size, enlarged_y_size, enlarged_x_size, enlarged_y_size);
}

std::vector<std::pair<int32_t, bool>> DRCEngine::expandAdjacentOneLayer(Violation& violation)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  int32_t violation_layer_idx = violation.get_violation_shape().get_layer_idx();

  std::vector<std::pair<int32_t, bool>> layer_routing_list;
  layer_routing_list.emplace_back(violation_layer_idx, true);
  if (0 < violation_layer_idx) {
    layer_routing_list.emplace_back(violation_layer_idx - 1, true);
  }
  if (violation_layer_idx < (static_cast<int32_t>(routing_layer_list.size()) - 1)) {
    layer_routing_list.emplace_back(violation_layer_idx + 1, true);
  }
  return layer_routing_list;
}

std::vector<std::pair<int32_t, bool>> DRCEngine::expandUpOneLayer(Violation& violation)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  int32_t violation_layer_idx = violation.get_violation_shape().get_layer_idx();

  std::vector<std::pair<int32_t, bool>> layer_routing_list;
  layer_routing_list.emplace_back(violation_layer_idx, true);
  if (violation_layer_idx < (static_cast<int32_t>(routing_layer_list.size()) - 1)) {
    layer_routing_list.emplace_back(violation_layer_idx + 1, true);
  }
  return layer_routing_list;
}

std::vector<std::pair<int32_t, bool>> DRCEngine::expandUpTwoLayer(Violation& violation)
{
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();

  int32_t violation_layer_idx = violation.get_violation_shape().get_layer_idx();

  std::vector<std::pair<int32_t, bool>> layer_routing_list;
  layer_routing_list.emplace_back(violation_layer_idx, true);
  if (violation_layer_idx < (static_cast<int32_t>(routing_layer_list.size()) - 1)) {
    layer_routing_list.emplace_back(violation_layer_idx + 1, true);
  }
  if (violation_layer_idx < (static_cast<int32_t>(routing_layer_list.size()) - 2)) {
    layer_routing_list.emplace_back(violation_layer_idx + 2, true);
  }
  return layer_routing_list;
}

#endif

}  // namespace irt
