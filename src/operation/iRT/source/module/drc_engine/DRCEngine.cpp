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

std::vector<Violation> DRCEngine::getViolationList(std::string top_name, std::vector<std::pair<EXTLayerRect*, bool>>& env_shape_list,
                                                   std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>>& net_pin_shape_map,
                                                   std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_fixed_result_map,
                                                   std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_routing_result_map,
                                                   std::string stage)
{
  // return getViolationListBySelf(top_name, env_shape_list, net_pin_shape_map, net_fixed_result_map, net_routing_result_map, stage);
  return getViolationListByOther(top_name, env_shape_list, net_pin_shape_map, net_fixed_result_map, net_routing_result_map, stage);
}

// private

DRCEngine* DRCEngine::_de_instance = nullptr;

std::vector<Violation> DRCEngine::getViolationListBySelf(std::string top_name, std::vector<std::pair<EXTLayerRect*, bool>>& env_shape_list,
                                                         std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>>& net_pin_shape_map,
                                                         std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_fixed_result_map,
                                                         std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_routing_result_map,
                                                         std::string stage)
{
  int32_t micron_dbu = RTDM.getDatabase().get_micron_dbu();
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  Die& die = RTDM.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  std::map<std::string, int32_t>& routing_layer_name_to_idx_map = RTDM.getDatabase().get_routing_layer_name_to_idx_map();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();
  std::string& de_temp_directory_path = RTDM.getConfig().de_temp_directory_path;

  // 定义路径
  std::string top_dir_path = RTUTIL.getString(de_temp_directory_path, top_name);
  std::string def_file_path = RTUTIL.getString(top_dir_path, "/clean.def");
  std::string netlist_file_path = RTUTIL.getString(top_dir_path, "/clean.v");
  std::string prepared_file_path = RTUTIL.getString(top_dir_path, "/prepared");
  std::string finished_file_path = RTUTIL.getString(top_dir_path, "/finished");
  std::string violation_file_path = RTUTIL.getString(top_dir_path, "/drc.txt");

  // 获取所有net
  std::set<int> net_idx_set;
  for (auto& [net_idx, pin_shape_list] : net_pin_shape_map) {
    net_idx_set.insert(net_idx);
  }
  for (auto& [net_idx, segment_list] : net_routing_result_map) {
    net_idx_set.insert(net_idx);
  }
  // 删除再构建top文件夹
  {
    RTUTIL.removeDir(top_dir_path);
    RTUTIL.createDir(top_dir_path);
    // 修改文件夹权限
    std::filesystem::perms permissions
        = std::filesystem::perms::owner_all | std::filesystem::perms::group_all | std::filesystem::perms::others_all;
    RTUTIL.changePermissions(top_dir_path, permissions);
  }
  // 构建def
  {
    std::ofstream* def_file = RTUTIL.getOutputFileStream(def_file_path);
    // 构建header
    {
      RTUTIL.pushStream(def_file, "VERSION 5.8 ;", "\n");
      RTUTIL.pushStream(def_file, "DESIGN ", top_name, " ;", "\n");
      RTUTIL.pushStream(def_file, "UNITS DISTANCE MICRONS ", micron_dbu, " ;", "\n");
      RTUTIL.pushStream(def_file, "DIEAREA ( ", die.get_real_ll_x(), " ", die.get_real_ll_y(), " ) ( ", die.get_real_ur_x(), " ",
                        die.get_real_ur_y(), " ) ;", "\n");
      RTUTIL.pushStream(def_file, "\n");
    }
    // 构建track
    {
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
    }
    // 构建blockage
    {
      RTUTIL.pushStream(def_file, "BLOCKAGES ", env_shape_list.size(), " ;", "\n");
      for (auto& [env_shape, is_routing] : env_shape_list) {
        std::string layer_name;
        if (is_routing) {
          layer_name = routing_layer_list[env_shape->get_layer_idx()].get_layer_name();
        } else {
          layer_name = cut_layer_list[env_shape->get_layer_idx()].get_layer_name();
        }
        RTUTIL.pushStream(def_file, "   - LAYER ", layer_name, " RECT ( ", env_shape->get_real_ll_x(), " ", env_shape->get_real_ll_y(),
                          " ) ( ", env_shape->get_real_ur_x(), " ", env_shape->get_real_ur_y(), " ) ;", "\n");
      }
      RTUTIL.pushStream(def_file, "END BLOCKAGES", "\n");
      RTUTIL.pushStream(def_file, "\n");
    }
    // 构建net
    {
      RTUTIL.pushStream(def_file, "NETS ", net_idx_set.size(), " ;", "\n");
      for (int32_t net_idx : net_idx_set) {
        std::string flag = "  + ROUTED";

        RTUTIL.pushStream(def_file, "- net_", net_idx, "\n");
        if (RTUTIL.exist(net_pin_shape_map, net_idx)) {
          for (auto& [pin_shape, is_routing] : net_pin_shape_map[net_idx]) {
            std::string layer_name;
            if (is_routing) {
              layer_name = routing_layer_list[pin_shape->get_layer_idx()].get_layer_name();
            } else {
              layer_name = cut_layer_list[pin_shape->get_layer_idx()].get_layer_name();
            }
            PlanarRect& real_rect = pin_shape->get_real_rect();
            PlanarCoord mid_point = real_rect.getMidPoint();
            RTUTIL.pushStream(def_file, flag, " ", layer_name, " ( ", mid_point.get_x(), " ", mid_point.get_y(), " ) RECT ( ",
                              real_rect.get_ll_x() - mid_point.get_x(), " ", real_rect.get_ll_y() - mid_point.get_y(), " ",
                              real_rect.get_ur_x() - mid_point.get_x(), " ", real_rect.get_ur_y() - mid_point.get_y(), " )", "\n");
            flag = "    NEW";
          }
        }
        if (RTUTIL.exist(net_fixed_result_map, net_idx)) {
          for (Segment<LayerCoord> segment : net_fixed_result_map[net_idx]) {
            LayerCoord& first_coord = segment.get_first();
            LayerCoord& second_coord = segment.get_second();
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
              if (RTUTIL.isHorizontal(first_coord, second_coord)) {
                std::string layer_name = routing_layer_list[first_layer_idx].get_layer_name();
                RTUTIL.pushStream(def_file, flag, " ", layer_name, " ( ", first_coord.get_x(), " ", first_coord.get_y(), " ) ( ",
                                  second_coord.get_x(), " * )", "\n");
                flag = "    NEW";
              } else if (RTUTIL.isVertical(first_coord, second_coord)) {
                std::string layer_name = routing_layer_list[first_layer_idx].get_layer_name();
                RTUTIL.pushStream(def_file, flag, " ", layer_name, " ( ", first_coord.get_x(), " ", first_coord.get_y(), " ) ( * ",
                                  second_coord.get_y(), " )", "\n");
                flag = "    NEW";
              }
            }
          }
        }
        if (RTUTIL.exist(net_routing_result_map, net_idx)) {
          for (Segment<LayerCoord> segment : net_routing_result_map[net_idx]) {
            LayerCoord& first_coord = segment.get_first();
            LayerCoord& second_coord = segment.get_second();
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
              if (RTUTIL.isHorizontal(first_coord, second_coord)) {
                std::string layer_name = routing_layer_list[first_layer_idx].get_layer_name();
                RTUTIL.pushStream(def_file, flag, " ", layer_name, " ( ", first_coord.get_x(), " ", first_coord.get_y(), " ) ( ",
                                  second_coord.get_x(), " * )", "\n");
                flag = "    NEW";
              } else if (RTUTIL.isVertical(first_coord, second_coord)) {
                std::string layer_name = routing_layer_list[first_layer_idx].get_layer_name();
                RTUTIL.pushStream(def_file, flag, " ", layer_name, " ( ", first_coord.get_x(), " ", first_coord.get_y(), " ) ( * ",
                                  second_coord.get_y(), " )", "\n");
                flag = "    NEW";
              }
            }
          }
        }
        RTUTIL.pushStream(def_file, " ;\n");
      }
      RTUTIL.pushStream(def_file, "END NETS", "\n");
      RTUTIL.pushStream(def_file, "\n");
    }
    // 构建footer
    {
      RTUTIL.pushStream(def_file, "END DESIGN", "\n");
      RTUTIL.pushStream(def_file, "\n");
    }
    RTUTIL.closeFileStream(def_file);
  }
  // 构建netlist
  {
    std::ofstream* netlist_file = RTUTIL.getOutputFileStream(netlist_file_path);
    // 构建header
    {
      RTUTIL.pushStream(netlist_file, "module ", top_name, " ();", "\n");
      RTUTIL.pushStream(netlist_file, "\n");
    }
    // 构建net
    {
      for (int32_t net_idx : net_idx_set) {
        RTUTIL.pushStream(netlist_file, "wire net_", net_idx, " ;", "\n");
      }
      RTUTIL.pushStream(netlist_file, "\n");
    }
    // 构建footer
    {
      RTUTIL.pushStream(netlist_file, "endmodule", "\n");
      RTUTIL.pushStream(netlist_file, "\n");
    }
    RTUTIL.closeFileStream(netlist_file);
  }
  // 构建任务状态文件
  {
    std::ofstream* prepared_file = RTUTIL.getOutputFileStream(prepared_file_path);
    RTUTIL.pushStream(prepared_file, " ");
    RTUTIL.closeFileStream(prepared_file);
  }
  // 等待直到任务结束
  {
    int waiting_time = 0;
    while (!RTUTIL.existFile(finished_file_path)) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      waiting_time++;
      if (waiting_time % 500 == 0) {
        RTLOG.warn(Loc::current(), "The task ", top_name, " waited for ", waiting_time, " seconds");
      }
    }
  }
  // 从中得到违例信息
  std::vector<Violation> voilation_list;
  if (RTUTIL.existFile(violation_file_path)) {
    std::regex geometric_regex(R"(^(.+?): \( (.+?) \) (.+?)  \( (.+?) \)$)");
    std::regex single_net_regex(R"(^Regular Wire of Net ([^&\n]+)$)");
    std::regex double_net_regex(R"(^Regular Wire of Net (.+?) & Regular Wire of Net (.+?)$)");
    std::regex net_env_regex(R"(^Regular Wire of Net (.+?) & Routing Blockage$)");
    std::regex bounds_regex(R"(^Bounds : \( (.+?), (.+?) \) \( (.+?), (.+?) \)$)");

    std::ifstream* violation_file = RTUTIL.getInputFileStream(violation_file_path);

    std::string new_line;
    while (getline(*violation_file, new_line)) {
      std::smatch geometric_match;
      if (std::regex_search(new_line, geometric_match, geometric_regex)) {
        std::string drc_type;
        std::set<std::string> net_name_set;
        std::string layer_name;
        std::string ll_x_string;
        std::string ll_y_string;
        std::string ur_x_string;
        std::string ur_y_string;
        // 读取
        {
          drc_type = geometric_match[2];
          std::string net_line = geometric_match[3];
          std::smatch net_match;
          if (std::regex_search(net_line, net_match, single_net_regex)) {
            net_name_set.insert(net_match[1]);
          } else if (std::regex_search(net_line, net_match, double_net_regex)) {
            net_name_set.insert(net_match[1]);
            net_name_set.insert(net_match[2]);
          } else if (std::regex_search(net_line, net_match, net_env_regex)) {
            net_name_set.insert("net_-1");
            net_name_set.insert(net_match[1]);
          } else {
            RTLOG.error(Loc::current(), "The net regex did not match!");
          }
          layer_name = geometric_match[4];
          if (getline(*violation_file, new_line)) {
            std::smatch bounds_match;
            if (std::regex_search(new_line, bounds_match, bounds_regex)) {
              ll_x_string = bounds_match[1];
              ll_y_string = bounds_match[2];
              ur_x_string = bounds_match[3];
              ur_y_string = bounds_match[4];
            } else {
              RTLOG.error(Loc::current(), "The bounds regex did not match!");
            }
          } else {
            RTLOG.error(Loc::current(), "Failed to read the next line for bounds!");
          }
        }
        // 解析
        {
          std::map<std::string, std::string> rule_layer_map;
          // skip
          rule_layer_map["Floating Patch"] = "skip";         // 由于cell没有加载,所以pin shape属于漂浮
          rule_layer_map["Off Grid or Wrong Way"] = "skip";  // 不在track上的布线结果
          rule_layer_map["Minimum Width"] = "skip";          // 最小宽度违例,实际上是Floating Patch的最小宽度
          rule_layer_map["MinStep"] = "skip";                // 金属层min step
          rule_layer_map["Minimum Area"] = "skip";           // 金属层面积过小
          rule_layer_map["Minimum Cut"] = "skip";            // 对一些时钟树net需要多cut
          rule_layer_map["Enclosure Parallel"] = "skip";     // enclosure与merge的shepe的spacing
          rule_layer_map["EnclosureEdge"] = "skip";          // enclosure与merge的shepe的spacing
          rule_layer_map["Enclosure"] = "skip";              // enclosure与merge的shepe的spacing
          // metal 表示本层违例
          rule_layer_map["Metal Short"] = "metal";                   // 短路,不同一个net
          rule_layer_map["Non-sufficient Metal Overlap"] = "metal";  // 同net的wire边碰一起
          rule_layer_map["ParallelRunLength Spacing"] = "metal";     // 平行线spacing
          rule_layer_map["EndOfLine Spacing"] = "metal";             // EOL spacing
          // cut 以below的metal层举例
          rule_layer_map["Cut EolSpacing"] = "cut";               // EOL spacing
          rule_layer_map["Cut Short"] = "cut";                    // 短路
          rule_layer_map["Different Layer Cut Spacing"] = "cut";  // 不同层的cut spacing问题
          rule_layer_map["Same Layer Cut Spacing"] = "cut";       // 同层的cut spacing问题
          rule_layer_map["MaxViaStack"] = "cut";                  // 叠的通孔太多了

          if (!RTUTIL.exist(rule_layer_map, drc_type)) {
            RTLOG.warn(Loc::current(), "Unknow rule! '", drc_type, "'");
            drc_type = "Metal Short";
          }
          std::string layer_type = rule_layer_map[drc_type];
          if (layer_type == "skip") {
            continue;
          }
          int32_t layer_idx = -1;
          bool is_routing = true;
          if (layer_type == "metal") {
            layer_idx = routing_layer_name_to_idx_map[layer_name];
            is_routing = true;
          } else if (layer_type == "cut") {
            int32_t below_layer_idx = routing_layer_name_to_idx_map[layer_name];
            layer_idx = layer_via_master_list[below_layer_idx].front().get_cut_layer_idx();
            is_routing = false;
          } else {
            RTLOG.error(Loc::current(), "Unknow layer type!");
          }
          EXTLayerRect ext_layer_rect;
          ext_layer_rect.set_real_ll_x(static_cast<int32_t>(std::stod(ll_x_string) * micron_dbu));
          ext_layer_rect.set_real_ll_y(static_cast<int32_t>(std::stod(ll_y_string) * micron_dbu));
          ext_layer_rect.set_real_ur_x(static_cast<int32_t>(std::stod(ur_x_string) * micron_dbu));
          ext_layer_rect.set_real_ur_y(static_cast<int32_t>(std::stod(ur_y_string) * micron_dbu));
          ext_layer_rect.set_grid_rect(RTUTIL.getClosedGCellGridRect(ext_layer_rect.get_real_rect(), gcell_axis));
          ext_layer_rect.set_layer_idx(layer_idx);
          std::set<int32_t> violation_net_set;
          for (const std::string& net_name : net_name_set) {
            violation_net_set.insert(std::stoi(RTUTIL.splitString(net_name, '_').back()));
          }
          Violation violation;
          violation.set_violation_shape(ext_layer_rect);
          violation.set_is_routing(is_routing);
          violation.set_violation_net_set(violation_net_set);
          voilation_list.push_back(violation);
        }
      }
    }
    RTUTIL.closeFileStream(violation_file);
  } else {
    RTLOG.warn(Loc::current(), "The task ", top_name, " violation_file_path is not exist!");
  }
  // 删除文件夹
  {
    RTUTIL.removeDir(top_dir_path);
  }
  return voilation_list;
}

std::vector<Violation> DRCEngine::getViolationListByOther(std::string top_name, std::vector<std::pair<EXTLayerRect*, bool>>& env_shape_list,
                                                          std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>>& net_pin_shape_map,
                                                          std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_fixed_result_map,
                                                          std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_routing_result_map,
                                                          std::string stage)
{
  std::map<int32_t, std::vector<Segment<LayerCoord>>> net_result_map;
  for (auto& [net_idx, segment_list] : net_fixed_result_map) {
    for (Segment<LayerCoord>& segment : segment_list) {
      net_result_map[net_idx].push_back(segment);
    }
  }
  for (auto& [net_idx, segment_list] : net_routing_result_map) {
    for (Segment<LayerCoord>& segment : segment_list) {
      net_result_map[net_idx].push_back(segment);
    }
  }
  return RTI.getViolationList(env_shape_list, net_pin_shape_map, net_result_map, stage);
}

}  // namespace irt
