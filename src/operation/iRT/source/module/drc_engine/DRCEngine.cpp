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

std::vector<Violation> DRCEngine::getViolationList(DETask& de_task)
{
  // return {};
  // return getViolationListBySelf(de_task);
  return getViolationListByOther(de_task);
}

// private

DRCEngine* DRCEngine::_de_instance = nullptr;

std::vector<Violation> DRCEngine::getViolationListBySelf(DETask& de_task)
{
  buildTask(de_task);
  writeTask(de_task);
  readTask(de_task);
  return de_task.get_violation_list();
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

  std::string& top_name = de_task.get_top_name();
  std::vector<std::pair<EXTLayerRect*, bool>>& env_shape_list = de_task.get_env_shape_list();
  std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>>& net_pin_shape_map = de_task.get_net_pin_shape_map();
  std::map<int32_t, std::vector<Segment<LayerCoord>*>>& net_access_result_map = de_task.get_net_access_result_map();
  std::map<int32_t, std::vector<EXTLayerRect*>>& net_access_patch_map = de_task.get_net_access_patch_map();
  std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_detailed_result_map = de_task.get_net_detailed_result_map();
  std::map<int32_t, std::vector<EXTLayerRect>>& net_detailed_patch_map = de_task.get_net_detailed_patch_map();
  std::string& def_file_path = de_task.get_def_file_path();
  std::string& netlist_file_path = de_task.get_netlist_file_path();
  std::string& prepared_file_path = de_task.get_prepared_file_path();

  std::set<int32_t> net_idx_set;
  // 获取所有net
  {
    for (auto& [net_idx, pin_shape_list] : net_pin_shape_map) {
      net_idx_set.insert(net_idx);
    }
    for (auto& [net_idx, segment_list] : net_access_result_map) {
      net_idx_set.insert(net_idx);
    }
    for (auto& [net_idx, patch_list] : net_access_patch_map) {
      net_idx_set.insert(net_idx);
    }
    for (auto& [net_idx, segment_list] : net_detailed_result_map) {
      net_idx_set.insert(net_idx);
    }
    for (auto& [net_idx, patch_list] : net_detailed_patch_map) {
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
    // 构建blockage
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
    // 构建net
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
      if (RTUTIL.exist(net_access_result_map, net_idx)) {
        for (Segment<LayerCoord>* segment : net_access_result_map[net_idx]) {
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
      if (RTUTIL.exist(net_access_patch_map, net_idx)) {
        for (EXTLayerRect* patch : net_access_patch_map[net_idx]) {
          std::string layer_name = routing_layer_list[patch->get_layer_idx()].get_layer_name();
          PlanarRect& real_rect = patch->get_real_rect();
          PlanarCoord mid_point = real_rect.getMidPoint();
          RTUTIL.pushStream(def_file, flag, " ", layer_name, " ( ", mid_point.get_x(), " ", mid_point.get_y(), " ) RECT ( ",
                            real_rect.get_ll_x() - mid_point.get_x(), " ", real_rect.get_ll_y() - mid_point.get_y(), " ",
                            real_rect.get_ur_x() - mid_point.get_x(), " ", real_rect.get_ur_y() - mid_point.get_y(), " )", "\n");
          flag = "    NEW";
        }
      }
      if (RTUTIL.exist(net_detailed_result_map, net_idx)) {
        for (Segment<LayerCoord> segment : net_detailed_result_map[net_idx]) {
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
      if (RTUTIL.exist(net_detailed_patch_map, net_idx)) {
        for (EXTLayerRect& patch : net_detailed_patch_map[net_idx]) {
          std::string layer_name = routing_layer_list[patch.get_layer_idx()].get_layer_name();
          PlanarRect& real_rect = patch.get_real_rect();
          PlanarCoord mid_point = real_rect.getMidPoint();
          RTUTIL.pushStream(def_file, flag, " ", layer_name, " ( ", mid_point.get_x(), " ", mid_point.get_y(), " ) RECT ( ",
                            real_rect.get_ll_x() - mid_point.get_x(), " ", real_rect.get_ll_y() - mid_point.get_y(), " ",
                            real_rect.get_ur_x() - mid_point.get_x(), " ", real_rect.get_ur_y() - mid_point.get_y(), " )", "\n");
          flag = "    NEW";
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
      RTUTIL.pushStream(netlist_file, "wire net_", net_idx, " ;", "\n");
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
  ScaleAxis& gcell_axis = RTDM.getDatabase().get_gcell_axis();
  std::map<std::string, int32_t>& routing_layer_name_to_idx_map = RTDM.getDatabase().get_routing_layer_name_to_idx_map();
  std::map<int32_t, std::vector<int32_t>>& routing_to_adjacent_cut_map = RTDM.getDatabase().get_routing_to_adjacent_cut_map();

  std::map<ViolationType, DEProcessType>& violation_process_map = getViolationProcessMap();

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
    std::string drc_rule_name;
    std::set<std::string> net_name_set;
    std::string layer_name;
    std::string ll_x_string;
    std::string ll_y_string;
    std::string ur_x_string;
    std::string ur_y_string;
    while (std::getline(*violation_file, new_line)) {
      if (new_line.empty()) {
        continue;
      }
      // 读取
      {
        drc_rule_name = new_line;
        std::getline(*violation_file, new_line);
        std::istringstream net_name_set_stream(new_line);
        std::string net_name;
        while (std::getline(net_name_set_stream, net_name, ',')) {
          if (!net_name.empty()) {
            net_name_set.insert(net_name);
          }
        }
        std::getline(*violation_file, new_line);
        std::istringstream layer_bound_stream(new_line);
        layer_bound_stream >> layer_name >> ll_x_string >> ll_y_string >> ur_x_string >> ur_y_string;
      }
      // 解析
      {
        ViolationType violation_type = GetViolationTypeByName()(drc_rule_name);
        if (!RTUTIL.exist(violation_process_map, violation_type)) {
          // 未知规则舍弃
          RTLOG.warn(Loc::current(), "Unknow rule! '", drc_rule_name, "'");
          continue;
        }
        DEProcessType process_type = violation_process_map[violation_type];
        if (!RTUTIL.exist(de_task.get_process_type_set(), process_type)) {
          // 非处理类型舍弃
          continue;
        }
        PlanarRect real_rect(
            static_cast<int32_t>(std::stod(ll_x_string) * micron_dbu), static_cast<int32_t>(std::stod(ll_y_string) * micron_dbu),
            static_cast<int32_t>(std::stod(ur_x_string) * micron_dbu), static_cast<int32_t>(std::stod(ur_y_string) * micron_dbu));
        if (!RTUTIL.isOverlap(de_task.get_check_region(), real_rect)) {
          // 不在检查区域内的舍弃
          continue;
        }
        std::set<int32_t> violation_net_set;
        for (const std::string& net_name : net_name_set) {
          int32_t net_idx = std::stoi(RTUTIL.splitString(net_name, '_').back());
          if (net_idx == -1 || RTUTIL.exist(de_task.get_net_detailed_result_map(), net_idx)) {
            violation_net_set.insert(net_idx);
          }
        }
        if (violation_net_set.empty() || (violation_net_set.size() == 1 && *violation_net_set.begin() == -1)) {
          // net不是布线net的舍弃
          continue;
        }
        std::vector<std::pair<int32_t, bool>> layer_routing_list;
        int32_t routing_layer_idx = routing_layer_name_to_idx_map[layer_name];
        layer_routing_list.emplace_back(routing_layer_idx, true);
        if (process_type == DEProcessType::kCutCost) {
          for (int32_t cut_layer_idx : routing_to_adjacent_cut_map[routing_layer_idx]) {
            layer_routing_list.emplace_back(cut_layer_idx, false);
          }
        }
        if (false) {
          // 因为不可布线层需要打via,对于不在可布线层的违例也有意义
          continue;
        }
        for (std::pair<int32_t, bool> layer_routing : layer_routing_list) {
          EXTLayerRect ext_layer_rect;
          ext_layer_rect.set_real_rect(real_rect);
          ext_layer_rect.set_grid_rect(RTUTIL.getClosedGCellGridRect(ext_layer_rect.get_real_rect(), gcell_axis));
          ext_layer_rect.set_layer_idx(layer_routing.first);

          Violation violation;
          violation.set_violation_type(violation_type);
          violation.set_violation_shape(ext_layer_rect);
          violation.set_is_routing(layer_routing.second);
          violation.set_violation_net_set(violation_net_set);
          de_task.get_violation_list().push_back(violation);
        }
      }
    }
    RTUTIL.closeFileStream(violation_file);
  } else {
    RTLOG.warn(Loc::current(), "The task ", top_name, " violation_file_path is not exist!");
  }
  // 删除文件夹
  RTUTIL.removeDir(de_task.get_top_dir_path());
}

std::map<ViolationType, DEProcessType>& DRCEngine::getViolationProcessMap()
{
  static std::map<ViolationType, DEProcessType> violation_process_map;
  static std::once_flag init_flag;

  std::call_once(init_flag, []() {
    /**
     * skip             暂时无法处理的规则
     * routing_cost     routing层加cost
     * cut_cost         cut层及相邻两routing层加cost
     * routing_patch    routing层加patch
     */
    // skip
    violation_process_map[ViolationType::kFloatingPatch] = DEProcessType::kSkip;      // 由于cell没有加载,所以pin shape属于漂浮
    violation_process_map[ViolationType::kOffGridOrWrongWay] = DEProcessType::kSkip;  // 不在track上的布线结果
    violation_process_map[ViolationType::kMinimumWidth] = DEProcessType::kSkip;  // 最小宽度违例,实际上是Floating Patch的最小宽度
    violation_process_map[ViolationType::kMinimumCut] = DEProcessType::kSkip;  // 对一些时钟树net需要多cut
    violation_process_map[ViolationType::kOutOfDie] = DEProcessType::kSkip;    // shape超出die
    //
    violation_process_map[ViolationType::kEnclosureParallel] = DEProcessType::kSkip;          // enclosure与merge的shepe的spacing
    violation_process_map[ViolationType::kEnclosureEdge] = DEProcessType::kSkip;              // enclosure与merge的shepe的spacing
    violation_process_map[ViolationType::kEnclosure] = DEProcessType::kSkip;                  // enclosure与merge的shepe的spacing
    violation_process_map[ViolationType::kMinHole] = DEProcessType::kSkip;                    // 围起的hole面积太小
    violation_process_map[ViolationType::kNotchSpacing] = DEProcessType::kSkip;               //
    violation_process_map[ViolationType::kCornerFillSpacing] = DEProcessType::kSkip;          //
    violation_process_map[ViolationType::kNonsufficientMetalOverlap] = DEProcessType::kSkip;  // 同net的wire边碰一起
    // routing_cost
    violation_process_map[ViolationType::kMetalShort] = DEProcessType::kRoutingCost;                // 短路,不同一个net
    violation_process_map[ViolationType::kParallelRunLengthSpacing] = DEProcessType::kRoutingCost;  // 平行线spacing
    violation_process_map[ViolationType::kEndOfLineSpacing] = DEProcessType::kRoutingCost;          // EOL spacing
    // cut_cost
    violation_process_map[ViolationType::kCutEOLSpacing] = DEProcessType::kCutCost;             // EOL spacing
    violation_process_map[ViolationType::kCutShort] = DEProcessType::kCutCost;                  // 短路
    violation_process_map[ViolationType::kDifferentLayerCutSpacing] = DEProcessType::kCutCost;  // 不同层的cut spacing问题
    violation_process_map[ViolationType::kSameLayerCutSpacing] = DEProcessType::kCutCost;       // 同层的cut spacing问题
    violation_process_map[ViolationType::kMaxViaStack] = DEProcessType::kCutCost;               // 叠的通孔太多了
    // routing_patch
    violation_process_map[ViolationType::kMinStep] = DEProcessType::kRoutingPatch;      // 金属层min step
    violation_process_map[ViolationType::kMinimumArea] = DEProcessType::kRoutingPatch;  // 金属层面积过小
  });

  return violation_process_map;
}

std::vector<Violation> DRCEngine::getViolationListByOther(DETask& de_task)
{
  std::map<int32_t, std::vector<Segment<LayerCoord>>> net_result_map;
  for (auto& [net_idx, segment_list] : de_task.get_net_access_result_map()) {
    for (Segment<LayerCoord>* segment : segment_list) {
      net_result_map[net_idx].push_back(*segment);
    }
  }
  for (auto& [net_idx, segment_list] : de_task.get_net_detailed_result_map()) {
    for (Segment<LayerCoord>& segment : segment_list) {
      net_result_map[net_idx].push_back(segment);
    }
  }
  return RTI.getViolationList(de_task.get_env_shape_list(), de_task.get_net_pin_shape_map(), net_result_map);
}

}  // namespace irt
