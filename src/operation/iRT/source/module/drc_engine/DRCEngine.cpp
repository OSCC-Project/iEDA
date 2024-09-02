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
                                                   std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_result_map, std::string stage)
{
  return RTI.getViolationList(env_shape_list, net_pin_shape_map, net_result_map, stage);
  // return getViolationListBySelf(top_name, env_shape_list, net_pin_shape_map, net_result_map, stage);
}

// private

DRCEngine* DRCEngine::_de_instance = nullptr;

std::vector<Violation> DRCEngine::getViolationListBySelf(std::string top_name, std::vector<std::pair<EXTLayerRect*, bool>>& env_shape_list,
                                                         std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>>& net_pin_shape_map,
                                                         std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_result_map,
                                                         std::string stage)
{
  int32_t micron_dbu = RTDM.getDatabase().get_micron_dbu();
  Die& die = RTDM.getDatabase().get_die();
  std::vector<RoutingLayer>& routing_layer_list = RTDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = RTDM.getDatabase().get_cut_layer_list();
  std::vector<std::vector<ViaMaster>>& layer_via_master_list = RTDM.getDatabase().get_layer_via_master_list();
  std::string& de_temp_directory_path = RTDM.getConfig().de_temp_directory_path;

  std::string top_dir_path = RTUTIL.getString(de_temp_directory_path, top_name);
  std::string def_file_path = RTUTIL.getString(top_dir_path, "/clean.def");
  std::string netlist_file_path = RTUTIL.getString(top_dir_path, "/clean.v");

  // 获取所有net
  std::set<int> net_idx_set;
  for (auto& [net_idx, pin_shape_list] : net_pin_shape_map) {
    net_idx_set.insert(net_idx);
  }
  for (auto& [net_idx, segment_list] : net_result_map) {
    net_idx_set.insert(net_idx);
  }

  // 构建top文件夹
  {
    RTUTIL.createDir(top_dir_path);
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
        if (RTUTIL.exist(net_result_map, net_idx)) {
          for (Segment<LayerCoord> segment : net_result_map[net_idx]) {
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
#if 1
  // 定义局部静态变量
  static std::mutex mtx;
  static std::condition_variable cv;
  static int32_t active_threads = 0;
  static const int32_t max_threads = 5;

  // 获取锁并等待，直到 active_threads < max_threads
  {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [] { return active_threads < max_threads; });
    ++active_threads;
  }
#endif

  // 执行命令
  std::string command
      = RTUTIL.getString("cd /home/zengzhisheng/debug_workspace/drc_engine && ./run_case.sh ", de_temp_directory_path, " ", top_name);
  int32_t command_result = std::system(command.c_str());
  if (command_result != 0) {
    RTLOG.error(Loc::current(), "Unable to execute '", command, "' return '", command_result, "'");
  }

#if 1
  // 释放锁并减少 active_threads 计数
  {
    std::lock_guard<std::mutex> lock(mtx);
    --active_threads;
    cv.notify_one();
  }
#endif

  std::vector<Violation> voilation_list;
  return voilation_list;
}

}  // namespace irt
