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
#pragma once

#include "DEProcessType.hpp"
#include "RTHeader.hpp"
#include "Violation.hpp"

namespace irt {

class DETask
{
 public:
  DETask() = default;
  ~DETask() = default;
  // getter
  std::set<DEProcessType>& get_process_type_set() { return _process_type_set; }
  std::string& get_top_name() { return _top_name; }
  PlanarRect& get_check_region() { return _check_region; }
  std::vector<std::pair<EXTLayerRect*, bool>>& get_env_shape_list() { return _env_shape_list; }
  std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>>& get_net_pin_shape_map() { return _net_pin_shape_map; }
  std::map<int32_t, std::vector<Segment<LayerCoord>*>>& get_net_access_result_map() { return _net_access_result_map; }
  std::map<int32_t, std::vector<EXTLayerRect*>>& get_net_access_patch_map() { return _net_access_patch_map; }
  std::map<int32_t, std::vector<Segment<LayerCoord>>>& get_net_detailed_result_map() { return _net_detailed_result_map; }
  std::map<int32_t, std::vector<EXTLayerRect>>& get_net_detailed_patch_map() { return _net_detailed_patch_map; }
  std::string& get_top_dir_path() { return _top_dir_path; }
  std::string& get_def_file_path() { return _def_file_path; }
  std::string& get_netlist_file_path() { return _netlist_file_path; }
  std::string& get_prepared_file_path() { return _prepared_file_path; }
  std::string& get_finished_file_path() { return _finished_file_path; }
  std::string& get_violation_file_path() { return _violation_file_path; }
  std::vector<Violation>& get_violation_list() { return _violation_list; }
  // setter
  void set_process_type_set(const std::set<DEProcessType>& process_type_set) { _process_type_set = process_type_set; }
  void set_top_name(const std::string& top_name) { _top_name = top_name; }
  void set_check_region(const PlanarRect& check_region) { _check_region = check_region; }
  void set_env_shape_list(const std::vector<std::pair<EXTLayerRect*, bool>>& env_shape_list) { _env_shape_list = env_shape_list; }
  void set_net_pin_shape_map(const std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>>& net_pin_shape_map)
  {
    _net_pin_shape_map = net_pin_shape_map;
  }
  void set_net_access_result_map(const std::map<int32_t, std::vector<Segment<LayerCoord>*>>& net_access_result_map)
  {
    _net_access_result_map = net_access_result_map;
  }
  void set_net_access_patch_map(const std::map<int32_t, std::vector<EXTLayerRect*>>& net_access_patch_map)
  {
    _net_access_patch_map = net_access_patch_map;
  }
  void set_net_detailed_result_map(const std::map<int32_t, std::vector<Segment<LayerCoord>>>& net_detailed_result_map)
  {
    _net_detailed_result_map = net_detailed_result_map;
  }
  void set_net_detailed_patch_map(const std::map<int32_t, std::vector<EXTLayerRect>>& net_detailed_patch_map)
  {
    _net_detailed_patch_map = net_detailed_patch_map;
  }
  void set_top_dir_path(const std::string& top_dir_path) { _top_dir_path = top_dir_path; }
  void set_def_file_path(const std::string& def_file_path) { _def_file_path = def_file_path; }
  void set_netlist_file_path(const std::string& netlist_file_path) { _netlist_file_path = netlist_file_path; }
  void set_prepared_file_path(const std::string& prepared_file_path) { _prepared_file_path = prepared_file_path; }
  void set_finished_file_path(const std::string& finished_file_path) { _finished_file_path = finished_file_path; }
  void set_violation_file_path(const std::string& violation_file_path) { _violation_file_path = violation_file_path; }
  void set_violation_list(const std::vector<Violation>& violation_list) { _violation_list = violation_list; }
  // function
 private:
  std::set<DEProcessType> _process_type_set;
  std::string _top_name;
  PlanarRect _check_region;
  std::vector<std::pair<EXTLayerRect*, bool>> _env_shape_list;
  std::map<int32_t, std::vector<std::pair<EXTLayerRect*, bool>>> _net_pin_shape_map;
  std::map<int32_t, std::vector<Segment<LayerCoord>*>> _net_access_result_map;
  std::map<int32_t, std::vector<EXTLayerRect*>> _net_access_patch_map;
  std::map<int32_t, std::vector<Segment<LayerCoord>>> _net_detailed_result_map;
  std::map<int32_t, std::vector<EXTLayerRect>> _net_detailed_patch_map;
  std::string _top_dir_path;
  std::string _def_file_path;
  std::string _netlist_file_path;
  std::string _prepared_file_path;
  std::string _finished_file_path;
  std::string _violation_file_path;
  std::vector<Violation> _violation_list;
};

}  // namespace irt
