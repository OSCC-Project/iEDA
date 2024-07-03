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
#include "tcl_plconfig.h"

#include <any>
#include <iomanip>

#include "tcl_util.h"

namespace tcl {

CmdPLConfig::CmdPLConfig(const char* cmd_name) : TclCmd(cmd_name)
{
    // config_json_path string
    _config_list.push_back(std::make_pair("-config_json_path", ValueType::kString));
    // pl_dir string
    _config_list.push_back(std::make_pair("-pl_dir", ValueType::kString));
    // is_timing_effort int
    _config_list.push_back(std::make_pair("-is_timing_effort", ValueType::kInt));
    // is_congestion_effort int
    _config_list.push_back(std::make_pair("-is_congestion_effort", ValueType::kInt));
    // num_threads int
    _config_list.push_back(std::make_pair("-num_threads", ValueType::kInt));
    // target_density double
    _config_list.push_back(std::make_pair("-target_density", ValueType::kDouble));
    // bin_cnt_x int
    _config_list.push_back(std::make_pair("-bin_cnt_x", ValueType::kInt));
    // bin_cnt_y int
    _config_list.push_back(std::make_pair("-bin_cnt_y", ValueType::kInt));
    // target_overflow double
    _config_list.push_back(std::make_pair("-target_overflow", ValueType::kDouble));
    // global_right_padding int
    _config_list.push_back(std::make_pair("-global_right_padding", ValueType::kInt));
    // max_displacement int
    _config_list.push_back(std::make_pair("-max_displacement", ValueType::kInt));
    // global_right_padding int
    _config_list.push_back(std::make_pair("-global_right_padding", ValueType::kInt));
    // enable_networkflow int
    _config_list.push_back(std::make_pair("-enable_networkflow", ValueType::kInt));

  TclUtil::addOption(this, _config_list);
}

unsigned CmdPLConfig::exec()
{
  std::map<std::string, std::any> config_map = TclUtil::getConfigMap(this, _config_list);

  if (config_map.empty()) {
    return 0;
  }
  std::string config_json_path = std::any_cast<std::string>(config_map["-config_json_path"]);
  config_map.erase("-config_json_path");
  TclUtil::alterJsonConfig(config_json_path, config_map);
  return 1;
}

}  // namespace tcl