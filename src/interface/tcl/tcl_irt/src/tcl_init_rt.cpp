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
#include "RTInterface.hpp"
#include "tcl_rt.h"
#include "tcl_util.h"

namespace tcl {

// public

TclInitRT::TclInitRT(const char* cmd_name) : TclCmd(cmd_name)
{
  // std::string temp_directory_path;       // required
  _config_list.push_back(std::make_pair("-temp_directory_path", ValueType::kString));
  // int32_t thread_number;                 // optional
  _config_list.push_back(std::make_pair("-thread_number", ValueType::kInt));
  // std::string bottom_routing_layer;      // optional
  _config_list.push_back(std::make_pair("-bottom_routing_layer", ValueType::kString));
  // std::string top_routing_layer;         // optional
  _config_list.push_back(std::make_pair("-top_routing_layer", ValueType::kString));
  // int32_t output_inter_result;           // optional
  _config_list.push_back(std::make_pair("-output_inter_result", ValueType::kInt));
  // int32_t enable_notification;           // optional
  _config_list.push_back(std::make_pair("-enable_notification", ValueType::kInt));
  // int32_t enable_timing;                 // optional
  _config_list.push_back(std::make_pair("-enable_timing", ValueType::kInt));
  // int32_t enable_fast_mode;              // optional
  _config_list.push_back(std::make_pair("-enable_fast_mode", ValueType::kInt));
  // int32_t enable_lsa;                    // optional
  _config_list.push_back(std::make_pair("-enable_lsa", ValueType::kInt));

  TclUtil::addOption(this, _config_list);
}

unsigned TclInitRT::exec()
{
  if (!check()) {
    return 0;
  }
  std::map<std::string, std::any> config_map = TclUtil::getConfigMap(this, _config_list);
  RTI.initRT(config_map);
  return 1;
}

// private

}  // namespace tcl
