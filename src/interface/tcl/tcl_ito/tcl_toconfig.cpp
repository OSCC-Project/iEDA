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
#include <any>
#include <iomanip>

#include "tcl_toconfig.h"
#include "tcl_util.h"

namespace tcl {

CmdTOConfig::CmdTOConfig(const char* cmd_name) : TclCmd(cmd_name)
{
    // config_json_path string      required
    _config_list.push_back(std::make_pair("-config_json_path", ValueType::kString));
    // setup_target_slack double
    _config_list.push_back(std::make_pair("-setup_target_slack", ValueType::kDouble));
    // hold_target_slack double
    _config_list.push_back(std::make_pair("-hold_target_slack", ValueType::kDouble));
    // max_insert_instance_percent double
    _config_list.push_back(std::make_pair("-max_insert_instance_percent", ValueType::kDouble));
    // max_core_utilization double
    _config_list.push_back(std::make_pair("-max_core_utilization", ValueType::kDouble));
    // routing_tree string
    _config_list.push_back(std::make_pair("-routing_tree", ValueType::kString));
    // fix_fanout bool

    //optimize_drv bool

    // optimize_hold bool

    // optimize_setup bool

    // drv_insert_buffers stringlist
    _config_list.push_back(std::make_pair("-drv_insert_buffers", ValueType::kStringList));
    // hold_insert_buffers stringlist
    _config_list.push_back(std::make_pair("-hold_insert_buffers", ValueType::kStringList));
    // setup_insert_buffers stringlist
    _config_list.push_back(std::make_pair("-setup_insert_buffers", ValueType::kStringList));
    // number_of_decreasing_slack_iter int
    _config_list.push_back(std::make_pair("-number_of_decreasing_slack_iter", ValueType::kInt));
    // max_allowed_buffering_fanout int
    _config_list.push_back(std::make_pair("-max_allowed_buffering_fanout", ValueType::kInt));
    // min_divide_fanout int
    _config_list.push_back(std::make_pair("-min_divide_fanoutt", ValueType::kInt));
    // optimize_endpoints_percent double
    _config_list.push_back(std::make_pair("-optimize_endpoints_percent", ValueType::kDouble));
    // drv_optimize_iter_number int
    _config_list.push_back(std::make_pair("-drv_optimize_iter_number", ValueType::kInt));

    TclUtil::addOption(this, _config_list);
}

unsigned CmdTOConfig::exec()
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