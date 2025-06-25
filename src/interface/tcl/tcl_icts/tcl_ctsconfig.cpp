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

#include "tcl_ctsconfig.h"
#include "tcl_util.h"

namespace tcl {

CmdCTSConfig::CmdCTSConfig(const char* cmd_name) : TclCmd(cmd_name)
{
    // config_json_path string      required
    _config_list.push_back(std::make_pair("-config_json_path", ValueType::kString));
    // skew_bound
    _config_list.push_back(std::make_pair("-skew_bound", ValueType::kDouble));
    // max_buf_tran
    _config_list.push_back(std::make_pair("-max_buf_tran", ValueType::kDouble));
    // max_sink_tran
    _config_list.push_back(std::make_pair("-max_sink_tran", ValueType::kDouble));
    // max_cap
    _config_list.push_back(std::make_pair("-max_cap", ValueType::kDouble));
    // max_fanout
    _config_list.push_back(std::make_pair("-max_fanout", ValueType::kInt));
    // max_length
    _config_list.push_back(std::make_pair("-max_length", ValueType::kInt));
    // min_length
    _config_list.push_back(std::make_pair("-min_length", ValueType::kInt));
    // scale_size int
    _config_list.push_back(std::make_pair("-scale_size", ValueType::kInt));
    // cluster_size int
    _config_list.push_back(std::make_pair("-cluster_size", ValueType::kInt));
    // routing_layer intlist
    _config_list.push_back(std::make_pair("-routing_layer", ValueType::kIntList));
    // buffer_type stringlist
    _config_list.push_back(std::make_pair("-buffer_type", ValueType::kStringList));
    // root_buffer_type string
    _config_list.push_back(std::make_pair("-root_buffer_type", ValueType::kString));
    // root_buffer_required string
    _config_list.push_back(std::make_pair("-root_buffer_required", ValueType::kString));
    // inherit_root string
    _config_list.push_back(std::make_pair("-inherit_root", ValueType::kString));
    // break_long_wire string
    _config_list.push_back(std::make_pair("-break_long_wire", ValueType::kString));
    // level_max_length intlist
    _config_list.push_back(std::make_pair("-level_max_length", ValueType::kIntList));
    // level_max_fanout intlist
    _config_list.push_back(std::make_pair("-level_max_fanout", ValueType::kIntList));
    // level_max_cap doublelist
    _config_list.push_back(std::make_pair("-level_max_cap", ValueType::kDoubleList));
    // level_skew_bound doublelist
    _config_list.push_back(std::make_pair("-level_skew_bound", ValueType::kDoubleList));
    // level_cluster_ratio doublelist
    _config_list.push_back(std::make_pair("-level_cluster_ratio", ValueType::kDoubleList));
    // shift_level int
    _config_list.push_back(std::make_pair("-shift_level", ValueType::kInt));
    // latency_opt_level int
    _config_list.push_back(std::make_pair("-latency_opt_level", ValueType::kInt));
    // global_latency_opt_ratio double
    _config_list.push_back(std::make_pair("-global_latency_opt_ratio", ValueType::kDouble));
    // local_latency_opt_ratio double
    _config_list.push_back(std::make_pair("-local_latency_opt_ratio", ValueType::kDouble));
    // external_model string
    _config_list.push_back(std::make_pair("-external_model", ValueType::kString));
    // use_netlist string
    _config_list.push_back(std::make_pair("-use_netlist", ValueType::kString));

    TclUtil::addOption(this, _config_list);
}

unsigned CmdCTSConfig::exec()
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