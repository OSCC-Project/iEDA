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

#include "tcl_noconfig.h"
#include "tcl_util.h"

namespace tcl {

CmdNOConfig::CmdNOConfig(const char* cmd_name) : TclCmd(cmd_name)
{
    // config_json_path string      required
    _config_list.push_back(std::make_pair("-config_json_path", ValueType::kString));
    // max_fanout int 
    _config_list.push_back(std::make_pair("-max_fanout", ValueType::kInt));
    // insert_buffer string
    _config_list.push_back(std::make_pair("-insert_buffer", ValueType::kString));

    TclUtil::addOption(this, _config_list);
}

unsigned CmdNOConfig::exec()
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