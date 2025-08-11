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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "flow_config.h"

#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#include "json_parser.h"
// #include "flow.h"

namespace iplf {
PLFConfig* PLFConfig::_instance = nullptr;
//   std::mutex PLFConfig::_mutex;

bool PLFConfig::initConfig(string path)
{
  _path = path;

  std::ifstream& config_stream = ieda::getInputFileStream(_path);

  {
    nlohmann::json json;
    config_stream >> json;
    /// read tools option
    _tools_config.run_tcl = ieda::getJsonData(json, {"Tools", "TCL"});

    /// read flow
    _flow_config.run_synthesis = ieda::getJsonData(json, {"Flow", "Synthesis"});
    _flow_config.run_floorplan = ieda::getJsonData(json, {"Flow", "Floorplan"});
    _flow_config.run_placer = ieda::getJsonData(json, {"Flow", "Placer"});
    _flow_config.run_cts = ieda::getJsonData(json, {"Flow", "CTS"});
    _flow_config.run_router = ieda::getJsonData(json, {"Flow", "Router"});
    _flow_config.run_drc = ieda::getJsonData(json, {"Flow", "DRC"});
    _flow_config.run_gui = ieda::getJsonData(json, {"Flow", "GUI"});
    _flow_config.run_to = ieda::getJsonData(json, {"Flow", "TO"});

    /// read config path
    _config_path.idb_path = ieda::getJsonData(json, {"ConfigPath", "idb_path"});
    _config_path.ifp_path = ieda::getJsonData(json, {"ConfigPath", "ifp_path"});
    _config_path.ipl_path = ieda::getJsonData(json, {"ConfigPath", "ipl_path"});
    _config_path.icts_path = ieda::getJsonData(json, {"ConfigPath", "icts_path"});
    _config_path.irt_path = ieda::getJsonData(json, {"ConfigPath", "irt_path"});
    _config_path.idrc_path = ieda::getJsonData(json, {"ConfigPath", "idrc_path"});
    _config_path.ito_path = ieda::getJsonData(json, {"ConfigPath", "ito_path"});
    _config_path.ipnp_path = ieda::getJsonData(json, {"ConfigPath", "ipnp_path"});
  }

  ieda::closeFileStream(config_stream);

  return true;
}

}  // namespace iplf
