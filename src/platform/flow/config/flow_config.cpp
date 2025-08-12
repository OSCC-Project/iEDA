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
