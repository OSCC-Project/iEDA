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
/**
 * @file FastPlacer.cpp
 * @author Jianrong Su
 * @brief
 * @version 1.0
 * @date 2025-06-23
 */

#include "FastPlacer.hh"
#include "PLAPI.hh"
#include "log/Log.hh"
#include "PNPConfig.hh"

namespace ipnp {

void FastPlacer::runFastPlacer(idb::IdbBuilder* idb_builder)
{
  std::string pl_json_file;
  
  PNPConfig* temp_config = new PNPConfig();
  if (!temp_config->get_pl_default_config_path().empty()) {
    pl_json_file = temp_config->get_pl_default_config_path();
  }
  else {
    pl_json_file = "../src/operation/iPNP/example/pl_default_config.json";
  }
  delete temp_config;
  
  ipl::PLAPI& plapi = ipl::PLAPI::getInst();

  plapi.initAPI(pl_json_file, idb_builder);
  plapi.runGP();
  plapi.runLG();
  plapi.writeBackSourceDataBase();
  // plapi.runFlow();

  plapi.destoryInst();
}


}  // namespace ipnp
