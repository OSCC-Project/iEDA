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
#include "ino_io.h"

#include "builder.h"
#include "flow_config.h"
#include "iNO/api/NoApi.hpp"
#include "idm.h"
#include "usage/usage.hh"

namespace iplf {
NoIO* NoIO::_instance = nullptr;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool NoIO::runNOFixIO(std::string config)
{
  // if (config.empty()) {
  //   /// set config path
  //   config = flowConfigInst->get_ito_path();
  // }

  flowConfigInst->set_status_stage("iNO - FixIO");

  ieda::Stats stats;

  /// set data config
  NoApiInst.initNO(config);
  /// reset lib & sdc
  resetConfig(NoApiInst.get_no_config());

  NoApiInst.iNODataInit(dmInst->get_idb_builder(), nullptr);
  NoApiInst.fixIO();

  flowConfigInst->add_status_runtime(stats.elapsedRunTime());
  flowConfigInst->set_status_memmory(stats.memoryDelta());

  return true;
}

bool NoIO::runNOFixFanout(std::string config)
{
  // if (config.empty()) {
  //   /// set config path
  //   config = flowConfigInst->get_ito_path();
  // }

  flowConfigInst->set_status_stage("iNO - FixFanout");

  ieda::Stats stats;

  /// set data config
  NoApiInst.initNO(config);
  /// reset lib & sdc
  resetConfig(NoApiInst.get_no_config());

  NoApiInst.iNODataInit(dmInst->get_idb_builder(), nullptr);
  NoApiInst.fixFanout();

  flowConfigInst->add_status_runtime(stats.elapsedRunTime());
  flowConfigInst->set_status_memmory(stats.memoryDelta());

  return true;
}

void NoIO::resetConfig(ino::NoConfig* no_config)
{
  if (no_config == nullptr) {
    return;
  }

  idm::DataConfig& db_config = dmInst->get_config();

  if (db_config.get_lib_paths().size() > 0) {
    no_config->set_lib_files(db_config.get_lib_paths());
  }

  if (!db_config.get_sdc_path().empty()) {
    no_config->set_sdc_file(db_config.get_sdc_path());
  }
}

}  // namespace iplf
