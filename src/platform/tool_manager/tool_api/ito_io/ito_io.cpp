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
#include "ito_io.h"

#include "ToApi.hpp"
#include "builder.h"
#include "flow_config.h"
#include "idm.h"

namespace iplf {
ToIO* ToIO::_instance = nullptr;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool ToIO::runTO(std::string config)
{
  if (config.empty()) {
    /// set config path
    config = flowConfigInst->get_ito_path();
  }

  flowConfigInst->set_status_stage("iTO - Timing Optimization");

  ieda::Stats stats;

  /// set data config
  ToApiInst.initTO(config);
  /// reset lib & sdc
  resetConfig();

  ToApiInst.iTODataInit(dmInst->get_idb_builder(), nullptr);
  ToApiInst.runTO();

  flowConfigInst->add_status_runtime(stats.elapsedRunTime());
  flowConfigInst->set_status_memmory(stats.memoryDelta());

  return true;
}

bool ToIO::runTOFixFanout(std::string config)
{
  if (config.empty()) {
    /// set config path
    config = flowConfigInst->get_ito_path();
  }

  flowConfigInst->set_status_stage("iNO - Fix Fanout");

  ieda::Stats stats;

  /// set data config
  ToApiInst.initTO(config);
  /// reset lib & sdc
  resetConfig();

  ToApiInst.iTODataInit(dmInst->get_idb_builder(), nullptr);

  flowConfigInst->add_status_runtime(stats.elapsedRunTime());
  flowConfigInst->set_status_memmory(stats.memoryDelta());

  return true;
}

bool ToIO::runTODrv(std::string config)
{
  if (config.empty()) {
    /// set config path
    config = flowConfigInst->get_ito_path();
  }

  flowConfigInst->set_status_stage("iTO - Fix DRV");

  ieda::Stats stats;

  /// set data config
  ToApiInst.initTO(config);
  /// reset lib & sdc
  resetConfig();

  ToApiInst.iTODataInit(dmInst->get_idb_builder(), nullptr);
  ToApiInst.optimizeDesignViolation();

  flowConfigInst->add_status_runtime(stats.elapsedRunTime());
  flowConfigInst->set_status_memmory(stats.memoryDelta());

  return true;
}

bool ToIO::runTOHold(std::string config)
{
  if (config.empty()) {
    /// set config path
    config = flowConfigInst->get_ito_path();
  }

  flowConfigInst->set_status_stage("iTO - Fix Hold");

  ieda::Stats stats;

  /// set data config
  ToApiInst.initTO(config);
  /// reset lib & sdc
  resetConfig();

  ToApiInst.iTODataInit(dmInst->get_idb_builder(), nullptr);
  ToApiInst.optimizeHold();

  flowConfigInst->add_status_runtime(stats.elapsedRunTime());
  flowConfigInst->set_status_memmory(stats.memoryDelta());

  return true;
}

bool ToIO::runTOSetup(std::string config)
{
  if (config.empty()) {
    /// set config path
    config = flowConfigInst->get_ito_path();
  }

  flowConfigInst->set_status_stage("iTO - Fix Setup");

  ieda::Stats stats;

  /// set data config
  ToApiInst.initTO(config);
  /// reset lib & sdc
  resetConfig();

  ToApiInst.iTODataInit(dmInst->get_idb_builder(), nullptr);
  ToApiInst.optimizeSetup();

  flowConfigInst->add_status_runtime(stats.elapsedRunTime());
  flowConfigInst->set_status_memmory(stats.memoryDelta());

  return true;
}

void ToIO::resetConfig()
{
  idm::DataConfig& db_config = dmInst->get_config();

  if (db_config.get_lib_paths().size() > 0) {
    ToApiInst.resetConfigLibs(db_config.get_lib_paths());
  }

  if (!db_config.get_sdc_path().empty()) {
    ToApiInst.resetConfigSdc(db_config.get_sdc_path());
  }
}

}  // namespace iplf
