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
#include "ipw_io.h"

#include <filesystem>

#include "IdbEnum.h"
#include "TimingEngine.hh"
#include "TimingIDBAdapter.hh"
#include "api/Power.hh"
#include "builder.h"
#include "flow_config.h"
#include "idm.h"
#include "ista_io.h"

namespace iplf {
PowerIO* PowerIO::_instance = nullptr;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @Brief : run sta
 * @param  path path is not a must, if path is empty, using the DB Config output
 * path
 * @return true
 * @return false
 */
bool PowerIO::autoRunPower(std::string path)
{
  flowConfigInst->set_status_stage("iPA - Power Analysis");

  ieda::Stats stats;

  if (!staInst->isInitSTA()) {
    /// init
    staInst->setStaWorkDirectory(path);
    std::vector<std::string> paths;
    staInst->runLiberty(paths);
    staInst->readIdb();
    staInst->runSDC();
  }

  /// run
  reportSummaryPower();

  flowConfigInst->add_status_runtime(stats.elapsedRunTime());
  flowConfigInst->set_status_memmory(stats.memoryDelta());

  return true;
}

/**
 * @brief report power
 *
 * @return true
 * @return false
 */
bool PowerIO::reportSummaryPower()
{
  auto* timing_engine = ista::TimingEngine::getOrCreateTimingEngine();

  if (!timing_engine->isBuildGraph()) {
    timing_engine->buildGraph();
    timing_engine->updateTiming();
  }

  ista::Sta* ista = ista::Sta::getOrCreateSta();
  ipower::Power* ipower = ipower::Power::getOrCreatePower(&(ista->get_graph()));

  ipower->runCompleteFlow();

  return true;
}

}  // namespace iplf
