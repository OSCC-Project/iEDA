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
#include "ipnp_io.h"

#include "iPNPApi.hh"
#include "iPNP.hh"  
#include "flow_config.h"
#include "idm.h"
#include "usage/usage.hh"

#include <iostream>

namespace iplf {
PnpIO* PnpIO::_instance = nullptr;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool PnpIO::runPNP(std::string config)
{
  if (config.empty()) {
    config = "../src/operation/iPNP/example/pnp_config.json";
    std::cout << "Using default iPNP config: " << config << std::endl;
  }

  flowConfigInst->set_status_stage("iPNP - Power Network Planning");

  ieda::Stats stats;

  try {
    ipnp::iPNP ipnp_tool(config);

    ipnp::iPNPApi::setInstance(&ipnp_tool);

    std::cout << "Running iPNP with config: " << config << std::endl;

    ipnp_tool.run();

    std::cout << "iPNP completed successfully" << std::endl;

    flowConfigInst->add_status_runtime(stats.elapsedRunTime());
    flowConfigInst->set_status_memmory(stats.memoryDelta());

    return true;

  } catch (const std::exception& e) {
    std::cerr << "iPNP execution failed: " << e.what() << std::endl;
    flowConfigInst->add_status_runtime(stats.elapsedRunTime());
    flowConfigInst->set_status_memmory(stats.memoryDelta());
    return false;
  }
}

}  // namespace iplf
