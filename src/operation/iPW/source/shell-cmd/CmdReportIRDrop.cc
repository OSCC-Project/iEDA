// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file CmdReportIRDrop.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief cmd to report IR drop.
 * @version 0.1
 * @date 2025-03-04
 * 
 */
#include "PowerShellCmd.hh"
#include "api/PowerEngine.hh"

namespace ipower {

CmdReportIRDrop::CmdReportIRDrop(const char* cmd_name) : TclCmd(cmd_name) {
    auto* net_name_option = new TclStringOption("-net_name", 0, nullptr);
    addOption(net_name_option);
}

unsigned CmdReportIRDrop::check() { 
  TclOption* net_name_option = getOptionOrArg("-net_name");
  LOG_FATAL_IF(!net_name_option) << "net name should be specified";
  return 1;
}

unsigned CmdReportIRDrop::exec() {
  if (!check()) {
    return 0;
  }

  TclOption* net_name_option = getOptionOrArg("-net_name");
  auto* power_net_name = net_name_option->getStringVal();
  if (!power_net_name) {
    LOG_FATAL << "net name should be specified";
  }

  PowerEngine* power_engine = PowerEngine::getOrCreatePowerEngine();
  power_engine->runIRAnalysis(power_net_name);
  power_engine->reportIRAnalysis();

  return 1;
}
}