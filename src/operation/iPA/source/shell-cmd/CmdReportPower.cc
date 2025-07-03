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
 * @file CmdReportPower.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief cmd to report power.
 * @version 0.1
 * @date 2023-05-04
 */

#include "PowerShellCmd.hh"
#include "sta/Sta.hh"

namespace ipower {

CmdReportPower::CmdReportPower(const char* cmd_name) : TclCmd(cmd_name) {
  auto* default_toggle = new TclDoubleOption("-toggle", 0, 0.02);
  addOption(default_toggle);

  auto* enable_json_output = new TclSwitchOption("-json");
  addOption(enable_json_output);
}

unsigned CmdReportPower::check() { return 1; }

unsigned CmdReportPower::exec() {
  if (!check()) {
    return 0;
  }

  Sta* ista = Sta::getOrCreateSta();
  Power* ipower = Power::getOrCreatePower(&(ista->get_graph()));

  auto* default_toggle_option = getOptionOrArg("-toggle");
  double default_toggle = default_toggle_option->getDoubleVal();

  auto* enable_json_output_option = getOptionOrArg("-json");
  if (enable_json_output_option->is_set_val()) {
    ipower->enableJsonReport();
  }

  ipower->set_default_toggle(default_toggle);

  ipower->runCompleteFlow();

  return 1;
}

}  // namespace ipower