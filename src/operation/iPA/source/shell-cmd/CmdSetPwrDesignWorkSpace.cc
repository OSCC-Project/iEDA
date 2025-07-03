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
 * @file CmdSetPwrDesignWorkSpace.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The command for set the power design workspace.
 * @version 0.1
 * @date 2021-12-14
 */

#include "PowerShellCmd.hh"
#include "api/Power.hh"
#include "sta/Sta.hh"

namespace ipower {
CmdSetPwrDesignWorkSpace::CmdSetPwrDesignWorkSpace(const char* cmd_name)
    : TclCmd(cmd_name) {
  auto* design_workspace_arg =
      new TclStringOption("design_workspace", 1, nullptr);
  addOption(design_workspace_arg);
}

unsigned CmdSetPwrDesignWorkSpace::check() {
  TclOption* design_workspace_arg = getOptionOrArg("design_workspace");
  LOG_FATAL_IF(!design_workspace_arg);
  return 1;
}

unsigned CmdSetPwrDesignWorkSpace::exec() {
  if (!check()) {
    return 0;
  }

  TclOption* design_workspace_arg = getOptionOrArg("design_workspace");
  auto* design_workspace = design_workspace_arg->getStringVal();

  auto* ista = ista::Sta::getOrCreateSta();
  auto* ipower = ipower::Power::getOrCreatePower(&(ista->get_graph()));
  ipower->set_design_work_space(design_workspace);

  return 1;
}
}  // namespace ipower