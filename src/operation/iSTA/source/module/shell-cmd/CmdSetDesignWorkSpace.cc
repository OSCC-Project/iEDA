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
 * @file CmdSetDesignWorkSpace.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The command for set the sta design workspace.
 * @version 0.1
 * @date 2021-12-14
 */

#include "ShellCmd.hh"

namespace ista {
CmdSetDesignWorkSpace::CmdSetDesignWorkSpace(const char* cmd_name)
    : TclCmd(cmd_name) {
  auto* design_workspace_arg =
      new TclStringOption("design_workspace", 1, nullptr);
  addOption(design_workspace_arg);
}

unsigned CmdSetDesignWorkSpace::check() {
  TclOption* design_workspace_arg = getOptionOrArg("design_workspace");
  LOG_FATAL_IF(!design_workspace_arg);
  return 1;
}

unsigned CmdSetDesignWorkSpace::exec() {
  if (!check()) {
    return 0;
  }

  TclOption* design_workspace_arg = getOptionOrArg("design_workspace");
  auto* design_workspace = design_workspace_arg->getStringVal();

  Sta* ista = Sta::getOrCreateSta();
  ista->set_design_work_space(design_workspace);

  return 1;
}
}  // namespace ista