/**
 * @file CmdSetDesignWorkSpace.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The command for set the sta design workspace.
 * @version 0.1
 * @date 2021-12-14
 *
 * @copyright Copyright (c) 2021
 *
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