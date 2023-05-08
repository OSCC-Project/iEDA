/**
 * @file CmdLinkDesign.cc
 * @author Wang Hao (harry0789@qq.com)
 * @brief
 * @version 0.1
 * @date 2021-10-12
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "ShellCmd.hh"
#include "sta/Sta.hh"

namespace ista {
CmdLinkDesign::CmdLinkDesign(const char* cmd_name) : TclCmd(cmd_name) {
  auto* cell_name_option = new TclStringOption("cell_name", 1, nullptr);
  addOption(cell_name_option);
}

unsigned CmdLinkDesign::check() {
  TclOption* cell_name_option = getOptionOrArg("cell_name");
  LOG_FATAL_IF(!cell_name_option);
  return 1;
}

unsigned CmdLinkDesign::exec() {
  if (!check()) {
    return 0;
  }

  TclOption* cell_name_option = getOptionOrArg("cell_name");
  auto* cell_name = cell_name_option->getStringVal();

  Sta* ista = Sta::getOrCreateSta();
  ista->set_top_module_name(cell_name);
  ista->linkDesign(cell_name);

  return 1;
}
}  // namespace ista