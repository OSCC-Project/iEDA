/**
 * @file CmdReadLiberty.cc
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
CmdReadLiberty::CmdReadLiberty(const char* cmd_name) : TclCmd(cmd_name) {
  auto* file_name_option = new TclStringListOption("file_name", 1);
  addOption(file_name_option);
  // -corner_name
  // -min
  // -max
}

unsigned CmdReadLiberty::check() {
  TclOption* file_name_option = getOptionOrArg("file_name");
  LOG_FATAL_IF(!file_name_option);
  return 1;
}

unsigned CmdReadLiberty::exec() {
  if (!check()) {
    return 0;
  }

  TclOption* file_name_option = getOptionOrArg("file_name");
  auto liberty_files = file_name_option->getStringList();

  Sta* ista = Sta::getOrCreateSta();
  ista->readLiberty(liberty_files);

  return 1;
}
}  // namespace ista