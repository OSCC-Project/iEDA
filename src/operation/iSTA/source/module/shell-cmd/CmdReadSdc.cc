/**
 * @file CmdReadSdc.cc
 * @author Wang Hao (harry0789@qq.com)
 * @brief
 * @version 0.1
 * @date 2021-10-12
 */
#include "ShellCmd.hh"
#include "sta/Sta.hh"

namespace ista {
CmdReadSdc::CmdReadSdc(const char* cmd_name) : TclCmd(cmd_name) {
  auto* file_name_option = new TclStringOption("file_name", 1, nullptr);
  addOption(file_name_option);
  // -echo
}

unsigned CmdReadSdc::check() {
  TclOption* file_name_option = getOptionOrArg("file_name");
  LOG_FATAL_IF(!file_name_option);
  return 1;
}

unsigned CmdReadSdc::exec() {
  if (!check()) {
    return 0;
  }

  TclOption* file_name_option = getOptionOrArg("file_name");
  auto* sdc_script_file = file_name_option->getStringVal();

  Sta* ista = Sta::getOrCreateSta();
  return ista->readSdc(sdc_script_file);
}
}  // namespace ista