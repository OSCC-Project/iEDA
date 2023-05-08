/**
 * @file CmdReadVcd.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief cmd to read a vcd.
 * @version 0.1
 * @date 2023-05-04
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "PowerShellCmd.hh"
#include "sta/Sta.hh"

namespace ipower {
CmdReadVcd::CmdReadVcd(const char* cmd_name) : TclCmd(cmd_name) {
  auto* file_name_option = new TclStringOption("file_name", 1, nullptr);
  addOption(file_name_option);

  auto* top_instance_name_option = new TclStringOption("-top_name", 0, nullptr);
  addOption(top_instance_name_option);

  //   auto* begin_time_option = new TclIntOption("-begin_time", 0, 0);
  //   addOption(begin_time_option);

  //   auto* end_time_option = new TclIntOption("-end_time", 0, 0);
  //   addOption(end_time_option);
}

unsigned CmdReadVcd::check() {
  TclOption* file_name_option = getOptionOrArg("file_name");
  TclOption* top_instance_name_option = getOptionOrArg("top_instance_name");
  LOG_FATAL_IF(!file_name_option);
  LOG_FATAL_IF(!top_instance_name_option);
  return 1;
}

unsigned CmdReadVcd::exec() {
  if (!check()) {
    return 0;
  }

  TclOption* file_name_option = getOptionOrArg("file_name");
  auto vcd_file = file_name_option->getStringVal();

  TclOption* top_instance_name_option = getOptionOrArg("top_instance_name");
  auto top_instance_name = top_instance_name_option->getStringVal();

  Sta* ista = Sta::getOrCreateSta();
  Power* ipower = Power::getOrCreatePower(&(ista->get_graph()));
  // TODO set begin end time
  return ipower->readVCD(vcd_file, top_instance_name);
}

}  // namespace ipower