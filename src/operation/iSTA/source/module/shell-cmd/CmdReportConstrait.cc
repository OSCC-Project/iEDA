/**
 * @file CmdReportConstraint.cc
 * @author Wang Hao (harry0789@qq.com)
 * @brief
 * @version 0.1
 * @date 2021-10-12
 */
#include "ShellCmd.hh"

namespace ista {
CmdReportConstraint::CmdReportConstraint(const char* cmd_name)
    : TclCmd(cmd_name) {
  auto* digits_option = new TclIntOption("-digits", 0, 0);
  addOption(digits_option);
  auto* violators_option = new TclSwitchOption("-violators");
  addOption(violators_option);
  auto* max_slew_option = new TclSwitchOption("-max_slew");
  addOption(max_slew_option);
  auto* file_name_option = new TclStringOption(">", 0, nullptr);
  addOption(file_name_option);
}

unsigned CmdReportConstraint::check() {
  TclOption* file_name_option = getOptionOrArg("file_name");
  LOG_FATAL_IF(!file_name_option);
  return 1;
}

unsigned CmdReportConstraint::exec() {
  if (!check()) {
    return 0;
  }

  // TclOption* digits_option = getOptionOrArg("-digits");
  // if (digits_option->is_set_val()) {
  //   auto digits = digits_option->getIntVal();
  // }

  // TclOption* path_delay_option = getOptionOrArg("-violators");
  // if (path_delay_option->is_set_val()) {
  //   auto path_delay = path_delay_option->getStringVal();
  // }

  // TclOption* file_name_option = getOptionOrArg(">");
  // if (file_name_option->is_set_val()) {
  //   auto file_name = file_name_option->getStringVal();
  //   // todo: print to file
  // } else {
  //   // todo: print to shell
  // }

  return 1;
}
}  // namespace ista