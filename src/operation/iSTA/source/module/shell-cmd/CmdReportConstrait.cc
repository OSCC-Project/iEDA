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