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
#include "tcl_power.h"

#include "tool_manager.h"

namespace tcl {

CmdPowerRun::CmdPowerRun(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* file_name_option = new TclStringOption(TCL_OUTPUT_PATH, 1, nullptr);
  addOption(file_name_option);
}

unsigned CmdPowerRun::check()
{
  // TclOption* file_name_option = getOptionOrArg(TCL_OUTPUT_PATH);
  // LOG_FATAL_IF(!file_name_option);
  return 1;
}

unsigned CmdPowerRun::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_OUTPUT_PATH);
  auto path = option->getStringVal() != nullptr ? option->getStringVal() : "";

  if (iplf::tmInst->autoRunPower(path)) {
    std::cout << "iPA run successfully." << std::endl;
  }

  return 1;
}

}  // namespace tcl
