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
#include "tcl_sta.h"

#include "tool_manager.h"

namespace tcl {

CmdSTARun::CmdSTARun(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* file_name_option = new TclStringOption(TCL_OUTPUT_PATH, 1, nullptr);
  addOption(file_name_option);
}

unsigned CmdSTARun::check()
{
  // TclOption* file_name_option = getOptionOrArg(TCL_OUTPUT_PATH);
  // LOG_FATAL_IF(!file_name_option);
  return 1;
}

unsigned CmdSTARun::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_OUTPUT_PATH);
  auto path = option->getStringVal() != nullptr ? option->getStringVal() : "";

  if (iplf::tmInst->autoRunSTA(path)) {
    std::cout << "iSTA run successfully." << std::endl;
  }

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CmdBuildClockTree::CmdBuildClockTree(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* file_name_option = new TclStringOption(TCL_OUTPUT_PATH, 1, nullptr);
  auto* data_option = new TclStringOption("clock_data_path", 1, nullptr);

  addOption(file_name_option);
  addOption(data_option);
}

unsigned CmdBuildClockTree::check()
{
  //   TclOption* file_name_option = getOptionOrArg(TCL_OUTPUT_PATH);
  // LOG_FATAL_IF(!file_name_option);
  return 1;
}

unsigned CmdBuildClockTree::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_OUTPUT_PATH);
  auto path = option->getStringVal() != nullptr ? option->getStringVal() : "";

  TclOption* data_option = getOptionOrArg("clock_data_path");
  auto data_path = data_option->getStringVal() != nullptr ? data_option->getStringVal() : "";

  iplf::tmInst->buildClockTree(path, data_path);

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CmdSTAInit::CmdSTAInit(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* file_name_option = new TclStringOption(TCL_OUTPUT_PATH, 1, nullptr);
  addOption(file_name_option);
}

unsigned CmdSTAInit::check()
{
  // TclOption* file_name_option = getOptionOrArg(TCL_OUTPUT_PATH);
  // LOG_FATAL_IF(!file_name_option);
  return 1;
}

unsigned CmdSTAInit::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_OUTPUT_PATH);
  auto path = option->getStringVal() != nullptr ? option->getStringVal() : "";

  if (iplf::tmInst->initSTA(path)) {
    std::cout << "iSTA run successfully." << std::endl;
  }

  return 1;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CmdSTAReport::CmdSTAReport(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* file_name_option = new TclStringOption(TCL_OUTPUT_PATH, 1, nullptr);
  addOption(file_name_option);
}

unsigned CmdSTAReport::check()
{
  // TclOption* file_name_option = getOptionOrArg(TCL_OUTPUT_PATH);
  // LOG_FATAL_IF(!file_name_option);
  return 1;
}

unsigned CmdSTAReport::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_OUTPUT_PATH);
  auto path = option->getStringVal() != nullptr ? option->getStringVal() : "";

  if (iplf::tmInst->runSTA(path)) {
    std::cout << "iSTA run successfully." << std::endl;
  }

  return 1;
}

}  // namespace tcl
