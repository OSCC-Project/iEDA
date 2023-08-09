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
 * @File Name: tcl_web.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */
#include "tcl_web.h"

#include "gui_io.h"
#include "tool_manager.h"

namespace tcl {

CmdCaptureDesign::CmdCaptureDesign(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* type = new TclStringOption(TCL_PATH, 1, nullptr);
  addOption(type);
}

unsigned CmdCaptureDesign::check()
{
  // TclOption *file_name_option = getOptionOrArg("-path");
  // LOG_FATAL_IF(!file_name_option);
  return 1;
}

unsigned CmdCaptureDesign::exec()
{
  if (!check()) {
    return 0;
  }

  std::string path = "";
  TclOption* opt = getOptionOrArg(TCL_PATH);
  if (opt->getStringVal() != nullptr) {
    path = opt->getStringVal();
  }

  iplf::tmInst->guiCaptrueDesign(path);

  return 1;
}

}  // namespace tcl
