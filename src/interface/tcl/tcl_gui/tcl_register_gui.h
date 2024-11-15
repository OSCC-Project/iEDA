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
#pragma once
/**
 * @File Name: tcl_register.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */
#include "ScriptEngine.hh"
#include "UserShell.hh"
#include "tcl_gui.h"
#include "tcl_qt.h"
#include "tcl_web.h"
#include "tool_manager.h"

using namespace ieda;

namespace tcl {

int registerCmdGUI()
{
  registerTclCmd(CmdGuiStart, "gui_start");
  registerTclCmd(CmdGuiShow, "gui_show");
  registerTclCmd(CmdGuiHide, "gui_hide");
  registerTclCmd(CmdGuiShowDrc, "gui_show_drc");
  registerTclCmd(CmdGuiShowClockTree, "gui_show_cts");
  registerTclCmd(CmdGuiShowPlacement, "gui_show_pl");
  registerTclCmd(CmdGuiShowGraph, "gui_show_graph");

  // web
  registerTclCmd(CmdCaptureDesign, "capture_design");

  GuiTclNotifier::setup();
  // run Qt's event loop
  Tcl_SetMainLoop([]() { iplf::tmInst->guiExec(); });
  std::cout << "Tcl_SetMainLoop......" << std::endl;

  return EXIT_SUCCESS;
}

}  // namespace tcl