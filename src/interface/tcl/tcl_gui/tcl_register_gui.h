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
#include "tcl_qt/tcl_qt.h"
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

  GuiTclNotifier::setup();
  // run Qt's event loop
  Tcl_SetMainLoop([]() { iplf::tmInst->guiExec(); });
  std::cout << "Tcl_SetMainLoop......" << std::endl;

  return EXIT_SUCCESS;
}

}  // namespace tcl