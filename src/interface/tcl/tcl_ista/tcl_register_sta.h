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
#include "tcl_sta.h"

#define TCL_USERSHELL

#ifdef TCL_USERSHELL
// #include "sdc-cmd/Cmd.hh"
#include "shell-cmd/ShellCmd.hh"
#endif

using namespace ieda;

namespace tcl {
int registerCmdSTA()
{
  registerTclCmd(CmdSTARun, "run_sta");
  registerTclCmd(CmdSTAInit, "init_sta");
  registerTclCmd(CmdSTAReport, "report_sta");
  registerTclCmd(CmdBuildClockTree, "build_clock_tree");

  /// sta tcl
  registerTclCmd(ista::CmdTESTSLL, "test_string_list_list");
  registerTclCmd(ista::CmdSetDesignWorkSpace, "set_design_workspace");
  registerTclCmd(ista::CmdReadVerilog, "read_netlist");
  registerTclCmd(ista::CmdReadLiberty, "read_liberty");
  registerTclCmd(ista::CmdLinkDesign, "link_design");
  registerTclCmd(ista::CmdReadSpef, "read_spef");
  registerTclCmd(ista::CmdReadSdc, "read_sdc");
  registerTclCmd(ista::CmdReportTiming, "report_timing");
  registerTclCmd(ista::CmdReportConstraint, "report_constraint");
  return EXIT_SUCCESS;
}

}  // namespace tcl