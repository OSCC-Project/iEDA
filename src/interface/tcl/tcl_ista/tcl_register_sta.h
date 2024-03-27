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
#include "tcl_sta.h"

#define TCL_USERSHELL

#ifdef TCL_USERSHELL
#include "sdc-cmd/Cmd.hh"
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
  registerTclCmd(ista::CmdReadLefDef, "read_lef_def");
  registerTclCmd(ista::CmdReadLiberty, "read_liberty");
  registerTclCmd(ista::CmdLinkDesign, "link_design");
  registerTclCmd(ista::CmdReadSpef, "read_spef");
  registerTclCmd(ista::CmdReadSdc, "read_sdc");
  registerTclCmd(ista::CmdReportTiming, "report_timing");
  registerTclCmd(ista::CmdReportConstraint, "report_constraint");
  registerTclCmd(ista::CmdDefToVerilog, "def_to_verilog");
  registerTclCmd(ista::CmdVerilogToDef, "verilog_to_def");
  registerTclCmd(ista::CmdGetLibs, "get_libs");
  return EXIT_SUCCESS;
}

}  // namespace tcl