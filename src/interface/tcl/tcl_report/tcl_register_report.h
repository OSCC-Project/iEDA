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
#include "./tcl_report_db/tcl_report_db.h"
#include "./tcl_report_evl/tcl_report_evl.h"
#include "./tcl_report_pr/tcl_report_route.h"
#include "ScriptEngine.hh"
#include "UserShell.hh"
#include "./tcl_report_pr/tcl_report_place.h"
using namespace ieda;
namespace tcl {

int registerCmdReportDb()
{
  registerTclCmd(CmdReportDbSummary, "report_db");

  return EXIT_SUCCESS;
}

int registerCmdReportEval()
{
  registerTclCmd(CmdReportWL, "report_wirelength");
  registerTclCmd(CmdReportCong, "report_congestion");
  // registerTclCmd(CmdReportDanglingNet, "report_dangling_net");

  return EXIT_SUCCESS;
}

int registerCmdReport()
{
  registerCmdReportDb();
  registerCmdReportEval();
  registerTclCmd(CmdReportRoute, "report_route");
  registerTclCmd(CmdReportPlaceDistro, "report_inst_distro");
  registerTclCmd(CmdReportPrefixedInst, "report_prefixed_instance");
  return EXIT_SUCCESS;
}

}  // namespace tcl