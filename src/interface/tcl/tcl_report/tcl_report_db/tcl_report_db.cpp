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
#include "tcl_report_db.h"

#include "report_manager.h"

namespace tcl {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CmdReportDbSummary::CmdReportDbSummary(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* file_name_option = new TclStringOption(TCL_PATH, 1, EMPTY_STR);
  addOption(file_name_option);
}

unsigned CmdReportDbSummary::check()
{
  TclOption* file_name_option = getOptionOrArg(TCL_PATH);
  LOG_FATAL_IF(!file_name_option);
  return 1;
}

unsigned CmdReportDbSummary::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_PATH);

  rptInst->reportDBSummary(option->getStringVal());

  return 1;
}

CmdReportDanglingNet::CmdReportDanglingNet(const char* name) : TclCmd(name)
{
  addOption(new TclStringOption(TCL_PATH, 1, EMPTY_STR));
}

unsigned CmdReportDanglingNet::check()
{
  return 1;
}

unsigned CmdReportDanglingNet::exec()
{
  if (!check()) {
    return 0;
  }
  TclOption* option = getOptionOrArg(TCL_PATH);

  return rptInst->reportDanglingNet(option->getStringVal());
}

}  // namespace tcl
