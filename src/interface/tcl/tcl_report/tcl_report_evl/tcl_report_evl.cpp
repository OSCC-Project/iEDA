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
#include "tcl_report_evl.h"
#include "report_manager.h"

namespace tcl {

CmdReportWL::CmdReportWL(const char* cmd) : TclCmd(cmd)
{
  addOption(new TclStringOption(TCL_PATH, 1));
}
unsigned CmdReportWL::check()
{
  return 1;
}

CMD_CLASS_DEFAULT_EXEC(CmdReportWL,
                       rptInst->reportWL(getOptionOrArg(TCL_PATH)->getStringVal() ? getOptionOrArg(TCL_PATH)->getStringVal() : "");)

CmdReportCong::CmdReportCong(const char* cmd) : TclCmd(cmd)
{
  addOption(new TclStringOption(TCL_PATH, 1));
}
unsigned CmdReportCong::check()
{
  return 1;
}
CMD_CLASS_DEFAULT_EXEC(CmdReportCong,
                       rptInst->reportCongestion(getOptionOrArg(TCL_PATH)->getStringVal() ? getOptionOrArg(TCL_PATH)->getStringVal() : ""));

}  // namespace tcl