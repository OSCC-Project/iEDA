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
#include "tcl_report_route.h"

namespace tcl {

static const auto TCL_NET = "-net";
static const auto TCL_SUMMARY = "-summary";

CmdReportRoute::CmdReportRoute(const char* cmd_name) : TclCmd(cmd_name)
{
  addOption(new TclStringOption(TCL_PATH, 1, EMPTY_STR));
  addOption(new TclStringOption(TCL_NET, 1, EMPTY_STR));
  addOption(new ieda::TclIntOption(TCL_SUMMARY, 1, 1));
}
unsigned CmdReportRoute::exec()
{
  auto* file_path = getOptionOrArg(TCL_PATH)->getStringVal();
  auto* net = getOptionOrArg(TCL_NET)->getStringVal();
  auto summary = getOptionOrArg(TCL_SUMMARY)->getIntVal();
  rptInst->reportRoute(file_path, net, summary);
  return 1;
};

}  // namespace tcl