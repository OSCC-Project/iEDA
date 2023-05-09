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
#include "tcl_report_place.h"

#include "report_manager.h"
namespace tcl {

using ieda::TclOption;
using ieda::TclStringListOption;
using ieda::TclStringOption;

static const auto PREFIXES = "-prefixes";
CmdReportPlaceDistro::CmdReportPlaceDistro(const char* cmd_name) : TclCmd(cmd_name)
{
  addOption(new TclStringOption(TCL_PATH, 1, EMPTY_STR));
  addOption(new TclStringListOption(PREFIXES, 1));
}

unsigned CmdReportPlaceDistro::exec()
{
  auto prefixes = getOptionOrArg(PREFIXES)->getStringList();
  rptInst->reportPlaceDistribution(prefixes);
  return 1;
}
static const auto PREFIX = "-prefix";
static const auto LEVEL = "-level";
static const auto THRESHOLD = "-mincount";
CmdReportPrefixedInst::CmdReportPrefixedInst(const char* cmd_name) : TclCmd(cmd_name)
{
  addOption(new TclStringOption("-prefix", 1, EMPTY_STR));
  addOption(new ieda::TclIntOption("-level", 1, 1));
  addOption(new ieda::TclIntOption(THRESHOLD, 1, 1));
}
unsigned CmdReportPrefixedInst::exec()
{
  auto* prefix = getOptionOrArg(PREFIX)->getStringVal();
  auto level = getOptionOrArg(LEVEL)->getIntVal();
  auto mincount = getOptionOrArg(THRESHOLD)->getIntVal();
  rptInst->reportInstLevel(prefix, level, mincount);
  return 1;
}
}  // namespace tcl