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
#include "tcl_db.h"

#include "idm.h"
#include "report_manager.h"
#include "tool_manager.h"

namespace tcl {

static const char* const INST_OPT = "-inst";
static const char* const NET_OPT = "-net";
CmdIdbGet::CmdIdbGet(const char* name) : TclCmd(name)
{
  static const char* empty_str = "";
  for (const char* arg : {TCL_PATH, INST_OPT, NET_OPT}) {
    addOption(new TclStringOption(arg, 1, empty_str));
  }
}
unsigned CmdIdbGet::check()
{
  return 1;
}
/**
 * @brief idb_get -path ./result/test/test.rpt -inst xxx
 *        idb_get -path ./result/test/test.rpt -net xxx
 *
 * @return unsigned
 */
unsigned CmdIdbGet::exec()
{
  std::string path = getOptionOrArg(TCL_PATH)->getStringVal();
  std::string inst_name = getOptionOrArg(INST_OPT)->getStringVal();

  if (not inst_name.empty()) {
    rptInst->reportInstance(path, inst_name);
  } else if (std::string net_name = getOptionOrArg(NET_OPT)->getStringVal(); !net_name.empty()) {
    rptInst->reportNet(path, net_name);
  }
  return 1;
}

}  // namespace tcl
