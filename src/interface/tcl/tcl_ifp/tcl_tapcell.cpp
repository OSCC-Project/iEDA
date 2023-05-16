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
#include "Str.hh"
#include "idm.h"
#include "ifp_api.h"
#include "tcl_ifp.h"
#include "tool_manager.h"

namespace tcl {

TclFpTapCell::TclFpTapCell(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* tap_name = new TclStringOption("-tapcell", 0, nullptr);
  auto* distance = new TclDoubleOption("-distance", 0, 0);
  auto* endcap_name = new TclStringOption("-endcap", 0);

  addOption(tap_name);
  addOption(distance);
  addOption(endcap_name);
}

unsigned TclFpTapCell::check()
{
  TclOption* tap_name = getOptionOrArg("-tapcell");
  TclOption* distance = getOptionOrArg("-distance");
  TclOption* endcap_name = getOptionOrArg("-endcap");

  LOG_FATAL_IF(!tap_name);
  LOG_FATAL_IF(!distance);
  LOG_FATAL_IF(!endcap_name);

  return 1;
}

unsigned TclFpTapCell::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* tap_name = getOptionOrArg("-tapcell");
  TclOption* distance = getOptionOrArg("-distance");
  TclOption* endcap_name = getOptionOrArg("-endcap");

  auto tap = tap_name->getStringVal();
  auto dis = distance->getDoubleVal();
  auto end = endcap_name->getStringVal();

  fpApiInst->tapCells(dis, tap, end);

  return 1;
}

}  // namespace tcl
