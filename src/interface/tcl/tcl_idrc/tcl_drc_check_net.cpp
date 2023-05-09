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
#include "idm.h"
#include "tcl_drc.h"

namespace tcl {

TclDrcCheckNet::TclDrcCheckNet(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* file_name_option = new TclStringOption(TCL_NAME, 1, nullptr);
  addOption(file_name_option);
}

unsigned TclDrcCheckNet::check()
{
  TclOption* file_name_option = getOptionOrArg(TCL_NAME);
  LOG_FATAL_IF(!file_name_option);

  return 1;
}

unsigned TclDrcCheckNet::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_NAME);
  if (option != nullptr) {
    std::string net_name = option->getStringVal();
    dmInst->isNetConnected(net_name);
  }

  return 1;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TclDrcCheckAllNet::TclDrcCheckAllNet(const char* cmd_name) : TclCmd(cmd_name)
{
}

unsigned TclDrcCheckAllNet::check()
{
  return 1;
}

unsigned TclDrcCheckAllNet::exec()
{
  if (!check()) {
    return 0;
  }

  auto result = dmInst->isAllNetConnected();

  return 1;
}
}  // namespace tcl
