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

#include "tcl_ipnp.h"

#include "ipnp_api.hh"
#include "tcl_util.h"
#include "tool_manager.h"

namespace tcl {

CmdRunPnp::CmdRunPnp(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* config_option = new TclStringOption("-config", 1, nullptr);
  addOption(config_option);
}

unsigned CmdRunPnp::check()
{
  TclOption* config_option = getOptionOrArg("-config");
  LOG_FATAL_IF(!config_option);
  return 1;
}

unsigned CmdRunPnp::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* config_option = getOptionOrArg("-config");
  std::string config_file = config_option->getStringVal();

  std::cout << "Running iPNP with config: " << config_file << std::endl;

  bool success = iplf::tmInst->autoRunPNP(config_file);

  if (success) {
    std::cout << "iPNP execution completed successfully" << std::endl;
  } else {
    std::cout << "iPNP execution failed" << std::endl;
  }

  return success ? 1 : 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CmdAddVIA1::CmdAddVIA1(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* config_option = new TclStringOption(TCL_CONFIG, 0, "");
  addOption(config_option);
}

unsigned CmdAddVIA1::check()
{
  TclOption* file_name_option = getOptionOrArg(TCL_CONFIG);
  LOG_FATAL_IF(!file_name_option);

  return 1;  // check success
}

unsigned CmdAddVIA1::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* config_option = getOptionOrArg(TCL_CONFIG);

  if (config_option) {
    auto* config_file = config_option->getStringVal();

    PNPApiInst->connect_M2_M1(config_file);

    LOG_INFO << "VIA1 connections added successfully.";
  }

  return 1;
}

}  // namespace tcl
