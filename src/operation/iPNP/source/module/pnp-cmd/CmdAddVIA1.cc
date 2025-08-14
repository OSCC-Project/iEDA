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
/**
 * @file CmdAddVIA1.cc
 * @author Jianrong Su
 * @brief Command to add VIA1 connections between M2 and M1 layers
 * @version 1.0
 * @date 2025-06-30
 */

#include <filesystem>

#include "PNP.hh"
#include "PNPShellCmd.hh"
#include "ipnp_api.hh"
#include "log/Log.hh"

namespace ipnp {

CmdAddVIA1::CmdAddVIA1(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* config_option = new TclStringOption("-config", 0, "");
  addOption(config_option);
}

unsigned CmdAddVIA1::check()
{
  TclOption* config_option = getOptionOrArg("-config");

  if (config_option) {
    auto* config_file = config_option->getStringVal();
    if (!std::filesystem::exists(config_file)) {
      LOG_ERROR << "Configuration file not found: " << config_file;
      return 0;
    }
  }
  return 1;  // check success
}

unsigned CmdAddVIA1::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* config_option = getOptionOrArg("-config");

  if (config_option) {
    auto* config_file = config_option->getStringVal();

    LOG_INFO << "Adding VIA1 connections between M2 and M1 layers with configuration: " << config_file;

    // 直接调用静态方法
    ipnp::PNPApi::connect_M2_M1(config_file);

    LOG_INFO << "VIA1 connections added successfully.";
  } else {
    LOG_ERROR << "Configuration file is required for adding VIA1 connections.";
    return 0;
  }

  return 1;
}

}  // namespace ipnp