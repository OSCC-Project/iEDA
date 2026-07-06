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
 * @file tcl_init_rcx.cpp
 * @author Yipei Xu (yipeix@163.com)
 * @brief
 * @version 0.1
 * @date 2025-12-09
 */
#include <iostream>

#include "RCXAPI.hh"
#include "log/Log.hh"
#include "tcl_ircx.h"
#include "tcl_util.h"

namespace tcl {

TclInitRCX::TclInitRCX(const char* cmd_name) : TclCmd(cmd_name)
{
  addOption(new TclStringOption("-config", 0, nullptr));
}

unsigned TclInitRCX::check()
{
  TclOption* config_option = getOptionOrArg("-config");
  if (!config_option || !config_option->is_set_val()) {
    LOG_ERROR << "init_rcx requires -config.";
    return 0;
  }

  const char* config_file = config_option->getStringVal();
  if (config_file == nullptr || config_file[0] == '\0') {
    LOG_ERROR << "init_rcx requires a non-empty -config file.";
    return 0;
  }

  return 1;
}

unsigned TclInitRCX::exec()
{
  if (!check()) {
    return 0;
  }

  const std::string hello_info =
      "\033[49;32m***************************\n"
      "  _  _____   _____ __   __\n"
      " (_)|  __ \\ /  __ \\\\ \\ / /\n"
      "  _ | |__) || |     \\ V / \n"
      " | ||  _  / | |      > <  \n"
      " | || | \\ \\ | |____ / . \\ \n"
      " |_||_|  \\_\\ \\_____/_/ \\_\\\n"
      "***************************\n"
      "WELCOME TO iRCX TCL-shell interface. \e[0m";
  std::cout << hello_info << std::endl;

  TclOption* config_option = getOptionOrArg("-config");
  return RCX_API_INST.init(config_option->getStringVal()) ? 1U : 0U;
}

}  // namespace tcl
