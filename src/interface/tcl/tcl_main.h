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
#pragma once
/**
 * @File Name: tcl_main.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */
#include <chrono>
#include <thread>

#include "UserShell.hh"
#include "tool_manager.h"
#include "tcl_register.h"

namespace tcl {

int tcl_start(int tcl_argc, char** tcl_argv)
{
#ifdef BUILD_GUI
  iplf::tmInst->guiInit();
#endif

  // this_thread::sleep_for(chrono::seconds(10));

  auto shell = ieda::UserShell::getShell();
  shell->set_init_func(registerCommands);
  shell->userMain(tcl_argc, tcl_argv);

  return 0;
}

}  // namespace tcl