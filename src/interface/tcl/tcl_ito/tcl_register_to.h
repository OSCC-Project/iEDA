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
 * @File Name: tcl_register.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */
#include "ScriptEngine.hh"
#include "UserShell.hh"
#include "tcl_ito.h"
#include "tcl_toconfig.h"

using namespace ieda;
namespace tcl {

int registerCmdTO()
{
  registerTclCmd(CmdTOAutoRun, "run_to");
  registerTclCmd(CmdTORunDrv, "run_to_drv");
  registerTclCmd(CmdTORunDrvSpecialNet, "run_to_drv_special_net");
  registerTclCmd(CmdTORunHold, "run_to_hold");
  registerTclCmd(CmdTORunSetup, "run_to_setup");
  registerTclCmd(CmdTOBuffering, "run_to_buffering");
  registerTclCmd(CmdTOConfig, "to_config");

  return EXIT_SUCCESS;
}

}  // namespace tcl