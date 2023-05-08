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

using namespace ieda;
namespace tcl {

int registerCmdTO()
{
  registerTclCmd(CmdTOAutoRun, "run_to");
  registerTclCmd(CmdTORunDrv, "run_to_drv");
  registerTclCmd(CmdTORunHold, "run_to_hold");
  registerTclCmd(CmdTORunSetup, "run_to_setup");

  return EXIT_SUCCESS;
}

}  // namespace tcl