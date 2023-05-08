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
#include "flow.h"

using namespace ieda;

namespace tcl {

int registerCmdFlow()
{
  registerTclCmd(CmdFlowAutoRun, "flow_run");
  registerTclCmd(CmdFlowExit, "flow_exit");
  return EXIT_SUCCESS;
}

}  // namespace tcl