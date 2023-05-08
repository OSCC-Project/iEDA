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
#include "tcl_ino.h"

using namespace ieda;
namespace tcl {

int registerCmdNO()
{
  registerTclCmd(CmdNORunFixFanout, "run_no_fixfanout");
  return EXIT_SUCCESS;
}

}  // namespace tcl