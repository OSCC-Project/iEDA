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
#include "tcl_config.h"

using namespace ieda;
namespace tcl {

int registerConfig()
{
  registerTclCmd(CmdFlowInitConfig, "flow_init");
  registerTclCmd(CmdDbConfigSetting, "db_init");

  return EXIT_SUCCESS;
}

}  // namespace tcl