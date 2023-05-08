/**
 * @File Name: tcl_flow.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */
#include "tcl_flow.h"

#include <iostream>

#include "flow.h"

namespace tcl {

CmdFlowAutoRun::CmdFlowAutoRun(const char* cmd_name) : TclCmd(cmd_name)
{
  // auto* file_name_option = new TclStringOption(TCL_CONFIG, 1, nullptr);
  // addOption(file_name_option);
}

unsigned CmdFlowAutoRun::check()
{
  // TclOption* file_name_option = getOptionOrArg(TCL_CONFIG);
  // LOG_FATAL_IF(!file_name_option);
  return 1;
}

unsigned CmdFlowAutoRun::exec()
{
  if (!check()) {
    return 0;
  }

  // TclOption* option = getOptionOrArg(TCL_CONFIG);
  // auto data_config  = option->getStringVal();
  iplf::plfInst->runFlow();

  return 1;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CmdFlowExit::CmdFlowExit(const char* cmd_name) : TclCmd(cmd_name)
{
}

unsigned CmdFlowExit::check()
{
  return 1;
}

unsigned CmdFlowExit::exec()
{
  if (!check()) {
    return 0;
  }

  exit(0);

  return 1;
}

}  // namespace tcl
