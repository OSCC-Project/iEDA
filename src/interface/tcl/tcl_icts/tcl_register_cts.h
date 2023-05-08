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
#include "tcl_cts.h"

using namespace ieda;
namespace tcl {

int registerCmdCTS() {
  registerTclCmd(CmdCTSAutoRun, "run_cts");
  registerTclCmd(CmdCTSReport, "cts_report");
  registerTclCmd(CmdCTSSaveTree, "cts_save_tree");

  return EXIT_SUCCESS;
}

}  // namespace tcl