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
#include "tcl_drc.h"

using namespace ieda;
namespace tcl {

int registerCmdDRC()
{
  registerTclCmd(CmdDRCAutoRun, "run_drc");
  registerTclCmd(CmdDRCSaveDetailFile, "save_drc");
  registerTclCmd(CmdDRCReadDetailFile, "read_drc");

  registerTclCmd(TclInitDrc, "init_drc");
  registerTclCmd(TclDrcCheckDef, "drc_check_def");
  registerTclCmd(TclDestroyDrc, "destroy_drc");

  registerTclCmd(TclInitDrcAPI, "init_drc_api");
  registerTclCmd(TclDestroyDrcAPI, "destroy_drc_api");

  registerTclCmd(TclDrcCheckNet, "check_net");
  registerTclCmd(TclDrcCheckAllNet, "check_all_net");
  registerTclCmd(TclCheckDrc, "check_drc");

  return EXIT_SUCCESS;
}

}  // namespace tcl