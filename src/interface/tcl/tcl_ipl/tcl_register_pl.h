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
#include "tcl_ipl.h"

using namespace ieda;
namespace tcl {

int registerCmdPlacer()
{
  registerTclCmd(CmdPlacerAutoRun, "run_placer");
  registerTclCmd(CmdPlacerFiller, "run_filler");
  registerTclCmd(CmdPlacerIncrementalFlow, "run_incremental_flow");
  registerTclCmd(CmdPlacerIncrementalLG, "run_incremental_lg");
  registerTclCmd(CmdPlacerCheckLegality, "placer_check_legality");
  registerTclCmd(CmdPlacerReport, "placer_report");

  registerTclCmd(CmdPlacerInit, "init_pl");
  registerTclCmd(CmdPlacerDestroy, "destroy_pl");
  registerTclCmd(CmdPlacerRunMP, "placer_run_mp");
  registerTclCmd(CmdPlacerRunGP, "placer_run_gp");
  registerTclCmd(CmdPlacerRunLG, "placer_run_lg");
  registerTclCmd(CmdPlacerRunDP, "placer_run_dp");

  return EXIT_SUCCESS;
}

}  // namespace tcl