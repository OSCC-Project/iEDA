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
#ifdef BUILD_GUI
#include "tcl_register_gui.h"
#endif

#include "flow.h"
#include "tcl_flow.h"
#include "tcl_register_config.h"
#include "tcl_register_cts.h"
#include "tcl_register_drc.h"
#include "tcl_register_flow.h"
#include "tcl_register_fp.h"
#include "tcl_register_idb.h"
#include "tcl_register_inst.h"
#include "tcl_register_irt.h"
#include "tcl_register_pdn.h"
#include "tcl_register_pl.h"
#include "tcl_register_report.h"
#include "tcl_register_sta.h"
#include "tcl_register_to.h"
#include "tcl_register_no.h"

using namespace ieda;
namespace tcl {

int registerCommands()
{
  /// config
  registerConfig();

  /// flow
  registerCmdFlow();

  /// db
  registerCmdDB();

  /// instance operation
  registerCmdInstance();

  /// FP
  registerCmdFP();

  /// PDN
  registerCmdPDN();

  // /// Placer
  registerCmdPlacer();

  /// CTS
  registerCmdCTS();

  /// NO
  registerCmdNO();

  /// TO
  registerCmdTO();

  /// Router
  registerCmdRT();

  /// DRC
  registerCmdDRC();

  /// STA
  registerCmdSTA();

#ifdef BUILD_GUI
  /// gui
  registerCmdGUI();
#endif

  registerCmdReport();

  return EXIT_SUCCESS;
}

}  // namespace tcl