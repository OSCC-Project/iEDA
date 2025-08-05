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
#ifdef BUILD_GUI
#include "tcl_register_gui.h"
#endif

#include "flow.h"
#include "tcl_flow.h"
#include "tcl_register_config.h"
#include "tcl_register_cts.h"
#include "tcl_register_eco.h"
#include "tcl_register_eval.h"
#include "tcl_register_feature.h"
#include "tcl_register_flow.h"
#include "tcl_register_fp.h"
#include "tcl_register_idb.h"
#include "tcl_register_idrc.h"
#include "tcl_register_inst.h"
#include "tcl_register_irt.h"
#include "tcl_register_no.h"
#include "tcl_register_pdn.h"
#include "tcl_register_pl.h"
#include "tcl_register_power.h"
#include "tcl_register_report.h"
#include "tcl_register_sta.h"
#include "tcl_register_to.h"
#include "tcl_register_vec.h"
#include "tcl_register_pnp.h"

#ifdef CONTEST
#include "tcl_register_contest.h"
#endif

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

  /// Power
  registerCmdPower();

#ifdef BUILD_GUI
  /// gui
  registerCmdGUI();
#endif

  registerCmdReport();

  registerCmdFeature();

  registerCmdEval();

  registerCmdECO();

  registerCmdVectorization();

  /// PNP
  registerCmdPNP();

#ifdef CONTEST
  registerCmdContest();
#endif

  return EXIT_SUCCESS;
}

}  // namespace tcl