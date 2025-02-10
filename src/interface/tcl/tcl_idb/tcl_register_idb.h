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

#include "tcl_db.h"
#include "tcl_db_file.h"
#include "tcl_db_operate.h"

using namespace ieda;
namespace tcl {

int registerCmdDB()
{
  registerTclCmd(CmdInitIdb, "idb_init");
  registerTclCmd(CmdInitTechLef, "tech_lef_init");
  registerTclCmd(CmdInitLef, "lef_init");
  registerTclCmd(CmdInitDef, "def_init");
  registerTclCmd(CmdInitVerilog, "verilog_init");
  registerTclCmd(CmdSaveDef, "def_save");
  registerTclCmd(CmdSaveNetlist, "netlist_save");
  registerTclCmd(CmdSaveJSON, "json_save");
  registerTclCmd(CmdSaveGDS, "gds_save");
  registerTclCmd(CmdGenerateMPScript, "aimp_random");

  /// idb operator
  registerTclCmd(CmdIdbSetNet, "set_net");
  registerTclCmd(CmdIdbMergeNets, "merge_nets");

  registerTclCmd(CmdIdbClearBlockageExceptPgNet, "remove_except_pg_net");
  registerTclCmd(CmdIdbClearBlockage, "clear_blockage");

  registerTclCmd(CmdIdbGet, "idb_get");
  registerTclCmd(CmdIdbDeleteInstance, "delete_inst");
  registerTclCmd(CmdIdbDeleteNet, "delete_net");

  registerTclCmd(CmdIdbCreateInstance, "create_inst");
  registerTclCmd(CmdIdbCreateNet, "create_net");
  return EXIT_SUCCESS;
}

}  // namespace tcl