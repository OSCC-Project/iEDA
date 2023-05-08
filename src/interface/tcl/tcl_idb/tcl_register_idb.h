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
  registerTclCmd(CmdSaveGDS, "gds_save");

  /// idb operator
  registerTclCmd(CmdIdbSetNet, "set_net");

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