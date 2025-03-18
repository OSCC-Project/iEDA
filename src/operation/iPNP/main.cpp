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
/**
 * @file main.cpp
 * @author Xinhao li
 * @brief The main function of iPNP. function: Launch the tcl console and interact with the user. Refer to iSTA/main.cc.
 * @version 0.1
 * @date 2024-07-15
 */

#include <iostream>
#include <string>
#include <filesystem>

#include "iPNP.hh"
#include "api/iPNPApi.hh"

using namespace ipnp;
using namespace idb;

// int registerCommands() {
//   registerTclCmd(CmdSetDesignWorkSpace, "set_design_workspace");
//   registerTclCmd(CmdReadVerilog, "read_netlist");
//   registerTclCmd(CmdReadLefDef, "read_lef_def");
//   registerTclCmd(CmdReadLiberty, "read_liberty");
//   registerTclCmd(CmdLinkDesign, "link_design");
//   registerTclCmd(CmdReadSpef, "read_spef");
//   registerTclCmd(CmdReadSdc, "read_sdc");
//   registerTclCmd(CmdReportTiming, "report_timing");
//   registerTclCmd(CmdReportConstraint, "report_constraint");
//   registerTclCmd(CmdDefToVerilog, "def_to_verilog");
//   registerTclCmd(CmdVerilogToDef, "verilog_to_def");

//   return EXIT_SUCCESS;
// }

int main(int argc, char** argv)
{
  std::vector<std::string> lef_files{
    "/home/taosimin/T28/tlef/tsmcn28_9lm6X2ZUTRDL.tlef",
    "/home/taosimin/T28/lef/PLLTS28HPMLAINT.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140opplvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140lvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140uhvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140hvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140oppuhvt.lef",
    "/home/taosimin/T28/lef/ts5n28hpcplvta256x32m4fw_130a.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140cg.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140oppuhvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140mbhvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140ulvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140uhvt.lef",
    "/home/taosimin/T28/lef/ts5n28hpcplvta64x100m2fw_130a.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140hvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140oppulvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140mb.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140cgcwhvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140lvt.lef",
    "/home/taosimin/T28/lef/tpbn28v_9lm.lef",
    "/home/taosimin/T28/lef/ts5n28hpcplvta64x128m2f_130a.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140uhvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140mblvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140cgcw.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140mbhvt.lef",
    "/home/taosimin/T28/lef/tpbn28v.lef",
    "/home/taosimin/T28/lef/ts5n28hpcplvta64x128m2fw_130a.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140lvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140ulvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140opphvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140cgehvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140mb.lef",
    "/home/taosimin/T28/lef/tphn28hpcpgv18_9lm.lef",
    "/home/taosimin/T28/lef/ts5n28hpcplvta64x88m2fw_130a.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140mb.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140cghvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140opp.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140cghvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140oppehvt.lef",
    "/home/taosimin/T28/lef/ts1n28hpcplvtb2048x48m8sw_180a.lef",
    "/home/taosimin/T28/lef/ts5n28hpcplvta64x92m2fw_130a.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140mblvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140cg.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140opplvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140cg.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140opphvt.lef",
    "/home/taosimin/T28/lef/ts1n28hpcplvtb512x128m4sw_180a.lef",
    "/home/taosimin/T28/lef/ts5n28hpcplvta64x96m2fw_130a.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140opphvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140hvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140oppuhvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140cguhvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140opp.lef",
    "/home/taosimin/T28/lef/ts1n28hpcplvtb512x64m4sw_180a.lef",
    "/home/taosimin/T28/lef/ts6n28hpcplvta2048x32m8sw_130a.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp30p140opp.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp35p140oppulvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140ehvt.lef",
    "/home/taosimin/T28/lef/tcbn28hpcplusbwp40p140opplvt.lef",
    "/home/taosimin/T28/lef/ts1n28hpcplvtb8192x64m8sw_180a.lef" };

  std::string def_path = "/home/taosimin/ir_example/aes/aes.def";

  pnpApiInst->initializeiPNP("");
  pnpApiInst->readDeftoiPNP(lef_files, def_path);

  iPNP* ipnp = pnpApiInst->get_ipnp();
  ipnp->set_output_def_path("/home/sujianrong/iEDA/src/operation/iPNP/data/test/output.def");
  ipnp->run();

  return 0;
}
