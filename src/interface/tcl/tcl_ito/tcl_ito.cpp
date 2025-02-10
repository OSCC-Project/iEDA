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
#include "tcl_ito.h"

#include "ToApi.hpp"
#include "tool_manager.h"

namespace tcl {

CmdTOAutoRun::CmdTOAutoRun(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* file_name_option = new TclStringOption(TCL_CONFIG, 1, nullptr);
  addOption(file_name_option);
}

unsigned CmdTOAutoRun::check()
{
  TclOption* file_name_option = getOptionOrArg(TCL_CONFIG);
  LOG_FATAL_IF(!file_name_option);
  return 1;
}

unsigned CmdTOAutoRun::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_CONFIG);
  auto data_config = option->getStringVal();

  if (iplf::tmInst->autoRunTO(data_config)) {
    std::cout << "iTO auto run successfully." << std::endl;
  }

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CmdTORunDrv::CmdTORunDrv(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* file_name_option = new TclStringOption(TCL_CONFIG, 1, nullptr);
  addOption(file_name_option);
}

unsigned CmdTORunDrv::check()
{
  TclOption* file_name_option = getOptionOrArg(TCL_CONFIG);
  LOG_FATAL_IF(!file_name_option);
  return 1;
}

unsigned CmdTORunDrv::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_CONFIG);
  auto data_config = option->getStringVal();

  if (iplf::tmInst->RunTODrv(data_config)) {
    std::cout << "iTO Drv run successfully." << std::endl;
  }

  return 1;
}

CmdTORunDrvSpecialNet::CmdTORunDrvSpecialNet(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* file_name_option = new TclStringOption(TCL_CONFIG, 1, nullptr);
  auto* net_name_option = new TclStringOption(TCL_NAME, 1, nullptr);
  addOption(file_name_option);
  addOption(net_name_option);
}

unsigned CmdTORunDrvSpecialNet::check()
{
  TclOption* file_name_option = getOptionOrArg(TCL_CONFIG);
  TclOption* net_name_option = getOptionOrArg(TCL_NAME);
  LOG_FATAL_IF(!file_name_option);
  LOG_FATAL_IF(!net_name_option);
  return 1;
}

unsigned CmdTORunDrvSpecialNet::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_CONFIG);
  auto data_config = option->getStringVal();

  TclOption* name_option = getOptionOrArg(TCL_NAME);
  auto net_name = name_option->getStringVal();

  if (iplf::tmInst->RunTODrvSpecialNet(data_config, net_name)) {
    std::cout << "iTO optimize '" << net_name << "' Drv run successfully." << std::endl;
  }

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CmdTORunHold::CmdTORunHold(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* file_name_option = new TclStringOption(TCL_CONFIG, 1, nullptr);
  addOption(file_name_option);
}

unsigned CmdTORunHold::check()
{
  TclOption* file_name_option = getOptionOrArg(TCL_CONFIG);
  LOG_FATAL_IF(!file_name_option);
  return 1;
}

unsigned CmdTORunHold::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_CONFIG);
  auto data_config = option->getStringVal();
  std::cout << "iTO run Hold." << std::endl;
  if (iplf::tmInst->RunTOHold(data_config)) {
    std::cout << "iTO Hold run successfully." << std::endl;
  }

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CmdTORunSetup::CmdTORunSetup(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* file_name_option = new TclStringOption(TCL_CONFIG, 1, nullptr);
  addOption(file_name_option);
}

unsigned CmdTORunSetup::check()
{
  TclOption* file_name_option = getOptionOrArg(TCL_CONFIG);
  LOG_FATAL_IF(!file_name_option);
  return 1;
}

unsigned CmdTORunSetup::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_CONFIG);
  auto data_config = option->getStringVal();

  if (iplf::tmInst->RunTOSetup(data_config)) {
    std::cout << "iTO Setup run successfully." << std::endl;
  }

  return 1;
}

CmdTOBuffering::CmdTOBuffering(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* file_name_option = new TclStringOption(TCL_CONFIG, 1, nullptr);
  auto* net_name_option = new TclStringOption(TCL_NAME, 1, nullptr);
  addOption(file_name_option);
  addOption(net_name_option);
}

unsigned CmdTOBuffering::check()
{
  TclOption* file_name_option = getOptionOrArg(TCL_CONFIG);
  TclOption* net_name_option = getOptionOrArg(TCL_NAME);
  LOG_FATAL_IF(!file_name_option);
  LOG_FATAL_IF(!net_name_option);
  return 1;
}

unsigned CmdTOBuffering::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_CONFIG);
  auto data_config = option->getStringVal();

  TclOption* name_option = getOptionOrArg(TCL_NAME);
  auto net_name = name_option->getStringVal();

  if (iplf::tmInst->RunTOBuffering(data_config, net_name)) {
    std::cout << "iTO '" << net_name << "' buffering run successfully." << std::endl;
  }

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CmdTOApiInitConfig::CmdTOApiInitConfig(const char* cmd_name) : TclCmd(cmd_name)
{
  addOption(new TclStringOption(TCL_CONFIG, 1, nullptr));
}
unsigned CmdTOApiInitConfig::check()
{
  TclOption* opt = getOptionOrArg(TCL_CONFIG);
  if (!opt || !opt->getStringVal()) {
    std::cerr << "Please specify an ito config file";
    return 0;
  }
  return 1;
}
CMD_CLASS_DEFAULT_EXEC(CmdTOApiInitConfig, ToApiInst.init(getOptionOrArg(TCL_CONFIG)->getStringVal());)

CMD_CLASS_DEFAULT_DEFINITION(CmdTOApiRunFlow)
CMD_CLASS_DEFAULT_EXEC(CmdTOApiRunFlow, ToApiInst.runTO())

CMD_CLASS_DEFAULT_DEFINITION(CmdTOApiOptDRV);
CMD_CLASS_DEFAULT_EXEC(CmdTOApiOptDRV, ToApiInst.optimizeDrv());

CMD_CLASS_DEFAULT_DEFINITION(CmdTOApiOptHold);
CMD_CLASS_DEFAULT_EXEC(CmdTOApiOptHold, ToApiInst.optimizeHold());

CMD_CLASS_DEFAULT_DEFINITION(CmdTOApiOptSetup);
CMD_CLASS_DEFAULT_EXEC(CmdTOApiOptSetup, ToApiInst.optimizeSetup());

CmdTOApiSaveDef::CmdTOApiSaveDef(const char* cmd_name) : TclCmd(cmd_name)
{
  addOption(new TclStringOption(TCL_PATH, 1, nullptr));
}
unsigned CmdTOApiSaveDef::check()
{
  TclOption* opt = getOptionOrArg(TCL_PATH);
  if (!opt || !opt->getStringVal()) {
    std::cerr << "Please specify save path.";
    return 0;
  }
  return 1;
}
CMD_CLASS_DEFAULT_EXEC(CmdTOApiSaveDef, ToApiInst.saveDef(getOptionOrArg(TCL_PATH)->getStringVal()));

CMD_CLASS_DEFAULT_DEFINITION(CmdTOApiReportTiming);
CMD_CLASS_DEFAULT_EXEC(CmdTOApiReportTiming, ToApiInst.reportTiming())
}  // namespace tcl
