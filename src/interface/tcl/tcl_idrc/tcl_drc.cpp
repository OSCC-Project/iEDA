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
#include "tcl_drc.h"

#include "tool_manager.h"

namespace tcl {

CmdDRCAutoRun::CmdDRCAutoRun(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* file_name_option = new TclStringOption(TCL_CONFIG, 1, nullptr);
  auto* file_path_option = new TclStringOption(TCL_PATH, 1, nullptr);
  addOption(file_name_option);
  addOption(file_path_option);
}

unsigned CmdDRCAutoRun::check()
{
  TclOption* file_name_option = getOptionOrArg(TCL_CONFIG);
  TclOption* file_path_option = getOptionOrArg(TCL_PATH);
  LOG_FATAL_IF(!file_name_option);
  LOG_FATAL_IF(!file_path_option);
  return 1;
}

unsigned CmdDRCAutoRun::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_CONFIG);
  auto data_config = option->getStringVal();

  TclOption* path_option = getOptionOrArg(TCL_PATH);
  auto data_path = path_option->getStringVal();

  if (iplf::tmInst->autoRunDRC(data_config, data_path)) {
    std::cout << "iDRC run successfully." << std::endl;
  }

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CmdDRCSaveDetailFile::CmdDRCSaveDetailFile(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* file_path_option = new TclStringOption(TCL_PATH, 1, nullptr);
  addOption(file_path_option);
}

unsigned CmdDRCSaveDetailFile::check()
{
  TclOption* file_path_option = getOptionOrArg(TCL_PATH);
  LOG_FATAL_IF(!file_path_option);
  return 1;
}

unsigned CmdDRCSaveDetailFile::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* path_option = getOptionOrArg(TCL_PATH);
  auto data_path = path_option->getStringVal();

  if (iplf::tmInst->saveDrcDetailToFile(data_path)) {
    std::cout << "iDRC save detail drc to file success. path = " << data_path << std::endl;
  }

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CmdDRCDiagnosis::CmdDRCDiagnosis(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* third_json_file_option = new TclStringOption("-third_json_file", 1, nullptr);
  addOption(third_json_file_option);
  auto* idrc_json_file_option = new TclStringOption("-idrc_json_file", 1, nullptr);
  addOption(idrc_json_file_option);
}

unsigned CmdDRCDiagnosis::check()
{
  TclOption* third_json_file_option = getOptionOrArg("-third_json_file");
  LOG_FATAL_IF(!third_json_file_option);
  TclOption* idrc_json_file_option = getOptionOrArg("-idrc_json_file");
  LOG_FATAL_IF(!idrc_json_file_option);
  return 1;
}

unsigned CmdDRCDiagnosis::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* third_json_file_option = getOptionOrArg("-third_json_file");
  auto third_json_file = third_json_file_option->getStringVal();
  TclOption* idrc_json_file_option = getOptionOrArg("-idrc_json_file");
  auto idrc_json_file = idrc_json_file_option->getStringVal();

  idrc::DrcApi drc_api;
  drc_api.diagnosis(third_json_file, idrc_json_file);

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CmdDRCReadDetailFile::CmdDRCReadDetailFile(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* file_path_option = new TclStringOption(TCL_PATH, 1, nullptr);
  addOption(file_path_option);
}

unsigned CmdDRCReadDetailFile::check()
{
  TclOption* file_path_option = getOptionOrArg(TCL_PATH);

  LOG_FATAL_IF(!file_path_option);
  return 1;
}

unsigned CmdDRCReadDetailFile::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* path_option = getOptionOrArg(TCL_PATH);
  auto data_path = path_option->getStringVal();

  if (iplf::tmInst->readDrcDetailFromFile(data_path)) {
    std::cout << "iDRC read detail file successfully." << std::endl;
  }

  return 1;
}

}  // namespace tcl
