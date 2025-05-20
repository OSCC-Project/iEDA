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
#include "tcl_db_file.h"

#include "idm.h"
#include "report_manager.h"
#include "tool_manager.h"
namespace tcl {

CmdInitIdb::CmdInitIdb(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* option = new TclStringOption(TCL_CONFIG, 1, nullptr);

  addOption(option);
}

unsigned CmdInitIdb::check()
{
  TclOption* option = getOptionOrArg(TCL_CONFIG);

  LOG_FATAL_IF(!option);

  return 1;
}

unsigned CmdInitIdb::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_CONFIG);

  auto data_config = option->getStringVal();

  if (iplf::tmInst->idbStart(data_config)) {
    std::cout << "idb start." << std::endl;
  }

  return 1;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CmdInitTechLef::CmdInitTechLef(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* option = new TclStringOption(TCL_PATH, 1, nullptr);
  addOption(option);
}

unsigned CmdInitTechLef::check()
{
  //   TclOption* option = getOptionOrArg(TCL_PATH);
  //   LOG_FATAL_IF(!option);
  return 1;
}

unsigned CmdInitTechLef::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_PATH);
  auto path = option->getStringVal();
  if (path != nullptr) {
    vector<string> path_list;
    path_list.push_back(path);
    dmInst->get_config().set_tech_lef_path(path);
    dmInst->readLef(path_list, true);
    return 1;
  } else {
    if (dmInst->get_config().get_tech_lef_path() != "") {
      vector<string> path_list;
      path_list.push_back(dmInst->get_config().get_tech_lef_path());
      dmInst->readLef(path_list, true);
    }
  }

  return 1;
}

CmdInitLef::CmdInitLef(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* name_list = new TclStringListOption(TCL_PATH, 1);
  addOption(name_list);
}

unsigned CmdInitLef::check()
{
  //   TclOption* name_list = getOptionOrArg(TCL_PATH);
  //   LOG_FATAL_IF(!name_list);
  return 1;
}

unsigned CmdInitLef::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* name_list_option = getOptionOrArg(TCL_PATH);
  auto lef_path = name_list_option->getStringList();
  if (!lef_path.empty()) {
    dmInst->get_config().set_lef_paths(lef_path);
    dmInst->readLef(lef_path);
    return 1;
  } else {
    if (dmInst->get_config().get_lef_paths().size() > 0) {
      dmInst->readLef(dmInst->get_config().get_lef_paths());
    }
  }

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CmdInitDef::CmdInitDef(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* path = new TclStringOption(TCL_PATH, 1);
  addOption(path);
}

unsigned CmdInitDef::check()
{
  TclOption* path = getOptionOrArg(TCL_PATH);
  LOG_FATAL_IF(!path);
  return 1;
}

unsigned CmdInitDef::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* def_name = getOptionOrArg(TCL_PATH);
  auto def_path = def_name->getStringVal();
  if (def_path != nullptr) {
    dmInst->get_config().set_def_path(def_path);
    dmInst->readDef(def_path);
    return 1;
  }
  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CmdInitVerilog::CmdInitVerilog(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* path = new TclStringOption(TCL_PATH, 1);
  auto* top = new TclStringOption(TCL_VERILOG_TOP, 1);
  addOption(path);
  addOption(top);
}

unsigned CmdInitVerilog::check()
{
  TclOption* path = getOptionOrArg(TCL_PATH);
  TclOption* top = getOptionOrArg(TCL_VERILOG_TOP);
  LOG_FATAL_IF(!path);
  LOG_FATAL_IF(!top);
  return 1;
}

unsigned CmdInitVerilog::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* path = getOptionOrArg(TCL_PATH);
  TclOption* top = getOptionOrArg(TCL_VERILOG_TOP);

  auto path_string = path->getStringVal();
  auto top_module = top->getStringVal();
  if (path_string != nullptr && top_module != nullptr) {
    dmInst->get_config().set_verilog_path(path_string);
    dmInst->readVerilog(path_string, top_module);
    return 1;
  }

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CmdSaveDef::CmdSaveDef(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* option = new TclStringOption(TCL_NAME, 1, nullptr);
  addOption(option);

  auto* path = new TclStringOption(TCL_PATH, 1);
  addOption(path);
}

unsigned CmdSaveDef::check()
{
  TclOption* option = getOptionOrArg(TCL_NAME);
  LOG_FATAL_IF(!option);

  TclOption* path = getOptionOrArg(TCL_PATH);
  LOG_FATAL_IF(!path);
  return 1;
}

unsigned CmdSaveDef::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_NAME);
  auto name = option->getStringVal();
  if (name != nullptr) {
    if (iplf::tmInst->idbSave(name)) {
      std::cout << "idb save success." << std::endl;
      return 1;
    }
  }

  TclOption* def_path = getOptionOrArg(TCL_PATH);
  auto str_path = def_path->getStringVal();
  if (str_path != nullptr) {
    dmInst->saveDef(str_path);
    return 1;
  }

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CmdSaveLef::CmdSaveLef(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* path = new TclStringOption(TCL_PATH, 1);
  addOption(path);
}

unsigned CmdSaveLef::check()
{
  TclOption* path = getOptionOrArg(TCL_PATH);
  LOG_FATAL_IF(!path);
  return 1;
}

unsigned CmdSaveLef::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* path = getOptionOrArg(TCL_PATH);
  auto str_path = path->getStringVal();
  if (str_path != nullptr) {
    dmInst->saveLef(str_path);
    return 1;
  }

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CmdSaveNetlist::CmdSaveNetlist(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* option = new TclStringOption(TCL_NAME, 1, nullptr);
  addOption(option);

  auto* path = new TclStringOption(TCL_PATH, 1);
  addOption(path);

  auto* exclude_cell_names = new TclStringListOption(EXCLUDE_CELL_NAMES, 1, {});
  addOption(exclude_cell_names);

  auto* is_add_space = new TclSwitchOption(TCL_ADD_SPACE);
  addOption(is_add_space);
}

unsigned CmdSaveNetlist::check()
{
  TclOption* option = getOptionOrArg(TCL_NAME);
  LOG_FATAL_IF(!option);

  TclOption* path = getOptionOrArg(TCL_PATH);
  LOG_FATAL_IF(!path);

  TclOption* exclude_cell_names = getOptionOrArg(EXCLUDE_CELL_NAMES);
  LOG_FATAL_IF(!exclude_cell_names);

  TclOption* is_add_space = getOptionOrArg(TCL_ADD_SPACE);
  LOG_FATAL_IF(!is_add_space);

  return 1;
}

unsigned CmdSaveNetlist::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_NAME);
  auto name = option->getStringVal();
  if (name != nullptr) {
    if (iplf::tmInst->idbSave(name)) {
      std::cout << "idb save success." << std::endl;
      return 1;
    }
  }

  TclOption* verilog_path = getOptionOrArg(TCL_PATH);
  auto str_path = verilog_path->getStringVal();

  TclOption* exclude_cell_names_option = getOptionOrArg(EXCLUDE_CELL_NAMES);
  auto exclude_cell_names = exclude_cell_names_option->getStringList();

  std::set<std::string> new_exclude_cell_names;
  for (auto& exclude_cell_name : exclude_cell_names) {
    new_exclude_cell_names.insert(exclude_cell_name);
  }

  TclOption* is_add_space_option = getOptionOrArg(TCL_ADD_SPACE);
  bool is_add_space = false;
  if (is_add_space_option->is_set_val()) {
    is_add_space = true;
  }

  if (str_path != nullptr) {
    dmInst->saveVerilog(str_path, std::move(new_exclude_cell_names), is_add_space);
    return 1;
  }

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CmdSaveGDS::CmdSaveGDS(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* option = new TclStringOption(TCL_NAME, 1, nullptr);
  addOption(option);

  auto* path = new TclStringOption(TCL_PATH, 1);
  addOption(path);
}

unsigned CmdSaveGDS::check()
{
  TclOption* option = getOptionOrArg(TCL_NAME);
  LOG_FATAL_IF(!option);

  TclOption* path = getOptionOrArg(TCL_PATH);
  LOG_FATAL_IF(!path);
  return 1;
}

unsigned CmdSaveGDS::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* def_path = getOptionOrArg(TCL_PATH);
  auto str_path = def_path->getStringVal();
  if (str_path != nullptr) {
    dmInst->saveGDSII(str_path);
    return 1;
  }
  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CmdSaveJSON::CmdSaveJSON(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* option = new TclStringOption(TCL_NAME, 1, nullptr);
  addOption(option);

  auto* path = new TclStringOption(TCL_PATH, 1);
  addOption(path);

  auto* discard = new TclStringOption(TCL_JSON_OPTION, 1, nullptr);
  addOption(discard);
}

unsigned CmdSaveJSON::check()
{
  TclOption* option = getOptionOrArg(TCL_NAME);
  LOG_FATAL_IF(!option);

  TclOption* discard = getOptionOrArg(TCL_JSON_OPTION);
  LOG_FATAL_IF(!discard);

  TclOption* path = getOptionOrArg(TCL_PATH);
  LOG_FATAL_IF(!path);
  return 1;
}

unsigned CmdSaveJSON::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* def_path = getOptionOrArg(TCL_PATH);
  TclOption* discard = getOptionOrArg(TCL_JSON_OPTION);
  auto str_path = def_path->getStringVal();
  auto str_option = discard->getStringVal();
  // std::cout<<str_path<<std::endl;
  if (str_path != nullptr) {
    dmInst->saveJSON(str_path, str_option);
    return 1;
  }

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CmdGenerateMPScript::CmdGenerateMPScript(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* dir = new TclStringOption(TCL_DIRECTORY, 1);
  auto* name = new TclStringOption(TCL_NAME, 1, nullptr);
  auto* number = new TclIntOption("-number", 0);
  addOption(dir);
  addOption(name);
  addOption(number);
}

unsigned CmdGenerateMPScript::check()
{
  TclOption* dir = getOptionOrArg(TCL_DIRECTORY);
  LOG_FATAL_IF(!dir);

  TclOption* name = getOptionOrArg(TCL_NAME);
  LOG_FATAL_IF(!name);

  auto* number = new TclIntOption("-number", 0);
  LOG_FATAL_IF(!number);
  return 1;
}
/*
example script : aimp_random -dir ./result/mp -name ariane_macro_loc -number 100
*/
unsigned CmdGenerateMPScript::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* dir = getOptionOrArg(TCL_DIRECTORY);
  TclOption* name = getOptionOrArg(TCL_NAME);
  TclOption* number = getOptionOrArg("-number");
  auto str_dir = dir->getStringVal();
  auto str_name = name->getStringVal();
  auto int_number = number->getIntVal();
  if (str_dir != nullptr && str_name != nullptr) {
    dmInst->place_macro_generate_tcl(str_dir, str_name, int_number);
    return 1;
  }

  return 1;
}

}  // namespace tcl
