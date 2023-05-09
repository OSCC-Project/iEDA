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
 * @File Name: tcl_config.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */
#include "tcl_config.h"

#include <iostream>

#include "dm_config.h"
#include "flow.h"
#include "idm.h"

namespace tcl {

CmdFlowInitConfig::CmdFlowInitConfig(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* file_name_option = new TclStringOption(TCL_CONFIG, 1, nullptr);
  addOption(file_name_option);
}

unsigned CmdFlowInitConfig::check()
{
  TclOption* file_name_option = getOptionOrArg(TCL_CONFIG);
  LOG_FATAL_IF(!file_name_option);
  return 1;
}

unsigned CmdFlowInitConfig::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_CONFIG);
  auto data_config = option->getStringVal();
  iplf::plfInst->initFlow(data_config);

  std::cout << "Init Flow Config success." << std::endl;
  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CmdDbConfigSetting::CmdDbConfigSetting(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* tcl_option = new TclStringOption(TCL_CONFIG, 1, nullptr);
  addOption(tcl_option);

  tcl_option = new TclStringOption("-tech_lef_path", 1, nullptr);
  addOption(tcl_option);

  tcl_option = new TclStringOption("-def_path", 1, nullptr);
  addOption(tcl_option);

  tcl_option = new TclStringOption("-verilog_path", 1, nullptr);
  addOption(tcl_option);

  tcl_option = new TclStringOption("-output_dir_path", 1, nullptr);
  addOption(tcl_option);

  tcl_option = new TclStringOption("-sdc_path", 1, nullptr);
  addOption(tcl_option);

  auto* lef_list = new TclStringListOption("-lef_paths", 1);
  addOption(lef_list);

  auto* lib_list = new TclStringListOption("-lib_path", 1);
  addOption(lib_list);

  tcl_option = new TclStringOption("-spef_path", 1, nullptr);
  addOption(tcl_option);

  tcl_option = new TclStringOption("-routing_layer_1st", 1, nullptr);
  addOption(tcl_option);
}

unsigned CmdDbConfigSetting::check()
{
  // TclOption* file_name_option = getOptionOrArg(TCL_CONFIG);
  // LOG_FATAL_IF(!file_name_option);
  return 1;
}

unsigned CmdDbConfigSetting::exec()
{
  if (!check()) {
    return 0;
  }

  idm::DataConfig& dm_config = dmInst->get_config();

  TclOption* tcl_option = nullptr;

  tcl_option = getOptionOrArg(TCL_CONFIG);
  if (tcl_option != nullptr) {
    auto path = tcl_option->getStringVal();
    if (path != nullptr) {
      dm_config.initConfig(path);
    }
  }

  tcl_option = getOptionOrArg("-tech_lef_path");
  if (tcl_option != nullptr) {
    auto path = tcl_option->getStringVal();
    if (path != nullptr) {
      dm_config.set_tech_lef_path(path);
    }
  }

  TclOption* lef_list_option = getOptionOrArg("-lef_paths");
  if (lef_list_option != nullptr) {
    auto lef_path = lef_list_option->getStringList();
    if (!lef_path.empty()) {
      dm_config.set_lef_paths(lef_path);
    }
  }

  tcl_option = getOptionOrArg("-def_path");
  if (tcl_option != nullptr) {
    auto path = tcl_option->getStringVal();
    if (path != nullptr) {
      dm_config.set_def_path(path);
    }
  }

  tcl_option = getOptionOrArg("-verilog_path");
  if (tcl_option != nullptr) {
    auto path = tcl_option->getStringVal();
    if (path != nullptr) {
      dm_config.set_verilog_path(path);
    }
  }

  tcl_option = getOptionOrArg("-output_dir_path");
  if (tcl_option != nullptr) {
    auto path = tcl_option->getStringVal();
    if (path != nullptr) {
      dm_config.set_output_path(path);
    }
  }

  TclOption* lib_list_option = getOptionOrArg("-lib_path");
  if (lib_list_option != nullptr) {
    auto lib_path = lib_list_option->getStringList();
    if (!lib_path.empty()) {
      dm_config.set_lib_paths(lib_path);
    }
  }

  tcl_option = getOptionOrArg("-sdc_path");
  if (tcl_option != nullptr) {
    auto path = tcl_option->getStringVal();
    if (path != nullptr) {
      dm_config.set_sdc_path(path);
    }
  }

  tcl_option = getOptionOrArg("-routing_layer_1st");
  if (tcl_option != nullptr) {
    auto value = tcl_option->getStringVal();
    if (value != nullptr) {
      dm_config.set_routing_layer_1st(value);
    }
  }

  tcl_option = getOptionOrArg("-spef_path");
  if (tcl_option != nullptr) {
    auto path = tcl_option->getStringVal();
    if (path != nullptr) {
      dm_config.set_spef_path(path);
    }
  }

  return 1;
}

}  // namespace tcl
