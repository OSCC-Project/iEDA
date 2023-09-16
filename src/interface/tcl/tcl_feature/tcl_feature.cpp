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
 * @File Name: tcl_flow.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */
#include "tcl_feature.h"

#include <iostream>

#include "feature_manager.h"
#include "idm.h"

namespace tcl {

CmdFeatureGenerateLayout::CmdFeatureGenerateLayout(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* path_option = new TclStringOption(TCL_PATH, 1, nullptr);
  addOption(path_option);
}

unsigned CmdFeatureGenerateLayout::check()
{
  TclOption* path_option = getOptionOrArg(TCL_PATH);
  LOG_FATAL_IF(!path_option);
  return 1;
}

unsigned CmdFeatureGenerateLayout::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_PATH);
  auto path = option->getStringVal();

  iplf::FeatureManager feature_parser(dmInst->get_idb_layout(), dmInst->get_idb_design());

  feature_parser.save_layout(path);

  return 1;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CmdFeatureGenerateInstances::CmdFeatureGenerateInstances(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* path_option = new TclStringOption(TCL_PATH, 1, nullptr);
  addOption(path_option);
}

unsigned CmdFeatureGenerateInstances::check()
{
  TclOption* path_option = getOptionOrArg(TCL_PATH);
  LOG_FATAL_IF(!path_option);
  return 1;
}

unsigned CmdFeatureGenerateInstances::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_PATH);
  auto path = option->getStringVal();

  iplf::FeatureManager feature_parser(dmInst->get_idb_layout(), dmInst->get_idb_design());

  feature_parser.save_instances(path);

  return 1;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CmdFeatureGenerateNets::CmdFeatureGenerateNets(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* path_option = new TclStringOption(TCL_PATH, 1, nullptr);
  addOption(path_option);
}

unsigned CmdFeatureGenerateNets::check()
{
  TclOption* path_option = getOptionOrArg(TCL_PATH);
  LOG_FATAL_IF(!path_option);
  return 1;
}

unsigned CmdFeatureGenerateNets::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_PATH);
  auto path = option->getStringVal();

  iplf::FeatureManager feature_parser(dmInst->get_idb_layout(), dmInst->get_idb_design());

  feature_parser.save_nets(path);

  return 1;
}

}  // namespace tcl
