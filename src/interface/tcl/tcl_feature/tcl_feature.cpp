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

namespace tcl {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * feature_summary -path "xxxx.json"
 */
CmdFeatureSummary::CmdFeatureSummary(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* path_option = new TclStringOption(TCL_PATH, 1, nullptr);
  auto* step_option = new TclStringOption(TCL_STEP, 1, nullptr);
  addOption(path_option);
  addOption(step_option);
}

unsigned CmdFeatureSummary::check()
{
  TclOption* path_option = getOptionOrArg(TCL_PATH);
  // TclOption* step_option = getOptionOrArg(TCL_STEP);
  LOG_FATAL_IF(!path_option);
  //   LOG_FATAL_IF(!step_option);
  return 1;
}

unsigned CmdFeatureSummary::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_PATH);
  auto path = option->getStringVal();

  std::string step = "";
  TclOption* step_option = getOptionOrArg(TCL_STEP);
  if (step_option->getStringVal() != nullptr) {
    step = step_option->getStringVal();
  }

  featureInst->save_summary(path);

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * CmdFeatureTool -path "xxxx.json"
 */
CmdFeatureTool::CmdFeatureTool(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* path_option = new TclStringOption(TCL_PATH, 1, nullptr);
  auto* step_option = new TclStringOption(TCL_STEP, 1, nullptr);
  addOption(path_option);
  addOption(step_option);
}

unsigned CmdFeatureTool::check()
{
  TclOption* path_option = getOptionOrArg(TCL_PATH);
  TclOption* step_option = getOptionOrArg(TCL_STEP);
  LOG_FATAL_IF(!path_option);
  //   LOG_FATAL_IF(!step_option);
  return 1;
}

unsigned CmdFeatureTool::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_PATH);
  auto path = option->getStringVal();

  std::string step = "";
  TclOption* step_option = getOptionOrArg(TCL_STEP);
  if (step_option->getStringVal() != nullptr) {
    step = step_option->getStringVal();
  }

  featureInst->save_tools(path, step);

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * feature_eval_map -path "xxxx.csv"
 */
CmdFeatureEvalMap::CmdFeatureEvalMap(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* path_option = new TclStringOption(TCL_PATH, 1, nullptr);
  auto* bin_cnt_x = new TclIntOption("-bin_cnt_x", 0);
  auto* bin_cnt_y = new TclIntOption("-bin_cnt_y", 0);
  addOption(path_option);
  addOption(bin_cnt_x);
  addOption(bin_cnt_y);
}

unsigned CmdFeatureEvalMap::check()
{
  TclOption* path_option = getOptionOrArg(TCL_PATH);
  TclOption* bin_cnt_x = getOptionOrArg("-bin_cnt_x");
  TclOption* bin_cnt_y = getOptionOrArg("-bin_cnt_y");
  LOG_FATAL_IF(!path_option);
  LOG_FATAL_IF(!bin_cnt_x);
  LOG_FATAL_IF(!bin_cnt_y);
  return 1;
}

unsigned CmdFeatureEvalMap::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_PATH);
  auto* opt_bin_cnt_x = getOptionOrArg("-bin_cnt_x");
  auto* opt_bin_cnt_y = getOptionOrArg("-bin_cnt_y");
  auto path = option->getStringVal();
  auto bin_cnt_x = opt_bin_cnt_x->getIntVal();
  auto bin_cnt_y = opt_bin_cnt_y->getIntVal();

  featureInst->save_eval_map(path, bin_cnt_x, bin_cnt_y);

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * feature_route -path "xxxx.json"
 */
CmdFeatureRoute::CmdFeatureRoute(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* path_option = new TclStringOption(TCL_PATH, 1, nullptr);
  addOption(path_option);
}

unsigned CmdFeatureRoute::check()
{
  TclOption* path_option = getOptionOrArg(TCL_PATH);

  return 1;
}

unsigned CmdFeatureRoute::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_PATH);
  auto path = option->getStringVal();

  featureInst->save_route_data(path);

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * feature_route_read -path "xxxx.json"
 */
CmdFeatureRouteRead::CmdFeatureRouteRead(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* path_option = new TclStringOption(TCL_PATH, 1, nullptr);
  addOption(path_option);
}

unsigned CmdFeatureRouteRead::check()
{
  TclOption* path_option = getOptionOrArg(TCL_PATH);

  return 1;
}

unsigned CmdFeatureRouteRead::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_PATH);
  auto path = option->getStringVal();

  featureInst->read_route_data(path);

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * feature_cong_map -dir "/your/dir/path/"
 */
CmdFeatureCongMap::CmdFeatureCongMap(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* dir_option = new TclStringOption(TCL_DIRECTORY, 1, nullptr);
  auto* step_option = new TclStringOption(TCL_STEP, 1, nullptr);
  addOption(dir_option);
  addOption(step_option);
}

unsigned CmdFeatureCongMap::check()
{
  TclOption* dir_option = getOptionOrArg(TCL_DIRECTORY);
  TclOption* step_option = getOptionOrArg(TCL_STEP);

  LOG_FATAL_IF(!dir_option);
  LOG_FATAL_IF(!step_option);

  return 1;
}

unsigned CmdFeatureCongMap::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_DIRECTORY);
  auto dir_path = option->getStringVal();

  std::string stage = "";
  TclOption* step_option = getOptionOrArg(TCL_STEP);
  if (step_option->getStringVal() != nullptr) {
    stage = step_option->getStringVal();
  }

  featureInst->save_cong_map(stage, dir_path);

  return 1;
}

}  // namespace tcl
