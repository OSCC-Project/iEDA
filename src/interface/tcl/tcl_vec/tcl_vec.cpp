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
 * @File Name: tcl_eco.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2024-08-26
 *
 */
#include "tcl_vec.h"

#include <iostream>

#include "vec_api.h"

namespace tcl {
/**
 * type : shape, pattern
 * default type : shape
 */
CmdVecLayoutPatchs::CmdVecLayoutPatchs(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* option = new TclStringOption(TCL_PATH, 1, nullptr);
  addOption(option);
}

unsigned CmdVecLayoutPatchs::check()
{
  TclOption* option = getOptionOrArg(TCL_PATH);
  LOG_FATAL_IF(!option);

  return 1;
}

unsigned CmdVecLayoutPatchs::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_PATH);
  if (option != nullptr) {
    auto path_option = option->getStringVal();
    std::string path = path_option == nullptr ? "./layout_patchs.csv" : path_option;

    ivec::VectorizationApi vec_api;
    vec_api.buildVectorizationLayoutData(path);
  }

  return 1;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CmdVecLayoutGraph::CmdVecLayoutGraph(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* option = new TclStringOption(TCL_PATH, 1, nullptr);
  addOption(option);
}

unsigned CmdVecLayoutGraph::check()
{
  TclOption* option = getOptionOrArg(TCL_PATH);
  LOG_FATAL_IF(!option);

  return 1;
}

unsigned CmdVecLayoutGraph::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* option = getOptionOrArg(TCL_PATH);
  if (option != nullptr) {
    auto path_option = option->getStringVal();
    std::string path = path_option == nullptr ? "./layout_graph.json" : path_option;

    ivec::VectorizationApi vec_api;
    vec_api.buildVectorizationGraphData(path);
  }

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CmdVecFeature::CmdVecFeature(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* dir_option = new TclStringOption(TCL_DIRECTORY, 1, nullptr);
  addOption(dir_option);

  auto* row_step_option = new TclIntOption(TCL_PATCH_ROW_STEP, 0, 9);  
  addOption(row_step_option);

  auto* col_step_option = new TclIntOption(TCL_PATCH_COL_STEP, 0, 9);  
  addOption(col_step_option);
}

unsigned CmdVecFeature::check()
{
  TclOption* dir_option = getOptionOrArg(TCL_DIRECTORY);
  LOG_FATAL_IF(!dir_option); 

  TclOption* row_step_option = getOptionOrArg(TCL_PATCH_ROW_STEP);
  if (row_step_option && row_step_option->getIntVal() <= 0) {
    LOG_ERROR << "patch_row_step must be positive";
    return 0;
  }

  TclOption* col_step_option = getOptionOrArg(TCL_PATCH_COL_STEP);
  if (col_step_option && col_step_option->getIntVal() <= 0) {
    LOG_ERROR << "patch_col_step must be positive";
    return 0;
  }

  return 1;
}

unsigned CmdVecFeature::exec()
{
  if (!check()) {
    return 0;
  }

  TclOption* dir_option = getOptionOrArg(TCL_DIRECTORY);
  if (dir_option != nullptr) {
    auto path_option = dir_option->getStringVal();
    std::string path = path_option == nullptr ? "./vectors" : path_option;

    int patch_row_step = 9; 
    TclOption* row_step_option = getOptionOrArg(TCL_PATCH_ROW_STEP);
    if (row_step_option != nullptr) {
      patch_row_step = row_step_option->getIntVal();
    }

    int patch_col_step = 9;  
    TclOption* col_step_option = getOptionOrArg(TCL_PATCH_COL_STEP);
    if (col_step_option != nullptr) {
      patch_col_step = col_step_option->getIntVal();
    }

    ivec::VectorizationApi vec_api;
    vec_api.buildVectorizationFeature(path, patch_row_step, patch_col_step);
  }

  return 1;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace tcl
