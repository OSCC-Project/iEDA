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
 * @File Name: tcl_gui.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */
#include "tcl_gui.h"

#include "gui_io.h"
#include "vec_api.h"
#include "tool_manager.h"

namespace tcl {

CmdGuiStart::CmdGuiStart(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* type = new TclStringOption(TCL_TYPE, 1, nullptr);
  addOption(type);
}

unsigned CmdGuiStart::check()
{
  // TclOption *file_name_option = getOptionOrArg("-path");
  // LOG_FATAL_IF(!file_name_option);
  return 1;
}

unsigned CmdGuiStart::exec()
{
  if (!check()) {
    return 0;
  }

  std::cout << "gui start." << std::endl;

  std::string type = "";
  TclOption* opt = getOptionOrArg(TCL_TYPE);
  if (opt->getStringVal() != nullptr) {
    type = opt->getStringVal();
  }

  iplf::tmInst->guiStart(type);

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CmdGuiShow::CmdGuiShow(const char* cmd_name) : TclCmd(cmd_name)
{
  // auto *file_name_option = new TclStringOption("-path", 1, nullptr);
  // addOption(file_name_option);
}

unsigned CmdGuiShow::check()
{
  // TclOption *file_name_option = getOptionOrArg("-path");
  // LOG_FATAL_IF(!file_name_option);
  return 1;
}

unsigned CmdGuiShow::exec()
{
  if (!check()) {
    return 0;
  }

  std::cout << "gui show." << std::endl;

  iplf::tmInst->guiShow();

  return 1;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CmdGuiHide::CmdGuiHide(const char* cmd_name) : TclCmd(cmd_name)
{
  // auto *file_name_option = new TclStringOption("-path", 1, nullptr);
  // addOption(file_name_option);
}

unsigned CmdGuiHide::check()
{
  // TclOption *file_name_option = getOptionOrArg("-path");
  // LOG_FATAL_IF(!file_name_option);
  return 1;
}

unsigned CmdGuiHide::exec()
{
  if (!check()) {
    return 0;
  }

  std::cout << "gui hide." << std::endl;

  iplf::tmInst->guiHide();

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CmdGuiShowDrc::CmdGuiShowDrc(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* path_opt = new TclStringOption(TCL_PATH, 1, nullptr);
  addOption(path_opt);
  auto* max_num_opt = new TclIntOption(TCL_MAX_NUM, 0);
  addOption(max_num_opt);
}

unsigned CmdGuiShowDrc::check()
{
  // TclOption *file_name_option = getOptionOrArg("-path");
  // LOG_FATAL_IF(!file_name_option);
  return 1;
}

unsigned CmdGuiShowDrc::exec()
{
  if (!check()) {
    return 0;
  }

  std::string path = "";
  TclOption* path_opt = getOptionOrArg(TCL_PATH);
  if (path_opt != nullptr) {
    path = path_opt->getStringVal();
  }

  TclOption* max_num_opt = getOptionOrArg(TCL_MAX_NUM);
  int max_num = -1;
  if (max_num_opt != nullptr) {
    max_num = max_num_opt->getIntVal();
  }

  iplf::tmInst->guiShowDrc(path, max_num);

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CmdGuiShowClockTree::CmdGuiShowClockTree(const char* cmd_name) : TclCmd(cmd_name)
{
  //   auto* path_opt = new TclStringOption(TCL_PATH, 1, nullptr);
  //   addOption(path_opt);
}

unsigned CmdGuiShowClockTree::check()
{
  // TclOption *file_name_option = getOptionOrArg("-path");
  // LOG_FATAL_IF(!file_name_option);
  return 1;
}

unsigned CmdGuiShowClockTree::exec()
{
  if (!check()) {
    return 0;
  }

  iplf::tmInst->guiShowClockTree();

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CmdGuiShowPlacement::CmdGuiShowPlacement(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* dir_opt = new TclStringOption(TCL_DIRECTORY, 1, nullptr);
  addOption(dir_opt);
}

unsigned CmdGuiShowPlacement::check()
{
  TclOption* dir_opt = getOptionOrArg(TCL_DIRECTORY);
  LOG_FATAL_IF(!dir_opt);
  return 1;
}

unsigned CmdGuiShowPlacement::exec()
{
  if (!check()) {
    return 0;
  }

  std::string dir = "";
  TclOption* opt = getOptionOrArg(TCL_DIRECTORY);
  if (opt != nullptr) {
    dir = opt->getStringVal();
  }

  guiInst->guiUpdateInstanceInFastMode(dir, true);

  iplf::tmInst->guiTimerStart(10);

  return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
CmdGuiShowGraph::CmdGuiShowGraph(const char* cmd_name) : TclCmd(cmd_name)
{
  auto* path_opt = new TclStringOption(TCL_PATH, 1, nullptr);
  addOption(path_opt);
}

unsigned CmdGuiShowGraph::check()
{
  // TclOption *file_name_option = getOptionOrArg("-path");
  // LOG_FATAL_IF(!file_name_option);
  return 1;
}

unsigned CmdGuiShowGraph::exec()
{
  if (!check()) {
    return 0;
  }

  std::string path = "";
  TclOption* path_opt = getOptionOrArg(TCL_PATH);
  if (path_opt != nullptr) {
    path = path_opt->getStringVal();
  }

  ivec::VectorizationApi lm_api;
  auto data = lm_api.getGraph(path);

  iplf::tmInst->guiShowGraph(data);

  return 1;
}

}  // namespace tcl
