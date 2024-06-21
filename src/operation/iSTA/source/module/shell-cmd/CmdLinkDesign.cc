// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file CmdLinkDesign.cc
 * @author Wang Hao (harry0789@qq.com)
 * @brief
 * @version 0.1
 * @date 2021-10-12
 */
#include "ShellCmd.hh"
#include "sta/Sta.hh"

namespace ista {
CmdLinkDesign::CmdLinkDesign(const char* cmd_name) : TclCmd(cmd_name) {
  auto* cell_name_option = new TclStringOption("cell_name", 1, nullptr);
  addOption(cell_name_option);
}

unsigned CmdLinkDesign::check() {
  TclOption* cell_name_option = getOptionOrArg("cell_name");
  LOG_FATAL_IF(!cell_name_option);
  return 1;
}

unsigned CmdLinkDesign::exec() {
  if (!check()) {
    return 0;
  }

  TclOption* cell_name_option = getOptionOrArg("cell_name");
  auto* cell_name = cell_name_option->getStringVal();

  Sta* ista = Sta::getOrCreateSta();
  ista->set_top_module_name(cell_name);

  ista->linkDesignWithRustParser(cell_name);

  return 1;
}
}  // namespace ista