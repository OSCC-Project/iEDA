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
 * @file CmdReadVerilog.cc
 * @author Wang Hao (harry0789@qq.com)
 * @brief
 * @version 0.1
 * @date 2021-10-12
 */
#include "ShellCmd.hh"
#include "sta/Sta.hh"

namespace ista {
CmdReadVerilog::CmdReadVerilog(const char* cmd_name) : TclCmd(cmd_name) {
  auto* file_name_option = new TclStringOption("file_name", 1, nullptr);
  addOption(file_name_option);
}

unsigned CmdReadVerilog::check() {
  TclOption* file_name_option = getOptionOrArg("file_name");
  LOG_FATAL_IF(!file_name_option);

  return 1;
}

unsigned CmdReadVerilog::exec() {
  if (!check()) {
    return 0;
  }

  TclOption* file_name_option = getOptionOrArg("file_name");
  auto* verilog_file = file_name_option->getStringVal();

  Sta* ista = Sta::getOrCreateSta();

  return ista->readVerilogWithRustParser(verilog_file);
}
}  // namespace ista