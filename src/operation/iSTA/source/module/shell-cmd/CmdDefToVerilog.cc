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
 * @file CmdDefToVerilog.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of def to verilog command.
 * @version 0.1
 * @date 2023-06-12
 */
#include "ShellCmd.hh"
#include "builder.h"

namespace ista {

CmdDefToVerilog::CmdDefToVerilog(const char* cmd_name) : TclCmd(cmd_name) {
  auto* def_option = new TclStringOption("-def", 0, nullptr);
  addOption(def_option);

  auto* lef_option = new TclStringListOption("-lef", 0);
  addOption(lef_option);

  auto* verilog_option = new TclStringOption("-verilog", 0, nullptr);
  addOption(verilog_option);

  auto* add_space_for_escape_name_option = new TclSwitchOption("-addspace");
  addOption(add_space_for_escape_name_option);
}

unsigned CmdDefToVerilog::check() { return 1; }

unsigned CmdDefToVerilog::exec() {
  if (!check()) {
    return 0;
  }

  auto* db_builder = new idb::IdbBuilder();

  TclOption* lef_option = getOptionOrArg("-lef");
  auto lef_files = lef_option->getStringList();

  db_builder->buildLef(lef_files);

  TclOption* def_option = getOptionOrArg("-def");
  auto* def_file = def_option->getStringVal();
  db_builder->buildDef(def_file);

  TclOption* verilog_option = getOptionOrArg("-verilog");
  auto* verilog_file = verilog_option->getStringVal();

  TclOption* add_space_for_escape_name_option = getOptionOrArg("-addspace");
  bool is_add_space_for_escape_name = false;
  if (add_space_for_escape_name_option->is_set_val()) {
    is_add_space_for_escape_name = true;
  }

  std::set<std::string> exclude_cell_names;
  db_builder->saveVerilog(verilog_file, exclude_cell_names,
                          is_add_space_for_escape_name);

  return 1;
}

}  // namespace ista
