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
 * @file CmdVerilogToDef.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of verilog to def command.
 * @version 0.1
 * @date 2023-09-16
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "ShellCmd.hh"
#include "builder.h"

namespace ista {

CmdVerilogToDef::CmdVerilogToDef(const char* cmd_name) : TclCmd(cmd_name) {
  auto* def_option = new TclStringOption("-def", 0, nullptr);
  addOption(def_option);

  auto* lef_option = new TclStringListOption("-lef", 0);
  addOption(lef_option);

  auto* verilog_option = new TclStringOption("-verilog", 0, nullptr);
  addOption(verilog_option);

  auto* top_option = new TclStringOption("-top", 0, nullptr);
  addOption(top_option);

  auto* die_are_option = new TclDoubleListOption("-die_area", 0);
  addOption(die_are_option);
}

unsigned CmdVerilogToDef::check() { return 1; }

unsigned CmdVerilogToDef::exec() {
  if (!check()) {
    return 0;
  }

  auto* db_builder = new idb::IdbBuilder();

  TclOption* lef_option = getOptionOrArg("-lef");
  auto lef_files = lef_option->getStringList();

  db_builder->buildLef(lef_files);

  TclOption* verilog_option = getOptionOrArg("-verilog");
  auto* verilog_file = verilog_option->getStringVal();

  TclOption* top_option = getOptionOrArg("-top");
  auto* top = top_option->getStringVal();

  db_builder->rustBuildVerilog(verilog_file, top);

  // set die area
  TclOption* die_area_option = getOptionOrArg("-die_area");
  std::vector<double> die_area{0.0, 0.0, 0.0, 0.0};
  if (die_area_option->is_set_val()) {
    die_area = die_area_option->getDoubleList();
  }
  LOG_FATAL_IF(die_area.size() != 4) << "Invalid die area";

  auto* idb_layout = db_builder->get_def_service()->get_layout();
  idb_layout->get_die()->add_point(idb_layout->transUnitDB(die_area[0]),
                                   idb_layout->transUnitDB(die_area[1]));

  idb_layout->get_die()->add_point(idb_layout->transUnitDB(die_area[2]),
                                   idb_layout->transUnitDB(die_area[3]));

  TclOption* def_option = getOptionOrArg("-def");
  const auto* def_file = def_option->getStringVal();
  if (const auto ret = db_builder->saveDef(def_file); ret) {
    LOG_ERROR << "Fail to Save DEF file!";
  }

  // the below two lines is used to test idb verilog.
  // std::set<std::string> exclude_cell_names = {};
  // db_builder->saveVerilog(def_file, exclude_cell_names);

  return 1;
}

}  // namespace ista
