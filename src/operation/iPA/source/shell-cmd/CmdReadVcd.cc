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
 * @file CmdReadVcd.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief cmd to read a vcd.
 * @version 0.1
 * @date 2023-05-04
 */

#include "PowerShellCmd.hh"
#include "sta/Sta.hh"

namespace ipower {
CmdReadVcd::CmdReadVcd(const char* cmd_name) : TclCmd(cmd_name) {
  auto* file_name_option = new TclStringOption("file_name", 1, nullptr);
  addOption(file_name_option);

  auto* top_instance_name_option = new TclStringOption("-top_name", 0, nullptr);
  addOption(top_instance_name_option);

  //   auto* begin_time_option = new TclIntOption("-begin_time", 0, 0);
  //   addOption(begin_time_option);

  //   auto* end_time_option = new TclIntOption("-end_time", 0, 0);
  //   addOption(end_time_option);
}

unsigned CmdReadVcd::check() {
  TclOption* file_name_option = getOptionOrArg("file_name");
  TclOption* top_instance_name_option = getOptionOrArg("-top_name");
  LOG_FATAL_IF(!file_name_option) << "vcd file should be specified";
  LOG_FATAL_IF(!top_instance_name_option) << "top instance name should be specified";
  return 1;
}

unsigned CmdReadVcd::exec() {
  if (!check()) {
    return 0;
  }

  TclOption* file_name_option = getOptionOrArg("file_name");
  auto vcd_file = file_name_option->getStringVal();

  TclOption* top_instance_name_option = getOptionOrArg("-top_name");
  auto* top_instance_name = top_instance_name_option->getStringVal();
  LOG_FATAL_IF(!top_instance_name) << "netlist top instance name should be specified";

  Sta* ista = Sta::getOrCreateSta();
  Power* ipower = Power::getOrCreatePower(&(ista->get_graph()));
  // TODO set begin end time
  return ipower->readRustVCD(vcd_file, top_instance_name);
}

}  // namespace ipower