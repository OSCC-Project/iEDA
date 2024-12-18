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
 * @file CmdDumpGraphData.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of dump graph data command.
 * @version 0.1
 * @date 2024-11-06
 */
#include "ShellCmd.hh"

namespace ista {

CmdDumpGraphData::CmdDumpGraphData(const char* cmd_name) : TclCmd(cmd_name) {
  auto* graph_option = new TclStringOption("graph_file", 1, nullptr);
  addOption(graph_option);

}

unsigned CmdDumpGraphData::check() { return 1; }

unsigned CmdDumpGraphData::exec() {
  if (!check()) {
    return 0;
  }

  TclOption* graph_file_option = getOptionOrArg("graph_file");
  auto* graph_file = graph_file_option->getStringVal();

  auto* ista = ista::Sta::getOrCreateSta();
  ista->dumpGraphData(graph_file);

  return 1;
}

} 