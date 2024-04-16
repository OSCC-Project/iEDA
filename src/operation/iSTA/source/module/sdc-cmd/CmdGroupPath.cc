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
 * @file CmdGroupPath.cc
 * @author long shuaiying (longshy@pcl.ac.cn)
 * @brief support `group_path` command in sdc
 * @version 0.1
 * @date 2024-04-09
 */
#include "Cmd.hh"

namespace ista {
CmdGroupPath::CmdGroupPath(const char* cmd_name) : TclCmd(cmd_name) {
  auto* name_option = new TclStringOption("-name", 0, nullptr);
  addOption(name_option);

  auto* from = new TclStringListOption("-from", 0);
  addOption(from);

  auto* to = new TclStringListOption("-to", 0);
  addOption(to);
}

unsigned CmdGroupPath::check() { return 1; }

unsigned CmdGroupPath::exec() { return 1; }

}  // namespace ista