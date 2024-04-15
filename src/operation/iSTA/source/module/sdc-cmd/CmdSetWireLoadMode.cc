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
 * @file CmdSetWireLoadMode.cc
 * @author long shuaiying (longshy@pcl.ac.cn)
 * @brief support `set_wire_load_mode` command in sdc
 * @version 0.1
 * @date 2024-04-09
 */
#include "Cmd.hh"

namespace ista {
CmdSetWireLoadMode::CmdSetWireLoadMode(const char* cmd_name)
    : TclCmd(cmd_name) {
  auto* mode_name_arg = new TclStringOption("mode_name", 1, nullptr);
  addOption(mode_name_arg);
}

unsigned CmdSetWireLoadMode::check() { return 1; }

unsigned CmdSetWireLoadMode::exec() { return 1; }

}  // namespace ista