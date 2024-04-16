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
 * @file CmdSetOperatingConditions.cc
 * @author long shuaiying (longshy@pcl.ac.cn)
 * @brief support `set_operating_conditions` command in sdc
 * @version 0.1
 * @date 2024-04-09
 */
#include "Cmd.hh"

namespace ista {
CmdSetOperatingConditions::CmdSetOperatingConditions(const char* cmd_name)
    : TclCmd(cmd_name) {
  auto* analysis_type_option =
      new TclStringOption("-analysis_type", 0, nullptr);
  addOption(analysis_type_option);

  auto* library_option = new TclStringOption("-library", 0, nullptr);
  addOption(library_option);
}

unsigned CmdSetOperatingConditions::check() { return 1; }

unsigned CmdSetOperatingConditions::exec() { return 1; }

}  // namespace ista