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
 * @file CmdReadPGSpef.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief read_pg_spef command
 * @version 0.1
 * @date 2025-03-04
 * 
 */
#include "PowerShellCmd.hh"
#include "api/PowerEngine.hh"

namespace ipower {

CmdReadPGSpef::CmdReadPGSpef(const char* cmd_name) : TclCmd(cmd_name) {
  auto* file_name_option = new TclStringOption("file_name", 1, nullptr);
  addOption(file_name_option);
}

unsigned CmdReadPGSpef::check() { return 1; }

unsigned CmdReadPGSpef::exec() {
  if (!check()) {
    return 0;
  }

  TclOption* file_name_option = getOptionOrArg("file_name");
  auto pg_spef_file = file_name_option->getStringVal();

  PowerEngine* power_engine = PowerEngine::getOrCreatePowerEngine();
  return power_engine->readPGSpef(pg_spef_file);
}

}  // namespace ipower