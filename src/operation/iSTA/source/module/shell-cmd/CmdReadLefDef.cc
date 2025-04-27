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
 * @file CmdReadLefDef.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of read lef def command.
 * @version 0.1
 * @date 2023-06-12
 */
#include "ShellCmd.hh"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"

namespace ista {

CmdReadLefDef::CmdReadLefDef(const char* cmd_name) : TclCmd(cmd_name) {
  auto* def_option = new TclStringOption("-def", 0, nullptr);
  addOption(def_option);

  auto* lef_option = new TclStringListOption("-lef", 0);
  addOption(lef_option);
}

unsigned CmdReadLefDef::check() { return 1; }

unsigned CmdReadLefDef::exec() {
  if (!check()) {
    return 0;
  }


  TclOption* lef_option = getOptionOrArg("-lef");
  auto lef_files = lef_option->getStringList();

  TclOption* def_option = getOptionOrArg("-def");
  auto* def_file = def_option->getStringVal();

  auto* timing_engine = TimingEngine::getOrCreateTimingEngine();
  timing_engine->readDefDesign(def_file, lef_files);

  return 1;
}

}  // namespace ista