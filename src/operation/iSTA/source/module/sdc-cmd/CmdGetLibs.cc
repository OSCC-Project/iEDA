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
 * @file CmdGetLibs.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The sdc cmd of get_libs
 * @version 0.1
 * @date 2024-03-19
 */
#include "Cmd.hh"
#include "sdc/SdcCollection.hh"
#include "wildcards/single_include/wildcards.hpp"

namespace ista {

CmdGetLibs::CmdGetLibs(const char* cmd_name) : TclCmd(cmd_name) {
  auto* patterns_arg = new TclStringListOption("patterns", 1, {});
  addOption(patterns_arg);

  auto* used_option = new TclSwitchOption("-used");
  addOption(used_option);
}

unsigned CmdGetLibs::check() { return 1; }

/**
 * @brief execute the get_libs.
 *
 * @return unsigned
 */
unsigned CmdGetLibs::exec() {
  Sta* ista = Sta::getOrCreateSta();

  auto* lib_patterns_arg = getOptionOrArg("patterns");
  auto lib_patterns = lib_patterns_arg->getStringList();

  auto* used_option = getOptionOrArg("-used");

  if (used_option->is_set_val()) {
    auto used_libs = ista->getUsedLibs();
    for (auto* used_lib : used_libs) {
      std::string lib_name = used_lib->get_file_name();

      for (auto& pattern : lib_patterns) {
        // match lib name all str.
        std::string_view pattern_str = pattern;
        if (wildcards::match(lib_name, pattern_str)) {
            LOG_INFO << "used lib: " << lib_name;
        }
      }
    }
  } else {
    // TODO(to taosimin), get all lib as collection.
  }

  return 1;
}

}  // namespace ista