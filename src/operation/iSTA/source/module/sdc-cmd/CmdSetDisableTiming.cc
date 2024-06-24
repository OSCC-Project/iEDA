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
 * @file CmdSetDisableTiming.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The sdc cmd of set_disable_timing
 * @version 0.1
 * @date 2024-06-24
 *
 */

#include "Cmd.hh"
#include "sdc/SdcCollection.hh"

namespace ista {

CmdSetDisableTiming::CmdSetDisableTiming(const char* cmd_name)
    : TclCmd(cmd_name) {
  auto* from_option = new TclStringListOption("-from", 0, {});
  addOption(from_option);

  auto* to_option = new TclStringListOption("-to", 0, {});
  addOption(to_option);

  auto* through_option = new TclStringListListOption("-through", 0);
  addOption(through_option);

  auto* disable_objs_arg = new TclStringOption("disable_objs", 1);
  addOption(disable_objs_arg);
}

unsigned CmdSetDisableTiming::check() { return 1; }

/**
 * @brief execute the get_libs.
 *
 * @return unsigned
 */
unsigned CmdSetDisableTiming::exec() {
  // TODO(to taosimin), implement disable timing.

  return 1;
}

}  // namespace ista