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
 * @file CmdSetClockGroups.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The set_clock_groups sdc cmd implemention.
 * @version 0.1
 * @date 2022-07-11
 */
#include "Cmd.hh"

namespace ista {

CmdSetClockGroups::CmdSetClockGroups(const char* cmd_name) : TclCmd(cmd_name) {
  auto* name_option = new TclStringOption("-name", 0, nullptr);
  addOption(name_option);

  auto* asynchronous_option = new TclSwitchOption("-asynchronous");
  addOption(asynchronous_option);

  auto* exclusive_option = new TclSwitchOption("-exclusive");
  addOption(exclusive_option);

  auto* physically_exclusive_option =
      new TclSwitchOption("-physically_exclusive");
  addOption(physically_exclusive_option);

  auto* logically_exclusive =
  new TclSwitchOption("-logically_exclusive");
  addOption(logically_exclusive);

  auto* group = new TclStringListListOption("-group", 0);
  addOption(group);
}

/**
 * @brief The set_clock_group cmd legally check.
 *
 * @return unsigned
 */
unsigned CmdSetClockGroups::check() { return 1; }

/**
 * @brief The set_clock_group execute body.
 *
 * @return unsigned
 */
unsigned CmdSetClockGroups::exec() {
  if (!check()) {
    return 0;
  }

  Sta* ista = Sta::getOrCreateSta();
  SdcConstrain* the_constrain = ista->getConstrain();
  Netlist* design_nl = ista->get_netlist();

  std::string group_name;
  auto* name_option = getOptionOrArg("-name");
  if (name_option->is_set_val()) {
    group_name = name_option->getStringVal();
  }

  TclOption* group_option = getOptionOrArg("-group");
  if (group_option->is_set_val()) {
    auto clock_groups = std::make_unique<SdcClockGroups>(std::move(group_name));
    auto group_list_list = group_option->getStringListList();
    for (auto&& group_list : group_list_list) {
      for (auto&& group_clock : group_list) {
        SdcClockGroup clock_group;
        auto clock_names =
            GetClockName(group_clock.c_str(), design_nl, the_constrain);
        for (auto& clock_name : clock_names) {
          clock_group.addClock(std::move(clock_name));
        }
        clock_groups->addClockGroup(std::move(clock_group));
      }
    }
    the_constrain->addClockGroups(std::move(clock_groups));
  }
  return 1;
}

}  // namespace ista
