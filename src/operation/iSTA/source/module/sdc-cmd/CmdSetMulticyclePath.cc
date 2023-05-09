// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file CmdSetMulticyclePath.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The sdc set_multicycle_path cmd implemention.
 * @version 0.1
 * @date 2022-07-17
 */
#include "Cmd.hh"
#include "netlist/DesignObject.hh"
#include "sdc/SdcException.hh"
#include "sta/Sta.hh"

namespace ista {

CmdSetMulticyclePath::CmdSetMulticyclePath(const char* cmd_name)
    : TclCmd(cmd_name) {
  auto* from_option = new TclStringListOption("-from", 0, {});
  addOption(from_option);

  auto* to_option = new TclStringListOption("-to", 0, {});
  addOption(to_option);

  auto* through_option = new TclStringListListOption("-through", 0);
  addOption(through_option);

  auto* setup_option = new TclSwitchOption("-setup");
  addOption(setup_option);

  auto* hold_option = new TclSwitchOption("-hold");
  addOption(hold_option);

  auto* start_option = new TclSwitchOption("-start");
  addOption(start_option);

  auto* end_option = new TclSwitchOption("-end");
  addOption(end_option);

  auto* rise_option = new TclSwitchOption("-rise");
  addOption(rise_option);

  auto* fall_option = new TclSwitchOption("-fall");
  addOption(fall_option);

  auto* path_multiplier_arg = new TclIntOption("path_multiplier", 1, 1);
  addOption(path_multiplier_arg);
}

unsigned CmdSetMulticyclePath::check() { return 1; }

unsigned CmdSetMulticyclePath::exec() {
  if (!check()) {
    return 0;
  }

  auto* path_multiplier = getOptionOrArg("path_multiplier");

  auto* set_multicycle_path =
      new SdcMulticyclePath(path_multiplier->getIntVal());
  {
    auto* setup_option = getOptionOrArg("-setup");
    auto* hold_option = getOptionOrArg("-hold");
    // -setup -hold default set, if one set, other not set, we set other not
    // set.
    if (setup_option->is_set_val() && !hold_option->is_set_val()) {
      set_multicycle_path->set_hold(false);
    }

    if (hold_option->is_set_val() && !setup_option->is_set_val()) {
      set_multicycle_path->set_setup(false);
    }
  }

  {
    auto* rise_option = getOptionOrArg("-rise");
    auto* fall_option = getOptionOrArg("-fall");
    // -rise -fall default set, if one set, other not set, we set other not set.
    if (rise_option->is_set_val() && !fall_option->is_set_val()) {
      set_multicycle_path->set_fall(false);
    }

    if (fall_option->is_set_val() && !rise_option->is_set_val()) {
      set_multicycle_path->set_rise(false);
    }
  }

  {
    auto* start_option = getOptionOrArg("-start");
    auto* end_option = getOptionOrArg("-end");
    if (start_option->is_set_val() && !end_option->is_set_val()) {
      set_multicycle_path->set_end(false);
    }

    if (end_option->is_set_val() && !start_option->is_set_val()) {
      set_multicycle_path->set_start(false);
    }
  }

  {
    auto* from_option = getOptionOrArg("-from");
    auto from_list = from_option->getStringList();
    set_multicycle_path->set_prop_froms(std::move(from_list));

    auto* through_option = getOptionOrArg("-through");
    auto through_list_list = through_option->getStringListList();
    set_multicycle_path->set_prop_throughs(std::move(through_list_list));

    auto* to_option = getOptionOrArg("-to");
    auto to_list = to_option->getStringList();
    set_multicycle_path->set_prop_tos(std::move(to_list));
  }

  Sta* ista = Sta::getOrCreateSta();
  SdcConstrain* the_constrain = ista->getConstrain();

  the_constrain->addSdcException(set_multicycle_path);
  return 1;
}

}  // namespace ista