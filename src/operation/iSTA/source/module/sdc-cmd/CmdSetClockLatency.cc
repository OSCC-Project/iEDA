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
 * @file CmdSetClockLatency.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The sdc set_clock_latency implemention.
 * @version 0.1
 * @date 2021-10-19
 */

#include "Cmd.hh"
#include "sdc/SdcSetClockLatency.hh"

namespace ista {

CmdSetClockLatency::CmdSetClockLatency(const char* cmd_name)
    : TclCmd(cmd_name) {
  auto* rise_option = new TclSwitchOption("-rise");
  addOption(rise_option);

  auto* fall_option = new TclSwitchOption("-fall");
  addOption(fall_option);

  auto* min_option = new TclSwitchOption("-min");
  addOption(min_option);

  auto* max_option = new TclSwitchOption("-max");
  addOption(max_option);

  auto* early_option = new TclSwitchOption("-early");
  addOption(early_option);

  auto* late_option = new TclSwitchOption("-late");
  addOption(late_option);

  auto* delay_arg = new TclDoubleOption("delay", 1, 0.0);
  addOption(delay_arg);

  auto* object_list_arg = new TclStringOption("object_list", 1, {});
  addOption(object_list_arg);
}

unsigned CmdSetClockLatency::check() { return 1; }

unsigned CmdSetClockLatency::exec() {
  auto* delay_value = getOptionOrArg("delay");

  Sta* ista = Sta::getOrCreateSta();
  SdcConstrain* the_constrain = ista->getConstrain();

  Netlist* design_nl = ista->get_netlist();

  SdcSetClockLatency* clock_latency = new SdcSetClockLatency(
      ista->convertTimeUnit(delay_value->getDoubleVal()));

  auto* rise_option = getOptionOrArg("-rise");
  auto* fall_option = getOptionOrArg("-fall");
  if (rise_option->is_set_val() || !fall_option->is_set_val()) {
    clock_latency->set_rise();
  }
  if (fall_option->is_set_val() || !rise_option->is_set_val()) {
    clock_latency->set_fall();
  }

  auto* max_option = getOptionOrArg("-max");
  auto* min_option = getOptionOrArg("-min");
  if (max_option->is_set_val() || !min_option->is_set_val()) {
    clock_latency->set_max();
  }
  if (min_option->is_set_val() || !max_option->is_set_val()) {
    clock_latency->set_min();
  }

  auto* early_option = getOptionOrArg("-early");
  auto* late_option = getOptionOrArg("-late");
  if (early_option->is_set_val() || !late_option->is_set_val()) {
    clock_latency->set_early();
  }
  if (late_option->is_set_val() || !early_option->is_set_val()) {
    clock_latency->set_late();
  }

  TclOption* object_list_option = getOptionOrArg("object_list");

  auto pin_ports =
      design_nl->findObj(object_list_option->getStringVal(), false, false);
  std::set<DesignObject*> pins;
  for (auto* design_obj : pin_ports) {
    pins.insert(design_obj);
  }

  clock_latency->set_objs(std::move(pins));
  the_constrain->addTimingLatency(clock_latency);

  return 1;
}

}  // namespace ista
