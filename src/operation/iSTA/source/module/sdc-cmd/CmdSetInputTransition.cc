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
 * @file CmdSetInputTransition.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The sdc set_input_transition cmd implemention.
 * @version 0.1
 * @date 2021-04-14
 */

#include "Cmd.hh"
#include "netlist/DesignObject.hh"
#include "sdc/SdcConstrain.hh"
#include "sdc/SdcSetInputTransition.hh"
#include "sta/Sta.hh"
#include "sdc/SdcAllPorts.hh"

namespace ista {
CmdSetInputTransition::CmdSetInputTransition(const char* cmd_name)
    : TclCmd(cmd_name) {
  auto* rise_option = new TclSwitchOption("-rise");
  addOption(rise_option);

  auto* fall_option = new TclSwitchOption("-fall");
  addOption(fall_option);

  auto* min_option = new TclSwitchOption("-min");
  addOption(min_option);

  auto* max_option = new TclSwitchOption("-max");
  addOption(max_option);

  auto* clock_option = new TclStringOption("-clock", 0, nullptr);
  addOption(clock_option);

  auto* transition_arg = new TclDoubleOption("transition", 1, 0.0);
  addOption(transition_arg);

  // The pin port arg should be string list, fix me.
  auto* port_list_arg = new TclStringOption("pin_port_list", 1, nullptr);
  addOption(port_list_arg);
}

/**
 * @brief The set_input_transition cmd legally check.
 *
 * @return unsigned
 */
unsigned CmdSetInputTransition::check() {
  // TODO(to taosimin) fix check
  return 1;
}

/**
 * @brief The set_input_transition execute body.
 *
 * @return unsigned success return 1, else return 0.
 */
unsigned CmdSetInputTransition::exec() {
  if (!check()) {
    return 0;
  }

  Sta* ista = Sta::getOrCreateSta();
  SdcConstrain* the_constrain = ista->getConstrain();
  Netlist* design_nl = ista->get_netlist();

  auto* transition_value = getOptionOrArg("transition");

  auto* set_input_transiton = new SdcSetInputTransition(
      get_cmd_name(), ista->convertTimeUnit(transition_value->getDoubleVal()));

  auto* rise_option = getOptionOrArg("-rise");
  auto* fall_option = getOptionOrArg("-fall");
  // -rise -fall default set, if one set, other not set, we set other not set.
  if (rise_option->is_set_val() && !fall_option->is_set_val()) {
    set_input_transiton->set_fall(false);
  }

  if (fall_option->is_set_val() && !rise_option->is_set_val()) {
    set_input_transiton->set_rise(false);
  }

  auto* max_option = getOptionOrArg("-max");
  auto* min_option = getOptionOrArg("-min");
  if (max_option->is_set_val() && !min_option->is_set_val()) {
    set_input_transiton->set_min(false);
  }

  if (min_option->is_set_val() && !max_option->is_set_val()) {
    set_input_transiton->set_max(false);
  }

  TclOption* pin_port_list_option = getOptionOrArg("pin_port_list");
  std::string pin_port_name = pin_port_list_option->getStringVal();
  LOG_INFO << "pin_port_name: " << pin_port_name;
  auto pin_port_list = FindObjOfSdc(pin_port_name, design_nl);
  LOG_FATAL_IF(pin_port_list.empty()) << "pin_port_list is empty.";

  std::set<DesignObject*> pin_ports;
  for (auto obj : pin_port_list) {
    std::visit(overloaded{
                   [&pin_ports](SdcCommandObj* sdc_obj) {
                     if (sdc_obj->isAllInputPorts()) {
                       auto* all_input_ports =
                           dynamic_cast<SdcAllInputPorts*>(sdc_obj);
                       auto& input_ports = all_input_ports->get_input_ports();
                       std::ranges::for_each(input_ports,
                                             [&pin_ports](auto* input_port) {
                                               pin_ports.insert(input_port);
                                             });
                     }
                   },
                   [&pin_ports](DesignObject* design_obj) {
                     pin_ports.insert(design_obj);
                   },
               },
               obj);
  }

  LOG_FATAL_IF(pin_ports.empty()) << "pin_port_list is empty.";
  set_input_transiton->set_objs(std::move(pin_ports));
  the_constrain->addIOConstrain(set_input_transiton);

  return 1;
}

}  // namespace ista
