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
 * @file CmdSetLoad.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The sdc set_load cmd implemention.
 * @version 0.1
 * @date 2021-05-17
 */

#include "Cmd.hh"
#include "netlist/DesignObject.hh"
#include "sdc/SdcConstrain.hh"
#include "sdc/SdcSetLoad.hh"
#include "sta/Sta.hh"

namespace ista {

CmdSetLoad::CmdSetLoad(const char* cmd_name) : TclCmd(cmd_name) {
  auto* rise_option = new TclSwitchOption("-rise");
  addOption(rise_option);

  auto* fall_option = new TclSwitchOption("-fall");
  addOption(fall_option);

  auto* min_option = new TclSwitchOption("-min");
  addOption(min_option);

  auto* max_option = new TclSwitchOption("-max");
  addOption(max_option);

  auto* pin_load_option = new TclSwitchOption("-pin_load");
  addOption(pin_load_option);

  auto* wire_load_option = new TclSwitchOption("-wire_load");
  addOption(wire_load_option);

  auto* allow_negative_cap_option = new TclSwitchOption("-allow_negative_cap");
  addOption(allow_negative_cap_option);

  auto* subtract_pin_load_option = new TclSwitchOption("-subtract_pin_load");
  addOption(subtract_pin_load_option);

  auto* capacitance_arg = new TclDoubleOption("capacitance", 1, 0.0);
  addOption(capacitance_arg);

  // The objects_arg arg should be string list, fix me.
  auto* objects_arg = new TclStringOption("objects", 1, nullptr);
  addOption(objects_arg);
}

/**
 * @brief The set_load cmd legally check.
 *
 * @return unsigned
 */
unsigned CmdSetLoad::check() { return 1; }

/**
 * @brief The set_load execute body.
 *
 * @return unsigned success return 1, else return 0.
 */
unsigned CmdSetLoad::exec() {
  if (!check()) {
    return 0;
  }

  auto* load_value = getOptionOrArg("capacitance");

  auto* set_load = new SdcSetLoad(get_cmd_name(), load_value->getDoubleVal());

  auto* rise_option = getOptionOrArg("-rise");
  auto* fall_option = getOptionOrArg("-fall");
  // -rise -fall if one set, other both two not set, then we set.
  if (rise_option->is_set_val() || !fall_option->is_set_val()) {
    set_load->set_rise();
  }

  if (fall_option->is_set_val() || !rise_option->is_set_val()) {
    set_load->set_fall();
  }

  auto* max_option = getOptionOrArg("-max");
  auto* min_option = getOptionOrArg("-min");
  if (max_option->is_set_val() || !min_option->is_set_val()) {
    set_load->set_max();
  }

  if (min_option->is_set_val() || !max_option->is_set_val()) {
    set_load->set_min();
  }

  TclOption* port_list_option = getOptionOrArg("objects");
  auto* port_list_str = port_list_option->getStringVal();

  Sta* ista = Sta::getOrCreateSta();
  Netlist* design_nl = ista->get_netlist();

  std::set<DesignObject*> ports;
  auto object_list = FindObjOfSdc(port_list_str, design_nl);
  LOG_FATAL_IF(object_list.empty()) << "object list is empty.";

  for (auto& object : object_list) {
    std::visit(
        overloaded{
            [](SdcCommandObj* sdc_obj) { LOG_FATAL << "not support sdc obj."; },
            [&ports](DesignObject* design_obj) { ports.insert(design_obj); },
        },
        object);
  }

  set_load->set_objs(std::move(ports));
  SdcConstrain* the_constrain = ista->getConstrain();
  the_constrain->addIOConstrain(set_load);

  return 1;
}

}  // namespace ista
