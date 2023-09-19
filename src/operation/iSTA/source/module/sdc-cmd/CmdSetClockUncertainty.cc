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
 * @file CmdSetClockUncertainty.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The sdc set_clock_uncertainty cmd implemention.
 * @version 0.1
 * @date 2021-09-21
 */

#include "Cmd.hh"
#include "sdc/SdcSetClockUncertainty.hh"

namespace ista {
CmdSetClockUncertainty::CmdSetClockUncertainty(const char* cmd_name)
    : TclCmd(cmd_name) {
  auto* uncertainty_arg = new TclDoubleOption("uncertainty", 1, 0.0);
  addOption(uncertainty_arg);

  auto* obj_list_option = new TclStringListOption("object_list", 1, {});
  addOption(obj_list_option);

  auto* setup_option = new TclSwitchOption("-setup");
  addOption(setup_option);

  auto* hold_option = new TclSwitchOption("-hold");
  addOption(hold_option);

  auto* rise_option = new TclSwitchOption("-rise");
  addOption(rise_option);

  auto* fall_option = new TclSwitchOption("-fall");
  addOption(fall_option);
}

/**
 * @brief The set_clock_uncertainty cmd legally check.
 *
 * @return unsigned
 */
unsigned CmdSetClockUncertainty::check() {
  // TODO(to taosimin) fix check
  return 1;
}

/**
 * @brief The set_clock_uncertainty execute body.
 *
 * @return unsigned
 */
unsigned CmdSetClockUncertainty::exec() {
  if (!check()) {
    return 0;
  }
  Sta* ista = Sta::getOrCreateSta();

  auto* uncertainty_value = getOptionOrArg("uncertainty");

  auto* clock_uncertainty = new SdcSetClockUncertainty(
      ista->convertTimeUnit(uncertainty_value->getDoubleVal()));

  auto* rise_option = getOptionOrArg("-rise");
  auto* fall_option = getOptionOrArg("-fall");
  if (rise_option->is_set_val() || !fall_option->is_set_val()) {
    clock_uncertainty->set_fall(false);
  }
  if (fall_option->is_set_val() || !rise_option->is_set_val()) {
    clock_uncertainty->set_rise(false);
  }

  auto* setup_option = getOptionOrArg("-setup");
  auto* hold_option = getOptionOrArg("-hold");
  if (setup_option->is_set_val() || !hold_option->is_set_val()) {
    clock_uncertainty->set_hold(false);
  }
  if (hold_option->is_set_val() || !setup_option->is_set_val()) {
    clock_uncertainty->set_setup(false);
  }

  auto* object_list_option = getOptionOrArg("object_list");

  SdcConstrain* the_constrain = ista->getConstrain();

  // Netlist* design_nl = ista->get_netlist();
  std::vector<std::string> pin_port_strs = object_list_option->getStringList();
  std::set<SdcCollectionObj> objs;
  for (auto& pin_port_name : pin_port_strs) {
    if (Str::startWith(pin_port_name.c_str(),
                       TclEncodeResult::get_encode_preamble())) {
      auto* obj_collection = static_cast<SdcCollection*>(
          TclEncodeResult::decode(pin_port_name.c_str()));
      auto& obj_list = obj_collection->get_collection_objs();
      for (auto obj : obj_list) {
        std::visit(
            overloaded{
                [&objs](SdcCommandObj* sdc_obj) { objs.insert(sdc_obj); },
                [&objs](DesignObject* design_obj) { objs.insert(design_obj); },
            },
            obj);
      }
    } else {
      Netlist* design_nl = ista->get_netlist();
      auto pin_ports = design_nl->findObj(pin_port_name.c_str(), false, false);

      for (auto* design_obj : pin_ports) {
        objs.insert(design_obj);
      }
    }
  }

  clock_uncertainty->set_objs(std::move(objs));

  the_constrain->addTimingUncertainty(clock_uncertainty);

  return 1;
}

}  // namespace ista
