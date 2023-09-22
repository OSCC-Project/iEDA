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
 * @file CmdSetMaxCapacitance.cc
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-10-14
 */
#include "Cmd.hh"
#include "sdc/SdcCollection.hh"

namespace ista {

CmdSetMaxCapacitance::CmdSetMaxCapacitance(const char* cmd_name)
    : TclCmd(cmd_name) {
  auto* capacitance_arg = new TclDoubleOption("capacitance_value", 1, 0.0);
  addOption(capacitance_arg);

  auto* clock_path_option = new TclSwitchOption("-clock_path");
  addOption(clock_path_option);

  auto* data_path_option = new TclSwitchOption("-data_path");
  addOption(data_path_option);

  auto* rise_option = new TclSwitchOption("-rise");
  addOption(rise_option);

  auto* fall_option = new TclSwitchOption("-fall");
  addOption(fall_option);

  auto* pin_port_arg = new TclStringListOption("object_list", 1, {});
  addOption(pin_port_arg);
}

/**
 * @brief The SetMaxCapacitance cmd legally check.
 *
 * @return unsigned
 */
unsigned CmdSetMaxCapacitance::check() {
  TclOption* pin_port_arg = getOptionOrArg("object_list");
  TclOption* capacitance_option = getOptionOrArg("capacitance_value");
  LOG_FATAL_IF(!pin_port_arg);
  LOG_FATAL_IF(!capacitance_option);
  if (!(pin_port_arg->is_set_val() && capacitance_option->is_set_val())) {
    LOG_ERROR << "'object_list' and 'capacitance_value' are required.";
    return 0;
  }

  double capacitance_value = capacitance_option->getDoubleVal();
  if (capacitance_value < 0) {
    LOG_ERROR << "'capacitance_value'>=0 required";
    return 0;
  }

  return 1;
}

/**
 * @brief The SetMaxCapacitance execute body.
 *
 * @return unsigned success return 1, else return 0.
 */
unsigned CmdSetMaxCapacitance::exec() {
  if (!check()) {
    return 0;
  }
  Sta* ista = Sta::getOrCreateSta();

  auto* capacitance_option = getOptionOrArg("capacitance_value");

  double capacitance_value =
      ista->convertCapUnit(capacitance_option->getDoubleVal());
  auto* set_max_cap = new SetMaxCapacitance(capacitance_value);

  setClockDataPath(set_max_cap);
  setRiseFall(set_max_cap);
  setObjectLists(set_max_cap);

  addTimingDRC2Constrain(set_max_cap);

  return 1;
}

void CmdSetMaxCapacitance::addTimingDRC2Constrain(
    SetMaxCapacitance* set_max_cap) {
  Sta* ista = Sta::getOrCreateSta();
  SdcConstrain* the_constrain = ista->getConstrain();
  the_constrain->addTimingDRC(set_max_cap);
}

void CmdSetMaxCapacitance::setClockDataPath(SetMaxCapacitance* set_max_cap) {
  TclOption* clock_path_option = getOptionOrArg("-clock_path");
  TclOption* data_path_option = getOptionOrArg("-data_path");

  if (clock_path_option->is_set_val() || !data_path_option->is_set_val()) {
    set_max_cap->set_is_clock_path();
  }

  if (data_path_option->is_set_val() || !clock_path_option->is_set_val()) {
    set_max_cap->set_is_data_path();
  }
}

void CmdSetMaxCapacitance::setRiseFall(SetMaxCapacitance* set_max_cap) {
  auto* rise_option = getOptionOrArg("-rise");
  auto* fall_option = getOptionOrArg("-fall");
  if (rise_option->is_set_val() || !fall_option->is_set_val()) {
    set_max_cap->set_is_rise();
  }

  if (fall_option->is_set_val() || !rise_option->is_set_val()) {
    set_max_cap->set_is_fall();
  }
}

void CmdSetMaxCapacitance::setObjectLists(SetMaxCapacitance* set_max_cap) {
  Sta* ista = Sta::getOrCreateSta();
  TclOption* pin_port_option = getOptionOrArg("object_list");
  std::vector<std::string> pin_port_strs = pin_port_option->getStringList();
  std::set<SdcCollectionObj> objs;

  for (auto& pin_port_name : pin_port_strs) {
    if (Str::startWith(pin_port_name.c_str(),
                       TclEncodeResult::get_encode_preamble())) {
      auto* obj_collection = static_cast<SdcCollection*>(
          TclEncodeResult::decode(pin_port_name.c_str()));
      if (obj_collection->isNetlistCollection()) {
        LOG_INFO << "set max capacitance to all design obj.";
      } else {
        auto& obj_list = obj_collection->get_collection_objs();
        for (auto obj : obj_list) {
          std::visit(overloaded{
                         [&objs](SdcCommandObj* sdc_obj) {
                           LOG_INFO << "set max capacitance to clock obj.";
                           objs.insert(sdc_obj);
                         },
                         [&objs](DesignObject* design_obj) {
                           objs.insert(design_obj);
                         },
                     },
                     obj);
        }
      }
    } else {
      Netlist* design_nl = ista->get_netlist();
      auto pin_ports = design_nl->findObj(pin_port_name.c_str(), false, false);

      for (auto design_obj : pin_ports) {
        objs.insert(design_obj);
        DLOG_INFO << " pin is " << design_obj->getFullName();
      }
    }
  }

  set_max_cap->set_objs(std::move(objs));
}

}  // namespace ista