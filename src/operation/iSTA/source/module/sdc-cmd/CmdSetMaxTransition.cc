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
 * @file CmdSetMaxTransition.cc
 * @author Wang Hao (harry0789@qq.com)
 * @brief
 * @version 0.1
 * @date 2021-10-06
 */

#include "Cmd.hh"
#include "sdc/SdcCollection.hh"

namespace ista {

CmdSetMaxTransition::CmdSetMaxTransition(const char* cmd_name)
    : TclCmd(cmd_name) {
  auto* clock_path_option = new TclSwitchOption("-clock_path");
  addOption(clock_path_option);

  auto* data_path_option = new TclSwitchOption("-data_path");
  addOption(data_path_option);

  auto* rise_option = new TclSwitchOption("-rise");
  addOption(rise_option);

  auto* fall_option = new TclSwitchOption("-fall");
  addOption(fall_option);

  auto* transition_option = new TclDoubleOption("transition_value", 1, 0.0);
  addOption(transition_option);

  auto* pin_port_arg = new TclStringListOption("object_list", 1, {});
  addOption(pin_port_arg);
}

/**
 * @brief The SetMaxTransition cmd legally check.
 *
 * @return unsigned
 */
unsigned CmdSetMaxTransition::check() {
  TclOption* pin_port_arg = getOptionOrArg("object_list");
  TclOption* transition_option = getOptionOrArg("transition_value");
  LOG_FATAL_IF(!pin_port_arg);
  LOG_FATAL_IF(!transition_option);
  if (!(pin_port_arg->is_set_val() && transition_option->is_set_val())) {
    LOG_ERROR << "'object_list' and 'transition_value' are required.";
    return 0;
  }

  return 1;
}

/**
 * @brief The create_generate_clock execute body.
 *
 * @return unsigned success return 1, else return 0.
 */
unsigned CmdSetMaxTransition::exec() {
  if (!check()) {
    return 0;
  }

  auto* transition_option = getOptionOrArg("transition_value");

  double transition_value = transition_option->getDoubleVal();
  auto* set_max_transtion = new SetMaxTransition(transition_value);

  setClockDataPath(set_max_transtion);
  setRiseFall(set_max_transtion);
  setObjectLists(set_max_transtion);

  addTimingDRC2Constrain(set_max_transtion);

  return 1;
}

void CmdSetMaxTransition::addTimingDRC2Constrain(
    SetMaxTransition* set_max_transition) {
  Sta* ista = Sta::getOrCreateSta();
  SdcConstrain* the_constrain = ista->getConstrain();
  the_constrain->addTimingDRC(set_max_transition);
}

void CmdSetMaxTransition::setClockDataPath(
    SetMaxTransition* set_max_transition) {
  TclOption* clock_path_option = getOptionOrArg("-clock_path");
  TclOption* data_path_option = getOptionOrArg("-data_path");

  if (clock_path_option->is_set_val() || !data_path_option->is_set_val()) {
    set_max_transition->set_is_clock_path();
  }

  if (data_path_option->is_set_val() || !clock_path_option->is_set_val()) {
    set_max_transition->set_is_data_path();
  }
}

void CmdSetMaxTransition::setRiseFall(SetMaxTransition* set_max_transition) {
  auto* rise_option = getOptionOrArg("-rise");
  auto* fall_option = getOptionOrArg("-fall");
  if (rise_option->is_set_val() || !fall_option->is_set_val()) {
    set_max_transition->set_is_rise();
  }

  if (fall_option->is_set_val() || !rise_option->is_set_val()) {
    set_max_transition->set_is_fall();
  }
}

void CmdSetMaxTransition::setObjectLists(SetMaxTransition* set_max_transition) {
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
        LOG_INFO << "set max transition to all design obj.";
      } else {
        auto& obj_list = obj_collection->get_collection_objs();
        for (auto obj : obj_list) {
          std::visit(
              overloaded{
                  [&objs](SdcCommandObj* sdc_obj) {
                    LOG_INFO
                        << "set max transition to clock obj "
                        << dynamic_cast<SdcClock*>(sdc_obj)->get_clock_name();
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

  set_max_transition->set_objs(std::move(objs));
}

}  // namespace ista