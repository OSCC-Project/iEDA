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
 * @file CmdSetDrivingCell.cc
 * @author longshy (longshy@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2023-08-23
 */
#include "Cmd.hh"
#include "netlist/DesignObject.hh"
#include "sdc/SdcConstrain.hh"
#include "sdc/SdcSetInputTransition.hh"
#include "sta/Sta.hh"

namespace ista {
CmdSetDrivingCell ::CmdSetDrivingCell(const char* cmd_name) : TclCmd(cmd_name) {
  auto* rise_option = new TclSwitchOption("-rise");
  addOption(rise_option);

  auto* fall_option = new TclSwitchOption("-fall");
  addOption(fall_option);

  auto* min_option = new TclSwitchOption("-min");
  addOption(min_option);

  auto* max_option = new TclSwitchOption("-max");
  addOption(max_option);

  auto* lib_cell_option = new TclStringOption("-lib_cell", 0, nullptr);
  addOption(lib_cell_option);

  auto* pin_option = new TclStringOption("-pin", 0, nullptr);
  addOption(pin_option);

  auto* port_list_arg = new TclStringOption("pin_port_list", 1, nullptr);
  addOption(port_list_arg);

  auto* input_transition_rise_option =
      new TclDoubleOption("-input_transition_rise", 0, 0.0);
  addOption(input_transition_rise_option);

  auto* input_transition_fall_option =
      new TclDoubleOption("-input_transition_fall", 0, 0.0);
  addOption(input_transition_fall_option);
}

/**
 * @brief The set_driving_cell cmd legally check.
 *
 * @return unsigned
 */
unsigned CmdSetDrivingCell::check() {
  // TODO(to longshuaiying) fix check
  return 1;
}

/**
 * @brief The set_driving_cell execute body.
 *
 * @return unsigned success return 1, else return 0.
 */
unsigned CmdSetDrivingCell::exec() {
  if (!check()) {
    return 0;
  }

  Sta* ista = Sta::getOrCreateSta();
  auto* lib_cell_option = getOptionOrArg("-lib_cell");
  auto* lib_cell_name = lib_cell_option->getStringVal();
  auto* lib_cell = ista->findLibertyCell(lib_cell_name);
  auto* pin_option = getOptionOrArg("-pin");
  auto* outpin_name = pin_option->getStringVal();

  // shijian add.
  TclOption* pin_port_list_option = getOptionOrArg("pin_port_list");
  std::string pin_port_name = pin_port_list_option->getStringVal();
  // LOG_INFO << "pin_port_name: " << pin_port_name;
  Netlist* design_nl = ista->get_netlist();
  auto pin_port_list = FindObjOfSdc(pin_port_name, design_nl);
  LOG_FATAL_IF(pin_port_list.empty()) << "pin_port_list is empty.";

  std::set<DesignObject*> pin_ports;
  for (auto obj : pin_port_list) {
    std::visit(overloaded{
                   [](SdcCommandObj* sdc_obj) {
                     LOG_FATAL
                         << "set_input_transition not support sdc obj yet.";
                   },
                   [&pin_ports](DesignObject* design_obj) {
                     pin_ports.insert(design_obj);
                   },
               },
               obj);
  }

  LOG_FATAL_IF(pin_ports.empty()) << "pin_port_list is empty.";
  auto* input_transition_rise_option = getOptionOrArg("-input_transition_rise");
  auto* input_transition_fall_option = getOptionOrArg("-input_transition_fall");
  // as the lib cell is buffer, only one cell arc set. if problem, fix me.
  LibArc* cell_arc = nullptr;
  for (auto& cell_arc_set : lib_cell->get_cell_arcs()) {
    for (auto& arc : cell_arc_set.get()->get_arcs()) {
      if (strcmp(arc.get()->get_snk_port(), outpin_name) == 0) {
        cell_arc = cell_arc_set->front();
      }
    }
  }
  LibArc::TimingSense timing_sense = cell_arc->get_timing_sense();

  SdcConstrain* the_constrain = ista->getConstrain();

  std::stack<SdcSetIODelay*> the_existed_constrains;
  auto& sdc_io_constraints = the_constrain->get_sdc_io_constraints();
  for (auto& sdc_io_constraint : sdc_io_constraints) {
    if (!dynamic_cast<SdcSetIODelay*>(sdc_io_constraint.get())) {
      continue;
    }

    auto& objs =
        dynamic_cast<SdcSetIODelay*>(sdc_io_constraint.get())->get_objs();
    for (auto& obj : objs) {
      if (Str::equal(obj->get_name(), (*pin_ports.begin())->get_name())) {
        auto* the_existed_constrain =
            dynamic_cast<SdcSetIODelay*>(sdc_io_constraint.get());
        the_existed_constrains.push(the_existed_constrain);
      }
    }
  }
  LOG_FATAL_IF(the_existed_constrains.empty());

  if (the_existed_constrains.size() == 1) {
    auto* the_existed_constrain = the_existed_constrains.top();
    auto* copy_constrain = the_existed_constrain->copy();
    the_constrain->addIOConstrain(copy_constrain);
    the_existed_constrains.push(copy_constrain);
  }

  double delay_value_rise = cell_arc->getDelayOrConstrainCheckNs(
      TransType::kRise,
      ista->convertTimeUnit(input_transition_rise_option->getDoubleVal()), 0);
  double delay_value_fall = cell_arc->getDelayOrConstrainCheckNs(
      TransType::kFall,
      ista->convertTimeUnit(input_transition_fall_option->getDoubleVal()), 0);

  bool is_rise_need_assign = true;
  bool is_fall_need_assign = true;
  while (!the_existed_constrains.empty()) {
    auto* the_existed_constrain = the_existed_constrains.top();
    if (timing_sense == LibArc::TimingSense::kNegativeUnate) {
      the_existed_constrain->set_clock_fall();
    }

    if (the_existed_constrain->isRise() && is_rise_need_assign) {
      the_existed_constrain->set_delay_value(delay_value_rise);
      the_existed_constrain->set_fall(false);
      is_rise_need_assign = false;
    }

    if (the_existed_constrain->isFall() && is_fall_need_assign) {
      the_existed_constrain->set_delay_value(delay_value_fall);
      the_existed_constrain->set_rise(false);
      is_fall_need_assign = false;
    }

    the_existed_constrains.pop();
  }

  return 1;
}

}  // namespace ista