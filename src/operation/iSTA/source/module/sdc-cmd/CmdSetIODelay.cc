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
 * @file CmdSetIODelay.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The io_delay and set_output_delay cmd implemention.
 * @version 0.1
 * @date 2021-05-24
 */
#include <ranges>

#include "Cmd.hh"
#include "log/Log.hh"
#include "netlist/DesignObject.hh"
#include "sdc/SdcAllPorts.hh"
#include "sdc/SdcCommand.hh"
#include "sdc/SdcConstrain.hh"
#include "sdc/SdcSetIODelay.hh"
#include "sta/Sta.hh"
namespace ista {
/**
 * @brief The common add option of io delay.
 *
 * @param cmd
 */
void AddOption(TclCmd* cmd) {
  auto* rise_option = new TclSwitchOption("-rise");
  cmd->addOption(rise_option);

  auto* fall_option = new TclSwitchOption("-fall");
  cmd->addOption(fall_option);

  auto* min_option = new TclSwitchOption("-min");
  cmd->addOption(min_option);

  auto* max_option = new TclSwitchOption("-max");
  cmd->addOption(max_option);

  auto* clock_option = new TclStringOption("-clock", 0, nullptr);
  cmd->addOption(clock_option);

  auto* clock_fall_option = new TclSwitchOption("-clock_fall");
  cmd->addOption(clock_fall_option);

  auto* add_delay_option = new TclSwitchOption("-add_delay");
  cmd->addOption(add_delay_option);

  auto* transition_arg = new TclDoubleOption("delay_value", 1, 0.0);
  cmd->addOption(transition_arg);

  // The pin port arg should be string list, fix me.
  auto* port_list_arg = new TclStringOption("port_list", 1, nullptr);
  cmd->addOption(port_list_arg);
}

CmdSetInputDelay::CmdSetInputDelay(const char* cmd_name) : TclCmd(cmd_name) {
  AddOption(this);
}

CmdSetOutputDelay::CmdSetOutputDelay(const char* cmd_name) : TclCmd(cmd_name) {
  AddOption(this);
}

/**
 * @brief The set_input_delay cmd legally check.
 *
 * @return unsigned
 */
unsigned CmdSetInputDelay::check() { return 1; }

/**
 * @brief The set_output_delay cmd legally check.
 *
 * @return unsigned
 */
unsigned CmdSetOutputDelay::check() { return 1; }

/**
 * @brief The common execute body of io delay.
 *
 * @param io_delay
 */
void commonExec(TclCmd* cmd, SdcSetIODelay* io_delay) {
  auto* rise_option = cmd->getOptionOrArg("-rise");
  auto* fall_option = cmd->getOptionOrArg("-fall");

  // -rise -fall default set, if one set, other not set, we set other not set.
  if (rise_option->is_set_val() && !fall_option->is_set_val()) {
    io_delay->set_fall(false);
  }

  if (fall_option->is_set_val() && !rise_option->is_set_val()) {
    io_delay->set_rise(false);
  }

  auto* max_option = cmd->getOptionOrArg("-max");
  auto* min_option = cmd->getOptionOrArg("-min");
  if (max_option->is_set_val() && !min_option->is_set_val()) {
    io_delay->set_min(false);
  }

  if (min_option->is_set_val() && !max_option->is_set_val()) {
    io_delay->set_max(false);
  }

  auto* clock_fall_option = cmd->getOptionOrArg("-clock_fall");
  if (clock_fall_option->is_set_val()) {
    io_delay->set_clock_fall();
  }

  TclOption* port_list_option = cmd->getOptionOrArg("port_list");
  auto* port_list_str = port_list_option->getStringVal();

  Sta* ista = Sta::getOrCreateSta();
  Netlist* design_nl = ista->get_netlist();

  auto object_list = FindObjOfSdc(port_list_str, design_nl);
  LOG_FATAL_IF(object_list.empty())
      << "object list " << port_list_str << " is empty.";

  std::set<DesignObject*> ports;
  for (auto& object : object_list) {
    std::visit(
        overloaded{
            [&ports](SdcCommandObj* sdc_obj) {
              if (sdc_obj->isAllInputPorts()) {
                auto* all_input_ports =
                    dynamic_cast<SdcAllInputPorts*>(sdc_obj);
                auto& input_ports = all_input_ports->get_input_ports();
                std::ranges::for_each(input_ports, [&ports](auto* input_port) {
                  ports.insert(input_port);
                });
              } else if (sdc_obj->isAllOutputPorts()) {
                auto* all_output_ports =
                    dynamic_cast<SdcAllOutputPorts*>(sdc_obj);
                auto& output_ports = all_output_ports->get_output_ports();
                std::ranges::for_each(
                    output_ports,
                    [&ports](auto* output_port) { ports.insert(output_port); });
              }
            },
            [&ports](DesignObject* design_obj) { ports.insert(design_obj); },
        },
        object);
  }

  io_delay->set_objs(std::move(ports));

  SdcConstrain* the_constrain = ista->getConstrain();
  the_constrain->addIOConstrain(io_delay);
}

/**
 * @brief The set_input_delay execute body.
 *
 * @return unsigned success return 1, else return 0.
 */
unsigned CmdSetInputDelay::exec() {
  if (!check()) {
    return 0;
  }
  Sta* ista = Sta::getOrCreateSta();
  Netlist* design_nl = ista->get_netlist();

  auto* clock_option = getOptionOrArg("-clock");
  auto* clock_str = clock_option->getStringVal();
  SdcConstrain* the_constrain = ista->getConstrain();
  std::string clock_name =
      GetClockName(clock_str, design_nl, the_constrain).front();

  auto* delay_value = getOptionOrArg("delay_value");

  auto* set_input_delay =
      new SdcSetInputDelay(get_cmd_name(), clock_name.c_str(),
                           ista->convertTimeUnit(delay_value->getDoubleVal()));

  LOG_INFO << "set input delay sdc line no : "
           << set_input_delay->get_line_no();
  commonExec(this, set_input_delay);

  return 1;
}

/**
 * @brief The set_output_delay execute body.
 *
 * @return unsigned success return 1, else return 0.
 */
unsigned CmdSetOutputDelay::exec() {
  if (!check()) {
    return 0;
  }

  Sta* ista = Sta::getOrCreateSta();
  Netlist* design_nl = ista->get_netlist();

  auto* clock_option = getOptionOrArg("-clock");
  auto* clock_str = clock_option->getStringVal();
  SdcConstrain* the_constrain = ista->getConstrain();
  std::string clock_name =
      GetClockName(clock_str, design_nl, the_constrain).front();
  auto* delay_value = getOptionOrArg("delay_value");

  auto* set_output_delay =
      new SdcSetOutputDelay(get_cmd_name(), clock_name.c_str(),
                            ista->convertTimeUnit(delay_value->getDoubleVal()));

  LOG_INFO << "set output delay sdc line no : "
           << set_output_delay->get_line_no();
  commonExec(this, set_output_delay);

  return 1;
}

}  // namespace ista
