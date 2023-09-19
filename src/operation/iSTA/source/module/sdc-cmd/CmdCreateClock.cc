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
 * @file CmdCreateClock.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The sdc create_clock cmd implemention.
 * @version 0.1
 * @date 2021-03-02
 */

#include <set>

#include "Cmd.hh"
#include "netlist/DesignObject.hh"
#include "sdc/SdcClock.hh"
#include "sdc/SdcCollection.hh"
#include "sdc/SdcConstrain.hh"
#include "sta/Sta.hh"

namespace ista {

CmdCreateClock::CmdCreateClock(const char* cmd_name) : TclCmd(cmd_name) {
  auto* name_option = new TclStringOption("-name", 0, nullptr);
  addOption(name_option);

  auto* period_option = new TclDoubleOption("-period", 0, 0.0);
  addOption(period_option);

  auto* waveform_option = new TclDoubleListOption("-waveform", 0, {});
  addOption(waveform_option);

  auto* pin_port_arg = new TclStringListOption("source_objects", 1, {});
  addOption(pin_port_arg);
}

/**
 * @brief The create_clock cmd legally check.
 *
 * @return unsigned
 */
unsigned CmdCreateClock::check() {
  TclOption* period_option = getOptionOrArg("-period");
  if (!period_option->is_set_val()) {
    LOG_ERROR << "The period is not set.";
    return 0;
  }

  if (period_option->getDoubleVal() < 0.0) {
    LOG_ERROR << "The period must be greater than or equal to zero.";
    return 0;
  }

  return 1;
}

/**
 * @brief The create_clock execute body.
 *
 * @return unsigned success return 1, else return 0.
 */
unsigned CmdCreateClock::exec() {
  if (!check()) {
    return 0;
  }

  TclOption* name_option = getOptionOrArg("-name");
  TclOption* period_option = getOptionOrArg("-period");

  Sta* ista = Sta::getOrCreateSta();
  SdcConstrain* the_constrain = ista->getConstrain();
  const char* clk_name = name_option->getStringVal();
  LOG_FATAL_IF(!clk_name) << "clock name can not be empty.";
  SdcClock* the_clock = new SdcClock(clk_name);
  double period = ista->convertTimeUnit(period_option->getDoubleVal());

  the_clock->set_period(period);

  TclOption* waveform_option = getOptionOrArg("-waveform");
  if (waveform_option && waveform_option->is_set_val()) {
    auto waveform_val = waveform_option->getDoubleList();
    SdcClock::SdcWaveform edges = {waveform_val[0], waveform_val[1]};
    the_clock->set_edges(std::move(edges));
  } else {
    SdcClock::SdcWaveform edges = {0.0, period / 2};
    the_clock->set_edges(std::move(edges));
  }
  Netlist* design_nl = ista->get_netlist();

  TclOption* pin_port_option = getOptionOrArg("source_objects");
  std::vector<std::string> pin_port_strs = pin_port_option->getStringList();
  std::set<DesignObject*> pins;
  for (auto& pin_port_name : pin_port_strs) {
    auto object_list = FindObjOfSdc(pin_port_name, design_nl);
    LOG_FATAL_IF(object_list.empty()) << "object list is empty.";

    for (auto& object : object_list) {
      std::visit(overloaded{
                     [](SdcCommandObj* sdc_obj) {
                       LOG_FATAL << "not support sdc obj.";
                     },
                     [&pins, clk_name](DesignObject* design_obj) {
                       DLOG_INFO
                           << "create clock " << clk_name
                           << " for pin/port: " << design_obj->getFullName();
                       pins.insert(design_obj);
                     },
                 },
                 object);
    }
  }

  the_clock->set_objs(std::move(pins));

  the_constrain->addClock(the_clock);
  return 1;
}

}  // namespace ista
