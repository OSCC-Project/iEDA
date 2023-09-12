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
 * @file CmdSetPropagatedClock.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The sdc set_propagated_clock implemention.
 * @version 0.1
 * @date 2022-02-23
 */
#include "Cmd.hh"
#include "sdc/SdcClock.hh"

namespace ista {

CmdSetPropagatedClock::CmdSetPropagatedClock(const char* cmd_name)
    : TclCmd(cmd_name) {
  auto* object_list_arg = new TclStringOption("object_list", 1, {});
  addOption(object_list_arg);
}

unsigned CmdSetPropagatedClock::check() { return 1; }

unsigned CmdSetPropagatedClock::exec() {
  Sta* ista = Sta::getOrCreateSta();
  Netlist* design_nl = ista->get_netlist();

  TclOption* object_list_option = getOptionOrArg("object_list");
  auto* object_list_str = object_list_option->getStringVal();

  auto object_list = FindObjOfSdc(object_list_str, design_nl);
  LOG_FATAL_IF(object_list.empty()) << "object list is empty.";

  for (auto& object : object_list) {
    std::visit(
        overloaded{
            [](SdcCommandObj* sdc_obj) {
              if (sdc_obj->isAllClock()) {
                auto* all_clocks = dynamic_cast<SdcAllClocks*>(sdc_obj);
                auto& sdc_clocks = all_clocks->get_clocks();
                for (auto* sdc_clock : sdc_clocks) {
                  sdc_clock->set_is_propagated();
                  LOG_INFO << sdc_clock->get_clock_name() << " is propagated.";
                }
                delete all_clocks;
              }
            },
            [](DesignObject* design_obj) {
              LOG_FATAL << "set_clock_latency not support design object yet.";
            },
        },
        object);
  }

  DLOG_INFO << "exec set_propagated_clock cmd.";

  return 1;
}

}  // namespace ista
