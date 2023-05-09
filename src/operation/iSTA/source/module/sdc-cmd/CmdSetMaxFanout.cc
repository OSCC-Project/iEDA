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
 * @file CmdSetMaxFanout.cc
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-10-14
 */
#include "Cmd.hh"
#include "sdc/SdcCollection.hh"

namespace ista {

CmdSetMaxFanout::CmdSetMaxFanout(const char* cmd_name) : TclCmd(cmd_name) {
  auto* source_option = new TclDoubleOption("fanout_value", 1, 0.0);
  addOption(source_option);

  auto* pin_port_arg = new TclStringListOption("object_list", 1, {});
  addOption(pin_port_arg);
}

/**
 * @brief The set_max_fanout cmd legally check.
 *
 * @return unsigned
 */
unsigned CmdSetMaxFanout::check() {
  TclOption* fanout_option = getOptionOrArg("fanout_value");
  TclOption* pin_port_arg = getOptionOrArg("object_list");
  LOG_FATAL_IF(!fanout_option);
  LOG_FATAL_IF(!pin_port_arg);
  if (!(fanout_option->is_set_val() && pin_port_arg->is_set_val())) {
    LOG_ERROR << "'fanout_value' 'object_list' are missing.";
    return 0;
  }

  // double fanout_value = fanout_option->getDoubleVal();
  if (fanout_option->getDoubleVal() < 0) {
    LOG_ERROR << "'fanout_value'>=0 required";
    return 0;
  }

  return 1;
}

/**
 * @brief The set_max_fanout execute body.
 *
 * @return unsigned success return 1, else return 0.
 */
unsigned CmdSetMaxFanout::exec() {
  if (!check()) {
    return 0;
  }

  auto* fanout_value_option = getOptionOrArg("fanout_value");

  double fanout_value = fanout_value_option->getDoubleVal();
  auto* set_max_fanout = new SetMaxFanout(fanout_value);

  setObjectLists(set_max_fanout);
  addTimingDRC2Constrain(set_max_fanout);

  return 1;
}

void CmdSetMaxFanout::addTimingDRC2Constrain(SetMaxFanout* set_max_fanout) {
  Sta* ista = Sta::getOrCreateSta();
  SdcConstrain* the_constrain = ista->getConstrain();
  the_constrain->addTimingDRC(set_max_fanout);
}

void CmdSetMaxFanout::setObjectLists(SetMaxFanout* set_max_fanout) {
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
        LOG_INFO << "set_max_fanout to all design obj.";
      } else {
        auto& obj_list = obj_collection->get_collection_objs();
        for (auto obj : obj_list) {
          std::visit(overloaded{
                         [](SdcCommandObj* sdc_obj) {
                           LOG_FATAL << "set max fanout not support clock yet.";
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

      for (auto* design_obj : pin_ports) {
        objs.insert(design_obj);
        DLOG_INFO << " pin is " << design_obj->getFullName();
      }
    }
  }

  set_max_fanout->set_objs(std::move(objs));
}
}  // namespace ista