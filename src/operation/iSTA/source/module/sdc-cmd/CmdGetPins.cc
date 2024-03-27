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
 * @file CmdGetPins.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2021-10-14
 */
#include "Cmd.hh"
#include "sdc/SdcCollection.hh"

namespace ista {

CmdGetPins::CmdGetPins(const char* cmd_name) : TclCmd(cmd_name) {
  auto* patterns_arg = new TclStringListOption("patterns", 1, {});
  addOption(patterns_arg);
}

unsigned CmdGetPins::check() { return 1; }

/**
 * @brief execute the get_pins.
 *
 * @return unsigned
 */
unsigned CmdGetPins::exec() {
  Sta* ista = Sta::getOrCreateSta();
  auto& the_constrain = ista->get_constrains();

  auto* pin_list_arg = getOptionOrArg("patterns");
  auto pin_list = pin_list_arg->getStringList();

  std::vector<SdcCollectionObj> obj_list;

  Netlist* design_nl = ista->get_netlist();
  for (auto& pin_name : pin_list) {
    auto pin_ports = design_nl->findObj(pin_name.c_str(), false, false);
    for (auto* design_obj : pin_ports) {
      obj_list.emplace_back(design_obj);
    }
  }

  auto* sdc_collection = new SdcCollection(SdcCollection::CollectionType::kPin,
                                           std::move(obj_list));

  the_constrain->addSdcCollection(sdc_collection);

  char* result = TclEncodeResult::encode(sdc_collection);
  ScriptEngine::getOrCreateInstance()->setResult(result);

  return 1;
}
}  // namespace ista
