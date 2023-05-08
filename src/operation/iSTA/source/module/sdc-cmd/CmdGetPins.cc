/**
 * @file CmdGetPins.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2021-10-14
 *
 * @copyright Copyright (c) 2021
 *
 */
#include "Cmd.hh"
#include "sdc/SdcCollection.hh"

namespace ista {

CmdGetPins::CmdGetPins(const char* cmd_name) : TclCmd(cmd_name) {
  auto* clock_list_arg = new TclStringListOption("patterns", 1, {});
  addOption(clock_list_arg);
}

unsigned CmdGetPins::check() { return 1; }

/**
 * @brief execute the nl.
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
