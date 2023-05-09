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
 * @file CmdGetClocks.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2021-10-14
 */
#include "Cmd.hh"
#include "sdc/SdcCollection.hh"

namespace ista {

CmdGetClocks::CmdGetClocks(const char* cmd_name) : TclCmd(cmd_name) {
  auto* clock_list_arg = new TclStringListOption("patterns", 1, {});
  addOption(clock_list_arg);
}

unsigned CmdGetClocks::check() { return 1; }

/**
 * @brief execute the nl.
 *
 * @return unsigned
 */
unsigned CmdGetClocks::exec() {
  Sta* ista = Sta::getOrCreateSta();
  auto& the_constrain = ista->get_constrains();
  auto& the_clocks = the_constrain->get_sdc_clocks();
  auto* clock_list_arg = getOptionOrArg("patterns");
  auto clock_list = clock_list_arg->getStringList();

  std::vector<SdcCollectionObj> obj_list;

  for (auto& [clock_name, the_clock] : the_clocks) {
    auto it = std::find_if(
        clock_list.begin(), clock_list.end(), [clock_name](auto clock_pattern) {
          if ((clock_pattern == "*") || (clock_name == clock_pattern)) {
            return true;
          }
          // TODO(to taosimin) wildcard match
          return false;
        });
    if (it != clock_list.end()) {
      obj_list.emplace_back(the_clock.get());
    }
  }

  auto* sdc_collection = new SdcCollection(
      SdcCollection::CollectionType::kClock, std::move(obj_list));
  the_constrain->addSdcCollection(sdc_collection);

  char* result = TclEncodeResult::encode(sdc_collection);
  ScriptEngine::getOrCreateInstance()->setResult(result);

  return 1;
}
}  // namespace ista
