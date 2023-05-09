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
 * @file CmdAllClock.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The sdc all_clock implemention.
 * @version 0.1
 * @date 2022-02-23
 */
#include <tuple>

#include "Cmd.hh"
#include "sdc/SdcClock.hh"

namespace ista {

CmdAllClocks::CmdAllClocks(const char* cmd_name) : TclCmd(cmd_name) {}

unsigned CmdAllClocks::check() { return 1; }

/**
 * @brief execute the all_clocks cmd.
 *
 * @return unsigned
 */
unsigned CmdAllClocks::exec() {
  Sta* ista = Sta::getOrCreateSta();
  SdcConstrain* the_constrain = ista->getConstrain();
  auto& sdc_clocks = the_constrain->get_sdc_clocks();

  auto* all_clocks = new SdcAllClocks();
  for (auto& [clock_name, clock] : sdc_clocks) {
    all_clocks->addClock(clock.get());
  }
  SdcCollectionObj collection_obj(all_clocks);

  auto* all_clocks_collection = new SdcCollection(
      SdcCollection::CollectionType::kAllClocks, {collection_obj});

  char* result = TclEncodeResult::encode(all_clocks_collection);
  ScriptEngine::getOrCreateInstance()->setResult(result);

  DLOG_INFO << "exec all_clocks cmd.";

  return 1;
}

}  // namespace ista