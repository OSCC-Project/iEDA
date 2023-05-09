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
 * @file CmdCurrentDesign.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2021-10-07
 */

#include "Cmd.hh"
#include "sdc/SdcCollection.hh"

namespace ista {

CmdCurrentDesign::CmdCurrentDesign(const char* cmd_name) : TclCmd(cmd_name) {}

unsigned CmdCurrentDesign::check() { return 1; }

/**
 * @brief execute the current_design cmd.
 *
 * @return unsigned
 */
unsigned CmdCurrentDesign::exec() {
  Sta* ista = Sta::getOrCreateSta();
  auto* nl = ista->get_netlist();
  SdcCollectionObj collection_obj(nl);

  auto* nl_collection = new SdcCollection(
      SdcCollection::CollectionType::kNetlist, {collection_obj});

  char* result = TclEncodeResult::encode(nl_collection);
  ScriptEngine::getOrCreateInstance()->setResult(result);

  return 1;
}

}  // namespace ista
