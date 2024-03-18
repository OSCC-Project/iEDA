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
 * @file CmdAllPorts.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The sdc all_inputs all_outputs command implemention.
 * @version 0.1
 * @date 2024-02-06
 */
#include "Cmd.hh"
#include "sdc/SdcAllPorts.hh"

namespace ista {

CmdAllInputs::CmdAllInputs(const char* cmd_name) : TclCmd(cmd_name) {}

unsigned CmdAllInputs::check() { return 1; }

/**
 * @brief execute the all_inputs cmd.
 *
 * @return unsigned
 */
unsigned CmdAllInputs::exec() {
  Sta* ista = Sta::getOrCreateSta();
  auto* netlist = ista->get_netlist();

  auto* all_input_ports = new SdcAllInputPorts();

  Port* port;
  FOREACH_PORT(netlist, port) {
    if (port->isInput()) {
      all_input_ports->addPort(port);
    }
  }

  SdcCollectionObj collection_obj(all_input_ports);

  auto* all_input_ports_collection = new SdcCollection(
      SdcCollection::CollectionType::kAllInputPorts, {collection_obj});

  char* result = TclEncodeResult::encode(all_input_ports_collection);
  ScriptEngine::getOrCreateInstance()->setResult(result);

  DLOG_INFO << "exec all_inputs cmd.";

  return 1;
}

CmdAllOutputs::CmdAllOutputs(const char* cmd_name) : TclCmd(cmd_name) {}

unsigned CmdAllOutputs::check() { return 1; }

/**
 * @brief execute the all_outputs cmd.
 *
 * @return unsigned
 */
unsigned CmdAllOutputs::exec() {
  Sta* ista = Sta::getOrCreateSta();
  auto* netlist = ista->get_netlist();

  auto* all_output_ports = new SdcAllOutputPorts();

  Port* port;
  FOREACH_PORT(netlist, port) {
    if (port->isOutput()) {
      all_output_ports->addPort(port);
    }
  }

  SdcCollectionObj collection_obj(all_output_ports);

  auto* all_output_ports_collection = new SdcCollection(
      SdcCollection::CollectionType::kAllOutputPorts, {collection_obj});

  char* result = TclEncodeResult::encode(all_output_ports_collection);
  ScriptEngine::getOrCreateInstance()->setResult(result);

  DLOG_INFO << "exec all_outputs cmd.";

  return 1;
}

}  // namespace ista