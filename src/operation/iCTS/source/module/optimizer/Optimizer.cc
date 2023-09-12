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
#include "Optimizer.h"

#include "CTSAPI.hpp"
#include "CtsSignalWire.hh"
#include "Topology.h"

namespace icts {

void Optimizer::update() {
  auto *design = CTSAPIInst.get_design();
  auto *db_wrapper = CTSAPIInst.get_db_wrapper();
  int size = _idb_nets.size();
  for (int i = 0; i < size; ++i) {
    auto *idb_net = _idb_nets[i];
    auto &signal_wires = _topos[i];

    auto *clk_net = design->findNet(idb_net->get_net_name());
    if (clk_net == nullptr) {
      auto *new_net = db_wrapper->idbToCts(idb_net);
      new_net->setSignalWire(signal_wires.begin(), signal_wires.end());
      design->addNet(new_net);
    } else {
      clk_net->setSignalWire(signal_wires.begin(), signal_wires.end());
      // The loads of clock net has changed.
      // Need to change loads of clock net later.
    }
  }
}

void Optimizer::optimize(NetIterator begin, NetIterator end) {
  for (auto itr = begin; itr != end; itr++) {
    auto *net = *itr;
    if (net->is_newly()) {
      auto idb_nets = CTSAPIInst.fix(OptiNet(net));
      for (auto *idb_net : idb_nets) {
        _idb_nets.push_back(idb_net);
      }
      reroute(idb_nets.begin(), idb_nets.end());
    }
  }
}

void Optimizer::reroute(IdbNetIterator begin, IdbNetIterator end) {
  for (auto itr = begin; itr != end; itr++) {
    auto topo = reroute(*itr);
    _topos.push_back(topo);
  }
}
std::vector<CtsSignalWire> Optimizer::reroute(IdbNet *idb_net) {
  std::vector<CtsInstance *> load_insts;
  auto idb_load_pins = idb_net->get_load_pins();
  for (auto *idb_load_pin : idb_load_pins) {
    auto *idb_load_inst = idb_load_pin->get_instance();
    auto *load_inst = get_cts_inst(idb_load_inst);
    load_insts.push_back(load_inst);
  }

  Router router;
  Topology<Endpoint> topo;
  router.topoligize(topo, load_insts);
  router.rerouteDME(topo);

  std::vector<CtsSignalWire> result;
  auto edge_itr = topo.edges();
  for (auto itr = edge_itr.first; itr != edge_itr.second; ++itr) {
    auto edge = *itr;
    CtsSignalWire signal_wire(edge.first, edge.second);
    result.push_back(signal_wire);
  }
  auto *idb_driver = idb_net->get_driving_pin()->get_instance();
  Endpoint driver{idb_driver->get_name(), get_location(idb_driver)};
  Endpoint load = topo.value(topo.root());
  CtsSignalWire signal_wire(driver, load);
  result.push_back(signal_wire);

  return result;
}

IdbNet *Optimizer::get_idb_net(const OptiNet &opti_net) const {
  auto *db_wrapper = CTSAPIInst.get_db_wrapper();
  return db_wrapper->ctsToIdb(opti_net.get_clk_net());
}
CtsInstance *Optimizer::get_cts_inst(IdbInstance *idb_inst) const {
  auto *db_wrapper = CTSAPIInst.get_db_wrapper();
  return db_wrapper->idbToCts(idb_inst);
}
Point Optimizer::get_location(IdbInstance *idb_inst) const {
  auto *db_wrapper = CTSAPIInst.get_db_wrapper();
  return db_wrapper->idbToCts(*idb_inst->get_coordinate());
}

}  // namespace icts