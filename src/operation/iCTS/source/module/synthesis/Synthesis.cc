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
#include "Synthesis.h"

#include "CTSAPI.hpp"
namespace icts {

void Synthesis::init()
{
  printLog();
}

void Synthesis::printLog()
{
  LOG_INFO << "\033[1;31m";
  LOG_INFO << R"(                  _   _               _     )";
  LOG_INFO << R"(                 | | | |             (_)    )";
  LOG_INFO << R"(  ___ _   _ _ __ | |_| |__   ___  ___ _ ___ )";
  LOG_INFO << R"( / __| | | | '_ \| __| '_ \ / _ \/ __| / __|)";
  LOG_INFO << R"( \__ \ |_| | | | | |_| | | |  __/\__ \ \__ \)";
  LOG_INFO << R"( |___/\__, |_| |_|\__|_| |_|\___||___/_|___/)";
  LOG_INFO << R"(       __/ |                                )";
  LOG_INFO << R"(      |___/                                 )";
  LOG_INFO << "\033[0m";
  LOG_INFO << "Enter synthesis!";
}

void Synthesis::update()
{
  auto* design = CTSAPIInst.get_design();
  for (auto* net : _nets) {
    design->addNet(net);

    auto& pins = net->get_pins();
    for (auto* pin : pins) {
      auto* inst = pin->get_instance();
      design->addPin(pin);
      design->addInstance(inst);
    }
  }
}

void Synthesis::place(CtsInstance* inst)
{
  _placer->placeInstance(inst);
}

void Synthesis::cancelPlace(CtsInstance* inst)
{
  _placer->cancelPlaceInstance(inst);
}

void Synthesis::incrementalInsertInstance(ClockTopo& clk_topo)
{
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  auto* design = CTSAPIInst.get_design();
  auto* driver = clk_topo.get_driver();
#ifdef DEBUG_ICTS_SYNTHESIS
  DLOG_INFO << "Incremental insert inst " << driver->get_name();
#endif
  if (!driver->is_virtual() && design->findInstance(driver->get_name()) == nullptr) {
    // insert inst to idb
    auto idb_driver = CTSAPIInst.makeIdbInstance(driver->get_name(), driver->get_cell_master());
    // link coordinate and synchronize to cts db
    db_wrapper->linkInstanceCood(driver, idb_driver);
    // DLOG_INFO << "Instance " << driver->get_name() << " has been inserted";
  }
}

void Synthesis::incrementalInsertNet(ClockTopo& clk_topo)
{
#ifdef DEBUG_ICTS_SYNTHESIS
  DLOG_INFO << "Incremental insert net " << clk_topo.get_name();
#endif
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  auto* design = CTSAPIInst.get_design();
  IdbNet* idb_new_net = nullptr;
  CtsNet* net = design->findNet(clk_topo.get_name());
  // get or create clock net.
  auto* driver = clk_topo.get_driver();
  if (net == nullptr) {
    auto idb_driver = db_wrapper->ctsToIdb(driver);
    idb_new_net = CTSAPIInst.makeIdbNet(clk_topo.get_name());
    auto idb_out_pin = db_wrapper->ctsToIdb(driver->get_out_pin());

    CTSAPIInst.connect(idb_driver, idb_out_pin->get_pin_name(), idb_new_net);
    // link net to cts (idbtocts)
    net = db_wrapper->idbToCts(idb_new_net);  // include out pin connection with net
  } else {
    net = driver->get_out_pin()->get_net();
    idb_new_net = db_wrapper->ctsToIdb(net);
    for (auto* load_pin : net->get_load_pins()) {
      auto idb_pin = db_wrapper->ctsToIdb(load_pin);
      CTSAPIInst.disconnect(idb_pin);
      db_wrapper->ctsDisconnect(load_pin);
    }
  }

  // add instance and pin to clock net.
  auto& load_insts = clk_topo.get_loads();
  for (auto* load_inst : load_insts) {
    CtsPin* load_pin = load_inst->get_load_pin();
    auto idb_pin = db_wrapper->ctsToIdb(load_pin);
    if (idb_pin->get_net() != nullptr) {
      if (idb_pin->get_net() != idb_new_net) {
        CTSAPIInst.disconnect(idb_pin);
        db_wrapper->ctsDisconnect(load_pin);
      } else {
        continue;
      }
    }
    auto idb_load = db_wrapper->ctsToIdb(load_inst);
    CTSAPIInst.connect(idb_load, idb_pin->get_pin_name(), idb_new_net);
    db_wrapper->ctsConnect(load_inst, load_pin, net);
  }
  if (!driver->is_virtual() && design->findInstance(driver->get_name()) == nullptr) {
    CTSAPIInst.insertBuffer(driver->get_name());
  }
  // add signal wire to clock net.
  net->clearWires();
  for (auto& signal_wire : clk_topo.get_signal_wires()) {
    net->addSignalWire(signal_wire);
  }
  if (net->is_newly()) {
    _nets.push_back(net);
  }

#ifdef USE_ROUTING

  CTSAPIInst.routingWire(net);

#endif
  // DLOG_INFO << "Net " << net->get_net_name() << " has been inserted";
}

void Synthesis::insertInstance(CtsInstance* inst)
{
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  auto idb_inst = db_wrapper->makeIdbInstance(inst);
  // link coordinate and synchronize to cts db
  db_wrapper->linkInstanceCood(inst, idb_inst);
}

void Synthesis::insertInstance(ClockTopo& clk_topo)
{
  auto* design = CTSAPIInst.get_design();
  auto* driver = clk_topo.get_driver();
  auto* config = CTSAPIInst.get_config();
  auto router_type = config->get_router_type();
  if (router_type == "ZST" || router_type == "BST" || router_type == "UST") {
    place(driver);
  }
#ifdef DEBUG_ICTS_SYNTHESIS
  DLOG_INFO << "Insert inst " << driver->get_name();
#endif
  if (design->findInstance(driver->get_name()) == nullptr) {
    auto* db_wrapper = CTSAPIInst.get_db_wrapper();
    // insert inst to idb
    auto idb_inst = db_wrapper->makeIdbInstance(driver);
    // link coordinate and synchronize to cts db
    db_wrapper->linkInstanceCood(driver, idb_inst);
    // DLOG_INFO << "Instance " << driver->get_name() << " has been inserted";
  }
}

void Synthesis::insertNet(ClockTopo& clk_topo)
{
#ifdef DEBUG_ICTS_SYNTHESIS
  DLOG_INFO << "Insert net " << clk_topo.get_name();
#endif
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  auto* design = CTSAPIInst.get_design();
  CtsNet* net = design->findNet(clk_topo.get_name());
  // get or create clock net.
  auto* driver = clk_topo.get_driver();
  if (net == nullptr) {
    net = new CtsNet(clk_topo.get_name());
    db_wrapper->makeIdbNet(net);
    // connect driver to origin net
    db_wrapper->idbConnect(driver, driver->get_out_pin(), net);
  } else {
    net = driver->get_out_pin()->get_net();
    for (auto* load_pin : net->get_load_pins()) {
      db_wrapper->idbDisconnect(load_pin);
    }
  }
  // add instance and pin to clock net.
  auto& load_insts = clk_topo.get_loads();
  for (auto* load_inst : load_insts) {
    CtsPin* load_pin = load_inst->get_load_pin();
    LOG_FATAL_IF(!load_pin) << "Can't found load pin in inst: " << load_inst->get_name();
    if (load_pin->get_net() != nullptr) {
      if (load_pin->get_net() != net) {
        db_wrapper->idbDisconnect(load_pin);
      } else {
        continue;
      }
    }
    db_wrapper->idbConnect(load_inst, load_pin, net);
  }
  // add signal wire to clock net.
  net->clearWires();
  for (auto& signal_wire : clk_topo.get_signal_wires()) {
    net->addSignalWire(signal_wire);
  }
  if (net->is_newly()) {
    _nets.push_back(net);
  }

#ifdef USE_ROUTING

  CTSAPIInst.routingWire(net);

#endif
  // DLOG_INFO << "Net " << net->get_net_name() << " has been inserted";
}
void Synthesis::insertCtsNetlist()
{
  auto* design = CTSAPIInst.get_design();
  auto clk_topos = design->get_clock_topos();
  for (auto& clk_topo : clk_topos) {
    insertInstance(clk_topo);
  }
#ifdef USE_ROUTING
  CTSAPIInst.iRTinit();
#endif
  for (auto& clk_topo : clk_topos) {
    insertNet(clk_topo);
  }
#ifdef USE_ROUTING
  CTSAPIInst.iRTdestroy();
#endif
  CTSAPIInst.convertDBToTimingEngine();
}

void Synthesis::incrementalInsertCtsNetlist()
{
  auto* design = CTSAPIInst.get_design();
  auto clk_topos = design->get_clock_topos();
  for (auto& clk_topo : clk_topos) {
    incrementalInsertInstance(clk_topo);
  }
#ifdef USE_ROUTING
  CTSAPIInst.iRTinit();
#endif
  for (auto& clk_topo : clk_topos) {
    incrementalInsertNet(clk_topo);
  }
#ifdef USE_ROUTING
  CTSAPIInst.iRTdestroy();
#endif
}

}  // namespace icts