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
#include "Router.h"

#include "CTSAPI.hpp"
#include "CtsDBWrapper.h"
#include "GOCA.hh"
#include "TimingPropagator.hh"
namespace icts {
void Router::init()
{
  CTSAPIInst.saveToLog("\n\nRouter Log");
  auto* design = CTSAPIInst.get_design();
  // report unit res & cap
  CTSAPIInst.saveToLog("\nRouter unit res: ", CTSAPIInst.getClockUnitRes());
  CTSAPIInst.saveToLog("Router unit cap: ", CTSAPIInst.getClockUnitCap());
  printLog();
  auto& clocks = design->get_clocks();
  for (auto& clock : clocks) {
    _clocks.push_back(clock);
    for (auto* clk_net : clock->get_clock_nets()) {
      auto& clk_pins = clk_net->get_pins();
      for (auto* clk_pin : clk_pins) {
        auto* clk_inst = clk_pin->get_instance();
        auto& inst_name = clk_inst->get_name();
        if (_name_to_inst.count(inst_name) == 0) {
          _name_to_inst[inst_name] = clk_inst;
        }
      }
    }
  }
}
void Router::update()
{
  auto* design = CTSAPIInst.get_design();
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  for (auto& net : design->get_nets()) {
    // debug
    if (net->get_net_name() == "sys_clk_25m_buf") {
      int a = 1;
    }
    for (auto* load_pin : net->get_load_pins()) {
      auto* load_inst = load_pin->get_instance();
      if (load_pin->get_net() != nullptr) {
        if (load_pin->get_net() != net) {
          db_wrapper->idbDisconnect(load_pin);
        } else {
          continue;
        }
      }
      db_wrapper->idbConnect(load_inst, load_pin, net);
    }
  }
  for (auto& net : _insert_nets) {
    auto* driver_inst = net->get_driver_inst();
    db_wrapper->updateCell(driver_inst);
    db_wrapper->makeIdbNet(net);
    for (auto* load_pin : net->get_load_pins()) {
      auto* load_inst = load_pin->get_instance();
      if (load_pin->get_net() != nullptr) {
        if (load_pin->get_net() != net) {
          db_wrapper->idbDisconnect(load_pin);
        } else {
          continue;
        }
      }
      db_wrapper->idbConnect(load_inst, load_pin, net);
    }
    std::ranges::for_each(driver_inst->get_pin_list(), [&design](CtsPin* pin) { design->addPin(pin); });
    design->addInstance(driver_inst);
    design->addNet(net);
  }
  CTSAPIInst.convertDBToTimingEngine();
}

void Router::build()
{
  for (auto* clock : _clocks) {
    auto* design = CTSAPIInst.get_design();
    auto& clock_nets = clock->get_clock_nets();
    for (auto* clk_net : clock_nets) {
      design->resetId();
      // debug
      if (clk_net->get_net_name() == "sys_clk_25m_buf") {
        int a = 0;
      }
      if (!routeAble(clk_net)) {
        continue;
      }
      gocaRouting(clk_net);
      clk_net->setClockRouted();
      breakLongWire(clk_net);
    }
  }
}

void Router::printLog()
{
  LOG_INFO << "\033[1;31m";
  LOG_INFO << R"(                  _             )";
  LOG_INFO << R"(                 | |            )";
  LOG_INFO << R"(  _ __ ___  _   _| |_ ___ _ __  )";
  LOG_INFO << R"( | '__/ _ \| | | | __/ _ \ '__| )";
  LOG_INFO << R"( | | | (_) | |_| | ||  __/ |    )";
  LOG_INFO << R"( |_|  \___/ \__,_|\__\___|_|    )";
  LOG_INFO << "\033[0m";
  LOG_INFO << "Enter router!";
}

void Router::gocaRouting(CtsNet* clk_net)
{
  auto pins = getSinkPins(clk_net);
  if (pins.size() <= 1) {
    return;
  }
  auto net_name = clk_net->get_net_name();
  // total topology
  auto goca = GOCA(net_name, pins);
  goca.run();
  auto clk_nets = goca.get_clk_nets();
  for (auto& net : clk_nets) {
    _insert_nets.emplace_back(net);
  }
  if (clk_nets.empty()) {
    return;
  }
  auto* root_net = clk_nets.back();
  auto* root_driver = root_net->get_driver_pin();
  auto* root_inst = root_driver->get_instance();
  auto* load_pin = root_inst->get_load_pin();
  removeSinkPin(clk_net);
  clk_net->addPin(load_pin);
}

std::vector<CtsPin*> Router::getSinkPins(CtsNet* clk_net)
{
  std::vector<CtsPin*> pins;
  for (auto* load_pin : clk_net->get_load_pins()) {
    auto* load_inst = load_pin->get_instance();
    auto inst_type = load_inst->get_type();
    if (inst_type != CtsInstanceType::kBuffer) {
      pins.push_back(load_pin);
    }
  }
  return pins;
}

std::vector<CtsPin*> Router::getBufferPins(CtsNet* clk_net)
{
  std::vector<CtsPin*> pins;
  for (auto* load_pin : clk_net->get_load_pins()) {
    auto* load_inst = load_pin->get_instance();
    auto inst_type = load_inst->get_type();
    if (inst_type != CtsInstanceType::kSink) {
      pins.push_back(load_pin);
    }
  }
  return pins;
}

void Router::removeSinkPin(CtsNet* clk_net)
{
  auto* driver_pin = clk_net->get_driver_pin();
  auto pins = getBufferPins(clk_net);
  clk_net->clear();
  for (auto* pin : pins) {
    clk_net->addPin(pin);
  }
  clk_net->addPin(driver_pin);
}

void Router::breakLongWire(CtsNet* clk_net)
{
  std::string net_name = clk_net->get_net_name();
  auto* driver_inst = clk_net->get_driver_inst();
  auto load_pins = clk_net->get_load_pins();
  CtsInstance* root_buf = nullptr;
  if (load_pins.size() > 1) {
    // total topology
    auto goca = GOCA(net_name + "_rebuild", load_pins);
    goca.simpleRun();
    auto clk_nets = goca.get_clk_nets();
    for (auto& net : clk_nets) {
      _insert_nets.emplace_back(net);
    }
    root_buf = clk_nets.back()->get_driver_inst();
  } else {
    root_buf = load_pins.front()->get_instance();
  }
  auto parent_loc = driver_inst->get_location();
  auto child_loc = root_buf->get_location();
  auto length = TimingPropagator::calcLen(parent_loc, child_loc);
  int required_buf_num = std::ceil(length / TimingPropagator::getMaxLength());
  if (required_buf_num > 1) {
    auto* db_wrapper = CTSAPIInst.get_db_wrapper();
    auto cur_loc = child_loc;
    auto delta_loc = (parent_loc - child_loc) / (required_buf_num + 1);
    auto* cur_load = root_buf;
    while (required_buf_num--) {
      cur_loc += delta_loc;
      auto sub_net_name = CTSAPIInst.toString(net_name, "_break_", required_buf_num);
      auto* cur_net = new CtsNet(sub_net_name);
      auto* buf
          = new CtsInstance(sub_net_name + "_buf", TimingPropagator::getMinSizeLib()->get_cell_master(), CtsInstanceType::kBuffer, cur_loc);
      db_wrapper->linkIdb(buf);
      cur_net->addPin(buf->get_out_pin());
      cur_net->addPin(cur_load->get_in_pin());
      int steiner_id = 0;
      if (pgl::rectilinear(cur_loc, cur_load->get_location())) {
        cur_net->addSignalWire(CtsSignalWire(Endpoint{buf->get_name(), cur_loc}, Endpoint{cur_load->get_name(), cur_load->get_location()}));
      } else {
        auto trunk_loc = Point(cur_loc.x(), cur_load->get_location().y());
        auto trunk_name = "steiner_" + std::to_string(steiner_id++);
        cur_net->addSignalWire(CtsSignalWire(Endpoint{buf->get_name(), cur_loc}, Endpoint{trunk_name, trunk_loc}));
        cur_net->addSignalWire(CtsSignalWire(Endpoint{trunk_name, trunk_loc}, Endpoint{cur_load->get_name(), cur_load->get_location()}));
      }
      _insert_nets.emplace_back(cur_net);
      cur_load = buf;
    }
    root_buf = cur_load;
  }
  for (auto pin : load_pins) {
    clk_net->removePin(pin);
  }
  clk_net->addPin(root_buf->get_load_pin());
  int steiner_id = 0;
  if (pgl::rectilinear(parent_loc, child_loc)) {
    clk_net->addSignalWire(CtsSignalWire(Endpoint{driver_inst->get_name(), parent_loc}, Endpoint{root_buf->get_name(), child_loc}));
  } else {
    auto trunk_loc = Point(parent_loc.x(), child_loc.y());
    auto trunk_name = "steiner_" + std::to_string(steiner_id++);
    clk_net->addSignalWire(CtsSignalWire(Endpoint{driver_inst->get_name(), parent_loc}, Endpoint{trunk_name, trunk_loc}));
    clk_net->addSignalWire(CtsSignalWire(Endpoint{trunk_name, trunk_loc}, Endpoint{root_buf->get_name(), child_loc}));
  }
}

bool Router::routeAble(CtsNet* clk_net)
{
  auto sink_pins = getSinkPins(clk_net);
  if (sink_pins.size() <= 1) {
    auto* driver = clk_net->get_driver_inst();
    // ignore multiple load pin instance
    if (driver->isMux()) {
      return false;
    }
    // ignore mux clk
    if (driver->isSink()) {
      return false;
    }
    // ignore io net
    auto pins = clk_net->get_pins();
    bool is_io = false;
    for (auto pin : pins) {
      if (pin->is_io()) {
        is_io = true;
        break;
      }
    }
    if (is_io) {
      return false;
    }
  }
  return true;
}

}  // namespace icts