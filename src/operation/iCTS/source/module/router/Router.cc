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
 * @file Router.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "Router.hh"

#include "CTSAPI.hh"
#include "CtsDBWrapper.hh"
#include "Solver.hh"
#include "TimingPropagator.hh"
#include "usage/usage.hh"
namespace icts {
void Router::init()
{
  CTSAPIInst.saveToLog("--RC Info--");
  auto* design = CTSAPIInst.get_design();
  // report unit res & cap
  CTSAPIInst.saveToLog("Unit RES (H): ", CTSAPIInst.getClockUnitRes(LayerPattern::kH), " ohm");
  CTSAPIInst.saveToLog("Unit CAP (H): ", CTSAPIInst.getClockUnitCap(LayerPattern::kH), " pF");
  CTSAPIInst.saveToLog("Unit RES (V): ", CTSAPIInst.getClockUnitRes(LayerPattern::kV), " ohm");
  CTSAPIInst.saveToLog("Unit CAP (V): ", CTSAPIInst.getClockUnitCap(LayerPattern::kV), " pF");
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
  CTSAPIInst.saveToLog("");
}
void Router::build()
{
  ieda::Stats stats;
  CTSAPIInst.saveToLog("--Clock Net Info--");
  for (auto* clock : _clocks) {
    auto* design = CTSAPIInst.get_design();
    auto& clock_nets = clock->get_clock_nets();
    for (auto* clk_net : clock_nets) {
      CTSAPIInst.saveToLog("Net name: ", clk_net->get_net_name());
      LOG_INFO << "Net name: " << clk_net->get_net_name();
      design->resetId();
      auto sink_pins = getSinkPins(clk_net);
      auto buf_pins = getBufferPins(clk_net);
      CTSAPIInst.saveToLog("\tSink pins num: ", sink_pins.size());
      LOG_INFO << "\tSink pins num: " << sink_pins.size();
      CTSAPIInst.saveToLog("\tBuffer pins num: ", buf_pins.size());
      LOG_INFO << "\tBuffer pins num: " << buf_pins.size();
      routing(clk_net);
      clk_net->setClockRouted();
    }
  }
  CTSAPIInst.saveToLog("");
}
void Router::update()
{
  LOG_INFO << "Synthesis data to cts design...";
  // update to cts design, idb and sta
  std::ranges::for_each(_solver_set.get_nets(), [&](Net* net) {
    std::ranges::for_each(net->get_pins(), [&](Pin* pin) { synthesisPin(pin); });
    synthesisNet(net);
  });
  LOG_INFO << "Convert data to timing engine...";
  CTSAPIInst.convertDBToTimingEngine();
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

void Router::routing(CtsNet* clk_net)
{
  auto pins = clk_net->get_load_pins();
  if (pins.empty()) {
    LOG_WARNING << "Net: " << clk_net->get_net_name() << " is empty!";
    return;
  }
  auto net_name = clk_net->get_net_name();
  // total topology
  auto solver = Solver(net_name, clk_net->get_driver_pin(), pins);
  solver.run();
  auto clk_nets = solver.get_solver_nets();
  if (clk_nets.empty()) {
    return;
  }
  std::ranges::for_each(clk_nets, [&](Net* net) {
    _solver_set.add_net(net);
    std::ranges::for_each(net->get_pins(), [&](Pin* pin) { _solver_set.add_pin(pin); });
  });
}

std::vector<CtsPin*> Router::getSinkPins(CtsNet* clk_net)
{
  std::vector<CtsPin*> pins;
  for (auto* load_pin : clk_net->get_load_pins()) {
    auto* load_inst = load_pin->get_instance();
    auto inst_type = load_inst->get_type();
    if (inst_type != CtsInstanceType::kBuffer && inst_type != CtsInstanceType::kMux) {
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

void Router::synthesisPin(Pin* pin)
{
  auto* design = CTSAPIInst.get_design();
  if (design->findPin(pin->get_name()) != nullptr) {
    return;
  }

  // It's a new insert buffer pin
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  auto buf = pin->get_inst();
  if (!buf) {
    // is CLK
    return;
  }
  auto* cts_buf = new CtsInstance(buf->get_name(), buf->get_cell_master(), CtsInstanceType::kBuffer, buf->get_location());
  db_wrapper->linkIdb(cts_buf);

  // update driver pin name
  auto* driver_pin = buf->get_driver_pin();
  auto* cts_driver_pin = cts_buf->get_out_pin();
  driver_pin->set_name(cts_driver_pin->get_full_name());
  // update load pin name
  auto* load_pin = buf->get_load_pin();
  auto* cts_load_pin = cts_buf->get_in_pin();
  load_pin->set_name(cts_load_pin->get_full_name());

  design->addInstance(cts_buf);
  design->addPin(cts_driver_pin);
  design->addPin(cts_load_pin);
}

void Router::synthesisNet(Net* net)
{
  auto* design = CTSAPIInst.get_design();
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  auto* cts_net = design->findNet(net->get_name());
  if (!cts_net) {
    cts_net = new CtsNet(net->get_name());
    db_wrapper->makeIdbNet(cts_net);
    design->addNet(cts_net);
  }
  // It's a new insert buffer net
  std::ranges::for_each(net->get_pins(), [&](Pin* pin) {
    auto* cts_pin = design->findPin(pin->get_name());
    LOG_FATAL_IF(!cts_pin) << "Can't found pin " << pin->get_name() << " in net " << net->get_name();
    if (cts_pin->is_io()) {
      return;
    }
    db_wrapper->idbDisconnect(cts_pin);
    db_wrapper->idbConnect(cts_pin, cts_net);
  });
  // synthesis wire
  auto* driver_pin = net->get_driver_pin();
  auto id = driver_pin->getMaxId();
  driver_pin->preOrder([&](Node* node) {
    auto* parent = node->get_parent();
    if (parent == nullptr) {
      return;
    }
    auto current_loc = node->get_location();
    auto parent_loc = parent->get_location();
    auto parent_name = parent->isPin() ? dynamic_cast<Pin*>(parent)->get_inst()->get_name() : parent->get_name();
    auto current_name = node->isPin() ? dynamic_cast<Pin*>(node)->get_inst()->get_name() : node->get_name();
    auto require_nake = node->get_required_snake();
    if (require_nake > 0) {
      auto require_snake = std::ceil(require_nake * TimingPropagator::getDbUnit());
      auto delta_x = std::abs(current_loc.x() - parent_loc.x());
      auto trunk_x = (parent_loc.x() + current_loc.x() + delta_x + require_snake) / 2;
      auto snake_p1 = Point(trunk_x, parent_loc.y());
      auto snake_p2 = Point(trunk_x, current_loc.y());
      if (!(CTSAPIInst.isInDie(snake_p1) && CTSAPIInst.isInDie(snake_p2))) {
        // is not in die
        trunk_x = (parent_loc.x() + current_loc.x() - delta_x - require_snake) / 2;
        snake_p1 = Point(trunk_x, parent_loc.y());
        snake_p2 = Point(trunk_x, current_loc.y());
      }
      std::vector<std::string> name_vec = {parent_name, "steiner_" + std::to_string(++id), "steiner_" + std::to_string(++id), current_name};
      std::vector<Point> point_vec = {parent_loc, snake_p1, snake_p2, current_loc};
      for (size_t i = 0; i < name_vec.size() - 1; ++i) {
        cts_net->add_signal_wire(CtsSignalWire(Endpoint{name_vec[i], point_vec[i]}, Endpoint{name_vec[i + 1], point_vec[i + 1]}));
      }
    } else {
      if (Point::isRectilinear(parent_loc, current_loc)) {
        cts_net->add_signal_wire(CtsSignalWire(Endpoint{parent_name, parent_loc}, Endpoint{current_name, current_loc}));
      } else {
        auto trunk_loc = Point(parent_loc.x(), current_loc.y());
        auto trunk_name = "steiner_" + std::to_string(++id);
        cts_net->add_signal_wire(CtsSignalWire(Endpoint{parent_name, parent_loc}, Endpoint{trunk_name, trunk_loc}));
        cts_net->add_signal_wire(CtsSignalWire(Endpoint{trunk_name, trunk_loc}, Endpoint{current_name, current_loc}));
      }
    }
  });
  design->addSolverNet(net);
}

}  // namespace icts