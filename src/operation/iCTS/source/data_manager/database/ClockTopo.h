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
#pragma once
#include <algorithm>
#include <vector>

#include "CtsInstance.h"
#include "CtsSignalWire.h"

namespace icts {
using std::vector;

class ClockTopo {
 public:
  ClockTopo() = default;
  ClockTopo(const string& name) : _topo_name(name) {}
  ClockTopo(const ClockTopo& clk_tree) = default;
  ~ClockTopo() = default;

  string get_name() const { return _topo_name; }
  int get_level() const;
  CtsInstance* get_driver() const { return _driver_inst; }
  vector<CtsInstance*>& get_loads() { return _load_insts; }
  vector<CtsSignalWire>& get_signal_wires() { return _signal_wires; }

  void set_name(const string& name) { _topo_name = name; }

  void add_driver(CtsInstance* inst) { _driver_inst = inst; }
  void add_load(CtsInstance* inst) { _load_insts.emplace_back(inst); }
  void add_signal_wire(const CtsSignalWire& wire) {
    _signal_wires.emplace_back(wire);
  }

  void connect_load(CtsInstance* load) {
    auto driver_loc = _driver_inst->get_location();
    auto load_loc = load->get_location();
    add_load(load);
    Endpoint epl{_driver_inst->get_name(), driver_loc};
    Endpoint epr{load->get_name(), load_loc};
    if (pgl::rectilinear(driver_loc, load_loc)) {
      CtsSignalWire wire(epl, epr);
      add_signal_wire(wire);
    } else {
      auto trunk_loc = Point(driver_loc.x(), load_loc.y());
      Endpoint ep_trunk = {"steiner_999", trunk_loc};
      CtsSignalWire wire_1(epl, ep_trunk);
      add_signal_wire(wire_1);
      CtsSignalWire wire_2(ep_trunk, epr);
      add_signal_wire(wire_2);
    }
  }

  void disconnect_load(CtsInstance* load) {
    _load_insts.erase(std::remove_if(_load_insts.begin(), _load_insts.end(),
                                     [&load](const CtsInstance* inst) {
                                       return load == inst;
                                     }),
                      _load_insts.end());
  }

  void clearWires() { _signal_wires.clear(); }

  void buildWires() {
    clearWires();
    Endpoint epl{_driver_inst->get_name(), _driver_inst->get_location()};
    int steiner_id = 0;
    for (auto* load : _load_insts) {
      Endpoint epr{load->get_name(), load->get_location()};
      if (pgl::rectilinear(_driver_inst->get_location(),
                           load->get_location())) {
        add_signal_wire(CtsSignalWire(epl, epr));
      } else {
        auto trunk_loc =
            Point(_driver_inst->get_location().x(), load->get_location().y());
        Endpoint ep_trunk{"steiner_" + std::to_string(steiner_id++), trunk_loc};
        add_signal_wire(CtsSignalWire(epl, ep_trunk));
        add_signal_wire(CtsSignalWire(ep_trunk, epr));
      }
    }
  }

 private:
  std::string _topo_name;
  CtsInstance* _driver_inst = nullptr;
  std::vector<CtsInstance*> _load_insts;
  std::vector<CtsSignalWire> _signal_wires;
};

inline int ClockTopo::get_level() const { return _driver_inst->get_level(); }

}  // namespace icts