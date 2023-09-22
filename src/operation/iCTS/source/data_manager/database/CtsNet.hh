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
 * @file CtsNet.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once

#include <string>
#include <vector>

#include "CtsInstance.hh"
#include "CtsPin.hh"
#include "CtsSignalWire.hh"
#include "DesignObject.hh"

namespace icts {

class CtsPin;
class CtsInstance;
class CtsNet : public DesignObject
{
 public:
  CtsNet() = default;
  explicit CtsNet(const std::string& net_name) : _net_name(net_name) {}
  CtsNet(const CtsNet& net) = default;
  ~CtsNet() = default;

  // getter
  const std::string& get_net_name() const { return _net_name; }
  CtsPin* get_driver_pin() const;
  std::vector<CtsPin*> get_load_pins() const;
  CtsInstance* get_driver_inst(CtsPin* pin = nullptr) const;
  std::vector<CtsInstance*> get_load_insts() const;
  std::vector<CtsPin*>& get_pins() { return _pins; }
  std::vector<CtsInstance*> get_instances() const;
  std::vector<CtsSignalWire>& get_signal_wires();
  // setter
  void set_net_name(const std::string& net_name) { _net_name = net_name; }
  template <typename WireIterator>
  void set_signal_wire(WireIterator begin, WireIterator end)
  {
    _signal_wires.clear();
    for (auto itr = begin; itr != end; itr++) {
      add_signal_wire(*itr);
    }
  }

  // operator

  void clear()
  {
    _pins.clear();
    clearWires();
  }
  bool isClockRouted() const { return _is_clock_routed; }
  void setClockRouted(const bool& is_clock_routed = true) { _is_clock_routed = is_clock_routed; }
  CtsPin* findPin(const std::string& pin_name);
  void addPin(CtsPin* pin);
  void removePin(CtsPin* pin);
  void add_signal_wire(const CtsSignalWire& signal_wire) { _signal_wires.push_back(signal_wire); }
  void clearWires() { _signal_wires.clear(); }

  CtsInstance* findInstance(const std::string& inst_name);

 private:
  std::string _net_name;
  std::vector<CtsPin*> _pins;
  std::vector<CtsSignalWire> _signal_wires;
  bool _is_clock_routed = false;
};

}  // namespace icts