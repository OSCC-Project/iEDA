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
#pragma once

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "CtsConfig.hh"
#include "CtsDesign.hh"
#include "CtsInstance.hh"
#include "CtsNet.hh"
#include "CtsSignalWire.hh"
#include "Net.hh"
#include "Pin.hh"
namespace icts {
class SolverSet
{
 public:
  SolverSet() = default;
  SolverSet(const SolverSet&) = default;
  ~SolverSet() = default;
  // add
  void add_pin(Pin* pin)
  {
    if (_pin_map.count(pin->get_name()) == 0) {
      _pins.push_back(pin);
      _pin_map.insert(std::make_pair(pin->get_name(), pin));
    }
  }
  void add_net(Net* net)
  {
    if (_net_map.count(net->get_name()) == 0) {
      _nets.push_back(net);
      _net_map.insert(std::make_pair(net->get_name(), net));
    }
  }
  // get
  std::vector<Net*> get_nets() { return _nets; }
  Net* get_last_net() const { return _nets.back(); }

  // find
  Pin* find_pin(const std::string& pin_name)
  {
    if (_pin_map.count(pin_name) == 0) {
      return nullptr;
    }
    return _pin_map[pin_name];
  }
  Net* find_net(const std::string& net_name)
  {
    if (_net_map.count(net_name) == 0) {
      return nullptr;
    }
    return _net_map[net_name];
  }

 private:
  std::vector<Pin*> _pins;
  std::vector<Net*> _nets;
  std::unordered_map<std::string, Pin*> _pin_map;
  std::unordered_map<std::string, Net*> _net_map;
};
class Router
{
 public:
  Router() = default;
  Router(const Router&) = default;
  ~Router() = default;
  void init();
  void build();
  void update();

 private:
  void printLog();
  void routing(CtsNet* clk_net);
  std::vector<CtsPin*> getSinkPins(CtsNet* clk_net);
  std::vector<CtsPin*> getBufferPins(CtsNet* clk_net);

  void synthesisPin(Pin* pin);
  void synthesisNet(Net* net);

  SolverSet _solver_set;
  std::vector<CtsClock*> _clocks;
  std::unordered_map<std::string, CtsInstance*> _name_to_inst;
};

}  // namespace icts