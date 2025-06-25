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
 * @file CtsDesign.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once

#include <unordered_map>
#include <utility>
#include <vector>

#include "CtsClock.hh"
#include "CtsInstance.hh"
#include "CtsNet.hh"
#include "CtsPin.hh"
#include "Net.hh"
namespace icts {
class CtsDesign
{
 public:
  CtsDesign() = default;
  CtsDesign(const CtsDesign&) = default;
  ~CtsDesign();

  void resetId() { _id = 0; }
  int nextId() { return _id++; }
  bool isClockTopNet(const std::string& net_name) const
  {
    for (auto [clock, clock_net_name] : _clock_net_names) {
      if (clock_net_name == net_name) {
        return true;
      }
    }
    return false;
  }
  std::vector<CtsClock*>& get_clocks() { return _clocks; }
  std::vector<std::pair<std::string, std::string>>& get_clock_net_names() { return _clock_net_names; }
  std::vector<CtsInstance*>& get_insts() { return _insts; }

  std::vector<CtsNet*>& get_nets() { return _nets; }

  void addClockNetName(const std::string& clock_name, const std::string& clock_net_name);
  void addClockNet(const std::string& clock_name, CtsNet* net);
  void addClock(CtsClock* clock) { _clocks.push_back(clock); }

  void addPin(CtsPin* pin)
  {
    auto name = pin->is_io() ? pin->get_pin_name() : pin->get_full_name();
    if (_pin_map.count(name) == 0) {
      _pins.push_back(pin);
      _pin_map.insert(std::make_pair(name, pin));
    }
  }
  void addNet(CtsNet* net)
  {
    if (_net_map.count(net->get_net_name()) == 0) {
      _nets.push_back(net);
      _net_map.insert(std::make_pair(net->get_net_name(), net));
    }
  }
  void addInstance(CtsInstance* inst)
  {
    if (_inst_map.count(inst->get_name()) == 0) {
      _insts.push_back(inst);
      _inst_map.insert(std::make_pair(inst->get_name(), inst));
    }
  }

  void addSolverNet(Net* net)
  {
    if (_solver_map.count(net->get_name()) == 0) {
      _solver_map.insert(std::make_pair(net->get_name(), net));
    }
  }

  CtsNet* findNet(const std::string& net_name) const;
  CtsPin* findPin(const std::string& pin_full_name) const;
  CtsInstance* findInstance(const std::string& instance_name) const;

  Net* findSolverNet(const std::string& net_name) const;

 private:
  int _id = 0;
  std::vector<std::pair<std::string, std::string>> _clock_net_names;

  std::vector<CtsClock*> _clocks;

  std::vector<CtsNet*> _nets;
  std::vector<CtsInstance*> _insts;
  std::vector<CtsPin*> _pins;

  std::unordered_map<std::string, CtsNet*> _net_map;
  std::unordered_map<std::string, CtsInstance*> _inst_map;
  std::unordered_map<std::string, CtsPin*> _pin_map;
  std::unordered_map<std::string, Net*> _solver_map;
};
}  // namespace icts