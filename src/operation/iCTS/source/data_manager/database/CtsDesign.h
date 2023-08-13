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

#include <unordered_map>
#include <utility>
#include <vector>

#include "ClockTopo.h"
#include "CtsClock.h"
#include "CtsInstance.h"
#include "CtsNet.h"
#include "CtsPin.h"
#include "HCTS.h"
#include "Net.hh"
#include "TimingCalculator.h"
namespace icts {

using std::make_pair;
using std::pair;
using std::vector;

class CtsDesign
{
 public:
  CtsDesign() = default;
  CtsDesign(const CtsDesign&) = default;
  ~CtsDesign();

  bool isClockTopNet(const std::string& net_name) const
  {
    for (auto [clock, clock_net_name] : _clock_net_names) {
      if (clock_net_name == net_name) {
        return true;
      }
    }
    return false;
  }

  void resetId() { _id = 0; }
  int nextId() { return _id++; }
  vector<CtsClock*>& get_clocks() { return _clocks; }
  vector<ClockTopo>& get_clock_topos() { return _clock_topos; }
  vector<pair<string, string>>& get_clock_net_names() { return _clock_net_names; }
  vector<CtsInstance*>& get_insts() { return _insts; }

  vector<CtsNet*>& get_nets() { return _nets; }
  std::vector<Net*> get_goca_nets() { return _goca_nets; }
  vector<TimingNode*>& get_timing_nodes() { return _inst_timing_nodes; }

  void addClockNetName(const string& clock_name, const string& clock_net_name);
  void addClockNet(const string& clock_name, CtsNet* net);
  void addClock(CtsClock* clock) { _clocks.push_back(clock); }
  void addClockTopo(const ClockTopo& clock_topo) { _clock_topos.push_back(clock_topo); }

  void addPin(CtsPin* pin)
  {
    if (_pin_map.count(pin->get_pin_name()) == 0) {
      _pins.push_back(pin);
      _pin_map.insert(std::make_pair(pin->get_full_name(), pin));
    }
  }
  void addNet(CtsNet* net)
  {
    if (_net_map.count(net->get_net_name()) == 0) {
      _nets.push_back(net);
      _net_map.insert(std::make_pair(net->get_net_name(), net));
    }
  }
  void addGocaNet(Net* net)
  {
    if (_goca_net_map.count(net->get_name()) == 0) {
      _goca_nets.push_back(net);
      _goca_net_map.insert(std::make_pair(net->get_name(), net));
    }
  }
  void addInstance(CtsInstance* inst)
  {
    if (_inst_map.count(inst->get_name()) == 0) {
      _insts.push_back(inst);
      _inst_map.insert(std::make_pair(inst->get_name(), inst));
    }
  }
  void addTimingNode(TimingNode* node)
  {
    if (_inst_timing_node_map.count(node->get_name()) == 0) {
      _inst_timing_nodes.push_back(node);
      _inst_timing_node_map.insert(std::make_pair(node->get_name(), node));
    }
  }

  void addHCtsNode(HNode* node)
  {
    if (_hcts_node_map.count(node->getName()) == 0) {
      _hcts_nodes.push_back(node);
      _hcts_node_map.insert(std::make_pair(node->getName(), node));
    }
  }

  // wait to realize find operator | vector -> set
  ClockTopo& findClockTopo(const string& topo_name)
  {
    for (auto& topo : _clock_topos) {
      if (topo.get_name() == topo_name) {
        return topo;
      }
    }
    LOG_FATAL << "can not find clock topo: " << topo_name;
  }

  CtsNet* findNet(const string& net_name) const;
  Net* findGocaNet(const string& net_name) const;
  CtsPin* findPin(const string& pin_full_name) const;
  CtsInstance* findInstance(const string& instance_name) const;
  TimingNode* findTimingNode(const string& node_name) const;
  HNode* findHCtsNode(const string& node_name) const;

 private:
  int _id = 0;
  vector<pair<string, string>> _clock_net_names;

  vector<CtsClock*> _clocks;

  vector<ClockTopo> _clock_topos;

  vector<CtsNet*> _nets;
  vector<CtsInstance*> _insts;
  vector<CtsPin*> _pins;
  vector<TimingNode*> _inst_timing_nodes;
  vector<HNode*> _hcts_nodes;

  std::unordered_map<std::string, CtsNet*> _net_map;
  std::unordered_map<std::string, CtsInstance*> _inst_map;
  std::unordered_map<std::string, CtsPin*> _pin_map;
  std::unordered_map<std::string, TimingNode*> _inst_timing_node_map;
  std::unordered_map<std::string, HNode*> _hcts_node_map;

  // goca temp part
  vector<Net*> _goca_nets;
  std::unordered_map<std::string, Net*> _goca_net_map;
};
}  // namespace icts