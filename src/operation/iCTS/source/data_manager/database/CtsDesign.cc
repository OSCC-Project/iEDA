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
 * @file CtsDesign.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "CtsDesign.hh"

namespace icts {
CtsDesign::~CtsDesign()
{
  for (auto* net : _nets) {
    delete net;
  }
  for (auto* inst : _insts) {
    delete inst;
  }
  for (auto* pin : _pins) {
    delete pin;
  }
}

void CtsDesign::addClockNetName(const std::string& clock_name, const std::string& net_name)
{
  for (auto& clock_net_name : _clock_net_names) {
    if (clock_net_name.first == clock_name && clock_net_name.second == net_name) {
      return;
    }
  }
  _clock_net_names.push_back(std::make_pair(clock_name, net_name));
}

void CtsDesign::addClockNet(const std::string& clock_name, CtsNet* net)
{
  bool found = false;
  for (auto* clock : _clocks) {
    if (clock->get_clock_name() == clock_name) {
      clock->addClockNet(net);
      found = true;
      break;
    }
  }
  if (!found) {
    CtsClock* clock = new CtsClock(clock_name);
    clock->addClockNet(net);
    _clocks.push_back(clock);
  }
}

CtsNet* CtsDesign::findNet(const std::string& net_name) const
{
  for (auto* net : _nets) {
    if (net_name == net->get_net_name()) {
      return net;
    }
  }
  return nullptr;
}

CtsPin* CtsDesign::CtsDesign::findPin(const std::string& pin_full_name) const
{
  if (_pin_map.count(pin_full_name)) {
    return _pin_map.at(pin_full_name);
  }
  return nullptr;
}
CtsInstance* CtsDesign::findInstance(const std::string& instance_name) const
{
  if (_inst_map.count(instance_name)) {
    return _inst_map.at(instance_name);
  }
  return nullptr;
}

Net* CtsDesign::findSolverNet(const std::string& net_name) const
{
  if (_solver_map.count(net_name)) {
    return _solver_map.at(net_name);
  }
  return nullptr;
}

}  // namespace icts