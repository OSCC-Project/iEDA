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
#include <map>
#include <set>
#include <string>
#include <vector>

#include "ClockTopo.h"
#include "CtsNet.hh"
#include "TimingCalculator.h"

namespace icts {
// balance by master clock latency
struct BalanceTrait {
  double latency;
  double clock_at;
  TimingNode* clock_node;
  TimingNode* root_node;
  CtsNet* clk_net;
};
class Balancer {
 public:
  Balancer() = default;
  ~Balancer() = default;
  void init();
  void balance();

  double calcDumpCapOut(CtsNet* clock_net) const;
  std::map<std::string, std::string> getNetNameToClockNameMap() const;
  std::map<std::set<std::string>, std::vector<CtsNet*>>
  getClkNetsGroupByMasterClocks() const;

 private:
};

}  // namespace icts