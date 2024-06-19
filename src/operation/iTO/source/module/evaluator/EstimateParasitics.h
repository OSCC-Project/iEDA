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

#include <set>
#include <string>
#include <vector>

#include <unordered_set>

#include "DbInterface.h"
#include "RoutingTree.h"
#include "Utility.h"

namespace ito {
using ito::dbuToMeters;
using ito::metersToDbu;

class EstimateParasitics {
 public:
  EstimateParasitics(DbInterface *dbintreface);

  EstimateParasitics(TimingEngine *timing_engine, int dbu);
  EstimateParasitics() = default;
  ~EstimateParasitics() = default;

  void excuteParasiticsEstimate();

  void estimateAllNetParasitics();

  void estimateNetParasitics(Net *net);

  void invalidNetRC(Net *net);

  void estimateInvalidNetParasitics(DesignObject *drvr_pin_port, Net *net);

  void excuteWireParasitic(DesignObject *drvr_pin_port, Net *curr_net,
                           TimingDBAdapter *db_adapter);

  std::unordered_set<ista::Net *> get_parasitics_invalid_net() {
    return _parasitics_invalid_nets;
  }

 private:
  void RctNodeConnectPin(Net *net, int index, RctNode *rcnode, RoutingTree *tree);

  DbInterface     *_db_interface = nullptr;
  TimingEngine    *_timing_engine = nullptr;
  TimingDBAdapter *_db_adapter = nullptr;
  int              _dbu;

  std::unordered_set<ista::Net *> _parasitics_invalid_nets;

  bool _have_estimated_parasitics = false;
};

} // namespace ito