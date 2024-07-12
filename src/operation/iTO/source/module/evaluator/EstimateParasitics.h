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
#include <unordered_set>
#include <vector>

#include "Utility.h"
#include "tree_build/TreeBuild.h"

namespace ito {
using ito::dbuToMeters;
using ito::metersToDbu;

#define toEvalInst EstimateParasitics::get_instance()
class EstimateParasitics
{
 public:
  static EstimateParasitics* get_instance();
  static void destroy_instance();

  void excuteParasiticsEstimate();
  void estimateAllNetParasitics();
  void estimateNetParasitics(Net* net);
  void invalidNetRC(Net* net);
  void estimateInvalidNetParasitics(Net* net, DesignObject* driver_pin_port);
  void excuteWireParasitic(Net* curr_net);
  std::unordered_set<ista::Net*> get_parasitics_invalid_net() { return _parasitics_invalid_nets; }

 private:
  static EstimateParasitics* _instance;
  bool _have_estimated_parasitics = false;
  std::unordered_set<ista::Net*> _parasitics_invalid_nets;

  EstimateParasitics();
  ~EstimateParasitics() = default;

  void RctNodeConnectPins(int index1, RctNode* node1, int index2, RctNode* node2, Net* net, TreeBuild* tree);
  void updateParastic(Net* curr_net, int index1, int index2, int length_per_wire, TreeBuild* tree);
};

}  // namespace ito