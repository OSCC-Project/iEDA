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
 * @date 2026-03-08
 * @brief Unified routing dispatch facade
 */

#pragma once

#include <vector>

#include "module/routing/local_legalization/LocalLegalization.hh"
#include "routing/RoutingTerminal.hh"
#include "routing/SteinerTree.hh"
#include "timing/RCTree.hh"

namespace icts {

struct BSTRoutingConfig;
struct ClockRouteSegmentRc;
class Net;
class Pin;

class Router
{
 public:
  using ClockTerminal = ClockRoutingTerminal;
  using ClockSteinerTreeType = ClockSteinerTree<>;
  using RCTreeType = RCTree;
  using LegalizationRegion = LocalLegalization::RegionType;
  using LegalizationConfig = LocalLegalization::Config;
  using LegalizationOutput = LocalLegalization::Output;

  Router() = delete;
  ~Router() = default;

  static auto buildFluteTree(const ClockTerminal& driver_terminal, const std::vector<ClockTerminal>& load_terminals)
      -> ClockSteinerTreeType;
  static auto buildSaltTree(const ClockTerminal& driver_terminal, const std::vector<ClockTerminal>& load_terminals) -> ClockSteinerTreeType;
  static auto buildBstTree(const std::vector<ClockTerminal>& load_terminals, const BSTRoutingConfig& parameters) -> ClockSteinerTreeType;
  static auto buildCbsTree(const std::vector<ClockTerminal>& load_terminals, const BSTRoutingConfig& parameters) -> ClockSteinerTreeType;
  static auto buildClockNetTree(const Net& net) -> ClockSteinerTreeType;

  static auto legalizePins(std::vector<Pin*>& movable_pins, const std::vector<Pin*>& fixed_pins, const LegalizationRegion& feasible_region,
                           const LegalizationRegion& block_region) -> LegalizationOutput;
  static auto legalizePins(std::vector<Pin*>& movable_pins, const std::vector<Pin*>& fixed_pins, const LegalizationRegion& feasible_region,
                           const LegalizationRegion& block_region, const LegalizationConfig& config) -> LegalizationOutput;

  static auto buildRCTree(const ClockSteinerTreeType& clock_tree, const ClockRouteSegmentRc& route_segment_rc) -> RCTreeType;
};

}  // namespace icts
