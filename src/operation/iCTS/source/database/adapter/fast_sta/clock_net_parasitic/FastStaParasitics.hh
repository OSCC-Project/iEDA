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
 * @file FastStaParasitics.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief OpenSTA-style RC reduction data path for CTS fast STA.
 */

#pragma once

#include <vector>

#include "FastSta.hh"

namespace icts {

class Net;
struct FastStaClockContext;
template <typename T>
class ClockSteinerTree;

class FastStaParasitics
{
 public:
  FastStaParasitics() = delete;

  static auto updateNetLoads(FastStaClockContext& context) -> void;
  static auto updateNetLoads(FastStaClockContext& context, const std::vector<FastStaNetId>& net_ids) -> void;
  static auto buildNetParasiticFromSegments(FastStaClockContext& context, FastStaNetId net_id,
                                            const std::vector<FastStaRcSegment>& segments) -> bool;
  static auto buildNetParasiticFromRouteTree(FastStaClockContext& context, FastStaNetId net_id, const Net& net,
                                             const ClockSteinerTree<int>& route_tree) -> bool;
  static auto reduceToPiElmore(FastStaClockContext& context, FastStaNetId net_id) -> bool;
};

}  // namespace icts
