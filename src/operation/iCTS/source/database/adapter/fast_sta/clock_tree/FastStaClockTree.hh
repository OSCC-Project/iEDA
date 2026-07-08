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
 * @file FastStaClockTree.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief CTS clock-tree graph construction helpers for fast STA.
 */

#pragma once

#include "clock_state/FastStaClockState.hh"

namespace icts {

class Clock;
struct FastStaClockRouteGeometry;

class FastStaClockTree
{
 public:
  FastStaClockTree() = delete;

  static auto buildFromClock(const Clock& clock) -> FastStaClockContext;
  static auto buildFromClockRouteGeometry(const Clock& clock, const FastStaClockRouteGeometry& route_geometry) -> FastStaClockContext;
  static auto applyRouteGeometry(FastStaClockContext& context, const FastStaClockRouteGeometry& route_geometry) -> void;
};

}  // namespace icts
