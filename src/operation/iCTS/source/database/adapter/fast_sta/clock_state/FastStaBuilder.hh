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
 * @file FastStaBuilder.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief Initialization bridge from committed CTS state to fast STA context.
 */

#pragma once

#include "clock_state/FastStaClockState.hh"

namespace icts {

class Net;
struct FastStaClockBuildInput;
struct FastStaEnvironment;
template <typename T>
class ClockSteinerTree;

class FastStaBuilder
{
 public:
  FastStaBuilder() = delete;

  static auto buildClockContext(const FastStaEnvironment& environment, const FastStaClockBuildInput& input) -> FastStaClockContext;
  static auto injectNetRouteTree(FastStaClockContext& context, const Net& net, const ClockSteinerTree<int>& route_tree) -> bool;
};

}  // namespace icts
