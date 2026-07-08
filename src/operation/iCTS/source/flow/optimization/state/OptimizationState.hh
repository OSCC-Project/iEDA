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
 * @file OptimizationState.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Fast STA state capture contracts for CTS optimization.
 */

#pragma once

#include <cstddef>
#include <vector>

#include "FastSta.hh"
#include "optimization/model/ClockSizingOptimizationData.hh"

namespace icts::clock_sizing_optimization {

auto FirstClockSizingEditBufferIndex(const std::vector<ClockSizingEdit>& edits) -> std::size_t;
auto CaptureState(const FastSTA& fast_sta, FastStaClockId clock_id, const std::vector<ClockSizingCapLimit>& cap_baseline,
                  const std::vector<ClockSizingSlewLimit>& slew_baseline) -> ClockSizingTimingState;
auto CaptureStateWithArea(const FastSTA& fast_sta, FastStaClockId clock_id, const std::vector<ClockSizingCapLimit>& cap_baseline,
                          const std::vector<ClockSizingSlewLimit>& slew_baseline, double area_um2) -> ClockSizingTimingState;
auto TargetMet(const ClockSizingTimingState& state, double target_skew_ns) -> bool;
auto StateImproves(const ClockSizingTimingState& current, const ClockSizingTimingState& candidate, double target_skew_ns) -> bool;

}  // namespace icts::clock_sizing_optimization
