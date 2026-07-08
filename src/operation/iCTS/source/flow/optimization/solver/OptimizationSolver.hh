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
 * @file OptimizationSolver.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Solver contracts for CTS post-synthesis optimization.
 */

#pragma once

#include <vector>

#include "FastSta.hh"
#include "optimization/model/ClockSizingOptimizationData.hh"

namespace icts::clock_sizing_optimization {

struct ScalableSolverDecisionInput
{
  const FastSTA* fast_sta = nullptr;
  FastStaClockId clock_id = kInvalidFastStaClockId;
  const std::vector<ClockSizingBuffer>* buffers = nullptr;
};

auto SolveClock(FastSTA& fast_sta, FastStaClockId clock_id, std::vector<ClockSizingBuffer>& buffers,
                const std::vector<ClockSizingCapLimit>& cap_baseline, const std::vector<ClockSizingSlewLimit>& slew_baseline,
                double target_skew_ns) -> ClockSizingSummary;
auto SolveClockScalable(FastSTA& fast_sta, FastStaClockId clock_id, std::vector<ClockSizingBuffer>& buffers,
                        const std::vector<ClockSizingCapLimit>& cap_baseline, const std::vector<ClockSizingSlewLimit>& slew_baseline,
                        double target_skew_ns) -> ClockSizingSummary;
auto ShouldUseScalableSolver(const ScalableSolverDecisionInput& input) -> bool;

}  // namespace icts::clock_sizing_optimization
