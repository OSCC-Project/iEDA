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
 * @file OptimizationState.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Fast STA state capture and improvement checks for CTS optimization.
 */

#include "optimization/state/OptimizationState.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

#include "FastSta.hh"
#include "optimization/model/ClockSizingOptimizationData.hh"

namespace icts::clock_sizing_optimization {

namespace {

auto CheckCapLegality(const FastSTA& fast_sta, FastStaClockId clock_id, const std::vector<ClockSizingCapLimit>& baseline)
    -> ClockSizingCapCheck
{
  ClockSizingCapCheck result;
  const auto graph_profile = fast_sta.queryClockGraphProfile(clock_id);
  if (!graph_profile.has_value()) {
    result.legal = false;
    return result;
  }
  for (FastStaNetId net_id = 0U; net_id < graph_profile->net_count; ++net_id) {
    const auto cap_status = fast_sta.queryCapStatus(clock_id, net_id);
    if (!cap_status.has_value()) {
      result.legal = false;
      ++result.violation_count;
      continue;
    }
    if (cap_status->max_cap_pf <= 0.0) {
      continue;
    }
    const auto baseline_load = net_id < baseline.size() ? baseline.at(net_id).load_cap_pf : 0.0;
    const auto baseline_violated = net_id < baseline.size() && baseline.at(net_id).violated;
    const bool legal = baseline_violated ? cap_status->load_cap_pf <= baseline_load + kClockSizingEpsilon : !cap_status->violated;
    if (!legal) {
      result.legal = false;
      ++result.violation_count;
    }
  }
  return result;
}

auto ResolveSlewRole(const std::optional<FastStaSlewStatus>& slew_status, FastStaNodeId node_id,
                     const std::vector<ClockSizingSlewLimit>& baseline) -> FastStaSlewRole
{
  if (slew_status.has_value()) {
    return slew_status->role;
  }
  if (node_id < baseline.size()) {
    return baseline.at(node_id).role;
  }
  return FastStaSlewRole::kUnknown;
}

auto CountSlewViolationRole(ClockSizingSlewCheck& result, FastStaSlewRole role) -> void
{
  ++result.violation_count;
  switch (role) {
    case FastStaSlewRole::kBufferInput:
      ++result.buffer_violation_count;
      break;
    case FastStaSlewRole::kSink:
      ++result.sink_violation_count;
      break;
    case FastStaSlewRole::kUnknown:
      break;
  }
}

auto CheckSlewLegality(const FastSTA& fast_sta, FastStaClockId clock_id, const std::vector<ClockSizingSlewLimit>& baseline)
    -> ClockSizingSlewCheck
{
  ClockSizingSlewCheck result;
  const auto graph_profile = fast_sta.queryClockGraphProfile(clock_id);
  if (!graph_profile.has_value()) {
    result.legal = false;
    return result;
  }
  for (FastStaNodeId node_id = 0U; node_id < graph_profile->node_count; ++node_id) {
    const auto slew_status = fast_sta.querySlewStatus(clock_id, node_id);
    const auto baseline_available = node_id < baseline.size() && baseline.at(node_id).available;
    auto max_slew_ns = 0.0;
    if (slew_status.has_value()) {
      max_slew_ns = slew_status->max_slew_ns;
    } else if (baseline_available) {
      max_slew_ns = baseline.at(node_id).max_slew_ns;
    }
    if (max_slew_ns <= 0.0) {
      continue;
    }
    const auto role = ResolveSlewRole(slew_status, node_id, baseline);
    if (!slew_status.has_value()) {
      result.legal = false;
      CountSlewViolationRole(result, role);
      continue;
    }
    const auto baseline_slew = baseline_available ? baseline.at(node_id).slew_ns : 0.0;
    const auto baseline_violated = baseline_available && baseline.at(node_id).violated;
    const bool legal = baseline_violated ? slew_status->slew_ns <= baseline_slew + kClockSizingEpsilon : !slew_status->violated;
    if (!legal) {
      result.legal = false;
      CountSlewViolationRole(result, role);
    }
  }
  return result;
}

}  // namespace

auto FirstClockSizingEditBufferIndex(const std::vector<ClockSizingEdit>& edits) -> std::size_t
{
  auto first_index = std::numeric_limits<std::size_t>::max();
  for (const auto& edit : edits) {
    first_index = std::min(first_index, edit.buffer_index);
  }
  return first_index;
}

auto CaptureState(const FastSTA& fast_sta, FastStaClockId clock_id, const std::vector<ClockSizingCapLimit>& cap_baseline,
                  const std::vector<ClockSizingSlewLimit>& slew_baseline) -> ClockSizingTimingState
{
  ClockSizingTimingState state;
  state.skew = fast_sta.querySkew(clock_id);
  state.power = fast_sta.queryPower(clock_id);
  state.cap = CheckCapLegality(fast_sta, clock_id, cap_baseline);
  state.slew = CheckSlewLegality(fast_sta, clock_id, slew_baseline);
  state.valid = state.skew.valid && state.cap.legal && state.slew.legal;
  return state;
}

auto CaptureStateWithArea(const FastSTA& fast_sta, FastStaClockId clock_id, const std::vector<ClockSizingCapLimit>& cap_baseline,
                          const std::vector<ClockSizingSlewLimit>& slew_baseline, double area_um2) -> ClockSizingTimingState
{
  auto state = CaptureState(fast_sta, clock_id, cap_baseline, slew_baseline);
  state.power.area_um2 = area_um2;
  return state;
}

auto TargetMet(const ClockSizingTimingState& state, double target_skew_ns) -> bool
{
  return state.valid && state.skew.skew_ns <= target_skew_ns + kClockSizingEpsilon;
}

auto StateImproves(const ClockSizingTimingState& current, const ClockSizingTimingState& candidate, double target_skew_ns) -> bool
{
  if (!candidate.valid) {
    return false;
  }
  const bool current_met = TargetMet(current, target_skew_ns);
  const bool candidate_met = TargetMet(candidate, target_skew_ns);
  if (current_met) {
    if (!candidate_met) {
      return false;
    }
    if (candidate.power.area_um2 < current.power.area_um2 - kClockSizingEpsilon) {
      return true;
    }
    return std::abs(candidate.power.area_um2 - current.power.area_um2) <= kClockSizingEpsilon
           && candidate.skew.skew_ns < current.skew.skew_ns - kClockSizingEpsilon;
  }
  if (candidate_met) {
    return true;
  }
  return candidate.skew.skew_ns < current.skew.skew_ns - kClockSizingEpsilon;
}

}  // namespace icts::clock_sizing_optimization
