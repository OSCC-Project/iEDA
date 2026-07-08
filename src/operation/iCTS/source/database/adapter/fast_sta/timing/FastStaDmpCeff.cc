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
 * @file FastStaDmpCeff.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief OpenSTA-style DMP effective capacitance and load slew calculation for CTS fast STA.
 */

#include "FastStaDmpCeff.hh"

#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>

#include "FastStaDmpCeffSolver.hh"
#include "FastStaLibertyModel.hh"
#include "timing/FastStaClockTiming.hh"

namespace icts {
namespace {

using fast_sta_dmp::DmpExp;
using fast_sta_dmp::FindRoot;
using fast_sta_dmp::InputThreshold;
using fast_sta_dmp::kEpsilon;
using fast_sta_dmp::kFindRootMaxIter;
using fast_sta_dmp::kThresholdTimeTolerance;
using fast_sta_dmp::SlewDerate;
using fast_sta_dmp::SlewLowerThreshold;
using fast_sta_dmp::SlewUpperThreshold;

auto vl0(const FastStaDmpDriverResult& state, double t_ns, double p3_per_ns) -> std::pair<double, double>
{
  if (state.algorithm == FastStaDmpAlgorithm::kPi) {
    const auto denom1 = state.pole1_per_ns - p3_per_ns;
    const auto denom2 = state.pole2_per_ns - p3_per_ns;
    if (std::abs(denom1) <= kEpsilon || std::abs(denom2) <= kEpsilon || std::abs(p3_per_ns) <= kEpsilon) {
      return {0.0, 0.0};
    }
    const auto d1 = state.k0 * (state.k1 - state.k2 / p3_per_ns);
    const auto d3 = -p3_per_ns * state.k0 * state.k3 / denom1;
    const auto d4 = -p3_per_ns * state.k0 * state.k4 / denom2;
    const auto d5 = state.k0 * (state.k2 / p3_per_ns - state.k1 + p3_per_ns * state.k3 / denom1 + p3_per_ns * state.k4 / denom2);
    const auto exp_p1 = DmpExp(-state.pole1_per_ns * t_ns);
    const auto exp_p2 = DmpExp(-state.pole2_per_ns * t_ns);
    const auto exp_p3 = DmpExp(-p3_per_ns * t_ns);
    return {d1 + t_ns + d3 * exp_p1 + d4 * exp_p2 + d5 * exp_p3,
            1.0 - d3 * state.pole1_per_ns * exp_p1 - d4 * state.pole2_per_ns * exp_p2 - d5 * p3_per_ns * exp_p3};
  }
  if (state.algorithm == FastStaDmpAlgorithm::kZeroC2) {
    const auto denom1 = state.pole1_per_ns - p3_per_ns;
    if (std::abs(denom1) <= kEpsilon || std::abs(p3_per_ns) <= kEpsilon) {
      return {0.0, 0.0};
    }
    const auto d1 = state.k0 * (state.k1 - state.k2 / p3_per_ns);
    const auto d3 = -p3_per_ns * state.k0 * state.k3 / denom1;
    const auto d5 = state.k0 * (state.k2 / p3_per_ns - state.k1 + p3_per_ns * state.k3 / denom1);
    const auto exp_p1 = DmpExp(-state.pole1_per_ns * t_ns);
    const auto exp_p3 = DmpExp(-p3_per_ns * t_ns);
    return {d1 + t_ns + d3 * exp_p1 + d5 * exp_p3, 1.0 - d3 * state.pole1_per_ns * exp_p1 - d5 * p3_per_ns * exp_p3};
  }
  return {0.0, 0.0};
}

auto vl(const FastStaDmpDriverResult& state, double t_ns, double p3_per_ns) -> std::pair<double, double>
{
  const auto t1_ns = t_ns - state.t0_ns;
  if (t1_ns <= 0.0 || state.dt_ns <= kEpsilon) {
    return {0.0, 0.0};
  }
  if (t1_ns <= state.dt_ns) {
    const auto [value, deriv] = vl0(state, t1_ns, p3_per_ns);
    return {value / state.dt_ns, deriv / state.dt_ns};
  }
  const auto [value, deriv] = vl0(state, t1_ns, p3_per_ns);
  const auto [dt_value, dt_deriv] = vl0(state, t1_ns - state.dt_ns, p3_per_ns);
  return {(value - dt_value) / state.dt_ns, (deriv - dt_deriv) / state.dt_ns};
}

auto findVlCrossing(const FastStaDmpDriverResult& state, double p3_per_ns, double threshold, double lower, double upper)
    -> std::optional<double>
{
  if (upper <= lower) {
    upper = lower + std::max(state.dt_ns, 1e-6);
  }
  return FindRoot(
      [&](double t, double& y_value, double& dy_value) -> void {
        const auto [vl_value, dvl_dt] = vl(state, t, p3_per_ns);
        y_value = vl_value - threshold;
        dy_value = dvl_dt;
      },
      lower, upper, kThresholdTimeTolerance, kFindRootMaxIter);
}

auto vlCrossingUpperBound(const FastStaDmpDriverResult& state, double elmore_delay_ns) -> double
{
  if (state.algorithm == FastStaDmpAlgorithm::kPi) {
    return state.t0_ns + state.dt_ns + (state.near_cap_pf + state.far_cap_pf) * (state.rd_ns_per_pf + state.rpi_ns_per_pf) * 2.0
           + elmore_delay_ns * 2.0;
  }
  if (state.algorithm == FastStaDmpAlgorithm::kZeroC2) {
    return state.t0_ns + state.dt_ns + state.far_cap_pf * (state.rd_ns_per_pf + state.rpi_ns_per_pf) * 2.0 + elmore_delay_ns * 2.0;
  }
  return state.t0_ns + state.dt_ns + elmore_delay_ns * 2.0;
}

auto applyThresholdAdjust(const FastStaDmpDriverResult& driver_timing, const FastStaLibertyCell* load_cell, double& wire_delay_ns,
                          double& load_slew_ns) -> void
{
  if (load_cell == nullptr) {
    return;
  }
  const auto load_vth = InputThreshold(*load_cell, driver_timing.transition);
  const auto load_lower = SlewLowerThreshold(*load_cell, driver_timing.transition);
  const auto load_upper = SlewUpperThreshold(*load_cell, driver_timing.transition);
  const auto load_derate = SlewDerate(*load_cell);
  const auto driver_span = driver_timing.slew_upper_threshold - driver_timing.slew_lower_threshold;
  const auto load_span = load_upper - load_lower;
  if (driver_span <= kEpsilon || load_span <= kEpsilon) {
    return;
  }

  const auto delay_delta = load_slew_ns * ((load_vth - driver_timing.output_threshold) / driver_span);
  wire_delay_ns += driver_timing.transition == FastStaTransition::kRise ? delay_delta : -delay_delta;
  load_slew_ns *= (load_span / load_derate) / (driver_span / driver_timing.slew_derate);
  wire_delay_ns = std::max(0.0, wire_delay_ns);
  load_slew_ns = std::max(0.0, load_slew_ns);
}

}  // namespace

auto FastStaDmpCeff::calcDriverTiming(const FastStaLibertyCell& driver_cell, const FastStaPiModel& pi, FastStaTransition transition,
                                      double input_slew_ns) -> FastStaDmpDriverResult
{
  fast_sta_dmp::DmpSolver solver(driver_cell, pi, transition, input_slew_ns);
  return solver.solve();
}

auto FastStaDmpCeff::calcLoadDelaySlew(const FastStaDmpDriverResult& driver_timing, double elmore_delay_ns,
                                       const FastStaLibertyCell* load_cell) -> FastStaDmpLoadResult
{
  const auto elmore_ns = std::max(0.0, elmore_delay_ns);
  auto wire_delay_ns = elmore_ns;
  auto load_slew_ns = std::max(0.0, driver_timing.driver_slew_ns);

  if (driver_timing.valid && driver_timing.driver_waveform_valid && elmore_ns > 0.0
      && elmore_ns >= std::max(0.0, driver_timing.driver_slew_ns) * 1e-3) {
    const auto p3_per_ns = 1.0 / elmore_ns;
    const auto upper = vlCrossingUpperBound(driver_timing, elmore_ns);
    const auto load_delay = findVlCrossing(driver_timing, p3_per_ns, driver_timing.output_threshold, driver_timing.t0_ns, upper);
    if (load_delay.has_value()) {
      const auto tl = findVlCrossing(driver_timing, p3_per_ns, driver_timing.slew_lower_threshold, driver_timing.t0_ns, *load_delay);
      const auto th = findVlCrossing(driver_timing, p3_per_ns, driver_timing.slew_upper_threshold, *load_delay, upper);
      if (tl.has_value() && th.has_value()) {
        wire_delay_ns = *load_delay - driver_timing.driver_waveform_delay_ns;
        load_slew_ns = (*th - *tl) / driver_timing.slew_derate;
        if (wire_delay_ns < 0.0) {
          wire_delay_ns = elmore_ns;
        }
        load_slew_ns = std::max(load_slew_ns, driver_timing.driver_slew_ns);
      }
    }
  }

  applyThresholdAdjust(driver_timing, load_cell, wire_delay_ns, load_slew_ns);
  return FastStaDmpLoadResult{.valid = true, .wire_delay_ns = std::max(0.0, wire_delay_ns), .load_slew_ns = std::max(0.0, load_slew_ns)};
}

auto FastStaDmpCeff::calcInputPortDelaySlew(double input_slew_ns, double elmore_delay_ns, FastStaTransition transition,
                                            const FastStaLibertyCell* load_cell) -> FastStaDmpLoadResult
{
  const auto elmore_ns = std::max(0.0, elmore_delay_ns);
  const auto vth = load_cell != nullptr ? fast_sta_dmp::InputThreshold(*load_cell, transition) : 0.5;
  const auto vl_threshold = load_cell != nullptr ? fast_sta_dmp::SlewLowerThreshold(*load_cell, transition) : 0.2;
  const auto vh_threshold = load_cell != nullptr ? fast_sta_dmp::SlewUpperThreshold(*load_cell, transition) : 0.8;
  const auto derate = load_cell != nullptr ? fast_sta_dmp::SlewDerate(*load_cell) : 1.0;
  auto wire_delay_ns = 0.0;
  auto load_slew_ns = std::max(0.0, input_slew_ns);
  if (elmore_ns > 0.0 && vth > 0.0 && vth < 1.0 && vl_threshold > 0.0 && vh_threshold > vl_threshold && vh_threshold < 1.0) {
    wire_delay_ns = -elmore_ns * std::log(1.0 - vth);
    load_slew_ns += elmore_ns * std::log((1.0 - vl_threshold) / (1.0 - vh_threshold)) / derate;
  }
  return FastStaDmpLoadResult{.valid = true, .wire_delay_ns = std::max(0.0, wire_delay_ns), .load_slew_ns = std::max(0.0, load_slew_ns)};
}

}  // namespace icts
