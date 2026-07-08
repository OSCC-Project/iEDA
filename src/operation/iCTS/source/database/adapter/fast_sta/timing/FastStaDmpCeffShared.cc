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
 * @file FastStaDmpCeffShared.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Shared DMP effective-capacitance numeric helpers for CTS fast STA.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "FastStaDmpCeffSolver.hh"
#include "FastStaLibertyModel.hh"
#include "clock_net_parasitic/FastStaClockNetParasitic.hh"

namespace icts::fast_sta_dmp {
namespace {

auto thresholdInRange(double value, double default_value) -> double
{
  return value > 0.0 && value < 1.0 ? value : default_value;
}

auto selectTable(const std::vector<FastStaLibertyTable>& tables, FastStaTransition transition) -> const FastStaLibertyTable*
{
  for (const auto& table : tables) {
    if (table.transition == transition && !table.empty()) {
      return &table;
    }
  }
  for (const auto& table : tables) {
    if (!table.empty()) {
      return &table;
    }
  }
  return nullptr;
}

auto lookupTable(const std::vector<FastStaLibertyTable>& tables, FastStaTransition transition, double input_slew_ns, double output_load_pf)
    -> double
{
  const auto* table = selectTable(tables, transition);
  if (table == nullptr) {
    return 0.0;
  }
  return table->lookup(input_slew_ns, output_load_pf).value_or(0.0);
}

}  // namespace

auto OutputThreshold(const FastStaLibertyCell& cell, FastStaTransition transition) -> double
{
  return thresholdInRange(transition == FastStaTransition::kRise ? cell.output_threshold_rise : cell.output_threshold_fall, 0.5);
}

auto InputThreshold(const FastStaLibertyCell& cell, FastStaTransition transition) -> double
{
  return thresholdInRange(transition == FastStaTransition::kRise ? cell.input_threshold_rise : cell.input_threshold_fall, 0.5);
}

auto SlewLowerThreshold(const FastStaLibertyCell& cell, FastStaTransition transition) -> double
{
  return thresholdInRange(transition == FastStaTransition::kRise ? cell.slew_lower_threshold_rise : cell.slew_lower_threshold_fall, 0.3);
}

auto SlewUpperThreshold(const FastStaLibertyCell& cell, FastStaTransition transition) -> double
{
  return thresholdInRange(transition == FastStaTransition::kRise ? cell.slew_upper_threshold_rise : cell.slew_upper_threshold_fall, 0.7);
}

auto SlewDerate(const FastStaLibertyCell& cell) -> double
{
  return cell.slew_derate_from_library > 0.0 ? cell.slew_derate_from_library : 1.0;
}

auto GateDelaySlew(const FastStaLibertyCell& cell, FastStaTransition transition, double input_slew_ns, double ceff_pf)
    -> std::pair<double, double>
{
  return {lookupTable(cell.timing_arc.delay_tables, transition, input_slew_ns, ceff_pf),
          lookupTable(cell.timing_arc.slew_tables, transition, input_slew_ns, ceff_pf)};
}

auto GateModelRdNsPerPf(const FastStaLibertyCell& cell, const FastStaPiModel& pi, FastStaTransition transition, double input_slew_ns)
    -> double
{
  const auto cap1_pf = std::max(0.0, pi.near_cap_pf + pi.far_cap_pf);
  const auto cap2_pf = cap1_pf + kCapDeltaPf;
  const auto [delay1_ns, unused_slew1] = GateDelaySlew(cell, transition, input_slew_ns, cap1_pf);
  const auto [delay2_ns, unused_slew2] = GateDelaySlew(cell, transition, input_slew_ns, cap2_pf);
  (void) unused_slew1;
  (void) unused_slew2;
  const auto vth = OutputThreshold(cell, transition);
  if (!std::isfinite(delay1_ns) || !std::isfinite(delay2_ns) || vth <= 0.0) {
    return 0.0;
  }
  return -std::log(vth) * std::abs(delay1_ns - delay2_ns) / kCapDeltaPf;
}

auto DmpExp(double x) -> double
{
  if (x < -12.0) {
    return 0.0;
  }
  auto y = 1.0 + x / 4096.0;
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  y *= y;
  return y;
}

auto FindRoot(const RootFunc& func, double x1, double x2, double x_tolerance, std::size_t max_iter) -> std::optional<double>
{
  if (!std::isfinite(x1) || !std::isfinite(x2) || max_iter == 0U) {
    return std::nullopt;
  }
  if (x1 == x2) {
    return std::nullopt;
  }

  double y1 = 0.0;
  double dy1 = 0.0;
  double y2 = 0.0;
  double dy2 = 0.0;
  func(x1, y1, dy1);
  func(x2, y2, dy2);
  if ((y1 > 0.0 && y2 > 0.0) || (y1 < 0.0 && y2 < 0.0)) {
    return std::nullopt;
  }
  if (y1 == 0.0) {
    return x1;
  }
  if (y2 == 0.0) {
    return x2;
  }
  if (y1 > 0.0) {
    std::swap(x1, x2);
  }

  auto root = (x1 + x2) * 0.5;
  auto dx_prev = std::abs(x2 - x1);
  auto dx = dx_prev;
  double y = 0.0;
  double dy = 0.0;
  func(root, y, dy);
  for (std::size_t iter = 0U; iter < max_iter; ++iter) {
    if (std::abs(dy) <= kEpsilon || (((root - x2) * dy - y) * ((root - x1) * dy - y) > 0.0)
        || (std::abs(2.0 * y) > std::abs(dx_prev * dy))) {
      dx_prev = dx;
      dx = (x2 - x1) * 0.5;
      root = x1 + dx;
    } else {
      dx_prev = dx;
      dx = y / dy;
      root -= dx;
    }
    if (std::abs(dx) <= x_tolerance * std::max(std::abs(root), 1e-12)) {
      return root;
    }

    func(root, y, dy);
    if (y < 0.0) {
      x1 = root;
    } else {
      x2 = root;
    }
  }
  return std::nullopt;
}

}  // namespace icts::fast_sta_dmp
