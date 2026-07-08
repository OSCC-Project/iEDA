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
 * @file FastStaLibertyModel.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-18
 * @brief CTS fast STA Liberty model lookup helpers.
 */

#include "FastStaLibertyModel.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <utility>

namespace icts {
namespace {

auto findAxisInterval(const std::vector<double>& axis, double value) -> std::pair<std::size_t, std::size_t>
{
  if (axis.empty()) {
    return {0U, 0U};
  }
  if (axis.size() == 1U) {
    return {0U, 0U};
  }
  if (value <= axis.front()) {
    return {0U, 1U};
  }
  if (value >= axis.back()) {
    return {axis.size() - 2U, axis.size() - 1U};
  }

  const auto upper = std::ranges::upper_bound(axis, value);
  const auto high_index = static_cast<std::size_t>(std::distance(axis.begin(), upper));
  return {high_index - 1U, high_index};
}

auto axisQueryValue(const FastStaLibertyAxis& axis, double input_slew_ns, double output_load_pf) -> double
{
  switch (axis.kind) {
    case FastStaLibertyAxisKind::kInputSlew:
      return input_slew_ns;
    case FastStaLibertyAxisKind::kOutputLoad:
      return output_load_pf;
    case FastStaLibertyAxisKind::kUnknown:
      return axis.values.empty() ? 0.0 : axis.values.front();
  }
  return 0.0;
}

auto linearInterpolate(double x1, double x2, double y1, double y2, double x) -> double
{
  if (std::abs(x2 - x1) <= 1e-18) {
    return y1;
  }
  return y1 + (y2 - y1) * (x - x1) / (x2 - x1);
}

}  // namespace

auto FastStaLibertyTable::lookup(double input_slew_ns, double output_load_pf) const -> std::optional<double>
{
  if (values.empty()) {
    return std::nullopt;
  }
  if (axes.empty()) {
    return values.front();
  }

  if (axes.size() == 1U) {
    const auto& axis = axes.front();
    if (axis.values.empty()) {
      return values.front();
    }
    const auto [low, high] = findAxisInterval(axis.values, axisQueryValue(axis, input_slew_ns, output_load_pf));
    if (low >= values.size() || high >= values.size()) {
      return std::nullopt;
    }
    return linearInterpolate(axis.values.at(low), axis.values.at(high), values.at(low), values.at(high),
                             axisQueryValue(axis, input_slew_ns, output_load_pf));
  }

  const auto& x_axis = axes.at(0U);
  const auto& y_axis = axes.at(1U);
  if (x_axis.values.empty() || y_axis.values.empty()) {
    return values.front();
  }

  const auto [x_low, x_high] = findAxisInterval(x_axis.values, axisQueryValue(x_axis, input_slew_ns, output_load_pf));
  const auto [y_low, y_high] = findAxisInterval(y_axis.values, axisQueryValue(y_axis, input_slew_ns, output_load_pf));
  const auto y_size = y_axis.values.size();
  const auto q11_index = x_low * y_size + y_low;
  const auto q21_index = x_high * y_size + y_low;
  const auto q12_index = x_low * y_size + y_high;
  const auto q22_index = x_high * y_size + y_high;
  if (q11_index >= values.size() || q21_index >= values.size() || q12_index >= values.size() || q22_index >= values.size()) {
    return std::nullopt;
  }

  const auto x = axisQueryValue(x_axis, input_slew_ns, output_load_pf);
  const auto y = axisQueryValue(y_axis, input_slew_ns, output_load_pf);
  const auto r1 = linearInterpolate(x_axis.values.at(x_low), x_axis.values.at(x_high), values.at(q11_index), values.at(q21_index), x);
  const auto r2 = linearInterpolate(x_axis.values.at(x_low), x_axis.values.at(x_high), values.at(q12_index), values.at(q22_index), x);
  return linearInterpolate(y_axis.values.at(y_low), y_axis.values.at(y_high), r1, r2, y);
}

}  // namespace icts
