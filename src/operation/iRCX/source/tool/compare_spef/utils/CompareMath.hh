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
#pragma once

#include <cmath>
#include <optional>

namespace ircx {
namespace compare_spef {
namespace math {

inline constexpr double kEpsilon = 1e-12;

inline auto roundToSignificantDigitsImpl(double value, int digits, double epsilon) -> double
{
  if (std::abs(value) <= epsilon || !std::isfinite(value)) {
    return value;
  }

  const double scale = std::pow(10.0, static_cast<double>(digits - 1) - std::floor(std::log10(std::abs(value))));
  return std::round(value * scale) / scale;
}

inline auto roundToSignificantDigitsHalfEvenImpl(double value, int digits, double epsilon) -> double
{
  if (std::abs(value) <= epsilon || !std::isfinite(value)) {
    return value;
  }

  const double scale = std::pow(10.0, static_cast<double>(digits - 1) - std::floor(std::log10(std::abs(value))));
  const double scaled_value = value * scale;
  const double lower = std::floor(scaled_value);
  const double fraction = scaled_value - lower;
  if (std::abs(fraction - 0.5) <= 1e-9) {
    return (std::fmod(lower, 2.0) == 0.0 ? lower : lower + 1.0) / scale;
  }
  return std::round(scaled_value) / scale;
}

inline auto absoluteRelativeDelta(double test, double reference) -> std::optional<double>
{
  if (std::abs(reference) <= kEpsilon) {
    return std::nullopt;
  }
  return (test - reference) / reference;
}

inline auto roundToSignificantDigits(double value, int digits = 6) -> double
{
  return roundToSignificantDigitsImpl(value, digits, kEpsilon);
}

inline auto roundToSignificantDigitsHalfEven(double value, int digits = 6) -> double
{
  return roundToSignificantDigitsHalfEvenImpl(value, digits, kEpsilon);
}

inline auto capacitanceRelativeDelta(double test, double reference) -> std::optional<double>
{
  return absoluteRelativeDelta(roundToSignificantDigits(test), roundToSignificantDigits(reference));
}

inline auto couplingRelativeDelta(double test, double reference, double denominator) -> std::optional<double>
{
  const double rounded_denominator = roundToSignificantDigits(denominator);
  if (std::abs(rounded_denominator) <= kEpsilon) {
    return std::nullopt;
  }
  return (roundToSignificantDigitsHalfEven(test) - roundToSignificantDigitsHalfEven(reference)) / rounded_denominator;
}

}  // namespace math
}  // namespace compare_spef
}  // namespace ircx
