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
 * @file ValueLattice.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-21
 * @brief Uniform lattice helper shared by characterization and H-tree composition.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

namespace icts {

constexpr double kValueLatticeEpsilon = 1e-9;

inline auto CeilDivUnsigned(unsigned numerator, unsigned denominator) -> unsigned
{
  return denominator == 0U ? 0U : (numerator + denominator - 1U) / denominator;
}

class UniformValueLattice
{
 public:
  UniformValueLattice() = default;

  UniformValueLattice(double step_value, unsigned steps) : _step_value(step_value), _steps(steps) {}

  static auto buildFromMax(double max_value, unsigned steps) -> UniformValueLattice
  {
    if (max_value <= 0.0 || steps == 0U) {
      return UniformValueLattice{};
    }
    return UniformValueLattice(max_value / static_cast<double>(steps), steps);
  }

  auto isValid() const -> bool { return _step_value > 0.0 && _steps > 0U; }
  auto stepValue() const -> double { return _step_value; }
  auto steps() const -> unsigned { return _steps; }
  auto maxValue() const -> double { return _step_value * static_cast<double>(_steps); }

  auto valueForIndex(unsigned index) const -> double { return static_cast<double>(index) * _step_value; }

  auto coveringIndex(double value) const -> unsigned
  {
    if (!isValid() || value <= 0.0) {
      return 0U;
    }
    return static_cast<unsigned>(std::ceil((value / _step_value) - kValueLatticeEpsilon));
  }

  auto tryObservedIndex(double value) const -> std::optional<unsigned>
  {
    if (!isValid() || value <= 0.0) {
      return std::nullopt;
    }
    if (value > maxValue() + kValueLatticeEpsilon) {
      return std::nullopt;
    }
    return std::clamp(coveringIndex(value), 1U, _steps);
  }

  auto sampleValues() const -> std::vector<double>
  {
    std::vector<double> samples;
    if (!isValid()) {
      return samples;
    }

    samples.reserve(_steps);
    for (unsigned step_index = 1U; step_index <= _steps; ++step_index) {
      samples.push_back(valueForIndex(step_index));
    }
    return samples;
  }

 private:
  double _step_value = 0.0;
  unsigned _steps = 0U;
};

}  // namespace icts
