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
 * @file AnalyticalModel.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-14
 * @brief Analytical characterization model and structural capacitance catalog types.
 */

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "PatternId.hh"

namespace icts::analytical {

enum class AnalyticalMetric
{
  kOutputSlew,
  kDelay,
  kPower,
  kSourceBoundaryNetSwitchPower,
};

enum class AnalyticalModelBasis
{
  kAffine,
  kQuadratic,
};

struct AnalyticalDomain
{
  double slew_min_ns = 0.0;
  double slew_max_ns = 0.0;
  double cap_min_pf = 0.0;
  double cap_max_pf = 0.0;

  auto isValid() const -> bool;
  auto contains(double input_slew_ns, double load_cap_pf, double epsilon = 1e-12) const -> bool;
  auto slewScale() const -> double;
  auto capScale() const -> double;
};

struct AnalyticalFitQuality
{
  std::size_t sample_count = 0U;
  double rmse = 0.0;
  double max_abs_residual = 0.0;
  double max_relative_residual = 0.0;
  double max_bucket_residual = 0.0;
  double r2 = 0.0;
  bool r2_valid = false;
  bool monotonicity_passed = true;
  bool bucket_aware_passed = true;
  bool accepted = false;
  std::string failure_reason;
};

struct AnalyticalSurfaceModel
{
  AnalyticalMetric metric = AnalyticalMetric::kDelay;
  AnalyticalModelBasis basis = AnalyticalModelBasis::kAffine;
  AnalyticalDomain domain;
  std::vector<double> coefficients;
  AnalyticalFitQuality quality;

  auto isValid() const -> bool;
  auto evaluate(double input_slew_ns, double load_cap_pf) const -> std::optional<double>;
  auto evaluateUnsafe(double input_slew_ns, double load_cap_pf) const -> double;
  auto evaluateConservativeUpper(double input_slew_ns, double load_cap_pf) const -> std::optional<double>;
};

struct StructuralCapOperator
{
  double alpha = 1.0;
  double eta_pf = 0.0;
  bool physical = true;
  bool bucket_compatible = true;
  std::string source;

  auto isValid() const -> bool;
  auto apply(double downstream_cap_pf) const -> double;

  static auto identity() -> StructuralCapOperator;
  static auto wire(double wire_cap_pf) -> StructuralCapOperator;
  static auto buffered(double first_buffer_input_cap_pf, double pre_buffer_wire_cap_pf) -> StructuralCapOperator;
  static auto fanout(double fanout, double junction_cap_pf) -> StructuralCapOperator;
  static auto compose(const StructuralCapOperator& upstream, const StructuralCapOperator& downstream) -> StructuralCapOperator;
};

struct AnalyticalModelKey
{
  PatternId pattern_id = PatternId::segment(0U);
  unsigned length_idx = 0U;

  auto operator==(const AnalyticalModelKey& rhs) const -> bool = default;
};

struct AnalyticalModelKeyHash
{
  auto operator()(const AnalyticalModelKey& key) const noexcept -> std::size_t;
};

struct AnalyticalModelSet
{
  AnalyticalModelKey key;
  std::optional<AnalyticalSurfaceModel> output_slew_model = std::nullopt;
  std::optional<AnalyticalSurfaceModel> delay_model = std::nullopt;
  std::optional<AnalyticalSurfaceModel> power_model = std::nullopt;
  std::optional<AnalyticalSurfaceModel> source_boundary_power_model = std::nullopt;
  std::optional<StructuralCapOperator> source_cap_operator = std::nullopt;

  auto isComplete() const -> bool;
  auto findMetric(AnalyticalMetric metric) const -> const AnalyticalSurfaceModel*;
};

class AnalyticalModelCatalog
{
 public:
  auto addModelSet(AnalyticalModelSet model_set) -> void;
  auto find(const AnalyticalModelKey& key) const -> const AnalyticalModelSet*;
  auto find(PatternId pattern_id, unsigned length_idx) const -> const AnalyticalModelSet*;
  auto size() const -> std::size_t { return _models.size(); }
  auto empty() const -> bool { return _models.empty(); }
  auto get_model_sets() const -> const std::unordered_map<AnalyticalModelKey, AnalyticalModelSet, AnalyticalModelKeyHash>&
  {
    return _models;
  }

 private:
  std::unordered_map<AnalyticalModelKey, AnalyticalModelSet, AnalyticalModelKeyHash> _models;
};

auto AnalyticalBasisTermCount(AnalyticalModelBasis basis) -> std::size_t;
auto BuildAnalyticalFeatures(AnalyticalModelBasis basis, const AnalyticalDomain& domain, double input_slew_ns, double load_cap_pf)
    -> std::vector<double>;
auto ToString(AnalyticalMetric metric) -> std::string;
auto ToString(AnalyticalModelBasis basis) -> std::string;

}  // namespace icts::analytical
