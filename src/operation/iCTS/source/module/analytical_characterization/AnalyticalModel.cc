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
 * @file AnalyticalModel.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-14
 * @brief Analytical characterization model and structural capacitance catalog implementation.
 */

#include "AnalyticalModel.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace icts::analytical {
namespace {

constexpr double kPositiveScaleFloor = 1e-30;

auto SafeScale(double value) -> double
{
  return value > kPositiveScaleFloor ? value : 1.0;
}

}  // namespace

auto AnalyticalDomain::isValid() const -> bool
{
  return slew_min_ns > 0.0 && slew_max_ns >= slew_min_ns && cap_min_pf > 0.0 && cap_max_pf >= cap_min_pf;
}

auto AnalyticalDomain::contains(double input_slew_ns, double load_cap_pf, double epsilon) const -> bool
{
  return isValid() && input_slew_ns + epsilon >= slew_min_ns && input_slew_ns <= slew_max_ns + epsilon
         && load_cap_pf + epsilon >= cap_min_pf && load_cap_pf <= cap_max_pf + epsilon;
}

auto AnalyticalDomain::slewScale() const -> double
{
  return SafeScale(slew_max_ns);
}

auto AnalyticalDomain::capScale() const -> double
{
  return SafeScale(cap_max_pf);
}

auto AnalyticalSurfaceModel::isValid() const -> bool
{
  return domain.isValid() && coefficients.size() == AnalyticalBasisTermCount(basis) && quality.accepted;
}

auto AnalyticalSurfaceModel::evaluate(double input_slew_ns, double load_cap_pf) const -> std::optional<double>
{
  if (!isValid() || !domain.contains(input_slew_ns, load_cap_pf)) {
    return std::nullopt;
  }
  return evaluateUnsafe(input_slew_ns, load_cap_pf);
}

auto AnalyticalSurfaceModel::evaluateUnsafe(double input_slew_ns, double load_cap_pf) const -> double
{
  const auto features = BuildAnalyticalFeatures(basis, domain, input_slew_ns, load_cap_pf);
  double value = 0.0;
  for (std::size_t index = 0U; index < coefficients.size() && index < features.size(); ++index) {
    value += coefficients.at(index) * features.at(index);
  }
  return value;
}

auto AnalyticalSurfaceModel::evaluateConservativeUpper(double input_slew_ns, double load_cap_pf) const -> std::optional<double>
{
  const auto value = evaluate(input_slew_ns, load_cap_pf);
  if (!value.has_value()) {
    return std::nullopt;
  }
  return *value + quality.max_abs_residual;
}

auto StructuralCapOperator::isValid() const -> bool
{
  return std::isfinite(alpha) && std::isfinite(eta_pf) && alpha >= 0.0 && eta_pf >= 0.0;
}

auto StructuralCapOperator::apply(double downstream_cap_pf) const -> double
{
  return alpha * downstream_cap_pf + eta_pf;
}

auto StructuralCapOperator::identity() -> StructuralCapOperator
{
  StructuralCapOperator op{};
  op.source = "identity";
  return op;
}

auto StructuralCapOperator::wire(double wire_cap_pf) -> StructuralCapOperator
{
  StructuralCapOperator op{};
  op.eta_pf = std::max(0.0, wire_cap_pf);
  op.source = "wire";
  return op;
}

auto StructuralCapOperator::buffered(double first_buffer_input_cap_pf, double pre_buffer_wire_cap_pf) -> StructuralCapOperator
{
  StructuralCapOperator op{};
  op.alpha = 0.0;
  op.eta_pf = std::max(0.0, first_buffer_input_cap_pf + pre_buffer_wire_cap_pf);
  op.source = "buffered";
  return op;
}

auto StructuralCapOperator::fanout(double fanout, double junction_cap_pf) -> StructuralCapOperator
{
  StructuralCapOperator op{};
  op.alpha = std::max(0.0, fanout);
  op.eta_pf = std::max(0.0, junction_cap_pf);
  op.source = "fanout";
  return op;
}

auto StructuralCapOperator::compose(const StructuralCapOperator& upstream, const StructuralCapOperator& downstream) -> StructuralCapOperator
{
  StructuralCapOperator op{};
  op.alpha = upstream.alpha * downstream.alpha;
  op.eta_pf = upstream.alpha * downstream.eta_pf + upstream.eta_pf;
  op.physical = upstream.physical && downstream.physical;
  op.bucket_compatible = upstream.bucket_compatible && downstream.bucket_compatible;
  op.source = upstream.source.empty() ? downstream.source : upstream.source + "_of_" + downstream.source;
  return op;
}

auto AnalyticalModelKeyHash::operator()(const AnalyticalModelKey& key) const noexcept -> std::size_t
{
  std::size_t seed = std::hash<unsigned>{}(key.pattern_id.pack());
  seed ^= std::hash<unsigned>{}(key.length_idx) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
  return seed;
}

auto AnalyticalModelSet::isComplete() const -> bool
{
  return output_slew_model.has_value() && delay_model.has_value() && power_model.has_value() && source_boundary_power_model.has_value()
         && source_cap_operator.has_value() && source_cap_operator->isValid() && source_cap_operator->bucket_compatible;
}

auto AnalyticalModelSet::findMetric(AnalyticalMetric metric) const -> const AnalyticalSurfaceModel*
{
  switch (metric) {
    case AnalyticalMetric::kOutputSlew:
      return output_slew_model.has_value() ? &*output_slew_model : nullptr;
    case AnalyticalMetric::kDelay:
      return delay_model.has_value() ? &*delay_model : nullptr;
    case AnalyticalMetric::kPower:
      return power_model.has_value() ? &*power_model : nullptr;
    case AnalyticalMetric::kSourceBoundaryNetSwitchPower:
      return source_boundary_power_model.has_value() ? &*source_boundary_power_model : nullptr;
  }
  return nullptr;
}

auto AnalyticalModelCatalog::addModelSet(AnalyticalModelSet model_set) -> void
{
  _models[model_set.key] = std::move(model_set);
}

auto AnalyticalModelCatalog::find(const AnalyticalModelKey& key) const -> const AnalyticalModelSet*
{
  const auto it = _models.find(key);
  return it == _models.end() ? nullptr : &it->second;
}

auto AnalyticalModelCatalog::find(PatternId pattern_id, unsigned length_idx) const -> const AnalyticalModelSet*
{
  return find(AnalyticalModelKey{.pattern_id = pattern_id, .length_idx = length_idx});
}

auto AnalyticalBasisTermCount(AnalyticalModelBasis basis) -> std::size_t
{
  switch (basis) {
    case AnalyticalModelBasis::kAffine:
      return 3U;
    case AnalyticalModelBasis::kQuadratic:
      return 6U;
  }
  return 0U;
}

auto BuildAnalyticalFeatures(AnalyticalModelBasis basis, const AnalyticalDomain& domain, double input_slew_ns, double load_cap_pf)
    -> std::vector<double>
{
  const double s_norm = input_slew_ns / domain.slewScale();
  const double c_norm = load_cap_pf / domain.capScale();
  if (basis == AnalyticalModelBasis::kAffine) {
    return {1.0, s_norm, c_norm};
  }
  return {1.0, s_norm, c_norm, s_norm * c_norm, s_norm * s_norm, c_norm * c_norm};
}

auto ToString(AnalyticalMetric metric) -> std::string
{
  switch (metric) {
    case AnalyticalMetric::kOutputSlew:
      return "output_slew";
    case AnalyticalMetric::kDelay:
      return "delay";
    case AnalyticalMetric::kPower:
      return "power";
    case AnalyticalMetric::kSourceBoundaryNetSwitchPower:
      return "source_boundary_net_switch_power";
  }
  return "unknown";
}

auto ToString(AnalyticalModelBasis basis) -> std::string
{
  switch (basis) {
    case AnalyticalModelBasis::kAffine:
      return "affine";
    case AnalyticalModelBasis::kQuadratic:
      return "quadratic";
  }
  return "unknown";
}

}  // namespace icts::analytical
