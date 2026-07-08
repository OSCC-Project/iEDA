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
 * @file AnalyticalFit.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-14
 * @brief Least-squares fitting helpers for analytical characterization surfaces.
 */

#include "AnalyticalFit.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace icts::analytical {
namespace {

constexpr double kPivotEpsilon = 1e-18;
constexpr double kVarianceEpsilon = 1e-30;

auto MakeFailure(std::string reason) -> AnalyticalFitBuild
{
  return AnalyticalFitBuild{
      .output = {},
      .summary = AnalyticalFitSummary{.success = false, .failure_reason = std::move(reason)},
  };
}

auto BuildDomain(const std::vector<AnalyticalFitSample>& samples) -> AnalyticalDomain
{
  AnalyticalDomain domain;
  if (samples.empty()) {
    return domain;
  }

  domain.slew_min_ns = samples.front().input_slew_ns;
  domain.slew_max_ns = samples.front().input_slew_ns;
  domain.cap_min_pf = samples.front().load_cap_pf;
  domain.cap_max_pf = samples.front().load_cap_pf;
  for (const auto& sample : samples) {
    domain.slew_min_ns = std::min(domain.slew_min_ns, sample.input_slew_ns);
    domain.slew_max_ns = std::max(domain.slew_max_ns, sample.input_slew_ns);
    domain.cap_min_pf = std::min(domain.cap_min_pf, sample.load_cap_pf);
    domain.cap_max_pf = std::max(domain.cap_max_pf, sample.load_cap_pf);
  }
  return domain;
}

auto SolveLinearSystem(std::vector<std::vector<double>> matrix, std::vector<double> rhs) -> std::optional<std::vector<double>>
{
  const std::size_t size = matrix.size();
  if (size == 0U || rhs.size() != size) {
    return std::nullopt;
  }
  for (const auto& row : matrix) {
    if (row.size() != size) {
      return std::nullopt;
    }
  }

  for (std::size_t pivot = 0U; pivot < size; ++pivot) {
    std::size_t pivot_row = pivot;
    double best_abs = std::abs(matrix.at(pivot).at(pivot));
    for (std::size_t row = pivot + 1U; row < size; ++row) {
      const double candidate_abs = std::abs(matrix.at(row).at(pivot));
      if (candidate_abs > best_abs) {
        best_abs = candidate_abs;
        pivot_row = row;
      }
    }
    if (best_abs <= kPivotEpsilon) {
      return std::nullopt;
    }
    if (pivot_row != pivot) {
      std::swap(matrix.at(pivot), matrix.at(pivot_row));
      std::swap(rhs.at(pivot), rhs.at(pivot_row));
    }

    const double pivot_value = matrix.at(pivot).at(pivot);
    for (std::size_t col = pivot; col < size; ++col) {
      matrix.at(pivot).at(col) /= pivot_value;
    }
    rhs.at(pivot) /= pivot_value;

    for (std::size_t row = 0U; row < size; ++row) {
      if (row == pivot) {
        continue;
      }
      const double factor = matrix.at(row).at(pivot);
      if (factor == 0.0) {
        continue;
      }
      for (std::size_t col = pivot; col < size; ++col) {
        matrix.at(row).at(col) -= factor * matrix.at(pivot).at(col);
      }
      rhs.at(row) -= factor * rhs.at(pivot);
    }
  }
  return rhs;
}

auto BuildNormalEquation(const std::vector<AnalyticalFitSample>& samples, AnalyticalModelBasis basis, const AnalyticalDomain& domain)
    -> std::pair<std::vector<std::vector<double>>, std::vector<double>>
{
  const std::size_t term_count = AnalyticalBasisTermCount(basis);
  std::vector<std::vector<double>> normal_matrix(term_count, std::vector<double>(term_count, 0.0));
  std::vector<double> normal_rhs(term_count, 0.0);

  for (const auto& sample : samples) {
    const auto features = BuildAnalyticalFeatures(basis, domain, sample.input_slew_ns, sample.load_cap_pf);
    for (std::size_t row = 0U; row < term_count; ++row) {
      normal_rhs.at(row) += features.at(row) * sample.value;
      for (std::size_t col = 0U; col < term_count; ++col) {
        normal_matrix.at(row).at(col) += features.at(row) * features.at(col);
      }
    }
  }
  return {std::move(normal_matrix), std::move(normal_rhs)};
}

auto CheckMonotonicity(const AnalyticalSurfaceModel& model, const std::vector<AnalyticalFitSample>& samples, double tolerance) -> bool
{
  for (const auto& lhs : samples) {
    const double lhs_value = model.evaluateUnsafe(lhs.input_slew_ns, lhs.load_cap_pf);
    for (const auto& rhs : samples) {
      if (lhs.input_slew_ns <= rhs.input_slew_ns && lhs.load_cap_pf <= rhs.load_cap_pf) {
        const double rhs_value = model.evaluateUnsafe(rhs.input_slew_ns, rhs.load_cap_pf);
        if (lhs_value > rhs_value + tolerance) {
          return false;
        }
      }
    }
  }
  return true;
}

auto EvaluateQuality(AnalyticalSurfaceModel& model, const std::vector<AnalyticalFitSample>& samples, const AnalyticalFitConfig& config)
    -> void
{
  auto& quality = model.quality;
  quality.sample_count = samples.size();

  const double mean_value = std::accumulate(samples.begin(), samples.end(), 0.0,
                                            [](double total, const AnalyticalFitSample& sample) -> double { return total + sample.value; })
                            / static_cast<double>(samples.size());

  double square_error = 0.0;
  double square_total = 0.0;
  for (const auto& sample : samples) {
    const double prediction = model.evaluateUnsafe(sample.input_slew_ns, sample.load_cap_pf);
    const double residual = prediction - sample.value;
    const double abs_residual = std::abs(residual);
    square_error += residual * residual;
    square_total += (sample.value - mean_value) * (sample.value - mean_value);
    quality.max_abs_residual = std::max(quality.max_abs_residual, abs_residual);
    const double relative_denominator = std::max(std::abs(sample.value), config.relative_floor);
    quality.max_relative_residual = std::max(quality.max_relative_residual, abs_residual / relative_denominator);
    if (config.bucket_size > 0.0) {
      quality.max_bucket_residual = std::max(quality.max_bucket_residual, abs_residual / config.bucket_size);
    }
  }
  quality.rmse = std::sqrt(square_error / static_cast<double>(samples.size()));
  quality.r2_valid = square_total > kVarianceEpsilon;
  quality.r2 = quality.r2_valid ? 1.0 - square_error / square_total : 1.0;
  quality.monotonicity_passed = !config.require_monotonic || CheckMonotonicity(model, samples, config.monotonic_tolerance);
}

auto AcceptQuality(const AnalyticalFitQuality& quality, const AnalyticalFitConfig& config) -> std::string
{
  if (config.max_abs_residual > 0.0 && quality.max_abs_residual > config.max_abs_residual) {
    return "max_abs_residual_exceeded";
  }
  if (config.max_relative_residual > 0.0 && quality.max_relative_residual > config.max_relative_residual) {
    return "max_relative_residual_exceeded";
  }
  if (config.max_bucket_residual > 0.0 && quality.max_bucket_residual > config.max_bucket_residual) {
    return "max_bucket_residual_exceeded";
  }
  if (config.min_r2 > 0.0 && quality.r2_valid && quality.r2 < config.min_r2) {
    return "r2_below_threshold";
  }
  if (!quality.monotonicity_passed) {
    return "monotonicity_check_failed";
  }
  return {};
}

}  // namespace

auto FitAnalyticalSurface(const std::vector<AnalyticalFitSample>& samples, const AnalyticalFitConfig& config) -> AnalyticalFitBuild
{
  const std::size_t term_count = AnalyticalBasisTermCount(config.basis);
  if (samples.size() < term_count) {
    return MakeFailure("insufficient_samples");
  }

  const auto domain = BuildDomain(samples);
  if (!domain.isValid()) {
    return MakeFailure("invalid_domain");
  }

  const auto [normal_matrix, normal_rhs] = BuildNormalEquation(samples, config.basis, domain);
  const auto coefficients = SolveLinearSystem(normal_matrix, normal_rhs);
  if (!coefficients.has_value()) {
    return MakeFailure("singular_normal_equation");
  }

  AnalyticalSurfaceModel model{};
  model.metric = config.metric;
  model.basis = config.basis;
  model.domain = domain;
  model.coefficients = *coefficients;
  EvaluateQuality(model, samples, config);

  const auto rejection_reason = AcceptQuality(model.quality, config);
  if (!rejection_reason.empty()) {
    model.quality.accepted = false;
    model.quality.failure_reason = rejection_reason;
    return AnalyticalFitBuild{
        .output = AnalyticalFitOutput{.model = model},
        .summary = AnalyticalFitSummary{.success = false, .failure_reason = rejection_reason},
    };
  }

  model.quality.accepted = true;
  return AnalyticalFitBuild{
      .output = AnalyticalFitOutput{.model = model},
      .summary = AnalyticalFitSummary{.success = true, .failure_reason = {}},
  };
}

}  // namespace icts::analytical
