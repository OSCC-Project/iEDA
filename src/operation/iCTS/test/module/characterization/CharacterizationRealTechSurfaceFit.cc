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
 * @file CharacterizationRealTechSurfaceFit.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Surface fit helpers for exact real-tech characterization regression tests.
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "module/characterization/CharacterizationRealTechExactRegression.hh"

namespace icts_test {

auto MakeExactRegressionCharBuilderContract() -> realtech_fixture::RuntimeCharBuilderContract
{
  auto contract = realtech_fixture::MakeRuntimeCharBuilderContract();
  const double length_step_um = contract.config.wirelength_unit_um.value_or(realtech_fixture::kRealTechCharWirelengthUnitUm);
  if (length_step_um <= 0.0) {
    return contract;
  }

  const unsigned required_iterations = realtech_fixture::MakeLengthIndex(realtech_fixture::kExactRootLevelLengthUm, length_step_um);
  contract.config.wirelength_iterations = std::max(contract.config.wirelength_iterations.value_or(1U), required_iterations);
  return contract;
}

auto MakeIterOneExperimentCharBuilderContract() -> realtech_fixture::RuntimeCharBuilderContract
{
  auto contract = realtech_fixture::MakeRuntimeCharBuilderContract();
  contract.config.wirelength_iterations = std::max(contract.config.wirelength_iterations.value_or(1U), 3U);
  return contract;
}

auto FitBasisName(FitBasisKind basis_kind) -> std::string
{
  return basis_kind == FitBasisKind::kLinear ? "linear" : "quadratic";
}

auto FitBasisSize(FitBasisKind basis_kind) -> std::size_t
{
  return basis_kind == FitBasisKind::kLinear ? 3U : 6U;
}

auto MakeFitBasis(FitBasisKind basis_kind, double s, double c) -> std::array<double, 6>
{
  std::array<double, 6> basis{1.0, s, c, s * c, s * s, c * c};
  if (basis_kind == FitBasisKind::kLinear) {
    basis[3] = 0.0;
    basis[4] = 0.0;
    basis[5] = 0.0;
  }
  return basis;
}

auto MakeFitBasis(FitBasisKind basis_kind, const icts::SegmentChar& entry) -> std::array<double, 6>
{
  const auto s = static_cast<double>(entry.get_input_slew_idx());
  const auto c = static_cast<double>(entry.get_load_cap_idx());
  return MakeFitBasis(basis_kind, s, c);
}

auto SolveSmallLinearSystem(std::array<std::array<double, 6>, 6> matrix, std::array<double, 6> rhs, std::size_t size,
                            std::array<double, 6>& solution) -> bool
{
  for (std::size_t pivot_col = 0U; pivot_col < size; ++pivot_col) {
    std::size_t pivot_row = pivot_col;
    double pivot_abs = std::abs(matrix[pivot_col][pivot_col]);
    for (std::size_t row = pivot_col + 1U; row < size; ++row) {
      const double candidate_abs = std::abs(matrix[row][pivot_col]);
      if (candidate_abs > pivot_abs) {
        pivot_abs = candidate_abs;
        pivot_row = row;
      }
    }

    if (pivot_abs <= 1e-18) {
      return false;
    }

    if (pivot_row != pivot_col) {
      std::swap(matrix[pivot_row], matrix[pivot_col]);
      std::swap(rhs[pivot_row], rhs[pivot_col]);
    }

    const double pivot = matrix[pivot_col][pivot_col];
    for (std::size_t row = pivot_col + 1U; row < size; ++row) {
      const double factor = matrix[row][pivot_col] / pivot;
      matrix[row][pivot_col] = 0.0;
      for (std::size_t col = pivot_col + 1U; col < size; ++col) {
        matrix[row][col] -= factor * matrix[pivot_col][col];
      }
      rhs[row] -= factor * rhs[pivot_col];
    }
  }

  solution.fill(0.0);
  for (std::size_t row_from_end = 0U; row_from_end < size; ++row_from_end) {
    const std::size_t row = size - 1U - row_from_end;
    double residual = rhs[row];
    for (std::size_t col = row + 1U; col < size; ++col) {
      residual -= matrix[row][col] * solution[col];
    }
    solution[row] = residual / matrix[row][row];
  }
  return true;
}

struct GroupFitResult
{
  std::size_t sample_count = 0U;
  double sum_abs_y = 0.0;
  double max_abs_y = 0.0;
  double sse = 0.0;
  double sst = 0.0;
  double rmse = 0.0;
  double r2 = 1.0;
};

struct FitStats
{
  std::size_t fitted_groups = 0U;
  std::size_t skipped_groups = 0U;
  std::size_t sample_count = 0U;
  double sum_abs_y = 0.0;
  double max_abs_y = 0.0;
  double sse = 0.0;
  double sst = 0.0;
  double max_group_rmse = 0.0;
  double min_group_r2 = 1.0;
};

auto TryFitGroup(const std::vector<const icts::SegmentChar*>& group, const std::function<double(const icts::SegmentChar&)>& metric_fn,
                 FitBasisKind basis_kind) -> std::optional<GroupFitResult>
{
  const std::size_t basis_size = FitBasisSize(basis_kind);
  if (group.size() < basis_size) {
    return std::nullopt;
  }

  double sum_y = 0.0;
  for (const auto* entry : group) {
    sum_y += metric_fn(*entry);
  }
  const double mean_y = sum_y / static_cast<double>(group.size());

  std::array<std::array<double, 6>, 6> normal_matrix{};
  std::array<double, 6> normal_rhs{};
  for (const auto* entry : group) {
    const auto basis = MakeFitBasis(basis_kind, *entry);
    const double y = metric_fn(*entry);
    for (std::size_t row = 0U; row < basis_size; ++row) {
      normal_rhs[row] += basis[row] * y;
      for (std::size_t col = 0U; col < basis_size; ++col) {
        normal_matrix[row][col] += basis[row] * basis[col];
      }
    }
  }

  std::array<double, 6> coefficients{};
  if (!SolveSmallLinearSystem(normal_matrix, normal_rhs, basis_size, coefficients)) {
    return std::nullopt;
  }

  GroupFitResult result;
  result.sample_count = group.size();
  for (const auto* entry : group) {
    const auto basis = MakeFitBasis(basis_kind, *entry);
    double predicted = 0.0;
    for (std::size_t term = 0U; term < basis_size; ++term) {
      predicted += coefficients[term] * basis[term];
    }

    const double y = metric_fn(*entry);
    const double error = predicted - y;
    result.sse += error * error;
    const double centered = y - mean_y;
    result.sst += centered * centered;
    const double abs_y = std::abs(y);
    result.sum_abs_y += abs_y;
    result.max_abs_y = std::max(result.max_abs_y, abs_y);
  }

  result.rmse = std::sqrt(result.sse / static_cast<double>(result.sample_count));
  if (result.sst > 1e-30) {
    result.r2 = 1.0 - result.sse / result.sst;
  } else {
    result.r2 = result.sse <= 1e-30 ? 1.0 : 0.0;
  }
  return result;
}

auto TryFitSurfaceCoefficients(const std::vector<const icts::SegmentChar*>& group,
                               const std::function<double(const icts::SegmentChar&)>& metric_fn, const realtech_fixture::CharGrid& grid,
                               FitBasisKind basis_kind) -> std::optional<std::array<double, 6>>
{
  const std::size_t basis_size = FitBasisSize(basis_kind);
  if (group.size() < basis_size) {
    return std::nullopt;
  }

  std::array<std::array<double, 6>, 6> normal_matrix{};
  std::array<double, 6> normal_rhs{};
  for (const auto* entry : group) {
    const double input_slew_ns = static_cast<double>(entry->get_input_slew_idx()) * grid.slew_step_ns;
    const double load_cap_pf = static_cast<double>(entry->get_load_cap_idx()) * grid.cap_step_pf;
    const auto basis = MakeFitBasis(basis_kind, input_slew_ns, load_cap_pf);
    const double y = metric_fn(*entry);
    for (std::size_t row = 0U; row < basis_size; ++row) {
      normal_rhs[row] += basis[row] * y;
      for (std::size_t col = 0U; col < basis_size; ++col) {
        normal_matrix[row][col] += basis[row] * basis[col];
      }
    }
  }

  std::array<double, 6> coefficients{};
  if (!SolveSmallLinearSystem(normal_matrix, normal_rhs, basis_size, coefficients)) {
    return std::nullopt;
  }
  return coefficients;
}

auto FitMetricByPattern(const std::vector<icts::SegmentChar>& entries, unsigned length_idx,
                        const std::function<double(const icts::SegmentChar&)>& metric_fn, FitBasisKind basis_kind) -> FitStats
{
  std::unordered_map<icts::PatternId, std::vector<const icts::SegmentChar*>> groups;
  for (const auto& entry : entries) {
    if (entry.get_length_idx() == length_idx) {
      groups[entry.get_pattern_id()].push_back(&entry);
    }
  }

  FitStats stats;
  for (const auto& [pattern_id, group] : groups) {
    (void) pattern_id;
    const auto group_result = TryFitGroup(group, metric_fn, basis_kind);
    if (!group_result.has_value()) {
      ++stats.skipped_groups;
      continue;
    }

    ++stats.fitted_groups;
    stats.sample_count += group_result->sample_count;
    stats.sum_abs_y += group_result->sum_abs_y;
    stats.max_abs_y = std::max(stats.max_abs_y, group_result->max_abs_y);
    stats.sse += group_result->sse;
    stats.sst += group_result->sst;
    stats.max_group_rmse = std::max(stats.max_group_rmse, group_result->rmse);
    stats.min_group_r2 = std::min(stats.min_group_r2, group_result->r2);
  }
  return stats;
}

auto OverallRmse(const FitStats& stats) -> double
{
  return stats.sample_count == 0U ? 0.0 : std::sqrt(stats.sse / static_cast<double>(stats.sample_count));
}

auto OverallR2(const FitStats& stats) -> double
{
  if (stats.sst > 1e-30) {
    return 1.0 - stats.sse / stats.sst;
  }
  return stats.sse <= 1e-30 ? 1.0 : 0.0;
}

auto RelativeRmse(const FitStats& stats) -> double
{
  if (stats.sample_count == 0U) {
    return 0.0;
  }
  const double mean_abs_y = stats.sum_abs_y / static_cast<double>(stats.sample_count);
  const double denominator = std::max(mean_abs_y, 1e-30);
  return OverallRmse(stats) / denominator;
}

auto AppendFitStats(std::ostringstream& report_stream, const std::string& metric_name, FitBasisKind basis_kind, const FitStats& stats)
    -> void
{
  report_stream << "iter1_fit{metric=" << metric_name << ",basis=" << FitBasisName(basis_kind) << ",fitted_groups=" << stats.fitted_groups
                << ",skipped_groups=" << stats.skipped_groups << ",samples=" << stats.sample_count << ",rmse=" << OverallRmse(stats)
                << ",relative_rmse=" << RelativeRmse(stats) << ",r2=" << OverallR2(stats) << ",max_group_rmse=" << stats.max_group_rmse
                << ",min_group_r2=" << stats.min_group_r2 << "}\n";
}

auto AppendIterOneFitReport(std::ostringstream& report_stream, const std::vector<icts::SegmentChar>& entries,
                            const realtech_fixture::CharGrid& grid, unsigned length_idx) -> void
{
  std::size_t length_one_sample_count = 0U;
  std::unordered_map<icts::PatternId, std::size_t> pattern_sample_counts;
  for (const auto& entry : entries) {
    if (entry.get_length_idx() != length_idx) {
      continue;
    }
    ++length_one_sample_count;
    ++pattern_sample_counts[entry.get_pattern_id()];
  }

  report_stream << "iter1_fit_input_space=input_slew_idx,load_cap_idx\n";
  report_stream << "iter1_fit_length_idx=" << length_idx << "\n";
  report_stream << "iter1_fit_length_um=" << (static_cast<double>(length_idx) * grid.length_step_um) << "\n";
  report_stream << "iter1_fit_raw_sample_count=" << length_one_sample_count << "\n";
  report_stream << "iter1_fit_pattern_group_count=" << pattern_sample_counts.size() << "\n";

  const auto append_metric = [&](const std::string& metric_name, const std::function<double(const icts::SegmentChar&)>& metric_fn) -> void {
    AppendFitStats(report_stream, metric_name, FitBasisKind::kLinear,
                   FitMetricByPattern(entries, length_idx, metric_fn, FitBasisKind::kLinear));
    AppendFitStats(report_stream, metric_name, FitBasisKind::kQuadratic,
                   FitMetricByPattern(entries, length_idx, metric_fn, FitBasisKind::kQuadratic));
  };

  append_metric("output_slew_ns", [&grid](const icts::SegmentChar& entry) -> double {
    return static_cast<double>(entry.get_output_slew_idx()) * grid.slew_step_ns;
  });
  append_metric("driven_cap_pf", [&grid](const icts::SegmentChar& entry) -> double {
    return static_cast<double>(entry.get_driven_cap_idx()) * grid.cap_step_pf;
  });
  append_metric("delay_ns", [](const icts::SegmentChar& entry) -> double { return entry.get_delay(); });
  append_metric("power_w", [](const icts::SegmentChar& entry) -> double { return entry.get_power(); });
  append_metric("source_boundary_net_switch_power_w",
                [](const icts::SegmentChar& entry) -> double { return entry.get_source_boundary_net_switch_power(); });
}

}  // namespace icts_test
