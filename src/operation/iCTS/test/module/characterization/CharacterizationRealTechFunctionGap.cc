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
 * @file CharacterizationRealTechFunctionGap.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Functional compose gap helpers for real-tech characterization regression tests.
 */
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "module/characterization/CharacterizationRealTechExactRegression.hh"

namespace icts_test {

auto AddMetricGap(MetricGapAccumulator& accumulator, double direct_value, double predicted_value, const std::string& example) -> void
{
  ++accumulator.count;
  accumulator.sum_direct += direct_value;
  accumulator.sum_predicted += predicted_value;
  const double delta = predicted_value - direct_value;
  const double abs_delta = std::abs(delta);
  accumulator.sum_abs_delta += abs_delta;
  accumulator.sum_sq_delta += delta * delta;
  accumulator.max_rel_delta = std::max(accumulator.max_rel_delta, abs_delta / std::max(std::abs(direct_value), 1e-30));
  if (delta < -1e-15) {
    ++accumulator.predicted_lower_count;
  } else if (delta > 1e-15) {
    ++accumulator.predicted_higher_count;
  }
  if (abs_delta > accumulator.max_abs_delta) {
    accumulator.max_abs_delta = abs_delta;
    accumulator.worst_example = example;
  }
}

auto MetricRmse(const MetricGapAccumulator& accumulator) -> double
{
  return accumulator.count == 0U ? 0.0 : std::sqrt(accumulator.sum_sq_delta / static_cast<double>(accumulator.count));
}

auto MetricMeanAbs(const MetricGapAccumulator& accumulator) -> double
{
  return accumulator.count == 0U ? 0.0 : accumulator.sum_abs_delta / static_cast<double>(accumulator.count);
}

auto AnalyzeFunctionComposeGap(const std::string& source_label, unsigned target_length_idx, FitBasisKind basis_kind,
                               const std::vector<icts::SegmentChar>& direct_entries,
                               const realtech_fixture::SegmentFrontierContext& segment_context,
                               const std::unordered_map<std::string, FunctionalSurfaceModel>& model_by_unit_key,
                               const realtech_fixture::CharGrid& grid, double max_slew_ns, double max_cap_pf) -> FunctionComposeGapStats
{
  FunctionComposeGapStats stats;
  stats.source_label = source_label;
  stats.target_length_idx = target_length_idx;
  stats.basis_kind = basis_kind;

  for (const auto& direct_entry : direct_entries) {
    if (direct_entry.get_length_idx() != target_length_idx) {
      continue;
    }
    ++stats.direct_count;

    const auto pattern_it = segment_context.patterns.find(direct_entry.get_pattern_id());
    if (pattern_it == segment_context.patterns.end()) {
      continue;
    }
    const auto unit_keys = DecomposeToUnitPatternKeys(pattern_it->second, target_length_idx);
    if (!unit_keys.has_value()) {
      continue;
    }
    ++stats.decomposable_count;

    std::vector<const FunctionalSurfaceModel*> unit_models;
    unit_models.reserve(unit_keys->size());
    bool has_missing_model = false;
    for (const auto& unit_key : *unit_keys) {
      const auto model_it = model_by_unit_key.find(unit_key);
      if (model_it == model_by_unit_key.end()) {
        has_missing_model = true;
        if (stats.missing_model_example.empty()) {
          stats.missing_model_example = unit_key;
        }
        break;
      }
      unit_models.push_back(&model_it->second);
    }
    if (has_missing_model) {
      ++stats.missing_model_count;
      continue;
    }

    const double input_slew_ns = static_cast<double>(direct_entry.get_input_slew_idx()) * grid.slew_step_ns;
    const double load_cap_pf = static_cast<double>(direct_entry.get_load_cap_idx()) * grid.cap_step_pf;
    const auto prediction = PredictFunctionalCompose(unit_models, input_slew_ns, load_cap_pf, max_slew_ns, max_cap_pf);
    stats.max_fixed_point_residual = std::max(stats.max_fixed_point_residual, prediction.residual);
    stats.max_fixed_point_iterations = std::max(stats.max_fixed_point_iterations, prediction.iterations);
    if (!prediction.converged) {
      ++stats.convergence_failure_count;
      if (stats.convergence_failure_example.empty()) {
        stats.convergence_failure_example = realtech_fixture::FormatSegmentChar(direct_entry, grid);
      }
      continue;
    }
    if (!prediction.is_valid) {
      ++stats.invalid_prediction_count;
      if (stats.invalid_prediction_example.empty()) {
        stats.invalid_prediction_example = realtech_fixture::FormatSegmentChar(direct_entry, grid);
      }
      continue;
    }
    if (prediction.is_out_of_domain) {
      ++stats.out_of_domain_count;
    }

    ++stats.evaluated_count;
    std::ostringstream example;
    example << "direct=" << realtech_fixture::FormatSegmentChar(direct_entry, grid)
            << ", predicted{output_slew_ns=" << prediction.output_slew_ns << ",driven_cap_pf=" << prediction.driven_cap_pf
            << ",delay_ns=" << prediction.delay_ns << ",power_w=" << prediction.power_w
            << ",source_boundary_power_w=" << prediction.source_boundary_power_w << "}";
    const auto example_text = example.str();

    AddMetricGap(stats.output_slew, static_cast<double>(direct_entry.get_output_slew_idx()) * grid.slew_step_ns, prediction.output_slew_ns,
                 example_text);
    AddMetricGap(stats.driven_cap, static_cast<double>(direct_entry.get_driven_cap_idx()) * grid.cap_step_pf, prediction.driven_cap_pf,
                 example_text);
    AddMetricGap(stats.delay, direct_entry.get_delay(), prediction.delay_ns, example_text);
    AddMetricGap(stats.power, direct_entry.get_power(), prediction.power_w, example_text);
    AddMetricGap(stats.source_boundary_power, direct_entry.get_source_boundary_net_switch_power(), prediction.source_boundary_power_w,
                 example_text);
  }

  return stats;
}

auto AnalyzeStructuralCapFunctionComposeGap(const std::string& source_label, unsigned target_length_idx, FitBasisKind basis_kind,
                                            const std::vector<icts::SegmentChar>& direct_entries,
                                            const realtech_fixture::SegmentFrontierContext& segment_context,
                                            const std::unordered_map<std::string, FunctionalSurfaceModel>& model_by_unit_key,
                                            const std::unordered_map<std::string, StructuralCapOperator>& cap_operator_by_unit_key,
                                            const realtech_fixture::CharGrid& grid, const icts::UniformValueLattice& cap_lattice,
                                            double max_slew_ns, double max_cap_pf) -> FunctionComposeGapStats
{
  FunctionComposeGapStats stats;
  stats.source_label = source_label;
  stats.target_length_idx = target_length_idx;
  stats.basis_kind = basis_kind;

  for (const auto& direct_entry : direct_entries) {
    if (direct_entry.get_length_idx() != target_length_idx) {
      continue;
    }
    ++stats.direct_count;

    const auto pattern_it = segment_context.patterns.find(direct_entry.get_pattern_id());
    if (pattern_it == segment_context.patterns.end()) {
      continue;
    }
    const auto unit_keys = DecomposeToUnitPatternKeys(pattern_it->second, target_length_idx);
    if (!unit_keys.has_value()) {
      continue;
    }
    ++stats.decomposable_count;

    std::vector<const FunctionalSurfaceModel*> unit_models;
    std::vector<const StructuralCapOperator*> cap_operators;
    unit_models.reserve(unit_keys->size());
    cap_operators.reserve(unit_keys->size());
    bool has_missing_model = false;
    for (const auto& unit_key : *unit_keys) {
      const auto model_it = model_by_unit_key.find(unit_key);
      const auto cap_operator_it = cap_operator_by_unit_key.find(unit_key);
      if (model_it == model_by_unit_key.end() || cap_operator_it == cap_operator_by_unit_key.end()) {
        has_missing_model = true;
        if (stats.missing_model_example.empty()) {
          stats.missing_model_example = unit_key;
        }
        break;
      }
      unit_models.push_back(&model_it->second);
      cap_operators.push_back(&cap_operator_it->second);
    }
    if (has_missing_model) {
      ++stats.missing_model_count;
      continue;
    }

    const double input_slew_ns = static_cast<double>(direct_entry.get_input_slew_idx()) * grid.slew_step_ns;
    const double load_cap_pf = static_cast<double>(direct_entry.get_load_cap_idx()) * grid.cap_step_pf;
    const auto prediction
        = PredictStructuralCapFunctionalCompose(unit_models, cap_operators, input_slew_ns, load_cap_pf, max_slew_ns, max_cap_pf);
    stats.max_fixed_point_residual = std::max(stats.max_fixed_point_residual, prediction.residual);
    stats.max_fixed_point_iterations = std::max(stats.max_fixed_point_iterations, prediction.iterations);
    if (!prediction.converged) {
      ++stats.convergence_failure_count;
      if (stats.convergence_failure_example.empty()) {
        stats.convergence_failure_example = realtech_fixture::FormatSegmentChar(direct_entry, grid);
      }
      continue;
    }
    if (!prediction.is_valid) {
      ++stats.invalid_prediction_count;
      if (stats.invalid_prediction_example.empty()) {
        stats.invalid_prediction_example = realtech_fixture::FormatSegmentChar(direct_entry, grid);
      }
      continue;
    }
    if (prediction.is_out_of_domain) {
      ++stats.out_of_domain_count;
    }

    ++stats.evaluated_count;
    std::ostringstream example;
    example << "direct=" << realtech_fixture::FormatSegmentChar(direct_entry, grid)
            << ", predicted{output_slew_ns=" << prediction.output_slew_ns << ",driven_cap_pf=" << prediction.driven_cap_pf
            << ",delay_ns=" << prediction.delay_ns << ",power_w=" << prediction.power_w
            << ",source_boundary_power_w=" << prediction.source_boundary_power_w << "}";
    const auto example_text = example.str();
    const double predicted_driven_cap_bucket_pf
        = static_cast<double>(cap_lattice.coveringIndex(prediction.driven_cap_pf)) * grid.cap_step_pf;

    AddMetricGap(stats.output_slew, static_cast<double>(direct_entry.get_output_slew_idx()) * grid.slew_step_ns, prediction.output_slew_ns,
                 example_text);
    AddMetricGap(stats.driven_cap, static_cast<double>(direct_entry.get_driven_cap_idx()) * grid.cap_step_pf,
                 predicted_driven_cap_bucket_pf, example_text);
    AddMetricGap(stats.delay, direct_entry.get_delay(), prediction.delay_ns, example_text);
    AddMetricGap(stats.power, direct_entry.get_power(), prediction.power_w, example_text);
    AddMetricGap(stats.source_boundary_power, direct_entry.get_source_boundary_net_switch_power(), prediction.source_boundary_power_w,
                 example_text);
  }

  return stats;
}

auto AppendMetricGap(std::ostringstream& report_stream, const std::string& source_label, FitBasisKind basis_kind,
                     unsigned target_length_idx, const std::string& metric_name, const MetricGapAccumulator& accumulator) -> void
{
  report_stream << "function_compose_metric{source=" << source_label << ",basis=" << FitBasisName(basis_kind)
                << ",target_length_idx=" << target_length_idx << ",metric=" << metric_name << ",count=" << accumulator.count
                << ",sum_direct=" << accumulator.sum_direct << ",sum_predicted=" << accumulator.sum_predicted
                << ",ratio_predicted_over_direct=" << SafeRatio(accumulator.sum_predicted, accumulator.sum_direct)
                << ",rmse=" << MetricRmse(accumulator) << ",mean_abs=" << MetricMeanAbs(accumulator)
                << ",max_abs=" << accumulator.max_abs_delta << ",max_rel=" << accumulator.max_rel_delta
                << ",predicted_lower_count=" << accumulator.predicted_lower_count
                << ",predicted_higher_count=" << accumulator.predicted_higher_count << "}\n";
  report_stream << "function_compose_metric_worst{source=" << source_label << ",basis=" << FitBasisName(basis_kind)
                << ",target_length_idx=" << target_length_idx << ",metric=" << metric_name << "," << accumulator.worst_example << "}\n";
}

auto AppendFunctionComposeGapStats(std::ostringstream& report_stream, const FunctionComposeGapStats& stats,
                                   const realtech_fixture::CharGrid& grid) -> void
{
  const std::string label = "function_compose_gap{source=" + stats.source_label + ",basis=" + FitBasisName(stats.basis_kind)
                            + ",target_length_idx=" + std::to_string(stats.target_length_idx);
  report_stream << label << ",target_length_um=" << (static_cast<double>(stats.target_length_idx) * grid.length_step_um)
                << ",direct_count=" << stats.direct_count << ",decomposable_count=" << stats.decomposable_count
                << ",missing_model_count=" << stats.missing_model_count << ",convergence_failure_count=" << stats.convergence_failure_count
                << ",invalid_prediction_count=" << stats.invalid_prediction_count << ",out_of_domain_count=" << stats.out_of_domain_count
                << ",evaluated_count=" << stats.evaluated_count << ",max_fixed_point_residual=" << stats.max_fixed_point_residual
                << ",max_fixed_point_iterations=" << stats.max_fixed_point_iterations << "}\n";
  AppendMetricGap(report_stream, stats.source_label, stats.basis_kind, stats.target_length_idx, "output_slew_ns", stats.output_slew);
  AppendMetricGap(report_stream, stats.source_label, stats.basis_kind, stats.target_length_idx, "driven_cap_pf", stats.driven_cap);
  AppendMetricGap(report_stream, stats.source_label, stats.basis_kind, stats.target_length_idx, "delay_ns", stats.delay);
  AppendMetricGap(report_stream, stats.source_label, stats.basis_kind, stats.target_length_idx, "power_w", stats.power);
  AppendMetricGap(report_stream, stats.source_label, stats.basis_kind, stats.target_length_idx, "source_boundary_power_w",
                  stats.source_boundary_power);
  report_stream << "function_compose_missing_model_example{source=" << stats.source_label << ",basis=" << FitBasisName(stats.basis_kind)
                << ",target_length_idx=" << stats.target_length_idx << "," << stats.missing_model_example << "}\n";
  report_stream << "function_compose_convergence_failure_example{source=" << stats.source_label
                << ",basis=" << FitBasisName(stats.basis_kind) << ",target_length_idx=" << stats.target_length_idx << ","
                << stats.convergence_failure_example << "}\n";
  report_stream << "function_compose_invalid_prediction_example{source=" << stats.source_label
                << ",basis=" << FitBasisName(stats.basis_kind) << ",target_length_idx=" << stats.target_length_idx << ","
                << stats.invalid_prediction_example << "}\n";
}

auto AppendStructuralCapOperatorStats(std::ostringstream& report_stream,
                                      const std::unordered_map<std::string, StructuralCapOperator>& cap_operator_by_unit_key) -> void
{
  std::size_t wire_operator_count = 0U;
  std::size_t buffered_operator_count = 0U;
  std::size_t sample_count = 0U;
  double max_abs_residual_pf = 0.0;
  for (const auto& [unit_key, cap_operator] : cap_operator_by_unit_key) {
    (void) unit_key;
    if (cap_operator.alpha == 1.0) {
      ++wire_operator_count;
    } else {
      ++buffered_operator_count;
    }
    sample_count += cap_operator.sample_count;
    max_abs_residual_pf = std::max(max_abs_residual_pf, cap_operator.max_abs_residual_pf);
  }

  report_stream << "structural_cap_operator_count{total=" << cap_operator_by_unit_key.size() << ",wire=" << wire_operator_count
                << ",buffered=" << buffered_operator_count << ",samples=" << sample_count << ",max_abs_residual_pf=" << max_abs_residual_pf
                << "}\n";
}

auto AppendStructuralCapOperatorSampleGap(std::ostringstream& report_stream,
                                          const std::unordered_map<std::string, StructuralCapOperator>& cap_operator_by_unit_key,
                                          const std::vector<icts::SegmentChar>& entries,
                                          const realtech_fixture::SegmentFrontierContext& segment_context,
                                          const realtech_fixture::CharGrid& grid, const icts::UniformValueLattice& cap_lattice) -> void
{
  MetricGapAccumulator physical_gap;
  MetricGapAccumulator bucket_gap;
  std::size_t missing_operator_count = 0U;
  for (const auto& entry : entries) {
    if (entry.get_length_idx() != 1U) {
      continue;
    }
    const auto pattern_it = segment_context.patterns.find(entry.get_pattern_id());
    if (pattern_it == segment_context.patterns.end()) {
      continue;
    }
    const auto operator_it = cap_operator_by_unit_key.find(MakeUnitPatternKey(pattern_it->second));
    if (operator_it == cap_operator_by_unit_key.end()) {
      ++missing_operator_count;
      continue;
    }

    const double direct_pf = static_cast<double>(entry.get_driven_cap_idx()) * grid.cap_step_pf;
    const double load_cap_pf = static_cast<double>(entry.get_load_cap_idx()) * grid.cap_step_pf;
    const double predicted_physical_pf = (operator_it->second.alpha * load_cap_pf) + operator_it->second.eta_pf;
    const double predicted_bucket_pf
        = static_cast<double>(cap_lattice.tryObservedIndex(predicted_physical_pf).value_or(0U)) * grid.cap_step_pf;
    const std::string example = realtech_fixture::FormatSegmentChar(entry, grid);
    AddMetricGap(physical_gap, direct_pf, predicted_physical_pf, example);
    AddMetricGap(bucket_gap, direct_pf, predicted_bucket_pf, example);
  }

  report_stream << "structural_cap_operator_sample_gap{missing_operator_count=" << missing_operator_count << "}\n";
  AppendMetricGap(report_stream, "iter1_unit_physical_structural_cap", FitBasisKind::kLinear, 1U, "driven_cap_pf", physical_gap);
  AppendMetricGap(report_stream, "iter1_unit_bucket_structural_cap", FitBasisKind::kLinear, 1U, "driven_cap_pf", bucket_gap);
}

}  // namespace icts_test
