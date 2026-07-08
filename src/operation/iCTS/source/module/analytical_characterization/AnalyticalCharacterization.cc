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
 * @file AnalyticalCharacterization.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-14
 * @brief Analytical characterization catalog builder implementation.
 */

#include "AnalyticalCharacterization.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "AnalyticalFit.hh"
#include "BufferingPattern.hh"
#include "ClockRouteSegmentRc.hh"
#include "PatternId.hh"
#include "SegmentChar.hh"
#include "ValueLattice.hh"
#include "characterization/Characterization.hh"

namespace icts::analytical {
namespace {

struct GroupedSegmentChars
{
  AnalyticalModelKey key;
  std::vector<SegmentChar> chars;
};

auto FindPattern(PatternId pattern_id, const std::vector<BufferingPattern>& buffering_patterns) -> const BufferingPattern*
{
  const auto it = std::ranges::find_if(buffering_patterns,
                                       [&](const BufferingPattern& pattern) -> bool { return pattern.get_pattern_id() == pattern_id; });
  return it == buffering_patterns.end() ? nullptr : &*it;
}

auto GroupSegmentChars(const std::vector<SegmentChar>& segment_chars) -> std::vector<GroupedSegmentChars>
{
  std::unordered_map<AnalyticalModelKey, std::vector<SegmentChar>, AnalyticalModelKeyHash> grouped;
  std::vector<AnalyticalModelKey> stable_keys;
  for (const auto& segment_char : segment_chars) {
    AnalyticalModelKey key{.pattern_id = segment_char.get_pattern_id(), .length_idx = segment_char.get_length_idx()};
    if (!grouped.contains(key)) {
      stable_keys.push_back(key);
    }
    grouped[key].push_back(segment_char);
  }

  std::ranges::sort(stable_keys, [](const AnalyticalModelKey& lhs, const AnalyticalModelKey& rhs) -> bool {
    if (lhs.length_idx != rhs.length_idx) {
      return lhs.length_idx < rhs.length_idx;
    }
    return lhs.pattern_id.pack() < rhs.pattern_id.pack();
  });

  std::vector<GroupedSegmentChars> result;
  result.reserve(stable_keys.size());
  for (const auto& key : stable_keys) {
    result.push_back(GroupedSegmentChars{.key = key, .chars = std::move(grouped[key])});
  }
  return result;
}

auto BuildSamples(const std::vector<SegmentChar>& chars, const UniformValueLattice& slew_lattice, const UniformValueLattice& cap_lattice,
                  AnalyticalMetric metric) -> std::vector<AnalyticalFitSample>
{
  std::vector<AnalyticalFitSample> samples;
  samples.reserve(chars.size());
  for (const auto& segment_char : chars) {
    double value = 0.0;
    switch (metric) {
      case AnalyticalMetric::kOutputSlew:
        value = slew_lattice.valueForIndex(segment_char.get_output_slew_idx());
        break;
      case AnalyticalMetric::kDelay:
        value = segment_char.get_delay();
        break;
      case AnalyticalMetric::kPower:
        value = segment_char.get_power();
        break;
      case AnalyticalMetric::kSourceBoundaryNetSwitchPower:
        value = segment_char.get_source_boundary_net_switch_power();
        break;
    }
    samples.push_back(AnalyticalFitSample{
        .input_slew_ns = slew_lattice.valueForIndex(segment_char.get_input_slew_idx()),
        .load_cap_pf = cap_lattice.valueForIndex(segment_char.get_load_cap_idx()),
        .value = value,
    });
  }
  return samples;
}

auto MakeFitConfig(AnalyticalMetric metric, const AnalyticalCharacterizationConfig& config) -> AnalyticalFitConfig
{
  AnalyticalFitConfig fit_config;
  fit_config.metric = metric;
  fit_config.min_r2 = config.min_r2;
  switch (metric) {
    case AnalyticalMetric::kOutputSlew:
      fit_config.basis = config.output_slew_basis;
      fit_config.max_abs_residual = config.max_output_slew_abs_residual_ns;
      fit_config.max_bucket_residual = config.max_output_slew_bucket_residual;
      fit_config.require_monotonic = config.require_monotonic_output_slew;
      break;
    case AnalyticalMetric::kDelay:
      fit_config.basis = config.delay_basis;
      fit_config.max_relative_residual = config.max_delay_relative_residual;
      fit_config.require_monotonic = config.require_monotonic_delay;
      break;
    case AnalyticalMetric::kPower:
      fit_config.basis = config.power_basis;
      fit_config.max_relative_residual = config.max_power_relative_residual;
      fit_config.require_monotonic = config.require_monotonic_power;
      break;
    case AnalyticalMetric::kSourceBoundaryNetSwitchPower:
      fit_config.basis = config.source_boundary_power_basis;
      fit_config.max_relative_residual = config.max_power_relative_residual;
      fit_config.require_monotonic = config.require_monotonic_source_boundary_power;
      break;
  }
  return fit_config;
}

auto CanUseSparseConstantModel(const std::string& failure_reason) -> bool
{
  return failure_reason == "insufficient_samples" || failure_reason == "singular_normal_equation" || failure_reason == "invalid_domain";
}

auto AssignAnalyticalSurfaceModel(AnalyticalSurfaceModel& dst, const AnalyticalSurfaceModel& src) -> void
{
  dst.metric = src.metric;
  dst.basis = src.basis;
  dst.domain = src.domain;
  dst.coefficients = src.coefficients;
  dst.quality = src.quality;
}

auto AssignStructuralCapOperator(StructuralCapOperator& dst, const StructuralCapOperator& src) -> void
{
  dst.alpha = src.alpha;
  dst.eta_pf = src.eta_pf;
  dst.physical = src.physical;
  dst.bucket_compatible = src.bucket_compatible;
  dst.source = src.source;
}

auto BuildSparseConstantModel(const std::vector<AnalyticalFitSample>& samples, const AnalyticalFitConfig& fit_config,
                              const UniformValueLattice& slew_lattice, const UniformValueLattice& cap_lattice)
    -> std::optional<AnalyticalSurfaceModel>
{
  if (samples.empty() || !slew_lattice.isValid() || !cap_lattice.isValid()) {
    return std::nullopt;
  }

  double total_value = 0.0;
  for (const auto& sample : samples) {
    total_value += sample.value;
  }
  const double mean_value = total_value / static_cast<double>(samples.size());

  AnalyticalSurfaceModel model{};
  model.metric = fit_config.metric;
  model.basis = fit_config.basis;
  model.domain = AnalyticalDomain{
      .slew_min_ns = slew_lattice.stepValue(),
      .slew_max_ns = slew_lattice.maxValue(),
      .cap_min_pf = cap_lattice.stepValue(),
      .cap_max_pf = cap_lattice.maxValue(),
  };
  model.coefficients.assign(AnalyticalBasisTermCount(model.basis), 0.0);
  if (model.coefficients.empty()) {
    return std::nullopt;
  }
  model.coefficients.front() = mean_value;

  auto& quality = model.quality;
  quality.sample_count = samples.size();
  double square_error = 0.0;
  for (const auto& sample : samples) {
    const double residual = mean_value - sample.value;
    const double abs_residual = std::abs(residual);
    square_error += residual * residual;
    quality.max_abs_residual = std::max(quality.max_abs_residual, abs_residual);
    quality.max_relative_residual
        = std::max(quality.max_relative_residual, abs_residual / std::max(std::abs(sample.value), fit_config.relative_floor));
    if (fit_config.bucket_size > 0.0) {
      quality.max_bucket_residual = std::max(quality.max_bucket_residual, abs_residual / fit_config.bucket_size);
    }
  }
  quality.rmse = std::sqrt(square_error / static_cast<double>(samples.size()));
  quality.r2_valid = false;
  quality.r2 = 1.0;
  quality.monotonicity_passed = true;
  quality.bucket_aware_passed = true;
  quality.accepted = true;
  quality.failure_reason = "sparse_constant_model";
  return model;
}

auto FitMetric(const GroupedSegmentChars& group, const UniformValueLattice& slew_lattice, const UniformValueLattice& cap_lattice,
               AnalyticalMetric metric, const AnalyticalCharacterizationConfig& config, AnalyticalCharacterizationBuild& result,
               std::optional<AnalyticalSurfaceModel>& model_slot) -> void
{
  auto fit_config = MakeFitConfig(metric, config);
  if (metric == AnalyticalMetric::kOutputSlew) {
    fit_config.bucket_size = slew_lattice.stepValue();
  }
  auto samples = BuildSamples(group.chars, slew_lattice, cap_lattice, metric);
  auto fit = FitAnalyticalSurface(samples, fit_config);
  if (fit.summary.success && fit.output.model.has_value()) {
    AssignAnalyticalSurfaceModel(model_slot.emplace(), *fit.output.model);
    return;
  }

  if (config.allow_sparse_constant_model && CanUseSparseConstantModel(fit.summary.failure_reason)) {
    auto sparse_model = BuildSparseConstantModel(samples, fit_config, slew_lattice, cap_lattice);
    if (sparse_model.has_value()) {
      AssignAnalyticalSurfaceModel(model_slot.emplace(), *sparse_model);
      return;
    }
  }

  ++result.summary.rejected_fit_count;
  result.summary.failures.push_back(AnalyticalCharacterizationFailure{
      .key = group.key,
      .metric = metric,
      .reason = fit.summary.failure_reason.empty() ? std::string{"fit_rejected"} : fit.summary.failure_reason,
  });
}

auto ValidateBucketCompatibility(const StructuralCapOperator& op, const std::vector<SegmentChar>& grouped_chars,
                                 const UniformValueLattice& cap_lattice) -> bool
{
  for (const auto& segment_char : grouped_chars) {
    const double load_cap_pf = cap_lattice.valueForIndex(segment_char.get_load_cap_idx());
    const unsigned predicted_bucket = cap_lattice.coveringIndex(op.apply(load_cap_pf));
    if (predicted_bucket != segment_char.get_driven_cap_idx()) {
      return false;
    }
  }
  return true;
}

auto MakeExactStructuralCapOperator(const BufferingPattern* pattern, const AnalyticalCharacterizationConfig& config)
    -> std::optional<StructuralCapOperator>
{
  if (pattern == nullptr || config.length_unit_um <= 0.0 || !std::isfinite(config.clock_route_segment_rc.capacitance_per_um_pf)
      || config.clock_route_segment_rc.capacitance_per_um_pf <= 0.0) {
    return std::nullopt;
  }

  const double total_length_um = static_cast<double>(pattern->get_length_idx()) * config.length_unit_um;
  if (total_length_um <= 0.0) {
    return std::nullopt;
  }

  const auto calc_wire_cap_pf
      = [&](double wirelength_um) -> double { return std::max(0.0, wirelength_um) * config.clock_route_segment_rc.capacitance_per_um_pf; };

  if (pattern->isWirePattern()) {
    auto op = StructuralCapOperator::wire(calc_wire_cap_pf(total_length_um));
    op.source = "exact_wire";
    return op;
  }

  const auto& buffer_positions = pattern->get_buffer_positions();
  const auto& cell_masters = pattern->get_cell_masters();
  if (buffer_positions.empty() || cell_masters.empty() || buffer_positions.size() != cell_masters.size()) {
    return std::nullopt;
  }

  const double first_buffer_position = buffer_positions.front();
  if (first_buffer_position <= 0.0 || first_buffer_position > 1.0) {
    return std::nullopt;
  }

  const auto input_cap_iter = config.buffer_input_cap_pf_by_cell_master.find(cell_masters.front());
  if (input_cap_iter == config.buffer_input_cap_pf_by_cell_master.end()) {
    return std::nullopt;
  }

  const double first_buffer_input_cap_pf = input_cap_iter->second;
  if (!std::isfinite(first_buffer_input_cap_pf) || first_buffer_input_cap_pf <= 0.0) {
    return std::nullopt;
  }

  const double pre_buffer_wire_length_um = total_length_um * first_buffer_position;
  auto op = StructuralCapOperator::buffered(first_buffer_input_cap_pf, calc_wire_cap_pf(pre_buffer_wire_length_um));
  op.source = "exact_buffered";
  return op;
}

auto MakeBucketRepresentativeOperator(const BufferingPattern* pattern, const std::vector<SegmentChar>& grouped_chars,
                                      const UniformValueLattice& cap_lattice) -> std::optional<StructuralCapOperator>
{
  if (grouped_chars.empty()) {
    return std::nullopt;
  }

  StructuralCapOperator op{};
  op.physical = false;
  op.source = "bucket_representative";
  if (pattern != nullptr && pattern->isBufferPattern()) {
    op.alpha = 0.0;
    op.eta_pf = cap_lattice.valueForIndex(grouped_chars.front().get_driven_cap_idx());
    return op;
  }

  op.alpha = 1.0;
  double min_eta_pf = std::numeric_limits<double>::infinity();
  for (const auto& segment_char : grouped_chars) {
    const double driven_cap_pf = cap_lattice.valueForIndex(segment_char.get_driven_cap_idx());
    const double load_cap_pf = cap_lattice.valueForIndex(segment_char.get_load_cap_idx());
    min_eta_pf = std::min(min_eta_pf, std::max(0.0, driven_cap_pf - load_cap_pf));
  }
  op.eta_pf = std::isfinite(min_eta_pf) ? min_eta_pf : 0.0;
  return op;
}

auto BuildStructuralCapOperator(PatternId pattern_id, const std::vector<BufferingPattern>& buffering_patterns,
                                const std::vector<SegmentChar>& grouped_chars, const UniformValueLattice& cap_lattice,
                                const AnalyticalCharacterizationConfig& config, std::optional<StructuralCapOperator>& operator_slot) -> void
{
  const auto* pattern = FindPattern(pattern_id, buffering_patterns);
  auto op = config.prefer_exact_structural_cap ? MakeExactStructuralCapOperator(pattern, config)
                                               : MakeBucketRepresentativeOperator(pattern, grouped_chars, cap_lattice);
  if (!op.has_value() && !config.prefer_exact_structural_cap) {
    return;
  }
  if (!op.has_value()) {
    return;
  }

  op->bucket_compatible = ValidateBucketCompatibility(*op, grouped_chars, cap_lattice);
  AssignStructuralCapOperator(operator_slot.emplace(), *op);
}

}  // namespace

auto BuildBucketCompatibleStructuralCapOperator(PatternId pattern_id, const std::vector<BufferingPattern>& buffering_patterns,
                                                const std::vector<SegmentChar>& grouped_chars, const UniformValueLattice& cap_lattice)
    -> std::optional<StructuralCapOperator>
{
  const auto* pattern = FindPattern(pattern_id, buffering_patterns);
  auto op = MakeBucketRepresentativeOperator(pattern, grouped_chars, cap_lattice);
  if (!op.has_value()) {
    return std::nullopt;
  }
  op->bucket_compatible = ValidateBucketCompatibility(*op, grouped_chars, cap_lattice);
  return op;
}

auto AnalyticalCharacterization::buildFromCharBuilder(const CharBuilder& char_builder, const AnalyticalCharacterizationConfig& config)
    -> AnalyticalCharacterizationBuild
{
  auto builder_config = config;
  if (builder_config.length_unit_um <= 0.0) {
    builder_config.length_unit_um = char_builder.get_wirelength_unit_um();
  }
  builder_config.clock_route_segment_rc = char_builder.get_clock_route_segment_rc();
  builder_config.buffer_input_cap_pf_by_cell_master.clear();
  for (const auto& buffer_cell : char_builder.get_characterization_buffer_cells()) {
    builder_config.buffer_input_cap_pf_by_cell_master[buffer_cell.cell_master] = buffer_cell.input_cap_pf;
  }
  return buildFromSegmentChars(char_builder.get_segment_chars(), char_builder.get_buffering_patterns(), char_builder.get_slew_lattice(),
                               char_builder.get_cap_lattice(), builder_config);
}

auto AnalyticalCharacterization::buildFromSegmentChars(const std::vector<SegmentChar>& segment_chars,
                                                       const std::vector<BufferingPattern>& buffering_patterns,
                                                       const UniformValueLattice& slew_lattice, const UniformValueLattice& cap_lattice,
                                                       const AnalyticalCharacterizationConfig& config) -> AnalyticalCharacterizationBuild
{
  AnalyticalCharacterizationBuild result;
  if (segment_chars.empty() || !slew_lattice.isValid() || !cap_lattice.isValid()) {
    result.summary.failures.push_back(AnalyticalCharacterizationFailure{
        .key = {},
        .metric = AnalyticalMetric::kDelay,
        .reason = "empty_samples_or_invalid_lattice",
    });
    return result;
  }

  for (const auto& group : GroupSegmentChars(segment_chars)) {
    AnalyticalModelSet model_set;
    model_set.key = group.key;
    FitMetric(group, slew_lattice, cap_lattice, AnalyticalMetric::kOutputSlew, config, result, model_set.output_slew_model);
    FitMetric(group, slew_lattice, cap_lattice, AnalyticalMetric::kDelay, config, result, model_set.delay_model);
    FitMetric(group, slew_lattice, cap_lattice, AnalyticalMetric::kPower, config, result, model_set.power_model);
    FitMetric(group, slew_lattice, cap_lattice, AnalyticalMetric::kSourceBoundaryNetSwitchPower, config, result,
              model_set.source_boundary_power_model);
    BuildStructuralCapOperator(group.key.pattern_id, buffering_patterns, group.chars, cap_lattice, config, model_set.source_cap_operator);
    if (model_set.source_cap_operator.has_value()) {
      ++result.summary.structural_cap_operator_count;
      if (!model_set.source_cap_operator->bucket_compatible) {
        result.summary.failures.push_back(AnalyticalCharacterizationFailure{
            .key = group.key,
            .metric = AnalyticalMetric::kDelay,
            .reason = "structural_cap_bucket_incompatible",
        });
      }
    } else {
      result.summary.failures.push_back(AnalyticalCharacterizationFailure{
          .key = group.key,
          .metric = AnalyticalMetric::kDelay,
          .reason = "missing_structural_cap_operator",
      });
    }

    if (model_set.isComplete()) {
      result.output.catalog.addModelSet(std::move(model_set));
      ++result.summary.model_set_count;
    }
  }

  result.summary.success = result.summary.model_set_count > 0U && result.summary.rejected_fit_count == 0U;
  return result;
}

}  // namespace icts::analytical
