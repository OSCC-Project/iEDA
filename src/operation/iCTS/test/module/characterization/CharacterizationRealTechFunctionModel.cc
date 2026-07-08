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
 * @file CharacterizationRealTechFunctionModel.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Functional surface model helpers for real-tech characterization regression tests.
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "database/io/Wrapper.hh"
#include "module/characterization/CharacterizationRealTechExactRegression.hh"

namespace icts_test {

auto MakeUnitPatternKey(const std::vector<double>& buffer_positions, const std::vector<std::string>& cell_masters) -> std::string
{
  std::ostringstream stream;
  stream.setf(std::ostringstream::fixed, std::ostringstream::floatfield);
  stream << std::setprecision(9);
  stream << "pos=";
  for (double position : buffer_positions) {
    stream << position << ",";
  }
  stream << ";masters=";
  for (const auto& cell_master : cell_masters) {
    stream << cell_master << ",";
  }
  return stream.str();
}

auto MakeUnitPatternKey(const icts::BufferingPattern& pattern) -> std::string
{
  return MakeUnitPatternKey(pattern.get_buffer_positions(), pattern.get_cell_masters());
}

auto EvalSurface(const FunctionalSurfaceModel& model, const std::array<double, 6>& coefficients, double input_slew_ns, double load_cap_pf)
    -> double
{
  const auto basis = MakeFitBasis(model.basis_kind, input_slew_ns, load_cap_pf);
  double value = 0.0;
  for (std::size_t term = 0U; term < FitBasisSize(model.basis_kind); ++term) {
    value += coefficients[term] * basis[term];
  }
  return value;
}

auto EvalFunctionalSurfaceModel(const FunctionalSurfaceModel& model, double input_slew_ns, double load_cap_pf) -> FunctionalSurfaceSample
{
  return FunctionalSurfaceSample{
      .output_slew_ns = EvalSurface(model, model.output_slew_coefficients, input_slew_ns, load_cap_pf),
      .driven_cap_pf = EvalSurface(model, model.driven_cap_coefficients, input_slew_ns, load_cap_pf),
      .delay_ns = EvalSurface(model, model.delay_coefficients, input_slew_ns, load_cap_pf),
      .power_w = EvalSurface(model, model.power_coefficients, input_slew_ns, load_cap_pf),
      .source_boundary_power_w = EvalSurface(model, model.source_boundary_power_coefficients, input_slew_ns, load_cap_pf),
  };
}

auto IsFinitePositive(double value) -> bool
{
  return std::isfinite(value) && value > 0.0;
}

auto BuildFunctionalSurfaceModels(const std::vector<icts::SegmentChar>& entries,
                                  const realtech_fixture::SegmentFrontierContext& segment_context, const realtech_fixture::CharGrid& grid,
                                  FitBasisKind basis_kind) -> std::unordered_map<std::string, FunctionalSurfaceModel>
{
  std::unordered_map<icts::PatternId, std::vector<const icts::SegmentChar*>> groups;
  for (const auto& entry : entries) {
    if (entry.get_length_idx() == 1U) {
      groups[entry.get_pattern_id()].push_back(&entry);
    }
  }

  std::unordered_map<std::string, FunctionalSurfaceModel> models;
  for (const auto& [pattern_id, group] : groups) {
    const auto pattern_it = segment_context.patterns.find(pattern_id);
    if (pattern_it == segment_context.patterns.end()) {
      continue;
    }

    const auto output_slew_coefficients = TryFitSurfaceCoefficients(
        group,
        [&grid](const icts::SegmentChar& entry) -> double { return static_cast<double>(entry.get_output_slew_idx()) * grid.slew_step_ns; },
        grid, basis_kind);
    const auto driven_cap_coefficients = TryFitSurfaceCoefficients(
        group,
        [&grid](const icts::SegmentChar& entry) -> double { return static_cast<double>(entry.get_driven_cap_idx()) * grid.cap_step_pf; },
        grid, basis_kind);
    const auto delay_coefficients
        = TryFitSurfaceCoefficients(group, [](const icts::SegmentChar& entry) -> double { return entry.get_delay(); }, grid, basis_kind);
    const auto power_coefficients
        = TryFitSurfaceCoefficients(group, [](const icts::SegmentChar& entry) -> double { return entry.get_power(); }, grid, basis_kind);
    const auto source_boundary_power_coefficients = TryFitSurfaceCoefficients(
        group, [](const icts::SegmentChar& entry) -> double { return entry.get_source_boundary_net_switch_power(); }, grid, basis_kind);
    if (!output_slew_coefficients.has_value() || !driven_cap_coefficients.has_value() || !delay_coefficients.has_value()
        || !power_coefficients.has_value() || !source_boundary_power_coefficients.has_value()) {
      continue;
    }

    const std::string unit_key = MakeUnitPatternKey(pattern_it->second);
    models[unit_key] = FunctionalSurfaceModel{
        .unit_key = unit_key,
        .basis_kind = basis_kind,
        .output_slew_coefficients = *output_slew_coefficients,
        .driven_cap_coefficients = *driven_cap_coefficients,
        .delay_coefficients = *delay_coefficients,
        .power_coefficients = *power_coefficients,
        .source_boundary_power_coefficients = *source_boundary_power_coefficients,
    };
  }

  return models;
}

auto BuildPhysicalStructuralCapOperators(const realtech_fixture::SegmentFrontierContext& segment_context,
                                         const icts::CharBuilder::Config& config, const realtech_fixture::CharGrid& grid)
    -> std::unordered_map<std::string, StructuralCapOperator>
{
  std::unordered_map<std::string, StructuralCapOperator> operators;
  if (!config.routing_layer.has_value() || *config.routing_layer <= 0) {
    return operators;
  }
  const int routing_layer = *config.routing_layer;
  for (const auto& [pattern_id, pattern] : segment_context.patterns) {
    (void) pattern_id;
    if (pattern.get_length_idx() != 1U) {
      continue;
    }

    const auto& buffer_positions = pattern.get_buffer_positions();
    const auto& cell_masters = pattern.get_cell_masters();
    if (buffer_positions.size() != cell_masters.size()) {
      continue;
    }

    const double unit_length_um = static_cast<double>(pattern.get_length_idx()) * grid.length_step_um;
    double alpha = 1.0;
    double eta_pf = icts_test::runtime::CurrentRuntime().wrapper.queryWireCapacitance(routing_layer, unit_length_um, config.wire_width_um);
    if (!cell_masters.empty()) {
      alpha = 0.0;
      const double first_buffer_position = buffer_positions.front();
      const double prewire_length_um = std::clamp(first_buffer_position, 0.0, 1.0) * unit_length_um;
      eta_pf = icts_test::runtime::CurrentRuntime().wrapper.queryCharInputPinCap(cell_masters.front());
      eta_pf += icts_test::runtime::CurrentRuntime().wrapper.queryWireCapacitance(routing_layer, prewire_length_um, config.wire_width_um);
    }

    if (!std::isfinite(eta_pf) || eta_pf < 0.0) {
      continue;
    }

    operators[MakeUnitPatternKey(pattern)] = StructuralCapOperator{
        .unit_key = MakeUnitPatternKey(pattern),
        .alpha = alpha,
        .eta_pf = eta_pf,
        .sample_count = 0U,
        .max_abs_residual_pf = 0.0,
    };
  }

  return operators;
}

auto DecomposeToUnitPatternKeys(const icts::BufferingPattern& pattern, unsigned target_length_idx)
    -> std::optional<std::vector<std::string>>
{
  if (target_length_idx == 0U) {
    return std::nullopt;
  }

  std::vector<std::vector<double>> slot_positions(target_length_idx);
  std::vector<std::vector<std::string>> slot_masters(target_length_idx);

  const auto& positions = pattern.get_buffer_positions();
  const auto& masters = pattern.get_cell_masters();
  if (positions.size() != masters.size()) {
    return std::nullopt;
  }

  constexpr double position_epsilon = 1e-7;
  for (std::size_t index = 0U; index < positions.size(); ++index) {
    const double scaled_position = positions.at(index) * static_cast<double>(target_length_idx);
    auto one_based_slot = static_cast<int>(std::ceil(scaled_position - position_epsilon));
    one_based_slot = std::clamp(one_based_slot, 1, static_cast<int>(target_length_idx));
    const auto slot_index = static_cast<unsigned>(one_based_slot - 1);
    double local_position = scaled_position - static_cast<double>(slot_index);
    if (std::abs(local_position - 1.0) <= position_epsilon) {
      local_position = 1.0;
    }
    if (local_position <= position_epsilon || local_position > 1.0 + position_epsilon) {
      return std::nullopt;
    }

    slot_positions.at(slot_index).push_back(local_position);
    slot_masters.at(slot_index).push_back(masters.at(index));
  }

  std::vector<std::string> unit_keys;
  unit_keys.reserve(target_length_idx);
  for (unsigned slot_index = 0U; slot_index < target_length_idx; ++slot_index) {
    unit_keys.push_back(MakeUnitPatternKey(slot_positions.at(slot_index), slot_masters.at(slot_index)));
  }
  return unit_keys;
}

auto PredictFunctionalCompose(const std::vector<const FunctionalSurfaceModel*>& unit_models, double input_slew_ns, double load_cap_pf,
                              double max_slew_ns, double max_cap_pf) -> FunctionalComposePrediction
{
  FunctionalComposePrediction prediction;
  const std::size_t unit_count = unit_models.size();
  if (unit_count == 0U || !IsFinitePositive(input_slew_ns) || !IsFinitePositive(load_cap_pf)) {
    return prediction;
  }

  std::vector<double> slews(unit_count + 1U, input_slew_ns);
  std::vector<double> load_caps(unit_count + 1U, load_cap_pf);
  load_caps.back() = load_cap_pf;

  constexpr unsigned max_iterations = 100U;
  constexpr double convergence_tolerance = 1e-12;
  for (unsigned iteration = 1U; iteration <= max_iterations; ++iteration) {
    slews.front() = input_slew_ns;
    bool finite_iteration = true;
    for (std::size_t unit_index = 0U; unit_index < unit_count; ++unit_index) {
      const auto response = EvalFunctionalSurfaceModel(*unit_models.at(unit_index), slews.at(unit_index), load_caps.at(unit_index + 1U));
      if (!std::isfinite(response.output_slew_ns)) {
        finite_iteration = false;
        break;
      }
      slews.at(unit_index + 1U) = response.output_slew_ns;
    }
    if (!finite_iteration) {
      return prediction;
    }

    auto next_load_caps = load_caps;
    next_load_caps.back() = load_cap_pf;
    for (std::size_t reverse_index = 0U; reverse_index < unit_count; ++reverse_index) {
      const std::size_t unit_index = unit_count - 1U - reverse_index;
      const auto response
          = EvalFunctionalSurfaceModel(*unit_models.at(unit_index), slews.at(unit_index), next_load_caps.at(unit_index + 1U));
      if (!std::isfinite(response.driven_cap_pf)) {
        return prediction;
      }
      next_load_caps.at(unit_index) = response.driven_cap_pf;
    }

    double residual = 0.0;
    for (std::size_t index = 0U; index < unit_count; ++index) {
      residual = std::max(residual, std::abs(next_load_caps.at(index) - load_caps.at(index)));
    }
    load_caps = std::move(next_load_caps);
    prediction.iterations = iteration;
    prediction.residual = residual;
    if (residual <= convergence_tolerance) {
      prediction.converged = true;
      break;
    }
  }

  if (!prediction.converged) {
    return prediction;
  }

  std::vector<FunctionalSurfaceSample> responses;
  responses.reserve(unit_count);
  slews.front() = input_slew_ns;
  for (std::size_t unit_index = 0U; unit_index < unit_count; ++unit_index) {
    const auto response = EvalFunctionalSurfaceModel(*unit_models.at(unit_index), slews.at(unit_index), load_caps.at(unit_index + 1U));
    if (!std::isfinite(response.output_slew_ns) || !std::isfinite(response.driven_cap_pf) || !std::isfinite(response.delay_ns)
        || !std::isfinite(response.power_w) || !std::isfinite(response.source_boundary_power_w)) {
      return prediction;
    }
    responses.push_back(response);
    slews.at(unit_index + 1U) = response.output_slew_ns;
  }

  prediction.output_slew_ns = responses.back().output_slew_ns;
  prediction.driven_cap_pf = responses.front().driven_cap_pf;
  prediction.source_boundary_power_w = responses.front().source_boundary_power_w;
  for (std::size_t unit_index = 0U; unit_index < unit_count; ++unit_index) {
    prediction.delay_ns += responses.at(unit_index).delay_ns;
    prediction.power_w += responses.at(unit_index).power_w;
    if (unit_index > 0U) {
      prediction.power_w -= responses.at(unit_index).source_boundary_power_w;
    }
  }

  prediction.is_valid = IsFinitePositive(prediction.output_slew_ns) && IsFinitePositive(prediction.driven_cap_pf)
                        && std::isfinite(prediction.delay_ns) && prediction.delay_ns >= 0.0 && std::isfinite(prediction.power_w)
                        && prediction.power_w >= 0.0 && std::isfinite(prediction.source_boundary_power_w)
                        && prediction.source_boundary_power_w >= 0.0;
  if (!prediction.is_valid) {
    return prediction;
  }

  for (double slew_ns : slews) {
    if (slew_ns <= 0.0 || slew_ns > max_slew_ns) {
      prediction.is_out_of_domain = true;
    }
  }
  for (double cap_pf : load_caps) {
    if (cap_pf <= 0.0 || cap_pf > max_cap_pf) {
      prediction.is_out_of_domain = true;
    }
  }
  return prediction;
}

auto PredictStructuralCapFunctionalCompose(const std::vector<const FunctionalSurfaceModel*>& unit_models,
                                           const std::vector<const StructuralCapOperator*>& cap_operators, double input_slew_ns,
                                           double load_cap_pf, double max_slew_ns, double max_cap_pf) -> FunctionalComposePrediction
{
  FunctionalComposePrediction prediction;
  const std::size_t unit_count = unit_models.size();
  if (unit_count == 0U || unit_count != cap_operators.size() || !IsFinitePositive(input_slew_ns) || !IsFinitePositive(load_cap_pf)) {
    return prediction;
  }

  std::vector<double> load_caps(unit_count + 1U, load_cap_pf);
  load_caps.back() = load_cap_pf;
  for (std::size_t reverse_index = 0U; reverse_index < unit_count; ++reverse_index) {
    const std::size_t unit_index = unit_count - 1U - reverse_index;
    const auto& cap_operator = *cap_operators.at(unit_index);
    load_caps.at(unit_index) = (cap_operator.alpha * load_caps.at(unit_index + 1U)) + cap_operator.eta_pf;
    if (!std::isfinite(load_caps.at(unit_index))) {
      return prediction;
    }
  }

  std::vector<FunctionalSurfaceSample> responses;
  responses.reserve(unit_count);
  std::vector<double> slews(unit_count + 1U, input_slew_ns);
  slews.front() = input_slew_ns;
  for (std::size_t unit_index = 0U; unit_index < unit_count; ++unit_index) {
    const auto response = EvalFunctionalSurfaceModel(*unit_models.at(unit_index), slews.at(unit_index), load_caps.at(unit_index + 1U));
    if (!std::isfinite(response.output_slew_ns) || !std::isfinite(response.delay_ns) || !std::isfinite(response.power_w)
        || !std::isfinite(response.source_boundary_power_w)) {
      return prediction;
    }
    responses.push_back(response);
    slews.at(unit_index + 1U) = response.output_slew_ns;
  }

  prediction.converged = true;
  prediction.iterations = 1U;
  prediction.residual = 0.0;
  prediction.output_slew_ns = responses.back().output_slew_ns;
  prediction.driven_cap_pf = load_caps.front();
  prediction.source_boundary_power_w = responses.front().source_boundary_power_w;
  for (std::size_t unit_index = 0U; unit_index < unit_count; ++unit_index) {
    prediction.delay_ns += responses.at(unit_index).delay_ns;
    prediction.power_w += responses.at(unit_index).power_w;
    if (unit_index > 0U) {
      prediction.power_w -= responses.at(unit_index).source_boundary_power_w;
    }
  }

  prediction.is_valid = IsFinitePositive(prediction.output_slew_ns) && IsFinitePositive(prediction.driven_cap_pf)
                        && std::isfinite(prediction.delay_ns) && prediction.delay_ns >= 0.0 && std::isfinite(prediction.power_w)
                        && prediction.power_w >= 0.0 && std::isfinite(prediction.source_boundary_power_w)
                        && prediction.source_boundary_power_w >= 0.0;
  if (!prediction.is_valid) {
    return prediction;
  }

  for (double slew_ns : slews) {
    if (slew_ns <= 0.0 || slew_ns > max_slew_ns) {
      prediction.is_out_of_domain = true;
    }
  }
  for (double cap_pf : load_caps) {
    if (cap_pf <= 0.0 || cap_pf > max_cap_pf) {
      prediction.is_out_of_domain = true;
    }
  }
  return prediction;
}

}  // namespace icts_test
