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
 * @file MathHtreeProblemBuilder.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-25
 * @brief Analytical H-tree solve-problem to mathematical slot-choice model adapter implementation.
 */

#include "synthesis/htree/analytical_solver/model/MathHtreeProblemBuilder.hh"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "BufferingPattern.hh"
#include "analytical_characterization/AnalyticalModel.hh"
#include "characterization/Characterization.hh"
#include "synthesis/htree/HTree.hh"
#include "synthesis/htree/analytical_solver/AnalyticalSolver.hh"
#include "synthesis/htree/analytical_solver/model/AnalyticalSolverModel.hh"
#include "synthesis/htree/constraint/Constraint.hh"
#include "synthesis/htree/topology_pruning/TopologyPruning.hh"

namespace icts::htree::analytical_solver {
namespace {

auto ConvertAffineFunction(const icts::analytical::AnalyticalSurfaceModel& model) -> MathHtreeAffineFunction
{
  MathHtreeAffineFunction function;
  if (model.coefficients.size() >= 3U) {
    function.constant = model.coefficients.at(0U);
    function.input_slew_coefficient = model.coefficients.at(1U) / model.domain.slewScale();
    function.load_cap_coefficient = model.coefficients.at(2U) / model.domain.capScale();
  }
  return function;
}

auto MergeDomain(const icts::analytical::AnalyticalModelSet& model_set) -> MathHtreeDomain
{
  MathHtreeDomain domain{
      .input_slew_min_ns = 0.0,
      .input_slew_max_ns = std::numeric_limits<double>::infinity(),
      .load_cap_min_pf = 0.0,
      .load_cap_max_pf = std::numeric_limits<double>::infinity(),
  };
  for (const auto metric :
       {icts::analytical::AnalyticalMetric::kOutputSlew, icts::analytical::AnalyticalMetric::kDelay,
        icts::analytical::AnalyticalMetric::kPower, icts::analytical::AnalyticalMetric::kSourceBoundaryNetSwitchPower}) {
    const auto* model = model_set.findMetric(metric);
    if (model == nullptr || !model->domain.isValid()) {
      continue;
    }
    domain.input_slew_min_ns = std::max(domain.input_slew_min_ns, model->domain.slew_min_ns);
    domain.input_slew_max_ns = std::min(domain.input_slew_max_ns, model->domain.slew_max_ns);
    domain.load_cap_min_pf = std::max(domain.load_cap_min_pf, model->domain.cap_min_pf);
    domain.load_cap_max_pf = std::min(domain.load_cap_max_pf, model->domain.cap_max_pf);
  }
  return domain;
}

auto IsAffineComplete(const icts::analytical::AnalyticalModelSet& model_set) -> bool
{
  if (!model_set.isComplete()) {
    return false;
  }
  for (const auto metric :
       {icts::analytical::AnalyticalMetric::kOutputSlew, icts::analytical::AnalyticalMetric::kDelay,
        icts::analytical::AnalyticalMetric::kPower, icts::analytical::AnalyticalMetric::kSourceBoundaryNetSwitchPower}) {
    const auto* model = model_set.findMetric(metric);
    if (model == nullptr || model->basis != icts::analytical::AnalyticalModelBasis::kAffine || !model->isValid()) {
      return false;
    }
  }
  return model_set.source_cap_operator.has_value() && model_set.source_cap_operator->isValid();
}

auto MakeChoice(const analytical_solver::UnitModelRef& unit_model) -> std::optional<MathHtreeChoice>
{
  const auto& model_set = *unit_model.model_set;
  const auto* output_slew_model = model_set.findMetric(icts::analytical::AnalyticalMetric::kOutputSlew);
  const auto* delay_model = model_set.findMetric(icts::analytical::AnalyticalMetric::kDelay);
  const auto* power_model = model_set.findMetric(icts::analytical::AnalyticalMetric::kPower);
  const auto* source_boundary_power_model = model_set.findMetric(icts::analytical::AnalyticalMetric::kSourceBoundaryNetSwitchPower);
  if (output_slew_model == nullptr || delay_model == nullptr || power_model == nullptr || source_boundary_power_model == nullptr
      || !model_set.source_cap_operator.has_value()) {
    return std::nullopt;
  }
  const auto& source_cap_operator = *model_set.source_cap_operator;
  MathHtreeChoice choice;
  choice.name = "unit_pattern_" + std::to_string(unit_model.pattern_id.local_id);
  choice.domain = MergeDomain(model_set);
  choice.output_slew_ns = ConvertAffineFunction(*output_slew_model);
  choice.delay_ns = ConvertAffineFunction(*delay_model);
  choice.power_w = ConvertAffineFunction(*power_model);
  choice.source_boundary_power_w = ConvertAffineFunction(*source_boundary_power_model);
  choice.source_cap_pf = MathHtreeAffineFunction{
      .constant = source_cap_operator.eta_pf,
      .input_slew_coefficient = 0.0,
      .load_cap_coefficient = source_cap_operator.alpha,
  };
  choice.source_has_buffer = unit_model.composition_state.monotonic_boundary_state.source.has_buffer;
  choice.source_strength_rank = unit_model.composition_state.monotonic_boundary_state.source.strength_rank;
  choice.sink_has_buffer = unit_model.composition_state.monotonic_boundary_state.sink.has_buffer;
  choice.sink_strength_rank = unit_model.composition_state.monotonic_boundary_state.sink.strength_rank;
  choice.terminal_branch_buffer = unit_model.composition_state.terminal_semantic == TerminalSemantic::kBranchBuffered;
  return choice;
}

auto MakeSlotName(std::size_t level_index, unsigned unit_index) -> std::string
{
  return "level_" + std::to_string(level_index) + "_unit_" + std::to_string(unit_index);
}

}  // namespace

auto BuildMathHtreeProblem(const analytical_solver::AnalyticalHTreeSolveProblem& solve_problem) -> MathHtreeProblemBuildResult
{
  MathHtreeProblemBuildResult result;
  const auto validation_failure = analytical_solver::ValidateSolveProblem(solve_problem);
  if (!validation_failure.empty()) {
    result.failure_reason = validation_failure;
    return result;
  }
  if (!solve_problem.config.use_functional_unit_compose) {
    result.failure_reason = "math_solver_requires_functional_unit_compose";
    return result;
  }

  const auto unit_models = analytical_solver::CollectUnitModelRefs(solve_problem);
  if (unit_models.empty()) {
    result.failure_reason = "empty_unit_model_catalog";
    return result;
  }

  std::vector<MathHtreeChoice> unit_choices;
  std::vector<MathHtreeSlotChoiceRef> unit_choice_refs;
  unit_choices.reserve(unit_models.size());
  unit_choice_refs.reserve(unit_models.size());
  for (const auto& unit_model : unit_models) {
    if (unit_model.model_set == nullptr || !IsAffineComplete(*unit_model.model_set)) {
      continue;
    }
    auto choice = MakeChoice(unit_model);
    if (!choice.has_value() || !choice->domain.isValid()) {
      continue;
    }
    unit_choices.push_back(std::move(*choice));
    unit_choice_refs.push_back(MathHtreeSlotChoiceRef{
        .unit_pattern_id = unit_model.pattern_id,
    });
  }
  if (unit_choices.empty()) {
    result.failure_reason = "empty_affine_unit_choice_set";
    return result;
  }

  result.problem.name = "analytical_htree_depth_" + std::to_string(solve_problem.levels->size());
  result.problem.root_input_slew_ns = analytical_solver::ResolveAnalyticalRootProbeSlewNs(solve_problem);
  result.problem.leaf_load_cap_pf = solve_problem.config.representative_leaf_load_cap_pf;
  result.problem.solve_time_limit_ms = 5000.0;
  result.problem.max_fanout = solve_problem.fanout_config.max_fanout;

  for (std::size_t level_index = 0U; level_index < solve_problem.levels->size(); ++level_index) {
    const auto& level = solve_problem.levels->at(level_index);
    if (solve_problem.config.unit_length_idx == 0U || level.aligned_length_idx == 0U
        || level.aligned_length_idx % solve_problem.config.unit_length_idx != 0U) {
      result.failure_reason = "level_length_not_unit_aligned";
      return result;
    }
    const unsigned unit_count = level.aligned_length_idx / solve_problem.config.unit_length_idx;
    const std::size_t first_slot_index = result.problem.slots.size();
    result.problem.levels.push_back(MathHtreeLevel{
        .name = "level_" + std::to_string(level_index),
        .first_slot_index = first_slot_index,
        .slot_count = unit_count,
        .is_leaf_level = level.is_leaf_level,
        .require_terminal_branch_buffer = solve_problem.boundary_constraints.force_branch_buffer,
    });
    result.level_slot_counts.push_back(unit_count);
    for (unsigned unit_index = 0U; unit_index < unit_count; ++unit_index) {
      MathHtreeSlot slot;
      slot.name = MakeSlotName(level_index, unit_index);
      slot.level_index = level_index;
      slot.delay_weight = 1.0;
      slot.power_weight = static_cast<double>(std::size_t{1U} << std::min<std::size_t>(level_index, sizeof(std::size_t) * 8U - 1U));
      slot.source_boundary_power_weight = unit_index == 0U && level_index == 0U ? 0.0 : slot.power_weight;
      slot.downstream_cap_multiplier = 1.0;
      if (unit_index + 1U == unit_count && level_index + 1U < solve_problem.levels->size()) {
        slot.downstream_cap_multiplier = 2.0;
      }
      slot.choices = unit_choices;
      result.problem.slots.push_back(std::move(slot));
      result.choice_refs_by_slot.push_back(unit_choice_refs);
    }
  }

  std::string problem_failure;
  if (!result.problem.isValid(problem_failure)) {
    result.failure_reason = problem_failure;
    return result;
  }
  result.success = true;
  return result;
}

}  // namespace icts::htree::analytical_solver
