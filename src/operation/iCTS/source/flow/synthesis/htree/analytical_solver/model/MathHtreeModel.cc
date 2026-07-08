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
 * @file MathHtreeModel.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-25
 * @brief Mathematical H-tree slot-choice optimization model contract implementation.
 */

#include "synthesis/htree/analytical_solver/model/MathHtreeModel.hh"

#include <cmath>
#include <cstddef>
#include <string>

namespace icts::htree::analytical_solver {
namespace {

auto IsPositiveFinite(double value) -> bool
{
  return std::isfinite(value) && value > 0.0;
}

auto IsNonNegativeFinite(double value) -> bool
{
  return std::isfinite(value) && value >= 0.0;
}

}  // namespace

auto MathHtreeAffineFunction::evaluate(double input_slew_ns, double load_cap_pf) const -> double
{
  return constant + input_slew_coefficient * input_slew_ns + load_cap_coefficient * load_cap_pf;
}

auto MathHtreeDomain::isValid() const -> bool
{
  return IsPositiveFinite(input_slew_min_ns) && IsPositiveFinite(input_slew_max_ns) && input_slew_max_ns >= input_slew_min_ns
         && IsPositiveFinite(load_cap_min_pf) && IsPositiveFinite(load_cap_max_pf) && load_cap_max_pf >= load_cap_min_pf;
}

auto MathHtreeChoice::hasAnyBuffer() const -> bool
{
  return source_has_buffer || sink_has_buffer;
}

auto MathHtreeLevel::lastSlotIndex() const -> std::size_t
{
  return first_slot_index + slot_count - 1U;
}

auto MathHtreeProblem::isValid(std::string& failure_reason) const -> bool
{
  if (!IsPositiveFinite(root_input_slew_ns)) {
    failure_reason = "invalid_root_input_slew";
    return false;
  }
  if (!IsPositiveFinite(leaf_load_cap_pf)) {
    failure_reason = "invalid_leaf_load_cap";
    return false;
  }
  if (slots.empty()) {
    failure_reason = "empty_slot_sequence";
    return false;
  }
  if (levels.empty()) {
    failure_reason = "empty_level_sequence";
    return false;
  }
  std::size_t expected_first_slot_index = 0U;
  for (const auto& level : levels) {
    if (level.slot_count == 0U || level.first_slot_index != expected_first_slot_index || level.lastSlotIndex() >= slots.size()) {
      failure_reason = "invalid_level_slot_range";
      return false;
    }
    expected_first_slot_index += level.slot_count;
  }
  if (expected_first_slot_index != slots.size()) {
    failure_reason = "level_slot_range_mismatch";
    return false;
  }
  for (const auto& slot : slots) {
    if (slot.level_index >= levels.size()) {
      failure_reason = "invalid_slot_level_index";
      return false;
    }
    if (!IsPositiveFinite(slot.delay_weight) || !IsPositiveFinite(slot.power_weight)
        || !IsNonNegativeFinite(slot.source_boundary_power_weight)) {
      failure_reason = "invalid_slot_weight";
      return false;
    }
    if (!IsPositiveFinite(slot.downstream_cap_multiplier)) {
      failure_reason = "invalid_downstream_cap_multiplier";
      return false;
    }
    if (slot.choices.empty()) {
      failure_reason = "empty_slot_choice_set";
      return false;
    }
    for (const auto& choice : slot.choices) {
      if (!choice.domain.isValid()) {
        failure_reason = "invalid_choice_domain";
        return false;
      }
      if (!IsNonNegativeFinite(choice.max_output_slew_ns) && !std::isinf(choice.max_output_slew_ns)) {
        failure_reason = "invalid_choice_max_output_slew";
        return false;
      }
      if (!IsNonNegativeFinite(choice.max_source_cap_pf) && !std::isinf(choice.max_source_cap_pf)) {
        failure_reason = "invalid_choice_max_source_cap";
        return false;
      }
    }
  }
  return true;
}

auto MathHtreeSolution::hasUsableIntegerSolution() const -> bool
{
  return status == MathHtreeSolveStatus::kOptimal || status == MathHtreeSolveStatus::kFeasible
         || status == MathHtreeSolveStatus::kFeasibleWithGap;
}

auto ToString(MathHtreeSolveStatus status) -> std::string
{
  switch (status) {
    case MathHtreeSolveStatus::kNotSolved:
      return "not_solved";
    case MathHtreeSolveStatus::kOptimal:
      return "optimal";
    case MathHtreeSolveStatus::kFeasible:
      return "feasible";
    case MathHtreeSolveStatus::kFeasibleWithGap:
      return "feasible_with_gap";
    case MathHtreeSolveStatus::kTimeout:
      return "timeout";
    case MathHtreeSolveStatus::kInfeasible:
      return "infeasible";
    case MathHtreeSolveStatus::kUnbounded:
      return "unbounded";
    case MathHtreeSolveStatus::kAbnormal:
      return "abnormal";
    case MathHtreeSolveStatus::kModelInvalid:
      return "model_invalid";
    case MathHtreeSolveStatus::kSolverUnavailable:
      return "solver_unavailable";
  }
  return "unknown";
}

auto ToString(MathHtreeObjective objective) -> std::string
{
  switch (objective) {
    case MathHtreeObjective::kMinDelay:
      return "min_delay";
    case MathHtreeObjective::kMinPower:
      return "min_power";
    case MathHtreeObjective::kNormalizedDelayPower:
      return "normalized_delay_power";
  }
  return "unknown";
}

}  // namespace icts::htree::analytical_solver
