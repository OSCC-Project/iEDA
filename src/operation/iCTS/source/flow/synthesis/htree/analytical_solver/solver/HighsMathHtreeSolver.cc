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
 * @file HighsMathHtreeSolver.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-26
 * @brief In-process HiGHS MILP backend implementation for the mathematical H-tree model.
 */

#include "synthesis/htree/analytical_solver/solver/HighsMathHtreeSolver.hh"

#include <stdint.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <ratio>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "Highs.h"
#include "lp_data/HConst.h"
#include "lp_data/HStruct.h"
#include "lp_data/HighsInfo.h"
#include "lp_data/HighsLp.h"
#include "lp_data/HighsStatus.h"
#include "model/HighsModel.h"
#include "util/HighsInt.h"
#include "util/HighsSparseMatrix.h"

namespace icts::htree::analytical_solver {
namespace {

constexpr double kAnchorFloor = 1e-30;
constexpr double kUsableAnchorGap = 0.25;
constexpr double kCoefficientFloor = 1e-30;
constexpr const char* kBackendName = "HiGHS in-process";

struct MathHtreeModelStats
{
  std::size_t binary_variable_count = 0U;
  std::size_t continuous_variable_count = 0U;
  std::size_t constraint_count = 0U;
};

struct LinearTerm
{
  HighsInt variable = 0;
  double coefficient = 0.0;
};

struct HighsVariable
{
  double cost = 0.0;
  double lower = 0.0;
  double upper = 0.0;
  HighsVarType type = HighsVarType::kContinuous;
};

struct HighsConstraint
{
  double lower = -kHighsInf;
  double upper = kHighsInf;
  std::vector<LinearTerm> terms;
};

class HighsMipModelBuilder
{
 public:
  auto addContinuousVariable(double lower, double upper, double cost) -> HighsInt
  {
    const auto index = static_cast<HighsInt>(_variables.size());
    _variables.push_back(HighsVariable{
        .cost = cost,
        .lower = lower,
        .upper = upper,
        .type = HighsVarType::kContinuous,
    });
    return index;
  }

  auto addBinaryVariable(double cost = 0.0) -> HighsInt
  {
    const auto index = static_cast<HighsInt>(_variables.size());
    _variables.push_back(HighsVariable{
        .cost = cost,
        .lower = 0.0,
        .upper = 1.0,
        .type = HighsVarType::kInteger,
    });
    return index;
  }

  auto addConstraint(double lower, double upper, std::vector<LinearTerm> terms) -> void
  {
    _constraints.push_back(HighsConstraint{
        .lower = lower,
        .upper = upper,
        .terms = std::move(terms),
    });
  }

  auto addEquality(std::vector<LinearTerm> terms, double rhs) -> void { addConstraint(rhs, rhs, std::move(terms)); }
  auto addLessOrEqual(std::vector<LinearTerm> terms, double rhs) -> void { addConstraint(-kHighsInf, rhs, std::move(terms)); }
  auto addGreaterOrEqual(std::vector<LinearTerm> terms, double rhs) -> void { addConstraint(rhs, kHighsInf, std::move(terms)); }

  auto variableCount() const -> std::size_t { return _variables.size(); }
  auto constraintCount() const -> std::size_t { return _constraints.size(); }

  auto buildModel() const -> HighsModel
  {
    HighsModel model;
    auto& lp = model.lp_;
    lp.num_col_ = static_cast<HighsInt>(_variables.size());
    lp.num_row_ = static_cast<HighsInt>(_constraints.size());
    lp.sense_ = ObjSense::kMinimize;
    lp.offset_ = 0.0;
    lp.col_cost_.reserve(_variables.size());
    lp.col_lower_.reserve(_variables.size());
    lp.col_upper_.reserve(_variables.size());
    lp.integrality_.reserve(_variables.size());
    for (const auto& variable : _variables) {
      lp.col_cost_.push_back(variable.cost);
      lp.col_lower_.push_back(variable.lower);
      lp.col_upper_.push_back(variable.upper);
      lp.integrality_.push_back(variable.type);
    }
    lp.row_lower_.reserve(_constraints.size());
    lp.row_upper_.reserve(_constraints.size());

    std::vector<std::vector<LinearTerm>> column_terms(_variables.size());
    for (std::size_t row_index = 0U; row_index < _constraints.size(); ++row_index) {
      const auto& constraint = _constraints.at(row_index);
      lp.row_lower_.push_back(constraint.lower);
      lp.row_upper_.push_back(constraint.upper);
      for (const auto& term : constraint.terms) {
        if (term.variable < 0 || static_cast<std::size_t>(term.variable) >= _variables.size()) {
          continue;
        }
        column_terms.at(static_cast<std::size_t>(term.variable))
            .push_back(LinearTerm{.variable = static_cast<HighsInt>(row_index), .coefficient = term.coefficient});
      }
    }

    auto& matrix = lp.a_matrix_;
    matrix.format_ = MatrixFormat::kColwise;
    matrix.start_.clear();
    matrix.index_.clear();
    matrix.value_.clear();
    matrix.p_end_.clear();
    matrix.start_.reserve(_variables.size() + 1U);
    matrix.start_.push_back(0);
    for (const auto& column : column_terms) {
      for (const auto& term : column) {
        matrix.index_.push_back(term.variable);
        matrix.value_.push_back(term.coefficient);
      }
      matrix.start_.push_back(static_cast<HighsInt>(matrix.index_.size()));
    }
    return model;
  }

 private:
  std::vector<HighsVariable> _variables;
  std::vector<HighsConstraint> _constraints;
};

enum class AffineVariableKind
{
  kEval,
  kActual,
};

struct ChoiceVariableSet
{
  HighsInt z = -1;
  HighsInt si = -1;
  HighsInt cl = -1;
  HighsInt se = -1;
  HighsInt ce = -1;
};

struct SlotVariableSet
{
  HighsInt s_in = -1;
  HighsInt s_eval = -1;
  HighsInt s_out = -1;
  HighsInt c_load = -1;
  HighsInt c_eval = -1;
  HighsInt c_src = -1;
  HighsInt d = -1;
  HighsInt p = -1;
  HighsInt pb = -1;
  HighsInt op = -1;
  std::vector<ChoiceVariableSet> choices;
};

struct BuiltHighsModel
{
  HighsModel model;
  MathHtreeModelStats stats;
  std::vector<SlotVariableSet> slots;
  std::vector<HighsInt> level_buffer_variables;
};

auto AppendTerm(std::vector<LinearTerm>& terms, HighsInt variable, double coefficient) -> void
{
  if (std::abs(coefficient) <= kCoefficientFloor) {
    return;
  }
  terms.push_back(LinearTerm{.variable = variable, .coefficient = coefficient});
}

auto BoundMax(const std::vector<double>& values) -> double
{
  double bound = 0.0;
  for (const double value : values) {
    if (std::isfinite(value)) {
      bound = std::max(bound, std::abs(value));
    }
  }
  return bound;
}

auto EvalRangeBound(const MathHtreeAffineFunction& function, const MathHtreeDomain& domain) -> double
{
  return BoundMax({
      function.evaluate(domain.input_slew_min_ns, domain.load_cap_min_pf),
      function.evaluate(domain.input_slew_min_ns, domain.load_cap_max_pf),
      function.evaluate(domain.input_slew_max_ns, domain.load_cap_min_pf),
      function.evaluate(domain.input_slew_max_ns, domain.load_cap_max_pf),
  });
}

auto ResolveGlobalInputSlewUpper(const MathHtreeProblem& problem) -> double
{
  double upper = problem.root_input_slew_ns;
  for (const auto& slot : problem.slots) {
    for (const auto& choice : slot.choices) {
      upper = std::max(upper, choice.domain.input_slew_max_ns);
      upper = std::max(upper, EvalRangeBound(choice.output_slew_ns, choice.domain));
      if (std::isfinite(choice.max_output_slew_ns)) {
        upper = std::max(upper, choice.max_output_slew_ns);
      }
    }
  }
  return std::max(upper, problem.root_input_slew_ns);
}

auto ResolveGlobalLoadCapUpper(const MathHtreeProblem& problem) -> double
{
  double upper = problem.leaf_load_cap_pf;
  for (const auto& slot : problem.slots) {
    for (const auto& choice : slot.choices) {
      upper = std::max(upper, choice.domain.load_cap_max_pf);
      upper = std::max(upper, EvalRangeBound(choice.source_cap_pf, choice.domain));
      if (std::isfinite(choice.max_source_cap_pf)) {
        upper = std::max(upper, choice.max_source_cap_pf);
      }
    }
  }
  return std::max(upper, problem.leaf_load_cap_pf);
}

auto ResolveGlobalMetricUpper(const MathHtreeProblem& problem, bool delay) -> double
{
  double upper = 0.0;
  for (const auto& slot : problem.slots) {
    double slot_upper = 0.0;
    for (const auto& choice : slot.choices) {
      const auto& metric = delay ? choice.delay_ns : choice.power_w;
      slot_upper = std::max(slot_upper, EvalRangeBound(metric, choice.domain));
      slot_upper = std::max(slot_upper, -metric.evaluate(choice.domain.input_slew_min_ns, choice.domain.load_cap_min_pf));
      slot_upper = std::max(slot_upper, -metric.evaluate(choice.domain.input_slew_min_ns, choice.domain.load_cap_max_pf));
      slot_upper = std::max(slot_upper, -metric.evaluate(choice.domain.input_slew_max_ns, choice.domain.load_cap_min_pf));
      slot_upper = std::max(slot_upper, -metric.evaluate(choice.domain.input_slew_max_ns, choice.domain.load_cap_max_pf));
      if (!delay) {
        slot_upper = std::max(slot_upper, EvalRangeBound(choice.source_boundary_power_w, choice.domain));
      }
    }
    upper = std::max(upper, slot_upper);
  }
  return std::max(upper, kAnchorFloor);
}

auto FloorLog2(std::size_t value) -> unsigned
{
  unsigned result = 0U;
  while (value > 1U) {
    value >>= 1U;
    ++result;
  }
  return result;
}

auto CountChoiceConstraints(const MathHtreeProblem& problem) -> std::size_t
{
  std::size_t constraint_count = 2U;
  for (const auto& slot : problem.slots) {
    constraint_count += 13U;
    constraint_count += slot.choices.size() * 8U;
    for (const auto& choice : slot.choices) {
      if (std::isfinite(choice.max_output_slew_ns)) {
        constraint_count += 1U;
      }
      if (std::isfinite(choice.max_source_cap_pf)) {
        constraint_count += 1U;
      }
    }
  }
  if (problem.slots.size() > 1U) {
    constraint_count += 2U * (problem.slots.size() - 1U);
  }
  return constraint_count;
}

auto CountExposedBoundaryCompatibilityConstraints(const MathHtreeProblem& problem) -> std::size_t
{
  std::size_t constraint_count = 0U;
  for (std::size_t upstream_slot_index = 0U; upstream_slot_index + 1U < problem.slots.size(); ++upstream_slot_index) {
    for (const auto& upstream_choice : problem.slots.at(upstream_slot_index).choices) {
      if (!upstream_choice.sink_has_buffer) {
        continue;
      }
      for (std::size_t downstream_slot_index = upstream_slot_index + 1U; downstream_slot_index < problem.slots.size();
           ++downstream_slot_index) {
        for (const auto& downstream_choice : problem.slots.at(downstream_slot_index).choices) {
          if (!downstream_choice.source_has_buffer || upstream_choice.sink_strength_rank >= downstream_choice.source_strength_rank) {
            continue;
          }
          ++constraint_count;
        }
      }
    }
  }
  return constraint_count;
}

auto CountLevelLegalityConstraints(const MathHtreeProblem& problem) -> std::size_t
{
  std::size_t constraint_count = 0U;
  for (const auto& level : problem.levels) {
    constraint_count += 1U;
    for (std::size_t slot_offset = 0U; slot_offset < level.slot_count; ++slot_offset) {
      const auto& slot = problem.slots.at(level.first_slot_index + slot_offset);
      for (const auto& choice : slot.choices) {
        if (choice.hasAnyBuffer()) {
          ++constraint_count;
        }
      }
    }
    if (level.is_leaf_level && problem.max_fanout > 0U) {
      ++constraint_count;
    }
    if (level.require_terminal_branch_buffer) {
      ++constraint_count;
    }
  }
  if (problem.max_fanout > 1U && !problem.levels.empty()) {
    const unsigned max_bufferless_run = FloorLog2(problem.max_fanout) - 1U;
    const std::size_t window_size = static_cast<std::size_t>(max_bufferless_run) + 1U;
    if (window_size <= problem.levels.size()) {
      constraint_count += problem.levels.size() - window_size + 1U;
    }
  }
  return constraint_count;
}

auto MakeModelStats(const MathHtreeProblem& problem) -> MathHtreeModelStats
{
  MathHtreeModelStats stats;
  stats.continuous_variable_count = problem.slots.size() * 10U;
  for (const auto& slot : problem.slots) {
    stats.binary_variable_count += slot.choices.size();
    stats.continuous_variable_count += slot.choices.size() * 4U;
  }
  stats.binary_variable_count += problem.levels.size();
  stats.constraint_count
      = CountChoiceConstraints(problem) + CountExposedBoundaryCompatibilityConstraints(problem) + CountLevelLegalityConstraints(problem);
  return stats;
}

auto SubtractAffineFunction(const MathHtreeAffineFunction& lhs, const MathHtreeAffineFunction& rhs, double rhs_scale)
    -> MathHtreeAffineFunction
{
  return MathHtreeAffineFunction{
      .constant = lhs.constant - rhs_scale * rhs.constant,
      .input_slew_coefficient = lhs.input_slew_coefficient - rhs_scale * rhs.input_slew_coefficient,
      .load_cap_coefficient = lhs.load_cap_coefficient - rhs_scale * rhs.load_cap_coefficient,
  };
}

auto AppendChoiceAffineTerms(std::vector<LinearTerm>& terms, const MathHtreeAffineFunction& function, const ChoiceVariableSet& choice_vars,
                             AffineVariableKind variable_kind, double scale) -> void
{
  const HighsInt slew_variable = variable_kind == AffineVariableKind::kEval ? choice_vars.se : choice_vars.si;
  const HighsInt cap_variable = variable_kind == AffineVariableKind::kEval ? choice_vars.ce : choice_vars.cl;
  AppendTerm(terms, slew_variable, scale * function.input_slew_coefficient);
  AppendTerm(terms, cap_variable, scale * function.load_cap_coefficient);
  AppendTerm(terms, choice_vars.z, scale * function.constant);
}

auto AddLocalSumEquality(HighsMipModelBuilder& builder, HighsInt total_variable, const std::vector<ChoiceVariableSet>& choice_vars,
                         HighsInt ChoiceVariableSet::* local_member) -> void
{
  std::vector<LinearTerm> terms;
  terms.reserve(choice_vars.size() + 1U);
  AppendTerm(terms, total_variable, 1.0);
  for (const auto& choice_var : choice_vars) {
    AppendTerm(terms, choice_var.*local_member, -1.0);
  }
  builder.addEquality(std::move(terms), 0.0);
}

auto AddSelectedContributionRange(HighsMipModelBuilder& builder, HighsInt value_variable, HighsInt selected_variable, double lower,
                                  double upper) -> void
{
  std::vector<LinearTerm> lower_terms;
  lower_terms.reserve(2U);
  AppendTerm(lower_terms, value_variable, 1.0);
  AppendTerm(lower_terms, selected_variable, -lower);
  builder.addGreaterOrEqual(std::move(lower_terms), 0.0);

  std::vector<LinearTerm> upper_terms;
  upper_terms.reserve(2U);
  AppendTerm(upper_terms, value_variable, 1.0);
  AppendTerm(upper_terms, selected_variable, -upper);
  builder.addLessOrEqual(std::move(upper_terms), 0.0);
}

auto AddChoiceAffineSumEquality(HighsMipModelBuilder& builder, HighsInt value_variable, const MathHtreeSlot& slot,
                                const SlotVariableSet& slot_vars, MathHtreeAffineFunction MathHtreeChoice::* function_member,
                                AffineVariableKind variable_kind) -> void
{
  std::vector<LinearTerm> terms;
  terms.reserve(slot.choices.size() * 3U + 1U);
  AppendTerm(terms, value_variable, 1.0);
  for (std::size_t choice_index = 0U; choice_index < slot.choices.size(); ++choice_index) {
    AppendChoiceAffineTerms(terms, slot.choices.at(choice_index).*function_member, slot_vars.choices.at(choice_index), variable_kind, -1.0);
  }
  builder.addEquality(std::move(terms), 0.0);
}

auto AddOwnedPowerEquality(HighsMipModelBuilder& builder, const MathHtreeSlot& slot, const SlotVariableSet& slot_vars) -> void
{
  std::vector<LinearTerm> terms;
  terms.reserve(slot.choices.size() * 3U + 1U);
  AppendTerm(terms, slot_vars.op, 1.0);
  const double source_boundary_ratio = slot.power_weight <= 0.0 ? 0.0 : slot.source_boundary_power_weight / slot.power_weight;
  for (std::size_t choice_index = 0U; choice_index < slot.choices.size(); ++choice_index) {
    const auto& choice = slot.choices.at(choice_index);
    AppendChoiceAffineTerms(terms, SubtractAffineFunction(choice.power_w, choice.source_boundary_power_w, source_boundary_ratio),
                            slot_vars.choices.at(choice_index), AffineVariableKind::kEval, -1.0);
  }
  builder.addEquality(std::move(terms), 0.0);
}

auto AddChoiceAffineUpper(HighsMipModelBuilder& builder, const MathHtreeAffineFunction& function, const ChoiceVariableSet& choice_vars,
                          AffineVariableKind variable_kind, double upper) -> void
{
  std::vector<LinearTerm> terms;
  terms.reserve(4U);
  AppendChoiceAffineTerms(terms, function, choice_vars, variable_kind, 1.0);
  AppendTerm(terms, choice_vars.z, -upper);
  builder.addLessOrEqual(std::move(terms), 0.0);
}

auto AddChoiceConstraints(HighsMipModelBuilder& builder, const MathHtreeProblem& problem, const BuiltHighsModel& built_model) -> void
{
  std::vector<LinearTerm> root_terms;
  AppendTerm(root_terms, built_model.slots.front().s_in, 1.0);
  builder.addEquality(std::move(root_terms), problem.root_input_slew_ns);

  std::vector<LinearTerm> leaf_terms;
  AppendTerm(leaf_terms, built_model.slots.back().c_load, 1.0);
  builder.addEquality(std::move(leaf_terms), problem.leaf_load_cap_pf);

  for (std::size_t slot_index = 0U; slot_index < problem.slots.size(); ++slot_index) {
    const auto& slot = problem.slots.at(slot_index);
    const auto& slot_vars = built_model.slots.at(slot_index);

    std::vector<LinearTerm> choice_sum_terms;
    choice_sum_terms.reserve(slot.choices.size());
    for (const auto& choice_var : slot_vars.choices) {
      AppendTerm(choice_sum_terms, choice_var.z, 1.0);
    }
    builder.addEquality(std::move(choice_sum_terms), 1.0);

    AddLocalSumEquality(builder, slot_vars.s_in, slot_vars.choices, &ChoiceVariableSet::si);
    AddLocalSumEquality(builder, slot_vars.c_load, slot_vars.choices, &ChoiceVariableSet::cl);
    AddLocalSumEquality(builder, slot_vars.s_eval, slot_vars.choices, &ChoiceVariableSet::se);
    AddLocalSumEquality(builder, slot_vars.c_eval, slot_vars.choices, &ChoiceVariableSet::ce);

    std::vector<LinearTerm> slew_eval_floor_terms;
    AppendTerm(slew_eval_floor_terms, slot_vars.s_eval, 1.0);
    AppendTerm(slew_eval_floor_terms, slot_vars.s_in, -1.0);
    builder.addGreaterOrEqual(std::move(slew_eval_floor_terms), 0.0);

    std::vector<LinearTerm> cap_eval_floor_terms;
    AppendTerm(cap_eval_floor_terms, slot_vars.c_eval, 1.0);
    AppendTerm(cap_eval_floor_terms, slot_vars.c_load, -1.0);
    builder.addGreaterOrEqual(std::move(cap_eval_floor_terms), 0.0);

    AddChoiceAffineSumEquality(builder, slot_vars.s_out, slot, slot_vars, &MathHtreeChoice::output_slew_ns, AffineVariableKind::kEval);
    AddChoiceAffineSumEquality(builder, slot_vars.d, slot, slot_vars, &MathHtreeChoice::delay_ns, AffineVariableKind::kEval);
    AddChoiceAffineSumEquality(builder, slot_vars.p, slot, slot_vars, &MathHtreeChoice::power_w, AffineVariableKind::kEval);
    AddChoiceAffineSumEquality(builder, slot_vars.pb, slot, slot_vars, &MathHtreeChoice::source_boundary_power_w,
                               AffineVariableKind::kEval);
    AddOwnedPowerEquality(builder, slot, slot_vars);
    AddChoiceAffineSumEquality(builder, slot_vars.c_src, slot, slot_vars, &MathHtreeChoice::source_cap_pf, AffineVariableKind::kActual);

    for (std::size_t choice_index = 0U; choice_index < slot.choices.size(); ++choice_index) {
      const auto& choice = slot.choices.at(choice_index);
      const auto& choice_vars = slot_vars.choices.at(choice_index);
      AddSelectedContributionRange(builder, choice_vars.si, choice_vars.z, 0.0, choice.domain.input_slew_max_ns);
      AddSelectedContributionRange(builder, choice_vars.cl, choice_vars.z, 0.0, choice.domain.load_cap_max_pf);
      AddSelectedContributionRange(builder, choice_vars.se, choice_vars.z, choice.domain.input_slew_min_ns,
                                   choice.domain.input_slew_max_ns);
      AddSelectedContributionRange(builder, choice_vars.ce, choice_vars.z, choice.domain.load_cap_min_pf, choice.domain.load_cap_max_pf);

      std::vector<LinearTerm> eval_slew_actual_terms;
      AppendTerm(eval_slew_actual_terms, choice_vars.se, 1.0);
      AppendTerm(eval_slew_actual_terms, choice_vars.si, -1.0);
      builder.addGreaterOrEqual(std::move(eval_slew_actual_terms), 0.0);

      std::vector<LinearTerm> eval_cap_actual_terms;
      AppendTerm(eval_cap_actual_terms, choice_vars.ce, 1.0);
      AppendTerm(eval_cap_actual_terms, choice_vars.cl, -1.0);
      builder.addGreaterOrEqual(std::move(eval_cap_actual_terms), 0.0);

      if (std::isfinite(choice.max_output_slew_ns)) {
        AddChoiceAffineUpper(builder, choice.output_slew_ns, choice_vars, AffineVariableKind::kEval, choice.max_output_slew_ns);
      }
      if (std::isfinite(choice.max_source_cap_pf)) {
        AddChoiceAffineUpper(builder, choice.source_cap_pf, choice_vars, AffineVariableKind::kActual, choice.max_source_cap_pf);
      }
    }

    if (slot_index + 1U < problem.slots.size()) {
      const auto& next_slot_vars = built_model.slots.at(slot_index + 1U);
      std::vector<LinearTerm> slew_continuity_terms;
      AppendTerm(slew_continuity_terms, next_slot_vars.s_in, 1.0);
      AppendTerm(slew_continuity_terms, slot_vars.s_out, -1.0);
      builder.addEquality(std::move(slew_continuity_terms), 0.0);

      std::vector<LinearTerm> cap_continuity_terms;
      AppendTerm(cap_continuity_terms, slot_vars.c_load, 1.0);
      AppendTerm(cap_continuity_terms, next_slot_vars.c_src, -slot.downstream_cap_multiplier);
      builder.addEquality(std::move(cap_continuity_terms), 0.0);
    }
  }
}

auto AddExposedBoundaryCompatibilityConstraints(HighsMipModelBuilder& builder, const MathHtreeProblem& problem,
                                                const BuiltHighsModel& built_model) -> void
{
  for (std::size_t upstream_slot_index = 0U; upstream_slot_index + 1U < problem.slots.size(); ++upstream_slot_index) {
    const auto& upstream_slot = problem.slots.at(upstream_slot_index);
    for (std::size_t upstream_choice_index = 0U; upstream_choice_index < upstream_slot.choices.size(); ++upstream_choice_index) {
      const auto& upstream_choice = upstream_slot.choices.at(upstream_choice_index);
      if (!upstream_choice.sink_has_buffer) {
        continue;
      }
      for (std::size_t downstream_slot_index = upstream_slot_index + 1U; downstream_slot_index < problem.slots.size();
           ++downstream_slot_index) {
        const auto& downstream_slot = problem.slots.at(downstream_slot_index);
        for (std::size_t downstream_choice_index = 0U; downstream_choice_index < downstream_slot.choices.size();
             ++downstream_choice_index) {
          const auto& downstream_choice = downstream_slot.choices.at(downstream_choice_index);
          if (!downstream_choice.source_has_buffer || upstream_choice.sink_strength_rank >= downstream_choice.source_strength_rank) {
            continue;
          }

          std::vector<LinearTerm> terms;
          AppendTerm(terms, built_model.slots.at(upstream_slot_index).choices.at(upstream_choice_index).z, 1.0);
          AppendTerm(terms, built_model.slots.at(downstream_slot_index).choices.at(downstream_choice_index).z, 1.0);
          for (std::size_t intermediate_slot_index = upstream_slot_index + 1U; intermediate_slot_index < downstream_slot_index;
               ++intermediate_slot_index) {
            const auto& intermediate_slot = problem.slots.at(intermediate_slot_index);
            for (std::size_t intermediate_choice_index = 0U; intermediate_choice_index < intermediate_slot.choices.size();
                 ++intermediate_choice_index) {
              if (intermediate_slot.choices.at(intermediate_choice_index).sink_has_buffer) {
                AppendTerm(terms, built_model.slots.at(intermediate_slot_index).choices.at(intermediate_choice_index).z, -1.0);
              }
            }
          }
          builder.addLessOrEqual(std::move(terms), 1.0);
        }
      }
    }
  }
}

auto HasBufferChoice(const MathHtreeLevel& level, const MathHtreeProblem& problem) -> bool
{
  for (std::size_t slot_offset = 0U; slot_offset < level.slot_count; ++slot_offset) {
    const auto& slot = problem.slots.at(level.first_slot_index + slot_offset);
    for (const auto& choice : slot.choices) {
      if (choice.hasAnyBuffer()) {
        return true;
      }
    }
  }
  return false;
}

auto HasTerminalBranchChoice(const MathHtreeSlot& slot) -> bool
{
  for (const auto& choice : slot.choices) {
    if (choice.terminal_branch_buffer) {
      return true;
    }
  }
  return false;
}

auto AddImpossibleBinaryConstraint(HighsMipModelBuilder& builder, HighsInt binary_variable) -> void
{
  std::vector<LinearTerm> terms;
  AppendTerm(terms, binary_variable, 1.0);
  builder.addEquality(std::move(terms), 2.0);
}

auto AddLevelLegalityConstraints(HighsMipModelBuilder& builder, const MathHtreeProblem& problem, const BuiltHighsModel& built_model) -> void
{
  for (std::size_t level_index = 0U; level_index < problem.levels.size(); ++level_index) {
    const auto& level = problem.levels.at(level_index);
    const HighsInt level_buffer_var = built_model.level_buffer_variables.at(level_index);
    std::vector<LinearTerm> upper_terms;
    AppendTerm(upper_terms, level_buffer_var, 1.0);
    for (std::size_t slot_offset = 0U; slot_offset < level.slot_count; ++slot_offset) {
      const std::size_t slot_index = level.first_slot_index + slot_offset;
      const auto& slot = problem.slots.at(slot_index);
      for (std::size_t choice_index = 0U; choice_index < slot.choices.size(); ++choice_index) {
        if (slot.choices.at(choice_index).hasAnyBuffer()) {
          AppendTerm(upper_terms, built_model.slots.at(slot_index).choices.at(choice_index).z, -1.0);
        }
      }
    }
    builder.addLessOrEqual(std::move(upper_terms), 0.0);

    for (std::size_t slot_offset = 0U; slot_offset < level.slot_count; ++slot_offset) {
      const std::size_t slot_index = level.first_slot_index + slot_offset;
      const auto& slot = problem.slots.at(slot_index);
      for (std::size_t choice_index = 0U; choice_index < slot.choices.size(); ++choice_index) {
        if (!slot.choices.at(choice_index).hasAnyBuffer()) {
          continue;
        }
        std::vector<LinearTerm> lower_terms;
        AppendTerm(lower_terms, built_model.slots.at(slot_index).choices.at(choice_index).z, 1.0);
        AppendTerm(lower_terms, level_buffer_var, -1.0);
        builder.addLessOrEqual(std::move(lower_terms), 0.0);
      }
    }

    if (level.is_leaf_level && problem.max_fanout > 0U) {
      if (HasBufferChoice(level, problem)) {
        std::vector<LinearTerm> terms;
        AppendTerm(terms, level_buffer_var, 1.0);
        builder.addEquality(std::move(terms), 1.0);
      } else {
        AddImpossibleBinaryConstraint(builder, level_buffer_var);
      }
    }

    if (level.require_terminal_branch_buffer) {
      const std::size_t last_slot_index = level.lastSlotIndex();
      if (!HasTerminalBranchChoice(problem.slots.at(last_slot_index))) {
        AddImpossibleBinaryConstraint(builder, level_buffer_var);
        continue;
      }
      std::vector<LinearTerm> terminal_terms;
      for (std::size_t choice_index = 0U; choice_index < problem.slots.at(last_slot_index).choices.size(); ++choice_index) {
        if (problem.slots.at(last_slot_index).choices.at(choice_index).terminal_branch_buffer) {
          AppendTerm(terminal_terms, built_model.slots.at(last_slot_index).choices.at(choice_index).z, 1.0);
        }
      }
      builder.addEquality(std::move(terminal_terms), 1.0);
    }
  }

  if (problem.max_fanout <= 1U || problem.levels.empty()) {
    return;
  }
  const unsigned max_bufferless_run = FloorLog2(problem.max_fanout) - 1U;
  const std::size_t window_size = static_cast<std::size_t>(max_bufferless_run) + 1U;
  for (std::size_t first_level_index = 0U; first_level_index + window_size <= problem.levels.size(); ++first_level_index) {
    std::vector<LinearTerm> terms;
    terms.reserve(window_size);
    for (std::size_t offset = 0U; offset < window_size; ++offset) {
      AppendTerm(terms, built_model.level_buffer_variables.at(first_level_index + offset), 1.0);
    }
    builder.addGreaterOrEqual(std::move(terms), 1.0);
  }
}

auto ObjectiveCostForDelay(const MathHtreeProblem& problem, const MathHtreeSlot& slot, MathHtreeObjective objective) -> double
{
  if (objective == MathHtreeObjective::kMinDelay) {
    return slot.delay_weight;
  }
  if (objective == MathHtreeObjective::kNormalizedDelayPower) {
    return slot.delay_weight / std::max(problem.min_delay_anchor_ns, kAnchorFloor);
  }
  return 0.0;
}

auto ObjectiveCostForPower(const MathHtreeProblem& problem, const MathHtreeSlot& slot, MathHtreeObjective objective) -> double
{
  if (objective == MathHtreeObjective::kMinPower) {
    return slot.power_weight;
  }
  if (objective == MathHtreeObjective::kNormalizedDelayPower) {
    return slot.power_weight / std::max(problem.min_power_anchor_w, kAnchorFloor);
  }
  return 0.0;
}

auto BuildHighsModel(const MathHtreeProblem& problem, MathHtreeObjective objective) -> BuiltHighsModel
{
  const double input_slew_upper = ResolveGlobalInputSlewUpper(problem);
  const double load_cap_upper = ResolveGlobalLoadCapUpper(problem);
  const double delay_upper = ResolveGlobalMetricUpper(problem, true);
  const double power_upper = ResolveGlobalMetricUpper(problem, false);

  HighsMipModelBuilder builder;
  BuiltHighsModel built_model;
  built_model.stats = MakeModelStats(problem);
  built_model.slots.resize(problem.slots.size());

  for (std::size_t slot_index = 0U; slot_index < problem.slots.size(); ++slot_index) {
    const auto& slot = problem.slots.at(slot_index);
    auto& slot_vars = built_model.slots.at(slot_index);
    slot_vars.s_in = builder.addContinuousVariable(0.0, input_slew_upper, 0.0);
    slot_vars.s_eval = builder.addContinuousVariable(0.0, input_slew_upper, 0.0);
    slot_vars.s_out = builder.addContinuousVariable(0.0, input_slew_upper, 0.0);
    slot_vars.c_load = builder.addContinuousVariable(0.0, load_cap_upper, 0.0);
    slot_vars.c_eval = builder.addContinuousVariable(0.0, load_cap_upper, 0.0);
    slot_vars.c_src = builder.addContinuousVariable(0.0, load_cap_upper, 0.0);
    slot_vars.d = builder.addContinuousVariable(0.0, delay_upper, ObjectiveCostForDelay(problem, slot, objective));
    slot_vars.p = builder.addContinuousVariable(0.0, power_upper, 0.0);
    slot_vars.pb = builder.addContinuousVariable(0.0, power_upper, 0.0);
    slot_vars.op = builder.addContinuousVariable(0.0, power_upper, ObjectiveCostForPower(problem, slot, objective));

    slot_vars.choices.resize(slot.choices.size());
    for (auto& choice_vars : slot_vars.choices) {
      choice_vars.z = builder.addBinaryVariable();
      choice_vars.si = builder.addContinuousVariable(0.0, input_slew_upper, 0.0);
      choice_vars.cl = builder.addContinuousVariable(0.0, load_cap_upper, 0.0);
      choice_vars.se = builder.addContinuousVariable(0.0, input_slew_upper, 0.0);
      choice_vars.ce = builder.addContinuousVariable(0.0, load_cap_upper, 0.0);
    }
  }

  built_model.level_buffer_variables.reserve(problem.levels.size());
  for (std::size_t level_index = 0U; level_index < problem.levels.size(); ++level_index) {
    built_model.level_buffer_variables.push_back(builder.addBinaryVariable());
  }

  AddChoiceConstraints(builder, problem, built_model);
  AddExposedBoundaryCompatibilityConstraints(builder, problem, built_model);
  AddLevelLegalityConstraints(builder, problem, built_model);
  built_model.model = builder.buildModel();
  built_model.stats.constraint_count = builder.constraintCount();
  return built_model;
}

auto MakeStatusSolution(MathHtreeSolveStatus status, std::string failure_reason) -> MathHtreeSolution
{
  MathHtreeSolution solution;
  solution.status = status;
  solution.failure_reason = std::move(failure_reason);
  solution.backend_name = kBackendName;
  return solution;
}

auto HasHighsIncumbent(const HighsInfo& info, const HighsSolution& highs_solution, std::size_t variable_count) -> bool
{
  return info.primal_solution_status == kSolutionStatusFeasible && highs_solution.col_value.size() >= variable_count;
}

auto MapHighsModelStatus(HighsModelStatus status, bool has_incumbent) -> MathHtreeSolveStatus
{
  switch (status) {
    case HighsModelStatus::kOptimal:
      return MathHtreeSolveStatus::kOptimal;
    case HighsModelStatus::kInfeasible:
      return MathHtreeSolveStatus::kInfeasible;
    case HighsModelStatus::kUnbounded:
    case HighsModelStatus::kUnboundedOrInfeasible:
      return MathHtreeSolveStatus::kUnbounded;
    case HighsModelStatus::kTimeLimit:
    case HighsModelStatus::kIterationLimit:
    case HighsModelStatus::kSolutionLimit:
    case HighsModelStatus::kInterrupt:
      return has_incumbent ? MathHtreeSolveStatus::kFeasibleWithGap : MathHtreeSolveStatus::kTimeout;
    case HighsModelStatus::kObjectiveBound:
    case HighsModelStatus::kObjectiveTarget:
    case HighsModelStatus::kUnknown:
      return has_incumbent ? MathHtreeSolveStatus::kFeasible : MathHtreeSolveStatus::kAbnormal;
    case HighsModelStatus::kNotset:
    case HighsModelStatus::kLoadError:
    case HighsModelStatus::kModelError:
    case HighsModelStatus::kPresolveError:
    case HighsModelStatus::kSolveError:
    case HighsModelStatus::kPostsolveError:
    case HighsModelStatus::kModelEmpty:
    case HighsModelStatus::kMemoryLimit:
      return MathHtreeSolveStatus::kAbnormal;
  }
  return MathHtreeSolveStatus::kAbnormal;
}

auto SelectedChoiceIndex(const HighsSolution& highs_solution, const SlotVariableSet& slot_vars) -> std::size_t
{
  std::size_t selected_index = 0U;
  double selected_value = -1.0;
  for (std::size_t choice_index = 0U; choice_index < slot_vars.choices.size(); ++choice_index) {
    const auto variable_index = static_cast<std::size_t>(slot_vars.choices.at(choice_index).z);
    const double value = variable_index < highs_solution.col_value.size() ? highs_solution.col_value.at(variable_index) : 0.0;
    if (value > selected_value) {
      selected_value = value;
      selected_index = choice_index;
    }
  }
  return selected_index;
}

auto ValueOrZero(const HighsSolution& highs_solution, HighsInt variable) -> double
{
  if (variable < 0 || static_cast<std::size_t>(variable) >= highs_solution.col_value.size()) {
    return 0.0;
  }
  return highs_solution.col_value.at(static_cast<std::size_t>(variable));
}

auto CollectSlotSolutions(const MathHtreeProblem& problem, const BuiltHighsModel& built_model, const HighsSolution& highs_solution,
                          MathHtreeSolution& solution) -> void
{
  solution.slots.reserve(problem.slots.size());
  for (std::size_t slot_index = 0U; slot_index < problem.slots.size(); ++slot_index) {
    const auto& slot_vars = built_model.slots.at(slot_index);
    MathHtreeSlotSolution slot_solution;
    slot_solution.selected_choice_index = SelectedChoiceIndex(highs_solution, slot_vars);
    slot_solution.input_slew_ns = ValueOrZero(highs_solution, slot_vars.s_in);
    slot_solution.output_slew_ns = ValueOrZero(highs_solution, slot_vars.s_out);
    slot_solution.load_cap_pf = ValueOrZero(highs_solution, slot_vars.c_load);
    slot_solution.source_cap_pf = ValueOrZero(highs_solution, slot_vars.c_src);
    slot_solution.delay_ns = ValueOrZero(highs_solution, slot_vars.d);
    slot_solution.power_w = ValueOrZero(highs_solution, slot_vars.p);
    slot_solution.source_boundary_power_w = ValueOrZero(highs_solution, slot_vars.pb);
    solution.total_delay_ns += problem.slots.at(slot_index).delay_weight * slot_solution.delay_ns;
    solution.total_power_w += problem.slots.at(slot_index).power_weight * slot_solution.power_w
                              - problem.slots.at(slot_index).source_boundary_power_weight * slot_solution.source_boundary_power_w;
    solution.slots.push_back(slot_solution);
  }
}

auto ApplyHighsOptions(Highs& highs, double time_limit_ms) -> void
{
  static_cast<void>(highs.setOptionValue("output_flag", false));
  static_cast<void>(highs.setOptionValue("log_to_console", false));
  if (time_limit_ms > 0.0 && std::isfinite(time_limit_ms)) {
    static_cast<void>(highs.setOptionValue("time_limit", time_limit_ms / 1000.0));
  }
}

auto RunHighs(const MathHtreeProblem& problem, MathHtreeObjective objective) -> MathHtreeSolution
{
  std::string failure_reason;
  if (!problem.isValid(failure_reason)) {
    return MakeStatusSolution(MathHtreeSolveStatus::kModelInvalid, failure_reason);
  }
  if (objective == MathHtreeObjective::kNormalizedDelayPower
      && (problem.min_delay_anchor_ns <= kAnchorFloor || problem.min_power_anchor_w <= kAnchorFloor)) {
    return MakeStatusSolution(MathHtreeSolveStatus::kModelInvalid, "invalid_normalized_objective_anchor");
  }

  auto built_model = BuildHighsModel(problem, objective);
  MathHtreeSolution solution;
  solution.backend_name = kBackendName;
  solution.binary_variable_count = built_model.stats.binary_variable_count;
  solution.continuous_variable_count = built_model.stats.continuous_variable_count;
  solution.variable_count = solution.binary_variable_count + solution.continuous_variable_count;
  solution.constraint_count = built_model.stats.constraint_count;
  solution.min_delay_anchor_ns = problem.min_delay_anchor_ns;
  solution.min_power_anchor_w = problem.min_power_anchor_w;

  Highs highs;
  ApplyHighsOptions(highs, problem.solve_time_limit_ms);
  const auto pass_status = highs.passModel(std::move(built_model.model));
  if (pass_status == HighsStatus::kError) {
    solution.status = MathHtreeSolveStatus::kModelInvalid;
    solution.failure_reason = "highs_pass_model_failed";
    return solution;
  }

  const auto start_time = std::chrono::steady_clock::now();
  const auto run_status = highs.run();
  const auto end_time = std::chrono::steady_clock::now();
  solution.solve_wall_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

  const auto& info = highs.getInfo();
  const auto& highs_solution = highs.getSolution();
  solution.has_solver_incumbent = HasHighsIncumbent(info, highs_solution, solution.variable_count);
  solution.status = MapHighsModelStatus(highs.getModelStatus(), solution.has_solver_incumbent);
  solution.objective_value = info.objective_function_value;
  solution.primal_bound = solution.has_solver_incumbent ? info.objective_function_value : std::numeric_limits<double>::quiet_NaN();
  solution.dual_bound = info.mip_dual_bound;
  solution.optimality_gap = info.mip_gap;
  solution.branch_and_bound_node_count = static_cast<std::size_t>(std::max<int64_t>(0, info.mip_node_count));

  if (run_status == HighsStatus::kError && solution.status == MathHtreeSolveStatus::kAbnormal) {
    solution.failure_reason = "highs_run_failed";
    return solution;
  }

  if (solution.hasUsableIntegerSolution() && solution.has_solver_incumbent) {
    CollectSlotSolutions(problem, built_model, highs_solution, solution);
  } else if (solution.status == MathHtreeSolveStatus::kAbnormal && solution.failure_reason.empty()) {
    solution.failure_reason = "highs_solution_unavailable";
  }
  return solution;
}

auto IsUsableAnchorSolution(const MathHtreeSolution& solution) -> bool
{
  if (!solution.hasUsableIntegerSolution()) {
    return false;
  }
  if (solution.status != MathHtreeSolveStatus::kFeasibleWithGap) {
    return true;
  }
  return std::isfinite(solution.optimality_gap) && solution.optimality_gap <= kUsableAnchorGap;
}

auto AppendAnchorGapFailureReason(MathHtreeSolution& solution, const std::string& anchor_name) -> void
{
  std::ostringstream reason;
  reason << anchor_name << "_anchor_unavailable:" << ToString(solution.status);
  if (solution.status == MathHtreeSolveStatus::kFeasibleWithGap) {
    reason << ":gap=" << std::setprecision(6) << solution.optimality_gap;
  } else if (!solution.failure_reason.empty()) {
    reason << ':' << solution.failure_reason;
  } else {
    reason << ":no_detail";
  }
  solution.failure_reason = reason.str();
}

auto MakeTimedProblem(const MathHtreeProblem& problem, double time_limit_ms) -> MathHtreeProblem
{
  auto timed_problem = problem;
  timed_problem.solve_time_limit_ms = time_limit_ms;
  return timed_problem;
}

}  // namespace

HighsMathHtreeSolver::HighsMathHtreeSolver(HighsMathHtreeSolverOptions options) : _options(options)
{
}

auto HighsMathHtreeSolver::solve(const MathHtreeProblem& problem, MathHtreeObjective objective) -> MathHtreeSolution
{
  return RunHighs(problem, objective);
}

auto HighsMathHtreeSolver::solveNormalizedTradeoff(const MathHtreeProblem& problem) const -> MathHtreeSolution
{
  std::string failure_reason;
  if (!problem.isValid(failure_reason)) {
    return MakeStatusSolution(MathHtreeSolveStatus::kModelInvalid, failure_reason);
  }

  MathHtreeProblem anchored_problem = problem;
  auto min_delay_solution = solve(MakeTimedProblem(anchored_problem, _options.anchor_time_limit_ms), MathHtreeObjective::kMinDelay);
  if (!IsUsableAnchorSolution(min_delay_solution) || min_delay_solution.total_delay_ns <= kAnchorFloor) {
    AppendAnchorGapFailureReason(min_delay_solution, "min_delay");
    return min_delay_solution;
  }
  anchored_problem.min_delay_anchor_ns = min_delay_solution.total_delay_ns;

  auto min_power_solution = solve(MakeTimedProblem(anchored_problem, _options.anchor_time_limit_ms), MathHtreeObjective::kMinPower);
  if (!IsUsableAnchorSolution(min_power_solution) || min_power_solution.total_power_w <= kAnchorFloor) {
    AppendAnchorGapFailureReason(min_power_solution, "min_power");
    return min_power_solution;
  }
  anchored_problem.min_power_anchor_w = min_power_solution.total_power_w;

  auto normalized_solution
      = solve(MakeTimedProblem(anchored_problem, _options.normalized_time_limit_ms), MathHtreeObjective::kNormalizedDelayPower);
  normalized_solution.min_delay_anchor_ns = anchored_problem.min_delay_anchor_ns;
  normalized_solution.min_power_anchor_w = anchored_problem.min_power_anchor_w;
  normalized_solution.solve_wall_time_ms += min_delay_solution.solve_wall_time_ms + min_power_solution.solve_wall_time_ms;
  normalized_solution.branch_and_bound_node_count
      += min_delay_solution.branch_and_bound_node_count + min_power_solution.branch_and_bound_node_count;
  normalized_solution.optimality_gap
      = std::max({normalized_solution.optimality_gap, min_delay_solution.optimality_gap, min_power_solution.optimality_gap});
  return normalized_solution;
}

}  // namespace icts::htree::analytical_solver
