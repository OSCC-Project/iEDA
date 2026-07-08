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
 * @file MathHtreeModel.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-25
 * @brief Mathematical H-tree slot-choice optimization model contracts.
 */

#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace icts::htree::analytical_solver {

enum class MathHtreeSolveStatus
{
  kNotSolved,
  kOptimal,
  kFeasible,
  kFeasibleWithGap,
  kTimeout,
  kInfeasible,
  kUnbounded,
  kAbnormal,
  kModelInvalid,
  kSolverUnavailable,
};

enum class MathHtreeObjective
{
  kMinDelay,
  kMinPower,
  kNormalizedDelayPower,
};

struct MathHtreeAffineFunction
{
  double constant = 0.0;
  double input_slew_coefficient = 0.0;
  double load_cap_coefficient = 0.0;

  auto evaluate(double input_slew_ns, double load_cap_pf) const -> double;
};

struct MathHtreeDomain
{
  double input_slew_min_ns = 0.0;
  double input_slew_max_ns = 0.0;
  double load_cap_min_pf = 0.0;
  double load_cap_max_pf = 0.0;

  auto isValid() const -> bool;
};

struct MathHtreeChoice
{
  std::string name;
  MathHtreeDomain domain;
  MathHtreeAffineFunction output_slew_ns;
  MathHtreeAffineFunction delay_ns;
  MathHtreeAffineFunction power_w;
  MathHtreeAffineFunction source_boundary_power_w;
  MathHtreeAffineFunction source_cap_pf;
  double max_output_slew_ns = std::numeric_limits<double>::infinity();
  double max_source_cap_pf = std::numeric_limits<double>::infinity();
  bool source_has_buffer = false;
  unsigned source_strength_rank = 0U;
  bool sink_has_buffer = false;
  unsigned sink_strength_rank = 0U;
  bool terminal_branch_buffer = false;

  auto hasAnyBuffer() const -> bool;
};

struct MathHtreeSlot
{
  std::string name;
  std::size_t level_index = 0U;
  double delay_weight = 1.0;
  double power_weight = 1.0;
  double source_boundary_power_weight = 0.0;
  double downstream_cap_multiplier = 1.0;
  std::vector<MathHtreeChoice> choices;
};

struct MathHtreeLevel
{
  std::string name;
  std::size_t first_slot_index = 0U;
  std::size_t slot_count = 0U;
  bool is_leaf_level = false;
  bool require_terminal_branch_buffer = false;

  auto lastSlotIndex() const -> std::size_t;
};

struct MathHtreeProblem
{
  std::string name;
  double root_input_slew_ns = 0.0;
  double leaf_load_cap_pf = 0.0;
  double min_delay_anchor_ns = 0.0;
  double min_power_anchor_w = 0.0;
  double solve_time_limit_ms = 5000.0;
  std::size_t max_fanout = 0U;
  std::vector<MathHtreeSlot> slots;
  std::vector<MathHtreeLevel> levels;

  auto isValid(std::string& failure_reason) const -> bool;
};

struct MathHtreeSlotSolution
{
  std::size_t selected_choice_index = 0U;
  double input_slew_ns = 0.0;
  double output_slew_ns = 0.0;
  double load_cap_pf = 0.0;
  double source_cap_pf = 0.0;
  double delay_ns = 0.0;
  double power_w = 0.0;
  double source_boundary_power_w = 0.0;
};

struct MathHtreeSolution
{
  MathHtreeSolveStatus status = MathHtreeSolveStatus::kNotSolved;
  std::string failure_reason;
  std::string backend_name;
  std::size_t variable_count = 0U;
  std::size_t binary_variable_count = 0U;
  std::size_t continuous_variable_count = 0U;
  std::size_t constraint_count = 0U;
  std::size_t branch_and_bound_node_count = 0U;
  double solve_wall_time_ms = 0.0;
  double optimality_gap = std::numeric_limits<double>::infinity();
  double primal_bound = std::numeric_limits<double>::quiet_NaN();
  double dual_bound = std::numeric_limits<double>::quiet_NaN();
  double objective_value = 0.0;
  double total_delay_ns = 0.0;
  double total_power_w = 0.0;
  double min_delay_anchor_ns = 0.0;
  double min_power_anchor_w = 0.0;
  bool has_solver_incumbent = false;
  std::vector<MathHtreeSlotSolution> slots;

  auto hasUsableIntegerSolution() const -> bool;
};

auto ToString(MathHtreeSolveStatus status) -> std::string;
auto ToString(MathHtreeObjective objective) -> std::string;

}  // namespace icts::htree::analytical_solver
