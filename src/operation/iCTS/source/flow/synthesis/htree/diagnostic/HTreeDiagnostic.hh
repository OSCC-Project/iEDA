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
 * @file HTreeDiagnostic.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-25
 * @brief Internal diagnostic H-tree build contract for reports, experiments, and tests.
 */

#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "synthesis/htree/HTree.hh"

namespace icts::htree {

struct RootDriverCompensationReport
{
  bool enabled = false;
  bool valid = false;
  std::string method;
  std::string cell_master;
  std::string load_source;
  std::string route_estimator;
  double input_slew_ns = 0.0;
  unsigned load_bucket_idx = 0U;
  double load_cap_pf = 0.0;
  unsigned source_boundary_bucket_idx = 0U;
  double source_boundary_load_cap_pf = 0.0;
  std::size_t source_boundary_branch_count = 0U;
  double terminal_pin_cap_pf = 0.0;
  double wire_cap_pf = 0.0;
  double routed_wirelength_um = 0.0;
  std::size_t terminal_count = 0U;
  double clock_period_ns = 0.0;
  std::string clock_period_source;
  double output_slew_ns = 0.0;
  unsigned output_slew_bucket_idx = 0U;
  double cell_delay_ns = 0.0;
  double internal_power_w = 0.0;
  double leakage_power_w = 0.0;
  double cell_power_w = 0.0;
  double raw_delay_ns = 0.0;
  double raw_power_w = 0.0;
  double compensated_delay_ns = 0.0;
  double compensated_power_w = 0.0;
};

struct Diagnostics
{
  std::optional<unsigned> failure_level = std::nullopt;
  std::optional<unsigned> failure_length_idx = std::nullopt;
  double char_wirelength_unit_um = 0.0;
  unsigned char_wirelength_iterations = 0U;
  unsigned char_unique_level_bins = 0U;
  bool char_grid_adapted = false;
  double char_max_slew_ns = 0.0;
  double char_max_cap_pf = 0.0;
  unsigned char_slew_steps = 0U;
  unsigned char_cap_steps = 0U;
  bool force_branch_buffer = false;
  bool root_driver_sizing_enabled = true;
  std::optional<unsigned> target_depth = std::nullopt;
  unsigned depth_explore_window = 0U;
  std::size_t depth_candidate_count = 0U;
  std::size_t selected_final_frontier_count = 0U;
  std::size_t selected_candidate_solution_count = 0U;
  std::size_t selected_candidate_frontier_entry_count = 0U;
  std::size_t selected_feasible_solution_count = 0U;
  std::size_t selected_feasible_frontier_entry_count = 0U;
  std::optional<double> min_top_input_slew_ns = std::nullopt;
  std::optional<unsigned> top_input_slew_covering_idx = std::nullopt;
  std::size_t htree_load_group_count = 0U;
  double htree_load_cap_min_pf = 0.0;
  double htree_load_cap_max_pf = 0.0;
  double htree_load_cap_mean_pf = 0.0;
  double htree_load_cap_median_pf = 0.0;
  std::string selected_root_driver_cell_master;
  RootDriverCompensationReport root_driver_compensation;
  std::optional<double> boundary_relaxation_score = std::nullopt;
  std::string boundary_relaxation_reason;
  std::size_t pruned_leaf_single_load_buffers = 0U;
  bool analytical_mode_enabled = false;
  bool analytical_mode_selected = false;
  std::string analytical_failure_reason;
  std::size_t analytical_model_set_count = 0U;
  std::size_t analytical_rejected_fit_count = 0U;
  std::size_t analytical_structural_cap_operator_count = 0U;
  std::size_t analytical_evaluated_segment_count = 0U;
  std::size_t analytical_generated_candidate_count = 0U;
  std::size_t analytical_validated_candidate_count = 0U;
  std::size_t analytical_validated_pareto_count = 0U;
  std::size_t analytical_selected_pareto_power_rank = 0U;
  double analytical_validated_delay_min_ns = 0.0;
  double analytical_validated_delay_median_ns = 0.0;
  double analytical_validated_delay_max_ns = 0.0;
  double analytical_validated_power_min_w = 0.0;
  double analytical_validated_power_median_w = 0.0;
  double analytical_validated_power_max_w = 0.0;
  std::string analytical_solver_backend;
  std::string analytical_solver_status;
  std::size_t analytical_solver_variable_count = 0U;
  std::size_t analytical_solver_binary_variable_count = 0U;
  std::size_t analytical_solver_continuous_variable_count = 0U;
  std::size_t analytical_solver_constraint_count = 0U;
  double analytical_solver_wall_time_ms = 0.0;
  double analytical_solver_objective_value = 0.0;
  double analytical_solver_optimality_gap = 0.0;
  double analytical_solver_min_delay_anchor_ns = 0.0;
  double analytical_solver_min_power_anchor_w = 0.0;
  double analytical_solver_total_delay_ns = 0.0;
  double analytical_solver_total_power_w = 0.0;
  HTree::LogContext log_context;
  std::string object_name_prefix;
};

struct DiagnosticBuild : HTree::Build
{
  DiagnosticBuild() = default;

  DiagnosticBuild(const DiagnosticBuild&) = delete;
  auto operator=(const DiagnosticBuild&) -> DiagnosticBuild& = delete;

  DiagnosticBuild(DiagnosticBuild&& rhs) noexcept = default;
  auto operator=(DiagnosticBuild&& rhs) noexcept -> DiagnosticBuild& = default;

  Diagnostics diagnostics;
};

auto BuildWithDiagnostics(const HTree::Input& input, const HTree::Config& config) -> DiagnosticBuild;

}  // namespace icts::htree
