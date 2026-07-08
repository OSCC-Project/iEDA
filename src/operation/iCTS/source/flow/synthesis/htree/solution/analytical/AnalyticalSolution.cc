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
 * @file AnalyticalSolution.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Analytical H-tree build-stage orchestration
 */

#include "synthesis/htree/solution/analytical/AnalyticalSolution.hh"

#include <glog/logging.h>

#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "HTreeTopologyChar.hh"
#include "Log.hh"
#include "logger/Schema.hh"
#include "synthesis/htree/analytical_solver/selection/AnalyticalSelection.hh"
#include "synthesis/htree/compensation/RootDriverCompensation.hh"
#include "synthesis/htree/diagnostic/HTreeDiagnostic.hh"
#include "synthesis/htree/plan/DepthPlan.hh"
#include "synthesis/htree/solution/finalization/SolutionFinalizer.hh"
#include "synthesis/htree/solution/report/StageReport.hh"
#include "synthesis/htree/synthesis_state/SynthesisState.hh"
#include "synthesis/htree/topology_pruning/TopologyPruning.hh"

namespace icts::htree::analytical_solution {
namespace as = analytical_selection;

auto SelectAnalyticalHTreeSolution(HTreeSynthesisState& state) -> HTreeSelectionBuild
{
  HTreeSelectionBuild selection_build;
  selection_build.engine = htree::HTreeSelectionEngine::kAnalytical;
  LOG_FATAL_IF(state.input == nullptr) << "HTree analytical solution requires synthesis input.";
  LOG_FATAL_IF(state.config == nullptr) << "HTree analytical solution requires synthesis config.";

  auto& result = state.result;
  const auto& input = *state.input;
  auto& segment_pattern_library = state.segmentPatterns();
  const auto& char_builder = state.charBuilder();

  LOG_FATAL_IF(input.design == nullptr) << "HTree analytical solution requires explicit Design dependency.";
  LOG_FATAL_IF(input.wrapper == nullptr) << "HTree analytical solution requires explicit Wrapper dependency.";
  LOG_FATAL_IF(input.reporter == nullptr) << "HTree analytical solution requires explicit reporter dependency.";
  auto& reporter = *input.reporter;
  auto analytical_stage = reporter.beginStage("HTree", "Select analytical topology candidates",
                                              {
                                                  {"depth_candidates", std::to_string(state.depth_candidates.size())},
                                                  {"max_depth", std::to_string(state.max_depth)},
                                                  {"solver", "math_milp_normalized_delay_power"},
                                              },
                                              DetailStageReportOptions());
  const auto analytical_attempt
      = as::TrySolveAnalyticalHTree(result.output.topology, state.full_level_plans, state.depth_candidates, segment_pattern_library,
                                    state.search_boundary_constraints, state.fanout_pruning_config, state.root_driver_compensation_input,
                                    state.sink_load_region_input, char_builder, result.diagnostics.char_slew_steps);
  result.diagnostics.analytical_model_set_count = analytical_attempt.model_set_count;
  result.diagnostics.analytical_rejected_fit_count = analytical_attempt.rejected_fit_count;
  result.diagnostics.analytical_structural_cap_operator_count = analytical_attempt.structural_cap_operator_count;
  result.diagnostics.analytical_evaluated_segment_count = analytical_attempt.evaluated_segment_count;
  result.diagnostics.analytical_generated_candidate_count = analytical_attempt.generated_candidate_count;
  result.diagnostics.analytical_validated_candidate_count = analytical_attempt.validated_candidate_count;
  result.diagnostics.analytical_validated_pareto_count = analytical_attempt.validated_pareto_count;
  result.diagnostics.analytical_selected_pareto_power_rank = analytical_attempt.selected_pareto_power_rank;
  result.diagnostics.analytical_validated_delay_min_ns = analytical_attempt.validated_delay_min_ns;
  result.diagnostics.analytical_validated_delay_median_ns = analytical_attempt.validated_delay_median_ns;
  result.diagnostics.analytical_validated_delay_max_ns = analytical_attempt.validated_delay_max_ns;
  result.diagnostics.analytical_validated_power_min_w = analytical_attempt.validated_power_min_w;
  result.diagnostics.analytical_validated_power_median_w = analytical_attempt.validated_power_median_w;
  result.diagnostics.analytical_validated_power_max_w = analytical_attempt.validated_power_max_w;
  result.diagnostics.analytical_solver_backend = analytical_attempt.backend_name;
  result.diagnostics.analytical_solver_status = analytical_attempt.solver_status;
  result.diagnostics.analytical_solver_variable_count = analytical_attempt.solver_variable_count;
  result.diagnostics.analytical_solver_binary_variable_count = analytical_attempt.solver_binary_variable_count;
  result.diagnostics.analytical_solver_continuous_variable_count = analytical_attempt.solver_continuous_variable_count;
  result.diagnostics.analytical_solver_constraint_count = analytical_attempt.solver_constraint_count;
  result.diagnostics.analytical_solver_wall_time_ms = analytical_attempt.solver_wall_time_ms;
  result.diagnostics.analytical_solver_objective_value = analytical_attempt.solver_objective_value;
  result.diagnostics.analytical_solver_optimality_gap = analytical_attempt.solver_optimality_gap;
  result.diagnostics.analytical_solver_min_delay_anchor_ns = analytical_attempt.solver_min_delay_anchor_ns;
  result.diagnostics.analytical_solver_min_power_anchor_w = analytical_attempt.solver_min_power_anchor_w;
  result.diagnostics.analytical_solver_total_delay_ns = analytical_attempt.solver_total_delay_ns;
  result.diagnostics.analytical_solver_total_power_w = analytical_attempt.solver_total_power_w;

  if (analytical_attempt.selected && analytical_attempt.selected_evaluation.best_char.has_value()) {
    result.diagnostics.analytical_mode_selected = true;
    analytical_stage.finished({
        {"selected_depth", std::to_string(analytical_attempt.selected_evaluation.depth)},
        {"model_sets", std::to_string(analytical_attempt.model_set_count)},
        {"rejected_fits", std::to_string(analytical_attempt.rejected_fit_count)},
        {"evaluated_segments", std::to_string(analytical_attempt.evaluated_segment_count)},
        {"scored_segments", std::to_string(analytical_attempt.scored_segment_count)},
        {"missing_models", std::to_string(analytical_attempt.missing_model_count)},
        {"decomposition_rejections", std::to_string(analytical_attempt.decomposition_rejected_count)},
        {"metric_rejections", std::to_string(analytical_attempt.metric_evaluation_rejected_count)},
        {"domain_slew_rejections", std::to_string(analytical_attempt.domain_slew_rejected_count)},
        {"domain_cap_rejections", std::to_string(analytical_attempt.domain_cap_rejected_count)},
        {"domain_slew_floors", std::to_string(analytical_attempt.domain_slew_floor_count)},
        {"domain_cap_floors", std::to_string(analytical_attempt.domain_cap_floor_count)},
        {"max_domain_rejected_cap_pf", std::to_string(analytical_attempt.max_domain_rejected_cap_pf)},
        {"materialization_attempts", std::to_string(analytical_attempt.materialization_attempt_count)},
        {"root_fanout_rejections", std::to_string(analytical_attempt.root_fanout_rejected_count)},
        {"lattice_rejections", std::to_string(analytical_attempt.lattice_rejected_count)},
        {"solver_backend", analytical_attempt.backend_name.empty() ? "unknown" : analytical_attempt.backend_name},
        {"solver_status", analytical_attempt.solver_status.empty() ? "unknown" : analytical_attempt.solver_status},
        {"solver_variables", std::to_string(analytical_attempt.solver_variable_count)},
        {"solver_binary_variables", std::to_string(analytical_attempt.solver_binary_variable_count)},
        {"solver_continuous_variables", std::to_string(analytical_attempt.solver_continuous_variable_count)},
        {"solver_constraints", std::to_string(analytical_attempt.solver_constraint_count)},
        {"solver_wall_time_ms", std::to_string(analytical_attempt.solver_wall_time_ms)},
        {"solver_objective_value", std::to_string(analytical_attempt.solver_objective_value)},
        {"solver_optimality_gap", std::to_string(analytical_attempt.solver_optimality_gap)},
        {"solver_min_delay_anchor_ns", std::to_string(analytical_attempt.solver_min_delay_anchor_ns)},
        {"solver_min_power_anchor_w", std::to_string(analytical_attempt.solver_min_power_anchor_w)},
        {"solver_total_delay_ns", std::to_string(analytical_attempt.solver_total_delay_ns)},
        {"solver_total_power_w", std::to_string(analytical_attempt.solver_total_power_w)},
        {"generated_candidates", std::to_string(analytical_attempt.generated_candidate_count)},
        {"validated_candidates", std::to_string(analytical_attempt.validated_candidate_count)},
        {"validated_pareto_candidates", std::to_string(analytical_attempt.validated_pareto_count)},
        {"selected_pareto_power_rank", std::to_string(analytical_attempt.selected_pareto_power_rank)},
        {"validated_delay_min_ns", std::to_string(analytical_attempt.validated_delay_min_ns)},
        {"validated_delay_median_ns", std::to_string(analytical_attempt.validated_delay_median_ns)},
        {"validated_delay_max_ns", std::to_string(analytical_attempt.validated_delay_max_ns)},
        {"validated_power_min_w", std::to_string(analytical_attempt.validated_power_min_w)},
        {"validated_power_median_w", std::to_string(analytical_attempt.validated_power_median_w)},
        {"validated_power_max_w", std::to_string(analytical_attempt.validated_power_max_w)},
    });

    auto selected_evaluation = analytical_attempt.selected_evaluation;
    auto selected_summary = analytical_attempt.selected_summary;
    result.diagnostics.depth_candidate_count = state.depth_candidates.size();

    htree::DepthSearchBuild analytical_exploration;
    as::ApplyAnalyticalRootDriverStats(analytical_exploration, analytical_attempt, state.root_driver_compensation_input);
    const htree::HTreeSelectedSolution selected_solution{
        .engine = htree::HTreeSelectionEngine::kAnalytical,
        .evaluation = selected_evaluation,
        .summary = selected_summary,
        .compensation_stats = analytical_exploration.summary.root_driver_compensation_stats,
        .compensation_detail = analytical_attempt.selected_compensation_detail,
        .root_driver_clock_period_source = state.root_driver_clock_period_source,
        .used_boundary_relaxation = false,
        .boundary_relaxation_reason = "",
        .boundary_relaxation_score = std::nullopt,
    };
    selection_build.selected = true;
    selection_build.selected_solution = selected_solution;
    return selection_build;
  }

  result.diagnostics.analytical_failure_reason
      = analytical_attempt.failure_reason.empty() ? "analytical_candidate_unavailable" : analytical_attempt.failure_reason;
  analytical_stage.failed({
      {"selected_depth", "none"},
      {"reason", result.diagnostics.analytical_failure_reason},
      {"model_sets", std::to_string(analytical_attempt.model_set_count)},
      {"rejected_fits", std::to_string(analytical_attempt.rejected_fit_count)},
      {"evaluated_segments", std::to_string(analytical_attempt.evaluated_segment_count)},
      {"scored_segments", std::to_string(analytical_attempt.scored_segment_count)},
      {"missing_models", std::to_string(analytical_attempt.missing_model_count)},
      {"decomposition_rejections", std::to_string(analytical_attempt.decomposition_rejected_count)},
      {"metric_rejections", std::to_string(analytical_attempt.metric_evaluation_rejected_count)},
      {"domain_slew_rejections", std::to_string(analytical_attempt.domain_slew_rejected_count)},
      {"domain_cap_rejections", std::to_string(analytical_attempt.domain_cap_rejected_count)},
      {"domain_slew_floors", std::to_string(analytical_attempt.domain_slew_floor_count)},
      {"domain_cap_floors", std::to_string(analytical_attempt.domain_cap_floor_count)},
      {"max_domain_rejected_cap_pf", std::to_string(analytical_attempt.max_domain_rejected_cap_pf)},
      {"materialization_attempts", std::to_string(analytical_attempt.materialization_attempt_count)},
      {"root_fanout_rejections", std::to_string(analytical_attempt.root_fanout_rejected_count)},
      {"lattice_rejections", std::to_string(analytical_attempt.lattice_rejected_count)},
      {"solver_backend", analytical_attempt.backend_name.empty() ? "unknown" : analytical_attempt.backend_name},
      {"solver_status", analytical_attempt.solver_status.empty() ? "unknown" : analytical_attempt.solver_status},
      {"solver_variables", std::to_string(analytical_attempt.solver_variable_count)},
      {"solver_binary_variables", std::to_string(analytical_attempt.solver_binary_variable_count)},
      {"solver_continuous_variables", std::to_string(analytical_attempt.solver_continuous_variable_count)},
      {"solver_constraints", std::to_string(analytical_attempt.solver_constraint_count)},
      {"solver_wall_time_ms", std::to_string(analytical_attempt.solver_wall_time_ms)},
      {"solver_objective_value", std::to_string(analytical_attempt.solver_objective_value)},
      {"solver_optimality_gap", std::to_string(analytical_attempt.solver_optimality_gap)},
      {"solver_min_delay_anchor_ns", std::to_string(analytical_attempt.solver_min_delay_anchor_ns)},
      {"solver_min_power_anchor_w", std::to_string(analytical_attempt.solver_min_power_anchor_w)},
      {"solver_total_delay_ns", std::to_string(analytical_attempt.solver_total_delay_ns)},
      {"solver_total_power_w", std::to_string(analytical_attempt.solver_total_power_w)},
      {"generated_candidates", std::to_string(analytical_attempt.generated_candidate_count)},
      {"validated_candidates", std::to_string(analytical_attempt.validated_candidate_count)},
      {"validated_pareto_candidates", std::to_string(analytical_attempt.validated_pareto_count)},
      {"selected_pareto_power_rank", std::to_string(analytical_attempt.selected_pareto_power_rank)},
      {"validated_delay_min_ns", std::to_string(analytical_attempt.validated_delay_min_ns)},
      {"validated_delay_median_ns", std::to_string(analytical_attempt.validated_delay_median_ns)},
      {"validated_delay_max_ns", std::to_string(analytical_attempt.validated_delay_max_ns)},
      {"validated_power_min_w", std::to_string(analytical_attempt.validated_power_min_w)},
      {"validated_power_median_w", std::to_string(analytical_attempt.validated_power_median_w)},
      {"validated_power_max_w", std::to_string(analytical_attempt.validated_power_max_w)},
  });
  EmitDiagnostic(reporter, DiagnosticLevel::kError, "HTree", "analytical H-tree candidate selection did not produce a validated candidate.",
                 {
                     {"reason", result.diagnostics.analytical_failure_reason},
                     {"model_sets", std::to_string(result.diagnostics.analytical_model_set_count)},
                     {"generated_candidates", std::to_string(result.diagnostics.analytical_generated_candidate_count)},
                     {"validated_candidates", std::to_string(result.diagnostics.analytical_validated_candidate_count)},
                 });
  selection_build.failure_reason = result.diagnostics.analytical_failure_reason;
  return selection_build;
}

}  // namespace icts::htree::analytical_solution
