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
 * @file AnalyticalSelection.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Analytical candidate selection helpers for H-tree synthesis
 */

#include "synthesis/htree/analytical_solver/selection/AnalyticalSelection.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "AnalyticalCharacterization.hh"
#include "AnalyticalModel.hh"
#include "ClockRouteSegmentRc.hh"
#include "HTreeTopologyChar.hh"
#include "Log.hh"
#include "SegmentChar.hh"
#include "ValueLattice.hh"
#include "characterization/Characterization.hh"
#include "synthesis/htree/HTree.hh"
#include "synthesis/htree/analytical_solver/AnalyticalSolver.hh"
#include "synthesis/htree/analytical_solver/candidate/AnalyticalCandidate.hh"
#include "synthesis/htree/analytical_solver/selection/AnalyticalValidation.hh"
#include "synthesis/htree/compensation/RootDriverCompensation.hh"
#include "synthesis/htree/constraint/Constraint.hh"
#include "synthesis/htree/plan/DepthPlan.hh"
#include "synthesis/htree/plan/Plan.hh"
#include "synthesis/htree/region/SinkLoadRegion.hh"
#include "synthesis/htree/segment_pruning/TopologyPatternLibrary.hh"
#include "synthesis/htree/topology_pruning/TopologyPruning.hh"

namespace icts {
class BufferingPattern;
}  // namespace icts

namespace icts::htree::analytical_selection {
namespace {

auto ResolveAnalyticalRepresentativeLeafLoadCapPf(const UniformValueLattice& cap_lattice) -> double
{
  if (!cap_lattice.isValid()) {
    return 0.0;
  }
  return cap_lattice.valueForIndex(1U);
}

auto ShouldPreferAnalyticalCandidate(const analytical_solver::AnalyticalCandidate& lhs, const analytical_solver::AnalyticalCandidate& rhs)
    -> bool
{
  if (lhs.materialized_char.has_value() && rhs.materialized_char.has_value()) {
    const auto& lhs_char = *lhs.materialized_char;
    const auto& rhs_char = *rhs.materialized_char;
    if (lhs_char.get_delay() != rhs_char.get_delay()) {
      return lhs_char.get_delay() < rhs_char.get_delay();
    }
    if (lhs_char.get_power() != rhs_char.get_power()) {
      return lhs_char.get_power() < rhs_char.get_power();
    }
  }
  return analytical_solver::PreferAnalyticalCandidate(lhs, rhs);
}

auto PreferAnalyticalPowerOrder(const analytical_solver::AnalyticalCandidate& lhs, const analytical_solver::AnalyticalCandidate& rhs)
    -> bool
{
  if (lhs.materialized_char.has_value() && rhs.materialized_char.has_value()) {
    const auto& lhs_char = *lhs.materialized_char;
    const auto& rhs_char = *rhs.materialized_char;
    if (lhs_char.get_power() != rhs_char.get_power()) {
      return lhs_char.get_power() < rhs_char.get_power();
    }
    if (lhs_char.get_delay() != rhs_char.get_delay()) {
      return lhs_char.get_delay() < rhs_char.get_delay();
    }
    if (lhs_char.get_driven_cap_idx() != rhs_char.get_driven_cap_idx()) {
      return lhs_char.get_driven_cap_idx() < rhs_char.get_driven_cap_idx();
    }
    if (lhs_char.get_output_slew_idx() != rhs_char.get_output_slew_idx()) {
      return lhs_char.get_output_slew_idx() < rhs_char.get_output_slew_idx();
    }
  }
  return analytical_solver::PreferAnalyticalCandidate(lhs, rhs);
}

auto RequireMaterializedAnalyticalChar(const AnalyticalValidatedCandidate& candidate) -> const HTreeTopologyChar&
{
  LOG_FATAL_IF(!candidate.candidate.materialized_char.has_value()) << "HTree: missing materialized analytical topology char.";
  return *candidate.candidate.materialized_char;
}

auto CollectAnalyticalDelayPowerParetoRefs(const std::vector<AnalyticalValidatedCandidate>& candidates)
    -> std::vector<const AnalyticalValidatedCandidate*>
{
  std::vector<const AnalyticalValidatedCandidate*> sorted_candidates;
  sorted_candidates.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    if (candidate.candidate.materialized_char.has_value()) {
      sorted_candidates.push_back(&candidate);
    }
  }

  std::ranges::sort(sorted_candidates, [](const AnalyticalValidatedCandidate* lhs, const AnalyticalValidatedCandidate* rhs) -> bool {
    LOG_FATAL_IF(lhs == nullptr || rhs == nullptr) << "HTree: null analytical candidate during Pareto sorting.";
    return ShouldPreferAnalyticalCandidate(lhs->candidate, rhs->candidate);
  });

  std::vector<const AnalyticalValidatedCandidate*> pareto_front;
  pareto_front.reserve(sorted_candidates.size());
  double best_power_before_delay = std::numeric_limits<double>::infinity();
  std::size_t delay_group_begin = 0U;
  while (delay_group_begin < sorted_candidates.size()) {
    std::size_t delay_group_end = delay_group_begin + 1U;
    const auto& delay_group_anchor_char = RequireMaterializedAnalyticalChar(*sorted_candidates.at(delay_group_begin));
    while (delay_group_end < sorted_candidates.size()) {
      const auto& delay_group_next_char = RequireMaterializedAnalyticalChar(*sorted_candidates.at(delay_group_end));
      if (delay_group_next_char.get_delay() != delay_group_anchor_char.get_delay()) {
        break;
      }
      ++delay_group_end;
    }

    double delay_group_min_power = std::numeric_limits<double>::infinity();
    for (std::size_t index = delay_group_begin; index < delay_group_end; ++index) {
      const auto& materialized_char = RequireMaterializedAnalyticalChar(*sorted_candidates.at(index));
      delay_group_min_power = std::min(delay_group_min_power, materialized_char.get_power());
    }

    for (std::size_t index = delay_group_begin; index < delay_group_end; ++index) {
      const auto* candidate = sorted_candidates.at(index);
      const auto& materialized_char = RequireMaterializedAnalyticalChar(*candidate);
      if (best_power_before_delay <= materialized_char.get_power() || delay_group_min_power < materialized_char.get_power()) {
        continue;
      }
      pareto_front.push_back(candidate);
    }

    best_power_before_delay = std::min(best_power_before_delay, delay_group_min_power);
    delay_group_begin = delay_group_end;
  }

  std::ranges::sort(pareto_front, [](const AnalyticalValidatedCandidate* lhs, const AnalyticalValidatedCandidate* rhs) -> bool {
    LOG_FATAL_IF(lhs == nullptr || rhs == nullptr) << "HTree: null analytical candidate during Pareto delay ordering.";
    return ShouldPreferAnalyticalCandidate(lhs->candidate, rhs->candidate);
  });
  return pareto_front;
}

auto SelectAnalyticalParetoPowerGuardedMinDelay(const std::vector<AnalyticalValidatedCandidate>& candidates,
                                                AnalyticalHTreeAttempt& attempt) -> const AnalyticalValidatedCandidate*
{
  auto pareto_front = CollectAnalyticalDelayPowerParetoRefs(candidates);
  attempt.validated_pareto_count = pareto_front.size();
  if (pareto_front.empty()) {
    return nullptr;
  }

  double min_pareto_power_w = std::numeric_limits<double>::infinity();
  for (const auto* candidate : pareto_front) {
    LOG_FATAL_IF(candidate == nullptr || !candidate->candidate.materialized_char.has_value())
        << "HTree: invalid analytical Pareto candidate during power guard calculation.";
    min_pareto_power_w = std::min(min_pareto_power_w, candidate->candidate.materialized_char->get_power());
  }

  const auto* selected_candidate = pareto_front.front();
  const double guarded_power_limit_w = min_pareto_power_w * (1.0 + kAnalyticalParetoPowerSlackRatio);
  if (std::isfinite(guarded_power_limit_w) && guarded_power_limit_w > 0.0) {
    for (const auto* candidate : pareto_front) {
      LOG_FATAL_IF(candidate == nullptr || !candidate->candidate.materialized_char.has_value())
          << "HTree: invalid analytical Pareto candidate during power-guarded delay selection.";
      if (candidate->candidate.materialized_char->get_power() <= guarded_power_limit_w) {
        selected_candidate = candidate;
        break;
      }
    }
  }

  auto power_ordered_pareto_front = pareto_front;
  std::ranges::sort(
      power_ordered_pareto_front, [](const AnalyticalValidatedCandidate* lhs, const AnalyticalValidatedCandidate* rhs) -> bool {
        LOG_FATAL_IF(lhs == nullptr || rhs == nullptr) << "HTree: null analytical candidate during selected power rank calculation.";
        return PreferAnalyticalPowerOrder(lhs->candidate, rhs->candidate);
      });
  const auto selected_rank_it = std::ranges::find(power_ordered_pareto_front, selected_candidate);
  attempt.selected_pareto_power_rank
      = selected_rank_it == power_ordered_pareto_front.end()
            ? 0U
            : static_cast<std::size_t>(std::distance(power_ordered_pareto_front.begin(), selected_rank_it)) + 1U;
  return selected_candidate;
}

auto RecordAnalyticalValidatedDistribution(const std::vector<AnalyticalValidatedCandidate>& candidates, AnalyticalHTreeAttempt& attempt)
    -> void
{
  std::vector<double> delays;
  std::vector<double> powers;
  delays.reserve(candidates.size());
  powers.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    if (!candidate.candidate.materialized_char.has_value()) {
      continue;
    }
    delays.push_back(candidate.candidate.materialized_char->get_delay());
    powers.push_back(candidate.candidate.materialized_char->get_power());
  }
  if (delays.empty() || powers.empty()) {
    return;
  }

  std::ranges::sort(delays);
  std::ranges::sort(powers);
  const std::size_t median_index = (delays.size() - 1U) / 2U;
  attempt.validated_delay_min_ns = delays.front();
  attempt.validated_delay_median_ns = delays.at(median_index);
  attempt.validated_delay_max_ns = delays.back();
  attempt.validated_power_min_w = powers.front();
  attempt.validated_power_median_w = powers.at(median_index);
  attempt.validated_power_max_w = powers.back();
}

auto MakeAnalyticalCandidateEvaluation(const analytical_solver::AnalyticalCandidate& candidate, const std::vector<HTree::LevelPlan>& levels,
                                       const BoundaryConstraints& boundary_constraints, std::size_t final_frontier_count)
    -> htree::CandidateBuildEvaluation
{
  htree::CandidateBuildEvaluation evaluation;
  evaluation.depth = candidate.depth;
  evaluation.leaf_count = candidate.leaf_count;
  evaluation.boundary_constraints = boundary_constraints;
  evaluation.levels = levels;
  evaluation.success = candidate.materialized_char.has_value();
  evaluation.failure_reason = evaluation.success ? "" : "missing_analytical_materialized_char";
  evaluation.final_frontier_count = final_frontier_count;
  evaluation.candidate_solution_count = final_frontier_count;
  evaluation.feasible_solution_count = final_frontier_count;
  if (candidate.materialized_char.has_value()) {
    evaluation.candidate_frontier_entries = {*candidate.materialized_char};
    evaluation.feasible_frontier_entries = {*candidate.materialized_char};
    evaluation.best_char = *candidate.materialized_char;
  }
  evaluation.topology_pattern_library = candidate.topology_pattern_library;
  return evaluation;
}

auto MakeAnalyticalDepthSummary(const CandidateBuildEvaluation& evaluation, const SinkLoadRegionLegalitySummary& sink_legality)
    -> DepthSummary
{
  return htree::DepthSummary{
      .depth = evaluation.depth,
      .leaf_count = evaluation.leaf_count,
      .success = evaluation.success,
      .selected = true,
      .used_explicit_target_depth = false,
      .failure_reason = evaluation.failure_reason,
      .htree_load_group_count = sink_legality.cap_distribution.group_count,
      .htree_load_cap_min_pf = sink_legality.cap_distribution.cap_min_pf,
      .htree_load_cap_max_pf = sink_legality.cap_distribution.cap_max_pf,
      .htree_load_cap_mean_pf = sink_legality.cap_distribution.cap_mean_pf,
      .htree_load_cap_median_pf = sink_legality.cap_distribution.cap_median_pf,
      .final_frontier_count = evaluation.final_frontier_count,
      .candidate_solution_count = evaluation.candidate_solution_count,
      .candidate_frontier_entry_count = evaluation.candidate_frontier_entries.size(),
      .feasible_solution_count = evaluation.feasible_solution_count,
      .feasible_frontier_entry_count = evaluation.feasible_frontier_entries.size(),
      .used_boundary_relaxation = false,
      .selected_power_w = evaluation.best_char.has_value() ? evaluation.best_char->get_power() : 0.0,
      .selected_delay_ns = evaluation.best_char.has_value() ? evaluation.best_char->get_delay() : 0.0,
  };
}

auto CollectAnalyticalUnitSegmentChars(const CharBuilder& char_builder, unsigned unit_length_idx) -> std::vector<SegmentChar>
{
  std::vector<SegmentChar> segment_chars;
  for (const auto& segment_char : char_builder.get_segment_chars()) {
    if (segment_char.get_length_idx() == unit_length_idx) {
      segment_chars.push_back(segment_char);
    }
  }
  return segment_chars;
}

auto BuildAnalyticalModelCatalog(const std::vector<SegmentChar>& segment_chars, const std::vector<BufferingPattern>& buffering_patterns,
                                 const CharBuilder& char_builder, AnalyticalHTreeAttempt& attempt)
    -> analytical::AnalyticalCharacterizationBuild
{
  analytical::AnalyticalCharacterizationConfig analytical_options;
  analytical_options.require_monotonic_output_slew = false;
  analytical_options.require_monotonic_delay = false;
  analytical_options.require_monotonic_power = false;
  analytical_options.require_monotonic_source_boundary_power = false;
  analytical_options.allow_sparse_constant_model = false;
  analytical_options.prefer_exact_structural_cap = true;
  analytical_options.length_unit_um = char_builder.get_wirelength_unit_um();
  analytical_options.clock_route_segment_rc = char_builder.get_clock_route_segment_rc();
  for (const auto& buffer_cell : char_builder.get_characterization_buffer_cells()) {
    analytical_options.buffer_input_cap_pf_by_cell_master[buffer_cell.cell_master] = buffer_cell.input_cap_pf;
  }

  auto char_result = analytical::AnalyticalCharacterization::buildFromSegmentChars(
      segment_chars, buffering_patterns, char_builder.get_slew_lattice(), char_builder.get_cap_lattice(), analytical_options);
  attempt.model_set_count = char_result.summary.model_set_count;
  attempt.rejected_fit_count = char_result.summary.rejected_fit_count;
  attempt.structural_cap_operator_count = char_result.summary.structural_cap_operator_count;
  return char_result;
}

}  // namespace

auto TrySolveAnalyticalHTree(const Tree& topology, const std::vector<HTree::LevelPlan>& full_level_plans,
                             const std::vector<unsigned>& depth_candidates, BufferPatternLibrary& segment_pattern_library,
                             const BoundaryConstraints& search_boundary_constraints, const HTreeFanoutPruningConfig& fanout_pruning_config,
                             const RootDriverCompensationInput& root_driver_compensation_input,
                             const SinkLoadRegionLegalityInput& sink_load_region_input, const CharBuilder& char_builder,
                             unsigned char_slew_steps) -> AnalyticalHTreeAttempt
{
  AnalyticalHTreeAttempt attempt;
  if (depth_candidates.empty()) {
    attempt.failure_reason = "empty_depth_candidates";
    return attempt;
  }

  const auto analytical_segment_chars = CollectAnalyticalUnitSegmentChars(char_builder, kAnalyticalUnitLengthIdx);
  const auto& analytical_buffering_patterns = char_builder.get_buffering_patterns();
  auto char_result = BuildAnalyticalModelCatalog(analytical_segment_chars, analytical_buffering_patterns, char_builder, attempt);
  if (char_result.output.catalog.empty()) {
    attempt.failure_reason
        = char_result.summary.failures.empty() ? "empty_analytical_model_catalog" : char_result.summary.failures.front().reason;
    return attempt;
  }

  htree::SinkLoadRegionLegalityContext sink_load_region_legality_context{
      .result_by_signature = {},
      .max_monotone_failed_level = std::numeric_limits<int>::min(),
      .cap_lattice = char_builder.get_cap_lattice(),
      .input = sink_load_region_input,
  };
  htree::RootDriverCompensationPass compensation_pass(root_driver_compensation_input);
  std::vector<AnalyticalValidatedCandidate> legal_candidates;
  std::string first_solver_failure;
  std::string first_validation_failure;
  const std::vector<unsigned> solved_depth_candidates = {depth_candidates.front()};

  for (const unsigned depth : solved_depth_candidates) {
    auto candidate_levels = htree::MakeCandidateLevelPlans(full_level_plans, depth);
    analytical_solver::AnalyticalHTreeSolveProblem solve_problem;
    solve_problem.levels = &candidate_levels;
    solve_problem.segment_pattern_library = &segment_pattern_library;
    solve_problem.mutable_segment_pattern_library = &segment_pattern_library;
    solve_problem.model_catalog = &char_result.output.catalog;
    solve_problem.boundary_constraints = search_boundary_constraints;
    solve_problem.fanout_config = fanout_pruning_config;
    solve_problem.slew_lattice = char_builder.get_slew_lattice();
    solve_problem.cap_lattice = char_builder.get_cap_lattice();
    solve_problem.config.root_input_slew_ns = root_driver_compensation_input.input_slew_ns;
    solve_problem.config.representative_leaf_load_cap_pf = ResolveAnalyticalRepresentativeLeafLoadCapPf(char_builder.get_cap_lattice());
    solve_problem.config.use_functional_unit_compose = true;
    solve_problem.config.unit_length_idx = kAnalyticalUnitLengthIdx;

    auto solver_result = analytical_solver::SolveAnalyticalHTreeCandidates(solve_problem);
    attempt.evaluated_segment_count += solver_result.summary.evaluated_segment_count;
    attempt.generated_candidate_count += solver_result.summary.generated_candidate_count;
    attempt.scored_segment_count += solver_result.summary.scored_segment_count;
    attempt.missing_model_count += solver_result.summary.missing_model_count;
    attempt.decomposition_rejected_count += solver_result.summary.decomposition_rejected_count;
    attempt.metric_evaluation_rejected_count += solver_result.summary.metric_evaluation_rejected_count;
    attempt.domain_slew_rejected_count += solver_result.summary.domain_slew_rejected_count;
    attempt.domain_cap_rejected_count += solver_result.summary.domain_cap_rejected_count;
    attempt.domain_slew_floor_count += solver_result.summary.domain_slew_floor_count;
    attempt.domain_cap_floor_count += solver_result.summary.domain_cap_floor_count;
    attempt.max_domain_rejected_cap_pf = std::max(attempt.max_domain_rejected_cap_pf, solver_result.summary.max_domain_rejected_cap_pf);
    attempt.materialization_attempt_count += solver_result.summary.materialization_attempt_count;
    attempt.root_fanout_rejected_count += solver_result.summary.root_fanout_rejected_count;
    attempt.lattice_rejected_count += solver_result.summary.lattice_rejected_count;
    if (!solver_result.summary.backend_name.empty()) {
      attempt.backend_name = solver_result.summary.backend_name;
    }
    if (!solver_result.summary.solver_status.empty()) {
      attempt.solver_status = solver_result.summary.solver_status;
    }
    attempt.solver_variable_count += solver_result.summary.solver_variable_count;
    attempt.solver_binary_variable_count += solver_result.summary.solver_binary_variable_count;
    attempt.solver_continuous_variable_count += solver_result.summary.solver_continuous_variable_count;
    attempt.solver_constraint_count += solver_result.summary.solver_constraint_count;
    attempt.solver_wall_time_ms += solver_result.summary.solver_wall_time_ms;
    attempt.solver_objective_value = solver_result.summary.solver_objective_value;
    attempt.solver_optimality_gap = std::max(attempt.solver_optimality_gap, solver_result.summary.solver_optimality_gap);
    attempt.solver_primal_bound = solver_result.summary.solver_primal_bound;
    attempt.solver_dual_bound = solver_result.summary.solver_dual_bound;
    attempt.solver_min_delay_anchor_ns = solver_result.summary.solver_min_delay_anchor_ns;
    attempt.solver_min_power_anchor_w = solver_result.summary.solver_min_power_anchor_w;
    attempt.solver_total_delay_ns = solver_result.summary.solver_total_delay_ns;
    attempt.solver_total_power_w = solver_result.summary.solver_total_power_w;
    if (!solver_result.summary.success) {
      if (first_solver_failure.empty()) {
        first_solver_failure
            = solver_result.summary.failure_reason.empty() ? "analytical_solver_failed" : solver_result.summary.failure_reason;
      }
      continue;
    }

    for (auto& candidate : solver_result.output.candidates) {
      htree::RootDriverCompensationPass candidate_compensation_pass(root_driver_compensation_input);
      auto validation = analytical_solver::ValidateAnalyticalCandidate(
          candidate, analytical_solver::AnalyticalCandidateLegalityCheck{
                         .topology = &topology,
                         .segment_pattern_library = &segment_pattern_library,
                         .sink_load_region_legality_context = &sink_load_region_legality_context,
                         .root_driver_compensation_pass = &candidate_compensation_pass,
                         .validate_sink_load_region = true,
                         .validate_root_driver_compensation = root_driver_compensation_input.enabled,
                     });
      if (!validation.legal) {
        if (first_validation_failure.empty()) {
          first_validation_failure
              = validation.failure_reason.empty() ? "analytical_candidate_validation_failed" : validation.failure_reason;
        }
        continue;
      }
      ++attempt.validated_candidate_count;
      if (candidate.materialized_char.has_value() && validation.root_driver_compensation.enabled) {
        candidate.materialized_char->set_root_driver_compensation(validation.root_driver_compensation.cell_delay_ns,
                                                                  validation.root_driver_compensation.cell_power_w);
      }
      legal_candidates.push_back(AnalyticalValidatedCandidate{
          .candidate = std::move(candidate),
          .validation = std::move(validation),
          .original_index = legal_candidates.size(),
      });
    }
  }

  attempt.root_driver_compensation_stats = compensation_pass.get_stats();
  if (legal_candidates.empty()) {
    if (!first_validation_failure.empty()) {
      attempt.failure_reason = first_validation_failure;
    } else if (!first_solver_failure.empty()) {
      attempt.failure_reason = first_solver_failure;
    } else {
      attempt.failure_reason = "no_legal_analytical_candidate";
    }
    return attempt;
  }

  RecordAnalyticalValidatedDistribution(legal_candidates, attempt);
  const auto* selected_candidate = SelectAnalyticalParetoPowerGuardedMinDelay(legal_candidates, attempt);
  if (selected_candidate == nullptr || !selected_candidate->candidate.materialized_char.has_value()) {
    attempt.failure_reason = "missing_best_analytical_candidate";
    return attempt;
  }

  attempt.selected = true;
  attempt.selected_evaluation = MakeAnalyticalCandidateEvaluation(
      selected_candidate->candidate, htree::MakeCandidateLevelPlans(full_level_plans, selected_candidate->candidate.depth),
      search_boundary_constraints, attempt.validated_pareto_count);
  attempt.selected_sink_load_region_legality = selected_candidate->validation.sink_load_region;
  attempt.selected_compensation_detail = selected_candidate->validation.root_driver_compensation;
  attempt.selected_summary = MakeAnalyticalDepthSummary(attempt.selected_evaluation, attempt.selected_sink_load_region_legality);
  attempt.root_driver_compensation_stats = compensation_pass.get_stats();
  (void) char_slew_steps;
  return attempt;
}

auto ApplyAnalyticalRootDriverStats(DepthSearchBuild& exploration, const AnalyticalHTreeAttempt& attempt,
                                    const RootDriverCompensationInput& compensation_input) -> void
{
  exploration.summary.root_driver_compensation_stats = attempt.root_driver_compensation_stats;
  exploration.summary.root_driver_compensation_stats.enabled = compensation_input.enabled;
  exploration.summary.root_driver_compensation_stats.input_slew_ns = compensation_input.input_slew_ns;
  exploration.summary.root_driver_compensation_stats.clock_period_ns = compensation_input.clock_period_ns;
  exploration.summary.root_driver_compensation_stats.method = compensation_input.enabled ? "direct" : "disabled";
}

}  // namespace icts::htree::analytical_selection
