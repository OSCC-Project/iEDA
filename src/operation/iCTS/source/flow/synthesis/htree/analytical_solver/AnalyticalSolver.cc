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
 * @file AnalyticalSolver.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-14
 * @brief Mathematical analytical H-tree solver implementation.
 */

#include "synthesis/htree/analytical_solver/AnalyticalSolver.hh"

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "HTreeTopologyChar.hh"
#include "PatternId.hh"
#include "characterization/Characterization.hh"
#include "synthesis/htree/analytical_solver/candidate/AnalyticalCandidate.hh"
#include "synthesis/htree/analytical_solver/model/AnalyticalSolverModel.hh"
#include "synthesis/htree/analytical_solver/model/MathHtreeModel.hh"
#include "synthesis/htree/analytical_solver/model/MathHtreeProblemBuilder.hh"
#include "synthesis/htree/analytical_solver/solver/MathHtreeMilpSolver.hh"
#include "synthesis/htree/segment_pruning/TopologyPatternLibrary.hh"
#include "synthesis/htree/topology_pruning/TopologyPruning.hh"

namespace icts::htree::analytical_solver {
namespace {

auto MarkFailure(AnalyticalSolverBuild& result, std::string reason) -> AnalyticalSolverBuild
{
  result.summary.success = false;
  result.summary.failure_reason = std::move(reason);
  return result;
}

auto MakeFunctionalContext(const AnalyticalHTreeSolveProblem& solve_problem) -> FunctionalComposeContext
{
  FunctionalComposeContext context;
  context.unit_models = CollectUnitModelRefs(solve_problem);
  context.unit_pattern_by_cell_master_and_terminal_semantic = BuildUnitPatternByCellMaster(solve_problem, context.unit_models);
  const auto* segment_pattern_library = ResolveSegmentPatternLibrary(solve_problem);
  context.next_segment_pattern_id = segment_pattern_library == nullptr ? 0U : ResolveNextSegmentPatternId(*segment_pattern_library);
  return context;
}

auto SelectedUnitPatternIds(const MathHtreeProblemBuildResult& build_result, const MathHtreeSolution& solution) -> std::vector<PatternId>
{
  std::vector<PatternId> unit_pattern_ids;
  unit_pattern_ids.reserve(solution.slots.size());
  for (std::size_t slot_index = 0U; slot_index < solution.slots.size(); ++slot_index) {
    const auto selected_choice_index = solution.slots.at(slot_index).selected_choice_index;
    if (slot_index >= build_result.choice_refs_by_slot.size()
        || selected_choice_index >= build_result.choice_refs_by_slot.at(slot_index).size()) {
      return {};
    }
    unit_pattern_ids.push_back(build_result.choice_refs_by_slot.at(slot_index).at(selected_choice_index).unit_pattern_id);
  }
  return unit_pattern_ids;
}

auto GroupUnitPatternIdsByLevel(const std::vector<PatternId>& unit_pattern_ids, const std::vector<unsigned>& level_slot_counts)
    -> std::vector<std::vector<PatternId>>
{
  std::vector<std::vector<PatternId>> grouped;
  grouped.reserve(level_slot_counts.size());
  std::size_t cursor = 0U;
  for (const unsigned slot_count : level_slot_counts) {
    if (cursor + slot_count > unit_pattern_ids.size()) {
      return {};
    }
    grouped.emplace_back(unit_pattern_ids.begin() + static_cast<std::ptrdiff_t>(cursor),
                         unit_pattern_ids.begin() + static_cast<std::ptrdiff_t>(cursor + slot_count));
    cursor += slot_count;
  }
  return cursor == unit_pattern_ids.size() ? grouped : std::vector<std::vector<PatternId>>{};
}

auto MaterializeLevelSegmentPatterns(const AnalyticalHTreeSolveProblem& solve_problem,
                                     const std::vector<std::vector<PatternId>>& level_unit_pattern_ids, FunctionalComposeContext& context)
    -> std::optional<std::vector<PatternId>>
{
  if (solve_problem.mutable_segment_pattern_library == nullptr) {
    return std::nullopt;
  }
  std::vector<PatternId> level_segment_pattern_ids;
  level_segment_pattern_ids.reserve(level_unit_pattern_ids.size());
  for (const auto& unit_pattern_ids : level_unit_pattern_ids) {
    auto segment_pattern_id
        = MaterializeFunctionalSegmentPattern(unit_pattern_ids, context, *solve_problem.mutable_segment_pattern_library);
    if (!segment_pattern_id.has_value()) {
      return std::nullopt;
    }
    level_segment_pattern_ids.push_back(*segment_pattern_id);
  }
  return level_segment_pattern_ids;
}

auto ScoreSelectedLevelSegments(const AnalyticalHTreeSolveProblem& solve_problem,
                                const std::vector<std::vector<PatternId>>& level_unit_pattern_ids,
                                const std::vector<PatternId>& level_segment_pattern_ids, const std::vector<unsigned>& level_slot_counts,
                                const MathHtreeSolution& solution, AnalyticalSolverBuild& result)
    -> std::optional<std::vector<ScoredSegment>>
{
  if (solve_problem.levels == nullptr || level_unit_pattern_ids.size() != solve_problem.levels->size()
      || level_segment_pattern_ids.size() != solve_problem.levels->size() || level_slot_counts.size() != solve_problem.levels->size()) {
    return std::nullopt;
  }

  std::vector<ScoredSegment> scored_levels(solve_problem.levels->size());
  std::size_t slot_cursor = 0U;
  for (std::size_t level_index = 0U; level_index < solve_problem.levels->size(); ++level_index) {
    const unsigned level_slot_count = level_slot_counts.at(level_index);
    if (level_slot_count == 0U || slot_cursor + level_slot_count > solution.slots.size()) {
      return std::nullopt;
    }
    const std::size_t level_begin_slot = slot_cursor;
    const std::size_t level_end_slot = slot_cursor + level_slot_count - 1U;
    auto scored = ScoreFunctionalUnitSequence(
        solve_problem, level_unit_pattern_ids.at(level_index), level_segment_pattern_ids.at(level_index),
        solve_problem.levels->at(level_index).aligned_length_idx, solution.slots.at(level_begin_slot).input_slew_ns,
        solution.slots.at(level_end_slot).load_cap_pf, false, result);
    if (!scored.has_value()) {
      return std::nullopt;
    }
    scored_levels.at(level_index) = *scored;
    slot_cursor += level_slot_count;
  }
  return slot_cursor == solution.slots.size() ? std::optional<std::vector<ScoredSegment>>(std::move(scored_levels)) : std::nullopt;
}

auto BuildCandidateFromScoredLevels(const AnalyticalHTreeSolveProblem& solve_problem,
                                    const std::vector<PatternId>& level_segment_pattern_ids,
                                    const std::vector<ScoredSegment>& scored_levels, AnalyticalSolverBuild& result)
    -> std::optional<AnalyticalCandidate>
{
  if (solve_problem.levels == nullptr || level_segment_pattern_ids.size() != scored_levels.size() || scored_levels.empty()) {
    return std::nullopt;
  }
  ++result.summary.materialization_attempt_count;

  AnalyticalCandidate candidate;
  candidate.depth = static_cast<unsigned>(solve_problem.levels->size());
  candidate.leaf_load_cap_pf = solve_problem.config.representative_leaf_load_cap_pf;
  candidate.root_input_slew_ns = solve_problem.config.root_input_slew_ns;
  candidate.leaf_count = candidate.depth >= sizeof(std::size_t) * 8U ? 0U : (std::size_t{1U} << candidate.depth);
  candidate.level_segment_pattern_ids = level_segment_pattern_ids;
  candidate.output_slew_ns = scored_levels.back().output_slew_ns;
  candidate.root_source_cap_pf = scored_levels.front().source_cap_pf;
  candidate.conservative_slew_ns = candidate.output_slew_ns;
  candidate.branch_buffer_legal = true;
  candidate.fanout_legal = true;
  candidate.trace.reserve(scored_levels.size());

  for (std::size_t level_index = 0U; level_index < scored_levels.size(); ++level_index) {
    const auto& scored = scored_levels.at(level_index);
    candidate.trace.push_back(MakeSegmentChoice(level_index, scored));
    candidate.raw_delay_ns += scored.delay_ns;
    candidate.raw_power_w = AccumulateHTreePower(candidate.raw_power_w, level_index, scored);
    candidate.conservative_delay_ns += scored.delay_upper_ns;
    candidate.conservative_power_w = AccumulateHTreePower(candidate.conservative_power_w, level_index, scored);
  }

  const auto* segment_pattern_library = ResolveSegmentPatternLibrary(solve_problem);
  if (segment_pattern_library == nullptr) {
    candidate.rejection_reason = "missing_segment_pattern_library";
    return std::nullopt;
  }
  auto topology_pattern_library = BuildAnalyticalTopologyPattern(candidate.level_segment_pattern_ids, *segment_pattern_library,
                                                                 solve_problem.fanout_config.max_fanout);
  if (!topology_pattern_library.has_value()) {
    candidate.rejection_reason = "topology_pattern_composition_illegal";
    ++result.summary.root_fanout_rejected_count;
    return std::nullopt;
  }

  candidate.topology_pattern_library = std::move(*topology_pattern_library);
  const PatternId topology_pattern_id = PatternId::topology(
      candidate.topology_pattern_library.nodes.empty() ? 0U : static_cast<unsigned>(candidate.topology_pattern_library.nodes.size() - 1U));
  const auto composition_state = candidate.topology_pattern_library.getCompositionState(topology_pattern_id);
  candidate.fanout_legal = IsBinarySourceFanoutLegal(composition_state.source_exposed_load_count, solve_problem.fanout_config.max_fanout);
  if (!candidate.fanout_legal) {
    candidate.rejection_reason = "root_fanout_illegal";
    ++result.summary.root_fanout_rejected_count;
    return std::nullopt;
  }

  candidate.materialized_char = MaterializeAnalyticalTopologyChar(candidate, solve_problem.slew_lattice, solve_problem.cap_lattice);
  if (!candidate.materialized_char.has_value()) {
    candidate.rejection_reason = "materialized_char_out_of_lattice";
    ++result.summary.lattice_rejected_count;
    return std::nullopt;
  }
  return candidate;
}

auto SolveMathHtreeProblem(const MathHtreeProblem& problem) -> MathHtreeSolution
{
  MathHtreeMilpSolver solver;
  return solver.solveNormalizedTradeoff(problem);
}

}  // namespace

auto SolveAnalyticalHTreeCandidates(const AnalyticalHTreeSolveProblem& solve_problem) -> AnalyticalSolverBuild
{
  const auto validation_failure = ValidateSolveProblem(solve_problem);
  if (!validation_failure.empty()) {
    return MakeFailure(validation_failure);
  }

  auto build_result = BuildMathHtreeProblem(solve_problem);
  if (!build_result.success) {
    return MakeFailure(build_result.failure_reason);
  }

  const auto solution = SolveMathHtreeProblem(build_result.problem);
  AnalyticalSolverBuild result;
  result.summary.backend_name = solution.backend_name;
  result.summary.solver_status = ToString(solution.status);
  result.summary.solver_variable_count = solution.variable_count;
  result.summary.solver_binary_variable_count = solution.binary_variable_count;
  result.summary.solver_continuous_variable_count = solution.continuous_variable_count;
  result.summary.solver_constraint_count = solution.constraint_count;
  result.summary.solver_wall_time_ms = solution.solve_wall_time_ms;
  result.summary.solver_objective_value = solution.objective_value;
  result.summary.solver_optimality_gap = solution.optimality_gap;
  result.summary.solver_primal_bound = solution.primal_bound;
  result.summary.solver_dual_bound = solution.dual_bound;
  result.summary.solver_min_delay_anchor_ns = solution.min_delay_anchor_ns;
  result.summary.solver_min_power_anchor_w = solution.min_power_anchor_w;
  result.summary.solver_total_delay_ns = solution.total_delay_ns;
  result.summary.solver_total_power_w = solution.total_power_w;
  if (!solution.hasUsableIntegerSolution()) {
    result.summary.success = false;
    result.summary.failure_reason = solution.failure_reason.empty() ? ToString(solution.status) : solution.failure_reason;
    return result;
  }

  auto selected_unit_pattern_ids = SelectedUnitPatternIds(build_result, solution);
  if (selected_unit_pattern_ids.empty()) {
    return MarkFailure(result, "invalid_math_solution_choice_mapping");
  }
  auto level_unit_pattern_ids = GroupUnitPatternIdsByLevel(selected_unit_pattern_ids, build_result.level_slot_counts);
  if (level_unit_pattern_ids.empty()) {
    return MarkFailure(result, "invalid_math_solution_level_grouping");
  }

  auto functional_context = MakeFunctionalContext(solve_problem);
  auto level_segment_pattern_ids = MaterializeLevelSegmentPatterns(solve_problem, level_unit_pattern_ids, functional_context);
  if (!level_segment_pattern_ids.has_value()) {
    return MarkFailure(result, "math_solution_segment_materialization_failed");
  }

  auto scored_levels = ScoreSelectedLevelSegments(solve_problem, level_unit_pattern_ids, *level_segment_pattern_ids,
                                                  build_result.level_slot_counts, solution, result);
  if (!scored_levels.has_value()) {
    return MarkFailure(result, "math_solution_level_scoring_failed");
  }
  auto candidate = BuildCandidateFromScoredLevels(solve_problem, *level_segment_pattern_ids, *scored_levels, result);
  if (!candidate.has_value()) {
    return MarkFailure(result,
                       result.summary.root_fanout_rejected_count > 0U ? "root_fanout_illegal" : "math_candidate_materialization_failed");
  }

  result.output.candidates.push_back(std::move(*candidate));
  result.summary.generated_candidate_count = result.output.candidates.size();
  result.summary.evaluated_segment_count = solution.slots.size();
  result.summary.scored_segment_count = scored_levels->size();
  result.summary.success = true;
  return result;
}

}  // namespace icts::htree::analytical_solver
