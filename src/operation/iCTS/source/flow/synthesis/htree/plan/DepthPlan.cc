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
 * @file DepthSearch.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief H-tree topology depth search and global frontier pool construction.
 */

#include "synthesis/htree/plan/DepthPlan.hh"

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "HTreeTopologyChar.hh"
#include "ValueLattice.hh"
#include "synthesis/htree/compensation/RootDriverCompensation.hh"
#include "synthesis/htree/constraint/Constraint.hh"
#include "synthesis/htree/plan/Plan.hh"

namespace icts::htree {

auto EvaluateTopologyDepthCandidate(const Tree& topology, const std::vector<HTree::LevelPlan>& full_level_plans, unsigned depth,
                                    const SegmentFrontierCatalog& segment_frontier_catalog, BufferPatternLibrary& segment_pattern_library,
                                    const BoundaryConstraints& base_boundary_constraints,
                                    SinkLoadRegionLegalityContext& sink_load_region_legality_context, unsigned char_slew_steps,
                                    RootDriverCompensationPass& compensation_pass, SchemaWriter& reporter,
                                    const HTreeFanoutPruningConfig& fanout_config) -> DepthCandidateBuild
{
  auto candidate_levels = MakeCandidateLevelPlans(full_level_plans, depth);
  const std::size_t leaf_count = CountCandidateLeafNodes(topology, depth);
  const auto candidate_constraints = base_boundary_constraints;

  DepthCandidateBuild candidate_result;
  candidate_result.leaf_count = leaf_count;
  candidate_result.evaluation = EvaluateCandidateBuild(candidate_levels, segment_frontier_catalog, segment_pattern_library,
                                                       candidate_constraints, topology, sink_load_region_legality_context, leaf_count,
                                                       depth, char_slew_steps, compensation_pass, reporter, fanout_config);
  return candidate_result;
}

auto RecordTopologyDepthCandidateBuild(unsigned depth, bool used_explicit_target_depth, const DepthCandidateBuild& candidate_result,
                                       std::vector<DepthSummary>& depth_summaries) -> void
{
  const auto& evaluation = candidate_result.evaluation;
  depth_summaries.push_back(DepthSummary{
      .depth = depth,
      .leaf_count = candidate_result.leaf_count,
      .success = evaluation.success,
      .selected = false,
      .used_explicit_target_depth = used_explicit_target_depth,
      .failure_reason = evaluation.failure_reason,
      .final_frontier_count = evaluation.final_frontier_count,
      .candidate_solution_count = evaluation.candidate_solution_count,
      .candidate_frontier_entry_count = evaluation.candidate_frontier_entries.size(),
      .feasible_solution_count = evaluation.feasible_solution_count,
      .feasible_frontier_entry_count = evaluation.feasible_frontier_entries.size(),
      .used_boundary_relaxation = evaluation.used_boundary_relaxation,
      .selected_power_w = evaluation.best_char.has_value() ? evaluation.best_char->get_power() : 0.0,
      .selected_delay_ns = evaluation.best_char.has_value() ? evaluation.best_char->get_delay() : 0.0,
  });
}

auto AppendGlobalCandidateRefs(std::size_t candidate_index, const CandidateBuildEvaluation& evaluation,
                               std::vector<CandidateCharRef>& global_feasible_pool, std::vector<CandidateCharRef>& global_candidate_pool)
    -> void
{
  for (const auto& entry : evaluation.feasible_frontier_entries) {
    global_feasible_pool.push_back(CandidateCharRef{
        .candidate_index = candidate_index,
        .entry = &entry,
    });
  }
  for (const auto& entry : evaluation.candidate_frontier_entries) {
    global_candidate_pool.push_back(CandidateCharRef{
        .candidate_index = candidate_index,
        .entry = &entry,
    });
  }
}

auto SearchTopologyDepthCandidates(const Tree& topology, const std::vector<HTree::LevelPlan>& full_level_plans,
                                   const std::vector<unsigned>& depth_candidates, const SegmentFrontierCatalog& segment_frontier_catalog,
                                   BufferPatternLibrary& segment_pattern_library, const BoundaryConstraints& base_boundary_constraints,
                                   const UniformValueLattice& cap_lattice, unsigned char_slew_steps, bool used_explicit_target_depth,
                                   const RootDriverCompensationInput& compensation_input,
                                   const SinkLoadRegionLegalityInput& sink_load_region_input, SchemaWriter& reporter,
                                   const HTreeFanoutPruningConfig& fanout_config) -> DepthSearchBuild
{
  DepthSearchBuild exploration;
  exploration.output.candidate_evaluations.reserve(depth_candidates.size());
  exploration.summary.depth_summaries.reserve(depth_candidates.size());
  exploration.output.sink_load_region_legality_context = SinkLoadRegionLegalityContext{
      .result_by_signature = {},
      .max_monotone_failed_level = std::numeric_limits<int>::min(),
      .cap_lattice = cap_lattice,
      .input = sink_load_region_input,
  };
  RootDriverCompensationPass compensation_pass(compensation_input);

  for (const unsigned depth : depth_candidates) {
    auto candidate_result = EvaluateTopologyDepthCandidate(
        topology, full_level_plans, depth, segment_frontier_catalog, segment_pattern_library, base_boundary_constraints,
        exploration.output.sink_load_region_legality_context, char_slew_steps, compensation_pass, reporter, fanout_config);
    RecordTopologyDepthCandidateBuild(depth, used_explicit_target_depth, candidate_result, exploration.summary.depth_summaries);
    ReduceCandidateBuildEvaluationForGlobalSelection(candidate_result.evaluation, topology, segment_pattern_library,
                                                     exploration.output.sink_load_region_legality_context,
                                                     fanout_config.allow_boundary_relaxation);
    exploration.output.candidate_evaluations.push_back(std::move(candidate_result.evaluation));
    const auto candidate_index = exploration.output.candidate_evaluations.size() - 1U;
    AppendGlobalCandidateRefs(candidate_index, exploration.output.candidate_evaluations.back(), exploration.output.global_feasible_pool,
                              exploration.output.global_candidate_pool);
  }
  exploration.summary.root_driver_compensation_stats = compensation_pass.get_stats();

  return exploration;
}

}  // namespace icts::htree
