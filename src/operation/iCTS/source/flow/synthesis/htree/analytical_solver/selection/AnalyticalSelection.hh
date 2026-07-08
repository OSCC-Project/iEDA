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
 * @file AnalyticalSelection.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Analytical H-tree candidate selection contracts.
 */

#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "synthesis/htree/HTree.hh"
#include "synthesis/htree/analytical_solver/candidate/AnalyticalCandidate.hh"
#include "synthesis/htree/analytical_solver/selection/AnalyticalValidation.hh"
#include "synthesis/htree/compensation/RootDriverCompensation.hh"
#include "synthesis/htree/plan/DepthPlan.hh"
#include "synthesis/htree/region/SinkLoadRegion.hh"
#include "synthesis/htree/topology_pruning/TopologyPruning.hh"

namespace icts {

class CharBuilder;
class Tree;

namespace htree {
struct BoundaryConstraints;
struct BufferPatternLibrary;
}  // namespace htree

namespace htree::analytical_selection {

inline constexpr unsigned kAnalyticalUnitLengthIdx = 1U;
inline constexpr double kAnalyticalParetoPowerSlackRatio = 0.20;

struct AnalyticalHTreeAttempt
{
  bool selected = false;
  std::string failure_reason;
  CandidateBuildEvaluation selected_evaluation;
  DepthSummary selected_summary;
  RootDriverCompensationDetail selected_compensation_detail;
  SinkLoadRegionLegalitySummary selected_sink_load_region_legality;
  RootDriverCompensationStats root_driver_compensation_stats;
  std::size_t model_set_count = 0U;
  std::size_t rejected_fit_count = 0U;
  std::size_t structural_cap_operator_count = 0U;
  std::size_t evaluated_segment_count = 0U;
  std::size_t generated_candidate_count = 0U;
  std::size_t validated_candidate_count = 0U;
  std::size_t scored_segment_count = 0U;
  std::size_t missing_model_count = 0U;
  std::size_t decomposition_rejected_count = 0U;
  std::size_t metric_evaluation_rejected_count = 0U;
  std::size_t domain_slew_rejected_count = 0U;
  std::size_t domain_cap_rejected_count = 0U;
  std::size_t domain_slew_floor_count = 0U;
  std::size_t domain_cap_floor_count = 0U;
  double max_domain_rejected_cap_pf = 0.0;
  std::size_t materialization_attempt_count = 0U;
  std::size_t root_fanout_rejected_count = 0U;
  std::size_t lattice_rejected_count = 0U;
  std::string backend_name;
  std::string solver_status;
  std::size_t solver_variable_count = 0U;
  std::size_t solver_binary_variable_count = 0U;
  std::size_t solver_continuous_variable_count = 0U;
  std::size_t solver_constraint_count = 0U;
  double solver_wall_time_ms = 0.0;
  double solver_objective_value = 0.0;
  double solver_optimality_gap = 0.0;
  double solver_primal_bound = std::numeric_limits<double>::quiet_NaN();
  double solver_dual_bound = std::numeric_limits<double>::quiet_NaN();
  double solver_min_delay_anchor_ns = 0.0;
  double solver_min_power_anchor_w = 0.0;
  double solver_total_delay_ns = 0.0;
  double solver_total_power_w = 0.0;
  std::size_t validated_pareto_count = 0U;
  std::size_t selected_pareto_power_rank = 0U;
  double validated_delay_min_ns = 0.0;
  double validated_delay_median_ns = 0.0;
  double validated_delay_max_ns = 0.0;
  double validated_power_min_w = 0.0;
  double validated_power_median_w = 0.0;
  double validated_power_max_w = 0.0;
};

struct AnalyticalValidatedCandidate
{
  analytical_solver::AnalyticalCandidate candidate;
  analytical_solver::AnalyticalValidationSummary validation;
  std::size_t original_index = 0U;
};

auto TrySolveAnalyticalHTree(const Tree& topology, const std::vector<HTree::LevelPlan>& full_level_plans,
                             const std::vector<unsigned>& depth_candidates, BufferPatternLibrary& segment_pattern_library,
                             const BoundaryConstraints& search_boundary_constraints, const HTreeFanoutPruningConfig& fanout_pruning_config,
                             const RootDriverCompensationInput& root_driver_compensation_input,
                             const SinkLoadRegionLegalityInput& sink_load_region_input, const CharBuilder& char_builder,
                             unsigned char_slew_steps) -> AnalyticalHTreeAttempt;
auto ApplyAnalyticalRootDriverStats(DepthSearchBuild& exploration, const AnalyticalHTreeAttempt& attempt,
                                    const RootDriverCompensationInput& compensation_input) -> void;

}  // namespace htree::analytical_selection
}  // namespace icts
