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
 * @file DepthSearch.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-01
 * @brief H-tree topology depth search contracts and result data.
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "synthesis/htree/HTree.hh"
#include "synthesis/htree/compensation/RootDriverCompensation.hh"
#include "synthesis/htree/region/SinkLoadRegion.hh"
#include "synthesis/htree/topology_pruning/TopologyPruning.hh"

namespace icts {

class SchemaWriter;
class Tree;
class UniformValueLattice;
}  // namespace icts

namespace icts::htree {

struct BoundaryConstraints;
struct BufferPatternLibrary;
class SegmentFrontierCatalog;

struct DepthSummary
{
  unsigned depth = 0U;
  std::size_t leaf_count = 0U;
  bool success = false;
  bool selected = false;
  bool used_explicit_target_depth = false;
  std::string failure_reason;
  std::size_t htree_load_group_count = 0U;
  double htree_load_cap_min_pf = 0.0;
  double htree_load_cap_max_pf = 0.0;
  double htree_load_cap_mean_pf = 0.0;
  double htree_load_cap_median_pf = 0.0;
  std::size_t final_frontier_count = 0U;
  std::size_t candidate_solution_count = 0U;
  std::size_t candidate_frontier_entry_count = 0U;
  std::size_t feasible_solution_count = 0U;
  std::size_t feasible_frontier_entry_count = 0U;
  bool used_boundary_relaxation = false;
  double selected_power_w = 0.0;
  double selected_delay_ns = 0.0;
};

struct DepthSearchOutput
{
  std::vector<CandidateBuildEvaluation> candidate_evaluations;
  std::vector<CandidateCharRef> global_feasible_pool;
  std::vector<CandidateCharRef> global_candidate_pool;
  SinkLoadRegionLegalityContext sink_load_region_legality_context;
};

struct DepthSearchSummary
{
  std::vector<DepthSummary> depth_summaries;
  RootDriverCompensationStats root_driver_compensation_stats;
};

struct DepthSearchBuild
{
  DepthSearchOutput output;
  DepthSearchSummary summary;
};

struct DepthCandidateBuild
{
  CandidateBuildEvaluation evaluation;
  std::size_t leaf_count = 0U;
};

auto EvaluateTopologyDepthCandidate(const Tree& topology, const std::vector<HTree::LevelPlan>& full_level_plans, unsigned depth,
                                    const SegmentFrontierCatalog& segment_frontier_catalog, BufferPatternLibrary& segment_pattern_library,
                                    const BoundaryConstraints& base_boundary_constraints,
                                    SinkLoadRegionLegalityContext& sink_load_region_legality_context, unsigned char_slew_steps,
                                    RootDriverCompensationPass& compensation_pass, SchemaWriter& reporter,
                                    const HTreeFanoutPruningConfig& fanout_config) -> DepthCandidateBuild;
auto RecordTopologyDepthCandidateBuild(unsigned depth, bool used_explicit_target_depth, const DepthCandidateBuild& candidate_result,
                                       std::vector<DepthSummary>& depth_summaries) -> void;
auto AppendGlobalCandidateRefs(std::size_t candidate_index, const CandidateBuildEvaluation& evaluation,
                               std::vector<CandidateCharRef>& global_feasible_pool, std::vector<CandidateCharRef>& global_candidate_pool)
    -> void;
auto SearchTopologyDepthCandidates(const Tree& topology, const std::vector<HTree::LevelPlan>& full_level_plans,
                                   const std::vector<unsigned>& depth_candidates, const SegmentFrontierCatalog& segment_frontier_catalog,
                                   BufferPatternLibrary& segment_pattern_library, const BoundaryConstraints& base_boundary_constraints,
                                   const UniformValueLattice& cap_lattice, unsigned char_slew_steps, bool used_explicit_target_depth,
                                   const RootDriverCompensationInput& compensation_input,
                                   const SinkLoadRegionLegalityInput& sink_load_region_input, SchemaWriter& reporter,
                                   const HTreeFanoutPruningConfig& fanout_config) -> DepthSearchBuild;
auto EmitDepthCandidateSummary(SchemaWriter& reporter, const std::vector<DepthSummary>& depth_summaries) -> void;

}  // namespace icts::htree
