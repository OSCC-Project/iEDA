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
 * @file AnalyticalSolver.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-14
 * @brief Mathematical analytical H-tree solver entry point and result contracts.
 */

#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "ValueLattice.hh"
#include "synthesis/htree/HTree.hh"
#include "synthesis/htree/analytical_solver/candidate/AnalyticalCandidate.hh"
#include "synthesis/htree/constraint/Constraint.hh"
#include "synthesis/htree/topology_pruning/TopologyPruning.hh"

namespace icts::analytical {
class AnalyticalModelCatalog;
}  // namespace icts::analytical

namespace icts::htree {
struct BufferPatternLibrary;
}  // namespace icts::htree

namespace icts::htree::analytical_solver {

struct AnalyticalSolverConfig
{
  double root_input_slew_ns = 0.0;
  double representative_leaf_load_cap_pf = 0.0;
  bool use_conservative_scoring = true;
  bool use_functional_unit_compose = false;
  unsigned unit_length_idx = 1U;
};

struct AnalyticalHTreeSolveProblem
{
  const std::vector<HTree::LevelPlan>* levels = nullptr;
  const BufferPatternLibrary* segment_pattern_library = nullptr;
  BufferPatternLibrary* mutable_segment_pattern_library = nullptr;
  const icts::analytical::AnalyticalModelCatalog* model_catalog = nullptr;
  BoundaryConstraints boundary_constraints;
  HTreeFanoutPruningConfig fanout_config;
  UniformValueLattice slew_lattice;
  UniformValueLattice cap_lattice;
  AnalyticalSolverConfig config;
};

struct AnalyticalSolverOutput
{
  std::vector<AnalyticalCandidate> candidates;
};

struct AnalyticalSolverSummary
{
  bool success = false;
  std::string failure_reason;
  std::size_t evaluated_segment_count = 0U;
  std::size_t generated_candidate_count = 0U;
  std::size_t materialization_attempt_count = 0U;
  std::size_t scored_segment_count = 0U;
  std::size_t missing_model_count = 0U;
  std::size_t decomposition_rejected_count = 0U;
  std::size_t metric_evaluation_rejected_count = 0U;
  std::size_t domain_slew_rejected_count = 0U;
  std::size_t domain_cap_rejected_count = 0U;
  std::size_t domain_slew_floor_count = 0U;
  std::size_t domain_cap_floor_count = 0U;
  double max_domain_rejected_cap_pf = 0.0;
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
};

struct AnalyticalSolverBuild
{
  AnalyticalSolverOutput output;
  AnalyticalSolverSummary summary;
};

auto SolveAnalyticalHTreeCandidates(const AnalyticalHTreeSolveProblem& solve_problem) -> AnalyticalSolverBuild;

}  // namespace icts::htree::analytical_solver
