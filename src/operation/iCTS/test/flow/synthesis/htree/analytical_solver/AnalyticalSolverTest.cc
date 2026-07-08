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
 * @file AnalyticalSolverTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-26
 * @brief Unit tests for the mathematical analytical H-tree solver.
 */

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

#include "Flow.hh"
#include "analytical_characterization/AnalyticalModel.hh"
#include "common/CTSTestRuntime.hh"
#include "database/characterization/BufferingPattern.hh"
#include "database/characterization/HTreeTopologyChar.hh"
#include "database/characterization/PatternId.hh"
#include "database/characterization/ValueLattice.hh"
#include "synthesis/htree/HTree.hh"
#include "synthesis/htree/analytical_solver/AnalyticalSolver.hh"
#include "synthesis/htree/analytical_solver/candidate/AnalyticalCandidate.hh"
#include "synthesis/htree/constraint/Constraint.hh"
#include "synthesis/htree/segment_pruning/SegmentPatternLibrary.hh"
#include "synthesis/htree/segment_pruning/TopologyPatternLibrary.hh"
#include "synthesis/htree/topology_pruning/TopologyPruning.hh"

namespace icts_test {
namespace {

auto MakeAcceptedModel(icts::analytical::AnalyticalMetric metric, double constant) -> icts::analytical::AnalyticalSurfaceModel
{
  icts::analytical::AnalyticalSurfaceModel model;
  model.metric = metric;
  model.basis = icts::analytical::AnalyticalModelBasis::kAffine;
  model.domain = icts::analytical::AnalyticalDomain{.slew_min_ns = 0.01, .slew_max_ns = 0.20, .cap_min_pf = 0.01, .cap_max_pf = 0.50};
  model.coefficients = {constant, 0.0, 0.0};
  model.quality.accepted = true;
  model.quality.sample_count = 9U;
  return model;
}

auto MakeModelSet(icts::PatternId pattern_id, unsigned length_idx, double delay_ns, double power_w) -> icts::analytical::AnalyticalModelSet
{
  return icts::analytical::AnalyticalModelSet{
      .key = icts::analytical::AnalyticalModelKey{.pattern_id = pattern_id, .length_idx = length_idx},
      .output_slew_model = MakeAcceptedModel(icts::analytical::AnalyticalMetric::kOutputSlew, 0.04),
      .delay_model = MakeAcceptedModel(icts::analytical::AnalyticalMetric::kDelay, delay_ns),
      .power_model = MakeAcceptedModel(icts::analytical::AnalyticalMetric::kPower, power_w),
      .source_boundary_power_model = MakeAcceptedModel(icts::analytical::AnalyticalMetric::kSourceBoundaryNetSwitchPower, power_w * 0.25),
      .source_cap_operator = icts::analytical::StructuralCapOperator::wire(0.01),
  };
}

auto MakeQuadraticModelSet(icts::PatternId pattern_id, unsigned length_idx) -> icts::analytical::AnalyticalModelSet
{
  auto model_set = MakeModelSet(pattern_id, length_idx, 0.10, 1e-6);
  if (model_set.delay_model.has_value()) {
    model_set.delay_model->basis = icts::analytical::AnalyticalModelBasis::kQuadratic;
    model_set.delay_model->coefficients = {0.10, 0.0, 0.0, 0.0, 0.0, 0.0};
  }
  return model_set;
}

auto MakeBoundaryState(bool source_has_buffer, bool sink_has_buffer) -> icts::MonotonicBoundaryState
{
  return icts::MonotonicBoundaryState{
      .source = icts::BoundaryBufferState{.has_buffer = source_has_buffer, .strength_rank = source_has_buffer ? 1U : 0U},
      .sink = icts::BoundaryBufferState{.has_buffer = sink_has_buffer, .strength_rank = sink_has_buffer ? 1U : 0U},
  };
}

auto MakeLevelPlan(unsigned length_idx, bool leaf_level = false) -> icts::HTree::LevelPlan
{
  return icts::HTree::LevelPlan{
      .requested_length_dbu = 0,
      .requested_length_um = 0.0,
      .aligned_length_idx = length_idx,
      .aligned_length_um = 0.0,
      .is_leaf_level = leaf_level,
      .selected_has_any_buffer = false,
      .selected_has_terminal_branch_buffer = false,
      .selected_leaf_buffer_cell_master = {},
      .selected_terminal_cell_master = {},
      .selected_buffer_count = 0U,
      .selected_buffer_area_um2 = 0.0,
      .selected_weighted_buffer_count = 0U,
      .selected_weighted_buffer_area_um2 = 0.0,
      .segment_pattern_id = icts::PatternId::segment(0U),
  };
}

auto MakeSolveProblem(std::vector<icts::HTree::LevelPlan>& levels, icts::htree::BufferPatternLibrary& segment_patterns,
                      icts::analytical::AnalyticalModelCatalog& catalog) -> icts::htree::analytical_solver::AnalyticalHTreeSolveProblem
{
  icts::htree::analytical_solver::AnalyticalHTreeSolveProblem solve_problem;
  solve_problem.levels = &levels;
  solve_problem.segment_pattern_library = &segment_patterns;
  solve_problem.mutable_segment_pattern_library = &segment_patterns;
  solve_problem.model_catalog = &catalog;
  solve_problem.slew_lattice = icts::UniformValueLattice(0.01, 20U);
  solve_problem.cap_lattice = icts::UniformValueLattice(0.01, 50U);
  solve_problem.config.root_input_slew_ns = 0.02;
  solve_problem.config.representative_leaf_load_cap_pf = 0.05;
  solve_problem.config.use_functional_unit_compose = true;
  solve_problem.config.unit_length_idx = 1U;
  return solve_problem;
}

TEST(AnalyticalSolverTest, SolvesNormalizedTradeoffWithContinuousUnitSlots)
{
  const auto pattern_fast = icts::PatternId::segment(1U);
  const auto pattern_low_power = icts::PatternId::segment(2U);
  const unsigned length_idx = 1U;

  icts::htree::BufferPatternLibrary segment_patterns(icts_test::runtime::CurrentRuntime().wrapper);
  segment_patterns.add(icts::BufferingPattern(length_idx, pattern_fast, {}, {}, false));
  segment_patterns.add(icts::BufferingPattern(length_idx, pattern_low_power, {}, {}, false));

  icts::analytical::AnalyticalModelCatalog catalog;
  catalog.addModelSet(MakeModelSet(pattern_fast, length_idx, 0.10, 2e-6));
  catalog.addModelSet(MakeModelSet(pattern_low_power, length_idx, 0.30, 1e-6));

  std::vector<icts::HTree::LevelPlan> levels = {
      MakeLevelPlan(length_idx),
  };
  auto solve_problem = MakeSolveProblem(levels, segment_patterns, catalog);

  const auto result = icts::htree::analytical_solver::SolveAnalyticalHTreeCandidates(solve_problem);

  ASSERT_TRUE(result.summary.success) << result.summary.failure_reason;
  ASSERT_EQ(result.output.candidates.size(), 1U);
  EXPECT_EQ(result.output.candidates.front().level_segment_pattern_ids.front(), pattern_fast);
  EXPECT_TRUE(result.output.candidates.front().materialized_char.has_value());
  EXPECT_GT(result.summary.solver_variable_count, 0U);
  EXPECT_GT(result.summary.solver_binary_variable_count, 0U);
  EXPECT_GT(result.summary.solver_constraint_count, 0U);
  EXPECT_EQ(result.summary.backend_name, "HiGHS in-process");
  EXPECT_FALSE(result.summary.solver_status.empty());
  EXPECT_NEAR(result.summary.solver_min_delay_anchor_ns, 0.10, 1e-9);
  EXPECT_NEAR(result.summary.solver_min_power_anchor_w, 1e-6, 1e-15);
}

TEST(AnalyticalSolverTest, ComposesUnitSlotsIntoDiscreteSegmentPattern)
{
  const auto unit_pattern = icts::PatternId::segment(1U);
  const unsigned unit_length_idx = 1U;
  const unsigned level_length_idx = 2U;

  icts::htree::BufferPatternLibrary segment_patterns(icts_test::runtime::CurrentRuntime().wrapper);
  segment_patterns.add(icts::BufferingPattern(unit_length_idx, unit_pattern, {}, {}, false));

  icts::analytical::AnalyticalModelCatalog catalog;
  catalog.addModelSet(MakeModelSet(unit_pattern, unit_length_idx, 0.10, 1e-6));

  std::vector<icts::HTree::LevelPlan> levels = {
      MakeLevelPlan(level_length_idx),
  };
  auto solve_problem = MakeSolveProblem(levels, segment_patterns, catalog);

  const auto result = icts::htree::analytical_solver::SolveAnalyticalHTreeCandidates(solve_problem);

  ASSERT_TRUE(result.summary.success) << result.summary.failure_reason;
  ASSERT_EQ(result.output.candidates.size(), 1U);
  const auto materialized_pattern_id = result.output.candidates.front().level_segment_pattern_ids.front();
  EXPECT_NE(materialized_pattern_id, unit_pattern);
  const auto* materialized_pattern = segment_patterns.find(materialized_pattern_id);
  ASSERT_NE(materialized_pattern, nullptr);
  EXPECT_EQ(materialized_pattern->get_length_idx(), level_length_idx);
  EXPECT_NEAR(result.output.candidates.front().raw_delay_ns, 0.20, 1e-12);
  EXPECT_TRUE(result.output.candidates.front().materialized_char.has_value());
}

TEST(AnalyticalSolverTest, RejectsNonAffineModelSet)
{
  const auto pattern_id = icts::PatternId::segment(1U);
  const unsigned length_idx = 1U;

  icts::htree::BufferPatternLibrary segment_patterns(icts_test::runtime::CurrentRuntime().wrapper);
  segment_patterns.add(icts::BufferingPattern(length_idx, pattern_id, {}, {}, false));

  icts::analytical::AnalyticalModelCatalog catalog;
  catalog.addModelSet(MakeQuadraticModelSet(pattern_id, length_idx));

  std::vector<icts::HTree::LevelPlan> levels = {
      MakeLevelPlan(length_idx),
  };
  auto solve_problem = MakeSolveProblem(levels, segment_patterns, catalog);

  const auto result = icts::htree::analytical_solver::SolveAnalyticalHTreeCandidates(solve_problem);

  EXPECT_FALSE(result.summary.success);
  EXPECT_EQ(result.summary.failure_reason, "empty_affine_unit_choice_set");
}

TEST(AnalyticalSolverTest, RespectsTerminalBranchBufferConstraint)
{
  const auto pattern_plain = icts::PatternId::segment(1U);
  const auto pattern_branch = icts::PatternId::segment(2U);
  const unsigned length_idx = 1U;

  icts::htree::BufferPatternLibrary segment_patterns(icts_test::runtime::CurrentRuntime().wrapper);
  segment_patterns.add(icts::BufferingPattern(length_idx, pattern_plain, {}, {}, false));
  segment_patterns.add(icts::BufferingPattern(length_idx, pattern_branch, {1.0}, {"BUF_X1"}, true, MakeBoundaryState(true, true)));

  icts::analytical::AnalyticalModelCatalog catalog;
  catalog.addModelSet(MakeModelSet(pattern_plain, length_idx, 0.05, 1e-6));
  catalog.addModelSet(MakeModelSet(pattern_branch, length_idx, 0.20, 2e-6));

  std::vector<icts::HTree::LevelPlan> levels = {
      MakeLevelPlan(length_idx),
  };
  auto solve_problem = MakeSolveProblem(levels, segment_patterns, catalog);
  solve_problem.boundary_constraints.force_branch_buffer = true;
  solve_problem.fanout_config.max_fanout = 2U;

  const auto result = icts::htree::analytical_solver::SolveAnalyticalHTreeCandidates(solve_problem);

  ASSERT_TRUE(result.summary.success) << result.summary.failure_reason;
  ASSERT_EQ(result.output.candidates.size(), 1U);
  EXPECT_EQ(result.output.candidates.front().level_segment_pattern_ids.front(), pattern_branch);
  EXPECT_TRUE(result.output.candidates.front().fanout_legal);
}

TEST(AnalyticalSolverTest, AnalyticalTopologyPatternRejectsInternalFanoutHiddenByRootBuffer)
{
  const auto plain_pattern = icts::PatternId::segment(1U);
  const auto root_buffer_pattern = icts::PatternId::segment(2U);
  const unsigned length_idx = 1U;

  icts::htree::BufferPatternLibrary segment_patterns(icts_test::runtime::CurrentRuntime().wrapper);
  segment_patterns.add(icts::BufferingPattern(length_idx, plain_pattern, {}, {}, false));
  segment_patterns.add(icts::BufferingPattern(length_idx, root_buffer_pattern, {0.1}, {"BUF_X1"}, false, MakeBoundaryState(true, true)));

  const auto topology_pattern = icts::htree::analytical_solver::BuildAnalyticalTopologyPattern(
      {root_buffer_pattern, plain_pattern, plain_pattern}, segment_patterns, 2U);

  EXPECT_FALSE(topology_pattern.has_value());
}

}  // namespace
}  // namespace icts_test
