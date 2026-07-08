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
 * @file CharacterizationRealTechFunctionComposeRegressionTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Function compose regression coverage on real-tech assets.
 */

#include <gtest/gtest.h>

#include <iomanip>
#include <optional>
#include <sstream>

#include "module/characterization/CharacterizationRealTechExactRegression.hh"

namespace icts_test {
namespace {

namespace realtech_fixture = characterization::realtech;

TEST(CharacterizationRealTechExactRegressionTest, IterOneFunctionComposeGapReport)
{
  realtech_fixture::RealTechCharFixture char_fixture;
  if (const auto prepare_error = char_fixture.prepare("iter1_function_compose_gap", std::nullopt, 0.0, 0.0); prepare_error.has_value()) {
    GTEST_SKIP() << *prepare_error;
    return;
  }

  icts::CharBuilder builder;
  const auto contract = MakeIterOneExperimentCharBuilderContract();
  builder.init(contract.input, contract.config);
  builder.build();

  ASSERT_FALSE(builder.get_segment_chars().empty());
  auto segment_context = realtech_fixture::BuildSegmentFrontierContext(builder.get_buffering_patterns());
  const auto grid = realtech_fixture::CalcCharGrid(builder);
  ASSERT_GT(grid.length_step_um, 0.0);
  ASSERT_GT(grid.slew_step_ns, 0.0);
  ASSERT_GT(grid.cap_step_pf, 0.0);

  auto direct_frontier_by_length = realtech_fixture::BuildSegmentLengthFrontiers(builder.get_segment_chars(), segment_context);
  ASSERT_TRUE(direct_frontier_by_length.contains(2U));
  ASSERT_TRUE(direct_frontier_by_length.contains(3U));
  ASSERT_FALSE(direct_frontier_by_length.at(2U).empty());
  ASSERT_FALSE(direct_frontier_by_length.at(3U).empty());

  const auto linear_models = BuildFunctionalSurfaceModels(builder.get_segment_chars(), segment_context, grid, FitBasisKind::kLinear);
  const auto quadratic_models = BuildFunctionalSurfaceModels(builder.get_segment_chars(), segment_context, grid, FitBasisKind::kQuadratic);
  ASSERT_FALSE(linear_models.empty());
  ASSERT_FALSE(quadratic_models.empty());

  const auto linear_raw_length_two
      = AnalyzeFunctionComposeGap("raw", 2U, FitBasisKind::kLinear, builder.get_segment_chars(), segment_context, linear_models, grid,
                                  builder.get_max_slew(), builder.get_max_cap());
  const auto linear_raw_length_three
      = AnalyzeFunctionComposeGap("raw", 3U, FitBasisKind::kLinear, builder.get_segment_chars(), segment_context, linear_models, grid,
                                  builder.get_max_slew(), builder.get_max_cap());
  const auto quadratic_raw_length_two
      = AnalyzeFunctionComposeGap("raw", 2U, FitBasisKind::kQuadratic, builder.get_segment_chars(), segment_context, quadratic_models, grid,
                                  builder.get_max_slew(), builder.get_max_cap());
  const auto quadratic_raw_length_three
      = AnalyzeFunctionComposeGap("raw", 3U, FitBasisKind::kQuadratic, builder.get_segment_chars(), segment_context, quadratic_models, grid,
                                  builder.get_max_slew(), builder.get_max_cap());
  const auto linear_frontier_length_two
      = AnalyzeFunctionComposeGap("frontier", 2U, FitBasisKind::kLinear, direct_frontier_by_length.at(2U), segment_context, linear_models,
                                  grid, builder.get_max_slew(), builder.get_max_cap());
  const auto linear_frontier_length_three
      = AnalyzeFunctionComposeGap("frontier", 3U, FitBasisKind::kLinear, direct_frontier_by_length.at(3U), segment_context, linear_models,
                                  grid, builder.get_max_slew(), builder.get_max_cap());
  const auto quadratic_frontier_length_two
      = AnalyzeFunctionComposeGap("frontier", 2U, FitBasisKind::kQuadratic, direct_frontier_by_length.at(2U), segment_context,
                                  quadratic_models, grid, builder.get_max_slew(), builder.get_max_cap());
  const auto quadratic_frontier_length_three
      = AnalyzeFunctionComposeGap("frontier", 3U, FitBasisKind::kQuadratic, direct_frontier_by_length.at(3U), segment_context,
                                  quadratic_models, grid, builder.get_max_slew(), builder.get_max_cap());

  ASSERT_GT(linear_raw_length_two.evaluated_count, 0U);
  ASSERT_GT(linear_raw_length_three.evaluated_count, 0U);
  ASSERT_GT(quadratic_raw_length_two.evaluated_count, 0U);
  ASSERT_GT(quadratic_raw_length_three.evaluated_count, 0U);
  ASSERT_GT(linear_frontier_length_two.evaluated_count, 0U);
  ASSERT_GT(linear_frontier_length_three.evaluated_count, 0U);
  ASSERT_GT(quadratic_frontier_length_two.evaluated_count, 0U);
  ASSERT_GT(quadratic_frontier_length_three.evaluated_count, 0U);

  std::ostringstream report_stream;
  report_stream.setf(std::ostringstream::scientific, std::ostringstream::floatfield);
  report_stream << std::setprecision(12);
  report_stream << "scenario=iter1_function_compose_gap\n";
  report_stream << "wirelength_unit_um=" << grid.length_step_um << "\n";
  report_stream << "slew_step_ns=" << grid.slew_step_ns << "\n";
  report_stream << "cap_step_pf=" << grid.cap_step_pf << "\n";
  report_stream << "wirelength_iterations=" << builder.get_wirelength_iterations() << "\n";
  report_stream << "raw_segment_char_count=" << builder.get_segment_chars().size() << "\n";
  report_stream << "direct_frontier_count{length_idx=2,count=" << direct_frontier_by_length.at(2U).size() << "}\n";
  report_stream << "direct_frontier_count{length_idx=3,count=" << direct_frontier_by_length.at(3U).size() << "}\n";
  report_stream << "function_model_count{basis=linear,count=" << linear_models.size() << "}\n";
  report_stream << "function_model_count{basis=quadratic,count=" << quadratic_models.size() << "}\n";
  AppendFunctionComposeGapStats(report_stream, linear_raw_length_two, grid);
  AppendFunctionComposeGapStats(report_stream, linear_raw_length_three, grid);
  AppendFunctionComposeGapStats(report_stream, quadratic_raw_length_two, grid);
  AppendFunctionComposeGapStats(report_stream, quadratic_raw_length_three, grid);
  AppendFunctionComposeGapStats(report_stream, linear_frontier_length_two, grid);
  AppendFunctionComposeGapStats(report_stream, linear_frontier_length_three, grid);
  AppendFunctionComposeGapStats(report_stream, quadratic_frontier_length_two, grid);
  AppendFunctionComposeGapStats(report_stream, quadratic_frontier_length_three, grid);

  ASSERT_TRUE(
      realtech_fixture::WriteScenarioLog("iter1_function_compose_gap", "iter1_function_compose_gap_report.txt", report_stream.str()));
}

TEST(CharacterizationRealTechExactRegressionTest, IterOneStructuralCapFunctionComposeGapReport)
{
  realtech_fixture::RealTechCharFixture char_fixture;
  if (const auto prepare_error = char_fixture.prepare("iter1_structural_cap_function_compose_gap", std::nullopt, 0.0, 0.0);
      prepare_error.has_value()) {
    GTEST_SKIP() << *prepare_error;
    return;
  }

  icts::CharBuilder builder;
  const auto contract = MakeIterOneExperimentCharBuilderContract();
  builder.init(contract.input, contract.config);
  builder.build();

  ASSERT_FALSE(builder.get_segment_chars().empty());
  auto segment_context = realtech_fixture::BuildSegmentFrontierContext(builder.get_buffering_patterns());
  const auto grid = realtech_fixture::CalcCharGrid(builder);
  ASSERT_GT(grid.length_step_um, 0.0);
  ASSERT_GT(grid.slew_step_ns, 0.0);
  ASSERT_GT(grid.cap_step_pf, 0.0);

  auto direct_frontier_by_length = realtech_fixture::BuildSegmentLengthFrontiers(builder.get_segment_chars(), segment_context);
  ASSERT_TRUE(direct_frontier_by_length.contains(2U));
  ASSERT_TRUE(direct_frontier_by_length.contains(3U));
  ASSERT_FALSE(direct_frontier_by_length.at(2U).empty());
  ASSERT_FALSE(direct_frontier_by_length.at(3U).empty());

  const auto linear_models = BuildFunctionalSurfaceModels(builder.get_segment_chars(), segment_context, grid, FitBasisKind::kLinear);
  const auto quadratic_models = BuildFunctionalSurfaceModels(builder.get_segment_chars(), segment_context, grid, FitBasisKind::kQuadratic);
  const auto cap_operators = BuildPhysicalStructuralCapOperators(segment_context, contract.config, grid);
  ASSERT_FALSE(linear_models.empty());
  ASSERT_FALSE(quadratic_models.empty());
  ASSERT_FALSE(cap_operators.empty());

  const auto linear_raw_length_two = AnalyzeStructuralCapFunctionComposeGap(
      "raw_structural_cap", 2U, FitBasisKind::kLinear, builder.get_segment_chars(), segment_context, linear_models, cap_operators, grid,
      builder.get_cap_lattice(), builder.get_max_slew(), builder.get_max_cap());
  const auto linear_raw_length_three = AnalyzeStructuralCapFunctionComposeGap(
      "raw_structural_cap", 3U, FitBasisKind::kLinear, builder.get_segment_chars(), segment_context, linear_models, cap_operators, grid,
      builder.get_cap_lattice(), builder.get_max_slew(), builder.get_max_cap());
  const auto quadratic_raw_length_two = AnalyzeStructuralCapFunctionComposeGap(
      "raw_structural_cap", 2U, FitBasisKind::kQuadratic, builder.get_segment_chars(), segment_context, quadratic_models, cap_operators,
      grid, builder.get_cap_lattice(), builder.get_max_slew(), builder.get_max_cap());
  const auto quadratic_raw_length_three = AnalyzeStructuralCapFunctionComposeGap(
      "raw_structural_cap", 3U, FitBasisKind::kQuadratic, builder.get_segment_chars(), segment_context, quadratic_models, cap_operators,
      grid, builder.get_cap_lattice(), builder.get_max_slew(), builder.get_max_cap());
  const auto linear_frontier_length_two = AnalyzeStructuralCapFunctionComposeGap(
      "frontier_structural_cap", 2U, FitBasisKind::kLinear, direct_frontier_by_length.at(2U), segment_context, linear_models, cap_operators,
      grid, builder.get_cap_lattice(), builder.get_max_slew(), builder.get_max_cap());
  const auto linear_frontier_length_three = AnalyzeStructuralCapFunctionComposeGap(
      "frontier_structural_cap", 3U, FitBasisKind::kLinear, direct_frontier_by_length.at(3U), segment_context, linear_models, cap_operators,
      grid, builder.get_cap_lattice(), builder.get_max_slew(), builder.get_max_cap());
  const auto quadratic_frontier_length_two = AnalyzeStructuralCapFunctionComposeGap(
      "frontier_structural_cap", 2U, FitBasisKind::kQuadratic, direct_frontier_by_length.at(2U), segment_context, quadratic_models,
      cap_operators, grid, builder.get_cap_lattice(), builder.get_max_slew(), builder.get_max_cap());
  const auto quadratic_frontier_length_three = AnalyzeStructuralCapFunctionComposeGap(
      "frontier_structural_cap", 3U, FitBasisKind::kQuadratic, direct_frontier_by_length.at(3U), segment_context, quadratic_models,
      cap_operators, grid, builder.get_cap_lattice(), builder.get_max_slew(), builder.get_max_cap());

  ASSERT_GT(linear_raw_length_two.evaluated_count, 0U);
  ASSERT_GT(linear_raw_length_three.evaluated_count, 0U);
  ASSERT_GT(quadratic_raw_length_two.evaluated_count, 0U);
  ASSERT_GT(quadratic_raw_length_three.evaluated_count, 0U);
  ASSERT_GT(linear_frontier_length_two.evaluated_count, 0U);
  ASSERT_GT(linear_frontier_length_three.evaluated_count, 0U);
  ASSERT_GT(quadratic_frontier_length_two.evaluated_count, 0U);
  ASSERT_GT(quadratic_frontier_length_three.evaluated_count, 0U);

  std::ostringstream report_stream;
  report_stream.setf(std::ostringstream::scientific, std::ostringstream::floatfield);
  report_stream << std::setprecision(12);
  report_stream << "scenario=iter1_structural_cap_function_compose_gap\n";
  report_stream << "wirelength_unit_um=" << grid.length_step_um << "\n";
  report_stream << "slew_step_ns=" << grid.slew_step_ns << "\n";
  report_stream << "cap_step_pf=" << grid.cap_step_pf << "\n";
  report_stream << "wirelength_iterations=" << builder.get_wirelength_iterations() << "\n";
  report_stream << "raw_segment_char_count=" << builder.get_segment_chars().size() << "\n";
  report_stream << "direct_frontier_count{length_idx=2,count=" << direct_frontier_by_length.at(2U).size() << "}\n";
  report_stream << "direct_frontier_count{length_idx=3,count=" << direct_frontier_by_length.at(3U).size() << "}\n";
  report_stream << "function_model_count{basis=linear,count=" << linear_models.size() << "}\n";
  report_stream << "function_model_count{basis=quadratic,count=" << quadratic_models.size() << "}\n";
  AppendStructuralCapOperatorStats(report_stream, cap_operators);
  AppendStructuralCapOperatorSampleGap(report_stream, cap_operators, builder.get_segment_chars(), segment_context, grid,
                                       builder.get_cap_lattice());
  AppendFunctionComposeGapStats(report_stream, linear_raw_length_two, grid);
  AppendFunctionComposeGapStats(report_stream, linear_raw_length_three, grid);
  AppendFunctionComposeGapStats(report_stream, quadratic_raw_length_two, grid);
  AppendFunctionComposeGapStats(report_stream, quadratic_raw_length_three, grid);
  AppendFunctionComposeGapStats(report_stream, linear_frontier_length_two, grid);
  AppendFunctionComposeGapStats(report_stream, linear_frontier_length_three, grid);
  AppendFunctionComposeGapStats(report_stream, quadratic_frontier_length_two, grid);
  AppendFunctionComposeGapStats(report_stream, quadratic_frontier_length_three, grid);

  ASSERT_TRUE(realtech_fixture::WriteScenarioLog("iter1_structural_cap_function_compose_gap",
                                                 "iter1_structural_cap_function_compose_gap_report.txt", report_stream.str()));
}

}  // namespace
}  // namespace icts_test
