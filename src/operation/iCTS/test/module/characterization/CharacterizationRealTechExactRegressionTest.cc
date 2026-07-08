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
 * @file CharacterizationRealTechExactRegressionTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Exact compose and fit-gap regression coverage on real-tech assets.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>

#include "module/characterization/CharacterizationRealTechExactRegression.hh"

namespace icts_test {
namespace {

namespace realtech_fixture = characterization::realtech;

TEST(CharacterizationRealTechExactRegressionTest, ExactComposeAndExactJoinRemainUsable)
{
  realtech_fixture::RealTechCharFixture char_fixture;
  if (const auto prepare_error = char_fixture.prepare("exact_regression", std::nullopt, 0.0, 0.0); prepare_error.has_value()) {
    GTEST_SKIP() << *prepare_error;
    return;
  }

  const auto usable_buffers = realtech_fixture::CollectUsableBufferMasters(realtech_fixture::CollectConfiguredBufferLimitInfo());
  if (usable_buffers.empty()) {
    GTEST_SKIP() << "No configured buffer has both slew and cap limits via port or table limits.";
  }

  icts::CharBuilder builder;
  const auto contract = MakeExactRegressionCharBuilderContract();
  builder.init(contract.input, contract.config);
  builder.build();

  ASSERT_FALSE(builder.get_segment_chars().empty());
  auto segment_context = realtech_fixture::BuildSegmentFrontierContext(builder.get_buffering_patterns());
  const auto lattice_summary = realtech_fixture::SummarizeSegmentCharLattice(builder.get_segment_chars(), builder);
  EXPECT_EQ(lattice_summary.out_of_range_entries, 0U) << realtech_fixture::FormatSegmentCharLatticeSummary(lattice_summary, builder);
  EXPECT_LE(lattice_summary.max_length_idx, builder.get_wirelength_iterations());
  EXPECT_LE(lattice_summary.max_input_slew_idx, builder.get_slew_steps());
  EXPECT_LE(lattice_summary.max_output_slew_idx, builder.get_slew_steps());
  EXPECT_LE(lattice_summary.max_driven_cap_idx, builder.get_cap_steps());
  EXPECT_LE(lattice_summary.max_load_cap_idx, builder.get_cap_steps());
  const auto grid = realtech_fixture::CalcCharGrid(builder);
  ASSERT_GT(grid.length_step_um, 0.0);

  const unsigned length_25_idx = realtech_fixture::MakeLengthIndex(realtech_fixture::kExactLeafLevelLengthUm, grid.length_step_um);
  const unsigned length_50_idx = realtech_fixture::MakeLengthIndex(realtech_fixture::kExactMidLevelLengthUm, grid.length_step_um);
  const unsigned length_100_idx = realtech_fixture::MakeLengthIndex(realtech_fixture::kExactRootLevelLengthUm, grid.length_step_um);

  auto frontier_by_length = realtech_fixture::BuildSegmentLengthFrontiers(builder.get_segment_chars(), segment_context);
  ASSERT_TRUE(frontier_by_length.contains(length_25_idx));
  ASSERT_TRUE(frontier_by_length.contains(length_50_idx));
  ASSERT_TRUE(frontier_by_length.contains(length_100_idx));
  ASSERT_FALSE(frontier_by_length.at(length_25_idx).empty());
  ASSERT_FALSE(frontier_by_length.at(length_50_idx).empty());
  ASSERT_FALSE(frontier_by_length.at(length_100_idx).empty());

  auto exact_segment_100_raw = realtech_fixture::ComposeSegmentEntriesExact(frontier_by_length.at(length_50_idx),
                                                                            frontier_by_length.at(length_50_idx), segment_context);
  ASSERT_FALSE(exact_segment_100_raw.empty());
  auto exact_segment_100_frontier = realtech_fixture::BuildSegmentStateFrontier(exact_segment_100_raw, segment_context);
  ASSERT_FALSE(exact_segment_100_frontier.empty());
  EXPECT_TRUE(std::ranges::all_of(exact_segment_100_frontier, [length_100_idx](const icts::SegmentChar& entry) -> bool {
    return entry.get_length_idx() == length_100_idx;
  }));

  realtech_fixture::HTreeFrontierContext htree_context;

  realtech_fixture::HTreeStageSummary leaf_stage;
  leaf_stage.label = "leaf_25um";
  leaf_stage.raw_entries = realtech_fixture::MakeHTreeSeedEntries(frontier_by_length.at(length_25_idx), segment_context, htree_context);
  leaf_stage.frontier_entries = realtech_fixture::BuildHTreeStateFrontier(leaf_stage.raw_entries, htree_context);
  ASSERT_FALSE(leaf_stage.frontier_entries.empty());

  realtech_fixture::HTreeStageSummary mid_stage;
  mid_stage.label = "mid_50um_to_25um";
  mid_stage.raw_entries = realtech_fixture::ComposeHTreeEntriesExact(
      realtech_fixture::MakeHTreeSeedEntries(frontier_by_length.at(length_50_idx), segment_context, htree_context),
      leaf_stage.frontier_entries, htree_context);
  mid_stage.frontier_entries = realtech_fixture::BuildHTreeStateFrontier(mid_stage.raw_entries, htree_context);
  ASSERT_FALSE(mid_stage.frontier_entries.empty());

  realtech_fixture::HTreeStageSummary root_stage;
  root_stage.label = "root_100um_to_50um_to_25um";
  root_stage.raw_entries = realtech_fixture::ComposeHTreeEntriesExact(
      realtech_fixture::MakeHTreeSeedEntries(frontier_by_length.at(length_100_idx), segment_context, htree_context),
      mid_stage.frontier_entries, htree_context);
  root_stage.frontier_entries = realtech_fixture::BuildHTreeStateFrontier(root_stage.raw_entries, htree_context);
  ASSERT_FALSE(root_stage.frontier_entries.empty());

  EXPECT_GE(exact_segment_100_raw.size(), exact_segment_100_frontier.size());
  EXPECT_GE(mid_stage.raw_entries.size(), mid_stage.frontier_entries.size());
  EXPECT_GE(root_stage.raw_entries.size(), root_stage.frontier_entries.size());

  const auto best_exact_htree = realtech_fixture::SelectBestHTreeChar(root_stage.frontier_entries);
  if (!best_exact_htree.has_value()) {
    GTEST_FAIL() << "Failed to select exact H-tree characterization entry.";
    return;
  }

  std::ostringstream report_stream;
  report_stream.setf(std::ostringstream::fixed, std::ostringstream::floatfield);
  report_stream << std::setprecision(3);
  report_stream << "scenario=exact_regression\n";
  report_stream << "usable_buffers=" << realtech_fixture::JoinStrings(usable_buffers) << "\n";
  report_stream << "segment_char_lattice=" << realtech_fixture::FormatSegmentCharLatticeSummary(lattice_summary, builder) << "\n";
  report_stream << "observed_sample_bounds{output_slew_overflow_samples=" << builder.get_output_slew_overflow_samples()
                << ",max_observed_output_slew_ns=" << builder.get_max_observed_output_slew_ns()
                << ",max_observed_output_slew_idx=" << builder.get_max_observed_output_slew_idx()
                << ",driven_cap_overflow_samples=" << builder.get_driven_cap_overflow_samples()
                << ",max_observed_driven_cap_pf=" << builder.get_max_observed_driven_cap_pf()
                << ",max_observed_driven_cap_idx=" << builder.get_max_observed_driven_cap_idx() << "}\n";
  report_stream << "exact_segment_compose{lhs=50um,rhs=50um,target=100um,raw_count=" << exact_segment_100_raw.size()
                << ",frontier_count=" << exact_segment_100_frontier.size() << "}\n";
  report_stream << "exact_htree_frontier_counts{leaf=" << leaf_stage.frontier_entries.size() << ",mid=" << mid_stage.frontier_entries.size()
                << ",root=" << root_stage.frontier_entries.size() << "}\n";
  realtech_fixture::AppendExamples(
      report_stream, "exact_segment_100_example=", exact_segment_100_frontier,
      [&](const icts::SegmentChar& entry) -> std::string { return realtech_fixture::FormatSegmentChar(entry, grid); });
  realtech_fixture::AppendExamples(
      report_stream, "exact_htree_leaf_example=", leaf_stage.frontier_entries,
      [&](const icts::HTreeTopologyChar& entry) -> std::string { return realtech_fixture::FormatHTreeChar(entry, grid); });
  realtech_fixture::AppendExamples(
      report_stream, "exact_htree_mid_example=", mid_stage.frontier_entries,
      [&](const icts::HTreeTopologyChar& entry) -> std::string { return realtech_fixture::FormatHTreeChar(entry, grid); });
  realtech_fixture::AppendExamples(
      report_stream, "exact_htree_root_example=", root_stage.frontier_entries,
      [&](const icts::HTreeTopologyChar& entry) -> std::string { return realtech_fixture::FormatHTreeChar(entry, grid); });
  report_stream << "best_exact_htree_char=" << realtech_fixture::FormatHTreeChar(best_exact_htree.value(), grid) << "\n";

  ASSERT_TRUE(realtech_fixture::WriteScenarioLog("exact_regression", "exact_regression_report.txt", report_stream.str()));
}

TEST(CharacterizationRealTechExactRegressionTest, IterOneFitAndComposedFrontierGapReport)
{
  realtech_fixture::RealTechCharFixture char_fixture;
  if (const auto prepare_error = char_fixture.prepare("iter1_fit_compose_gap", std::nullopt, 0.0, 0.0); prepare_error.has_value()) {
    GTEST_SKIP() << *prepare_error;
    return;
  }

  icts::CharBuilder builder;
  const auto contract = MakeIterOneExperimentCharBuilderContract();
  builder.init(contract.input, contract.config);
  builder.build();

  ASSERT_FALSE(builder.get_segment_chars().empty());
  auto direct_segment_context = realtech_fixture::BuildSegmentFrontierContext(builder.get_buffering_patterns());
  const auto grid = realtech_fixture::CalcCharGrid(builder);
  ASSERT_GT(grid.length_step_um, 0.0);

  auto direct_frontier_by_length = realtech_fixture::BuildSegmentLengthFrontiers(builder.get_segment_chars(), direct_segment_context);
  ASSERT_TRUE(direct_frontier_by_length.contains(1U));
  ASSERT_TRUE(direct_frontier_by_length.contains(2U));
  ASSERT_TRUE(direct_frontier_by_length.contains(3U));
  ASSERT_FALSE(direct_frontier_by_length.at(1U).empty());
  ASSERT_FALSE(direct_frontier_by_length.at(2U).empty());
  ASSERT_FALSE(direct_frontier_by_length.at(3U).empty());

  auto iter_one_segment_context = realtech_fixture::BuildSegmentFrontierContext(builder.get_buffering_patterns());
  std::unordered_map<unsigned, std::vector<icts::SegmentChar>> iter_one_frontier_by_length;
  iter_one_frontier_by_length[1U] = direct_frontier_by_length.at(1U);
  ASSERT_TRUE(realtech_fixture::SynthesizeSegmentFrontierExactOnly(iter_one_frontier_by_length, 2U, iter_one_segment_context));
  ASSERT_TRUE(realtech_fixture::SynthesizeSegmentFrontierExactOnly(iter_one_frontier_by_length, 3U, iter_one_segment_context));
  ASSERT_FALSE(iter_one_frontier_by_length.at(2U).empty());
  ASSERT_FALSE(iter_one_frontier_by_length.at(3U).empty());

  const auto length_two_gap = CompareComposedFrontierToDirect(2U, direct_frontier_by_length.at(2U), iter_one_frontier_by_length.at(2U),
                                                              iter_one_segment_context, grid);
  const auto length_three_gap = CompareComposedFrontierToDirect(3U, direct_frontier_by_length.at(3U), iter_one_frontier_by_length.at(3U),
                                                                iter_one_segment_context, grid);
  ASSERT_GT(length_two_gap.matched_count, 0U);
  ASSERT_GT(length_three_gap.matched_count, 0U);

  std::ostringstream report_stream;
  report_stream.setf(std::ostringstream::scientific, std::ostringstream::floatfield);
  report_stream << std::setprecision(12);
  report_stream << "scenario=iter1_fit_compose_gap\n";
  report_stream << "wirelength_unit_um=" << grid.length_step_um << "\n";
  report_stream << "slew_step_ns=" << grid.slew_step_ns << "\n";
  report_stream << "cap_step_pf=" << grid.cap_step_pf << "\n";
  report_stream << "wirelength_iterations=" << builder.get_wirelength_iterations() << "\n";
  report_stream << "raw_segment_char_count=" << builder.get_segment_chars().size() << "\n";
  report_stream << "direct_frontier_count{length_idx=1,count=" << direct_frontier_by_length.at(1U).size() << "}\n";
  report_stream << "direct_frontier_count{length_idx=2,count=" << direct_frontier_by_length.at(2U).size() << "}\n";
  report_stream << "direct_frontier_count{length_idx=3,count=" << direct_frontier_by_length.at(3U).size() << "}\n";
  report_stream << "iter1_composed_frontier_count{length_idx=2,count=" << iter_one_frontier_by_length.at(2U).size() << "}\n";
  report_stream << "iter1_composed_frontier_count{length_idx=3,count=" << iter_one_frontier_by_length.at(3U).size() << "}\n";
  AppendIterOneFitReport(report_stream, builder.get_segment_chars(), grid, 1U);
  AppendComposeGapStats(report_stream, length_two_gap, grid);
  AppendComposeGapStats(report_stream, length_three_gap, grid);

  ASSERT_TRUE(realtech_fixture::WriteScenarioLog("iter1_fit_compose_gap", "iter1_fit_compose_gap_report.txt", report_stream.str()));
}

}  // namespace
}  // namespace icts_test
