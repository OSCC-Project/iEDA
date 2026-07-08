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
 * @file CharacterizationRealTechPowerAccountingTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief Exact compose power-accounting regression coverage on real-tech assets.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

#include "module/characterization/CharacterizationRealTechExactRegression.hh"

namespace icts_test {
namespace {

namespace realtech_fixture = characterization::realtech;

TEST(CharacterizationRealTechExactRegressionTest, ExactComposePowerAccountingProducesComparableDirectReport)
{
  realtech_fixture::RealTechCharFixture char_fixture;
  if (const auto prepare_error = char_fixture.prepare("exact_compose_power_accounting", std::nullopt, 0.0, 0.0);
      prepare_error.has_value()) {
    GTEST_SKIP() << *prepare_error;
    return;
  }

  icts::CharBuilder builder;
  const auto contract = MakeExactRegressionCharBuilderContract();
  builder.init(contract.input, contract.config);
  builder.build();

  ASSERT_FALSE(builder.get_segment_chars().empty());
  auto segment_context = realtech_fixture::BuildSegmentFrontierContext(builder.get_buffering_patterns());
  const auto grid = realtech_fixture::CalcCharGrid(builder);
  ASSERT_GT(grid.length_step_um, 0.0);

  const unsigned length_50_idx = realtech_fixture::MakeLengthIndex(realtech_fixture::kExactMidLevelLengthUm, grid.length_step_um);
  const unsigned length_100_idx = realtech_fixture::MakeLengthIndex(realtech_fixture::kExactRootLevelLengthUm, grid.length_step_um);

  auto frontier_by_length = realtech_fixture::BuildSegmentLengthFrontiers(builder.get_segment_chars(), segment_context);
  ASSERT_TRUE(frontier_by_length.contains(length_50_idx));
  ASSERT_TRUE(frontier_by_length.contains(length_100_idx));
  ASSERT_FALSE(frontier_by_length.at(length_50_idx).empty());
  ASSERT_FALSE(frontier_by_length.at(length_100_idx).empty());

  auto exact_segment_100_raw = realtech_fixture::ComposeSegmentEntriesExact(frontier_by_length.at(length_50_idx),
                                                                            frontier_by_length.at(length_50_idx), segment_context);
  ASSERT_FALSE(exact_segment_100_raw.empty());
  auto exact_segment_100_frontier = realtech_fixture::BuildSegmentStateFrontier(exact_segment_100_raw, segment_context);
  ASSERT_FALSE(exact_segment_100_frontier.empty());

  std::unordered_map<SegmentCompareKey, icts::SegmentChar, SegmentCompareKeyHash> direct_entries;
  direct_entries.reserve(frontier_by_length.at(length_100_idx).size());
  for (const auto& entry : frontier_by_length.at(length_100_idx)) {
    direct_entries.emplace(MakeSegmentCompareKey(segment_context, entry), entry);
  }

  std::size_t matched_entry_count = 0U;
  std::size_t exact_lower_power_count = 0U;
  std::size_t exact_higher_power_count = 0U;
  std::size_t exact_lower_delay_count = 0U;
  std::size_t exact_higher_delay_count = 0U;
  double sum_direct_power_w = 0.0;
  double sum_exact_power_w = 0.0;
  double sum_direct_delay_ns = 0.0;
  double sum_exact_delay_ns = 0.0;
  double worst_power_underestimate_w = 0.0;
  double worst_delay_underestimate_ns = 0.0;
  std::string worst_power_example;
  std::string worst_delay_example;

  for (const auto& exact_entry : exact_segment_100_frontier) {
    const auto direct_it = direct_entries.find(MakeSegmentCompareKey(segment_context, exact_entry));
    if (direct_it == direct_entries.end()) {
      continue;
    }

    const auto& direct_entry = direct_it->second;
    ++matched_entry_count;
    sum_direct_power_w += direct_entry.get_power();
    sum_exact_power_w += exact_entry.get_power();
    sum_direct_delay_ns += direct_entry.get_delay();
    sum_exact_delay_ns += exact_entry.get_delay();

    const double power_delta_w = exact_entry.get_power() - direct_entry.get_power();
    if (power_delta_w < -1e-15) {
      ++exact_lower_power_count;
      const double power_underestimate_w = -power_delta_w;
      if (power_underestimate_w > worst_power_underestimate_w) {
        worst_power_underestimate_w = power_underestimate_w;
        std::ostringstream example;
        example << "key{input_slew_idx=" << exact_entry.get_input_slew_idx() << ",output_slew_idx=" << exact_entry.get_output_slew_idx()
                << ",driven_cap_idx=" << exact_entry.get_driven_cap_idx() << ",load_cap_idx=" << exact_entry.get_load_cap_idx()
                << "} direct=" << realtech_fixture::FormatSegmentChar(direct_entry, grid)
                << " exact=" << realtech_fixture::FormatSegmentChar(exact_entry, grid);
        worst_power_example = example.str();
      }
    } else if (power_delta_w > 1e-15) {
      ++exact_higher_power_count;
    }

    const double delay_delta_ns = exact_entry.get_delay() - direct_entry.get_delay();
    if (delay_delta_ns < -1e-15) {
      ++exact_lower_delay_count;
      const double delay_underestimate_ns = -delay_delta_ns;
      if (delay_underestimate_ns > worst_delay_underestimate_ns) {
        worst_delay_underestimate_ns = delay_underestimate_ns;
        std::ostringstream example;
        example << "key{input_slew_idx=" << exact_entry.get_input_slew_idx() << ",output_slew_idx=" << exact_entry.get_output_slew_idx()
                << ",driven_cap_idx=" << exact_entry.get_driven_cap_idx() << ",load_cap_idx=" << exact_entry.get_load_cap_idx()
                << "} direct=" << realtech_fixture::FormatSegmentChar(direct_entry, grid)
                << " exact=" << realtech_fixture::FormatSegmentChar(exact_entry, grid);
        worst_delay_example = example.str();
      }
    } else if (delay_delta_ns > 1e-15) {
      ++exact_higher_delay_count;
    }
  }

  ASSERT_GT(matched_entry_count, 0U);

  std::ostringstream report_stream;
  report_stream.setf(std::ostringstream::scientific, std::ostringstream::floatfield);
  report_stream << std::setprecision(12);
  report_stream << "scenario=exact_compose_power_accounting\n";
  report_stream << "direct_length_100_frontier_count=" << frontier_by_length.at(length_100_idx).size() << "\n";
  report_stream << "exact_compose_length_100_frontier_count=" << exact_segment_100_frontier.size() << "\n";
  report_stream << "matched_entry_count=" << matched_entry_count << "\n";
  report_stream << "exact_lower_power_count=" << exact_lower_power_count << "\n";
  report_stream << "exact_higher_power_count=" << exact_higher_power_count << "\n";
  report_stream << "exact_lower_delay_count=" << exact_lower_delay_count << "\n";
  report_stream << "exact_higher_delay_count=" << exact_higher_delay_count << "\n";
  report_stream << "sum_direct_power_w=" << sum_direct_power_w << "\n";
  report_stream << "sum_exact_power_w=" << sum_exact_power_w << "\n";
  report_stream << "sum_power_delta_w=" << (sum_exact_power_w - sum_direct_power_w) << "\n";
  report_stream << "power_ratio_exact_over_direct=" << (sum_direct_power_w > 0.0 ? (sum_exact_power_w / sum_direct_power_w) : 0.0) << "\n";
  report_stream << "sum_direct_delay_ns=" << sum_direct_delay_ns << "\n";
  report_stream << "sum_exact_delay_ns=" << sum_exact_delay_ns << "\n";
  report_stream << "sum_delay_delta_ns=" << (sum_exact_delay_ns - sum_direct_delay_ns) << "\n";
  report_stream << "delay_ratio_exact_over_direct=" << (sum_direct_delay_ns > 0.0 ? (sum_exact_delay_ns / sum_direct_delay_ns) : 0.0)
                << "\n";
  report_stream << "worst_power_underestimate_w=" << worst_power_underestimate_w << "\n";
  report_stream << "worst_power_underestimate_example=" << worst_power_example << "\n";
  report_stream << "worst_delay_underestimate_ns=" << worst_delay_underestimate_ns << "\n";
  report_stream << "worst_delay_underestimate_example=" << worst_delay_example << "\n";

  ASSERT_TRUE(realtech_fixture::WriteScenarioLog("exact_compose_power_accounting", "exact_compose_power_accounting_report.txt",
                                                 report_stream.str()));
}

}  // namespace
}  // namespace icts_test
