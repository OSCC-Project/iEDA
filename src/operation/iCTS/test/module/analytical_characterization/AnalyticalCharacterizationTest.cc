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
 * @file AnalyticalCharacterizationTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-14
 * @brief Unit tests for analytical characterization catalog construction.
 */

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ClockRouteSegmentRc.hh"
#include "analytical_characterization/AnalyticalCharacterization.hh"
#include "analytical_characterization/AnalyticalModel.hh"
#include "database/characterization/BufferingPattern.hh"
#include "database/characterization/CharCore.hh"
#include "database/characterization/PatternId.hh"
#include "database/characterization/SegmentChar.hh"
#include "database/characterization/ValueLattice.hh"

namespace icts_test {
namespace {

auto MakeSegmentChar(unsigned input_slew_idx, unsigned output_slew_idx, unsigned driven_cap_idx, unsigned load_cap_idx,
                     icts::PatternId pattern_id, unsigned length_idx) -> icts::SegmentChar
{
  const double input_slew = static_cast<double>(input_slew_idx) * 0.01;
  const double load_cap = static_cast<double>(load_cap_idx) * 0.02;
  const double delay = 0.1 + input_slew + load_cap;
  const double power = 1e-6 + 1e-7 * input_slew_idx + 2e-7 * load_cap_idx;
  const double source_boundary_power = 0.25 * power;
  return icts::SegmentChar(
      icts::CharCore(input_slew_idx, output_slew_idx, driven_cap_idx, load_cap_idx, delay, power, pattern_id, source_boundary_power),
      length_idx);
}

TEST(AnalyticalCharacterizationTest, BuildsCatalogFromSyntheticSegmentRows)
{
  const auto pattern_id = icts::PatternId::segment(7U);
  const unsigned length_idx = 1U;
  const std::vector<icts::BufferingPattern> patterns = {
      icts::BufferingPattern(length_idx, pattern_id, {}, {}, false),
  };

  std::vector<icts::SegmentChar> chars;
  for (unsigned slew_idx = 1U; slew_idx <= 3U; ++slew_idx) {
    for (unsigned cap_idx = 1U; cap_idx <= 3U; ++cap_idx) {
      chars.push_back(MakeSegmentChar(slew_idx, slew_idx + cap_idx, cap_idx + 1U, cap_idx, pattern_id, length_idx));
    }
  }

  icts::analytical::AnalyticalCharacterizationConfig config;
  config.require_monotonic_power = true;
  config.require_monotonic_source_boundary_power = true;
  const auto result = icts::analytical::AnalyticalCharacterization::buildFromSegmentChars(
      chars, patterns, icts::UniformValueLattice(0.01, 16U), icts::UniformValueLattice(0.02, 16U), config);

  ASSERT_TRUE(result.summary.success);
  EXPECT_EQ(result.summary.model_set_count, 1U);
  EXPECT_EQ(result.summary.structural_cap_operator_count, 1U);
  const auto* model_set = result.output.catalog.find(pattern_id, length_idx);
  ASSERT_NE(model_set, nullptr);
  ASSERT_TRUE(model_set->isComplete());
  if (!model_set->source_cap_operator.has_value()) {
    ADD_FAILURE() << "Expected source cap operator.";
    return;
  }
  const auto& source_cap_operator = model_set->source_cap_operator.value();
  EXPECT_FALSE(source_cap_operator.physical);
  EXPECT_TRUE(source_cap_operator.bucket_compatible);
}

TEST(AnalyticalCharacterizationTest, ExactStructuralCapUsesExplicitRouteRcAndBufferCatalog)
{
  const auto pattern_id = icts::PatternId::segment(8U);
  const unsigned length_idx = 2U;
  const std::vector<icts::BufferingPattern> patterns = {
      icts::BufferingPattern(length_idx, pattern_id, {0.5}, {"BUF_X1"}, false),
  };

  std::vector<icts::SegmentChar> chars;
  for (unsigned slew_idx = 1U; slew_idx <= 3U; ++slew_idx) {
    for (unsigned cap_idx = 1U; cap_idx <= 3U; ++cap_idx) {
      chars.push_back(MakeSegmentChar(slew_idx, slew_idx + cap_idx, 2U, cap_idx, pattern_id, length_idx));
    }
  }

  icts::analytical::AnalyticalCharacterizationConfig config;
  config.prefer_exact_structural_cap = true;
  config.length_unit_um = 10.0;
  config.clock_route_segment_rc = icts::ClockRouteSegmentRc{
      .dbu_per_um = 1000,
      .resistance_per_um_ohm = 0.001,
      .capacitance_per_um_pf = 0.001,
  };
  config.buffer_input_cap_pf_by_cell_master.emplace("BUF_X1", 0.03);
  config.require_monotonic_power = true;
  config.require_monotonic_source_boundary_power = true;

  const auto result = icts::analytical::AnalyticalCharacterization::buildFromSegmentChars(
      chars, patterns, icts::UniformValueLattice(0.01, 16U), icts::UniformValueLattice(0.02, 16U), config);

  ASSERT_TRUE(result.summary.success);
  const auto* model_set = result.output.catalog.find(pattern_id, length_idx);
  ASSERT_NE(model_set, nullptr);
  if (!model_set->source_cap_operator.has_value()) {
    ADD_FAILURE() << "Expected exact buffered source-cap operator.";
    return;
  }
  const auto& source_cap_operator = model_set->source_cap_operator.value();
  EXPECT_TRUE(source_cap_operator.physical);
  EXPECT_TRUE(source_cap_operator.bucket_compatible);
  EXPECT_EQ(source_cap_operator.source, "exact_buffered");
  EXPECT_DOUBLE_EQ(source_cap_operator.alpha, 0.0);
  EXPECT_NEAR(source_cap_operator.eta_pf, 0.04, 1e-12);
}

TEST(AnalyticalCharacterizationTest, PreservesPatternAndLengthKey)
{
  const auto pattern_a = icts::PatternId::segment(1U);
  const auto pattern_b = icts::PatternId::segment(2U);
  const std::vector<icts::BufferingPattern> patterns = {
      icts::BufferingPattern(1U, pattern_a, {}, {}, false),
      icts::BufferingPattern(2U, pattern_b, {}, {}, false),
  };

  std::vector<icts::SegmentChar> chars;
  for (unsigned slew_idx = 1U; slew_idx <= 3U; ++slew_idx) {
    for (unsigned cap_idx = 1U; cap_idx <= 3U; ++cap_idx) {
      chars.push_back(MakeSegmentChar(slew_idx, slew_idx + cap_idx, cap_idx + 1U, cap_idx, pattern_a, 1U));
      chars.push_back(MakeSegmentChar(slew_idx, slew_idx + cap_idx, cap_idx + 2U, cap_idx, pattern_b, 2U));
    }
  }

  const auto result = icts::analytical::AnalyticalCharacterization::buildFromSegmentChars(
      chars, patterns, icts::UniformValueLattice(0.01, 16U), icts::UniformValueLattice(0.02, 16U),
      icts::analytical::AnalyticalCharacterizationConfig{});

  ASSERT_TRUE(result.summary.success);
  EXPECT_NE(result.output.catalog.find(pattern_a, 1U), nullptr);
  EXPECT_NE(result.output.catalog.find(pattern_b, 2U), nullptr);
  EXPECT_EQ(result.output.catalog.find(pattern_a, 2U), nullptr);
  EXPECT_EQ(result.output.catalog.find(pattern_b, 1U), nullptr);
}

}  // namespace
}  // namespace icts_test
