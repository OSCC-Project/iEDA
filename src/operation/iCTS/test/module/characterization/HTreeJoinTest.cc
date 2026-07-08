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
 * @file HTreeJoinTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief H-tree characterization join tests.
 */

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "characterization/Characterization.hh"
#include "database/characterization/HTreeTopologyChar.hh"
#include "database/characterization/PatternId.hh"
#include "module/characterization/fixture/CharacterizationUnitCaseData.hh"

namespace icts_test {
namespace {

namespace char_cases = characterization;

TEST(HTreeJoinTest, HalfCapJoin)
{
  icts::HTreeTopologyCharTable upstream;
  icts::HTreeTopologyCharTable downstream;

  upstream.addChar(char_cases::MakeHTreeChar(char_cases::kSlew80, char_cases::kSlew100, char_cases::kCap40, char_cases::kCap100,
                                             char_cases::kDelay1p0, char_cases::kPower0p5,
                                             char_cases::HTreeShape{.pattern_id = char_cases::kPattern1, .levels = 1}));

  downstream.addChar(char_cases::MakeHTreeChar(char_cases::kSlew100, char_cases::kSlew120, char_cases::kCap50, char_cases::kCap60,
                                               char_cases::kDelay2p0, char_cases::kPower0p3,
                                               char_cases::HTreeShape{.pattern_id = char_cases::kPattern2, .levels = 1}));
  downstream.addChar(char_cases::MakeHTreeChar(char_cases::kSlew100, char_cases::kSlew130, char_cases::kCap51, char_cases::kCap70,
                                               char_cases::kDelay3p0, char_cases::kPower0p4,
                                               char_cases::HTreeShape{.pattern_id = char_cases::kPattern3, .levels = 1}));

  const icts::TopologyPatternCombiner combiner(char_cases::kBoundaryKey);
  auto result = upstream.concatWith(downstream, combiner);

  ASSERT_EQ(result.size(), 1U);

  const auto& merged = result.get_chars().front();
  EXPECT_EQ(merged.get_input_slew_idx(), char_cases::kSlew80);
  EXPECT_EQ(merged.get_output_slew_idx(), char_cases::kSlew120);
  EXPECT_EQ(merged.get_driven_cap_idx(), char_cases::kCap40);
  EXPECT_EQ(merged.get_leaf_load_cap_idx(), char_cases::kCap60);
  EXPECT_EQ(merged.get_load_cap_idx(), char_cases::kCap60);
  EXPECT_DOUBLE_EQ(merged.get_delay(), char_cases::kDelay3p0);
  EXPECT_DOUBLE_EQ(merged.get_power(), char_cases::kMergedPower1p1);
  EXPECT_EQ(merged.get_levels(), 2U);
  EXPECT_EQ(merged.get_pattern_id().domain, icts::PatternDomain::kTopologyPattern);
}

TEST(HTreeJoinTest, OddCapHalvingUsesCeilHalfBin)
{
  icts::HTreeTopologyCharTable upstream;
  icts::HTreeTopologyCharTable downstream;

  upstream.addChar(char_cases::MakeHTreeChar(char_cases::kSlew80, char_cases::kSlew100, char_cases::kCap40, char_cases::kCap101,
                                             char_cases::kDelay1p0, char_cases::kPower0p5,
                                             char_cases::HTreeShape{.pattern_id = char_cases::kPattern1, .levels = 1}));
  downstream.addChar(char_cases::MakeHTreeChar(char_cases::kSlew100, char_cases::kSlew120, char_cases::kCap50, char_cases::kCap60,
                                               char_cases::kDelay2p0, char_cases::kPower0p3,
                                               char_cases::HTreeShape{.pattern_id = char_cases::kPattern2, .levels = 1}));
  downstream.addChar(char_cases::MakeHTreeChar(char_cases::kSlew100, char_cases::kSlew130, char_cases::kCap51, char_cases::kCap70,
                                               char_cases::kDelay3p0, char_cases::kPower0p4,
                                               char_cases::HTreeShape{.pattern_id = char_cases::kPattern3, .levels = 1}));

  const icts::TopologyPatternCombiner combiner(char_cases::kBoundaryKey);
  auto result = upstream.concatWith(downstream, combiner);

  EXPECT_EQ(result.size(), 1U);
  EXPECT_EQ(result.get_chars().front().get_output_slew_idx(), char_cases::kSlew130);
  EXPECT_EQ(result.get_chars().front().get_driven_cap_idx(), char_cases::kCap40);
  EXPECT_EQ(result.get_chars().front().get_leaf_load_cap_idx(), char_cases::kCap70);
  EXPECT_EQ(result.get_chars().front().get_load_cap_idx(), char_cases::kCap70);
}

TEST(HTreeJoinTest, PowerDoubling)
{
  icts::HTreeTopologyCharTable upstream;
  icts::HTreeTopologyCharTable downstream;

  upstream.addChar(char_cases::MakeHTreeChar(char_cases::kSlew80, char_cases::kSlew100, char_cases::kCap40, char_cases::kCap100,
                                             char_cases::kDelay1p0, char_cases::kPower10p0,
                                             char_cases::HTreeShape{.pattern_id = char_cases::kPattern1, .levels = 1}));
  downstream.addChar(char_cases::MakeHTreeChar(char_cases::kSlew100, char_cases::kSlew120, char_cases::kCap50, char_cases::kCap60,
                                               char_cases::kDelay2p0, char_cases::kPower5p0,
                                               char_cases::HTreeShape{.pattern_id = char_cases::kPattern2, .levels = 1}));

  const icts::TopologyPatternCombiner combiner(char_cases::kBoundaryKey);
  auto result = upstream.concatWith(downstream, combiner);

  ASSERT_EQ(result.size(), 1U);
  EXPECT_DOUBLE_EQ(result.get_chars().front().get_power(), char_cases::kMergedPower20p0);
}

TEST(HTreeJoinTest, PowerDoublingSubtractsDownstreamSourceBoundarySwitchPower)
{
  icts::HTreeTopologyCharTable upstream;
  icts::HTreeTopologyCharTable downstream;

  upstream.addChar(char_cases::MakeHTreeChar(char_cases::kSlew80, char_cases::kSlew100, char_cases::kCap40, char_cases::kCap100,
                                             char_cases::kDelay1p0, char_cases::kPower10p0,
                                             char_cases::HTreeShape{.pattern_id = char_cases::kPattern1, .levels = 1}, 0.20));
  downstream.addChar(char_cases::MakeHTreeChar(char_cases::kSlew100, char_cases::kSlew120, char_cases::kCap50, char_cases::kCap60,
                                               char_cases::kDelay2p0, char_cases::kPower5p0,
                                               char_cases::HTreeShape{.pattern_id = char_cases::kPattern2, .levels = 1}, 1.50));

  const icts::TopologyPatternCombiner combiner(char_cases::kBoundaryKey);
  auto result = upstream.concatWith(downstream, combiner);

  ASSERT_EQ(result.size(), 1U);
  EXPECT_DOUBLE_EQ(result.get_chars().front().get_power(), 17.0);
  EXPECT_DOUBLE_EQ(result.get_chars().front().get_source_boundary_net_switch_power(), 0.20);
}

}  // namespace
}  // namespace icts_test
