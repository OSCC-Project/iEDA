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
 * @file PrunerTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Frontier helper tests for characterization composition states.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "characterization/Characterization.hh"
#include "database/characterization/PatternId.hh"
#include "database/characterization/SegmentChar.hh"
#include "module/characterization/fixture/CharacterizationUnitCaseData.hh"

namespace icts_test {
namespace {

namespace char_cases = characterization;

TEST(PrunerTest, CostDominatesUsesDelayPowerOrdering)
{
  auto better_entry = char_cases::MakeSegmentChar(
      char_cases::kSlew80, char_cases::kSlew90, char_cases::kCap40, char_cases::kCap60, char_cases::kDelay1p0, char_cases::kPower0p5,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern1, .length_idx = char_cases::kLength1000});
  auto worse_entry = char_cases::MakeSegmentChar(
      char_cases::kSlew80, char_cases::kSlew100, char_cases::kCap40, char_cases::kCap50, char_cases::kDelay2p0, char_cases::kPower0p6,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern1, .length_idx = char_cases::kLength1000});

  EXPECT_TRUE(icts::CostDominates(better_entry, worse_entry));
  EXPECT_FALSE(icts::CostDominates(worse_entry, better_entry));
}

TEST(PrunerTest, CostDominatesRejectsTradeoffs)
{
  auto lower_slew_entry = char_cases::MakeSegmentChar(
      char_cases::kSlew80, char_cases::kSlew90, char_cases::kCap40, char_cases::kCap50, char_cases::kDelay1p0, char_cases::kPower0p5,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern1, .length_idx = char_cases::kLength1000});
  auto higher_cap_entry = char_cases::MakeSegmentChar(
      char_cases::kSlew80, char_cases::kSlew100, char_cases::kCap40, char_cases::kCap60, char_cases::kDelay1p0, char_cases::kPower0p5,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern1, .length_idx = char_cases::kLength1000});

  EXPECT_FALSE(icts::CostDominates(lower_slew_entry, higher_cap_entry));
  EXPECT_FALSE(icts::CostDominates(higher_cap_entry, lower_slew_entry));
}

TEST(PrunerTest, SegmentStateFrontierPrunesDominatedSameGroupEntries)
{
  const auto cheaper_entry = char_cases::MakeSegmentChar(
      char_cases::kSlew80, char_cases::kSlew100, char_cases::kCap40, char_cases::kCap60, char_cases::kDelay1p0, char_cases::kPower0p5,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern1, .length_idx = char_cases::kLength1000});
  const auto dominated_entry = char_cases::MakeSegmentChar(
      char_cases::kSlew80, char_cases::kSlew100, char_cases::kCap40, char_cases::kCap60, char_cases::kDelay2p0, char_cases::kPower0p6,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern2, .length_idx = char_cases::kLength1000});

  const auto frontier = icts::BuildSegmentStateFrontier(
      std::vector<icts::SegmentChar>{dominated_entry, cheaper_entry},
      [](const icts::SegmentChar&) -> icts::TerminalSemantic { return icts::TerminalSemantic::kLeafUnbuffered; });

  ASSERT_EQ(frontier.size(), 1U);
  EXPECT_EQ(frontier.front().get_pattern_id().local_id, char_cases::kPattern1);
}

TEST(PrunerTest, SegmentStateFrontierPreservesDistinctExactJoinBoundaries)
{
  const auto stronger_but_different_boundary = char_cases::MakeSegmentChar(
      char_cases::kSlew80, char_cases::kSlew90, char_cases::kCap40, char_cases::kCap70, char_cases::kDelay1p0, char_cases::kPower0p5,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern1, .length_idx = char_cases::kLength1000});
  const auto join_critical_boundary = char_cases::MakeSegmentChar(
      char_cases::kSlew80, char_cases::kSlew100, char_cases::kCap40, char_cases::kCap60, char_cases::kDelay2p0, char_cases::kPower0p6,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern2, .length_idx = char_cases::kLength1000});

  const auto frontier = icts::BuildSegmentStateFrontier(
      std::vector<icts::SegmentChar>{stronger_but_different_boundary, join_critical_boundary},
      [](const icts::SegmentChar&) -> icts::TerminalSemantic { return icts::TerminalSemantic::kLeafUnbuffered; });

  ASSERT_EQ(frontier.size(), 2U);

  icts::SegmentCharTable upstream;
  upstream.addChar(frontier.front());
  upstream.addChar(frontier.back());

  icts::SegmentCharTable downstream;
  downstream.addChar(char_cases::MakeSegmentChar(
      char_cases::kSlew100, char_cases::kSlew110, char_cases::kCap60, char_cases::kCap70, char_cases::kDelay1p5, char_cases::kPower0p2,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern3, .length_idx = char_cases::kLength2000}));

  const icts::SegmentPatternCombiner combiner(char_cases::kBoundaryKey);
  const auto result = upstream.concatWith(downstream, combiner).get_chars();
  ASSERT_EQ(result.size(), 1U);
  EXPECT_EQ(result.front().get_input_slew_idx(), char_cases::kSlew80);
  EXPECT_EQ(result.front().get_output_slew_idx(), char_cases::kSlew110);
  EXPECT_EQ(result.front().get_load_cap_idx(), char_cases::kCap70);
}

TEST(PrunerTest, SegmentStateFrontierDoesNotMergeTerminalSemantics)
{
  const auto leaf_unbuffered_entry = char_cases::MakeSegmentChar(
      char_cases::kSlew80, char_cases::kSlew100, char_cases::kCap40, char_cases::kCap60, char_cases::kDelay2p0, char_cases::kPower0p6,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern1, .length_idx = char_cases::kLength1000});
  const auto branch_buffered_entry = char_cases::MakeSegmentChar(
      char_cases::kSlew80, char_cases::kSlew100, char_cases::kCap40, char_cases::kCap60, char_cases::kDelay1p0, char_cases::kPower0p5,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern2, .length_idx = char_cases::kLength1000});

  const auto frontier = icts::BuildSegmentStateFrontier(std::vector<icts::SegmentChar>{leaf_unbuffered_entry, branch_buffered_entry},
                                                        [](const icts::SegmentChar& entry) -> icts::TerminalSemantic {
                                                          return entry.get_pattern_id().local_id == char_cases::kPattern2
                                                                     ? icts::TerminalSemantic::kBranchBuffered
                                                                     : icts::TerminalSemantic::kLeafUnbuffered;
                                                        });

  ASSERT_EQ(frontier.size(), 2U);
  std::vector<unsigned> pattern_ids;
  pattern_ids.reserve(frontier.size());
  for (const auto& entry : frontier) {
    pattern_ids.push_back(entry.get_pattern_id().local_id);
  }
  std::ranges::sort(pattern_ids);
  EXPECT_EQ(pattern_ids, (std::vector<unsigned>{char_cases::kPattern1, char_cases::kPattern2}));
}

}  // namespace
}  // namespace icts_test
