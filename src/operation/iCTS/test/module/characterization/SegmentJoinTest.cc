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
 * @file SegmentJoinTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Segment characterization join tests.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "characterization/Characterization.hh"
#include "database/characterization/BufferingPattern.hh"
#include "database/characterization/PatternId.hh"
#include "database/characterization/SegmentChar.hh"
#include "module/characterization/fixture/CharacterizationUnitCaseData.hh"

namespace icts_test {
namespace {

namespace char_cases = characterization;

auto MakePatternCompositionState(unsigned source_strength_rank, unsigned sink_strength_rank,
                                 icts::TerminalSemantic terminal_semantic = icts::TerminalSemantic::kLeafUnbuffered)
    -> icts::PatternCompositionState
{
  return icts::PatternCompositionState{
      .terminal_semantic = terminal_semantic,
      .monotonic_boundary_state = icts::MonotonicBoundaryState{
          .source = icts::BoundaryBufferState{.has_buffer = source_strength_rank != 0U, .strength_rank = source_strength_rank},
          .sink = icts::BoundaryBufferState{.has_buffer = sink_strength_rank != 0U, .strength_rank = sink_strength_rank},
      },
  };
}

class MonotonicSegmentPatternCombiner
{
 public:
  explicit MonotonicSegmentPatternCombiner(std::unordered_map<icts::PatternId, icts::PatternCompositionState> pattern_states,
                                           unsigned start_id = 1000U)
      : _pattern_states(std::move(pattern_states)), _next_id(start_id)
  {
  }

  auto canCompose(icts::PatternId upstream, icts::PatternId downstream) const -> bool
  {
    return _pattern_states.at(upstream).monotonic_boundary_state.canComposeWith(_pattern_states.at(downstream).monotonic_boundary_state);
  }

  auto combine(icts::PatternId upstream, icts::PatternId downstream) const -> icts::PatternId
  {
    const auto upstream_state = _pattern_states.at(upstream);
    const auto downstream_state = _pattern_states.at(downstream);

    const auto merged_pattern_id = icts::PatternId::segment(_next_id++);
    _pattern_states[merged_pattern_id] = icts::PatternCompositionState{
        .terminal_semantic = downstream_state.terminal_semantic,
        .monotonic_boundary_state
        = icts::MonotonicBoundaryState::compose(upstream_state.monotonic_boundary_state, downstream_state.monotonic_boundary_state),
    };
    return merged_pattern_id;
  }

 private:
  mutable std::unordered_map<icts::PatternId, icts::PatternCompositionState> _pattern_states;
  mutable unsigned _next_id = 0U;
};

TEST(SegmentJoinTest, BasicEqualJoin)
{
  icts::SegmentCharTable upstream;
  icts::SegmentCharTable downstream;

  upstream.addChar(char_cases::MakeSegmentChar(
      char_cases::kSlew80, char_cases::kSlew100, char_cases::kCap40, char_cases::kCap50, char_cases::kDelay1p0, char_cases::kPower0p5,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern1, .length_idx = char_cases::kLength1000}));

  downstream.addChar(char_cases::MakeSegmentChar(
      char_cases::kSlew100, char_cases::kSlew120, char_cases::kCap50, char_cases::kCap60, char_cases::kDelay2p0, char_cases::kPower0p3,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern2, .length_idx = char_cases::kLength2000}));
  downstream.addChar(char_cases::MakeSegmentChar(
      char_cases::kSlew100, char_cases::kSlew130, char_cases::kCap55, char_cases::kCap70, char_cases::kDelay3p0, char_cases::kPower0p4,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern3, .length_idx = char_cases::kLength3000}));

  const icts::SegmentPatternCombiner combiner(char_cases::kBoundaryKey);
  auto result = upstream.concatWith(downstream, combiner);

  ASSERT_EQ(result.size(), 1U);

  const auto& merged = result.get_chars().front();
  EXPECT_EQ(merged.get_input_slew_idx(), char_cases::kSlew80);
  EXPECT_EQ(merged.get_output_slew_idx(), char_cases::kSlew120);
  EXPECT_EQ(merged.get_driven_cap_idx(), char_cases::kCap40);
  EXPECT_EQ(merged.get_load_cap_idx(), char_cases::kCap60);
  EXPECT_DOUBLE_EQ(merged.get_delay(), char_cases::kDelay3p0);
  EXPECT_DOUBLE_EQ(merged.get_power(), char_cases::kMergedPower0p8);
  EXPECT_EQ(merged.get_length_idx(), char_cases::kLengthSum3000);
  EXPECT_EQ(merged.get_pattern_id().domain, icts::PatternDomain::kSegmentPattern);
}

TEST(SegmentJoinTest, MultipleMatches)
{
  icts::SegmentCharTable upstream;
  icts::SegmentCharTable downstream;

  upstream.addChar(char_cases::MakeSegmentChar(
      char_cases::kSlew80, char_cases::kSlew100, char_cases::kCap40, char_cases::kCap50, char_cases::kDelay1p0, char_cases::kPower0p5,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern1, .length_idx = char_cases::kLength1000}));
  upstream.addChar(char_cases::MakeSegmentChar(
      char_cases::kSlew90, char_cases::kSlew100, char_cases::kCap45, char_cases::kCap50, char_cases::kDelay1p5, char_cases::kPower0p6,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern2, .length_idx = char_cases::kLength1500}));

  downstream.addChar(char_cases::MakeSegmentChar(
      char_cases::kSlew100, char_cases::kSlew120, char_cases::kCap50, char_cases::kCap60, char_cases::kDelay2p0, char_cases::kPower0p3,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern3, .length_idx = char_cases::kLength2000}));

  const icts::SegmentPatternCombiner combiner(char_cases::kBoundaryKey);
  auto result = upstream.concatWith(downstream, combiner);

  EXPECT_EQ(result.size(), 2U);
}

TEST(SegmentJoinTest, NoMatches)
{
  icts::SegmentCharTable upstream;
  icts::SegmentCharTable downstream;

  upstream.addChar(char_cases::MakeSegmentChar(
      char_cases::kSlew80, char_cases::kSlew100, char_cases::kCap40, char_cases::kCap50, char_cases::kDelay1p0, char_cases::kPower0p5,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern1, .length_idx = char_cases::kLength1000}));
  downstream.addChar(char_cases::MakeSegmentChar(
      char_cases::kSlew100, char_cases::kSlew120, char_cases::kCap51, char_cases::kCap60, char_cases::kDelay2p0, char_cases::kPower0p3,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern2, .length_idx = char_cases::kLength2000}));

  const icts::SegmentPatternCombiner combiner(char_cases::kBoundaryKey);
  auto result = upstream.concatWith(downstream, combiner);

  EXPECT_EQ(result.size(), 0U);
}

TEST(SegmentJoinTest, ExactJoinRejectsNonMonotonicBoundaryStrengthIncrease)
{
  icts::SegmentCharTable upstream;
  icts::SegmentCharTable downstream;

  upstream.addChar(char_cases::MakeSegmentChar(
      char_cases::kSlew80, char_cases::kSlew100, char_cases::kCap40, char_cases::kCap50, char_cases::kDelay1p0, char_cases::kPower0p5,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern1, .length_idx = char_cases::kLength1000}));

  downstream.addChar(char_cases::MakeSegmentChar(
      char_cases::kSlew100, char_cases::kSlew120, char_cases::kCap50, char_cases::kCap60, char_cases::kDelay2p0, char_cases::kPower0p3,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern2, .length_idx = char_cases::kLength2000}));
  downstream.addChar(char_cases::MakeSegmentChar(
      char_cases::kSlew100, char_cases::kSlew110, char_cases::kCap50, char_cases::kCap60, char_cases::kDelay1p5, char_cases::kPower0p2,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern3, .length_idx = char_cases::kLength2000}));

  MonotonicSegmentPatternCombiner combiner({
      {icts::PatternId::segment(char_cases::kPattern1), MakePatternCompositionState(3U, 1U)},
      {icts::PatternId::segment(char_cases::kPattern2), MakePatternCompositionState(2U, 2U)},
      {icts::PatternId::segment(char_cases::kPattern3), MakePatternCompositionState(1U, 1U)},
  });
  const auto result = upstream.concatWith(downstream, combiner).get_chars();

  ASSERT_EQ(result.size(), 1U);
  EXPECT_EQ(result.front().get_output_slew_idx(), char_cases::kSlew110);
  EXPECT_DOUBLE_EQ(result.front().get_delay(), 2.5);
  EXPECT_DOUBLE_EQ(result.front().get_power(), 0.7);
}

TEST(SegmentJoinTest, SegmentStateFrontierKeepsDistinctMonotonicBoundaryStates)
{
  const auto weaker_sink_boundary = char_cases::MakeSegmentChar(
      char_cases::kSlew80, char_cases::kSlew100, char_cases::kCap40, char_cases::kCap60, char_cases::kDelay1p0, char_cases::kPower0p5,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern1, .length_idx = char_cases::kLength1000});
  const auto stronger_sink_boundary = char_cases::MakeSegmentChar(
      char_cases::kSlew80, char_cases::kSlew100, char_cases::kCap40, char_cases::kCap60, char_cases::kDelay1p0, char_cases::kPower0p2,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern2, .length_idx = char_cases::kLength1000});

  const auto frontier = icts::BuildSegmentStateFrontier(std::vector<icts::SegmentChar>{weaker_sink_boundary, stronger_sink_boundary},
                                                        [](const icts::SegmentChar& entry) -> icts::PatternCompositionState {
                                                          if (entry.get_pattern_id().local_id == char_cases::kPattern2) {
                                                            return MakePatternCompositionState(3U, 2U);
                                                          }
                                                          return MakePatternCompositionState(3U, 1U);
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

TEST(SegmentJoinTest, JoinSubtractsDownstreamSourceBoundarySwitchPower)
{
  icts::SegmentCharTable upstream;
  icts::SegmentCharTable downstream;

  upstream.addChar(char_cases::MakeSegmentChar(
      char_cases::kSlew80, char_cases::kSlew100, char_cases::kCap40, char_cases::kCap50, char_cases::kDelay1p0, char_cases::kPower0p5,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern1, .length_idx = char_cases::kLength1000}, 0.05));
  downstream.addChar(char_cases::MakeSegmentChar(
      char_cases::kSlew100, char_cases::kSlew120, char_cases::kCap50, char_cases::kCap60, char_cases::kDelay2p0, char_cases::kPower0p3,
      char_cases::SegmentShape{.pattern_id = char_cases::kPattern2, .length_idx = char_cases::kLength2000}, 0.10));

  const icts::SegmentPatternCombiner combiner(char_cases::kBoundaryKey);
  auto result = upstream.concatWith(downstream, combiner);

  ASSERT_EQ(result.size(), 1U);
  EXPECT_DOUBLE_EQ(result.get_chars().front().get_power(), 0.7);
  EXPECT_DOUBLE_EQ(result.get_chars().front().get_source_boundary_net_switch_power(), 0.05);
}

}  // namespace
}  // namespace icts_test
