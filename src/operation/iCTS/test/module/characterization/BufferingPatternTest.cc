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
 * @file BufferingPatternTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-16
 * @brief Unit coverage for BufferingPattern terminal branch-buffer metadata.
 */

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "database/characterization/BufferingPattern.hh"
#include "database/characterization/PatternId.hh"

namespace icts_test {
namespace {

TEST(BufferingPatternTest, ConcatPreservesTerminalBranchBufferFromDownstream)
{
  const icts::BufferingPattern upstream(2U, icts::PatternId::segment(1U), std::vector<double>{0.5}, std::vector<std::string>{"BUF_X4"},
                                        false);
  const icts::BufferingPattern downstream(3U, icts::PatternId::segment(2U), std::vector<double>{1.0}, std::vector<std::string>{"BUF_X2"},
                                          true);

  const auto merged = icts::BufferingPattern::concat(upstream, downstream);

  EXPECT_TRUE(merged.hasTerminalBranchBuffer());
  ASSERT_EQ(merged.get_buffer_positions().size(), 2U);
  EXPECT_DOUBLE_EQ(merged.get_buffer_positions().at(0), 0.2);
  EXPECT_DOUBLE_EQ(merged.get_buffer_positions().at(1), 1.0);
  ASSERT_EQ(merged.get_cell_masters().size(), 2U);
  EXPECT_EQ(merged.get_cell_masters().at(0), "BUF_X4");
  EXPECT_EQ(merged.get_cell_masters().at(1), "BUF_X2");
}

TEST(BufferingPatternTest, ConcatUsesDownstreamTerminalBranchBufferSemantics)
{
  const icts::BufferingPattern upstream(2U, icts::PatternId::segment(1U), std::vector<double>{1.0}, std::vector<std::string>{"BUF_X4"},
                                        true);
  const icts::BufferingPattern downstream(3U, icts::PatternId::segment(2U), std::vector<double>{0.5}, std::vector<std::string>{"BUF_X2"},
                                          false);

  const auto merged = icts::BufferingPattern::concat(upstream, downstream);

  EXPECT_FALSE(merged.hasTerminalBranchBuffer());
  ASSERT_EQ(merged.get_buffer_positions().size(), 2U);
  EXPECT_DOUBLE_EQ(merged.get_buffer_positions().at(0), 0.4);
  EXPECT_DOUBLE_EQ(merged.get_buffer_positions().at(1), 0.7);
}

}  // namespace
}  // namespace icts_test
