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
 * @file TopologyRealTechMatrixTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Real-tech BP placement matrix coverage for Topology flow.
 */

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "TopologyRealTechScenario.hh"

namespace icts_test {
namespace {

namespace smoke = synthesis_realtech_smoke;

TEST(TopologyRealTechSmokeTest, BpBeTopFullSinkNonClusteredExperimentMatrix)
{
  const auto matrix_result = smoke::EvaluateBpBeTopFullSinkNonClusteredExperimentMatrix();
  if (matrix_result.skipped) {
    GTEST_SKIP() << matrix_result.skip_reason;
    return;
  }

  for (const auto& failure_message : matrix_result.failure_messages) {
    ADD_FAILURE() << failure_message;
  }
  EXPECT_TRUE(matrix_result.report_written);
}

TEST(TopologyRealTechSmokeTest, Arm9FullSinkTopologyToleranceComparison)
{
  const auto comparison_result = smoke::EvaluateArm9FullSinkTopologyToleranceComparison();
  if (comparison_result.skipped) {
    GTEST_SKIP() << comparison_result.skip_reason;
    return;
  }

  ASSERT_EQ(comparison_result.records.size(), 2U);
  EXPECT_DOUBLE_EQ(comparison_result.records.at(0).htree_topology_tolerance, 0.1);
  EXPECT_DOUBLE_EQ(comparison_result.records.at(1).htree_topology_tolerance, 0.0);
  for (const auto& failure_message : comparison_result.failure_messages) {
    ADD_FAILURE() << failure_message;
  }
  EXPECT_TRUE(comparison_result.report_written);
}

}  // namespace
}  // namespace icts_test
