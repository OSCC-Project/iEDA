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
 * @file TopologyGenTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Thin topology generation test entry.
 */

#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "module/topology/topology_gen/fixture/TopologyGenScenario.hh"

namespace icts_test::topology_gen {
namespace {

class TopologyGenTestInterface : public ::testing::TestWithParam<TopologyCase>
{
};

auto GetTopologyCases() -> const std::vector<TopologyCase>&
{
  static const std::vector<TopologyCase> topology_cases = BuildTopologyCases();
  return topology_cases;
}

TEST_P(TopologyGenTestInterface, BuildAndVisualize)
{
  RunBuildAndVisualize(GetParam());
}

INSTANTIATE_TEST_SUITE_P(TopologyCases, TopologyGenTestInterface, ::testing::ValuesIn(GetTopologyCases()));

}  // namespace
}  // namespace icts_test::topology_gen
