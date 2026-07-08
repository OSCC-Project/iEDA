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
 * @file LocalLegalizationTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-03-17
 * @brief Unit tests for standalone local legalization.
 */

#include <gtest/gtest.h>

#include <set>
#include <string>
#include <vector>

#include "database/spatial/Point.hh"
#include "database/spatial/Rect.hh"
#include "database/spatial/Region.hh"
#include "module/routing/local_legalization/LocalLegalization.hh"

namespace icts_test {
namespace {

constexpr int kLargeRegionMax = 10;
constexpr int kExpandedRegionMax = 5;
constexpr int kDisjointSourceX = 6;
constexpr int kDisjointStartX = 10;
constexpr int kDisjointEndX = 12;

using Point = icts::Point<int>;
using Rect = icts::Rect<int>;
using Region = icts::Region<int>;
using Legalizer = icts::LocalLegalization;

auto ToSet(const std::vector<Point>& points) -> std::set<Point>
{
  return {points.begin(), points.end()};
}

TEST(LocalLegalizationTest, KeepsAlreadyLegalUniquePoints)
{
  Legalizer::Problem problem;
  problem.movable_points = {Point(1, 1), Point(3, 3)};
  problem.fixed_points = {Point(kExpandedRegionMax, kExpandedRegionMax)};
  problem.feasible_region = Region(Rect(0, 0, kLargeRegionMax, kLargeRegionMax));

  auto result = Legalizer::legalize(problem);
  ASSERT_TRUE(result.success);
  EXPECT_EQ(result.legalized_points, problem.movable_points);
  EXPECT_EQ(result.total_displacement, 0);
}

TEST(LocalLegalizationTest, ResolvesMovableAndFixedCollision)
{
  Legalizer::Problem problem;
  problem.movable_points = {Point(1, 1)};
  problem.fixed_points = {Point(1, 1)};
  problem.feasible_region = Region(Rect(0, 0, 3, 3));

  auto result = Legalizer::legalize(problem);
  ASSERT_TRUE(result.success);
  EXPECT_NE(result.legalized_points.front(), Point(1, 1));
  EXPECT_TRUE(problem.feasible_region.contains(result.legalized_points.front()));
}

TEST(LocalLegalizationTest, ResolvesDuplicateMovablePointsUniquely)
{
  Legalizer::Problem problem;
  problem.movable_points = {Point(2, 2), Point(2, 2), Point(2, 2)};
  problem.feasible_region = Region(Rect(0, 0, kExpandedRegionMax, kExpandedRegionMax));

  auto result = Legalizer::legalize(problem);
  ASSERT_TRUE(result.success);
  EXPECT_EQ(ToSet(result.legalized_points).size(), result.legalized_points.size());
}

TEST(LocalLegalizationTest, RespectsBlockedRegion)
{
  Legalizer::Problem problem;
  problem.movable_points = {Point(2, 2)};
  problem.feasible_region = Region(Rect(0, 0, 4, 4));
  problem.block_region = Region(Rect(2, 2, 3, 3));

  auto result = Legalizer::legalize(problem);
  ASSERT_TRUE(result.success);
  EXPECT_FALSE(problem.block_region.contains(result.legalized_points.front()));
  EXPECT_TRUE(problem.feasible_region.contains(result.legalized_points.front()));
}

TEST(LocalLegalizationTest, UsesDisjointFeasibleIslands)
{
  Legalizer::Problem problem;
  problem.movable_points = {Point(kDisjointSourceX, 1)};
  problem.feasible_region = Region({Rect(0, 0, 2, 2), Rect(kDisjointStartX, 0, kDisjointEndX, 2)});

  auto result = Legalizer::legalize(problem);
  ASSERT_TRUE(result.success);
  EXPECT_EQ(result.legalized_points.front(), Point(2, 1));
}

TEST(LocalLegalizationTest, GlobalAssignmentBeatsGreedyOrdering)
{
  Legalizer::Problem problem;
  problem.movable_points = {Point(0, 0), Point(1, 0)};
  problem.fixed_points = {Point(0, 0)};
  problem.feasible_region = Region(Rect(0, 0, 2, 1));

  auto result = Legalizer::legalize(problem);
  ASSERT_TRUE(result.success);
  EXPECT_EQ(result.total_displacement, 1);
  EXPECT_EQ(ToSet(result.legalized_points), (std::set<Point>{Point(0, 1), Point(1, 0)}));
}

}  // namespace
}  // namespace icts_test
