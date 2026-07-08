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
 * @file SpatialRegionTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-03-17
 * @brief Unit tests for spatial Rect and Region helpers.
 */

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

#include "database/spatial/Point.hh"
#include "database/spatial/Rect.hh"
#include "database/spatial/Region.hh"

namespace icts_test {
namespace {

constexpr int kRectMaxX = 10;
constexpr int kRectMaxY = 5;
constexpr int kHoleMax = 7;
constexpr int kIslandLeftMinX = 10;
constexpr int kIslandLeftMaxX = 12;

using Point = icts::Point<int>;
using Rect = icts::Rect<int>;
using Region = icts::Region<int>;

TEST(SpatialRegionTest, RectContainsAndClamp)
{
  const Rect rect(0, 0, kRectMaxX, kRectMaxY);
  EXPECT_TRUE(rect.contains(Point(0, 0)));
  EXPECT_TRUE(rect.contains(Point(kRectMaxX, kRectMaxY)));
  EXPECT_FALSE(rect.contains(Point(kRectMaxX + 1, kRectMaxY)));
  EXPECT_EQ(rect.clamp(Point(-3, 9)), Point(0, kRectMaxY));
}

TEST(SpatialRegionTest, RegionSubtractRectKeepsExpectedPieces)
{
  Region region(Rect(0, 0, kRectMaxX, kRectMaxX));
  region.subtract(Rect(3, 3, kHoleMax, kHoleMax));

  EXPECT_FALSE(region.empty());
  EXPECT_FALSE(region.contains(Point(5, 5)));
  EXPECT_TRUE(region.contains(Point(1, 1)));
  EXPECT_TRUE(region.contains(Point(9, 9)));
  EXPECT_TRUE(region.contains(Point(2, 5)));
  EXPECT_EQ(region.rects().size(), 4U);
}

TEST(SpatialRegionTest, RegionNormalizationMergesTouchingSpansOnly)
{
  Region region;
  region.add_rect(Rect(0, 0, 4, 2));
  region.add_rect(Rect(0, 3, 4, kRectMaxY));
  EXPECT_EQ(region.rects().size(), 1U);
  EXPECT_TRUE(region.contains(Point(2, 4)));

  const Region l_shape({Rect(0, 0, 4, 1), Rect(3, 0, kRectMaxY, 4)});
  EXPECT_EQ(l_shape.rects().size(), 2U);
  EXPECT_FALSE(l_shape.contains(Point(1, 3)));
}

TEST(SpatialRegionTest, RegionProjectNearestSelectsClosestLegalPoint)
{
  const Region region({Rect(0, 0, 2, 2), Rect(kIslandLeftMinX, 0, kIslandLeftMaxX, 2)});
  const auto projected = region.project_nearest(Point(7, 1));
  ASSERT_TRUE(projected.has_value());
  const Point projected_point = projected.value_or(Point(0, 0));
  EXPECT_EQ(projected_point, Point(kIslandLeftMinX, 1));

  const auto inside = region.project_nearest(Point(1, 2));
  ASSERT_TRUE(inside.has_value());
  const Point inside_point = inside.value_or(Point(0, 0));
  EXPECT_EQ(inside_point, Point(1, 2));
}

}  // namespace
}  // namespace icts_test
