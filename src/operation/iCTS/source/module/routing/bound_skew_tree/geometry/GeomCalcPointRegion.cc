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
 * @file GeomCalcPointRegion.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Point sorting and region geometry helpers for bound-skew tree routing
 */
#include <algorithm>
#include <cstddef>
#include <limits>
#include <ranges>
#include <vector>

#include "bound_skew_tree/component/Components.hh"
#include "bound_skew_tree/geometry/GeomCalc.hh"

namespace icts::bst {
namespace {

constexpr size_t kLinePointCount = 2;

}  // namespace

auto GeomCalc::sortPointsByFront(Points& points) -> void
{
  std::ranges::for_each(points, [&](Point& point) -> void { point.val = distance(point, points.front()); });
  sortPointsByValue(points);
}

auto GeomCalc::sortPointsByValue(Points& points) -> void
{
  if (points.empty()) {
    return;
  }
  std::ranges::sort(points, [](const Point& lhs_point, const Point& rhs_point) -> bool { return lhs_point.val < rhs_point.val; });
}

auto GeomCalc::sortPointsByValueDesc(Points& points) -> void
{
  if (points.empty()) {
    return;
  }
  std::ranges::sort(points, [](const Point& lhs_point, const Point& rhs_point) -> bool { return lhs_point.val > rhs_point.val; });
}

auto GeomCalc::uniquePointLocations(std::vector<Point>& points) -> void
{
  if (points.size() < kLinePointCount) {
    return;
  }

  std::vector<Point> unique_points = {points.front()};
  std::ranges::for_each(points, [&](const Point& point) -> void {
    if (!isSame(point, unique_points.back())) {
      unique_points.push_back(point);
    }
  });
  if (unique_points.size() > 1 && isSame(unique_points.front(), unique_points.back())) {
    unique_points.pop_back();
  }
  points = unique_points;
}

auto GeomCalc::uniquePointValues(std::vector<Point>& points) -> void
{
  auto [first, last] = std::ranges::unique(
      points, [](const Point& lhs_point, const Point& rhs_point) -> bool { return Equal(lhs_point.val, rhs_point.val); });
  points.erase(first, last);
}

auto GeomCalc::convexHull(std::vector<Point>& points) -> void
{
  if (points.size() < kLinePointCount) {
    return;
  }
  if (points.size() == kLinePointCount) {
    const auto dist = distance(points.front(), points.back());
    if (Equal(dist, 0)) {
      points.pop_back();
    }
    return;
  }

  std::ranges::sort(points, [](const Point& lhs_point, const Point& rhs_point) -> bool {
    return lhs_point.x + kEpsilon < rhs_point.x || (Equal(lhs_point.x, rhs_point.x) && lhs_point.y < rhs_point.y);
  });

  std::vector<Point> hull_points(2 * points.size());
  size_t hull_size = 0;
  for (const auto& point : points) {
    while (hull_size > 1 && crossProduct(hull_points.at(hull_size - 2), hull_points.at(hull_size - 1), point) <= kEpsilon) {
      --hull_size;
    }
    hull_points.at(hull_size++) = point;
  }
  for (size_t point_index = points.size() - 1, lower_hull_size = hull_size + 1; point_index > 0; --point_index) {
    while (hull_size >= lower_hull_size
           && crossProduct(hull_points.at(hull_size - 2), hull_points.at(hull_size - 1), points.at(point_index - 1)) <= kEpsilon) {
      --hull_size;
    }
    hull_points.at(hull_size++) = points.at(point_index - 1);
  }
  points = {hull_points.begin(), hull_points.begin() + static_cast<std::ptrdiff_t>(hull_size - 1)};
}

auto GeomCalc::centerPoint(const std::vector<Point>& points) -> Point
{
  if (points.empty()) {
    return {0, 0};
  }

  double x_sum = 0.0;
  double y_sum = 0.0;
  std::ranges::for_each(points, [&](const Point& point) -> void {
    x_sum += point.x;
    y_sum += point.y;
  });
  const auto point_count = static_cast<double>(points.size());
  return {x_sum / point_count, y_sum / point_count};
}

auto GeomCalc::isRegionContain(const Point& point, const std::vector<Point>& region) -> bool
{
  auto is_in_region = false;
  const auto point_x = point.x;
  const auto point_y = point.y;
  const auto region_point_count = region.size();
  auto previous_point_index = region_point_count - 1;

  for (size_t point_index = 0; point_index < region_point_count; previous_point_index = point_index, ++point_index) {
    const auto source_x = region.at(point_index).x;
    const auto source_y = region.at(point_index).y;
    const auto target_x = region.at(previous_point_index).x;
    const auto target_y = region.at(previous_point_index).y;
    if ((source_y < point_y && target_y >= point_y) || (target_y < point_y && source_y >= point_y)) {
      if (source_x + ((point_y - source_y) / (target_y - source_y) * (target_x - source_x)) < point_x) {
        is_in_region = !is_in_region;
      }
    }
  }
  if (is_in_region) {
    return true;
  }

  Point point_on_boundary = point;
  for (size_t point_index = 0; point_index < region.size(); ++point_index) {
    const auto next_point_index = (point_index + 1) % region.size();
    if (onLine(point_on_boundary, {region.at(point_index), region.at(next_point_index)})) {
      return true;
    }
  }
  return false;
}

auto GeomCalc::closestPointOnRegion(const Point& point, const std::vector<Point>& region) -> Point
{
  if (isRegionContain(point, region)) {
    return point;
  }

  Point closest_point_on_line;
  Point best_point;
  auto min_distance = std::numeric_limits<double>::max();
  for (size_t point_index = 0; point_index < region.size(); ++point_index) {
    const auto next_point_index = (point_index + 1) % region.size();
    const auto dist = pointToLineDistance(point, {region.at(point_index), region.at(next_point_index)}, closest_point_on_line);
    if (dist < min_distance) {
      min_distance = dist;
      best_point = closest_point_on_line;
    }
  }
  return best_point;
}

}  // namespace icts::bst
