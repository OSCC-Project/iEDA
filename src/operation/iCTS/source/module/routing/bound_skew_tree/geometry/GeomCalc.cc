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
 * @file GeomCalc.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-03-08
 * @brief Geometry helpers for bound-skew tree routing
 */
#include "bound_skew_tree/geometry/GeomCalc.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>
#include <vector>

#include "Log.hh"
#include "bound_skew_tree/component/Components.hh"

namespace icts::bst {
namespace {

constexpr size_t kLinePointCount = 2;

[[nodiscard]] auto HeadPoint(const Line& line) -> const Point&
{
  return line.at(kHead);
}

[[nodiscard]] auto TailPoint(const Line& line) -> const Point&
{
  return line.at(kTail);
}

}  // namespace

auto GeomCalc::isSame(const Point& first_point, const Point& second_point) -> bool
{
  const auto dist = distance(first_point, second_point);
  return Equal(dist, 0);
}

auto GeomCalc::distance(const Point& first_point, const Point& second_point) -> double
{
  return std::abs(first_point.x - second_point.x) + std::abs(first_point.y - second_point.y);
}

auto GeomCalc::pointToLineDistanceManhattan(const Point& point, const Line& line, Point& closest_point) -> double
{
  TransformedRect point_transformed_rect(point, 0);
  TransformedRect line_transformed_rect;
  lineToTransformedRect(line_transformed_rect, line);
  const auto dist = transformedRectDistance(point_transformed_rect, line_transformed_rect);
  TransformedRect expanded_point_transformed_rect;
  expanded_point_transformed_rect.makeDiamond(point, dist);
  TransformedRect intersection_transformed_rect;
  makeIntersection(expanded_point_transformed_rect, line_transformed_rect, intersection_transformed_rect);
  coreMidPoint(intersection_transformed_rect, closest_point);
  return dist;
}

auto GeomCalc::pointToLineDistanceNonManhattan(const Point& point, const Line& line, Point& closest_point) -> double
{
  std::vector<Point> candidate_points;

  if (!Equal(HeadPoint(line).x, TailPoint(line).x) && (point.x - HeadPoint(line).x) * (point.x - TailPoint(line).x) <= 0) {
    Point candidate_point;
    candidate_point.x = point.x;
    candidate_point.y = ((TailPoint(line).x - point.x) * (HeadPoint(line).y - TailPoint(line).y) / (TailPoint(line).x - HeadPoint(line).x))
                        + TailPoint(line).y;
    candidate_points.push_back(candidate_point);
  }

  if (!Equal(HeadPoint(line).y, TailPoint(line).y) && (point.y - HeadPoint(line).y) * (point.y - TailPoint(line).y) <= 0) {
    Point candidate_point;
    candidate_point.x = ((TailPoint(line).y - point.y) * (HeadPoint(line).x - TailPoint(line).x) / (TailPoint(line).y - HeadPoint(line).y))
                        + TailPoint(line).x;
    candidate_point.y = point.y;
    candidate_points.push_back(candidate_point);
  }

  if (candidate_points.size() < kLinePointCount) {
    candidate_points.push_back(HeadPoint(line));
    candidate_points.push_back(TailPoint(line));
  }

  double min_distance = std::numeric_limits<double>::max();
  std::ranges::for_each(candidate_points, [&](const Point& candidate_point) -> void {
    const auto dist = distance(point, candidate_point);
    if (dist < min_distance) {
      min_distance = dist;
      closest_point = candidate_point;
    }
  });

  return min_distance;
}

auto GeomCalc::pointToLineDistance(const Point& point, const Line& line, Point& closest_point) -> double
{
  auto min_distance = std::numeric_limits<double>::max();
  const auto delta_x = std::abs(line.at(kHead).x - line.at(kTail).x);
  const auto delta_y = std::abs(line.at(kHead).y - line.at(kTail).y);

  if (isSame(line.at(kHead), line.at(kTail))) {
    closest_point = line.at(kHead);
    min_distance = distance(point, closest_point);
  } else {
    Point snapped_point = point;
    if (onLine(snapped_point, line)) {
      closest_point = snapped_point;
      min_distance = 0.0;
    } else if (Equal(delta_x, delta_y)) {
      min_distance = pointToLineDistanceManhattan(point, line, closest_point);
    } else {
      min_distance = pointToLineDistanceNonManhattan(point, line, closest_point);
    }
  }

  Point validated_point = closest_point;
  LOG_FATAL_IF(!onLine(validated_point, line)) << "closest point is not on line";
  return min_distance;
}

auto GeomCalc::pointToTransformedRectDistance(const Point& point, TransformedRect& transformed_rect) -> double
{
  TransformedRect point_transformed_rect(point, 0);
  if (containsTransformedRect(point_transformed_rect, transformed_rect)) {
    return 0;
  }

  std::vector<TransformedRect> candidate_transformed_rects(4, transformed_rect);
  candidate_transformed_rects.at(kLeft + kHead).y_low(transformed_rect.y_high());
  candidate_transformed_rects.at(kLeft + kTail).y_high(transformed_rect.y_low());
  candidate_transformed_rects.at(kRight + kHead).x_low(transformed_rect.x_high());
  candidate_transformed_rects.at(kRight + kTail).x_high(transformed_rect.x_low());

  double min_distance = std::numeric_limits<double>::max();
  std::ranges::for_each(candidate_transformed_rects, [&](TransformedRect& candidate_transformed_rect) -> void {
    const auto dist = transformedRectDistance(point_transformed_rect, candidate_transformed_rect);
    min_distance = std::min(min_distance, dist);
  });
  return min_distance;
}

auto GeomCalc::calcCoord(Point& point, const Line& line, const double& shift) -> void
{
  const auto current_line_type = lineType(line);
  double head_distance = 0.0;
  double tail_distance = 0.0;
  if (current_line_type == LineType::kHorizontal || current_line_type == LineType::kFlat) {
    head_distance = std::abs(line.at(kHead).x - shift);
    tail_distance = std::abs(line.at(kTail).x - shift);
  } else {
    head_distance = std::abs(line.at(kHead).y - shift);
    tail_distance = std::abs(line.at(kTail).y - shift);
  }

  point.x = ((line.at(kHead).x * tail_distance) + (line.at(kTail).x * head_distance)) / (head_distance + tail_distance);
  point.y = ((line.at(kHead).y * tail_distance) + (line.at(kTail).y * head_distance)) / (head_distance + tail_distance);
}

auto GeomCalc::calcRelativeCoord(Point& point, const RelativeType& type, const double& shift) -> void
{
  switch (type) {
    case RelativeType::kLeft:
      point.x += shift;
      break;
    case RelativeType::kRight:
      point.x -= shift;
      break;
    case RelativeType::kTop:
      point.y -= shift;
      break;
    case RelativeType::kBottom:
      point.y += shift;
      break;
    default:
      break;
  }
}

auto GeomCalc::crossProduct(const Point& origin_point, const Point& lhs_point, const Point& rhs_point) -> double
{
  return ((lhs_point.x - origin_point.x) * (rhs_point.y - origin_point.y))
         - ((rhs_point.x - origin_point.x) * (lhs_point.y - origin_point.y));
}

auto GeomCalc::inBoundBox(const Point& point, const Line& line) -> bool
{
  return point.x <= std::max(line.at(kHead).x, line.at(kTail).x) + kEpsilon
         && point.x >= std::min(line.at(kHead).x, line.at(kTail).x) - kEpsilon
         && point.y <= std::max(line.at(kHead).y, line.at(kTail).y) + kEpsilon
         && point.y >= std::min(line.at(kHead).y, line.at(kTail).y) - kEpsilon;
}

auto GeomCalc::boundBoxOverlap(const Line& lhs_line, const Line& rhs_line, const double& epsilon) -> bool
{
  const BoundingBox lhs_box{.min_x = std::min(lhs_line.at(kHead).x, lhs_line.at(kTail).x),
                            .min_y = std::min(lhs_line.at(kHead).y, lhs_line.at(kTail).y),
                            .max_x = std::max(lhs_line.at(kHead).x, lhs_line.at(kTail).x),
                            .max_y = std::max(lhs_line.at(kHead).y, lhs_line.at(kTail).y)};
  const BoundingBox rhs_box{.min_x = std::min(rhs_line.at(kHead).x, rhs_line.at(kTail).x),
                            .min_y = std::min(rhs_line.at(kHead).y, rhs_line.at(kTail).y),
                            .max_x = std::max(rhs_line.at(kHead).x, rhs_line.at(kTail).x),
                            .max_y = std::max(rhs_line.at(kHead).y, rhs_line.at(kTail).y)};
  return boundBoxOverlap(lhs_box, rhs_box, epsilon);
}

auto GeomCalc::boundBoxOverlap(const BoundingBox& lhs_box, const BoundingBox& rhs_box, const double& overlap_epsilon) -> bool
{
  if (lhs_box.min_x - rhs_box.max_x >= overlap_epsilon) {
    return false;
  }
  if (rhs_box.min_x - lhs_box.max_x >= overlap_epsilon) {
    return false;
  }
  if (lhs_box.min_y - rhs_box.max_y >= overlap_epsilon) {
    return false;
  }
  return rhs_box.min_y - lhs_box.max_y < overlap_epsilon;
}

}  // namespace icts::bst
