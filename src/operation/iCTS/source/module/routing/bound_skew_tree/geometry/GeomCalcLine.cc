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
 * @file GeomCalcLine.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Line classification, intersection, and distance helpers for bound-skew tree routing
 */
#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <ostream>

#include "Log.hh"
#include "bound_skew_tree/component/Components.hh"
#include "bound_skew_tree/geometry/GeomCalc.hh"

namespace icts::bst {
namespace {

constexpr double kAverageFactor = 2.0;
constexpr size_t kLinePointCount = 2;

struct EndpointIntersectionContext
{
  Point* intersection_point = nullptr;
  size_t* intersection_count = nullptr;
};

struct LinePairView
{
  const Line* source_line = nullptr;
  const Line* target_line = nullptr;
};

[[nodiscard]] auto HeadPoint(const Line& line) -> const Point&
{
  return line.at(kHead);
}

[[nodiscard]] auto TailPoint(const Line& line) -> const Point&
{
  return line.at(kTail);
}

[[nodiscard]] auto HasZeroLength(const Line& line) -> bool
{
  return Equal(GeomCalc::distance(HeadPoint(line), TailPoint(line)), 0.0);
}

[[nodiscard]] auto IsSameDirectedLine(const Line& first_line, const Line& second_line) -> bool
{
  return GeomCalc::distance(HeadPoint(first_line), HeadPoint(second_line))
             + GeomCalc::distance(TailPoint(first_line), TailPoint(second_line))
         < kEpsilon;
}

auto AccumulateEndpointIntersections(const EndpointIntersectionContext& context, const LinePairView& line_pair) -> void
{
  LOG_FATAL_IF(context.intersection_point == nullptr || context.intersection_count == nullptr) << "endpoint intersection context is null";
  LOG_FATAL_IF(line_pair.source_line == nullptr || line_pair.target_line == nullptr) << "line pair is null";

  for (const Point& point_source : *line_pair.source_line) {
    Point candidate_point = point_source;
    if (GeomCalc::onLine(candidate_point, *line_pair.target_line)) {
      *context.intersection_point = candidate_point;
      ++(*context.intersection_count);
    }
  }
}

auto RemoveDuplicatedEndpointIntersections(size_t& intersection_count, const std::array<LinePairView, kLinePointCount>& line_pairs) -> void
{
  const LinePairView& left_to_right = line_pairs.at(kLeft);
  const LinePairView& right_to_left = line_pairs.at(kRight);
  LOG_FATAL_IF(left_to_right.source_line == nullptr || right_to_left.source_line == nullptr) << "line pair is null";

  for (const Point& first_point : *left_to_right.source_line) {
    for (const Point& second_point : *right_to_left.source_line) {
      if (GeomCalc::distance(first_point, second_point) < kEpsilon) {
        --intersection_count;
      }
    }
  }
}

[[nodiscard]] auto CountEndpointIntersections(Point& intersection_point, const Line& first_line, const Line& second_line) -> size_t
{
  size_t intersection_count = 0;
  const EndpointIntersectionContext context{.intersection_point = &intersection_point, .intersection_count = &intersection_count};
  const std::array<LinePairView, kLinePointCount> line_pairs = {
      LinePairView{.source_line = &first_line, .target_line = &second_line},
      LinePairView{.source_line = &second_line, .target_line = &first_line},
  };

  for (const LinePairView& line_pair : line_pairs) {
    AccumulateEndpointIntersections(context, line_pair);
  }
  RemoveDuplicatedEndpointIntersections(intersection_count, line_pairs);
  return intersection_count;
}

[[nodiscard]] auto CalcLineSlope(const Line& line) -> double
{
  return (TailPoint(line).y - HeadPoint(line).y) / (TailPoint(line).x - HeadPoint(line).x);
}

[[nodiscard]] auto CalcLineYAtX(const Line& line, const double target_x) -> double
{
  return ((HeadPoint(line).y - TailPoint(line).y) * (target_x - HeadPoint(line).x) / (HeadPoint(line).x - TailPoint(line).x))
         + HeadPoint(line).y;
}

[[nodiscard]] auto SolveInfiniteLineIntersection(Point& intersection_point, const Line& first_line, const Line& second_line) -> bool
{
  const bool first_line_is_vertical = Equal(HeadPoint(first_line).x, TailPoint(first_line).x);
  const bool second_line_is_vertical = Equal(HeadPoint(second_line).x, TailPoint(second_line).x);

  if (first_line_is_vertical && second_line_is_vertical) {
    return false;
  }

  if (first_line_is_vertical) {
    intersection_point.x = HeadPoint(first_line).x;
    intersection_point.y = CalcLineYAtX(second_line, intersection_point.x);
    return true;
  }

  if (second_line_is_vertical) {
    intersection_point.x = HeadPoint(second_line).x;
    intersection_point.y = CalcLineYAtX(first_line, intersection_point.x);
    return true;
  }

  const double first_line_slope = CalcLineSlope(first_line);
  const double second_line_slope = CalcLineSlope(second_line);
  if (Equal(first_line_slope, second_line_slope)) {
    return false;
  }

  intersection_point.x = (HeadPoint(second_line).y - HeadPoint(first_line).y + (HeadPoint(first_line).x * first_line_slope)
                          - (HeadPoint(second_line).x * second_line_slope))
                         / (first_line_slope - second_line_slope);
  intersection_point.y
      = (HeadPoint(first_line).y + HeadPoint(second_line).y + (first_line_slope * (intersection_point.x - HeadPoint(first_line).x))
         + (second_line_slope * (intersection_point.x - HeadPoint(second_line).x)))
        / kAverageFactor;
  return true;
}

}  // namespace

auto GeomCalc::lineType(const Line& line) -> LineType
{
  return lineType(line.at(kHead), line.at(kTail));
}

auto GeomCalc::lineType(const Point& first_point, const Point& second_point) -> LineType
{
  const auto delta_x = std::abs(first_point.x - second_point.x);
  const auto delta_y = std::abs(first_point.y - second_point.y);
  if (Equal(delta_x, delta_y)) {
    return LineType::kManhattan;
  }
  if (Equal(delta_x, 0)) {
    return LineType::kVertical;
  }
  if (Equal(delta_y, 0)) {
    return LineType::kHorizontal;
  }
  if (delta_x > delta_y) {
    return LineType::kFlat;
  }
  return LineType::kTilt;
}

auto GeomCalc::lineIntersect(Point& intersection_point, const Line& first_line, const Line& second_line) -> IntersectType
{
  LOG_FATAL_IF(HasZeroLength(first_line) || HasZeroLength(second_line)) << "line length is zero";
  if (!boundBoxOverlap(first_line, second_line)) {
    return IntersectType::kNone;
  }
  if (IsSameDirectedLine(first_line, second_line)) {
    intersection_point = first_line.at(kHead);
    return IntersectType::kSame;
  }

  const size_t endpoint_intersection_count = CountEndpointIntersections(intersection_point, first_line, second_line);
  if (endpoint_intersection_count >= kLinePointCount) {
    return IntersectType::kOverlap;
  }

  if (!SolveInfiniteLineIntersection(intersection_point, first_line, second_line)) {
    const bool first_line_is_vertical = Equal(HeadPoint(first_line).x, TailPoint(first_line).x);
    const bool second_line_is_vertical = Equal(HeadPoint(second_line).x, TailPoint(second_line).x);
    if (!(first_line_is_vertical && second_line_is_vertical)) {
      return IntersectType::kNone;
    }
  }

  if (inBoundBox(intersection_point, first_line) && inBoundBox(intersection_point, second_line)) {
    return IntersectType::kCrossing;
  }
  return IntersectType::kNone;
}

auto GeomCalc::lineRelative(const Line& lhs_line, const Line& rhs_line, const size_t& reference_side) -> RelativeType
{
  const auto lhs_line_type = lineType(lhs_line);
  const auto rhs_line_type = lineType(rhs_line);
  LOG_FATAL_IF(lhs_line_type != rhs_line_type) << "line type is not same";

  if (lhs_line_type == LineType::kVertical || lhs_line_type == LineType::kTilt) {
    const auto lhs_max_x = std::max(lhs_line.at(kHead).x, lhs_line.at(kTail).x);
    const auto rhs_max_x = std::max(rhs_line.at(kHead).x, rhs_line.at(kTail).x);
    if ((lhs_max_x <= rhs_max_x && reference_side == kLeft) || (lhs_max_x >= rhs_max_x && reference_side == kRight)) {
      return RelativeType::kLeft;
    }
    return RelativeType::kRight;
  }

  if (lhs_line_type == LineType::kHorizontal || lhs_line_type == LineType::kFlat) {
    const auto lhs_max_y = std::max(lhs_line.at(kHead).y, lhs_line.at(kTail).y);
    const auto rhs_max_y = std::max(rhs_line.at(kHead).y, rhs_line.at(kTail).y);
    if ((lhs_max_y <= rhs_max_y && reference_side == kLeft) || (lhs_max_y >= rhs_max_y && reference_side == kRight)) {
      return RelativeType::kBottom;
    }
    return RelativeType::kTop;
  }

  if (lhs_line_type == LineType::kManhattan) {
    return RelativeType::kManhattanParallel;
  }

  LOG_FATAL << "line type error";
  return RelativeType::kManhattanParallel;
}

auto GeomCalc::lineDist(const Line& lhs_line, const Line& rhs_line) -> LineDistanceResult
{
  LineDistanceResult result;
  result.distance = std::numeric_limits<double>::max();

  Point intersection_point;
  const bool lhs_is_point = isSame(HeadPoint(lhs_line), TailPoint(lhs_line));
  const bool rhs_is_point = isSame(HeadPoint(rhs_line), TailPoint(rhs_line));
  const std::array<LinePairView, kLinePointCount> line_pairs = {
      LinePairView{.source_line = &lhs_line, .target_line = &rhs_line},
      LinePairView{.source_line = &rhs_line, .target_line = &lhs_line},
  };
  const std::array<size_t, kLinePointCount> point_counts = {
      lhs_is_point ? 1U : kLinePointCount,
      rhs_is_point ? 1U : kLinePointCount,
  };

  if (!lhs_is_point && !rhs_is_point && lineIntersect(intersection_point, lhs_line, rhs_line) != IntersectType::kNone) {
    result.closest_points.at(kLeft) = intersection_point;
    result.closest_points.at(kRight) = intersection_point;
    result.distance = 0.0;
    return result;
  }

  for (size_t side = 0; side < line_pairs.size(); ++side) {
    const size_t opposite_side = (side + 1) % kLinePointCount;
    const LinePairView& line_pair = line_pairs.at(side);
    LOG_FATAL_IF(line_pair.source_line == nullptr || line_pair.target_line == nullptr) << "line pair is null";

    for (size_t point_index = 0; point_index < point_counts.at(side); ++point_index) {
      Point closest_point_on_target_line;
      const Point& source_point = line_pair.source_line->at(point_index);
      const double dist = pointToLineDistance(source_point, *line_pair.target_line, closest_point_on_target_line);
      if (dist < result.distance) {
        result.distance = dist;
        result.closest_points.at(side) = source_point;
        result.closest_points.at(opposite_side) = closest_point_on_target_line;
      }
    }
  }

  return result;
}

auto GeomCalc::onLine(Point& point, const Line& line) -> bool
{
  const auto line_length = distance(line.at(kHead), line.at(kTail));
  const auto distance_to_head = distance(point, line.at(kHead));
  const auto distance_to_tail = distance(point, line.at(kTail));
  if (std::abs(distance_to_head + distance_to_tail - line_length) < 2 * kEpsilon) {
    if (Equal(distance_to_head, 0)) {
      point = line.at(kHead);
      return true;
    }
    if (Equal(distance_to_tail, 0)) {
      point = line.at(kTail);
      return true;
    }

    const auto delta_x = std::abs(line.at(kTail).x - line.at(kHead).x);
    const auto delta_y = std::abs(line.at(kTail).y - line.at(kHead).y);
    if (delta_y > delta_x) {
      const auto snapped_x = ((line.at(kTail).x - line.at(kHead).x) * (point.y - line.at(kHead).y) / (line.at(kTail).y - line.at(kHead).y))
                             + line.at(kHead).x;
      if (Equal(snapped_x, point.x)) {
        point.x = snapped_x;
        return true;
      }
    } else {
      const auto snapped_y = ((line.at(kTail).y - line.at(kHead).y) * (point.x - line.at(kHead).x) / (line.at(kTail).x - line.at(kHead).x))
                             + line.at(kHead).y;
      if (Equal(snapped_y, point.y)) {
        point.y = snapped_y;
        return true;
      }
    }
  }
  return false;
}

auto GeomCalc::isParallel(const Line& lhs_line, const Line& rhs_line) -> bool
{
  if (isSame(lhs_line.at(kHead), lhs_line.at(kTail)) || isSame(rhs_line.at(kHead), rhs_line.at(kTail))) {
    return false;
  }

  const auto lhs_delta_x = std::abs(lhs_line.at(kHead).x - lhs_line.at(kTail).x);
  const auto lhs_delta_y = std::abs(lhs_line.at(kHead).y - lhs_line.at(kTail).y);
  const auto rhs_delta_x = std::abs(rhs_line.at(kHead).x - rhs_line.at(kTail).x);
  const auto rhs_delta_y = std::abs(rhs_line.at(kHead).y - rhs_line.at(kTail).y);
  if (Equal(lhs_delta_x, 0) && Equal(rhs_delta_x, 0)) {
    return true;
  }
  if (!Equal(lhs_delta_x, 0) && !Equal(rhs_delta_x, 0) && Equal(lhs_delta_y / lhs_delta_x, rhs_delta_y / rhs_delta_x)) {
    return true;
  }
  return false;
}

}  // namespace icts::bst
