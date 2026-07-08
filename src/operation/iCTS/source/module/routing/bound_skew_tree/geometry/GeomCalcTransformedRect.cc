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
 * @file GeomCalcTransformedRect.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Transformed-rectangle geometry operations for bound-skew tree routing
 */
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <ostream>

#include "Log.hh"
#include "bound_skew_tree/component/Components.hh"
#include "bound_skew_tree/geometry/GeomCalc.hh"

namespace icts::bst {
auto GeomCalc::transformedRectDistance(TransformedRect& lhs_transformed_rect, TransformedRect& rhs_transformed_rect) -> double
{
  normalizeTransformedRect(lhs_transformed_rect);
  normalizeTransformedRect(rhs_transformed_rect);

  const auto lhs_x_low = lhs_transformed_rect.x_low();
  const auto lhs_x_high = lhs_transformed_rect.x_high();
  const auto lhs_y_low = lhs_transformed_rect.y_low();
  const auto lhs_y_high = lhs_transformed_rect.y_high();
  const auto rhs_x_low = rhs_transformed_rect.x_low();
  const auto rhs_x_high = rhs_transformed_rect.x_high();
  const auto rhs_y_low = rhs_transformed_rect.y_low();
  const auto rhs_y_high = rhs_transformed_rect.y_high();

  const bool x_is_intersect = (lhs_x_low <= rhs_x_high) && (rhs_x_low <= lhs_x_high);
  const bool y_is_intersect = (lhs_y_low <= rhs_y_high) && (rhs_y_low <= lhs_y_high);
  if (!x_is_intersect && !y_is_intersect) {
    const auto low_to_low = std::max(std::abs(lhs_x_low - rhs_x_low), std::abs(lhs_y_low - rhs_y_low));
    const auto low_to_high = std::max(std::abs(lhs_x_low - rhs_x_high), std::abs(lhs_y_low - rhs_y_high));
    const auto high_to_low = std::max(std::abs(lhs_x_high - rhs_x_low), std::abs(lhs_y_high - rhs_y_low));
    const auto high_to_high = std::max(std::abs(lhs_x_high - rhs_x_high), std::abs(lhs_y_high - rhs_y_high));
    return std::min({low_to_low, low_to_high, high_to_low, high_to_high});
  }
  if (!x_is_intersect) {
    const auto lhs_low_to_rhs_high = lhs_x_low - rhs_x_high;
    const auto rhs_low_to_lhs_high = rhs_x_low - lhs_x_high;
    return std::max(lhs_low_to_rhs_high, rhs_low_to_lhs_high);
  }
  if (!y_is_intersect) {
    const auto lhs_low_to_rhs_high = lhs_y_low - rhs_y_high;
    const auto rhs_low_to_lhs_high = rhs_y_low - lhs_y_high;
    return std::max(lhs_low_to_rhs_high, rhs_low_to_lhs_high);
  }
  return 0;
}

auto GeomCalc::makeIntersection(const TransformedRect& first_transformed_rect, const TransformedRect& second_transformed_rect,
                                TransformedRect& intersection) -> void
{
  intersection.x_low(std::max(first_transformed_rect.x_low(), second_transformed_rect.x_low()));
  intersection.x_high(std::min(first_transformed_rect.x_high(), second_transformed_rect.x_high()));
  intersection.y_low(std::max(first_transformed_rect.y_low(), second_transformed_rect.y_low()));
  intersection.y_high(std::min(first_transformed_rect.y_high(), second_transformed_rect.y_high()));
  normalizeTransformedRect(intersection);
}

auto GeomCalc::coreMidPoint(TransformedRect& transformed_rect, Point& midpoint) -> void
{
  const auto transformed_x = (transformed_rect.x_low() + transformed_rect.x_high()) / 2;
  const auto transformed_y = (transformed_rect.y_low() + transformed_rect.y_high()) / 2;
  midpoint.x = (transformed_x + transformed_y) / 2;
  midpoint.y = (transformed_y - transformed_x) / 2;
}

auto GeomCalc::containsTransformedRect(const TransformedRect& inner_transformed_rect, const TransformedRect& outer_transformed_rect) -> bool
{
  return inner_transformed_rect.x_high() <= outer_transformed_rect.x_high() + kEpsilon
         && inner_transformed_rect.x_low() >= outer_transformed_rect.x_low() - kEpsilon
         && inner_transformed_rect.y_high() <= outer_transformed_rect.y_high() + kEpsilon
         && inner_transformed_rect.y_low() >= outer_transformed_rect.y_low() - kEpsilon;
}

auto GeomCalc::buildTransformedRect(const TransformedRect& transformed_rect, const double& radius,
                                    TransformedRect& expanded_transformed_rect) -> void
{
  expanded_transformed_rect.x_low(transformed_rect.x_low() - radius);
  expanded_transformed_rect.x_high(transformed_rect.x_high() + radius);
  expanded_transformed_rect.y_low(transformed_rect.y_low() - radius);
  expanded_transformed_rect.y_high(transformed_rect.y_high() + radius);
}

auto GeomCalc::transformedRectCore(const TransformedRect& transformed_rect, TransformedRect& core_transformed_rect) -> void
{
  if (transformed_rect.x_high() - transformed_rect.x_low() < transformed_rect.y_high() - transformed_rect.y_low()) {
    core_transformed_rect.y_low(transformed_rect.y_low());
    core_transformed_rect.y_high(transformed_rect.y_high());
    const auto center_x = (transformed_rect.x_low() + transformed_rect.x_high()) / 2;
    core_transformed_rect.x_low(center_x);
    core_transformed_rect.x_high(center_x);
    return;
  }

  core_transformed_rect.x_low(transformed_rect.x_low());
  core_transformed_rect.x_high(transformed_rect.x_high());
  const auto center_y = (transformed_rect.y_low() + transformed_rect.y_high()) / 2;
  core_transformed_rect.y_low(center_y);
  core_transformed_rect.y_high(center_y);
}

auto GeomCalc::transformedRectToPoint(const TransformedRect& transformed_rect, Point& point) -> void
{
  point.x = (transformed_rect.y_low() + transformed_rect.x_high()) / 2;
  point.y = (transformed_rect.y_low() - transformed_rect.x_high()) / 2;
}

auto GeomCalc::transformedRectToRegion(TransformedRect& transformed_rect, Region& region) -> void
{
  const auto transformed_width = transformed_rect.x_high() - transformed_rect.x_low();
  const auto transformed_height = transformed_rect.y_high() - transformed_rect.y_low();
  if (Equal(transformed_width, 0) && Equal(transformed_height, 0)) {
    Point point;
    transformedRectToPoint(transformed_rect, point);
    region.push_back(point);
    return;
  }
  if (Equal(transformed_width, 0) || Equal(transformed_height, 0)) {
    Point head_point;
    Point tail_point;
    transformedRectToLine(transformed_rect, head_point, tail_point);
    region.push_back(head_point);
    region.push_back(tail_point);
    return;
  }

  TransformedRect edge_transformed_rect(transformed_rect.x_high(), transformed_rect.x_high(), transformed_rect.y_low(),
                                        transformed_rect.y_high());
  Point head_point;
  Point tail_point;

  transformedRectToLine(edge_transformed_rect, head_point, tail_point);
  region.push_back(head_point);
  region.push_back(tail_point);

  edge_transformed_rect.x_low(transformed_rect.x_low());
  edge_transformed_rect.x_high(transformed_rect.x_low());

  transformedRectToLine(edge_transformed_rect, head_point, tail_point);
  region.push_back(head_point);
  region.push_back(tail_point);
}

auto GeomCalc::isSegmentTransformedRect(const TransformedRect& transformed_rect) -> bool
{
  return Equal(transformed_rect.x_low(), transformed_rect.x_high()) || Equal(transformed_rect.y_low(), transformed_rect.y_high());
}

auto GeomCalc::lineToTransformedRect(TransformedRect& transformed_rect, const Line& line) -> void
{
  lineToTransformedRect(transformed_rect, line.at(kHead), line.at(kTail));
}

auto GeomCalc::lineToTransformedRect(TransformedRect& transformed_rect, const Point& first_point, const Point& second_point) -> void
{
  if (first_point.y <= second_point.y) {
    transformed_rect.x_low(second_point.x - second_point.y);
    transformed_rect.x_high(first_point.x - first_point.y);
    transformed_rect.y_low(first_point.x + first_point.y);
    transformed_rect.y_high(second_point.x + second_point.y);
  } else {
    transformed_rect.x_low(first_point.x - first_point.y);
    transformed_rect.x_high(second_point.x - second_point.y);
    transformed_rect.y_low(second_point.x + second_point.y);
    transformed_rect.y_high(first_point.x + first_point.y);
  }
  normalizeTransformedRect(transformed_rect);
}

auto GeomCalc::transformedRectToLine(TransformedRect& transformed_rect, Line& line) -> void
{
  transformedRectToLine(transformed_rect, line.at(kHead), line.at(kTail));
}

auto GeomCalc::transformedRectToLine(TransformedRect& transformed_rect, Point& first_point, Point& second_point) -> void
{
  normalizeTransformedRect(transformed_rect);
  first_point.x = (transformed_rect.y_low() + transformed_rect.x_high()) / 2;
  first_point.y = (transformed_rect.y_low() - transformed_rect.x_high()) / 2;
  second_point.x = (transformed_rect.y_high() + transformed_rect.x_low()) / 2;
  second_point.y = (transformed_rect.y_high() - transformed_rect.x_low()) / 2;
  LOG_FATAL_IF(first_point.y > second_point.y) << "second point y should be larger than first point y";
}

auto GeomCalc::normalizeTransformedRect(TransformedRect& transformed_rect) -> void
{
  const auto x_low = transformed_rect.x_low();
  const auto x_high = transformed_rect.x_high();
  if (Equal(x_low, x_high)) {
    const auto average_x = (x_low + x_high) / 2;
    transformed_rect.x_low(average_x);
    transformed_rect.x_high(average_x);
  }

  const auto y_low = transformed_rect.y_low();
  const auto y_high = transformed_rect.y_high();
  if (Equal(y_low, y_high)) {
    const auto average_y = (y_low + y_high) / 2;
    transformed_rect.y_low(average_y);
    transformed_rect.y_high(average_y);
  }
  LOG_FATAL_IF(transformed_rect.x_low() > transformed_rect.x_high() || transformed_rect.y_low() > transformed_rect.y_high())
      << "transformed rect is not valid";
}

}  // namespace icts::bst
