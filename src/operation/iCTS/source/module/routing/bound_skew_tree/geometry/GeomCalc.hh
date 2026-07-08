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
 * @file GeomCalc.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-03-08
 * @brief Geometry primitives and helpers for bound-skew tree routing
 */
#pragma once

#include <cstddef>
#include <vector>

#include "bound_skew_tree/component/Components.hh"

namespace icts::bst {
enum class LineType
{
  kVertical,
  kHorizontal,
  kManhattan,
  kFlat,
  kTilt,
};

enum class IntersectType
{
  kNone,
  kEndpoint,  // intersect point is endpoint of line segment
  kCrossing,  // intersect point is in both line segment
  kOverlap,   // intersect some part of line segment
  kSame,      // two line are same
};

enum class RelativeType
{
  kLeft,
  kRight,
  kTop,
  kBottom,
  kManhattanParallel,
};

struct BoundingBox
{
  double min_x = 0.0;
  double min_y = 0.0;
  double max_x = 0.0;
  double max_y = 0.0;
};

struct LineDistanceResult
{
  PointPair closest_points;
  double distance = 0.0;
};

class GeomCalc
{
 public:
  GeomCalc() = delete;

  /* Calculate */
  // point
  static auto isSame(const Point& first_point, const Point& second_point) -> bool;

  static auto distance(const Point& first_point, const Point& second_point) -> double;

  static auto pointToLineDistanceManhattan(const Point& point, const Line& line, Point& closest_point) -> double;

  static auto pointToLineDistanceNonManhattan(const Point& point, const Line& line, Point& closest_point) -> double;

  static auto pointToLineDistance(const Point& point, const Line& line, Point& closest_point) -> double;

  static auto pointToTransformedRectDistance(const Point& point, TransformedRect& transformed_rect) -> double;

  static auto calcCoord(Point& point, const Line& line, const double& shift) -> void;

  static auto calcRelativeCoord(Point& point, const RelativeType& type, const double& shift) -> void;

  static auto crossProduct(const Point& origin_point, const Point& lhs_point, const Point& rhs_point) -> double;

  // line
  static auto lineType(const Line& line) -> LineType;

  static auto lineType(const Point& first_point, const Point& second_point) -> LineType;

  static auto lineIntersect(Point& intersection_point, const Line& first_line, const Line& second_line) -> IntersectType;

  static auto lineRelative(const Line& lhs_line, const Line& rhs_line, const size_t& reference_side) -> RelativeType;

  static auto lineDist(const Line& lhs_line, const Line& rhs_line) -> LineDistanceResult;

  static auto onLine(Point& point, const Line& line) -> bool;

  static auto isParallel(const Line& lhs_line, const Line& rhs_line) -> bool;

  // box
  static auto inBoundBox(const Point& point, const Line& line) -> bool;

  static auto boundBoxOverlap(const Line& lhs_line, const Line& rhs_line, const double& epsilon = kEpsilon) -> bool;

  static auto boundBoxOverlap(const BoundingBox& lhs_box, const BoundingBox& rhs_box, const double& overlap_epsilon = kEpsilon) -> bool;

  // transformed rect
  static auto transformedRectDistance(TransformedRect& lhs_transformed_rect, TransformedRect& rhs_transformed_rect) -> double;
  static auto makeIntersection(const TransformedRect& first_transformed_rect, const TransformedRect& second_transformed_rect,
                               TransformedRect& intersection) -> void;
  static auto coreMidPoint(TransformedRect& transformed_rect, Point& midpoint) -> void;
  static auto containsTransformedRect(const TransformedRect& inner_transformed_rect, const TransformedRect& outer_transformed_rect) -> bool;
  static auto buildTransformedRect(const TransformedRect& transformed_rect, const double& radius,
                                   TransformedRect& expanded_transformed_rect) -> void;
  static auto transformedRectCore(const TransformedRect& transformed_rect, TransformedRect& core_transformed_rect) -> void;
  static auto transformedRectToPoint(const TransformedRect& transformed_rect, Point& point) -> void;
  static auto transformedRectToRegion(TransformedRect& transformed_rect, Region& region) -> void;
  static auto isSegmentTransformedRect(const TransformedRect& transformed_rect) -> bool;

  // points
  static auto sortPointsByFront(Points& points) -> void;
  static auto sortPointsByValue(Points& points) -> void;
  static auto sortPointsByValueDesc(Points& points) -> void;
  static auto uniquePointLocations(std::vector<Point>& points) -> void;
  static auto uniquePointValues(std::vector<Point>& points) -> void;

  // region
  static auto convexHull(std::vector<Point>& points) -> void;
  static auto centerPoint(const std::vector<Point>& points) -> Point;
  static auto isRegionContain(const Point& point, const std::vector<Point>& region) -> bool;
  static auto closestPointOnRegion(const Point& point, const std::vector<Point>& region) -> Point;

  /* Convert */
  static auto lineToTransformedRect(TransformedRect& transformed_rect, const Line& line) -> void;
  static auto lineToTransformedRect(TransformedRect& transformed_rect, const Point& first_point, const Point& second_point) -> void;
  static auto transformedRectToLine(TransformedRect& transformed_rect, Line& line) -> void;
  static auto transformedRectToLine(TransformedRect& transformed_rect, Point& first_point, Point& second_point) -> void;

  /* Check */
  static auto normalizeTransformedRect(TransformedRect& transformed_rect) -> void;
};

}  // namespace icts::bst
