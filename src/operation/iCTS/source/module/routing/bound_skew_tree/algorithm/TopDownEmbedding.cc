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
 * @file TopDownEmbedding.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief Top-down embedding and point-delay propagation implementation for bound-skew trees (stage C).
 */

#include "bound_skew_tree/algorithm/TopDownEmbedding.hh"

#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

#include "Log.hh"
#include "bound_skew_tree/algorithm/BoundSkewTreeImpl.hh"
#include "bound_skew_tree/component/Components.hh"
#include "bound_skew_tree/config/BSTRoutingConfig.hh"
#include "bound_skew_tree/geometry/GeomCalc.hh"

namespace icts::bst::detail {

auto TopDownEmbedding::embedChild(const EmbeddingStep& embedding_target) -> void
{
  auto* parent = embedding_target.parent;
  auto* child = embedding_target.child;
  const size_t side = embedding_target.side;
  Point child_loc;
  const auto parent_loc = parent->get_location();
  auto merge_region = child->get_merge_region();
  if (merge_region.size() == 4 && isTransformedRectArea(child)) {
    TransformedRect transformed_rect;
    mergeRegionToTransformedRect(merge_region, transformed_rect);
    const auto dist = Geom::pointToTransformedRectDistance(parent_loc, transformed_rect);
    const TransformedRect parent_transformed_rect(parent_loc, dist);
    TransformedRect merge_segment;
    Geom::makeIntersection(parent_transformed_rect, transformed_rect, merge_segment);
    Geom::coreMidPoint(merge_segment, child_loc);
  } else {
    const auto joining_segment_line = parent->get_line(side);
    auto head = BoundSkewTreeImpl::linePoint(joining_segment_line, kHead);
    auto tail = BoundSkewTreeImpl::linePoint(joining_segment_line, kTail);
    Line temp;
    locateBoundarySegment(child, head, temp);
    locateBoundarySegment(child, tail, temp);
    auto horizontal_distance = std::abs(head.x - tail.x);
    auto vertical_distance = std::abs(head.y - tail.y);
    if (Equal(horizontal_distance, 0) && Equal(vertical_distance, 0)) {
      // kHead loc is same as kTail loc
      child_loc = head;
    } else if (Equal(horizontal_distance, 0)) {
      // vertical
      child_loc.x = head.x;
      child_loc.y = parent_loc.y;
    } else if (Equal(vertical_distance, 0)) {
      // horizontal
      child_loc.x = parent_loc.x;
      child_loc.y = head.y;
    } else {
      // others
      Geom::pointToLineDistance(parent_loc, joining_segment_line, child_loc);
    }
    LOG_FATAL_IF(!Geom::onLine(child_loc, joining_segment_line)) << "child loc is not on joining_segment line";
  }
  child->set_location(child_loc);
  if (parent->get_edge_len(side) >= 0) {
    LOG_FATAL_IF(parent->get_edge_len(side) < Geom::distance(parent_loc, child_loc) - kEpsilon) << "edge len is less than distance";
  } else {
    parent->set_edge_len(side, Geom::distance(parent_loc, child_loc));
  }
}

auto TopDownEmbedding::isTransformedRectArea(Area* current_area) -> bool
{
  if (isManhattanArea(current_area)) {
    return true;
  }
  auto merge_region = current_area->get_merge_region();
  if (merge_region.size() != 4) {
    return false;
  }
  int manhattan_line_count = 0;
  for (const auto& line : current_area->getMergeRegionLines()) {
    if (Geom::lineType(line) == LineType::kManhattan) {
      ++manhattan_line_count;
    }
    auto min_delay_delta = std::abs(line.at(kHead).min - line.at(kTail).min);
    auto max_delay_delta = std::abs(line.at(kHead).max - line.at(kTail).max);
    if (min_delay_delta > kEpsilon || max_delay_delta > kEpsilon) {
      return false;
    }
  }
  return manhattan_line_count == 2;
}

auto TopDownEmbedding::isManhattanArea(Area* current_area) -> bool
{
  auto merge_region = current_area->get_merge_region();
  if (merge_region.size() == 1) {
    return true;
  }
  if (merge_region.size() == 2 && Geom::lineType(merge_region.at(kHead), merge_region.at(kTail)) == LineType::kManhattan) {
    return true;
  }
  return false;
}

auto TopDownEmbedding::mergeRegionToTransformedRect(const Region& merge_region, TransformedRect& transformed_rect) -> void
{
  if (merge_region.size() == 1) {
    transformed_rect.makeDiamond(merge_region.front(), 0);
    return;
  }
  if (merge_region.size() == 2) {
    Geom::lineToTransformedRect(transformed_rect, merge_region.at(kHead), merge_region.at(kTail));
    return;
  }
  if (merge_region.size() == 4) {
    TransformedRect left_transformed_rect;
    if (Geom::lineType(merge_region.at(0), merge_region.at(1)) == LineType::kManhattan) {
      Geom::lineToTransformedRect(left_transformed_rect, merge_region.at(0), merge_region.at(1));
    } else {
      LOG_FATAL_IF(Geom::lineType(merge_region.at(2), merge_region.at(1)) != LineType::kManhattan) << "merge_region is not manhattan";
      Geom::lineToTransformedRect(left_transformed_rect, merge_region.at(1), merge_region.at(2));
    }
    TransformedRect right_transformed_rect;
    if (Geom::lineType(merge_region.at(2), merge_region.at(3)) == LineType::kManhattan) {
      Geom::lineToTransformedRect(right_transformed_rect, merge_region.at(2), merge_region.at(3));
    } else {
      LOG_FATAL_IF(Geom::lineType(merge_region.at(0), merge_region.at(3)) != LineType::kManhattan) << "merge_region is not manhattan";
      Geom::lineToTransformedRect(right_transformed_rect, merge_region.at(3), merge_region.at(0));
    }
    transformed_rect = left_transformed_rect;
    transformed_rect.enclose(right_transformed_rect);
    return;
  }
  LOG_FATAL << "merge_region size is not 1, 2 or 4";
}

auto TopDownEmbedding::calcAreaLineType(const Area& current_area) -> LineType
{
  auto line = current_area.get_line(kLeft);
  return Geom::lineType(line);
}

auto TopDownEmbedding::calcConvexHull(Area* current_area) -> void
{
  auto merge_region = current_area->get_merge_region();
  Geom::convexHull(merge_region);
  current_area->set_convex_hull(merge_region);
}

auto TopDownEmbedding::calcJoiningRegionArea(const Line& first_line, const Line& second_line) -> double
{
  auto min_x = std::min({first_line.at(kHead).x, first_line.at(kTail).x, second_line.at(kHead).x, second_line.at(kTail).x});
  auto max_x = std::max({first_line.at(kHead).x, first_line.at(kTail).x, second_line.at(kHead).x, second_line.at(kTail).x});
  auto min_y = std::min({first_line.at(kHead).y, first_line.at(kTail).y, second_line.at(kHead).y, second_line.at(kTail).y});
  auto max_y = std::max({first_line.at(kHead).y, first_line.at(kTail).y, second_line.at(kHead).y, second_line.at(kTail).y});
  auto bound_area = (max_x - min_x) * (max_y - min_y);
  auto tri_area_1 = BoundSkewTreeImpl::kHalfFactor * std::abs(first_line.at(kHead).x - first_line.at(kTail).x)
                    * std::abs(first_line.at(kHead).y - first_line.at(kTail).y);
  auto tri_area_2 = BoundSkewTreeImpl::kHalfFactor * std::abs(second_line.at(kHead).x - second_line.at(kTail).x)
                    * std::abs(second_line.at(kHead).y - second_line.at(kTail).y);
  auto jr_area = bound_area - tri_area_1 - tri_area_2;
  LOG_FATAL_IF(jr_area < 0) << "joining_region area is negative";
  return jr_area;
}

auto TopDownEmbedding::locateBoundarySegment(Area* current_area, Point& point, Line& boundary_segment) -> void
{
  for (auto merge_region_line : current_area->getMergeRegionLines()) {
    boundary_segment = merge_region_line;
    if (Geom::onLine(point, boundary_segment)) {
      return;
    }
  }
  LOG_FATAL << "point is not located in area";
}

auto TopDownEmbedding::calcSimplePointDelays(Point& point, Line& boundary_segment) const -> bool
{
  LOG_FATAL_IF(!Geom::onLine(point, boundary_segment)) << "point is not located in line";
  const auto dist = Geom::distance(point, boundary_segment.at(kHead));
  const auto horizontal_distance = std::abs(boundary_segment.at(kHead).x - boundary_segment.at(kTail).x);
  const auto vertical_distance = std::abs(boundary_segment.at(kHead).y - boundary_segment.at(kTail).y);
  const auto length = horizontal_distance + vertical_distance;
  if (Equal(dist, 0)) {
    point.min = boundary_segment.at(kHead).min;
    point.max = boundary_segment.at(kHead).max;
    return true;
  }
  if (Geom::isSame(point, boundary_segment.at(kTail))) {
    point.min = boundary_segment.at(kTail).min;
    point.max = boundary_segment.at(kTail).max;
    return true;
  }
  if (Equal(horizontal_distance, vertical_distance)) {
    // line is manhattan arc
    LOG_FATAL_IF(!Equal(boundary_segment.at(kHead).min, boundary_segment.at(kTail).min)
                 || !Equal(boundary_segment.at(kHead).max, boundary_segment.at(kTail).max))
        << "manhattan arc endpoint's delay is not same";
    point.min = boundary_segment.at(kHead).min = boundary_segment.at(kTail).min;
    point.max = boundary_segment.at(kHead).max = boundary_segment.at(kTail).max;
    return true;
  }
  if (Equal(horizontal_distance, 0) || Equal(vertical_distance, 0)) {
    // line is vertical or horizontal
    auto alpha = Equal(horizontal_distance, 0) ? _impl._delay_quadratic_factor.vertical : _impl._delay_quadratic_factor.horizontal;
    auto beta = ((boundary_segment.at(kTail).min - boundary_segment.at(kHead).min) / length) - (alpha * length);
    point.min = boundary_segment.at(kHead).min + (alpha * dist * dist) + (beta * dist);
    beta = ((boundary_segment.at(kTail).max - boundary_segment.at(kHead).max) / length) - (alpha * length);
    point.max = boundary_segment.at(kHead).max + (alpha * dist * dist) + (beta * dist);
    return true;
  }
  return false;
}

auto TopDownEmbedding::calcSegmentPointDelays(Point& point, Line& boundary_segment) const -> void
{
  if (!calcSimplePointDelays(point, boundary_segment)) {
    LOG_FATAL << "segment-only point delay calculation requires area context";
    return;
  }
  checkPointDelay(point);
}

auto TopDownEmbedding::calcPointDelays(const Area& current_area, Point& point, Line& boundary_segment) const -> void
{
  if (!calcSimplePointDelays(point, boundary_segment)) {
    LOG_FATAL_IF(!Equal(pointSkew(boundary_segment.at(kHead)), _impl._skew_bound)
                 || !Equal(pointSkew(boundary_segment.at(kTail)), _impl._skew_bound))
        << "thera are skew reservation in line";
    calcIrregularPointDelays(current_area, point, boundary_segment);
  }
  checkPointDelay(point);
}

auto TopDownEmbedding::updatePointDelaysByEndSide(const Area& current_area, const size_t& end_side, Point& point) const -> void
{
  auto* left_child = current_area.get_left();
  auto* right_child = current_area.get_right();
  if (left_child == nullptr || right_child == nullptr) {
    LOG_FATAL << "updatePointDelaysByEndSide requires both child areas";
    return;
  }

  const auto left_line = current_area.get_line(kLeft);
  const auto right_line = current_area.get_line(kRight);
  const auto left_endpoint = BoundSkewTreeImpl::linePoint(left_line, end_side);
  const auto right_endpoint = BoundSkewTreeImpl::linePoint(right_line, end_side);
  const auto delay_left = pointDelayIncrease(point, left_endpoint, left_child->get_cap_load(), _impl._rc_pattern);
  const auto delay_right = pointDelayIncrease(point, right_endpoint, right_child->get_cap_load(), _impl._rc_pattern);
  point.min = std::min(left_endpoint.min + delay_left, right_endpoint.min + delay_right);
  point.max = std::max(left_endpoint.max + delay_left, right_endpoint.max + delay_right);
}

auto TopDownEmbedding::calcIrregularPointDelays(const Area& current_area, Point& point, Line& boundary_segment) const -> void
{
  auto* left_child = current_area.get_left();
  auto* right_child = current_area.get_right();
  if (left_child == nullptr || right_child == nullptr) {
    LOG_FATAL << "calcIrregularPointDelays requires both child areas";
    return;
  }

  auto horizontal_distance = std::abs(boundary_segment.at(kHead).x - boundary_segment.at(kTail).x);
  auto vertical_distance = std::abs(boundary_segment.at(kHead).y - boundary_segment.at(kTail).y);
  auto left_line = current_area.get_line(kLeft);
  auto right_line = current_area.get_line(kRight);
  auto joining_segment_type = Geom::lineType(current_area.get_line(kLeft));
  if (joining_segment_type == LineType::kManhattan) {
    LOG_FATAL_IF(!Geom::isSame(left_line.at(kHead), left_line.at(kTail)) || !Geom::isSame(right_line.at(kHead), right_line.at(kTail)))
        << "endpoint should be same, left head: [" << left_line.at(kHead).x << ", " << left_line.at(kHead).y << "], left tail: ["
        << left_line.at(kTail).x << ", " << left_line.at(kTail).y << "], right head: [" << right_line.at(kHead).x << ", "
        << right_line.at(kHead).y << "], right tail: [" << right_line.at(kTail).x << ", " << right_line.at(kTail).y << "]";

    auto delay_left = pointDelayIncrease(left_line.at(kHead), point, left_child->get_cap_load(), _impl._rc_pattern);
    auto delay_right = pointDelayIncrease(right_line.at(kHead), point, right_child->get_cap_load(), _impl._rc_pattern);
    point.min = std::min(left_line.at(kHead).min + delay_left, right_line.at(kHead).min + delay_right);
    point.max = std::max(left_line.at(kHead).max + delay_left, right_line.at(kHead).max + delay_right);
    LOG_FATAL_IF(pointSkew(point) >= _impl._skew_bound + kEpsilon) << "skew is larger than skew bound";
  } else {
    LOG_FATAL_IF(joining_segment_type != LineType::kVertical && joining_segment_type != LineType::kHorizontal)
        << "joining_segment type is not vertical or horizontal";
    auto dist = Geom::distance(point, boundary_segment.at(kHead));
    auto length = horizontal_distance + vertical_distance;
    double alpha = 0;
    if (horizontal_distance > vertical_distance) {
      auto slope = vertical_distance / horizontal_distance;
      auto ratio = std::pow(1 + std::abs(slope), 2);
      alpha = (_impl._delay_quadratic_factor.horizontal + (slope * slope * _impl._delay_quadratic_factor.vertical)) / ratio;
    } else {
      auto slope = horizontal_distance / vertical_distance;
      auto ratio = std::pow(1 + std::abs(slope), 2);
      alpha = (_impl._delay_quadratic_factor.vertical + (slope * slope * _impl._delay_quadratic_factor.horizontal)) / ratio;
    }
    auto beta = ((boundary_segment.at(kTail).max - boundary_segment.at(kHead).max) / length) - (alpha * length);
    point.max = boundary_segment.at(kHead).max + (alpha * dist * dist) + (beta * dist);
    beta = ((boundary_segment.at(kTail).min - boundary_segment.at(kHead).min) / length) - (alpha * length);
    point.min = boundary_segment.at(kHead).min + (alpha * dist * dist) + (beta * dist);
  }
}

auto TopDownEmbedding::pointDelayIncrease(const Point& lhs_point, const Point& rhs_point, const double& cap_load,
                                          const BSTRoutingRCPattern& rc_pattern) const -> double
{
  auto delay = calcDelayIncrease(std::abs(lhs_point.x - rhs_point.x), std::abs(lhs_point.y - rhs_point.y), cap_load, rc_pattern);
  LOG_FATAL_IF(delay < 0) << "point increase delay is negative";
  return delay;
}

auto TopDownEmbedding::pointDelayIncrease(const Point& lhs_point, const Point& rhs_point, const double& length, const double& cap_load,
                                          const BSTRoutingRCPattern& rc_pattern) const -> double
{
  auto [horizontal_distance, vertical_distance] = BoundSkewTreeImpl::calcManhattanDistanceComponents(lhs_point, rhs_point);
  LOG_FATAL_IF(!Equal(length, horizontal_distance + vertical_distance) && length < horizontal_distance + vertical_distance)
      << "length is less than horizontal_distance + vertical_distance";
  double delay = 0;
  if (Equal(horizontal_distance, 0)) {
    delay = calcDelayIncrease(0, length, cap_load, rc_pattern);
  } else if (Equal(vertical_distance, 0)) {
    delay = calcDelayIncrease(length, 0, cap_load, rc_pattern);
  } else {
    delay = calcDelayIncrease(horizontal_distance, vertical_distance, cap_load, rc_pattern);
    if (length > horizontal_distance + vertical_distance) {
      delay += calcDelayIncrease(
          0, length - horizontal_distance - vertical_distance,
          cap_load + (_impl._unit_horizontal_capacitance * horizontal_distance) + (_impl._unit_vertical_capacitance * vertical_distance),
          rc_pattern);
    }
  }
  LOG_FATAL_IF(delay < 0) << "point increase delay is negative";
  return delay;
}

auto TopDownEmbedding::calcDelayIncrease(const double& horizontal_length, const double& vertical_length, const double& cap_load,
                                         const BSTRoutingRCPattern& rc_pattern) const -> double
{
  double delay = 0;
  switch (rc_pattern) {
    case BSTRoutingRCPattern::kHV:
      delay = (_impl._unit_horizontal_resistance * horizontal_length
               * ((_impl._unit_horizontal_capacitance * horizontal_length / 2) + cap_load))
              + (_impl._unit_vertical_resistance * vertical_length
                 * ((_impl._unit_vertical_capacitance * vertical_length / 2) + cap_load
                    + (horizontal_length * _impl._unit_horizontal_capacitance)));
      break;
    case BSTRoutingRCPattern::kVH:
      delay = (_impl._unit_vertical_resistance * vertical_length * ((_impl._unit_vertical_capacitance * vertical_length / 2) + cap_load))
              + (_impl._unit_horizontal_resistance * horizontal_length
                 * ((_impl._unit_horizontal_capacitance * horizontal_length / 2) + cap_load
                    + (vertical_length * _impl._unit_vertical_capacitance)));
      break;
    case BSTRoutingRCPattern::kSingle:
      delay = _impl._unit_horizontal_resistance * (horizontal_length + vertical_length)
              * ((_impl._unit_horizontal_capacitance * (horizontal_length + vertical_length) / 2) + cap_load);
      break;
    default:
      LOG_FATAL << "unknown rc_pattern";
      break;
  }
  return delay;
}

auto TopDownEmbedding::pointSkew(const Point& point) -> double
{
  return point.max - point.min;
}

auto TopDownEmbedding::getJoiningRegionLine(const size_t& side) const -> Line
{
  return Line{_impl.joiningRegionPoint(side, kHead), _impl.joiningRegionPoint(side, kTail)};
}

auto TopDownEmbedding::getJoiningSegmentLine(const size_t& side) const -> Line
{
  return Line{_impl.joiningSegmentPoint(side, kHead), _impl.joiningSegmentPoint(side, kTail)};
}

auto TopDownEmbedding::setJoiningRegionLine(const size_t& side, const Line& line) -> void
{
  _impl.joiningRegionPoint(side, kHead) = BoundSkewTreeImpl::linePoint(line, kHead);
  _impl.joiningRegionPoint(side, kTail) = BoundSkewTreeImpl::linePoint(line, kTail);
}

auto TopDownEmbedding::setJoiningSegmentLine(const size_t& side, const Line& line) -> void
{
  _impl.joiningSegmentPoint(side, kHead) = BoundSkewTreeImpl::linePoint(line, kHead);
  _impl.joiningSegmentPoint(side, kTail) = BoundSkewTreeImpl::linePoint(line, kTail);
}

auto TopDownEmbedding::checkPointDelay(Point& point) -> void
{
  // LOG_ERROR_IF(point.min <= -kEpsilon) << "point min delay is negative";
  LOG_FATAL_IF(point.max - point.min <= -kEpsilon) << "point skew is negative";
  if (point.min < -kEpsilon) {
    point.min = 0;
  }
  if (point.max < point.min + kEpsilon) {
    point.max = point.min;
  }
}

auto TopDownEmbedding::checkJoiningSegmentMergeSegment() const -> void
{
  TransformedRect left;
  TransformedRect right;
  auto left_joining_segment = getJoiningSegmentLine(kLeft);
  auto right_joining_segment = getJoiningSegmentLine(kRight);
  Geom::lineToTransformedRect(left, left_joining_segment);
  Geom::lineToTransformedRect(right, right_joining_segment);
  LOG_FATAL_IF(!Geom::containsTransformedRect(left, _impl.mergeSegment(kLeft)))
      << "left joining_segment is not contain in left merge_segment";
  LOG_FATAL_IF(!Geom::containsTransformedRect(right, _impl.mergeSegment(kRight)))
      << "right joining_segment is not contain in right merge_segment";
}

auto TopDownEmbedding::checkUpdatedJoiningSegment(const Area* current_area, Line& left_line, Line& right_line) const -> void
{
  const auto is_parallel = Geom::isParallel(left_line, right_line);
  const auto line_type = Geom::lineType(left_line);
  if (is_parallel) {
    LOG_FATAL_IF(line_type == LineType::kFlat || line_type == LineType::kTilt) << "not consider case";
  }
  const auto left_joining_segment = getJoiningSegmentLine(kLeft);
  const auto right_joining_segment = getJoiningSegmentLine(kRight);
  const auto line_distance = Geom::lineDist(left_joining_segment, right_joining_segment);
  const auto dist = line_distance.distance;
  LOG_FATAL_IF(
      !Geom::isSame(BoundSkewTreeImpl::linePoint(left_joining_segment, kHead), BoundSkewTreeImpl::linePoint(left_joining_segment, kTail))
      && !Geom::isSame(BoundSkewTreeImpl::linePoint(right_joining_segment, kHead),
                       BoundSkewTreeImpl::linePoint(right_joining_segment, kTail))
      && !Geom::isParallel(left_joining_segment, right_joining_segment))
      << "joining_segment line error";
  LOG_FATAL_IF(!Equal(dist, current_area->get_radius())) << "distance between joinsegments not equal to radius";
  auto left_joining_segment_head = BoundSkewTreeImpl::linePoint(left_joining_segment, kHead);
  auto left_joining_segment_tail = BoundSkewTreeImpl::linePoint(left_joining_segment, kTail);
  LOG_FATAL_IF(!Geom::onLine(left_joining_segment_head, left_line) || !Geom::onLine(left_joining_segment_tail, left_line))
      << "left_joining_segment not in left section";
  auto right_joining_segment_head = BoundSkewTreeImpl::linePoint(right_joining_segment, kHead);
  auto right_joining_segment_tail = BoundSkewTreeImpl::linePoint(right_joining_segment, kTail);
  LOG_FATAL_IF(!Geom::onLine(right_joining_segment_head, right_line) || !Geom::onLine(right_joining_segment_tail, right_line))
      << "left_joining_segment not in left section";
}

}  // namespace icts::bst::detail
