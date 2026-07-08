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
 * @file BottomUpMergeJoining.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief Bound-skew tree joining segment and joining region construction implementation (stage B.1).
 */

#include "bound_skew_tree/algorithm/BottomUpMergeJoining.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iterator>
#include <limits>
#include <ostream>
#include <string_view>
#include <utility>
#include <vector>

#include "Log.hh"
#include "bound_skew_tree/algorithm/BottomUpMergeBalance.hh"
#include "bound_skew_tree/algorithm/BottomUpMergeInfeasibility.hh"
#include "bound_skew_tree/algorithm/BoundSkewTreeImpl.hh"
#include "bound_skew_tree/algorithm/TopDownEmbedding.hh"
#include "bound_skew_tree/component/Components.hh"
#include "bound_skew_tree/geometry/GeomCalc.hh"

namespace icts::bst::detail {
namespace {

auto checkMatchingEndpoint(const Point& joining_segment_point, const Point& line_point, const std::string_view side_name,
                           const std::string_view endpoint_name) -> void
{
  LOG_FATAL_IF(!Geom::isSame(joining_segment_point, line_point))
      << side_name << " joining segment is not same as " << side_name << " line at " << endpoint_name;
}

}  // namespace

auto BottomUpMergeJoining::calcJoiningSegment(const MergeAreas& merge_areas) -> void
{
  auto* parent = merge_areas.parent;
  auto* left = merge_areas.left;
  auto* right = merge_areas.right;
  initSide();
  parent->set_radius(std::numeric_limits<double>::max());
  auto left_lines = left->getConvexHullLines();
  auto right_lines = right->getConvexHullLines();
  std::ranges::for_each(left_lines, [&](Line& left_line) -> void {
    std::ranges::for_each(right_lines, [&](Line& right_line) -> void { calcJoiningSegment(parent, left_line, right_line); });
  });
  calcJoiningSegmentDelay(left, right);
  if (Geom::lineType(_impl.topDownEmbedding().getJoiningSegmentLine(kLeft)) == LineType::kManhattan) {
    _impl.topDownEmbedding().checkJoiningSegmentMergeSegment();
  }
}

auto BottomUpMergeJoining::processJoiningSegment(Area* current_area) -> void
{
  FOR_EACH_BST_SIDE(side)
  {
    auto& joining_segment_points = _impl.joiningSegmentPoints(side);
    auto& head_point = BoundSkewTreeImpl::pointAt(joining_segment_points, kHead);
    auto& tail_point = BoundSkewTreeImpl::pointAt(joining_segment_points, kTail);
    if (Equal(head_point.y, tail_point.y)) {
      if (head_point.x < tail_point.x) {
        std::swap(head_point, tail_point);
      }
    } else if (head_point.y < tail_point.y) {
      std::swap(head_point, tail_point);
    }
  }
  FOR_EACH_BST_SIDE(side)
  {
    _impl.topDownEmbedding().setJoiningRegionLine(side, _impl.topDownEmbedding().getJoiningSegmentLine(side));
    current_area->set_line(side, _impl.topDownEmbedding().getJoiningSegmentLine(side));
  }
}

auto BottomUpMergeJoining::constructMergeRegion(const MergeAreas& merge_areas) -> void
{
  auto* parent = merge_areas.parent;
  calcJoiningRegion(merge_areas);
  calcJoiningRegionCorner(*parent);
  _impl.bottomUpMergeBalance().calcBalancePoint(*parent);
  _impl.bottomUpMergeBalance().calcFeasibleMergeSegmentPoints(*parent);
  if (_impl.bottomUpMergeBalance().hasFeasibleMergeSegmentOnJoiningRegion()) {
    _impl.bottomUpMergeBalance().constructFeasibleMergeRegion(parent);
  } else {
    _impl.bottomUpMergeInfeasibility().constructInfeasibleMergeRegion(parent);
  }
  if (Geom::lineType(parent->get_line(kLeft)) == LineType::kManhattan && parent->get_edge_len(kLeft) >= 0) {
    LOG_FATAL_IF(parent->get_edge_len(kRight) < 0.0) << "right edge length is negative";
    _impl.bottomUpMergeInfeasibility().constructTransformedRectMergeRegion(parent);
  }
  auto merge_region = parent->get_merge_region();
  Geom::uniquePointLocations(merge_region);
  parent->set_merge_region(merge_region);
  TopDownEmbedding::calcConvexHull(parent);
}

auto BottomUpMergeJoining::initSide() -> void
{
  FOR_EACH_BST_SIDE(side)
  {
    _impl.joiningRegionPoints(side) = {Point(), Point()};
    _impl.joiningSegmentPoints(side) = {Point(), Point()};
  }
}

auto BottomUpMergeJoining::calcJoiningSegment(Area* current_area, Line& left_line, Line& right_line) -> void
{
  auto line_distance = Geom::lineDist(left_line, right_line);
  auto line_dist = line_distance.distance;
  auto closest_pair = line_distance.closest_points;
  auto left_joining_segment_backup = _impl.topDownEmbedding().getJoiningSegmentLine(kLeft);
  auto right_joining_segment_backup = _impl.topDownEmbedding().getJoiningSegmentLine(kRight);
  auto left_merge_segment_backup = _impl.mergeSegment(kLeft);
  auto right_merge_segment_backup = _impl.mergeSegment(kRight);
  if (Equal(line_dist, current_area->get_radius())) {
    current_area->set_radius(line_dist);
    updateJoiningSegment(current_area, left_line, right_line, closest_pair);
    auto origin_area = TopDownEmbedding::calcJoiningRegionArea(left_joining_segment_backup, right_joining_segment_backup);
    auto new_area = TopDownEmbedding::calcJoiningRegionArea(_impl.topDownEmbedding().getJoiningSegmentLine(kLeft),
                                                            _impl.topDownEmbedding().getJoiningSegmentLine(kRight));
    if (origin_area >= new_area) {
      _impl.topDownEmbedding().setJoiningSegmentLine(kLeft, left_joining_segment_backup);
      _impl.topDownEmbedding().setJoiningSegmentLine(kRight, right_joining_segment_backup);
      if (Geom::lineType(left_joining_segment_backup) == LineType::kManhattan) {
        _impl.mergeSegment(kLeft) = left_merge_segment_backup;
        _impl.mergeSegment(kRight) = right_merge_segment_backup;
      }
    }
  } else if (line_dist < current_area->get_radius()) {
    current_area->set_radius(line_dist);
    updateJoiningSegment(current_area, left_line, right_line, closest_pair);
  }
  if (Geom::lineType(_impl.topDownEmbedding().getJoiningSegmentLine(kLeft)) == LineType::kManhattan) {
    _impl.topDownEmbedding().checkJoiningSegmentMergeSegment();
  }
}

auto BottomUpMergeJoining::calcJoiningSegmentDelay(Area* left, Area* right) -> void
{
  FOR_EACH_BST_SIDE(left_side)
  {
    Line line;
    auto& left_point = BoundSkewTreeImpl::pointAt(_impl.joiningSegmentPoints(kLeft), left_side);
    TopDownEmbedding::locateBoundarySegment(left, left_point, line);
    _impl.topDownEmbedding().calcPointDelays(*left, left_point, line);
  }
  FOR_EACH_BST_SIDE(right_side)
  {
    Line line;
    auto& right_point = BoundSkewTreeImpl::pointAt(_impl.joiningSegmentPoints(kRight), right_side);
    TopDownEmbedding::locateBoundarySegment(right, right_point, line);
    _impl.topDownEmbedding().calcPointDelays(*right, right_point, line);
  }
}

auto BottomUpMergeJoining::updateJoiningSegment(Area* current_area, Line& left_line, Line& right_line, PointPair closest_pair) -> void
{
  auto left_type = Geom::lineType(left_line);
  auto right_type = Geom::lineType(right_line);
  auto left_is_manhattan = left_type == LineType::kManhattan;
  auto right_is_manhattan = right_type == LineType::kManhattan;
  TransformedRect left_merge_segment;
  TransformedRect right_merge_segment;
  if (left_is_manhattan) {
    Geom::lineToTransformedRect(left_merge_segment, left_line);
  }
  if (right_is_manhattan) {
    Geom::lineToTransformedRect(right_merge_segment, right_line);
  }
  const auto left_closest_point = BoundSkewTreeImpl::linePoint(closest_pair, kLeft);
  const auto right_closest_point = BoundSkewTreeImpl::linePoint(closest_pair, kRight);
  if (!left_is_manhattan && right_is_manhattan) {
    left_merge_segment.makeDiamond(left_closest_point, 0);
  }
  if (left_is_manhattan && !right_is_manhattan) {
    right_merge_segment.makeDiamond(right_closest_point, 0);
  }
  _impl.topDownEmbedding().setJoiningSegmentLine(kLeft, {left_closest_point, left_closest_point});
  _impl.topDownEmbedding().setJoiningSegmentLine(kRight, {right_closest_point, right_closest_point});
  if (left_is_manhattan || right_is_manhattan) {
    auto dist = Geom::transformedRectDistance(left_merge_segment, right_merge_segment);
    LOG_FATAL_IF(std::abs(dist - current_area->get_radius()) > kEpsilon) << "merge_segment distance is not equal to radius";
    current_area->set_radius(dist);
    _impl.mergeSegment(kLeft) = left_merge_segment;
    _impl.mergeSegment(kRight) = right_merge_segment;
    TransformedRect left_bound;
    TransformedRect right_bound;
    TransformedRect left_intersect;
    TransformedRect right_intersect;
    Geom::buildTransformedRect(left_merge_segment, dist, left_bound);
    Geom::buildTransformedRect(right_merge_segment, dist, right_bound);
    Geom::makeIntersection(right_bound, left_merge_segment, left_intersect);
    Geom::makeIntersection(left_bound, right_merge_segment, right_intersect);
    Geom::transformedRectToLine(left_intersect, _impl.joiningSegmentPoint(kLeft, kHead), _impl.joiningSegmentPoint(kLeft, kTail));
    Geom::transformedRectToLine(right_intersect, _impl.joiningSegmentPoint(kRight, kHead), _impl.joiningSegmentPoint(kRight, kTail));
  } else if (Geom::isParallel(left_line, right_line)) {
    const auto& left_head = BoundSkewTreeImpl::linePoint(left_line, kHead);
    const auto& left_tail = BoundSkewTreeImpl::linePoint(left_line, kTail);
    const auto& right_head = BoundSkewTreeImpl::linePoint(right_line, kHead);
    const auto& right_tail = BoundSkewTreeImpl::linePoint(right_line, kTail);
    const auto min_x = std::max(std::min(left_head.x, left_tail.x), std::min(right_head.x, right_tail.x));
    const auto max_x = std::min(std::max(left_head.x, left_tail.x), std::max(right_head.x, right_tail.x));
    const auto min_y = std::max(std::min(left_head.y, left_tail.y), std::min(right_head.y, right_tail.y));
    const auto max_y = std::min(std::max(left_head.y, left_tail.y), std::max(right_head.y, right_tail.y));
    if ((left_type == LineType::kVertical || left_type == LineType::kTilt) && max_y >= min_y) {
      Geom::calcCoord(_impl.joiningSegmentPoint(kLeft, kHead), left_line, min_y);
      Geom::calcCoord(_impl.joiningSegmentPoint(kLeft, kTail), left_line, max_y);
      Geom::calcCoord(_impl.joiningSegmentPoint(kRight, kHead), right_line, min_y);
      Geom::calcCoord(_impl.joiningSegmentPoint(kRight, kTail), right_line, max_y);
    } else if ((left_type == LineType::kHorizontal || left_type == LineType::kFlat) && max_x >= min_x) {
      Geom::calcCoord(_impl.joiningSegmentPoint(kLeft, kHead), left_line, min_x);
      Geom::calcCoord(_impl.joiningSegmentPoint(kLeft, kTail), left_line, max_x);
      Geom::calcCoord(_impl.joiningSegmentPoint(kRight, kHead), right_line, min_x);
      Geom::calcCoord(_impl.joiningSegmentPoint(kRight, kTail), right_line, max_x);
    }
  } else {
    // single point case
  }
  if (Geom::lineType(_impl.topDownEmbedding().getJoiningSegmentLine(kLeft)) == LineType::kManhattan && left_type != LineType::kManhattan
      && right_type != LineType::kManhattan) {
    _impl.mergeSegment(kLeft).makeDiamond(left_closest_point, 0);
    _impl.mergeSegment(kRight).makeDiamond(right_closest_point, 0);
  }
  // _impl.topDownEmbedding().checkUpdatedJoiningSegment(current_area, left_line, right_line);
}

auto BottomUpMergeJoining::addJoiningSegmentPoints(const MergeAreas& merge_areas) -> void
{
  auto* parent = merge_areas.parent;
  auto* left = merge_areas.left;
  auto* right = merge_areas.right;
  // add points on origin joining_segment lines
  FOR_EACH_BST_SIDE(side)
  {
    auto& segment_points = _impl.joiningSegmentPoints(side);
    LOG_FATAL_IF(Geom::isSame(BoundSkewTreeImpl::pointAt(segment_points, kHead), BoundSkewTreeImpl::pointAt(segment_points, kTail)))
        << "join segment is a point";
    auto merge_region = side == kLeft ? left->get_merge_region() : right->get_merge_region();
    for (auto point : merge_region) {
      if (Geom::onLine(point, _impl.topDownEmbedding().getJoiningSegmentLine(side))
          && !Geom::isSame(point, BoundSkewTreeImpl::pointAt(segment_points, kHead))
          && !Geom::isSame(point, BoundSkewTreeImpl::pointAt(segment_points, kTail))) {
        segment_points.push_back(point);
      }
    }
    Geom::sortPointsByFront(segment_points);
  }
  // add points on other side
  auto updated_joining_segments = _impl._joining_segment;
  FOR_EACH_BST_SIDE(side)
  {
    const auto other_side = BoundSkewTreeImpl::otherSide(side);
    const auto other_merge_region = other_side == kLeft ? left->get_merge_region() : right->get_merge_region();
    const auto relative_type = Geom::lineRelative(_impl.topDownEmbedding().getJoiningSegmentLine(kLeft),
                                                  _impl.topDownEmbedding().getJoiningSegmentLine(kRight), other_side);
    const auto& segment_points = _impl.joiningSegmentPoints(side);
    auto& updated_segment_points = updated_joining_segments.forSide(side);
    for (auto point : other_merge_region) {
      Geom::calcRelativeCoord(point, relative_type, parent->get_radius());
      for (size_t point_index = 0; point_index + 1 < segment_points.size(); ++point_index) {
        Line line = {BoundSkewTreeImpl::pointAt(segment_points, point_index), BoundSkewTreeImpl::pointAt(segment_points, point_index + 1)};
        if (Geom::onLine(point, line) && !Geom::isSame(point, BoundSkewTreeImpl::pointAt(segment_points, point_index))
            && !Geom::isSame(point, BoundSkewTreeImpl::pointAt(segment_points, point_index + 1))) {
          _impl.topDownEmbedding().calcSegmentPointDelays(point, line);
          updated_segment_points.push_back(point);
          break;
        }
      }
    }
    Geom::sortPointsByFront(updated_segment_points);
  }
  FOR_EACH_BST_SIDE(side)
  {
    _impl.joiningSegmentPoints(side) = updated_joining_segments.forSide(side);
  }
}

auto BottomUpMergeJoining::delayFromJoiningSegment(const JoiningSegmentDelayQuery& query, const SideDelay& delay_from) const -> double
{
  const auto& segment_point = _impl.joiningSegmentPoint(query.segment_side, query.point_index);
  double delay = query.timing_type == kMin ? segment_point.min : segment_point.max;
  delay += query.joining_region_side == query.segment_side ? 0.0 : delay_from.get(query.segment_side);
  return delay;
}

auto BottomUpMergeJoining::calcJoiningRegion(const MergeAreas& merge_areas) -> void
{
  auto* parent = merge_areas.parent;
  if (TopDownEmbedding::calcAreaLineType(*parent) == LineType::kManhattan) {
    calcJoiningRegionEndpoints(*parent);
  } else {
    calcNonManhattanJoiningRegionEndpoints(merge_areas);
  }
  addFeasibleMergeSegmentToJoiningRegion();
}

auto BottomUpMergeJoining::calcJoiningRegionEndpoints(const Area& current_area) -> void
{
  const auto left_line = current_area.get_line(kLeft);
  const auto right_line = current_area.get_line(kRight);
  checkMatchingEndpoint(_impl.joiningSegmentPoint(kLeft, kHead), BoundSkewTreeImpl::linePoint(left_line, kHead), "left", "head");
  checkMatchingEndpoint(_impl.joiningSegmentPoint(kLeft, kTail), BoundSkewTreeImpl::linePoint(left_line, kTail), "left", "tail");
  checkMatchingEndpoint(_impl.joiningSegmentPoint(kRight, kHead), BoundSkewTreeImpl::linePoint(right_line, kHead), "right", "head");
  checkMatchingEndpoint(_impl.joiningSegmentPoint(kRight, kTail), BoundSkewTreeImpl::linePoint(right_line, kTail), "right", "tail");

  _impl.joiningRegionPoint(kLeft, kHead) = _impl.joiningSegmentPoint(kLeft, kHead);
  _impl.joiningRegionPoint(kLeft, kTail) = _impl.joiningSegmentPoint(kLeft, kTail);
  _impl.joiningRegionPoint(kRight, kHead) = _impl.joiningSegmentPoint(kRight, kHead);
  _impl.joiningRegionPoint(kRight, kTail) = _impl.joiningSegmentPoint(kRight, kTail);
  _impl.topDownEmbedding().updatePointDelaysByEndSide(current_area, kHead, _impl.joiningRegionPoint(kLeft, kHead));
  _impl.topDownEmbedding().updatePointDelaysByEndSide(current_area, kHead, _impl.joiningRegionPoint(kRight, kHead));
  _impl.topDownEmbedding().updatePointDelaysByEndSide(current_area, kTail, _impl.joiningRegionPoint(kLeft, kTail));
  _impl.topDownEmbedding().updatePointDelaysByEndSide(current_area, kTail, _impl.joiningRegionPoint(kRight, kTail));
}

auto BottomUpMergeJoining::calcNonManhattanJoiningRegionEndpoints(const MergeAreas& merge_areas) -> void
{
  auto* left = merge_areas.left;
  auto* right = merge_areas.right;
  addJoiningSegmentPoints(merge_areas);
  const SideDelay delay_from{
      .left = _impl.topDownEmbedding().pointDelayIncrease(_impl.joiningSegmentPoint(kLeft, kHead), _impl.joiningSegmentPoint(kRight, kHead),
                                                          left->get_cap_load(), _impl._rc_pattern),
      .right = _impl.topDownEmbedding().pointDelayIncrease(
          _impl.joiningSegmentPoint(kLeft, kHead), _impl.joiningSegmentPoint(kRight, kHead), right->get_cap_load(), _impl._rc_pattern)};
  FOR_EACH_BST_SIDE(side)
  {
    const auto other_side = BoundSkewTreeImpl::otherSide(side);
    auto& joining_region_points = _impl.joiningRegionPoints(side);
    const auto& segment_points = _impl.joiningSegmentPoints(side);
    const auto& other_segment_points = _impl.joiningSegmentPoints(other_side);
    joining_region_points = segment_points;
    for (size_t point_index = 0; point_index < segment_points.size(); ++point_index) {
      auto point = BoundSkewTreeImpl::pointAt(segment_points, point_index);
      const auto& other_point = BoundSkewTreeImpl::pointAt(other_segment_points, point_index);
      point.min = std::min(point.min, other_point.min + delay_from.get(other_side));
      point.max = std::max(point.max, other_point.max + delay_from.get(other_side));
      BoundSkewTreeImpl::pointAt(joining_region_points, point_index) = point;
    }
    Geom::uniquePointLocations(joining_region_points);
  }
  FOR_EACH_BST_SIDE(side)
  {
    const auto other_side = BoundSkewTreeImpl::otherSide(side);
    const auto section_count = _impl.joiningRegionPoints(side).size() - 1;
    for (size_t point_index = 0; point_index < section_count; ++point_index) {
      const auto min_delta = (_impl.joiningSegmentPoint(side, point_index).min - _impl.joiningSegmentPoint(other_side, point_index).min
                              - delay_from.get(other_side))
                             * (_impl.joiningSegmentPoint(side, point_index + 1).min
                                - _impl.joiningSegmentPoint(other_side, point_index + 1).min - delay_from.get(other_side));
      if (min_delta < -kEpsilon) {
        addTurnPoint(side, point_index, kMin, delay_from);
      }
      const auto max_delta = (_impl.joiningSegmentPoint(side, point_index).max - _impl.joiningSegmentPoint(other_side, point_index).max
                              - delay_from.get(other_side))
                             * (_impl.joiningSegmentPoint(side, point_index + 1).max
                                - _impl.joiningSegmentPoint(other_side, point_index + 1).max - delay_from.get(other_side));
      if (max_delta < -kEpsilon) {
        addTurnPoint(side, point_index, kMax, delay_from);
      }
    }
    auto& joining_region_points = _impl.joiningRegionPoints(side);
    Geom::sortPointsByFront(joining_region_points);
    Geom::uniquePointLocations(joining_region_points);
  }
  FOR_EACH_BST_SIDE(side)
  {
    auto& joining_region_points = _impl.joiningRegionPoints(side);
    for (size_t point_index = 0; point_index + 1 < joining_region_points.size(); ++point_index) {
      const auto first_point = BoundSkewTreeImpl::pointAt(joining_region_points, point_index);
      const auto second_point = BoundSkewTreeImpl::pointAt(joining_region_points, point_index + 1);
      const auto distance = Geom::distance(first_point, second_point);
      LOG_FATAL_IF(Equal(distance, 0)) << "distance is zero";
      BoundSkewTreeImpl::pointAt(joining_region_points, point_index).val
          = (TopDownEmbedding::pointSkew(second_point) - TopDownEmbedding::pointSkew(first_point)) / distance;
    }
    Points increasing_points = {joining_region_points.front()};
    for (size_t point_index = 1; point_index + 1 < joining_region_points.size(); ++point_index) {
      const auto current_value = increasing_points.back().val;
      const auto next_value = BoundSkewTreeImpl::pointAt(joining_region_points, point_index).val;
      LOG_FATAL_IF(current_value > next_value + (100 * kEpsilon))
          << "current_value: " << current_value << "> next_value: " << next_value << ", skew slope is not strictly monotone increasing";
      if (next_value > current_value) {
        increasing_points.push_back(BoundSkewTreeImpl::pointAt(joining_region_points, point_index));
      }
    }
    increasing_points.push_back(joining_region_points.back());
    joining_region_points = increasing_points;
  }
}

auto BottomUpMergeJoining::addTurnPoint(const size_t& side, const size_t& point_index, const size_t& timing_type,
                                        const SideDelay& delay_from) -> void
{
  const auto first_point = _impl.joiningRegionPoint(side, point_index);
  const auto second_point = _impl.joiningRegionPoint(side, point_index + 1);
  const double alpha
      = Equal(first_point.x, second_point.x) ? _impl._delay_quadratic_factor.vertical : _impl._delay_quadratic_factor.horizontal;
  const auto distance = Geom::distance(first_point, second_point);
  LOG_FATAL_IF(Equal(distance, 0)) << "distance is zero";

  SideState<TimingState<double>> beta;
  FOR_EACH_BST_SIDE(segment_side)
  {
    FOR_EACH_BST_SIDE(current_timing_type)
    {
      const auto first_delay = delayFromJoiningSegment(
          JoiningSegmentDelayQuery{
              .joining_region_side = side, .segment_side = segment_side, .point_index = point_index, .timing_type = current_timing_type},
          delay_from);
      const auto second_delay = delayFromJoiningSegment(JoiningSegmentDelayQuery{.joining_region_side = side,
                                                                                 .segment_side = segment_side,
                                                                                 .point_index = point_index + 1,
                                                                                 .timing_type = current_timing_type},
                                                        delay_from);
      beta.forSide(segment_side).forTiming(current_timing_type) = ((second_delay - first_delay) / distance) - (alpha * distance);
    }
  }

  const auto left_delay = delayFromJoiningSegment(
      JoiningSegmentDelayQuery{.joining_region_side = kLeft, .segment_side = side, .point_index = point_index, .timing_type = timing_type},
      delay_from);
  const auto right_delay = delayFromJoiningSegment(
      JoiningSegmentDelayQuery{.joining_region_side = kRight, .segment_side = side, .point_index = point_index, .timing_type = timing_type},
      delay_from);
  const auto beta_delta = beta.right.forTiming(timing_type) - beta.left.forTiming(timing_type);
  const auto turn_distance = (left_delay - right_delay) / beta_delta;
  LOG_FATAL_IF(turn_distance <= 0 || turn_distance >= distance) << "turn dist is not in range";

  const auto reference_distance = distance - turn_distance;
  Point turn_point(((first_point.x * reference_distance) + (second_point.x * turn_distance)) / distance,
                   ((first_point.y * reference_distance) + (second_point.y * turn_distance)) / distance);
  SideState<TimingState<double>> delay_bound;
  FOR_EACH_BST_SIDE(segment_side)
  {
    FOR_EACH_BST_SIDE(current_timing_type)
    {
      delay_bound.forSide(segment_side).forTiming(current_timing_type)
          = delayFromJoiningSegment(JoiningSegmentDelayQuery{.joining_region_side = side,
                                                             .segment_side = segment_side,
                                                             .point_index = point_index,
                                                             .timing_type = current_timing_type},
                                    delay_from)
            + (alpha * turn_distance * turn_distance) + (beta.forSide(segment_side).forTiming(current_timing_type) * turn_distance);
    }
  }
  turn_point.min = std::min(delay_bound.left.min, delay_bound.right.min);
  turn_point.max = std::max(delay_bound.left.max, delay_bound.right.max);
  _impl.joiningRegionPoints(side).push_back(turn_point);
}

auto BottomUpMergeJoining::addFeasibleMergeSegmentToJoiningRegion() -> void
{
  FOR_EACH_BST_SIDE(side)
  {
    auto& joining_region_points = _impl.joiningRegionPoints(side);
    for (size_t point_index = 0; point_index + 1 < joining_region_points.size(); ++point_index) {
      const auto current_point = BoundSkewTreeImpl::pointAt(joining_region_points, point_index);
      const auto next_point = BoundSkewTreeImpl::pointAt(joining_region_points, point_index + 1);
      const auto current_delta = TopDownEmbedding::pointSkew(current_point) - _impl._skew_bound;
      const auto next_delta = TopDownEmbedding::pointSkew(next_point) - _impl._skew_bound;
      if (current_delta * next_delta < 0 && !Equal(current_delta, 0) && !Equal(next_delta, 0)) {
        const auto distance = Geom::distance(current_point, next_point);
        const auto turn_distance = ((_impl._skew_bound - TopDownEmbedding::pointSkew(current_point)) * distance)
                                   / (TopDownEmbedding::pointSkew(next_point) - TopDownEmbedding::pointSkew(current_point));
        const auto reference_distance = distance - turn_distance;
        Point turn_point{((current_point.x * reference_distance) + (next_point.x * turn_distance)) / distance,
                         ((current_point.y * reference_distance) + (next_point.y * turn_distance)) / distance};
        Line line = {current_point, next_point};
        _impl.topDownEmbedding().calcSegmentPointDelays(turn_point, line);
        const auto insert_offset = static_cast<Points::difference_type>(point_index + 1);
        joining_region_points.insert(std::next(joining_region_points.begin(), insert_offset), turn_point);
      }
    }
  }
}

auto BottomUpMergeJoining::calcJoiningRegionCorner(const Area& current_area) -> void
{
  FOR_EACH_BST_SIDE(side)
  {
    const auto& segment_points = _impl.joiningSegmentPoints(side);
    LOG_FATAL_IF(segment_points.front().y + kEpsilon < segment_points.back().y) << "join segment direction is not correct";
  }
  if (TopDownEmbedding::calcAreaLineType(current_area) == LineType::kManhattan && !Equal(current_area.get_radius(), 0)) {
    FOR_EACH_BST_SIDE(end_side)
    {
      if (joiningRegionCornerExists(end_side)) {
        const auto left_point = BoundSkewTreeImpl::pointAt(_impl.joiningSegmentPoints(kLeft), end_side);
        const auto right_point = BoundSkewTreeImpl::pointAt(_impl.joiningSegmentPoints(kRight), end_side);
        auto& joining_corner_point = _impl.joiningCornerPoint(end_side);
        if ((left_point.x - right_point.x) * (left_point.y - right_point.y) < 0) {
          if (end_side == kHead) {
            joining_corner_point = {std::max(left_point.x, right_point.x), std::max(left_point.y, right_point.y)};
          } else {
            joining_corner_point = {std::min(left_point.x, right_point.x), std::min(left_point.y, right_point.y)};
          }
        } else {
          if (end_side == kHead) {
            joining_corner_point = {std::min(left_point.x, right_point.x), std::max(left_point.y, right_point.y)};
          } else {
            joining_corner_point = {std::max(left_point.x, right_point.x), std::min(left_point.y, right_point.y)};
          }
        }
        _impl.topDownEmbedding().updatePointDelaysByEndSide(current_area, end_side, joining_corner_point);
      }
    }
  }
}

auto BottomUpMergeJoining::joiningRegionCornerExists(const size_t& end_side) const -> bool
{
  const auto first_point = BoundSkewTreeImpl::pointAt(_impl.joiningSegmentPoints(kLeft), end_side);
  const auto second_point = BoundSkewTreeImpl::pointAt(_impl.joiningSegmentPoints(kRight), end_side);
  return !Equal(first_point.x, second_point.x) && !Equal(first_point.y, second_point.y);
}

}  // namespace icts::bst::detail
