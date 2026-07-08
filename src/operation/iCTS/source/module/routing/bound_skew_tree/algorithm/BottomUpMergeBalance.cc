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
 * @file BottomUpMergeBalance.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief Bound-skew tree balance-point and merge-region construction implementation (stage B.2).
 */

#include "bound_skew_tree/algorithm/BottomUpMergeBalance.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

#include "Log.hh"
#include "bound_skew_tree/algorithm/BottomUpMergeJoining.hh"
#include "bound_skew_tree/algorithm/BoundSkewTreeImpl.hh"
#include "bound_skew_tree/algorithm/TopDownEmbedding.hh"
#include "bound_skew_tree/component/Components.hh"
#include "bound_skew_tree/config/BSTRoutingConfig.hh"
#include "bound_skew_tree/geometry/GeomCalc.hh"

namespace icts::bst::detail {

auto BottomUpMergeBalance::calcBalancePoint(const Area& current_area) -> void
{
  auto* left_child = current_area.get_left();
  auto* right_child = current_area.get_right();
  if (left_child == nullptr || right_child == nullptr) {
    LOG_FATAL << "calcBalancePoint requires both child areas";
    return;
  }

  FOR_EACH_BST_SIDE(end_side)
  {
    _impl.balancePoints(end_side).clear();
  }
  if (Equal(current_area.get_radius(), 0)) {
    return;
  }
  FOR_EACH_BST_SIDE(end_side)
  {
    const auto left_line = current_area.get_line(kLeft);
    const auto right_line = current_area.get_line(kRight);
    auto left_point = BoundSkewTreeImpl::linePoint(left_line, end_side);
    auto right_point = BoundSkewTreeImpl::linePoint(right_line, end_side);
    left_point.val = left_child->get_cap_load();
    right_point.val = right_child->get_cap_load();
    const auto default_balance_ref_axis = end_side == kHead ? BalanceRefAxis::kX : BalanceRefAxis::kY;
    const auto swapped_balance_ref_axis = end_side == kHead ? BalanceRefAxis::kY : BalanceRefAxis::kX;
    const auto balance_ref_axis
        = (left_point.x - right_point.x) * (left_point.y - right_point.y) < 0 ? swapped_balance_ref_axis : default_balance_ref_axis;
    FOR_EACH_BST_SIDE(timing_type)
    {
      const BalancePointQuery query{.first_point = left_point,
                                    .second_point = right_point,
                                    .timing_type = timing_type,
                                    .balance_ref_axis = balance_ref_axis,
                                    .rc_pattern = _impl._rc_pattern};
      BalancePointResult result;
      calcBalanceBetweenPoints(query, result);
      if (!Equal(result.distance_to_first, 0) && !Equal(result.distance_to_second, 0)) {
        _impl.topDownEmbedding().updatePointDelaysByEndSide(current_area, end_side, result.balance_point);
        _impl.balancePoints(end_side).push_back(result.balance_point);
      }
    }
  }
}

auto BottomUpMergeBalance::calcBalanceBetweenPoints(const BalancePointQuery& query, BalancePointResult& result) const -> void
{
  auto horizontal_distance = std::abs(query.first_point.x - query.second_point.x);
  auto vertical_distance = std::abs(query.first_point.y - query.second_point.y);
  if (Equal(horizontal_distance, 0) || Equal(vertical_distance, 0)) {
    calcBalancePointOnLine(query, result);
  } else if (query.first_point.x <= query.second_point.x) {
    calcBalancePointOffLine(query, result);
  } else {
    BalancePointQuery swapped_query = query;
    swapped_query.first_point = query.second_point;
    swapped_query.second_point = query.first_point;
    calcBalancePointOffLine(swapped_query, result);
    std::swap(result.distance_to_first, result.distance_to_second);
  }
}

auto BottomUpMergeBalance::calcBalancePointOnLine(const BalancePointQuery& query, BalancePointResult& result) const -> void
{
  auto horizontal_distance = std::abs(query.first_point.x - query.second_point.x);
  auto vertical_distance = std::abs(query.first_point.y - query.second_point.y);
  LOG_FATAL_IF(!Equal(horizontal_distance, 0) && !Equal(vertical_distance, 0))
      << "h and v are not zero, which balance point is not on line";

  auto first_delay = query.timing_type == kMin ? query.first_point.min : query.first_point.max;
  auto second_delay = query.timing_type == kMin ? query.second_point.min : query.second_point.max;
  auto unit_resistance = Equal(horizontal_distance, 0) ? _impl._unit_vertical_resistance : _impl._unit_horizontal_resistance;
  auto unit_capacitance = Equal(horizontal_distance, 0) ? _impl._unit_vertical_capacitance : _impl._unit_horizontal_capacitance;
  const auto merge_distances = calcMergeDist(unit_resistance, unit_capacitance, query.first_point.val, first_delay, query.second_point.val,
                                             second_delay, horizontal_distance + vertical_distance);
  result.distance_to_first = merge_distances.distance_to_first;
  result.distance_to_second = merge_distances.distance_to_second;
  calcPointCoordOnLine(query.first_point, query.second_point, result.distance_to_first, result.distance_to_second, result.balance_point);
  double first_delay_increase = 0.0;
  double second_delay_increase = 0.0;
  if (Equal(horizontal_distance, 0)) {
    first_delay_increase = _impl.topDownEmbedding().calcDelayIncrease(0, result.distance_to_first, query.first_point.val, query.rc_pattern);
    second_delay_increase
        = _impl.topDownEmbedding().calcDelayIncrease(0, result.distance_to_second, query.second_point.val, query.rc_pattern);
  } else {
    first_delay_increase = _impl.topDownEmbedding().calcDelayIncrease(result.distance_to_first, 0, query.first_point.val, query.rc_pattern);
    second_delay_increase
        = _impl.topDownEmbedding().calcDelayIncrease(result.distance_to_second, 0, query.second_point.val, query.rc_pattern);
  }
  result.balance_point.min = std::min(query.first_point.min + first_delay_increase, query.second_point.min + second_delay_increase);
  result.balance_point.max = std::max(query.first_point.max + first_delay_increase, query.second_point.max + second_delay_increase);
}

auto BottomUpMergeBalance::calcBalancePointOffLine(const BalancePointQuery& query, BalancePointResult& result) const -> void
{
  auto first_point = query.first_point;
  auto second_point = query.second_point;
  auto horizontal_distance = std::abs(first_point.x - second_point.x);
  auto vertical_distance = std::abs(first_point.y - second_point.y);
  LOG_FATAL_IF(Equal(horizontal_distance, 0) || Equal(vertical_distance, 0)) << "h or v is zero, which balance point is on line";
  LOG_FATAL_IF(first_point.x > second_point.x) << "first_point is not left of second_point";

  auto first_delay = query.timing_type == kMin ? first_point.min : first_point.max;
  auto second_delay = query.timing_type == kMin ? second_point.min : second_point.max;
  auto x_position = calcXBalancePosition(first_delay, second_delay, first_point.val, second_point.val, horizontal_distance,
                                         vertical_distance, query.balance_ref_axis);
  double y_position = 0.0;
  if (x_position < 0) {
    y_position = query.balance_ref_axis == BalanceRefAxis::kX
                     ? calcYBalancePosition(first_delay, second_delay, first_point.val, second_point.val, horizontal_distance,
                                            vertical_distance, query.balance_ref_axis)
                     : -1;
    x_position = y_position >= 0 ? 0 : x_position;
  } else if (x_position > horizontal_distance) {
    y_position = query.balance_ref_axis == BalanceRefAxis::kX
                     ? vertical_distance + 1
                     : calcYBalancePosition(first_delay, second_delay, first_point.val, second_point.val, horizontal_distance,
                                            vertical_distance, query.balance_ref_axis);
    x_position = y_position <= vertical_distance ? horizontal_distance : x_position;
  } else {
    y_position = query.balance_ref_axis == BalanceRefAxis::kX ? vertical_distance : 0;
  }

  if (x_position < 0) {
    LOG_FATAL_IF(y_position >= 0) << "y is illegal";
    auto adjusted_point = first_point;
    auto delay_increase
        = _impl.topDownEmbedding().calcDelayIncrease(horizontal_distance, vertical_distance, second_point.val, query.rc_pattern);
    adjusted_point.min = second_point.min + delay_increase;
    adjusted_point.max = second_point.max + delay_increase;
    adjusted_point.val = second_point.val + (_impl._unit_horizontal_capacitance * horizontal_distance)
                         + (_impl._unit_vertical_capacitance * vertical_distance);
    calcBalancePointOnLine(BalancePointQuery{.first_point = first_point,
                                             .second_point = adjusted_point,
                                             .timing_type = query.timing_type,
                                             .balance_ref_axis = query.balance_ref_axis,
                                             .rc_pattern = query.rc_pattern},
                           result);
    LOG_FATAL_IF(result.distance_to_first > kEpsilon) << "dist to first_point should be zero";
    auto new_delay_increase
        = _impl.topDownEmbedding().calcDelayIncrease(0, result.distance_to_second, adjusted_point.val, query.rc_pattern);
    LOG_FATAL_IF(!Equal(first_delay, delay_increase + new_delay_increase + second_delay)) << "delay is not equal";
    result.distance_to_second += horizontal_distance + vertical_distance;
  } else if (x_position > horizontal_distance) {
    LOG_FATAL_IF(y_position <= vertical_distance) << "y: " << y_position << " is not greater than v: " << vertical_distance;
    auto adjusted_point = second_point;
    auto delay_increase
        = _impl.topDownEmbedding().calcDelayIncrease(horizontal_distance, vertical_distance, first_point.val, query.rc_pattern);
    adjusted_point.min = first_point.min + delay_increase;
    adjusted_point.max = first_point.max + delay_increase;
    adjusted_point.val = first_point.val + (_impl._unit_horizontal_capacitance * horizontal_distance)
                         + (_impl._unit_vertical_capacitance * vertical_distance);
    calcBalancePointOnLine(BalancePointQuery{.first_point = adjusted_point,
                                             .second_point = second_point,
                                             .timing_type = query.timing_type,
                                             .balance_ref_axis = query.balance_ref_axis,
                                             .rc_pattern = query.rc_pattern},
                           result);
    LOG_FATAL_IF(result.distance_to_second > kEpsilon) << "dist to second_point should be zero";
    auto new_delay_increase = _impl.topDownEmbedding().calcDelayIncrease(0, result.distance_to_first, adjusted_point.val, query.rc_pattern);
    LOG_FATAL_IF(!Equal(second_delay, delay_increase + new_delay_increase + first_delay)) << "delay is not equal";
    result.distance_to_first += horizontal_distance + vertical_distance;
  } else {
    LOG_FATAL_IF(y_position < -kEpsilon || y_position > vertical_distance + kEpsilon)
        << "y: " << y_position << " is not in range [0, " << vertical_distance << "]";
    result.balance_point.x = first_point.x + x_position;
    result.balance_point.y = first_point.y < second_point.y ? first_point.y + y_position : first_point.y - y_position;
    auto first_delay_increase = _impl.topDownEmbedding().calcDelayIncrease(x_position, y_position, first_point.val, query.rc_pattern);
    auto second_delay_increase = _impl.topDownEmbedding().calcDelayIncrease(
        horizontal_distance - x_position, vertical_distance - y_position, second_point.val, query.rc_pattern);
    result.balance_point.min = std::min(first_point.min + first_delay_increase, second_point.min + second_delay_increase);
    result.balance_point.max = std::max(first_point.max + first_delay_increase, second_point.max + second_delay_increase);
    result.distance_to_first = x_position + y_position;
    result.distance_to_second = horizontal_distance + vertical_distance - result.distance_to_first;
    LOG_FATAL_IF(!Equal(first_delay_increase + first_delay, second_delay_increase + second_delay)) << "delay is not equal";
  }
  LOG_FATAL_IF(result.distance_to_first + result.distance_to_second < horizontal_distance + vertical_distance - kEpsilon)
      << "dist out of range";
}

auto BottomUpMergeBalance::calcMergeDist(const double& unit_resistance, const double& unit_capacitance, const double& cap_load_1,
                                         const double& delay_1, const double& cap_load_2, const double& delay_2,
                                         const double& total_distance) -> MergeDistances
{
  MergeDistances merge_distances;
  auto distance_to_merge = (delay_2 - delay_1 + (unit_resistance * total_distance * (cap_load_2 + (unit_capacitance * total_distance / 2))))
                           / (unit_resistance * (cap_load_1 + cap_load_2 + (unit_capacitance * total_distance)));
  if (distance_to_merge < 0) {
    auto capacitance_ratio = cap_load_2 / unit_capacitance;
    distance_to_merge
        = std::sqrt((capacitance_ratio * capacitance_ratio) + (2 * (delay_1 - delay_2) / (unit_resistance * unit_capacitance)))
          - capacitance_ratio;
    merge_distances.distance_to_first = 0;
    merge_distances.distance_to_second = distance_to_merge;
  } else if (distance_to_merge > total_distance) {
    auto capacitance_ratio = cap_load_1 / unit_capacitance;
    distance_to_merge
        = std::sqrt((capacitance_ratio * capacitance_ratio) + (2 * (delay_2 - delay_1) / (unit_resistance * unit_capacitance)))
          - capacitance_ratio;
    merge_distances.distance_to_first = distance_to_merge;
    merge_distances.distance_to_second = 0;
  } else {
    merge_distances.distance_to_first = distance_to_merge;
    merge_distances.distance_to_second = total_distance - distance_to_merge;
  }
  return merge_distances;
}

auto BottomUpMergeBalance::calcPointCoordOnLine(const Point& first_point, const Point& second_point, const double& distance_to_first,
                                                const double& distance_to_second, Point& point) -> void
{
  auto total_distance = distance_to_first + distance_to_second;
  auto point_distance = Geom::distance(first_point, second_point);
  LOG_FATAL_IF(!Equal(total_distance, point_distance) && total_distance < point_distance) << "distance is less than points distance";
  if (Equal(distance_to_first, 0)) {
    point = first_point;
  } else if (Equal(distance_to_second, 0)) {
    point = second_point;
  } else {
    point = {((first_point.x * distance_to_second) + (second_point.x * distance_to_first)) / total_distance,
             ((first_point.y * distance_to_second) + (second_point.y * distance_to_first)) / total_distance};
  }
}

auto BottomUpMergeBalance::calcXBalancePosition(const double& delay_1, const double& delay_2, const double& cap_load_1,
                                                const double& cap_load_2, const double& horizontal_distance,
                                                const double& vertical_distance, BalanceRefAxis balance_ref_axis) const -> double
{
  auto resistance_capacitance_cross_term = _impl._rc_pattern == BSTRoutingRCPattern::kHV
                                               ? _impl._unit_vertical_resistance * _impl._unit_horizontal_capacitance
                                               : _impl._unit_horizontal_resistance * _impl._unit_vertical_capacitance;
  double numerator = 0;
  if (balance_ref_axis == BalanceRefAxis::kX) {
    // assume (x, vertical_distance-y) and (horizontal_distance-x, y), then set y = 0
    numerator = delay_2 - delay_1 + (_impl._delay_quadratic_factor.horizontal * horizontal_distance * horizontal_distance)
                - (_impl._delay_quadratic_factor.vertical * vertical_distance * vertical_distance)
                + (_impl._unit_horizontal_resistance * horizontal_distance * cap_load_2)
                - (_impl._unit_vertical_resistance * vertical_distance * cap_load_1);
  } else {
    // assume (x, y) and (horizontal_distance-x, vertical_distance-y), then set y = 0
    numerator = delay_2 - delay_1 + (_impl._delay_quadratic_factor.horizontal * horizontal_distance * horizontal_distance)
                + (_impl._delay_quadratic_factor.vertical * vertical_distance * vertical_distance)
                + (cap_load_2
                   * ((_impl._unit_horizontal_resistance * horizontal_distance) + (_impl._unit_vertical_resistance * vertical_distance)))
                + (resistance_capacitance_cross_term * horizontal_distance * vertical_distance);
  }
  auto denominator = (_impl._unit_horizontal_resistance * (cap_load_1 + cap_load_2))
                     + (resistance_capacitance_cross_term * vertical_distance)
                     + (2 * horizontal_distance * _impl._delay_quadratic_factor.horizontal);
  return numerator / denominator;
}

auto BottomUpMergeBalance::calcYBalancePosition(const double& delay_1, const double& delay_2, const double& cap_load_1,
                                                const double& cap_load_2, const double& horizontal_distance,
                                                const double& vertical_distance, BalanceRefAxis balance_ref_axis) const -> double
{
  auto resistance_capacitance_cross_term = _impl._rc_pattern == BSTRoutingRCPattern::kHV
                                               ? _impl._unit_vertical_resistance * _impl._unit_horizontal_capacitance
                                               : _impl._unit_horizontal_resistance * _impl._unit_vertical_capacitance;
  double numerator = 0;
  auto denominator = (_impl._unit_vertical_resistance * (cap_load_1 + cap_load_2))
                     + (2 * vertical_distance * _impl._delay_quadratic_factor.vertical)
                     + (resistance_capacitance_cross_term * horizontal_distance);
  double y_position = 0;
  if (balance_ref_axis == BalanceRefAxis::kX) {
    // assume (x, y) and (horizontal_distance-x, vertical_distance-y), then set x = 0
    numerator = delay_2 - delay_1 + (_impl._delay_quadratic_factor.horizontal * horizontal_distance * horizontal_distance)
                + (_impl._delay_quadratic_factor.vertical * vertical_distance * vertical_distance)
                + (cap_load_2
                   * ((_impl._unit_horizontal_resistance * horizontal_distance) + (_impl._unit_vertical_resistance * vertical_distance)))
                + (resistance_capacitance_cross_term * horizontal_distance * vertical_distance);
    y_position = numerator / denominator;
    LOG_FATAL_IF(y_position > vertical_distance + kEpsilon)
        << "y: " << y_position << " is larger than vertical_distance: " << vertical_distance;
  } else {
    // assume (horizontal_distance-x, y) and (x, vertical_distance-y), then set x = 0
    numerator = delay_2 - delay_1 + (_impl._delay_quadratic_factor.vertical * vertical_distance * vertical_distance)
                - (_impl._delay_quadratic_factor.horizontal * horizontal_distance * horizontal_distance)
                + (_impl._unit_vertical_resistance * vertical_distance * cap_load_2)
                - (_impl._unit_horizontal_resistance * horizontal_distance * cap_load_1);
    y_position = numerator / denominator;
    LOG_FATAL_IF(y_position < -kEpsilon) << "y: " << y_position << " is less than 0";
  }
  return y_position;
}

auto BottomUpMergeBalance::calcFeasibleMergeSegmentPoints(const Area& current_area) -> void
{
  FOR_EACH_BST_SIDE(end_side)
  {
    _impl.feasibleMergeSegmentPoints(end_side).clear();
    SideState<Point> candidate;
    if (end_side == kHead) {
      candidate.left = _impl.joiningRegionPoints(kLeft).front();
      candidate.right = _impl.joiningRegionPoints(kRight).front();
    } else {
      candidate.left = _impl.joiningRegionPoints(kLeft).back();
      candidate.right = _impl.joiningRegionPoints(kRight).back();
    }
    bool exist = false;
    if (_impl.bottomUpMergeJoining().joiningRegionCornerExists(end_side)) {
      const auto& joining_corner = _impl.joiningCornerPoint(end_side);
      exist = calcFeasibleMergeSegmentOnLine(current_area, candidate.left, joining_corner, end_side);
      if (exist) {
        exist = calcFeasibleMergeSegmentOnLine(current_area, candidate.right, joining_corner, end_side);
        if (!exist) {
          exist = calcFeasibleMergeSegmentOnLine(current_area, _impl.joiningCornerPoint(end_side), candidate.left, end_side);
          LOG_FATAL_IF(!exist) << "can't find feasible merge section on line";
        }
      } else {
        exist = calcFeasibleMergeSegmentOnLine(current_area, _impl.joiningCornerPoint(end_side), candidate.right, end_side);
        if (exist) {
          exist = calcFeasibleMergeSegmentOnLine(current_area, candidate.right, joining_corner, end_side);
          LOG_FATAL_IF(!exist) << "can't find feasible merge section on line";
        }
      }
    } else {
      exist = calcFeasibleMergeSegmentOnLine(current_area, candidate.left, candidate.right, end_side);
      if (exist) {
        calcFeasibleMergeSegmentOnLine(current_area, candidate.right, candidate.left, end_side);
      }
    }
    Geom::uniquePointLocations(_impl.feasibleMergeSegmentPoints(end_side));
  }
}

auto BottomUpMergeBalance::calcFeasibleMergeSegmentOnLine(const Area& current_area, Point& point, const Point& reference_point,
                                                          const size_t& end_side) -> bool
{
  auto skew = TopDownEmbedding::pointSkew(point);
  if (Equal(skew, _impl._skew_bound) || skew < _impl._skew_bound) {
    _impl.feasibleMergeSegmentPoints(end_side).push_back(point);
    return true;
  }
  auto nearest_balance_point = reference_point;
  std::ranges::for_each(_impl.balancePoints(end_side), [&nearest_balance_point, &point](const Point& balance_point) -> void {
    if (Geom::distance(point, balance_point) < Geom::distance(point, nearest_balance_point)) {
      nearest_balance_point = balance_point;
    }
  });
  skew = TopDownEmbedding::pointSkew(nearest_balance_point);
  if (Equal(skew, _impl._skew_bound)) {
    _impl.feasibleMergeSegmentPoints(end_side).push_back(nearest_balance_point);
    return true;
  }
  if (skew < _impl._skew_bound) {
    Point feasible_merge_point;
    calcFeasibleMergeSegmentBetweenPoints(point, nearest_balance_point, feasible_merge_point);
    _impl.topDownEmbedding().updatePointDelaysByEndSide(current_area, end_side, feasible_merge_point);
    if (!Equal(TopDownEmbedding::pointSkew(feasible_merge_point), _impl._skew_bound)) {
      auto first_point = point;
      auto second_point = nearest_balance_point;
      _impl.topDownEmbedding().updatePointDelaysByEndSide(current_area, end_side, first_point);
      _impl.topDownEmbedding().updatePointDelaysByEndSide(current_area, end_side, second_point);
      LOG_FATAL << "feasible merge section point should in skew bound";
    }
    _impl.feasibleMergeSegmentPoints(end_side).push_back(feasible_merge_point);
    return true;
  }
  return false;
}

auto BottomUpMergeBalance::calcFeasibleMergeSegmentBetweenPoints(const Point& high_skew_point, const Point& low_skew_point,
                                                                 Point& feasible_merge_point) const -> void
{
  auto high_skew = TopDownEmbedding::pointSkew(high_skew_point);
  auto low_skew = TopDownEmbedding::pointSkew(low_skew_point);
  LOG_FATAL_IF(low_skew > _impl._skew_bound) << "low skew is larger than skew bound";
  LOG_FATAL_IF(high_skew < low_skew + kEpsilon) << "high skew is less than low skew";
  auto dist = Geom::distance(high_skew_point, low_skew_point);
  LOG_FATAL_IF(dist <= kEpsilon) << "distance is less than epsilon";
  auto dist_to_low = dist * (_impl._skew_bound - low_skew) / (high_skew - low_skew);
  calcPointCoordOnLine(high_skew_point, low_skew_point, dist - dist_to_low, dist_to_low, feasible_merge_point);
}

auto BottomUpMergeBalance::hasFeasibleMergeSegmentOnJoiningRegion() const -> bool
{
  if (!_impl.feasibleMergeSegmentPoints(kHead).empty() || !_impl.feasibleMergeSegmentPoints(kTail).empty()) {
    return true;
  }
  FOR_EACH_BST_SIDE(side)
  {
    for (const auto& point : _impl.joiningRegionPoints(side)) {
      if (TopDownEmbedding::pointSkew(point) <= _impl._skew_bound) {
        return true;
      }
    }
  }
  return false;
}

auto BottomUpMergeBalance::constructFeasibleMergeRegion(Area* parent) const -> void
{
  if (TopDownEmbedding::calcAreaLineType(*parent) == LineType::kManhattan) {
    // parallel manhattan arc
    addMergeRegionBetweenJoiningSegments(parent, kHead);
    if (!parent->get_merge_region().empty() && !isJoiningRegionLine()) {
      addMergeRegionBetweenJoiningSegments(parent, kTail);
    }
  } else {
    // parallel horizontal or vertical arc
    addMergeRegionBetweenJoiningSegments(parent, kHead);
    addMergeRegionOnJoiningSegment(parent, kLeft);
    addMergeRegionBetweenJoiningSegments(parent, kTail);
    if (parent->get_radius() > kEpsilon) {
      addMergeRegionOnJoiningSegment(parent, kRight);
    }
  }
}

auto BottomUpMergeBalance::isJoiningRegionLine() const -> bool
{
  const auto& left_segment_points = _impl.joiningSegmentPoints(kLeft);
  const auto& right_segment_points = _impl.joiningSegmentPoints(kRight);
  const auto left_head = BoundSkewTreeImpl::pointAt(left_segment_points, kHead);
  const auto left_tail = BoundSkewTreeImpl::pointAt(left_segment_points, kTail);
  const auto right_head = BoundSkewTreeImpl::pointAt(right_segment_points, kHead);
  const auto right_tail = BoundSkewTreeImpl::pointAt(right_segment_points, kTail);
  const auto min_x = std::min({left_head.x, left_tail.x, right_head.x, right_tail.x});
  const auto min_y = std::min({left_head.y, left_tail.y, right_head.y, right_tail.y});
  const auto max_x = std::max({left_head.x, left_tail.x, right_head.x, right_tail.x});
  const auto max_y = std::max({left_head.y, left_tail.y, right_head.y, right_tail.y});
  return Equal(min_x, max_x) || Equal(min_y, max_y);
}

auto BottomUpMergeBalance::addMergeRegionBetweenJoiningSegments(Area* current_area, const size_t& end_side) const -> void
{
  Points merge_region_points;
  std::ranges::for_each(_impl.balancePoints(end_side),
                        [&merge_region_points](const Point& point) -> void { merge_region_points.push_back(point); });
  std::ranges::for_each(_impl.feasibleMergeSegmentPoints(end_side),
                        [&merge_region_points](const Point& point) -> void { merge_region_points.push_back(point); });
  if (_impl.bottomUpMergeJoining().joiningRegionCornerExists(end_side)
      && TopDownEmbedding::pointSkew(_impl.joiningCornerPoint(end_side)) < _impl._skew_bound + kEpsilon) {
    merge_region_points.push_back(_impl.joiningCornerPoint(end_side));
  }
  if (merge_region_points.empty()) {
    return;
  }
  const auto left_line = current_area->get_line(kLeft);
  const auto right_line = current_area->get_line(kRight);
  Point reference_joining_segment_point
      = end_side == kHead ? BoundSkewTreeImpl::linePoint(left_line, end_side) : BoundSkewTreeImpl::linePoint(right_line, end_side);
  std::ranges::for_each(merge_region_points,
                        [&](Point& point) -> void { point.val = Geom::distance(point, reference_joining_segment_point); });
  Geom::sortPointsByValueDesc(merge_region_points);
  Geom::uniquePointLocations(merge_region_points);
  std::ranges::for_each(merge_region_points, [&](const Point& point) -> void { current_area->add_merge_region_point(point); });
}

auto BottomUpMergeBalance::addMergeRegionOnJoiningSegment(Area* current_area, const size_t& side) const -> void
{
  const auto& side_joining_region = _impl.joiningRegionPoints(side);
  const auto other_side = BoundSkewTreeImpl::otherSide(side);
  const auto& other_joining_region = _impl.joiningRegionPoints(other_side);
  LOG_FATAL_IF(side_joining_region.size() < 2) << "join region size is less than 2";

  const auto first_point = side_joining_region.front();
  const auto other_first_point = other_joining_region.front();
  size_t joining_region_left_index = calcMergeRegionLeftIndex(side);
  if (_impl.feasibleMergeSegmentPoints(kHead).empty()
      && TopDownEmbedding::pointSkew(first_point) < TopDownEmbedding::pointSkew(other_first_point)) {
    joining_region_left_index = side_joining_region.size() - 1;
    for (size_t point_index = 1; point_index + 1 < side_joining_region.size(); ++point_index) {
      if (Equal(TopDownEmbedding::pointSkew(BoundSkewTreeImpl::pointAt(side_joining_region, point_index)), _impl._skew_bound)) {
        joining_region_left_index = point_index;
        break;
      }
    }
  }

  const auto last_point = side_joining_region.back();
  const auto other_last_point = other_joining_region.back();
  auto merge_region_span = calcMergeRegionSpan(side, joining_region_left_index);
  auto& joining_region_right_index = merge_region_span.right_index;
  if (_impl.feasibleMergeSegmentPoints(kTail).empty()
      && TopDownEmbedding::pointSkew(last_point) < TopDownEmbedding::pointSkew(other_last_point)) {
    while (joining_region_right_index >= joining_region_left_index) {
      if (Equal(TopDownEmbedding::pointSkew(BoundSkewTreeImpl::pointAt(side_joining_region, joining_region_right_index)),
                _impl._skew_bound)) {
        break;
      }
      if (joining_region_right_index == joining_region_left_index) {
        break;
      }
      --joining_region_right_index;
    }
  }

  appendMergeRegionPointsOnSegment(current_area, merge_region_span);
}

auto BottomUpMergeBalance::calcMergeRegionLeftIndex(const size_t& side) const -> size_t
{
  LOG_FATAL_IF(_impl.joiningRegionPoints(side).size() < 2) << "join region size is less than 2";
  return 1;
}

auto BottomUpMergeBalance::calcMergeRegionSpan(const size_t& side, const size_t& left_index) const -> MergeRegionSpan
{
  LOG_FATAL_IF(_impl.joiningRegionPoints(side).size() < 2) << "join region size is less than 2";
  const auto right_index = _impl.joiningRegionPoints(side).size() - 2;
  return MergeRegionSpan{.side = side, .left_index = left_index, .right_index = right_index < left_index ? left_index - 1 : right_index};
}

auto BottomUpMergeBalance::appendMergeRegionPointsOnSegment(Area* current_area, const MergeRegionSpan& merge_region_span) const -> void
{
  if (merge_region_span.right_index < merge_region_span.left_index) {
    return;
  }
  if (merge_region_span.side == kLeft) {
    for (size_t point_index = merge_region_span.left_index; point_index <= merge_region_span.right_index; ++point_index) {
      addMergeRegionPointFromJoiningRegion(current_area, merge_region_span.side, point_index);
    }
    return;
  }

  size_t point_index = merge_region_span.right_index;
  while (point_index >= merge_region_span.left_index) {
    addMergeRegionPointFromJoiningRegion(current_area, merge_region_span.side, point_index);
    if (point_index == merge_region_span.left_index) {
      break;
    }
    --point_index;
  }
}

auto BottomUpMergeBalance::addMergeRegionPointFromJoiningRegion(Area* current_area, const size_t& side, const size_t& point_index) const
    -> void
{
  auto point = _impl.joiningRegionPoint(side, point_index);
  const auto original_point = point;
  const auto slope = calcSkewSlope(*current_area);
  const auto dist = (TopDownEmbedding::pointSkew(point) - _impl._skew_bound) / slope;
  if (dist <= 0) {
    current_area->add_merge_region_point(point);
  } else if (dist <= current_area->get_radius()) {
    const auto relative_type = Geom::lineRelative(_impl.topDownEmbedding().getJoiningSegmentLine(kLeft),
                                                  _impl.topDownEmbedding().getJoiningSegmentLine(kRight), side);
    Geom::calcRelativeCoord(point, relative_type, dist);
    const auto horizontal_distance = std::abs(point.x - original_point.x);
    const auto vertical_distance = std::abs(point.y - original_point.y);
    LOG_FATAL_IF(!Equal(horizontal_distance, 0) && !Equal(vertical_distance, 0)) << "not horizontal or vertical";
    const auto incr_delay = side == kLeft
                                ? _impl.topDownEmbedding().calcDelayIncrease(horizontal_distance, vertical_distance,
                                                                             current_area->get_left()->get_cap_load(), _impl._rc_pattern)
                                : _impl.topDownEmbedding().calcDelayIncrease(horizontal_distance, vertical_distance,
                                                                             current_area->get_right()->get_cap_load(), _impl._rc_pattern);
    point.min += incr_delay;
    point.max = _impl._skew_bound + point.min;
    current_area->add_merge_region_point(point);
  }
}

auto BottomUpMergeBalance::calcSkewSlope(const Area& current_area) const -> double
{
  auto* left_child = current_area.get_left();
  auto* right_child = current_area.get_right();
  if (left_child == nullptr || right_child == nullptr) {
    LOG_FATAL << "calcSkewSlope requires both child areas";
    return 0.0;
  }

  const auto left_line = current_area.get_line(kLeft);
  const auto right_line = current_area.get_line(kRight);
  const auto left_x_coord = BoundSkewTreeImpl::linePoint(left_line, kHead).x;
  const auto left_y_coord = BoundSkewTreeImpl::linePoint(left_line, kHead).y;
  const auto right_x_coord = BoundSkewTreeImpl::linePoint(right_line, kHead).x;
  const auto right_y_coord = BoundSkewTreeImpl::linePoint(right_line, kHead).y;
  const auto left_cap_load = left_child->get_cap_load();
  const auto right_cap_load = right_child->get_cap_load();
  if (Equal(left_x_coord, right_x_coord)) {
    return _impl._unit_vertical_resistance
           * (left_cap_load + right_cap_load + (current_area.get_radius() * _impl._unit_vertical_capacitance));
  }
  if (Equal(left_y_coord, right_y_coord)) {
    return _impl._unit_horizontal_resistance
           * (left_cap_load + right_cap_load + (current_area.get_radius() * _impl._unit_horizontal_capacitance));
  }
  LOG_FATAL << "line is not horizontal or vertical";
  return 0.0;
}

}  // namespace icts::bst::detail
