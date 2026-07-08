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
 * @file BottomUpMergeInfeasibility.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief Bound-skew tree infeasibility / transformed-rectangle merge fallback (stage B.3).
 */

#include "bound_skew_tree/algorithm/BottomUpMergeInfeasibility.hh"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <ostream>
#include <utility>

#include "Log.hh"
#include "bound_skew_tree/algorithm/BottomUpMergeBalance.hh"
#include "bound_skew_tree/algorithm/BoundSkewTreeImpl.hh"
#include "bound_skew_tree/algorithm/TopDownEmbedding.hh"
#include "bound_skew_tree/component/Components.hh"

namespace icts::bst::detail {

auto BottomUpMergeInfeasibility::constructInfeasibleMergeRegion(Area* parent) const -> void
{
  calcMinSkewSection(parent);
  calcDetourEdgeLength(parent);
  refineMergeRegionDelay(parent);
}

auto BottomUpMergeInfeasibility::calcMinSkewSection(Area* current_area) const -> void
{
  auto min_skew = std::numeric_limits<double>::max();
  auto min_skew_side = kLeft;
  FOR_EACH_BST_SIDE(side)
  {
    auto min_side_point_skew = std::numeric_limits<double>::max();
    std::ranges::for_each(_impl.joiningRegionPoints(side), [&](const Point& point) -> void {
      min_side_point_skew = std::min(min_side_point_skew, TopDownEmbedding::pointSkew(point));
    });
    if (min_side_point_skew < min_skew) {
      min_skew = min_side_point_skew;
      min_skew_side = side;
    }
  }
  std::ranges::for_each(_impl.joiningRegionPoints(min_skew_side), [&](const Point& point) -> void {
    if (Equal(TopDownEmbedding::pointSkew(point), min_skew)) {
      current_area->add_merge_region_point(point);
    }
  });
}

auto BottomUpMergeInfeasibility::calcDetourEdgeLength(Area* current_area) const -> void
{
  const auto left_line = current_area->get_line(kLeft);
  const auto right_line = current_area->get_line(kRight);
  auto left_point = BoundSkewTreeImpl::linePoint(left_line, kHead);
  auto right_point = BoundSkewTreeImpl::linePoint(right_line, kHead);
  left_point.val = current_area->get_left()->get_cap_load();
  right_point.val = current_area->get_right()->get_cap_load();
  auto delta = TopDownEmbedding::pointSkew(current_area->get_merge_region().front()) - _impl._skew_bound;
  LOG_FATAL_IF(delta <= 0) << "remain skew less than 0";
  auto [horizontal_distance, vertical_distance] = BoundSkewTreeImpl::calcManhattanDistanceComponents(left_point, right_point);
  if (left_point.max > right_point.max) {
    right_point.max
        = left_point.max - delta
          - _impl.topDownEmbedding().calcDelayIncrease(horizontal_distance, vertical_distance, right_point.val, _impl._rc_pattern);
    BalancePointResult result;
    _impl.bottomUpMergeBalance().calcBalanceBetweenPoints(BalancePointQuery{.first_point = left_point,
                                                                            .second_point = right_point,
                                                                            .timing_type = kMax,
                                                                            .balance_ref_axis = BalanceRefAxis::kX,
                                                                            .rc_pattern = _impl._rc_pattern},
                                                          result);
    LOG_FATAL_IF(result.distance_to_first > kEpsilon) << "dist to left_point should be zero";
    current_area->set_edge_len(kLeft, 0);
    current_area->set_edge_len(kRight, result.distance_to_second);
  } else {
    left_point.max
        = right_point.max - delta
          - _impl.topDownEmbedding().calcDelayIncrease(horizontal_distance, vertical_distance, left_point.val, _impl._rc_pattern);
    BalancePointResult result;
    _impl.bottomUpMergeBalance().calcBalanceBetweenPoints(BalancePointQuery{.first_point = left_point,
                                                                            .second_point = right_point,
                                                                            .timing_type = kMax,
                                                                            .balance_ref_axis = BalanceRefAxis::kX,
                                                                            .rc_pattern = _impl._rc_pattern},
                                                          result);
    LOG_FATAL_IF(result.distance_to_second > kEpsilon) << "dist to right_point should be zero";
    current_area->set_edge_len(kLeft, result.distance_to_first);
    current_area->set_edge_len(kRight, 0);
  }
}

auto BottomUpMergeInfeasibility::refineMergeRegionDelay(Area* current_area) const -> void
{
  auto merge_region = current_area->get_merge_region();
  std::ranges::for_each(merge_region, [&](Point& point) -> void { point.min = point.max - _impl._skew_bound; });
  current_area->set_merge_region(merge_region);
}

auto BottomUpMergeInfeasibility::constructTransformedRectMergeRegion(Area* current_area) const -> void
{
  TransformedRect left_transformed_rect;
  Geom::buildTransformedRect(_impl.mergeSegment(kLeft), current_area->get_edge_len(kLeft), left_transformed_rect);
  TransformedRect right_transformed_rect;
  Geom::buildTransformedRect(_impl.mergeSegment(kRight), current_area->get_edge_len(kRight), right_transformed_rect);

  TransformedRect intersect;
  Geom::makeIntersection(left_transformed_rect, right_transformed_rect, intersect);
  Geom::transformedRectCore(intersect, intersect);
  Region merge_region;
  Geom::transformedRectToRegion(intersect, merge_region);
  const auto reference_point = current_area->get_merge_region().front();
  std::ranges::for_each(merge_region, [&](Point& point) -> void {
    point.min = reference_point.min;
    point.max = reference_point.max;
  });
  current_area->set_merge_region(merge_region);
}

}  // namespace icts::bst::detail
