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
 * @file BottomUpMergeBalance.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief Stage B.2 · Balance-point computation, feasible-merge-region
 *        construction, and merge-region span emission for bound-skew trees
 *        during bottom-up merging.
 */

#pragma once

#include <cstddef>

#include "bound_skew_tree/algorithm/BoundSkewTreeImpl.hh"
#include "bound_skew_tree/component/Components.hh"

namespace icts::bst::detail {

class BottomUpMergeBalance
{
 public:
  explicit BottomUpMergeBalance(BoundSkewTreeImpl& impl) : _impl(impl) {}
  ~BottomUpMergeBalance() = default;
  BottomUpMergeBalance(const BottomUpMergeBalance&) = delete;
  auto operator=(const BottomUpMergeBalance&) -> BottomUpMergeBalance& = delete;

  // entry points called by joining component / pipeline
  auto calcBalancePoint(const Area& current_area) -> void;
  auto calcBalanceBetweenPoints(const BalancePointQuery& query, BalancePointResult& result) const -> void;
  auto calcFeasibleMergeSegmentPoints(const Area& current_area) -> void;
  auto hasFeasibleMergeSegmentOnJoiningRegion() const -> bool;
  auto constructFeasibleMergeRegion(Area* parent) const -> void;

 private:
  auto calcBalancePointOnLine(const BalancePointQuery& query, BalancePointResult& result) const -> void;
  auto calcBalancePointOffLine(const BalancePointQuery& query, BalancePointResult& result) const -> void;
  static auto calcMergeDist(const double& unit_resistance, const double& unit_capacitance, const double& cap_load_1, const double& delay_1,
                            const double& cap_load_2, const double& delay_2, const double& total_distance) -> MergeDistances;
  static auto calcPointCoordOnLine(const Point& first_point, const Point& second_point, const double& distance_to_first,
                                   const double& distance_to_second, Point& point) -> void;
  auto calcXBalancePosition(const double& delay_1, const double& delay_2, const double& cap_load_1, const double& cap_load_2,
                            const double& horizontal_distance, const double& vertical_distance, BalanceRefAxis balance_ref_axis) const
      -> double;
  auto calcYBalancePosition(const double& delay_1, const double& delay_2, const double& cap_load_1, const double& cap_load_2,
                            const double& horizontal_distance, const double& vertical_distance, BalanceRefAxis balance_ref_axis) const
      -> double;
  auto calcFeasibleMergeSegmentOnLine(const Area& current_area, Point& point, const Point& reference_point, const size_t& end_side) -> bool;
  auto calcFeasibleMergeSegmentBetweenPoints(const Point& high_skew_point, const Point& low_skew_point, Point& feasible_merge_point) const
      -> void;
  auto isJoiningRegionLine() const -> bool;
  auto addMergeRegionBetweenJoiningSegments(Area* current_area, const size_t& end_side) const -> void;
  auto addMergeRegionOnJoiningSegment(Area* current_area, const size_t& side) const -> void;
  auto calcMergeRegionLeftIndex(const size_t& side) const -> size_t;
  auto calcMergeRegionSpan(const size_t& side, const size_t& left_index) const -> MergeRegionSpan;
  auto appendMergeRegionPointsOnSegment(Area* current_area, const MergeRegionSpan& merge_region_span) const -> void;
  auto addMergeRegionPointFromJoiningRegion(Area* current_area, const size_t& side, const size_t& point_index) const -> void;
  auto calcSkewSlope(const Area& current_area) const -> double;

  BoundSkewTreeImpl& _impl;
};

}  // namespace icts::bst::detail
