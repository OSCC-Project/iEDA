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
 * @file BottomUpMergeJoining.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief Stage B.1 · Joining-segment / joining-region construction for
 *        bound-skew trees during bottom-up merging. Computes the merge
 *        segment between two child areas, derives the joining region and
 *        its corner / turn-point structure.
 */

#pragma once

#include <cstddef>

#include "bound_skew_tree/component/Components.hh"

namespace icts::bst::detail {

class BoundSkewTreeImpl;
struct MergeAreas;
struct JoiningSegmentDelayQuery;
struct SideDelay;

class BottomUpMergeJoining
{
 public:
  explicit BottomUpMergeJoining(BoundSkewTreeImpl& impl) : _impl(impl) {}
  ~BottomUpMergeJoining() = default;
  BottomUpMergeJoining(const BottomUpMergeJoining&) = delete;
  auto operator=(const BottomUpMergeJoining&) -> BottomUpMergeJoining& = delete;

  // entry points called by the pipeline
  auto calcJoiningSegment(const MergeAreas& merge_areas) -> void;
  auto processJoiningSegment(Area* current_area) -> void;
  auto constructMergeRegion(const MergeAreas& merge_areas) -> void;

  // entry points called from infeasibility fallback / balance components
  auto calcJoiningRegionCorner(const Area& current_area) -> void;
  auto joiningRegionCornerExists(const size_t& end_side) const -> bool;

  auto delayFromJoiningSegment(const JoiningSegmentDelayQuery& query, const SideDelay& delay_from) const -> double;

 private:
  auto initSide() -> void;
  auto calcJoiningSegment(Area* current_area, Line& left_line, Line& right_line) -> void;
  auto calcJoiningSegmentDelay(Area* left_area, Area* right_area) -> void;
  auto updateJoiningSegment(Area* current_area, Line& left_line, Line& right_line, PointPair closest_pair) -> void;
  auto addJoiningSegmentPoints(const MergeAreas& merge_areas) -> void;
  auto calcJoiningRegion(const MergeAreas& merge_areas) -> void;
  auto calcJoiningRegionEndpoints(const Area& current_area) -> void;
  auto calcNonManhattanJoiningRegionEndpoints(const MergeAreas& merge_areas) -> void;
  auto addTurnPoint(const size_t& side, const size_t& point_index, const size_t& timing_type, const SideDelay& delay_from) -> void;
  auto addFeasibleMergeSegmentToJoiningRegion() -> void;

  BoundSkewTreeImpl& _impl;
};

}  // namespace icts::bst::detail
