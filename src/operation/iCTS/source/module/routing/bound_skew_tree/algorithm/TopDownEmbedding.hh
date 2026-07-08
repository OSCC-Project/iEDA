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
 * @file TopDownEmbedding.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief Stage C · Top-down embedding, point-delay propagation, and the
 *        low-level delay / merge-region geometry math used by the
 *        bound-skew tree algorithm. The instance methods read the per-run
 *        unit RC / pattern state on `BoundSkewTreeImpl`; the static methods
 *        are pure geometry helpers exposed to the bottom-up components.
 */

#pragma once

#include <cstddef>

#include "bound_skew_tree/component/Components.hh"
#include "bound_skew_tree/geometry/GeomCalc.hh"

namespace icts {
enum class BSTRoutingRCPattern;
}  // namespace icts

namespace icts::bst::detail {

class BoundSkewTreeImpl;
struct EmbeddingStep;

class TopDownEmbedding
{
 public:
  explicit TopDownEmbedding(BoundSkewTreeImpl& impl) : _impl(impl) {}
  ~TopDownEmbedding() = default;
  TopDownEmbedding(const TopDownEmbedding&) = delete;
  auto operator=(const TopDownEmbedding&) -> TopDownEmbedding& = delete;

  // ----- pure geometry helpers (static, no state) ---------------------------
  static auto embedChild(const EmbeddingStep& embedding_target) -> void;
  static auto isTransformedRectArea(Area* current_area) -> bool;
  static auto isManhattanArea(Area* current_area) -> bool;
  static auto mergeRegionToTransformedRect(const Region& merge_region, TransformedRect& transformed_rect) -> void;
  static auto calcAreaLineType(const Area& current_area) -> LineType;
  static auto calcConvexHull(Area* current_area) -> void;
  static auto calcJoiningRegionArea(const Line& first_line, const Line& second_line) -> double;
  static auto locateBoundarySegment(Area* current_area, Point& point, Line& boundary_segment) -> void;
  static auto pointSkew(const Point& point) -> double;
  static auto checkPointDelay(Point& point) -> void;

  // ----- joining-line accessors that need impl state ------------------------
  auto getJoiningRegionLine(const size_t& side) const -> Line;
  auto getJoiningSegmentLine(const size_t& side) const -> Line;
  auto setJoiningRegionLine(const size_t& side, const Line& line) -> void;
  auto setJoiningSegmentLine(const size_t& side, const Line& line) -> void;
  auto checkJoiningSegmentMergeSegment() const -> void;
  auto checkUpdatedJoiningSegment(const Area* current_area, Line& left_line, Line& right_line) const -> void;

  // ----- delay-update math (uses per-run unit RC / pattern) ----------------
  auto calcSimplePointDelays(Point& point, Line& boundary_segment) const -> bool;
  auto calcSegmentPointDelays(Point& point, Line& boundary_segment) const -> void;
  auto calcPointDelays(const Area& current_area, Point& point, Line& boundary_segment) const -> void;
  auto updatePointDelaysByEndSide(const Area& current_area, const size_t& end_side, Point& point) const -> void;
  auto calcIrregularPointDelays(const Area& current_area, Point& point, Line& boundary_segment) const -> void;
  auto pointDelayIncrease(const Point& lhs_point, const Point& rhs_point, const double& cap_load,
                          const BSTRoutingRCPattern& rc_pattern) const -> double;
  auto pointDelayIncrease(const Point& lhs_point, const Point& rhs_point, const double& length, const double& cap_load,
                          const BSTRoutingRCPattern& rc_pattern) const -> double;
  auto calcDelayIncrease(const double& horizontal_length, const double& vertical_length, const double& cap_load,
                         const BSTRoutingRCPattern& rc_pattern) const -> double;

 private:
  BoundSkewTreeImpl& _impl;
};

}  // namespace icts::bst::detail
