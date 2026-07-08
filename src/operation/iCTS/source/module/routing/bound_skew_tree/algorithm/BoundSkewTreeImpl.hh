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
 * @file BoundSkewTreeImpl.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief Private Pimpl aggregate holding the bound-skew tree algorithm state
 *        plus the 6 cooperating components (binary topology, bottom-up merge
 *        joining / balance / infeasibility, top-down embedding, and the
 *        A->B->C pipeline orchestrator). Lives entirely behind the
 *        BoundSkewTree facade. All private nested types and shared data
 *        members of the algorithm live here at namespace `icts::bst::detail`.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "bound_skew_tree/component/Components.hh"
#include "bound_skew_tree/config/BSTRoutingConfig.hh"
#include "bound_skew_tree/geometry/GeomCalc.hh"

namespace icts::bst::detail {

using Geom = GeomCalc;

struct KMeansConfig
{
  size_t cluster_count = 0;
  uint32_t seed = 0;
  size_t max_iter = 10;
};

struct MergeAreas
{
  Area* parent = nullptr;
  Area* left = nullptr;
  Area* right = nullptr;
};

struct EmbeddingStep
{
  Area* parent = nullptr;
  Area* child = nullptr;
  size_t side = 0;
};

enum class BalanceRefAxis : size_t
{
  kX = 0,
  kY = 1,
};

struct JoiningSegmentDelayQuery
{
  size_t joining_region_side = 0;
  size_t segment_side = 0;
  size_t point_index = 0;
  size_t timing_type = 0;
};

struct SideDelay
{
  double left = 0.0;
  double right = 0.0;

  auto get(const size_t& side) const -> double { return side == kLeft ? left : right; }
};

struct BalancePointQuery
{
  Point first_point;
  Point second_point;
  size_t timing_type = 0;
  BalanceRefAxis balance_ref_axis = BalanceRefAxis::kX;
  BSTRoutingRCPattern rc_pattern = BSTRoutingRCPattern::kHV;
};

struct BalancePointResult
{
  double distance_to_first = 0.0;
  double distance_to_second = 0.0;
  Point balance_point;
};

struct MergeDistances
{
  double distance_to_first = 0.0;
  double distance_to_second = 0.0;
};

struct MergeRegionSpan
{
  size_t side = 0;
  size_t left_index = 0;
  size_t right_index = 0;
};

template <typename T>
struct SideState
{
  T left{};
  T right{};

  auto forSide(const size_t side) -> T& { return side == kLeft ? left : right; }
  auto forSide(const size_t side) const -> const T& { return side == kLeft ? left : right; }
};

template <typename T>
struct EndState
{
  T head{};
  T tail{};

  auto forEnd(const size_t end_side) -> T& { return end_side == kHead ? head : tail; }
  auto forEnd(const size_t end_side) const -> const T& { return end_side == kHead ? head : tail; }
};

template <typename T>
struct TimingState
{
  T min{};
  T max{};

  auto forTiming(const size_t timing_type) -> T& { return timing_type == kMin ? min : max; }
  auto forTiming(const size_t timing_type) const -> const T& { return timing_type == kMin ? min : max; }
};

struct AxisDelayFactor
{
  double horizontal = 0.0;
  double vertical = 0.0;
};

class BstPipeline;
class BinaryTopology;
class BottomUpMergeJoining;
class BottomUpMergeBalance;
class TopDownEmbedding;
class BottomUpMergeInfeasibility;

/**
 * @brief Pimpl aggregate for BoundSkewTree.
 *
 * Holds:
 *   - the full algorithm state (formerly private fields of BoundSkewTree);
 *   - the 6 cooperating components (one `std::unique_ptr` each);
 *   - shared math helpers used across multiple components (cost / merge / etc.).
 *
 * Components are friends so they can read/write the algorithm state directly,
 * and call cross-component helpers via the public accessors below.
 */
class BoundSkewTreeImpl
{
  friend class BstPipeline;
  friend class BinaryTopology;
  friend class BottomUpMergeJoining;
  friend class BottomUpMergeBalance;
  friend class TopDownEmbedding;
  friend class BottomUpMergeInfeasibility;

 public:
  static constexpr double kHalfFactor = 0.5;
  using CostFunc = std::function<double(Area*, Area*)>;

  BoundSkewTreeImpl(std::vector<std::unique_ptr<Area>> load_areas, const BSTRoutingConfig& parameters,
                    const BSTRoutingTopologyMode& topology_mode);
  BoundSkewTreeImpl(std::vector<std::unique_ptr<Area>> owned_areas, Area* root, const BSTRoutingConfig& parameters);

  BoundSkewTreeImpl(const BoundSkewTreeImpl&) = delete;
  BoundSkewTreeImpl(BoundSkewTreeImpl&&) = delete;
  auto operator=(const BoundSkewTreeImpl&) -> BoundSkewTreeImpl& = delete;
  auto operator=(BoundSkewTreeImpl&&) -> BoundSkewTreeImpl& = delete;
  ~BoundSkewTreeImpl();

  auto run() -> void;
  auto get_root() const -> Area* { return _root; }
  auto set_root_guide(const double& x_coord, const double& y_coord) -> void { _root_guide = Point(x_coord, y_coord, 0, 0, 0); }
  auto set_rc_pattern(const BSTRoutingRCPattern& rc_pattern) -> void { _rc_pattern = rc_pattern; }

  auto pipeline() -> BstPipeline&;
  auto binaryTopology() -> BinaryTopology&;
  auto bottomUpMergeJoining() -> BottomUpMergeJoining&;
  auto bottomUpMergeBalance() -> BottomUpMergeBalance&;
  auto topDownEmbedding() -> TopDownEmbedding&;
  auto bottomUpMergeInfeasibility() -> BottomUpMergeInfeasibility&;

  auto getBestMatch(const CostFunc& cost_func) const -> Match;
  auto mergeCost(Area* left, Area* right) const -> double;
  static auto distanceCost(Area* left, Area* right) -> double;
  auto makeArea(const size_t& area_id) -> Area*;
  auto merge(Area* left, Area* right) -> Area*;
  auto areaReset() -> void;
  static auto resetPointValues(Area* root) -> void;
  static auto calcManhattanDistanceComponents(const Point& first_point, const Point& second_point) -> std::pair<double, double>;

  static auto otherSide(const size_t& side) -> size_t { return side == kLeft ? kRight : kLeft; }
  static auto oppositeEnd(const size_t& end_side) -> size_t { return end_side == kHead ? kTail : kHead; }
  static auto pointAt(Points& points, const size_t& point_index) -> Point& { return points.at(point_index); }
  static auto pointAt(const Points& points, const size_t& point_index) -> const Point& { return points.at(point_index); }
  static auto linePoint(Line& line, const size_t& end_side) -> Point& { return line.at(end_side); }
  static auto linePoint(const Line& line, const size_t& end_side) -> const Point& { return line.at(end_side); }

  auto joiningRegionPoints(const size_t& side) -> Points& { return _joining_region.forSide(side); }
  auto joiningRegionPoints(const size_t& side) const -> const Points& { return _joining_region.forSide(side); }
  auto joiningSegmentPoints(const size_t& side) -> Points& { return _joining_segment.forSide(side); }
  auto joiningSegmentPoints(const size_t& side) const -> const Points& { return _joining_segment.forSide(side); }
  auto mergeSegment(const size_t& side) -> TransformedRect& { return _merge_segment.forSide(side); }
  auto mergeSegment(const size_t& side) const -> const TransformedRect& { return _merge_segment.forSide(side); }
  auto balancePoints(const size_t& end_side) -> Points& { return _balance_points.forEnd(end_side); }
  auto balancePoints(const size_t& end_side) const -> const Points& { return _balance_points.forEnd(end_side); }
  auto joiningCornerPoint(const size_t& end_side) -> Point& { return _joining_corner.forEnd(end_side); }
  auto joiningCornerPoint(const size_t& end_side) const -> const Point& { return _joining_corner.forEnd(end_side); }
  auto feasibleMergeSegmentPoints(const size_t& end_side) -> Points& { return _feasible_merge_segment_points.forEnd(end_side); }
  auto feasibleMergeSegmentPoints(const size_t& end_side) const -> const Points& { return _feasible_merge_segment_points.forEnd(end_side); }
  auto joiningSegmentPoint(const size_t& side, const size_t& point_index) -> Point&
  {
    return pointAt(joiningSegmentPoints(side), point_index);
  }
  auto joiningSegmentPoint(const size_t& side, const size_t& point_index) const -> const Point&
  {
    return pointAt(joiningSegmentPoints(side), point_index);
  }
  auto joiningRegionPoint(const size_t& side, const size_t& point_index) -> Point&
  {
    return pointAt(joiningRegionPoints(side), point_index);
  }
  auto joiningRegionPoint(const size_t& side, const size_t& point_index) const -> const Point&
  {
    return pointAt(joiningRegionPoints(side), point_index);
  }

 private:
  size_t _id = 0;

  std::vector<std::unique_ptr<Area>> _owned_areas;
  std::vector<Area*> _unmerged_nodes;
  std::optional<Point> _root_guide;
  BSTRoutingTopologyMode _topology_mode = BSTRoutingTopologyMode::kSourceRouteTree;

  Area* _root = nullptr;

  double _skew_bound = 0;
  BSTRoutingRCPattern _rc_pattern = BSTRoutingRCPattern::kHV;

  double _unit_horizontal_capacitance = 0.0;
  double _unit_horizontal_resistance = 0.0;
  double _unit_vertical_capacitance = 0.0;
  double _unit_vertical_resistance = 0.0;
  AxisDelayFactor _delay_quadratic_factor;

  SideState<Points> _joining_region;
  SideState<Points> _joining_segment;
  SideState<TransformedRect> _merge_segment;
  EndState<Points> _balance_points;
  EndState<Point> _joining_corner;
  EndState<Points> _feasible_merge_segment_points;

  std::unique_ptr<BstPipeline> _pipeline;
  std::unique_ptr<BinaryTopology> _binary_topology;
  std::unique_ptr<BottomUpMergeJoining> _bottom_up_merge_joining;
  std::unique_ptr<BottomUpMergeBalance> _bottom_up_merge_balance;
  std::unique_ptr<TopDownEmbedding> _top_down_embedding;
  std::unique_ptr<BottomUpMergeInfeasibility> _bottom_up_merge_infeasibility;
};

}  // namespace icts::bst::detail
