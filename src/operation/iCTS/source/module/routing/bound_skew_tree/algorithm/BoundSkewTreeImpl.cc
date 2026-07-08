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
 * @file BoundSkewTreeImpl.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-20
 * @brief Bound-skew tree Pimpl implementation: constructors / destructor,
 *        public-API forwarders, component accessors, and shared math used by
 *        multiple cooperating components.
 */

#include "bound_skew_tree/algorithm/BoundSkewTreeImpl.hh"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <utility>
#include <vector>

#include "Log.hh"
#include "Point.hh"
#include "bound_skew_tree/algorithm/BinaryTopology.hh"
#include "bound_skew_tree/algorithm/BottomUpMergeBalance.hh"
#include "bound_skew_tree/algorithm/BottomUpMergeInfeasibility.hh"
#include "bound_skew_tree/algorithm/BottomUpMergeJoining.hh"
#include "bound_skew_tree/algorithm/BstPipeline.hh"
#include "bound_skew_tree/algorithm/TopDownEmbedding.hh"

namespace icts::bst::detail {

BoundSkewTreeImpl::BoundSkewTreeImpl(std::vector<std::unique_ptr<Area>> load_areas, const BSTRoutingConfig& parameters,
                                     const BSTRoutingTopologyMode& topology_mode)
    : _owned_areas(std::move(load_areas)),
      _topology_mode(topology_mode),
      _skew_bound(parameters.skew_bound),
      _rc_pattern(parameters.rc_pattern),
      _unit_horizontal_capacitance(parameters.unit_h_cap),
      _unit_horizontal_resistance(parameters.unit_h_res),
      _unit_vertical_capacitance(parameters.unit_v_cap),
      _unit_vertical_resistance(parameters.unit_v_res),
      _delay_quadratic_factor{.horizontal = kHalfFactor * _unit_horizontal_resistance * _unit_horizontal_capacitance,
                              .vertical = kHalfFactor * _unit_vertical_resistance * _unit_vertical_capacitance}
{
  LOG_FATAL_IF(topology_mode == BSTRoutingTopologyMode::kSourceRouteTree) << "Normal BST construction cannot use source-route-tree mode.";
  _unmerged_nodes.reserve(_owned_areas.size());
  for (const auto& area : _owned_areas) {
    _unmerged_nodes.push_back(area.get());
  }
  if (parameters.root_guide.has_value()) {
    set_root_guide(parameters.root_guide->get_x(), parameters.root_guide->get_y());
  }
  _pipeline = std::make_unique<BstPipeline>(*this);
  _binary_topology = std::make_unique<BinaryTopology>(*this);
  _bottom_up_merge_joining = std::make_unique<BottomUpMergeJoining>(*this);
  _bottom_up_merge_balance = std::make_unique<BottomUpMergeBalance>(*this);
  _top_down_embedding = std::make_unique<TopDownEmbedding>(*this);
  _bottom_up_merge_infeasibility = std::make_unique<BottomUpMergeInfeasibility>(*this);
}

BoundSkewTreeImpl::BoundSkewTreeImpl(std::vector<std::unique_ptr<Area>> owned_areas, Area* root, const BSTRoutingConfig& parameters)
    : _owned_areas(std::move(owned_areas)),
      _root(root),
      _skew_bound(parameters.skew_bound),
      _rc_pattern(parameters.rc_pattern),
      _unit_horizontal_capacitance(parameters.unit_h_cap),
      _unit_horizontal_resistance(parameters.unit_h_res),
      _unit_vertical_capacitance(parameters.unit_v_cap),
      _unit_vertical_resistance(parameters.unit_v_res),
      _delay_quadratic_factor{.horizontal = kHalfFactor * _unit_horizontal_resistance * _unit_horizontal_capacitance,
                              .vertical = kHalfFactor * _unit_vertical_resistance * _unit_vertical_capacitance}
{
  LOG_FATAL_IF(root == nullptr) << "BST source-route-tree root area is null.";
  if (parameters.root_guide.has_value()) {
    set_root_guide(parameters.root_guide->get_x(), parameters.root_guide->get_y());
  }
  _pipeline = std::make_unique<BstPipeline>(*this);
  _binary_topology = std::make_unique<BinaryTopology>(*this);
  _bottom_up_merge_joining = std::make_unique<BottomUpMergeJoining>(*this);
  _bottom_up_merge_balance = std::make_unique<BottomUpMergeBalance>(*this);
  _top_down_embedding = std::make_unique<TopDownEmbedding>(*this);
  _bottom_up_merge_infeasibility = std::make_unique<BottomUpMergeInfeasibility>(*this);
}

BoundSkewTreeImpl::~BoundSkewTreeImpl() = default;

auto BoundSkewTreeImpl::run() -> void
{
  _pipeline->run();
}

auto BoundSkewTreeImpl::pipeline() -> BstPipeline&
{
  return *_pipeline;
}

auto BoundSkewTreeImpl::binaryTopology() -> BinaryTopology&
{
  return *_binary_topology;
}

auto BoundSkewTreeImpl::bottomUpMergeJoining() -> BottomUpMergeJoining&
{
  return *_bottom_up_merge_joining;
}

auto BoundSkewTreeImpl::bottomUpMergeBalance() -> BottomUpMergeBalance&
{
  return *_bottom_up_merge_balance;
}

auto BoundSkewTreeImpl::topDownEmbedding() -> TopDownEmbedding&
{
  return *_top_down_embedding;
}

auto BoundSkewTreeImpl::bottomUpMergeInfeasibility() -> BottomUpMergeInfeasibility&
{
  return *_bottom_up_merge_infeasibility;
}

auto BoundSkewTreeImpl::getBestMatch(const CostFunc& cost_func) const -> Match
{
  auto min_cost = std::numeric_limits<double>::max();
  Match best_match;
  for (size_t i = 0; i < _unmerged_nodes.size(); ++i) {
    for (size_t j = i + 1; j < _unmerged_nodes.size(); ++j) {
      auto cost = cost_func(_unmerged_nodes.at(i), _unmerged_nodes.at(j));
      if (cost < min_cost) {
        min_cost = cost;
        best_match = {.left = _unmerged_nodes.at(i), .right = _unmerged_nodes.at(j), .merge_cost = cost};
      }
    }
  }
  return best_match;
}

auto BoundSkewTreeImpl::mergeCost(Area* left, Area* right) const -> double
{
  auto min_distance = std::numeric_limits<double>::max();
  auto left_merge_region = left->get_merge_region();
  auto right_merge_region = right->get_merge_region();
  Point closest_left_point;
  Point closest_right_point;
  for (const auto& left_point : left_merge_region) {
    for (const auto& right_point : right_merge_region) {
      const auto current_distance = Geom::distance(left_point, right_point);
      if (current_distance >= min_distance) {
        continue;
      }
      min_distance = current_distance;
      closest_left_point = left_point;
      closest_right_point = right_point;
    }
  }
  auto left_max = closest_left_point.max;
  auto right_max = closest_right_point.max;
  auto factor = left->get_cap_load() + right->get_cap_load() + (_unit_horizontal_capacitance * min_distance);
  auto len_to_left = (((right_max - left_max) / _unit_horizontal_resistance)
                      + (kHalfFactor * _unit_horizontal_capacitance * min_distance * min_distance) + (min_distance * right->get_cap_load()))
                     / factor;
  if (len_to_left < 0) {
    len_to_left = -len_to_left;
  } else if (len_to_left > min_distance) {
    len_to_left -= min_distance;
  }
  auto latency = left_max + (kHalfFactor * _unit_horizontal_resistance * _unit_horizontal_capacitance * len_to_left * len_to_left)
                 + (_unit_horizontal_resistance * len_to_left * left->get_cap_load());
  return latency;
}

auto BoundSkewTreeImpl::distanceCost(Area* left, Area* right) -> double
{
  auto min_distance = std::numeric_limits<double>::max();
  auto left_merge_region = left->get_merge_region();
  auto right_merge_region = right->get_merge_region();
  for (const auto& left_point : left_merge_region) {
    for (const auto& right_point : right_merge_region) {
      min_distance = std::min(min_distance, Geom::distance(left_point, right_point));
    }
  }
  return min_distance;
}

auto BoundSkewTreeImpl::calcManhattanDistanceComponents(const Point& first_point, const Point& second_point) -> std::pair<double, double>
{
  return {std::abs(first_point.x - second_point.x), std::abs(first_point.y - second_point.y)};
}

auto BoundSkewTreeImpl::makeArea(const size_t& area_id) -> Area*
{
  auto area = std::make_unique<Area>(area_id);
  auto* area_ptr = area.get();
  _owned_areas.push_back(std::move(area));
  return area_ptr;
}

auto BoundSkewTreeImpl::merge(Area* left, Area* right) -> Area*
{
  auto* parent = makeArea(++_id);
  parent->set_rc_pattern(_rc_pattern);
  parent->set_left(left);
  parent->set_right(right);
  left->set_parent(parent);
  right->set_parent(parent);
  return parent;
}

auto BoundSkewTreeImpl::areaReset() -> void
{
  _unmerged_nodes.clear();
  _unmerged_nodes.push_back(_root);
  resetPointValues(_root);
}

auto BoundSkewTreeImpl::resetPointValues(Area* root) -> void
{
  if (root == nullptr) {
    return;
  }

  std::vector<Area*> stack{root};
  while (!stack.empty()) {
    auto* current = stack.back();
    stack.pop_back();

    auto point = current->get_location();
    point.val = 0;
    current->set_location(point);

    if (current->get_right() != nullptr) {
      stack.push_back(current->get_right());
    }
    if (current->get_left() != nullptr) {
      stack.push_back(current->get_left());
    }
  }
}

}  // namespace icts::bst::detail
