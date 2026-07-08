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
 * @file FastClusteringGeometry.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Geometry and scoring helpers for fast topology clustering.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "Pin.hh"
#include "Point.hh"
#include "TopologyConfig.hh"
#include "cluster_draft/FastClusteringDraft.hh"

namespace icts::fast_clustering {

auto IsEmpty(const Bounds& bounds) -> bool
{
  return bounds.min_x > bounds.max_x || bounds.min_y > bounds.max_y;
}

auto ExtendBounds(Bounds bounds, const Point<int>& point) -> Bounds
{
  bounds.min_x = std::min(bounds.min_x, point.get_x());
  bounds.min_y = std::min(bounds.min_y, point.get_y());
  bounds.max_x = std::max(bounds.max_x, point.get_x());
  bounds.max_y = std::max(bounds.max_y, point.get_y());
  return bounds;
}

auto CalcDiameter(const Bounds& bounds) -> int
{
  if (IsEmpty(bounds)) {
    return 0;
  }
  return (bounds.max_x - bounds.min_x) + (bounds.max_y - bounds.min_y);
}

auto CalcClusterBounds(const std::vector<Pin*>& cluster) -> Bounds
{
  Bounds bounds;
  for (const auto* pin : cluster) {
    if (pin == nullptr) {
      continue;
    }
    bounds = ExtendBounds(bounds, pin->get_location());
  }
  return bounds;
}

auto CalcClusterBounds(const std::vector<std::size_t>& entry_ids, const std::vector<LoadEntry>& entries) -> Bounds
{
  Bounds bounds;
  for (const auto entry_id : entry_ids) {
    bounds = ExtendBounds(bounds, entries.at(entry_id).location);
  }
  return bounds;
}

auto CalcManhattanDistance(const Point<int>& lhs, const Point<int>& rhs) -> double
{
  const auto dx = std::abs(static_cast<long long>(lhs.get_x()) - static_cast<long long>(rhs.get_x()));
  const auto dy = std::abs(static_cast<long long>(lhs.get_y()) - static_cast<long long>(rhs.get_y()));
  return static_cast<double>(dx + dy);
}

namespace {

auto CalcEntryCenter(const std::vector<std::size_t>& entry_ids, const std::vector<LoadEntry>& entries) -> Point<int>
{
  if (entry_ids.empty()) {
    return {0, 0};
  }

  long long x_sum = 0;
  long long y_sum = 0;
  for (const auto entry_id : entry_ids) {
    const auto location = entries.at(entry_id).location;
    x_sum += location.get_x();
    y_sum += location.get_y();
  }
  return {static_cast<int>(std::lround(static_cast<double>(x_sum) / static_cast<double>(entry_ids.size()))),
          static_cast<int>(std::lround(static_cast<double>(y_sum) / static_cast<double>(entry_ids.size())))};
}

auto CalcEntryMedian(const std::vector<std::size_t>& entry_ids, const std::vector<LoadEntry>& entries) -> Point<int>
{
  std::vector<int> x_coords;
  std::vector<int> y_coords;
  x_coords.reserve(entry_ids.size());
  y_coords.reserve(entry_ids.size());
  for (const auto entry_id : entry_ids) {
    const auto location = entries.at(entry_id).location;
    x_coords.push_back(location.get_x());
    y_coords.push_back(location.get_y());
  }
  if (x_coords.empty()) {
    return {0, 0};
  }

  const auto middle = static_cast<std::ptrdiff_t>(x_coords.size() / 2U);
  auto x_middle = x_coords.begin() + middle;
  auto y_middle = y_coords.begin() + middle;
  std::nth_element(x_coords.begin(), x_middle, x_coords.end());
  std::nth_element(y_coords.begin(), y_middle, y_coords.end());
  return {*x_middle, *y_middle};
}

}  // namespace

auto ResolveDraftRoot(const std::vector<std::size_t>& entry_ids, const std::vector<LoadEntry>& entries, const ClusterConfig& config)
    -> Point<int>
{
  return config.root_policy == ClusterRootPolicy::kCenter ? CalcEntryCenter(entry_ids, entries) : CalcEntryMedian(entry_ids, entries);
}

namespace {

auto CalcRoutingCapProxy(const std::vector<std::size_t>& entry_ids, const std::vector<LoadEntry>& entries, const ClusterConfig& config)
    -> double
{
  if (entry_ids.size() <= 1U) {
    return 0.0;
  }

  const auto root = ResolveDraftRoot(entry_ids, entries, config);
  double proxy = 0.0;
  for (const auto entry_id : entry_ids) {
    proxy += CalcManhattanDistance(root, entries.at(entry_id).location);
  }
  return proxy;
}

}  // namespace

auto BuildDraft(std::vector<std::size_t> entry_ids, const std::vector<LoadEntry>& entries, const ClusterConfig& config) -> ClusterDraft
{
  auto bounds = CalcClusterBounds(entry_ids, entries);
  auto routing_cap_proxy = CalcRoutingCapProxy(entry_ids, entries, config);
  return ClusterDraft{
      .entry_ids = std::move(entry_ids),
      .bounds = bounds,
      .routing_cap_proxy = routing_cap_proxy,
  };
}

auto CalcCenter(const std::vector<Pin*>& cluster) -> Point<int>
{
  if (cluster.empty()) {
    return {0, 0};
  }

  long long x_sum = 0;
  long long y_sum = 0;
  std::size_t pin_count = 0;
  for (const auto* pin : cluster) {
    if (pin == nullptr) {
      continue;
    }
    const auto location = pin->get_location();
    x_sum += location.get_x();
    y_sum += location.get_y();
    ++pin_count;
  }
  if (pin_count == 0U) {
    return {0, 0};
  }
  return {static_cast<int>(std::lround(static_cast<double>(x_sum) / static_cast<double>(pin_count))),
          static_cast<int>(std::lround(static_cast<double>(y_sum) / static_cast<double>(pin_count)))};
}

namespace {

auto CalcMedian(const std::vector<Pin*>& cluster) -> Point<int>
{
  std::vector<int> x_coords;
  std::vector<int> y_coords;
  x_coords.reserve(cluster.size());
  y_coords.reserve(cluster.size());
  for (const auto* pin : cluster) {
    if (pin == nullptr) {
      continue;
    }
    const auto location = pin->get_location();
    x_coords.push_back(location.get_x());
    y_coords.push_back(location.get_y());
  }
  if (x_coords.empty()) {
    return {0, 0};
  }

  const auto middle = static_cast<std::ptrdiff_t>(x_coords.size() / 2U);
  auto x_middle = x_coords.begin() + middle;
  auto y_middle = y_coords.begin() + middle;
  std::nth_element(x_coords.begin(), x_middle, x_coords.end());
  std::nth_element(y_coords.begin(), y_middle, y_coords.end());
  return {*x_middle, *y_middle};
}

}  // namespace

auto ResolveEvaluationRoot(const std::vector<Pin*>& cluster, const ClusterConfig& config) -> Point<int>
{
  return config.root_policy == ClusterRootPolicy::kCenter ? CalcCenter(cluster) : CalcMedian(cluster);
}

auto ResolvePackingFanoutLimit(const ClusterConfig& config, std::size_t load_count) -> std::size_t
{
  if (load_count == 0U) {
    return 0U;
  }
  if (config.max_fanout > 0U) {
    return std::min(config.max_fanout, load_count);
  }
  return std::min<std::size_t>(load_count, kDefaultPackingFanout);
}

auto IsFanoutLegal(std::size_t fanout, const ClusterConfig& config) -> bool
{
  return config.max_fanout == 0U || fanout <= config.max_fanout;
}

auto IsDiameterLegal(const Bounds& bounds, const ClusterConfig& config) -> bool
{
  return config.max_diameter <= 0 || CalcDiameter(bounds) <= config.max_diameter;
}

auto IsDraftGeometryLegal(const ClusterDraft& draft, const ClusterConfig& config) -> bool
{
  return IsFanoutLegal(draft.entry_ids.size(), config) && IsDiameterLegal(draft.bounds, config);
}

auto ClusterScoreProxy(const ClusterDraft& draft, const ClusterConfig& config) -> double
{
  if (draft.entry_ids.empty()) {
    return std::numeric_limits<double>::infinity();
  }

  const auto diameter = CalcDiameter(draft.bounds);
  if (config.scoring_strategy == ClusterScoringStrategy::kTotalWirelength) {
    return config.wirelength_weight * static_cast<double>(diameter);
  }
  if (diameter > 0) {
    return static_cast<double>(diameter);
  }
  return config.max_diameter > 0 ? static_cast<double>(config.max_diameter) : 0.0;
}

auto CalcRoutingCapVariancePenalty(double routing_cap_proxy, double target_routing_cap_proxy) -> double
{
  const auto safe_target = std::max(1.0, target_routing_cap_proxy);
  const auto delta = routing_cap_proxy - target_routing_cap_proxy;
  return delta * delta / safe_target;
}

auto DraftObjective(const ClusterDraft& draft, const ClusterConfig& config, double target_routing_cap_proxy,
                    double routing_cap_balance_weight) -> double
{
  return ClusterScoreProxy(draft, config)
         + routing_cap_balance_weight * CalcRoutingCapVariancePenalty(draft.routing_cap_proxy, target_routing_cap_proxy);
}

auto CalcDraftAggregate(const std::vector<ClusterDraft>& drafts) -> DraftAggregate
{
  DraftAggregate aggregate;
  for (const auto& draft : drafts) {
    if (!draft.active || draft.entry_ids.empty()) {
      continue;
    }
    ++aggregate.active_count;
    aggregate.total_routing_cap_proxy += draft.routing_cap_proxy;
  }
  return aggregate;
}

auto CalcMeanRoutingCapProxy(const DraftAggregate& aggregate) -> double
{
  if (aggregate.active_count == 0U) {
    return 0.0;
  }
  return aggregate.total_routing_cap_proxy / static_cast<double>(aggregate.active_count);
}

auto CalcBoundsDistance(const Bounds& lhs, const Bounds& rhs) -> long long
{
  if (IsEmpty(lhs) || IsEmpty(rhs)) {
    return 0;
  }

  long long dx = 0;
  if (lhs.max_x < rhs.min_x) {
    dx = static_cast<long long>(rhs.min_x) - lhs.max_x;
  } else if (rhs.max_x < lhs.min_x) {
    dx = static_cast<long long>(lhs.min_x) - rhs.max_x;
  }

  long long dy = 0;
  if (lhs.max_y < rhs.min_y) {
    dy = static_cast<long long>(rhs.min_y) - lhs.max_y;
  } else if (rhs.max_y < lhs.min_y) {
    dy = static_cast<long long>(lhs.min_y) - rhs.max_y;
  }
  return dx + dy;
}

auto DistanceToBounds(const Point<int>& point, const Bounds& bounds) -> long long
{
  if (IsEmpty(bounds)) {
    return 0;
  }

  long long dx = 0;
  if (point.get_x() < bounds.min_x) {
    dx = static_cast<long long>(bounds.min_x) - point.get_x();
  } else if (point.get_x() > bounds.max_x) {
    dx = static_cast<long long>(point.get_x()) - bounds.max_x;
  }

  long long dy = 0;
  if (point.get_y() < bounds.min_y) {
    dy = static_cast<long long>(bounds.min_y) - point.get_y();
  } else if (point.get_y() > bounds.max_y) {
    dy = static_cast<long long>(point.get_y()) - bounds.max_y;
  }
  return dx + dy;
}

}  // namespace icts::fast_clustering
