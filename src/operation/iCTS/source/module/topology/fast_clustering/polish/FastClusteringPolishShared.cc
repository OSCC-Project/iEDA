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
 * @file FastClusteringPolishShared.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Shared polish, neighbor, and diagnostic helpers for fast topology clustering.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "cluster_draft/FastClusteringDraft.hh"

namespace icts {
struct ClusterConfig;
}  // namespace icts

namespace icts::fast_clustering {
namespace {

struct NeighborCandidate
{
  std::size_t cluster_id = 0;
  long long distance = 0;
};

struct BucketCoordinate
{
  std::size_t x = 0U;
  std::size_t y = 0U;
};

struct ClusterSpatialPoint
{
  std::size_t cluster_id = 0U;
  long long x = 0;
  long long y = 0;
};

auto IsBetterNeighbor(const NeighborCandidate& lhs, const NeighborCandidate& rhs) -> bool
{
  if (lhs.distance != rhs.distance) {
    return lhs.distance < rhs.distance;
  }
  return lhs.cluster_id < rhs.cluster_id;
}

auto InsertBoundedNeighbor(std::vector<NeighborCandidate>& candidates, NeighborCandidate candidate, std::size_t max_candidate_count) -> void
{
  const auto insertion_point = std::ranges::upper_bound(candidates, candidate, IsBetterNeighbor);
  if (candidates.size() < max_candidate_count) {
    candidates.insert(insertion_point, candidate);
  } else if (insertion_point != candidates.end()) {
    candidates.insert(insertion_point, candidate);
    candidates.pop_back();
  }
}

auto BuildNeighborIdsFromCandidates(const std::vector<NeighborCandidate>& candidates) -> std::vector<std::size_t>
{
  std::vector<std::size_t> neighbor_ids;
  neighbor_ids.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    neighbor_ids.push_back(candidate.cluster_id);
  }
  return neighbor_ids;
}

auto CalcBoundsCenterX(const Bounds& bounds) -> long long
{
  return (static_cast<long long>(bounds.min_x) + static_cast<long long>(bounds.max_x)) / 2LL;
}

auto CalcBoundsCenterY(const Bounds& bounds) -> long long
{
  return (static_cast<long long>(bounds.min_y) + static_cast<long long>(bounds.max_y)) / 2LL;
}

auto ResolveBucketCount(std::size_t active_count, std::size_t max_candidate_count, long long span_x, long long span_y)
    -> std::pair<std::size_t, std::size_t>
{
  const auto target_bucket_load = static_cast<double>(std::max<std::size_t>(max_candidate_count, 8U));
  const auto target_bucket_count = std::max(1.0, static_cast<double>(active_count) / target_bucket_load);
  const auto safe_span_x = static_cast<double>(std::max<long long>(span_x, 1LL));
  const auto safe_span_y = static_cast<double>(std::max<long long>(span_y, 1LL));
  const auto aspect = safe_span_x / safe_span_y;
  auto x_count = static_cast<std::size_t>(std::ceil(std::sqrt(target_bucket_count * aspect)));
  x_count = std::clamp<std::size_t>(x_count, 1U, std::max<std::size_t>(1U, active_count));
  auto y_count = static_cast<std::size_t>(std::ceil(target_bucket_count / static_cast<double>(x_count)));
  y_count = std::clamp<std::size_t>(y_count, 1U, std::max<std::size_t>(1U, active_count));
  return {x_count, y_count};
}

auto ResolveBucketCoordinate(const ClusterSpatialPoint& point, long long min_x, long long min_y, long long span_x, long long span_y,
                             std::size_t x_bucket_count, std::size_t y_bucket_count) -> BucketCoordinate
{
  auto resolve_axis = [](long long value, long long min_value, long long span, std::size_t bucket_count) -> std::size_t {
    if (bucket_count <= 1U || span <= 0LL) {
      return 0U;
    }
    const auto normalized = static_cast<double>(value - min_value) / static_cast<double>(span);
    auto bucket = static_cast<std::size_t>(std::floor(normalized * static_cast<double>(bucket_count)));
    return std::min(bucket, bucket_count - 1U);
  };
  return {
      .x = resolve_axis(point.x, min_x, span_x, x_bucket_count),
      .y = resolve_axis(point.y, min_y, span_y, y_bucket_count),
  };
}

auto BucketIndex(std::size_t x, std::size_t y, std::size_t y_bucket_count) -> std::size_t
{
  return x * y_bucket_count + y;
}

auto SelectNearestFromSpatialCandidates(std::size_t cluster_id, const std::vector<ClusterDraft>& clusters,
                                        const std::vector<std::size_t>& candidate_ids, std::size_t max_candidate_count)
    -> std::vector<std::size_t>
{
  std::vector<NeighborCandidate> candidates;
  candidates.reserve(max_candidate_count);
  const auto& cluster = clusters.at(cluster_id);
  for (const auto neighbor_id : candidate_ids) {
    if (neighbor_id == cluster_id || !clusters.at(neighbor_id).active || clusters.at(neighbor_id).entry_ids.empty()) {
      continue;
    }
    InsertBoundedNeighbor(candidates,
                          NeighborCandidate{
                              .cluster_id = neighbor_id,
                              .distance = CalcBoundsDistance(cluster.bounds, clusters.at(neighbor_id).bounds),
                          },
                          max_candidate_count);
  }
  return BuildNeighborIdsFromCandidates(candidates);
}

}  // namespace

auto ElapsedSeconds(TimePoint start) -> double
{
  return std::chrono::duration<double>(SteadyClock::now() - start).count();
}

auto FormatSeconds(double seconds) -> std::string
{
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(3) << seconds;
  return stream.str();
}

auto FormatRatio(double value) -> std::string
{
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(3) << value;
  return stream.str();
}

auto BuildMergedDraft(const ClusterDraft& lhs, const ClusterDraft& rhs, const std::vector<LoadEntry>& entries, const ClusterConfig& config)
    -> ClusterDraft
{
  std::vector<std::size_t> merged_entry_ids;
  merged_entry_ids.reserve(lhs.entry_ids.size() + rhs.entry_ids.size());
  merged_entry_ids.insert(merged_entry_ids.end(), lhs.entry_ids.begin(), lhs.entry_ids.end());
  merged_entry_ids.insert(merged_entry_ids.end(), rhs.entry_ids.begin(), rhs.entry_ids.end());
  return BuildDraft(std::move(merged_entry_ids), entries, config);
}

auto SelectNearestActiveNeighbors(std::size_t cluster_id, const std::vector<ClusterDraft>& clusters, std::size_t max_candidate_count)
    -> std::vector<std::size_t>
{
  if (max_candidate_count == 0U) {
    return {};
  }

  std::vector<NeighborCandidate> candidates;
  candidates.reserve(max_candidate_count);
  const auto& cluster = clusters.at(cluster_id);
  for (std::size_t neighbor_id = 0; neighbor_id < clusters.size(); ++neighbor_id) {
    if (neighbor_id == cluster_id || !clusters.at(neighbor_id).active || clusters.at(neighbor_id).entry_ids.empty()) {
      continue;
    }
    const NeighborCandidate candidate{
        .cluster_id = neighbor_id,
        .distance = CalcBoundsDistance(cluster.bounds, clusters.at(neighbor_id).bounds),
    };
    InsertBoundedNeighbor(candidates, candidate, max_candidate_count);
  }

  return BuildNeighborIdsFromCandidates(candidates);
}

auto BuildSpatialNeighborGraph(const std::vector<ClusterDraft>& clusters, std::size_t max_candidate_count) -> NeighborGraph
{
  NeighborGraph graph;
  graph.neighbor_ids.resize(clusters.size());
  if (max_candidate_count == 0U || clusters.empty()) {
    return graph;
  }

  std::vector<ClusterSpatialPoint> active_points;
  active_points.reserve(clusters.size());
  long long min_x = std::numeric_limits<long long>::max();
  long long min_y = std::numeric_limits<long long>::max();
  long long max_x = std::numeric_limits<long long>::min();
  long long max_y = std::numeric_limits<long long>::min();
  for (std::size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
    const auto& cluster = clusters.at(cluster_id);
    if (!cluster.active || cluster.entry_ids.empty() || IsEmpty(cluster.bounds)) {
      continue;
    }
    const auto x = CalcBoundsCenterX(cluster.bounds);
    const auto y = CalcBoundsCenterY(cluster.bounds);
    active_points.push_back(ClusterSpatialPoint{
        .cluster_id = cluster_id,
        .x = x,
        .y = y,
    });
    min_x = std::min(min_x, x);
    min_y = std::min(min_y, y);
    max_x = std::max(max_x, x);
    max_y = std::max(max_y, y);
  }

  if (active_points.size() <= 1U) {
    return graph;
  }

  const auto span_x = std::max(1LL, max_x - min_x);
  const auto span_y = std::max(1LL, max_y - min_y);
  const auto [x_bucket_count, y_bucket_count] = ResolveBucketCount(active_points.size(), max_candidate_count, span_x, span_y);
  std::vector<std::vector<std::size_t>> buckets(x_bucket_count * y_bucket_count);
  std::vector<BucketCoordinate> cluster_buckets(clusters.size());
  for (const auto& point : active_points) {
    const auto coordinate = ResolveBucketCoordinate(point, min_x, min_y, span_x, span_y, x_bucket_count, y_bucket_count);
    cluster_buckets.at(point.cluster_id) = coordinate;
    buckets.at(BucketIndex(coordinate.x, coordinate.y, y_bucket_count)).push_back(point.cluster_id);
  }

  const auto max_radius = std::max(x_bucket_count, y_bucket_count);
  const auto target_candidate_count = std::max<std::size_t>(max_candidate_count * 4U, max_candidate_count);
  for (const auto& point : active_points) {
    const auto coordinate = cluster_buckets.at(point.cluster_id);
    std::vector<std::size_t> candidate_ids;
    for (std::size_t radius = 0U; radius <= max_radius; ++radius) {
      const auto min_bucket_x = coordinate.x > radius ? coordinate.x - radius : 0U;
      const auto max_bucket_x = std::min(x_bucket_count - 1U, coordinate.x + radius);
      const auto min_bucket_y = coordinate.y > radius ? coordinate.y - radius : 0U;
      const auto max_bucket_y = std::min(y_bucket_count - 1U, coordinate.y + radius);
      candidate_ids.clear();
      for (std::size_t bucket_x = min_bucket_x; bucket_x <= max_bucket_x; ++bucket_x) {
        for (std::size_t bucket_y = min_bucket_y; bucket_y <= max_bucket_y; ++bucket_y) {
          const auto& bucket = buckets.at(BucketIndex(bucket_x, bucket_y, y_bucket_count));
          candidate_ids.insert(candidate_ids.end(), bucket.begin(), bucket.end());
        }
      }
      if (candidate_ids.size() > target_candidate_count || radius == max_radius) {
        break;
      }
    }
    auto neighbor_ids = SelectNearestFromSpatialCandidates(point.cluster_id, clusters, candidate_ids, max_candidate_count);
    if (neighbor_ids.size() < max_candidate_count && active_points.size() > max_candidate_count) {
      neighbor_ids = SelectNearestActiveNeighbors(point.cluster_id, clusters, max_candidate_count);
    }
    graph.neighbor_ids.at(point.cluster_id) = std::move(neighbor_ids);
  }
  return graph;
}

auto SelectNearestActiveNeighbors(std::size_t cluster_id, const std::vector<ClusterDraft>& clusters, const NeighborGraph* neighbor_graph,
                                  std::size_t max_candidate_count) -> std::vector<std::size_t>
{
  if (neighbor_graph != nullptr && cluster_id < neighbor_graph->neighbor_ids.size()) {
    std::vector<std::size_t> active_neighbor_ids;
    active_neighbor_ids.reserve(std::min(max_candidate_count, neighbor_graph->neighbor_ids.at(cluster_id).size()));
    for (const auto neighbor_id : neighbor_graph->neighbor_ids.at(cluster_id)) {
      if (active_neighbor_ids.size() == max_candidate_count) {
        break;
      }
      if (neighbor_id < clusters.size() && neighbor_id != cluster_id && clusters.at(neighbor_id).active
          && !clusters.at(neighbor_id).entry_ids.empty()) {
        active_neighbor_ids.push_back(neighbor_id);
      }
    }
    if (!active_neighbor_ids.empty()) {
      return active_neighbor_ids;
    }
  }
  return SelectNearestActiveNeighbors(cluster_id, clusters, max_candidate_count);
}

auto PairObjective(const ClusterDraft& lhs, const ClusterDraft& rhs, const ClusterConfig& config, double target_routing_cap_proxy,
                   double routing_cap_balance_weight) -> double
{
  return DraftObjective(lhs, config, target_routing_cap_proxy, routing_cap_balance_weight)
         + DraftObjective(rhs, config, target_routing_cap_proxy, routing_cap_balance_weight);
}

}  // namespace icts::fast_clustering
