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
 * @file FastClusteringBoundaryPolish.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Boundary-load polish orchestration for fast topology clustering.
 */

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "Log.hh"
#include "cluster_draft/FastClusteringDraft.hh"

namespace icts {
struct ClusterConfig;
}  // namespace icts

namespace icts::fast_clustering {
namespace {

struct BoundarySourceOrder
{
  std::vector<std::size_t> cluster_ids;
  std::size_t active_count = 0U;
  double mean_proxy = 0.0;
  double stddev_proxy = 0.0;
  double threshold_proxy = 0.0;
};

auto BuildCapHeavyClusterOrder(const std::vector<ClusterDraft>& clusters) -> BoundarySourceOrder
{
  BoundarySourceOrder order;
  std::vector<std::size_t> cluster_order(clusters.size());
  std::iota(cluster_order.begin(), cluster_order.end(), 0U);
  double proxy_sum = 0.0;
  for (const auto& cluster : clusters) {
    if (!cluster.active || cluster.entry_ids.size() <= 1U) {
      continue;
    }
    ++order.active_count;
    proxy_sum += cluster.routing_cap_proxy;
  }
  if (order.active_count == 0U) {
    return order;
  }
  order.mean_proxy = proxy_sum / static_cast<double>(order.active_count);
  double squared_delta_sum = 0.0;
  for (const auto& cluster : clusters) {
    if (!cluster.active || cluster.entry_ids.size() <= 1U) {
      continue;
    }
    const auto delta = cluster.routing_cap_proxy - order.mean_proxy;
    squared_delta_sum += delta * delta;
  }
  order.stddev_proxy = std::sqrt(squared_delta_sum / static_cast<double>(order.active_count));
  order.threshold_proxy = order.mean_proxy + kBoundaryHeavyStddevFactor * order.stddev_proxy;

  std::ranges::sort(cluster_order, [&clusters](std::size_t lhs, std::size_t rhs) -> bool {
    if (std::abs(clusters.at(lhs).routing_cap_proxy - clusters.at(rhs).routing_cap_proxy) > kScoreEpsilon) {
      return clusters.at(lhs).routing_cap_proxy > clusters.at(rhs).routing_cap_proxy;
    }
    return lhs < rhs;
  });

  const auto max_source_count = std::max<std::size_t>(
      1U, static_cast<std::size_t>(std::ceil(static_cast<double>(order.active_count) * kBoundaryMaxSourceFraction)));
  order.cluster_ids.reserve(max_source_count);
  for (const auto cluster_id : cluster_order) {
    if (cluster_id >= clusters.size() || !clusters.at(cluster_id).active || clusters.at(cluster_id).entry_ids.size() <= 1U) {
      continue;
    }
    if (clusters.at(cluster_id).routing_cap_proxy + kScoreEpsilon < order.threshold_proxy) {
      break;
    }
    order.cluster_ids.push_back(cluster_id);
    if (order.cluster_ids.size() == max_source_count) {
      break;
    }
  }
  return order;
}

auto ResolveBoundaryMinMoveCount(std::size_t considered_sources) -> std::size_t
{
  return std::max<std::size_t>(16U, considered_sources / 1000U);
}

}  // namespace

auto PolishBoundaryLoads(std::vector<ClusterDraft>& clusters, const std::vector<LoadEntry>& entries, const ClusterConfig& config) -> void
{
  for (std::size_t round = 0; round < kBoundaryPolishRoundCount; ++round) {
    const auto round_start = SteadyClock::now();
    bool changed = false;
    const auto order_start = SteadyClock::now();
    const auto source_order = BuildCapHeavyClusterOrder(clusters);
    const auto neighbor_graph = BuildSpatialNeighborGraph(clusters, kMaxBoundaryNeighborCandidates);
    const auto order_elapsed_seconds = ElapsedSeconds(order_start);
    const auto search_start = SteadyClock::now();
    std::size_t considered_sources = 0U;
    std::size_t moved_loads = 0U;
    auto aggregate = CalcDraftAggregate(clusters);
    for (const auto source_id : source_order.cluster_ids) {
      if (source_id >= clusters.size() || !clusters.at(source_id).active || clusters.at(source_id).entry_ids.size() <= 1U) {
        continue;
      }

      ++considered_sources;
      auto best_move = FindBestBoundaryMove(source_id, clusters, entries, config, aggregate, &neighbor_graph);
      if (!best_move.has_value()) {
        continue;
      }

      aggregate.total_routing_cap_proxy += best_move->source_after.routing_cap_proxy + best_move->target_after.routing_cap_proxy
                                           - clusters.at(source_id).routing_cap_proxy - clusters.at(best_move->target_id).routing_cap_proxy;
      clusters.at(source_id) = std::move(best_move->source_after);
      clusters.at(best_move->target_id) = std::move(best_move->target_after);
      ++moved_loads;
      changed = true;
    }
    const auto search_elapsed_seconds = ElapsedSeconds(search_start);
    const auto min_move_count = ResolveBoundaryMinMoveCount(considered_sources);
    LOG_INFO << "Fast clustering boundary polish round " << round << ": active_sources=" << source_order.active_count
             << ", selected_sources=" << source_order.cluster_ids.size() << ", considered_sources=" << considered_sources
             << ", moved_loads=" << moved_loads << ", min_moves_for_next_round=" << min_move_count
             << ", changed=" << (changed ? "true" : "false") << ", order_elapsed_time=" << FormatSeconds(order_elapsed_seconds)
             << " s, search_elapsed_time=" << FormatSeconds(search_elapsed_seconds)
             << " s, elapsed_time=" << FormatSeconds(ElapsedSeconds(round_start)) << " s";
    if (!changed || moved_loads < min_move_count) {
      break;
    }
  }
}

}  // namespace icts::fast_clustering
