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
 * @file FastClusteringPolish.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Merge and boundary polishing orchestration for fast topology clustering.
 */

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <ostream>
#include <ranges>
#include <string>
#include <vector>

#include "Log.hh"
#include "cluster_draft/FastClusteringDraft.hh"

namespace icts {
struct ClusterConfig;
}  // namespace icts

namespace icts::fast_clustering {
namespace {

auto CountActiveClusters(const std::vector<ClusterDraft>& clusters) -> std::size_t
{
  std::size_t count = 0U;
  for (const auto& cluster : clusters) {
    if (cluster.active && !cluster.entry_ids.empty()) {
      ++count;
    }
  }
  return count;
}

auto ResolveMergeUtilizationThreshold(std::size_t fanout_limit) -> std::size_t
{
  if (fanout_limit <= 1U) {
    return 1U;
  }
  return std::max<std::size_t>(2U, (fanout_limit * 3U + 3U) / 4U);
}

auto ShouldAttemptMerge(const ClusterDraft& cluster, std::size_t fanout_limit) -> bool
{
  if (!cluster.active || cluster.entry_ids.empty()) {
    return false;
  }
  if (cluster.entry_ids.size() <= 1U) {
    return true;
  }
  return cluster.entry_ids.size() < ResolveMergeUtilizationThreshold(fanout_limit);
}

}  // namespace

auto PolishSmallClusters(std::vector<ClusterDraft>& clusters, const std::vector<LoadEntry>& entries, const ClusterConfig& config) -> void
{
  const auto polish_start = SteadyClock::now();
  double merge_elapsed_seconds = 0.0;
  std::size_t total_attempted_clusters = 0U;
  std::size_t total_merges = 0U;
  const auto fanout_limit = ResolvePackingFanoutLimit(config, entries.size());
  for (std::size_t round = 0; round < kMergeRoundCount; ++round) {
    const auto round_start = SteadyClock::now();
    const auto active_before = CountActiveClusters(clusters);
    bool changed = false;
    std::size_t attempted_clusters = 0U;
    std::size_t round_merges = 0U;
    const auto neighbor_graph = BuildSpatialNeighborGraph(clusters, kMaxMergeNeighborCandidates);
    std::vector<std::size_t> cluster_order(clusters.size());
    std::iota(cluster_order.begin(), cluster_order.end(), 0U);
    std::ranges::sort(cluster_order, [&clusters](std::size_t lhs, std::size_t rhs) -> bool {
      if (clusters.at(lhs).entry_ids.size() != clusters.at(rhs).entry_ids.size()) {
        return clusters.at(lhs).entry_ids.size() < clusters.at(rhs).entry_ids.size();
      }
      return lhs < rhs;
    });

    for (const auto cluster_id : cluster_order) {
      if (!ShouldAttemptMerge(clusters.at(cluster_id), fanout_limit)) {
        continue;
      }
      ++attempted_clusters;
      const bool merged = MergeDraftsIfUseful(cluster_id, clusters, entries, config, &neighbor_graph);
      if (merged) {
        ++round_merges;
      }
      changed = merged || changed;
    }
    const auto round_elapsed_seconds = ElapsedSeconds(round_start);
    merge_elapsed_seconds += round_elapsed_seconds;
    total_attempted_clusters += attempted_clusters;
    total_merges += round_merges;
    LOG_INFO << "Fast clustering merge polish round " << round << ": active_before=" << active_before
             << ", active_after=" << CountActiveClusters(clusters) << ", attempted_clusters=" << attempted_clusters
             << ", merges=" << round_merges << ", changed=" << (changed ? "true" : "false")
             << ", elapsed_time=" << FormatSeconds(round_elapsed_seconds) << " s";
    if (!changed) {
      break;
    }
  }

  const auto boundary_start = SteadyClock::now();
  const auto active_before_boundary = CountActiveClusters(clusters);
  PolishBoundaryLoads(clusters, entries, config);
  const auto boundary_elapsed_seconds = ElapsedSeconds(boundary_start);
  LOG_INFO << "Fast clustering boundary polish: active_before=" << active_before_boundary
           << ", active_after=" << CountActiveClusters(clusters) << ", elapsed_time=" << FormatSeconds(boundary_elapsed_seconds) << " s";

  const auto inactive_tail
      = std::ranges::remove_if(clusters, [](const ClusterDraft& cluster) -> bool { return !cluster.active || cluster.entry_ids.empty(); });
  clusters.erase(inactive_tail.begin(), inactive_tail.end());
  LOG_INFO << "Fast clustering polish done: active_clusters=" << clusters.size()
           << ", merge_attempted_clusters=" << total_attempted_clusters << ", total_merges=" << total_merges
           << ", merge_elapsed_time=" << FormatSeconds(merge_elapsed_seconds)
           << " s, boundary_elapsed_time=" << FormatSeconds(boundary_elapsed_seconds)
           << " s, total_elapsed_time=" << FormatSeconds(ElapsedSeconds(polish_start)) << " s";
}

}  // namespace icts::fast_clustering
