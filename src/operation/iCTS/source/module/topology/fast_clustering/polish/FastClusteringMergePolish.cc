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
 * @file FastClusteringMergePolish.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Small-cluster merge polishing for fast topology clustering.
 */

#include <cstddef>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "cluster_draft/FastClusteringDraft.hh"

namespace icts {
struct ClusterConfig;
}  // namespace icts

namespace icts::fast_clustering {
namespace {

auto CanUseMergedDraft(const ClusterDraft& lhs, const ClusterDraft& rhs, const ClusterDraft& merged, const ClusterConfig& config) -> bool
{
  return lhs.active && rhs.active && !lhs.entry_ids.empty() && !rhs.entry_ids.empty() && IsDraftGeometryLegal(merged, config);
}

auto MergeDrafts(ClusterDraft& target, ClusterDraft& source, const std::vector<LoadEntry>& entries, const ClusterConfig& config) -> void
{
  auto merged = BuildMergedDraft(target, source, entries, config);
  target.entry_ids = std::move(merged.entry_ids);
  target.bounds = merged.bounds;
  target.routing_cap_proxy = merged.routing_cap_proxy;
  target.active = true;
  source.entry_ids.clear();
  source.routing_cap_proxy = 0.0;
  source.active = false;
}

}  // namespace

auto MergeDraftsIfUseful(std::size_t cluster_id, std::vector<ClusterDraft>& clusters, const std::vector<LoadEntry>& entries,
                         const ClusterConfig& config, const NeighborGraph* neighbor_graph) -> bool
{
  const auto aggregate = CalcDraftAggregate(clusters);
  const auto before_target_proxy = CalcMeanRoutingCapProxy(aggregate);
  const auto& cluster = clusters.at(cluster_id);
  const auto neighbor_ids = SelectNearestActiveNeighbors(cluster_id, clusters, neighbor_graph, kMaxMergeNeighborCandidates);

  std::optional<std::size_t> best_neighbor;
  double best_delta = std::numeric_limits<double>::infinity();
  for (const auto neighbor_id : neighbor_ids) {
    const auto& neighbor = clusters.at(neighbor_id);
    const auto merged = BuildMergedDraft(neighbor, cluster, entries, config);
    if (!CanUseMergedDraft(cluster, neighbor, merged, config)) {
      continue;
    }

    const auto after_count = aggregate.active_count > 1U ? aggregate.active_count - 1U : aggregate.active_count;
    const auto after_total_proxy
        = aggregate.total_routing_cap_proxy - cluster.routing_cap_proxy - neighbor.routing_cap_proxy + merged.routing_cap_proxy;
    const auto after_target_proxy = after_count == 0U ? 0.0 : after_total_proxy / static_cast<double>(after_count);
    const auto separate_score = PairObjective(cluster, neighbor, config, before_target_proxy, kMergeRoutingCapBalanceWeight);
    const auto merged_score = DraftObjective(merged, config, after_target_proxy, kMergeRoutingCapBalanceWeight);
    const auto forced_small_merge = cluster.entry_ids.size() == 1U || neighbor.entry_ids.size() == 1U;
    if (!forced_small_merge && merged_score > separate_score + kScoreEpsilon) {
      continue;
    }

    const auto score_delta = merged_score - separate_score;
    if (score_delta < best_delta) {
      best_delta = score_delta;
      best_neighbor = neighbor_id;
    }
  }

  if (!best_neighbor.has_value()) {
    return false;
  }

  MergeDrafts(clusters.at(*best_neighbor), clusters.at(cluster_id), entries, config);
  return true;
}

}  // namespace icts::fast_clustering
