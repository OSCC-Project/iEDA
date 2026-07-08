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
 * @file FastClusteringBoundarySearch.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Boundary-load move search for fast topology clustering.
 */

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "cluster_draft/FastClusteringDraft.hh"

namespace icts {
struct ClusterConfig;
}  // namespace icts

namespace icts::fast_clustering {
namespace {

auto IsBoundaryTargetUseful(const ClusterDraft& target, double target_routing_cap_proxy) -> bool
{
  return target.active && !target.entry_ids.empty() && target.routing_cap_proxy < target_routing_cap_proxy;
}

auto CalcAfterTargetRoutingCapProxy(const DraftAggregate& aggregate, const ClusterDraft& source, const ClusterDraft& target,
                                    const ClusterDraft& source_after, const ClusterDraft& target_after) -> double
{
  const auto after_total_routing_cap_proxy = aggregate.total_routing_cap_proxy - source.routing_cap_proxy - target.routing_cap_proxy
                                             + source_after.routing_cap_proxy + target_after.routing_cap_proxy;
  return aggregate.active_count == 0U ? 0.0 : after_total_routing_cap_proxy / static_cast<double>(aggregate.active_count);
}

auto BuildBoundaryMoveIfImproves(std::size_t target_id, const ClusterDraft& source, const ClusterDraft& target, std::size_t moved_entry_id,
                                 const std::vector<LoadEntry>& entries, const ClusterConfig& config, const DraftAggregate& aggregate,
                                 double target_routing_cap_proxy) -> std::optional<BoundaryMove>
{
  ClusterDraft source_after;
  ClusterDraft target_after;
  if (!TryBuildDraftAfterMove(source, target, moved_entry_id, entries, config, source_after, target_after)) {
    return std::nullopt;
  }

  const auto before_score = PairObjective(source, target, config, target_routing_cap_proxy, kBoundaryRoutingCapBalanceWeight);
  const auto after_target_routing_cap_proxy = CalcAfterTargetRoutingCapProxy(aggregate, source, target, source_after, target_after);
  const auto after_score
      = PairObjective(source_after, target_after, config, after_target_routing_cap_proxy, kBoundaryRoutingCapBalanceWeight);
  const auto score_delta = after_score - before_score;
  if (score_delta + kScoreEpsilon >= 0.0) {
    return std::nullopt;
  }

  return BoundaryMove{
      .target_id = target_id,
      .source_after = std::move(source_after),
      .target_after = std::move(target_after),
      .score_delta = score_delta,
  };
}

auto SelectBetterBoundaryMove(std::optional<BoundaryMove> best_move, BoundaryMove candidate) -> std::optional<BoundaryMove>
{
  if (!best_move.has_value() || candidate.score_delta < best_move->score_delta) {
    return candidate;
  }
  return best_move;
}

}  // namespace

auto FindBestBoundaryMove(std::size_t source_id, const std::vector<ClusterDraft>& clusters, const std::vector<LoadEntry>& entries,
                          const ClusterConfig& config, const DraftAggregate& aggregate, const NeighborGraph* neighbor_graph)
    -> std::optional<BoundaryMove>
{
  const auto& source = clusters.at(source_id);
  const auto target_routing_cap_proxy = CalcMeanRoutingCapProxy(aggregate);
  if (!source.active || source.entry_ids.size() <= 1U || source.routing_cap_proxy <= target_routing_cap_proxy) {
    return std::nullopt;
  }

  std::optional<BoundaryMove> best_move;
  const auto neighbor_ids = SelectNearestActiveNeighbors(source_id, clusters, neighbor_graph, kMaxBoundaryNeighborCandidates);
  for (const auto target_id : neighbor_ids) {
    const auto& target = clusters.at(target_id);
    if (!IsBoundaryTargetUseful(target, target_routing_cap_proxy)) {
      continue;
    }

    const auto entry_candidates = BuildBoundaryEntryCandidates(source, target, entries, config);
    for (const auto moved_entry_id : entry_candidates) {
      auto candidate
          = BuildBoundaryMoveIfImproves(target_id, source, target, moved_entry_id, entries, config, aggregate, target_routing_cap_proxy);
      if (candidate.has_value()) {
        best_move = SelectBetterBoundaryMove(std::move(best_move), std::move(*candidate));
      }
    }
  }

  return best_move;
}

}  // namespace icts::fast_clustering
