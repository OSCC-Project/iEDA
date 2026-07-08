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
 * @file FastClusteringDraft.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Fast sink-load clustering draft data and helper declarations.
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "Clustering.hh"
#include "Point.hh"
#include "TopologyConfig.hh"

namespace icts::fast_clustering {

using SteadyClock = std::chrono::steady_clock;
using TimePoint = SteadyClock::time_point;

inline constexpr std::size_t kDefaultPackingFanout = 32;
inline constexpr std::size_t kMergeRoundCount = 2;
inline constexpr std::size_t kSplitCandidateWindow = 4;
inline constexpr std::size_t kMaxMergeNeighborCandidates = 12;
inline constexpr std::size_t kBoundaryPolishRoundCount = 2;
inline constexpr std::size_t kMaxBoundaryNeighborCandidates = 8;
inline constexpr std::size_t kMaxBoundaryEntryCandidates = 8;
inline constexpr double kSplitRoutingCapBalanceWeight = 0.25;
inline constexpr double kSplitUtilizationBalanceWeight = 2.0;
inline constexpr double kMergeRoutingCapBalanceWeight = 0.18;
inline constexpr double kBoundaryRoutingCapBalanceWeight = 0.35;
inline constexpr double kBoundaryHeavyStddevFactor = 0.25;
inline constexpr double kBoundaryMaxSourceFraction = 0.35;
inline constexpr double kScoreEpsilon = 1e-9;

struct LoadEntry
{
  Pin* pin = nullptr;
  Point<int> location = Point<int>(0, 0);
  std::size_t original_index = 0;
};

struct Bounds
{
  int min_x = std::numeric_limits<int>::max();
  int min_y = std::numeric_limits<int>::max();
  int max_x = std::numeric_limits<int>::min();
  int max_y = std::numeric_limits<int>::min();
};

struct ClusterDraft
{
  std::vector<std::size_t> entry_ids;
  Bounds bounds;
  double routing_cap_proxy = 0.0;
  bool active = true;
};

struct DraftAggregate
{
  std::size_t active_count = 0;
  double total_routing_cap_proxy = 0.0;
};

struct BoundaryMove
{
  std::size_t target_id = 0;
  ClusterDraft source_after;
  ClusterDraft target_after;
  double score_delta = 0.0;
};

struct NeighborGraph
{
  std::vector<std::vector<std::size_t>> neighbor_ids;
};

auto ElapsedSeconds(TimePoint start) -> double;
auto FormatSeconds(double seconds) -> std::string;
auto FormatRatio(double value) -> std::string;
auto IsEmpty(const Bounds& bounds) -> bool;
auto ExtendBounds(Bounds bounds, const Point<int>& point) -> Bounds;
auto CalcDiameter(const Bounds& bounds) -> int;
auto CalcClusterBounds(const std::vector<Pin*>& cluster) -> Bounds;
auto CalcClusterBounds(const std::vector<std::size_t>& entry_ids, const std::vector<LoadEntry>& entries) -> Bounds;
auto CalcManhattanDistance(const Point<int>& lhs, const Point<int>& rhs) -> double;
auto ResolveDraftRoot(const std::vector<std::size_t>& entry_ids, const std::vector<LoadEntry>& entries, const ClusterConfig& config)
    -> Point<int>;
auto BuildDraft(std::vector<std::size_t> entry_ids, const std::vector<LoadEntry>& entries, const ClusterConfig& config) -> ClusterDraft;
auto CalcCenter(const std::vector<Pin*>& cluster) -> Point<int>;
auto ResolveEvaluationRoot(const std::vector<Pin*>& cluster, const ClusterConfig& config) -> Point<int>;
auto ResolvePackingFanoutLimit(const ClusterConfig& config, std::size_t load_count) -> std::size_t;
auto IsFanoutLegal(std::size_t fanout, const ClusterConfig& config) -> bool;
auto IsDiameterLegal(const Bounds& bounds, const ClusterConfig& config) -> bool;
auto IsDraftGeometryLegal(const ClusterDraft& draft, const ClusterConfig& config) -> bool;
auto ClusterScoreProxy(const ClusterDraft& draft, const ClusterConfig& config) -> double;
auto CalcRoutingCapVariancePenalty(double routing_cap_proxy, double target_routing_cap_proxy) -> double;
auto DraftObjective(const ClusterDraft& draft, const ClusterConfig& config, double target_routing_cap_proxy,
                    double routing_cap_balance_weight) -> double;
auto CalcDraftAggregate(const std::vector<ClusterDraft>& drafts) -> DraftAggregate;
auto CalcMeanRoutingCapProxy(const DraftAggregate& aggregate) -> double;
auto CalcBoundsDistance(const Bounds& lhs, const Bounds& rhs) -> long long;
auto DistanceToBounds(const Point<int>& point, const Bounds& bounds) -> long long;
auto BuildMergedDraft(const ClusterDraft& lhs, const ClusterDraft& rhs, const std::vector<LoadEntry>& entries, const ClusterConfig& config)
    -> ClusterDraft;
auto SelectNearestActiveNeighbors(std::size_t cluster_id, const std::vector<ClusterDraft>& clusters, std::size_t max_candidate_count)
    -> std::vector<std::size_t>;
auto BuildSpatialNeighborGraph(const std::vector<ClusterDraft>& clusters, std::size_t max_candidate_count) -> NeighborGraph;
auto SelectNearestActiveNeighbors(std::size_t cluster_id, const std::vector<ClusterDraft>& clusters, const NeighborGraph* neighbor_graph,
                                  std::size_t max_candidate_count) -> std::vector<std::size_t>;
auto PairObjective(const ClusterDraft& lhs, const ClusterDraft& rhs, const ClusterConfig& config, double target_routing_cap_proxy,
                   double routing_cap_balance_weight) -> double;
auto CollectEntries(const std::vector<Pin*>& loads) -> std::vector<LoadEntry>;
auto BuildSpatialRecursiveClusters(const std::vector<LoadEntry>& entries, const ClusterConfig& config) -> std::vector<ClusterDraft>;
auto TryBuildDraftAfterMove(const ClusterDraft& source, const ClusterDraft& target, std::size_t moved_entry_id,
                            const std::vector<LoadEntry>& entries, const ClusterConfig& config, ClusterDraft& source_after,
                            ClusterDraft& target_after) -> bool;
auto BuildBoundaryEntryCandidates(const ClusterDraft& source, const ClusterDraft& target, const std::vector<LoadEntry>& entries,
                                  const ClusterConfig& config) -> std::vector<std::size_t>;
auto FindBestBoundaryMove(std::size_t source_id, const std::vector<ClusterDraft>& clusters, const std::vector<LoadEntry>& entries,
                          const ClusterConfig& config, const DraftAggregate& aggregate, const NeighborGraph* neighbor_graph)
    -> std::optional<BoundaryMove>;
auto MergeDraftsIfUseful(std::size_t cluster_id, std::vector<ClusterDraft>& clusters, const std::vector<LoadEntry>& entries,
                         const ClusterConfig& config, const NeighborGraph* neighbor_graph) -> bool;
auto PolishBoundaryLoads(std::vector<ClusterDraft>& clusters, const std::vector<LoadEntry>& entries, const ClusterConfig& config) -> void;
auto PolishSmallClusters(std::vector<ClusterDraft>& clusters, const std::vector<LoadEntry>& entries, const ClusterConfig& config) -> void;
auto FinalizeClusters(const std::vector<ClusterDraft>& drafts, const std::vector<LoadEntry>& entries, const ClusterConfig& config)
    -> std::optional<ClusterOutput>;
auto CountAssignedLoads(const ClusterOutput& result) -> std::size_t;

}  // namespace icts::fast_clustering
