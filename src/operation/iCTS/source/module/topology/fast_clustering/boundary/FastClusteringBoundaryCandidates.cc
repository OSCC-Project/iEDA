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
 * @file FastClusteringBoundaryCandidates.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Boundary-load candidate generation for fast topology clustering.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "Point.hh"
#include "cluster_draft/FastClusteringDraft.hh"

namespace icts {
struct ClusterConfig;
}  // namespace icts

namespace icts::fast_clustering {

auto TryBuildDraftAfterMove(const ClusterDraft& source, const ClusterDraft& target, std::size_t moved_entry_id,
                            const std::vector<LoadEntry>& entries, const ClusterConfig& config, ClusterDraft& source_after,
                            ClusterDraft& target_after) -> bool
{
  if (source.entry_ids.size() <= 1U) {
    return false;
  }

  bool found_source_entry = false;
  std::vector<std::size_t> source_ids;
  source_ids.reserve(source.entry_ids.size() - 1U);
  for (const auto entry_id : source.entry_ids) {
    if (entry_id == moved_entry_id) {
      found_source_entry = true;
      continue;
    }
    source_ids.push_back(entry_id);
  }
  if (!found_source_entry || source_ids.empty()) {
    return false;
  }

  std::vector<std::size_t> target_ids = target.entry_ids;
  target_ids.push_back(moved_entry_id);
  source_after = BuildDraft(std::move(source_ids), entries, config);
  target_after = BuildDraft(std::move(target_ids), entries, config);
  return IsDraftGeometryLegal(source_after, config) && IsDraftGeometryLegal(target_after, config);
}

auto BuildBoundaryEntryCandidates(const ClusterDraft& source, const ClusterDraft& target, const std::vector<LoadEntry>& entries,
                                  const ClusterConfig& config) -> std::vector<std::size_t>
{
  struct EntryCandidate
  {
    std::size_t entry_id = 0;
    long long distance_to_target = 0;
    double distance_from_source_root = 0.0;
    std::size_t original_index = 0;
  };

  const auto source_root = ResolveDraftRoot(source.entry_ids, entries, config);
  std::vector<EntryCandidate> candidates;
  candidates.reserve(source.entry_ids.size());
  for (const auto entry_id : source.entry_ids) {
    const auto location = entries.at(entry_id).location;
    candidates.push_back(EntryCandidate{
        .entry_id = entry_id,
        .distance_to_target = DistanceToBounds(location, target.bounds),
        .distance_from_source_root = CalcManhattanDistance(location, source_root),
        .original_index = entries.at(entry_id).original_index,
    });
  }

  std::ranges::sort(candidates, [](const EntryCandidate& lhs, const EntryCandidate& rhs) -> bool {
    if (lhs.distance_to_target != rhs.distance_to_target) {
      return lhs.distance_to_target < rhs.distance_to_target;
    }
    if (std::abs(lhs.distance_from_source_root - rhs.distance_from_source_root) > kScoreEpsilon) {
      return lhs.distance_from_source_root > rhs.distance_from_source_root;
    }
    return lhs.original_index < rhs.original_index;
  });

  std::vector<std::size_t> entry_ids;
  entry_ids.reserve(std::min(kMaxBoundaryEntryCandidates, candidates.size()));
  for (const auto& candidate : candidates) {
    if (entry_ids.size() == kMaxBoundaryEntryCandidates) {
      break;
    }
    entry_ids.push_back(candidate.entry_id);
  }
  return entry_ids;
}

}  // namespace icts::fast_clustering
