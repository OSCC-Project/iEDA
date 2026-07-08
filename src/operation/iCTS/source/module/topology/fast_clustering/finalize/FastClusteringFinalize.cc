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
 * @file FastClusteringFinalize.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Final legal cluster materialization for fast topology clustering.
 */

#include <glog/logging.h>

#include <algorithm>
#include <compare>
#include <cstddef>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "ClusterConstraintEvaluation.hh"
#include "ClusterConstraintEvaluator.hh"
#include "Clustering.hh"
#include "Log.hh"
#include "Pin.hh"
#include "Point.hh"
#include "TopologyConfig.hh"
#include "cluster_draft/FastClusteringDraft.hh"

namespace icts::fast_clustering {
namespace {

auto MaterializeCluster(const ClusterDraft& draft, const std::vector<LoadEntry>& entries) -> std::vector<Pin*>
{
  std::vector<Pin*> cluster;
  cluster.reserve(draft.entry_ids.size());
  for (const auto entry_id : draft.entry_ids) {
    auto* pin = entries.at(entry_id).pin;
    if (pin != nullptr) {
      cluster.push_back(pin);
    }
  }
  std::ranges::sort(cluster, [](const Pin* lhs, const Pin* rhs) -> bool {
    if (lhs == nullptr || rhs == nullptr) {
      return lhs < rhs;
    }
    return lhs->get_name() < rhs->get_name();
  });
  return cluster;
}

auto NeedExactCap(const ClusterConfig& config) -> bool
{
  return config.enable_exact_cap;
}

auto ToElectricalSummary(const ConstraintEvaluation& evaluation) -> ClusterElectricalSummary
{
  const auto& metrics = evaluation.metrics;
  return ClusterElectricalSummary{
      .exact = metrics.electrical.exact,
      .route_success = metrics.electrical.route_success,
      .sink_count = metrics.fanout,
      .diameter_dbu = metrics.diameter,
      .pin_cap_pf = metrics.electrical.pin_cap,
      .wire_cap_pf = metrics.electrical.wire_cap,
      .total_cap_pf = metrics.electrical.total_cap,
      .wirelength_dbu = metrics.wirelength,
  };
}

auto SplitClusterByLongestAxis(std::vector<Pin*> cluster) -> std::pair<std::vector<Pin*>, std::vector<Pin*>>
{
  const auto bounds = CalcClusterBounds(cluster);
  const bool split_by_x = (bounds.max_x - bounds.min_x) >= (bounds.max_y - bounds.min_y);
  std::ranges::sort(cluster, [split_by_x](const Pin* lhs, const Pin* rhs) -> bool {
    const auto lhs_location = lhs->get_location();
    const auto rhs_location = rhs->get_location();
    const auto lhs_primary = split_by_x ? lhs_location.get_x() : lhs_location.get_y();
    const auto rhs_primary = split_by_x ? rhs_location.get_x() : rhs_location.get_y();
    if (lhs_primary != rhs_primary) {
      return lhs_primary < rhs_primary;
    }
    const auto lhs_secondary = split_by_x ? lhs_location.get_y() : lhs_location.get_x();
    const auto rhs_secondary = split_by_x ? rhs_location.get_y() : rhs_location.get_x();
    if (lhs_secondary != rhs_secondary) {
      return lhs_secondary < rhs_secondary;
    }
    return lhs->get_name() < rhs->get_name();
  });

  const auto middle = cluster.begin() + static_cast<std::ptrdiff_t>(cluster.size() / 2U);
  std::vector<Pin*> lhs(cluster.begin(), middle);
  std::vector<Pin*> rhs(middle, cluster.end());
  return {std::move(lhs), std::move(rhs)};
}

auto AppendFinalCluster(const std::vector<Pin*>& cluster, const ClusterConfig& config, ClusterOutput& result) -> bool
{
  std::vector<std::vector<Pin*>> pending;
  pending.push_back(cluster);
  while (!pending.empty()) {
    auto current_cluster = std::move(pending.back());
    pending.pop_back();
    if (current_cluster.empty()) {
      continue;
    }

    const auto root = ResolveEvaluationRoot(current_cluster, config);
    const auto evaluation = ClusterConstraintEvaluator::evaluateLoads(current_cluster, root, config, NeedExactCap(config));
    if (evaluation.legal) {
      result.clusters.push_back(current_cluster);
      result.centers.push_back(CalcCenter(current_cluster));
      result.electrical_summaries.push_back(ToElectricalSummary(evaluation));
      continue;
    }

    if (current_cluster.size() <= 1U) {
      LOG_WARNING << "Fast clustering could not legalize singleton cluster: violation=" << static_cast<int>(evaluation.violation)
                  << ", pin=" << (current_cluster.front() == nullptr ? std::string("<null>") : current_cluster.front()->get_name());
      return false;
    }

    auto [lhs, rhs] = SplitClusterByLongestAxis(std::move(current_cluster));
    if (lhs.empty() || rhs.empty()) {
      return false;
    }
    pending.push_back(std::move(rhs));
    pending.push_back(std::move(lhs));
  }
  return true;
}

}  // namespace

auto FinalizeClusters(const std::vector<ClusterDraft>& drafts, const std::vector<LoadEntry>& entries, const ClusterConfig& config)
    -> std::optional<ClusterOutput>
{
  ClusterOutput result;
  result.clusters.reserve(drafts.size());
  result.centers.reserve(drafts.size());
  result.electrical_summaries.reserve(drafts.size());

  for (const auto& draft : drafts) {
    auto cluster = MaterializeCluster(draft, entries);
    if (!AppendFinalCluster(cluster, config, result)) {
      return std::nullopt;
    }
  }
  return result;
}

auto CountAssignedLoads(const ClusterOutput& result) -> std::size_t
{
  std::size_t count = 0;
  for (const auto& cluster : result.clusters) {
    count += cluster.size();
  }
  return count;
}

}  // namespace icts::fast_clustering
