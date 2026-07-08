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
 * @file ClusterGeometryMetrics.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Shared cluster geometry and metric helpers for clustering tests.
 */

#include "common/clustering/metrics/ClusterGeometryMetrics.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "database/design/Pin.hh"
#include "database/spatial/Point.hh"
#include "module/topology/clustering/Clustering.hh"

namespace icts_test::common::clustering {
namespace {

constexpr std::size_t kSingletonClusterSize = 1U;

}  // namespace

auto CalcClusterDiameter(const std::vector<icts::Pin*>& cluster) -> int
{
  if (cluster.empty()) {
    return 0;
  }

  int min_x = cluster.front()->get_location().get_x();
  int min_y = cluster.front()->get_location().get_y();
  int max_x = min_x;
  int max_y = min_y;
  for (const auto* pin : cluster) {
    if (pin == nullptr) {
      continue;
    }
    const auto& location = pin->get_location();
    min_x = std::min(min_x, location.get_x());
    min_y = std::min(min_y, location.get_y());
    max_x = std::max(max_x, location.get_x());
    max_y = std::max(max_y, location.get_y());
  }
  return (max_x - min_x) + (max_y - min_y);
}

auto CalcClusterCenter(const std::vector<icts::Pin*>& cluster) -> icts::Point<int>
{
  if (cluster.empty()) {
    return {0, 0};
  }

  long long sum_x = 0;
  long long sum_y = 0;
  std::size_t valid_count = 0;
  for (const auto* pin : cluster) {
    if (pin == nullptr) {
      continue;
    }
    ++valid_count;
    const auto& location = pin->get_location();
    sum_x += location.get_x();
    sum_y += location.get_y();
  }
  if (valid_count == 0) {
    return {0, 0};
  }

  return {static_cast<int>(std::lround(static_cast<double>(sum_x) / static_cast<double>(valid_count))),
          static_cast<int>(std::lround(static_cast<double>(sum_y) / static_cast<double>(valid_count)))};
}

auto CollectClusterMetrics(const icts::ClusterOutput& result) -> ClusterMetrics
{
  ClusterMetrics metrics;
  if (result.clusters.empty()) {
    return metrics;
  }

  metrics.cluster_count = result.clusters.size();
  metrics.min_cluster_size = std::numeric_limits<std::size_t>::max();
  metrics.min_cluster_diameter = std::numeric_limits<int>::max();

  std::size_t total_load_count = 0;
  for (const auto& cluster : result.clusters) {
    if (cluster.empty()) {
      continue;
    }
    const auto cluster_size = cluster.size();
    const auto cluster_diameter = CalcClusterDiameter(cluster);
    metrics.singleton_cluster_count += cluster_size == kSingletonClusterSize ? 1U : 0U;
    metrics.min_cluster_size = std::min(metrics.min_cluster_size, cluster_size);
    metrics.max_cluster_size = std::max(metrics.max_cluster_size, cluster_size);
    metrics.min_cluster_diameter = std::min(metrics.min_cluster_diameter, cluster_diameter);
    metrics.max_cluster_diameter = std::max(metrics.max_cluster_diameter, cluster_diameter);
    total_load_count += cluster_size;
  }

  if (metrics.min_cluster_size == std::numeric_limits<std::size_t>::max()) {
    metrics.min_cluster_size = 0;
  }
  if (metrics.min_cluster_diameter == std::numeric_limits<int>::max()) {
    metrics.min_cluster_diameter = 0;
  }
  metrics.avg_cluster_size
      = metrics.cluster_count == 0 ? 0.0 : static_cast<double>(total_load_count) / static_cast<double>(metrics.cluster_count);
  return metrics;
}

}  // namespace icts_test::common::clustering
