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
 * @file Clustering.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-01-16
 * @brief Clustering for topology embedding.
 */

#include "Clustering.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ClusterConstraintEvaluation.hh"
#include "ClusterConstraintEvaluator.hh"
#include "FastClustering.hh"
#include "Geometry.hh"
#include "KMeans.hh"
#include "MinCostFlow.hh"
#include "Pin.hh"
#include "Point.hh"
#include "TopologyConfig.hh"

namespace icts {
namespace {

constexpr double kDefaultMaxRatio = 0.6;
constexpr int kDefaultMaxIter = 10;
constexpr int kDefaultConvergeThreshold = 1000;
constexpr std::size_t kDefaultKMeansIterCount = 5;
constexpr int kBipartitionClusterCount = 2;

auto SplitByPosition(const std::vector<Pin*>& loads, std::size_t min_cluster_size) -> std::vector<std::vector<Pin*>>
{
  std::vector<Pin*> sorted = loads;
  std::ranges::sort(sorted, [](Pin* lhs, Pin* rhs) -> auto {
    const auto& lhs_point = lhs->get_location();
    const auto& rhs_point = rhs->get_location();
    if (lhs_point.get_x() != rhs_point.get_x()) {
      return lhs_point.get_x() < rhs_point.get_x();
    }
    return lhs_point.get_y() < rhs_point.get_y();
  });

  std::vector<std::vector<Pin*>> clusters(kBipartitionClusterCount);
  if (sorted.size() <= 1) {
    clusters.at(0) = std::move(sorted);
    return clusters;
  }

  std::size_t left_size = std::max<std::size_t>(min_cluster_size, sorted.size() / 2);
  if (sorted.size() - left_size < min_cluster_size) {
    left_size = sorted.size() - min_cluster_size;
  }
  left_size = std::max<std::size_t>(1, left_size);

  const auto middle_iter = std::next(sorted.begin(), static_cast<std::ptrdiff_t>(left_size));
  clusters.at(0).assign(sorted.begin(), middle_iter);
  clusters.at(1).assign(middle_iter, sorted.end());
  return clusters;
}

auto CalcCenters(const std::vector<std::vector<Pin*>>& clusters) -> std::vector<Point<double>>
{
  std::vector<Point<double>> centers;
  centers.reserve(clusters.size());
  for (const auto& cluster : clusters) {
    centers.push_back(geometry::CalcCenter(cluster, [](Pin* pin) -> auto { return pin->get_location(); }));
  }
  return centers;
}

auto ToClusterElectricalViolation(ConstraintViolation violation) -> ClusterElectricalViolation
{
  switch (violation) {
    case ConstraintViolation::kNone:
      return ClusterElectricalViolation::kNone;
    case ConstraintViolation::kEmptyCluster:
      return ClusterElectricalViolation::kEmptyCluster;
    case ConstraintViolation::kFanout:
      return ClusterElectricalViolation::kFanout;
    case ConstraintViolation::kDiameter:
      return ClusterElectricalViolation::kDiameter;
    case ConstraintViolation::kCapacitance:
      return ClusterElectricalViolation::kCapacitance;
    case ConstraintViolation::kRoutingFailed:
      return ClusterElectricalViolation::kRoutingFailed;
  }
  return ClusterElectricalViolation::kEmptyCluster;
}

}  // namespace

auto Clustering::biPartition(const std::vector<Pin*>& loads, std::size_t min_cluster_size) -> ClusterOutput
{
  BiPartitionConfig config;
  config.max_ratio = kDefaultMaxRatio;
  config.max_iter = kDefaultMaxIter;
  config.converge_threshold = kDefaultConvergeThreshold;
  config.kmeans_iter_count = kDefaultKMeansIterCount;
  return biPartition(loads, min_cluster_size, config);
}

auto Clustering::biPartition(const std::vector<Pin*>& loads, std::size_t min_cluster_size, const BiPartitionConfig& config) -> ClusterOutput
{
  ClusterOutput result;
  if (loads.empty()) {
    return result;
  }
  if (loads.size() == 1) {
    result.clusters = {loads};
    result.centers = {loads.front()->get_location()};
    return result;
  }

  const std::size_t safe_min = std::max<std::size_t>(1, std::min(min_cluster_size, loads.size() / 2));

  auto max_cluster_size = static_cast<std::size_t>(std::ceil(config.max_ratio * static_cast<double>(loads.size())));
  max_cluster_size = std::max(max_cluster_size, safe_min);
  max_cluster_size = std::min(max_cluster_size, loads.size() - safe_min);
  if (config.max_cluster_size > 0U) {
    const auto max_legal_size = std::max(safe_min, std::min(config.max_cluster_size, loads.size() - safe_min));
    max_cluster_size = std::min(max_cluster_size, max_legal_size);
  }
  max_cluster_size = std::max<std::size_t>(1, max_cluster_size);

  const KMeans<Pin*> kmeans;
  auto kmeans_result
      = kmeans.run(loads, kBipartitionClusterCount, [](Pin* pin) -> auto { return pin->get_location(); }, config.kmeans_iter_count);

  auto centers = kmeans_result.centers;
  if (centers.size() < 2) {
    auto position_split_clusters = SplitByPosition(loads, safe_min);
    centers = CalcCenters(position_split_clusters);
  }

  std::vector<std::vector<Pin*>> clusters;
  for (int iter = 0; iter < config.max_iter; ++iter) {
    MinCostFlow<Pin*> mcf;
    for (auto* pin : loads) {
      const auto& loc = pin->get_location();
      mcf.addNode(loc.get_x(), loc.get_y(), pin);
    }
    for (const auto& center : centers) {
      mcf.addCenter(center.get_x(), center.get_y());
    }

    clusters = mcf.run(max_cluster_size);
    if (clusters.size() < 2) {
      clusters = SplitByPosition(loads, safe_min);
    }

    auto new_centers = CalcCenters(clusters);
    if (new_centers.size() < 2) {
      break;
    }

    double max_shift = 0.0;
    const std::size_t loop_count = std::min(centers.size(), new_centers.size());
    for (std::size_t i = 0; i < loop_count; ++i) {
      max_shift = std::max(max_shift, geometry::Manhattan(centers.at(i), new_centers.at(i)));
    }
    centers = std::move(new_centers);
    if (max_shift < config.converge_threshold) {
      break;
    }
  }

  result.clusters = clusters;
  if (result.clusters.empty()) {
    result.clusters = SplitByPosition(loads, safe_min);
  }

  if (centers.size() != result.clusters.size()) {
    centers = CalcCenters(result.clusters);
  }
  result.centers.reserve(centers.size());
  for (const auto& center : centers) {
    result.centers.emplace_back(static_cast<int>(std::lround(center.get_x())), static_cast<int>(std::lround(center.get_y())));
  }

  return result;
}

auto Clustering::fastClustering(const std::vector<Pin*>& loads) -> ClusterOutput
{
  return defaultFastClustering(loads, ClusterConfig{});
}

auto Clustering::defaultFastClustering(const std::vector<Pin*>& loads, const ClusterConfig& base_config) -> ClusterOutput
{
  return FastClustering::runDefault(loads, base_config);
}

auto Clustering::fastClustering(const std::vector<Pin*>& loads, const ClusterConfig& config) -> ClusterOutput
{
  return FastClustering::run(loads, config);
}

auto Clustering::evaluateClusterElectrical(const std::vector<Pin*>& loads, const Point<int>& anchor, const ClusterConfig& config)
    -> ClusterElectricalEvaluation
{
  const bool need_exact_cap = config.enable_exact_cap || config.always_build_exact_cap || IsFiniteCapLimit(config.max_cap);
  return evaluateClusterElectrical(loads, anchor, config, need_exact_cap);
}

auto Clustering::evaluateClusterElectrical(const std::vector<Pin*>& loads, const Point<int>& anchor, const ClusterConfig& config,
                                           bool need_exact_cap) -> ClusterElectricalEvaluation
{
  const auto evaluation = ClusterConstraintEvaluator::evaluateLoads(loads, anchor, config, need_exact_cap);
  const auto& metrics = evaluation.metrics;
  return ClusterElectricalEvaluation{
      .legal = evaluation.legal,
      .violation = ToClusterElectricalViolation(evaluation.violation),
      .summary =
          ClusterElectricalSummary{
              .exact = metrics.electrical.exact,
              .route_success = metrics.electrical.route_success,
              .sink_count = metrics.fanout,
              .diameter_dbu = metrics.diameter,
              .pin_cap_pf = metrics.electrical.pin_cap,
              .wire_cap_pf = metrics.electrical.wire_cap,
              .total_cap_pf = metrics.electrical.total_cap,
              .wirelength_dbu = metrics.wirelength,
          },
  };
}

}  // namespace icts
