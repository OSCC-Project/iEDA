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
 * @file FastClusteringRealTechBenchmarkMetrics.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Metric evaluation for the fast-clustering real-tech benchmark.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "FastClusteringRealTechBenchmarkFixture.hh"
#include "database/config/Config.hh"
#include "database/design/Pin.hh"
#include "database/io/Wrapper.hh"
#include "database/spatial/Point.hh"
#include "module/topology/clustering/Clustering.hh"
#include "module/topology/config/TopologyConfig.hh"
#include "module/topology/fast_clustering/FastClustering.hh"

namespace icts_test::fast_clustering::realtech {

auto CalcClusterDiameter(const std::vector<icts::Pin*>& loads) -> int
{
  if (loads.empty()) {
    return 0;
  }

  int min_x = loads.front()->get_location().get_x();
  int min_y = loads.front()->get_location().get_y();
  int max_x = min_x;
  int max_y = min_y;
  for (const auto* pin : loads) {
    if (pin == nullptr) {
      continue;
    }
    const auto location = pin->get_location();
    min_x = std::min(min_x, location.get_x());
    min_y = std::min(min_y, location.get_y());
    max_x = std::max(max_x, location.get_x());
    max_y = std::max(max_y, location.get_y());
  }
  return (max_x - min_x) + (max_y - min_y);
}

namespace {

auto CalcManhattanDistance(const icts::Point<int>& lhs, const icts::Point<int>& rhs) -> double
{
  const auto dx = std::abs(static_cast<long long>(lhs.get_x()) - static_cast<long long>(rhs.get_x()));
  const auto dy = std::abs(static_cast<long long>(lhs.get_y()) - static_cast<long long>(rhs.get_y()));
  return static_cast<double>(dx + dy);
}

auto CalcClusterCenter(const std::vector<icts::Pin*>& cluster) -> icts::Point<int>
{
  long long x_sum = 0;
  long long y_sum = 0;
  std::size_t pin_count = 0;
  for (const auto* pin : cluster) {
    if (pin == nullptr) {
      continue;
    }
    const auto location = pin->get_location();
    x_sum += location.get_x();
    y_sum += location.get_y();
    ++pin_count;
  }
  if (pin_count == 0U) {
    return {0, 0};
  }
  return {static_cast<int>(std::lround(static_cast<double>(x_sum) / static_cast<double>(pin_count))),
          static_cast<int>(std::lround(static_cast<double>(y_sum) / static_cast<double>(pin_count)))};
}

auto CalcClusterMedian(const std::vector<icts::Pin*>& cluster) -> icts::Point<int>
{
  std::vector<int> x_coords;
  std::vector<int> y_coords;
  x_coords.reserve(cluster.size());
  y_coords.reserve(cluster.size());
  for (const auto* pin : cluster) {
    if (pin == nullptr) {
      continue;
    }
    const auto location = pin->get_location();
    x_coords.push_back(location.get_x());
    y_coords.push_back(location.get_y());
  }
  if (x_coords.empty()) {
    return {0, 0};
  }

  const auto middle = static_cast<std::ptrdiff_t>(x_coords.size() / 2U);
  auto x_middle = x_coords.begin() + middle;
  auto y_middle = y_coords.begin() + middle;
  std::nth_element(x_coords.begin(), x_middle, x_coords.end());
  std::nth_element(y_coords.begin(), y_middle, y_coords.end());
  return {*x_middle, *y_middle};
}

auto ResolveClusterRoot(const std::vector<icts::Pin*>& cluster, const icts::ClusterConfig& config) -> icts::Point<int>
{
  return config.root_policy == icts::ClusterRootPolicy::kCenter ? CalcClusterCenter(cluster) : CalcClusterMedian(cluster);
}

auto CalcRoutingCapProxy(const std::vector<icts::Pin*>& cluster, const icts::ClusterConfig& config) -> double
{
  if (cluster.size() <= 1U) {
    return 0.0;
  }

  const auto root = ResolveClusterRoot(cluster, config);
  double proxy = 0.0;
  for (const auto* pin : cluster) {
    if (pin == nullptr) {
      continue;
    }
    proxy += CalcManhattanDistance(root, pin->get_location());
  }
  return proxy;
}

auto CalcPopulationVariance(const std::vector<double>& values, double mean) -> double
{
  if (values.empty()) {
    return 0.0;
  }

  double variance = 0.0;
  for (const auto value : values) {
    const auto delta = value - mean;
    variance += delta * delta;
  }
  return variance / static_cast<double>(values.size());
}

auto ScoreCluster(const std::vector<icts::Pin*>& cluster, const icts::ClusterElectricalSummary* summary, const icts::ClusterConfig& config)
    -> double
{
  const auto diameter = CalcClusterDiameter(cluster);
  if (config.scoring_strategy == icts::ClusterScoringStrategy::kTotalWirelength) {
    if (summary != nullptr && summary->wirelength_dbu > 0.0) {
      return config.wirelength_weight * summary->wirelength_dbu;
    }
    return config.wirelength_weight * static_cast<double>(diameter);
  }
  if (diameter > 0) {
    return static_cast<double>(diameter);
  }
  return config.max_diameter > 0 ? static_cast<double>(config.max_diameter) : 0.0;
}

}  // namespace

auto BuildBenchmarkConfig(const std::vector<icts::Pin*>& loads) -> icts::ClusterConfig
{
  auto config = icts::FastClustering::buildElectricalBaseConfig(icts_test::runtime::CurrentRuntime().config.get_max_fanout(),
                                                                icts_test::runtime::CurrentRuntime().config.get_max_cap());
  config.clock_route_segment_rc
      = icts_test::runtime::CurrentRuntime().wrapper.queryConfiguredClockRouteSegmentRc(icts_test::runtime::CurrentRuntime().config);
  config.sink_pin_cap_pf_by_pin.reserve(loads.size());
  for (const auto* pin : loads) {
    if (pin == nullptr) {
      continue;
    }
    config.sink_pin_cap_pf_by_pin.emplace(pin, std::max(0.0, icts_test::runtime::CurrentRuntime().wrapper.queryPinCapacitance(pin)));
  }
  config.enable_exact_cap = false;
  config.always_build_exact_cap = false;
  return config;
}

auto EvaluateResult(const std::string& algorithm, const icts::ClusterOutput& result, const icts::ClusterConfig& config,
                    std::size_t expected_load_count, double runtime_ms) -> ResultMetrics
{
  ResultMetrics metrics;
  metrics.algorithm = algorithm;
  metrics.runtime_ms = runtime_ms;
  metrics.expected_load_count = expected_load_count;
  metrics.cluster_count = result.clusters.size();
  metrics.legal = !result.clusters.empty();

  std::vector<double> routing_cap_proxies;
  routing_cap_proxies.reserve(result.clusters.size());
  for (std::size_t cluster_id = 0; cluster_id < result.clusters.size(); ++cluster_id) {
    const auto& cluster = result.clusters.at(cluster_id);
    metrics.load_count += cluster.size();
    metrics.max_fanout = std::max(metrics.max_fanout, cluster.size());
    if (cluster.size() == 1U) {
      ++metrics.singleton_count;
    }
    const auto diameter = CalcClusterDiameter(cluster);
    metrics.max_diameter = std::max(metrics.max_diameter, diameter);
    if (config.max_fanout > 0U && cluster.size() > config.max_fanout) {
      ++metrics.fanout_violations;
    }
    if (config.max_diameter > 0 && diameter > config.max_diameter) {
      ++metrics.diameter_violations;
    }

    const auto* summary = cluster_id < result.electrical_summaries.size() ? &result.electrical_summaries.at(cluster_id) : nullptr;
    if (summary != nullptr) {
      metrics.total_wirelength += summary->wirelength_dbu;
      if (config.max_cap > 0.0 && std::isfinite(config.max_cap) && summary->total_cap_pf > config.max_cap) {
        ++metrics.cap_violations;
      }
      if (summary->exact && !summary->route_success) {
        ++metrics.route_failures;
      }
    }
    metrics.total_score += ScoreCluster(cluster, summary, config);
    const auto routing_cap_proxy = CalcRoutingCapProxy(cluster, config);
    metrics.total_routing_cap_proxy += routing_cap_proxy;
    routing_cap_proxies.push_back(routing_cap_proxy);
  }

  if (!routing_cap_proxies.empty()) {
    metrics.avg_routing_cap_proxy = metrics.total_routing_cap_proxy / static_cast<double>(routing_cap_proxies.size());
    metrics.routing_cap_proxy_variance = CalcPopulationVariance(routing_cap_proxies, metrics.avg_routing_cap_proxy);
    metrics.routing_cap_proxy_stddev = std::sqrt(metrics.routing_cap_proxy_variance);
  }

  if (metrics.load_count < metrics.expected_load_count) {
    metrics.missing_load_count = metrics.expected_load_count - metrics.load_count;
  }
  metrics.legal = metrics.legal && metrics.fanout_violations == 0U && metrics.diameter_violations == 0U && metrics.cap_violations == 0U
                  && metrics.route_failures == 0U && metrics.load_count == metrics.expected_load_count;
  return metrics;
}

}  // namespace icts_test::fast_clustering::realtech
