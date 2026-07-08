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
 * @file FastClusteringSyntheticTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-24
 * @brief Synthetic fast clustering API and legality regression tests.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ClockRouteSegmentRc.hh"
#include "common/data/pin_factory/PinFactory.hh"
#include "common/dataset/TestDataset.hh"
#include "database/design/Pin.hh"
#include "database/spatial/Point.hh"
#include "module/topology/clustering/Clustering.hh"
#include "module/topology/config/TopologyConfig.hh"
#include "module/topology/fast_clustering/FastClustering.hh"

namespace icts_test::fast_clustering::synthetic {
namespace {

auto BuildClusteredPoints() -> std::vector<icts::Point<int>>
{
  std::vector<icts::Point<int>> points;
  for (int cluster_x = 0; cluster_x < 4; ++cluster_x) {
    for (int cluster_y = 0; cluster_y < 3; ++cluster_y) {
      const int base_x = cluster_x * 1000;
      const int base_y = cluster_y * 1000;
      for (int offset = 0; offset < 5; ++offset) {
        points.emplace_back(base_x + offset * 10, base_y + offset * 7);
      }
    }
  }
  return points;
}

auto CalcDiameter(const std::vector<icts::Pin*>& cluster) -> int
{
  if (cluster.empty()) {
    return 0;
  }

  int min_x = cluster.front()->get_location().get_x();
  int min_y = cluster.front()->get_location().get_y();
  int max_x = min_x;
  int max_y = min_y;
  for (const auto* pin : cluster) {
    const auto location = pin->get_location();
    min_x = std::min(min_x, location.get_x());
    min_y = std::min(min_y, location.get_y());
    max_x = std::max(max_x, location.get_x());
    max_y = std::max(max_y, location.get_y());
  }
  return (max_x - min_x) + (max_y - min_y);
}

auto CountAssignedLoads(const icts::ClusterOutput& result) -> std::size_t
{
  std::size_t assigned_count = 0;
  for (const auto& cluster : result.clusters) {
    assigned_count += cluster.size();
  }
  return assigned_count;
}

auto MakeSyntheticClockRouteSegmentRc() -> icts::ClockRouteSegmentRc
{
  return icts::ClockRouteSegmentRc{
      .dbu_per_um = 1000,
      .resistance_per_um_ohm = 0.002,
      .capacitance_per_um_pf = 0.000001,
  };
}

auto AddSyntheticLoadPinCaps(const std::vector<icts::Pin*>& loads, icts::ClusterConfig& config) -> void
{
  for (const auto* pin : loads) {
    if (pin == nullptr) {
      continue;
    }
    config.sink_pin_cap_pf_by_pin.emplace(pin, 0.002);
  }
}

TEST(FastClusteringSyntheticTest, FacadeProducesCompleteLegalClusters)
{
  auto generated = common::data::pin_factory::BuildPinsFromPoints(BuildClusteredPoints(), {.width = 5000, .height = 4000}, "fast_pin_");
  icts::ClusterConfig config;
  config.max_fanout = 6;
  config.max_diameter = 160;
  config.max_cap = std::numeric_limits<double>::infinity();
  config.enable_exact_cap = false;

  const auto result = icts::FastClustering::run(generated.loads, config);

  ASSERT_FALSE(result.clusters.empty());
  EXPECT_EQ(CountAssignedLoads(result), generated.loads.size());
  EXPECT_EQ(result.centers.size(), result.clusters.size());
  EXPECT_EQ(result.electrical_summaries.size(), result.clusters.size());

  std::set<const icts::Pin*> seen_pins;
  for (const auto& cluster : result.clusters) {
    EXPECT_LE(cluster.size(), config.max_fanout);
    EXPECT_LE(CalcDiameter(cluster), config.max_diameter);
    for (const auto* pin : cluster) {
      EXPECT_TRUE(seen_pins.insert(pin).second) << pin->get_name();
    }
  }
  EXPECT_EQ(seen_pins.size(), generated.loads.size());
}

TEST(FastClusteringSyntheticTest, ExactCapUsesExplicitClusterLoadPinCaps)
{
  auto generated = common::data::pin_factory::BuildPinsFromPoints(BuildClusteredPoints(), {.width = 5000, .height = 4000}, "cap_pin_");
  icts::ClusterConfig config;
  config.max_fanout = 8;
  config.max_cap = 1.0;
  config.clock_route_segment_rc = MakeSyntheticClockRouteSegmentRc();
  AddSyntheticLoadPinCaps(generated.loads, config);

  const auto result = icts::FastClustering::run(generated.loads, config);

  ASSERT_FALSE(result.clusters.empty());
  ASSERT_EQ(result.electrical_summaries.size(), result.clusters.size());
  EXPECT_EQ(CountAssignedLoads(result), generated.loads.size());
  for (const auto& summary : result.electrical_summaries) {
    EXPECT_TRUE(summary.exact);
    EXPECT_TRUE(summary.route_success);
    EXPECT_GT(summary.pin_cap_pf, 0.0);
    EXPECT_GE(summary.total_cap_pf, summary.pin_cap_pf);
  }
}

TEST(FastClusteringSyntheticTest, ClusteringFacadeMatchesFastClusteringFacade)
{
  auto generated = common::data::pin_factory::BuildPinsFromPoints(BuildClusteredPoints(), {.width = 5000, .height = 4000}, "facade_pin_");
  icts::ClusterConfig config;
  config.max_fanout = 8;
  config.max_cap = std::numeric_limits<double>::infinity();
  config.enable_exact_cap = false;

  const auto topology_result = icts::FastClustering::run(generated.loads, config);
  const auto clustering_result = icts::Clustering::fastClustering(generated.loads, config);

  EXPECT_EQ(topology_result.clusters.size(), clustering_result.clusters.size());
  EXPECT_EQ(CountAssignedLoads(topology_result), CountAssignedLoads(clustering_result));
}

}  // namespace
}  // namespace icts_test::fast_clustering::synthetic
