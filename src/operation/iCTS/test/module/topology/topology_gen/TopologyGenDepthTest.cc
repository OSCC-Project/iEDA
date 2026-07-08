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
 * @file TopologyGenDepthTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-18
 * @brief Unit tests for explicit TopologyGen target depth handling.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "database/design/Pin.hh"
#include "database/spatial/Point.hh"
#include "database/spatial/Tree.hh"
#include "geometry/Geometry.hh"
#include "module/topology/TopologyGen.hh"
#include "module/topology/config/TopologyConfig.hh"

namespace icts_test::topology_gen {
namespace {

auto BuildLoads() -> std::vector<std::unique_ptr<icts::Pin>>
{
  std::vector<std::unique_ptr<icts::Pin>> storage;
  storage.emplace_back(std::make_unique<icts::Pin>("load0", icts::PinType::kClock, icts::Point<int>(0, 0)));
  storage.emplace_back(std::make_unique<icts::Pin>("load1", icts::PinType::kClock, icts::Point<int>(10, 0)));
  storage.emplace_back(std::make_unique<icts::Pin>("load2", icts::PinType::kClock, icts::Point<int>(20, 0)));
  storage.emplace_back(std::make_unique<icts::Pin>("load3", icts::PinType::kClock, icts::Point<int>(30, 0)));
  storage.emplace_back(std::make_unique<icts::Pin>("load4", icts::PinType::kClock, icts::Point<int>(0, 20)));
  storage.emplace_back(std::make_unique<icts::Pin>("load5", icts::PinType::kClock, icts::Point<int>(10, 20)));
  storage.emplace_back(std::make_unique<icts::Pin>("load6", icts::PinType::kClock, icts::Point<int>(20, 20)));
  storage.emplace_back(std::make_unique<icts::Pin>("load7", icts::PinType::kClock, icts::Point<int>(30, 20)));
  return storage;
}

auto BorrowLoads(const std::vector<std::unique_ptr<icts::Pin>>& storage) -> std::vector<icts::Pin*>
{
  std::vector<icts::Pin*> loads;
  loads.reserve(storage.size());
  for (const auto& pin : storage) {
    loads.push_back(pin.get());
  }
  return loads;
}

auto BuildTwoLoadStorage() -> std::vector<std::unique_ptr<icts::Pin>>
{
  std::vector<std::unique_ptr<icts::Pin>> storage;
  storage.emplace_back(std::make_unique<icts::Pin>("load0", icts::PinType::kClock, icts::Point<int>(0, 0)));
  storage.emplace_back(std::make_unique<icts::Pin>("load1", icts::PinType::kClock, icts::Point<int>(1000, 0)));
  return storage;
}

auto CollectFirstLevelDistances(const icts::Tree& tree) -> std::vector<int>
{
  std::vector<int> distances;
  const auto levels = tree.levels();
  if (levels.size() <= 1U) {
    return distances;
  }

  const auto* root_node = tree.get_node(tree.get_root());
  if (root_node == nullptr) {
    return distances;
  }

  distances.reserve(levels.at(1).size());
  for (const auto node_id : levels.at(1)) {
    const auto* node = tree.get_node(node_id);
    if (node == nullptr) {
      continue;
    }
    distances.push_back(icts::geometry::Manhattan(root_node->get_position(), node->get_position()));
  }
  return distances;
}

TEST(TopologyGenDepthTest, DefaultBuildUsesDeepestPowerOfTwoDepth)
{
  const auto storage = BuildLoads();
  const auto loads = BorrowLoads(storage);

  const auto tree = icts::TopologyGen::build(loads, icts::TopologyGen::Input{}, icts::TopologyGen::Config{});
  const auto levels = tree.levels();

  ASSERT_EQ(levels.size(), 4U);
  EXPECT_EQ(levels.back().size(), 8U);
}

TEST(TopologyGenDepthTest, FixedRootLocationOverridesLoadMedian)
{
  const auto storage = BuildLoads();
  const auto loads = BorrowLoads(storage);
  const icts::Point<int> fixed_root(500, 700);

  icts::TopologyGen::Input input;
  input.fixed_root_location = fixed_root;
  const auto tree = icts::TopologyGen::build(loads, input, icts::TopologyGen::Config{});

  const auto* root_node = tree.get_node(tree.get_root());
  ASSERT_NE(root_node, nullptr);
  EXPECT_EQ(root_node->get_position().get_x(), fixed_root.get_x());
  EXPECT_EQ(root_node->get_position().get_y(), fixed_root.get_y());
}

TEST(TopologyGenDepthTest, ExplicitTargetDepthBuildsRequestedLeafCount)
{
  const auto storage = BuildLoads();
  const auto loads = BorrowLoads(storage);

  auto config = icts::TopologyGen::Config{};
  config.target_depth = 2U;
  const auto tree = icts::TopologyGen::build(loads, icts::TopologyGen::Input{}, config);
  const auto levels = tree.levels();

  ASSERT_EQ(levels.size(), 3U);
  ASSERT_EQ(levels.back().size(), 4U);

  std::unordered_set<const icts::Pin*> covered_loads;
  for (const auto node_id : levels.back()) {
    const auto* node = tree.get_node(node_id);
    ASSERT_NE(node, nullptr);
    ASSERT_FALSE(node->get_loads().empty());
    for (const auto* load : node->get_loads()) {
      ASSERT_NE(load, nullptr);
      covered_loads.insert(load);
    }
  }
  EXPECT_EQ(covered_loads.size(), loads.size());
}

TEST(TopologyGenDepthTest, ExplicitTargetDepthClampsToMaxDepth)
{
  const auto storage = BuildLoads();
  const auto loads = BorrowLoads(storage);

  auto config = icts::TopologyGen::Config{};
  config.target_depth = 8U;
  const auto tree = icts::TopologyGen::build(loads, icts::TopologyGen::Input{}, config);
  const auto levels = tree.levels();

  ASSERT_EQ(levels.size(), 4U);
  EXPECT_EQ(levels.back().size(), 8U);
}

TEST(TopologyGenDepthTest, TopologyToleranceKeepsEdgesInsideBaselineWindow)
{
  const auto storage = BuildTwoLoadStorage();
  const auto loads = BorrowLoads(storage);

  icts::BiPartitionConfig exact_config;
  exact_config.htree_topology_tolerance = 0.0;
  const auto exact_tree
      = icts::TopologyGen::build(loads, icts::TopologyGen::Input{}, icts::TopologyGen::Config{.partition_config = exact_config});
  const auto exact_distances = CollectFirstLevelDistances(exact_tree);
  ASSERT_EQ(exact_distances.size(), 2U);
  EXPECT_NEAR(exact_distances.at(0), exact_distances.at(1), 1);

  icts::BiPartitionConfig tolerant_config;
  tolerant_config.htree_topology_tolerance = 1.0;
  const auto tolerant_tree
      = icts::TopologyGen::build(loads, icts::TopologyGen::Input{}, icts::TopologyGen::Config{.partition_config = tolerant_config});
  const auto tolerant_distances = CollectFirstLevelDistances(tolerant_tree);
  ASSERT_EQ(tolerant_distances.size(), 2U);
  EXPECT_TRUE(std::ranges::any_of(tolerant_distances, [](int distance) -> bool { return distance == 0; }));
  EXPECT_TRUE(std::ranges::any_of(tolerant_distances, [](int distance) -> bool { return distance == 1000; }));
}

}  // namespace
}  // namespace icts_test::topology_gen
