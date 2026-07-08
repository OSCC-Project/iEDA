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
 * @file RouterClockTreeTest.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-03-29
 * @brief Unit tests for clock routing tree specialization.
 */

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "ClockRouteSegmentRc.hh"
#include "SteinerTree.hh"
#include "database/spatial/Point.hh"
#include "module/routing/router/Router.hh"

namespace icts_test {
namespace {

using Router = icts::Router;
using Point = icts::Point<int>;

constexpr int kTerminalX = 100;
constexpr int kTerminalY = 200;
constexpr int kSteinerX = 80;
constexpr int kSteinerY = 180;
constexpr int kRootX = 10;
constexpr int kRootY = 20;
constexpr double kTerminalPinCap = 0.18;
constexpr double kTerminalInsertionDelay = 0.9;
constexpr double kRootPinCap = 0.23;
constexpr double kDriverPinCap = 0.26;
constexpr double kDriverInsertionDelay = 0.35;
constexpr double kLoadPinCap0 = 0.11;
constexpr double kLoadPinCap1 = 0.13;
constexpr double kLoadPinCap2 = 0.17;
constexpr double kLoadInsertionDelay0 = 0.51;
constexpr double kLoadInsertionDelay1 = 0.63;
constexpr double kLoadInsertionDelay2 = 0.74;
constexpr int kTestDbuPerUm = 1000;
constexpr double kTestResistancePerUmOhm = 0.002;
constexpr double kTestCapacitancePerUmPf = 0.0004;
constexpr int kFluteLoad0X = 100;
constexpr int kFluteLoad0Y = 20;
constexpr int kFluteLoad1X = 20;
constexpr int kFluteLoad1Y = 130;
constexpr int kFluteLoad2X = 150;
constexpr int kFluteLoad2Y = 120;
constexpr int kSaltLoad0X = 95;
constexpr int kSaltLoad0Y = 15;
constexpr int kSaltLoad1X = 40;
constexpr int kSaltLoad1Y = 125;
constexpr int kSaltLoad2X = 170;
constexpr int kSaltLoad2Y = 90;

auto MakeTestClockRouteSegmentRc() -> icts::ClockRouteSegmentRc
{
  return icts::ClockRouteSegmentRc{
      .dbu_per_um = kTestDbuPerUm,
      .resistance_per_um_ohm = kTestResistancePerUmOhm,
      .capacitance_per_um_pf = kTestCapacitancePerUmPf,
  };
}

auto ExpectTerminalNodeMetadata(const Router::ClockSteinerTreeType& clock_tree, const Router::ClockTerminal& terminal) -> void
{
  const auto* node = clock_tree.findNode(terminal.name);
  ASSERT_NE(node, nullptr);
  EXPECT_TRUE(node->is_terminal);
  EXPECT_DOUBLE_EQ(node->pin_cap, terminal.pin_cap);
  EXPECT_DOUBLE_EQ(node->insertion_delay, terminal.insertion_delay);
}

auto ExpectSteinerNodeDefaults(const Router::ClockSteinerTreeType& clock_tree) -> void
{
  for (const auto& node : clock_tree.get_nodes()) {
    if (node.is_terminal) {
      continue;
    }
    EXPECT_DOUBLE_EQ(node.pin_cap, 0.0);
    EXPECT_DOUBLE_EQ(node.insertion_delay, 0.0);
  }
}

auto ExpectRCTreeLumpedCapMatchesClockTree(const Router::ClockSteinerTreeType& clock_tree) -> void
{
  const auto rc_tree = Router::buildRCTree(clock_tree, MakeTestClockRouteSegmentRc());
  for (const auto& node : clock_tree.get_nodes()) {
    const auto* vertex = rc_tree.findVertex(node.name);
    ASSERT_NE(vertex, nullptr);
    EXPECT_DOUBLE_EQ(vertex->lumped_cap, node.pin_cap);
  }
}

TEST(RouterClockTreeTest, ClockSteinerNodeDefaultsAndExplicitMetadata)
{
  Router::ClockSteinerTreeType clock_tree;

  auto terminal_id = clock_tree.addNode("sink0", Point(kTerminalX, kTerminalY), true, kTerminalPinCap, kTerminalInsertionDelay);
  auto steiner_id = clock_tree.addNode("steiner0", Point(kSteinerX, kSteinerY), false);

  const auto* terminal = clock_tree.get_node(terminal_id);
  const auto* steiner = clock_tree.get_node(steiner_id);
  ASSERT_NE(terminal, nullptr);
  ASSERT_NE(steiner, nullptr);

  EXPECT_DOUBLE_EQ(terminal->pin_cap, kTerminalPinCap);
  EXPECT_DOUBLE_EQ(terminal->insertion_delay, kTerminalInsertionDelay);
  EXPECT_DOUBLE_EQ(steiner->pin_cap, 0.0);
  EXPECT_DOUBLE_EQ(steiner->insertion_delay, 0.0);
}

TEST(RouterClockTreeTest, BuildRCTreeUsesClockNodePinCap)
{
  Router::ClockSteinerTreeType clock_tree;
  auto clock_root_id = clock_tree.addNode("clock_root", Point(kRootX, kRootY), true, kRootPinCap, 0.0);
  auto steiner_id = clock_tree.addNode("steiner", Point(kSteinerX, kSteinerY), false);
  clock_tree.setRoot(clock_root_id);
  ASSERT_NE(clock_tree.addEdge(clock_root_id, steiner_id, 10, 10), Router::ClockSteinerTreeType::kInvalidId);

  auto clock_rc_tree = Router::buildRCTree(clock_tree, MakeTestClockRouteSegmentRc());
  const auto* clock_vertex = clock_rc_tree.findVertex("clock_root");
  const auto* steiner_vertex = clock_rc_tree.findVertex("steiner");
  ASSERT_NE(clock_vertex, nullptr);
  ASSERT_NE(steiner_vertex, nullptr);
  EXPECT_DOUBLE_EQ(clock_vertex->lumped_cap, kRootPinCap);
  EXPECT_DOUBLE_EQ(steiner_vertex->lumped_cap, 0.0);
}

TEST(RouterClockTreeTest, BuildFluteClockTreePreservesTerminalMetadataAndRCTreeCap)
{
  Router::ClockTerminal driver;
  driver.name = "driver";
  driver.location = Point(0, 0);
  driver.pin_cap = kDriverPinCap;
  driver.insertion_delay = kDriverInsertionDelay;

  Router::ClockTerminal load0;
  load0.name = "load0";
  load0.location = Point(kFluteLoad0X, kFluteLoad0Y);
  load0.pin_cap = kLoadPinCap0;
  load0.insertion_delay = kLoadInsertionDelay0;

  Router::ClockTerminal load1;
  load1.name = "load1";
  load1.location = Point(kFluteLoad1X, kFluteLoad1Y);
  load1.pin_cap = kLoadPinCap1;
  load1.insertion_delay = kLoadInsertionDelay1;

  Router::ClockTerminal load2;
  load2.name = "load2";
  load2.location = Point(kFluteLoad2X, kFluteLoad2Y);
  load2.pin_cap = kLoadPinCap2;
  load2.insertion_delay = kLoadInsertionDelay2;

  const std::vector<Router::ClockTerminal> loads = {load0, load1, load2};

  const auto clock_tree = Router::buildFluteTree(driver, loads);
  EXPECT_TRUE(clock_tree.validate());

  ExpectTerminalNodeMetadata(clock_tree, driver);
  for (const auto& load : loads) {
    ExpectTerminalNodeMetadata(clock_tree, load);
  }
  ExpectSteinerNodeDefaults(clock_tree);
  ExpectRCTreeLumpedCapMatchesClockTree(clock_tree);
}

TEST(RouterClockTreeTest, BuildFluteClockTreeLegalizesOverlappingTerminals)
{
  Router::ClockTerminal driver;
  driver.name = "driver";
  driver.location = Point(0, 0);
  driver.pin_cap = kDriverPinCap;
  driver.insertion_delay = kDriverInsertionDelay;

  Router::ClockTerminal load0;
  load0.name = "load0";
  load0.location = Point(0, 0);
  load0.pin_cap = kLoadPinCap0;
  load0.insertion_delay = kLoadInsertionDelay0;

  Router::ClockTerminal load1;
  load1.name = "load1";
  load1.location = Point(0, 0);
  load1.pin_cap = kLoadPinCap1;
  load1.insertion_delay = kLoadInsertionDelay1;

  Router::ClockTerminal load2;
  load2.name = "load2";
  load2.location = Point(kFluteLoad2X, kFluteLoad2Y);
  load2.pin_cap = kLoadPinCap2;
  load2.insertion_delay = kLoadInsertionDelay2;

  const std::vector<Router::ClockTerminal> loads = {load0, load1, load2};

  const auto clock_tree = Router::buildFluteTree(driver, loads);
  ASSERT_TRUE(clock_tree.validate());

  ExpectTerminalNodeMetadata(clock_tree, driver);
  for (const auto& load : loads) {
    ExpectTerminalNodeMetadata(clock_tree, load);
  }

  const auto* driver_node = clock_tree.findNode(driver.name);
  const auto* load0_node = clock_tree.findNode(load0.name);
  const auto* load1_node = clock_tree.findNode(load1.name);
  const auto* load2_node = clock_tree.findNode(load2.name);
  ASSERT_NE(driver_node, nullptr);
  ASSERT_NE(load0_node, nullptr);
  ASSERT_NE(load1_node, nullptr);
  ASSERT_NE(load2_node, nullptr);

  EXPECT_EQ(driver_node->location, driver.location);
  EXPECT_NE(load0_node->location, driver_node->location);
  EXPECT_NE(load1_node->location, driver_node->location);
  EXPECT_NE(load0_node->location, load1_node->location);
  EXPECT_NE(load2_node->location, driver_node->location);
}

TEST(RouterClockTreeTest, BuildSaltClockTreePreservesTerminalMetadataAndRCTreeCap)
{
  Router::ClockTerminal driver;
  driver.name = "driver";
  driver.location = Point(0, 0);
  driver.pin_cap = kDriverPinCap;
  driver.insertion_delay = kDriverInsertionDelay;

  Router::ClockTerminal load0;
  load0.name = "load0";
  load0.location = Point(kSaltLoad0X, kSaltLoad0Y);
  load0.pin_cap = kLoadPinCap0;
  load0.insertion_delay = kLoadInsertionDelay0;

  Router::ClockTerminal load1;
  load1.name = "load1";
  load1.location = Point(kSaltLoad1X, kSaltLoad1Y);
  load1.pin_cap = kLoadPinCap1;
  load1.insertion_delay = kLoadInsertionDelay1;

  Router::ClockTerminal load2;
  load2.name = "load2";
  load2.location = Point(kSaltLoad2X, kSaltLoad2Y);
  load2.pin_cap = kLoadPinCap2;
  load2.insertion_delay = kLoadInsertionDelay2;

  const std::vector<Router::ClockTerminal> loads = {load0, load1, load2};

  const auto clock_tree = Router::buildSaltTree(driver, loads);
  EXPECT_TRUE(clock_tree.validate());

  ExpectTerminalNodeMetadata(clock_tree, driver);
  for (const auto& load : loads) {
    ExpectTerminalNodeMetadata(clock_tree, load);
  }
  ExpectSteinerNodeDefaults(clock_tree);
  ExpectRCTreeLumpedCapMatchesClockTree(clock_tree);
}

}  // namespace
}  // namespace icts_test
