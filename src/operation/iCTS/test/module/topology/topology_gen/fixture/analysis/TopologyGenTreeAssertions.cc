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
 * @file TopologyGenTreeAssertions.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Topology validation and summary helpers for topology generation tests.
 */

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "Log.hh"
#include "Point.hh"
#include "Tree.hh"
#include "common/dataset/TestDataset.hh"
#include "common/topology/TopologyAnalysis.hh"
#include "module/topology/topology_gen/fixture/TopologyGenCaseFixture.hh"
#include "utils/geometry/Geometry.hh"

namespace icts {
class Pin;
}  // namespace icts

namespace icts_test::topology_gen::detail {
namespace {

constexpr std::size_t kInvalidNodeId = std::numeric_limits<std::size_t>::max();

struct EdgeValidationStats
{
  std::size_t invalid_parent_count = 0;
  std::size_t invalid_pos_count = 0;
  std::size_t missing_edge_count = 0;
  std::size_t zero_edge_count = 0;
};

auto IsValidPos(const icts::Point<int>& position) -> bool
{
  return position.get_x() >= 0 && position.get_y() >= 0;
}

auto ValidateNodeEdge(const icts::Tree& tree, std::size_t node_id, EdgeValidationStats& stats) -> void
{
  const auto* node = tree.get_node(node_id);
  if (node == nullptr) {
    return;
  }
  const auto parent_id = node->get_parent();
  if (parent_id == kInvalidNodeId || parent_id >= tree.get_size()) {
    ++stats.invalid_parent_count;
    LOG_WARNING << "Edge issue: node=" << node_id << " invalid parent id=" << parent_id;
    return;
  }

  const auto* parent = tree.get_node(parent_id);
  if (parent == nullptr) {
    ++stats.invalid_parent_count;
    LOG_WARNING << "Edge issue: node=" << node_id << " parent missing id=" << parent_id;
    return;
  }

  const auto& child_pos = node->get_position();
  if (!IsValidPos(child_pos)) {
    ++stats.invalid_pos_count;
    LOG_WARNING << "Edge issue: node=" << node_id << " invalid child pos=(" << child_pos.get_x() << "," << child_pos.get_y() << ")";
    return;
  }

  const auto& parent_pos = parent->get_position();
  if (!IsValidPos(parent_pos)) {
    ++stats.missing_edge_count;
    LOG_WARNING << "Edge issue: node=" << node_id << " parent=" << parent_id << " invalid parent pos=(" << parent_pos.get_x() << ","
                << parent_pos.get_y() << ")";
    return;
  }

  const auto dist = icts::geometry::Manhattan(child_pos, parent_pos);
  if (dist == 0) {
    ++stats.zero_edge_count;
    LOG_WARNING << "Edge issue: node=" << node_id << " parent=" << parent_id << " zero-length edge child=(" << child_pos.get_x() << ","
                << child_pos.get_y() << ") parent=(" << parent_pos.get_x() << "," << parent_pos.get_y() << ")";
  }
}

}  // namespace

auto AnalyzeBuiltTopology(const icts::Tree& tree, const std::vector<icts::Pin*>& loads, TopologyArtifacts& artifacts) -> void
{
  std::string error;

  ASSERT_TRUE(common::topology::AnalyzeTopology(tree, loads, artifacts.stats, artifacts.cluster_map, artifacts.centers, error)) << error;

  EXPECT_EQ(artifacts.stats.tree_size, tree.get_size());
  EXPECT_GE(artifacts.stats.leaf_count, 1U);
  EXPECT_LE(artifacts.stats.leaf_count, loads.size());
  EXPECT_EQ(artifacts.cluster_map.size(), loads.size());
}

auto LogTopologySummary(const TopologyStats& stats) -> void
{
  std::ostringstream summary;
  summary << "Tree size=" << stats.tree_size << ", leafs=" << stats.leaf_count << ", leaf_load[min/max/avg]=" << stats.min_leaf_load << "/"
          << stats.max_leaf_load << "/" << stats.avg_leaf_load << ", empty_leafs=" << stats.empty_leaf_count;
  LOG_INFO << summary.str();
}

auto ValidateTreeEdges(const icts::Tree& tree) -> void
{
  EdgeValidationStats stats;

  for (std::size_t id = 0; id < tree.get_size(); ++id) {
    if (id == tree.get_root()) {
      continue;
    }
    ValidateNodeEdge(tree, id, stats);
  }

  EXPECT_EQ(stats.invalid_parent_count, 0U) << "Invalid parent count: " << stats.invalid_parent_count;
  EXPECT_EQ(stats.invalid_pos_count, 0U) << "Invalid child positions: " << stats.invalid_pos_count;
  EXPECT_EQ(stats.missing_edge_count, 0U) << "Missing edges (parent position invalid): " << stats.missing_edge_count;
  EXPECT_EQ(stats.zero_edge_count, 0U) << "Zero-length edges: " << stats.zero_edge_count;
}

}  // namespace icts_test::topology_gen::detail
