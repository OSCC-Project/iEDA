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
 * @file TopologyAnalysis.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Shared topology analysis utilities for iCTS tests.
 */

#include "common/topology/TopologyAnalysis.hh"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "Pin.hh"
#include "Point.hh"
#include "common/dataset/TestDataset.hh"
#include "database/spatial/Tree.hh"

namespace icts_test::common::topology {
namespace {

constexpr int kInvalidCoord = -1;
constexpr std::size_t kInvalidNodeId = std::numeric_limits<std::size_t>::max();

auto ComputeLoadCentroid(const std::vector<icts::Pin*>& loads) -> icts::Point<int>
{
  if (loads.empty()) {
    return {kInvalidCoord, kInvalidCoord};
  }
  long long sum_x = 0;
  long long sum_y = 0;
  for (const auto* pin : loads) {
    if (pin == nullptr) {
      continue;
    }
    const auto& location = pin->get_location();
    sum_x += location.get_x();
    sum_y += location.get_y();
  }
  return {static_cast<int>(sum_x / static_cast<long long>(loads.size())), static_cast<int>(sum_y / static_cast<long long>(loads.size()))};
}

auto CollectLoadsUnderNode(const icts::Tree& tree, std::size_t node_id, std::vector<icts::Pin*>& collected_loads) -> void
{
  std::queue<std::size_t> pending_nodes;
  pending_nodes.push(node_id);

  while (!pending_nodes.empty()) {
    const std::size_t current_node_id = pending_nodes.front();
    pending_nodes.pop();

    const auto* node = tree.get_node(current_node_id);
    if (node == nullptr) {
      continue;
    }

    for (auto* pin : node->get_loads()) {
      collected_loads.push_back(pin);
    }
    for (auto child_id : node->get_children()) {
      if (child_id != kInvalidNodeId) {
        pending_nodes.push(child_id);
      }
    }
  }
}

}  // namespace

auto AnalyzeTopology(const icts::Tree& tree, const std::vector<icts::Pin*>& loads, TopologyStats& stats,
                     std::unordered_map<const icts::Pin*, std::size_t>& cluster_map, std::vector<icts::Point<int>>& centers,
                     std::string& error) -> bool
{
  stats = {};
  cluster_map.clear();
  centers.clear();

  if (tree.get_size() == 0) {
    error = "tree is empty";
    return false;
  }
  if (tree.get_root() == kInvalidNodeId) {
    error = "tree root is invalid";
    return false;
  }

  stats.tree_size = tree.get_size();
  std::vector<std::size_t> leaf_ids;
  leaf_ids.reserve(tree.get_size());
  for (std::size_t id = 0; id < tree.get_size(); ++id) {
    const auto* node = tree.get_node(id);
    if (node == nullptr) {
      continue;
    }
    if (node->isLeaf()) {
      leaf_ids.push_back(id);
    }
  }

  if (leaf_ids.empty()) {
    error = "tree has no leaves";
    return false;
  }

  stats.leaf_count = leaf_ids.size();
  stats.min_leaf_load = std::numeric_limits<std::size_t>::max();
  stats.max_leaf_load = 0;
  std::size_t total_loads = 0;

  cluster_map.reserve(loads.size());
  centers.reserve(leaf_ids.size());

  for (std::size_t index = 0; index < leaf_ids.size(); ++index) {
    const auto* node = tree.get_node(leaf_ids.at(index));
    if (node == nullptr) {
      continue;
    }
    const std::size_t load_count = node->get_loads().size();
    total_loads += load_count;
    if (load_count == 0) {
      ++stats.empty_leaf_count;
    }
    stats.min_leaf_load = std::min(stats.min_leaf_load, load_count);
    stats.max_leaf_load = std::max(stats.max_leaf_load, load_count);
    centers.push_back(node->get_position());

    for (const auto* pin : node->get_loads()) {
      cluster_map[pin] = index;
    }
  }

  if (stats.min_leaf_load == std::numeric_limits<std::size_t>::max()) {
    stats.min_leaf_load = 0;
  }
  stats.avg_leaf_load = stats.leaf_count == 0 ? 0.0 : static_cast<double>(total_loads) / static_cast<double>(stats.leaf_count);

  if (total_loads != loads.size()) {
    std::ostringstream stream;
    stream << "load count mismatch: expected " << loads.size() << ", got " << total_loads;
    error = stream.str();
    return false;
  }

  if (cluster_map.size() != loads.size()) {
    std::ostringstream stream;
    stream << "cluster map size mismatch: expected " << loads.size() << ", got " << cluster_map.size();
    error = stream.str();
    return false;
  }
  return true;
}

auto AnalyzeFirstLevelClusters(const icts::Tree& tree, const std::vector<icts::Pin*>& loads,
                               std::unordered_map<const icts::Pin*, std::size_t>& cluster_map, std::vector<icts::Point<int>>& centers,
                               std::string& error) -> bool
{
  cluster_map.clear();
  centers.clear();

  if (tree.get_size() == 0) {
    error = "tree is empty";
    return false;
  }
  const std::size_t root_id = tree.get_root();
  if (root_id == kInvalidNodeId) {
    error = "tree root is invalid";
    return false;
  }

  const auto* root_node = tree.get_node(root_id);
  if (root_node == nullptr) {
    error = "root node is null";
    return false;
  }

  const auto& children = root_node->get_children();
  std::vector<std::size_t> valid_children;
  for (auto child_id : children) {
    if (child_id != kInvalidNodeId) {
      valid_children.push_back(child_id);
    }
  }

  if (valid_children.empty()) {
    for (const auto* pin : loads) {
      cluster_map[pin] = 0;
    }
    centers.push_back(ComputeLoadCentroid(loads));
    return true;
  }

  cluster_map.reserve(loads.size());
  centers.reserve(valid_children.size());

  for (std::size_t index = 0; index < valid_children.size(); ++index) {
    std::vector<icts::Pin*> child_loads;
    CollectLoadsUnderNode(tree, valid_children.at(index), child_loads);

    for (const auto* pin : child_loads) {
      cluster_map[pin] = index;
    }
    centers.push_back(ComputeLoadCentroid(child_loads));
  }

  if (cluster_map.size() != loads.size()) {
    std::ostringstream stream;
    stream << "first-level cluster map size mismatch: expected " << loads.size() << ", got " << cluster_map.size();
    error = stream.str();
    return false;
  }

  return true;
}

}  // namespace icts_test::common::topology
