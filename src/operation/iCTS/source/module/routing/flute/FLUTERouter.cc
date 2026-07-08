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
 * @file FLUTERouter.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-03-15
 * @brief FLUTE router adapter implementation for Steiner tree construction.
 */

#include "FLUTERouter.hh"

#include <glog/logging.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Log.hh"
#include "Point.hh"
#include "SaltPinBuilder.hh"
#include "geometry/Geometry.hh"
#include "salt/base/flute.h"
#include "salt/base/net.h"
#include "salt/base/tree.h"

namespace icts {

auto FLUTERouter::buildTree(const ClockTerminal& driver_terminal, const std::vector<ClockTerminal>& load_terminals)
    -> FLUTERouter::ClockSteinerTreeType
{
  ClockSteinerTreeType clock_tree;
  auto root_id
      = clock_tree.addNode(driver_terminal.name, driver_terminal.location, true, driver_terminal.pin_cap, driver_terminal.insertion_delay);
  clock_tree.setRoot(root_id);

  std::unordered_map<int, std::size_t> salt_to_tree_id;
  salt_to_tree_id[0] = root_id;
  for (std::size_t i = 0; i < load_terminals.size(); ++i) {
    const auto& terminal = load_terminals.at(i);
    auto node_id = clock_tree.addNode(terminal.name, terminal.location, true, terminal.pin_cap, terminal.insertion_delay);
    salt_to_tree_id[static_cast<int>(i + 1)] = node_id;
  }

  if (load_terminals.empty()) {
    return clock_tree;
  }

  auto salt_pins = BuildSaltPins(driver_terminal, load_terminals);
  salt::Net net;
  net.init(0, "FLUTE", salt_pins);

  salt::Tree tree;
  salt::FluteBuilder flute_builder;
  flute_builder.Run(net, tree);
  tree.UpdateId();

  auto ensure_node = [&](const std::shared_ptr<salt::TreeNode>& salt_node) -> std::size_t {
    auto iter = salt_to_tree_id.find(salt_node->id);
    if (iter != salt_to_tree_id.end()) {
      return iter->second;
    }

    auto node_id = clock_tree.addNode(std::string("steiner_") + std::to_string(salt_node->id),
                                      Point<int>(salt_node->loc.x, salt_node->loc.y), false, 0.0, 0.0);
    salt_to_tree_id[salt_node->id] = node_id;
    return node_id;
  };

  auto source = tree.source;
  auto source_id = ensure_node(source);
  clock_tree.setRoot(source_id);

  auto connect_node_func = [&](const std::shared_ptr<salt::TreeNode>& salt_node) -> void {
    auto current_id = ensure_node(salt_node);
    if (salt_node->id == source->id) {
      return;
    }

    auto parent_id = ensure_node(salt_node->parent);
    const auto* current_node = clock_tree.get_node(current_id);
    const auto* parent_node = clock_tree.get_node(parent_id);
    LOG_FATAL_IF(current_node == nullptr || parent_node == nullptr) << "FLUTE clock routing tree node is null.";

    const auto distance = geometry::Manhattan(parent_node->location, current_node->location);
    LOG_FATAL_IF(distance < 0) << "FLUTE embedded edge distance is negative.";
    auto edge_id = clock_tree.addEdge(parent_id, current_id, distance, distance);
    LOG_FATAL_IF(edge_id == ClockSteinerTreeType::kInvalidId) << "Failed to add edge when building FLUTE ClockSteinerTree.";
  };
  salt::TreeNode::preOrder(source, connect_node_func);

  LOG_FATAL_IF(!clock_tree.validate()) << "Constructed FLUTE ClockSteinerTree is invalid.";
  return clock_tree;
}

}  // namespace icts
