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
 * @file BSTRouterExport.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief ClockSteinerTree export helpers for the bounded-skew router adapter.
 */

#include <glog/logging.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Geometry.hh"
#include "Log.hh"
#include "Point.hh"
#include "SteinerTree.hh"
#include "bound_skew_tree/BSTRouter.hh"
#include "bound_skew_tree/clock_tree_conversion/BstClockTreeConversion.hh"
#include "bound_skew_tree/component/Components.hh"

namespace icts {
namespace {

using bst::Area;
using bst::kLeft;
using bst::kRight;

auto ExportAreaNode(const Area* area, const BSTRoutingConfig& parameters, BSTRouter::ClockSteinerTreeType& tree,
                    std::unordered_map<const Area*, std::size_t>& area_to_node_id) -> std::size_t
{
  enum class ExportState : std::uint8_t
  {
    kEnter,
    kAfterLeftChild,
    kAfterRightChild,
  };

  struct ExportFrame
  {
    const Area* area = nullptr;
    std::size_t node_id = BSTRouter::ClockSteinerTreeType::kInvalidId;
    std::size_t child_id = BSTRouter::ClockSteinerTreeType::kInvalidId;
    ExportState state = ExportState::kEnter;
  };

  auto add_edge = [&](std::size_t parent_id, std::size_t child_id, const Area* parent_area, std::size_t side) -> void {
    const auto* parent_node = tree.get_node(parent_id);
    const auto* child_node = tree.get_node(child_id);
    LOG_FATAL_IF(parent_node == nullptr || child_node == nullptr) << "BST exported node is null.";

    const auto distance = geometry::Manhattan(parent_node->location, child_node->location);
    LOG_FATAL_IF(distance < 0) << "BST embedded edge distance is negative.";

    const auto routed_distance = static_cast<int>(std::lround(parent_area->get_edge_len(side) * parameters.dbu_per_um));
    LOG_FATAL_IF(routed_distance < distance) << "BST routed edge length is shorter than embedded Manhattan distance.";

    auto edge_id = tree.addEdge(parent_id, child_id, distance, routed_distance);
    LOG_FATAL_IF(edge_id == BSTRouter::ClockSteinerTreeType::kInvalidId) << "Failed to add edge when exporting BST ClockSteinerTree.";
  };

  std::size_t return_node_id = BSTRouter::ClockSteinerTreeType::kInvalidId;
  std::vector<ExportFrame> frame_stack;
  frame_stack.push_back(ExportFrame{.area = area});

  while (!frame_stack.empty()) {
    auto& frame = frame_stack.back();
    switch (frame.state) {
      case ExportState::kEnter: {
        auto iter = area_to_node_id.find(frame.area);
        if (iter != area_to_node_id.end()) {
          return_node_id = iter->second;
          frame_stack.pop_back();
          break;
        }

        const auto& location = frame.area->get_location();
        auto insertion_delay = frame.area->is_fixed_terminal() ? location.max : 0.0;
        auto pin_cap = frame.area->is_fixed_terminal() ? frame.area->get_cap_load() : 0.0;
        frame.node_id = tree.addNode(frame.area->get_name(),
                                     Point<int>(static_cast<int>(std::lround(location.x * parameters.dbu_per_um)),
                                                static_cast<int>(std::lround(location.y * parameters.dbu_per_um))),
                                     frame.area->is_fixed_terminal(), pin_cap, insertion_delay);
        LOG_FATAL_IF(frame.node_id == BSTRouter::ClockSteinerTreeType::kInvalidId)
            << "Failed to add node when exporting BST ClockSteinerTree.";
        area_to_node_id[frame.area] = frame.node_id;

        if (frame.area->get_left() == nullptr) {
          frame.state = ExportState::kAfterLeftChild;
          break;
        }
        frame.state = ExportState::kAfterLeftChild;
        frame_stack.push_back(ExportFrame{.area = frame.area->get_left()});
        break;
      }
      case ExportState::kAfterLeftChild:
        if (frame.area->get_left() != nullptr) {
          frame.child_id = return_node_id;
          add_edge(frame.node_id, frame.child_id, frame.area, kLeft);
        }
        if (frame.area->get_right() == nullptr) {
          return_node_id = frame.node_id;
          frame_stack.pop_back();
          break;
        }
        frame.state = ExportState::kAfterRightChild;
        frame_stack.push_back(ExportFrame{.area = frame.area->get_right()});
        break;
      case ExportState::kAfterRightChild:
        frame.child_id = return_node_id;
        add_edge(frame.node_id, frame.child_id, frame.area, kRight);
        return_node_id = frame.node_id;
        frame_stack.pop_back();
        break;
    }
  }

  return return_node_id;
}

}  // namespace

auto ExportBstClockTree(const bst::Area* root, const BSTRoutingConfig& parameters) -> BSTRouter::ClockSteinerTreeType
{
  BSTRouter::ClockSteinerTreeType tree;
  LOG_FATAL_IF(root == nullptr) << "BST root area is null when exporting ClockSteinerTree.";

  std::unordered_map<const bst::Area*, std::size_t> area_to_node_id;
  auto root_id = ExportAreaNode(root, parameters, tree, area_to_node_id);
  tree.setRoot(root_id);
  LOG_FATAL_IF(!tree.validate()) << "Constructed BST ClockSteinerTree is invalid.";
  return tree;
}

}  // namespace icts
