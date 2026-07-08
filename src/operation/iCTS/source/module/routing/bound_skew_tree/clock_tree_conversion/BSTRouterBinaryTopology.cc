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
 * @file BSTRouterBinaryTopology.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-05-19
 * @brief source-route-tree adaptation helpers for the bounded-skew router.
 */

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <ostream>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include "Geometry.hh"
#include "Log.hh"
#include "Point.hh"
#include "SteinerTree.hh"
#include "bound_skew_tree/BSTRouter.hh"
#include "bound_skew_tree/algorithm/BoundSkewTree.hh"
#include "bound_skew_tree/clock_tree_conversion/BstClockTreeConversion.hh"
#include "bound_skew_tree/component/Components.hh"

namespace icts {
namespace {

using bst::Area;
using bst::BoundSkewTree;
using bst::kLeft;
using bst::kRight;

constexpr int kInvalidCoordinate = -1;
constexpr double kPiElmoreQuadraticFactor = 2.0;
constexpr double kTreeCoordinateScale = 1.0;
constexpr std::size_t kTraversalOrderReserve = 64;

struct BinaryTopologyNode
{
  std::string name;
  Point<int> location = Point<int>(kInvalidCoordinate, kInvalidCoordinate);
  bool is_terminal = false;
  double min_delay = 0.0;
  double max_delay = 0.0;
  double cap_load = 0.0;
  double sub_len = 0.0;
  BSTRoutingRCPattern rc_pattern = BSTRoutingRCPattern::kHV;
  BinaryTopologyNode* left = nullptr;
  BinaryTopologyNode* right = nullptr;
};

struct TerminalElectrical
{
  double pin_cap = 0.0;
  double insertion_delay = 0.0;
};

using BinaryNodeStore = std::vector<std::unique_ptr<BinaryTopologyNode>>;
using AreaStore = std::vector<std::unique_ptr<Area>>;

auto CreateBinaryNode(const std::string& name, const Point<int>& location, bool is_terminal, const TerminalElectrical& electrical,
                      const BSTRoutingConfig& parameters, BinaryNodeStore& owned_nodes) -> BinaryTopologyNode*
{
  auto node = std::make_unique<BinaryTopologyNode>();
  auto* node_ptr = node.get();
  node_ptr->name = name;
  node_ptr->location = location;
  node_ptr->is_terminal = is_terminal;
  node_ptr->rc_pattern = parameters.rc_pattern;
  if (is_terminal) {
    node_ptr->cap_load = electrical.pin_cap;
    node_ptr->min_delay = electrical.insertion_delay;
    node_ptr->max_delay = electrical.insertion_delay;
    if (node_ptr->max_delay - node_ptr->min_delay > parameters.skew_bound) {
      node_ptr->min_delay = node_ptr->max_delay - parameters.skew_bound;
    }
  }
  owned_nodes.push_back(std::move(node));
  return node_ptr;
}

auto CollectChildNodeIds(const BSTRouter::ClockSteinerTreeType& source_route_tree, std::size_t node_id) -> std::vector<std::size_t>
{
  std::vector<std::size_t> child_node_ids;
  const auto* node = source_route_tree.get_node(node_id);
  LOG_FATAL_IF(node == nullptr) << "BST source-route-tree node is null.";
  child_node_ids.reserve(node->child_edge_ids.size());
  for (auto edge_id : node->child_edge_ids) {
    const auto* edge = source_route_tree.get_edge(edge_id);
    LOG_FATAL_IF(edge == nullptr) << "BST source-route-tree edge is null.";
    child_node_ids.push_back(edge->target_node_id);
  }
  return child_node_ids;
}

enum class BuildState : std::uint8_t
{
  kEnter,
  kAfterOnlyChild,
  kAfterLeftChild,
  kAfterRightChild,
  kAfterTrunkLeftChild,
  kAfterTrunkRightChild,
  kAfterThirdChild,
  kAfterLeftCopyLeftChild,
  kAfterLeftCopyRightChild,
  kAfterRightCopyLeftChild,
  kAfterRightCopyRightChild,
};

struct BuildFrame
{
  std::size_t node_id = 0;
  BuildState state = BuildState::kEnter;
  BinaryTopologyNode* node = nullptr;
  std::vector<std::size_t> child_ids;
  BinaryTopologyNode* first_child = nullptr;
  BinaryTopologyNode* second_child = nullptr;
  BinaryTopologyNode* third_child = nullptr;
  BinaryTopologyNode* fourth_child = nullptr;
  BinaryTopologyNode* trunk = nullptr;
  BinaryTopologyNode* left_copy = nullptr;
  BinaryTopologyNode* right_copy = nullptr;
};

auto MakeBuildFrame(std::size_t child_id) -> BuildFrame
{
  BuildFrame frame;
  frame.node_id = child_id;
  return frame;
}

auto PushBuildFrame(std::vector<BuildFrame>& frame_stack, std::size_t child_id) -> void
{
  frame_stack.push_back(MakeBuildFrame(child_id));
}

auto CreateSteinerNode(const Point<int>& location, const BSTRoutingConfig& parameters, std::size_t& next_steiner_id,
                       BinaryNodeStore& owned_nodes) -> BinaryTopologyNode*
{
  return CreateBinaryNode(std::string("steiner_") + std::to_string(next_steiner_id++), location, false, TerminalElectrical{}, parameters,
                          owned_nodes);
}

auto CollectNonNullChildren(BinaryTopologyNode* node) -> std::vector<BinaryTopologyNode*>
{
  std::vector<BinaryTopologyNode*> children;
  children.reserve(2);
  if (node->left != nullptr) {
    children.push_back(node->left);
  }
  if (node->right != nullptr) {
    children.push_back(node->right);
  }
  return children;
}

auto EnterBuildFrame(BuildFrame& frame, const BSTRouter::ClockSteinerTreeType& source_route_tree, const BSTRoutingConfig& parameters,
                     std::size_t& next_steiner_id, BinaryNodeStore& owned_nodes, BinaryTopologyNode*& return_value,
                     std::vector<BuildFrame>& frame_stack) -> void
{
  const auto* tree_node = source_route_tree.get_node(frame.node_id);
  LOG_FATAL_IF(tree_node == nullptr) << "BST source-route-tree node is null.";

  auto node_name = tree_node->name.empty() ? std::string("steiner_") + std::to_string(frame.node_id) : tree_node->name;
  auto electrical = TerminalElectrical{.pin_cap = tree_node->pin_cap, .insertion_delay = tree_node->insertion_delay};
  frame.node = CreateBinaryNode(node_name, tree_node->location, tree_node->is_terminal, electrical, parameters, owned_nodes);
  frame.child_ids = CollectChildNodeIds(source_route_tree, frame.node_id);

  if (frame.child_ids.size() > 2) {
    std::ranges::sort(frame.child_ids, [&](std::size_t lhs_id, std::size_t rhs_id) -> bool {
      const auto* lhs = source_route_tree.get_node(lhs_id);
      const auto* rhs = source_route_tree.get_node(rhs_id);
      if (lhs == nullptr || rhs == nullptr) {
        LOG_FATAL << "BST source-route-tree child node is null.";
        return false;
      }
      return geometry::Manhattan(tree_node->location, lhs->location) < geometry::Manhattan(tree_node->location, rhs->location);
    });
  }

  switch (frame.child_ids.size()) {
    case 0:
      LOG_FATAL_IF(!frame.node->is_terminal) << "BST source-route-tree Steiner node has no children.";
      return_value = frame.node;
      frame_stack.pop_back();
      return;
    case 1:
      frame.state = BuildState::kAfterOnlyChild;
      PushBuildFrame(frame_stack, frame.child_ids.front());
      return;
    case 2:
      frame.state = BuildState::kAfterLeftChild;
      PushBuildFrame(frame_stack, frame.child_ids.at(kLeft));
      return;
    case 3:
      frame.trunk = CreateSteinerNode(frame.node->location, parameters, next_steiner_id, owned_nodes);
      frame.state = BuildState::kAfterTrunkLeftChild;
      PushBuildFrame(frame_stack, frame.child_ids.at(0));
      return;
    case 4:
      frame.left_copy = CreateSteinerNode(frame.node->location, parameters, next_steiner_id, owned_nodes);
      frame.state = BuildState::kAfterLeftCopyLeftChild;
      PushBuildFrame(frame_stack, frame.child_ids.at(0));
      return;
    default:
      LOG_FATAL << "BST source-route-tree node child size " << frame.child_ids.size() << " is unsupported.";
  }
}

auto ExitSingleChildFrame(BuildFrame& frame, std::size_t& next_steiner_id, BinaryNodeStore& owned_nodes, BinaryTopologyNode*& return_value,
                          std::vector<BuildFrame>& frame_stack) -> void
{
  auto* child = return_value;
  LOG_FATAL_IF(child == nullptr) << "BST source-route-tree child build returned null.";

  if (frame.node->is_terminal) {
    auto copy = std::make_unique<BinaryTopologyNode>(*frame.node);
    auto* copy_ptr = copy.get();
    copy_ptr->name = std::string("steiner_") + std::to_string(next_steiner_id++);
    copy_ptr->is_terminal = false;
    copy_ptr->left = child;
    copy_ptr->right = frame.node;
    owned_nodes.push_back(std::move(copy));
    frame.node->left = nullptr;
    frame.node->right = nullptr;
    return_value = copy_ptr;
    frame_stack.pop_back();
    return;
  }

  auto grandchildren = CollectNonNullChildren(child);
  LOG_FATAL_IF(grandchildren.empty()) << "BST source-route-tree single-child Steiner node flatten result is empty.";

  frame.node->left = grandchildren.front();
  frame.node->right = grandchildren.size() > 1 ? grandchildren.at(kRight) : nullptr;
  return_value = frame.node;
  frame_stack.pop_back();
}

auto ProcessBuildFrame(BuildFrame& frame, const BSTRouter::ClockSteinerTreeType& source_route_tree, const BSTRoutingConfig& parameters,
                       std::size_t& next_steiner_id, BinaryNodeStore& owned_nodes, BinaryTopologyNode*& return_value,
                       std::vector<BuildFrame>& frame_stack) -> void
{
  switch (frame.state) {
    case BuildState::kEnter:
      EnterBuildFrame(frame, source_route_tree, parameters, next_steiner_id, owned_nodes, return_value, frame_stack);
      return;
    case BuildState::kAfterOnlyChild:
      ExitSingleChildFrame(frame, next_steiner_id, owned_nodes, return_value, frame_stack);
      return;
    case BuildState::kAfterLeftChild:
      frame.first_child = return_value;
      frame.state = BuildState::kAfterRightChild;
      PushBuildFrame(frame_stack, frame.child_ids.at(kRight));
      return;
    case BuildState::kAfterRightChild:
      frame.second_child = return_value;
      frame.node->left = frame.first_child;
      frame.node->right = frame.second_child;
      return_value = frame.node;
      frame_stack.pop_back();
      return;
    case BuildState::kAfterTrunkLeftChild:
      frame.first_child = return_value;
      frame.state = BuildState::kAfterTrunkRightChild;
      PushBuildFrame(frame_stack, frame.child_ids.at(1));
      return;
    case BuildState::kAfterTrunkRightChild:
      frame.second_child = return_value;
      frame.trunk->left = frame.first_child;
      frame.trunk->right = frame.second_child;
      frame.node->left = frame.trunk;
      frame.state = BuildState::kAfterThirdChild;
      PushBuildFrame(frame_stack, frame.child_ids.at(2));
      return;
    case BuildState::kAfterThirdChild:
      frame.third_child = return_value;
      frame.node->right = frame.third_child;
      return_value = frame.node;
      frame_stack.pop_back();
      return;
    case BuildState::kAfterLeftCopyLeftChild:
      frame.first_child = return_value;
      frame.state = BuildState::kAfterLeftCopyRightChild;
      PushBuildFrame(frame_stack, frame.child_ids.at(1));
      return;
    case BuildState::kAfterLeftCopyRightChild:
      frame.second_child = return_value;
      frame.left_copy->left = frame.first_child;
      frame.left_copy->right = frame.second_child;
      frame.right_copy = CreateSteinerNode(frame.node->location, parameters, next_steiner_id, owned_nodes);
      frame.state = BuildState::kAfterRightCopyLeftChild;
      PushBuildFrame(frame_stack, frame.child_ids.at(2));
      return;
    case BuildState::kAfterRightCopyLeftChild:
      frame.third_child = return_value;
      frame.state = BuildState::kAfterRightCopyRightChild;
      PushBuildFrame(frame_stack, frame.child_ids.at(3));
      return;
    case BuildState::kAfterRightCopyRightChild:
      frame.fourth_child = return_value;
      frame.right_copy->left = frame.third_child;
      frame.right_copy->right = frame.fourth_child;
      frame.node->left = frame.left_copy;
      frame.node->right = frame.right_copy;
      return_value = frame.node;
      frame_stack.pop_back();
      return;
  }
}

auto BuildBinaryTopologyNode(const BSTRouter::ClockSteinerTreeType& source_route_tree, std::size_t node_id,
                             const BSTRoutingConfig& parameters, std::size_t& next_steiner_id, BinaryNodeStore& owned_nodes)
    -> BinaryTopologyNode*
{
  std::vector<BuildFrame> frame_stack;
  PushBuildFrame(frame_stack, node_id);
  BinaryTopologyNode* return_value = nullptr;

  while (!frame_stack.empty()) {
    ProcessBuildFrame(frame_stack.back(), source_route_tree, parameters, next_steiner_id, owned_nodes, return_value, frame_stack);
  }

  return return_value;
}

auto FinalizeElectricalState(BinaryTopologyNode* node, const BSTRoutingConfig& parameters) -> void
{
  if (node == nullptr) {
    return;
  }

  std::vector<BinaryTopologyNode*> traversal_stack;
  traversal_stack.push_back(node);
  std::vector<BinaryTopologyNode*> traversal_order;
  traversal_order.reserve(kTraversalOrderReserve);

  while (!traversal_stack.empty()) {
    auto* current = traversal_stack.back();
    traversal_stack.pop_back();
    if (current == nullptr) {
      continue;
    }
    traversal_order.push_back(current);
    if (current->left != nullptr) {
      traversal_stack.push_back(current->left);
    }
    if (current->right != nullptr) {
      traversal_stack.push_back(current->right);
    }
  }

  for (auto* current : std::ranges::reverse_view(traversal_order)) {
    if (current->left == nullptr && current->right == nullptr) {
      continue;
    }

    auto children = CollectNonNullChildren(current);

    double cap_load = 0.0;
    for (auto* child : children) {
      auto wire_cap = parameters.unit_h_cap * child->sub_len;
      cap_load += child->cap_load + wire_cap;
    }
    current->cap_load = cap_load;

    double max_delay = 0.0;
    double min_delay = std::numeric_limits<double>::max();
    for (auto* child : children) {
      auto delay = parameters.unit_h_res * child->sub_len
                   * ((parameters.unit_h_cap * child->sub_len / kPiElmoreQuadraticFactor) + child->cap_load);
      max_delay = std::max(max_delay, child->max_delay + delay);
      min_delay = std::min(min_delay, child->min_delay + delay);
    }
    current->max_delay = max_delay;
    current->min_delay = min_delay == std::numeric_limits<double>::max() ? 0.0 : min_delay;
    if (current->max_delay - current->min_delay > parameters.skew_bound) {
      current->min_delay = current->max_delay - parameters.skew_bound;
    }
  }
}

auto BuildAreaTree(BinaryTopologyNode* node, const BSTRoutingConfig& parameters, AreaStore& owned_areas) -> Area*
{
  if (node == nullptr) {
    LOG_FATAL << "BST binary topology node is null.";
    return nullptr;
  }

  enum class BuildAreaState : std::uint8_t
  {
    kEnter,
    kAfterLeftChild,
    kAfterRightChild,
  };

  struct BuildAreaFrame
  {
    BinaryTopologyNode* node = nullptr;
    Area* area = nullptr;
    Area* left_area = nullptr;
    Area* right_area = nullptr;
    BuildAreaState state = BuildAreaState::kEnter;
  };

  auto create_area = [&](BinaryTopologyNode* current_node) -> Area* {
    auto area = std::make_unique<Area>(current_node->name, kTreeCoordinateScale * current_node->location.get_x() / parameters.dbu_per_um,
                                       kTreeCoordinateScale * current_node->location.get_y() / parameters.dbu_per_um,
                                       current_node->cap_load, current_node->min_delay, current_node->max_delay,
                                       current_node->sub_len / parameters.dbu_per_um, current_node->rc_pattern, current_node->is_terminal);
    auto* area_ptr = area.get();
    owned_areas.push_back(std::move(area));
    return area_ptr;
  };

  Area* return_area = nullptr;
  std::vector<BuildAreaFrame> frame_stack;
  frame_stack.push_back(
      BuildAreaFrame{.node = node, .area = nullptr, .left_area = nullptr, .right_area = nullptr, .state = BuildAreaState::kEnter});

  while (!frame_stack.empty()) {
    auto& frame = frame_stack.back();
    switch (frame.state) {
      case BuildAreaState::kEnter:
        frame.area = create_area(frame.node);
        if (frame.node->left == nullptr && frame.node->right == nullptr) {
          return_area = frame.area;
          frame_stack.pop_back();
          break;
        }
        if (frame.node->left == nullptr || frame.node->right == nullptr) {
          LOG_FATAL << "BST area adaptation expects binary topology.";
          return frame.area;
        }
        frame.state = BuildAreaState::kAfterLeftChild;
        frame_stack.push_back(BuildAreaFrame{
            .node = frame.node->left, .area = nullptr, .left_area = nullptr, .right_area = nullptr, .state = BuildAreaState::kEnter});
        break;
      case BuildAreaState::kAfterLeftChild:
        frame.left_area = return_area;
        frame.state = BuildAreaState::kAfterRightChild;
        frame_stack.push_back(BuildAreaFrame{
            .node = frame.node->right, .area = nullptr, .left_area = nullptr, .right_area = nullptr, .state = BuildAreaState::kEnter});
        break;
      case BuildAreaState::kAfterRightChild:
        frame.right_area = return_area;
        LOG_FATAL_IF(frame.area == nullptr || frame.left_area == nullptr || frame.right_area == nullptr)
            << "BST area adaptation produced null child linkage.";
        frame.area->set_left(frame.left_area);
        frame.area->set_right(frame.right_area);
        frame.left_area->set_parent(frame.area);
        frame.right_area->set_parent(frame.area);
        return_area = frame.area;
        frame_stack.pop_back();
        break;
    }
  }

  return return_area;
}

}  // namespace

auto BuildBstFromSourceRouteTree(const BSTRouter::ClockSteinerTreeType& source_route_tree, BSTRoutingConfig parameters)
    -> BSTRouter::ClockSteinerTreeType
{
  parameters.topology_mode = BSTRoutingTopologyMode::kSourceRouteTree;

  const BSTRouter::ClockSteinerTreeType empty_tree;
  if (source_route_tree.node_count() == 0) {
    return empty_tree;
  }

  LOG_FATAL_IF(!source_route_tree.validate()) << "Source route tree for BST routing is invalid.";

  std::size_t next_steiner_id = source_route_tree.node_count();
  BinaryNodeStore owned_nodes;
  auto root_id = source_route_tree.get_root();
  auto* root = BuildBinaryTopologyNode(source_route_tree, root_id, parameters, next_steiner_id, owned_nodes);
  FinalizeElectricalState(root, parameters);

  AreaStore owned_areas;
  auto* root_area = BuildAreaTree(root, parameters, owned_areas);
  BoundSkewTree solver(std::move(owned_areas), root_area, parameters);
  solver.run();
  return ExportBstClockTree(solver.get_root(), parameters);
}

}  // namespace icts
