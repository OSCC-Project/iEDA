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
 * @file Tree.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-01-16
 * @brief Topology tree data structure for CTS.
 */

#pragma once

#include <limits>
#include <memory>
#include <queue>
#include <vector>

#include "Point.hh"

namespace icts {

class Pin;

class TreeNode
{
 public:
  TreeNode() = default;
  explicit TreeNode(std::size_t id) : _id(id) {}
  ~TreeNode() = default;

  auto get_id() const -> std::size_t { return _id; }
  auto get_parent() const -> std::size_t { return _parent; }
  auto get_children() const -> const std::vector<std::size_t>& { return _children; }
  auto get_children() -> std::vector<std::size_t>& { return _children; }
  auto get_position() const -> const Point<int>& { return _position; }
  auto get_position() -> Point<int>& { return _position; }
  auto get_loads() const -> const std::vector<Pin*>& { return _loads; }
  auto get_loads() -> std::vector<Pin*>& { return _loads; }

  auto set_parent(std::size_t parent) -> void { _parent = parent; }
  auto set_child(std::size_t index, std::size_t child) -> void
  {
    if (index >= _children.size()) {
      _children.resize(index + 1, std::numeric_limits<std::size_t>::max());
    }
    _children[index] = child;
  }
  auto add_child(std::size_t child) -> void { _children.push_back(child); }
  auto isLeaf() const -> bool
  {
    for (auto child : _children) {
      if (child != std::numeric_limits<std::size_t>::max()) {
        return false;
      }
    }
    return true;
  }

 private:
  std::size_t _id = std::numeric_limits<std::size_t>::max();
  std::size_t _parent = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> _children;
  Point<int> _position = Point<int>(-1, -1);
  std::vector<Pin*> _loads;
};

class Tree
{
 public:
  Tree() = default;
  Tree(const Tree&) = delete;
  auto operator=(const Tree&) -> Tree& = delete;
  Tree(Tree&&) = default;
  auto operator=(Tree&&) -> Tree& = default;
  ~Tree() = default;

  auto create_node() -> std::size_t
  {
    std::size_t id = _nodes.size();
    _nodes.push_back(std::make_unique<TreeNode>(id));
    return id;
  }

  auto add_child(std::size_t parent, std::size_t index) -> std::size_t
  {
    auto* parent_node = get_node(parent);
    if (parent_node == nullptr) {
      return std::numeric_limits<std::size_t>::max();
    }
    std::size_t child = create_node();
    auto* child_node = get_node(child);
    child_node->set_parent(parent);
    parent_node->set_child(index, child);
    return child;
  }

  auto add_child(std::size_t parent) -> std::size_t
  {
    auto* parent_node = get_node(parent);
    if (parent_node == nullptr) {
      return std::numeric_limits<std::size_t>::max();
    }
    std::size_t child = create_node();
    auto* child_node = get_node(child);
    child_node->set_parent(parent);
    parent_node->add_child(child);
    return child;
  }

  auto get_node(std::size_t id) -> TreeNode*
  {
    if (id >= _nodes.size()) {
      return nullptr;
    }
    return _nodes[id].get();
  }
  auto get_node(std::size_t id) const -> const TreeNode*
  {
    if (id >= _nodes.size()) {
      return nullptr;
    }
    return _nodes[id].get();
  }

  auto set_root(std::size_t root) -> void { _root = root; }
  auto get_root() const -> std::size_t { return _root; }
  auto get_size() const -> std::size_t { return _nodes.size(); }

  auto levels() const -> std::vector<std::vector<std::size_t>>
  {
    std::vector<std::vector<std::size_t>> result;
    if (_root == std::numeric_limits<std::size_t>::max()) {
      return result;
    }

    std::queue<std::size_t> queue;
    queue.push(_root);
    while (!queue.empty()) {
      std::size_t level_size = queue.size();
      std::vector<std::size_t> level;
      level.reserve(level_size);
      for (std::size_t i = 0; i < level_size; ++i) {
        std::size_t id = queue.front();
        queue.pop();
        level.push_back(id);

        const auto* node_ptr = get_node(id);
        if (node_ptr == nullptr) {
          continue;
        }
        for (auto child : node_ptr->get_children()) {
          if (child != std::numeric_limits<std::size_t>::max()) {
            queue.push(child);
          }
        }
      }
      result.push_back(std::move(level));
    }

    return result;
  }

 private:
  std::vector<std::unique_ptr<TreeNode>> _nodes;
  std::size_t _root = std::numeric_limits<std::size_t>::max();
};

}  // namespace icts
