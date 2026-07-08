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
 * @file SteinerTree.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-03-11
 * @brief Point-based Steiner tree data structures for iCTS routing.
 */

#pragma once

#include <cstddef>
#include <limits>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Point.hh"

namespace icts {

template <typename T = int>
struct SteinerNode
{
  static constexpr std::size_t kInvalidId = std::numeric_limits<std::size_t>::max();

  std::size_t id = kInvalidId;
  std::string name = "";
  Point<T> location;
  bool is_terminal = false;
  std::size_t parent_edge_id = kInvalidId;
  std::vector<std::size_t> child_edge_ids;
};

template <typename T = int>
struct ClockSteinerNode : public SteinerNode<T>
{
  double pin_cap = 0.0;
  double insertion_delay = 0.0;
};

template <typename T = int>
struct SteinerEdge
{
  static constexpr std::size_t kInvalidId = std::numeric_limits<std::size_t>::max();

  SteinerEdge() = default;
  SteinerEdge(std::size_t edge_id, std::size_t source_id, std::size_t target_id, const T& edge_distance)
      : id(edge_id), source_node_id(source_id), target_node_id(target_id), distance(edge_distance)
  {
  }

  std::size_t id = kInvalidId;
  std::size_t source_node_id = kInvalidId;
  std::size_t target_node_id = kInvalidId;
  T distance{};  // Embedded geometric edge distance in coordinate type T.
};

template <typename T = int, typename NodeT = SteinerNode<T>, typename EdgeT = SteinerEdge<T>>
class SteinerTree
{
 public:
  using CoordType = T;
  using NodeType = NodeT;
  using EdgeType = EdgeT;

  static constexpr std::size_t kInvalidId = std::numeric_limits<std::size_t>::max();

  SteinerTree() = default;
  SteinerTree(const SteinerTree&) = default;
  auto operator=(const SteinerTree&) -> SteinerTree& = default;
  SteinerTree(SteinerTree&&) = default;
  auto operator=(SteinerTree&&) -> SteinerTree& = default;
  virtual ~SteinerTree() = default;

  auto reserveNodes(std::size_t n) -> void { _nodes.reserve(n); }
  auto reserveEdges(std::size_t n) -> void { _edges.reserve(n); }

  virtual auto addNode(const std::string& name, const Point<T>& location, bool is_terminal = false) -> std::size_t
  {
    auto node = buildNode(name, location, is_terminal);
    return appendNode(std::move(node));
  }

  auto setRoot(std::size_t node_id) -> void
  {
    if (node_id < _nodes.size()) {
      _root_node_id = node_id;
    }
  }

  template <typename... EdgeArgs>
  auto addEdge(std::size_t source_node_id, std::size_t target_node_id, const T& distance, EdgeArgs&&... edge_args) -> std::size_t
  {
    if (source_node_id >= _nodes.size() || target_node_id >= _nodes.size() || source_node_id == target_node_id) {
      return kInvalidId;
    }

    auto& target_node = _nodes[target_node_id];
    if (target_node.parent_edge_id != kInvalidId) {
      return kInvalidId;
    }

    std::size_t edge_id = _edges.size();
    _edges.emplace_back(edge_id, source_node_id, target_node_id, distance, std::forward<EdgeArgs>(edge_args)...);
    _nodes[source_node_id].child_edge_ids.push_back(edge_id);
    target_node.parent_edge_id = edge_id;
    return edge_id;
  }

  auto get_node(std::size_t id) -> NodeType*
  {
    if (id >= _nodes.size()) {
      return nullptr;
    }
    return &_nodes[id];
  }
  auto get_node(std::size_t id) const -> const NodeType*
  {
    if (id >= _nodes.size()) {
      return nullptr;
    }
    return &_nodes[id];
  }

  auto get_edge(std::size_t id) -> EdgeType*
  {
    if (id >= _edges.size()) {
      return nullptr;
    }
    return &_edges[id];
  }
  auto get_edge(std::size_t id) const -> const EdgeType*
  {
    if (id >= _edges.size()) {
      return nullptr;
    }
    return &_edges[id];
  }

  auto findNode(const std::string& name) -> NodeType*
  {
    auto iter = _node_name_to_id.find(name);
    return iter == _node_name_to_id.end() ? nullptr : get_node(iter->second);
  }
  auto findNode(const std::string& name) const -> const NodeType*
  {
    auto iter = _node_name_to_id.find(name);
    return iter == _node_name_to_id.end() ? nullptr : get_node(iter->second);
  }
  auto get_node_name_map() const -> const std::unordered_map<std::string, std::size_t>& { return _node_name_to_id; }

  template <typename... EdgeArgs>
  auto addEdge(const std::string& source_node_name, const std::string& target_node_name, const T& distance, EdgeArgs&&... edge_args)
      -> std::size_t
  {
    auto source_iter = _node_name_to_id.find(source_node_name);
    auto target_iter = _node_name_to_id.find(target_node_name);
    if (source_iter == _node_name_to_id.end() || target_iter == _node_name_to_id.end()) {
      return kInvalidId;
    }
    return addEdge(source_iter->second, target_iter->second, distance, std::forward<EdgeArgs>(edge_args)...);
  }

  auto get_nodes() const -> const std::vector<NodeType>& { return _nodes; }
  auto get_edges() const -> const std::vector<EdgeType>& { return _edges; }
  auto get_edges() -> std::vector<EdgeType>& { return _edges; }

  auto get_root() const -> std::size_t { return _root_node_id; }
  auto node_count() const -> std::size_t { return _nodes.size(); }
  auto edge_count() const -> std::size_t { return _edges.size(); }

  auto is_root(std::size_t node_id) const -> bool { return node_id == _root_node_id; }

  auto is_leaf(std::size_t node_id) const -> bool
  {
    const auto* node = get_node(node_id);
    return node != nullptr && node->child_edge_ids.empty();
  }

  auto validate() const -> bool
  {
    if (_nodes.empty()) {
      return _root_node_id == kInvalidId && _edges.empty();
    }
    if (_root_node_id >= _nodes.size()) {
      return false;
    }
    if (_edges.size() != _nodes.size() - 1) {
      return false;
    }

    std::vector<std::size_t> indegree(_nodes.size(), 0);
    for (std::size_t edge_id = 0; edge_id < _edges.size(); ++edge_id) {
      const auto& edge = _edges[edge_id];
      if (edge.id != edge_id || edge.source_node_id >= _nodes.size() || edge.target_node_id >= _nodes.size()
          || edge.source_node_id == edge.target_node_id) {
        return false;
      }
      ++indegree[edge.target_node_id];
    }

    std::unordered_map<std::string, std::size_t> validated_name_to_id;
    for (std::size_t node_id = 0; node_id < _nodes.size(); ++node_id) {
      const auto& node = _nodes[node_id];
      if (node.id != node_id || node.name.empty()) {
        return false;
      }
      if (validated_name_to_id.contains(node.name)) {
        return false;
      }
      validated_name_to_id[node.name] = node_id;
      if (is_root(node_id)) {
        if (node.parent_edge_id != kInvalidId || indegree[node_id] != 0) {
          return false;
        }
      } else {
        if (node.parent_edge_id >= _edges.size() || indegree[node_id] != 1) {
          return false;
        }
        if (_edges[node.parent_edge_id].target_node_id != node_id) {
          return false;
        }
      }
      for (auto edge_id : node.child_edge_ids) {
        if (edge_id >= _edges.size()) {
          return false;
        }
        if (_edges[edge_id].source_node_id != node_id) {
          return false;
        }
      }
    }

    std::vector<bool> visited(_nodes.size(), false);
    std::queue<std::size_t> queue;
    queue.push(_root_node_id);
    visited[_root_node_id] = true;
    std::size_t visited_count = 0;

    while (!queue.empty()) {
      auto node_id = queue.front();
      queue.pop();
      ++visited_count;

      const auto& node = _nodes[node_id];
      for (auto edge_id : node.child_edge_ids) {
        auto child_id = _edges[edge_id].target_node_id;
        if (visited[child_id]) {
          return false;
        }
        visited[child_id] = true;
        queue.push(child_id);
      }
    }

    return visited_count == _nodes.size() && validated_name_to_id == _node_name_to_id;
  }

 protected:
  virtual auto buildNode(const std::string& name, const Point<T>& location, bool is_terminal) -> NodeType
  {
    NodeType node;
    node.name = name;
    node.location = location;
    node.is_terminal = is_terminal;
    return node;
  }

  auto appendNode(NodeType node) -> std::size_t
  {
    if (node.name.empty() || _node_name_to_id.contains(node.name)) {
      return kInvalidId;
    }

    const std::size_t id = _nodes.size();
    node.id = id;
    node.parent_edge_id = kInvalidId;
    node.child_edge_ids.clear();
    _nodes.push_back(std::move(node));
    _node_name_to_id[_nodes.back().name] = id;
    return id;
  }

  std::vector<NodeType> _nodes;
  std::vector<EdgeType> _edges;
  std::size_t _root_node_id = kInvalidId;
  std::unordered_map<std::string, std::size_t> _node_name_to_id;
};

template <typename T = int>
struct ClockSteinerEdge : public SteinerEdge<T>
{
  ClockSteinerEdge() = default;
  ClockSteinerEdge(std::size_t edge_id, std::size_t source_id, std::size_t target_id, const T& edge_distance, const T& routed_wire_distance)
      : SteinerEdge<T>(edge_id, source_id, target_id, edge_distance), routed_distance(routed_wire_distance)
  {
  }

  T routed_distance{};  // Routed wire distance in coordinate type T.
};

template <typename T = int>
class ClockSteinerTree : public SteinerTree<T, ClockSteinerNode<T>, ClockSteinerEdge<T>>
{
 public:
  using BaseType = SteinerTree<T, ClockSteinerNode<T>, ClockSteinerEdge<T>>;
  using CoordType = typename BaseType::CoordType;
  using NodeType = typename BaseType::NodeType;
  using EdgeType = typename BaseType::EdgeType;
  using BaseType::kInvalidId;

  ClockSteinerTree() = default;
  ~ClockSteinerTree() override = default;

  auto addNode(const std::string& name, const Point<T>& location, bool is_terminal = false) -> std::size_t override
  {
    return addNode(name, location, is_terminal, 0.0, 0.0);
  }

  auto addNode(const std::string& name, const Point<T>& location, bool is_terminal, double pin_cap, double insertion_delay) -> std::size_t
  {
    NodeType node;
    node.name = name;
    node.location = location;
    node.is_terminal = is_terminal;
    node.pin_cap = pin_cap;
    node.insertion_delay = insertion_delay;
    return this->appendNode(std::move(node));
  }
};

}  // namespace icts
