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
 * @file MinCostFlow.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @brief Min-cost flow helper for topology clustering.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <utility>
#include <vector>

#include "Geometry.hh"
#include "lemon/list_graph.h"
#include "lemon/maps.h"
#include "lemon/network_simplex.h"

namespace icts {
/**
 * @brief MinCostFlow template class for solving clustering by min-cost flow.
 *        input: nodes, centers
 *        constraint: max_cluster_size (like max_fanout)
 *        output: clusters
 *
 * @tparam Value
 */
template <typename Value>
class MinCostFlow
{
 public:
  using CapacityType = int;
  using CostType = int;

  MinCostFlow() = default;
  ~MinCostFlow() = default;

  auto addNode(double x, double y, Value& value) -> void { _nodes.push_back({FlowPoint{x, y}, value}); }

  auto addCenter(double x, double y) -> void { _centers.push_back({x, y}); }

  auto run(const size_t& max_cluster_size) -> std::vector<std::vector<Value>>
  {
    // Flow problem:
    // virtual source -> [sinks] -> [buffers] -> virtual target
    // requirement: meet the max_cluster_size (like max_fanout) constraint

    // Define the node and arc types
    using Node = lemon::ListDigraph::Node;
    using NodeMap = lemon::ListDigraph::NodeMap<std::pair<int, int>>;
    using Arc = lemon::ListDigraph::Arc;
    using ArcMap = lemon::ListDigraph::ArcMap<CostType>;
    using ArcIt = lemon::ListDigraph::ArcIt;
    using MinCostFlowSolver = lemon::NetworkSimplex<lemon::ListDigraph, CapacityType, CostType>;

    const auto cluster_capacity = static_cast<CapacityType>(max_cluster_size);
    const auto total_supply = static_cast<CapacityType>(_nodes.size());

    // Define the network
    lemon::ListDigraph network;

    // Add nodes to the network
    Node source = network.addNode();
    Node target = network.addNode();
    std::vector<Node> sinks, buffers;
    sinks.reserve(_nodes.size());
    buffers.reserve(_centers.size());

    std::ranges::transform(_nodes, std::back_inserter(sinks), [&](auto&) -> auto { return network.addNode(); });

    std::ranges::transform(_centers, std::back_inserter(buffers), [&](auto&) -> auto { return network.addNode(); });

    // Add arcs to the network
    std::vector<Arc> source_sink_arcs, sink_buffer_arcs, buffer_target_arcs;
    source_sink_arcs.reserve(sinks.size());
    sink_buffer_arcs.reserve(sinks.size() * buffers.size());
    buffer_target_arcs.reserve(buffers.size());

    // source -> sink arcs
    for (const auto& sink : sinks) {
      source_sink_arcs.emplace_back(network.addArc(source, sink));
    }

    // sink -> buffer arcs (with distance costs)
    std::vector<CostType> dist_costs;
    dist_costs.reserve(sinks.size() * buffers.size());

    for (size_t i = 0; i < sinks.size(); ++i) {
      const auto& sink = sinks[i];
      const auto& sink_pt = _nodes[i].point;

      for (size_t j = 0; j < buffers.size(); ++j) {
        auto dist = calcCost(sink_pt, _centers[j]);
        sink_buffer_arcs.emplace_back(network.addArc(sink, buffers[j]));
        dist_costs.emplace_back(dist);
      }
    }

    // buffer -> target arcs
    for (const auto& buffer : buffers) {
      buffer_target_arcs.emplace_back(network.addArc(buffer, target));
    }

    // cost
    ArcMap arc_cost(network), arc_capacity(network);
    for (const auto& arc : source_sink_arcs) {
      arc_capacity[arc] = 1;
    }
    for (size_t i = 0; i < sink_buffer_arcs.size(); ++i) {
      const auto& arc = sink_buffer_arcs[i];
      arc_capacity[arc] = 1;
      arc_cost[arc] = dist_costs[i];
    }
    for (const auto& arc : buffer_target_arcs) {
      arc_capacity[arc] = cluster_capacity;
    }

    // mcf solver by lemon
    MinCostFlowSolver mcf(network);
    mcf.costMap(arc_cost);
    mcf.upperMap(arc_capacity);
    mcf.stSupply(source, target, total_supply);
    mcf.run();
    ArcMap solution(network);
    mcf.flowMap(solution);

    // init the node map
    NodeMap node_map(network);
    for (size_t i = 0; i < sinks.size(); ++i) {
      node_map[sinks[i]] = {i, -1};
    }

    for (size_t i = 0; i < buffers.size(); ++i) {
      node_map[buffers[i]] = {-1, i};
    }

    std::pair<int, int> virtual_node = {-2, -2};
    node_map[source] = virtual_node;
    node_map[target] = virtual_node;

    // get the clusters
    std::vector<std::vector<Value>> clusters(_centers.size());
    for (ArcIt it(network); it != lemon::INVALID; ++it) {
      if (solution[it] == 0) {
        continue;
      }
      if (node_map[network.source(it)].second == -1 && node_map[network.target(it)].first == -1) {
        const auto cluster_id = static_cast<std::size_t>(node_map[network.target(it)].second);
        const auto node_index = static_cast<std::size_t>(node_map[network.source(it)].first);
        clusters[cluster_id].emplace_back(_nodes[node_index].value);
      }
    }
    // remove empty cluster
    clusters.erase(std::remove_if(clusters.begin(), clusters.end(), [](auto& cluster) -> auto { return cluster.empty(); }), clusters.end());
    return clusters;
  }

 private:
  struct FlowPoint
  {
    double x;
    double y;
  };
  struct FlowNode
  {
    FlowPoint point;
    Value value;
  };
  static auto calcCost(const FlowPoint& p1, const FlowPoint& p2) -> CostType
  {
    const auto dist = geometry::Manhattan(Point<double>(p1.x, p1.y), Point<double>(p2.x, p2.y));
    return static_cast<CostType>(dist);
  }
  std::vector<FlowPoint> _centers{};
  std::vector<FlowNode> _nodes{};
};

}  // namespace icts
