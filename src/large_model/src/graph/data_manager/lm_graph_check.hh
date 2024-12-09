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
 * @file lm_layout_check.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @version 1.0
 * @date 2024-11-15
 * @brief Check layout data (connectivity of nets, wires, etc.), and provide interfaces for graph construction.
 */

#pragma once

#include <boost/graph/adjacency_list.hpp>

#include "lm_net.h"
namespace ilm {
struct GraphLabel
{
  int x;
  int y;
  int32_t layer_id;
  int pin_id;
};

using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, GraphLabel>;
using Vertex = Graph::vertex_descriptor;
using Edge = Graph::edge_descriptor;

struct EdgeHash
{
  size_t operator()(const Edge& e) const
  {
    size_t h1 = std::hash<size_t>()(e.m_source);
    size_t h2 = std::hash<size_t>()(e.m_target);
    return h1 ^ (h2 << 1);
  }
};
class GraphCheckerBase
{
 public:
  GraphCheckerBase() {}
  ~GraphCheckerBase() = default;

 protected:
  virtual bool isConnectivity(const Graph& graph) const;
  virtual void writeToDot(const Graph& graph, const std::string& path) const;
  virtual void writeToPy(LmNet& net, const std::string& path) const;
  virtual void writeToPy(const Graph& graph, LmNet& net, const std::string& path, const bool& mark_break = false,
                         const bool& mark_pin_id = true) const;
};

class LmNetChecker : public GraphCheckerBase
{
 public:
  LmNetChecker() {}
  ~LmNetChecker() = default;

  // check the connectivity of the net, wire, etc.
  bool isLocalConnectivity(LmNet& net) const;
  bool isLocalConnectivity(LmNetWire& wire) const;

  // convert the net to graph
  Graph convertToGraph(LmNet& net) const;

  // write to py for debug
  void writeToDot(const Graph& graph, const std::string& path) const;
  void writeToPy(const Graph& graph, LmNet& net, const std::string& path, const bool& mark_break = false,
                 const bool& mark_pin_id = true) const;

 private:
};

class LmLayoutChecker : public GraphCheckerBase
{
 public:
  LmLayoutChecker() {}
  ~LmLayoutChecker() = default;

  bool checkLayout(std::map<int, LmNet> net_map);

  void checkPinConnection(std::map<int, LmNet> net_map);

 private:
};
}  // namespace ilm