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
};

using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, GraphLabel>;

class GraphCheckerBase
{
 public:
  GraphCheckerBase() {}
  ~GraphCheckerBase() = default;

 protected:
  virtual bool isConnectivity(const Graph& graph);
  virtual void writeToDot(const Graph& graph, const std::string& path);
  virtual void writeToPy(LmNet& net, const std::string& path);
  virtual void writeToPy(const Graph& graph, LmNet& net, const std::string& path);
};

class LmNetChecker : public GraphCheckerBase
{
 public:
  LmNetChecker() {}
  ~LmNetChecker() = default;

  // check the connectivity of the net, wire, etc.
  bool isLocalConnectivity(LmNet& net);
  bool isLocalConnectivity(LmNetWire& wire);

  // convert the net to graph
  Graph convertToGraph(LmNet& net);

 private:
};

class LmLayoutChecker : public GraphCheckerBase
{
 public:
  LmLayoutChecker() {}
  ~LmLayoutChecker() = default;

  bool checkLayout(std::map<int, LmNet> net_map);

  bool addNet(LmNet& net);
  bool isConnectivity();
  Graph getGraph() { return _graph; }

 private:
  size_t _node_id = 0;
  std::unordered_map<std::tuple<int, int, int32_t>, size_t, boost::hash<std::tuple<int, int, int32_t>>> _node_to_id;
  Graph _graph;
  std::vector<LmNet> _nets;
};
}  // namespace ilm