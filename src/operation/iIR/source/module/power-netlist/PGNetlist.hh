// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file PGNetlist.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The pg netlist for wire topo analysis, and esitmate the wire R.
 * @version 0.1
 * @date 2025-02-22
 */
#pragma once

#include <tuple>
#include <vector>

#include "builder.h"
#include "def_service.h"
#include "lef_service.h"

#include "IdbLayer.h"
#include "IdbLayout.h"
#include "IdbNet.h"
#include "IdbPins.h"
#include "IdbSpecialNet.h"

namespace iir {

using IRNodeCoord = std::pair<int, int>;

/**
 * @brief PG network node.
 *
 */
class IRPGNode {
public:
  IRPGNode(IRNodeCoord coord, int layer_id)
      : _coord(coord), _layer_id(layer_id) {}
  ~IRPGNode() = default;

 private:
  IRNodeCoord _coord;  //!< The coord of the node.
  int _layer_id;       //!< The layer id of the node.
};

/**
 * @brief PG network edge.
 *
 */
class IRPGEdge {
public:
  IRPGEdge(IRPGNode& node1, IRPGNode& node2)
      : _node1(node1), _node2(node2) {}
  ~IRPGEdge() = default;

 private:
  IRPGNode& _node1;  //!< The first node.
  IRPGNode& _node2;  //!< The second node.
};

/**
 * @brief PG network for wire topo analysis.
 *
 */
class IRPGNetlist {
 public:
 IRPGNetlist() = default;
 ~IRPGNetlist() = default;

 IRPGNode& addNode(IRNodeCoord coord, int layer_id) {
    auto& one_node = _nodes.emplace_back(coord, layer_id);
    return one_node;
 }
 auto& get_nodes() { return _nodes; }
 IRPGEdge& addEdge(IRPGNode& node1, IRPGNode& node2) {
    auto& one_edge = _edges.emplace_back(node1, node2);
    return one_edge;
 }
 auto& get_edges() { return _edges; }

 private:
  std::vector<IRPGNode> _nodes;  //!< The nodes of the netlist.
  std::vector<IRPGEdge> _edges;  //!< The edges of the netlist.
};

/**
 * @brief The pg netlist builder.
 * 
 */
class IRPGNetlistBuilder {
 public:
  IRPGNetlistBuilder() = default;
  ~IRPGNetlistBuilder() = default;

  IRPGNetlist build(idb::IdbSpecialNet* special_net);

 private:
  std::vector<IRPGNetlist> _pg_netlists;
};

}  // namespace iir