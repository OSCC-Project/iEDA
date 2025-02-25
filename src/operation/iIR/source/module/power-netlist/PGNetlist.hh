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

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/segment.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <tuple>
#include <vector>
#include <list>
#include <ranges>

#include "IdbLayer.h"
#include "IdbLayout.h"
#include "IdbNet.h"
#include "IdbPins.h"
#include "IdbSpecialNet.h"
#include "builder.h"
#include "def_service.h"
#include "lef_service.h"
#include "log/Log.hh"


namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

using BGPoint = bg::model::point<int64_t, 3, bg::cs::cartesian>;
using BGSegment = bg::model::segment<BGPoint>;
using BGRect = bg::model::box<BGPoint>;
using BGValue = std::pair<BGRect, unsigned>;

namespace iir {

using IRNodeCoord = std::pair<int64_t, int64_t>;

/**
 * @brief PG network node.
 *
 */
class IRPGNode {
 public:
  IRPGNode(IRNodeCoord coord, int layer_id)
      : _coord(coord), _layer_id(layer_id) {}
  ~IRPGNode() = default;

  auto get_coord() const { return _coord; }
  auto get_layer_id() const { return _layer_id; }

  void set_node_id(unsigned id) { _node_id = id; }
  auto get_node_id() const { return _node_id; }

  void set_is_instance_pin() { _is_instance_pin = true; }
  bool is_instance_pin() const { return _is_instance_pin; }

 private:
  IRNodeCoord _coord;  //!< The coord of the node.
  int _layer_id;       //!< The layer id of the node.

  int _node_id = -1; //!< The node id of the pg nodes.

  bool _is_instance_pin = false; //!< The node is instance VDD/GND.
};

/**
 * @brief node comparator for store IR Node according to the coord order.
 * 
 */
struct IRNodeComparator {
  bool operator()(const IRPGNode* lhs, const IRPGNode* rhs) const {
    auto lhs_coord = lhs->get_coord();
    auto rhs_coord = rhs->get_coord();

    if (lhs_coord.first != rhs_coord.first) {
      return lhs_coord.first < rhs_coord.first;
    }
    return lhs_coord.second < rhs_coord.second;
  }
};

/**
 * @brief PG network edge.
 *
 */
class IRPGEdge {
 public:
  IRPGEdge(IRPGNode* node1, IRPGNode* node2)
      : _node1(node1->get_node_id()), _node2(node2->get_node_id()) {}
  ~IRPGEdge() = default;
  auto& get_node1() const { return _node1; }
  auto& get_node2() const { return _node2; }

 private:
  int64_t _node1;  //!< The first node id.
  int64_t _node2;  //!< The second node id.
};

/**
 * @brief PG network for wire topo analysis.
 *
 */
class IRPGNetlist {
 public:
  IRPGNetlist() = default;
  ~IRPGNetlist() = default;

  std::string& get_net_name() { return _net_name; }

  IRPGNode& addNode(IRNodeCoord coord, int layer_id) {
    auto& one_node = _nodes.emplace_back(coord, layer_id);
    one_node.set_node_id(_nodes.size() - 1);
    return one_node;
  }
  IRPGNode* findNode(IRNodeCoord coord, int layer_id) {
    auto result = std::ranges::find_if(_nodes, [&](const IRPGNode& node) {
      return node.get_coord() == coord && node.get_layer_id() == layer_id;
    });
    if (result != _nodes.end()) {
      return &(*result);
    }

    return nullptr;
  }
  auto& get_nodes() { return _nodes; }
  IRPGEdge& addEdge(IRPGNode* node1, IRPGNode* node2) {
    LOG_FATAL_IF(node1->get_node_id() == node2->get_node_id());
    auto& one_edge = _edges.emplace_back(node1, node2);
    return one_edge;
  }
  auto& get_edges() { return _edges; }
  auto getEdgeNum() { return _edges.size(); }

  void printToYaml(std::string yaml_path);

 private:
  std::list<IRPGNode> _nodes;  //!< The nodes of the netlist.
  std::vector<IRPGEdge> _edges;  //!< The edges of the netlist.

  std::string _net_name;
};

/**
 * @brief The pg netlist builder.
 *
 */
class IRPGNetlistBuilder {
 public:
  IRPGNetlistBuilder() = default;
  ~IRPGNetlistBuilder() = default;

  std::vector<BGSegment> buildBGSegments(idb::IdbSpecialNet* special_net, unsigned& line_segment_num);

  void build(idb::IdbSpecialNet* special_net);
  void createRustPGNetlist();
  void estimateRC();

 private:
  bgi::rtree<BGValue, bgi::quadratic<16>> _rtree;

  std::vector<IRPGNetlist> _pg_netlists; //!< The builded pg netlist.
  std::vector<const void*> _rust_pg_netlists; //!< The rust pg netlist.
};

}  // namespace iir