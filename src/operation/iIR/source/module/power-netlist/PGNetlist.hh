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

  void set_node_name(const char* name) { _node_name = name; }
  auto get_node_name() const { return _node_name; }

  void set_is_bump() { _is_bump = true; }
  auto is_bump() const { return _is_bump; }

 private:
  IRNodeCoord _coord;  //!< The coord of the node.
  int _layer_id;       //!< The layer id of the node.
  int _node_id = -1; //!< The node id of the pg nodes.
  bool _is_instance_pin = false; //!< The node is instance VDD/GND.
  bool _is_bump = false; //!< The node is bump VDD/GND.
  const char* _node_name = nullptr; //!< The name of the node.
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
 * @brief node comparator for store IR Node according to the row order for row edge connected.
 * 
 */
struct IRNodeRowComparator {
  bool operator()(const IRPGNode* lhs, const IRPGNode* rhs) const {
    auto lhs_coord = lhs->get_coord();
    auto rhs_coord = rhs->get_coord();

    if (lhs_coord.second != rhs_coord.second) {
      return lhs_coord.second < rhs_coord.second;
    }
    return lhs_coord.first < rhs_coord.first;
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

  void set_resistance(double resistance) { _resistance = resistance; }
  double get_resistance() const { return _resistance; }

 private:
  int64_t _node1;  //!< The first node id.
  int64_t _node2;  //!< The second node id.

  double _resistance = 0.0; //!< The edge resistance.
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
  void set_net_name(const std::string& name) { _net_name = name; }

  IRPGNode& addNode(IRNodeCoord coord, int layer_id) {
    auto& one_node = _nodes.emplace_back(coord, layer_id);
    _nodes_image.push_back(&one_node);
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
  auto& get_nodes_image() { return _nodes_image; }
  IRPGNode* getNode(unsigned index) {
    return _nodes_image[index];
  }

  auto getEdgeNode(IRPGEdge& pg_edge) {
    auto node1_id = pg_edge.get_node1();
    auto* node1 = _nodes_image[node1_id];
    auto node2_id = pg_edge.get_node2();
    auto* node2 = _nodes_image[node2_id];

    return std::make_tuple(node1, node2);
  }

  IRPGEdge& addEdge(IRPGNode* node1, IRPGNode* node2) {
    LOG_FATAL_IF(node1->get_node_id() == node2->get_node_id());
    auto& one_edge = _edges.emplace_back(node1, node2);
    return one_edge;
  }
  auto& get_edges() { return _edges; }
  auto getEdgeNum() { return _edges.size(); }

  void addNodeIdToName(unsigned node_id, std::string name) {
    _node_id_to_name[node_id] = std::move(name);
  }
  auto& get_node_id_to_name() { return _node_id_to_name; }
  auto& getNodeName(unsigned node_id) {
    return _node_id_to_name[node_id];
  }

  void printToYaml(std::string yaml_path);

 private:
  std::list<IRPGNode> _nodes;  //!< The nodes of the netlist.
  std::vector<IRPGNode*> _nodes_image; //!< The nodes image for fast access.
  std::vector<IRPGEdge> _edges;  //!< The edges of the netlist.

  std::map<unsigned, std::string> _node_id_to_name; //!< The node id to node name.

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

  std::vector<BGSegment> buildBGSegments(idb::IdbSpecialNet* special_net,
                                         unsigned& line_segment_num);

  void build(idb::IdbSpecialNet* special_net, idb::IdbPin* io_pin,
             std::function<double(unsigned, unsigned)> calc_resistance);
  void createRustPGNetlist();
  void createRustRCData();

  auto* get_rust_rc_data() const { return _rust_rc_data; }

 private:
  bgi::rtree<BGValue, bgi::quadratic<16>> _rtree;

  std::list<IRPGNetlist> _pg_netlists; //!< The builded pg netlist.
  std::vector<const void*> _rust_pg_netlists; //!< The rust pg netlist.
  const void* _rust_rc_data = nullptr;
};

}  // namespace iir