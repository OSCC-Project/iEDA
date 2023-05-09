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
#pragma once

/**
 * @file check_connection.h
 * @author yell
 * @brief check net connectivity
 * @version 0.1
 * @date 2023-03-01
 */

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/properties.hpp>
#include <boost/property_map/property_map.hpp>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace idb {
class IdbPin;
class IdbRegularWireSegment;
class IdbNet;
}  // namespace idb

using namespace idb;

namespace idm {

enum class NodeType : int8_t
{
  kNone,
  kPin,
  kSegment,
  kMax
};

enum class CheckInfo : int8_t
{
  kNone,
  kConnected,
  kDisconnected,
  kHasRing,
  kMax
};

class CheckNode
{
 public:
  CheckNode(int id) { _id = id; }
  virtual ~CheckNode() = default;

  /// getter
  NodeType get_type() { return _type; }
  bool is_pin() { return _type == NodeType::kPin ? true : false; }
  bool is_seg() { return _type == NodeType::kSegment ? true : false; }
  int16_t get_id() { return _id; }
  int16_t get_graph_id() { return _graph_id; }
  bool is_visited() { return _graph_id == -1 ? false : true; }

  /// setter
  void set_type(NodeType type) { _type = type; }
  void set_id(int id) { _id = id; }
  void set_graph_id(int id) { _graph_id = id; }

  virtual bool isIntersection(CheckNode* node_dst) = 0;

 private:
  int16_t _graph_id = -1;
  int16_t _id = -1;
  NodeType _type = NodeType::kNone;
};

class CheckNodePin : public CheckNode
{
 public:
  CheckNodePin(IdbPin* pin, int id) : CheckNode(id)
  {
    _pin = pin;
    set_type(NodeType::kPin);
  }
  ~CheckNodePin() = default;

  /// getter
  IdbPin* get_pin() { return _pin; }

  bool isIntersection(CheckNode* node_dst) override;

 private:
  IdbPin* _pin = nullptr;
};

class CheckNodeSegment : public CheckNode
{
 public:
  CheckNodeSegment(IdbRegularWireSegment* segment, int id) : CheckNode(id)
  {
    _segment = segment;
    set_type(NodeType::kSegment);
  }
  ~CheckNodeSegment() = default;

  /// getter
  IdbRegularWireSegment* get_segment() { return _segment; }

  virtual bool isIntersection(CheckNode* node_dst) override;

 private:
  IdbRegularWireSegment* _segment = nullptr;
};

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> BglGraph;
typedef boost::graph_traits<BglGraph>::vertex_descriptor Vertex;
typedef boost::graph_traits<BglGraph>::edge_descriptor Edge;

class NetGraph
{
 public:
  NetGraph() {}
  ~NetGraph() = default;

  /// getter
  int get_id() { return _graph_id; }
  int get_pin_num() { return _connected_pins.size(); }
  int get_vertex_num() { return boost::num_vertices(_graph); }
  int get_edge_num() { return boost::num_edges(_graph); }
  BglGraph& get_graph() { return _graph; }
  bool has_ring()
  {
    int vertex_num = boost::num_vertices(_graph);
    int edge_num = boost::num_edges(_graph);

    return vertex_num > 0 && edge_num > 0 && vertex_num <= edge_num ? true : false;
  }

  /// setter
  void set_id(int graph_id) { _graph_id = graph_id; }

  /// operator
  void add_vertex(int vertex) { _graph.added_vertex(vertex); }
  void add_edge(int vertex_1, int vertex_2) { boost::add_edge(vertex_1, vertex_2, _graph); }
  int num_edge() { return boost::num_edges(_graph); }

  void addConnectedPin(int pin_index) { _connected_pins.emplace(pin_index); }

 private:
  std::set<int> _connected_pins;  /// connected pin index list in this graph
  BglGraph _graph;
  int _graph_id = -1;
};

class CheckNet
{
 public:
  CheckNet(IdbNet* net) { wrapNet(net); }
  ~CheckNet() = default;

  CheckInfo checkNetConnection();
  CheckInfo isAllPinConnected();
  bool hasRing();

 private:
  std::string _net_name;
  std::vector<CheckNode*> _node_list;
  std::vector<NetGraph> _graph_list;
  int _pin_num = -1;

  /// getter
  std::vector<CheckNode*> get_node_list(NodeType type);
  int get_pin_num()
  {
    int num = 0;
    for (auto node : _node_list) {
      if (node->is_pin()) {
        num++;
      }
    }
    return num;
  }

  int get_segment_num()
  {
    int num = 0;
    for (auto node : _node_list) {
      if (node->is_seg()) {
        num++;
      }
    }
    return num;
  }

  /// build graph
  void buildGraph();

  ///  wrapper
  void wrapNet(IdbNet* net);
  int wrapNetPinList(IdbNet* net);
  int wrapNetSegmentList(IdbNet* net);

  /// check
  bool isIntersection(CheckNode* node_src, CheckNode* node_dst);

  /// DFS
  void buildGraphBFS(NetGraph& graph, CheckNode* check_node);
};

}  // namespace idm