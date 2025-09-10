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
/*
 * @Author: S.J Chen
 * @Date: 2022-03-08 12:18:14
 * @LastEditTime: 2022-10-27 19:21:01
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/util/topology_manager/TopologyManager.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_UTIL_TOPOLOGY_MANAGER_H
#define IPL_UTIL_TOPOLOGY_MANAGER_H

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "data/Rectangle.hh"

namespace ipl {

enum class NETWORK_TYPE
{
  kNone,
  kSignal,
  kClock,
  kFakeNet
};

enum class GROUP_TYPE
{
  kNone,
  kLogic,
  kFlipflop,
  kClockBuffer,
  kLogicBuffer,
  kMacro,
  kIOCell
};

enum class ARC_TYPE
{
  kNone,
  kNetArc,
  kGroupArc
};

enum class NODE_TYPE
{
  kNone,
  kInput,
  kOutput,
  kInputOutput
};

class NetWork;
class Group;
class Arc;

class Node
{
 public:
  Node() = delete;
  explicit Node(std::string name);
  Node(const Node&) = delete;
  Node(Node&&) = delete;
  ~Node() = default;

  Node& operator=(const Node&) = delete;
  Node& operator=(Node&&) = delete;

  // getter.
  int32_t get_node_id() const { return _node_id; }
  NODE_TYPE get_node_type() const { return _node_type; }
  std::string get_name() const { return _name; }
  Point<int32_t> get_location() const { return _location; }
  bool is_io_node() const { return _io_flag; }

  NetWork* get_network() const { return _network; }
  Group* get_group() const { return _group; }
  const std::vector<Arc*>& get_input_arc_list() const { return _input_arc_list; }
  const std::vector<Arc*>& get_output_arc_list() const { return _output_arc_list; }
  int32_t get_topo_id() const { return _topo_id; }
  double get_centrality() const { return _centrality; }
  double get_criticality() const { return _criticality; }

  // setter.
  void set_node_id(int32_t id) { _node_id = id; }
  void set_node_type(NODE_TYPE node_type) { _node_type = node_type; }
  void set_is_io() { _io_flag = true; }
  void set_location(Point<int32_t> location) { _location = std::move(location); }

  void set_network(NetWork* network) { _network = network; }
  void set_group(Group* group) { _group = group; }
  void add_input_arc(Arc* arc) { _input_arc_list.push_back(arc); }
  void add_output_arc(Arc* arc) { _output_arc_list.push_back(arc); }
  void set_topo_id(int32_t topo_id) { _topo_id = topo_id; }
  void set_centrality(double centrality) { _centrality = centrality; }
  void set_criticality(double criticality) { _criticality = criticality; }

 private:
  int32_t _node_id;
  NODE_TYPE _node_type;
  bool _io_flag;
  std::string _name;
  Point<int32_t> _location;

  NetWork* _network;
  Group* _group;

  std::vector<Arc*> _input_arc_list;
  std::vector<Arc*> _output_arc_list;

  int32_t _topo_id;

  double _centrality;
  double _criticality;
};
inline Node::Node(std::string name)
    : _node_id(-1),
      _node_type(NODE_TYPE::kNone),
      _io_flag(false),
      _name(name),
      _network(nullptr),
      _group(nullptr),
      _topo_id(0),
      _centrality(0.0f),
      _criticality(0.0f)
{
}

class NetWork
{
 public:
  NetWork() = delete;
  explicit NetWork(std::string name);
  NetWork(const NetWork&) = delete;
  NetWork(NetWork&&) = delete;
  ~NetWork() = default;

  NetWork& operator=(const NetWork&) = delete;
  NetWork& operator=(NetWork&&) = delete;

  // getter.
  int32_t get_network_id() const { return _network_id; }
  NETWORK_TYPE get_network_type() const { return _network_type; }
  std::string get_name() const { return _name; }
  float get_net_weight() const { return _net_weight; }
  Node* get_transmitter() const { return _transmitter; }
  const std::vector<Node*>& get_receiver_list() const { return _receiver_list; }
  std::vector<Node*> get_node_list() const;
  bool isIgnoreNetwork();

  // setter.
  void set_network_id(int32_t id) { _network_id = id; }
  void set_network_type(NETWORK_TYPE network_type) { _network_type = network_type; }
  void set_net_weight(float weight) { _net_weight = weight; }
  void set_transmitter(Node* transmitter) { _transmitter = transmitter; }
  void add_receiver(Node* receiver) { _receiver_list.push_back(receiver); }

  // function.
  Rectangle<int32_t> obtainNetWorkShape();
  int32_t obtainTopoIndex();

 private:
  int32_t _network_id;
  NETWORK_TYPE _network_type;
  std::string _name;
  float _net_weight;
  Node* _transmitter;
  std::vector<Node*> _receiver_list;
};
inline NetWork::NetWork(std::string name)
    : _network_id(-1), _network_type(NETWORK_TYPE::kNone), _name(name), _net_weight(1.0), _transmitter(nullptr)
{
}

class Group
{
 public:
  Group() = delete;
  explicit Group(std::string name);
  Group(const Group&) = delete;
  Group(Group&&) = delete;
  ~Group() = default;

  Group& operator=(const Group&) = delete;
  Group& operator=(Group&&) = delete;

  // getter.
  int32_t get_group_id() const { return _group_id; }
  GROUP_TYPE get_group_type() const { return _group_type; }
  std::string get_name() const { return _name; }
  std::vector<Node*>& get_node_list() { return _node_list; }

  // setter.
  void set_group_id(int32_t id) { _group_id = id; }
  void set_group_type(GROUP_TYPE type) { _group_type = type; }
  void add_node(Node* node) { _node_list.push_back(node); }

  // function.
  std::vector<Node*> obtainInputNodes();
  std::vector<Node*> obtainOutputNodes();
  int32_t obtainTopoIndex();

 private:
  int32_t _group_id;
  GROUP_TYPE _group_type;
  std::string _name;
  std::vector<Node*> _node_list;
};
inline Group::Group(std::string name) : _group_id(-1), _group_type(GROUP_TYPE::kNone), _name(name)
{
}

class Arc
{
 public:
  Arc() = delete;
  Arc(Node* from_node, Node* to_node);
  Arc(const Arc&) = delete;
  Arc(Arc&&) = delete;
  ~Arc() = default;

  Arc& operator=(const Arc&) = delete;
  Arc& operator=(Arc&&) = delete;

  // getter
  int32_t get_arc_id() const { return _arc_id; }
  ARC_TYPE get_arc_type() const { return _arc_type; }
  Node* get_from_node() const { return _from_node; }
  Node* get_to_node() const { return _to_node; }
  double get_flow_value() const { return _flow_value; }

  // setter
  void set_arc_id(int32_t arc_id) { _arc_id = arc_id; }
  void set_arc_type(ARC_TYPE arc_type) { _arc_type = arc_type; }
  void set_flow_value(double value) { _flow_value = value; }

 private:
  int32_t _arc_id;
  ARC_TYPE _arc_type;

  Node* _from_node;
  Node* _to_node;

  double _flow_value;
};
inline Arc::Arc(Node* from_node, Node* to_node)
    : _arc_id(-1), _arc_type(ARC_TYPE::kNone), _from_node(from_node), _to_node(to_node), _flow_value(0.0)
{
}

class TopologyManager
{
 public:
  TopologyManager();
  TopologyManager(const TopologyManager&) = delete;
  TopologyManager(TopologyManager&&) = delete;
  ~TopologyManager();

  TopologyManager& operator=(const TopologyManager&) = delete;
  TopologyManager& operator=(TopologyManager&&) = delete;

  // getter.
  const std::vector<Node*>& get_node_list() const { return _node_list; }
  const std::vector<NetWork*>& get_network_list() const { return _network_list; }
  const std::vector<Group*>& get_group_list() const { return _group_list; }
  const std::vector<Arc*>& get_arc_list() const { return _arc_list; }
  const std::vector<Node*>& get_input_port_list() const { return _port_input_list; }
  const std::vector<Node*>& get_output_port_list() const { return _port_output_list; }

  std::vector<Node*> get_node_copy_list() const { return _node_list; }
  std::vector<NetWork*> get_network_copy_list() const { return _network_list; }
  std::vector<Group*> get_group_copy_list() const { return _group_list; }
  // setter.
  void add_node(Node* node);
  void add_network(NetWork* network);
  void add_group(Group* group);
  void add_arc(Arc* arc);

  Node* findNodeById(int32_t node_id);
  NetWork* findNetworkById(int32_t network_id);
  Group* findGroupById(int32_t group_id);
  Arc* findArcById(int32_t arc_id);

  void updateTopoId(Node* node);
  void updateALLNodeTopoId();

  // sort vec by their index
  void sortNodeList();
  void sortNetworkList();
  void sortGroupList();
  void sortArcList();

 private:
  std::vector<Node*> _node_list;
  std::vector<NetWork*> _network_list;
  std::vector<Group*> _group_list;
  std::vector<Arc*> _arc_list;

  std::vector<Node*> _port_input_list;
  std::vector<Node*> _port_output_list;

  int32_t _nodes_range;
  int32_t _networks_range;
  int32_t _groups_range;
  int32_t _arcs_range;
};
inline TopologyManager::TopologyManager() : _nodes_range(0), _networks_range(0), _groups_range(0), _arcs_range(0)
{
}

inline TopologyManager::~TopologyManager()
{
  for (auto* node : _node_list) {
    delete node;
  }
  _node_list.clear();
  _nodes_range = 0;

  for (auto* network : _network_list) {
    delete network;
  }
  _network_list.clear();
  _networks_range = 0;

  for (auto* group : _group_list) {
    delete group;
  }
  _group_list.clear();
  _groups_range = 0;

  for (auto* arc : _arc_list) {
    delete arc;
  }
  _arc_list.clear();
  _arcs_range = 0;
}

}  // namespace ipl

#endif