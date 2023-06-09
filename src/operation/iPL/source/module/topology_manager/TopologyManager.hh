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

class NetWork;
class Group;

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
  std::string get_name() const { return _name; }
  Point<int32_t> get_location() const { return _location; }
  NetWork* get_network() const { return _network; }
  Group* get_group() const { return _group; }

  // setter.
  void set_node_id(int32_t id) { _node_id = id; }
  void set_location(Point<int32_t> location) { _location = std::move(location); }
  void set_network(NetWork* network) { _network = network; }
  void set_group(Group* group) { _group = group; }

 private:
  int32_t _node_id;
  std::string _name;
  Point<int32_t> _location;
  NetWork* _network;
  Group* _group;
};
inline Node::Node(std::string name) : _node_id(-1), _name(name), _network(nullptr), _group(nullptr)
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
  std::string get_name() const { return _name; }
  float get_net_weight() const { return _net_weight; }
  Node* get_transmitter() const { return _transmitter; }
  const std::vector<Node*>& get_receiver_list() const { return _receiver_list; }
  std::vector<Node*> get_node_list() const;
  bool isIgnoreNetwork();

  // setter.
  void set_network_id(int32_t id) { _network_id = id; }
  void set_net_weight(float weight) { _net_weight = weight; }
  void set_transmitter(Node* transmitter) { _transmitter = transmitter; }
  void add_receiver(Node* receiver) { _receiver_list.push_back(receiver); }

  // function.
  Rectangle<int32_t> obtainNetWorkShape();

 private:
  int32_t _network_id;
  std::string _name;
  float _net_weight;
  Node* _transmitter;
  std::vector<Node*> _receiver_list;
};
inline NetWork::NetWork(std::string name) : _network_id(-1), _name(name), _net_weight(1.0), _transmitter(nullptr)
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
  std::string get_name() const { return _name; }
  std::vector<Node*>& get_node_list() { return _node_list; }

  // setter.
  void set_group_id(int32_t id) { _group_id = id; }
  void add_node(Node* node) { _node_list.push_back(node); }

 private:
  int32_t _group_id;
  std::string _name;
  std::vector<Node*> _node_list;
};
inline Group::Group(std::string name) : _group_id(-1), _name(name)
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

  // setter.
  void add_node(Node* node);
  void add_network(NetWork* network);
  void add_group(Group* group);

  Node* findNodeById(int32_t node_id);
  NetWork* findNetworkById(int32_t network_id);
  Group* findGroupById(int32_t group_id);

 private:
  std::vector<Node*> _node_list;
  std::vector<NetWork*> _network_list;
  std::vector<Group*> _group_list;

  int32_t _nodes_range;
  int32_t _networks_range;
  int32_t _groups_range;
};
inline TopologyManager::TopologyManager() : _nodes_range(0), _networks_range(0), _groups_range(0)
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
}

}  // namespace ipl

#endif