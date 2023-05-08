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
  Node(Node&&)      = delete;
  ~Node()           = default;

  Node& operator=(const Node&) = delete;
  Node& operator=(Node&&) = delete;

  // getter.
  std::string    get_name() const { return _name; }
  Point<int32_t> get_location() const { return _location; }
  NetWork*       get_network() const { return _network; }
  Group*         get_group() const { return _group; }

  // setter.
  void set_location(Point<int32_t> location) { _location = std::move(location); }
  void set_network(NetWork* network) { _network = network; }
  void set_group(Group* group) { _group = group; }

 private:
  std::string    _name;
  Point<int32_t> _location;
  NetWork*       _network;
  Group*         _group;
};
inline Node::Node(std::string name) : _name(name), _network(nullptr), _group(nullptr)
{
}

class NetWork
{
 public:
  NetWork() = delete;
  explicit NetWork(std::string name);
  NetWork(const NetWork&) = delete;
  NetWork(NetWork&&)      = delete;
  ~NetWork()              = default;

  NetWork& operator=(const NetWork&) = delete;
  NetWork& operator=(NetWork&&) = delete;

  // getter.
  std::string               get_name() const { return _name; }
  float                     get_net_weight() const { return _net_weight; }
  Node*                     get_transmitter() const { return _transmitter; }
  const std::vector<Node*>& get_receiver_list() const { return _receiver_list; }
  std::vector<Node*>        get_node_list() const;
  bool isIgnoreNetwork();

  // setter.
  void set_net_weight(float weight) { _net_weight = weight; }
  void set_transmitter(Node* transmitter) { _transmitter = transmitter; }
  void add_receiver(Node* receiver) { _receiver_list.push_back(receiver); }

  // function.
  Rectangle<int32_t> obtainNetWorkShape();

 private:
  std::string        _name;
  float              _net_weight;
  Node*              _transmitter;
  std::vector<Node*> _receiver_list;
};
inline NetWork::NetWork(std::string name) : _name(name), _net_weight(1.0), _transmitter(nullptr)
{
}

class Group
{
 public:
  Group() = delete;
  explicit Group(std::string name);
  Group(const Group&) = delete;
  Group(Group&&)      = delete;
  ~Group()            = default;

  Group& operator=(const Group&) = delete;
  Group& operator=(Group&&) = delete;

  // getter.
  std::string        get_name() const { return _name; }
  std::vector<Node*> get_node_list() const { return _node_list; }

  // setter.
  void add_node(Node* node) { _node_list.push_back(node); }

 private:
  std::string        _name;
  std::vector<Node*> _node_list;
};
inline Group::Group(std::string name) : _name(name)
{
}

class TopologyManager
{
 public:
  TopologyManager()                       = default;
  TopologyManager(const TopologyManager&) = delete;
  TopologyManager(TopologyManager&&)      = delete;
  ~TopologyManager();

  TopologyManager& operator=(const TopologyManager&) = delete;
  TopologyManager& operator=(TopologyManager&&) = delete;

  // getter.
  const std::vector<Node*>&    get_node_list() const { return _node_list; }
  const std::vector<NetWork*>& get_network_list() const { return _network_list; }
  const std::vector<Group*>&   get_group_list() const { return _group_list; }

  // std::unordered_map<std::string, Node*>    get_node_map() const { return _node_map; }
  // std::unordered_map<std::string, NetWork*> get_network_map() const { return _network_map; }
  // std::unordered_map<std::string, Group*>   get_group_map() const { return _group_map; }

  // setter.
  void add_node(std::string name, Node* node);
  void add_network(std::string name, NetWork* network);
  void add_group(std::string name, Group* group);

  Node*    findNode(std::string name);
  NetWork* findNetwork(std::string name);
  Group*   findGroup(std::string name);

 private:
  std::vector<Node*>    _node_list;
  std::vector<NetWork*> _network_list;
  std::vector<Group*>   _group_list;

  std::unordered_map<std::string, Node*>    _node_map;
  std::unordered_map<std::string, NetWork*> _network_map;
  std::unordered_map<std::string, Group*>   _group_map;
};
inline TopologyManager::~TopologyManager()
{
  for (auto pair : _node_map) {
    delete pair.second;
  }
  _node_map.clear();

  for (auto pair : _network_map) {
    delete pair.second;
  }
  _network_map.clear();

  for (auto pair : _group_map) {
    delete pair.second;
  }
  _group_map.clear();
}

}  // namespace ipl

#endif