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
 * @file Node.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#pragma once
#include <functional>
#include <string>

#include "CTSAPI.hpp"
#include "CtsInstance.h"

namespace icts {

enum class NodeType
{
  kSink,
  kBuffer,
  kSteiner
};
class Node
{
 public:
  Node()
  {
    _id = CTSAPIInst.genId();
    _name = "steiner_" + std::to_string(_id);
  }
  Node(const std::string& name, const Point& location) : _name(name), _location(location) { _id = CTSAPIInst.genId(); }
  ~Node()
  {
    _children.clear();
    if (_inst) {
      delete _inst;
      _inst = nullptr;
    }
    _parent = nullptr;
  }
  // get
  const size_t& get_id() const { return _id; }
  const std::string& get_name() const { return _name; }
  const Point& get_location() const { return _location; }
  const std::string& get_cell_master() const { return _cell_master; }
  CtsInstance* get_inst() const { return _inst; }
  const uint16_t& get_fanout() const { return _fanout; }
  const double& get_min_delay() const { return _min_delay; }
  const double& get_max_delay() const { return _max_delay; }
  const double& get_slew_in() const { return _slew_in; }
  const double& get_cap_out() const { return _cap_out; }
  const double& get_cap_load() const { return _cap_load; }
  const double& get_insert_delay() const { return _insert_delay; }
  const double& get_sub_net_length() const { return _sub_net_length; }
  const NodeType& get_type() const { return _type; }
  Node* get_parent() const { return _parent; }
  const std::vector<Node*>& get_children() const { return _children; }

  // set
  void set_id(const size_t& id) { _id = id; }
  void set_name(const std::string& name) { _name = name; }
  void set_location(const Point& location) { _location = location; }
  void set_cell_master(const std::string& cell_master) { _cell_master = cell_master; }
  void set_inst(CtsInstance* inst) { _inst = inst; }
  void set_fanout(const uint16_t& fanout) { _fanout = fanout; }
  void set_min_delay(const double& min_delay) { _min_delay = min_delay; }
  void set_max_delay(const double& max_delay) { _max_delay = max_delay; }
  void set_slew_in(const double& slew_in) { _slew_in = slew_in; }
  void set_cap_out(const double& cap_out) { _cap_out = cap_out; }
  void set_cap_load(const double& cap_load) { _cap_load = cap_load; }
  void set_insert_delay(const double& insert_delay) { _insert_delay = insert_delay; }
  void set_sub_net_length(const double& sub_net_length) { _sub_net_length = sub_net_length; }
  void set_type(const NodeType& type) { _type = type; }
  void set_parent(Node* parent) { _parent = parent; }
  void set_children(const std::vector<Node*>& children) { _children = children; }

  // add
  void add_child(Node* child) { _children.push_back(child); }

  // bool
  bool isSink() const { return _type == NodeType::kSink; }
  bool isBuffer() const { return _type == NodeType::kBuffer; }
  bool isSteiner() const { return _type == NodeType::kSteiner; }

  // traversal
  using NodeFunc = std::function<void(Node*)>;

  void preOrder(NodeFunc func)
  {
    func(this);
    for (auto child : _children) {
      child->preOrder(func);
    }
  }
  void postOrder(NodeFunc func)
  {
    for (auto child : _children) {
      child->postOrder(func);
    }
    func(this);
  }
  using StopFunc = std::function<bool(Node*)>;
  void preOrderBy(
      NodeFunc func, StopFunc stop_func = [](Node* node) { return node->isBuffer(); })
  {
    auto stop = stop_func(this);

    func(this);
    if (!stop) {
      for (auto child : _children) {
        child->preOrder(func);
      }
    }
  }
  void postOrderBy(
      NodeFunc func, StopFunc stop_func = [](Node* node) { return node->isBuffer(); })
  {
    auto stop = stop_func(this);

    if (!stop) {
      for (auto child : _children) {
        child->postOrder(func);
      }
    }
    func(this);
  }

 private:
  size_t _id = 0;
  std::string _name = "";
  Point _location = Point(-1, -1);
  std::string _cell_master = "";
  CtsInstance* _inst = nullptr;
  uint16_t _fanout = 1;  // fanout can't exceed 65535
  double _min_delay = 0;
  double _max_delay = 0;
  double _slew_in = 0;
  double _cap_out = 0;
  double _cap_load = 0;
  double _insert_delay = 0;
  double _sub_net_length = 0;
  NodeType _type = NodeType::kSteiner;
  Node* _parent = nullptr;
  std::vector<Node*> _children;
};
}  // namespace icts