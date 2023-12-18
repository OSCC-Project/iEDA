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
#include <algorithm>
#include <functional>
#include <optional>
#include <string>

#include "CTSAPI.hh"
#include "CtsConfig.hh"
#include "CtsInstance.hh"
#include "Enum.hh"
#include "log/Log.hh"
namespace icts {

class Node
{
 public:
  // basic
  Node(const int& id, const Point& location) : _id(id), _location(location) { _name = "steiner_" + std::to_string(id); }
  Node(const std::string& name, const Point& location) : _name(name), _location(location) {}
  // steiner
  // Node(const Point& location) : _location(location)
  // {
  //   _id = CTSAPIInst.genId();
  //   _name = "steiner_" + std::to_string(_id);
  // }
  Node(const Node& other) = default;
  Node(Node&& other) = default;

  virtual ~Node()
  {
    std::ranges::for_each(_children, [&](Node* child) { child->set_parent(nullptr); });
    _children.clear();
    _parent = nullptr;
  }
  // get
  const int& get_id() const { return _id; }
  const std::string& get_name() const { return _name; }
  const Point& get_location() const { return _location; }
  const double& get_sub_len() const { return _sub_len; }
  const double& get_cap_load() const { return _cap_load; }
  const double& get_slew_in() const { return _slew_in; }
  const double& get_min_delay() const { return _min_delay; }
  const double& get_max_delay() const { return _max_delay; }
  const double& get_required_snake() const { return _required_snake; }
  const NodeType& get_type() const { return _type; }
  Node* get_parent() const { return _parent; }
  const std::vector<Node*>& get_children() const { return _children; }
  const RCPattern& get_pattern() const { return _pattern; }
  // set
  void set_id(const int& id) { _id = id; }
  void set_name(const std::string& name) { _name = name; }
  void set_location(const Point& location) { _location = location; }
  void set_sub_len(const double& sub_len) { _sub_len = sub_len; }
  void set_cap_load(const double& cap_load) { _cap_load = cap_load; }
  void set_slew_in(const double& slew_in) { _slew_in = slew_in; }
  void set_min_delay(const double& min_delay) { _min_delay = min_delay; }
  void set_max_delay(const double& max_delay) { _max_delay = max_delay; }
  void set_required_snake(const double& required_snake) { _required_snake = required_snake; }
  void set_parent(Node* parent) { _parent = parent; }
  void set_children(const std::vector<Node*>& children) { _children = children; }
  void set_pattern(const RCPattern& pattern) { _pattern = pattern; }
  // add
  void add_child(Node* child) { _children.push_back(child); }

  // remove
  void remove_child(Node* child) { _children.erase(std::remove(_children.begin(), _children.end(), child), _children.end()); }

  // bool
  bool isPin() const { return !isSteiner(); }
  bool isSinkPin() const { return _type == NodeType::kSinkPin; }
  bool isBufferPin() const { return _type == NodeType::kBufferPin; }
  bool isSteiner() const { return _type == NodeType::kSteiner; }

  // traversal
  using NodeFunc = std::function<void(Node*)>;
  using NodePatternFunc = std::function<void(Node*, const RCPattern&)>;
  void preOrder(NodeFunc func)
  {
    func(this);
    std::ranges::for_each(_children, [&](auto child) { child->preOrder(func); });
  }
  void preOrder(NodePatternFunc func)
  {
    func(this, _pattern);
    std::ranges::for_each(_children, [&](auto child) { child->preOrder(func); });
  }
  void postOrder(NodeFunc func)
  {
    std::ranges::for_each(_children, [&](auto child) { child->postOrder(func); });
    func(this);
  }
  void postOrder(NodePatternFunc func)
  {
    std::ranges::for_each(_children, [&](auto child) { child->postOrder(func); });
    func(this, _pattern);
  }

  // for pin node
  virtual std::string get_cell_master() const { LOG_FATAL << "Node type have not cell master"; }

  virtual bool isDriver() const { return false; }
  virtual bool isLoad() const { return false; }

  // for id suffix
  int getMaxId()
  {
    int max_id = 0;
    preOrder([&](Node* node) {
      if (node->get_type() != NodeType::kSteiner) {
        return;
      }
      auto id = node->get_id();
      max_id = std::max(max_id, id);
    });
    return max_id;
  }

 protected:
  void set_type(const NodeType& type) { _type = type; }

 private:
  int _id = 0;
  std::string _name = "";
  Point _location = Point(-1, -1);
  double _sub_len = 0;
  double _cap_load = 0;
  double _slew_in = 0;
  double _min_delay = 0;
  double _max_delay = 0;
  double _required_snake = 0;
  NodeType _type = NodeType::kSteiner;
  Node* _parent = nullptr;
  std::vector<Node*> _children;
  RCPattern _pattern = RCPattern::kSingle;
};
}  // namespace icts