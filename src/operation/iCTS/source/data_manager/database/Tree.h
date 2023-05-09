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
#include <cassert>
#include <queue>
#include <utility>
#include <vector>

namespace icts {
using std::pair;
using std::queue;
using std::vector;
template <typename Value>
struct TreeNode {
  typedef TreeNode *link_type;
  typedef Value value_type;

  TreeNode() = default;
  TreeNode(const value_type &value) : _value(value), _parent(nullptr) {}

  value_type &value() { return _value; }
  link_type &parent() { return _parent; }
  std::vector<link_type> &childs() { return _childs; }
  void insert(link_type child) { _childs.push_back(child); }

  value_type _value;
  link_type _parent;
  std::vector<link_type> _childs;
};

template <typename Value>
class Tree {
 protected:
  typedef Value value_type;
  typedef value_type *pointer;
  typedef value_type &reference;
  typedef TreeNode<value_type> tree_node;
  typedef TreeNode<value_type> *link_type;

 public:
  Tree() : _header(nullptr), _node_count(0) {}
  Tree(value_type value);
  Tree(link_type root);
  Tree(const Tree &tree);
  ~Tree() { clear(); }

  link_type &find(value_type value) { return find(root(), value); }
  bool empty() const { return _header == nullptr; }
  size_t size() const { return _node_count; }
  int height() { return height(root()); }
  void clear();
  vector<value_type> nodes();
  vector<value_type> nodes(int level);

  link_type root() { return _header; }

  static link_type createNode(const value_type &value) {
    return new tree_node(value);
  }
  static void destroyNode(link_type p) { putNode(p); }

 protected:
  link_type get_node() { return new tree_node(); }
  static void putNode(link_type p) { delete p; }

 private:
  void erase(link_type x);
  link_type find(link_type root, value_type value);
  size_t size(link_type node) const;
  int height(link_type node);
  link_type copy(link_type node);

 private:
  link_type _header;
  int _node_count;
};

template <typename Value>
Tree<Value>::Tree(value_type value) {
  _header = createNode(value);
  _node_count = 1;
}

template <typename Value>
Tree<Value>::Tree(link_type root) {
  _header = root;
  _node_count = size(root);
}

template <typename Value>
Tree<Value>::Tree(const Tree &tree) {
  _header = copy(tree._header);
  _node_count = tree.size();
}

template <typename Value>
vector<Value> Tree<Value>::nodes() {
  vector<value_type> values;
  queue<link_type> que;

  que.push(root());
  while (!que.empty()) {
    link_type node = que.front();
    que.pop();

    values.push_back(node->value());

    auto &childs = node->childs();
    for (auto *child : childs) {
      que.push(child);
    }
  }

  return values;
}

template <typename Value>
vector<Value> Tree<Value>::nodes(int level) {
  vector<value_type> values;
  queue<link_type> que;

  assert(level > 0);
  if (level == 1) {
    values.push_back(root()->value());
    return values;
  }

  que.push(root());
  int cur_level = 1;
  int count = que.size();
  while (!que.empty()) {
    link_type node = que.front();
    for (auto *child : node->childs()) {
      que.push(child);
    }
    que.pop();
    count--;

    if (count == 0) {
      count = que.size();
      cur_level++;
    }
    if (cur_level == level) {
      break;
    }
  }
  while (!que.empty()) {
    values.push_back(que.front()->value());
    que.pop();
  }
  return values;
}

template <typename Value>
TreeNode<Value> *Tree<Value>::find(link_type root, value_type value) {
  if (root) {
    if (root->value() == value) {
      return root;
    }
    auto &childs = root->childs();
    for (auto &child : childs) {
      link_type node_ptr = find(child, value);
      if (node_ptr != nullptr) {
        return node_ptr;
      }
    }
  }
  return nullptr;
}

template <typename Value>
TreeNode<Value> *Tree<Value>::copy(link_type node) {
  if (node == nullptr) {
    return nullptr;
  }
  link_type root = createNode(node->value());
  for (auto *child : node->childs()) {
    root->insert(copy(child));
  }
  return root;
}

template <typename Value>
size_t Tree<Value>::size(link_type node) const {
  size_t node_count = 0;
  if (node) {
    for (link_type child : node->childs()) {
      node_count += size(child);
    }
  }
  return node_count;
}

template <typename Value>
int Tree<Value>::height(link_type node) {
  int high = 0;
  if (node) {
    for (auto child : node->childs()) {
      high = std::max(high, height(child));
    }
    high += 1;
  }
  return high;
}

template <typename Value>
void Tree<Value>::clear() {
  if (_header != nullptr) {
    erase(root());
    _header = nullptr;
    _node_count = 0;
  }
}

template <typename Value>
void Tree<Value>::erase(link_type x) {
  if (x) {
    for (auto child : x->childs()) {
      erase(child);
    }
    putNode(x);
  }
}

}  // namespace icts