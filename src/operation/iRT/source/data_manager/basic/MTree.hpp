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

#include "Logger.hpp"
#include "Segment.hpp"
#include "TNode.hpp"

namespace irt {

template <typename T>
class MTree
{
 public:
  MTree() = default;
  explicit MTree(TNode<T>* root) { _root = root; }
  MTree(const MTree& other) { copy(other); }
  MTree(MTree&& other) { move(std::forward<MTree>(other)); }
  ~MTree() { free(); }
  MTree& operator=(const MTree& other)
  {
    copy(other);
    return (*this);
  }
  MTree& operator=(MTree&& other)
  {
    move(std::forward<MTree>(other));
    return (*this);
  }
  // getter
  TNode<T>* get_root() { return _root; }
  // setter
  void set_root(TNode<T>* root)
  {
    free();
    _root = root;
  }
  // function
  void clear() { set_root(nullptr); }

 private:
  TNode<T>* _root = nullptr;
  // function
  inline void copy(const MTree& other);
  inline void free();
  inline void copyTree(TNode<T>* other_root);
  inline void move(MTree&& other);
};

template <typename T>
inline void MTree<T>::copy(const MTree& other)
{
  free();
  copyTree(other._root);
}

template <typename T>
inline void MTree<T>::free()
{
  if (_root == nullptr) {
    return;
  }
  std::queue<TNode<T>*> node_queue;
  node_queue.push(_root);

  while (!node_queue.empty()) {
    TNode<T>* node = node_queue.front();
    node_queue.pop();

    for (TNode<T>* child_node : node->get_child_list()) {
      node_queue.push(child_node);
    }

    delete node;
    node = nullptr;
  }
}

template <typename T>
inline void MTree<T>::copyTree(TNode<T>* other_root)
{
  if (other_root == nullptr) {
    return;
  }
  _root = new TNode<T>(other_root->value());

  std::queue<TNode<T>*> old_node_queue;
  old_node_queue.push(other_root);
  std::queue<TNode<T>*> new_node_queue;
  new_node_queue.push(_root);

  while (!old_node_queue.empty()) {
    TNode<T>* old_node = old_node_queue.front();
    old_node_queue.pop();
    TNode<T>* new_node = new_node_queue.front();
    new_node_queue.pop();

    for (TNode<T>* child_node : old_node->get_child_list()) {
      new_node->addChild(new TNode<T>(child_node->value()));
    }
    for (TNode<T>* old_child_node : old_node->get_child_list()) {
      old_node_queue.push(old_child_node);
    }
    for (TNode<T>* new_child_node : new_node->get_child_list()) {
      new_node_queue.push(new_child_node);
    }
  }
}

template <typename T>
inline void MTree<T>::move(MTree&& other)
{
  free();
  _root = other._root;
  other._root = nullptr;
}

}  // namespace irt
