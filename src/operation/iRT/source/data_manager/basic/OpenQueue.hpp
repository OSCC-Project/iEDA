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

#include <queue>

#include "OpenQueueNode.hpp"

namespace irt {

template <class T>
class OpenQueue : private std::priority_queue<OpenQueueNode<T>*, std::vector<OpenQueueNode<T>*>, CmpOpenQueueNodeCost<T>>
{
 public:
  using Base = std::priority_queue<OpenQueueNode<T>*, std::vector<OpenQueueNode<T>*>, CmpOpenQueueNodeCost<T>>;
  OpenQueue() = default;
  ~OpenQueue() = default;

  T* top()
  {
    while (!Base::empty() && Base::top()->get_state() == OpenQueueNodeState::kDel) {
      auto del_node = Base::top();
      Base::pop();
      delete del_node;
    }
    return Base::empty() ? nullptr : Base::top()->get_node();
  }
  T* pop()
  {
    T* node = top();
    if (!Base::empty()) {
      auto del_node = Base::top();
      _node_to_priority_queue_node_map.erase(node);
      Base::pop();
      delete del_node;
      real_size--;
    }
    return node;
  }
  void push(T* node)
  {
    if (_node_to_priority_queue_node_map.find(node) != _node_to_priority_queue_node_map.end()) {
      _node_to_priority_queue_node_map[node]->set_state(OpenQueueNodeState::kDel);
      real_size--;
    }
    OpenQueueNode<T>* pq_node = new OpenQueueNode<T>(node);
    _node_to_priority_queue_node_map[node] = pq_node;
    Base::push(pq_node);
    real_size++;
    if (Base::size() > real_size * 10) {
      restruct();
    }
  }
  void restruct()
  {
    std::priority_queue<OpenQueueNode<T>*, std::vector<OpenQueueNode<T>*>, CmpOpenQueueNodeCost<T>> temp;
    for (auto& pq_node : this->c) {
      if (pq_node->get_state() != OpenQueueNodeState::kDel) {
        temp.push(pq_node);
      } else {
        delete pq_node;
      }
    }
    temp.swap(*this);
  }
  void clear()
  {
    for (auto& pq_node : this->c) {
      delete pq_node;
    }
    real_size = 0;
    std::priority_queue<OpenQueueNode<T>*, std::vector<OpenQueueNode<T>*>, CmpOpenQueueNodeCost<T>>().swap(*this);
    _node_to_priority_queue_node_map.clear();
  }

 private:
  long unsigned int real_size = 0;
  std::map<T*, OpenQueueNode<T>*> _node_to_priority_queue_node_map;
};

}  // namespace irt