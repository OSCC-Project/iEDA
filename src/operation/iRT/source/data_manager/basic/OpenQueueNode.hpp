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

#include "RTHeader.hpp"
#include "Utility.hpp"

namespace irt {

#if 1  // astar
enum class OpenQueueNodeState
{
  kOpen = 1,
  kDel = 2
};
#endif

template <typename T>
class OpenQueueNode
{
 public:
  OpenQueueNode() = default;
  ~OpenQueueNode() = default;
  OpenQueueNode(T* node)
      : _node(node),
        _neighbor_node_map_size(node->get_neighbor_node_map().size()),
        _estimated_cost(node->get_estimated_cost()),
        _known_cost(node->get_known_cost())
  {
    set_state(OpenQueueNodeState::kOpen);
  }
  // getter
  double getEstimatedCost() const { return _estimated_cost; }
  double get_known_cost() const { return _known_cost; }
  double get_estimated_cost() const { return _estimated_cost; }
  double get_neighbor_node_map_size() const { return _neighbor_node_map_size; }
  T* get_node() const { return _node; }
  OpenQueueNodeState get_state() const { return _state; }
  // setter
  void set_known_cost(const double known_cost) { _known_cost = known_cost; }
  void set_estimated_cost(const double estimated_cost) { _estimated_cost = estimated_cost; }
  void set_neighbor_node_map_size(int32_t size) { _neighbor_node_map_size = size; }
  void set_node(T* node) { _node = node; }
  void set_state(OpenQueueNodeState state) { _state = state; }
  // function
  double getTotalCost() { return (_known_cost + _estimated_cost); }
  bool isOpen() { return _state == OpenQueueNodeState::kOpen; }
  bool isDel() { return _state == OpenQueueNodeState::kDel; }

 private:
  T* _node;
  int32_t _neighbor_node_map_size;
  double _estimated_cost;
  double _known_cost;
  OpenQueueNodeState _state;
};

#if 1  // astar
template <typename T>
struct CmpOpenQueueNodeCost
{
  bool operator()(OpenQueueNode<T>* a, OpenQueueNode<T>* b)
  {
    if (RTUTIL.equalDoubleByError(a->getTotalCost(), b->getTotalCost(), RT_ERROR)) {
      if (RTUTIL.equalDoubleByError(a->get_estimated_cost(), b->get_estimated_cost(), RT_ERROR)) {
        return a->get_neighbor_node_map_size() < b->get_neighbor_node_map_size();
      } else {
        return a->get_estimated_cost() > b->get_estimated_cost();
      }
    } else {
      return a->getTotalCost() > b->getTotalCost();
    }
  }
};
#endif

}  // namespace irt
