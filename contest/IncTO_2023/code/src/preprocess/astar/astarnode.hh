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
 * @File Name: astarnode.h
 * @Brief : 3d A* Node
 * @Author : GuoFan (guofan@ustc.edu)
 * @Version : 1.0
 * @Creat Date : 2023-09-27
 *
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cmath>

#include "mapnode.hh"

namespace ieda_contest {

namespace astar {

namespace bg = boost::geometry;
namespace bgm = boost::geometry::model;

enum class NodeStatus
{
  wild,
  open,
  closed
};

template <std::size_t DimensionCount>
class Node : public gridmap::Node<DimensionCount>
{
 public:
  Node() : _parent(this) {}

  int get_g_cost() const { return _g_cost; }
  int get_h_cost() const { return _h_cost; }
  int get_f_cost() const { return _f_cost; }
  void set_g_cost(int g_cost)
  {
    _g_cost = g_cost;
    _f_cost = _g_cost + _h_cost;
  }
  void set_h_cost(int h_cost)
  {
    _h_cost = h_cost;
    _f_cost = _g_cost + _h_cost;
  }
  Node* get_parent() const { return _parent; }
  void set_parent(Node* parent) { _parent = parent; }
  NodeStatus get_status() { return _status; }
  void set_status(NodeStatus s) { _status = s; }

  void refresh() override
  {
    gridmap::Node<DimensionCount>::refresh();
    _g_cost = 0;
    _h_cost = 0;
    _f_cost = 0;
    _status = NodeStatus::wild;
  }

  int distance(Node& n)
  {
    auto p1 = this->get_position();
    auto p2 = n.get_position();
    return std::abs(bg::get<0>(p1) - bg::get<0>(p2)) + std::abs(bg::get<1>(p1) - bg::get<1>(p2));
  }

  bool operator>(const Node& n) const { return _f_cost > n._f_cost; }

 private:
  int _g_cost = 0;
  int _h_cost = 0;
  int _f_cost = 0;

  NodeStatus _status = NodeStatus::wild;

  Node* _parent;
};

}  // namespace astar

}  // namespace ieda_contest
