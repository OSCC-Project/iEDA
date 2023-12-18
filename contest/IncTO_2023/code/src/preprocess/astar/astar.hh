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
 * @File Name: astar.h
 * @Brief : 3d A*
 * @Author : GuoFan (guofan@ustc.edu)
 * @Version : 1.0
 * @Creat Date : 2023-09-27
 *
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <boost/geometry.hpp>
#include <vector>

#include "astarnode.hh"
#include "gridmap.hh"
#include "iterableHeap.hh"

namespace ieda_contest {

namespace astar {

namespace bg = boost::geometry;
namespace bgm = boost::geometry::model;

template <std::size_t DimensionCount>
class PathFinder
{
  typedef bgm::point<int, DimensionCount, bg::cs::cartesian> point;
  typedef gridmap::Map<Node<DimensionCount>> map_t;
  typedef Node<DimensionCount> node_t;

 public:
  PathFinder() : _map(nullptr) {}
  PathFinder(map_t& map, const point& starting_position, const point& ending_position)
      : _map(&map), _starting_position(starting_position), _ending_position(ending_position)
  {
  }

  void set_starting_position(const point starting_position) { _starting_position = starting_position; }
  void set_ending_position(const point ending_position) { _ending_position = ending_position; }
  void set_map(map_t& map) { _map = &map; }

  std::vector<node_t*> findPath() { return findPath(*_map, _starting_position, _ending_position); };
  void showPath(const std::vector<node_t*>& path) { return showPath(*_map, path); }

  /// @brief try to find a path from start_node to target_node, using A* algorithm
  /// @param map where stores all nodes and provide getNeighbours(node_t&) method
  /// @param start_node path source
  /// @param target_node path sink
  /// @return {} if there is no path, otherwise the path
  static std::vector<node_t*> findPath(map_t& map, node_t* start_node, node_t* target_node)
  {
    iterableHeap<node_t*> open_set;

    map.refreshMap();

    start_node->set_status(NodeStatus::open);
    open_set.push(start_node);
    while (!open_set.empty()) {
      node_t* current_node = open_set.top();
      open_set.pop();

      current_node->set_status(NodeStatus::closed);

      if (*current_node == *target_node) {
        std::deque<node_t*> path;
        node_t* current_node = target_node;
        while (*current_node != *start_node) {
          path.push_front(current_node);
          current_node = current_node->get_parent();
        }
        path.push_front(current_node);
        return std::vector<node_t*>{path.begin(), path.end()};
      }

      for (auto neighbour : map.getNeighbours(*current_node)) {
        if (/*!neighbour->isWalkable() || */ neighbour->get_status() == NodeStatus::closed) {  // TODO: isWalkable()
          continue;
        }

        int walk_cost = neighbour->walkCost();
        int new_cost_to_neighbour = current_node->get_g_cost() + (current_node->distance(*neighbour) * walk_cost * walk_cost);
        bool is_neighbour_in_open_set = neighbour->get_status() == NodeStatus::open;
        if (new_cost_to_neighbour < neighbour->get_g_cost() || !is_neighbour_in_open_set) {
          neighbour->set_g_cost(new_cost_to_neighbour);  // ! may break heap structure
          neighbour->set_h_cost(neighbour->distance(*target_node));
          neighbour->set_parent(current_node);

          if (!is_neighbour_in_open_set) {
            neighbour->set_status(NodeStatus::open);
            open_set.push(neighbour);
          } else {
            std::make_heap(open_set.begin(), open_set.end());  // ! repair heap structure
          }
        }
      }
    }

    return {};
  }
  static std::vector<node_t*> findPath(map_t& map, point start_position, point end_position)
  {
    return findPath(map, &map[start_position], &map[end_position]);
  }

 private:
  point _starting_position;
  point _ending_position;
  map_t* _map;
};

}  // namespace astar

}  // namespace ieda_contest
