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
#include <algorithm>
#include <utility>
#include <vector>

#include "Topology.h"
#include "Traits.h"
#include "pgl.h"

namespace icts {

template <typename T>
void build_topo(Topology<T> &topo, const std::vector<T> &datas) {
  std::vector<TopoNode<T>> nodes;
  std::vector<size_t> pointers;
  for (size_t i = 0; i < datas.size(); i++) {
    nodes.emplace_back(TopoNode<T>{datas[i], -1, -1, -1,
                                   DataTraits<T>::getSubWirelength(datas[i])});
    pointers.push_back(i);
  }

  if (datas.size() == 0) return;
  auto edge_cost_compare = [&nodes](auto &edge1, auto &edge2) {
    auto value_1 = edge1.second;
    auto value_2 = edge2.second;
    if ((value_1 ^ value_2) < 0) {
      return value_1 > 0;
    } else {
      if (value_1 > 0) {
        return value_1 < value_2;
      } else {
        return value_1 > value_2;
      }
    }
  };
  while (pointers.size() > 1) {
    std::vector<std::pair<std::pair<size_t, size_t>, int>> edges_costs;
    for (size_t vi = 0; vi < pointers.size(); ++vi) {
      for (size_t vj = vi + 1; vj < pointers.size(); ++vj) {
        auto loc_l = DataTraits<T>::getPoint(nodes[pointers[vi]].data());
        auto loc_r = DataTraits<T>::getPoint(nodes[pointers[vj]].data());
        auto sub_wl_l = nodes[pointers[vi]].sub_wire_length();
        auto sub_wl_r = nodes[pointers[vj]].sub_wire_length();
        auto delta = std::abs(sub_wl_l - sub_wl_r);
        auto dist = pgl::manhattan_distance(loc_l, loc_r);
        // auto merge_cost = dist - delta;
        auto merge_cost = delta > dist ? dist + delta : dist;
        edges_costs.emplace_back(std::make_pair(
            std::make_pair(pointers[vi], pointers[vj]), merge_cost));
      }
    }

    std::sort(edges_costs.begin(), edges_costs.end(), edge_cost_compare);
    // auto n = pointers.size();
    // for (auto i = 1; i <= n / 2; ++i) {
    auto edge = edges_costs.front().first;
    T val;
    auto loc_l = DataTraits<T>::getPoint(nodes[edge.first].data());
    auto loc_r = DataTraits<T>::getPoint(nodes[edge.second].data());
    DataTraits<T>::setX(val, (loc_l.x() + loc_r.x()) / 2);
    DataTraits<T>::setY(val, (loc_l.y() + loc_r.y()) / 2);
    auto sub_wire_length = (nodes[edge.first].sub_wire_length() +
                            nodes[edge.second].sub_wire_length() +
                            pgl::manhattan_distance(loc_l, loc_r)) /
                           2;
    nodes.emplace_back(TopoNode<T>{val, -1, static_cast<int>(edge.first),
                                   static_cast<int>(edge.second),
                                   sub_wire_length});
    nodes[edge.first].parent(nodes.size() - 1);
    nodes[edge.second].parent(nodes.size() - 1);

    auto begin = std::remove_if(
        pointers.begin(), pointers.end(), [&edge](const auto &pointer) {
          return pointer == edge.first || pointer == edge.second;
        });
    pointers.erase(begin, pointers.end());
    pointers.push_back(nodes.size() - 1);
    auto edge_begin = std::remove_if(edges_costs.begin(), edges_costs.end(),
                                     [&edge](const auto &item) {
                                       auto temp_edge = item.first;
                                       return edge.first == temp_edge.first ||
                                              edge.first == temp_edge.second ||
                                              edge.second == temp_edge.first ||
                                              edge.second == temp_edge.second;
                                     });
    edges_costs.erase(edge_begin, edges_costs.end());
    // }
  }

  topo = Topology<T>(nodes, pointers[0]);
  assert(nodes.size() == 2 * datas.size() - 1);
}

}  // namespace icts