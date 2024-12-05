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

#include "lm_graph_dm.h"

#include "Log.hh"
#include "lm_graph_check.hh"
#include "lm_graph_init.h"
#include "lm_net_graph_gen.hh"
#include "omp.h"
#include "usage.hh"

namespace ilm {

bool LmGraphDataManager::buildGraphData(const std::string path)
{
  LmNetGraphGenerator gen;
  auto wire_graphs = gen.buildGraphs();

  std::ranges::for_each(wire_graphs, [&](auto& wire_graph) -> void {
    // travelsal the edge in wire_graph
    for (auto edge : boost::make_iterator_range(boost::edges(wire_graph))) {
      auto source = boost::source(edge, wire_graph);
      auto target = boost::target(edge, wire_graph);
      auto source_label = wire_graph[source];
      auto target_label = wire_graph[target];
      LOG_INFO << "Edge: " << source_label.x << "," << source_label.y << " " << source_label.layer_id << " " << target_label.x << " "
               << target_label.y << " " << target_label.layer_id;
      LOG_INFO << "Path: ";
      std::ranges::for_each(wire_graph[edge].path, [&](auto& point_pair) -> void {
        auto start_point = point_pair.first;
        auto end_point = point_pair.second;
        LOG_INFO << "(" << bg::get<0>(start_point) << "," << bg::get<1>(start_point) << "," << bg::get<2>(start_point) << ") - ("
                 << bg::get<0>(end_point) << "," << bg::get<1>(end_point) << "," << bg::get<2>(end_point) << ")";
      });
    }
  });
}

}  // namespace ilm