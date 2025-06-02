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
#include "idm.h"
#include "lm_graph_check.hh"
#include "lm_grid_info.h"
#include "lm_net_graph_gen.hh"
#include "omp.h"
#include "usage.hh"

namespace ilm {

#define debug_error 0

bool LmGraphDataManager::buildGraphData()
{
  auto get_nodes
      = [&](int x1, int y1, int layer1, int x2, int y2, int layer2, LmLayoutLayers& layout_layers) -> std::pair<LmNode*, LmNode*> {
    if (layer1 == layer2) {
      auto& grid = layout_layers.findLayoutLayer(layer1)->get_grid();

      auto [row1, col1] = gridInfoInst.findNodeID(x1, y1);
      auto* node1 = grid.get_node(row1, col1);

      auto [row2, col2] = gridInfoInst.findNodeID(x2, y2);
      auto* node2 = grid.get_node(row2, col2);

      if (node1 == nullptr || node1->get_node_data() == nullptr || node2 == nullptr || node2->get_node_data() == nullptr) {
        LOG_ERROR << "error node....";
      }

      node1->set_real_coordinate(x1, y1);
      node2->set_real_coordinate(x2, y2);

      return std::make_pair(node1, node2);
    } else {
      auto& grid1 = layout_layers.findLayoutLayer(layer1)->get_grid();
      auto [row1, col1] = gridInfoInst.findNodeID(x1, y1);
      auto* node1 = grid1.get_node(row1, col1);

      auto& grid2 = layout_layers.findLayoutLayer(layer2)->get_grid();
      auto [row2, col2] = gridInfoInst.findNodeID(x2, y2);
      auto* node2 = grid2.get_node(row2, col2);

      if (node1 == nullptr || node1->get_node_data() == nullptr || node2 == nullptr || node2->get_node_data() == nullptr) {
        LOG_ERROR << "error node....";
      }

      node1->set_real_coordinate(x1, y1);
      node2->set_real_coordinate(x2, y2);

      return std::make_pair(node1, node2);
    }
  };

  auto convert_to_routing_graph = [&](const WireGraph& wire_graph) -> NetRoutingGraph {
    NetRoutingGraph routing_graph;
    auto [v_iter, v_end] = boost::vertices(wire_graph);
    for (; v_iter != v_end; ++v_iter) {
      auto v = *v_iter;
      auto x = wire_graph[v].x;
      auto y = wire_graph[v].y;
      auto layer_id = wire_graph[v].layer_id;
      NetRoutingVertex vertex{
          .id = v,
          .is_pin = wire_graph[v].is_pin,
          .is_driver_pin = wire_graph[v].is_driver_pin,
          .point = {x, y, layer_id},
      };
      routing_graph.vertices.emplace_back(vertex);
    }
    auto [e_iter, e_end] = boost::edges(wire_graph);
    for (; e_iter != e_end; ++e_iter) {
      auto e = *e_iter;
      auto source = boost::source(e, wire_graph);
      auto target = boost::target(e, wire_graph);
      std::vector<std::pair<LayoutDefPoint, LayoutDefPoint>> wire_path = wire_graph[e].path;
      // convert wire_path to path
      if (wire_path.empty()) {
        LOG_WARNING << "Empty path for edge: " << source << " -> " << target;
        continue;
      }
      std::vector<NetRoutingPoint> path;
      std::ranges::for_each(wire_path, [&](const std::pair<LayoutDefPoint, LayoutDefPoint>& point_pair) -> void {
        path.emplace_back(NetRoutingPoint{bg::get<0>(point_pair.first), bg::get<1>(point_pair.first), bg::get<2>(point_pair.first)});
      });
      path.emplace_back(
          NetRoutingPoint{bg::get<0>(wire_path.back().second), bg::get<1>(wire_path.back().second), bg::get<2>(wire_path.back().second)});
      NetRoutingEdge edge{
          .source_id = source,
          .target_id = target,
          .path = path,
      };
      routing_graph.edges.emplace_back(edge);
    }
    return routing_graph;
  };

  ieda::Stats stats;

  auto* idb_design = dmInst->get_idb_design();
  auto idb_nets = idb_design->get_net_list()->get_net_list();
  LmNetGraphGenerator gen;

  auto& layout_graph = _layout->get_graph();
  auto& layout_layers = _layout->get_layout_layers();

  omp_lock_t lck;
  omp_init_lock(&lck);

  // auto* special_net_list = idb_design->get_special_net_list();
  // auto* vdd_net = special_net_list->find_net("VDD");
  // gen.buildTopoGraph(vdd_net);

  // #pragma omp parallel for schedule(dynamic)
  for (size_t net_id = 0; net_id < idb_nets.size(); ++net_id) {
    auto* idb_net = idb_nets[net_id];
    /// ignore net if pin number < 2
    if (idb_net->get_pin_number() < 2) {
      continue;
    }
    if (gen.isCornerCase(idb_net)) {
      continue;
    }
    auto wire_graph = gen.buildGraph(idb_net);
    auto routing_graph = convert_to_routing_graph(wire_graph);
    auto* lm_net = layout_graph.get_net(net_id);
    lm_net->clearWire();
    lm_net->set_routing_graph(routing_graph);
#if debug_error
    std::set<int> pin_ids_wires;
    std::set<int> pin_ids_paths;
#endif
    // travelsal the edge in wire_graph
    for (auto edge : boost::make_iterator_range(boost::edges(wire_graph))) {
      LmNetWire lm_wire;
      /// add wire
      {
        auto source = boost::source(edge, wire_graph);
        auto target = boost::target(edge, wire_graph);

        auto source_label = wire_graph[source];
        auto target_label = wire_graph[target];

        auto [node1, node2] = get_nodes(source_label.x, source_label.y, source_label.layer_id, target_label.x, target_label.y,
                                        target_label.layer_id, layout_layers);

        /// ignore same node
        if (node1 == node2) {
          continue;
        }

        lm_wire.set_start(node1);
        lm_wire.set_end(node2);
#if debug_error
        if (node1->get_node_data()->get_pin_id() >= 0) {
          pin_ids_wires.insert(node1->get_node_data()->get_pin_id());
        }
        if (node2->get_node_data()->get_pin_id() >= 0) {
          pin_ids_wires.insert(node2->get_node_data()->get_pin_id());
        }

        if ((node1->get_node_data()->get_pin_id() >= 0 && false == source_label.is_pin)
            || (node1->get_node_data()->get_pin_id() == -1 && source_label.is_pin)) {
          LOG_WARNING << "Warning, node1 pin mismatch. ";
        }

        if ((node2->get_node_data()->get_pin_id() >= 0 && false == target_label.is_pin)
            || (node2->get_node_data()->get_pin_id() == -1 && target_label.is_pin)) {
          LOG_WARNING << "Warning, node2 pin mismatch. ";
        }
#endif
      }

      /// add path

#if debug_error
      LmNode* bk_start = nullptr;
      LmNode* bk_end = nullptr;
      int i = 0;
#endif
      std::ranges::for_each(wire_graph[edge].path, [&](auto& point_pair) -> void {
        auto start_point = point_pair.first;
        auto end_point = point_pair.second;
        auto [node1, node2] = get_nodes(bg::get<0>(start_point), bg::get<1>(start_point), bg::get<2>(start_point), bg::get<0>(end_point),
                                        bg::get<1>(end_point), bg::get<2>(end_point), layout_layers);

        /// ignore same node
        if (node1 != node2) {
          lm_wire.add_path(node1, node2);
        }
#if debug_error
        if (i == 0) {
          bk_end = node2;
        } else {
          bk_start = node1;

          if (bk_end != bk_start) {
            std::ranges::for_each(wire_graph[edge].path, [&](auto& p) -> void {
              LOG_INFO << "[" << bg::get<0>(p.first) << "," << bg::get<1>(p.first) << "," << bg::get<2>(p.first) << "] ["
                       << bg::get<0>(p.second) << "," << bg::get<1>(p.second) << "," << bg::get<2>(p.second) << "]";
            });
          }

          bk_end = node2;
        }

        if (node1->get_node_data()->get_pin_id() >= 0) {
          pin_ids_paths.insert(node1->get_node_data()->get_pin_id());
        }
        if (node2->get_node_data()->get_pin_id() >= 0) {
          pin_ids_paths.insert(node2->get_node_data()->get_pin_id());
        }
#endif
      });

      lm_net->addWire(lm_wire);
    }

    if (net_id % 1000 == 0) {
      LOG_INFO << "Read nets : " << net_id << " / " << (int) idb_nets.size();
    }
#if debug_error
    if (pin_ids_wires.size() != lm_net->get_pin_ids().size()) {
      LOG_WARNING << "Warning, pin size mismatch, name : " << _layout->findNetName(net_id) << " ,net id : " << net_id
                  << " ,net num : " << lm_net->get_pin_ids().size() << " ,wire num : " << pin_ids_wires.size()
                  << " ,path num: " << pin_ids_paths.size();
    }
#endif
  }

  LOG_INFO << "Read nets : " << idb_nets.size() << " / " << (int) idb_nets.size();

  omp_destroy_lock(&lck);
  LOG_INFO << "LM memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "LM elapsed time " << stats.elapsedRunTime() << " s";
  LOG_INFO << "LM save json net end...";

  return true;
}

}  // namespace ilm