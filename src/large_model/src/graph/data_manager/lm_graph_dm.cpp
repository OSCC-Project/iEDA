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

bool LmGraphDataManager::buildGraphData()
{
  auto get_nodes = [&](int x1, int y1, int layer1, int x2, int y2, int layer2, LmPatchLayers& patch_layers) -> std::pair<LmNode*, LmNode*> {
    if (layer1 == layer2) {
      auto& grid = patch_layers.findPatchLayer(layer1)->get_grid();

      auto [row1, col1] = gridInfoInst.findNodeID(x1, y1);
      auto* node1 = grid.get_node(row1, col1);

      auto [row2, col2] = gridInfoInst.findNodeID(x2, y2);
      auto* node2 = grid.get_node(row2, col2);

      if (node1 == nullptr || node1->get_node_data() == nullptr || node2 == nullptr || node2->get_node_data() == nullptr) {
        LOG_ERROR << "error node....";
      }

      return std::make_pair(node1, node2);
    } else {
      auto& grid1 = patch_layers.findPatchLayer(layer1)->get_grid();
      auto [row1, col1] = gridInfoInst.findNodeID(x1, y1);
      auto* node1 = grid1.get_node(row1, col1);

      auto& grid2 = patch_layers.findPatchLayer(layer2)->get_grid();
      auto [row2, col2] = gridInfoInst.findNodeID(x2, y2);
      auto* node2 = grid2.get_node(row2, col2);

      if (node1 == nullptr || node1->get_node_data() == nullptr || node2 == nullptr || node2->get_node_data() == nullptr) {
        LOG_ERROR << "error node....";
      }

      return std::make_pair(node1, node2);
    }
  };
  auto* idb_design = dmInst->get_idb_design();
  auto idb_nets = idb_design->get_net_list()->get_net_list();
  LmNetGraphGenerator gen;

  auto& layout_graph = _layout->get_graph();
  auto& patch_layers = _layout->get_patch_layers();

  for (size_t net_id = 0; net_id < idb_nets.size(); ++net_id) {
    auto* idb_net = idb_nets[net_id];
    auto wire_graph = gen.buildGraph(idb_net);
    auto* lm_net = layout_graph.get_net(net_id);
    lm_net->clearWire();

    std::set<int> pin_ids_wires;
    std::set<int> pin_ids_paths;

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
                                        target_label.layer_id, patch_layers);
        lm_wire.set_start(node1);
        lm_wire.set_end(node2);
        if (node1->get_node_data()->get_pin_id() != 0) {
          pin_ids_wires.insert(node1->get_node_data()->get_pin_id());
        }
        if (node2->get_node_data()->get_pin_id() != 0) {
          pin_ids_wires.insert(node2->get_node_data()->get_pin_id());
        }
      }

      /// add path

#if 0
      LmNode* bk_start = nullptr;
      LmNode* bk_end = nullptr;
      int i = 0;
#endif
      std::ranges::for_each(wire_graph[edge].path, [&](auto& point_pair) -> void {
        auto start_point = point_pair.first;
        auto end_point = point_pair.second;
        auto [node1, node2] = get_nodes(bg::get<0>(start_point), bg::get<1>(start_point), bg::get<2>(start_point), bg::get<0>(end_point),
                                        bg::get<1>(end_point), bg::get<2>(end_point), patch_layers);

        lm_wire.add_path(node1, node2);
#if 0
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

        if (node1->get_node_data()->get_pin_id() != 0) {
          pin_ids_paths.insert(node1->get_node_data()->get_pin_id());
        }
        if (node2->get_node_data()->get_pin_id() != 0) {
          pin_ids_paths.insert(node2->get_node_data()->get_pin_id());
        }
#endif
      });

      lm_net->addWire(lm_wire);
    }

    if (net_id % 1000 == 0) {
      LOG_INFO << "Read nets : " << net_id << " / " << (int) idb_nets.size();
    }
#if 0
    if (pin_ids_wires.size() != lm_net->get_pin_ids().size()) {
      LOG_WARNING << "Warning, pin size mismatch, net id : " << net_id << " net num : " << lm_net->get_pin_ids().size()
                  << " wire num : " << pin_ids_wires.size() << " path num: " << pin_ids_paths.size();
    }
#endif
    net_id++;
  }

  LOG_INFO << "Read nets : " << idb_nets.size() << " / " << (int) idb_nets.size();

  return true;
}

}  // namespace ilm