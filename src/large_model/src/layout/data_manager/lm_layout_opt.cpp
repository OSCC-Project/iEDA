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

#include "lm_layout_opt.h"

#include "IdbGeometry.h"
#include "IdbLayer.h"
#include "IdbLayerShape.h"
#include "IdbNet.h"
#include "IdbRegularWire.h"
#include "IdbSpecialNet.h"
#include "IdbSpecialWire.h"
#include "Log.hh"
#include "idm.h"
#include "omp.h"
#include "usage.hh"

namespace ilm {

void LmLayoutOptimize::wirePruning()
{
  ieda::Stats stats;

  LOG_INFO << "LM optimize connections for routing layer start...";

  omp_lock_t lck;
  omp_init_lock(&lck);

  auto& net_map = _layout->get_graph().get_net_map();

  struct ClassifyMap
  {
    std::set<int> pin_ids;
    std::vector<LmNetWire*> wires;
  };

  int connected_num = 0;

  for (auto& [net_id, net] : net_map) {
    auto& pin_ids = net.get_pin_ids();
    auto& wires = net.get_wires();
    if (wires.size() <= 0) {
      continue;
    }

    std::map<int, ClassifyMap> sub_graph;

    std::set<int> count_pins;

    int sub_graph_id = 0;
    for (auto& wire : wires) {
      auto& [node1, node2] = wire.get_connected_nodes();
      if (node1->get_node_data().get_pin_id() != -1) {
        count_pins.insert(node1->get_node_data().get_pin_id());
      }
      if (node2->get_node_data().get_pin_id() != -1) {
        count_pins.insert(node2->get_node_data().get_pin_id());
      }
      //   bool b_insert = false;
      //   for (auto& [id, classify_map] : sub_graph) {
      //     for (auto& sub_graph_wire : classify_map.wires) {
      //       auto& [sub_node1, sub_node2] = sub_graph_wire->get_connected_nodes();
      //       if (node1 == sub_node1 || node1 == sub_node2 || node2 == sub_node1 || node2 == sub_node2) {
      //         classify_map.wires.push_back(&wire);
      //         if (node1->get_node_data().get_pin_id() != -1) {
      //           classify_map.pin_ids.insert(node1->get_node_data().get_pin_id());
      //           b_insert = true;
      //         }

      //         if (node2->get_node_data().get_pin_id() != -1) {
      //           classify_map.pin_ids.insert(node2->get_node_data().get_pin_id());
      //           b_insert = true;
      //         }
      //       }
      //     }
      //   }

      //   if (false == b_insert) {
      //     ClassifyMap new_classify_map;

      //     new_classify_map.wires.push_back(&wire);
      //     if (node1->get_node_data().get_pin_id() != -1) {
      //       new_classify_map.pin_ids.insert(node1->get_node_data().get_pin_id());
      //     }

      //     if (node2->get_node_data().get_pin_id() != -1) {
      //       new_classify_map.pin_ids.insert(node2->get_node_data().get_pin_id());
      //     }

      //     sub_graph.insert(std::make_pair(sub_graph_id++, new_classify_map));
      //   }
    }

    bool b_match = pin_ids.size() == count_pins.size();
    // for (auto& [id, classify_map] : sub_graph) {
    //   if (classify_map.pin_ids.size() == pin_ids.size()) {
    //     b_match = true;
    //   }
    // }

    if (b_match == false) {
      for (auto pin_id : count_pins) {
        LOG_INFO << "count_pin : " << pin_id;
      }

      LOG_INFO << "net disconnected" << net_id;
    } else {
      LOG_INFO << "net connected " << net_id;
      connected_num++;
    }
  }

  LOG_INFO << "net connected ratio " << connected_num << " / " << net_map.size();

  omp_destroy_lock(&lck);

  LOG_INFO << "LM optimize connections for routing layer end...";
}

}  // namespace ilm