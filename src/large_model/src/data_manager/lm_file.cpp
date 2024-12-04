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

#include "lm_file.h"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "Log.hh"
#include "idm.h"
#include "json.hpp"
#include "omp.h"
#include "usage.hh"

namespace ilm {

using json = nlohmann::ordered_json;

bool LmLayoutFileIO::saveJson(std::string path, std::map<int, LmNet>& net_map)
{
  ieda::Stats stats;

  LOG_INFO << "LM save json start... path = " << path;

  omp_lock_t lck;
  omp_init_lock(&lck);

  json json_root;
  json_root["net_num"] = net_map.size();
  json json_nets = json::array();

  int total = 0;
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < (int) net_map.size(); ++i) {
    auto it = net_map.begin();
    std::advance(it, i);

    auto* idb_net = dmInst->get_idb_design()->get_net_list()->get_net_list()[it->first];

    json json_net;
    {
      json_net["id"] = it->first;
      json_net["name"] = idb_net->get_net_name();

      json json_pins = json::array();
      int pin_num = 0;
      if (idb_net->has_io_pins()) {
        for (auto io_pin : idb_net->get_io_pins()->get_pin_list()) {
          json json_pin;
          json_pin["i"] = "";
          json_pin["p"] = io_pin->get_pin_name();
          json_pins.push_back(json_pin);
          pin_num++;
        }
      }

      for (auto inst_pin : idb_net->get_instance_pin_list()->get_pin_list()) {
        json json_pin;
        json_pin["i"] = inst_pin->get_instance()->get_name();
        json_pin["p"] = inst_pin->get_pin_name();
        json_pins.push_back(json_pin);
        pin_num++;
      }

      json_net["pin_num"] = pin_num;
      json_net["pins"] = json_pins;

      json_net["wire_num"] = it->second.get_wires().size();
      total += it->second.get_wires().size();

      json json_wires = json::array();
      {
        for (auto& wire : it->second.get_wires()) {
          json json_wire = {};
          {
            /// wire
            {
              auto& [node1, node2] = wire.get_connected_nodes();

              json json_node;
              json_node["x1"] = node1->get_x();
              json_node["y1"] = node1->get_y();
              json_node["r1"] = node1->get_row_id();    /// row
              json_node["c1"] = node1->get_col_id();    /// col
              json_node["l1"] = node1->get_layer_id();  /// layer order

              json_node["x2"] = node2->get_x();
              json_node["y2"] = node2->get_y();
              json_node["r2"] = node2->get_row_id();    /// row
              json_node["c2"] = node2->get_col_id();    /// col
              json_node["l2"] = node2->get_layer_id();  /// layer order

              json_wire["wire"] = json_node;
            }

            /// paths
            {
              json_wire["path_num"] = wire.get_paths().size();

              json json_paths = json::array();
              if (wire.get_paths().size() > 1) {
                for (auto& [node1, node2] : wire.get_paths()) {
                  json json_node;
                  json_node["x1"] = node1->get_x();
                  json_node["y1"] = node1->get_y();
                  json_node["r1"] = node1->get_row_id();    /// row
                  json_node["c1"] = node1->get_col_id();    /// col
                  json_node["l1"] = node1->get_layer_id();  /// layer order

                  json_node["x2"] = node2->get_x();
                  json_node["y2"] = node2->get_y();
                  json_node["r2"] = node2->get_row_id();    /// row
                  json_node["c2"] = node2->get_col_id();    /// col
                  json_node["l2"] = node2->get_layer_id();  /// layer order

                  json_paths.push_back(json_node);
                }
              }
              json_wire["paths"] = json_paths;
            }
          }

          json_wires.push_back(json_wire);
        }
      }
      json_net["wires"] = json_wires;
    }
    omp_set_lock(&lck);
    json_nets.push_back(json_net);
    omp_unset_lock(&lck);

    if (i % 1000 == 0) {
      LOG_INFO << "Save net : " << i << " / " << net_map.size();
    }
  }
  LOG_INFO << "Save net : " << net_map.size() << " / " << net_map.size();
  LOG_INFO << "wire size : " << total;

  json_root["nets"] = json_nets;

  omp_destroy_lock(&lck);

  std::ofstream file_stream(path);
  file_stream << std::setw(4) << json_root;

  file_stream.close();

  LOG_INFO << "LM memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "LM elapsed time " << stats.elapsedRunTime() << " s";
  LOG_INFO << "LM save json success... path = " << path;

  return true;
}

}  // namespace ilm