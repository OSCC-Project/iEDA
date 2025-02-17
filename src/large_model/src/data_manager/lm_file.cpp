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
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "Log.hh"
#include "idm.h"
#include "lm_grid_info.h"
#include "omp.h"
#include "usage.hh"

namespace ilm {

void LmLayoutFileIO::makeDir(std::string dir)
{
  namespace fs = std::filesystem;
  if (false == fs::exists(dir) || false == fs::is_directory(dir)) {
    fs::create_directories(dir);
  }
}

bool LmLayoutFileIO::saveJson()
{
  LOG_INFO << "LM save json start... dir = " << _dir;

  makeDir(_dir);

  /// save graph
  saveJsonNets();

  /// save patch
  saveJsonPatchs();

  LOG_INFO << "LM save json end... dir = " << _dir;

  return true;
}

bool LmLayoutFileIO::saveJsonNets()
{
  ieda::Stats stats;

  LOG_INFO << "LM save json net start...";

  makeDir(_dir + "/large_model/nets/");

  auto& net_map = _layout->get_graph().get_net_map();

  omp_lock_t lck;
  omp_init_lock(&lck);

  int total = 0;
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < (int) net_map.size(); ++i) {
    auto it = net_map.begin();
    std::advance(it, i);
    auto& net_id = it->first;
    auto& lm_net = it->second;
    auto* idb_net = dmInst->get_idb_design()->get_net_list()->get_net_list()[it->first];

    json json_net;
    {
      json_net["id"] = net_id;
      json_net["name"] = idb_net->get_net_name();

      /// net feature
      {
        json json_feature = {};
        auto net_feature = lm_net.get_feature();
        if (net_feature != nullptr) {
          json_feature["llx"] = net_feature->llx;
          json_feature["lly"] = net_feature->lly;
          json_feature["urx"] = net_feature->urx;
          json_feature["ury"] = net_feature->ury;
          json_feature["wire_len"] = net_feature->wire_len;
          json_feature["via_num"] = net_feature->via_num;
          json_feature["drc_num"] = net_feature->drc_num;
          json_feature["drc_type"] = net_feature->drc_type;
          json_feature["R"] = net_feature->R;
          json_feature["C"] = net_feature->C;
          json_feature["power"] = net_feature->power;
          json_feature["delay"] = net_feature->delay;
          json_feature["slew"] = net_feature->slew;
          json_feature["fanout"] = net_feature->fanout;
          json_feature["aspect_ratio"] = net_feature->aspect_ratio;
          json_feature["width"] = net_feature->width;
          json_feature["height"] = net_feature->height;
          json_feature["area"] = net_feature->area;
          json_feature["volume"] = net_feature->volume;
          json_feature["l_ness"] = net_feature->l_ness;
          json_feature["layer_raito"]  = net_feature->layer_ratio;
        }
        json_net["feature"] = json_feature;
      }

      /// pins
      {
        json json_pins = json::array();
        int pin_num = 0;
        if (idb_net->has_io_pins()) {
          for (auto io_pin : idb_net->get_io_pins()->get_pin_list()) {
            json json_pin;
            json_pin["id"] = _layout->findPinId("", io_pin->get_pin_name());
            json_pin["i"] = "";
            json_pin["p"] = io_pin->get_pin_name();
            json_pins.push_back(json_pin);
            pin_num++;
          }
        }
        for (auto inst_pin : idb_net->get_instance_pin_list()->get_pin_list()) {
          json json_pin;
          json_pin["id"] = _layout->findPinId(inst_pin->get_instance()->get_name(), inst_pin->get_pin_name());
          json_pin["i"] = inst_pin->get_instance()->get_name();
          json_pin["p"] = inst_pin->get_pin_name();
          json_pins.push_back(json_pin);
          pin_num++;
        }
        json_net["pin_num"] = pin_num;
        json_net["pins"] = json_pins;
      }

      /// wires
      json_net["wire_num"] = lm_net.get_wires().size();
      total += lm_net.get_wires().size();
      json json_wires = json::array();
      {
        for (auto& wire : lm_net.get_wires()) {
          json json_wire = {};
          {
            json_wire["id"] = wire.get_id();

            /// wire feature
            {
              json json_feature;
              auto wire_feature = wire.get_feature();
              if (wire_feature != nullptr) {
                json_feature["wire_width"] = wire_feature->wire_width;
                json_feature["wire_len"] = wire_feature->wire_len;
                json_feature["wire_density"] = wire_feature->wire_density;
                json_feature["drc_num"] = wire_feature->drc_num;
                json_feature["R"] = wire_feature->R;
                json_feature["C"] = wire_feature->C;
                json_feature["power"] = wire_feature->power;
                json_feature["delay"] = wire_feature->delay;
                json_feature["slew"] = wire_feature->slew;
                json_feature["congestion"] = wire_feature->congestion;
                json_feature["drc_type"] = wire_feature->drc_type;
              }
              json_wire["feature"] = json_feature;
            }

            /// wire nodes
            {
              auto& [node1, node2] = wire.get_connected_nodes();

              auto json_node = makeNodePair(node1, node2);

              json_wire["wire"] = json_node;
            }

            /// paths
            {
              json_wire["path_num"] = wire.get_paths().size();

              json json_paths = json::array();
              for (auto& [node1, node2] : wire.get_paths()) {
                json json_node = makeNodePair(node1, node2);

                json_paths.push_back(json_node);
              }

              json_wire["paths"] = json_paths;
            }

            /// patch id
            {
              json_wire["patch_num"] = wire.get_patchs().size();
              json json_patchs = json::array();
              for (auto& [patch_id, layer_ids] : wire.get_patchs()) {
                json_patchs.push_back(patch_id);
              }

              json_wire["patchs"] = json_patchs;
            }
          }

          json_wires.push_back(json_wire);
        }
      }
      json_net["wires"] = json_wires;
    }
    omp_set_lock(&lck);
    auto file_name = _dir + "/large_model/nets/net_" + std::to_string(net_id) + ".json";
    std::ofstream file_stream(file_name);
    if (net_id == 0) {
      file_stream << std::setw(4) << json_net;
    } else {
      file_stream << std::setw(0) << json_net;
    }
    // file_stream << std::setw(4) << json_net;
    file_stream.close();
    omp_unset_lock(&lck);

    if (i % 1000 == 0) {
      LOG_INFO << "Save net : " << i << " / " << net_map.size();
    }
  }
  LOG_INFO << "Save net : " << net_map.size() << " / " << net_map.size();
  LOG_INFO << "wire size : " << total;

  omp_destroy_lock(&lck);

  LOG_INFO << "LM memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "LM elapsed time " << stats.elapsedRunTime() << " s";
  LOG_INFO << "LM save json net end...";

  return true;
}

bool LmLayoutFileIO::saveJsonPatchs()
{
  ieda::Stats stats;

  LOG_INFO << "LM save json patchs start...";

  makeDir(_dir + "/large_model/patchs/");

  if (!_patch_grid) {
    return false;
  }

  auto& patchs = _patch_grid->get_patchs();

  omp_lock_t lck;
  omp_init_lock(&lck);

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < (int) patchs.size(); ++i) {
    auto it = patchs.begin();
    std::advance(it, i);
    auto& patch_id = it->first;
    auto& patch = it->second;

    json json_patch;
    {
      auto [llx, lly] = gridInfoInst.get_node_coodinate(patch.rowIdMin, patch.colIdMin);
      auto [urx, ury] = gridInfoInst.get_node_coodinate(patch.rowIdMax, patch.colIdMax);
      int area = (urx - llx) * (ury - lly);

      json_patch["id"] = patch.patch_id;
      json_patch["patch_id_row"] = patch.patch_id_row;
      json_patch["patch_id_col"] = patch.patch_id_col;
      json_patch["llx"] = llx;
      json_patch["lly"] = lly;
      json_patch["urx"] = urx;
      json_patch["ury"] = ury;
      json_patch["row_min"] = patch.rowIdMin;
      json_patch["row_max"] = patch.rowIdMax;
      json_patch["col_min"] = patch.colIdMin;
      json_patch["col_max"] = patch.colIdMax;
      json_patch["area"] = area;
      json_patch["cell_density"] = patch.cell_density;
      json_patch["pin_density"] = patch.pin_density;
      json_patch["net_density"] = patch.net_density;
      json_patch["macro_margin"] = patch.macro_margin;
      json_patch["RUDY_congestion"] = patch.RUDY_congestion;
      json_patch["EGR_congestion"] = patch.EGR_congestion;
      
      json json_layers = json::array();

      for (auto& [layer_id, patch_layer] : patch.get_layer_map()) {
        json json_layer = {};
        json_layer["id"] = layer_id;
        // layer feature
        {
        json json_layer_feature;

        json_layer_feature["wire_width"] = patch_layer.wire_width;
        json_layer_feature["wire_len"] = patch_layer.wire_len;
        json_layer_feature["wire_density"] = (patch_layer.wire_width * patch_layer.wire_len) / static_cast<double>(area);

        json_layer["feature"] = json_layer_feature;
        }
        /// sub net in patch for each layer
        json_layer["net_num"] = patch_layer.get_sub_nets().size();
        json json_nets = json::array();
        for (auto& [net_id, lm_net] : patch_layer.get_sub_nets()) {
          json json_net = {};
          json_net["id"] = net_id;

          /// wires
          json_net["wire_num"] = lm_net.get_wires().size();
          json json_wires = json::array();
          {
            for (auto& wire : lm_net.get_wires()) {
              json json_wire = {};
              {
                json_wire["id"] = wire.get_id();

                /// wire feature
                {
                  json json_feature;
                  auto wire_feature = wire.get_feature();
                  if (wire_feature != nullptr) {
                    json_feature["wire_len"] = wire_feature->wire_len;
                  }
                  json_wire["feature"] = json_feature;
                }

                /// paths
                {
                  json_wire["path_num"] = wire.get_paths().size();
                  json json_paths = json::array();
                  for (auto& [node1, node2] : wire.get_paths()) {
                    json json_node = makeNodePair(node1, node2);
                    json_paths.push_back(json_node);
                  }

                  json_wire["paths"] = json_paths;
                }
              }

              json_wires.push_back(json_wire);
            }
          }
          json_net["wires"] = json_wires;
          json_nets.push_back(json_net);
        }
        json_layer["nets"] = json_nets;

        json_layers.push_back(json_layer);
      }

      json_patch["patch_layer"] = json_layers;
    }

    omp_set_lock(&lck);
    auto file_name = _dir + "/large_model/patchs/patch_" + std::to_string(patch_id) + ".json";
    std::ofstream file_stream(file_name);
    if (i == 1000) {
      file_stream << std::setw(4) << json_patch;
    } else {
      file_stream << std::setw(0) << json_patch;
    }
    file_stream.close();
    omp_unset_lock(&lck);

    if (i % 1000 == 0) {
      LOG_INFO << "Save patch : " << i << " / " << patchs.size();
    }
  }

  omp_destroy_lock(&lck);

  LOG_INFO << "LM memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "LM elapsed time " << stats.elapsedRunTime() << " s";
  LOG_INFO << "LM save json patchs end...";

  return true;
}

json LmLayoutFileIO::makeNodePair(LmNode* node1, LmNode* node2)
{
  json json_node;

  json_node["id1"] = node1->get_node_id();
  json_node["x1"] = node1->get_x();
  json_node["y1"] = node1->get_y();
  json_node["r1"] = node1->get_row_id();                   /// row
  json_node["c1"] = node1->get_col_id();                   /// col
  json_node["l1"] = node1->get_layer_id();                 /// layer order
  json_node["p1"] = node1->get_node_data()->get_pin_id();  /// pin

  json_node["id2"] = node2->get_node_id();
  json_node["x2"] = node2->get_x();
  json_node["y2"] = node2->get_y();
  json_node["r2"] = node2->get_row_id();                   /// row
  json_node["c2"] = node2->get_col_id();                   /// col
  json_node["l2"] = node2->get_layer_id();                 /// layer order
  json_node["p2"] = node2->get_node_data()->get_pin_id();  /// pin
  return json_node;
}

}  // namespace ilm