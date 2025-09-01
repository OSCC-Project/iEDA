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

#include "vec_file.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "Log.hh"
#include "idm.h"
#include "json_parser.h"
#include "omp.h"
#include "usage.hh"
#include "vec_grid_info.h"

namespace ivec {

void VecLayoutFileIO::makeDir(std::string dir)
{
  namespace fs = std::filesystem;
  if (false == fs::exists(dir) || false == fs::is_directory(dir)) {
    fs::create_directories(dir);
  }
}

bool VecLayoutFileIO::saveJson()
{
  LOG_INFO << "Vectorization save json start... dir = " << _path;

  makeDir(_path);

  /// save tech
  saveJsonTech();

  /// save cells
  saveJsonCells();

  /// save instances
  saveJsonInstances();

  /// save graph
  saveJsonNets();

  /// save patch
  saveJsonPatchs();

  LOG_INFO << "Vectorization save json end... dir = " << _path;

  return true;
}

bool VecLayoutFileIO::saveJsonNets()
{
  ieda::Stats stats;
  LOG_INFO << "Vectorization save json net start...";
  makeDir(_path + "/nets/");

  auto& net_map = _layout->get_graph().get_net_map();
  const int BATCH_SIZE = 1500;  // 可根据系统性能调整批量大小
  const int num_threads = omp_get_max_threads();
  const int NETS_PER_FILE = 1000;  // 每个文件存储的net数量

  // 预先将map的键值对复制到vector中，避免O(N^2)的迭代复杂度
  std::vector<std::pair<int, VecNet*>> net_vec;
  net_vec.reserve(net_map.size());
  for (auto& [net_id, vec_net] : net_map) {
    net_vec.emplace_back(net_id, &vec_net);
  }

  // 计算需要的文件数量
  int num_files = (net_vec.size() + NETS_PER_FILE - 1) / NETS_PER_FILE;

  // 用于收集所有线程生成的JSON数据
  std::vector<std::vector<std::pair<int, json>>> thread_batches(num_threads);

  int total = 0;
#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    auto& local_batch = thread_batches[thread_id];
    local_batch.reserve(BATCH_SIZE + 100);  // 预分配空间

#pragma omp for schedule(dynamic, 100) reduction(+ : total)
    for (int i = 0; i < (int) net_vec.size(); ++i) {
      // 直接O(1)访问vector元素，而不是O(i)的std::advance
      const auto& [net_id, vec_net_ptr] = net_vec[i];
      auto& vec_net = *vec_net_ptr;
      auto* idb_net = dmInst->get_idb_design()->get_net_list()->get_net_list()[net_id];

      json json_net;
      {
        json_net["id"] = net_id;
        json_net["name"] = vec_net.get_net_name();

        /// net feature
        {
          json json_feature = {};
          auto net_feature = vec_net.get_feature();
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
            json_feature["layer_ratio"] = net_feature->layer_ratio;
            json_feature["rsmt"] = net_feature->rsmt;
          }
          json_net["feature"] = json_feature;
        }

        /// pins
        {
          json json_pins = json::array();
          auto& pin_list = vec_net.get_pin_list();

          for (auto& [pin_id, vec_pin] : pin_list) {
            json json_pin;
            json_pin["id"] = pin_id;
            json_pin["i"] = vec_pin.instance_name;
            json_pin["p"] = vec_pin.pin_name;
            json_pin["driver"] = vec_pin.is_driver ? 1 : 0;  /// 1 : driver, 0 : load
            json_pins.push_back(json_pin);
          }

          json_net["pin_num"] = vec_net.get_pin_list().size();
          json_net["pins"] = json_pins;
        }

        /// wires
        auto& wires = vec_net.get_wires();
        json_net["wire_num"] = wires.size();
        total += wires.size();

        json json_wires = json::array();

        for (auto& wire : wires) {
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
              auto& paths = wire.get_paths();
              json_wire["path_num"] = paths.size();

              json json_paths = json::array();

              for (auto& [node1, node2] : paths) {
                json json_node = makeNodePair(node1, node2);
                json_paths.push_back(json_node);
              }

              json_wire["paths"] = json_paths;
            }

            /// patch id
            {
              auto& patches = wire.get_patchs();
              json_wire["patch_num"] = patches.size();

              json json_patchs = json::array();

              for (auto& [patch_id, layer_ids] : patches) {
                json_patchs.push_back(patch_id);
              }

              json_wire["patchs"] = json_patchs;
            }
          }

          json_wires.push_back(json_wire);
        }
        json_net["wires"] = json_wires;

        // routing graph
        auto routing_graph = vec_net.get_routing_graph();
        json json_routing_graph = {};
        json json_routing_graph_vertices = json::array();
        json json_routing_graph_edges = json::array();
        std::ranges::for_each(routing_graph.vertices, [&](const NetRoutingVertex& vertex) {
          json json_vertex;
          json_vertex["id"] = vertex.id;
          json_vertex["is_pin"] = vertex.is_pin ? 1 : 0;                // 1: pin, 0: non-pin
          json_vertex["is_driver_pin"] = vertex.is_driver_pin ? 1 : 0;  // 1: driver pin, 0: load pin
          json_vertex["x"] = vertex.point.x;
          json_vertex["y"] = vertex.point.y;
          json_vertex["layer_id"] = vertex.point.layer_id;
          json_routing_graph_vertices.push_back(json_vertex);
        });
        std::ranges::for_each(routing_graph.edges, [&](const NetRoutingEdge& edge) {
          json json_edge;
          json_edge["source_id"] = edge.source_id;
          json_edge["target_id"] = edge.target_id;
          json_edge["path"] = json::array();
          std::ranges::for_each(edge.path, [&](const NetRoutingPoint& point) {
            json json_point;
            json_point["x"] = point.x;
            json_point["y"] = point.y;
            json_point["layer_id"] = point.layer_id;
            json_edge["path"].push_back(json_point);
          });
          json_routing_graph_edges.push_back(json_edge);
        });
        json_routing_graph["vertices"] = json_routing_graph_vertices;
        json_routing_graph["edges"] = json_routing_graph_edges;
        json_net["routing_graph"] = json_routing_graph;
      }

      // 将结果添加到本地批次中，存储net_id和对应的json数据
      local_batch.emplace_back(net_id, std::move(json_net));

      if (i % 1000 == 0) {
#pragma omp critical(log)
        {
          LOG_INFO << "Processing net : " << i << " / " << net_vec.size();
        }
      }
    }
  }

  // 并行区域结束后，合并所有线程的结果
  LOG_INFO << "JSON generation completed, merging results...";

  // 创建一个映射，将net_id映射到对应的json数据
  std::map<int, json> all_nets;
  for (const auto& batch : thread_batches) {
    for (const auto& [net_id, json_data] : batch) {
      all_nets[net_id] = json_data;
    }
  }

  // 批量写入文件
  LOG_INFO << "Starting batch file writing...";

  // 计算需要写入的文件数量
  int total_files = (all_nets.size() + NETS_PER_FILE - 1) / NETS_PER_FILE;

#pragma omp parallel for schedule(dynamic, 1) num_threads(std::min(num_threads, 8))
  for (int file_idx = 0; file_idx < total_files; ++file_idx) {
    // 计算当前文件包含的网络范围
    int start_net_idx = file_idx * NETS_PER_FILE;
    int end_net_idx = std::min((file_idx + 1) * NETS_PER_FILE - 1, (int) all_nets.size() - 1);

    // 创建文件名格式: net_START_END.json
    std::string filename = "net_" + std::to_string(start_net_idx) + "_" + std::to_string(end_net_idx) + ".json";
    std::string full_path = _path + "/nets/" + filename;

    // 创建一个包含当前批次网络的数组
    json batch_json = json::array();

    // 找到这个范围内的所有网络
    auto it = all_nets.begin();
    std::advance(it, start_net_idx);

    for (int i = start_net_idx; i <= end_net_idx && it != all_nets.end(); ++i, ++it) {
      batch_json.push_back(it->second);
    }

    std::ofstream file_stream(full_path);
    // file_stream << std::setw(4) << batch_json;
    file_stream << batch_json;
    file_stream.close();

#pragma omp critical(log)
    {
      LOG_INFO << "Writing files: " << (file_idx + 1) * NETS_PER_FILE << " / " << all_nets.size();
    }
  }

  LOG_INFO << "Saved nets: " << all_nets.size() << " / " << net_vec.size();
  LOG_INFO << "Total wires: " << total;
  LOG_INFO << "Vectorization memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "Vectorization elapsed time " << stats.elapsedRunTime() << " s";
  LOG_INFO << "Vectorization save json net end...";

  return true;
}

bool VecLayoutFileIO::saveJsonPatchs()
{
  ieda::Stats stats;
  LOG_INFO << "Vectorization save json patchs start...";
  makeDir(_path + "/patchs/");

  if (!_patch_grid) {
    return false;
  }

  auto& patchs = _patch_grid->get_patchs();
  const int BATCH_SIZE = 1500;  // 可根据系统性能调整批量大小
  const int num_threads = omp_get_max_threads();
  const int PATCHS_PER_FILE = 1000;  // 每个文件存储的patch数量

  // 预先将map的键值对复制到vector中，避免O(N²)的迭代复杂度
  std::vector<std::pair<int, VecPatch*>> patch_vec;
  patch_vec.reserve(patchs.size());
  for (auto& [patch_id, patch] : patchs) {
    patch_vec.emplace_back(patch_id, &patch);
  }

  // 计算需要的文件数量
  int num_files = (patch_vec.size() + PATCHS_PER_FILE - 1) / PATCHS_PER_FILE;

  // 用于收集所有线程生成的JSON数据
  std::vector<std::vector<std::pair<int, json>>> thread_batches(num_threads);

#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    auto& local_batch = thread_batches[thread_id];
    local_batch.reserve(BATCH_SIZE + 100);  // 预分配空间

#pragma omp for schedule(dynamic, 100)
    for (int i = 0; i < (int) patch_vec.size(); ++i) {
      const auto& [patch_id, patch_ptr] = patch_vec[i];
      auto& patch = *patch_ptr;

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
        json_patch["timing"] = patch.timing_map;
        json_patch["power"] = patch.power_map;
        json_patch["IR_drop"] = patch.ir_drop_map;

        json json_sub_nets = json::array();
        std::unordered_map<int, json> unique_json_sub_nets;

        json json_layers = json::array();

        std::vector<std::pair<int, VecPatchLayer*>> layer_vec;
        layer_vec.reserve(patch.get_layer_map().size());
        for (auto& [layer_id, patch_layer] : patch.get_layer_map()) {
          layer_vec.emplace_back(layer_id, &patch_layer);
        }

        for (const auto& [layer_id, patch_layer_ptr] : layer_vec) {
          auto& patch_layer = *patch_layer_ptr;
          json json_layer = {};
          json_layer["id"] = layer_id;
          // layer feature
          {
            json json_layer_feature;

            json_layer_feature["wire_width"] = patch_layer.wire_width;
            json_layer_feature["wire_len"] = patch_layer.wire_len;
            json_layer_feature["wire_density"] = (patch_layer.wire_width * patch_layer.wire_len) / static_cast<double>(area);
            json_layer_feature["congestion"] = patch_layer.congestion;

            json_layer["feature"] = json_layer_feature;
          }
          /// sub net in patch for each layer
          auto& sub_nets = patch_layer.get_sub_nets();
          json_layer["net_num"] = sub_nets.size();

          json json_nets = json::array();

          std::vector<std::pair<int, VecNet*>> subnet_vec;
          subnet_vec.reserve(sub_nets.size());
          for (auto& [net_id, vec_net] : sub_nets) {
            subnet_vec.emplace_back(net_id, &vec_net);
          }

          for (const auto& [net_id, vec_net_ptr] : subnet_vec) {
            auto& vec_net = *vec_net_ptr;
            json json_net = {};
            json_net["id"] = net_id;

            // subnet
            auto* idb_net = dmInst->get_idb_design()->get_net_list()->get_net_list()[net_id];
            json json_sub_net;
            json_sub_net["id"] = net_id;
            json_sub_net["llx"] = std::max(idb_net->get_bounding_box()->get_low_x(), llx);
            json_sub_net["lly"] = std::max(idb_net->get_bounding_box()->get_low_y(), lly);
            json_sub_net["urx"] = std::min(idb_net->get_bounding_box()->get_high_x(), urx);
            json_sub_net["ury"] = std::min(idb_net->get_bounding_box()->get_high_y(), ury);
            unique_json_sub_nets[net_id] = json_sub_net;

            /// wires
            auto& wires = vec_net.get_wires();
            json_net["wire_num"] = wires.size();

            json json_wires = json::array();

            for (auto& wire : wires) {
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
                  auto& paths = wire.get_paths();
                  json_wire["path_num"] = paths.size();

                  json json_paths = json::array();

                  for (auto& [node1, node2] : paths) {
                    json json_node = makeNodePair(node1, node2);
                    json_paths.push_back(json_node);
                  }

                  json_wire["paths"] = json_paths;
                }
              }

              json_wires.push_back(json_wire);
            }
            json_net["wires"] = json_wires;
            json_nets.push_back(json_net);
          }
          json_layer["nets"] = json_nets;

          json_layers.push_back(json_layer);
        }

        for (const auto& [net_id, json_sub_net] : unique_json_sub_nets) {
          json_sub_nets.push_back(json_sub_net);
        }
        json_patch["sub_nets"] = json_sub_nets;
        json_patch["patch_layer"] = json_layers;
      }

      // 将结果添加到本地批次中，存储patch_id和对应的json数据
      local_batch.emplace_back(patch_id, std::move(json_patch));

      if (i % 1000 == 0) {
#pragma omp critical(log)
        {
          LOG_INFO << "Processing patch : " << i << " / " << patch_vec.size();
        }
      }
    }
  }

  // 并行区域结束后，合并所有线程的结果
  LOG_INFO << "JSON generation completed, merging results...";

  // 创建一个映射，将patch_id映射到对应的json数据
  std::map<int, json> all_patches;
  for (const auto& batch : thread_batches) {
    for (const auto& [patch_id, json_data] : batch) {
      all_patches[patch_id] = json_data;
    }
  }

  // 批量写入文件
  LOG_INFO << "Starting batch file writing...";

  // 计算需要写入的文件数量
  int total_files = (all_patches.size() + PATCHS_PER_FILE - 1) / PATCHS_PER_FILE;

#pragma omp parallel for schedule(dynamic, 1) num_threads(std::min(num_threads, 8))
  for (int file_idx = 0; file_idx < total_files; ++file_idx) {
    // 计算当前文件包含的patch范围
    int start_patch_idx = file_idx * PATCHS_PER_FILE;
    int end_patch_idx = std::min((file_idx + 1) * PATCHS_PER_FILE - 1, (int) all_patches.size() - 1);

    // 创建文件名格式: patch_START_END.json
    std::string filename = "patch_" + std::to_string(start_patch_idx) + "_" + std::to_string(end_patch_idx) + ".json";
    std::string full_path = _path + "/patchs/" + filename;

    // 创建一个包含当前批次patch的数组
    json batch_json = json::array();

    // 找到这个范围内的所有patch
    auto it = all_patches.begin();
    std::advance(it, start_patch_idx);

    for (int i = start_patch_idx; i <= end_patch_idx && it != all_patches.end(); ++i, ++it) {
      batch_json.push_back(it->second);
    }

    std::ofstream file_stream(full_path);
    file_stream << batch_json;
    file_stream.close();

#pragma omp critical(log)
    {
      LOG_INFO << "Writing files: " << (file_idx + 1) * PATCHS_PER_FILE << " / " << all_patches.size();
    }
  }

  LOG_INFO << "Saved patches: " << all_patches.size() << " / " << patch_vec.size();
  LOG_INFO << "Vectorization memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "Vectorization elapsed time " << stats.elapsedRunTime() << " s";
  LOG_INFO << "Vectorization save json patchs end...";

  return true;
}

json VecLayoutFileIO::makeNodePair(VecNode* node1, VecNode* node2)
{
  json json_node;

  json_node["id1"] = node1->get_node_id();
  json_node["x1"] = node1->get_x();
  json_node["y1"] = node1->get_y();
  json_node["real_x1"] = node1->get_realx();
  json_node["real_y1"] = node1->get_realy();
  json_node["r1"] = node1->get_row_id();                   /// row
  json_node["c1"] = node1->get_col_id();                   /// col
  json_node["l1"] = node1->get_layer_id();                 /// layer order
  json_node["p1"] = node1->get_node_data()->get_pin_id();  /// pin

  json_node["id2"] = node2->get_node_id();
  json_node["x2"] = node2->get_x();
  json_node["y2"] = node2->get_y();
  json_node["real_x2"] = node2->get_realx();
  json_node["real_y2"] = node2->get_realy();
  json_node["r2"] = node2->get_row_id();                   /// row
  json_node["c2"] = node2->get_col_id();                   /// col
  json_node["l2"] = node2->get_layer_id();                 /// layer order
  json_node["p2"] = node2->get_node_data()->get_pin_id();  /// pin

  //   json_node["via"] = -1;  /// via id

  if (node1->get_layer_id() != node2->get_layer_id()) {
    /// save via id
    auto top_layer_node = node1->get_layer_id() > node2->get_layer_id() ? node1 : node2;
    json_node["via"] = top_layer_node->get_node_data()->get_via_id();
  }
  return json_node;
}

bool VecLayoutFileIO::saveJsonTech()
{
  ieda::Stats stats;
  LOG_INFO << "Vectorization save json tech start...";
  makeDir(_path + "/tech/");

  json json_tech;
  {
    /// layers
    {
      auto& layer_map = _layout->get_layout_layers().get_layout_layer_map();
      json_tech["layer_num"] = layer_map.size();

      auto json_layer_list = json::array();
      for (auto& [id, vec_layer] : layer_map) {
        json json_layer;
        json_layer["id"] = vec_layer.get_layer_order();
        json_layer["name"] = vec_layer.get_layer_name();

        json_layer_list.push_back(json_layer);
      }

      json_tech["layers"] = json_layer_list;
    }

    /// vias
    {
      auto& via_map = _layout->get_via_name_map();
      json_tech["via_num"] = via_map.size();

      auto json_via_list = json::array();
      for (auto& [via_name, id] : via_map) {
        json json_via;
        json_via["id"] = id;
        json_via["name"] = via_name;

        json_via_list.push_back(json_via);
      }

      json_tech["vias"] = json_via_list;
    }
  }

  std::string filename = _path + "/tech/tech.json";
  std::ofstream file_stream(filename);
  file_stream << json_tech;
  //   file_stream << std::setw(4) << json_tech;
  file_stream.close();

  LOG_INFO << "Vectorization memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "Vectorization elapsed time " << stats.elapsedRunTime() << " s";
  LOG_INFO << "Vectorization save json tech end...";

  return true;
}

bool VecLayoutFileIO::saveJsonCells()
{
  ieda::Stats stats;
  LOG_INFO << "Vectorization save json cells start...";
  makeDir(_path + "/tech/");

  json json_cells;
  {
    auto& cell_map = _layout->get_cells().get_cell_map();
    json_cells["cell_num"] = cell_map.size();

    auto json_cell_list = json::array();
    for (auto& [id, vec_cell] : cell_map) {
      json json_cell;
      json_cell["id"] = vec_cell.id;
      json_cell["name"] = vec_cell.name;
      json_cell["width"] = vec_cell.width;
      json_cell["height"] = vec_cell.height;

      json_cell_list.push_back(json_cell);
    }

    json_cells["cells"] = json_cell_list;
  }

  std::string filename = _path + "/tech/cells.json";
  std::ofstream file_stream(filename);
  file_stream << json_cells;
  file_stream.close();

  LOG_INFO << "Vectorization memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "Vectorization elapsed time " << stats.elapsedRunTime() << " s";
  LOG_INFO << "Vectorization save json cells end...";

  return true;
}

bool VecLayoutFileIO::saveJsonInstances()
{
  ieda::Stats stats;
  LOG_INFO << "Vectorization save json instances start...";
  makeDir(_path + "/instances/");

  json json_insts;
  {
    auto& inst_map = _layout->get_instances().get_instance_map();
    json_insts["instance_num"] = inst_map.size();

    auto json_inst_list = json::array();
    for (auto& [id, vec_inst] : inst_map) {
      json json_inst;
      json_inst["id"] = vec_inst.id;
      json_inst["cell_id"] = vec_inst.cell_id;
      json_inst["name"] = vec_inst.name;
      json_inst["x"] = vec_inst.x;
      json_inst["y"] = vec_inst.y;
      json_inst["width"] = vec_inst.width;
      json_inst["height"] = vec_inst.height;
      json_inst["llx"] = vec_inst.llx;
      json_inst["lly"] = vec_inst.lly;
      json_inst["urx"] = vec_inst.urx;
      json_inst["ury"] = vec_inst.ury;

      json_inst_list.push_back(json_inst);
    }

    json_insts["instances"] = json_inst_list;
  }

  std::string filename = _path + "/instances/instances.json";
  std::ofstream file_stream(filename);
  file_stream << json_insts;
  file_stream.close();

  LOG_INFO << "Vectorization memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "Vectorization elapsed time " << stats.elapsedRunTime() << " s";
  LOG_INFO << "Vectorization save json instances end...";

  return true;
}

bool VecLayoutFileIO::readJsonNets()
{
  namespace fs = std::filesystem;

  auto find_json_files = [&](const fs::path& folder) -> std::vector<fs::path> {
    std::vector<fs::path> result;
    for (const auto& entry : fs::directory_iterator(folder)) {
      if (entry.is_regular_file() && entry.path().extension() == ".json") {
        result.emplace_back(entry.path());
      }
    }
    return result;
  };

  auto* idb_nets = dmInst->get_idb_design()->get_net_list();
  auto* idb_layers = dmInst->get_idb_layout()->get_layers();
  auto* idb_vias = dmInst->get_idb_layout()->get_via_list();

  idb_nets->clear_wire_list();

  auto read_file = [&](std::string file) {
    LOG_INFO << "read " << file;
    nlohmann::json json;
    std::ifstream file_stream(file);
    file_stream >> json;

    /// parse vector nets
    for (auto& json_net : json.items()) {
      std::string net_name = json_net.value()["name"];
      auto* idb_net = idb_nets->find_net(net_name);
      if (idb_net == nullptr) {
        continue;
      }

      idb_net->clear_wire_list();
      auto* idb_wire_list = idb_net->get_wire_list();

      auto json_wires = json_net.value()["wires"];
      auto* idb_wire = idb_wire_list->add_wire();
      for (auto& json_wire : json_wires.items()) {
        idb_wire->set_wire_state(idb::IdbWiringStatement::kRouted);

        auto json_paths = json_wire.value()["paths"];
        int32_t path_new = false;
        for (auto& json_path : json_paths.items()) {
          int p1 = json_path.value()["p1"];
          int p2 = json_path.value()["p2"];
          if (p1 != -1 && p2 != -1 && p1 == p2) {
            /// ignore single pin shape
            continue;
          }

          int x1 = json_path.value()["real_x1"];
          int y1 = json_path.value()["real_y1"];
          int l1 = json_path.value()["l1"];
          std::string layer1 = _layout->findLayerName(l1);
          auto* idb_layer1 = idb_layers->find_layer(layer1);
          int x2 = json_path.value()["real_x2"];
          int y2 = json_path.value()["real_y2"];
          int l2 = json_path.value()["l2"];
          std::string layer2 = _layout->findLayerName(l2);
          auto* idb_layer2 = idb_layers->find_layer(layer2);

          auto* idb_segment = new IdbRegularWireSegment();
          if (l1 == l2) {
            if (x1 == x2 && y1 == y2) {
              delete idb_segment;
              continue;
            }

            if (x1 == x2 || y1 == y2) {
            } else {
              /// use grid coordinate
              x1 = json_path.value()["x1"];
              y1 = json_path.value()["y1"];
              x2 = json_path.value()["x2"];
              y2 = json_path.value()["y2"];
            }

            idb_segment->set_layer(idb_layer1);
            idb_segment->add_point(x1, y1);
            idb_segment->add_point(x2, y2);

          } else {
            int via_id = json_path.value()["via"];
            if (via_id == -1) {
              LOG_WARNING << "via id error";
              delete idb_segment;
              continue;
            }

            auto* top_layer = l1 > l2 ? idb_layer1 : idb_layer2;
            auto x = l1 > l2 ? x1 : x2;
            auto y = l1 > l2 ? y1 : y2;
            idb_segment->set_layer(top_layer);
            idb_segment->set_is_via(true);
            idb_segment->add_point(x, y);

            auto via_name = _layout->findViaName(via_id);
            if (via_name != "") {
              auto* idb_via = idb_vias->find_via(via_name);
              auto* idb_via_new = idb_segment->copy_via(idb_via);
              idb_via_new->set_coordinate(x, y);
            } else {
              /// use default via, tbd
              LOG_WARNING << "can not find via";
              delete idb_segment;
              continue;
            }
          }

          if (path_new == false) {
            idb_segment->set_layer_as_new();
            path_new = true;
          }

          idb_wire->add_segment(idb_segment);
        }
      }
    }

    file_stream.close();
  };

  omp_lock_t lck;
  omp_init_lock(&lck);

  auto net_dir = _path + "/nets/";
#pragma omp parallel for schedule(dynamic)
  for (auto& file : find_json_files(net_dir)) {
    // omp_set_lock(&lck);

    read_file(file);

    // omp_unset_lock(&lck);
  }

  omp_destroy_lock(&lck);

  LOG_INFO << "read nets success.";

  return true;
}

bool VecLayoutFileIO::readJsonNetsPattern()
{
  namespace fs = std::filesystem;

  auto* idb_nets = dmInst->get_idb_design()->get_net_list();
  auto* idb_layers = dmInst->get_idb_layout()->get_layers();
  auto* idb_vias = dmInst->get_idb_design()->get_via_list();

  idb_nets->clear_wire_list();

  LOG_INFO << "read " << _path;
  nlohmann::json json;
  std::ifstream file_stream(_path);
  file_stream >> json;

  /// parse vector nets
  auto json_nets = json["nets"];
  for (auto& json_net : json_nets) {
    std::string net_name = json_net["net_name"];
    auto* idb_net = idb_nets->find_net(net_name);
    if (idb_net == nullptr) {
      std::cout << "can not find net " << net_name << std::endl;
      continue;
    }

    idb_net->clear_wire_list();
    auto* idb_wire_list = idb_net->get_wire_list();

    auto json_edges = json_net["edges"];
    auto* idb_wire = idb_wire_list->add_wire();

    bool path_new = false;
    for (auto& json_edge : json_edges.items()) {
      idb_wire->set_wire_state(idb::IdbWiringStatement::kRouted);

      auto json_start = json_edge.value()["start"];
      int x_start = json_start["x"];
      int y_start = json_start["y"];
      int layer_index_start = json_start["layer"];

      auto json_end = json_edge.value()["end"];
      int x_end = json_end["x"];
      int y_end = json_end["y"];
      int layer_index_end = json_end["layer"];

      if (layer_index_start == layer_index_end) {
        if (x_start == x_end && y_start == y_end) {
          std::cout << "same node  " << x_start << "  " << y_start << " , " << x_end << "  " << y_end << std::endl;
          continue;
        }

        if (x_start == x_end || y_start == y_end) {
        } else {
          std::cout << "illegal node  " << x_start << "  " << y_start << " ,  " << x_end << "  " << y_end << std::endl;
          continue;
        }

        std::string layer_metal = _layout->findLayerName(layer_index_start);
        auto* idb_layer_metal = idb_layers->find_layer(layer_metal);

        auto* idb_segment = idb_wire->add_segment();

        idb_segment->set_layer(idb_layer_metal);
        idb_segment->add_point(x_start, y_start);
        idb_segment->add_point(x_end, y_end);

        if (path_new == false) {
          idb_segment->set_layer_as_new();
          path_new = true;
        }
      } else {
        auto top_order = std::max(layer_index_start, layer_index_end);
        auto bottom_order = std::min(layer_index_start, layer_index_end);

        for (auto layer_order = bottom_order; layer_order <= top_order; layer_order += 2) {
          std::string bottom_layer_name = _layout->findLayerName(layer_order);
          auto* bottom_layer = idb_layers->find_layer(bottom_layer_name);
          std::string top_layer_name = _layout->findLayerName(layer_order + 2);
          auto* top_layer = idb_layers->find_layer(top_layer_name);

          auto* idb_segment = idb_wire->add_segment();
          idb_segment->set_layer(top_layer);
          idb_segment->set_is_via(true);
          idb_segment->add_point(x_start, y_start);

          /// create default via
          auto cut_order = (bottom_layer->get_order() + top_layer->get_order()) / 2;
          auto idb_cut_layer = (IdbLayerCut*) idb_layers->find_layer_by_order(cut_order);
          auto via_name = idb_cut_layer->get_name() + "_vector_default";
          auto* idb_via = idb_vias->find_via(via_name);
          if (idb_via == nullptr) {
            idb_via = idb_vias->createVia(via_name, idb_cut_layer);
          }
          auto* idb_via_new = idb_segment->copy_via(idb_via);
          idb_via_new->set_coordinate(x_start, y_start);

          if (path_new == false) {
            idb_segment->set_layer_as_new();
            path_new = true;
          }
        }
      }
    }
  }

  file_stream.close();

  LOG_INFO << "read nets success.";

  return true;
}

}  // namespace ivec