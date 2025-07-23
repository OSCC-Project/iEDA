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
#include "guiConfig.h"
#include "idbfastsetup.h"
#include "omp.h"
/// this is hord code for debug graph
void IdbSpeedUpSetup::showGraph(std::map<int, ivec::VecNet> net_map) {
  std::cout << "begin show graph" << std::endl;
  for (auto [net_id, net] : net_map) {
    auto container = _gui_design->get_drc_container("Metal Short");
    if (container != nullptr) {
      for (auto wire : net.get_wires()) {
        auto& [node1, node2] = wire.get_connected_nodes();
        if (node1 == nullptr && node2 == nullptr) {
          std::cout << "error node...." << std::endl;
          continue;
        }

        std::string layer1 = "M" + std::to_string(node1->get_layer_id() / 2 + 1);
        auto drc_list1     = container->findDrcList(layer1);
        if (drc_list1 == nullptr) {
          std::cout << "error drc_list1...." << layer1 << std::endl;
          continue;
        }

        QRectF rect1 =
            _transform.db_to_guidb_rect(node1->get_x() - 20, node1->get_y() - 20, node1->get_x() + 20, node1->get_y() + 20);
        GuiSpeedupDrc* item1 = drc_list1->findItem(rect1.center());
        if (item1 == nullptr) {
          return;
        }
        item1->add_rect(rect1);

        std::string layer2 = "M" + std::to_string(node2->get_layer_id() / 2 + 1);
        auto drc_list2     = container->findDrcList(layer2);
        if (drc_list2 == nullptr) {
          std::cout << "error drc_list2...." << std::endl;
          continue;
        }

        int detal    = 20;
        QRectF rect2 = _transform.db_to_guidb_rect(node2->get_x() - detal, node2->get_y() - detal, node2->get_x() + detal,
                                                   node2->get_y() + detal);
        GuiSpeedupDrc* item2 = drc_list2->findItem(rect2.center());
        if (item2 == nullptr) {
          return;
        }
        item2->add_rect(rect2);

        for (auto& [path_node1, path_node2] : wire.get_paths()) {
          if (path_node1 == nullptr && path_node2 == nullptr) {
            std::cout << "error node...." << std::endl;
            continue;
          }
          std::string layer_path1 = "M" + std::to_string(path_node1->get_layer_id() / 2 + 1);
          std::string layer_path2 = "M" + std::to_string(path_node2->get_layer_id() / 2 + 1);
          if (layer_path1 != layer_path2) {
            continue;
          }

          int llx = std::min(path_node1->get_x(), path_node2->get_x());
          int lly = std::min(path_node1->get_y(), path_node2->get_y());
          int urx = std::max(path_node1->get_x(), path_node2->get_x());
          int ury = std::max(path_node1->get_y(), path_node2->get_y());

          auto path_list = container->findDrcList(layer_path1);
          if (path_list == nullptr) {
            std::cout << "error path_list...." << std::endl;
            continue;
          }

          /// horizontal
          if (lly == ury) {
            lly -= 3;
            ury += 3;
          } else {
            llx -= 3;
            urx += 3;
          }
          QRectF rect_path         = _transform.db_to_guidb_rect(llx, lly, urx, ury);
          GuiSpeedupDrc* item_path = path_list->findItem(rect_path.center());
          if (item_path == nullptr) {
            std::cout << "error item_path...." << std::endl;
            continue;
          }

          item_path->add_rect(rect_path);
        }
      }
    }
  }
  std::cout << "end show graph" << std::endl;
}