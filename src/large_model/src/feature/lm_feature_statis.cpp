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

#include "lm_feature_statis.h"

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

void LmFeatureStatis::build()
{
  ieda::Stats stats;

  LOG_INFO << "LM build statis feature start...";

  auto& patch_layers = _layout->get_patch_layers();

  auto& net_map = _layout->get_graph().get_net_map();
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < (int) net_map.size(); ++i) {
    auto it = net_map.begin();
    std::advance(it, i);
    auto& lm_net = it->second;

    int net_row_min = INT32_MAX;
    int net_row_max = INT32_MIN;
    int net_col_min = INT32_MAX;
    int net_col_max = INT32_MIN;

    auto* net_feature = lm_net.get_feature(true);
    for (auto& wire : lm_net.get_wires()) {
      auto* wire_feature = wire.get_feature(true);

      for (auto& [node1, node2] : wire.get_paths()) {
        if (node1->get_layer_id() == node2->get_layer_id()) {
          auto order = node1->get_layer_id();
          auto* patch_layer = patch_layers.findPatchLayer(order);
          auto& grid = patch_layer->get_grid();

          /// set feature
          wire_feature->wire_width = patch_layer->get_wire_width();

          int min_row = std::min(node1->get_row_id(), node2->get_row_id());
          int max_row = std::max(node1->get_row_id(), node2->get_row_id());
          int min_col = std::min(node1->get_col_id(), node2->get_col_id());
          int max_col = std::max(node1->get_col_id(), node2->get_col_id());

          net_row_min = std::min(net_row_min, min_row);
          net_row_max = std::max(net_row_max, max_row);
          net_col_min = std::min(net_col_min, min_col);
          net_col_max = std::max(net_col_max, max_col);

          int horizontal_len = (max_col - min_col) * gridInfoInst.x_step;
          int vertical_len = (max_row - min_row) * gridInfoInst.y_step;

          wire_feature->wire_len += (horizontal_len + vertical_len);

          /// some feature label on node
          for (int row = min_row; row <= max_row; ++row) {
            for (int col = min_col; col <= max_col; ++col) {
              auto* node = grid.get_node(row, col);
              if (node == nullptr || node->get_node_data() == nullptr) {
                continue;
              }
            }
          }
        } else {
          /// via feature
          net_feature->via_num += 1;
        }
      }

      net_feature->wire_len += wire_feature->wire_len;
      net_feature->llx = gridInfoInst.calculate_x(net_col_min);
      net_feature->urx = gridInfoInst.calculate_x(net_col_max);
      net_feature->lly = gridInfoInst.calculate_y(net_row_min);
      net_feature->ury = gridInfoInst.calculate_y(net_row_max);
      net_feature->fanout = lm_net.get_pin_ids().size() - 1;
    }

    if (i % 1000 == 0) {
      LOG_INFO << "Read nets : " << i << " / " << (int) net_map.size();
    }
  }

  LOG_INFO << "Read nets : " << (int) net_map.size() << " / " << (int) net_map.size();
  LOG_INFO << "LM memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "LM elapsed time " << stats.elapsedRunTime() << " s";
  LOG_INFO << "LM build statis feature end...";
}

}  // namespace ilm