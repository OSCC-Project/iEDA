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

#include "lm_feature_drc.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "IdbLayer.h"
#include "Log.hh"
#include "idrc_io.h"
#include "lm_grid_info.h"
#include "omp.h"
#include "usage.hh"

namespace ilm {

void LmFeatureDrc::build()
{
  markNodes();
  markNetsAndWires();
}

void LmFeatureDrc::markNodes()
{
  ieda::Stats stats;

  LOG_INFO << "LM mark nodes drc start...";

  omp_lock_t lck;
  omp_init_lock(&lck);

  auto& patch_layers = _layout->get_patch_layers();

  auto detail_drc_map = iplf::drcInst->getDetailCheckResult(_drc_path);
  for (auto& [rule, drc_spot_list] : detail_drc_map) {
    LOG_INFO << "LM mark nodes drc : " << rule << ", size :" << drc_spot_list.size();
#pragma omp parallel for schedule(dynamic)
    for (auto drc_spot : drc_spot_list) {
      /// get row & col
      auto* spot_rect = static_cast<idrc::DrcViolationRect*>(drc_spot);
      int middle_x = (spot_rect->get_llx() + spot_rect->get_urx()) / 2;
      int middle_y = (spot_rect->get_lly() + spot_rect->get_ury()) / 2;

      auto [row, col] = gridInfoInst.findNodeID(middle_x, middle_y);

      /// set to node if node data exist
      auto order = _layout->findLayerId(drc_spot->get_layer()->get_name());
      if (order < 0) {
        continue;
      }

      auto& grid = patch_layers.findPatchLayer(order)->get_grid();
      auto* node = grid.get_node(row, col);
      if (node != nullptr && node->get_node_data() != nullptr) {
        auto& node_feature = node->get_node_data()->get_feature();
        omp_set_lock(&lck);
        node_feature.drc_num += 1;
        omp_unset_lock(&lck);
      }
    }
  }

  omp_destroy_lock(&lck);

  LOG_INFO << "LM memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "LM elapsed time " << stats.elapsedRunTime() << " s";
  LOG_INFO << "LM mark nodes drc end...";
}

void LmFeatureDrc::markNetsAndWires()
{
  ieda::Stats stats;

  LOG_INFO << "LM mark net & wire drc start...";

  auto& patch_layers = _layout->get_patch_layers();

  auto& net_map = _layout->get_graph().get_net_map();
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < (int) net_map.size(); ++i) {
    auto it = net_map.begin();
    std::advance(it, i);
    auto& lm_net = it->second;

    auto* net_feature = lm_net.get_feature(true);
    for (auto& wire : lm_net.get_wires()) {
      auto* wire_feature = wire.get_feature(true);

      std::set<LmNode*> node_list;
      for (auto& [node1, node2] : wire.get_paths()) {
        if (node1->get_layer_id() == node2->get_layer_id()) {
          auto order = node1->get_layer_id();
          auto& grid = patch_layers.findPatchLayer(order)->get_grid();

          int min_row = std::min(node1->get_row_id(), node2->get_row_id());
          int max_row = std::max(node1->get_row_id(), node2->get_row_id());
          int min_col = std::min(node1->get_col_id(), node2->get_col_id());
          int max_col = std::max(node1->get_col_id(), node2->get_col_id());

          for (int row = min_row; row <= max_row; ++row) {
            for (int col = min_col; col <= max_col; ++col) {
              auto* node = grid.get_node(row, col);
              if (node == nullptr || node->get_node_data() == nullptr) {
                continue;
              }

              wire_feature->drc_num += node->get_node_data()->get_feature().drc_num;
              net_feature->drc_num += node->get_node_data()->get_feature().drc_num;
            }
          }
        } else {
          wire_feature->drc_num += node1->get_node_data()->get_feature().drc_num;
          net_feature->drc_num += node1->get_node_data()->get_feature().drc_num;

          wire_feature->drc_num += node2->get_node_data()->get_feature().drc_num;
          net_feature->drc_num += node2->get_node_data()->get_feature().drc_num;
        }
      }
    }

    if (i % 1000 == 0) {
      LOG_INFO << "Read nets : " << i << " / " << (int) net_map.size();
    }
  }

  LOG_INFO << "LM memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "LM elapsed time " << stats.elapsedRunTime() << " s";
  LOG_INFO << "LM mark net & wire drc end...";
}

}  // namespace ilm