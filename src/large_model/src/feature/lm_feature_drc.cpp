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
  int drc_id = 0;
  for (auto& [rule, drc_spot_list] : detail_drc_map) {
    LOG_INFO << "LM mark nodes drc : " << rule << ", size :" << drc_spot_list.size();
    origin_drc_num += drc_spot_list.size();
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int) drc_spot_list.size(); ++i) {
      // for (auto drc_spot : drc_spot_list) {
      /// set to node if node data exist
      auto& drc_spot = drc_spot_list[i];
      auto order = _layout->findLayerId(drc_spot->get_layer()->get_name());
      if (order < 0) {
        continue;
      }

      auto& grid = patch_layers.findPatchLayer(order)->get_grid();

      /// get row & col
      auto* spot_rect = static_cast<idrc::DrcViolationRect*>(drc_spot);
      auto [row_1, row_2, co_1, col_2]
          = gridInfoInst.get_node_id_range(spot_rect->get_llx(), spot_rect->get_urx(), spot_rect->get_lly(), spot_rect->get_ury());
      for (int row = row_1; row <= row_2; ++row) {
        for (int col = co_1; col <= col_2; ++col) {
          auto* node = grid.get_node(row, col);
          if (node != nullptr && node->get_node_data() != nullptr) {
            auto& node_feature = node->get_node_data()->get_feature();
            omp_set_lock(&lck);
            node_feature.drc_ids.push_back(drc_id + i);
            omp_unset_lock(&lck);
          }
        }
      }
    }

    drc_id += drc_spot_list.size();
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
      std::set<int> drc_ids;

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

              for (auto drc_id : node->get_node_data()->get_feature().drc_ids) {
                drc_ids.insert(drc_id);
              }
            }
          }

          wire_feature->drc_num += drc_ids.size();
          net_feature->drc_num += drc_ids.size();
          mark_drc_num += drc_ids.size();
        } else {
          wire_feature->drc_num += node1->get_node_data()->get_feature().drc_ids.size();
          net_feature->drc_num += node1->get_node_data()->get_feature().drc_ids.size();
          mark_drc_num += node1->get_node_data()->get_feature().drc_ids.size();

          wire_feature->drc_num += node2->get_node_data()->get_feature().drc_ids.size();
          net_feature->drc_num += node2->get_node_data()->get_feature().drc_ids.size();
          mark_drc_num += node2->get_node_data()->get_feature().drc_ids.size();
        }
      }
    }

    if (i % 1000 == 0) {
      LOG_INFO << "Read nets : " << i << " / " << (int) net_map.size();
    }
  }

  if (mark_drc_num != origin_drc_num) {
    LOG_WARNING << "[origin_drc_num] vs [mark_drc_num] : " << origin_drc_num << " vs " << mark_drc_num;
  }

  LOG_INFO << "Read nets : " << (int) net_map.size() << " / " << (int) net_map.size();
  LOG_INFO << "LM memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "LM elapsed time " << stats.elapsedRunTime() << " s";
  LOG_INFO << "LM mark net & wire drc end...";
}

}  // namespace ilm