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

  feature_graph();
  feature_patch();
}

void LmFeatureDrc::feature_graph()
{
  markWires();
  markNets();
}

void LmFeatureDrc::feature_patch()
{
}

void LmFeatureDrc::markNodes()
{
  ieda::Stats stats;

  LOG_INFO << "LM mark nodes drc start...";

  omp_lock_t lck;
  omp_init_lock(&lck);

  auto& layout_layers = _layout->get_layout_layers();

  auto detail_drc_map = iplf::drcInst->getDetailCheckResult(_drc_path);
  int drc_id = 0;
  int drc_without_net = 0;

  // 新增：用于记录 drc_id 和 drc_type（rule）的映射
  std::map<int, std::string> drc_id_to_type;

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

      auto& grid = layout_layers.findLayoutLayer(order)->get_grid();

      /// get row & col
      auto* spot_rect = static_cast<idrc::DrcViolationRect*>(drc_spot);
      bool b_find_net = false;
      auto [row_1, row_2, co_1, col_2] = gridInfoInst.get_node_id_range(spot_rect->get_llx() - 1, spot_rect->get_urx() + 1,
                                                                        spot_rect->get_lly() - 1, spot_rect->get_ury() + 1);
      for (int row = row_1; row <= row_2; ++row) {
        for (int col = co_1; col <= col_2; ++col) {
          auto* node = grid.get_node(row, col);
          if (node != nullptr && node->get_node_data() != nullptr) {
            auto& node_feature = node->get_node_data()->get_feature();
            omp_set_lock(&lck);
            node_feature.drc_ids.insert(drc_id + i);
            drc_id_to_type[drc_id + i] = rule; // 将 drc_id 和 rule 关联
            omp_unset_lock(&lck);

            if (node->get_node_data()->get_net_id() >= 0) {
              b_find_net = true;
            }
          }
        }
      }

      if (false == b_find_net) {
        omp_set_lock(&lck);
        drc_without_net++;
        omp_unset_lock(&lck);
      }
    }

    drc_id += drc_spot_list.size();
  }

  LOG_INFO << "drc with no net id : " << drc_without_net;

  omp_destroy_lock(&lck);

  LOG_INFO << "LM memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "LM elapsed time " << stats.elapsedRunTime() << " s";
  LOG_INFO << "LM mark nodes drc end...";

  _drc_id_to_type = std::move(drc_id_to_type);

}

void LmFeatureDrc::markWires()
{
  ieda::Stats stats;

  omp_lock_t lck;
  omp_init_lock(&lck);

  LOG_INFO << "LM mark wire drc start...";

  auto& layout_layers = _layout->get_layout_layers();

  auto& net_map = _layout->get_graph().get_net_map();
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < (int) net_map.size(); ++i) {
    auto it = net_map.begin();
    std::advance(it, i);
    auto& lm_net = it->second;

    // auto* net_feature = lm_net.get_feature(true);
    for (auto& wire : lm_net.get_wires()) {
      std::set<int> drc_ids;

      auto* wire_feature = wire.get_feature(true);

      std::set<LmNode*> node_list;
      for (auto& [node1, node2] : wire.get_paths()) {
        if (node1->get_layer_id() == node2->get_layer_id()) {
          auto order = node1->get_layer_id();
          auto& grid = layout_layers.findLayoutLayer(order)->get_grid();

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

              drc_ids.insert(node->get_node_data()->get_feature().drc_ids.begin(), node->get_node_data()->get_feature().drc_ids.end());
            }
          }

          wire_feature->drc_num += drc_ids.size();
          //   net_feature->drc_num += drc_ids.size();

          omp_set_lock(&lck);
          mark_drc_num += drc_ids.size();
          omp_unset_lock(&lck);
        } else {
          // 如果路径跨层，分别处理 node1 和 node2 的 drc_ids
          drc_ids.insert(node1->get_node_data()->get_feature().drc_ids.begin(), node1->get_node_data()->get_feature().drc_ids.end());
          drc_ids.insert(node2->get_node_data()->get_feature().drc_ids.begin(), node2->get_node_data()->get_feature().drc_ids.end());

          wire_feature->drc_num += node1->get_node_data()->get_feature().drc_ids.size();
          //   net_feature->drc_num += node1->get_node_data()->get_feature().drc_ids.size();

          wire_feature->drc_num += node2->get_node_data()->get_feature().drc_ids.size();
          //   net_feature->drc_num += node2->get_node_data()->get_feature().drc_ids.size();

          omp_set_lock(&lck);
          mark_drc_num += node1->get_node_data()->get_feature().drc_ids.size();
          mark_drc_num += node2->get_node_data()->get_feature().drc_ids.size();
          omp_unset_lock(&lck);
        }
      }
      // 填充 wire_feature->drc_type
      for (auto drc_id : drc_ids) {
        wire_feature->drc_type.push_back(_drc_id_to_type[drc_id]); // 根据 drc_id 获取对应的 drc_type
      }
    }

    if (i % 1000 == 0) {
      LOG_INFO << "Read nets : " << i << " / " << (int) net_map.size();
    }
  }

  omp_destroy_lock(&lck);

  if (mark_drc_num != origin_drc_num) {
    LOG_WARNING << "[origin_drc_num] vs [mark_drc_num] : " << origin_drc_num << " vs " << mark_drc_num;
  }

  LOG_INFO << "Read nets : " << (int) net_map.size() << " / " << (int) net_map.size();
  LOG_INFO << "LM memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "LM elapsed time " << stats.elapsedRunTime() << " s";
  LOG_INFO << "LM mark wire drc end...";
}

void LmFeatureDrc::markNets()
{
  mark_drc_num = 0;

  ieda::Stats stats;

  omp_lock_t lck;
  omp_init_lock(&lck);

  LOG_INFO << "LM mark net drc start...";

  std::map<int, std::set<int>> net_drc_map;

  auto& layout_layers = _layout->get_layout_layers();
  auto& layer_map = layout_layers.get_layout_layer_map();
  for (int layer_id = 0; layer_id < (int) layer_map.size(); ++layer_id) {
    auto& grid = layout_layers.findLayoutLayer(layer_id)->get_grid();
    // 这里改用get_all_nodes()替代原来的get_node_matrix()
    auto nodes = grid.get_all_nodes();
    
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)nodes.size(); i++) {
      LmNode* node = nodes[i];
      if (node != nullptr && node->get_node_data() != nullptr) {
        omp_set_lock(&lck);
        net_drc_map[node->get_node_data()->get_net_id()].insert(
            node->get_node_data()->get_feature().drc_ids.begin(),
            node->get_node_data()->get_feature().drc_ids.end());
        omp_unset_lock(&lck);
      }
    }
  }
  
  int test_num = 0;
  for (auto& [net_id, drc_ids] : net_drc_map) {
    test_num += drc_ids.size();
  }

  LOG_INFO << "test_num size : " << test_num;

  auto& net_map = _layout->get_graph().get_net_map();
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < (int) net_map.size(); ++i) {
    auto it = net_map.begin();
    std::advance(it, i);
    auto& lm_net = it->second;

    auto* net_feature = lm_net.get_feature(true);
    net_feature->drc_num += net_drc_map[i].size();

    // 填充 drc_type
    for (auto drc_id : net_drc_map[i]) {
      omp_set_lock(&lck);
      net_feature->drc_type.push_back(_drc_id_to_type[drc_id]); // 根据 drc_id 获取对应的 drc_type
      omp_unset_lock(&lck);
    }

    omp_set_lock(&lck);
    mark_drc_num += net_drc_map[i].size();
    omp_unset_lock(&lck);

    if (i % 1000 == 0) {
      LOG_INFO << "Read nets : " << i << " / " << (int) net_map.size();
    }
  }

  omp_destroy_lock(&lck);

  if (mark_drc_num != origin_drc_num) {
    LOG_WARNING << "[origin_drc_num] vs [mark_drc_num] : " << origin_drc_num << " vs " << mark_drc_num;
  }

  LOG_INFO << "Read nets : " << (int) net_map.size() << " / " << (int) net_map.size();
  LOG_INFO << "LM memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "LM elapsed time " << stats.elapsedRunTime() << " s";
  LOG_INFO << "LM mark net drc end...";
}

}  // namespace ilm