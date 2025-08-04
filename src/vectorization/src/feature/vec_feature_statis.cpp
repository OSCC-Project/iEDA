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

#include "vec_feature_statis.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "Log.hh"
#include "congestion_api.h"
#include "density_api.h"
#include "idm.h"
#include "omp.h"
#include "timing_api.hh"
#include "usage.hh"
#include "vec_grid_info.h"
#include "wirelength_api.h"

namespace ivec {
void VecFeatureStatis::build()
{
  feature_patch();
  feature_graph();
}

void VecFeatureStatis::feature_graph()
{
  ieda::Stats stats;

  LOG_INFO << "Vectorization build statis feature start...";

  auto& layout_layers = _layout->get_layout_layers();

  // get egr_layer_map, which is a map of layer name to a 2D vector of congestion value.
  auto egr_layer_map = CONGESTION_API_INST->getEGRMap();

  // get the number of egr_map's rows and cols.
  const auto& first_layer = egr_layer_map.begin()->second;
  size_t egr_rows = first_layer.size();
  size_t egr_cols = (egr_rows > 0) ? first_layer[0].size() : 0;

  // calculate the factor to convert egr_map's row and col to the layout's coordinate (x and y).
  double row_factor = static_cast<double>(gridInfoInst.ury - gridInfoInst.lly) / egr_rows;
  double col_factor = static_cast<double>(gridInfoInst.urx - gridInfoInst.llx) / egr_cols;

  CONGESTION_API_INST->evalNetInfo();
  WIRELENGTH_API_INST->evalNetFlute();

  auto& net_map = _layout->get_graph().get_net_map();
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < (int) net_map.size(); ++i) {
    auto it = net_map.begin();
    std::advance(it, i);
    auto& vec_net = it->second;

    int net_row_min = INT32_MAX;
    int net_row_max = INT32_MIN;
    int net_col_min = INT32_MAX;
    int net_col_max = INT32_MIN;

    auto* net_feature = vec_net.get_feature(true);

    std::string net_name = _layout->findNetName(it->first);
    net_feature->aspect_ratio = CONGESTION_API_INST->findAspectRatio(net_name);
    net_feature->width = CONGESTION_API_INST->findBBoxWidth(net_name);
    net_feature->height = CONGESTION_API_INST->findBBoxHeight(net_name);
    net_feature->area = CONGESTION_API_INST->findBBoxArea(net_name);
    net_feature->l_ness = CONGESTION_API_INST->findLness(net_name);
    net_feature->rsmt = WIRELENGTH_API_INST->findNetFLUTE(net_name);

    /// 初始化 layer_ratio
    int16_t min_order = INT8_MAX;  // 记录最小层
    int16_t max_order = INT8_MIN;  // 记录最大层
    int layer_order_top = layout_layers.get_layer_order_top();
    int layer_order_bottom = layout_layers.get_layer_order_bottom();
    int num_layers = layer_order_top - layer_order_bottom + 1;  // 总布线层数
    net_feature->layer_ratio = std::vector<int>(num_layers, 0);

    for (auto& wire : vec_net.get_wires()) {
      auto* wire_feature = wire.get_feature(true);

      for (auto& [node1, node2] : wire.get_paths()) {
        if (node1->get_layer_id() == node2->get_layer_id()) {
          auto order = node1->get_layer_id();
          auto* layout_layer = layout_layers.findLayoutLayer(order);
          auto& grid = layout_layer->get_grid();
          auto layer_name = layout_layer->get_layer_name();

          /// 更新最小层和最大层
          min_order = std::min(min_order, order);
          max_order = std::max(max_order, order);
          /// 更新 layer_ratio
          if (order >= layer_order_bottom && order <= layer_order_top) {
            int layer_index = order - layer_order_bottom;  // 计算层的索引
            net_feature->layer_ratio[layer_index] = 1;     // 设置为 1，表示该层有线经过
          }

          /// set feature
          wire_feature->wire_width = layout_layer->get_wire_width();

          int min_row = std::min(node1->get_row_id(), node2->get_row_id());
          int max_row = std::max(node1->get_row_id(), node2->get_row_id());
          int min_col = std::min(node1->get_col_id(), node2->get_col_id());
          int max_col = std::max(node1->get_col_id(), node2->get_col_id());

          net_row_min = std::min(net_row_min, min_row);
          net_row_max = std::max(net_row_max, max_row);
          net_col_min = std::min(net_col_min, min_col);
          net_col_max = std::max(net_col_max, max_col);

          // transform the row and col index of node to the row and col index of egr_layer_map
          int min_node_x = std::min(node1->get_x(), node2->get_x());
          int max_node_x = std::max(node1->get_x(), node2->get_x());
          int min_node_y = std::min(node1->get_y(), node2->get_y());
          int max_node_y = std::max(node1->get_y(), node2->get_y());

          int trans_min_row = static_cast<int>(min_node_y / row_factor);
          int trans_max_row = static_cast<int>(max_node_y / row_factor);
          int trans_min_col = static_cast<int>(min_node_x / col_factor);
          int trans_max_col = static_cast<int>(max_node_x / col_factor);

          /// congestion
          int sum_congestion = 0;
          int trans_grid_count = 0;

          for (int r = trans_min_row; r <= trans_max_row; ++r) {
            for (int c = trans_min_col; c <= trans_max_col; ++c) {
              sum_congestion += egr_layer_map[layer_name][r][c];
              trans_grid_count++;
            }
          }

          if (trans_grid_count > 0) {
            wire_feature->congestion += ((double) sum_congestion / trans_grid_count);
          } else {
            wire_feature->congestion = 0;
          }

          int horizontal_len = (max_col - min_col) * gridInfoInst.x_step;
          int vertical_len = (max_row - min_row) * gridInfoInst.y_step;

          wire_feature->wire_len += (horizontal_len + vertical_len);
          wire_feature->wire_density = wire_feature->wire_len * wire_feature->wire_width / static_cast<double>(net_feature->area);

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
      net_feature->fanout = vec_net.get_pin_ids().size() - 1;
    }

    /// 计算 BBox体积
    if (min_order != INT32_MAX && max_order != INT32_MIN) {
      int total_height = (max_order - min_order);
      if (total_height == 0) {
        total_height = 1;  // 如果只在一层，层高为单层高度
      }
      net_feature->volume = net_feature->area * total_height;
    } else {
      net_feature->volume = 0;  // 如果没有有效的层，体积为 0
    }

    if (i % 1000 == 0) {
      LOG_INFO << "Read nets : " << i << " / " << (int) net_map.size();
    }
  }

  LOG_INFO << "Read nets : " << (int) net_map.size() << " / " << (int) net_map.size();
  LOG_INFO << "Vectorization memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "Vectorization elapsed time " << stats.elapsedRunTime() << " s";
  LOG_INFO << "Vectorization build statis feature end...";
}

void VecFeatureStatis::feature_patch()
{
  ieda::Stats stats;

  LOG_INFO << "Vectorization feature patch start...";

  if (_patch_grid == nullptr) {
    LOG_WARNING << "Vectorization feature patch not exist.";
  }

  auto& patchs = _patch_grid->get_patchs();
  auto& patch_xy_map = _patch_grid->get_patch_xy_map();
  auto& layout_layers = _layout->get_layout_layers();

  omp_lock_t lck;
  omp_init_lock(&lck);

  // 评估器特征计算，返回的是 patch_id 和 value 的 map
  std::map<int, double> cell_power_map = TimingPower_API_INST->patchPowerMap(patch_xy_map);
  std::map<int, double> cell_timing_map = TimingPower_API_INST->patchTimingMap(patch_xy_map);
  std::map<int, double> cell_ir_map = TimingPower_API_INST->patchIRDropMap(patch_xy_map);

  std::map<int, int> pin_density_map = DENSITY_API_INST->patchPinDensity(patch_xy_map);
  LOG_INFO << "finish pin_density_map, runtime: " << stats.elapsedRunTime();

  std::map<int, double> cell_density_map = DENSITY_API_INST->patchCellDensity(patch_xy_map);
  LOG_INFO << "finish cell_density_map, runtime: " << stats.elapsedRunTime();

  std::map<int, double> net_density_map = DENSITY_API_INST->patchNetDensity(patch_xy_map);
  LOG_INFO << "finish net_density_map, runtime: " << stats.elapsedRunTime();

  std::map<int, int> macro_margin_map = DENSITY_API_INST->patchMacroMargin(patch_xy_map);
  LOG_INFO << "finish macro_margin_map, runtime: " << stats.elapsedRunTime();

  std::map<int, double> rudy_congestion_map = CONGESTION_API_INST->patchRUDYCongestion(patch_xy_map);
  LOG_INFO << "finish rudy_congestion_map, runtime: " << stats.elapsedRunTime();

  std::map<int, double> egr_congestion_map = CONGESTION_API_INST->patchEGRCongestion(patch_xy_map);
  LOG_INFO << "finish egr_congestion_map, runtime: " << stats.elapsedRunTime();

  std::map<int, std::map<std::string, double>> layer_congestion_map = CONGESTION_API_INST->patchLayerEGRCongestion(patch_xy_map);
  LOG_INFO << "finish layer_egr_congestion_map, runtime: " << stats.elapsedRunTime();

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < (int) patchs.size(); ++i) {
    auto it = patchs.begin();
    std::advance(it, i);
    auto& patch_id = it->first;
    auto& patch = it->second;

    // 根据 patch_id 获取 patch feature
    patch.pin_density = pin_density_map[patch_id];
    patch.cell_density = cell_density_map[patch_id];
    patch.net_density = net_density_map[patch_id];
    patch.macro_margin = macro_margin_map[patch_id];
    patch.RUDY_congestion = rudy_congestion_map[patch_id];
    patch.EGR_congestion = egr_congestion_map[patch_id];

    // // timing, power, ir drop map
    auto cell_timing_map_find = cell_timing_map.find(patch_id);
    if (cell_timing_map_find != cell_timing_map.end()) {
      patch.timing_map = cell_timing_map_find->second;
    }

    auto cell_power_map_find = cell_power_map.find(patch_id);
    if (cell_power_map_find != cell_power_map.end()) {
      patch.power_map = cell_power_map[patch_id];
    }

    auto cell_ir_map_find = cell_ir_map.find(patch_id);
    if (cell_ir_map_find != cell_ir_map.end()) {
      patch.ir_drop_map = cell_ir_map[patch_id];
    }

    for (auto& [layer_id, patch_layer] : patch.get_layer_map()) {
      patch_layer.wire_width = layout_layers.findLayoutLayer(layer_id)->get_wire_width();
      // 获取每一层对应patch的拥塞值
      std::string layer_name = layout_layers.findLayoutLayer(layer_id)->get_layer_name();
      patch_layer.congestion = layer_congestion_map[patch_id][layer_name];

      for (auto& [net_id, vec_net] : patch_layer.get_sub_nets()) {
        for (auto& wire : vec_net.get_wires()) {
          auto* wire_feature = wire.get_feature(true);

          for (auto& [node1, node2] : wire.get_paths()) {
            if (node1->get_layer_id() == node2->get_layer_id()) {
              auto order = node1->get_layer_id();
              auto* layout_layer = layout_layers.findLayoutLayer(order);
              auto layer_name = layout_layer->get_layer_name();

              int min_row = std::min(node1->get_row_id(), node2->get_row_id());
              int max_row = std::max(node1->get_row_id(), node2->get_row_id());
              int min_col = std::min(node1->get_col_id(), node2->get_col_id());
              int max_col = std::max(node1->get_col_id(), node2->get_col_id());
              int horizontal_len = (max_col - min_col) * gridInfoInst.x_step;
              int vertical_len = (max_row - min_row) * gridInfoInst.y_step;
              wire_feature->wire_len += (horizontal_len + vertical_len);
              patch_layer.wire_len += wire_feature->wire_len;
            }
          }
        }
      }
    }

    if (i % 1000 == 0) {
      LOG_INFO << "Feature patch : " << i << " / " << patchs.size();
    }
  }

  omp_destroy_lock(&lck);

  LOG_INFO << "Vectorization memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "Vectorization elapsed time " << stats.elapsedRunTime() << " s";
  LOG_INFO << "Vectorization feature patch end...";
}

}  // namespace ivec