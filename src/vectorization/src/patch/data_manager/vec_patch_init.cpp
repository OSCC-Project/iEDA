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

#include "vec_patch_init.h"

#include "Log.hh"
#include "omp.h"
#include "usage.hh"
#include "vec_grid_info.h"
#include "vec_net.h"

namespace ivec {

void VecPatchInit::init()
{
  init_patch_grid();
  initSubNet();
}

void VecPatchInit::init_patch_grid()
{
  auto init_patch = [&](int patch_row, int patch_col) -> VecPatch {
    VecPatch im_patch;

    /// build node id range
    if (patchInfoInst.patch_row_start == 0) {
      im_patch.rowIdMin = patch_row * patchInfoInst.patch_row_step;
      im_patch.rowIdMax = (patch_row + 1) * patchInfoInst.patch_row_step;
      im_patch.rowIdMax = im_patch.rowIdMax > gridInfoInst.node_row_num ? gridInfoInst.node_row_num : im_patch.rowIdMax;
    } else {
      if (patch_row == 0) {
        im_patch.rowIdMin = 0;
        im_patch.rowIdMax = patchInfoInst.patch_row_start;
      } else {
        im_patch.rowIdMin = patchInfoInst.patch_row_start + (patch_row - 1) * patchInfoInst.patch_row_step;
        im_patch.rowIdMax = patchInfoInst.patch_row_start + patch_row * patchInfoInst.patch_row_step;
      }
    }

    if (patchInfoInst.patch_col_start == 0) {
      im_patch.colIdMin = patch_col * patchInfoInst.patch_col_step;
      im_patch.colIdMax = (patch_col + 1) * patchInfoInst.patch_col_step;
      im_patch.colIdMax = im_patch.colIdMax > gridInfoInst.node_col_num ? gridInfoInst.node_col_num : im_patch.colIdMax;
    } else {
      if (patch_col == 0) {
        im_patch.colIdMin = 0;
        im_patch.colIdMax = patchInfoInst.patch_col_start;
      } else {
        im_patch.colIdMin = patchInfoInst.patch_col_start + (patch_col - 1) * patchInfoInst.patch_col_step;
        im_patch.colIdMax = patchInfoInst.patch_col_start + patch_col * patchInfoInst.patch_col_step;
      }
    }

    /// build layer
    auto& patch_layer_map = im_patch.get_layer_map();
    auto& layers = _layout->get_layout_layers();
    for (int layer_id = layers.get_layer_order_bottom(); layer_id <= layers.get_layer_order_top(); ++layer_id) {
      VecPatchLayer patch_layer;
      patch_layer.layer_id = layer_id;
      patch_layer.rowIdMin = im_patch.rowIdMin;
      patch_layer.rowIdMax = im_patch.rowIdMax;
      patch_layer.colIdMin = im_patch.colIdMin;
      patch_layer.colIdMax = im_patch.colIdMax;

      patch_layer_map.insert(std::make_pair(layer_id, patch_layer));
    }

    return im_patch;
  };

  ieda::Stats stats;

  LOG_INFO << "Vectorization patch grid init start...";

  auto& patchs = _patch_grid->get_patchs();
  auto& patch_xy_map = _patch_grid->get_patch_xy_map();

  for (int row = 0; row < patchInfoInst.patch_num_vertical; ++row) {
    for (int col = 0; col < patchInfoInst.patch_num_horizontal; ++col) {
      auto im_patch = init_patch(row, col);
      auto patch_id = patchInfoInst.patch_num_horizontal * row + col;

      im_patch.patch_id = patch_id;
      im_patch.patch_id_row = row;
      im_patch.patch_id_col = col;
      patchs.insert(std::make_pair(patch_id, im_patch));

      std::pair<int, int> lx_ly = gridInfoInst.get_node_coodinate(im_patch.rowIdMin, im_patch.colIdMin);
      std::pair<int, int> ux_uy = gridInfoInst.get_node_coodinate(im_patch.rowIdMax, im_patch.colIdMax);
      patch_xy_map[patch_id] = std::make_pair(lx_ly, ux_uy);

      if (patch_id % 1000 == 0) {
        LOG_INFO << "Init patch : " << patch_id;
      }
    }
  }

  LOG_INFO << "Init patch : " << patchInfoInst.patch_num_vertical * patchInfoInst.patch_num_horizontal;

  LOG_INFO << "Vectorization memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "Vectorization elapsed time " << stats.elapsedRunTime() << " s";
  LOG_INFO << "Vectorization patch grid init end...";
}

void VecPatchInit::initSubNet()
{
  ieda::Stats stats;

  LOG_INFO << "Vectorization patch init subnet start...";

  omp_lock_t lck;
  omp_init_lock(&lck);

  auto& net_map = _layout->get_graph().get_net_map();
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < (int) net_map.size(); ++i) {
    auto it = net_map.begin();
    std::advance(it, i);
    auto& net_id = it->first;
    auto& vec_net = it->second;
    /// wires
    for (auto& wire : vec_net.get_wires()) {
      /// split paths to patchs
      for (auto& [node1, node2] : wire.get_paths()) {
        if (node1->get_layer_id() == node2->get_layer_id()) {
          /// metal
          auto node_map = splitWirePath(node1, node2);
          for (auto& [patch_id, split_nodes] : node_map) {
            auto* patch = _patch_grid->findPatch(patch_id);
            auto* patch_layer = patch->findLayer(node1->get_layer_id());
            if (patch_layer->findNet(net_id) == nullptr) {
              omp_set_lock(&lck);

              patch_layer->addSubnet(net_id, wire.get_id(), split_nodes.first, split_nodes.second);

              omp_unset_lock(&lck);
            } else {
              patch_layer->addSubnet(net_id, wire.get_id(), split_nodes.first, split_nodes.second);
            }

            /// label at wire
            wire.addPatch(patch_id, node1->get_layer_id());
          }
        } else {
          /// via ??? no need to store in patch???
          int patch_id = patchInfoInst.get_patch_id(node1->get_row_id(), node1->get_col_id());
          auto* patch = _patch_grid->findPatch(patch_id);

          for (auto node_pair : {std::make_pair(node1, node1), std::make_pair(node1, node2), std::make_pair(node2, node2)}) {
            int layer_id = (node_pair.first->get_layer_id() + node_pair.second->get_layer_id()) / 2;
            auto* patch_layer = patch->findLayer(layer_id);
            auto sub_net = patch_layer->findNet(net_id);
            if (sub_net == nullptr) {
              omp_set_lock(&lck);

              patch_layer->addSubnet(net_id, wire.get_id(), node_pair.first, node_pair.second);

              omp_unset_lock(&lck);
            } else {
              patch_layer->addSubnet(sub_net, wire.get_id(), node_pair.first, node_pair.second);
            }

            /// label at wire
            wire.addPatch(patch_id, layer_id);
          }
        }
      }
    }

    omp_set_lock(&lck);

    omp_unset_lock(&lck);

    if (i % 1000 == 0) {
      LOG_INFO << "Init net : " << i << " / " << net_map.size();
    }
  }
  LOG_INFO << "Init net : " << net_map.size() << " / " << net_map.size();

  omp_destroy_lock(&lck);

  LOG_INFO << "Vectorization memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "Vectorization elapsed time " << stats.elapsedRunTime() << " s";
  LOG_INFO << "Vectorization patch init subnet end...";
}

std::map<int, std::pair<VecNode*, VecNode*>> VecPatchInit::splitWirePath(VecNode* node1, VecNode* node2)
{
  std::map<int, std::pair<VecNode*, VecNode*>> node_map;
  if (node1->get_layer_id() == node2->get_layer_id()) {
    auto& node_grid = _layout->get_layout_layers().findLayoutLayer(node1->get_layer_id())->get_grid();

    int row_min = std::min(node1->get_row_id(), node2->get_row_id());
    int row_max = std::max(node1->get_row_id(), node2->get_row_id());
    int col_min = std::min(node1->get_col_id(), node2->get_col_id());
    int col_max = std::max(node1->get_col_id(), node2->get_col_id());

    if (row_min == row_max) {
      /// horizontal
      int patch_row_id = patchInfoInst.get_patch_row_id(row_min);
      int patch_id_min = patchInfoInst.get_patch_col_id(col_min);
      int patch_id_max = patchInfoInst.get_patch_col_id(col_max);
      for (int patch_col_id = patch_id_min; patch_col_id <= patch_id_max; ++patch_col_id) {
        auto [split_node_id_min, split_node_id_max] = patchInfoInst.get_node_range(patch_col_id, true);

        if (patch_col_id == patch_id_min) {
          split_node_id_min = col_min;
        }

        if (patch_col_id == patch_id_max) {
          split_node_id_max = col_max;
        }

        int patch_id = patchInfoInst.patch_num_horizontal * patch_row_id + patch_col_id;

        auto* split_node_1 = node_grid.get_node(row_min, split_node_id_min);
        auto* split_node_2 = node_grid.get_node(row_min, split_node_id_max);
        if (split_node_1 == nullptr || split_node_2 == nullptr) {
          LOG_ERROR << "Node error after split wire";
        }
        node_map.emplace(patch_id, std::make_pair(split_node_1, split_node_2));
      }
    } else {
      /// vertical
      int patch_id_min = patchInfoInst.get_patch_row_id(row_min);
      int patch_id_max = patchInfoInst.get_patch_row_id(row_max);
      int patch_col_id = patchInfoInst.get_patch_col_id(col_min);
      for (int patch_row_id = patch_id_min; patch_row_id <= patch_id_max; ++patch_row_id) {
        auto [split_node_id_min, split_node_id_max] = patchInfoInst.get_node_range(patch_row_id, false);

        if (patch_row_id == patch_id_min) {
          split_node_id_min = row_min;
        }

        if (patch_row_id == patch_id_max) {
          split_node_id_max = row_max;
        }

        int patch_id = patchInfoInst.patch_num_horizontal * patch_row_id + patch_col_id;

        auto* split_node_1 = node_grid.get_node(split_node_id_min, col_min);
        auto* split_node_2 = node_grid.get_node(split_node_id_max, col_min);
        if (split_node_1 == nullptr || split_node_2 == nullptr) {
          LOG_ERROR << "Node error after split wire";
        }
        node_map.emplace(patch_id, std::make_pair(split_node_1, split_node_2));
      }
    }
  } else {
    /// via
  }

  return node_map;
}

void VecPatchInit::initLayoutPDN()
{
  auto& layout_layers = _layout->get_layout_layers();

  for (auto& [patch_id, patch] : _patch_grid->get_patchs()) {
    for (auto& [layer_id, patch_layer] : patch.get_layer_map()) {
      auto* layout_layer = layout_layers.findLayoutLayer(layer_id);
      if (nullptr == layout_layer) {
        LOG_WARNING << "Can not get layer order : " << layer_id;
        return;
      }
      auto& grid = layout_layer->get_grid();

      for (int row = patch_layer.rowIdMin; row <= patch_layer.rowIdMax; ++row) {
        for (int col = patch_layer.colIdMin; col <= patch_layer.colIdMax; ++col) {
        }
      }
    }
  }
}

void VecPatchInit::initLayoutInstance()
{
}

void VecPatchInit::initLayoutIO()
{
}

void VecPatchInit::initLayoutNets()
{
}

}  // namespace ivec