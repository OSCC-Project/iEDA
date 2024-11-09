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

#include "lm_layout_init.h"

#include "IdbGeometry.h"
#include "IdbLayer.h"
#include "IdbLayerShape.h"
#include "IdbNet.h"
#include "IdbRegularWire.h"
#include "IdbSpecialNet.h"
#include "IdbSpecialWire.h"
#include "Log.hh"
#include "idm.h"
#include "omp.h"

namespace ilm {

void LmLayoutInit::init()
{
  initViaIds();
  initCellMasters();
  initLayers();
  initDie();
  initCore();
  initTracks();
  initPDN();
  initInstances();
  initNets();
}

void LmLayoutInit::initViaIds()
{
  auto* idb_layout = dmInst->get_idb_layout();
  auto* idb_design = dmInst->get_idb_design();
  auto* idb_lef_vias = idb_layout->get_via_list();
  auto* idb_def_vias = idb_design->get_via_list();

  auto& via_map = _layout->get_via_id_map();

  int index = 0;
  for (auto* via : idb_lef_vias->get_via_list()) {
    via_map.insert(std::make_pair(via->get_name(), index++));
  }

  for (auto* via : idb_def_vias->get_via_list()) {
    via_map.insert(std::make_pair(via->get_name(), index++));
  }

  LOG_INFO << "Via number : " << index;
}

void LmLayoutInit::initCellMasters()
{
}

void LmLayoutInit::initDie()
{
  auto* idb_layout = dmInst->get_idb_layout();
  auto* idb_die = idb_layout->get_die();

  auto& patch_layers = _layout->get_patch_layers();
  for (auto& [order, patch_layer] : patch_layers.get_patch_layer_map()) {
    patch_layer.set_llx(idb_die->get_llx());
    patch_layer.set_lly(idb_die->get_lly());
    patch_layer.set_urx(idb_die->get_urx());
    patch_layer.set_ury(idb_die->get_ury());

    auto& grid = patch_layer.get_grid();
    grid.get_info().llx = idb_die->get_llx();
    grid.get_info().lly = idb_die->get_lly();
    grid.get_info().urx = idb_die->get_urx();
    grid.get_info().ury = idb_die->get_ury();
  }
}

void LmLayoutInit::initCore()
{
}

void LmLayoutInit::initLayers()
{
  auto* idb_layout = dmInst->get_idb_layout();
  auto* idb_layers = idb_layout->get_layers();
  auto idb_layer_1st = dmInst->get_config().get_routing_layer_1st();

  auto& layer_id_map = _layout->get_layer_id_map();
  auto& patch_layers = _layout->get_patch_layers();
  auto& patch_layer_map = patch_layers.get_patch_layer_map();

  bool b_record = false;
  int index = 0;
  for (auto* idb_layer : idb_layers->get_layers()) {
    if (idb_layer->get_name() == idb_layer_1st) {
      b_record = true;
    }

    if (true == b_record) {
      layer_id_map.insert(std::make_pair(idb_layer->get_name(), index));

      LmPatchLayer patch_layer;
      patch_layer.set_layer_name(idb_layer->get_name());
      patch_layer.set_layer_order(index);

      auto& grid = patch_layer.get_grid();
      grid.get_info().layer_order = index;

      /// init routing tracks
      //   if (idb_layer->is_routing()) {
      //     auto* idb_track_grid_prefer = (dynamic_cast<idb::IdbLayerRouting*>(idb_layer))->get_prefer_track_grid();
      //     auto* idb_track_grid_nonprefer = (dynamic_cast<idb::IdbLayerRouting*>(idb_layer))->get_nonprefer_track_grid();

      //     /// init current routing layer
      //     initTrackGrid(idb_track_grid_prefer, grid);
      //     initTrackGrid(idb_track_grid_nonprefer, grid);

      //     /// reset lower routing layer by current prefer tracks
      //     if (index >= 2) {
      //       /// lower routing layer
      //       auto* lower_routing_layer = patch_layers.findPatchLayer(index - 2);
      //       initTrackGrid(idb_track_grid_prefer, lower_routing_layer->get_grid());

      //       /// lower cut layer
      //       auto* lower_cut_layer = patch_layers.findPatchLayer(index - 1);
      //       initTrackGrid(idb_track_grid_prefer, lower_cut_layer->get_grid());
      //     }
      //   }

      //   if (idb_layer->is_cut()) {
      //     /// copy lower routing layer info to this cut layer
      //     auto* lower_routing_layer = patch_layers.findPatchLayer(index - 1);
      //     auto& info = patch_layer.get_grid().get_info();
      //     info.x_start = lower_routing_layer->get_grid().get_info().x_start;
      //     info.x_step = lower_routing_layer->get_grid().get_info().x_step;
      //     info.y_start = lower_routing_layer->get_grid().get_info().y_start;
      //     info.y_step = lower_routing_layer->get_grid().get_info().y_step;
      //   }

      patch_layer_map.insert(std::make_pair(index, patch_layer));

      index++;

      if (false == idb_layer->is_cut() && false == idb_layer->is_routing()) {
        break;
      }
    }
  }

  patch_layers.set_layer_order_bottom(0);
  patch_layers.set_layer_order_top(index - 1);

  LOG_INFO << "Layer number : " << index;
}

void LmLayoutInit::initTrackGrid(idb::IdbTrackGrid* idb_track_grid, LmLayerGrid& lm_grid)
{
  auto start = idb_track_grid->get_track()->get_start();
  auto pitch = idb_track_grid->get_track()->get_pitch();

  if (idb_track_grid->get_track()->is_track_vertical()) {
    lm_grid.get_info().x_start = start;
    lm_grid.get_info().x_step = pitch / 2;
  } else {
    lm_grid.get_info().y_start = start;
    lm_grid.get_info().y_step = pitch / 2;
  }
}

void LmLayoutInit::initTracks(std::string layername)
{
  /// find base track
  auto* idb_layout = dmInst->get_idb_layout();
  auto* idb_layers = idb_layout->get_layers();
  auto* idb_layer = idb_layers->find_layer(layername);
  auto* idb_track_grid_prefer = (dynamic_cast<idb::IdbLayerRouting*>(idb_layer))->get_prefer_track_grid();
  auto* idb_track_grid_nonprefer = (dynamic_cast<idb::IdbLayerRouting*>(idb_layer))->get_nonprefer_track_grid();

  auto& patch_layers = _layout->get_patch_layers();
  auto& patch_layer_map = patch_layers.get_patch_layer_map();

  for (auto& [order, patch_layer] : patch_layer_map) {
    auto& grid = patch_layer.get_grid();
    initTrackGrid(idb_track_grid_prefer, grid);
    initTrackGrid(idb_track_grid_nonprefer, grid);
  }

  buildPatchGrid();
}

void LmLayoutInit::buildPatchGrid()
{
  auto& patch_layers = _layout->get_patch_layers();
  auto& patch_layer_map = patch_layers.get_patch_layer_map();
  for (auto& [order, patch_layer] : patch_layer_map) {
    auto& grid = patch_layer.get_grid();
    grid.buildNodeMatrix();
  }
}

void LmLayoutInit::initPDN()
{
  ieda::Stats stats;

  LOG_INFO << "LM init PDN start...";

  auto& path_layers = _layout->get_patch_layers();

  auto* idb_design = dmInst->get_idb_design();
  auto* idb_pdn = idb_design->get_special_net_list();

  omp_lock_t lck;
  omp_init_lock(&lck);

  int segment_num = 0;
  for (auto* idb_net : idb_pdn->get_net_list()) {
    auto* idb_wires = idb_net->get_wire_list();
    for (auto* idb_wire : idb_wires->get_wire_list()) {
#pragma omp parallel for schedule(dynamic)
      for (int i = 0; i < (int) idb_wire->get_segment_list().size(); ++i) {
        auto* idb_segment = idb_wire->get_segment_list()[i];
        /// wire
        if (idb_segment->get_point_num() >= 2) {
          auto* routing_layer = dynamic_cast<IdbLayerRouting*>(idb_segment->get_layer());
          auto routing_width = idb_segment->get_route_width() == 0 ? routing_layer->get_width() : idb_segment->get_route_width();

          /// get bounding box
          int32_t ll_x = 0;
          int32_t ll_y = 0;
          int32_t ur_x = 0;
          int32_t ur_y = 0;

          auto* point_1 = idb_segment->get_point_start();
          auto* point_2 = idb_segment->get_point_second();
          if (point_1->get_y() == point_2->get_y()) {
            // horizontal
            ll_x = std::min(point_1->get_x(), point_2->get_x());
            ll_y = std::min(point_1->get_y(), point_2->get_y()) - routing_width / 2;
            ur_x = std::max(point_1->get_x(), point_2->get_x());
            ur_y = ll_y + routing_width;
          } else if (point_1->get_x() == point_2->get_x()) {
            // vertical
            ll_x = std::min(point_1->get_x(), point_2->get_x()) - routing_width / 2;
            ll_y = std::min(point_1->get_y(), point_2->get_y());
            ur_x = ll_x + routing_width;
            ur_y = std::max(point_1->get_y(), point_2->get_y());
          }

          /// build grid
          auto order = _layout->findLayerId(routing_layer->get_name());
          auto* patch_layer = path_layers.findPatchLayer(order);
          if (nullptr == patch_layer) {
            LOG_ERROR << "Can not get layer order : " << routing_layer->get_name();
            continue;
          }
          auto& grid = patch_layer->get_grid();
          auto [row_1, row_2, co_1, col_2] = grid.getNodeIdRange(ll_x, ur_x, ll_y, ur_y);
          for (int row = row_1; row <= row_2; ++row) {
            for (int col = co_1; col <= col_2; ++col) {
              auto& grid_node = grid.get_node(row, col);
              grid_node.set_node_type(LmNodeTYpe::lm_pdn);
            }
          }
        }

        if (i % 1000 == 0) {
          omp_set_lock(&lck);

          segment_num = segment_num + 1000;

          LOG_INFO << "Read segment : " << segment_num;

          omp_unset_lock(&lck);
        }
      }
    }
  }

  omp_destroy_lock(&lck);

  LOG_INFO << "LM memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "LM elapsed time " << stats.elapsedRunTime() << " s";

  LOG_INFO << "LM init PDN end...";
}

void LmLayoutInit::initInstances()
{
}

void LmLayoutInit::initIOPins()
{
}

void LmLayoutInit::initNets()
{
}

}  // namespace ilm