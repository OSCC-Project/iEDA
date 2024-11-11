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

  initLayers();
  initDie();

  initTracks();
  initPDN();
  initNets();
  initInstances();
  initIOPins();
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
  ieda::Stats stats;

  LOG_INFO << "LM init tracks start...";

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

  LOG_INFO << "LM memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "LM elapsed time " << stats.elapsedRunTime() << " s";

  LOG_INFO << "LM init tracks end...";
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

void LmLayoutInit::transVia(idb::IdbVia* idb_via, int net_id, LmNodeTYpe type)
{
  auto& patch_layers = _layout->get_patch_layers();

  auto cut_layer_shape = idb_via->get_cut_layer_shape();

  auto order = _layout->findLayerId(cut_layer_shape.get_layer()->get_name());
  auto* patch_layer = patch_layers.findPatchLayer(order);
  if (nullptr == patch_layer) {
    LOG_WARNING << "Can not get layer order : " << cut_layer_shape.get_layer()->get_name();
    return;
  }
  auto& grid = patch_layer->get_grid();

  for (auto* cut_rect : cut_layer_shape.get_rect_list()) {
    auto [row_1, row_2, co_1, col_2]
        = grid.getNodeIdRange(cut_rect->get_low_x(), cut_rect->get_high_x(), cut_rect->get_low_y(), cut_rect->get_high_y());
    for (int row = row_1; row <= row_2; ++row) {
      for (int col = co_1; col <= col_2; ++col) {
        /// set node data
        auto& node_data = grid.get_node(row, col).get_node_data();
        node_data.set_type(type);
        node_data.set_status(LmNodeStatus::lm_connected);
        node_data.set_net_id(net_id);
      }
    }
  }
}

void LmLayoutInit::initPDN()
{
  ieda::Stats stats;

  LOG_INFO << "LM init PDN start...";
  auto& pdn_id_map = _layout->get_pdn_id_map();
  auto& patch_layers = _layout->get_patch_layers();

  auto* idb_design = dmInst->get_idb_design();
  auto* idb_pdn = idb_design->get_special_net_list();

  omp_lock_t lck;
  omp_init_lock(&lck);

  int segment_total = 0;
  for (auto* idb_net : idb_pdn->get_net_list()) {
    auto* idb_wires = idb_net->get_wire_list();
    for (auto* idb_wire : idb_wires->get_wire_list()) {
      segment_total += idb_wire->get_num();
    }
  }

  int segment_num = 0;
  int net_id = 0;
  for (auto* idb_net : idb_pdn->get_net_list()) {
    /// init pdn id map
    pdn_id_map.insert(std::make_pair(idb_net->get_net_name(), net_id));

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
          auto* patch_layer = patch_layers.findPatchLayer(order);
          if (nullptr == patch_layer) {
            LOG_WARNING << "Can not get layer order : " << routing_layer->get_name();
            continue;
          }
          auto& grid = patch_layer->get_grid();
          auto [row_1, row_2, co_1, col_2] = grid.getNodeIdRange(ll_x, ur_x, ll_y, ur_y);
          for (int row = row_1; row <= row_2; ++row) {
            for (int col = co_1; col <= col_2; ++col) {
              /// set node data
              auto& node_data = grid.get_node(row, col).get_node_data();
              node_data.set_type(LmNodeTYpe::lm_pdn);
              node_data.set_status(LmNodeStatus::lm_connecting);
            }
          }
        }

        if (idb_segment->is_via()) {
          auto* idb_via = idb_segment->get_via();
          transVia(idb_via, net_id, LmNodeTYpe::lm_pdn);
        }

        if (i % 1000 == 0) {
          omp_set_lock(&lck);

          segment_num = segment_num + 1000;
          segment_num = segment_num > segment_total ? segment_total : segment_num;

          LOG_INFO << "Read segment : " << segment_num << " / " << segment_total;

          omp_unset_lock(&lck);
        }
      }
    }

    net_id++;
  }

  omp_destroy_lock(&lck);

  LOG_INFO << "LM memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "LM elapsed time " << stats.elapsedRunTime() << " s";

  LOG_INFO << "LM init PDN end...";
}

void LmLayoutInit::initInstances()
{
  ieda::Stats stats;

  LOG_INFO << "LM init instances start...";

  auto& patch_layers = _layout->get_patch_layers();

  auto* idb_design = dmInst->get_idb_design();
  auto* idb_insts = idb_design->get_instance_list();

  omp_lock_t lck;
  omp_init_lock(&lck);

  int inst_total = (int) idb_insts->get_instance_list().size();

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < inst_total; ++i) {
    auto* idb_inst = idb_insts->get_instance_list()[i];
    auto* idb_inst_pins = idb_inst->get_pin_list();
    for (auto* idb_pin : idb_inst_pins->get_pin_list()) {
      if (false == idb_pin->is_net_pin()) {
        transPin(idb_pin, -1);
      }
    }

    for (auto* layer_shape : idb_inst->get_obs_box_list()) {
      auto* layer = layer_shape->get_layer();
      auto order = _layout->findLayerId(layer->get_name());
      auto* patch_layer = patch_layers.findPatchLayer(order);
      if (nullptr == patch_layer) {
        LOG_WARNING << "Can not get layer order : " << layer->get_name();
        continue;
      }
    }
  }
}

void LmLayoutInit::initIOPins()
{
  auto* idb_design = dmInst->get_idb_design();
  auto* idb_iopins = idb_design->get_io_pin_list();

  auto& patch_layers = _layout->get_patch_layers();

  for (auto* io_pin : idb_iopins->get_pin_list()) {
    if (io_pin->is_special_net_pin()) {
      auto net_id = _layout->findPdnId(io_pin->get_net_name());
      transPin(io_pin, net_id);
    }
  }
}

void LmLayoutInit::transPin(idb::IdbPin* idb_pin, int net_id)
{
  auto& patch_layers = _layout->get_patch_layers();
  for (auto* layer_shape : idb_pin->get_port_box_list()) {
    auto order = _layout->findLayerId(layer_shape->get_layer()->get_name());
    auto* patch_layer = patch_layers.findPatchLayer(order);
    if (nullptr == patch_layer) {
      LOG_WARNING << "Can not get layer order : " << layer_shape->get_layer()->get_name();
      continue;
    }

    for (IdbRect* rect : layer_shape->get_rect_list()) {
      /// build grid
      auto& grid = patch_layer->get_grid();
      auto [row_id, col_id] = grid.findNodeID(rect->get_middle_point_x(), rect->get_middle_point_y());
      auto& node_data = grid.get_node(row_id, col_id).get_node_data();
      node_data.set_type(LmNodeTYpe::lm_pin);
      if (net_id >= 0) {
        auto status = layer_shape->is_via() ? LmNodeStatus::lm_connected : LmNodeStatus::lm_connecting;
        node_data.set_status(status);
      } else {
        node_data.set_status(LmNodeStatus::lm_fix);
      }
      node_data.set_net_id(net_id);
    }
  }

  /// init via in pins
  /// tbd
}

void LmLayoutInit::initNets()
{
  ieda::Stats stats;

  LOG_INFO << "LM init instances start...";
  auto& net_id_map = _layout->get_net_id_map();
  auto& patch_layers = _layout->get_patch_layers();

  auto* idb_design = dmInst->get_idb_design();
  auto* idb_nets = idb_design->get_net_list();

  for (int net_id = 0; net_id < (int) idb_nets->get_net_list().size(); ++net_id) {
    /// init net id map
    auto* idb_net = idb_nets->get_net_list()[net_id];
    net_id_map.insert(std::make_pair(idb_net->get_net_name(), net_id));
  }

#pragma omp parallel for schedule(dynamic)
  for (int net_id = 0; net_id < (int) idb_nets->get_net_list().size(); ++net_id) {
    auto* idb_net = idb_nets->get_net_list()[net_id];
    /// init pins
    /// instance pin
    for (auto* idb_inst_pin : idb_net->get_instance_pin_list()->get_pin_list()) {
      transPin(idb_inst_pin, net_id);
    }

    for (auto* io_pin : idb_net->get_io_pins()->get_pin_list()) {
      transPin(io_pin, net_id);
    }

    /// init wires
    auto* idb_wires = idb_net->get_wire_list();
    for (auto* idb_wire : idb_wires->get_wire_list()) {
      for (auto* idb_segment : idb_wire->get_segment_list()) {
        /// wire
        if (idb_segment->is_via()) {
          for (auto* idb_via : idb_segment->get_via_list()) {
            transVia(idb_via, net_id, LmNodeTYpe::lm_net);
          }
        } else if (idb_segment->is_rect()) {
          /// wire patch
          auto* coordinate = idb_segment->get_point_start();
          auto* rect_delta = idb_segment->get_delta_rect();
          auto* routing_layer = dynamic_cast<IdbLayerRouting*>(idb_segment->get_layer());

        } else {
          /// nothing to do
        }

        /// build wire
        if (idb_segment->get_point_number() >= 2) {
          auto* routing_layer = dynamic_cast<IdbLayerRouting*>(idb_segment->get_layer());
          auto routing_width = routing_layer->get_width();

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
          auto* patch_layer = patch_layers.findPatchLayer(order);
          if (nullptr == patch_layer) {
            LOG_WARNING << "Can not get layer order : " << routing_layer->get_name();
            continue;
          }
          auto& grid = patch_layer->get_grid();
          auto [row_1, row_2, col_1, col_2] = grid.getNodeIdRange(ll_x, ur_x, ll_y, ur_y);
          /// net wire must only occupy one grid size
          if (row_1 != row_2 && col_1 != col_2) {
            LOG_WARNING << "Net width maybe is out of range : " << idb_net->get_net_name();
          }

          bool b_horizontal = (col_2 - col_1) > (row_2 - row_1) ? true : false;
          for (int row = row_1; row <= row_2; ++row) {
            for (int col = col_1; col <= col_2; ++col) {
              /// set node data
              auto& node_data = grid.get_node(row, col).get_node_data();
              node_data.set_type(LmNodeTYpe::lm_net);
              node_data.set_net_id(net_id);
              if (b_horizontal) {
                LmNodeStatus status = col == col_1 || col == (col_2 - 1) ? LmNodeStatus::lm_connected : LmNodeStatus::lm_connecting;
                node_data.set_status(status);
              } else {
                LmNodeStatus status = row == row_1 || row == (row_2 - 1) ? LmNodeStatus::lm_connected : LmNodeStatus::lm_connecting;
                node_data.set_status(status);
              }
            }
          }
        }
      }
    }
    if (net_id % 1000 == 0) {
      LOG_INFO << "Read nets : " << net_id << " / " << (int) idb_nets->get_net_list().size();
    }
  }

  LOG_INFO << "Read nets : " << idb_nets->get_net_list().size() << " / " << (int) idb_nets->get_net_list().size();
  LOG_INFO << "LM memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "LM elapsed time " << stats.elapsedRunTime() << " s";

  LOG_INFO << "LM init nets end...";
}

}  // namespace ilm