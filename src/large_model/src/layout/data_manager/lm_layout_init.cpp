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
#include "usage.hh"

namespace ilm {

void LmLayoutInit::init()
{
  initViaIds();

  initLayers();
  initDie();

  initTracks();
  initPDN();
  initNets();
  //   initInstances();
  //   initIOPins();

  buildConnectedPoints();

  initNets(true);

  //   buildConnectedPoints();

  optConnectionsRoutingLayer();
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
      patch_layer.set_as_routing(idb_layer->is_routing());
      patch_layer.set_horizontal(idb_layer->is_routing());

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
    auto [node_row_num, node_col_num] = grid.buildNodeMatrix(order);

    LOG_INFO << "LM layer order : " << order << ", row = " << node_row_num << ", col = " << node_col_num;
  }
}

void LmLayoutInit::transVia(idb::IdbVia* idb_via, int net_id, LmNodeTYpe type)
{
  auto& patch_layers = _layout->get_patch_layers();

  /// cut layer
  auto cut_layer_shape = idb_via->get_cut_layer_shape();
  auto order = _layout->findLayerId(cut_layer_shape.get_layer()->get_name());
  auto* patch_layer = patch_layers.findPatchLayer(order);
  if (nullptr == patch_layer) {
    LOG_WARNING << "Can not get layer order : " << cut_layer_shape.get_layer()->get_name();
    return;
  }
  auto& grid = patch_layer->get_grid();
  auto* via_coordinate = idb_via->get_coordinate();
  auto [row, col] = grid.findNodeID(via_coordinate->get_x(), via_coordinate->get_y());
  auto& node_data = grid.get_node(row, col).get_node_data();
  node_data.set_type(type);
  node_data.set_status(LmNodeStatus::lm_connected);
  node_data.set_net_id(net_id);
  node_data.set_connect_type(LmNodeConnectType::lm_via);

  /// botttom
  auto enclosure_bottom = idb_via->get_bottom_layer_shape();
  for (auto* rect : enclosure_bottom.get_rect_list()) {
    transEnclosure(rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y(), enclosure_bottom.get_layer()->get_name(),
                   net_id, row, col);
  }

  /// top
  auto enclosure_top = idb_via->get_top_layer_shape();
  for (auto* rect : enclosure_top.get_rect_list()) {
    transEnclosure(rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y(), enclosure_top.get_layer()->get_name(),
                   net_id, row, col);
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
  for (auto* idb_net : idb_pdn->get_net_list()) {
    /// init pdn id map
    pdn_id_map.insert(std::make_pair(idb_net->get_net_name(), -1));

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
          auto [row_1, row_2, co_1, col_2] = grid.get_node_id_range(ll_x, ur_x, ll_y, ur_y);
          for (int row = row_1; row <= row_2; ++row) {
            for (int col = co_1; col <= col_2; ++col) {
              /// set node data
              auto& node_data = grid.get_node(row, col).get_node_data();
              node_data.set_type(LmNodeTYpe::lm_pdn);
              node_data.set_connect_type(LmNodeConnectType::lm_wire);
            }
          }
        }

        if (idb_segment->is_via()) {
          auto* idb_via = idb_segment->get_via();
          transVia(idb_via, -1, LmNodeTYpe::lm_pdn);
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
  int number = 0;

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

    if (i % 1000 == 0) {
      omp_set_lock(&lck);

      number += 1000;
      number = number > inst_total ? inst_total : number;
      LOG_INFO << "Read instance : " << number << " / " << (int) inst_total;

      omp_unset_lock(&lck);
    }
  }

  omp_destroy_lock(&lck);

  LOG_INFO << "LM memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "LM elapsed time " << stats.elapsedRunTime() << " s";

  LOG_INFO << "LM init instances end...";
}

void LmLayoutInit::initIOPins()
{
  auto* idb_design = dmInst->get_idb_design();
  auto* idb_iopins = idb_design->get_io_pin_list();

  auto& patch_layers = _layout->get_patch_layers();

  for (auto* io_pin : idb_iopins->get_pin_list()) {
    /// net io pin has been built in init net flow
    if (false == io_pin->is_net_pin()) {
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
    auto& grid = patch_layer->get_grid();
    if (layer_shape->is_via()) {
      for (IdbRect* rect : layer_shape->get_rect_list()) {
        /// build grid
        auto& grid = patch_layer->get_grid();
        auto [row_id, col_id] = grid.findNodeID(rect->get_middle_point_x(), rect->get_middle_point_y());
        auto& node_data = grid.get_node(row_id, col_id).get_node_data();
        node_data.set_type(LmNodeTYpe::lm_pin);
        node_data.set_status(LmNodeStatus::lm_fix);
        node_data.set_net_id(net_id);
        if (net_id > 0) {
          node_data.set_type(LmNodeTYpe::lm_net);
        }
        node_data.set_connect_type(LmNodeConnectType::lm_via);
      }
    } else {
      for (IdbRect* rect : layer_shape->get_rect_list()) {
        /// build grid
        auto [row_1, row_2, co_1, col_2]
            = grid.get_node_id_range(rect->get_low_x(), rect->get_high_x(), rect->get_low_y(), rect->get_high_y());
        for (int row = row_1; row <= row_2; ++row) {
          for (int col = co_1; col <= col_2; ++col) {
            auto& node_data = grid.get_node(row, col).get_node_data();
            node_data.set_type(LmNodeTYpe::lm_pin);
            node_data.set_status(LmNodeStatus::lm_fix);
            node_data.set_net_id(net_id);
            if (net_id > 0) {
              node_data.set_type(LmNodeTYpe::lm_net);
            }
            node_data.set_connect_type(LmNodeConnectType::lm_wire);
          }
        }
      }
    }
  }

  /// init via in pins
  /// tbd
}

void LmLayoutInit::transNetRect(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y, std::string layer_name, int net_id)
{
  auto& patch_layers = _layout->get_patch_layers();

  auto order = _layout->findLayerId(layer_name);
  auto* patch_layer = patch_layers.findPatchLayer(order);
  if (nullptr == patch_layer) {
    LOG_WARNING << "Can not get layer order : " << layer_name;
    return;
  }
  if (layer_name == "M5") {
    int a = 0;
    a += 1;
  }
  auto& grid = patch_layer->get_grid();
  auto [row_1, row_2, col_1, col_2] = grid.get_node_id_range(ll_x, ur_x, ll_y, ur_y);
  /// net wire must only occupy one grid size

  bool b_horizontal = (ur_x - ll_x) == (ur_y - ll_y) ? patch_layer->is_horizontal() : ((ur_x - ll_x) > (ur_y - ll_y) ? true : false);
  for (int row = row_1; row <= row_2; ++row) {
    for (int col = col_1; col <= col_2; ++col) {
      /// set node data
      auto& node_data = grid.get_node(row, col).get_node_data();
      node_data.set_type(LmNodeTYpe::lm_net);
      node_data.set_connect_type(LmNodeConnectType::lm_wire);
      node_data.set_net_id(net_id);
      if (b_horizontal) {
        if (col == col_1) {
          node_data.set_direction(LmNodeDirection::lm_right);
          bool b_cancel = node_data.is_enclosure() && node_data.is_wire() ? true : false;
          node_data.set_status(LmNodeStatus::lm_end_point, b_cancel);
        } else if (col == col_2) {
          node_data.set_direction(LmNodeDirection::lm_left);
          bool b_cancel = node_data.is_enclosure() && node_data.is_wire() ? true : false;
          node_data.set_status(LmNodeStatus::lm_end_point, b_cancel);
        } else {
          node_data.set_direction(LmNodeDirection::lm_left);
          node_data.set_direction(LmNodeDirection::lm_right);
        }
      } else {
        if (row == row_1) {
          node_data.set_direction(LmNodeDirection::lm_up);
          bool b_cancel = node_data.is_enclosure() && node_data.is_wire() ? true : false;
          node_data.set_status(LmNodeStatus::lm_end_point, b_cancel);
        } else if (row == row_2) {
          node_data.set_direction(LmNodeDirection::lm_down);
          bool b_cancel = node_data.is_enclosure() && node_data.is_wire() ? true : false;
          node_data.set_status(LmNodeStatus::lm_end_point, b_cancel);
        } else {
          node_data.set_direction(LmNodeDirection::lm_up);
          node_data.set_direction(LmNodeDirection::lm_down);
        }
      }
    }
  }
}

void LmLayoutInit::transEnclosure(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y, std::string layer_name, int net_id, int via_row,
                                  int via_col)
{
  auto& patch_layers = _layout->get_patch_layers();

  auto order = _layout->findLayerId(layer_name);
  auto* patch_layer = patch_layers.findPatchLayer(order);
  if (nullptr == patch_layer) {
    LOG_WARNING << "Can not get layer order : " << layer_name;
    return;
  }
  auto& grid = patch_layer->get_grid();
  auto [row_1, row_2, col_1, col_2] = grid.get_node_id_range(ll_x, ur_x, ll_y, ur_y);
  /// net wire must only occupy one grid size

  bool b_horizontal = (ur_x - ll_x) == (ur_y - ll_y) ? patch_layer->is_horizontal() : ((ur_x - ll_x) > (ur_y - ll_y) ? true : false);
  if (b_horizontal) {
    row_1 = via_row;
    row_2 = via_row;
  } else {
    col_1 = via_col;
    col_2 = via_col;
  }

  for (int row = row_1; row <= row_2; ++row) {
    for (int col = col_1; col <= col_2; ++col) {
      /// set node data
      auto& node_data = grid.get_node(row, col).get_node_data();
      node_data.set_type(LmNodeTYpe::lm_net);
      node_data.set_connect_type(LmNodeConnectType::lm_enclosure);
      node_data.set_net_id(net_id);
      if (b_horizontal) {
        if (via_row == row) {
          if (col == col_1 || col == col_2) {
            if (col == col_1) {
              bool b_cancel = node_data.is_wire() && node_data.is_enclosure() ? true : false;
              node_data.set_status(LmNodeStatus::lm_end_point, b_cancel);
              if (b_cancel) {
                node_data.set_direction(LmNodeDirection::lm_left);
                node_data.set_direction(LmNodeDirection::lm_right);
              } else {
                node_data.set_direction(LmNodeDirection::lm_right);
              }
            }
            if (col == col_2) {
              bool b_cancel = node_data.is_wire() && node_data.is_enclosure() ? true : false;
              node_data.set_status(LmNodeStatus::lm_end_point, b_cancel);

              if (b_cancel) {
                node_data.set_direction(LmNodeDirection::lm_left);
                node_data.set_direction(LmNodeDirection::lm_right);
              } else {
                node_data.set_direction(LmNodeDirection::lm_left);
              }
            }
          } else {
            node_data.set_direction(LmNodeDirection::lm_left);
            node_data.set_direction(LmNodeDirection::lm_right);
          }
        }
      } else {
        if (via_col == col) {
          if (row == row_1 || row == row_2) {
            if (row == row_1) {
              bool b_cancel = node_data.is_wire() && node_data.is_enclosure() ? true : false;
              node_data.set_status(LmNodeStatus::lm_end_point, b_cancel);
              if (b_cancel) {
                node_data.set_direction(LmNodeDirection::lm_up);
                node_data.set_direction(LmNodeDirection::lm_down);
              } else {
                node_data.set_direction(LmNodeDirection::lm_up);
              }
            }
            if (row == row_2) {
              bool b_cancel = node_data.is_wire() && node_data.is_enclosure() ? true : false;
              node_data.set_status(LmNodeStatus::lm_end_point, b_cancel);
              if (b_cancel) {
                node_data.set_direction(LmNodeDirection::lm_up);
                node_data.set_direction(LmNodeDirection::lm_down);
              } else {
                node_data.set_direction(LmNodeDirection::lm_down);
              }
            }
          } else {
            node_data.set_direction(LmNodeDirection::lm_up);
            node_data.set_direction(LmNodeDirection::lm_down);
          }
        }
      }
    }
  }
}

void LmLayoutInit::transNetDelta(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y, std::string layer_name, int net_id)
{
  auto& patch_layers = _layout->get_patch_layers();

  auto order = _layout->findLayerId(layer_name);
  auto* patch_layer = patch_layers.findPatchLayer(order);
  if (nullptr == patch_layer) {
    LOG_WARNING << "Can not get layer order : " << layer_name;
    return;
  }
  auto& grid = patch_layer->get_grid();
  auto [row_1, row_2, col_1, col_2] = grid.get_node_id_range(ll_x, ur_x, ll_y, ur_y);

  /////////////////////////////////////////////////////////////////////////////
  /// print
  /////////////////////////////////////////////////////////////////////////////
  {
    LOG_INFO << "*********************origin*************************";
    for (int row = row_1; row <= row_2; ++row) {
      std::string line = "";
      for (int col = col_1; col <= col_2; ++col) {
        auto& node = grid.get_node(row, col);
        auto& node_data = node.get_node_data();
        std::string type = " ";
        type = node_data.is_connected() ? type + "c" : (node_data.is_connecting() ? type + "L" : type + ".");
        type = node_data.is_wire() ? type + "w" : type + ".";
        type = node_data.is_via() ? type + "v" : type + ".";
        type = node_data.is_enclosure() ? type + "o" : type + ".";
        type = node_data.is_end_point() ? type + "e" : type + ".";
        type = node_data.is_pin() ? type + "p" : type + ".";

        line = line + type + " ";
      }
      LOG_INFO << line;
    }

    LOG_INFO << "****************************************************";
  }
  /////////////////////////////////////////////////////////////////////////////
  /// print
  /////////////////////////////////////////////////////////////////////////////

  std::set<LmNode*> node_connectds;
  std::set<LmNode*> node_endpoints;
  std::set<LmNode*> node_connecting;

  /// find connecting node
  for (int row = row_1; row <= row_2; ++row) {
    for (int col = col_1; col <= col_2; ++col) {
      auto& node = grid.get_node(row, col);
      auto& node_data = node.get_node_data();
      node_data.set_type(LmNodeTYpe::lm_net);
      node_data.set_connect_type(LmNodeConnectType::lm_delta);
      node_data.set_net_id(net_id);
      node_data.set_direction(LmNodeDirection::lm_middle);

      if (node_data.is_connected()) {
        node_connectds.insert(&node);
        node_connecting.insert(&node);
      } else {
        if (node_data.is_end_point()) {
          node_endpoints.insert(&node);
          node_connecting.insert(&node);
        }
      }
    }
  }

  /// draw line
  for (auto node_connectd : node_connectds) {
    auto node_row = node_connectd->get_row_id();
    auto node_col = node_connectd->get_col_id();
    for (int row_travel = row_1; row_travel <= row_2; ++row_travel) {
      grid.get_node(row_travel, node_col).get_node_data().set_direction(LmNodeDirection::lm_up);
      grid.get_node(row_travel, node_col).get_node_data().set_direction(LmNodeDirection::lm_down);
    }

    for (int col_travel = col_1; col_travel <= col_2; ++col_travel) {
      grid.get_node(node_row, col_travel).get_node_data().set_direction(LmNodeDirection::lm_left);
      grid.get_node(node_row, col_travel).get_node_data().set_direction(LmNodeDirection::lm_right);
    }
  }

  for (auto node_endpoint : node_endpoints) {
    auto& node_data = node_endpoint->get_node_data();
    /// skip if node is endpoint & connected point
    if (node_data.is_connected()) {
      continue;
    }

    auto node_row = node_endpoint->get_row_id();
    auto node_col = node_endpoint->get_col_id();

    if (node_data.is_direction(LmNodeDirection::lm_left) || node_data.is_direction(LmNodeDirection::lm_right)) {
      for (auto node_connectd : node_connectds) {
        auto connected_row = node_connectd->get_row_id();
        auto connected_col = node_connectd->get_col_id();
        /// connecting endpoint
        auto col_start = node_col < connected_col ? node_col : connected_col;
        auto col_end = node_col < connected_col ? connected_col : node_col;
        for (int col = col_start; col <= col_end; ++col) {
          auto& endpoint_data = grid.get_node(node_row, col).get_node_data();
          if (col == col_start || col == col_end) {
            if (col == col_start) {
              endpoint_data.set_direction(LmNodeDirection::lm_right);
            }
            if (col == col_end) {
              endpoint_data.set_direction(LmNodeDirection::lm_left);
            }
          } else {
            endpoint_data.set_direction(LmNodeDirection::lm_left);
            endpoint_data.set_direction(LmNodeDirection::lm_right);
          }
        }

        /// connecting connected point
        auto row_start = node_row < connected_row ? node_row : connected_row;
        auto row_end = node_row < connected_row ? connected_row : node_row;

        for (int row = row_start; row <= row_end; ++row) {
          auto& endpoint_data = grid.get_node(row, connected_col).get_node_data();
          if (row == row_start || row == row_end) {
            if (row == row_start) {
              endpoint_data.set_direction(LmNodeDirection::lm_up);
            }
            if (row == row_end) {
              endpoint_data.set_direction(LmNodeDirection::lm_down);
            }
          } else {
            endpoint_data.set_direction(LmNodeDirection::lm_up);
            endpoint_data.set_direction(LmNodeDirection::lm_down);
          }
        }
      }
    }

    if (node_data.is_direction(LmNodeDirection::lm_up) || node_data.is_direction(LmNodeDirection::lm_down)) {
      for (auto node_connectd : node_connectds) {
        auto connected_row = node_connectd->get_row_id();
        auto connected_col = node_connectd->get_col_id();

        /// connecting endpoint
        auto row_start = node_row < connected_row ? node_row : connected_row;
        auto row_end = node_row < connected_row ? connected_row : node_row;

        for (int row = row_start; row <= row_end; ++row) {
          auto& endpoint_data = grid.get_node(row, node_col).get_node_data();
          if (row == row_start || row == row_end) {
            if (row == row_start) {
              endpoint_data.set_direction(LmNodeDirection::lm_up);
            }
            if (row == row_end) {
              endpoint_data.set_direction(LmNodeDirection::lm_down);
            }
          } else {
            endpoint_data.set_direction(LmNodeDirection::lm_up);
            endpoint_data.set_direction(LmNodeDirection::lm_down);
          }
        }

        /// connecting connected point
        auto col_start = node_col < connected_col ? node_col : connected_col;
        auto col_end = node_col < connected_col ? connected_col : node_col;
        for (int col = col_start; col <= col_end; ++col) {
          auto& endpoint_data = grid.get_node(connected_row, col).get_node_data();
          if (col == col_start || col == col_end) {
            if (col == col_start) {
              endpoint_data.set_direction(LmNodeDirection::lm_right);
            }
            if (col == col_end) {
              endpoint_data.set_direction(LmNodeDirection::lm_left);
            }
          } else {
            endpoint_data.set_direction(LmNodeDirection::lm_left);
            endpoint_data.set_direction(LmNodeDirection::lm_right);
          }
        }
      }
    }
  }

  /// build connnecting point
  for (int row = row_1; row <= row_2; ++row) {
    for (int col = col_1; col <= col_2; ++col) {
      auto& node = grid.get_node(row, col);
      if (true == setConnectNode(node)) {
      }
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  /// print
  /////////////////////////////////////////////////////////////////////////////
  {
    LOG_INFO << "*********************change*************************";
    for (int row = row_1; row <= row_2; ++row) {
      std::string line = "";
      for (int col = col_1; col <= col_2; ++col) {
        auto& node = grid.get_node(row, col);
        auto& node_data = node.get_node_data();
        std::string type = " ";
        type = node_data.is_connected() ? type + "c" : (node_data.is_connecting() ? type + "L" : type + ".");
        type = node_data.is_wire() ? type + "w" : type + ".";
        type = node_data.is_via() ? type + "v" : type + ".";
        type = node_data.is_enclosure() ? type + "o" : type + ".";
        type = node_data.is_end_point() ? type + "e" : type + ".";
        type = node_data.is_pin() ? type + "p" : type + ".";

        line = line + type + " ";
      }
      LOG_INFO << line;
    }

    LOG_INFO << "****************************************************";
  }
  /////////////////////////////////////////////////////////////////////////////
  /// print
  /////////////////////////////////////////////////////////////////////////////
}

void LmLayoutInit::initNets(bool init_delta)
{
  ieda::Stats stats;

  LOG_INFO << "LM init nets start...";
  auto& net_id_map = _layout->get_net_id_map();

  auto* idb_design = dmInst->get_idb_design();
  auto* idb_nets = idb_design->get_net_list();

  if (false == init_delta) {
    for (int net_id = 0; net_id < (int) idb_nets->get_net_list().size(); ++net_id) {
      /// init net id map
      auto* idb_net = idb_nets->get_net_list()[net_id];
      net_id_map.insert(std::make_pair(idb_net->get_net_name(), net_id));
    }
  }

  // #pragma omp parallel for schedule(dynamic)
  for (int net_id = 0; net_id < (int) idb_nets->get_net_list().size(); ++net_id) {
    auto* idb_net = idb_nets->get_net_list()[net_id];
    // if ("core/sbox_inst/n1554" == idb_net->get_net_name()) {
    //   continue;
    // }

    /// init pins
    /// instance pin
    if (false == init_delta) {
      for (auto* idb_inst_pin : idb_net->get_instance_pin_list()->get_pin_list()) {
        transPin(idb_inst_pin, net_id);
      }

      for (auto* io_pin : idb_net->get_io_pins()->get_pin_list()) {
        transPin(io_pin, net_id);
      }
    }

    /// init wires
    auto* idb_wires = idb_net->get_wire_list();
    for (auto* idb_wire : idb_wires->get_wire_list()) {
      for (auto* idb_segment : idb_wire->get_segment_list()) {
        if (init_delta) {
          if (idb_segment->is_rect()) {
            /// wire patch
            auto* coordinate = idb_segment->get_point_start();
            auto* rect_delta = idb_segment->get_delta_rect();
            IdbRect* rect = new IdbRect(rect_delta);
            rect->moveByStep(coordinate->get_x(), coordinate->get_y());

            /// build grid
            transNetDelta(rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y(),
                          idb_segment->get_layer()->get_name(), net_id);

            delete rect;
          }
        } else {
          /// wire
          if (idb_segment->is_via()) {
            for (auto* idb_via : idb_segment->get_via_list()) {
              transVia(idb_via, net_id, LmNodeTYpe::lm_net);
            }
          } else if (idb_segment->is_rect()) {
            /// wire patch
            continue;
          } else {
            /// nothing to do
          }

          /// build wire
          if (idb_segment->get_point_number() >= 2) {
            auto* routing_layer = dynamic_cast<IdbLayerRouting*>(idb_segment->get_layer());

            /// get bounding box
            auto* point_1 = idb_segment->get_point_start();
            auto* point_2 = idb_segment->get_point_second();

            int32_t ll_x = std::min(point_1->get_x(), point_2->get_x());
            int32_t ll_y = std::min(point_1->get_y(), point_2->get_y());
            int32_t ur_x = std::max(point_1->get_x(), point_2->get_x());
            int32_t ur_y = std::max(point_1->get_y(), point_2->get_y());

            /// build grid
            transNetRect(ll_x, ll_y, ur_x, ur_y, routing_layer->get_name(), net_id);
          }
        }
      }
      if (net_id % 1000 == 0) {
        LOG_INFO << "Read nets : " << net_id << " / " << (int) idb_nets->get_net_list().size();
      }
    }
  }

  LOG_INFO << "Read nets : " << idb_nets->get_net_list().size() << " / " << (int) idb_nets->get_net_list().size();
  LOG_INFO << "LM memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "LM elapsed time " << stats.elapsedRunTime() << " s";

  LOG_INFO << "LM init nets end...";
}

void LmLayoutInit::buildConnectedPoints()
{
  ieda::Stats stats;

  LOG_INFO << "LM build connected points start...";

  /// cut layer must be set first
  int connected_num = buildConnectedPointsCutLayer();

  connected_num += buildConnectedPointsRoutingLayer();

  LOG_INFO << "LM connected point number : " << connected_num;
  LOG_INFO << "LM memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "LM elapsed time " << stats.elapsedRunTime() << " s";

  LOG_INFO << "LM build connected points end...";
}

int LmLayoutInit::buildConnectedPointsCutLayer()
{
  ieda::Stats stats;

  LOG_INFO << "LM build connected points for cut layer start...";
  int connected_num = 0;
  omp_lock_t lck;
  omp_init_lock(&lck);

  auto& patch_layers = _layout->get_patch_layers();
  for (int i = patch_layers.get_layer_order_bottom(); i <= patch_layers.get_layer_order_top(); ++i) {
    auto* patch_layer = patch_layers.findPatchLayer(i);
    if (patch_layer == nullptr) {
      LOG_ERROR << "Data error, layer not exits, id : " << i;
      continue;
    }

    if (false == patch_layer->is_routing()) {
      auto& grid = patch_layer->get_grid();
      auto& node_matrix = grid.get_node_matrix();

      /// cut layer
      auto* patch_layer_bottom = patch_layers.findPatchLayer(i - 1);  /// bottom routing layer
      auto* patch_layer_top = patch_layers.findPatchLayer(i + 1);     /// top routing layer
      if (patch_layer_bottom == nullptr || patch_layer_top == nullptr) {
        LOG_ERROR << "Data error, layer not exits, id : " << i - 1 << " and " << i + 1;
        continue;
      }

      auto& node_matrix_bottom = patch_layer_bottom->get_grid().get_node_matrix();
      auto& node_matrix_top = patch_layer_top->get_grid().get_node_matrix();
#pragma omp parallel for schedule(dynamic)
      for (int row = 0; row < grid.get_info().node_row_num; ++row) {
        for (int col = 0; col < grid.get_info().node_col_num; ++col) {
          if (node_matrix[row][col].get_node_data().is_net()) {
            node_matrix_bottom[row][col].get_node_data().set_type(LmNodeTYpe::lm_net);
            node_matrix_bottom[row][col].get_node_data().set_direction(LmNodeDirection::lm_up);
            node_matrix_bottom[row][col].get_node_data().set_status(LmNodeStatus::lm_connected);
            node_matrix_bottom[row][col].get_node_data().set_connect_type(LmNodeConnectType::lm_via);
            node_matrix_bottom[row][col].get_node_data().set_net_id(node_matrix[row][col].get_node_data().get_net_id());

            node_matrix_top[row][col].get_node_data().set_type(LmNodeTYpe::lm_net);
            node_matrix_top[row][col].get_node_data().set_direction(LmNodeDirection::lm_down);
            node_matrix_top[row][col].get_node_data().set_status(LmNodeStatus::lm_connected);
            node_matrix_top[row][col].get_node_data().set_connect_type(LmNodeConnectType::lm_via);
            node_matrix_top[row][col].get_node_data().set_net_id(node_matrix[row][col].get_node_data().get_net_id());

            omp_set_lock(&lck);
            connected_num = connected_num + 2;
            omp_unset_lock(&lck);
          }
        }
        if (row % 1000 == 0) {
          LOG_INFO << "Patch layer " << i << " Read rows : " << row << " / " << grid.get_info().node_row_num;
        }
      }
      LOG_INFO << "Patch layer " << i << " Read rows : " << grid.get_info().node_row_num << " / " << grid.get_info().node_row_num;
    }
  }

  omp_destroy_lock(&lck);

  LOG_INFO << "LM build connected points for cut layer end...";

  return connected_num;
}

bool LmLayoutInit::setConnectNode(LmNode& node)
{
  auto& node_data = node.get_node_data();
  /// steiner point
  if (node_data.is_net() && node.is_steiner_point()) {
    node_data.set_status(LmNodeStatus::lm_connected);
    if (node_data.is_end_point()) {
      node_data.set_status(LmNodeStatus::lm_end_point, true);
    }
    return true;
  }
  /// pin connected to wire
  /// tbd : node_data.is_net() && node_data.is_pin()
  //   if ((node_data.is_net() && node_data.is_pin())) {
  //     node_data.set_status(LmNodeStatus::lm_connected);
  //     return true;
  //   }
  /// tbd
  //   if ((node_data.is_net() && node_data.is_io())) {
  //     node_data.set_status(LmNodeStatus::lm_connected);
  //     return true;
  //   }

  /// set corner as connecting
  if (node.is_corner()) {
    node_data.set_status(LmNodeStatus::lm_connecting);
    if (node_data.is_end_point()) {
      node_data.set_status(LmNodeStatus::lm_end_point, true);
    }

    return true;
  }

  //   node_data.set_status(LmNodeStatus::lm_connecting, true);
  //   node_data.set_status(LmNodeStatus::lm_connected, true);

  return false;
}

int LmLayoutInit::buildConnectedPointsRoutingLayer()
{
  ieda::Stats stats;

  LOG_INFO << "LM build connected points for routing layer start...";

  int connected_num = 0;
  omp_lock_t lck;
  omp_init_lock(&lck);

  auto& patch_layers = _layout->get_patch_layers();
  for (int i = patch_layers.get_layer_order_bottom(); i <= patch_layers.get_layer_order_top(); ++i) {
    auto* patch_layer = patch_layers.findPatchLayer(i);
    if (patch_layer == nullptr) {
      LOG_ERROR << "Data error, layer not exits, id : " << i;
      continue;
    }
    auto& grid = patch_layer->get_grid();
    auto& node_matrix = grid.get_node_matrix();

    if (patch_layer->is_routing()) {
#pragma omp parallel for schedule(dynamic)
      for (int row = 0; row < grid.get_info().node_row_num; ++row) {
        for (int col = 0; col < grid.get_info().node_col_num; ++col) {
          /// set steiner points as connected
          if (true == setConnectNode(node_matrix[row][col])) {
            omp_set_lock(&lck);
            connected_num++;
            omp_unset_lock(&lck);
          }
        }
        if (row % 1000 == 0) {
          LOG_INFO << "Patch layer " << i << " Read rows : " << row << " / " << grid.get_info().node_row_num;
        }
      }

      LOG_INFO << "Patch layer " << i << " Read rows : " << grid.get_info().node_row_num << " / " << grid.get_info().node_row_num;
    }
  }

  omp_destroy_lock(&lck);

  LOG_INFO << "LM build connected points for routing layer end...";

  return connected_num;
}

void LmLayoutInit::optConnectionsRoutingLayer()
{
  ieda::Stats stats;

  LOG_INFO << "LM optimize connections for routing layer start...";

  omp_lock_t lck;
  omp_init_lock(&lck);

  auto& patch_layers = _layout->get_patch_layers();
  for (int i = patch_layers.get_layer_order_bottom(); i <= patch_layers.get_layer_order_top(); ++i) {
    auto* patch_layer = patch_layers.findPatchLayer(i);
    if (patch_layer == nullptr) {
      LOG_ERROR << "Data error, layer not exits, id : " << i;
      continue;
    }
    auto& grid = patch_layer->get_grid();
    auto& node_matrix = grid.get_node_matrix();

    if (patch_layer->is_routing()) {
      // #pragma omp parallel for schedule(dynamic)
      for (int row = 1; row < grid.get_info().node_row_num - 1; ++row) {
        std::set<LmNode*> connecting_nodes;
        for (int col = 1; col < grid.get_info().node_col_num - 1; ++col) {
          /// eliminate useless connectiong points
          auto& node_data = node_matrix[row][col].get_node_data();
          if (false == node_data.is_net() || false == node_data.is_connecting()) {
            continue;
          }

          /// eliminate connecting point
          auto& left_node = node_matrix[row][col - 1];
          auto& right_node = node_matrix[row][col + 1];
          auto& down_node = node_matrix[row - 1][col];
          auto& up_node = node_matrix[row + 1][col];

          if (false == node_data.is_direction(LmNodeDirection::lm_left)
              && node_data.get_net_id() == left_node.get_node_data().get_net_id()) {
            // node_data.set_status(LmNodeStatus::lm_connecting, true);
            node_data.set_direction(LmNodeDirection::lm_left);
          }

          if (false == node_data.is_direction(LmNodeDirection::lm_right)
              && node_data.get_net_id() == right_node.get_node_data().get_net_id()) {
            // node_data.set_status(LmNodeStatus::lm_connecting, true);
            node_data.set_direction(LmNodeDirection::lm_right);
          }

          if (false == node_data.is_direction(LmNodeDirection::lm_down)
              && node_data.get_net_id() == down_node.get_node_data().get_net_id()) {
            // node_data.set_status(LmNodeStatus::lm_connecting, true);
            node_data.set_direction(LmNodeDirection::lm_down);
          }

          if (false == node_data.is_direction(LmNodeDirection::lm_up) && node_data.get_net_id() == up_node.get_node_data().get_net_id()) {
            // node_data.set_status(LmNodeStatus::lm_connecting, true);
            node_data.set_direction(LmNodeDirection::lm_up);
          }
        }
        if (row % 1000 == 0) {
          LOG_INFO << "Patch layer " << i << " Read rows : " << row << " / " << grid.get_info().node_row_num;
        }
      }

      LOG_INFO << "Patch layer " << i << " Read rows : " << grid.get_info().node_row_num << " / " << grid.get_info().node_row_num;
    }
  }

  omp_destroy_lock(&lck);

  LOG_INFO << "LM optimize connections for routing layer end...";
}

}  // namespace ilm