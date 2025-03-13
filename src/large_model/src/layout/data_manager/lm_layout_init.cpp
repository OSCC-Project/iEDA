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
#include "lm_grid_info.h"
#include "omp.h"
#include "usage.hh"

namespace ilm {

void LmLayoutInit::init()
{
  initViaIds();

  initLayers();
  initDie();
  initTracks();
  //   initPDN();
  //   initInstances();
  //   initIOPins();

  initNets();
}

void LmLayoutInit::initViaIds()
{
  auto* idb_layout = dmInst->get_idb_layout();
  auto* idb_design = dmInst->get_idb_design();
  auto* idb_lef_vias = idb_layout->get_via_list();
  auto* idb_def_vias = idb_design->get_via_list();

  int index = 0;
  for (auto* via : idb_lef_vias->get_via_list()) {
    _layout->add_via_map(index++, via->get_name());
  }

  for (auto* via : idb_def_vias->get_via_list()) {
    _layout->add_via_map(index++, via->get_name());
  }

  LOG_INFO << "Via number : " << index;
}

void LmLayoutInit::initDie()
{
  auto* idb_layout = dmInst->get_idb_layout();
  auto* idb_die = idb_layout->get_die();

  auto& layout_layers = _layout->get_layout_layers();
  for (auto& [order, layout_layer] : layout_layers.get_layout_layer_map()) {
    layout_layer.set_llx(idb_die->get_llx());
    layout_layer.set_lly(idb_die->get_lly());
    layout_layer.set_urx(idb_die->get_urx());
    layout_layer.set_ury(idb_die->get_ury());

    gridInfoInst.llx = idb_die->get_llx();
    gridInfoInst.lly = idb_die->get_lly();
    gridInfoInst.urx = idb_die->get_urx();
    gridInfoInst.ury = idb_die->get_ury();
  }
}

void LmLayoutInit::initLayers()
{
  auto* idb_layout = dmInst->get_idb_layout();
  auto* idb_layers = idb_layout->get_layers();
  auto idb_layer_1st = dmInst->get_config().get_routing_layer_1st();

  auto& layout_layers = _layout->get_layout_layers();
  auto& layout_layer_map = layout_layers.get_layout_layer_map();

  bool b_record = false;
  int index = 0;
  for (auto* idb_layer : idb_layers->get_layers()) {
    if (idb_layer->get_name() == idb_layer_1st) {
      b_record = true;
    }

    if (true == b_record) {
      _layout->add_layer_map(index, idb_layer->get_name());

      LmLayoutLayer layout_layer;
      layout_layer.set_layer_name(idb_layer->get_name());
      layout_layer.set_layer_order(index);
      layout_layer.set_as_routing(idb_layer->is_routing());
      if (idb_layer->is_routing()) {
        auto* routing_layer = dynamic_cast<IdbLayerRouting*>(idb_layer);
        layout_layer.set_horizontal(routing_layer->is_horizontal());
        layout_layer.set_wire_width(routing_layer->get_width());
      }

      auto& grid = layout_layer.get_grid();
      grid.layer_order = index;

      layout_layer_map.insert(std::make_pair(index, layout_layer));

      index++;

      if (false == idb_layer->is_cut() && false == idb_layer->is_routing()) {
        break;
      }
    }
  }

  layout_layers.set_layer_order_bottom(0);
  layout_layers.set_layer_order_top(index - 1);

  LOG_INFO << "Layer number : " << index;
}

void LmLayoutInit::initTrackGrid(idb::IdbTrackGrid* idb_track_grid)
{
  auto start = idb_track_grid->get_track()->get_start();
  auto pitch = idb_track_grid->get_track()->get_pitch();

  if (idb_track_grid->get_track()->is_track_vertical()) {
    gridInfoInst.x_start = start;
    gridInfoInst.x_step = pitch / 2;
  } else {
    gridInfoInst.y_start = start;
    gridInfoInst.y_step = pitch / 2;
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

  auto& layout_layers = _layout->get_layout_layers();
  auto& layout_layer_map = layout_layers.get_layout_layer_map();

  initTrackGrid(idb_track_grid_prefer);
  initTrackGrid(idb_track_grid_nonprefer);

  buildLayoutGrid();

  LOG_INFO << "LM memory usage " << stats.memoryDelta() << " MB";
  LOG_INFO << "LM elapsed time " << stats.elapsedRunTime() << " s";

  LOG_INFO << "LM init tracks end...";
}

void LmLayoutInit::buildLayoutGrid()
{
  auto& layout_layers = _layout->get_layout_layers();
  auto& layout_layer_map = layout_layers.get_layout_layer_map();
  // #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < (int) layout_layer_map.size(); ++i) {
    //   for (auto& [order, layout_layer] : layout_layer_map) {
    auto it = layout_layer_map.begin();
    std::advance(it, i);
    auto order = it->first;
    auto& layout_layer = it->second;

    auto& grid = layout_layer.get_grid();
    auto [node_row_num, node_col_num] = grid.buildNodeMatrix(order);

    LOG_INFO << "LM layer order : " << order << ", row = " << node_row_num << ", col = " << node_col_num;
  }
}

void LmLayoutInit::transVia(idb::IdbVia* idb_via, int net_id, LmNodeTYpe type)
{
  auto& layout_layers = _layout->get_layout_layers();

  /// cut layer
  auto cut_layer_shape = idb_via->get_cut_layer_shape();
  auto order = _layout->findLayerId(cut_layer_shape.get_layer()->get_name());
  auto* layout_layer = layout_layers.findLayoutLayer(order);
  if (nullptr == layout_layer) {
    LOG_WARNING << "Can not get layer order : " << cut_layer_shape.get_layer()->get_name();
    return;
  }
  auto& grid = layout_layer->get_grid();
  auto* via_coordinate = idb_via->get_coordinate();
  auto [row, col] = gridInfoInst.findNodeID(via_coordinate->get_x(), via_coordinate->get_y());

  auto* node = grid.get_node(row, col, true);
  node->set_col_id(col);
  node->set_row_id(row);
  node->set_layer_id(order);

  LmNodeData* node_data = node->get_node_data(net_id, true);
  node_data->set_type(type);
  if (type == LmNodeTYpe::lm_net) {
    node_data->set_net_id(net_id);
  }
  if (type == LmNodeTYpe::lm_pdn) {
    node_data->set_pdn_id(net_id);
  }
  node_data->set_connect_type(LmNodeConnectType::lm_via);

  /// botttom
  auto enclosure_bottom = idb_via->get_bottom_layer_shape();
  for (auto* rect : enclosure_bottom.get_rect_list()) {
    transEnclosure(rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y(), enclosure_bottom.get_layer()->get_name(),
                   net_id, row, col, type);
  }

  /// top
  auto enclosure_top = idb_via->get_top_layer_shape();
  for (auto* rect : enclosure_top.get_rect_list()) {
    transEnclosure(rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y(), enclosure_top.get_layer()->get_name(),
                   net_id, row, col, type);
  }
}

void LmLayoutInit::initPDN()
{
  ieda::Stats stats;

  LOG_INFO << "LM init PDN start...";
  auto& layout_layers = _layout->get_layout_layers();

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
  for (int i = 0; i < idb_pdn->get_net_list().size(); ++i) {
    auto* idb_net = idb_pdn->get_net_list()[i];
    /// init pdn id map
    _layout->add_pdn_map(i, idb_net->get_net_name());

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
          auto* layout_layer = layout_layers.findLayoutLayer(order);
          if (nullptr == layout_layer) {
            LOG_WARNING << "Can not get layer order : " << routing_layer->get_name();
            continue;
          }
          auto& grid = layout_layer->get_grid();
          auto [row_1, row_2, co_1, col_2] = gridInfoInst.get_node_id_range(ll_x, ur_x, ll_y, ur_y);
          for (int row = row_1; row <= row_2; ++row) {
            for (int col = co_1; col <= col_2; ++col) {
              /// set node data
              auto* node = grid.get_node(row, col, true);
              LmNodeData* node_data = node->get_node_data(-1, true);
              node->set_col_id(col);
              node->set_row_id(row);
              node->set_layer_id(order);
              node_data->set_type(LmNodeTYpe::lm_pdn);
              node_data->set_connect_type(LmNodeConnectType::lm_wire);
              node_data->set_pdn_id(i);
            }
          }
        }

        if (idb_segment->is_via()) {
          auto* idb_via = idb_segment->get_via();
          transVia(idb_via, i, LmNodeTYpe::lm_pdn);
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

  auto& layout_layers = _layout->get_layout_layers();

  auto* idb_design = dmInst->get_idb_design();
  auto* idb_insts = idb_design->get_instance_list();

  omp_lock_t lck;
  omp_init_lock(&lck);

  int inst_total = (int) idb_insts->get_instance_list().size();
  int number = 0;

  // #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < inst_total; ++i) {
    auto* idb_inst = idb_insts->get_instance_list()[i];
    _layout->add_instance_map(i, idb_inst->get_name());

    auto* idb_inst_pins = idb_inst->get_pin_list();
    for (auto* idb_pin : idb_inst_pins->get_pin_list()) {
      if (false == idb_pin->is_net_pin()) {
        auto type = idb_pin->is_special_net_pin() ? LmNodeTYpe::lm_pdn : LmNodeTYpe::kNone;
        transPin(idb_pin, -1, type, i);
      }
    }

    // for (auto* layer_shape : idb_inst->get_obs_box_list()) {
    //   auto* layer = layer_shape->get_layer();
    //   auto order = _layout->findLayerId(layer->get_name());
    //   auto* layout_layer = layout_layers.findLayoutLayer(order);
    //   if (nullptr == layout_layer) {
    //     // LOG_WARNING << "Can not get layer order : " << layer->get_name();
    //     continue;
    //   }
    // }

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

  for (auto* io_pin : idb_iopins->get_pin_list()) {
    /// net io pin has been built in init net flow
    if (false == io_pin->is_net_pin()) {
      auto net_id = _layout->findPdnId(io_pin->get_net_name());
      /// net has been connected
      auto type = io_pin->is_special_net_pin() ? LmNodeTYpe::lm_pdn : LmNodeTYpe::kNone;
      transPin(io_pin, net_id, type);
    }
  }
}

void LmLayoutInit::transPin(idb::IdbPin* idb_pin, int net_id, LmNodeTYpe type, int instance_id, int pin_id, bool b_io)
{
  auto& layout_layers = _layout->get_layout_layers();
  for (auto* layer_shape : idb_pin->get_port_box_list()) {
    auto order = _layout->findLayerId(layer_shape->get_layer()->get_name());
    auto* layout_layer = layout_layers.findLayoutLayer(order);
    if (nullptr == layout_layer) {
      LOG_WARNING << "Can not get layer order : " << layer_shape->get_layer()->get_name();
      continue;
    }
    auto& grid = layout_layer->get_grid();

    if (layer_shape->is_via()) {
      for (IdbRect* rect : layer_shape->get_rect_list()) {
        /// build grid
        auto& grid = layout_layer->get_grid();
        auto [row_id, col_id] = gridInfoInst.findNodeID(rect->get_middle_point_x(), rect->get_middle_point_y());
        auto* node = grid.get_node(row_id, col_id, true);
        node->set_col_id(col_id);
        node->set_row_id(row_id);
        node->set_layer_id(order);

        LmNodeData* node_data = node->get_node_data(net_id, true);
        node_data->set_type(LmNodeTYpe::lm_pin);
        node_data->set_type(type);
        if (type == LmNodeTYpe::lm_net) {
          node_data->set_net_id(net_id);
        }

        if (type == LmNodeTYpe::lm_pdn) {
          node_data->set_pdn_id(net_id);
        }

        if (instance_id != -1) {
          node_data->set_instance_id(instance_id);
        }

        if (b_io) {
          node_data->set_type(LmNodeTYpe::lm_io);
        }
        if (node_data->get_pin_id() > -1 && pin_id != node_data->get_pin_id()) {
          error_pin_num++;
        }
        if (pin_id > -1) {
          node_data->set_pin_id(pin_id);
        }
        node_data->set_connect_type(LmNodeConnectType::lm_via);
      }
    } else {
      for (IdbRect* rect : layer_shape->get_rect_list()) {
        /// build grid
        int llx = rect->get_low_x();
        int urx = rect->get_high_x();
        int lly = rect->get_low_y();
        int ury = rect->get_high_y();

        auto [row_1, row_2, col_1, col_2] = gridInfoInst.get_node_id_range(llx, urx, lly, ury);

        for (int row = row_1; row <= row_2; ++row) {
          for (int col = col_1; col <= col_2; ++col) {
            auto* node = grid.get_node(row, col, true);
            node->set_col_id(col);
            node->set_row_id(row);
            node->set_layer_id(order);

            LmNodeData* node_data = node->get_node_data(net_id, true);
            node_data->set_type(LmNodeTYpe::lm_pin);
            node_data->set_type(type);
            if (type == LmNodeTYpe::lm_net) {
              node_data->set_net_id(net_id);
            }

            if (type == LmNodeTYpe::lm_pdn) {
              node_data->set_pdn_id(net_id);
            }

            if (instance_id != -1) {
              node_data->set_instance_id(instance_id);
            }

            if (b_io) {
              node_data->set_type(LmNodeTYpe::lm_io);
            }
            if (node_data->get_pin_id() > -1 && pin_id != node_data->get_pin_id()) {
              error_pin_num++;
            }

            if (pin_id > -1) {
              node_data->set_pin_id(pin_id);
            }
          }
        }
      }
    }
  }

  /// init via in pins
  /// tbd
}

void LmLayoutInit::transNetRect(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y, std::string layer_name, int net_id, LmNodeTYpe type)
{
  auto& layout_layers = _layout->get_layout_layers();

  auto order = _layout->findLayerId(layer_name);
  auto* layout_layer = layout_layers.findLayoutLayer(order);
  if (nullptr == layout_layer) {
    LOG_WARNING << "Can not get layer order : " << layer_name;
    return;
  }

  auto& grid = layout_layer->get_grid();
  auto [row_1, row_2, col_1, col_2] = gridInfoInst.get_node_id_range(ll_x, ur_x, ll_y, ur_y);
  /// net wire must only occupy one grid size
  for (int row = row_1; row <= row_2; ++row) {
    for (int col = col_1; col <= col_2; ++col) {
      /// set node data
      auto* node = grid.get_node(row, col, true);
      node->set_col_id(col);
      node->set_row_id(row);
      node->set_layer_id(order);
      LmNodeData* node_data = node->get_node_data(net_id, true);
      node_data->set_type(LmNodeTYpe::lm_net);
      node_data->set_connect_type(LmNodeConnectType::lm_wire);
      if (type == LmNodeTYpe::lm_net) {
        node_data->set_net_id(net_id);
      }

      if (type == LmNodeTYpe::lm_pdn) {
        node_data->set_pdn_id(net_id);
      }
    }
  }
}

void LmLayoutInit::transEnclosure(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y, std::string layer_name, int net_id, int via_row,
                                  int via_col, LmNodeTYpe type)
{
  auto& layout_layers = _layout->get_layout_layers();

  auto order = _layout->findLayerId(layer_name);
  auto* layout_layer = layout_layers.findLayoutLayer(order);
  if (nullptr == layout_layer) {
    LOG_WARNING << "Can not get layer : " << layer_name;
    return;
  }
  auto& grid = layout_layer->get_grid();
  auto [row_1, row_2, col_1, col_2] = gridInfoInst.get_node_id_range(ll_x, ur_x, ll_y, ur_y);

  for (int row = row_1; row <= row_2; ++row) {
    for (int col = col_1; col <= col_2; ++col) {
      auto* node = grid.get_node(row, col, true);
      node->set_col_id(col);
      node->set_row_id(row);
      node->set_layer_id(order);
      LmNodeData* node_data = node->get_node_data(net_id, true);
      node_data->set_type(type);
      node_data->set_connect_type(LmNodeConnectType::lm_enclosure);
      if (type == LmNodeTYpe::lm_net) {
        node_data->set_net_id(net_id);
      }

      if (type == LmNodeTYpe::lm_pdn) {
        node_data->set_pdn_id(net_id);
      }
    }
  }
}

void LmLayoutInit::transNetDelta(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y, std::string layer_name, int net_id,
                                 LmNodeTYpe type)
{
  auto& layout_layers = _layout->get_layout_layers();

  auto order = _layout->findLayerId(layer_name);
  auto* layout_layer = layout_layers.findLayoutLayer(order);
  if (nullptr == layout_layer) {
    LOG_WARNING << "Can not get layer order : " << layer_name;
    return;
  }
  auto& grid = layout_layer->get_grid();
  auto [row_1, row_2, col_1, col_2] = gridInfoInst.get_node_id_range(ll_x, ur_x, ll_y, ur_y);

  for (int row = row_1; row <= row_2; ++row) {
    for (int col = col_1; col <= col_2; ++col) {
      auto* node = grid.get_node(row, col, true);
      node->set_col_id(col);
      node->set_row_id(row);
      node->set_layer_id(order);
      LmNodeData* node_data = node->get_node_data(net_id, true);
      node_data->set_type(LmNodeTYpe::lm_net);
      node_data->set_connect_type(LmNodeConnectType::lm_delta);
      if (type == LmNodeTYpe::lm_net) {
        node_data->set_net_id(net_id);
      }

      if (type == LmNodeTYpe::lm_pdn) {
        node_data->set_pdn_id(net_id);
      }
    }
  }
}

void LmLayoutInit::initNets()
{
  ieda::Stats stats;

  LOG_INFO << "LM init nets start...";

  auto* idb_design = dmInst->get_idb_design();
  auto* idb_nets = idb_design->get_net_list();
  auto& graph = _layout->get_graph();
  int pin_id = 0;
  for (int net_id = 0; net_id < (int) idb_nets->get_net_list().size(); ++net_id) {
    /// init net id map
    auto* idb_net = idb_nets->get_net_list()[net_id];
    /// ignore net if pin number < 2
    if (idb_net->get_pin_number() < 2) {
      continue;
    }
    _layout->add_net_map(net_id, idb_net->get_net_name());

    auto* lm_net = graph.addNet(net_id);
    if (lm_net == nullptr) {
      continue;
    }

    auto* driver_pin = idb_net->get_driving_pin();
    {
      auto instance_name = driver_pin->is_io_pin() ? "" : driver_pin->get_instance()->get_name();

      LmPin lm_pin;
      lm_pin.pin_id = pin_id;
      lm_pin.pin_name = driver_pin->get_pin_name();
      lm_pin.instance_name = instance_name;
      lm_pin.is_driver = true;
      lm_net->addPin(pin_id, lm_pin);

      lm_net->addPinId(pin_id);
      _layout->add_pin_map(pin_id, instance_name, driver_pin->get_pin_name());

      pin_id++;
    }

    for (auto* load_pin : idb_net->get_load_pins()) {
      auto instance_name = load_pin->is_io_pin() ? "" : load_pin->get_instance()->get_name();

      LmPin lm_pin;
      lm_pin.pin_id = pin_id;
      lm_pin.pin_name = load_pin->get_pin_name();
      lm_pin.instance_name = instance_name;
      lm_pin.is_driver = false;
      lm_net->addPin(pin_id, lm_pin);

      lm_net->addPinId(pin_id);
      _layout->add_pin_map(pin_id, instance_name, load_pin->get_pin_name());

      pin_id++;
    }

    for (auto* io_pin : idb_net->get_io_pins()->get_pin_list()) {
      if (lm_net == nullptr) {
        continue;
      }
      lm_net->addPinId(pin_id);
      _layout->add_pin_map(pin_id, "", io_pin->get_pin_name());
      pin_id++;
    }
  }

#pragma omp parallel for schedule(dynamic)
  for (int net_id = 0; net_id < (int) idb_nets->get_net_list().size(); ++net_id) {
    auto* idb_net = idb_nets->get_net_list()[net_id];
    /// ignore net if pin number < 2
    if (idb_net->get_pin_number() < 2) {
      continue;
    }

    auto* lm_net = graph.get_net(net_id);

    for (auto* idb_inst_pin : idb_net->get_instance_pin_list()->get_pin_list()) {
      auto pin_id = _layout->findPinId(idb_inst_pin->get_instance()->get_name(), idb_inst_pin->get_pin_name());
      auto instance_id = _layout->findInstId(idb_inst_pin->get_instance()->get_name());
      transPin(idb_inst_pin, net_id, LmNodeTYpe::lm_net, instance_id, pin_id, false);
    }

    for (auto* io_pin : idb_net->get_io_pins()->get_pin_list()) {
      auto pin_id = _layout->findPinId("", io_pin->get_pin_name());
      transPin(io_pin, net_id, LmNodeTYpe::lm_net, -1, pin_id, true);
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
          IdbRect* rect = new IdbRect(rect_delta);
          rect->moveByStep(coordinate->get_x(), coordinate->get_y());

          /// build grid
          transNetDelta(rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y(), idb_segment->get_layer()->get_name(),
                        net_id, LmNodeTYpe::lm_net);

          delete rect;
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
          transNetRect(ll_x, ll_y, ur_x, ur_y, routing_layer->get_name(), net_id, LmNodeTYpe::lm_net);
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

}  // namespace ilm