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
#include "ieco_via_init.h"

#include "idm.h"
#include "ieco_dm.h"
#include "omp.h"

namespace ieco {

ECOViaInit::ECOViaInit(EcoDataManager* data_manager)
{
  _data_manager = data_manager;
}

ECOViaInit::~ECOViaInit()
{
}

void ECOViaInit::initData()
{
  init_via_masters();
  init_nets();
}

void ECOViaInit::init_via_masters()
{
  auto& eco_data = _data_manager->get_eco_data();
  auto idb_layout = dmInst->get_idb_layout();

  for (auto* idb_via : idb_layout->get_via_list()->get_via_list()) {
    auto& via_layer = eco_data.get_via_layer(idb_via);

    auto master_name = idb_via->get_name();
    auto via_master = idb_via->get_instance();

    via_layer.addViaMaster(master_name, via_master);
  }
}

std::map<int, std::vector<EcoDataVia>> ECOViaInit::get_net_vias(idb::IdbNet* idb_net)
{
  std::map<int, std::vector<EcoDataVia>> eco_vias;  // int : layer order

  /// init via for each net
  for (auto* idb_wire : idb_net->get_wire_list()->get_wire_list()) {
    for (auto* idb_segment : idb_wire->get_segment_list()) {
      if (idb_segment->is_via()) {
        for (auto* idb_via : idb_segment->get_via_list()) {
          auto cut_layer_shape = idb_via->get_cut_layer_shape();
          auto& via_list = eco_vias[cut_layer_shape.get_layer()->get_order()];
          via_list.emplace_back(EcoDataVia(idb_via));
          // eco_vias.insert(std::make_pair(cut_layer_shape.get_layer()->get_order(), EcoDataVia(idb_via)));
        }
      }
    }
  }

  return eco_vias;
}

void ECOViaInit::init_segment_rect(EcoDataVia& eco_via, idb::IdbRegularWireSegment* idb_segment, int cut_layer_order)
{
  if (false == idb_segment->is_wire() && false == idb_segment->is_rect()) {
    return;
  }

  /// routing layers
  auto* segment_layer = idb_segment->get_layer();
  /// must neighbour layer for via cut layer
  if (cut_layer_order - 1 != segment_layer->get_order() && cut_layer_order + 1 != segment_layer->get_order()) {
    return;
  }

  auto idb_rect = idb_segment->get_segment_rect();
  /// is bottom
  if (cut_layer_order - 1 == segment_layer->get_order()) {
    /// check intersection
    if (eco_via.intersectedBottom(idb_rect.get_low_x(), idb_rect.get_low_y(), idb_rect.get_high_x(), idb_rect.get_high_y())) {
      /// add to connected data
      eco_via.addConnectedBottom(idb_rect.get_low_x(), idb_rect.get_low_y(), idb_rect.get_high_x(), idb_rect.get_high_y());
    }
  }
  /// is top
  if (cut_layer_order + 1 == segment_layer->get_order()) {
    /// check intersection
    if (eco_via.intersectedTop(idb_rect.get_low_x(), idb_rect.get_low_y(), idb_rect.get_high_x(), idb_rect.get_high_y())) {
      /// add to connected data
      eco_via.addConnectedTop(idb_rect.get_low_x(), idb_rect.get_low_y(), idb_rect.get_high_x(), idb_rect.get_high_y());
    }
  }
}

void ECOViaInit::init_segment_via(EcoDataVia& eco_via, idb::IdbRegularWireSegment* idb_segment, int cut_layer_order)
{
  if (false == idb_segment->is_via()) {
    return;
  }

  for (auto* idb_via : idb_segment->get_via_list()) {
    /// ignore same via
    if (eco_via.get_idb_via() == idb_via) {
      continue;
    }

    auto cut_shape = idb_via->get_cut_layer_shape();
    /// idb via top, checking target eco via bottom
    if (cut_shape.get_layer()->get_order() + 2 == cut_layer_order) {
      bool is_intersected = false;

      auto top_shape = idb_via->get_top_layer_shape();
      for (auto* rect : top_shape.get_rect_list()) {
        if (eco_via.intersectedBottom(rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y())) {
          is_intersected = true;
          break;
        }
      }

      if (is_intersected) {
        for (auto* rect : top_shape.get_rect_list()) {
          /// add all rects of this layer to connected data
          eco_via.addConnectedBottom(rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y());
        }
      }
    }

    /// idb via bottom, , checking target eco via top
    if (cut_shape.get_layer()->get_order() - 2 == cut_layer_order) {
      bool is_intersected = false;

      auto bottom_shape = idb_via->get_bottom_layer_shape();
      for (auto* rect : bottom_shape.get_rect_list()) {
        if (eco_via.intersectedTop(rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y())) {
          is_intersected = true;
          break;
        }
      }

      if (is_intersected) {
        for (auto* rect : bottom_shape.get_rect_list()) {
          /// add all rects of this layer to connected data
          eco_via.addConnectedTop(rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y());
        }
      }
    }
  }
}

void ECOViaInit::init_pin(EcoDataVia& eco_via, idb::IdbNet* idb_net, int cut_layer_order)
{
  for (auto* idb_pin : idb_net->get_instance_pin_list()->get_pin_list()) {
    for (auto* layer_shape : idb_pin->get_port_box_list()) {
      if (layer_shape->is_via()) {
        /// do not check, tbd
      } else {
        /// is bottom
        if (cut_layer_order - 1 == layer_shape->get_layer()->get_order()) {
          bool is_intersected = false;
          for (auto* rect : layer_shape->get_rect_list()) {
            if (eco_via.intersectedBottom(rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y())) {
              is_intersected = true;
              break;
            }
          }

          if (is_intersected) {
            for (auto* rect : layer_shape->get_rect_list()) {
              /// add all rects of this layer to connected data
              eco_via.addConnectedBottom(rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y());
            }
            eco_via.set_bottom_connected_pin(true);
          }
        }
        /// is top
        if (cut_layer_order + 1 == layer_shape->get_layer()->get_order()) {
          bool is_intersected = false;
          for (auto* rect : layer_shape->get_rect_list()) {
            if (eco_via.intersectedTop(rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y())) {
              is_intersected = true;
              break;
            }
          }

          if (is_intersected) {
            for (auto* rect : layer_shape->get_rect_list()) {
              /// add all rects of this layer to connected data
              eco_via.addConnectedTop(rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y());
            }
            eco_via.set_top_connected_pin(true);
          }
        }
      }
    }
  }
}

void ECOViaInit::init_nets()
{
  auto& eco_data = _data_manager->get_eco_data();

  auto idb_design = dmInst->get_idb_design();

  omp_lock_t lck;
  omp_init_lock(&lck);
#pragma omp parallel for schedule(dynamic)
  for (auto* idb_net : idb_design->get_net_list()->get_net_list()) {
    if (idb_net->get_segment_num() == 0) {
      continue;
    }

    // int : layer order vias
    std::map<int, std::vector<EcoDataVia>> eco_vias = get_net_vias(idb_net);
    /// init top and bottom connected wire
    for (auto& [cut_layer_order, via_list] : eco_vias) {
      for (auto& eco_via : via_list) {
        for (auto* idb_wire : idb_net->get_wire_list()->get_wire_list()) {
          for (auto* idb_segment : idb_wire->get_segment_list()) {
            init_segment_via(eco_via, idb_segment, cut_layer_order);
            init_segment_rect(eco_via, idb_segment, cut_layer_order);
          }
        }

        init_pin(eco_via, idb_net, cut_layer_order);
      }
    }

    /// add to via layer
    omp_set_lock(&lck);
    for (auto& [cut_layer_order, via_list] : eco_vias) {
      for (auto& eco_via : via_list) {
        auto& via_layer = eco_data.get_via_layer(eco_via.get_idb_via());
        via_layer.addVia(eco_via);
      }
    }
    omp_unset_lock(&lck);
  }

  omp_destroy_lock(&lck);
}

}  // namespace ieco