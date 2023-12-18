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

#include "engine_init_def.h"

#include "IdbGeometry.h"
#include "IdbLayer.h"
#include "IdbLayerShape.h"
#include "IdbNet.h"
#include "IdbRegularWire.h"
#include "IdbSpecialNet.h"
#include "IdbSpecialWire.h"
#include "idm.h"
#include "idrc_engine_manager.h"

namespace idrc {
/**
 *  top flow to init all def data to geometry data
 */
void DrcEngineInitDef::init()
{
  initDataFromIOPins();
  initDataFromInstances();
  initDataFromPDN();
  initDataFromNets();
}

/**
 * build geometry data from rect
 */
void DrcEngineInitDef::initDataFromRect(idb::IdbRect* rect, LayoutType type, int layer_id, int net_id)
{
  _engine_manager->addRect(rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y(), layer_id, net_id, type);
}

/**
 * build geometry data from idb layer shape, including layer id, net id, rectangles
 */
void DrcEngineInitDef::initDataFromShape(idb::IdbLayerShape* idb_shape, int net_id)
{
  if (idb_shape == nullptr) {
    return;
  }

  auto* idb_layout = dmInst->get_idb_layout();
  auto* idb_layers = idb_layout->get_layers();

  /// shape must be on the above of bottom routing layer
  if (false == idb_layers->is_pr_layer(idb_shape->get_layer())) {
    return;
  }

  int layer_id = idb_shape->get_layer()->get_id();
  LayoutType type = idb_shape->get_layer()->is_routing() ? LayoutType::kRouting : LayoutType::kCut;
  for (idb::IdbRect* rect : idb_shape->get_rect_list()) {
    initDataFromRect(rect, type, layer_id, net_id);
  }
}
/**
 * build geometry data from two points
 */
void DrcEngineInitDef::initDataFromPoints(idb::IdbCoordinate<int>* point_1, idb::IdbCoordinate<int>* point_2, int routing_width,
                                          int layer_id, int net_id, bool b_pdn)
{
  /// calculate rectangle by two points
  int llx, lly, urx, ury;
  int extend_size = b_pdn ? 0 : routing_width / 2;
  if (point_1->get_y() == point_2->get_y()) {
    // horizontal
    llx = std::min(point_1->get_x(), point_2->get_x()) - extend_size;
    lly = point_1->get_y() - routing_width / 2;
    urx = std::max(point_1->get_x(), point_2->get_x()) + extend_size;
    ury = lly + routing_width;
  } else if (point_1->get_x() == point_2->get_x()) {
    // vertical
    llx = point_1->get_x() - routing_width / 2;
    lly = std::min(point_1->get_y(), point_2->get_y()) - extend_size;
    urx = llx + routing_width;
    ury = std::max(point_1->get_y(), point_2->get_y()) + extend_size;
  }

  _engine_manager->addRect(llx, lly, urx, ury, layer_id, net_id, LayoutType::kRouting);
}

/**
 * build geometry data from pin
 */
void DrcEngineInitDef::initDataFromPin(idb::IdbPin* idb_pin)
{
  int net_id = idb_pin->get_net() == nullptr ? NET_ID_ENVIRONMENT : idb_pin->get_net()->get_id();
  for (IdbLayerShape* layer_shape : idb_pin->get_port_box_list()) {
    initDataFromShape(layer_shape, net_id);
  }
}
/**
 * build geometry data from via
 */
void DrcEngineInitDef::initDataFromVia(idb::IdbVia* idb_via, int net_id)
{
  /// cut
  auto cut_layer_shape = idb_via->get_cut_layer_shape();
  initDataFromShape(&cut_layer_shape, net_id);

  /// bottom
  auto bottom_layer_shape = idb_via->get_bottom_layer_shape();
  initDataFromShape(&bottom_layer_shape, net_id);

  /// top
  auto top_layer_shape = idb_via->get_top_layer_shape();
  initDataFromShape(&top_layer_shape, net_id);
}

void DrcEngineInitDef::initDataFromIOPins()
{
  auto* idb_design = dmInst->get_idb_design();
  auto* idb_io_pins = idb_design->get_io_pin_list();

  for (auto* idb_io_pin : idb_io_pins->get_pin_list()) {
    if (idb_io_pin != nullptr && idb_io_pin->get_term()->is_placed()) {
      initDataFromPin(idb_io_pin);
    }
  }
}

void DrcEngineInitDef::initDataFromInstances()
{
  std::cout << "idrc : begin init data from instances" << std::endl;
  ieda::Stats stats;

  auto* idb_design = dmInst->get_idb_design();

  uint64_t number = 0;
  for (IdbInstance* idb_inst : idb_design->get_instance_list()->get_instance_list()) {
    if (idb_inst == nullptr || idb_inst->get_cell_master() == nullptr) {
      continue;
    }
    /// instance pins
    for (auto* idb_pin : idb_inst->get_pin_list()->get_pin_list()) {
      initDataFromPin(idb_pin);
    }

    /// obs
    for (auto* idb_obs : idb_inst->get_obs_box_list()) {
      initDataFromShape(idb_obs, NET_ID_OBS);
    }

    number++;
  }

  std::cout << "idrc : end init data from instances, instance number = " << number << " runtime = " << stats.elapsedRunTime()
            << " memory = " << stats.memoryDelta() << std::endl;
}

void DrcEngineInitDef::initDataFromPDN()
{
  std::cout << "idrc : begin init data from pdn" << std::endl;
  ieda::Stats stats;

  auto* idb_design = dmInst->get_idb_design();

  uint64_t number = 0;
  for (auto* idb_special_net : idb_design->get_special_net_list()->get_net_list()) {
    for (auto* idb_special_wire : idb_special_net->get_wire_list()->get_wire_list()) {
      for (auto* idb_segment : idb_special_wire->get_segment_list()) {
        /// add wire
        if (idb_segment->get_point_list().size() >= 2) {
          /// get routing width
          auto* routing_layer = dynamic_cast<IdbLayerRouting*>(idb_segment->get_layer());
          int32_t routing_width = idb_segment->get_route_width() == 0 ? routing_layer->get_width() : idb_segment->get_route_width();

          /// calculate rectangle by two points
          auto* point_1 = idb_segment->get_point_start();
          auto* point_2 = idb_segment->get_point_second();

          initDataFromPoints(point_1, point_2, routing_width, routing_layer->get_id(), NET_ID_ENVIRONMENT, true);
        }

        /// vias
        if (idb_segment->is_via()) {
          /// add via
          initDataFromVia(idb_segment->get_via(), NET_ID_ENVIRONMENT);
        }

        number++;
      }
    }
  }

  std::cout << "idrc : end init data from pdn, segment number = " << number << " runtime = " << stats.elapsedRunTime()
            << " memory = " << stats.memoryDelta() << std::endl;
}
/**
 * init
 */
void DrcEngineInitDef::initDataFromNet(idb::IdbNet* idb_net)
{
  for (auto* idb_wire : idb_net->get_wire_list()->get_wire_list()) {
    for (auto* idb_segment : idb_wire->get_segment_list()) {
      if (idb_segment->get_point_number() >= 2) {
        /// get routing width
        auto* routing_layer = dynamic_cast<IdbLayerRouting*>(idb_segment->get_layer());
        int32_t routing_width = routing_layer->get_width();

        /// calculate rectangle by two points
        auto* point_1 = idb_segment->get_point_start();
        auto* point_2 = idb_segment->get_point_second();

        initDataFromPoints(point_1, point_2, routing_width, routing_layer->get_id(), idb_net->get_id());
      } else {
        /// via
        if (idb_segment->is_via()) {
          for (auto* idb_via : idb_segment->get_via_list()) {
            initDataFromVia(idb_via, idb_net->get_id());
          }
        }
        /// patch
        if (idb_segment->is_rect()) {
          IdbCoordinate<int32_t>* coordinate = idb_segment->get_point_start();
          IdbRect* rect_delta = idb_segment->get_delta_rect();
          IdbRect* rect = new IdbRect(rect_delta);
          rect->moveByStep(coordinate->get_x(), coordinate->get_y());

          auto* routing_layer = dynamic_cast<IdbLayerRouting*>(idb_segment->get_layer());

          initDataFromRect(rect, LayoutType::kRouting, routing_layer->get_id(), idb_net->get_id());

          delete rect;
        }
      }
    }
  }
}
/**
 * the basic geometry unit is construct independently by layer id and net id,
 * so it enable to read net parallelly
 */
void DrcEngineInitDef::initDataFromNets()
{
  std::cout << "idrc : begin init data from nets" << std::endl;

  ieda::Stats stats;

  auto* idb_design = dmInst->get_idb_design();

  for (auto* idb_net : idb_design->get_net_list()->get_net_list()) {
    initDataFromNet(idb_net);
  }

  std::cout << "idrc : end init data from nets, net number = " << idb_design->get_net_list()->get_num()
            << " runtime = " << stats.elapsedRunTime() << " memory = " << stats.memoryDelta() << std::endl;
}

}  // namespace idrc