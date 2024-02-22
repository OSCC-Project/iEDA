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

#include "engine_init.h"

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
 * build geometry data from rect
 */
void DrcEngineInit::initDataFromRect(idb::IdbRect* rect, LayoutType type, idb::IdbLayer* layer, int net_id)
{
  _engine_manager->addRect(rect->get_low_x(), rect->get_low_y(), rect->get_high_x(), rect->get_high_y(), layer, net_id, type);
}

/**
 * build geometry data from idb layer shape, including layer id, net id, rectangles
 */
void DrcEngineInit::initDataFromShape(idb::IdbLayerShape* idb_shape, int net_id)
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

  auto* layer = idb_shape->get_layer();
  LayoutType type = idb_shape->get_layer()->is_routing() ? LayoutType::kRouting : LayoutType::kCut;
  for (idb::IdbRect* rect : idb_shape->get_rect_list()) {
    initDataFromRect(rect, type, layer, net_id);
  }
}
/**
 * build geometry data from two points
 */
void DrcEngineInit::initDataFromPoints(idb::IdbCoordinate<int>* point_1, idb::IdbCoordinate<int>* point_2, int routing_width,
                                       idb::IdbLayer* layer, int net_id, bool b_pdn)
{
  /// calculate rectangle by two points
  int llx = 0, lly = 0, urx = 0, ury = 0;
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

  _engine_manager->addRect(llx, lly, urx, ury, layer, net_id, LayoutType::kRouting);
}

/**
 * build geometry data from pin
 */
void DrcEngineInit::initDataFromPin(idb::IdbPin* idb_pin, int default_id)
{
  int net_id = idb_pin->get_net() == nullptr ? default_id : idb_pin->get_net()->get_id();
  for (IdbLayerShape* layer_shape : idb_pin->get_port_box_list()) {
    initDataFromShape(layer_shape, net_id);
  }
}
/**
 * build geometry data from via
 */
void DrcEngineInit::initDataFromVia(idb::IdbVia* idb_via, int net_id)
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

/**
 * init
 */
void DrcEngineInit::initDataFromNet(idb::IdbNet* idb_net)
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

        initDataFromPoints(point_1, point_2, routing_width, idb_segment->get_layer(), idb_net->get_id());
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

          initDataFromRect(rect, LayoutType::kRouting, idb_segment->get_layer(), idb_net->get_id());

          delete rect;
        }
      }
    }
  }
}

}  // namespace idrc