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

#include "engine_init_rt.h"

#include "IdbGeometry.h"
#include "IdbLayer.h"
#include "IdbLayerShape.h"
#include "IdbNet.h"
#include "IdbRegularWire.h"
#include "IdbSpecialNet.h"
#include "IdbSpecialWire.h"
#include "idm.h"
#include "idrc_dm.h"
#include "idrc_engine_manager.h"

namespace idrc {

/**
 * init engine data from RT data
 */
void DrcEngineInitRT::init()
{
  for (auto& [net_id, idb_segments] : *(_data_manager->get_routing_data())) {
    for (auto& idb_segment : idb_segments) {
      /// add wire
      if (idb_segment->get_point_list().size() >= 2) {
        /// get routing width
        auto* routing_layer = dynamic_cast<IdbLayerRouting*>(idb_segment->get_layer());
        int32_t routing_width = routing_layer->get_width();

        /// calculate rectangle by two points
        auto* point_1 = idb_segment->get_point_start();
        auto* point_2 = idb_segment->get_point_second();

        initDataFromPoints(point_1, point_2, routing_width, idb_segment->get_layer(), net_id);
      }

      /// vias
      if (idb_segment->is_via()) {
        /// add via
        for (auto& via : idb_segment->get_via_list()) {
          initDataFromVia(via, net_id);
        }
      }

      /// rects
      if (idb_segment->is_rect()) {
        // add rect
        auto type = idb_segment->get_layer()->is_routing() ? LayoutType::kRouting : LayoutType::kCut;
        initDataFromRect(idb_segment->get_delta_rect(), type, idb_segment->get_layer(), net_id);
      }
    }
  }
  for (auto& [net_id, pin_shape_list] : *(_data_manager->get_pin_data())) {
    for (auto& pin_shape : pin_shape_list) {
      for (auto& rect : pin_shape->get_rect_list()) {
        auto type = pin_shape->get_layer()->is_routing() ? LayoutType::kRouting : LayoutType::kCut;
        initDataFromRect(rect, type, pin_shape->get_layer(), net_id);
      }
    }
  }
  for (auto& env_shape : *(_data_manager->get_env_shapes())) {
    for (auto& rect : env_shape->get_rect_list()) {
      auto type = env_shape->get_layer()->is_routing() ? LayoutType::kRouting : LayoutType::kCut;
      initDataFromRect(rect, type, env_shape->get_layer());
    }
  }
}

}  // namespace idrc