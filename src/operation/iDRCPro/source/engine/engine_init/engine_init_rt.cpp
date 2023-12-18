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

#include "BaseRegion.hpp"
#include "BaseShape.hpp"
#include "idrc_dm.h"
#include "idrc_engine_manager.h"

namespace idrc {

/**
 * init engine data from RT data
 */
void DrcEngineInitRT::init()
{
  auto init_data_shape = [](irt::BaseShape* shape, DrcEngineManager* engine_manager) {
    int llx = shape->get_shape().min_corner().x();
    int lly = shape->get_shape().min_corner().y();
    int urx = shape->get_shape().max_corner().x();
    int ury = shape->get_shape().max_corner().y();
    int layer_id = shape->get_layer_idx();
    int net_id = shape->get_base_info().get_net_idx();
    LayoutType type = shape->get_is_routing() ? LayoutType::kRouting : LayoutType::kCut;
    engine_manager->addRect(llx, lly, urx, ury, layer_id, net_id, type);
  };

  // init environment data
  auto* region = _data_manager->get_region();
  if (region != nullptr) {
    // routing layers
    for (auto& [layer_id, bgi_rtree] : region->get_routing_region_map()) {
      for (auto& [rect, base_shape] : bgi_rtree) {
        init_data_shape(base_shape, _engine_manager);
      }
    }

    // cut layers
    for (auto& [layer_id, bgi_rtree] : region->get_cut_region_map()) {
      for (auto& [rect, base_shape] : bgi_rtree) {
        init_data_shape(base_shape, _engine_manager);
      }
    }
  }

  // init target shapes
  auto* target_shapes = _data_manager->get_target_shapes();
  if (target_shapes != nullptr) {
    for (auto& shape : *target_shapes) {
      init_data_shape(&shape, _engine_manager);
    }
  }
}

}  // namespace idrc