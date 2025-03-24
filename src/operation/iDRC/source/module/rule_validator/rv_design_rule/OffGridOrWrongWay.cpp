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
#include "RuleValidator.hpp"

namespace idrc {

void RuleValidator::verifyOffGridOrWrongWay(RVBox& rv_box)
{
  int32_t manufacture_grid = DRCDM.getDatabase().get_manufacture_grid();

  std::map<int32_t, std::map<int32_t, std::vector<PlanarRect>>> routing_net_rect_list_map;
  for (DRCShape* drc_shape : rv_box.get_drc_env_shape_list()) {
    if (!drc_shape->get_is_routing() || drc_shape->get_net_idx() == -1) {
      continue;
    }
    routing_net_rect_list_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()].push_back(drc_shape->get_rect());
  }
  for (DRCShape* drc_shape : rv_box.get_drc_result_shape_list()) {
    if (!drc_shape->get_is_routing()) {
      continue;
    }
    routing_net_rect_list_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()].push_back(drc_shape->get_rect());
  }
  for (auto& [routing_layer_idx, net_rect_list_map] : routing_net_rect_list_map) {
    for (auto& [net_idx, rect_list] : net_rect_list_map) {
      for (PlanarRect& rect : rect_list) {
        if (rect.get_ll_x() % manufacture_grid == 0 && rect.get_ll_y() % manufacture_grid == 0 && rect.get_ur_x() % manufacture_grid == 0
            && rect.get_ur_y() % manufacture_grid == 0) {
          continue;
        }
        Violation violation;
        violation.set_violation_type(ViolationType::kOffGridOrWrongWay);
        violation.set_is_routing(true);
        violation.set_violation_net_set({net_idx});
        violation.set_required_size(0);
        violation.set_layer_idx(routing_layer_idx);
        violation.set_rect(rect);
        rv_box.get_violation_list().push_back(violation);
      }
    }
  }
}

}  // namespace idrc
