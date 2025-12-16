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

void RuleValidator::verifyOffGridOrWrongWay(RVCluster& rv_cluster)
{
  int32_t manufacture_grid = DRCDM.getDatabase().get_off_grid_or_wrong_way_rule().manufacture_grid;
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = DRCDM.getDatabase().get_cut_to_adjacent_routing_map();

  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> routing_net_gtl_poly_set_map;
  for (DRCShape* drc_shape : rv_cluster.get_drc_result_shape_list()) {
    int32_t routing_layer_idx = -1;
    if (!drc_shape->get_is_routing()) {
      std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[drc_shape->get_layer_idx()];
      routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
    } else {
      routing_layer_idx = drc_shape->get_layer_idx();
    }
    routing_net_gtl_poly_set_map[routing_layer_idx][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
  }
  for (auto& [routing_layer_idx, net_gtl_poly_set_map] : routing_net_gtl_poly_set_map) {
    for (auto& [net_idx, gtl_poly_set] : net_gtl_poly_set_map) {
      std::vector<GTLPolyInt> gtl_poly_list;
      gtl_poly_set.get_polygons(gtl_poly_list);
      for (GTLPolyInt& gtl_poly : gtl_poly_list) {
        std::vector<PlanarCoord> coord_list;
        for (const GTLPointInt& gtl_point : gtl_poly) {
          coord_list.emplace_back(gtl_point.x(), gtl_point.y());
        }
        coord_list.push_back(coord_list.front());
        for (size_t i = 1; i < coord_list.size(); i++) {
          PlanarRect rect = DRCUTIL.getRect(coord_list[i - 1], coord_list[i]);
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
          rv_cluster.get_violation_list().push_back(violation);
        }
      }
    }
  }
}

}  // namespace idrc
