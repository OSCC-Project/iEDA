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

void RuleValidator::verifyMinimumArea(RVBox& rv_box)
{
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();

  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> routing_net_gtl_poly_set_map;
  for (DRCShape* drc_shape : rv_box.get_drc_env_shape_list()) {
    if (!drc_shape->get_is_routing() || drc_shape->get_net_idx() == -1) {
      continue;
    }
    routing_net_gtl_poly_set_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
  }
  for (DRCShape* drc_shape : rv_box.get_drc_result_shape_list()) {
    if (!drc_shape->get_is_routing()) {
      continue;
    }
    routing_net_gtl_poly_set_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
  }
  for (auto& [routing_layer_idx, net_gtl_poly_set_map] : routing_net_gtl_poly_set_map) {
    int32_t min_area = routing_layer_list[routing_layer_idx].get_minimum_area_rule().min_area;
    for (auto& [net_idx, gtl_poly_set] : net_gtl_poly_set_map) {
      std::vector<GTLPolyInt> gtl_poly_list;
      gtl_poly_set.get_polygons(gtl_poly_list);
      for (GTLPolyInt& gtl_poly : gtl_poly_list) {
        if (gtl::area(gtl_poly) >= min_area) {
          continue;
        }
        std::vector<GTLRectInt> gtl_rect_list;
        gtl::get_rectangles(gtl_rect_list, gtl_poly, gtl::HORIZONTAL);

        GTLRectInt best_gtl_rect;
        int32_t max_x_span = 0;
        for (GTLRectInt& gtl_rect : gtl_rect_list) {
          int32_t curr_x_span = std::abs(gtl::xl(gtl_rect) - gtl::xh(gtl_rect));
          if (max_x_span <= curr_x_span) {
            max_x_span = curr_x_span;
            best_gtl_rect = gtl_rect;
          }
        }
        Violation violation;
        violation.set_violation_type(ViolationType::kMinimumArea);
        violation.set_is_routing(true);
        violation.set_violation_net_set({net_idx});
        violation.set_required_size(min_area);
        violation.set_layer_idx(routing_layer_idx);
        violation.set_rect(DRCUTIL.convertToPlanarRect(best_gtl_rect));
        rv_box.get_violation_list().push_back(violation);
      }
    }
  }
}

}  // namespace idrc
