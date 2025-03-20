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
  std::vector<Violation>& violation_list = rv_box.get_violation_list();

  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> layer_net_gtl_poly_set;
  for (DRCShape* rect : rv_box.get_drc_env_shape_list()) {
    if (!rect->get_is_routing()) {
      continue;
    }
    if (rect->get_net_idx() == -1) {
      continue;
    }
    layer_net_gtl_poly_set[rect->get_layer_idx()][rect->get_net_idx()] += GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y());
  }
  for (DRCShape* rect : rv_box.get_drc_result_shape_list()) {
    if (!rect->get_is_routing()) {
      continue;
    }
    layer_net_gtl_poly_set[rect->get_layer_idx()][rect->get_net_idx()] += GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y());
  }
  for (auto& [layer_idx, net_gtl_poly_set] : layer_net_gtl_poly_set) {
    int32_t min_area = routing_layer_list[layer_idx].get_min_area();
    for (auto& [net_idx, gtl_poly_set] : net_gtl_poly_set) {
      std::vector<GTLPolyInt> gtl_poly_list;
      gtl_poly_set.get_polygons(gtl_poly_list);
      for (GTLPolyInt& gtl_poly : gtl_poly_list) {
        if (gtl::area(gtl_poly) < min_area) {
          std::vector<GTLRectInt> gtl_rect_list;
          gtl::get_rectangles(gtl_rect_list, gtl_poly, gtl::HORIZONTAL);

          GTLRectInt best_gtl_rect;
          int32_t max_area = 0;
          for (GTLRectInt& gtl_rect : gtl_rect_list) {
            int32_t curr_area = static_cast<int32_t>(gtl::area(gtl_rect));
            if (curr_area > max_area) {
              max_area = curr_area;
              best_gtl_rect = gtl_rect;
            }
          }
          Violation violation;
          violation.set_violation_type(ViolationType::kMinimumArea);
          violation.set_is_routing(true);
          violation.set_violation_net_set({net_idx});
          violation.set_required_size(min_area);
          violation.set_layer_idx(layer_idx);
          violation.set_rect(PlanarRect(gtl::xl(best_gtl_rect), gtl::yl(best_gtl_rect), gtl::xh(best_gtl_rect), gtl::yh(best_gtl_rect)));
          violation_list.push_back(violation);
        }
      }
    }
  }
}

}  // namespace idrc
