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

void RuleValidator::verifyMinimumWidth(RVCluster& rv_cluster)
{
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  const auto& layer_data = rv_cluster.get_layer_data();

  for (const auto& [routing_layer_idx, rv_layer_data] : layer_data) {
    int32_t min_width = routing_layer_list[routing_layer_idx].get_minimum_width_rule().min_width;
    for (const auto& [net_idx, routing_net] : rv_layer_data.nets) {
      if (net_idx == -1) {
        continue;
      }
      GTLPolySetInt violation_gtl_poly_set;
      for (const PolygonData& polygon_data : rv_layer_data.getPolygons(routing_net)) {
        if (polygon_data.isEnv) {
          continue;
        }
        GTLPolySetInt polygon_poly_set;
        polygon_poly_set += polygon_data.hole_poly;
        for (gtl::orientation_2d_enum gtl_orient : {gtl::HORIZONTAL, gtl::VERTICAL}) {
          std::vector<GTLRectInt> gtl_rect_list;
          gtl::get_rectangles(gtl_rect_list, polygon_poly_set, gtl_orient);
          for (GTLRectInt& gtl_rect : gtl_rect_list) {
            int32_t span = 0;
            if (gtl_orient == gtl::HORIZONTAL) {
              span = gtl::xh(gtl_rect) - gtl::xl(gtl_rect);
            } else {
              span = gtl::yh(gtl_rect) - gtl::yl(gtl_rect);
            }
            if (min_width <= span) {
              continue;
            }
            violation_gtl_poly_set += gtl_rect;
          }
        }
      }
      std::vector<GTLRectInt> gtl_rect_list;
      gtl::get_max_rectangles(gtl_rect_list, violation_gtl_poly_set);
      for (GTLRectInt& gtl_rect : gtl_rect_list) {
        Violation violation;
        violation.set_violation_type(ViolationType::kMinimumWidth);
        violation.set_is_routing(true);
        violation.set_violation_net_set({net_idx});
        violation.set_required_size(min_width);
        violation.set_layer_idx(routing_layer_idx);
        violation.set_rect(DRCUTIL.convertToPlanarRect(gtl_rect));
        rv_cluster.get_violation_list().push_back(violation);
      }
    }
  }
}

}  // namespace idrc
