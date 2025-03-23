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

void RuleValidator::verifyNonsufficientMetalOverlap(RVBox& rv_box)
{
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();

  std::map<int32_t, std::map<int32_t, std::vector<DRCShape*>>> routing_net_drc_shape_map;
  for (DRCShape* drc_shape : rv_box.get_drc_env_shape_list()) {
    if (!drc_shape->get_is_routing() || drc_shape->get_net_idx() == -1) {
      continue;
    }
    routing_net_drc_shape_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()].push_back(drc_shape);
  }
  for (DRCShape* drc_shape : rv_box.get_drc_result_shape_list()) {
    if (!drc_shape->get_is_routing()) {
      continue;
    }
    routing_net_drc_shape_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()].push_back(drc_shape);
  }
  for (auto& [routing_layer_idx, net_drc_shape_map] : routing_net_drc_shape_map) {
    RoutingLayer& routing_layer = routing_layer_list[routing_layer_idx];
    int32_t min_width = routing_layer.get_min_width();
    int32_t half_width = min_width / 2;
    for (auto& [net_idx, drc_shape_list] : net_drc_shape_map) {
      std::map<int32_t, GTLPolySetInt> scale_gtl_poly_set_map;
      for (DRCShape* drc_shape : drc_shape_list) {
        PlanarCoord mid_coord = drc_shape->getMidPoint();
        if (routing_layer.isPreferH()) {
          scale_gtl_poly_set_map[mid_coord.get_y()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
        } else {
          scale_gtl_poly_set_map[mid_coord.get_x()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
        }
      }
      std::vector<GTLPolySetInt> gtl_poly_set_list;
      for (auto& [scale, gtl_poly_set] : scale_gtl_poly_set_map) {
        gtl_poly_set_list.push_back(gtl_poly_set);
      }
      GTLPolySetInt overlap_gtl_poly_set;
      for (size_t i = 0; i < gtl_poly_set_list.size(); i++) {
        for (size_t j = i + 1; j < gtl_poly_set_list.size(); j++) {
          overlap_gtl_poly_set += (gtl_poly_set_list[i] & gtl_poly_set_list[j]);
        }
      }
      std::vector<GTLRectInt> overlap_gtl_rect_list;
      gtl::get_max_rectangles(overlap_gtl_rect_list, overlap_gtl_poly_set);
      for (GTLRectInt& overlap_gtl_rect : overlap_gtl_rect_list) {
        PlanarRect overlap_rect = DRCUTIL.convertToPlanarRect(overlap_gtl_rect);
        double diag_length = std::sqrt(std::pow(overlap_rect.getXSpan(), 2) + std::pow(overlap_rect.getYSpan(), 2));
        if (diag_length > min_width) {
          continue;
        }
        int32_t x_enlarge_size = 0;
        if (overlap_rect.getXSpan() < half_width) {
          x_enlarge_size = half_width - overlap_rect.getXSpan();
        }
        int32_t y_enlarge_size = 0;
        if (overlap_rect.getYSpan() < half_width) {
          y_enlarge_size = half_width - overlap_rect.getYSpan();
        }
        PlanarRect violation_rect = DRCUTIL.getEnlargedRect(overlap_rect, x_enlarge_size, y_enlarge_size, x_enlarge_size, y_enlarge_size);

        Violation violation;
        violation.set_violation_type(ViolationType::kNonsufficientMetalOverlap);
        violation.set_is_routing(true);
        violation.set_violation_net_set({net_idx});
        violation.set_required_size(0);
        violation.set_layer_idx(routing_layer_idx);
        violation.set_rect(violation_rect);
        rv_box.get_violation_list().push_back(violation);
      }
    }
  }
}

}  // namespace idrc
