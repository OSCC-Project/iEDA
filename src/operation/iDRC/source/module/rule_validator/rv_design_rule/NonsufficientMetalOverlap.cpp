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

  std::map<int32_t, std::map<int32_t, std::vector<PlanarRect>>> routing_net_rect_map;
  for (DRCShape* drc_shape : rv_box.get_drc_env_shape_list()) {
    if (!drc_shape->get_is_routing() || drc_shape->get_net_idx() == -1) {
      continue;
    }
    routing_net_rect_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()].push_back(drc_shape->get_rect());
  }
  for (DRCShape* drc_shape : rv_box.get_drc_result_shape_list()) {
    if (!drc_shape->get_is_routing()) {
      continue;
    }
    routing_net_rect_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()].push_back(drc_shape->get_rect());
  }
  for (auto& [routing_layer_idx, net_rect_map] : routing_net_rect_map) {
    RoutingLayer& routing_layer = routing_layer_list[routing_layer_idx];
    int32_t min_width = routing_layer.get_nonsufficient_metal_overlap_rule().min_width;
    int32_t half_width = min_width / 2;
    for (auto& [net_idx, rect_list] : net_rect_map) {
      std::map<int32_t, GTLPolySetInt> scale_gtl_poly_set_map;
      for (PlanarRect& rect : rect_list) {
        PlanarCoord mid_coord = rect.getMidPoint();
        if (routing_layer.isPreferH()) {
          scale_gtl_poly_set_map[mid_coord.get_y()] += DRCUTIL.convertToGTLRectInt(rect);
        } else {
          scale_gtl_poly_set_map[mid_coord.get_x()] += DRCUTIL.convertToGTLRectInt(rect);
        }
      }
      rect_list.clear();
      for (auto& [scale, gtl_poly_set] : scale_gtl_poly_set_map) {
        std::vector<GTLRectInt> gtl_rect_list;
        gtl::get_max_rectangles(gtl_rect_list, gtl_poly_set);
        for (GTLRectInt& gtl_rect : gtl_rect_list) {
          rect_list.push_back(DRCUTIL.convertToPlanarRect(gtl_rect));
        }
      }
      std::map<int32_t, GTLPolySetInt> rect_overlap_gtl_poly_set_map;
      for (size_t i = 0; i < rect_list.size(); i++) {
        for (size_t j = i + 1; j < rect_list.size(); j++) {
          if (!DRCUTIL.isClosedOverlap(rect_list[i], rect_list[j])) {
            continue;
          }
          GTLRectInt gtl_rect = DRCUTIL.convertToGTLRectInt(DRCUTIL.getEnlargedRect(DRCUTIL.getOverlap(rect_list[i], rect_list[j]), 1));
          rect_overlap_gtl_poly_set_map[static_cast<int32_t>(i)] += gtl_rect;
          rect_overlap_gtl_poly_set_map[static_cast<int32_t>(j)] += gtl_rect;
        }
      }
      std::vector<PlanarRect> overlap_rect_list;
      for (auto& [rect_idx, overlap_gtl_poly_set] : rect_overlap_gtl_poly_set_map) {
        std::vector<GTLPolyInt> gtl_poly_list;
        overlap_gtl_poly_set.get_polygons(gtl_poly_list);
        for (GTLPolyInt& gtl_poly : gtl_poly_list) {
          GTLRectInt gtl_rect;
          gtl::extents(gtl_rect, gtl_poly);
          PlanarRect overlap_rect = DRCUTIL.convertToPlanarRect(gtl_rect);
          if (!DRCUTIL.hasShrinkedRect(overlap_rect, 1)) {
            continue;
          }
          overlap_rect = DRCUTIL.getShrinkedRect(overlap_rect, 1);
          if (rect_list[rect_idx].getArea() <= overlap_rect.getArea()) {
            continue;
          }
          double diag_length = std::hypot(overlap_rect.getXSpan(), overlap_rect.getYSpan());
          if (diag_length >= min_width) {
            continue;
          }
          overlap_rect_list.push_back(overlap_rect);
        }
      }
      for (PlanarRect& overlap_rect : overlap_rect_list) {
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
