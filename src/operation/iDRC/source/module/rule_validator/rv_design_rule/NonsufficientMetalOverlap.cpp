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

void RuleValidator::verifyNonsufficientMetalOverlap(RVCluster& rv_cluster)
{
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();

  std::map<int32_t, std::map<int32_t, std::vector<PlanarRect>>> routing_net_rect_map;
  for (DRCShape* drc_shape : rv_cluster.get_drc_env_shape_list()) {
    if (!drc_shape->get_is_routing()) {
      continue;
    }
    routing_net_rect_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()].push_back(drc_shape->get_rect());
  }
  for (DRCShape* drc_shape : rv_cluster.get_drc_result_shape_list()) {
    if (!drc_shape->get_is_routing()) {
      continue;
    }
    routing_net_rect_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()].push_back(drc_shape->get_rect());
  }
  for (auto& [routing_layer_idx, net_rect_map] : routing_net_rect_map) {
    for (auto& [net_idx, rect_list] : net_rect_map) {
      GTLPolySetInt gtl_poly_set;
      for (PlanarRect& rect : rect_list) {
        gtl_poly_set += DRCUTIL.convertToGTLRectInt(rect);
      }
      rect_list.clear();
      std::vector<GTLRectInt> gtl_rect_list;
      gtl::get_max_rectangles(gtl_rect_list, gtl_poly_set);
      for (GTLRectInt& gtl_rect : gtl_rect_list) {
        rect_list.push_back(DRCUTIL.convertToPlanarRect(gtl_rect));
      }
    }
  }
  std::map<int32_t, std::map<int32_t, bgi::rtree<BGRectInt, bgi::quadratic<16>>>> routing_net_bg_rtree_map;
  for (auto& [routing_layer_idx, net_rect_map] : routing_net_rect_map) {
    for (auto& [net_idx, rect_list] : net_rect_map) {
      for (PlanarRect& rect : rect_list) {
        routing_net_bg_rtree_map[routing_layer_idx][net_idx].insert(DRCUTIL.convertToBGRectInt(rect));
      }
    }
  }
  for (auto& [routing_layer_idx, net_rect_map] : routing_net_rect_map) {
    RoutingLayer& routing_layer = routing_layer_list[routing_layer_idx];
    int32_t min_width = routing_layer.get_minimum_width_rule().min_width;
    int32_t half_width = min_width / 2;
    for (auto& [net_idx, rect_list] : net_rect_map) {
      for (PlanarRect& rect : rect_list) {
        std::vector<BGRectInt> bg_rect_list;
        {
          PlanarRect check_rect = rect;
          routing_net_bg_rtree_map[routing_layer_idx][net_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(check_rect)), std::back_inserter(bg_rect_list));
        }
        for (auto& bg_env_rect : bg_rect_list) {
          PlanarRect env_rect = DRCUTIL.convertToPlanarRect(bg_env_rect);
          if (!DRCUTIL.isClosedOverlap(rect, env_rect)) {
            continue;
          }
          if (env_rect == rect) {
            continue;
          }
          PlanarRect overlap_rect = DRCUTIL.getOverlap(rect, env_rect);
          double diag_length = std::hypot(overlap_rect.getXSpan(), overlap_rect.getYSpan());
          if (diag_length >= min_width) {
            continue;
          }
          if (overlap_rect.get_ll() == overlap_rect.get_ur()) {
            continue;
          }
          std::vector<BGRectInt> overlap_rect_env_list;
          {
            PlanarRect check_rect = overlap_rect;
            routing_net_bg_rtree_map[routing_layer_idx][net_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(check_rect)),
                                                                       std::back_inserter(overlap_rect_env_list));
          }
          bool is_inside = false;
          for (auto& overlap_rect_env : overlap_rect_env_list) {
            PlanarRect thirdrect = DRCUTIL.convertToPlanarRect(overlap_rect_env);
            if ((DRCUTIL.isInside(thirdrect, overlap_rect) && DRCUTIL.isOpenOverlap(thirdrect, overlap_rect)) && thirdrect != rect && thirdrect != env_rect
                && thirdrect.getWidth() >= min_width && DRCUTIL.getOverlap(thirdrect, rect) != overlap_rect
                && DRCUTIL.getOverlap(thirdrect, env_rect) != overlap_rect) {
              is_inside = true;
              break;
            }
          }
          if (is_inside) {
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
          rv_cluster.get_violation_list().push_back(violation);
        }
      }
    }
  }
}

}  // namespace idrc
