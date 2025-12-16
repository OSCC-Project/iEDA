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

void RuleValidator::verifyParallelRunLengthSpacing(RVCluster& rv_cluster)
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
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> routing_bg_rtree_map;
  for (auto& [routing_layer_idx, net_rect_map] : routing_net_rect_map) {
    for (auto& [net_idx, rect_list] : net_rect_map) {
      for (PlanarRect& rect : rect_list) {
        routing_bg_rtree_map[routing_layer_idx].insert(std::make_pair(DRCUTIL.convertToBGRectInt(rect), net_idx));
      }
    }
  }
  for (auto& [routing_layer_idx, net_rect_map] : routing_net_rect_map) {
    RoutingLayer& routing_layer = routing_layer_list[routing_layer_idx];
    int32_t min_width = routing_layer.get_minimum_width_rule().min_width;
    ParallelRunLengthSpacingRule& parallel_run_length_spacing_rule = routing_layer.get_parallel_run_length_spacing_rule();
    std::map<std::set<int32_t>, std::map<int32_t, std::vector<PlanarRect>>> net_required_violation_rect_map;
    for (auto& [net_idx, rect_list] : net_rect_map) {
      for (PlanarRect& rect : rect_list) {
        if (rect.getWidth() < min_width) {
          continue;
        }
        std::vector<std::pair<BGRectInt, int32_t>> bg_rect_net_pair_list;
        {
          PlanarRect check_rect = DRCUTIL.getEnlargedRect(rect, parallel_run_length_spacing_rule.getMaxSpacing());
          routing_bg_rtree_map[routing_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(check_rect)), std::back_inserter(bg_rect_net_pair_list));
        }
        for (auto& [bg_env_rect, env_net_idx] : bg_rect_net_pair_list) {
          if (net_idx == -1 && env_net_idx == -1) {
            continue;
          }
          PlanarRect env_rect = DRCUTIL.convertToPlanarRect(bg_env_rect);
          if (env_rect.getWidth() < min_width) {
            continue;
          }
          if (DRCUTIL.isClosedOverlap(rect, env_rect)) {
            continue;
          }
          int32_t required_size
              = parallel_run_length_spacing_rule.getSpacing(std::max(rect.getWidth(), env_rect.getWidth()), DRCUTIL.getParallelLength(rect, env_rect));
          if (required_size <= DRCUTIL.getEuclideanDistance(rect, env_rect)) {
            continue;
          }
          PlanarRect violation_rect = DRCUTIL.getSpacingRect(rect, env_rect);
          std::vector<PlanarRect> violation_env_rect_list;
          {
            std::vector<std::pair<BGRectInt, int32_t>> violation_bg_rect_net_pair_list;
            routing_bg_rtree_map[routing_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(violation_rect)),
                                                          std::back_inserter(violation_bg_rect_net_pair_list));
            for (auto& [violation_bg_env_rect, violation_env_net_idx] : violation_bg_rect_net_pair_list) {
              violation_env_rect_list.push_back(DRCUTIL.convertToPlanarRect(violation_bg_env_rect));
            }
          }
          if (violation_rect.getArea() == 0) {
            bool is_valid = true;
            for (PlanarRect& violation_env_rect : violation_env_rect_list) {
              if (DRCUTIL.isInside(violation_env_rect, violation_rect) && DRCUTIL.isInside(violation_env_rect, violation_rect.getMidPoint(), false)) {
                is_valid = false;
                break;
              }
            }
            if (is_valid) {
              net_required_violation_rect_map[{net_idx, env_net_idx}][required_size].push_back(violation_rect);
            }
          } else {
            GTLPolySetInt violation_poly_set;
            for (PlanarRect& violation_env_rect : violation_env_rect_list) {
              violation_poly_set += DRCUTIL.convertToGTLRectInt(violation_env_rect);
            }
            GTLRectInt gtl_violation_rect;
            gtl::extents(gtl_violation_rect, DRCUTIL.convertToGTLRectInt(violation_rect) - violation_poly_set);
            violation_rect = DRCUTIL.convertToPlanarRect(gtl_violation_rect);
            net_required_violation_rect_map[{net_idx, env_net_idx}][required_size].push_back(violation_rect);
          }
        }
      }
    }
    for (auto& [violation_net_set, required_violation_rect_map] : net_required_violation_rect_map) {
      for (auto& [required_size, violation_rect_list] : required_violation_rect_map) {
        for (PlanarRect& violation_rect : violation_rect_list) {
          bool is_inside = false;
          for (PlanarRect& other_violation_rect : violation_rect_list) {
            if (other_violation_rect == violation_rect) {
              continue;
            }
            if (DRCUTIL.isInside(other_violation_rect, violation_rect)) {
              is_inside = true;
              break;
            }
          }
          if (!is_inside) {
            Violation violation;
            violation.set_violation_type(ViolationType::kParallelRunLengthSpacing);
            violation.set_is_routing(true);
            violation.set_violation_net_set(violation_net_set);
            violation.set_layer_idx(routing_layer_idx);
            violation.set_rect(violation_rect);
            violation.set_required_size(required_size);
            rv_cluster.get_violation_list().push_back(violation);
          }
        }
      }
    }
  }
}

}  // namespace idrc
