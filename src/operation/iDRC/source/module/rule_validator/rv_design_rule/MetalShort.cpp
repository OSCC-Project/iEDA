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

void RuleValidator::verifyMetalShort(RVCluster& rv_cluster)
{
  std::map<int32_t, std::map<int32_t, std::vector<PlanarRect>>> routing_net_rect_map;
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> routing_bg_rtree_map;
  for (DRCShape* drc_shape : rv_cluster.get_drc_env_shape_list()) {
    if (!drc_shape->get_is_routing()) {
      continue;
    }
    routing_net_rect_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()].push_back(drc_shape->get_rect());
    routing_bg_rtree_map[drc_shape->get_layer_idx()].insert(std::make_pair(DRCUTIL.convertToBGRectInt(drc_shape->get_rect()), drc_shape->get_net_idx()));
  }
  for (DRCShape* drc_shape : rv_cluster.get_drc_result_shape_list()) {
    if (!drc_shape->get_is_routing()) {
      continue;
    }
    routing_net_rect_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()].push_back(drc_shape->get_rect());
    routing_bg_rtree_map[drc_shape->get_layer_idx()].insert(std::make_pair(DRCUTIL.convertToBGRectInt(drc_shape->get_rect()), drc_shape->get_net_idx()));
  }
  for (auto& [routing_layer_idx, net_rect_map] : routing_net_rect_map) {
    std::map<std::set<int32_t>, std::vector<PlanarRect>> net_violation_rect_map;
    for (auto& [net_idx, rect_list] : net_rect_map) {
      for (PlanarRect& rect : rect_list) {
        std::vector<std::pair<BGRectInt, int32_t>> bg_rect_net_pair_list;
        {
          routing_bg_rtree_map[routing_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(rect)), std::back_inserter(bg_rect_net_pair_list));
        }
        for (auto& [bg_env_rect, env_net_idx] : bg_rect_net_pair_list) {
          if (net_idx == env_net_idx) {
            continue;
          }
          PlanarRect env_rect = DRCUTIL.convertToPlanarRect(bg_env_rect);
          if (!DRCUTIL.isClosedOverlap(rect, env_rect)) {
            continue;
          }
          net_violation_rect_map[{net_idx, env_net_idx}].push_back(DRCUTIL.getOverlap(rect, env_rect));
        }
      }
    }
    for (auto& [violation_net_set, violation_rect_list] : net_violation_rect_map) {
      GTLPolySetInt violation_poly_set;
      for (PlanarRect& violation_rect : violation_rect_list) {
        violation_poly_set += DRCUTIL.convertToGTLRectInt(DRCUTIL.getEnlargedRect(violation_rect, 1));
      }
      std::vector<GTLRectInt> gtl_rect_list;
      gtl::get_max_rectangles(gtl_rect_list, violation_poly_set);
      for (GTLRectInt& gtl_rect : gtl_rect_list) {
        PlanarRect violation_rect = DRCUTIL.convertToPlanarRect(gtl_rect);
        if (DRCUTIL.hasShrinkedRect(violation_rect, 1)) {
          violation_rect = DRCUTIL.getShrinkedRect(violation_rect, 1);
        }
        Violation violation;
        violation.set_violation_type(ViolationType::kMetalShort);
        violation.set_is_routing(true);
        violation.set_violation_net_set(violation_net_set);
        violation.set_layer_idx(routing_layer_idx);
        violation.set_rect(violation_rect);
        violation.set_required_size(0);
        rv_cluster.get_violation_list().push_back(violation);
      }
    }
  }
}

}  // namespace idrc
