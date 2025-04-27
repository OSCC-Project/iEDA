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

void RuleValidator::verifyMetalShort(RVBox& rv_box)
{
  std::map<int32_t, std::map<int32_t, std::vector<PlanarRect>>> routing_net_rect_map;
  for (DRCShape* drc_shape : rv_box.get_drc_env_shape_list()) {
    if (!drc_shape->get_is_routing()) {
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
  std::map<int32_t, std::map<int32_t, PlanarRect>> routing_net_bbox_map;
  for (auto& [routing_layer_idx, net_rect_map] : routing_net_rect_map) {
    for (auto& [net_idx, rect_list] : net_rect_map) {
      routing_net_bbox_map[routing_layer_idx][net_idx] = DRCUTIL.getBoundingBox(rect_list);
    }
  }
  for (auto& [routing_layer_idx, net_rect_map] : routing_net_rect_map) {
    std::vector<std::pair<int32_t, std::vector<PlanarRect>>> net_rect_pair_list;
    for (auto& [net_idx, rect_list] : net_rect_map) {
      net_rect_pair_list.emplace_back(net_idx, rect_list);
    }
    std::map<int32_t, PlanarRect>& net_bbox_map = routing_net_bbox_map[routing_layer_idx];
    for (size_t i = 0; i < net_rect_pair_list.size(); i++) {
      for (size_t j = i + 1; j < net_rect_pair_list.size(); j++) {
        int32_t pre_net_idx = net_rect_pair_list[i].first;
        int32_t post_net_idx = net_rect_pair_list[j].first;
        std::vector<PlanarRect>& pre_rect_list = net_rect_pair_list[i].second;
        std::vector<PlanarRect>& post_rect_list = net_rect_pair_list[j].second;
        if (!DRCUTIL.isClosedOverlap(net_bbox_map[pre_net_idx], net_bbox_map[post_net_idx])) {
          continue;
        }
        GTLPolySetInt gtl_poly_set;
        for (PlanarRect& overlap_rect : DRCUTIL.getOverlap(pre_rect_list, post_rect_list)) {
          gtl_poly_set += DRCUTIL.convertToGTLRectInt(DRCUTIL.getEnlargedRect(overlap_rect, 1));
        }
        std::vector<GTLRectInt> gtl_rect_list;
        gtl::get_max_rectangles(gtl_rect_list, gtl_poly_set);
        for (GTLRectInt& gtl_rect : gtl_rect_list) {
          PlanarRect violation_rect = DRCUTIL.convertToPlanarRect(gtl_rect);
          if (!DRCUTIL.hasShrinkedRect(violation_rect, 1)) {
            continue;
          }
          violation_rect = DRCUTIL.getShrinkedRect(violation_rect, 1);
          Violation violation;
          violation.set_violation_type(ViolationType::kMetalShort);
          violation.set_required_size(0);
          violation.set_is_routing(true);
          violation.set_violation_net_set({pre_net_idx, post_net_idx});
          violation.set_layer_idx(routing_layer_idx);
          violation.set_rect(violation_rect);
          rv_box.get_violation_list().push_back(violation);
        }
      }
    }
  }
}

}  // namespace idrc
