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

void RuleValidator::verifyDifferentLayerCutSpacing(RVBox& rv_box)
{
  std::vector<CutLayer>& cut_layer_list = DRCDM.getDatabase().get_cut_layer_list();
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = DRCDM.getDatabase().get_cut_to_adjacent_routing_map();

  std::map<int32_t, std::map<int32_t, std::vector<PlanarRect>>> cut_net_rect_map;
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> cut_bg_rtree_map;
  for (DRCShape* drc_shape : rv_box.get_drc_env_shape_list()) {
    if (drc_shape->get_is_routing()) {
      continue;
    }
    cut_net_rect_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()].push_back(drc_shape->get_rect());
    cut_bg_rtree_map[drc_shape->get_layer_idx()].insert(std::make_pair(DRCUTIL.convertToBGRectInt(drc_shape->get_rect()), drc_shape->get_net_idx()));
  }
  for (DRCShape* drc_shape : rv_box.get_drc_result_shape_list()) {
    if (drc_shape->get_is_routing()) {
      continue;
    }
    cut_net_rect_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()].push_back(drc_shape->get_rect());
    cut_bg_rtree_map[drc_shape->get_layer_idx()].insert(std::make_pair(DRCUTIL.convertToBGRectInt(drc_shape->get_rect()), drc_shape->get_net_idx()));
  }
  for (auto& [cut_layer_idx, net_rect_map] : cut_net_rect_map) {
    int32_t below_cut_layer_idx = cut_layer_idx - 1;
    if (below_cut_layer_idx < 0) {
      continue;
    }
    int32_t routing_layer_idx = -1;
    {
      std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[below_cut_layer_idx];
      routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
    }
    CutLayer& cut_layer = cut_layer_list[cut_layer_idx];
    DifferentLayerCutSpacingRule& different_layer_cut_spacing_rule = cut_layer.get_different_layer_cut_spacing_rule();
    int32_t curr_spacing = different_layer_cut_spacing_rule.below_spacing;
    int32_t curr_prl_spacing = different_layer_cut_spacing_rule.below_prl_spacing;
    int32_t curr_prl = -1 * different_layer_cut_spacing_rule.below_prl;
    for (auto& [net_idx, rect_list] : net_rect_map) {
      for (PlanarRect& rect : rect_list) {
        GTLPolySetInt curr_spacing_poly_set;
        curr_spacing_poly_set += DRCUTIL.convertToGTLRectInt(DRCUTIL.getEnlargedRect(rect, curr_spacing));
        GTLPolySetInt prl_spacing_poly_set;
        prl_spacing_poly_set += DRCUTIL.convertToGTLRectInt(DRCUTIL.getEnlargedRect(rect, curr_prl, curr_prl_spacing));
        prl_spacing_poly_set += DRCUTIL.convertToGTLRectInt(DRCUTIL.getEnlargedRect(rect, curr_prl_spacing, curr_prl));
        std::vector<std::pair<BGRectInt, int32_t>> bg_rect_net_pair_list;
        {
          PlanarRect check_rect = DRCUTIL.getEnlargedRect(rect, std::max({curr_spacing, curr_prl, curr_prl_spacing}));
          cut_bg_rtree_map[below_cut_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(check_rect)), std::back_inserter(bg_rect_net_pair_list));
        }
        for (auto& [bg_env_rect, env_net_idx] : bg_rect_net_pair_list) {
          if (net_idx == -1 && env_net_idx == -1) {
            continue;
          }
          PlanarRect env_rect = DRCUTIL.convertToPlanarRect(bg_env_rect);
          if (DRCUTIL.isClosedOverlap(rect, env_rect)) {
            continue;
          }
          GTLRectInt gtl_env_rect = DRCUTIL.convertToGTLRectInt(bg_env_rect);
          if (gtl::area(prl_spacing_poly_set & gtl_env_rect) > 0 && net_idx != env_net_idx) {
            Violation violation;
            violation.set_violation_type(ViolationType::kDifferentLayerCutSpacing);
            violation.set_is_routing(true);
            violation.set_violation_net_set({net_idx, env_net_idx});
            violation.set_layer_idx(routing_layer_idx);
            violation.set_rect(DRCUTIL.getSpacingRect(rect, env_rect));
            violation.set_required_size(curr_prl_spacing);
            rv_box.get_violation_list().push_back(violation);
          } else if (gtl::area(curr_spacing_poly_set & gtl_env_rect) > 0 && net_idx != env_net_idx) {
            if (curr_spacing <= DRCUTIL.getEuclideanDistance(rect, env_rect)) {
              continue;
            }
            Violation violation;
            violation.set_violation_type(ViolationType::kDifferentLayerCutSpacing);
            violation.set_is_routing(true);
            violation.set_violation_net_set({net_idx, env_net_idx});
            violation.set_layer_idx(routing_layer_idx);
            violation.set_rect(DRCUTIL.getSpacingRect(rect, env_rect));
            violation.set_required_size(curr_spacing);
            rv_box.get_violation_list().push_back(violation);
          }
        }
      }
    }
  }
}

}  // namespace idrc
