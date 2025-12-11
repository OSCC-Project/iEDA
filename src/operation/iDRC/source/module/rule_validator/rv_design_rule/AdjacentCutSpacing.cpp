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

void RuleValidator::verifyAdjacentCutSpacing(RVCluster& rv_cluster)
{
#if 0
  /*
  规则: SPACING 0.155 ADJACENTCUTS 3 WITHIN 0.200 ;
  */
  struct AdjacentCutSpacingRule  // 这是cut spacing的子规则
  {
    int32_t cut_spacing = -1;
    int32_t adjacent_cuts = -1;
    int32_t within = -1;
  };
  std::map<int32_t, AdjacentCutSpacingRule> layer_adjacent_cut_spacing_rule;
  for (int32_t i = 1; i <= 4; i++) {
    AdjacentCutSpacingRule rule = {310, 3, 400};
    layer_adjacent_cut_spacing_rule[i] = rule;  // via2-via5
  }
  std::vector<CutLayer>& cut_layer_list = DRCDM.getDatabase().get_cut_layer_list();
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = DRCDM.getDatabase().get_cut_to_adjacent_routing_map();

  std::map<int32_t, std::map<int32_t, std::vector<PlanarRect>>> cut_net_rect_map;
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> cut_bg_rtree_map;
  for (DRCShape* drc_shape : rv_cluster.get_drc_env_shape_list()) {
    if (drc_shape->get_is_routing()) {
      continue;
    }
    cut_net_rect_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()].push_back(drc_shape->get_rect());
    cut_bg_rtree_map[drc_shape->get_layer_idx()].insert(std::make_pair(DRCUTIL.convertToBGRectInt(drc_shape->get_rect()), drc_shape->get_net_idx()));
  }
  for (DRCShape* drc_shape : rv_cluster.get_drc_result_shape_list()) {
    if (drc_shape->get_is_routing()) {
      continue;
    }
    cut_net_rect_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()].push_back(drc_shape->get_rect());
    cut_bg_rtree_map[drc_shape->get_layer_idx()].insert(std::make_pair(DRCUTIL.convertToBGRectInt(drc_shape->get_rect()), drc_shape->get_net_idx()));
  }
  for (auto& [cut_layer_idx, net_rect_map] : cut_net_rect_map) {
    int32_t routing_layer_idx = -1;
    {
      std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[cut_layer_idx];
      routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
    }
    if (!DRCUTIL.exist(layer_adjacent_cut_spacing_rule, cut_layer_idx)) {
      continue;
    }
    AdjacentCutSpacingRule& curr_rule = layer_adjacent_cut_spacing_rule[cut_layer_idx];
    int32_t cut_spacing = curr_rule.cut_spacing;
    int32_t adjacent_cuts = curr_rule.adjacent_cuts;
    int32_t within = curr_rule.within;
    for (auto& [net_idx, rect_list] : net_rect_map) {
      for (PlanarRect& rect : rect_list) {
        std::vector<std::pair<BGRectInt, int32_t>> bg_rect_net_pair_list;
        {
          PlanarRect check_rect = DRCUTIL.getEnlargedRect(rect, within);
          cut_bg_rtree_map[cut_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(check_rect)), std::back_inserter(bg_rect_net_pair_list));
        }
        int32_t adjacent_cut_count = 0;
        bool is_inside_spacing = false;
        std::vector<std::pair<int32_t, PlanarRect>> env_cut_rect_list;
        for (auto& [bg_env_rect, env_net_idx] : bg_rect_net_pair_list) {
          PlanarRect env_rect = DRCUTIL.convertToPlanarRect(bg_env_rect);
          if (env_rect == rect) {
            continue;  // 忽略自己
          }
          if (DRCUTIL.getEuclideanDistance(rect, env_rect) >= within) {
            continue;  // 忽略不在within范围内的
          }
          if (DRCUTIL.getEuclideanDistance(rect, env_rect) < cut_spacing) {
            is_inside_spacing = true;  // 在spacing范围内
          }
          env_cut_rect_list.push_back(std::make_pair(env_net_idx, env_rect));
          adjacent_cut_count++;
        }
        if (adjacent_cut_count < adjacent_cuts) {
          continue;  // 忽略不满足adjacent_cuts的
        }
        if (is_inside_spacing == false) {
          continue;
        }
        sort(env_cut_rect_list.begin(), env_cut_rect_list.end(), [&](const std::pair<int32_t, PlanarRect>& a, const std::pair<int32_t, PlanarRect>& b) {
          SortStatus sort_status = SortStatus::kEqual;
          double a_dis = DRCUTIL.getEuclideanDistance(rect, a.second);
          double b_dis = DRCUTIL.getEuclideanDistance(rect, b.second);
          // EuclideanDistance大的优先
          if (sort_status == SortStatus::kEqual) {
            if (a_dis > b_dis) {
              sort_status = SortStatus::kTrue;
            } else if (a_dis < b_dis) {
              sort_status = SortStatus::kFalse;
            } else {
              sort_status = SortStatus::kEqual;
            }
          }
          if (sort_status == SortStatus::kTrue) {
            return true;
          } else if (sort_status == SortStatus::kFalse) {
            return false;
          }
          return false;
        });
        int32_t env_idx = env_cut_rect_list.front().first;
        Violation violation;
        violation.set_violation_type(ViolationType::kAdjacentCutSpacing);
        violation.set_is_routing(true);
        violation.set_violation_net_set({net_idx, env_idx});
        violation.set_layer_idx(routing_layer_idx);
        violation.set_rect(rect);
        violation.set_required_size(cut_spacing);
        rv_cluster.get_violation_list().push_back(violation);
      }
    }
  }
#endif
}

}  // namespace idrc
