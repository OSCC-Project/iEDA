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

void RuleValidator::verifyMaxViaStack(RVCluster& rv_cluster)
{
  MaxViaStackRule& max_via_stack_rule = DRCDM.getDatabase().get_max_via_stack_rule();
  std::map<int32_t, std::vector<int32_t>>& routing_to_adjacent_cut_map = DRCDM.getDatabase().get_routing_to_adjacent_cut_map();
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
  int32_t max_via_stack_num = max_via_stack_rule.max_via_stack_num;
  int32_t bottom_cut_layer_idx = -1;
  {
    std::vector<int32_t>& cut_layer_idx_list = routing_to_adjacent_cut_map[max_via_stack_rule.bottom_routing_layer_idx];
    bottom_cut_layer_idx = *std::max_element(cut_layer_idx_list.begin(), cut_layer_idx_list.end());
  }
  int32_t top_cut_layer_idx = -1;
  {
    std::vector<int32_t>& cut_layer_idx_list = routing_to_adjacent_cut_map[max_via_stack_rule.top_routing_layer_idx];
    top_cut_layer_idx = *std::min_element(cut_layer_idx_list.begin(), cut_layer_idx_list.end());
  }
  for (auto& [cut_layer_idx, net_rect_map] : cut_net_rect_map) {
    if (cut_layer_idx < bottom_cut_layer_idx || top_cut_layer_idx < (cut_layer_idx + max_via_stack_num)) {
      continue;
    }
    std::vector<std::pair<int32_t, PlanarRect>> net_stack_rect_list;
    for (auto& [net_idx, rect_list] : net_rect_map) {
      for (PlanarRect& rect : rect_list) {
        net_stack_rect_list.emplace_back(net_idx, rect);
      }
    }
    for (std::pair<int32_t, PlanarRect>& net_stack_rect_pair : net_stack_rect_list) {
      std::map<int32_t, std::vector<std::pair<int32_t, PlanarRect>>> layer_net_stack_rect_map;
      layer_net_stack_rect_map[cut_layer_idx].push_back(net_stack_rect_pair);
      for (int32_t curr_cut_layer_idx = cut_layer_idx; curr_cut_layer_idx < top_cut_layer_idx; curr_cut_layer_idx++) {
        std::map<int32_t, std::set<PlanarRect, CmpPlanarRectByXASC>> net_used_rect_set;
        for (auto& [net_idx, stack_rect] : layer_net_stack_rect_map[curr_cut_layer_idx]) {
          std::vector<std::pair<BGRectInt, int32_t>> bg_rect_net_pair_list;
          PlanarRect check_rect = stack_rect;
          cut_bg_rtree_map[curr_cut_layer_idx + 1].query(bgi::intersects(DRCUTIL.convertToBGRectInt(check_rect)), std::back_inserter(bg_rect_net_pair_list));
          for (auto& [bg_env_rect, env_net_idx] : bg_rect_net_pair_list) {
            PlanarRect env_rect = DRCUTIL.convertToPlanarRect(bg_env_rect);
            if (!DRCUTIL.isOpenOverlap(stack_rect, env_rect)) {
              continue;
            }
            if (DRCUTIL.exist(net_used_rect_set[env_net_idx], env_rect)) {
              continue;
            }
            layer_net_stack_rect_map[curr_cut_layer_idx + 1].push_back({env_net_idx, env_rect});
            net_used_rect_set[env_net_idx].insert(env_rect);
          }
        }
      }
      if (static_cast<int32_t>(layer_net_stack_rect_map.size()) <= max_via_stack_num) {
        continue;
      }
      // 依次获取违例矩形
      for (auto& [curr_cut_layer_idx, net_stack_rect_map] : layer_net_stack_rect_map) {
        if (curr_cut_layer_idx < bottom_cut_layer_idx + max_via_stack_num || top_cut_layer_idx < curr_cut_layer_idx) {
          continue;
        }
        int32_t routing_layer_idx = -1;
        {
          std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[curr_cut_layer_idx];
          routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
        }
        for (auto& [curr_net_idx, curr_stack_rect] : net_stack_rect_map) {
          bool is_violation = true;
          if (DRCUTIL.exist(layer_net_stack_rect_map, curr_cut_layer_idx + 1)) {
            for (auto& [pre_net_idx, pre_stack_rect] : layer_net_stack_rect_map[curr_cut_layer_idx - 1]) {
              if (DRCUTIL.isOpenOverlap(pre_stack_rect, curr_stack_rect)) {
                is_violation = false;
                break;
              }
            }
          }
          if (!is_violation) {
            continue;
          }
          Violation violation;
          violation.set_violation_type(ViolationType::kMaxViaStack);
          violation.set_is_routing(true);
          violation.set_layer_idx(routing_layer_idx);
          violation.set_rect(curr_stack_rect);
          violation.set_violation_net_set({curr_net_idx});
          violation.set_required_size(max_via_stack_num);
          rv_cluster.get_violation_list().push_back(violation);
        }
      }
    }
  }
}

}  // namespace idrc
