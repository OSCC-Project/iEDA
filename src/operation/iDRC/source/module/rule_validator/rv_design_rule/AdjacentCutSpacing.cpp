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
  std::vector<CutLayer>& cut_layer_list = DRCDM.getDatabase().get_cut_layer_list();
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = DRCDM.getDatabase().get_cut_to_adjacent_routing_map();
  const auto& layer_data = rv_cluster.get_layer_data();

  for (const auto& [cut_layer_idx, cut_layer_data] : layer_data) {
    if (cut_layer_data.cut_pool.empty()) {
      continue;
    }

    int32_t routing_layer_idx = -1;
    {
      std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[cut_layer_idx];
      routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
    }
    CutLayer& cut_layer = cut_layer_list[cut_layer_idx];

    AdjacentCutSpacingRule& adj_cut_rule = cut_layer.get_adjacent_cut_rule();
    for (const CutData& cut_data : cut_layer_data.getCuts()) {
      GTLRectInt cut_gtl_rect = cut_data.rect;
      PlanarRect cut_rect = DRCUTIL.convertToPlanarRect(cut_gtl_rect);
      int32_t net_idx = cut_data.net_idx;

      std::vector<CutData> neighbor_cut_list;
      {
        PlanarRect check_rect = DRCUTIL.getEnlargedRect(cut_rect, adj_cut_rule.cut_within);
        cut_layer_data.queryCuts(DRCUTIL.convertToGTLRectInt(check_rect), std::back_inserter(neighbor_cut_list));
      }

      int32_t adjacent_cut_count = 0;
      bool has_violation = false;
      int32_t cur_remote_spacing = -1;
      int32_t cur_remote_env_net = -1;
      std::set<std::tuple<int32_t, int32_t, int32_t, int32_t>> seen_neighboor_rects;

      for (const CutData& neighbor_cut_data : neighbor_cut_list) {
        int32_t env_net_idx = neighbor_cut_data.net_idx;
        PlanarRect env_rect = DRCUTIL.convertToPlanarRect(neighbor_cut_data.rect);
        if (cut_rect == env_rect) {
          continue;
        }

        int32_t distance = DRCUTIL.getEuclideanDistance(cut_rect, env_rect);
        if (distance < adj_cut_rule.cut_within) {
          auto nb_rect_key = std::make_tuple(env_rect.get_ll_x(), env_rect.get_ll_y(), env_rect.get_ur_x(), env_rect.get_ur_y());
          if (!seen_neighboor_rects.insert(nb_rect_key).second) {
            continue;
          }
          adjacent_cut_count++;
          if (distance > cur_remote_spacing && env_net_idx != -1) {
            cur_remote_spacing = distance;
            cur_remote_env_net = env_net_idx;
          }
        }
        if (distance < adj_cut_rule.cut_spacing) {
          has_violation = true;
        }
      }
      if (net_idx == -1 && cur_remote_env_net == -1) {
        continue;
      }
      if (has_violation && (adjacent_cut_count >= adj_cut_rule.adjacnet_cuts)) {
        Violation violation;
        violation.set_violation_type(ViolationType::kAdjacentCutSpacing);
        violation.set_is_routing(true);
        violation.set_violation_net_set({net_idx, cur_remote_env_net});
        violation.set_layer_idx(routing_layer_idx);
        violation.set_rect(cut_rect);
        violation.set_required_size(adj_cut_rule.cut_spacing);
        rv_cluster.get_violation_list().push_back(violation);
      }
    }
  }
}

}  // namespace idrc
