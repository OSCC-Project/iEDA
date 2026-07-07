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

void RuleValidator::verifyDifferentLayerCutSpacing(RVCluster& rv_cluster)
{
  std::vector<CutLayer>& cut_layer_list = DRCDM.getDatabase().get_cut_layer_list();
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = DRCDM.getDatabase().get_cut_to_adjacent_routing_map();
  const auto& layer_data = rv_cluster.get_layer_data();

  for (const auto& [cut_layer_idx, cut_layer_data] : layer_data) {
    if (cut_layer_data.cut_pool.empty()) {
      continue;
    }
    int32_t below_cut_layer_idx = cut_layer_idx - 1;
    if (below_cut_layer_idx < 0) {
      continue;
    }
    auto below_cut_layer_it = layer_data.find(below_cut_layer_idx);
    if (below_cut_layer_it == layer_data.end() || below_cut_layer_it->second.cut_pool.empty()) {
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
    int32_t curr_prl = different_layer_cut_spacing_rule.below_prl;
    for (const CutData& cut_data : cut_layer_data.getCuts()) {
      GTLRectInt cut_gtl_rect = cut_data.rect;
      int32_t net_idx = cut_data.net_idx;
      PlanarRect cut_rect = DRCUTIL.convertToPlanarRect(cut_gtl_rect);
      std::vector<CutData> overlap_cut_list;
      {
        PlanarRect check_rect = DRCUTIL.getEnlargedRect(cut_rect, std::max({curr_spacing, curr_prl, curr_prl_spacing}));
        below_cut_layer_it->second.queryCuts(DRCUTIL.convertToGTLRectInt(check_rect), std::back_inserter(overlap_cut_list));
      }
      for (const CutData& overlap_cut_data : overlap_cut_list) {
        int32_t env_net_idx = overlap_cut_data.net_idx;
        if (cut_data.isEnv && overlap_cut_data.isEnv) {
          continue;
        }
        if (net_idx == env_net_idx) {
          continue;
        }
        PlanarRect env_rect = DRCUTIL.convertToPlanarRect(overlap_cut_data.rect);
        if (DRCUTIL.isClosedOverlap(cut_rect, env_rect)) {
          continue;
        }
        int32_t prl = DRCUTIL.getParallelLength(cut_rect, env_rect);
        int32_t required_size = prl > curr_prl ? curr_prl_spacing : curr_spacing;
        int32_t distance = DRCUTIL.getEuclideanDistance(cut_rect, env_rect);
        if (distance >= required_size) {
          continue;
        }
        Violation violation;
        violation.set_violation_type(ViolationType::kDifferentLayerCutSpacing);
        violation.set_is_routing(true);
        violation.set_violation_net_set({net_idx, env_net_idx});
        violation.set_layer_idx(routing_layer_idx);
        violation.set_rect(DRCUTIL.getSpacingRect(cut_rect, env_rect));
        violation.set_required_size(required_size);
        rv_cluster.get_violation_list().push_back(std::move(violation));
      }
    }
  }
}

}  // namespace idrc
