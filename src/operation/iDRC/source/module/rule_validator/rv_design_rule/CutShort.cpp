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

void RuleValidator::verifyCutShort(RVCluster& rv_cluster)
{
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = DRCDM.getDatabase().get_cut_to_adjacent_routing_map();
  const auto& layer_data = rv_cluster.get_layer_data();

  for (const auto& [cut_layer_idx, rv_layer_data] : layer_data) {
    if (rv_layer_data.cut_pool.empty()) {
      continue;
    }
    std::vector<Violation> layer_violations;
    int32_t routing_layer_idx = -1;
    {
      std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[cut_layer_idx];
      routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
    }

    for (const CutData& cut_data : rv_layer_data.getCuts()) {
      if (cut_data.isEnv) {
        continue;
      }
      GTLRectInt curr_cut_gtl_rect = cut_data.rect;
      PlanarRect cut_rect = DRCUTIL.convertToPlanarRect(curr_cut_gtl_rect);
      std::vector<CutData> overlap_cut_list;
      rv_layer_data.queryCuts(cut_data.rect, std::back_inserter(overlap_cut_list));
      for (const CutData& overlap_cut_data : overlap_cut_list) {
        if (overlap_cut_data.net_idx > cut_data.net_idx) {
          continue;
        }
        PlanarRect env_rect = DRCUTIL.convertToPlanarRect(overlap_cut_data.rect);
        if (cut_data.net_idx == overlap_cut_data.net_idx && cut_rect == env_rect) {
          continue;
        }
        Violation violation;
        violation.set_violation_type(ViolationType::kCutShort);
        violation.set_required_size(0);
        violation.set_is_routing(true);
        violation.set_violation_net_set({cut_data.net_idx, overlap_cut_data.net_idx});
        violation.set_layer_idx(routing_layer_idx);
        violation.set_rect(DRCUTIL.getOverlap(cut_rect, env_rect));
        layer_violations.push_back(std::move(violation));
      }
    }

    // sort and unique
    {
      if (layer_violations.size() > 1) {
        std::sort(layer_violations.begin(), layer_violations.end(), [](const Violation& a, const Violation& b) {
          CmpPlanarRectByXASC rect_cmp;
          if (rect_cmp(a.get_rect(), b.get_rect()))
            return true;
          if (rect_cmp(b.get_rect(), a.get_rect()))
            return false;

          return a.get_violation_net_set() < b.get_violation_net_set();
        });

        auto last = std::unique(layer_violations.begin(), layer_violations.end(), [](const Violation& a, const Violation& b) {
          return a.get_rect() == b.get_rect() && a.get_violation_net_set() == b.get_violation_net_set();
        });

        layer_violations.erase(last, layer_violations.end());
      }

      auto& final_list = rv_cluster.get_violation_list();
      final_list.insert(final_list.end(), std::make_move_iterator(layer_violations.begin()), std::make_move_iterator(layer_violations.end()));
    }
  }
}

}  // namespace idrc
