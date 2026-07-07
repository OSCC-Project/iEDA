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
#include "PlanarRect.hpp"
#include "RuleValidator.hpp"

namespace idrc {

void RuleValidator::verifySameLayerCutSpacing(RVCluster& rv_cluster)
{
  std::vector<CutLayer>& cut_layer_list = DRCDM.getDatabase().get_cut_layer_list();
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = DRCDM.getDatabase().get_cut_to_adjacent_routing_map();
  const auto& layer_data = rv_cluster.get_layer_data();

  for (const auto& [cut_layer_idx, cut_layer_data] : layer_data) {
    if (cut_layer_data.cut_pool.empty()) {
      continue;
    }
    std::vector<Violation> layer_violations;
    int32_t routing_layer_idx = -1;
    {
      std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[cut_layer_idx];
      routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
    }
    CutLayer& cut_layer = cut_layer_list[cut_layer_idx];
    SameLayerCutSpacingRule& same_layer_cut_spacing_rule = cut_layer.get_same_layer_cut_spacing_rule();
    bool has_same_net = false;
    int32_t curr_same_net_spacing = -1;
    int32_t curr_spacing = -1;
    int32_t curr_prl_spacing = -1;
    int32_t curr_prl = -1;
    for (auto& spacing_rule : same_layer_cut_spacing_rule.spacings) {
      if (spacing_rule.has_same_net) {
        has_same_net = true;
        curr_same_net_spacing = spacing_rule.curr_spacing;
      } else {
        curr_spacing = spacing_rule.curr_spacing;
      }
      curr_prl = -1 * spacing_rule.curr_prl;
      curr_prl_spacing = spacing_rule.curr_prl_spacing;
    }

    for (const CutData& cut_data : cut_layer_data.getCuts()) {
      GTLRectInt cut_gtl_rect = cut_data.rect;
      int32_t net_idx = cut_data.net_idx;
      if (cut_data.isEnv) {
        continue;
      }
      PlanarRect cut_rect = DRCUTIL.convertToPlanarRect(cut_gtl_rect);
      PlanarRect checking_region_vertical = DRCUTIL.getEnlargedRect(cut_rect, curr_prl, curr_prl_spacing);
      PlanarRect checking_region_horizontal = DRCUTIL.getEnlargedRect(cut_rect, curr_prl_spacing, curr_prl);
      std::vector<CutData> overlap_cut_list;
      {
        PlanarRect check_rect = DRCUTIL.getEnlargedRect(cut_rect, std::max({curr_spacing, curr_prl, curr_prl_spacing}));
        cut_layer_data.queryCuts(DRCUTIL.convertToGTLRectInt(check_rect), std::back_inserter(overlap_cut_list));
      }
      for (const CutData& overlap_cut_data : overlap_cut_list) {
        int32_t env_net_idx = overlap_cut_data.net_idx;
        int32_t net_spacing = curr_spacing;
        PlanarRect env_rect = DRCUTIL.convertToPlanarRect(overlap_cut_data.rect);
        PlanarRect violation_rect = DRCUTIL.getSpacingRect(cut_rect, env_rect);
        if ((env_net_idx == net_idx) && has_same_net) {
          net_spacing = curr_same_net_spacing;
        }
        //  ignore cutShort
        if (DRCUTIL.isClosedOverlap(cut_rect, env_rect)) {
          continue;
        }
        bool use_prl_spacing = false, use_spaing = false;
        use_prl_spacing = DRCUTIL.isOpenOverlap(checking_region_horizontal, env_rect) || DRCUTIL.isOpenOverlap(checking_region_vertical, env_rect);
        use_prl_spacing &= (curr_prl_spacing != -1);
        use_spaing = DRCUTIL.getEuclideanDistance(cut_rect, env_rect) < net_spacing;
        if (!use_prl_spacing && !use_spaing) {
          continue;
        }
        Violation violation;
        violation.set_violation_type(ViolationType::kSameLayerCutSpacing);
        violation.set_is_routing(true);
        violation.set_violation_net_set({net_idx, env_net_idx});
        violation.set_layer_idx(routing_layer_idx);
        violation.set_rect(violation_rect);
        violation.set_required_size(use_prl_spacing ? curr_prl_spacing : net_spacing);
        layer_violations.push_back(std::move(violation));
      }
    }

    // merge same net violation
    {
      if (layer_violations.size() > 1) {
        std::sort(layer_violations.begin(), layer_violations.end(), [](const Violation& a, const Violation& b) {
          if (a.get_violation_net_set() != b.get_violation_net_set())
            return a.get_violation_net_set() < b.get_violation_net_set();

          const auto& ra = a.get_rect();
          const auto& rb = b.get_rect();
          if (ra.get_ll_x() != rb.get_ll_x())
            return ra.get_ll_x() < rb.get_ll_x();
          if (ra.get_ur_x() != rb.get_ur_x())
            return ra.get_ur_x() > rb.get_ur_x();
          if (ra.get_ll_y() != rb.get_ll_y())
            return ra.get_ll_y() < rb.get_ll_y();
          return ra.get_ur_y() > rb.get_ur_y();
        });

        std::vector<Violation> results;
        results.reserve(layer_violations.size());
        std::vector<const Violation*> active_set;

        for (const auto& v : layer_violations) {
          bool is_redundant = false;
          const auto& cur_r = v.get_rect();

          if (!active_set.empty() && active_set.back()->get_violation_net_set() != v.get_violation_net_set()) {
            active_set.clear();
          }

          active_set.erase(
              std::remove_if(active_set.begin(), active_set.end(), [&](const Violation* p) { return p->get_rect().get_ur_x() < cur_r.get_ll_x(); }),
              active_set.end());

          for (const auto* p_active : active_set) {
            if (DRCUTIL.isInside(p_active->get_rect(), cur_r)) {
              is_redundant = true;
              break;
            }
          }

          if (!is_redundant) {
            results.push_back(std::move(const_cast<Violation&>(v)));
            active_set.push_back(&results.back());
          }
        }
        layer_violations = std::move(results);
      }

      auto& final_list = rv_cluster.get_violation_list();
      final_list.insert(final_list.end(), std::make_move_iterator(layer_violations.begin()), std::make_move_iterator(layer_violations.end()));
    }
  }
}

}  // namespace idrc
