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
  const auto& layer_data = rv_cluster.get_layer_data();
  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> routing_net_checking_polysets;

  // checking polyset keeps the result routing only.
  {
    std::map<int32_t, std::map<int32_t, std::vector<GTLRectInt>>> net_rects_to_merge_checking;
    for (DRCShape* drc_shape : rv_cluster.get_drc_result_shape_list()) {
      if (drc_shape->get_is_routing() && drc_shape->get_net_idx() != -1) {
        net_rects_to_merge_checking[drc_shape->get_layer_idx()][drc_shape->get_net_idx()].push_back(DRCUTIL.convertToGTLRectInt(drc_shape->get_rect()));
      }
    }
    for (auto& [layer_idx, net_map] : net_rects_to_merge_checking) {
      auto& target_net_map = routing_net_checking_polysets[layer_idx];
      for (auto& [net_idx, rect_vec] : net_map) {
        target_net_map[net_idx].insert(rect_vec.begin(), rect_vec.end());
      }
    }
  }

  // check rules
  for (const auto& [routing_layer_idx, net_polyset] : routing_net_checking_polysets) {
    auto layer_data_it = layer_data.find(routing_layer_idx);
    if (layer_data_it == layer_data.end()) {
      continue;
    }
    const RVLayerData& rv_layer_data = layer_data_it->second;
    std::vector<Violation> layer_violations;
    std::vector<std::pair<GTLRectInt, int32_t>> overlap_max_rects;
    for (auto& [net_idx, polyset] : net_polyset) {
      std::vector<GTLRectInt> rect_list;
      gtl::get_max_rectangles(rect_list, polyset);

      for (GTLRectInt& gtl_rect : rect_list) {
        PlanarRect rect = DRCUTIL.convertToPlanarRect(gtl_rect);

        overlap_max_rects.clear();
        rv_layer_data.queryMaxRects(gtl_rect, std::back_inserter(overlap_max_rects));
        for (auto [env_gtl_rect, env_max_rect_id] : overlap_max_rects) {
          int32_t env_net_idx = rv_layer_data.getNetIdxByMaxRectId(env_max_rect_id);
          if (net_idx == env_net_idx) {
            continue;
          }
          PlanarRect env_rect = DRCUTIL.convertToPlanarRect(env_gtl_rect);
          if (!DRCUTIL.isClosedOverlap(rect, env_rect)) {
            continue;
          }
          Violation violation;
          violation.set_violation_type(ViolationType::kMetalShort);
          violation.set_is_routing(true);
          violation.set_violation_net_set({net_idx, env_net_idx});
          violation.set_layer_idx(routing_layer_idx);
          violation.set_rect(DRCUTIL.getOverlap(rect, env_rect));
          violation.set_required_size(0);
          layer_violations.push_back(std::move(violation));
        }
      }
    }

    // postprocess, build final violations
    {
      if (layer_violations.size() > 1) {
        std::sort(layer_violations.begin(), layer_violations.end(), [](const Violation& a, const Violation& b) {
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
            results.push_back(v);
            active_set.push_back(&results.back());
          }
        }
        layer_violations = std::move(results);
      }

      rv_cluster.get_violation_list().insert(rv_cluster.get_violation_list().end(), std::make_move_iterator(layer_violations.begin()),
                                             std::make_move_iterator(layer_violations.end()));
    }
  }
}

}  // namespace idrc
