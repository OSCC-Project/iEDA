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

void RuleValidator::verifyNonsufficientMetalOverlap(RVCluster& rv_cluster)
{
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  const auto& layer_data = rv_cluster.get_layer_data();

  for (auto& [routing_layer_idx, rv_layer_data] : layer_data) {
    RoutingLayer& routing_layer = routing_layer_list[routing_layer_idx];
    int32_t min_width = routing_layer.get_minimum_width_rule().min_width;
    int32_t half_width = min_width / 2;
    for (auto& [net_idx, routing_net] : rv_layer_data.nets) {
      if (net_idx == -1) {
        continue;
      }
      for (const MaxRectData& max_rect_data : rv_layer_data.getMaxRects(routing_net)) {
        if (max_rect_data.isEnv) {
          continue;
        }
        const GTLRectInt& gtl_rect = max_rect_data.rect;
        PlanarRect rect = DRCUTIL.convertToPlanarRect(gtl_rect);
        std::vector<std::pair<GTLRectInt, int32_t>> env_rect_id_list;
        {
          PlanarRect check_rect = rect;
          rv_layer_data.queryMaxRects(DRCUTIL.convertToGTLRectInt(check_rect), std::back_inserter(env_rect_id_list));
        }
        for (const auto& [env_gtl_rect, env_max_rect_id] : env_rect_id_list) {
          if (rv_layer_data.getNetIdxByMaxRectId(env_max_rect_id) != net_idx) {
            continue;
          }
          PlanarRect env_rect = DRCUTIL.convertToPlanarRect(env_gtl_rect);
          if (!DRCUTIL.isClosedOverlap(rect, env_rect)) {
            continue;
          }
          if (env_rect == rect) {
            continue;
          }
          PlanarRect overlap_rect = DRCUTIL.getOverlap(rect, env_rect);
          double diag_length = std::hypot(overlap_rect.getXSpan(), overlap_rect.getYSpan());
          if (diag_length >= min_width) {
            continue;
          }
          std::vector<BGRectInt> overlap_rect_env_list;
          {
            PlanarRect check_rect = DRCUTIL.getEnlargedRect(overlap_rect, 1);
            std::vector<std::pair<GTLRectInt, int32_t>> overlap_rect_env_pairs;
            rv_layer_data.queryMaxRects(DRCUTIL.convertToGTLRectInt(check_rect), std::back_inserter(overlap_rect_env_pairs));
            overlap_rect_env_list.reserve(overlap_rect_env_pairs.size());
            for (const auto& [overlap_gtl_rect, overlap_max_rect_id] : overlap_rect_env_pairs) {
              if (rv_layer_data.getNetIdxByMaxRectId(overlap_max_rect_id) != net_idx) {
                continue;
              }
              GTLRectInt overlap_gtl_rect_copy = overlap_gtl_rect;
              overlap_rect_env_list.push_back(DRCUTIL.convertToBGRectInt(overlap_gtl_rect_copy));
            }
          }
          bool is_inside = false;
          for (auto& overlap_rect_env : overlap_rect_env_list) {
            PlanarRect thirdrect = DRCUTIL.convertToPlanarRect(overlap_rect_env);
            // 如果有第三个矩形包含overlap， 且和rect env_rect还有其他重叠，称该违例在金属中，跳过
            if ((DRCUTIL.isInside(thirdrect, overlap_rect) && DRCUTIL.isClosedOverlap(thirdrect, overlap_rect)) && thirdrect != rect && thirdrect != env_rect
                && thirdrect.getWidth() >= min_width && DRCUTIL.getOverlap(thirdrect, rect) != overlap_rect
                && DRCUTIL.getOverlap(thirdrect, env_rect) != overlap_rect) {
              is_inside = true;
              break;
            }
          }
          if (is_inside) {
            continue;
          }
          int32_t x_enlarge_size = 0;
          if (overlap_rect.getXSpan() < half_width) {
            x_enlarge_size = half_width - overlap_rect.getXSpan();
          }
          int32_t y_enlarge_size = 0;
          if (overlap_rect.getYSpan() < half_width) {
            y_enlarge_size = half_width - overlap_rect.getYSpan();
          }
          PlanarRect violation_rect = DRCUTIL.getEnlargedRect(overlap_rect, x_enlarge_size, y_enlarge_size, x_enlarge_size, y_enlarge_size);
          Violation violation;
          violation.set_violation_type(ViolationType::kNonsufficientMetalOverlap);
          violation.set_is_routing(true);
          violation.set_violation_net_set({net_idx});
          violation.set_required_size(0);
          violation.set_layer_idx(routing_layer_idx);
          violation.set_rect(violation_rect);
          rv_cluster.get_violation_list().push_back(violation);
        }
      }
    }
  }
}

}  // namespace idrc
