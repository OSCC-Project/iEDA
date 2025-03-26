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

void RuleValidator::verifyMinStep(RVBox& rv_box)
{
  /// 情况1  MINSTEP 0.050000 MAXEDGES 1 ;
  constexpr int32_t MAX_EDGES = 1;
  constexpr int32_t MIN_STEP = 100;
  /// 情况2  PROPERTY LEF58_MINSTEP "MINSTEP 0.05 MAXEDGES 1 MINADJACENTLENGTH 0.065 CONVEXCORNER ;" ;：
  constexpr int32_t RULE_EDGE_LENGTH = 100;
  constexpr int32_t RULE_MIN_ADJACENT_LENGTH = 130;

  /// 利用Rtree索引，判断当前违例矩形是否被某个违例矩形覆盖；如果是，忽略该违例。
  auto addIfNotContained = [](bgi::rtree<BGRectInt, bgi::quadratic<16>>& rtree, const BGRectInt& rect) -> bool {
    std::vector<BGRectInt> result;
    rtree.query(bgi::intersects(rect), std::back_inserter(result));

    for (const auto& existing : result) {
      if (bg::within(rect, existing)) {
        return false;
      }
    }
    rtree.insert(rect);
    return true;
  };

  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> routing_net_gtl_poly_set_map;
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();

  for (DRCShape* drc_shape : rv_box.get_drc_result_shape_list()) {
    if (!drc_shape->get_is_routing() || drc_shape->get_net_idx() == -1) {
      continue;
    }
    routing_net_gtl_poly_set_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
  }

  for (DRCShape* drc_shape : rv_box.get_drc_env_shape_list()) {
    if (!drc_shape->get_is_routing() || drc_shape->get_net_idx() == -1) {
      continue;
    }
    routing_net_gtl_poly_set_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
  }

  /*
  遍历每个属于同个net的多边形
  先记录顶点坐标、边长、凹凸性的信息；
  再遍历多边形，进行判断；
  std::vector<int32_t> edge_lengths(hole_poly_size); 当前点的边长表示 当前点和前一个点组成的边的边长。
  */
  for (auto& [routing_layer_idx, net_gtl_poly_set_map] : routing_net_gtl_poly_set_map) {
    RoutingLayer& routing_layer = routing_layer_list[routing_layer_idx];
    for (auto& [net_idx, gtl_poly_set] : net_gtl_poly_set_map) {
      std::vector<GTLHolePolyInt> gtl_hole_poly_list;
      gtl_poly_set.clean();
      gtl_poly_set.get_polygons(gtl_hole_poly_list);
      for (GTLHolePolyInt& gtl_hole_poly : gtl_hole_poly_list) {
        if (gtl_hole_poly.size() < 4) {
          continue;
        }
        // 初始化数据结构
        int32_t hole_poly_size = gtl_hole_poly.size();
        std::vector<int32_t> edge_lengths(hole_poly_size);
        std::vector<bool> corner_convex(hole_poly_size);
        std::vector<GTLPointInt> vertices(hole_poly_size);
        bgi::rtree<BGRectInt, bgi::quadratic<16>> rtree;
        auto get_index_shifted = [&](int index, int shift) { return (index + shift + hole_poly_size) % hole_poly_size; };

        // 遍历多边形，存储边长和凹凸性信息
        {
          auto it_begin = gtl_hole_poly.begin();
          auto it_end = gtl_hole_poly.end();

          auto prev_it = it_begin;
          auto curr_it = std::next(prev_it);
          auto next_it = std::next(curr_it);
          int corner_index = 0;

          do {
            vertices[corner_index] = *curr_it;

            // 计算当前边长度（从prev_it到curr_it）
            int32_t edge_length = static_cast<int32_t>(std::hypot((*curr_it).x() - (*prev_it).x(), (*curr_it).y() - (*prev_it).y()));

            // 计算拐角凹凸性（prev_it, curr_it, next_it构成的角）
            int32_t cross_product
                = ((*curr_it).x() - (*prev_it).x()) * ((*next_it).y() - (*curr_it).y()) - ((*curr_it).y() - (*prev_it).y()) * ((*next_it).x() - (*curr_it).x());

            bool is_convex = (cross_product > 0);  // true表示凸角，false表示凹角

            edge_lengths[corner_index] = edge_length;
            corner_convex[corner_index] = is_convex;

            ++corner_index;
            prev_it = curr_it;
            curr_it = next_it;
            ++next_it;
            if (next_it == it_end) {
              next_it = it_begin;
            }
          } while (prev_it != it_begin);
        }

        for (int32_t current_index = 0; current_index < hole_poly_size; ++current_index) {
          auto pre_index = get_index_shifted(current_index, -1);
          auto next_index = get_index_shifted(current_index, 1);

          /// 情况1
          if (edge_lengths[current_index] < MIN_STEP) {
            int consecutive_small_edges = 1;
            for (int32_t i = 1; i < hole_poly_size; ++i) {
              if (edge_lengths[get_index_shifted(current_index, i)] < MIN_STEP) {
                consecutive_small_edges++;
              } else {
                break;
              }
            }
            if (consecutive_small_edges > MAX_EDGES) {
              // 找到违例 - 创建覆盖所有涉及点的最大矩形
              int start_index = pre_index;
              int end_index = get_index_shifted(current_index, consecutive_small_edges - 1);

              // 初始化边界值
              int min_x = vertices[start_index].x();
              int min_y = vertices[start_index].y();
              int max_x = vertices[start_index].x();
              int max_y = vertices[start_index].y();

              // 遍历所有连续小边的顶点，找到最小/最大坐标
              int vertex_index = start_index;
              while (vertex_index != get_index_shifted(end_index, 1)) {
                min_x = std::min(min_x, vertices[vertex_index].x());
                min_y = std::min(min_y, vertices[vertex_index].y());
                max_x = std::max(max_x, vertices[vertex_index].x());
                max_y = std::max(max_y, vertices[vertex_index].y());

                vertex_index = get_index_shifted(vertex_index, 1);
              }

              // 创建覆盖所有点的最大矩形
              GTLRectInt rect(min_x, min_y, max_x, max_y);
              BGRectInt bgrect = DRCUTIL.convertToBGRectInt(rect);
              if (addIfNotContained(rtree, bgrect)) {
                PlanarRect violation_rect = DRCUTIL.convertToPlanarRect(rect);
                Violation violation;
                violation.set_violation_type(ViolationType::kMinStep);
                violation.set_required_size(100);
                violation.set_is_routing(true);
                violation.set_violation_net_set({net_idx});
                violation.set_layer_idx(routing_layer_idx);
                violation.set_rect(violation_rect);
                rv_box.get_violation_list().push_back(violation);
              }
            }
          }
          /// 情况2
          // 找到凹凸凹序列
          if (corner_convex[pre_index] == false && corner_convex[current_index] == true && corner_convex[next_index] == false) {
            if ((edge_lengths[current_index] < RULE_EDGE_LENGTH && edge_lengths[next_index] < RULE_MIN_ADJACENT_LENGTH)
                || (edge_lengths[current_index] < RULE_MIN_ADJACENT_LENGTH && edge_lengths[next_index] < RULE_EDGE_LENGTH)) {
              // Violation
              GTLRectInt rect(vertices[pre_index].x(), vertices[pre_index].y(), vertices[next_index].x(), vertices[next_index].y());
              BGRectInt bgrect = DRCUTIL.convertToBGRectInt(rect);
              if (addIfNotContained(rtree, bgrect)) {
                PlanarRect violation_rect = DRCUTIL.convertToPlanarRect(rect);
                Violation violation;
                violation.set_violation_type(ViolationType::kMinStep);
                violation.set_required_size(100);
                violation.set_is_routing(true);
                violation.set_violation_net_set({net_idx});
                violation.set_layer_idx(routing_layer_idx);
                violation.set_rect(violation_rect);
                rv_box.get_violation_list().push_back(violation);
              }
            }
          }
        }
      }
    }
  }
}
}  // namespace idrc
