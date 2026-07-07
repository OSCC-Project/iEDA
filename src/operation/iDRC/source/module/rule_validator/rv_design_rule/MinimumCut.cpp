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

// // 该规则计算距离不是算欧式距离，类似曼哈顿距离
// double via_routing_distance(GTLPolyInt& gtl_poly, PlanarRect& cut_rect, PlanarRect& routing_rect)
// {
//   if (DRCUTIL.isOverlap(cut_rect, routing_rect)) {
//     return 0.0;
//   }
//   std::vector<GTLRectInt> gtl_rect_list;
//   gtl::get_max_rectangles(gtl_rect_list, gtl_poly);

//   std::vector<PlanarRect> rects;
//   for (auto& r : gtl_rect_list) {
//     rects.push_back(DRCUTIL.convertToPlanarRect(r));
//   }
//   // 检查是否有单个矩形同时包含/接触两者（直线连接）
//   for (const auto& r : rects) {
//     if (DRCUTIL.isOverlap(cut_rect, r) && DRCUTIL.isOverlap(routing_rect, r)) {
//       return DRCUTIL.getEuclideanDistance(cut_rect, routing_rect);
//     }
//   }
//   // 处理复杂路径（多段折弯）
//   int32_t start_idx = -1;
//   int32_t end_idx = -1;
//   for (size_t i = 0; i < rects.size(); ++i) {
//     if (DRCUTIL.isOverlap(cut_rect, rects[i]))
//       start_idx = i;
//     if (DRCUTIL.isOverlap(routing_rect, rects[i]))
//       end_idx = i;
//   }
//   if (start_idx == -1 || end_idx == -1) {
//     return DRCUTIL.getEuclideanDistance(cut_rect, routing_rect);
//   }
//   // 使用广度优先搜索 (BFS) 找到最短矩形链路径
//   std::queue<int32_t> q;
//   q.push(start_idx);
//   std::vector<int32_t> parent(rects.size(), -1);
//   std::vector<bool> visited(rects.size(), false);
//   visited[start_idx] = true;

//   bool found_path = false;
//   while (!q.empty()) {
//     int32_t curr = q.front();
//     q.pop();

//     if (curr == end_idx) {
//       found_path = true;
//       break;
//     }
//     for (size_t next = 0; next < rects.size(); ++next) {
//       if (!visited[next] && DRCUTIL.isOverlap(rects[curr], rects[next])) {
//         visited[next] = true;
//         parent[next] = curr;
//         q.push(next);
//       }
//     }
//   }
//   // 计算路径长度
//   if (found_path) {
//     double total_dist = 0.0;
//     int32_t curr = end_idx;
//     PlanarRect last_target = routing_rect;

//     while (parent[curr] != -1) {
//       int32_t prev = parent[curr];
//       PlanarRect overlap = DRCUTIL.getOverlap(rects[curr], rects[prev]);

//       total_dist += DRCUTIL.getEuclideanDistance(last_target, overlap);

//       last_target = overlap;
//       curr = prev;
//     }
//     total_dist += DRCUTIL.getEuclideanDistance(last_target, cut_rect);
//     return total_dist;
//   }
//   return DRCUTIL.getEuclideanDistance(cut_rect, routing_rect);
// }

void RuleValidator::verifyMinimumCut(RVCluster& rv_cluster)
{
  // std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  // std::map<int32_t, std::vector<int32_t>>& routing_to_adjacent_cut_map = DRCDM.getDatabase().get_routing_to_adjacent_cut_map();

  // std::map<int32_t, std::map<int32_t, GTLPolySetInt>> routing_net_gtl_poly_set_map;
  // {
  //   for (DRCShape* drc_shape : rv_cluster.get_drc_env_shape_list()) {
  //     if (drc_shape->get_is_routing()) {
  //       routing_net_gtl_poly_set_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
  //     }
  //   }
  //   for (DRCShape* drc_shape : rv_cluster.get_drc_result_shape_list()) {
  //     if (drc_shape->get_is_routing()) {
  //       routing_net_gtl_poly_set_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
  //     }
  //   }
  // }
  // std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> cut_bg_rtree_map;
  // {
  //   for (DRCShape* drc_shape : rv_cluster.get_drc_env_shape_list()) {
  //     if (!drc_shape->get_is_routing()) {
  //       cut_bg_rtree_map[drc_shape->get_layer_idx()].insert(std::make_pair(DRCUTIL.convertToBGRectInt(drc_shape->get_rect()), drc_shape->get_net_idx()));
  //     }
  //   }
  //   for (DRCShape* drc_shape : rv_cluster.get_drc_result_shape_list()) {
  //     if (!drc_shape->get_is_routing()) {
  //       cut_bg_rtree_map[drc_shape->get_layer_idx()].insert(std::make_pair(DRCUTIL.convertToBGRectInt(drc_shape->get_rect()), drc_shape->get_net_idx()));
  //     }
  //   }
  // }
  // for (auto& [routing_layer_idx, net_gtl_poly_set_map] : routing_net_gtl_poly_set_map) {
  //   std::vector<MinimumCutRule>& minimum_cut_rule_list = routing_layer_list[routing_layer_idx].get_minimum_cut_rule_list();
  //   if (minimum_cut_rule_list.empty()) {
  //     continue;
  //   }
  //   std::vector<int32_t>& cut_layer_idx_list = routing_to_adjacent_cut_map[routing_layer_idx];
  //   int32_t above_cut_layer_idx = *std::max_element(cut_layer_idx_list.begin(), cut_layer_idx_list.end());
  //   int32_t below_cut_layer_idx = *std::min_element(cut_layer_idx_list.begin(), cut_layer_idx_list.end());
  //   for (auto& [net_idx, gtl_poly_set] : net_gtl_poly_set_map) {
  //     std::vector<GTLPolyInt> gtl_poly_list;
  //     gtl_poly_set.get_polygons(gtl_poly_list);
  //     for (GTLPolyInt& gtl_poly : gtl_poly_list) {
  //       std::vector<GTLRectInt> gtl_rect_list;
  //       gtl::get_max_rectangles(gtl_rect_list, gtl_poly);
  //       for (GTLRectInt& gtl_rect : gtl_rect_list) {
  //         PlanarRect routing_rect = DRCUTIL.convertToPlanarRect(gtl_rect);
  //         for (int32_t rule_idx = static_cast<int32_t>(minimum_cut_rule_list.size()) - 1; rule_idx >= 0; rule_idx--) {
  //           MinimumCutRule& curr_rule = minimum_cut_rule_list[rule_idx];
  //           if (routing_rect.getWidth() < curr_rule.width) {
  //             continue;
  //           }
  //           std::vector<int32_t> check_cut_layer_idx_list;
  //           if (curr_rule.has_from_above) {
  //             check_cut_layer_idx_list.push_back(above_cut_layer_idx);
  //           } else if (curr_rule.has_from_below) {
  //             check_cut_layer_idx_list.push_back(below_cut_layer_idx);
  //           } else {
  //             check_cut_layer_idx_list = cut_layer_idx_list;
  //           }
  //           std::map<int32_t, std::vector<std::vector<PlanarRect>>> cut_cut_rect_list_list_map;
  //           if (curr_rule.has_length && routing_rect.getLength() > curr_rule.length) {
  //             std::vector<GTLPolyInt> env_gtl_poly_list;
  //             {
  //               GTLPolySetInt env_gtl_poly_set = (gtl_poly - gtl_rect);
  //               env_gtl_poly_set.get_polygons(env_gtl_poly_list);
  //             }
  //             for (GTLPolyInt& env_gtl_poly : env_gtl_poly_list) {
  //               std::vector<GTLRectInt> env_gtl_rect_list;
  //               gtl::get_max_rectangles(env_gtl_rect_list, env_gtl_poly);
  //               for (int32_t cut_layer_idx : check_cut_layer_idx_list) {
  //                 std::vector<PlanarRect> cut_rect_list;
  //                 for (GTLRectInt& env_gtl_rect : env_gtl_rect_list) {
  //                   std::vector<std::pair<BGRectInt, int32_t>> cut_bg_rect_net_pair_list;
  //                   cut_bg_rtree_map[cut_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(env_gtl_rect)),
  //                                                         std::back_inserter(cut_bg_rect_net_pair_list));
  //                   for (auto& [bg_env_rect, env_net_idx] : cut_bg_rect_net_pair_list) {
  //                     PlanarRect cut_rect = DRCUTIL.convertToPlanarRect(bg_env_rect);
  //                     if (DRCUTIL.exist(cut_rect_list, cut_rect)) {
  //                       continue;
  //                     }
  //                     cut_rect_list.push_back(cut_rect);
  //                   }
  //                 }
  //                 if (cut_rect_list.empty()) {
  //                   continue;
  //                 }
  //                 std::vector<PlanarRect> cut_rect_within_list;
  //                 for (PlanarRect& cut_rect : cut_rect_list) {
  //                   auto distance = via_routing_distance(gtl_poly, cut_rect, routing_rect);
  //                   if (distance < curr_rule.distance) {
  //                     cut_rect_within_list.push_back(cut_rect);
  //                   }
  //                 }
  //                 cut_cut_rect_list_list_map[cut_layer_idx].push_back(cut_rect_within_list);
  //               }
  //             }
  //           }
  //           if (!curr_rule.has_length) {
  //             for (int32_t cut_layer_idx : check_cut_layer_idx_list) {
  //               std::vector<std::pair<BGRectInt, int32_t>> cut_bg_rect_net_pair_list;
  //               cut_bg_rtree_map[cut_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(routing_rect)),
  //               std::back_inserter(cut_bg_rect_net_pair_list)); std::vector<PlanarRect> cut_rect_list; for (auto& [bg_rect, cut_net_idx] :
  //               cut_bg_rect_net_pair_list) {
  //                 PlanarRect cut_rect = DRCUTIL.convertToPlanarRect(bg_rect);
  //                 if (!DRCUTIL.isClosedOverlap(cut_rect, routing_rect)) {
  //                   continue;
  //                 }
  //                 cut_rect_list.push_back(cut_rect);
  //               }
  //               cut_cut_rect_list_list_map[cut_layer_idx].push_back(cut_rect_list);
  //             }
  //           }
  //           for (auto& [cut_layer_idx, cut_rect_list_list] : cut_cut_rect_list_list_map) {
  //             for (std::vector<PlanarRect>& cut_rect_list : cut_rect_list_list) {
  //               for (PlanarRect& cut_rect : cut_rect_list) {
  //                 Violation violation;
  //                 violation.set_violation_type(ViolationType::kMinimumCut);
  //                 violation.set_is_routing(true);
  //                 violation.set_violation_net_set({net_idx});
  //                 violation.set_layer_idx(routing_layer_idx);
  //                 violation.set_rect(cut_rect);
  //                 violation.set_required_size(curr_rule.num_cuts);
  //                 rv_cluster.get_violation_list().push_back(violation);
  //               }
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
}

}  // namespace idrc
