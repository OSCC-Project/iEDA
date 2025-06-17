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

void RuleValidator::verifyEnclosureEdge(RVBox& rv_box)
{
  /*
  规则：
  PROPERTY LEF58_ENCLOSUREEDGE "
      VIA1
      ENCLOSUREEDGE CUTCLASS VSINGLECUT 0.015000 WIDTH 0.160500 PARALLEL 0.100000 WITHIN 0.130000 ;
      ENCLOSUREEDGE CUTCLASS VSINGLECUT ABOVE 0.010000 WIDTH 0.070500 PARALLEL 0.100000 WITHIN 0.100000 ;
      ENCLOSUREEDGE CUTCLASS VSINGLECUT ABOVE 0.005000 WIDTH 0.055500 PARALLEL 0.100000 WITHIN 0.065000 ;
      ENCLOSUREEDGE CUTCLASS VSINGLECUT ABOVE 0.005000 WIDTH 0.050500 PARALLEL 0.100000 WITHIN 0.060000 EXCEPTTWOEDGES ;
      ENCLOSUREEDGE CUTCLASS VSINGLECUT ABOVE 0.01 CONVEXCORNERS 0.120 0.060 PARALLEL 0.051 LENGTH 0.1  ;

      VIA2-6
      ENCLOSUREEDGE CUTCLASS VSINGLECUT 0.015000 WIDTH 0.160500 PARALLEL 0.100000 WITHIN 0.130000 ;
      ENCLOSUREEDGE CUTCLASS VSINGLECUT  0.010000 WIDTH 0.070500 PARALLEL 0.100000 WITHIN 0.100000 ;
      ENCLOSUREEDGE CUTCLASS VSINGLECUT  0.005000 WIDTH 0.055500 PARALLEL 0.100000 WITHIN 0.065000 ;
      ENCLOSUREEDGE CUTCLASS VSINGLECUT  0.005000 WIDTH 0.050500 PARALLEL 0.100000 WITHIN 0.060000 EXCEPTTWOEDGES ;
      ENCLOSUREEDGE CUTCLASS VSINGLECUT ABOVE 0.01 CONVEXCORNERS 0.120 0.060 PARALLEL 0.051 LENGTH 0.1  ;
      前四条可以看做是一种 rule1
      最后一条是一种      rule2目前没有违例，所以暂时不能确定
  */
  struct enclosure_edge_rule
  {
    bool has_above;
    bool has_below;  // below | above要么只会出现一个，要么都不出现
    int overhang;

    // 这条和下一条应该是同级互斥的 |
    int min_width;              // >=
    int par_length;             // >
    int par_within;             // <
    bool has_except_two_edges;  // 两边都要 这下面还有一个可选参数，t28没有就没有加了

    // // 和上面一条是 |
    // bool has_convex_corners;
    // int convex_length;      //<=
    // int adjacent_length;    //<=
    // int convex_par_within;  //<  lef文档中这个也叫做par_within
    // int length;             //>=
  };

  std::map<int, std::vector<enclosure_edge_rule>> layer_enclosure_edge_rule_list;

  std::vector<CutLayer>& cut_layer_list = DRCDM.getDatabase().get_cut_layer_list();
  for (CutLayer& cut_layer : cut_layer_list) {
    std::vector<enclosure_edge_rule> enclosure_edge_rule_list;
    for (EnclosureEdgeRule& enclosure_edge_rule : cut_layer.get_enclosure_edge_rule_list()) {
      enclosure_edge_rule_list.push_back({enclosure_edge_rule.has_above, enclosure_edge_rule.has_below, enclosure_edge_rule.overhang,
                                          enclosure_edge_rule.min_width, enclosure_edge_rule.par_length, enclosure_edge_rule.par_within,
                                          enclosure_edge_rule.has_except_two_edges});
    }
    layer_enclosure_edge_rule_list[cut_layer.get_layer_idx()] = enclosure_edge_rule_list;
  }

  // // t28在V1-V6有该规则，每层的规则是一个列表
  // // V1和V2-V6有些许不同 v2-v6的前几项没有above
  // for (int i = 1; i <= 6; i++) {
  //   enclosure_edge_rule_list.push_back({/*above below*/ false, false, /*overhang*/ 30, /*case1 */ 321, 200, 260, /*two edge*/ false});
  //   enclosure_edge_rule_list.push_back({/*above below*/ i == 1 ? true : false, false, /*overhang*/ 20, /*case1 */ 141, 200, 200, /*two edge*/ false});
  //   enclosure_edge_rule_list.push_back({/*above below*/ i == 1 ? true : false, false, /*overhang*/ 10, /*case1 */ 111, 200, 130, /*two edge*/ false});
  //   enclosure_edge_rule_list.push_back({/*above below*/ i == 1 ? true : false, false, /*overhang*/ 10, /*case1 */ 101, 200, 120, /*two edge*/ true});
  //   layer_enclosure_edge_rule_list[i] = enclosure_edge_rule_list;
  // }

// 工具类函数
#if 1
  auto get_overhang = [](const PlanarRect& cut_rect, const Segment<PlanarCoord>& metal_egde) {
    Orientation orientation = DRCUTIL.getOrientation(metal_egde.get_first(), metal_egde.get_second());
    int32_t overhang = 0;
    if (orientation == Orientation::kWest) {
      overhang = std::abs(cut_rect.get_ur_y() - metal_egde.get_first().get_y());
    } else if (orientation == Orientation::kEast) {
      overhang = std::abs(cut_rect.get_ll_y() - metal_egde.get_first().get_y());
    } else if (orientation == Orientation::kNorth) {
      overhang = std::abs(cut_rect.get_ur_x() - metal_egde.get_first().get_x());
    } else if (orientation == Orientation::kSouth) {
      overhang = std::abs(cut_rect.get_ll_x() - metal_egde.get_first().get_x());
    }
    return overhang;
  };
  auto get_convex_rule_check_rect = [](const PlanarRect& cut_rect, const Segment<PlanarCoord>& metal_egde, int32_t enlarge_size) {
    Orientation orientation = DRCUTIL.getOrientation(metal_egde.get_first(), metal_egde.get_second());
    PlanarRect check_rect;
    if (orientation == Orientation::kWest) {  // 往北
      check_rect = DRCUTIL.getEnlargedRect(cut_rect.get_ur(), cut_rect.getXSpan(), 0, 0, enlarge_size);
    } else if (orientation == Orientation::kEast) {  // 往南
      check_rect = DRCUTIL.getEnlargedRect(cut_rect.get_ll(), 0, enlarge_size, cut_rect.getXSpan(), 0);
    } else if (orientation == Orientation::kNorth) {  // 往东
      check_rect = DRCUTIL.getEnlargedRect(cut_rect.get_ur(), 0, cut_rect.getYSpan(), enlarge_size, 0);
    } else if (orientation == Orientation::kSouth) {  // 往西
      check_rect = DRCUTIL.getEnlargedRect(cut_rect.get_ll(), enlarge_size, 0, 0, cut_rect.getYSpan());
    }
    return check_rect;
  };
#endif
  // 得到基础数据
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  std::map<int32_t, std::vector<int32_t>>& routing_to_adjacent_cut_map = DRCDM.getDatabase().get_routing_to_adjacent_cut_map();
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = DRCDM.getDatabase().get_cut_to_adjacent_routing_map();
  // 使用R树查询检测
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> cut_bg_rtree_map;

  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> routing_net_gtl_poly_set_map;
  std::map<int32_t, std::vector<PlanarRect>> cut_layer_via_list;

  std::map<int32_t, std::map<int32_t, std::vector<PlanarRect>>> routing_net_rect_map;
  for (DRCShape* drc_shape : rv_box.get_drc_env_shape_list()) {
    if (!drc_shape->get_is_routing()) {
      // cut_layer_via_list[drc_shape->get_layer_idx()].push_back(drc_shape->get_rect()); env_cut貌似不检测，原因未知
      cut_bg_rtree_map[drc_shape->get_layer_idx()].insert(std::make_pair(DRCUTIL.convertToBGRectInt(drc_shape->get_rect()), drc_shape->get_net_idx()));
    } else {
      routing_net_rect_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()].push_back(drc_shape->get_rect());
    }
  }
  for (DRCShape* drc_shape : rv_box.get_drc_result_shape_list()) {
    if (!drc_shape->get_is_routing()) {
      cut_layer_via_list[drc_shape->get_layer_idx()].push_back(drc_shape->get_rect());
      cut_bg_rtree_map[drc_shape->get_layer_idx()].insert(std::make_pair(DRCUTIL.convertToBGRectInt(drc_shape->get_rect()), drc_shape->get_net_idx()));
    } else {
      routing_net_rect_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()].push_back(drc_shape->get_rect());
    }
  }
  for (auto& [routing_layer_idx, net_rect_map] : routing_net_rect_map) {
    for (auto& [net_idx, rect_list] : net_rect_map) {
      GTLPolySetInt gtl_poly_set;
      for (PlanarRect& rect : rect_list) {
        gtl_poly_set += DRCUTIL.convertToGTLRectInt(rect);
      }
      routing_net_gtl_poly_set_map[routing_layer_idx][net_idx] = gtl_poly_set;
      rect_list.clear();
      std::vector<GTLRectInt> gtl_rect_list;
      gtl::get_max_rectangles(gtl_rect_list, gtl_poly_set);
      for (GTLRectInt& gtl_rect : gtl_rect_list) {
        rect_list.push_back(DRCUTIL.convertToPlanarRect(gtl_rect));
      }
    }
  }

  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> routing_bg_rtree_map;
  for (auto& [routing_layer_idx, net_rect_map] : routing_net_rect_map) {
    for (auto& [net_idx, rect_list] : net_rect_map) {
      for (PlanarRect& rect : rect_list) {
        routing_bg_rtree_map[routing_layer_idx].insert(std::make_pair(DRCUTIL.convertToBGRectInt(rect), net_idx));
      }
    }
  }

  /*
  对于rule1，有两种方案：
  1.用via去搜索metal rect
  2.用metal rect去搜索via
  发现用第一种方案效果好
  */
  /// rule1
  for (auto& [cut_layer_idx, via_list] : cut_layer_via_list) {
    if (cut_layer_idx < 1 || cut_layer_idx > 6) {
      continue;  // 只处理V1-V6  layer CO也有该规则，先不管
    }

    std::vector<enclosure_edge_rule> enclosure_edge_rule_list = layer_enclosure_edge_rule_list[cut_layer_idx];
    std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[cut_layer_idx];
    int32_t above_routing_layer_idx = *std::max_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
    int32_t below_routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
    // 处理rule1
    for (auto& cut_rect : via_list) {
      for (int32_t routing_layer_idx : routing_layer_idx_list) {
        // query所有metal rect
        std::vector<std::pair<BGRectInt, int32_t>> metal_rect_result;
        routing_bg_rtree_map[routing_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(cut_rect)), std::back_inserter(metal_rect_result));
        // 用所有的metal rect更新当前的overhang
        int32_t west_overhang = 0;
        int32_t east_overhang = 0;
        int32_t north_overhang = 0;
        int32_t south_overhang = 0;
        for (auto& [metal_bg_rect, metal_net_idx] : metal_rect_result) {
          PlanarRect metal_rect = DRCUTIL.convertToPlanarRect(metal_bg_rect);
          if (DRCUTIL.isClosedOverlap(metal_rect, cut_rect) == false) {
            continue;
          }
          if (metal_rect.get_ll_x() <= cut_rect.get_ll_x())
            west_overhang = std::max(west_overhang, std::abs(cut_rect.get_ll_x() - metal_rect.get_ll_x()));
          if (metal_rect.get_ur_x() >= cut_rect.get_ur_x())
            east_overhang = std::max(east_overhang, std::abs(cut_rect.get_ur_x() - metal_rect.get_ur_x()));
          if (metal_rect.get_ur_y() >= cut_rect.get_ur_y())
            north_overhang = std::max(north_overhang, std::abs(cut_rect.get_ur_y() - metal_rect.get_ur_y()));
          if (metal_rect.get_ll_y() <= cut_rect.get_ll_y())
            south_overhang = std::max(south_overhang, std::abs(cut_rect.get_ll_y() - metal_rect.get_ll_y()));
        }
        for (auto& [metal_bg_rect, metal_net_idx] : metal_rect_result) {
          PlanarRect metal_rect = DRCUTIL.convertToPlanarRect(metal_bg_rect);
          if (DRCUTIL.isClosedOverlap(metal_rect, cut_rect) == false) {
            continue;
          }
          for (const Direction& metal_direction : {Direction::kHorizontal, Direction::kVertical}) {  // 两边都需要check
            int metal_rect_width = metal_direction == Direction::kHorizontal ? metal_rect.getYSpan() : metal_rect.getXSpan();
            // 根据width找到第一个适用的规则,这里取决于width的排列顺序 t28 tlef是从大到小的
            // 这里先不检测第二个规则的  !enclosure_edge_rule_list[i].has_convex_corners
            int rule_index = -1;
            for (int i = 0; i < enclosure_edge_rule_list.size(); i++) {
              if (metal_rect_width >= enclosure_edge_rule_list[i].min_width) {
                rule_index = i;
                break;
              }
            }
            if (rule_index == -1) {
              continue;
            }

            enclosure_edge_rule& cur_rule = enclosure_edge_rule_list[rule_index];
            // 判断当前是下层还是上层,有above则豁免下层，有below则豁免上层，否则就检查
            bool is_above = (routing_layer_idx == above_routing_layer_idx);
            if ((cur_rule.has_above && !is_above) || (cur_rule.has_below && is_above)) {
              continue;
            }
            std::vector<std::pair<BGRectInt, int32_t>> env_bg_rect_net_pair_list;
            {
              PlanarRect check_rect;  // 用于查找周围的矩形
              if (metal_direction == Direction::kHorizontal) {
                check_rect = DRCUTIL.getEnlargedRect(metal_rect, 0, cur_rule.par_within);
              } else {
                check_rect = DRCUTIL.getEnlargedRect(metal_rect, cur_rule.par_within, 0);
              }
              routing_bg_rtree_map[routing_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(check_rect)),
                                                            std::back_inserter(env_bg_rect_net_pair_list));
            }
            // 检测两个par区域
            PlanarRect left_par_rect;
            PlanarRect right_par_rect;
            std::set<int> left_par_net_list;
            std::set<int> right_par_net_list;
            if (metal_direction == Direction::kHorizontal) {
              left_par_rect = DRCUTIL.getEnlargedRect(metal_rect.get_ll(), 0, cur_rule.par_within, metal_rect.getXSpan(), 0);   // 往south
              right_par_rect = DRCUTIL.getEnlargedRect(metal_rect.get_ur(), metal_rect.getXSpan(), 0, 0, cur_rule.par_within);  // 往north
            } else {
              left_par_rect = DRCUTIL.getEnlargedRect(metal_rect.get_ll(), cur_rule.par_within, 0, 0, metal_rect.getYSpan());   // 往west
              right_par_rect = DRCUTIL.getEnlargedRect(metal_rect.get_ur(), 0, metal_rect.getYSpan(), cur_rule.par_within, 0);  // 往east
            }
            for (auto& [env_bg_rect, env_metal_net_idx] : env_bg_rect_net_pair_list) {
              PlanarRect env_metal_rect = DRCUTIL.convertToPlanarRect(env_bg_rect);
              // 跳过重叠矩形
              if (DRCUTIL.isClosedOverlap(metal_rect, env_metal_rect)) {
                continue;
              }
              PlanarRect env_rect = DRCUTIL.convertToPlanarRect(env_bg_rect);
              // within是小于,par是大于
              if (DRCUTIL.isOpenOverlap(env_rect, left_par_rect) && DRCUTIL.getParallelLength(env_rect, left_par_rect) > cur_rule.par_length) {
                left_par_net_list.insert(env_metal_net_idx);
              }
              if (DRCUTIL.isOpenOverlap(env_rect, right_par_rect) && DRCUTIL.getParallelLength(env_rect, right_par_rect) > cur_rule.par_length) {
                right_par_net_list.insert(env_metal_net_idx);
              }
            }
            // two edges
            if (cur_rule.has_except_two_edges && (!left_par_net_list.empty() && !right_par_net_list.empty())) {
              // 如果满足了两个区域的条件，则不需要继续检查
              continue;
            }

            std::set<std::pair<int32_t, int32_t>> violation_net_set;
            // check south west的overhang是否满足条件
            if ((metal_direction == Direction::kHorizontal && south_overhang < cur_rule.overhang)
                || (metal_direction == Direction::kVertical && west_overhang < cur_rule.overhang)) {
              for (int32_t net_idx : left_par_net_list) {
                violation_net_set.insert(std::make_pair(net_idx, metal_net_idx));
              }
            }

            // check north east的overhang是否满足条件
            if (metal_direction == Direction::kHorizontal && north_overhang < cur_rule.overhang
                || metal_direction == Direction::kVertical && east_overhang < cur_rule.overhang) {
              for (int32_t net_idx : right_par_net_list) {
                violation_net_set.insert(std::make_pair(net_idx, metal_net_idx));
              }
            }
            for (const auto& [env_metal_net_idx, metal_net_idx] : violation_net_set) {
              Violation violation;
              violation.set_violation_type(ViolationType::kEnclosureEdge);
              violation.set_is_routing(true);
              violation.set_violation_net_set({env_metal_net_idx, metal_net_idx});
              violation.set_layer_idx(below_routing_layer_idx);
              violation.set_rect(cut_rect);
              violation.set_required_size(cur_rule.overhang);
              rv_box.get_violation_list().push_back(violation);
            }
          }
        }
      }
    }
  }

  // /// rule2
  // for (auto& [routing_layer_idx, net_gtl_poly_set_map] : routing_net_gtl_poly_set_map) {
  //   std::vector<int32_t>& cut_layer_idx_list = routing_to_adjacent_cut_map[routing_layer_idx];
  //   int32_t above_cuting_layer_idx = *std::max_element(cut_layer_idx_list.begin(), cut_layer_idx_list.end());
  //   int32_t below_cuting_layer_idx = *std::min_element(cut_layer_idx_list.begin(), cut_layer_idx_list.end());
  //   // TODO 这里需要处理底层和顶层的情况 0和6的情况
  //   std::vector<enclosure_edge_rule> enclosure_edge_rule_list = layer_enclosure_edge_rule_list[below_cuting_layer_idx];  //
  //   目前只支持below的,也就是cut的above if (routing_layer_idx <= 0 || routing_layer_idx > 6) {  // 目前只care M2-M7 // 所以第一层的metal跳过
  //     continue;
  //   }
  //   for (enclosure_edge_rule& cur_rule : enclosure_edge_rule_list) {
  //     if (!cur_rule.has_convex_corners) {
  //       continue;
  //     }
  //     for (auto& [net_idx, gtl_poly_set] : net_gtl_poly_set_map) {
  //       std::vector<GTLHolePolyInt> gtl_hole_poly_list;
  //       gtl_poly_set.get(gtl_hole_poly_list);

  //       for (GTLHolePolyInt& gtl_hole_poly : gtl_hole_poly_list) {
  //         int32_t coord_size = static_cast<int32_t>(gtl_hole_poly.size());
  //         if (coord_size < 4) {
  //           continue;
  //         }

  //         std::vector<PlanarCoord> coord_list;
  //         std::vector<int32_t> edge_length_list;
  //         std::vector<bool> convex_corner_list;
  //         std::vector<Segment<PlanarCoord>> edge_list;

  //         for (auto iter = gtl_hole_poly.begin(); iter != gtl_hole_poly.end(); iter++) {
  //           coord_list.push_back(DRCUTIL.convertToPlanarCoord(*iter));
  //         }
  //         for (int32_t i = 0; i < coord_size; i++) {
  //           PlanarCoord& pre_coord = coord_list[getIdx(i - 1, coord_size)];
  //           PlanarCoord& curr_coord = coord_list[i];
  //           PlanarCoord& post_coord = coord_list[getIdx(i + 1, coord_size)];
  //           edge_length_list.push_back(DRCUTIL.getManhattanDistance(pre_coord, curr_coord));
  //           convex_corner_list.push_back(DRCUTIL.isConvexCorner(DRCUTIL.getRotation(gtl_hole_poly), pre_coord, curr_coord, post_coord));
  //           edge_list.emplace_back(pre_coord, curr_coord);
  //         }

  //         for (int32_t i = 0; i < coord_size; i++) {
  //           // 出现连续的三个凸角时才会满足条件
  //           if (!convex_corner_list[i] || !convex_corner_list[getIdx(i + 1, coord_size)] || !convex_corner_list[getIdx(i + 2, coord_size)]) {
  //             continue;
  //           }
  //           // 此时会有两种情况
  //           /**
  //            * 三凸角
  //            *
  //            *           i                                         i(length_edge)
  //            *      o---------o                               o---------o                               o
  //            *      |            o                            |                                         |            o
  //            *  i+1 |            | i+3                    i+1 |                                     i+1 |            | i+3
  //            *      |            |              (convex_edge) |                              (adj_edge) |            | (length_edge)
  //            *      o------------o                            o------------o                            o------------o
  //            *           i+2                                     i+2(adj_edge)                           i+2(convex_edge)
  //            *
  //            */
  //           for (auto [convex_edge_idx, adj_edge_idx, length_edge_idx] :
  //                {std::tuple(getIdx(i + 1, coord_size), getIdx(i + 2, coord_size), i),
  //                 std::tuple(getIdx(i + 2, coord_size), getIdx(i + 1, coord_size), getIdx(i + 3, coord_size))}) {
  //             int32_t convex_edge_length = edge_length_list[convex_edge_idx];
  //             int32_t adj_edge_length = edge_length_list[adj_edge_idx];
  //             int32_t length_edge_length = edge_length_list[length_edge_idx];
  //             if (!(cur_rule.convex_length <= convex_edge_length && cur_rule.adjacent_length <= adj_edge_length && cur_rule.length >= length_edge_length)) {
  //               continue;  // 不满足规则
  //             }
  //             // 满足长度规则，那么此时convex_edge将作为cut的overhang
  //             // 用convex边和短边构成矩形
  //             Segment<PlanarCoord>& short_edge = adj_edge_length < convex_edge_length ? edge_list[adj_edge_idx] : edge_list[convex_edge_idx];
  //             PlanarRect metal_rect = DRCUTIL.getBoundingBox(
  //                 {short_edge.get_first(), short_edge.get_second(), edge_list[convex_edge_idx].get_first(), edge_list[convex_edge_idx].get_second()});
  //             // 查询下面的cut,没有cut就跳过,理论上来说需要考虑above和below,这里为了方便先只考虑below也就是t28的情况
  //             std::vector<std::pair<BGRectInt, int32_t>> cut_bg_rect_net_pair_list;
  //             cut_bg_rtree_map[below_cuting_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(metal_rect)),
  //                                                            std::back_inserter(cut_bg_rect_net_pair_list));
  //             // 没有cut就跳过
  //             if (cut_bg_rect_net_pair_list.empty()) {
  //               continue;
  //             }
  //             // 拿到convex edge和adj edge的方向,这个方向依赖于boost对hole polygon的逆时针遍历,不然可能导致出错
  //             Orientation convex_edge_orientation = DRCUTIL.getOrientation(edge_list[convex_edge_idx].get_first(), edge_list[convex_edge_idx].get_second());
  //             Orientation adj_edge_orientation = DRCUTIL.getOrientation(edge_list[adj_edge_idx].get_first(), edge_list[adj_edge_idx].get_second());

  //             int32_t require_overhang = cur_rule.overhang;
  //             for (auto& [cut_bg_rect, cut_net_idx] : cut_bg_rect_net_pair_list) {
  //               PlanarRect cut_rect = DRCUTIL.convertToPlanarRect(cut_bg_rect);
  //               if (DRCUTIL.isInside(cut_rect, metal_rect) == false) {
  //                 continue;  // cut_rect不在metal_rect内
  //               }
  //               // 判断convex overhang是否满足，不满足跳过
  //               int32_t convex_overhang = get_overhang(cut_rect, edge_list[convex_edge_idx]);
  //               if (convex_overhang >= require_overhang) {
  //                 continue;  // overhang满足要求
  //               }
  //               int32_t adj_overhang = get_overhang(cut_rect, short_edge);  // 用来从cut往外扩展得到查询区域的
  //               // 因为要求与cut的prl不为0,所以从cut生成区域往外查
  //               PlanarRect convex_edge_check_rect
  //                   = get_convex_rule_check_rect(cut_rect, edge_list[convex_edge_idx], cur_rule.convex_par_within + convex_overhang);
  //               PlanarRect adj_edge_check_rect = get_convex_rule_check_rect(cut_rect, edge_list[adj_edge_idx], cur_rule.convex_par_within + adj_overhang);
  //               bool is_convex_edge_fulfilled = false;
  //               bool is_adj_edge_fulfilled = false;
  //               PlanarRect check_rect = DRCUTIL.getBoundingBox(
  //                   {adj_edge_check_rect.get_ll(), adj_edge_check_rect.get_ur(), convex_edge_check_rect.get_ll(), convex_edge_check_rect.get_ur()});
  //               std::vector<std::pair<BGRectInt, int32_t>> env_metal_bg_rect_net_pair_list;
  //               routing_bg_rtree_map[routing_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(check_rect)),
  //                                                             std::back_inserter(env_metal_bg_rect_net_pair_list));
  //               std::set<int32_t> env_metal_net_set;
  //               for (auto& [env_metal_bg_rect, env_metal_net_idx] : env_metal_bg_rect_net_pair_list) {
  //                 PlanarRect env_metal_rect = DRCUTIL.convertToPlanarRect(env_metal_bg_rect);
  //                 if (DRCUTIL.isClosedOverlap(env_metal_rect, metal_rect)) {
  //                   continue;  // 重叠的矩形不参与检查
  //                 }
  //                 if (DRCUTIL.isOpenOverlap(env_metal_rect, convex_edge_check_rect)) {
  //                   is_convex_edge_fulfilled = true;  // convex edge的par满足
  //                   env_metal_net_set.insert(env_metal_net_idx);
  //                 }
  //                 if (DRCUTIL.isOpenOverlap(env_metal_rect, adj_edge_check_rect)) {
  //                   is_adj_edge_fulfilled = true;  // adj edge的par满足
  //                   env_metal_net_set.insert(env_metal_net_idx);
  //                 }
  //               }
  //               if (!is_convex_edge_fulfilled || !is_adj_edge_fulfilled) {
  //                 // 两个有一个不满足都跳出
  //                 continue;
  //               }
  //               for (int32_t env_net_idx : env_metal_net_set) {
  //                 // 生成违例
  //                 Violation violation;
  //                 violation.set_violation_type(ViolationType::kEnclosureEdge);
  //                 violation.set_is_routing(true);
  //                 violation.set_violation_net_set({env_net_idx, cut_net_idx});
  //                 violation.set_layer_idx(below_cuting_layer_idx - 1);
  //                 violation.set_rect(cut_rect);
  //                 violation.set_required_size(100);
  //                 rv_box.get_violation_list().push_back(violation);
  //               }
  //               // cut
  //             }
  //           }
  //           // coord
  //         }
  //         // polygon
  //       }
  //       // polygonset
  //     }
  //     // rule
  //   }
  // }
}

}  // namespace idrc
