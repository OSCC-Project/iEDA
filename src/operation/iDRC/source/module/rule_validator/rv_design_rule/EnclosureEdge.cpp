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
      ENCLOSUREEDGE CUTCLASS VSINGLECUT 0.015000 WIDTH 0.160500 PARALLEL 0.100000 WITHIN 0.130000 ;
      ENCLOSUREEDGE CUTCLASS VSINGLECUT  0.010000 WIDTH 0.070500 PARALLEL 0.100000 WITHIN 0.100000 ;
      ENCLOSUREEDGE CUTCLASS VSINGLECUT  0.005000 WIDTH 0.055500 PARALLEL 0.100000 WITHIN 0.065000 ;
      ENCLOSUREEDGE CUTCLASS VSINGLECUT ABOVE 0.01 CONVEXCORNERS 0.120 0.060 PARALLEL 0.051 LENGTH 0.1  ;
      前三条规则视为rule1,第四条规则为rule2
  */

/// ENCLOSUREEDGE规则构造：
#if 1
  struct EnclosureEdgeRule1
  {
    int _overhang;   // overhang value
    int _minwidth;   // WIDTH value  ≥
    int _parlength;  // PARALLEL value  ＞
    int _within;     // WITHIN value     ＜
    bool _isabove;
    bool _issinglecut;
    bool _excepttwoedges;

    // 构造函数
    EnclosureEdgeRule1(int enc, int w, int prl, int wth, bool ia, bool isc, bool etd)
        : _overhang(enc), _minwidth(w), _parlength(prl), _within(wth), _isabove(ia), _issinglecut(isc), _excepttwoedges(etd)
    {
    }
  };

  struct EnclosureEdgeRule2
  {
    int _overhang;  // overhang value
    int _convexlength;
    int _adjacentlength;
    int _parWithin;
    int _length;
    bool _isabove;
    bool _issinglecut;

    // 构造函数
    EnclosureEdgeRule2(int oh, int cl, int al, int pw, int l, bool is, bool isc)
        : _overhang(oh), _convexlength(cl), _adjacentlength(al), _parWithin(pw), _length(l), _isabove(is), _issinglecut(isc)
    {
    }
  };

  // 存储所有规则的容器
  std::vector<EnclosureEdgeRule1> enclosure_edge_rules1
      = {EnclosureEdgeRule1(10, 110, 200, 120, false, true, true), EnclosureEdgeRule1(10, 111, 200, 130, false, true, false),
         EnclosureEdgeRule1(20, 141, 200, 200, false, true, false), EnclosureEdgeRule1(30, 321, 200, 260, false, true, false)};

  EnclosureEdgeRule2 enclosure_edge_rule2(20, 240, 120, 102, 200, false, true);

#endif

// 工具类函数 Rtree
#if 1
  auto addRectToRtree
      = [](std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>>& _query_tree, GTLRectInt rect, int32_t layer_idx, int32_t net_idx) {
          BGRectInt rtree_rect(BGPointInt(xl(rect), yl(rect)), BGPointInt(xh(rect), yh(rect)));
          _query_tree[layer_idx].insert(std::make_pair(rtree_rect, net_idx));
        };

  auto queryRectbyRtreeWithIntersects = [](std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>>& _query_tree, int32_t layer_idx,
                                           int32_t llx, int32_t lly, int32_t urx, int32_t ury) {
    std::set<int32_t> net_ids;
    std::vector<std::pair<BGRectInt, int32_t>> result;
    BGRectInt rect(BGPointInt(llx, lly), BGPointInt(urx, ury));
    _query_tree[layer_idx].query(bgi::intersects(rect), std::back_inserter(result));
    return result;
  };

  auto queryRectbyRtreeWithWithin = [](std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>>& _query_tree, int32_t layer_idx,
                                       int32_t llx, int32_t lly, int32_t urx, int32_t ury) {
    std::set<int32_t> net_ids;
    std::vector<std::pair<BGRectInt, int32_t>> result;
    BGRectInt rect(BGPointInt(llx, lly), BGPointInt(urx, ury));
    _query_tree[layer_idx].query(bgi::within(rect), std::back_inserter(result));
    return result;
  };

  auto queryRectbyRtreeWithOverlaps = [](std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>>& _query_tree, int32_t layer_idx,
                                         int32_t llx, int32_t lly, int32_t urx, int32_t ury) {
    std::set<int32_t> net_ids;
    std::vector<std::pair<BGRectInt, int32_t>> result;
    BGRectInt rect(BGPointInt(llx, lly), BGPointInt(urx, ury));
    _query_tree[layer_idx].query(bgi::overlaps(rect), std::back_inserter(result));
    return result;
  };

  /// 允许边界重叠
  auto queryRectbyRtreeWithCoveredBy = [](std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>>& _query_tree, int32_t layer_idx,
                                          int32_t llx, int32_t lly, int32_t urx, int32_t ury) {
    std::set<int32_t> net_ids;
    std::vector<std::pair<BGRectInt, int32_t>> result;
    BGRectInt rect(BGPointInt(llx, lly), BGPointInt(urx, ury));
    _query_tree[layer_idx].query(bgi::covered_by(rect), std::back_inserter(result));
    return result;
  };

  auto queryRectbyRtreeWithCovers = [](std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>>& _query_tree, int32_t layer_idx,
                                       int32_t llx, int32_t lly, int32_t urx, int32_t ury) {
    std::set<int32_t> net_ids;
    std::vector<std::pair<BGRectInt, int32_t>> result;
    BGRectInt rect(BGPointInt(llx, lly), BGPointInt(urx, ury));
    _query_tree[layer_idx].query(bgi::covers(rect), std::back_inserter(result));
    return result;
  };

  auto getOppositeDirection = [](gtl::direction_2d direction) {
    if (direction == gtl::NORTH) {
      return gtl::SOUTH;
    } else if (direction == gtl::SOUTH) {
      return gtl::NORTH;
    } else if (direction == gtl::EAST) {
      return gtl::WEST;
    } else {
      return gtl::EAST;
    }
  };

  auto getEdgeSegmentRect = [](const GTLRectInt& current_gtl_rect, gtl::direction_2d bloating_direction) {
    GTLRectInt segment_rect;
    if (bloating_direction == gtl::WEST) {
      segment_rect = GTLRectInt(gtl::xl(current_gtl_rect), gtl::yl(current_gtl_rect), gtl::xl(current_gtl_rect), gtl::yh(current_gtl_rect));
    } else if (bloating_direction == gtl::EAST) {
      segment_rect = GTLRectInt(gtl::xh(current_gtl_rect), gtl::yl(current_gtl_rect), gtl::xh(current_gtl_rect), gtl::yh(current_gtl_rect));
    } else if (bloating_direction == gtl::NORTH) {
      segment_rect = GTLRectInt(gtl::xl(current_gtl_rect), gtl::yh(current_gtl_rect), gtl::xh(current_gtl_rect), gtl::yh(current_gtl_rect));
    } else {
      segment_rect = GTLRectInt(gtl::xl(current_gtl_rect), gtl::yl(current_gtl_rect), gtl::xh(current_gtl_rect), gtl::yl(current_gtl_rect));
    }
    return segment_rect;
  };

  // 检查within和parallel字段是否满足规则
  auto checkParallelWithinCondition
      = [&](std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>>& _query_tree, int32_t layer_idx, GTLRectInt current_gtl_rect,
            int32_t current_rect_net_idx, gtl::direction_2d direction, EnclosureEdgeRule1 applicable_rule) -> std::pair<bool, int32_t> {
    int rule_within_value = applicable_rule._within;
    int rule_parlength_value = applicable_rule._parlength;
    auto current_planar_rect = DRCUTIL.convertToPlanarRect(current_gtl_rect);
    auto current_rect_orientation = gtl::guess_orientation(current_gtl_rect);
    auto current_rect_width_orientation = current_rect_orientation.get_perpendicular();
    int current_rect_width = gtl::delta(current_gtl_rect, current_rect_width_orientation);
    auto bloat_current_rect = current_gtl_rect;
    gtl::bloat(bloat_current_rect, direction, rule_within_value);
    std::vector<std::pair<BGRectInt, int32_t>> around_rect_result = queryRectbyRtreeWithOverlaps(
        _query_tree, layer_idx, gtl::xl(bloat_current_rect), gtl::yl(bloat_current_rect), gtl::xh(bloat_current_rect), gtl::yh(bloat_current_rect));
    for (auto& [around_bg_rect, around_rect_net_idx] : around_rect_result) {
      PlanarRect around_planar_rect = DRCUTIL.convertToPlanarRect(around_bg_rect);
      // 跳过重叠矩形
      if (DRCUTIL.isClosedOverlap(current_planar_rect, around_planar_rect)) {
        continue;
      }
      PlanarRect spacing_planar_rect = DRCUTIL.getSpacingRect(current_planar_rect, around_planar_rect);
      auto spacing_gtl_rect = DRCUTIL.convertToGTLRectInt(spacing_planar_rect);
      int32_t within_value = gtl::delta(spacing_gtl_rect, current_rect_width_orientation);
      int32_t parallel_value = gtl::delta(spacing_gtl_rect, current_rect_orientation);
      if (within_value < rule_within_value && parallel_value > rule_parlength_value) {
        return {true, around_rect_net_idx};
      }
    }
    return {false, -1};
  };
  /*
判断within和parallel字段是否满足规则
  */

  auto checkRule2Condiction = [](PlanarRect a, PlanarRect b, gtl::direction_2d direction, int rule_whthin_value, int rule_parallel_value) {
    int within_value;
    int parallel_value;

    // 利用方向枚举的规律简化计算
    // WEST=0, EAST=1 - 水平方向；SOUTH=2, NORTH=3 - 垂直方向
    bool is_vertical = direction.to_int() >= 2;
    bool is_positive = direction.to_int() & 1;  // EAST=1, NORTH=3 是正方向

    if (is_vertical) {
      // 垂直方向 - SOUTH/NORTH
      within_value = is_positive ? (b.get_ll_y() - a.get_ur_y()) :  // NORTH
                         (a.get_ll_y() - b.get_ur_y());             // SOUTH
      parallel_value = std::min(a.get_ur_x(), b.get_ur_x()) - std::max(a.get_ll_x(), b.get_ll_x());
    } else {
      // 水平方向 - WEST/EAST
      within_value = is_positive ? (b.get_ll_x() - a.get_ur_x()) :  // EAST
                         (a.get_ll_x() - b.get_ur_x());             // WEST
      parallel_value = std::min(a.get_ur_y(), b.get_ur_y()) - std::max(a.get_ll_y(), b.get_ll_y());
    }

    return (within_value < rule_whthin_value && parallel_value > rule_parallel_value);
  };
#endif
  // 得到基础数据
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  std::map<int32_t, std::vector<int32_t>>& routing_to_adjacent_cut_map = DRCDM.getDatabase().get_routing_to_adjacent_cut_map();
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = DRCDM.getDatabase().get_cut_to_adjacent_routing_map();
  // 使用R树查询检测
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> routing_layer_all_query_tree;
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> cut_layer_all_query_tree;

  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> routing_layer_net_gtl_all_poly_set;
  std::map<int32_t, std::map<int32_t, std::vector<GTLRectInt>>> routing_layer_net_gtl_all_maxrect_list;
  std::map<int32_t, std::map<int32_t, std::vector<GTLRectInt>>> cut_layer_net_gtl_res_maxrect_list;
  std::map<int32_t, std::vector<GTLRectInt>> cut_layer_via_list;

  std::map<int32_t, std::map<int32_t, int32_t>> routing_layer_net_rect_width;

  {
    for (DRCShape* rect : rv_box.get_drc_result_shape_list()) {
      if (rect->get_net_idx() == -1) {
        continue;
      }
      if (rect->get_is_routing() == false) {
        cut_layer_via_list[rect->get_layer_idx()].push_back(GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y()));
        cut_layer_net_gtl_res_maxrect_list[rect->get_layer_idx()][rect->get_net_idx()].push_back(
            GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y()));
        addRectToRtree(cut_layer_all_query_tree, GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y()), rect->get_layer_idx(),
                       rect->get_net_idx());
      } else {
        int32_t layer_idx = rect->get_layer_idx();
        int32_t net_idx = rect->get_net_idx();
        routing_layer_net_gtl_all_poly_set[layer_idx][net_idx] += DRCUTIL.convertToGTLRectInt(rect->get_rect());
      }
    }

    // 用max rect作为被查找的
    for (auto& [routing_layer_idx, net_all_gtl_poly_set] : routing_layer_net_gtl_all_poly_set) {
      for (auto& [net_idx, res_all_poly_set] : net_all_gtl_poly_set) {
        std::vector<GTLRectInt> rect_list;
        gtl::get_max_rectangles(rect_list, res_all_poly_set);
        for (GTLRectInt& rect : rect_list) {
          routing_layer_net_gtl_all_maxrect_list[routing_layer_idx][net_idx].push_back(rect);
          addRectToRtree(routing_layer_all_query_tree, rect, routing_layer_idx, net_idx);
        }
      }
    }
  }

  /*
  对于rule1，有两种方案：
  1.用via去搜索metal rect
  2.用metal rect去搜索via
  发现用第一种方案效果好
  */
// rule1
#if 1
  for (auto& [cut_layer_idx, via_list] : cut_layer_via_list) {
    std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[cut_layer_idx];
    for (auto& via_gtl_rect : via_list) {
      PlanarRect via_planar_rect = DRCUTIL.convertToPlanarRect(via_gtl_rect);
      for (int32_t routing_layer_idx : routing_layer_idx_list) {
        std::vector<std::pair<BGRectInt, int32_t>> metal_rect_result = queryRectbyRtreeWithIntersects(
            routing_layer_all_query_tree, routing_layer_idx, gtl::xl(via_gtl_rect), gtl::yl(via_gtl_rect), gtl::xh(via_gtl_rect), gtl::yh(via_gtl_rect));

        for (auto& [metal_bg_rect, metal_net_idx] : metal_rect_result) {
          PlanarRect current_planar_rect = DRCUTIL.convertToPlanarRect(metal_bg_rect);
          GTLRectInt current_gtl_rect = DRCUTIL.convertToGTLRectInt(metal_bg_rect);
          if (DRCUTIL.isOpenOverlap(via_planar_rect, current_planar_rect) == false) {
            continue;
          }
          auto current_rect_orientation = gtl::guess_orientation(current_gtl_rect);
          auto current_rect_width_orientation = current_rect_orientation.get_perpendicular();
          int current_rect_width = gtl::delta(current_gtl_rect, current_rect_width_orientation);

          // 根据width找到第一个适用的规则
          int rule_index = -1;
          for (int i = enclosure_edge_rules1.size() - 1; i >= 0; i--) {
            if (current_rect_width >= enclosure_edge_rules1[i]._minwidth) {
              rule_index = i;
              break;
            }
          }
          if (rule_index == -1) {
            continue;
          }

          // 使用找到的规则
          auto& applicable_rule = enclosure_edge_rules1[rule_index];
          auto checkWithDirection = [&](const gtl::direction_2d& direction) {
            // 检查 within 和 parallel
            auto check_condition_result
                = checkParallelWithinCondition(routing_layer_all_query_tree, routing_layer_idx, current_gtl_rect, metal_net_idx, direction, applicable_rule);
            if (check_condition_result.first == true) {
              // 判断是否豁免
              auto opposite_direction = getOppositeDirection(direction);
              if (applicable_rule._excepttwoedges == true
                  && checkParallelWithinCondition(routing_layer_all_query_tree, routing_layer_idx, current_gtl_rect, metal_net_idx, opposite_direction,
                                                  applicable_rule)
                             .first
                         == true) {
                return;
              }
              /// 符合条件，检测上面的通孔
              GTLRectInt edge_rect = getEdgeSegmentRect(current_gtl_rect, direction);
              PlanarRect edge_rect_planar = DRCUTIL.convertToPlanarRect(edge_rect);
              auto check_overhang_rect = DRCUTIL.getSpacingRect(via_planar_rect, edge_rect_planar);
              auto check_overhang_gtl_rect = DRCUTIL.convertToGTLRectInt(check_overhang_rect);
              if (gtl::delta(check_overhang_gtl_rect, current_rect_width_orientation) < applicable_rule._overhang) {
                Violation violation;
                violation.set_violation_type(ViolationType::kEnclosureEdge);
                violation.set_is_routing(true);
                violation.set_violation_net_set({metal_net_idx, check_condition_result.second});
                violation.set_layer_idx(cut_layer_idx - 1);
                violation.set_rect(via_planar_rect);
                violation.set_required_size(applicable_rule._overhang);
                rv_box.get_violation_list().push_back(violation);
              }
            }
          };

          // 根据矩形方向选择检查方向
          if (current_rect_width_orientation == gtl::HORIZONTAL) {
            for (auto& check_direction : {gtl::WEST, gtl::EAST}) {
              checkWithDirection(check_direction);
            }
          } else {
            for (auto& check_direction : {gtl::NORTH, gtl::SOUTH}) {
              checkWithDirection(check_direction);
            }
          }
        }
      }
    }
  }
#endif

/// rule2
#if 1
  for (auto& [routing_layer_idx, net_gtl_poly_set_map] : routing_layer_net_gtl_all_poly_set) {
    std::vector<int32_t>& cut_layer_idx_list = routing_to_adjacent_cut_map[routing_layer_idx];
    for (auto& [net_idx, gtl_poly_set] : net_gtl_poly_set_map) {
      std::vector<GTLHolePolyInt> gtl_hole_poly_list;
      std::vector<gtl::polygon_with_holes_data<int32_t>> view_poly_list;
      gtl_poly_set.get(gtl_hole_poly_list);
      gtl_poly_set.get(view_poly_list);

      for (GTLHolePolyInt& gtl_hole_poly : gtl_hole_poly_list) {
        int32_t coord_size = static_cast<int32_t>(gtl_hole_poly.size());
        if (coord_size < 4) {
          continue;
        }

        std::vector<PlanarCoord> coord_list;
        std::vector<int32_t> edge_length_list;
        std::vector<bool> convex_corner_list;

        {
          for (auto iter = gtl_hole_poly.begin(); iter != gtl_hole_poly.end(); iter++) {
            coord_list.push_back(DRCUTIL.convertToPlanarCoord(*iter));
          }
          for (int32_t i = 0; i < coord_size; i++) {
            PlanarCoord& pre_coord = coord_list[getIdx(i - 1, coord_size)];
            PlanarCoord& curr_coord = coord_list[i];
            PlanarCoord& post_coord = coord_list[getIdx(i + 1, coord_size)];
            edge_length_list.push_back(DRCUTIL.getManhattanDistance(pre_coord, curr_coord));
            convex_corner_list.push_back(DRCUTIL.isConvexCorner(DRCUTIL.getRotation(gtl_hole_poly), pre_coord, curr_coord, post_coord));
          }
        }
        for (int32_t i = 0; i < coord_size; i++) {
          int32_t pre_i = getIdx(i - 1, coord_size);
          int32_t pre_pre_i = getIdx(i - 2, coord_size);
          int32_t post_i = getIdx(i + 1, coord_size);
          auto edge_direction = DRCUTIL.getDirection(coord_list[i], coord_list[i - 1]);
          // 第一级条件：当前边的边长小于等于convexLength，并且当前边两个顶点是凸角
          // 第二级条件：当前边的一条相邻边，边长小于等于adjacentLength，并且相邻边的两个顶点是凸角；当前边的另一条相邻边，边长大于等于length
          if (edge_length_list[i] <= enclosure_edge_rule2._convexlength && convex_corner_list[i] == true && convex_corner_list[pre_i] == true) {
            if ((edge_length_list[pre_i] <= enclosure_edge_rule2._adjacentlength && convex_corner_list[pre_pre_i] == true
                 && edge_length_list[post_i] >= enclosure_edge_rule2._length)
                || (edge_length_list[post_i] <= enclosure_edge_rule2._adjacentlength && convex_corner_list[post_i] == true
                    && edge_length_list[pre_i] >= enclosure_edge_rule2._length)) {
              // 获取当前边的起点、终点、中点
              PlanarCoord edge_start_point = coord_list[pre_i];
              PlanarCoord edge_end_point = coord_list[i];
              PlanarCoord edge_middle_point(edge_start_point.get_x() + (edge_end_point.get_x() - edge_start_point.get_x()) / 2,
                                            edge_start_point.get_y() + (edge_end_point.get_y() - edge_start_point.get_y()) / 2);

              // 获取相邻边(长度较短的)的长度和方向
              bool use_prev_edge = edge_length_list[pre_i] <= edge_length_list[post_i];
              int adjacent_length = use_prev_edge ? edge_length_list[pre_i] : edge_length_list[post_i];
              PlanarCoord adjacent_start_point, adjacent_end_point;

              if (use_prev_edge) {
                adjacent_start_point = coord_list[pre_pre_i];
                adjacent_end_point = coord_list[pre_i];
              } else {
                adjacent_start_point = coord_list[i];
                adjacent_end_point = coord_list[post_i];
              }

              // 以当前边和相邻边构造矩形
              int ll_x, ll_y, ur_x, ur_y;
              gtl::direction_2d bloating_direction_1;  // 第一个膨胀搜索方向
              gtl::direction_2d bloating_direction_2;  // 第二个膨胀搜索方向
              int32_t adjacent_net_idx = -1;
              {
                if (edge_direction == Direction::kHorizontal) {
                  // 当前边是水平的，相邻边是垂直的
                  ll_x = std::min(edge_start_point.get_x(), edge_end_point.get_x());
                  ur_x = std::max(edge_start_point.get_x(), edge_end_point.get_x());

                  // 确定垂直方向的延伸
                  bool extend_down = (coord_list[pre_pre_i].get_y() < edge_start_point.get_y()) || (coord_list[post_i].get_y() < edge_end_point.get_y());

                  if (extend_down) {
                    ur_y = edge_start_point.get_y();
                    ll_y = ur_y - adjacent_length;
                    bloating_direction_1 = gtl::NORTH;  // 矩形向下延伸，膨胀方向是向上
                  } else {
                    ll_y = edge_start_point.get_y();
                    ur_y = ll_y + adjacent_length;
                    bloating_direction_1 = gtl::SOUTH;  // 矩形向上延伸，膨胀方向是向下
                  }

                  // 确定第二个膨胀方向 - 根据相邻边在当前边上的投影
                  if (adjacent_start_point.get_x() < edge_middle_point.get_x()) {
                    bloating_direction_2 = gtl::EAST;
                  } else {
                    bloating_direction_2 = gtl::WEST;
                  }
                } else {
                  // 当前边是垂直的，相邻边是水平的
                  ll_y = std::min(edge_start_point.get_y(), edge_end_point.get_y());
                  ur_y = std::max(edge_start_point.get_y(), edge_end_point.get_y());

                  // 确定水平方向的延伸
                  bool extend_left = (coord_list[pre_pre_i].get_x() < edge_start_point.get_x()) || (coord_list[post_i].get_x() < edge_end_point.get_x());

                  if (extend_left) {
                    ur_x = edge_start_point.get_x();
                    ll_x = ur_x - adjacent_length;
                    bloating_direction_1 = gtl::EAST;  // 矩形向左延伸，膨胀方向是向右
                  } else {
                    ll_x = edge_start_point.get_x();
                    ur_x = ll_x + adjacent_length;
                    bloating_direction_1 = gtl::WEST;  // 矩形向右延伸，膨胀方向是向左
                  }

                  // 确定第二个膨胀方向 - 根据相邻边在当前边上的投影
                  if (adjacent_start_point.get_y() < edge_middle_point.get_y()) {
                    bloating_direction_2 = gtl::NORTH;
                  } else {
                    bloating_direction_2 = gtl::SOUTH;
                  }
                }
              }

              PlanarRect query_planar_rect(ll_x, ll_y, ur_x, ur_y);
              GTLRectInt query_gtl_rect = DRCUTIL.convertToGTLRectInt(query_planar_rect);

              for (int32_t cut_layer_idx : cut_layer_idx_list) {
                // 判断rule2的ABOVE字段
                if (routing_layer_idx == 1 && cut_layer_idx == 0) {
                  continue;
                }
                // 搜索当前矩形内的via，无via则提前返回
                std::vector<std::pair<BGRectInt, int32_t>> vias_in_current_rect
                    = queryRectbyRtreeWithWithin(cut_layer_all_query_tree, cut_layer_idx, gtl::xl(query_gtl_rect), gtl::yl(query_gtl_rect),
                                                 gtl::xh(query_gtl_rect), gtl::yh(query_gtl_rect));
                if (vias_in_current_rect.empty()) {
                  continue;
                }

                // 检查特定方向的相邻shape
                auto checkParallelWithinCondition = [&](gtl::direction_2d direction) -> std::pair<bool, int32_t> {
                  GTLRectInt bloated_rect = query_gtl_rect;
                  gtl::bloat(bloated_rect, direction, enclosure_edge_rule2._parWithin);

                  std::vector<std::pair<BGRectInt, int32_t>> around_results
                      = queryRectbyRtreeWithIntersects(routing_layer_all_query_tree, routing_layer_idx, gtl::xl(bloated_rect), gtl::yl(bloated_rect),
                                                       gtl::xh(bloated_rect), gtl::yh(bloated_rect));

                  for (auto& [around_bg_rect, around_rect_net_idx] : around_results) {
                    PlanarRect around_planar_rect = DRCUTIL.convertToPlanarRect(around_bg_rect);

                    // 跳过同网络或重叠的矩形
                    if (net_idx == around_rect_net_idx || DRCUTIL.isClosedOverlap(query_planar_rect, around_planar_rect)) {
                      continue;
                    }

                    // 检查规则条件
                    if (checkRule2Condiction(query_planar_rect, around_planar_rect, direction, enclosure_edge_rule2._parWithin, 0)) {
                      return {true, around_rect_net_idx};
                    }
                  }

                  return {false, -1};
                };

                // 检查规则的overhang字段
                auto isViaViolating = [&](const PlanarRect& via_rect) -> bool {
                  return (bloating_direction_1 == gtl::NORTH && abs(query_planar_rect.get_ur_y() - via_rect.get_ur_y()) < enclosure_edge_rule2._overhang)
                         || (bloating_direction_1 == gtl::SOUTH && abs(via_rect.get_ll_y() - query_planar_rect.get_ll_y()) < enclosure_edge_rule2._overhang)
                         || (bloating_direction_1 == gtl::EAST && abs(query_planar_rect.get_ur_x() - via_rect.get_ur_x()) < enclosure_edge_rule2._overhang)
                         || (bloating_direction_1 == gtl::WEST && abs(via_rect.get_ll_x() - query_planar_rect.get_ll_x()) < enclosure_edge_rule2._overhang);
                };

                // 检查两个方向
                auto [check_condition1, _] = checkParallelWithinCondition(bloating_direction_1);
                if (check_condition1) {
                  auto [check_condition2, adjacent_net_idx] = checkParallelWithinCondition(bloating_direction_2);

                  if (check_condition2) {
                    // 检查所有vias是否违反规则
                    for (auto& via_in_current_rect : vias_in_current_rect) {
                      PlanarRect via_planar_Rect = DRCUTIL.convertToPlanarRect(via_in_current_rect.first);

                      if (isViaViolating(via_planar_Rect)) {
                        Violation violation;
                        violation.set_violation_type(ViolationType::kEnclosureEdge);
                        violation.set_is_routing(true);
                        violation.set_violation_net_set({net_idx, adjacent_net_idx});
                        violation.set_layer_idx(cut_layer_idx - 1);
                        violation.set_rect(via_planar_Rect);
                        violation.set_required_size(enclosure_edge_rule2._overhang);
                        rv_box.get_violation_list().push_back(violation);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
#endif
}

}  // namespace idrc
