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

void RuleValidator::verifyCornerFillSpacing(RVBox& rv_box)
{
#if 1  // 函数定义
  auto getIdx = [](int32_t idx, int32_t coord_size) { return (idx + coord_size) % coord_size; };
  auto addRectToRtree
      = [](std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>>& _query_tree, GTLRectInt rect, int32_t layer_idx, int32_t net_idx) {
          BGRectInt rtree_rect(BGPointInt(xl(rect), yl(rect)), BGPointInt(xh(rect), yh(rect)));
          _query_tree[layer_idx].insert(std::make_pair(rtree_rect, net_idx));
        };
  auto queryNetIdbyRtree = [](std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>>& _query_tree, int32_t layer_idx, int32_t llx,
                              int32_t lly, int32_t urx, int32_t ury) {
    std::set<int32_t> net_ids;
    std::vector<std::pair<BGRectInt, int32_t>> result;
    BGRectInt rect(BGPointInt(llx, lly), BGPointInt(urx, ury));
    _query_tree[layer_idx].query(bgi::intersects(rect), std::back_inserter(result));
    for (auto& pair : result) {
      net_ids.insert(pair.second);
    }
    return net_ids;
  };
  auto queryRectbyRtree = [](std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>>& _query_tree, int32_t layer_idx, int32_t llx,
                             int32_t lly, int32_t urx, int32_t ury) {
    std::set<int32_t> net_ids;
    std::vector<std::pair<BGRectInt, int32_t>> result;
    BGRectInt rect(BGPointInt(llx, lly), BGPointInt(urx, ury));
    _query_tree[layer_idx].query(bgi::intersects(rect), std::back_inserter(result));
    return result;
  };
  auto get_edge_direction = [](const PlanarCoord& p1, const PlanarCoord& p2) {
    return p1.get_x() == p2.get_x() ? (p2.get_y() > p1.get_y() ? gtl::NORTH : gtl::SOUTH) : (p2.get_x() > p1.get_x() ? gtl::EAST : gtl::WEST);
  };
#endif
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> layer_query_tree;
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> layer_used_query_tree;

  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> routing_net_gtl_poly_set_map;
  std::map<int32_t, GTLPolySetInt> routing_violation_gtl_poly_set_map;
  for (DRCShape* drc_shape : rv_box.get_drc_env_shape_list()) {
    if (!drc_shape->get_is_routing() || drc_shape->get_net_idx() == -1) {
      continue;
    }
    routing_net_gtl_poly_set_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
  }
  for (DRCShape* drc_shape : rv_box.get_drc_result_shape_list()) {
    if (!drc_shape->get_is_routing() || drc_shape->get_net_idx() == -1) {
      continue;
    }
    routing_net_gtl_poly_set_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
  }
  // 用max rect作为被查找的
  for (auto& [routing_layer_idx, net_gtl_poly_set_map] : routing_net_gtl_poly_set_map) {
    for (auto& [net_idx, gtl_poly_set] : net_gtl_poly_set_map) {
      std::vector<GTLRectInt> rect_list;
      gtl::get_max_rectangles(rect_list, gtl_poly_set);
      for (GTLRectInt& rect : rect_list) {
        addRectToRtree(layer_query_tree, rect, routing_layer_idx, net_idx);
      }
    }
  }
  // 规则值定义
  int32_t corner_fill_spacing = 50 * 2;
  int32_t edge_length_a = 50 * 2;
  int32_t edge_length_b = 120 * 2;
  int32_t adjacent_eol_length = 60 * 2;
  // 只有M3-M7层有
  std::vector<int32_t> corner_fill_layer_list = {2, 3, 4, 5, 6};

  for (auto& [routing_layer_idx, net_gtl_poly_set_map] : routing_net_gtl_poly_set_map) {
    if (!DRCUTIL.exist(corner_fill_layer_list, routing_layer_idx)) {
      continue;  // 跳过没有的层
    }
    for (auto& [net_idx, gtl_poly_set] : net_gtl_poly_set_map) {
      std::vector<GTLHolePolyInt> gtl_hole_poly_list;
      gtl_poly_set.get_polygons(gtl_hole_poly_list);
      for (GTLHolePolyInt& gtl_hole_poly : gtl_hole_poly_list) {
        int32_t coord_size = static_cast<int32_t>(gtl_hole_poly.size());
        if (coord_size < 4) {
          continue;
        }
        std::vector<PlanarCoord> coord_list;
        for (auto iter = gtl_hole_poly.begin(); iter != gtl_hole_poly.end(); iter++) {
          coord_list.push_back(DRCUTIL.convertToPlanarCoord(*iter));
        }
        std::vector<int32_t> edge_length_list;
        std::vector<bool> convex_corner_list;
        Rotation rotation = DRCUTIL.getRotation(gtl_hole_poly);
        for (int32_t i = 0; i < coord_size; i++) {
          PlanarCoord& pre_coord = coord_list[getIdx(i - 1, coord_size)];
          PlanarCoord& curr_coord = coord_list[i];
          PlanarCoord& post_coord = coord_list[getIdx(i + 1, coord_size)];
          edge_length_list.push_back(DRCUTIL.getManhattanDistance(pre_coord, curr_coord));
          convex_corner_list.push_back(DRCUTIL.isConvexCorner(rotation, pre_coord, curr_coord, post_coord));
        }
        // 拿到所有的EOL边
        std::vector<int32_t> eol_edge_index;
        for (int32_t i = 0; i < coord_size; i++) {
          int32_t post_coord_idx = getIdx(i - 1, coord_size);
          int32_t curr_coord_idx = i;
          if (convex_corner_list[post_coord_idx] && convex_corner_list[curr_coord_idx]) {
            eol_edge_index.push_back(i);
          }
        }
        // 依次检测每个凹角
        for (int32_t i = 0; i < coord_size; i++) {
          // debug
          if (coord_list[i] == PlanarCoord(481250, 173900)) {
            int32_t debug = 0;
          }
          if (convex_corner_list[i]) {
            continue;
          }
          int32_t pre_index = getIdx(i - 1, coord_size);
          int32_t post_index = getIdx(i + 1, coord_size);
          int32_t post_post_index = getIdx(i + 2, coord_size);

          PlanarCoord& corner_point = coord_list[i];
          int32_t corner_edge_a_length = edge_length_list[i];
          int32_t corner_edge_a_adj_length = edge_length_list[pre_index];
          int32_t corner_edge_b_length = edge_length_list[post_index];
          int32_t corner_edge_b_adj_length = edge_length_list[post_post_index];
          // 对应lef所说的两种判断情况
          if ((corner_edge_a_length < edge_length_a && corner_edge_b_length < edge_length_b && corner_edge_b_adj_length < adjacent_eol_length
               && DRCUTIL.exist(eol_edge_index, post_post_index))
              || (corner_edge_b_length < edge_length_a && corner_edge_a_length < edge_length_b && corner_edge_a_adj_length < adjacent_eol_length
                  && DRCUTIL.exist(eol_edge_index, pre_index))) {
            gtl::direction_2d corner_edge_a_dir = get_edge_direction(coord_list[pre_index], coord_list[i]);
            gtl::direction_2d corner_edge_b_dir = get_edge_direction(coord_list[i], coord_list[post_index]);
            GTLRectInt gtl_check_rect(corner_point.get_x(), corner_point.get_y(), corner_point.get_x(), corner_point.get_y());
            // 根据绕行方形的方向进行膨胀
            gtl::direction_1d poly_winding = gtl::winding(gtl_hole_poly);  // LOW是顺时针，HIGH是逆时针
            if (poly_winding == gtl::LOW) {
              gtl::bloat(gtl_check_rect, corner_edge_a_dir.left(), corner_fill_spacing + corner_edge_b_length);
              gtl::bloat(gtl_check_rect, corner_edge_b_dir.left(), corner_fill_spacing + corner_edge_a_length);
            } else {
              gtl::bloat(gtl_check_rect, corner_edge_a_dir.right(), corner_fill_spacing + corner_edge_b_length);
              gtl::bloat(gtl_check_rect, corner_edge_b_dir.right(), corner_fill_spacing + corner_edge_a_length);
            }
            // 首先求出corner位于哪个方位：1：右下 2：左下 3：左上 4：右上
            int32_t corner_position = 0;
            PlanarCoord right_bottom = PlanarCoord(gtl::xh(gtl_check_rect), gtl::yl(gtl_check_rect));
            PlanarCoord left_bottom = PlanarCoord(gtl::xl(gtl_check_rect), gtl::yl(gtl_check_rect));
            PlanarCoord left_top = PlanarCoord(gtl::xl(gtl_check_rect), gtl::yh(gtl_check_rect));
            PlanarCoord right_top = PlanarCoord(gtl::xh(gtl_check_rect), gtl::yh(gtl_check_rect));
            if (corner_point == right_bottom) {
              corner_position = 1;
            } else if (corner_point == left_bottom) {
              corner_position = 2;
            } else if (corner_point == left_top) {
              corner_position = 3;
            } else if (corner_point == right_top) {
              corner_position = 4;
            }

            // 把相关的poly的rect加入到最终的查询树中
            std::vector<GTLRectInt> gtl_hole_poly_rect_list;
            gtl::get_max_rectangles(gtl_hole_poly_rect_list, gtl_hole_poly);
            for (GTLRectInt& hole_poly_rect : gtl_hole_poly_rect_list) {
              addRectToRtree(layer_used_query_tree, hole_poly_rect, routing_layer_idx, net_idx);  // 只加入用到的rect到rtree中
            }
            // 找到有交集的max rect
            std::vector<std::pair<BGRectInt, int32_t>> overlap_rect_result
                = queryRectbyRtree(layer_query_tree, routing_layer_idx, gtl::xl(gtl_check_rect) + 1, gtl::yl(gtl_check_rect) + 1, gtl::xh(gtl_check_rect) - 1,
                                   gtl::yh(gtl_check_rect) - 1);  // 从原图形中查找对应的rect,不要贴边的
            // 在被查询范围内的rect也加入最终的查询树中
            std::vector<PlanarRect> overlap_rect_list;
            for (auto& [bgrect, q_net_idx] : overlap_rect_result) {
              PlanarRect violation_rect = DRCUTIL.convertToPlanarRect(bgrect);
              GTLRectInt gtl_violation_rect = DRCUTIL.convertToGTLRectInt(bgrect);
              if (gtl::area(gtl_violation_rect & gtl_hole_poly) == violation_rect.getArea()) {
                continue;  // 同一个polygon的忽略
              }
              overlap_rect_list.push_back(violation_rect);
              addRectToRtree(layer_used_query_tree, gtl_violation_rect, routing_layer_idx, q_net_idx);  // 只加入用到的rect到rtree中
            }
            for (PlanarRect& overlap_rect : overlap_rect_list) {
              PlanarRect inside_corner_rect = DRCUTIL.getOverlap(overlap_rect, DRCUTIL.convertToPlanarRect(gtl_check_rect));
              if (inside_corner_rect.getArea() <= 0) {
                continue;
              }
              /*
              重叠区域如下：
              +----------------+
              |       |        |
              |   a   |    d   |
              |-------+--------|
              |       |        |
              |   c   |    b   |
              +----------------+
              c是corner的区域，a，b是竖直水平拓展的区域，d是斜对角的区域
              */
              int32_t llx = gtl::xl(gtl_check_rect);
              int32_t lly = gtl::yl(gtl_check_rect);
              int32_t urx = gtl::xh(gtl_check_rect);
              int32_t ury = gtl::yh(gtl_check_rect);
              // 拿到d区，计算距离，处理corner的情况，corner的地方可能有大于corner spacing的情况
              PlanarRect d_rect;
              PlanarCoord d_point;
              PlanarCoord violation_point;  // 构成vio rect的另一个点,要么就是对角点，要么就是另一种
              if (corner_position == 1) {
                // 右下
                d_rect = PlanarRect(llx, ury - corner_fill_spacing, llx + corner_fill_spacing, ury);
                d_point = PlanarCoord(llx + corner_fill_spacing, ury - corner_fill_spacing);
                violation_point = PlanarCoord(inside_corner_rect.get_ur_x(), inside_corner_rect.get_ll_y());
                // 判断A区和B区
                if (corner_point.get_y() == inside_corner_rect.get_ll_y()) {  // A区
                  violation_point = PlanarCoord(inside_corner_rect.get_ur_x(), std::min(inside_corner_rect.get_ur_y(), d_point.get_y()));
                }
                if (corner_point.get_x() == inside_corner_rect.get_ur_x()) {  // B区
                  violation_point = PlanarCoord(std::max(inside_corner_rect.get_ll_x(), d_point.get_x()), inside_corner_rect.get_ll_y());
                }
              } else if (corner_position == 2) {
                // 左下
                d_rect = PlanarRect(urx - corner_fill_spacing, ury - corner_fill_spacing, urx, ury);
                d_point = PlanarCoord(urx - corner_fill_spacing, ury - corner_fill_spacing);
                violation_point = PlanarCoord(inside_corner_rect.get_ll_x(), inside_corner_rect.get_ll_y());
                // 判断A区和B区
                if (corner_point.get_x() == inside_corner_rect.get_ll_x()) {  // A区
                  violation_point = PlanarCoord(std::min(inside_corner_rect.get_ur_x(), d_point.get_x()), inside_corner_rect.get_ll_y());
                }
                if (corner_point.get_y() == inside_corner_rect.get_ll_y()) {  // B区
                  violation_point = PlanarCoord(inside_corner_rect.get_ll_x(), std::min(inside_corner_rect.get_ur_y(), d_point.get_y()));
                }
              } else if (corner_position == 3) {
                // 左上
                d_rect = PlanarRect(urx - corner_fill_spacing, lly, urx, lly + corner_fill_spacing);
                d_point = PlanarCoord(urx - corner_fill_spacing, lly + corner_fill_spacing);
                violation_point = PlanarCoord(inside_corner_rect.get_ll_x(), inside_corner_rect.get_ur_y());
                // 判断A区和B区
                if (corner_point.get_x() == inside_corner_rect.get_ll_x()) {  // A区
                  violation_point = PlanarCoord(std::min(inside_corner_rect.get_ur_x(), d_point.get_x()), inside_corner_rect.get_ur_y());
                }
                if (corner_point.get_y() == inside_corner_rect.get_ur_y()) {  // B区
                  violation_point = PlanarCoord(inside_corner_rect.get_ll_x(), std::max(inside_corner_rect.get_ll_y(), d_point.get_y()));
                }
              } else if (corner_position == 4) {
                // 右上
                d_rect = PlanarRect(llx, lly, llx + corner_fill_spacing, lly + corner_fill_spacing);
                d_point = PlanarCoord(llx + corner_fill_spacing, lly + corner_fill_spacing);
                violation_point = PlanarCoord(inside_corner_rect.get_ur_x(), inside_corner_rect.get_ur_y());
                // 判断A区和B区
                if (corner_point.get_x() == inside_corner_rect.get_ur_x()) {  // A区
                  violation_point = PlanarCoord(std::max(inside_corner_rect.get_ll_x(), d_point.get_x()), inside_corner_rect.get_ur_y());
                }
                if (corner_point.get_y() == inside_corner_rect.get_ur_y()) {  // B区
                  violation_point = PlanarCoord(inside_corner_rect.get_ur_x(), std::max(inside_corner_rect.get_ll_y(), d_point.get_y()));
                }
              }
              if (DRCUTIL.isInside(d_rect, inside_corner_rect)) {
                int32_t distance
                    = gtl::euclidean_distance(GTLPointInt(violation_point.get_x(), violation_point.get_y()), GTLPointInt(d_point.get_x(), d_point.get_y()));
                if (distance >= corner_fill_spacing) {
                  continue;  // corner case跳过
                }
              }
              routing_violation_gtl_poly_set_map[routing_layer_idx]
                  += GTLRectInt(std::min(corner_point.get_x(), violation_point.get_x()), std::min(corner_point.get_y(), violation_point.get_y()),
                                std::max(corner_point.get_x(), violation_point.get_x()), std::max(corner_point.get_y(), violation_point.get_y()));
              // end
            }
          }
        }
      }
    }
  }

  for (auto& [layer_idx, violation_gtl_poly_set] : routing_violation_gtl_poly_set_map) {
    std::vector<GTLRectInt> violation_rect_list;
    gtl::get_max_rectangles(violation_rect_list, violation_gtl_poly_set);
    int32_t required_size = corner_fill_spacing;
    for (GTLRectInt& violation_rect : violation_rect_list) {
      int32_t llx = gtl::xl(violation_rect);
      int32_t lly = gtl::yl(violation_rect);
      int32_t urx = gtl::xh(violation_rect);
      int32_t ury = gtl::yh(violation_rect);

      std::set<int32_t> net_set = queryNetIdbyRtree(layer_used_query_tree, layer_idx, llx, lly, urx, ury);
      Violation violation;
      violation.set_violation_type(ViolationType::kCornerFillSpacing);
      violation.set_required_size(required_size);
      violation.set_is_routing(true);
      violation.set_violation_net_set(net_set);
      violation.set_layer_idx(layer_idx);
      violation.set_rect(PlanarRect(llx, lly, urx, ury));

      rv_box.get_violation_list().push_back(violation);
    }
  }
}

}  // namespace idrc
