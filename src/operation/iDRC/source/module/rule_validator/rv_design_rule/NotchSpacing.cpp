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
using GTLSegment = gtl::segment_data<int32_t>;
void RuleValidator::verifyNotchSpacing(RVBox& rv_box)
{
/*
  t28中的notch spacing(110 没有该规则,110下直接跳过该规则的检查):
  PROPERTY LEF58_SPACING "
  SPACING 0.07 NOTCHLENGTH 0.155 CONCAVEENDS 0.055 ; " ;
 */
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
#endif
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  std::vector<Violation>& violation_list = rv_box.get_violation_list();
  /*R-tree新写的逻辑*/
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> layer_query_tree;
  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> layer_net_gtl_poly_set;
  std::map<int32_t, GTLPolySetInt> layer_violation_gtl_poly_set;
  for (DRCShape* rect : rv_box.get_drc_result_shape_list()) {
    if (!rect->get_is_routing() && rect->get_net_idx() == -1) {  // 不是routing layer或者net_idx为-1的跳过该检测
      continue;
    }
    int32_t layer_idx = rect->get_layer_idx();
    int32_t net_idx = rect->get_net_idx();
    addRectToRtree(layer_query_tree, GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y()), layer_idx, net_idx);
    layer_net_gtl_poly_set[layer_idx][net_idx] += GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y());
  }
  for (DRCShape* rect : rv_box.get_drc_env_shape_list()) {
    if (!rect->get_is_routing() && rect->get_net_idx() == -1) {  // 不是routing layer或者net_idx为-1的跳过该检测
      continue;
    }
    int32_t layer_idx = rect->get_layer_idx();
    int32_t net_idx = rect->get_net_idx();
    addRectToRtree(layer_query_tree, GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y()), layer_idx, net_idx);
    layer_net_gtl_poly_set[layer_idx][net_idx] += GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y());
  }

  for (auto& [routing_layer_idx, net_gtl_poly_set] : layer_net_gtl_poly_set) {
    int32_t notch_spacing = routing_layer_list[routing_layer_idx].get_notch_spacing();
    int32_t notch_length = routing_layer_list[routing_layer_idx].get_notch_length();
    std::optional<int32_t> concave_ends = routing_layer_list[routing_layer_idx].get_concave_ends();
    //主要检测函数
    auto check_notch_func = [&](GTLHolePolyInt gtl_hole_poly, GTLPolyInt hole, bool is_hole = false) {//hole需要的是凸角
      GTLHolePolyInt check_poly;
      if (is_hole) {
        check_poly.set(hole.begin(), hole.end());
      } else {
        check_poly = gtl_hole_poly;
      }
      int32_t coord_size = static_cast<int32_t>(check_poly.size());
      if (coord_size < 4) {
        return;
      }
      std::vector<PlanarCoord> coord_list;
      for (auto iter = check_poly.begin(); iter != check_poly.end(); iter++) {
        coord_list.push_back(DRCUTIL.convertToPlanarCoord(*iter));
      }
      std::vector<int32_t> edge_length_list;
      std::vector<bool> convex_corner_list;
      Rotation rotation = DRCUTIL.getRotation(check_poly);
      for (int32_t i = 0; i < coord_size; i++) {
        PlanarCoord& pre_coord = coord_list[getIdx(i - 1, coord_size)];
        PlanarCoord& curr_coord = coord_list[i];
        PlanarCoord& post_coord = coord_list[getIdx(i + 1, coord_size)];
        edge_length_list.push_back(DRCUTIL.getManhattanDistance(pre_coord, curr_coord));
        if (is_hole) {
          convex_corner_list.push_back(!DRCUTIL.isConvexCorner(rotation, pre_coord, curr_coord, post_coord));
        } else {
          convex_corner_list.push_back(DRCUTIL.isConvexCorner(rotation, pre_coord, curr_coord, post_coord));
        }
      }
      for (int32_t i = 0; i < coord_size; i++) {
        int32_t pre_idx = getIdx(i - 1, coord_size);
        int32_t curr_idx = i;
        int32_t post_idx = getIdx(i + 1, coord_size);
        int32_t post_post_idx = getIdx(i + 2, coord_size);
        int32_t post_post_post_idx = getIdx(i + 3, coord_size);
        if (convex_corner_list[curr_idx]) {
          continue;  // 凸角不需要检查
        }
        if (concave_ends.has_value() == false) {
          if (convex_corner_list[post_idx]) {
            continue;  // 只检查连续两个凹角即可
          }
          int32_t side_edge_length_a = edge_length_list[curr_idx];
          int32_t side_edge_length_b = edge_length_list[post_post_idx];  // 两个侧边
          int32_t spacing_edge_length = edge_length_list[post_idx];      // 底边
          if ((side_edge_length_a < notch_length || side_edge_length_b < notch_length) && (spacing_edge_length < notch_spacing)) {
            // 用第二条边和短的那条边构成违例区域
            PlanarCoord& violation_rect_point_a = side_edge_length_a < side_edge_length_b ? coord_list[post_idx] : coord_list[curr_idx];
            PlanarCoord& violation_rect_point_b = side_edge_length_a < side_edge_length_b ? coord_list[pre_idx] : coord_list[post_post_idx];

            layer_violation_gtl_poly_set[routing_layer_idx] += GTLRectInt(std::min(violation_rect_point_a.get_x(), violation_rect_point_b.get_x()),
                                                                          std::min(violation_rect_point_a.get_y(), violation_rect_point_b.get_y()),
                                                                          std::max(violation_rect_point_a.get_x(), violation_rect_point_b.get_x()),
                                                                          std::max(violation_rect_point_a.get_y(), violation_rect_point_b.get_y()));
          }
        } else {  // cocave_ends的情况
          // 只检查连续的三个凹角
          if (convex_corner_list[post_idx] || convex_corner_list[post_post_idx]) {
            continue;  // 只检查连续两个凹角即可
          }
          // 四个连续凹角的情况跳过，lef中跳过
          if (convex_corner_list[post_post_post_idx] == false || convex_corner_list[pre_idx] == false) {
            continue;
          }
          // 三凹角会有四条边，对应两种情况：1 2作为底边，2 3作为底边
          /*
         1-----
         |
         |          |
         2----------3 太废代码了，先空着
          */
          // case1:12作为底边
          // case2:23作为底边
          std::vector<std::array<int32_t, 3>> cases = {{post_post_idx, post_idx, curr_idx}, {post_idx, post_post_idx, post_post_post_idx}};
          for (auto [concave_edge_idx, spacing_edge_idx, side_edge_idx] : cases) {
            int32_t concave_edge_length = edge_length_list[concave_edge_idx];
            int32_t side_edge_length = edge_length_list[side_edge_idx];
            int32_t spacing_edge_length = edge_length_list[spacing_edge_idx];
            if (concave_edge_length < notch_length && side_edge_length >= notch_length && spacing_edge_length < notch_spacing) {
              // 检查两边的rect是否满足条件,拿到底边对应的两条边
              GTLSegment concave_edge = GTLSegment(
                  GTLPointInt(coord_list[concave_edge_idx].get_x(), coord_list[concave_edge_idx].get_y()),
                  GTLPointInt(coord_list[getIdx(concave_edge_idx - 1, coord_size)].get_x(), coord_list[getIdx(concave_edge_idx - 1, coord_size)].get_y()));
              GTLSegment side_edge = GTLSegment(
                  GTLPointInt(coord_list[side_edge_idx].get_x(), coord_list[side_edge_idx].get_y()),
                  GTLPointInt(coord_list[getIdx(side_edge_idx - 1, coord_size)].get_x(), coord_list[getIdx(side_edge_idx - 1, coord_size)].get_y()));
              GTLSegment spacing_edge = GTLSegment(
                  GTLPointInt(coord_list[spacing_edge_idx].get_x(), coord_list[spacing_edge_idx].get_y()),
                  GTLPointInt(coord_list[getIdx(spacing_edge_idx - 1, coord_size)].get_x(), coord_list[getIdx(spacing_edge_idx - 1, coord_size)].get_y()));
              gtl::orientation_2d_enum slice_dir
                  = gtl::x(spacing_edge.low()) == gtl::x(spacing_edge.high()) ? gtl::HORIZONTAL : gtl::VERTICAL;  // 水平竖切，竖直横切
              // 找到两条边对应的两个矩形，对于竖直凹槽，左边为矩形右边，右边为矩形左边，对于水平凹槽，上边为矩形下边，下边为矩形上边
              std::vector<GTLRectInt> slice_rect_list;
              gtl::get_max_rectangles(slice_rect_list, gtl_hole_poly);
              bool find_low = false;
              bool find_high = false;
              if (slice_dir == gtl::HORIZONTAL) {
                GTLSegment& low_edge = gtl::y(concave_edge.low()) < gtl::y(side_edge.low()) ? concave_edge : side_edge;
                GTLSegment& high_edge = gtl::y(concave_edge.low()) > gtl::y(side_edge.low()) ? concave_edge : side_edge;

                for (GTLRectInt& slice_rect : slice_rect_list) {
                  if (find_low && find_high) {
                    break;
                  }
                  GTLSegment slice_rect_low_edge
                      = GTLSegment(GTLPointInt(gtl::xl(slice_rect), gtl::yl(slice_rect)), GTLPointInt(gtl::xh(slice_rect), gtl::yl(slice_rect)));
                  GTLSegment slice_rect_high_edge
                      = GTLSegment(GTLPointInt(gtl::xl(slice_rect), gtl::yh(slice_rect)), GTLPointInt(gtl::xh(slice_rect), gtl::yh(slice_rect)));
                  if (gtl::contains(slice_rect_low_edge, high_edge)) {
                    int32_t rect_width = gtl::yh(slice_rect) - gtl::yl(slice_rect);
                    if (rect_width > concave_ends) {
                      find_high = false;  // 有满足宽度的矩形直接退出
                      break;
                    } else {
                      find_high = true;
                    }
                  }
                  if (gtl::contains(slice_rect_high_edge, low_edge)) {
                    int32_t rect_width = gtl::yh(slice_rect) - gtl::yl(slice_rect);
                    if (rect_width > concave_ends) {
                      find_low = false;  // 有满足宽度的矩形直接退出
                      break;
                    } else {
                      find_low = true;
                    }
                  }
                }
              } else {
                GTLSegment& low_edge = gtl::x(concave_edge.low()) < gtl::x(side_edge.low()) ? concave_edge : side_edge;
                GTLSegment& high_edge = gtl::x(concave_edge.low()) > gtl::x(side_edge.low()) ? concave_edge : side_edge;
                for (GTLRectInt& slice_rect : slice_rect_list) {
                  if (find_low && find_high) {
                    break;
                  }
                  GTLSegment slice_rect_low_edge
                      = GTLSegment(GTLPointInt(gtl::xl(slice_rect), gtl::yl(slice_rect)), GTLPointInt(gtl::xl(slice_rect), gtl::yh(slice_rect)));
                  GTLSegment slice_rect_high_edge
                      = GTLSegment(GTLPointInt(gtl::xh(slice_rect), gtl::yl(slice_rect)), GTLPointInt(gtl::xh(slice_rect), gtl::yh(slice_rect)));
                  if (gtl::contains(slice_rect_low_edge, high_edge)) {
                    int32_t rect_width = gtl::xh(slice_rect) - gtl::xl(slice_rect);
                    if (rect_width > concave_ends) {
                      find_high = false;  // 有满足宽度的矩形直接退出
                      break;
                    } else {
                      find_high = true;
                    }
                  }
                  if (gtl::contains(slice_rect_high_edge, low_edge)) {
                    int32_t rect_width = gtl::xh(slice_rect) - gtl::xl(slice_rect);
                    if (rect_width > concave_ends) {
                      find_low = false;  // 有满足宽度的矩形直接退出
                      break;
                    } else {
                      find_low = true;
                    }
                  }
                }
              }
              if (!find_low || !find_high) {
                continue;  // 两条边都满足条件,否则跳过
              }
              // 用第二条边和短的那条边构成违例区域，这里是通过底边来判断的
              int32_t prev_prev_spacing_idx = getIdx(spacing_edge_idx - 2, coord_size);
              int32_t prev_spacing_idx = getIdx(spacing_edge_idx - 1, coord_size);
              int32_t post_spacing_idx = getIdx(spacing_edge_idx + 1, coord_size);
              PlanarCoord& violation_rect_point_a
                  = edge_length_list[prev_spacing_idx] < edge_length_list[post_spacing_idx] ? coord_list[spacing_edge_idx] : coord_list[prev_spacing_idx];
              PlanarCoord& violation_rect_point_b
                  = edge_length_list[prev_spacing_idx] < edge_length_list[post_spacing_idx] ? coord_list[prev_prev_spacing_idx] : coord_list[post_spacing_idx];

              layer_violation_gtl_poly_set[routing_layer_idx] += GTLRectInt(std::min(violation_rect_point_a.get_x(), violation_rect_point_b.get_x()),
                                                                            std::min(violation_rect_point_a.get_y(), violation_rect_point_b.get_y()),
                                                                            std::max(violation_rect_point_a.get_x(), violation_rect_point_b.get_x()),
                                                                            std::max(violation_rect_point_a.get_y(), violation_rect_point_b.get_y()));
            }
          }
        }
      }
    };
    for (auto& [net_idx, gtl_poly_set] : net_gtl_poly_set) {
      std::vector<GTLHolePolyInt> gtl_hole_poly_list;
      gtl_poly_set.get(gtl_hole_poly_list);  // get会自动识别要变成的类型,用带hole能够避免识别到hole导致edge不对
      for (GTLHolePolyInt& gtl_hole_poly : gtl_hole_poly_list) {
        check_notch_func(gtl_hole_poly, GTLPolyInt(), false);
        // hole单独处理，hole的凸角对应polygon的凹角，除此之外处理逻辑完全一样
        GTLHolePolyInt::iterator_holes_type hole_iter = gtl_hole_poly.begin_holes();
        while (hole_iter != gtl_hole_poly.end_holes()) {
          GTLPolyInt hole = *hole_iter;
          check_notch_func(gtl_hole_poly, hole, true);
          hole_iter++;
        }
      }
    }
  }

  for (auto& [layer_idx, violation_poly_set] : layer_violation_gtl_poly_set) {
    std::vector<GTLRectInt> violation_rect_list;
    gtl::get_max_rectangles(violation_rect_list, violation_poly_set);
    int32_t required_size = 140;
    for (GTLRectInt& violation_rect : violation_rect_list) {
      int32_t llx = gtl::xl(violation_rect);
      int32_t lly = gtl::yl(violation_rect);
      int32_t urx = gtl::xh(violation_rect);
      int32_t ury = gtl::yh(violation_rect);

      std::set<int32_t> net_set = queryNetIdbyRtree(layer_query_tree, layer_idx, llx, lly, urx, ury);
      Violation violation;
      violation.set_violation_type(ViolationType::kNotchSpacing);
      violation.set_required_size(required_size);
      violation.set_is_routing(true);
      violation.set_violation_net_set(net_set);
      violation.set_layer_idx(layer_idx);
      violation.set_rect(PlanarRect(llx, lly, urx, ury));

      violation_list.push_back(violation);
    }
  }
}

}  // namespace idrc
