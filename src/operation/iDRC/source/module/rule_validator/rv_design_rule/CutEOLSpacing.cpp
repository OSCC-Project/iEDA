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
void RuleValidator::verifyCutEOLSpacing(RVBox& rv_box)
{
  /*
  对应的规则如下：
      PROPERTY LEF58_EOLSPACING "
      EOLSPACING 0.08 0.09 CUTCLASS VSINGLECUT TO VDOUBLECUT 0.085 0.09 ENDWIDTH 0.07 PRL -0.04
      ENCLOSURE 0.04 0.00
      EXTENSION 0.065 0.12 SPANLENGTH 0.055 ; " ;
  */
#if 1
  auto getMAXXYSpacing = [](GTLRectInt& rect_a, GTLRectInt& rect_b) {
    int32_t h_spacing = std::max(0, std::max(gtl::xl(rect_a) - gtl::xh(rect_b), gtl::xl(rect_b) - gtl::xh(rect_a)));
    int32_t v_spacing = std::max(0, std::max(gtl::yl(rect_a) - gtl::yh(rect_b), gtl::yl(rect_b) - gtl::yh(rect_a)));
    return std::max(h_spacing, v_spacing);
  };
  auto update_overhang = [](GTLRectInt& gtl_cut_rect, GTLRectInt& gtl_polygon_rect, std::vector<GTLSegment>& overhang_list) {
    // overhang_list被初始化成了cut rect的四条边
    // 该函数用来更新overhang_list,使用之前保证cut rect在polygon_rect里面
    // overhang顺序：east south west north
    if (gtl::contains(gtl_polygon_rect, gtl_cut_rect) == false) {
      // if(gtl::intersects(gtl_polygon_rect, gtl_cut_rect,false) == false) {
      return;  // 不完全包含的忽略
    }
    GTLSegment east_segment(GTLPointInt(gtl::xh(gtl_polygon_rect), gtl::yl(gtl_polygon_rect)),
                            GTLPointInt(gtl::xh(gtl_polygon_rect), gtl::yh(gtl_polygon_rect)));
    GTLSegment south_segment(GTLPointInt(gtl::xl(gtl_polygon_rect), gtl::yl(gtl_polygon_rect)),
                             GTLPointInt(gtl::xh(gtl_polygon_rect), gtl::yl(gtl_polygon_rect)));
    GTLSegment west_segment(GTLPointInt(gtl::xl(gtl_polygon_rect), gtl::yl(gtl_polygon_rect)),
                            GTLPointInt(gtl::xl(gtl_polygon_rect), gtl::yh(gtl_polygon_rect)));
    GTLSegment north_segment(GTLPointInt(gtl::xl(gtl_polygon_rect), gtl::yh(gtl_polygon_rect)),
                             GTLPointInt(gtl::xh(gtl_polygon_rect), gtl::yh(gtl_polygon_rect)));
    if (east_segment.low().x() >= overhang_list[0].low().x()) {
      overhang_list[0] = east_segment;
    }
    if (south_segment.low().y() <= overhang_list[1].low().y()) {
      overhang_list[1] = south_segment;
    }
    if (west_segment.low().x() <= overhang_list[2].low().x()) {
      overhang_list[2] = west_segment;
    }
    if (north_segment.low().y() >= overhang_list[3].low().y()) {
      overhang_list[3] = north_segment;
    }
  };
  auto is_polygon_interact_rect = [](GTLRectInt query_rect, GTLPolySetInt polygon_set, bool is_consider_egde = false) {
    GTLPolySetInt query_rect_set;
    query_rect_set += query_rect;
    if (is_consider_egde) {
      polygon_set.interact(query_rect_set);
    } else {
      polygon_set &= query_rect_set;
    }

    return gtl::area(polygon_set) > 0;
  };
#endif
  // 规则定义
  int32_t cut_spacing_a = 80 * 2;
  int32_t cut_spacing_b = 90 * 2;

  int32_t signle_double_cut_spacing_a = 85 * 2;  // double cut暂时不考虑
  int32_t signle_double_cut_spacing_b = 90 * 2;  // double cut暂时不考虑

  int32_t eol_width = 70 * 2;         // EOL edge的长度 要小于这个值
  int32_t prl = -1 * 40 * 2;          // spacing的prl
  int32_t smaller_overhang = 40 * 2;  // 有一个overhang小于这个值
  int32_t equal_overhang = 0;         // 正交边(相邻边)的overhang等于这个值

  int32_t side_ext = 65 * 2;       //
  int32_t backward_ext = 120 * 2;  //
  int32_t span_length = 55 * 2;    //

  std::vector<int32_t> cut_eol_spacing_layers = {1, 2, 3, 4, 5, 6};
  // 基础数据
  std::vector<Violation>& violation_list = rv_box.get_violation_list();
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DRCDM.getDatabase().get_cut_layer_list();

  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> cut_layer_res_query_tree;
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> cut_layer_all_query_tree;

  std::map<int32_t, bgi::rtree<BGRectInt, bgi::quadratic<16>>> routing_layer_query_tree;
  std::map<int32_t, GTLPolySetInt> routing_layer_gtl_poly_set;

  for (DRCShape* rect : rv_box.get_drc_env_shape_list()) {
    if (rect->get_is_routing()) {
      int32_t layer_idx = rect->get_layer_idx();
      int32_t net_idx = rect->get_net_idx();
      routing_layer_gtl_poly_set[layer_idx] += DRCUTIL.convertToGTLRectInt(rect->get_rect());
    } else {
      cut_layer_all_query_tree[rect->get_layer_idx()].insert(std::make_pair(DRCUTIL.convertToBGRectInt(rect->get_rect()), rect->get_net_idx()));
    }
  }

  for (DRCShape* rect : rv_box.get_drc_result_shape_list()) {
    if (rect->get_is_routing()) {
      int32_t layer_idx = rect->get_layer_idx();
      int32_t net_idx = rect->get_net_idx();
      routing_layer_gtl_poly_set[layer_idx] += DRCUTIL.convertToGTLRectInt(rect->get_rect());
    } else {
      cut_layer_all_query_tree[rect->get_layer_idx()].insert(std::make_pair(DRCUTIL.convertToBGRectInt(rect->get_rect()), rect->get_net_idx()));
      cut_layer_res_query_tree[rect->get_layer_idx()].insert(std::make_pair(DRCUTIL.convertToBGRectInt(rect->get_rect()), rect->get_net_idx()));
    }
  }

  // 用max rect作为被查找的
  for (auto& [routing_layer_idx, gtl_poly_set] : routing_layer_gtl_poly_set) {
    std::vector<GTLRectInt> rect_list;
    gtl::get_max_rectangles(rect_list, gtl_poly_set);
    for (GTLRectInt& rect : rect_list) {
      routing_layer_query_tree[routing_layer_idx].insert(DRCUTIL.convertToBGRectInt(rect));
    }
  }

  for (auto& [routing_layer_idx, gtl_poly_set] : routing_layer_gtl_poly_set) {
    if (DRCUTIL.exist(cut_eol_spacing_layers, routing_layer_idx) == false) {
      continue;  // 没有对应的cut layer  此时routing_layer_idx 刚好对应其下一层的cut layer idx
    }
    int32_t cut_layer_idx = routing_layer_idx;
    int32_t min_width = routing_layer_list[routing_layer_idx].get_min_width();

    std::vector<GTLHolePolyInt> gtl_hole_poly_list;
    gtl_poly_set.get(gtl_hole_poly_list);
    // polyset -> polygon -> maxrect -> cut
    for (GTLHolePolyInt& gtl_hole_poly : gtl_hole_poly_list) {
      // 获得eol 边
      int32_t coord_size = static_cast<int32_t>(gtl_hole_poly.size());
      if (coord_size < 4) {
        continue;
      }
      std::vector<PlanarCoord> coord_list;
      for (auto iter = gtl_hole_poly.begin(); iter != gtl_hole_poly.end(); iter++) {
        coord_list.push_back(DRCUTIL.convertToPlanarCoord(*iter));
      }
      std::vector<bool> convex_corner_list;
      std::vector<int32_t> edge_length_list;
      for (int32_t i = 0; i < coord_size; i++) {
        PlanarCoord& pre_coord = coord_list[getIdx(i - 1, coord_size)];
        PlanarCoord& curr_coord = coord_list[i];
        PlanarCoord& post_coord = coord_list[getIdx(i + 1, coord_size)];
        convex_corner_list.push_back(DRCUTIL.isConvexCorner(DRCUTIL.getRotation(gtl_hole_poly), pre_coord, curr_coord, post_coord));
        edge_length_list.push_back(DRCUTIL.getManhattanDistance(pre_coord, curr_coord));
      }
      std::set<int32_t> eol_edge_idx_set;
      for (int32_t i = 0; i < coord_size; i++) {
        if (convex_corner_list[getIdx(i - 1, coord_size)] && convex_corner_list[i]) {
          eol_edge_idx_set.insert(i);
        }
      }

      struct RectCompare  // gtlrect key的比较单位
      {
        bool operator()(const GTLRectInt& a, const GTLRectInt& b) const
        {
          if (gtl::xl(a) != gtl::xl(b))
            return gtl::xl(a) < gtl::xl(b);
          if (gtl::yl(a) != gtl::yl(b))
            return gtl::yl(a) < gtl::yl(b);
          if (gtl::xh(a) != gtl::xh(b))
            return gtl::xh(a) < gtl::xh(b);
          return gtl::yh(a) < gtl::yh(b);
        }
      };

      // net_idx -> map(rect,overhang_segment)
      std::map<int32_t, std::map<GTLRectInt, std::vector<GTLSegment>, RectCompare>> net_gtl_cut_overhang_list;

      std::vector<GTLRectInt> polygon_rect_list;
      gtl::get_max_rectangles(polygon_rect_list, gtl_hole_poly);

      // 用来判断cut
      for (GTLRectInt& gtl_polygon_rect : polygon_rect_list) {
        // 从max rect更新overhang
        int32_t width = std::min(gtl::delta(gtl_polygon_rect, gtl::HORIZONTAL), gtl::delta(gtl_polygon_rect, gtl::VERTICAL));
        if (width < min_width) {
          continue;  // 小于最小宽度的忽略,cut不会在这种矩形中
        }
        // 查出所有的cut
        std::vector<std::pair<BGRectInt, int32_t>> bg_cut_result;
        {
          GTLRectInt check_rect = gtl_polygon_rect;
          cut_layer_res_query_tree[cut_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(check_rect)), std::back_inserter(bg_cut_result));
        }
        // 更新overhang
        for (auto& [bg_cut, net_idx] : bg_cut_result) {
          GTLRectInt gtl_cut_rect = DRCUTIL.convertToGTLRectInt(bg_cut);
          if (DRCUTIL.exist(net_gtl_cut_overhang_list, net_idx) && DRCUTIL.exist(net_gtl_cut_overhang_list[net_idx], gtl_cut_rect)) {
            // 更新overhang
            update_overhang(gtl_cut_rect, gtl_polygon_rect, net_gtl_cut_overhang_list[net_idx][gtl_cut_rect]);
          } else {  // 初始化overhang
            // 顺序：east south west north
            net_gtl_cut_overhang_list[net_idx][gtl_cut_rect].push_back(
                GTLSegment(GTLPointInt(gtl::xh(gtl_cut_rect), gtl::yl(gtl_cut_rect)), GTLPointInt(gtl::xh(gtl_cut_rect), gtl::yh(gtl_cut_rect))));
            net_gtl_cut_overhang_list[net_idx][gtl_cut_rect].push_back(
                GTLSegment(GTLPointInt(gtl::xl(gtl_cut_rect), gtl::yl(gtl_cut_rect)), GTLPointInt(gtl::xh(gtl_cut_rect), gtl::yl(gtl_cut_rect))));
            net_gtl_cut_overhang_list[net_idx][gtl_cut_rect].push_back(
                GTLSegment(GTLPointInt(gtl::xl(gtl_cut_rect), gtl::yl(gtl_cut_rect)), GTLPointInt(gtl::xl(gtl_cut_rect), gtl::yh(gtl_cut_rect))));
            net_gtl_cut_overhang_list[net_idx][gtl_cut_rect].push_back(
                GTLSegment(GTLPointInt(gtl::xl(gtl_cut_rect), gtl::yh(gtl_cut_rect)), GTLPointInt(gtl::xh(gtl_cut_rect), gtl::yh(gtl_cut_rect))));

            update_overhang(gtl_cut_rect, gtl_polygon_rect, net_gtl_cut_overhang_list[net_idx][gtl_cut_rect]);
          }
        }
        // polygon 的maxrect为单位
      }
      // 根据overhang进行cut spacing的检查
      for (auto& [net_idx, gtl_cut_overhang_list] : net_gtl_cut_overhang_list) {
        for (auto& [gtl_cut, overhang_list] : gtl_cut_overhang_list) {
          // 先判断条件1：相邻两个overhang长度是否满足条件
          bool is_enclosure = false;
          std::vector<bool> is_enclosure_list(4, false);
          std::vector<int32_t> overhang_dis_list(4, 0);
          overhang_dis_list[0] = (std::abs(overhang_list[0].low().x() - gtl::xh(gtl_cut)));  // east
          overhang_dis_list[1] = (std::abs(overhang_list[1].low().y() - gtl::yl(gtl_cut)));  // south
          overhang_dis_list[2] = (std::abs(overhang_list[2].low().x() - gtl::xl(gtl_cut)));  // west
          overhang_dis_list[3] = (std::abs(overhang_list[3].low().y() - gtl::yh(gtl_cut)));  // north
          for (int32_t i = 0; i < overhang_dis_list.size(); i++) {
            int32_t pre_idx = getIdx(i - 1, overhang_dis_list.size());
            int32_t post_idx = getIdx(i + 1, overhang_dis_list.size());
            if (overhang_dis_list[i] < smaller_overhang && (overhang_dis_list[pre_idx] == equal_overhang || overhang_dis_list[post_idx] == equal_overhang)) {
              is_enclosure = true;
              is_enclosure_list[i] = true;
            }
          }

          if (is_enclosure == false) {
            continue;  // 不满足条件enclosure条件直接跳过
          }

          // 再判断条件2，eol edge是否满足
          bool is_eol_edge = false;
          std::vector<bool> is_eol_edge_list(4, false);
          for (auto& eol_idx : eol_edge_idx_set) {  // 遍历eol egde
            int32_t pre_eol_idx = getIdx(eol_idx - 1, coord_size);
            GTLPointInt a = GTLPointInt(coord_list[pre_eol_idx].get_x(), coord_list[pre_eol_idx].get_y());
            GTLPointInt b = GTLPointInt(coord_list[eol_idx].get_x(), coord_list[eol_idx].get_y());
            GTLSegment eol_edge = GTLSegment(a, b);
            for (int32_t i = 0; i < overhang_dis_list.size(); i++) {  // 找到对应的overhang segment
                                                                      // if (gtl::contains(eol_edge, overhang_list[i]) && gtl::length(eol_edge) < eol_width) {
              if (gtl::contains(eol_edge, overhang_list[i])) {
                is_eol_edge_list[i] = true;
                is_eol_edge = true;
              }
            }
          }

          // 用overhang的外接矩形膨胀后生成一个查找区域,作为该cut上层的wire rect
          PlanarRect overhang_rect(INT32_MAX, INT32_MAX, 0, 0);
          for (GTLSegment& overhang : overhang_list) {
            // 更新llx,lly,urx,ury
            if (overhang_rect.get_ll_x() > overhang.low().x()) {
              overhang_rect.set_ll_x(overhang.low().x());
            }
            if (overhang_rect.get_ll_y() > overhang.low().y()) {
              overhang_rect.set_ll_y(overhang.low().y());
            }
            if (overhang_rect.get_ur_x() < overhang.high().x()) {
              overhang_rect.set_ur_x(overhang.high().x());
            }
            if (overhang_rect.get_ur_y() < overhang.high().y()) {
              overhang_rect.set_ur_y(overhang.high().y());
            }
          }
          // 膨胀大小为 side_ext;
          GTLRectInt gtl_bloat_overhang_rect = DRCUTIL.convertToGTLRectInt(DRCUTIL.getEnlargedRect(overhang_rect, side_ext));
          std::vector<BGRectInt> bg_routing_overlap_rect_result;
          {
            GTLRectInt check_rect = gtl_bloat_overhang_rect;
            routing_layer_query_tree[cut_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(check_rect)),
                                                          std::back_inserter(bg_routing_overlap_rect_result));
          }

          std::vector<GTLRectInt> routing_gtl_overlap_rect_list;  // cut上层layer的wire周围的所有矩形
          GTLPolySetInt gtl_cut_poly;                             // 包含cut的poly wire

          int32_t max_x_span = 0;
          int32_t max_y_span = 0;
          for (BGRectInt& bgrect : bg_routing_overlap_rect_result) {
            GTLRectInt gtl_overlap_rect = DRCUTIL.convertToGTLRectInt(bgrect);
            if (gtl::intersects(gtl_overlap_rect, gtl_cut, false)) {  // 和cut intersects的才算做cut的wire?
              gtl_cut_poly += gtl_overlap_rect;

              // 更新cut 的xspan和yspan
              int32_t x_span = gtl::delta(gtl_overlap_rect, gtl::HORIZONTAL);
              int32_t y_span = gtl::delta(gtl_overlap_rect, gtl::VERTICAL);
              if (x_span > max_x_span) {
                max_x_span = x_span;
              }
              if (y_span > max_y_span) {
                max_y_span = y_span;
              }
              continue;  // 完全包含cut的忽略
            }
            routing_gtl_overlap_rect_list.push_back(gtl_overlap_rect);
          }

          GTLPolySetInt gtl_routing_overlap_poly;  // cut的wire周围的所有图形，需要去除和cut wire poly 相交的rect
          for (GTLRectInt& gtl_overlap_rect : routing_gtl_overlap_rect_list) {
            if (gtl::area(gtl_overlap_rect & gtl_cut_poly) > 0) {
              continue;  // 需要去除和cut wire poly 相交的rect
            }
            gtl_routing_overlap_poly += gtl_overlap_rect;
          }

          // 按照lef的解释，会有8个查找区域
          int32_t wire_llx = overhang_rect.get_ll_x();
          int32_t wire_lly = overhang_rect.get_ll_y();
          int32_t wire_urx = overhang_rect.get_ur_x();
          int32_t wire_ury = overhang_rect.get_ur_y();
          GTLRectInt gtl_north_left_rect = GTLRectInt(wire_llx, wire_ury, wire_llx + backward_ext, wire_ury + side_ext);
          GTLRectInt gtl_north_right_rect = GTLRectInt(wire_urx - backward_ext, wire_ury, wire_urx, wire_ury + side_ext);
          GTLRectInt gtl_south_left_rect = GTLRectInt(wire_llx, wire_lly - side_ext, wire_llx + backward_ext, wire_lly);
          GTLRectInt gtl_south_right_rect = GTLRectInt(wire_urx - backward_ext, wire_lly - side_ext, wire_urx, wire_lly);
          GTLRectInt gtl_west_down_rect = GTLRectInt(wire_llx - side_ext, wire_lly, wire_llx, wire_lly + backward_ext);
          GTLRectInt gtl_west_up_rect = GTLRectInt(wire_llx - side_ext, wire_ury - backward_ext, wire_llx, wire_ury);
          GTLRectInt gtl_east_down_rect = GTLRectInt(wire_urx, wire_lly, wire_urx + side_ext, wire_lly + backward_ext);
          GTLRectInt gtl_east_up_rect = GTLRectInt(wire_urx, wire_ury - backward_ext, wire_urx + side_ext, wire_ury);

          // 判断是否有overlap，用查找区域和gtl_routing_overlap_poly & 来判断能否查找
          std::vector<std::vector<bool>> is_overlap_list(4, std::vector<bool>(2, false));
          is_overlap_list[0][0]
              = is_polygon_interact_rect(gtl_north_right_rect, gtl_routing_overlap_poly) && !is_polygon_interact_rect(gtl_north_right_rect, gtl_cut_poly);
          is_overlap_list[0][1]
              = is_polygon_interact_rect(gtl_south_right_rect, gtl_routing_overlap_poly) && !is_polygon_interact_rect(gtl_south_right_rect, gtl_cut_poly);
          is_overlap_list[1][0]
              = is_polygon_interact_rect(gtl_east_down_rect, gtl_routing_overlap_poly) && !is_polygon_interact_rect(gtl_east_down_rect, gtl_cut_poly);
          is_overlap_list[1][1]
              = is_polygon_interact_rect(gtl_west_down_rect, gtl_routing_overlap_poly) && !is_polygon_interact_rect(gtl_west_down_rect, gtl_cut_poly);
          is_overlap_list[2][0]
              = is_polygon_interact_rect(gtl_south_left_rect, gtl_routing_overlap_poly) && !is_polygon_interact_rect(gtl_south_left_rect, gtl_cut_poly);
          is_overlap_list[2][1]
              = is_polygon_interact_rect(gtl_north_left_rect, gtl_routing_overlap_poly) && !is_polygon_interact_rect(gtl_north_left_rect, gtl_cut_poly);
          is_overlap_list[3][0]
              = is_polygon_interact_rect(gtl_west_up_rect, gtl_routing_overlap_poly) && !is_polygon_interact_rect(gtl_west_up_rect, gtl_cut_poly);
          is_overlap_list[3][1]
              = is_polygon_interact_rect(gtl_east_up_rect, gtl_routing_overlap_poly) && !is_polygon_interact_rect(gtl_east_up_rect, gtl_cut_poly);

          bool no_less_span = max_x_span >= span_length && max_y_span >= span_length;  // 都大于等于span只考虑一边，否则两边
          std::vector<bool> is_use_spacing_list(4, false);                             // 最终判断该边是否需要生成spacing区域
          bool is_use_spacing = false;                                                 // 是否需要生成spacing
          for (int32_t i = 0; i < is_enclosure_list.size(); i++) {
            if (is_enclosure_list[i] == false) {
              continue;  // 查找区域必须满足enclosure条件，即一边等于euqal_overhang,一边小于samller_overhang
            }
            int32_t pre_idx = getIdx(i - 1, is_enclosure_list.size());
            int32_t post_idx = getIdx(i + 1, is_enclosure_list.size());
            bool is_pre_equal_overhang = overhang_dis_list[pre_idx] == equal_overhang;
            bool is_post_equal_overhang = overhang_dis_list[post_idx] == equal_overhang;
            if (is_pre_equal_overhang
                && ((is_overlap_list[i][0] && !is_overlap_list[i][1] && !no_less_span)
                    || is_overlap_list[i][0] && no_less_span)) {  // 都大于等于span只考虑一边，否则两边
              is_use_spacing_list[getIdx(i + 2, is_enclosure_list.size())] = true;
              is_use_spacing_list[getIdx(pre_idx + 2, is_enclosure_list.size())] = true;
              is_use_spacing = true;
            }
            if (is_post_equal_overhang
                && ((is_overlap_list[i][1] && !is_overlap_list[i][0] && !no_less_span)
                    || is_overlap_list[i][1] && no_less_span)) {  // 都大于等于span只考虑一边，否则两边
              is_use_spacing_list[getIdx(i + 2, is_enclosure_list.size())] = true;
              is_use_spacing_list[getIdx(post_idx + 2, is_enclosure_list.size())] = true;
              is_use_spacing = true;
            }
          }

          if (is_use_spacing == false) {
            continue;  // 不需要生成spacing区域
          }
          //  生成查找区域，贴着边那个位置的不要，生成的时候要往外延伸一点
          GTLRectInt east_spacing_rect
              = GTLRectInt(gtl::xh(gtl_cut) + 1, gtl::yl(gtl_cut) - cut_spacing_b, gtl::xh(gtl_cut) + cut_spacing_b, gtl::yh(gtl_cut) + cut_spacing_b);
          GTLRectInt south_spacing_rect
              = GTLRectInt(gtl::xl(gtl_cut) - cut_spacing_b, gtl::yl(gtl_cut) - cut_spacing_b, gtl::xh(gtl_cut) + cut_spacing_b, gtl::yl(gtl_cut) - 1);
          GTLRectInt west_spacing_rect
              = GTLRectInt(gtl::xl(gtl_cut) - cut_spacing_b, gtl::yl(gtl_cut) - cut_spacing_b, gtl::xl(gtl_cut) - 1, gtl::yh(gtl_cut) + cut_spacing_b);
          GTLRectInt north_spacing_rect
              = GTLRectInt(gtl::xl(gtl_cut) - cut_spacing_b, gtl::yh(gtl_cut) + 1, gtl::xh(gtl_cut) + cut_spacing_b, gtl::yh(gtl_cut) + cut_spacing_b);

          GTLRectInt east_spacing_rect_b = GTLRectInt(gtl::xh(gtl_cut) + 1, gtl::yl(gtl_cut) + prl, gtl::xh(gtl_cut) + cut_spacing_b, gtl::yh(gtl_cut) - prl);
          GTLRectInt south_spacing_rect_b = GTLRectInt(gtl::xl(gtl_cut) + prl, gtl::yl(gtl_cut) - cut_spacing_b, gtl::xh(gtl_cut) - prl, gtl::yl(gtl_cut) - 1);
          GTLRectInt west_spacing_rect_b = GTLRectInt(gtl::xl(gtl_cut) - cut_spacing_b, gtl::yl(gtl_cut) + prl, gtl::xl(gtl_cut) - 1, gtl::yh(gtl_cut) - prl);
          GTLRectInt north_spacing_rect_b = GTLRectInt(gtl::xl(gtl_cut) + prl, gtl::yh(gtl_cut) + 1, gtl::xh(gtl_cut) - prl, gtl::yh(gtl_cut) + cut_spacing_b);

          // 生成spacing_b区域和spacing_a区域
          GTLPolySetInt spacing_a_region;
          GTLPolySetInt spacing_b_region;
          if (is_use_spacing_list[0]) {
            spacing_a_region += east_spacing_rect;
            spacing_b_region += east_spacing_rect_b;
          }
          if (is_use_spacing_list[1]) {
            spacing_a_region += south_spacing_rect;
            spacing_b_region += south_spacing_rect_b;
          }
          if (is_use_spacing_list[2]) {
            spacing_a_region += west_spacing_rect;
            spacing_b_region += west_spacing_rect_b;
          }
          if (is_use_spacing_list[3]) {
            spacing_a_region += north_spacing_rect;
            spacing_b_region += north_spacing_rect_b;
          }
          spacing_a_region -= spacing_b_region;  // 去掉b的部分就是a的部分

          std::vector<std::pair<BGRectInt, int32_t>> bg_overlap_cut_result;
          {
            GTLRectInt gtl_cut_rect = gtl_cut;
            PlanarRect check_rect = DRCUTIL.getEnlargedRect(DRCUTIL.convertToPlanarRect(gtl_cut_rect), cut_spacing_b);
            cut_layer_all_query_tree[cut_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(check_rect)), std::back_inserter(bg_overlap_cut_result));
          }
          for (auto& [bgrect, q_net_idx] : bg_overlap_cut_result) {
            if (net_idx == -1 && q_net_idx == -1) {
              continue;  // -1忽略
            }
            PlanarRect env_cut = DRCUTIL.convertToPlanarRect(bgrect);
            GTLRectInt gtl_env_cut = DRCUTIL.convertToGTLRectInt(bgrect);
            GTLRectInt gtl_origin_cut = gtl_cut;
            PlanarRect origin_cut = DRCUTIL.convertToPlanarRect(gtl_origin_cut);
            // 先check两个rect是否重叠
            if (env_cut == origin_cut) {
              continue;  // 同一个cut忽略
            }
            int32_t need_spacing;
            int32_t real_spacing;
            if (is_polygon_interact_rect(gtl_env_cut, spacing_b_region, true)) {  // 和区域b有接触（因为包括等于abs(prl)的，所以必须是包括边,先判断b再判断a
              need_spacing = cut_spacing_b;
              real_spacing = getMAXXYSpacing(gtl_env_cut, gtl_origin_cut);
              if (real_spacing >= need_spacing) {
                continue;  // 满足距离条件
              }
            } else if (is_polygon_interact_rect(gtl_env_cut, spacing_a_region, true)) {  // 和区域a有接触
              need_spacing = cut_spacing_a;
              real_spacing = DRCUTIL.getEuclideanDistance(env_cut, origin_cut);
              if (real_spacing >= need_spacing) {
                continue;  // 满足距离条件
              }
            } else {
              continue;  // 都不满足的忽略
            }
            PlanarRect violation_rect;
            if (DRCUTIL.isClosedOverlap(env_cut, origin_cut)) {
              violation_rect = DRCUTIL.getOverlap(env_cut, origin_cut);
            } else {
              violation_rect = DRCUTIL.getSpacingRect(env_cut, origin_cut);
            }

            Violation violation;
            violation.set_violation_type(ViolationType::kCutEOLSpacing);
            violation.set_is_routing(true);
            violation.set_violation_net_set({net_idx, q_net_idx});
            violation.set_layer_idx(cut_layer_idx - 1);
            violation.set_rect(violation_rect);
            violation.set_required_size(need_spacing);
            violation_list.push_back(violation);
          }

          // cut 为单位
        }
      }
      // polygon 为单位
    }
    // polygonset 为单位
  }
}

}  // namespace idrc
