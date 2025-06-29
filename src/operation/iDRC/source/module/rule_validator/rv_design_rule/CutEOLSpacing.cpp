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
void RuleValidator::verifyCutEOLSpacing(RVBox& rv_box)
{
  /*
  对应的规则如下：
      PROPERTY LEF58_EOLSPACING "
      EOLSPACING 0.08 0.09 CUTCLASS VSINGLECUT TO VDOUBLECUT 0.085 0.09 ENDWIDTH 0.07 PRL -0.04
      ENCLOSURE 0.04 0.00
      EXTENSION 0.065 0.12 SPANLENGTH 0.055 ; " ;
  */
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

#if 1
  auto update_cut_overhang = [](std::vector<Segment<PlanarCoord>>& overhang_list, PlanarRect& polygon_rect) {
    PlanarCoord ll = polygon_rect.get_ll();
    PlanarCoord ur = polygon_rect.get_ur();
    PlanarCoord ul = PlanarCoord(ll.get_x(), ur.get_y());
    PlanarCoord lr = PlanarCoord(ur.get_x(), ll.get_y());
    // 东南西北
    Segment<PlanarCoord> east_segment(lr, ur);
    Segment<PlanarCoord> south_segment(ll, lr);
    Segment<PlanarCoord> west_segment(ll, ul);
    Segment<PlanarCoord> north_segment(ul, ur);
    if (overhang_list.empty()) {
      overhang_list.push_back(east_segment);
      overhang_list.push_back(south_segment);
      overhang_list.push_back(west_segment);
      overhang_list.push_back(north_segment);
    } else {
      if (east_segment.get_first().get_x() >= overhang_list[0].get_first().get_x()) {
        overhang_list[0] = east_segment;
      }
      if (south_segment.get_first().get_y() <= overhang_list[1].get_first().get_y()) {
        overhang_list[1] = south_segment;
      }
      if (west_segment.get_first().get_x() <= overhang_list[2].get_first().get_x()) {
        overhang_list[2] = west_segment;
      }
      if (north_segment.get_first().get_y() >= overhang_list[3].get_first().get_y()) {
        overhang_list[3] = north_segment;
      }
    }
  };

  auto is_rect_interact_polygon = [](GTLRectInt query_rect, GTLPolySetInt polygon_set, bool is_consider_egde) {
    GTLPolySetInt query_rect_set;
    query_rect_set += query_rect;
    if (is_consider_egde) {
      polygon_set.interact(query_rect_set);
    } else {
      polygon_set &= query_rect_set;
    }

    return gtl::area(polygon_set) > 0;
  };

  auto gen_eol_search_rect = [&](PlanarRect& cut_wire_rect, int egde_dir) {
    std::vector<PlanarRect> search_rect_list;  // 0
    int32_t wire_llx = cut_wire_rect.get_ll_x();
    int32_t wire_lly = cut_wire_rect.get_ll_y();
    int32_t wire_urx = cut_wire_rect.get_ur_x();
    int32_t wire_ury = cut_wire_rect.get_ur_y();
    switch (egde_dir) {
      case 0:
        return std::vector<PlanarRect>{PlanarRect(wire_urx - backward_ext, wire_ury, wire_urx, wire_ury + side_ext),
                                       PlanarRect(wire_urx - backward_ext, wire_lly - side_ext, wire_urx, wire_lly)};
        break;
      case 1:
        return std::vector<PlanarRect>{PlanarRect(wire_urx, wire_lly, wire_urx + side_ext, wire_lly + backward_ext),
                                       PlanarRect(wire_llx - side_ext, wire_lly, wire_llx, wire_lly + backward_ext)};
        break;
      case 2:
        return std::vector<PlanarRect>{PlanarRect(wire_llx, wire_lly - side_ext, wire_llx + backward_ext, wire_lly),
                                       PlanarRect(wire_llx, wire_ury, wire_llx + backward_ext, wire_ury + side_ext)};
        break;
      case 3:
        return std::vector<PlanarRect>{PlanarRect(wire_llx - side_ext, wire_ury - backward_ext, wire_llx, wire_ury),
                                       PlanarRect(wire_urx, wire_ury - backward_ext, wire_urx + side_ext, wire_ury)};
        break;
      default:
        break;
    }
    return std::vector<PlanarRect>{};
  };
#endif

  std::vector<int32_t> cut_eol_spacing_layers = {1, 2, 3, 4, 5, 6};
  // 基础数据
  std::vector<Violation>& violation_list = rv_box.get_violation_list();
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DRCDM.getDatabase().get_cut_layer_list();
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = DRCDM.getDatabase().get_cut_to_adjacent_routing_map();

  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> cut_layer_all_query_tree;

  std::map<int32_t, bgi::rtree<BGRectInt, bgi::quadratic<16>>> routing_layer_query_tree;  // 所有的shape都融合在一起
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
    int32_t cut_layer_idx = routing_layer_idx;  // 对应routing layer底层的cut layer的idx

    int32_t violation_routing_layer_idx = -1;
    {
      std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[cut_layer_idx];
      violation_routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
    }
    if (DRCUTIL.exist(cut_eol_spacing_layers, cut_layer_idx) == false) {
      continue;  // 没有对应的cut layer  此时routing_layer_idx 刚好对应其下一层的cut layer idx
    }

    int32_t min_width = routing_layer_list[routing_layer_idx].get_minimum_width_rule().min_width;

    std::vector<GTLHolePolyInt> gtl_hole_poly_list;
    gtl_poly_set.get(gtl_hole_poly_list);

    // polyset -> polygon -> maxrect -> cut
    for (GTLHolePolyInt& gtl_hole_poly : gtl_hole_poly_list) {
      // 获得polygon 的信息
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
      std::vector<Segment<PlanarCoord>> edge_list;
      for (int32_t i = 0; i < coord_size; i++) {
        PlanarCoord& pre_coord = coord_list[getIdx(i - 1, coord_size)];
        PlanarCoord& curr_coord = coord_list[i];
        PlanarCoord& post_coord = coord_list[getIdx(i + 1, coord_size)];
        convex_corner_list.push_back(DRCUTIL.isConvexCorner(DRCUTIL.getRotation(gtl_hole_poly), pre_coord, curr_coord, post_coord));
        edge_length_list.push_back(DRCUTIL.getManhattanDistance(pre_coord, curr_coord));
        edge_list.push_back(Segment<PlanarCoord>(pre_coord, curr_coord));
      }
      std::set<int32_t> eol_edge_idx_set;
      for (int32_t i = 0; i < coord_size; i++) {
        if (convex_corner_list[getIdx(i - 1, coord_size)] && convex_corner_list[i]) {
          eol_edge_idx_set.insert(i);
        }
      }

      // 获得cut
      // net_idx -> map(rect,overhang_segment)
      struct RectCompare  //  cut rect 作为key
      {
        bool operator()(const PlanarRect& a, const PlanarRect& b) const
        {
          if (a.get_ll_x() != b.get_ll_x()) {
            return a.get_ll_x() < b.get_ll_x();
          } else if (a.get_ll_y() != b.get_ll_y()) {
            return a.get_ll_y() < b.get_ll_y();
          } else if (a.get_ur_x() != b.get_ur_x()) {
            return a.get_ur_x() < b.get_ur_x();
          }
          return a.get_ur_y() < b.get_ur_y();
        }
      };
      std::map<int32_t, std::map<PlanarRect, std::vector<Segment<PlanarCoord>>, RectCompare>> net_cut_overhang_list;  // 记录overhang
      std::map<int32_t, std::map<PlanarRect, std::vector<PlanarRect>, RectCompare>> net_cut_metal_rect_list;  // 记录cut的metal rect  --和span rect open overlap
      std::map<int32_t, std::map<PlanarRect, std::vector<PlanarRect>, RectCompare>> net_cut_span_rect_list;   // cut 用来算span的rect --和cut open overlap
      std::vector<GTLRectInt> polygon_rect_list;
      gtl::get_max_rectangles(polygon_rect_list, gtl_hole_poly);
      for (GTLRectInt& gtl_polygon_rect : polygon_rect_list) {
        PlanarRect polygon_rect = DRCUTIL.convertToPlanarRect(gtl_polygon_rect);
        if (polygon_rect.getWidth() < min_width) {
          continue;
        }
        // 查出所有的cut
        std::vector<std::pair<BGRectInt, int32_t>> bg_cut_result;
        cut_layer_all_query_tree[cut_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(polygon_rect)), std::back_inserter(bg_cut_result));
        for (auto& [bg_cut, net_idx] : bg_cut_result) {
          PlanarRect cut_rect = DRCUTIL.convertToPlanarRect(bg_cut);
          if (DRCUTIL.isOpenOverlap(cut_rect, polygon_rect)) {
            net_cut_span_rect_list[net_idx][cut_rect].push_back(polygon_rect);
          }
          // 更新cut的overhang segment
          if (DRCUTIL.isInside(polygon_rect, cut_rect)) {
            update_cut_overhang(net_cut_overhang_list[net_idx][cut_rect], polygon_rect);
          }
        }
      }

      // 根据overhang进行cut spacing的检查
      for (auto& [net_idx, cut_overhang_list] : net_cut_overhang_list) {
        for (auto& [cut, overhang_list] : cut_overhang_list) {
          // 更新cut的metal rect  --和span rect open overlap
          for (GTLRectInt& gtl_polygon_rect : polygon_rect_list) {
            PlanarRect polygon_rect = DRCUTIL.convertToPlanarRect(gtl_polygon_rect);
            for (PlanarRect& span_rect : net_cut_span_rect_list[net_idx][cut]) {
              if (DRCUTIL.isOpenOverlap(polygon_rect, span_rect)) {
                net_cut_metal_rect_list[net_idx][cut].push_back(polygon_rect);
                continue;
              }
            }
          }

          // ENCLOSURE条件
          bool is_enclosure = false;
          std::vector<int32_t> overhang_dis_list(4, 0);
          overhang_dis_list[0] = std::abs(overhang_list[0].get_first().get_x() - cut.get_ur_x());  // east
          overhang_dis_list[1] = std::abs(overhang_list[1].get_first().get_y() - cut.get_ll_y());  // south
          overhang_dis_list[2] = std::abs(overhang_list[2].get_first().get_x() - cut.get_ll_x());  // west
          overhang_dis_list[3] = std::abs(overhang_list[3].get_first().get_y() - cut.get_ur_y());  // north
          for (int32_t i = 0; i < overhang_dis_list.size(); i++) {
            int32_t pre_idx = getIdx(i - 1, overhang_dis_list.size());
            int32_t post_idx = getIdx(i + 1, overhang_dis_list.size());
            if (overhang_dis_list[i] < smaller_overhang && (overhang_dis_list[pre_idx] == equal_overhang || overhang_dis_list[post_idx] == equal_overhang)) {
              is_enclosure = true;
            }
          }

          // eol edge 条件
          // 再判断条件2，eol edge是否满足
          bool is_eol_edge = false;
          std::vector<bool> is_eol_edge_list(4, false);
          for (auto& eol_idx : eol_edge_idx_set) {  // 遍历eol egde
            Segment<PlanarCoord>& eol_segment = edge_list[eol_idx];
            PlanarRect eol_segment_rect = DRCUTIL.getBoundingBox({eol_segment.get_first(), eol_segment.get_second()});
            for (int32_t i = 0; i < overhang_dis_list.size(); i++) {  // 找到对应的overhang segment
              PlanarRect overhang_segment_rect = DRCUTIL.getBoundingBox({overhang_list[i].get_first(), overhang_list[i].get_second()});
              if (DRCUTIL.isInside(eol_segment_rect, overhang_segment_rect) && edge_length_list[eol_idx] < eol_width) {
                is_eol_edge_list[i] = true;
                is_eol_edge = true;
              }
            }
          }

          // 更新overhang segment
          for (auto& polygon_segment : edge_list) {
            PlanarRect polygon_segment_rect = DRCUTIL.getBoundingBox({polygon_segment.get_first(), polygon_segment.get_second()});
            for (auto& cut_overhang : overhang_list) {
              if (DRCUTIL.isInside(polygon_segment_rect, DRCUTIL.getBoundingBox({cut_overhang.get_first(), cut_overhang.get_second()}))) {
                cut_overhang = polygon_segment;
              }
            }
          }

          // EXTENSION 条件
          // span length 记录
          int32_t max_x_span = 0;
          int32_t max_y_span = 0;
          for (PlanarRect& span_rect : net_cut_span_rect_list[net_idx][cut]) {
            int32_t x_span = span_rect.getXSpan();
            int32_t y_span = span_rect.getYSpan();
            max_x_span = std::max(max_x_span, x_span);
            max_y_span = std::max(max_y_span, y_span);
          }

          // 查找周围的routing矩形，原矩形的基础上膨胀side ext
          std::vector<BGRectInt> bg_env_rect_list;
          {
            std::vector<PlanarCoord> overhang_coord_list;
            for (auto& overhang_segment : overhang_list) {
              overhang_coord_list.push_back(overhang_segment.get_first());
              overhang_coord_list.push_back(overhang_segment.get_second());
            }
            PlanarRect check_rect = DRCUTIL.getEnlargedRect(DRCUTIL.getBoundingBox(overhang_coord_list), side_ext);
            routing_layer_query_tree[cut_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(check_rect)), std::back_inserter(bg_env_rect_list));
          }

          GTLPolySetInt env_gtl_poly_set;  // 周围env的环境
          for (auto& bg_env_rect : bg_env_rect_list) {
            PlanarRect env_rect = DRCUTIL.convertToPlanarRect(bg_env_rect);
            env_gtl_poly_set += DRCUTIL.convertToGTLRectInt(env_rect);
          }
          // env里需要减去cut所在的metal
          for (PlanarRect& cut_metal_rect : net_cut_metal_rect_list[net_idx][cut]) {
            env_gtl_poly_set -= DRCUTIL.convertToGTLRectInt(cut_metal_rect);
          }

          std::vector<bool> is_use_spacing_list(4, false);  // 用于判断cut需要用spacing的边
          //  遍历四条overhang并检查区域情况
          for (int32_t i = 0; i < overhang_dis_list.size(); i++) {
            int32_t pre_idx = getIdx(i - 1, overhang_dis_list.size());
            int32_t post_idx = getIdx(i + 1, overhang_dis_list.size());
            if (overhang_dis_list[i] >= smaller_overhang) {
              continue;  // overhang不满足条件
            }
            Direction direction = DRCUTIL.getDirection(overhang_list[i].get_first(), overhang_list[i].get_second());
            bool is_span_length = direction == Direction::kHorizontal ? max_x_span >= span_length : max_y_span >= span_length;
            // 是否和pre满足
            if (overhang_dis_list[pre_idx] == equal_overhang) {
              // 大于span则不需要考虑eol
              // 小于span需要考虑eol
              // 两条对边启用距离规则
              PlanarRect wire_rect = DRCUTIL.getBoundingBox(
                  {overhang_list[i].get_first(), overhang_list[i].get_second(), overhang_list[pre_idx].get_first(), overhang_list[pre_idx].get_second()});
              std::vector<PlanarRect> eol_ext_rect_list = gen_eol_search_rect(wire_rect, i);
              bool is_overlap_pre = is_rect_interact_polygon(DRCUTIL.convertToGTLRectInt(eol_ext_rect_list[0]), env_gtl_poly_set, false);
              bool is_overlap_post = is_rect_interact_polygon(DRCUTIL.convertToGTLRectInt(eol_ext_rect_list[1]), env_gtl_poly_set, false);
              if (is_span_length && is_overlap_pre || (!is_span_length && is_overlap_pre && !is_overlap_post && is_eol_edge_list[i])) {
                // 两条对边启用距离规则
                is_use_spacing_list[getIdx(i + 2, overhang_dis_list.size())] = true;
                is_use_spacing_list[getIdx(pre_idx + 2, overhang_dis_list.size())] = true;
              }
            }
            // 是否和post满足
            if (overhang_dis_list[post_idx] == equal_overhang) {
              // 两条对边启用距离规则
              PlanarRect wire_rect = DRCUTIL.getBoundingBox(
                  {overhang_list[i].get_first(), overhang_list[i].get_second(), overhang_list[post_idx].get_first(), overhang_list[post_idx].get_second()});
              std::vector<PlanarRect> eol_ext_rect_list = gen_eol_search_rect(wire_rect, i);
              bool is_overlap_pre = is_rect_interact_polygon(DRCUTIL.convertToGTLRectInt(eol_ext_rect_list[0]), env_gtl_poly_set, false);
              bool is_overlap_post = is_rect_interact_polygon(DRCUTIL.convertToGTLRectInt(eol_ext_rect_list[1]), env_gtl_poly_set, false);
              if (is_span_length && is_overlap_post || (!is_span_length && is_overlap_post && !is_overlap_pre && is_eol_edge_list[i])) {
                // 两条对边启用距离规则
                is_use_spacing_list[getIdx(i + 2, overhang_dis_list.size())] = true;
                is_use_spacing_list[getIdx(post_idx + 2, overhang_dis_list.size())] = true;
              }
            }
          }
          //  cut
          //  东南西北生成一个spacing区域
          // 东南西北的边进行膨胀即可
          PlanarCoord cut_ll = cut.get_ll();
          PlanarCoord cut_ur = cut.get_ur();
          PlanarCoord cut_ul = PlanarCoord(cut_ll.get_x(), cut_ur.get_y());
          PlanarCoord cut_lr = PlanarCoord(cut_ur.get_x(), cut_ll.get_y());
          PlanarRect cut_east_segment_rect = DRCUTIL.getOffsetRect(DRCUTIL.getBoundingBox({cut_lr, cut_ur}), PlanarCoord(1, 0));
          PlanarRect cut_south_segment_rect = DRCUTIL.getOffsetRect(DRCUTIL.getBoundingBox({cut_ll, cut_lr}), PlanarCoord(0, -1));
          PlanarRect cut_west_segment_rect = DRCUTIL.getOffsetRect(DRCUTIL.getBoundingBox({cut_ll, cut_ul}), PlanarCoord(-1, 0));
          PlanarRect cut_north_segment_rect = DRCUTIL.getOffsetRect(DRCUTIL.getBoundingBox({cut_ul, cut_ur}), PlanarCoord(0, 1));

          std::vector<PlanarRect> cut_spacing_rect_list;
          std::vector<PlanarRect> cut_spacing_b_rect_list;
          if (is_use_spacing_list[0]) {
            cut_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(cut_east_segment_rect, 0, cut_spacing_b, cut_spacing_b, cut_spacing_b));
            cut_spacing_b_rect_list.push_back(DRCUTIL.getEnlargedRect(cut_east_segment_rect, 0, -1 * prl, cut_spacing_b, -1 * prl));
          }
          if (is_use_spacing_list[1]) {
            cut_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(cut_south_segment_rect, cut_spacing_b, cut_spacing_b, cut_spacing_b, 0));
            cut_spacing_b_rect_list.push_back(DRCUTIL.getEnlargedRect(cut_south_segment_rect, -1 * prl, cut_spacing_b, -1 * prl, 0));
          }
          if (is_use_spacing_list[2]) {
            cut_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(cut_west_segment_rect, cut_spacing_b, cut_spacing_b, 0, cut_spacing_b));
            cut_spacing_b_rect_list.push_back(DRCUTIL.getEnlargedRect(cut_west_segment_rect, cut_spacing_b, -1 * prl, 0, -1 * prl));
          }
          if (is_use_spacing_list[3]) {
            cut_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(cut_north_segment_rect, cut_spacing_b, 0, cut_spacing_b, cut_spacing_b));
            cut_spacing_b_rect_list.push_back(DRCUTIL.getEnlargedRect(cut_north_segment_rect, -1 * prl, 0, -1 * prl, cut_spacing_b));
          }
          if (cut_spacing_rect_list.empty()) {
            continue;  // 没有需要的spacing区域,跳过
          }
          GTLPolySetInt cut_spacing_a_region;
          GTLPolySetInt cut_spacing_b_region;
          for (PlanarRect& cut_spacing_rect : cut_spacing_rect_list) {
            cut_spacing_a_region += DRCUTIL.convertToGTLRectInt(cut_spacing_rect);
          }
          for (PlanarRect& cut_spacing_b_rect : cut_spacing_b_rect_list) {
            cut_spacing_b_region += DRCUTIL.convertToGTLRectInt(cut_spacing_b_rect);
          }
          // 由于b的rect 小于 a的rect，所以判断时要先判断b区域
          // 查找周围的cut rect
          std::vector<std::pair<BGRectInt, int32_t>> bg_env_cut_net_pair_list;
          {
            PlanarRect check_rect = DRCUTIL.getEnlargedRect(cut, cut_spacing_b);
            cut_layer_all_query_tree[cut_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(check_rect)),
                                                          std::back_inserter(bg_env_cut_net_pair_list));
          }

          // 和周围的cut逐个判断
          for (auto& [bg_env_cut, env_net_idx] : bg_env_cut_net_pair_list) {
            if (net_idx == -1 && env_net_idx == -1) {
              continue;  // -1忽略
            }

            PlanarRect env_cut = DRCUTIL.convertToPlanarRect(bg_env_cut);
            PlanarRect ori_cut = cut;
            if (env_cut == ori_cut) {
              continue;  // 自己和自己不算
            }
            int32_t required_size = cut_spacing_a;
            int32_t real_spacing = DRCUTIL.getEuclideanDistance(env_cut, ori_cut);
            // 是否在spacing b区域
            if (is_rect_interact_polygon(DRCUTIL.convertToGTLRectInt(env_cut), cut_spacing_b_region, true)) {
              required_size = cut_spacing_b;
            } else if (is_rect_interact_polygon(DRCUTIL.convertToGTLRectInt(env_cut), cut_spacing_a_region, true)) {
              required_size = cut_spacing_a;
            } else {
              continue;  // 不在spacing区域
            }
            if (real_spacing >= required_size) {
              continue;  // 满足条件
            }

            // 生成违例矩形
            PlanarRect violation_rect;
            if (DRCUTIL.isClosedOverlap(env_cut, ori_cut)) {
              violation_rect = DRCUTIL.getOverlap(env_cut, ori_cut);
            } else {
              violation_rect = DRCUTIL.getSpacingRect(env_cut, cut);
            }

            Violation violation;
            violation.set_violation_type(ViolationType::kCutEOLSpacing);
            violation.set_is_routing(true);
            violation.set_violation_net_set({net_idx, env_net_idx});
            violation.set_layer_idx(violation_routing_layer_idx);
            violation.set_rect(violation_rect);
            violation.set_required_size(required_size);
            violation_list.push_back(violation);
            //  env_cut
          }
        }
        // cut
      }
      // polygon
    }
    // polygon set
  }
}

}  // namespace idrc
