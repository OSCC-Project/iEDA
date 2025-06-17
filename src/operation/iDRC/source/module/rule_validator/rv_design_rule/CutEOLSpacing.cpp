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

  auto gen_eol_search_rect = [&](PlanarRect& cut_wire_rect, int egde_dir, CutLayer& cut_layer) {
    std::vector<PlanarRect> search_rect_list;  // 0
    int32_t wire_llx = cut_wire_rect.get_ll_x();
    int32_t wire_lly = cut_wire_rect.get_ll_y();
    int32_t wire_urx = cut_wire_rect.get_ur_x();
    int32_t wire_ury = cut_wire_rect.get_ur_y();
    switch (egde_dir) {
      case 0:
        return std::vector<PlanarRect>{PlanarRect(wire_urx - cut_layer.get_cut_eol_spacing_rule().backward_ext, wire_ury, wire_urx,
                                                  wire_ury + cut_layer.get_cut_eol_spacing_rule().side_ext),
                                       PlanarRect(wire_urx - cut_layer.get_cut_eol_spacing_rule().backward_ext,
                                                  wire_lly - cut_layer.get_cut_eol_spacing_rule().side_ext, wire_urx, wire_lly)};
        break;
      case 1:
        return std::vector<PlanarRect>{PlanarRect(wire_urx, wire_lly, wire_urx + cut_layer.get_cut_eol_spacing_rule().side_ext,
                                                  wire_lly + cut_layer.get_cut_eol_spacing_rule().backward_ext),
                                       PlanarRect(wire_llx - cut_layer.get_cut_eol_spacing_rule().side_ext, wire_lly, wire_llx,
                                                  wire_lly + cut_layer.get_cut_eol_spacing_rule().backward_ext)};
        break;
      case 2:
        return std::vector<PlanarRect>{PlanarRect(wire_llx, wire_lly - cut_layer.get_cut_eol_spacing_rule().side_ext,
                                                  wire_llx + cut_layer.get_cut_eol_spacing_rule().backward_ext, wire_lly),
                                       PlanarRect(wire_llx, wire_ury, wire_llx + cut_layer.get_cut_eol_spacing_rule().backward_ext,
                                                  wire_ury + cut_layer.get_cut_eol_spacing_rule().side_ext)};
        break;
      case 3:
        return std::vector<PlanarRect>{PlanarRect(wire_llx - cut_layer.get_cut_eol_spacing_rule().side_ext,
                                                  wire_ury - cut_layer.get_cut_eol_spacing_rule().backward_ext, wire_llx, wire_ury),
                                       PlanarRect(wire_urx, wire_ury - cut_layer.get_cut_eol_spacing_rule().backward_ext,
                                                  wire_urx + cut_layer.get_cut_eol_spacing_rule().side_ext, wire_ury)};
        break;
      default:
        break;
    }
    return std::vector<PlanarRect>{};
  };
#endif

  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DRCDM.getDatabase().get_cut_layer_list();
  std::map<int32_t, std::vector<int32_t>>& routing_to_adjacent_cut_map = DRCDM.getDatabase().get_routing_to_adjacent_cut_map();
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = DRCDM.getDatabase().get_cut_to_adjacent_routing_map();

  std::map<int32_t, GTLPolySetInt> routing_gtl_poly_set_map;
  {
    for (DRCShape* drc_shape : rv_box.get_drc_env_shape_list()) {
      if (drc_shape->get_is_routing()) {
        routing_gtl_poly_set_map[drc_shape->get_layer_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
      }
    }
    for (DRCShape* drc_shape : rv_box.get_drc_result_shape_list()) {
      if (drc_shape->get_is_routing()) {
        routing_gtl_poly_set_map[drc_shape->get_layer_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
      }
    }
  }
  std::map<int32_t, bgi::rtree<BGRectInt, bgi::quadratic<16>>> routing_bg_rtree_map;
  for (auto& [routing_layer_idx, gtl_poly_set] : routing_gtl_poly_set_map) {
    std::vector<GTLRectInt> gtl_rect_list;
    gtl::get_max_rectangles(gtl_rect_list, gtl_poly_set);
    for (GTLRectInt& gtl_rect : gtl_rect_list) {
      routing_bg_rtree_map[routing_layer_idx].insert(DRCUTIL.convertToBGRectInt(gtl_rect));
    }
  }
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> cut_bg_rtree_map;
  {
    for (DRCShape* drc_shape : rv_box.get_drc_env_shape_list()) {
      if (!drc_shape->get_is_routing()) {
        cut_bg_rtree_map[drc_shape->get_layer_idx()].insert(std::make_pair(DRCUTIL.convertToBGRectInt(drc_shape->get_rect()), drc_shape->get_net_idx()));
      }
    }
    for (DRCShape* drc_shape : rv_box.get_drc_result_shape_list()) {
      if (!drc_shape->get_is_routing()) {
        cut_bg_rtree_map[drc_shape->get_layer_idx()].insert(std::make_pair(DRCUTIL.convertToBGRectInt(drc_shape->get_rect()), drc_shape->get_net_idx()));
      }
    }
  }

  for (auto& [routing_layer_idx, gtl_poly_set] : routing_gtl_poly_set_map) {
    int32_t cut_layer_idx = -1;
    {
      std::vector<int32_t>& cut_layer_idx_list = routing_to_adjacent_cut_map[routing_layer_idx];
      cut_layer_idx = *std::min_element(cut_layer_idx_list.begin(), cut_layer_idx_list.end());
    }
    if (cut_to_adjacent_routing_map[cut_layer_idx].size() < 2) {
      continue;
    }
    CutLayer& cut_layer = cut_layer_list[cut_layer_idx];
    int32_t eol_spacing = cut_layer.get_cut_eol_spacing_rule().eol_spacing;
    int32_t eol_prl = cut_layer.get_cut_eol_spacing_rule().eol_prl;
    int32_t eol_prl_spacing = cut_layer.get_cut_eol_spacing_rule().eol_prl_spacing;
    int32_t eol_width = cut_layer.get_cut_eol_spacing_rule().eol_width;
    int32_t smaller_overhang = cut_layer.get_cut_eol_spacing_rule().smaller_overhang;
    int32_t equal_overhang = cut_layer.get_cut_eol_spacing_rule().equal_overhang;
    int32_t side_ext = cut_layer.get_cut_eol_spacing_rule().side_ext;
    int32_t backward_ext = cut_layer.get_cut_eol_spacing_rule().backward_ext;
    int32_t span_length = cut_layer.get_cut_eol_spacing_rule().span_length;

    std::vector<GTLHolePolyInt> gtl_hole_poly_list;
    gtl_poly_set.get(gtl_hole_poly_list);
    for (GTLHolePolyInt& gtl_hole_poly : gtl_hole_poly_list) {
      int32_t coord_size = static_cast<int32_t>(gtl_hole_poly.size());
      if (coord_size < 4) {
        continue;
      }
      std::vector<PlanarCoord> coord_list;
      for (auto iter = gtl_hole_poly.begin(); iter != gtl_hole_poly.end(); iter++) {
        coord_list.push_back(DRCUTIL.convertToPlanarCoord(*iter));
      }
      std::vector<bool> convex_corner_list;
      std::vector<Segment<PlanarCoord>> edge_list;
      std::vector<int32_t> edge_length_list;
      for (int32_t i = 0; i < coord_size; i++) {
        PlanarCoord& pre_coord = coord_list[getIdx(i - 1, coord_size)];
        PlanarCoord& curr_coord = coord_list[i];
        PlanarCoord& post_coord = coord_list[getIdx(i + 1, coord_size)];
        convex_corner_list.push_back(DRCUTIL.isConvexCorner(DRCUTIL.getRotation(gtl_hole_poly), pre_coord, curr_coord, post_coord));
        edge_list.push_back(Segment<PlanarCoord>(pre_coord, curr_coord));
        edge_length_list.push_back(DRCUTIL.getManhattanDistance(pre_coord, curr_coord));
      }
      std::set<int32_t> eol_edge_idx_set;
      for (int32_t i = 0; i < coord_size; i++) {
        if (convex_corner_list[getIdx(i - 1, coord_size)] && convex_corner_list[i]) {
          eol_edge_idx_set.insert(i);
        }
      }
      // 获得cut
      // net_idx -> map(rect,overhang_segment)
      std::map<int32_t, std::map<PlanarRect, std::vector<Segment<PlanarCoord>>, CmpPlanarRectByXASC>> net_cut_overhang_list;
      // cut 用来算span的rect --和cut open overlap
      std::map<int32_t, std::map<PlanarRect, std::vector<PlanarRect>, CmpPlanarRectByXASC>> net_cut_span_rect_list;
      std::vector<GTLRectInt> polygon_rect_list;
      gtl::get_max_rectangles(polygon_rect_list, gtl_hole_poly);
      for (GTLRectInt& gtl_polygon_rect : polygon_rect_list) {
        PlanarRect polygon_rect = DRCUTIL.convertToPlanarRect(gtl_polygon_rect);
        if (polygon_rect.getWidth() < routing_layer_list[routing_layer_idx].get_minimum_width_rule().min_width) {
          continue;
        }
        // 查出所有的cut
        std::vector<std::pair<BGRectInt, int32_t>> bg_cut_result;
        cut_bg_rtree_map[cut_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(polygon_rect)), std::back_inserter(bg_cut_result));
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

      std::map<int32_t, std::map<PlanarRect, std::vector<PlanarRect>, CmpPlanarRectByXASC>>
          net_cut_metal_rect_list;  // 记录cut的metal rect  --和span rect open overlap

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
            routing_bg_rtree_map[cut_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(check_rect)), std::back_inserter(bg_env_rect_list));
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
              std::vector<PlanarRect> eol_ext_rect_list = gen_eol_search_rect(wire_rect, i, cut_layer);
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
              std::vector<PlanarRect> eol_ext_rect_list = gen_eol_search_rect(wire_rect, i, cut_layer);
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
            cut_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(cut_east_segment_rect, 0, eol_prl_spacing, eol_prl_spacing, eol_prl_spacing));
            cut_spacing_b_rect_list.push_back(DRCUTIL.getEnlargedRect(cut_east_segment_rect, 0, -1 * eol_prl, eol_prl_spacing, -1 * eol_prl));
          }
          if (is_use_spacing_list[1]) {
            cut_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(cut_south_segment_rect, eol_prl_spacing, eol_prl_spacing, eol_prl_spacing, 0));
            cut_spacing_b_rect_list.push_back(DRCUTIL.getEnlargedRect(cut_south_segment_rect, -1 * eol_prl, eol_prl_spacing, -1 * eol_prl, 0));
          }
          if (is_use_spacing_list[2]) {
            cut_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(cut_west_segment_rect, eol_prl_spacing, eol_prl_spacing, 0, eol_prl_spacing));
            cut_spacing_b_rect_list.push_back(DRCUTIL.getEnlargedRect(cut_west_segment_rect, eol_prl_spacing, -1 * eol_prl, 0, -1 * eol_prl));
          }
          if (is_use_spacing_list[3]) {
            cut_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(cut_north_segment_rect, eol_prl_spacing, 0, eol_prl_spacing, eol_prl_spacing));
            cut_spacing_b_rect_list.push_back(DRCUTIL.getEnlargedRect(cut_north_segment_rect, -1 * eol_prl, 0, -1 * eol_prl, eol_prl_spacing));
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
            PlanarRect check_rect = DRCUTIL.getEnlargedRect(cut, eol_prl_spacing);
            cut_bg_rtree_map[cut_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(check_rect)), std::back_inserter(bg_env_cut_net_pair_list));
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
            int32_t required_size = eol_spacing;
            int32_t real_spacing = DRCUTIL.getEuclideanDistance(env_cut, ori_cut);
            // 是否在spacing b区域
            if (is_rect_interact_polygon(DRCUTIL.convertToGTLRectInt(env_cut), cut_spacing_b_region, true)) {
              required_size = eol_prl_spacing;
            } else if (is_rect_interact_polygon(DRCUTIL.convertToGTLRectInt(env_cut), cut_spacing_a_region, true)) {
              required_size = eol_spacing;
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

            int32_t violation_routing_layer_idx = -1;
            {
              std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[cut_layer_idx];
              violation_routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
            }

            Violation violation;
            violation.set_violation_type(ViolationType::kCutEOLSpacing);
            violation.set_is_routing(true);
            violation.set_violation_net_set({net_idx, env_net_idx});
            violation.set_layer_idx(violation_routing_layer_idx);
            violation.set_rect(violation_rect);
            violation.set_required_size(required_size);
            rv_box.get_violation_list().push_back(violation);
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
