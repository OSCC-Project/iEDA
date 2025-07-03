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
      std::vector<Segment<PlanarCoord>> edge_list;
      std::vector<int32_t> edge_length_list;
      std::set<int32_t> eol_edge_idx_set;
      {
        std::vector<PlanarCoord> coord_list;
        for (auto iter = gtl_hole_poly.begin(); iter != gtl_hole_poly.end(); iter++) {
          coord_list.push_back(DRCUTIL.convertToPlanarCoord(*iter));
        }
        std::vector<bool> convex_corner_list;
        for (int32_t i = 0; i < coord_size; i++) {
          PlanarCoord& pre_coord = coord_list[getIdx(i - 1, coord_size)];
          PlanarCoord& curr_coord = coord_list[i];
          PlanarCoord& post_coord = coord_list[getIdx(i + 1, coord_size)];
          convex_corner_list.push_back(DRCUTIL.isConvexCorner(DRCUTIL.getRotation(gtl_hole_poly), pre_coord, curr_coord, post_coord));
          edge_list.push_back(Segment<PlanarCoord>(pre_coord, curr_coord));
          edge_length_list.push_back(DRCUTIL.getManhattanDistance(pre_coord, curr_coord));
        }
        for (int32_t i = 0; i < coord_size; i++) {
          if (convex_corner_list[getIdx(i - 1, coord_size)] && convex_corner_list[i]) {
            eol_edge_idx_set.insert(i);
          }
        }
      }
      std::map<int32_t, std::map<PlanarRect, std::vector<Segment<PlanarCoord>>, CmpPlanarRectByXASC>> net_cut_rect_overhang_map;
      std::map<int32_t, std::map<PlanarRect, std::vector<PlanarRect>, CmpPlanarRectByXASC>> net_cut_rect_span_rect_map;
      std::map<int32_t, std::map<PlanarRect, std::vector<PlanarRect>, CmpPlanarRectByXASC>> net_cut_rect_routing_rect_map;
      {
        std::vector<GTLRectInt> gtl_rect_list;
        gtl::get_max_rectangles(gtl_rect_list, gtl_hole_poly);
        for (GTLRectInt& gtl_rect : gtl_rect_list) {
          PlanarRect routing_rect = DRCUTIL.convertToPlanarRect(gtl_rect);
          if (routing_rect.getWidth() < routing_layer_list[routing_layer_idx].get_minimum_width_rule().min_width) {
            continue;
          }
          std::vector<std::pair<BGRectInt, int32_t>> cut_bg_rect_net_pair_list;
          cut_bg_rtree_map[cut_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(routing_rect)), std::back_inserter(cut_bg_rect_net_pair_list));
          for (auto& [bg_env_rect, env_net_idx] : cut_bg_rect_net_pair_list) {
            PlanarRect cut_rect = DRCUTIL.convertToPlanarRect(bg_env_rect);
            if (DRCUTIL.isInside(routing_rect, cut_rect)) {
              std::vector<Segment<PlanarCoord>>& overhang_list = net_cut_rect_overhang_map[env_net_idx][cut_rect];
              Segment<PlanarCoord> east_segment = routing_rect.getOrientEdge(Orientation::kEast);
              Segment<PlanarCoord> south_segment = routing_rect.getOrientEdge(Orientation::kSouth);
              Segment<PlanarCoord> west_segment = routing_rect.getOrientEdge(Orientation::kWest);
              Segment<PlanarCoord> north_segment = routing_rect.getOrientEdge(Orientation::kNorth);
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
            }
            if (DRCUTIL.isOpenOverlap(routing_rect, cut_rect)) {
              net_cut_rect_span_rect_map[env_net_idx][cut_rect].push_back(routing_rect);
            }
          }
        }
        for (auto& [net_idx, cut_rect_overhang_map] : net_cut_rect_overhang_map) {
          for (auto& [cut_rect, overhang_list] : cut_rect_overhang_map) {
            for (GTLRectInt& gtl_rect : gtl_rect_list) {
              PlanarRect routing_rect = DRCUTIL.convertToPlanarRect(gtl_rect);
              for (PlanarRect& span_rect : net_cut_rect_span_rect_map[net_idx][cut_rect]) {
                if (DRCUTIL.isOpenOverlap(routing_rect, span_rect)) {
                  net_cut_rect_routing_rect_map[net_idx][cut_rect].push_back(routing_rect);
                  continue;
                }
              }
            }
          }
        }
      }
      for (auto& [net_idx, cut_rect_overhang_map] : net_cut_rect_overhang_map) {
        for (auto& [cut_rect, overhang_list] : cut_rect_overhang_map) {
          std::vector<bool> need_spacing_list(4, false);
          {
            int32_t max_x_span = 0;
            int32_t max_y_span = 0;
            for (PlanarRect& span_rect : net_cut_rect_span_rect_map[net_idx][cut_rect]) {
              max_x_span = std::max(max_x_span, span_rect.getXSpan());
              max_y_span = std::max(max_y_span, span_rect.getYSpan());
            }
            GTLPolySetInt env_gtl_poly_set;
            {
              std::vector<PlanarCoord> coord_list;
              for (Segment<PlanarCoord>& overhang : overhang_list) {
                coord_list.push_back(overhang.get_first());
                coord_list.push_back(overhang.get_second());
              }
              PlanarRect check_rect = DRCUTIL.getEnlargedRect(DRCUTIL.getBoundingBox(coord_list), side_ext);
              std::vector<BGRectInt> bg_rect_list;
              routing_bg_rtree_map[cut_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(check_rect)), std::back_inserter(bg_rect_list));
              for (auto& bg_rect : bg_rect_list) {
                env_gtl_poly_set += DRCUTIL.convertToGTLRectInt(bg_rect);
              }
              for (PlanarRect& routing_rect : net_cut_rect_routing_rect_map[net_idx][cut_rect]) {
                env_gtl_poly_set -= DRCUTIL.convertToGTLRectInt(routing_rect);
              }
            }
            std::vector<int32_t> overhang_distance_list(4, 0);
            overhang_distance_list[0] = std::abs(overhang_list[0].get_first().get_x() - cut_rect.get_ur_x());  // east
            overhang_distance_list[1] = std::abs(overhang_list[1].get_first().get_y() - cut_rect.get_ll_y());  // south
            overhang_distance_list[2] = std::abs(overhang_list[2].get_first().get_x() - cut_rect.get_ll_x());  // west
            overhang_distance_list[3] = std::abs(overhang_list[3].get_first().get_y() - cut_rect.get_ur_y());  // north
            for (int32_t i = 0; i < static_cast<int32_t>(overhang_distance_list.size()); i++) {
              if (overhang_distance_list[i] >= smaller_overhang) {
                continue;
              }
              bool is_span_length;
              if (DRCUTIL.isHorizontal(overhang_list[i].get_first(), overhang_list[i].get_second())) {
                is_span_length = max_x_span >= span_length;
              } else {
                is_span_length = max_y_span >= span_length;
              }
              bool is_eol_edge = false;
              for (int32_t eol_edge_idx : eol_edge_idx_set) {
                if (DRCUTIL.isInside(edge_list[eol_edge_idx], overhang_list[i]) && edge_length_list[eol_edge_idx] < eol_width) {
                  is_eol_edge = true;
                }
              }
              std::vector<std::pair<bool, int32_t>> is_pre_adj_i_list;
              is_pre_adj_i_list.emplace_back(true, getIdx(i - 1, overhang_distance_list.size()));
              is_pre_adj_i_list.emplace_back(false, getIdx(i + 1, overhang_distance_list.size()));
              for (auto& [is_pre, adj_i] : is_pre_adj_i_list) {
                if (overhang_distance_list[adj_i] != equal_overhang) {
                  continue;
                }
                bool need_spacing = false;
                {
                  PlanarRect rect = DRCUTIL.getBoundingBox(
                      {overhang_list[i].get_first(), overhang_list[i].get_second(), overhang_list[adj_i].get_first(), overhang_list[adj_i].get_second()});
                  GTLRectInt pre_rect;
                  GTLRectInt post_rect;
                  if (i == 0) {
                    pre_rect = GTLRectInt(rect.get_ur_x() - backward_ext, rect.get_ur_y(), rect.get_ur_x(), rect.get_ur_y() + side_ext);
                    post_rect = GTLRectInt(rect.get_ur_x() - backward_ext, rect.get_ll_y() - side_ext, rect.get_ur_x(), rect.get_ll_y());
                  } else if (i == 1) {
                    pre_rect = GTLRectInt(rect.get_ur_x(), rect.get_ll_y(), rect.get_ur_x() + side_ext, rect.get_ll_y() + backward_ext);
                    post_rect = GTLRectInt(rect.get_ll_x() - side_ext, rect.get_ll_y(), rect.get_ll_x(), rect.get_ll_y() + backward_ext);
                  } else if (i == 2) {
                    pre_rect = GTLRectInt(rect.get_ll_x(), rect.get_ll_y() - side_ext, rect.get_ll_x() + backward_ext, rect.get_ll_y());
                    post_rect = GTLRectInt(rect.get_ll_x(), rect.get_ur_y(), rect.get_ll_x() + backward_ext, rect.get_ur_y() + side_ext);
                  } else if (i == 3) {
                    pre_rect = GTLRectInt(rect.get_ll_x() - side_ext, rect.get_ur_y() - backward_ext, rect.get_ll_x(), rect.get_ur_y());
                    post_rect = GTLRectInt(rect.get_ur_x(), rect.get_ur_y() - backward_ext, rect.get_ur_x() + side_ext, rect.get_ur_y());
                  }
                  bool is_overlap_pre = DRCUTIL.isOverlap(env_gtl_poly_set, pre_rect, false);
                  bool is_overlap_post = DRCUTIL.isOverlap(env_gtl_poly_set, post_rect, false);
                  if (is_pre) {
                    need_spacing = ((is_span_length && is_overlap_pre) || (!is_span_length && is_overlap_pre && !is_overlap_post && is_eol_edge));
                  } else {
                    need_spacing = ((is_span_length && is_overlap_post) || (!is_span_length && is_overlap_post && !is_overlap_pre && is_eol_edge));
                  }
                }
                if (need_spacing) {
                  need_spacing_list[getIdx(i + 2, overhang_distance_list.size())] = true;
                  need_spacing_list[getIdx(adj_i + 2, overhang_distance_list.size())] = true;
                }
              }
            }
          }
          GTLPolySetInt cut_spacing_a_region;
          GTLPolySetInt cut_spacing_b_region;
          {
            if (need_spacing_list[0]) {
              Segment<PlanarCoord> segment = cut_rect.getOrientEdge(Orientation::kEast);
              PlanarRect segment_rect = DRCUTIL.getOffsetRect(DRCUTIL.getRect(segment.get_first(), segment.get_second()), PlanarCoord(1, 0));
              cut_spacing_a_region += DRCUTIL.convertToGTLRectInt(DRCUTIL.getEnlargedRect(segment_rect, 0, -1 * eol_prl, eol_prl_spacing, -1 * eol_prl));
              cut_spacing_b_region += DRCUTIL.convertToGTLRectInt(DRCUTIL.getEnlargedRect(segment_rect, 0, eol_prl_spacing, eol_prl_spacing, eol_prl_spacing));
            }
            if (need_spacing_list[1]) {
              Segment<PlanarCoord> segment = cut_rect.getOrientEdge(Orientation::kSouth);
              PlanarRect segment_rect = DRCUTIL.getOffsetRect(DRCUTIL.getRect(segment.get_first(), segment.get_second()), PlanarCoord(0, -1));
              cut_spacing_a_region += DRCUTIL.convertToGTLRectInt(DRCUTIL.getEnlargedRect(segment_rect, -1 * eol_prl, eol_prl_spacing, -1 * eol_prl, 0));
              cut_spacing_b_region += DRCUTIL.convertToGTLRectInt(DRCUTIL.getEnlargedRect(segment_rect, eol_prl_spacing, eol_prl_spacing, eol_prl_spacing, 0));
            }
            if (need_spacing_list[2]) {
              Segment<PlanarCoord> segment = cut_rect.getOrientEdge(Orientation::kWest);
              PlanarRect segment_rect = DRCUTIL.getOffsetRect(DRCUTIL.getRect(segment.get_first(), segment.get_second()), PlanarCoord(-1, 0));
              cut_spacing_a_region += DRCUTIL.convertToGTLRectInt(DRCUTIL.getEnlargedRect(segment_rect, eol_prl_spacing, -1 * eol_prl, 0, -1 * eol_prl));
              cut_spacing_b_region += DRCUTIL.convertToGTLRectInt(DRCUTIL.getEnlargedRect(segment_rect, eol_prl_spacing, eol_prl_spacing, 0, eol_prl_spacing));
            }
            if (need_spacing_list[3]) {
              Segment<PlanarCoord> segment = cut_rect.getOrientEdge(Orientation::kNorth);
              PlanarRect segment_rect = DRCUTIL.getOffsetRect(DRCUTIL.getRect(segment.get_first(), segment.get_second()), PlanarCoord(0, 1));
              cut_spacing_a_region += DRCUTIL.convertToGTLRectInt(DRCUTIL.getEnlargedRect(segment_rect, -1 * eol_prl, 0, -1 * eol_prl, eol_prl_spacing));
              cut_spacing_b_region += DRCUTIL.convertToGTLRectInt(DRCUTIL.getEnlargedRect(segment_rect, eol_prl_spacing, 0, eol_prl_spacing, eol_prl_spacing));
            }
          }
          if (gtl::area(cut_spacing_b_region) == 0) {
            continue;
          }
          std::vector<std::pair<BGRectInt, int32_t>> bg_cut_rect_net_pair_list;
          cut_bg_rtree_map[cut_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(DRCUTIL.getEnlargedRect(cut_rect, eol_prl_spacing))),
                                                std::back_inserter(bg_cut_rect_net_pair_list));
          for (auto& [bg_env_cut_rect, env_net_idx] : bg_cut_rect_net_pair_list) {
            if (net_idx == -1 && env_net_idx == -1) {
              continue;
            }
            PlanarRect env_cut_rect = DRCUTIL.convertToPlanarRect(bg_env_cut_rect);
            GTLRectInt gtl_env_cut_rect = DRCUTIL.convertToGTLRectInt(bg_env_cut_rect);
            if (cut_rect == env_cut_rect) {
              continue;
            }
            int32_t required_size = eol_spacing;
            if (DRCUTIL.isOverlap(cut_spacing_a_region, gtl_env_cut_rect, true)) {
              required_size = eol_prl_spacing;
            } else if (!DRCUTIL.isOverlap(cut_spacing_b_region, gtl_env_cut_rect, true)) {
              continue;
            }
            if (DRCUTIL.getEuclideanDistance(cut_rect, env_cut_rect) >= required_size) {
              continue;
            }
            int32_t violation_routing_layer_idx = -1;
            {
              std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[cut_layer_idx];
              violation_routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
            }
            PlanarRect violation_rect;
            if (DRCUTIL.isClosedOverlap(cut_rect, env_cut_rect)) {
              violation_rect = DRCUTIL.getOverlap(cut_rect, env_cut_rect);
            } else {
              violation_rect = DRCUTIL.getSpacingRect(cut_rect, env_cut_rect);
            }
            Violation violation;
            violation.set_violation_type(ViolationType::kCutEOLSpacing);
            violation.set_is_routing(true);
            violation.set_violation_net_set({net_idx, env_net_idx});
            violation.set_layer_idx(violation_routing_layer_idx);
            violation.set_rect(violation_rect);
            violation.set_required_size(required_size);
            rv_box.get_violation_list().push_back(violation);
          }
        }
      }
    }
  }
}

}  // namespace idrc
