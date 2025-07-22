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
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();

  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> routing_net_gtl_poly_set_map;
  for (DRCShape* drc_shape : rv_box.get_drc_env_shape_list()) {
    if (!drc_shape->get_is_routing() || drc_shape->get_net_idx() == -1) {
      continue;
    }
    routing_net_gtl_poly_set_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
  }
  for (DRCShape* drc_shape : rv_box.get_drc_result_shape_list()) {
    if (!drc_shape->get_is_routing()) {
      continue;
    }
    routing_net_gtl_poly_set_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
  }
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> routing_bg_rtree_map;
  for (auto& [routing_layer_idx, net_gtl_poly_set_map] : routing_net_gtl_poly_set_map) {
    for (auto& [net_idx, gtl_poly_set] : net_gtl_poly_set_map) {
      std::vector<GTLRectInt> gtl_rect_list;
      gtl::get_max_rectangles(gtl_rect_list, gtl_poly_set);
      for (GTLRectInt& gtl_rect : gtl_rect_list) {
        routing_bg_rtree_map[routing_layer_idx].insert(std::make_pair(DRCUTIL.convertToBGRectInt(gtl_rect), net_idx));
      }
    }
  }
  for (auto& [routing_layer_idx, net_gtl_poly_set_map] : routing_net_gtl_poly_set_map) {
    RoutingLayer& routing_layer = routing_layer_list[routing_layer_idx];
    CornerFillSpacingRule& corner_fill_spacing_rule = routing_layer.get_corner_fill_spacing_rule();
    if (!corner_fill_spacing_rule.has_corner_fill) {
      continue;
    }
    int32_t corner_fill_spacing = corner_fill_spacing_rule.corner_fill_spacing;
    int32_t edge_length_1 = corner_fill_spacing_rule.edge_length_1;
    int32_t edge_length_2 = corner_fill_spacing_rule.edge_length_2;
    int32_t adjacent_eol = corner_fill_spacing_rule.adjacent_eol;
    std::map<std::set<int32_t>, GTLPolySetInt> net_violation_poly_set_map;
    for (auto& [net_idx, gtl_poly_set] : net_gtl_poly_set_map) {
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
        std::set<int32_t> valid_idx_set;
        {
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
          for (int32_t i = 0; i < coord_size; i++) {
            if (convex_corner_list[i]) {
              continue;
            }
            if (edge_length_list[i] < edge_length_1 && edge_length_list[getIdx(i + 1, coord_size)] < edge_length_2) {
              if (DRCUTIL.exist(eol_edge_idx_set, getIdx(i + 2, coord_size)) && edge_length_list[getIdx(i + 2, coord_size)] < adjacent_eol) {
                valid_idx_set.insert(i);
              }
            }
            if (edge_length_list[getIdx(i + 1, coord_size)] < edge_length_1 && edge_length_list[i] < edge_length_2) {
              if (DRCUTIL.exist(eol_edge_idx_set, getIdx(i - 1, coord_size)) && edge_length_list[getIdx(i - 1, coord_size)] < adjacent_eol) {
                valid_idx_set.insert(i);
              }
            }
          }
        }
        for (int32_t valid_idx : valid_idx_set) {
          PlanarCoord curr_coord = coord_list[valid_idx];
          PlanarRect corner_fill_rect;
          std::vector<Orientation> orientation_list;
          {
            PlanarCoord pre_coord = coord_list[getIdx(valid_idx - 1, coord_size)];
            PlanarCoord post_coord = coord_list[getIdx(valid_idx + 1, coord_size)];
            PlanarCoord diag_coord(pre_coord.get_x() ^ curr_coord.get_x() ^ post_coord.get_x(), pre_coord.get_y() ^ curr_coord.get_y() ^ post_coord.get_y());
            corner_fill_rect = DRCUTIL.getRect(curr_coord, diag_coord);
            orientation_list = DRCUTIL.getOrientationList(curr_coord, diag_coord);
          }
          PlanarRect check_rect;
          {
            check_rect = corner_fill_rect;
            if (DRCUTIL.exist(orientation_list, Orientation::kEast)) {
              check_rect = DRCUTIL.getEnlargedRect(check_rect, 0, 0, corner_fill_spacing, 0);
            } else if (DRCUTIL.exist(orientation_list, Orientation::kWest)) {
              check_rect = DRCUTIL.getEnlargedRect(check_rect, corner_fill_spacing, 0, 0, 0);
            }
            if (DRCUTIL.exist(orientation_list, Orientation::kSouth)) {
              check_rect = DRCUTIL.getEnlargedRect(check_rect, 0, corner_fill_spacing, 0, 0);
            } else if (DRCUTIL.exist(orientation_list, Orientation::kNorth)) {
              check_rect = DRCUTIL.getEnlargedRect(check_rect, 0, 0, 0, corner_fill_spacing);
            }
          }
          std::vector<std::pair<BGRectInt, int32_t>> bg_rect_net_pair_list;
          routing_bg_rtree_map[routing_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(check_rect)), std::back_inserter(bg_rect_net_pair_list));
          for (auto& [bg_env_rect, env_net_idx] : bg_rect_net_pair_list) {
            PlanarRect env_rect = DRCUTIL.convertToPlanarRect(bg_env_rect);
            GTLRectInt env_gtl_rect = DRCUTIL.convertToGTLRectInt(bg_env_rect);
            if (gtl::area(gtl_hole_poly & env_gtl_rect) == gtl::area(env_gtl_rect)) {
              continue;
            }
            if (!DRCUTIL.isOpenOverlap(env_rect, check_rect)) {
              continue;
            }
            PlanarRect overlap_rect = DRCUTIL.getOverlap(env_rect, check_rect);
            if (DRCUTIL.getEuclideanDistance(corner_fill_rect, overlap_rect) >= corner_fill_spacing) {
              continue;
            }
            PlanarCoord violation_coord;
            if (DRCUTIL.exist(orientation_list, Orientation::kEast)) {
              violation_coord.set_x(corner_fill_rect.get_ll_x() == overlap_rect.get_ll_x() ? corner_fill_rect.get_ur_x() : overlap_rect.get_ll_x());
            } else if (DRCUTIL.exist(orientation_list, Orientation::kWest)) {
              violation_coord.set_x(corner_fill_rect.get_ur_x() == overlap_rect.get_ur_x() ? corner_fill_rect.get_ll_x() : overlap_rect.get_ur_x());
            }
            if (DRCUTIL.exist(orientation_list, Orientation::kSouth)) {
              violation_coord.set_y(corner_fill_rect.get_ur_y() == overlap_rect.get_ur_y() ? corner_fill_rect.get_ll_y() : overlap_rect.get_ur_y());
            } else if (DRCUTIL.exist(orientation_list, Orientation::kNorth)) {
              violation_coord.set_y(corner_fill_rect.get_ll_y() == overlap_rect.get_ll_y() ? corner_fill_rect.get_ur_y() : overlap_rect.get_ll_y());
            }
            net_violation_poly_set_map[{net_idx, env_net_idx}] += DRCUTIL.convertToGTLRectInt(DRCUTIL.getRect(curr_coord, violation_coord));
          }
        }
      }
    }
    for (auto& [net_idx_set, violation_poly_set] : net_violation_poly_set_map) {
      std::vector<GTLRectInt> violation_rect_list;
      gtl::get_max_rectangles(violation_rect_list, violation_poly_set);
      for (GTLRectInt& violation_rect : violation_rect_list) {
        Violation violation;
        violation.set_violation_type(ViolationType::kCornerFillSpacing);
        violation.set_required_size(corner_fill_spacing);
        violation.set_is_routing(true);
        violation.set_violation_net_set(net_idx_set);
        violation.set_layer_idx(routing_layer_idx);
        violation.set_rect(DRCUTIL.convertToPlanarRect(violation_rect));
        rv_box.get_violation_list().push_back(violation);
      }
    }
  }
}

}  // namespace idrc
