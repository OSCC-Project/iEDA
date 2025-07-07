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

void RuleValidator::verifyCornerSpacing(RVBox& rv_box)
{
  /*
    PROPERTY LEF58_CORNERSPACING "CORNERSPACING CONVEXCORNER EXCEPTEOL 0.08
  WIDTH 0.00 SPACING 0.10
  WIDTH 0.20 SPACING 0.20
  WIDTH 0.50 SPACING 0.30 ;" ;
  */
  // corner to corner rule
  struct CornerSpacingRule
  {
    bool has_convex_corner = false;
    bool has_except_eol = false;
    int32_t eolwidth = 0;
    std::vector<std::pair<int32_t, int32_t>> width_spacing_list;
  };
  std::map<int32_t, CornerSpacingRule> layer_corner_spacing_rule;  // M2-M7
  for (int32_t i = 1; i <= 6; i++) {
    CornerSpacingRule rule = {true, true, 160, {{0, 200}, {400, 400}, {1000, 600}}};
    layer_corner_spacing_rule[i] = rule;  // M2-M7
  }
///////////
#if 1  // 数据结构定义
  struct PolyInfo
  {
    int32_t coord_size = -1;
    std::vector<PlanarCoord> coord_list;
    std::vector<bool> convex_corner_list;
    std::vector<Segment<PlanarCoord>> edge_list;
    std::vector<int32_t> edge_length_list;
    std::set<int32_t> eol_edge_idx_set;
    GTLHolePolyInt gtl_hole_poly;
    int32_t poly_info_idx = -1;
  };
#endif
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  std::map<int32_t, std::vector<int32_t>>& routing_to_adjacent_cut_map = DRCDM.getDatabase().get_routing_to_adjacent_cut_map();

  std::map<int32_t, std::map<int32_t, std::vector<PolyInfo>>> routing_net_poly_info_map;
  {
    std::map<int32_t, std::map<int32_t, GTLPolySetInt>> routing_net_gtl_poly_set_map;
    for (DRCShape* drc_shape : rv_box.get_drc_env_shape_list()) {
      if (drc_shape->get_is_routing()) {
        routing_net_gtl_poly_set_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
      }
    }
    for (DRCShape* drc_shape : rv_box.get_drc_result_shape_list()) {
      if (drc_shape->get_is_routing()) {
        routing_net_gtl_poly_set_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
      }
    }
    for (auto& [routing_layer_idx, net_gtl_poly_set_map] : routing_net_gtl_poly_set_map) {
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
          routing_net_poly_info_map[routing_layer_idx][net_idx].emplace_back(coord_size, coord_list, convex_corner_list, edge_list, edge_length_list,
                                                                             eol_edge_idx_set, gtl_hole_poly);
        }
      }
    }
  }
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, std::pair<int32_t, int32_t>>, bgi::quadratic<16>>> routing_bg_rtree_map;
  {
    for (auto& [routing_layer_idx, net_poly_info_map] : routing_net_poly_info_map) {
      for (auto& [net_idx, poly_info_list] : net_poly_info_map) {
        for (int32_t i = 0; i < static_cast<int32_t>(poly_info_list.size()); i++) {
          std::vector<GTLRectInt> gtl_rect_list;
          gtl::get_max_rectangles(gtl_rect_list, poly_info_list[i].gtl_hole_poly);
          for (GTLRectInt& gtl_rect : gtl_rect_list) {
            routing_bg_rtree_map[routing_layer_idx].insert(std::make_pair(DRCUTIL.convertToBGRectInt(gtl_rect), std::make_pair(net_idx, i)));
          }
          poly_info_list[i].poly_info_idx = i;
        }
      }
    }
  }

  for (auto& [routing_layer_idx, net_poly_info_map] : routing_net_poly_info_map) {
    if (DRCUTIL.exist(layer_corner_spacing_rule, routing_layer_idx) == false) {
      continue;  // skip layer without corner spacing rule
    }
    CornerSpacingRule& curr_rule = layer_corner_spacing_rule[routing_layer_idx];
    for (auto& [net_idx, poly_info_list] : net_poly_info_map) {
      for (PolyInfo& poly_info : poly_info_list) {
        int32_t& coord_size = poly_info.coord_size;
        std::vector<PlanarCoord>& coord_list = poly_info.coord_list;
        std::vector<GTLRectInt> gtl_rect_list;
        gtl::get_max_rectangles(gtl_rect_list, poly_info.gtl_hole_poly);
        std::vector<PlanarRect> rect_list;
        std::set<int32_t> eol_edge_idx_set = poly_info.eol_edge_idx_set;
        for (GTLRectInt& gtl_rect : gtl_rect_list) {
          rect_list.push_back(DRCUTIL.convertToPlanarRect(gtl_rect));
        }
        for (int32_t eol_idx = 0; eol_idx < poly_info.convex_corner_list.size(); eol_idx++) {
          if (!poly_info.convex_corner_list[eol_idx]) {
            continue;  // skip non-convex corner
          }
          if(curr_rule.has_except_eol){
            int32_t eol_width = curr_rule.eolwidth;
            int32_t post_eol_idx = getIdx(eol_idx + 1, coord_size);
            if((DRCUTIL.exist(eol_edge_idx_set, eol_idx) && poly_info.edge_length_list[eol_idx] < eol_width) ||
               (DRCUTIL.exist(eol_edge_idx_set, post_eol_idx) && poly_info.edge_length_list[post_eol_idx] < eol_width)) {
              continue;  // skip if the adjacent edges are shorter than the eol width
            }
          }
          PlanarCoord eol_coord = coord_list[eol_idx];
          PlanarCoord pre_coord = coord_list[getIdx(eol_idx - 1, coord_size)];
          PlanarCoord post_coord = coord_list[getIdx(eol_idx + 1, coord_size)];
          PlanarRect corner_rect = DRCUTIL.getBoundingBox({pre_coord, eol_coord, post_coord});
          Orientation pre_orientation = DRCUTIL.getOrientation(pre_coord, eol_coord);
          Orientation post_orientation = DRCUTIL.getOrientation(eol_coord, post_coord);
          int32_t max_width = 0;
          for (PlanarRect& rect : rect_list) {
            if (DRCUTIL.isInside(rect, eol_coord)) {
              max_width = std::max(max_width, rect.getWidth());
            }
          }
          int32_t required_spacing = 0;
          for (auto& [width, spacing] : curr_rule.width_spacing_list) {
            if (max_width > width) {
              required_spacing = spacing;
            }
          }
          PlanarRect check_rect;
          if (pre_orientation == Orientation::kNorth && post_orientation == Orientation::kWest) {
            check_rect = DRCUTIL.getEnlargedRect(eol_coord, 0, 0, required_spacing, required_spacing);
          } else if (pre_orientation == Orientation::kWest && post_orientation == Orientation::kSouth) {
            check_rect = DRCUTIL.getEnlargedRect(eol_coord, required_spacing, 0, 0, required_spacing);
          } else if (pre_orientation == Orientation::kSouth && post_orientation == Orientation::kEast) {
            check_rect = DRCUTIL.getEnlargedRect(eol_coord, required_spacing, required_spacing, 0, 0);
          } else if (pre_orientation == Orientation::kEast && post_orientation == Orientation::kNorth) {
            check_rect = DRCUTIL.getEnlargedRect(eol_coord, 0, required_spacing, required_spacing, 0);
          } else {
            DRCLOG.error(Loc::current(), "Unrecognized orientation!");
          }

          std::map<int32_t, std::set<int32_t>> env_net_poly_info_idx_map;
          {
            std::vector<std::pair<BGRectInt, std::pair<int32_t, int32_t>>> bg_rect_net_pair_list;
            routing_bg_rtree_map[routing_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(check_rect)), std::back_inserter(bg_rect_net_pair_list));
            for (auto& [bg_env_rect, net_poly_info_idx_pair] : bg_rect_net_pair_list) {
              env_net_poly_info_idx_map[net_poly_info_idx_pair.first].insert(net_poly_info_idx_pair.second);
            }
          }
          std::map<Orientation, Orientation> orientation_map = {{Orientation::kEast, Orientation::kWest},
                                                                {Orientation::kWest, Orientation::kEast},
                                                                {Orientation::kSouth, Orientation::kNorth},
                                                                {Orientation::kNorth, Orientation::kSouth}};
          for (auto& [env_net_idx, env_poly_info_idx_set] : env_net_poly_info_idx_map) {
            for (int32_t env_poly_info_idx : env_poly_info_idx_set) {
              PolyInfo& env_poly_info = net_poly_info_map[env_net_idx][env_poly_info_idx];

              for (int32_t env_eol_idx = 0; env_eol_idx < env_poly_info.convex_corner_list.size(); env_eol_idx++) {
                if (!env_poly_info.convex_corner_list[env_eol_idx]) {
                  continue;  // skip non-convex corner
                }
                PlanarCoord env_eol_coord = env_poly_info.coord_list[env_eol_idx];
                PlanarCoord env_pre_coord = env_poly_info.coord_list[getIdx(env_eol_idx - 1, env_poly_info.coord_size)];
                PlanarCoord env_post_coord = env_poly_info.coord_list[getIdx(env_eol_idx + 1, env_poly_info.coord_size)];
                PlanarRect env_corner_rect = DRCUTIL.getBoundingBox({env_pre_coord, env_eol_coord, env_post_coord});
                if (DRCUTIL.getParallelLength(env_corner_rect, corner_rect) > 0) {
                  continue;  // skip if the corner rect is not parallel to the check rect
                }

                if (!(orientation_map[pre_orientation] == DRCUTIL.getOrientation(env_pre_coord, env_eol_coord)
                      && orientation_map[post_orientation] == DRCUTIL.getOrientation(env_eol_coord, env_post_coord))) {
                  continue;  // skip if the orientation is not matched
                }
                int32_t x_spacing = std::abs(env_eol_coord.get_x() - eol_coord.get_x());
                int32_t y_spacing = std::abs(env_eol_coord.get_y() - eol_coord.get_y());
                int32_t MAXXY_spacing = std::max(x_spacing, y_spacing);
                if (MAXXY_spacing >= required_spacing || MAXXY_spacing == 0) {
                  continue;  // skip if the spacing is not satisfied
                }
                if(DRCUTIL.isInside(check_rect, env_eol_coord) == false ) {
                  continue;  
                }
                PlanarRect violation_rect = DRCUTIL.getBoundingBox({eol_coord, env_eol_coord});
                Violation violation;
                violation.set_violation_type(ViolationType::kCornerSpacing);
                violation.set_required_size(required_spacing);
                violation.set_is_routing(true);
                violation.set_violation_net_set({net_idx, env_net_idx});
                violation.set_layer_idx(routing_layer_idx);
                violation.set_rect(violation_rect);
                rv_box.get_violation_list().push_back(violation);
              }
            }
          }
        }
      }
    }
  }
}
}  // namespace idrc
