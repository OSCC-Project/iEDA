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

void RuleValidator::verifyEnclosureParallel(RVCluster& rv_cluster)
{
#if 1  // 数据结构定义
  struct PolyInfo
  {
    int32_t coord_size = -1;
    std::vector<Segment<PlanarCoord>> edge_list;
    std::vector<int32_t> edge_length_list;
    std::set<int32_t> eol_edge_idx_set;
    GTLHolePolyInt gtl_hole_poly;
    int32_t poly_info_idx = -1;
  };
#endif
  std::vector<CutLayer>& cut_layer_list = DRCDM.getDatabase().get_cut_layer_list();
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = DRCDM.getDatabase().get_cut_to_adjacent_routing_map();

  std::map<int32_t, std::map<int32_t, std::vector<PolyInfo>>> routing_net_poly_info_map;
  {
    std::map<int32_t, std::map<int32_t, GTLPolySetInt>> routing_net_gtl_poly_set_map;
    for (DRCShape* drc_shape : rv_cluster.get_drc_env_shape_list()) {
      if (drc_shape->get_is_routing()) {
        routing_net_gtl_poly_set_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
      }
    }
    for (DRCShape* drc_shape : rv_cluster.get_drc_result_shape_list()) {
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
          routing_net_poly_info_map[routing_layer_idx][net_idx].emplace_back(coord_size, edge_list, edge_length_list, eol_edge_idx_set, gtl_hole_poly);
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
  std::map<int32_t, std::vector<PlanarRect>> cut_rect_map;
  for (DRCShape* drc_shape : rv_cluster.get_drc_result_shape_list()) {
    if (!drc_shape->get_is_routing()) {
      cut_rect_map[drc_shape->get_layer_idx()].push_back(drc_shape->get_rect());
    }
  }
  for (auto& [cut_layer_idx, cut_rect_list] : cut_rect_map) {
    std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[cut_layer_idx];
    if (routing_layer_idx_list.size() < 2) {
      continue;
    }
    int32_t above_routing_layer_idx = *std::max_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
    int32_t below_routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
    EnclosureParallelRule curr_rule = cut_layer_list[cut_layer_idx].get_enclosure_parallel_rule();
    for (PlanarRect& cut_rect : cut_rect_list) {
      std::set<Segment<PlanarCoord>, CmpSegmentXASC> processed_segment_set;
      for (int32_t routing_layer_idx : routing_layer_idx_list) {
        if (curr_rule.has_above && (routing_layer_idx != above_routing_layer_idx)) {
          continue;
        }
        if (curr_rule.has_below && (routing_layer_idx != below_routing_layer_idx)) {
          continue;
        }
        std::vector<std::pair<BGRectInt, std::pair<int32_t, int32_t>>> bg_rect_net_pair_list;
        routing_bg_rtree_map[routing_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(cut_rect)), std::back_inserter(bg_rect_net_pair_list));
        std::vector<Orientation> orientation_list;
        {
          std::map<Orientation, int32_t> orient_overhang_map;
          for (auto& [bg_rect, net_poly_info_idx_pair] : bg_rect_net_pair_list) {
            PlanarRect routing_rect = DRCUTIL.convertToPlanarRect(bg_rect);
            if (!DRCUTIL.isClosedOverlap(routing_rect, cut_rect)) {
              continue;
            }
            if (routing_rect.get_ll_x() <= cut_rect.get_ll_x()) {
              orient_overhang_map[Orientation::kWest]
                  = std::max(orient_overhang_map[Orientation::kWest], std::abs(cut_rect.get_ll_x() - routing_rect.get_ll_x()));
            }
            if (routing_rect.get_ur_x() >= cut_rect.get_ur_x()) {
              orient_overhang_map[Orientation::kEast]
                  = std::max(orient_overhang_map[Orientation::kEast], std::abs(cut_rect.get_ur_x() - routing_rect.get_ur_x()));
            }
            if (routing_rect.get_ur_y() >= cut_rect.get_ur_y()) {
              orient_overhang_map[Orientation::kNorth]
                  = std::max(orient_overhang_map[Orientation::kNorth], std::abs(cut_rect.get_ur_y() - routing_rect.get_ur_y()));
            }
            if (routing_rect.get_ll_y() <= cut_rect.get_ll_y()) {
              orient_overhang_map[Orientation::kSouth]
                  = std::max(orient_overhang_map[Orientation::kSouth], std::abs(cut_rect.get_ll_y() - routing_rect.get_ll_y()));
            }
          }
          for (auto& [orient, overhang] : orient_overhang_map) {
            if (overhang >= curr_rule.overhang) {
              continue;
            }
            orientation_list.push_back(orient);
          }
        }
        for (auto& [bg_rect, net_poly_info_idx_pair] : bg_rect_net_pair_list) {
          int32_t net_idx = net_poly_info_idx_pair.first;
          PlanarRect routing_rect = DRCUTIL.convertToPlanarRect(bg_rect);
          if (!DRCUTIL.isClosedOverlap(routing_rect, cut_rect)) {
            continue;
          }
          PolyInfo& poly_info = routing_net_poly_info_map[routing_layer_idx][net_idx][net_poly_info_idx_pair.second];
          for (Orientation& orient : orientation_list) {
            int32_t edge_idx = -1;
            for (int32_t eol_idx : poly_info.eol_edge_idx_set) {
              if (DRCUTIL.isInside(poly_info.edge_list[eol_idx], routing_rect.getOrientEdge(orient))) {
                edge_idx = eol_idx;
                break;
              }
            }
            if (edge_idx == -1) {
              continue;
            }
            Segment<PlanarCoord> curr_segment = poly_info.edge_list[getIdx(edge_idx, poly_info.coord_size)];
            if (DRCUTIL.exist(processed_segment_set, curr_segment)) {
              continue;
            }
            if (DRCUTIL.getManhattanDistance(curr_segment.get_first(), curr_segment.get_second()) >= curr_rule.eol_width) {
              continue;
            }
            int32_t pre_segment_length;
            int32_t post_segment_length;
            {
              Segment<PlanarCoord> pre_segment = poly_info.edge_list[getIdx(edge_idx - 1, poly_info.coord_size)];
              Segment<PlanarCoord> post_segment = poly_info.edge_list[getIdx(edge_idx + 1, poly_info.coord_size)];
              pre_segment_length = DRCUTIL.getManhattanDistance(pre_segment.get_first(), pre_segment.get_second());
              post_segment_length = DRCUTIL.getManhattanDistance(post_segment.get_first(), post_segment.get_second());
            }
            if (curr_rule.has_min_length && pre_segment_length < curr_rule.min_length && post_segment_length < curr_rule.min_length) {
              continue;
            }
            processed_segment_set.insert(curr_segment);
            std::set<std::set<int32_t>> violation_net_set_set;
            {
              PlanarRect left_par_rect;
              PlanarRect right_par_rect;
              {
                int32_t par_spacing = curr_rule.par_spacing - DRCUTIL.getManhattanDistance(curr_segment.get_first(), curr_segment.get_second());
                if (orient == Orientation::kEast) {
                  left_par_rect = DRCUTIL.getEnlargedRect(curr_segment.get_first(), curr_rule.backward_ext + 1, par_spacing, curr_rule.forward_ext + 1, 0);
                  right_par_rect = DRCUTIL.getEnlargedRect(curr_segment.get_second(), curr_rule.backward_ext + 1, 0, curr_rule.forward_ext + 1, par_spacing);
                } else if (orient == Orientation::kWest) {
                  left_par_rect = DRCUTIL.getEnlargedRect(curr_segment.get_first(), curr_rule.forward_ext + 1, 0, curr_rule.backward_ext + 1, par_spacing);
                  right_par_rect = DRCUTIL.getEnlargedRect(curr_segment.get_second(), curr_rule.forward_ext + 1, par_spacing, curr_rule.backward_ext + 1, 0);
                } else if (orient == Orientation::kSouth) {
                  left_par_rect = DRCUTIL.getEnlargedRect(curr_segment.get_first(), par_spacing, curr_rule.forward_ext + 1, 0, curr_rule.backward_ext + 1);
                  right_par_rect = DRCUTIL.getEnlargedRect(curr_segment.get_second(), 0, curr_rule.forward_ext + 1, par_spacing, curr_rule.backward_ext + 1);
                } else if (orient == Orientation::kNorth) {
                  left_par_rect = DRCUTIL.getEnlargedRect(curr_segment.get_first(), 0, curr_rule.backward_ext + 1, par_spacing, curr_rule.forward_ext + 1);
                  right_par_rect = DRCUTIL.getEnlargedRect(curr_segment.get_second(), par_spacing, curr_rule.backward_ext + 1, 0, curr_rule.forward_ext + 1);
                } else {
                  DRCLOG.error(Loc::current(), "The orientation is error!");
                }
              }
              std::vector<std::pair<BGRectInt, std::pair<int32_t, int32_t>>> env_bg_rect_net_pair_list;
              {
                PlanarRect check_rect;
                if (pre_segment_length >= curr_rule.min_length && post_segment_length >= curr_rule.min_length) {
                  check_rect = DRCUTIL.getBoundingBox({left_par_rect.get_ll(), left_par_rect.get_ur(), right_par_rect.get_ll(), right_par_rect.get_ur()});
                } else if (pre_segment_length >= curr_rule.min_length) {
                  check_rect = left_par_rect;
                } else if (post_segment_length >= curr_rule.min_length) {
                  check_rect = right_par_rect;
                }
                routing_bg_rtree_map[routing_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(check_rect)),
                                                              std::back_inserter(env_bg_rect_net_pair_list));
              }
              std::set<int32_t> left_par_net_idx_set;
              std::set<int32_t> right_par_net_idx_set;
              for (auto& [bg_env_rect, env_net_poly_info_idx_pair] : env_bg_rect_net_pair_list) {
                PlanarRect env_routing_rect = DRCUTIL.convertToPlanarRect(bg_env_rect);
                if (DRCUTIL.isClosedOverlap(routing_rect, env_routing_rect)) {
                  continue;
                }
                if (DRCUTIL.isOpenOverlap(env_routing_rect, left_par_rect)) {
                  left_par_net_idx_set.insert(env_net_poly_info_idx_pair.first);
                }
                if (DRCUTIL.isOpenOverlap(env_routing_rect, right_par_rect)) {
                  right_par_net_idx_set.insert(env_net_poly_info_idx_pair.first);
                }
              }
              for (int32_t left_par_net_idx : left_par_net_idx_set) {
                violation_net_set_set.insert({left_par_net_idx, net_idx});
              }
              for (int32_t right_par_net_idx : right_par_net_idx_set) {
                violation_net_set_set.insert({right_par_net_idx, net_idx});
              }
            }
            for (const std::set<int32_t>& violation_net_set : violation_net_set_set) {
              Violation violation;
              violation.set_violation_type(ViolationType::kEnclosureParallel);
              violation.set_is_routing(true);
              violation.set_violation_net_set(violation_net_set);
              violation.set_layer_idx(below_routing_layer_idx);
              violation.set_rect(cut_rect);
              violation.set_required_size(curr_rule.overhang);
              rv_cluster.get_violation_list().push_back(violation);
            }
          }
        }
      }
    }
  }
}

}  // namespace idrc
