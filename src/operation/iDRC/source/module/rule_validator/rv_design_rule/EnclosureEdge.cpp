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
  std::vector<CutLayer>& cut_layer_list = DRCDM.getDatabase().get_cut_layer_list();
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = DRCDM.getDatabase().get_cut_to_adjacent_routing_map();

  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> routing_bg_rtree_map;
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
        std::vector<GTLRectInt> gtl_rect_list;
        gtl::get_max_rectangles(gtl_rect_list, gtl_poly_set);
        for (GTLRectInt& gtl_rect : gtl_rect_list) {
          routing_bg_rtree_map[routing_layer_idx].insert(std::make_pair(DRCUTIL.convertToBGRectInt(gtl_rect), net_idx));
        }
      }
    }
  }
  std::map<int32_t, std::vector<PlanarRect>> cut_rect_map;
  for (DRCShape* drc_shape : rv_box.get_drc_result_shape_list()) {
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
    for (PlanarRect& cut_rect : cut_rect_list) {
      for (int32_t routing_layer_idx : routing_layer_idx_list) {
        std::vector<std::pair<BGRectInt, int32_t>> bg_rect_net_pair_list;
        {
          routing_bg_rtree_map[routing_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(cut_rect)), std::back_inserter(bg_rect_net_pair_list));
        }
        std::map<Orientation, int32_t> orient_overhang_map;
        for (auto& [bg_rect, net_idx] : bg_rect_net_pair_list) {
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
        for (auto& [bg_rect, net_idx] : bg_rect_net_pair_list) {
          PlanarRect routing_rect = DRCUTIL.convertToPlanarRect(bg_rect);
          if (!DRCUTIL.isClosedOverlap(routing_rect, cut_rect)) {
            continue;
          }
          for (const Direction& curr_direction : {Direction::kHorizontal, Direction::kVertical}) {
            EnclosureEdgeRule curr_rule;
            {
              int32_t routing_rect_width;
              if (curr_direction == Direction::kHorizontal) {
                routing_rect_width = routing_rect.getYSpan();
              } else {
                routing_rect_width = routing_rect.getXSpan();
              }
              for (EnclosureEdgeRule& enclosure_edge_rule : cut_layer_list[cut_layer_idx].get_enclosure_edge_rule_list()) {
                if (routing_rect_width >= enclosure_edge_rule.min_width) {
                  curr_rule = enclosure_edge_rule;
                  break;
                }
              }
            }
            if (curr_rule.overhang < 0) {
              continue;
            }
            if (curr_rule.has_above && (routing_layer_idx != above_routing_layer_idx)) {
              continue;
            }
            if (curr_rule.has_below && (routing_layer_idx != below_routing_layer_idx)) {
              continue;
            }
            std::set<int32_t> left_par_net_idx_set;
            std::set<int32_t> right_par_net_idx_set;
            {
              PlanarRect left_par_rect;
              PlanarRect right_par_rect;
              if (curr_direction == Direction::kHorizontal) {
                left_par_rect = DRCUTIL.getEnlargedRect(routing_rect.get_ll(), 0, curr_rule.par_within, routing_rect.getXSpan(), 0);
                right_par_rect = DRCUTIL.getEnlargedRect(routing_rect.get_ur(), routing_rect.getXSpan(), 0, 0, curr_rule.par_within);
              } else {
                left_par_rect = DRCUTIL.getEnlargedRect(routing_rect.get_ll(), curr_rule.par_within, 0, 0, routing_rect.getYSpan());
                right_par_rect = DRCUTIL.getEnlargedRect(routing_rect.get_ur(), 0, routing_rect.getYSpan(), curr_rule.par_within, 0);
              }
              std::vector<std::pair<BGRectInt, int32_t>> env_bg_rect_net_pair_list;
              {
                PlanarRect check_rect;
                if (curr_direction == Direction::kHorizontal) {
                  check_rect = DRCUTIL.getEnlargedRect(routing_rect, 0, curr_rule.par_within);
                } else {
                  check_rect = DRCUTIL.getEnlargedRect(routing_rect, curr_rule.par_within, 0);
                }
                routing_bg_rtree_map[routing_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(check_rect)),
                                                              std::back_inserter(env_bg_rect_net_pair_list));
              }
              for (auto& [bg_env_rect, env_net_idx] : env_bg_rect_net_pair_list) {
                PlanarRect env_routing_rect = DRCUTIL.convertToPlanarRect(bg_env_rect);
                if (DRCUTIL.isClosedOverlap(routing_rect, env_routing_rect)) {
                  continue;
                }
                if (DRCUTIL.isOpenOverlap(env_routing_rect, left_par_rect)) {
                  if (DRCUTIL.getParallelLength(env_routing_rect, left_par_rect) > curr_rule.par_length) {
                    left_par_net_idx_set.insert(env_net_idx);
                  }
                }
                if (DRCUTIL.isOpenOverlap(env_routing_rect, right_par_rect)) {
                  if (DRCUTIL.getParallelLength(env_routing_rect, right_par_rect) > curr_rule.par_length) {
                    right_par_net_idx_set.insert(env_net_idx);
                  }
                }
              }
            }
            if (curr_rule.has_except_two_edges && !left_par_net_idx_set.empty() && !right_par_net_idx_set.empty()) {
              continue;
            }
            std::set<std::set<int32_t>> violation_net_set_set;
            {
              if ((curr_direction == Direction::kHorizontal && orient_overhang_map[Orientation::kSouth] < curr_rule.overhang)
                  || (curr_direction == Direction::kVertical && orient_overhang_map[Orientation::kWest] < curr_rule.overhang)) {
                for (int32_t left_par_net_idx : left_par_net_idx_set) {
                  violation_net_set_set.insert({left_par_net_idx, net_idx});
                }
              }
              if ((curr_direction == Direction::kHorizontal && orient_overhang_map[Orientation::kNorth] < curr_rule.overhang)
                  || (curr_direction == Direction::kVertical && orient_overhang_map[Orientation::kEast] < curr_rule.overhang)) {
                for (int32_t right_par_net_idx : right_par_net_idx_set) {
                  violation_net_set_set.insert({right_par_net_idx, net_idx});
                }
              }
            }
            for (const std::set<int32_t>& violation_net_set : violation_net_set_set) {
              Violation violation;
              violation.set_violation_type(ViolationType::kEnclosureEdge);
              violation.set_is_routing(true);
              violation.set_violation_net_set(violation_net_set);
              violation.set_layer_idx(below_routing_layer_idx);
              violation.set_rect(cut_rect);
              violation.set_required_size(curr_rule.overhang);
              rv_box.get_violation_list().push_back(violation);
            }
          }
        }
      }
    }
  }
}

}  // namespace idrc
