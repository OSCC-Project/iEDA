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

void RuleValidator::verifyEndOfLineSpacing(RVBox& rv_box)
{
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
  for (auto& [routing_layer_idx, net_poly_info_map] : routing_net_poly_info_map) {
    std::vector<EndOfLineSpacingRule>& end_of_line_spacing_rule_list = routing_layer_list[routing_layer_idx].get_end_of_line_spacing_rule_list();
    bool need_cut_shape = false;
    for (EndOfLineSpacingRule& end_of_line_spacing_rule : end_of_line_spacing_rule_list) {
      if (end_of_line_spacing_rule.has_enclose_cut) {
        need_cut_shape = true;
        break;
      }
    }
    std::map<Segment<PlanarCoord>, std::vector<std::pair<Violation, Segment<PlanarCoord>>>, CmpSegmentXASC> edge_violation_edge_map;
    for (auto& [net_idx, poly_info_list] : net_poly_info_map) {
      for (PolyInfo& poly_info : poly_info_list) {
        if (DRCUTIL.getRotation(poly_info.gtl_hole_poly) != Rotation::kCounterclockwise) {
          DRCLOG.error(Loc::current(), "The poly is error!");
        }
        std::vector<PlanarRect> adjacent_cut_shape_list;
        if (need_cut_shape) {
          std::vector<GTLRectInt> gtl_rect_list;
          gtl::get_max_rectangles(gtl_rect_list, poly_info.gtl_hole_poly);
          for (GTLRectInt& gtl_rect : gtl_rect_list) {
            std::vector<std::pair<BGRectInt, int32_t>> cut_bg_rect_net_pair_list;
            {
              // !现在只能处理below的cut
              std::vector<int32_t>& cut_layer_idx_list = routing_to_adjacent_cut_map[routing_layer_idx];
              int32_t cut_layer_idx = *std::min_element(cut_layer_idx_list.begin(), cut_layer_idx_list.end());
              cut_bg_rtree_map[cut_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(gtl_rect)), std::back_inserter(cut_bg_rect_net_pair_list));
            }
            for (auto& [bg_env_rect, env_net_idx] : cut_bg_rect_net_pair_list) {
              adjacent_cut_shape_list.push_back(DRCUTIL.convertToPlanarRect(bg_env_rect));
            }
          }
        }
        for (int32_t eol_edge_idx : poly_info.eol_edge_idx_set) {
          PlanarCoord pre_coord = poly_info.coord_list[getIdx(eol_edge_idx - 1, poly_info.coord_size)];
          PlanarCoord curr_coord = poly_info.coord_list[getIdx(eol_edge_idx, poly_info.coord_size)];
          if (!DRCUTIL.isRightAngled(pre_coord, curr_coord)) {
            DRCLOG.error(Loc::current(), "The edge is error!");
          }
          Direction direction = DRCUTIL.getDirection(pre_coord, curr_coord);
          Orientation orientation = DRCUTIL.getOrientation(pre_coord, curr_coord);
          // all spacing rect
          std::vector<PlanarRect> eol_spacing_rect_list;
          std::vector<PlanarRect> ete_spacing_rect_list;
          std::vector<PlanarRect> left_par_spacing_rect_list;
          std::vector<PlanarRect> right_par_spacing_rect_list;
          for (EndOfLineSpacingRule& end_of_line_spacing_rule : end_of_line_spacing_rule_list) {
            int32_t eol_spacing = end_of_line_spacing_rule.eol_spacing;
            int32_t eol_within = end_of_line_spacing_rule.eol_within;
            int32_t ete_spacing = end_of_line_spacing_rule.ete_spacing;
            int32_t par_spacing = 0;
            if (end_of_line_spacing_rule.has_subtrace_eol_width) {
              par_spacing = end_of_line_spacing_rule.par_spacing - poly_info.edge_length_list[eol_edge_idx];
            } else {
              par_spacing = end_of_line_spacing_rule.par_spacing;
            }
            int32_t par_within = end_of_line_spacing_rule.par_within;
            if (orientation == Orientation::kWest) {
              eol_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(pre_coord, curr_coord, eol_within, 0, eol_within, eol_spacing));
              left_par_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(curr_coord, par_spacing, par_within, 0, eol_within));
              right_par_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(pre_coord, 0, par_within, par_spacing, eol_within));
              if (end_of_line_spacing_rule.has_ete) {
                ete_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(pre_coord, curr_coord, eol_within, 0, eol_within, ete_spacing));
              } else {
                ete_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(pre_coord, curr_coord, eol_within, 0, eol_within, eol_spacing));
              }
            } else if (orientation == Orientation::kEast) {
              eol_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(pre_coord, curr_coord, eol_within, eol_spacing, eol_within, 0));
              left_par_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(curr_coord, 0, eol_within, par_spacing, par_within));
              right_par_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(pre_coord, par_spacing, eol_within, 0, par_within));
              if (end_of_line_spacing_rule.has_ete) {
                ete_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(pre_coord, curr_coord, eol_within, ete_spacing, eol_within, 0));
              } else {
                ete_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(pre_coord, curr_coord, eol_within, eol_spacing, eol_within, 0));
              }
            } else if (orientation == Orientation::kSouth) {
              eol_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(pre_coord, curr_coord, eol_spacing, eol_within, 0, eol_within));
              left_par_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(curr_coord, eol_within, par_spacing, par_within, 0));
              right_par_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(pre_coord, eol_within, 0, par_within, par_spacing));
              if (end_of_line_spacing_rule.has_ete) {
                ete_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(pre_coord, curr_coord, ete_spacing, eol_within, 0, eol_within));
              } else {
                ete_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(pre_coord, curr_coord, eol_spacing, eol_within, 0, eol_within));
              }
            } else if (orientation == Orientation::kNorth) {
              eol_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(pre_coord, curr_coord, 0, eol_within, eol_spacing, eol_within));
              left_par_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(curr_coord, par_within, 0, eol_within, par_spacing));
              right_par_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(pre_coord, par_within, par_spacing, eol_within, 0));
              if (end_of_line_spacing_rule.has_ete) {
                ete_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(pre_coord, curr_coord, 0, eol_within, ete_spacing, eol_within));
              } else {
                ete_spacing_rect_list.push_back(DRCUTIL.getEnlargedRect(pre_coord, curr_coord, 0, eol_within, eol_spacing, eol_within));
              }
            }
          }
          // env_net_poly_info_idx_map
          std::map<int32_t, std::set<int32_t>> env_net_poly_info_idx_map;
          {
            int32_t max_eol_spacing = 0;
            int32_t max_eol_width = 0;
            int32_t max_par_spacing = 0;
            int32_t max_par_within = 0;
            for (const EndOfLineSpacingRule& end_of_line_spacing_rule : end_of_line_spacing_rule_list) {
              max_eol_spacing = std::max(max_eol_spacing, end_of_line_spacing_rule.eol_spacing);
              max_eol_width = std::max(max_eol_width, end_of_line_spacing_rule.eol_width);
              if (end_of_line_spacing_rule.has_subtrace_eol_width) {
                max_par_spacing = std::max(max_par_spacing, end_of_line_spacing_rule.par_spacing - poly_info.edge_length_list[eol_edge_idx]);
              } else {
                max_par_spacing = std::max(max_par_spacing, end_of_line_spacing_rule.par_spacing);
              }
              max_par_within = std::max(max_par_within, end_of_line_spacing_rule.par_within);
            }
            PlanarRect max_check_rect;
            if (orientation == Orientation::kEast) {
              max_check_rect = DRCUTIL.getEnlargedRect(pre_coord, curr_coord, std::max(max_eol_width, max_par_spacing), max_eol_spacing,
                                                       std::max(max_eol_width, max_par_spacing), max_par_within);
            } else if (orientation == Orientation::kWest) {
              max_check_rect = DRCUTIL.getEnlargedRect(pre_coord, curr_coord, std::max(max_eol_width, max_par_spacing), max_par_within,
                                                       std::max(max_eol_width, max_par_spacing), max_eol_spacing);
            } else if (orientation == Orientation::kSouth) {
              max_check_rect = DRCUTIL.getEnlargedRect(pre_coord, curr_coord, max_eol_spacing, std::max(max_eol_width, max_par_spacing), max_par_within,
                                                       std::max(max_eol_width, max_par_spacing));
            } else if (orientation == Orientation::kNorth) {
              max_check_rect = DRCUTIL.getEnlargedRect(pre_coord, curr_coord, max_par_within, std::max(max_eol_width, max_par_spacing), max_eol_spacing,
                                                       std::max(max_eol_width, max_par_spacing));
            } else {
              DRCLOG.error(Loc::current(), "The orientation is error!");
            }
            std::vector<std::pair<BGRectInt, std::pair<int32_t, int32_t>>> bg_rect_net_pair_list;
            routing_bg_rtree_map[routing_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(max_check_rect)),
                                                          std::back_inserter(bg_rect_net_pair_list));
            for (auto& [bg_env_rect, net_poly_info_idx_pair] : bg_rect_net_pair_list) {
              env_net_poly_info_idx_map[net_poly_info_idx_pair.first].insert(net_poly_info_idx_pair.second);
            }
          }
          // skip_rule_idx_set
          std::set<int32_t> skip_rule_idx_set;
          for (int32_t i = 0; i < static_cast<int32_t>(end_of_line_spacing_rule_list.size()); i++) {
            if (poly_info.edge_length_list[eol_edge_idx] >= end_of_line_spacing_rule_list[i].eol_width) {
              skip_rule_idx_set.insert(i);
            }
            if (end_of_line_spacing_rule_list[i].has_min_length) {
              if (poly_info.edge_length_list[getIdx(eol_edge_idx - 1, poly_info.coord_size)] < end_of_line_spacing_rule_list[i].min_length
                  && poly_info.edge_length_list[getIdx(eol_edge_idx + 1, poly_info.coord_size)] < end_of_line_spacing_rule_list[i].min_length) {
                skip_rule_idx_set.insert(i);
              }
            }
          }
          // par status
          std::vector<bool> left_par_status_list(end_of_line_spacing_rule_list.size(), false);
          std::vector<bool> right_par_status_list(end_of_line_spacing_rule_list.size(), false);
          for (auto& [env_net_idx, env_poly_info_idx_set] : env_net_poly_info_idx_map) {
            for (int32_t env_poly_info_idx : env_poly_info_idx_set) {
              if (env_net_idx == net_idx && env_poly_info_idx == poly_info.poly_info_idx) {
                continue;
              }
              for (Segment<PlanarCoord>& env_edge : net_poly_info_map[env_net_idx][env_poly_info_idx].edge_list) {
                if (DRCUTIL.getDirection(env_edge.get_first(), env_edge.get_second()) == direction) {
                  continue;
                }
                for (size_t i = 0; i < end_of_line_spacing_rule_list.size(); i++) {
                  if (!end_of_line_spacing_rule_list[i].has_par || end_of_line_spacing_rule_list[i].has_same_metal) {
                    continue;
                  }
                  if (DRCUTIL.isOpenOverlap(env_edge.get_first(), env_edge.get_second(), left_par_spacing_rect_list[i])) {
                    left_par_status_list[i] = true;
                  }
                  if (DRCUTIL.isOpenOverlap(env_edge.get_first(), env_edge.get_second(), right_par_spacing_rect_list[i])) {
                    right_par_status_list[i] = true;
                  }
                }
              }
            }
          }
          // check
          PlanarRect curr_edge_rect = DRCUTIL.getRect(pre_coord, curr_coord);
          PlanarRect curr_max_rect = DRCUTIL.getRect(pre_coord, poly_info.coord_list[getIdx(eol_edge_idx + 1, poly_info.coord_size)]);
          for (auto& [env_net_idx, env_poly_info_idx_set] : env_net_poly_info_idx_map) {
            if (net_idx == -1 && env_net_idx == -1) {
              continue;
            }
            for (int32_t env_poly_info_idx : env_poly_info_idx_set) {
              PolyInfo& env_poly_info = net_poly_info_map[env_net_idx][env_poly_info_idx];
              std::vector<Segment<PlanarCoord>>& env_edge_list = env_poly_info.edge_list;
              int32_t env_edge_size = static_cast<int32_t>(env_edge_list.size());
              for (int32_t env_edge_idx = 0; env_edge_idx < env_edge_size; env_edge_idx++) {
                Segment<PlanarCoord>& env_edge = env_edge_list[env_edge_idx];
                if (DRCUTIL.getDirection(env_edge.get_first(), env_edge.get_second()) != direction) {
                  continue;
                }
                if (DRCUTIL.getOrientation(env_edge.get_first(), env_edge.get_second()) == orientation) {
                  continue;
                }
                PlanarRect env_edge_rect = DRCUTIL.getRect(env_edge.get_first(), env_edge.get_second());
                for (int32_t eol_rule_idx = static_cast<int32_t>(end_of_line_spacing_rule_list.size()) - 1; eol_rule_idx >= 0; eol_rule_idx--) {
                  if (DRCUTIL.exist(skip_rule_idx_set, eol_rule_idx)) {
                    continue;
                  }
                  EndOfLineSpacingRule& curr_rule = end_of_line_spacing_rule_list[eol_rule_idx];
                  // has_enclose_cut
                  if (curr_rule.has_enclose_cut) {
                    bool is_cut_require = false;
                    for (PlanarRect& adjacent_cut_shape : adjacent_cut_shape_list) {
                      if (DRCUTIL.isClosedOverlap(curr_max_rect, adjacent_cut_shape)) {
                        if (DRCUTIL.getEuclideanDistance(adjacent_cut_shape, curr_edge_rect) < curr_rule.enclosed_dist
                            && DRCUTIL.getEuclideanDistance(adjacent_cut_shape, env_edge_rect) < curr_rule.cut_to_metal_spacing) {
                          is_cut_require = true;
                          break;
                        }
                      }
                    }
                    if (!is_cut_require) {
                      continue;
                    }
                  }
                  // has_par
                  if (curr_rule.has_par) {
                    if (curr_rule.has_same_metal) {
                      bool pre_left = false;
                      bool pre_right = false;
                      {
                        Segment<PlanarCoord> edge = env_edge_list[getIdx(env_edge_idx - 1, env_edge_size)];
                        PlanarRect rect = DRCUTIL.getRect(edge.get_first(), edge.get_second());
                        if (DRCUTIL.isOpenOverlap(rect, left_par_spacing_rect_list[eol_rule_idx])
                            && DRCUTIL.getParallelLength(rect, left_par_spacing_rect_list[eol_rule_idx]) < curr_rule.par_within) {
                          pre_left = true;
                        } else if (DRCUTIL.isOpenOverlap(rect, right_par_spacing_rect_list[eol_rule_idx])
                                   && DRCUTIL.getParallelLength(rect, right_par_spacing_rect_list[eol_rule_idx]) < curr_rule.par_within) {
                          pre_right = true;
                        }
                      }
                      bool post_left = false;
                      bool post_right = false;
                      {
                        Segment<PlanarCoord> edge = env_edge_list[getIdx(env_edge_idx + 1, env_edge_size)];
                        PlanarRect rect = DRCUTIL.getBoundingBox({edge.get_first(), edge.get_second()});
                        if (DRCUTIL.isOpenOverlap(rect, left_par_spacing_rect_list[eol_rule_idx])
                            && DRCUTIL.getParallelLength(rect, left_par_spacing_rect_list[eol_rule_idx]) < curr_rule.par_within) {
                          post_left = true;
                        } else if (DRCUTIL.isOpenOverlap(rect, right_par_spacing_rect_list[eol_rule_idx])
                                   && DRCUTIL.getParallelLength(rect, right_par_spacing_rect_list[eol_rule_idx]) < curr_rule.par_within) {
                          post_right = true;
                        }
                      }
                      if (curr_rule.has_two_edges) {
                        if (!(pre_left && post_right) && !(pre_right && post_left)) {
                          continue;
                        }
                      } else {
                        if (!(pre_left || pre_right) && !(post_left || post_right)) {
                          continue;
                        }
                      }
                    } else {
                      if (curr_rule.has_two_edges) {
                        if (!(left_par_status_list[eol_rule_idx] && right_par_status_list[eol_rule_idx])) {
                          continue;
                        }
                      } else {
                        if (!(left_par_status_list[eol_rule_idx] || right_par_status_list[eol_rule_idx])) {
                          continue;
                        }
                      }
                    }
                  }
                  // eol & ete
                  PlanarRect spacing_rect = eol_spacing_rect_list[eol_rule_idx];
                  if (curr_rule.has_ete) {
                    if (DRCUTIL.exist(env_poly_info.eol_edge_idx_set, env_edge_idx) && env_poly_info.edge_length_list[env_edge_idx] < curr_rule.eol_width) {
                      spacing_rect = ete_spacing_rect_list[eol_rule_idx];
                    }
                  }
                  if (!DRCUTIL.isOpenOverlap(spacing_rect, env_edge_rect)) {
                    continue;
                  }
                  Violation violation;
                  violation.set_violation_type(ViolationType::kEndOfLineSpacing);
                  violation.set_required_size(direction == Direction::kHorizontal ? spacing_rect.getYSpan() : spacing_rect.getXSpan());
                  violation.set_is_routing(true);
                  violation.set_violation_net_set({net_idx, env_net_idx});
                  violation.set_layer_idx(routing_layer_idx);
                  violation.set_rect(DRCUTIL.getSpacingRect(curr_edge_rect, env_edge_rect));
                  edge_violation_edge_map[poly_info.edge_list[eol_edge_idx]].push_back(std::make_pair(violation, env_edge_list[env_edge_idx]));
                  break;
                }
              }
            }
          }
        }
      }
    }
    std::set<Violation, CmpViolation> invalid_violation_set;
    for (auto& [edge, violation_edge_list] : edge_violation_edge_map) {
      int32_t min_length = INT32_MAX;
      for (auto& [violation, edge] : violation_edge_list) {
        if (DRCUTIL.isHorizontal(edge.get_first(), edge.get_second())) {
          min_length = std::min(min_length, violation.getYSpan());
        } else {
          min_length = std::min(min_length, violation.getXSpan());
        }
      }
      for (auto& [violation, edge] : violation_edge_list) {
        if (DRCUTIL.isHorizontal(edge.get_first(), edge.get_second())) {
          if (min_length != violation.getYSpan()) {
            invalid_violation_set.insert(violation);
          }
        } else {
          if (min_length != violation.getXSpan()) {
            invalid_violation_set.insert(violation);
          }
        }
      }
    }
    for (auto& [segment, violation_edge_list] : edge_violation_edge_map) {
      for (auto& [violation, edge] : violation_edge_list) {
        if (DRCUTIL.exist(invalid_violation_set, violation)) {
          continue;
        }
        rv_box.get_violation_list().push_back(violation);
      }
    }
  }
}

}  // namespace idrc
