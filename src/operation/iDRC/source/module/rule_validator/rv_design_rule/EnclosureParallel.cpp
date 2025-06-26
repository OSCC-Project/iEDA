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

void RuleValidator::verifyEnclosureParallel(RVBox& rv_box)
{
  /*
  对应lef条目：
  PROPERTY LEF58_EOLENCLOSURE "
      EOLENCLOSURE 0.070 CUTCLASS VSINGLECUT ABOVE 0.030 PARALLELEDGE 0.115 EXTENSION 0.070 0.025 MINLENGTH 0.050 ; " ;
  PROPERTY LEF58_EOLENCLOSURE "
      EOLENCLOSURE 0.070 CUTCLASS VDOUBLECUT ABOVE 0.030 PARALLELEDGE 0.115 EXTENSION 0.070 0.025 MINLENGTH 0.050 ; " ;
  目前只需要用第一条SingleCut即可
  */
  struct EolEnclosureRule  // 报的是EnclosureParallel
  {
    int32_t eol_width;
    /**/ bool has_above;
    /**/ bool has_below;
    int32_t overhang;
    /**/ bool has_paralleledge;
    /**/ int32_t par_space;
    /**/ int32_t backward_ext;
    /**/ int32_t forward_ext;
    /****/ bool has_minlength;
    /****/ int32_t min_length;
  };
  std::map<int32_t, EolEnclosureRule> layer_eol_enclosure_rule_map;
  for (int32_t i = 1; i <= 6; i++) {
    layer_eol_enclosure_rule_map[i] = EolEnclosureRule(140, true, false, 60, true, 230, 141, 51, true, 100);
  }
#if 1
  auto is_segment_inside = [](const Segment<PlanarCoord>& master, const Segment<PlanarCoord>& salve) {
    if (DRCUTIL.getDirection(master.get_first(), master.get_second()) != DRCUTIL.getDirection(salve.get_first(), salve.get_second())) {
      return false;
    }
    if (DRCUTIL.getDirection(master.get_first(), master.get_second()) == Direction::kHorizontal) {
      int32_t master_small_x = std::min(master.get_first().get_x(), master.get_second().get_x());
      int32_t master_large_x = std::max(master.get_first().get_x(), master.get_second().get_x());
      int32_t salve_small_x = std::min(salve.get_first().get_x(), salve.get_second().get_x());
      int32_t salve_large_x = std::max(salve.get_first().get_x(), salve.get_second().get_x());
      return (master_small_x <= salve_small_x && salve_large_x <= master_large_x) && master.get_first().get_y() == salve.get_first().get_y();
    } else {
      int32_t master_small_y = std::min(master.get_first().get_y(), master.get_second().get_y());
      int32_t master_large_y = std::max(master.get_first().get_y(), master.get_second().get_y());
      int32_t salve_small_y = std::min(salve.get_first().get_y(), salve.get_second().get_y());
      int32_t salve_large_y = std::max(salve.get_first().get_y(), salve.get_second().get_y());
      return (master_small_y <= salve_small_y && salve_large_y <= master_large_y) && master.get_first().get_x() == salve.get_first().get_x();
    }
  };
#endif
#if 1  // 数据结构定义
  struct PolyInfo//阉割版polygoninfo
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
          routing_net_poly_info_map[routing_layer_idx][net_idx].emplace_back(coord_size, edge_list, edge_length_list, eol_edge_idx_set,gtl_hole_poly);
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
  {
    // for (DRCShape* drc_shape : rv_box.get_drc_env_shape_list()) {
    //   if (!drc_shape->get_is_routing()) {
    //     cut_rect_map[drc_shape->get_layer_idx()].push_back(drc_shape->get_rect());
    //   }
    // }
    for (DRCShape* drc_shape : rv_box.get_drc_result_shape_list()) {
      if (!drc_shape->get_is_routing()) {
        cut_rect_map[drc_shape->get_layer_idx()].push_back(drc_shape->get_rect());
      }
    }
  }

  for (auto& [cut_layer_idx, cut_rect_list] : cut_rect_map) {
    std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[cut_layer_idx];
    if (routing_layer_idx_list.size() < 2) {
      continue;
    }
    int32_t above_routing_layer_idx = *std::max_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
    int32_t below_routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
    EolEnclosureRule curr_rule = layer_eol_enclosure_rule_map[cut_layer_idx];
    for (PlanarRect& cut_rect : cut_rect_list) {
      std::set<Segment<PlanarCoord>, CmpSegmentXASC> processed_segment_set;  // 用来存储已经处理过的segment
      for (int32_t routing_layer_idx : routing_layer_idx_list) {
        if (curr_rule.has_above && (routing_layer_idx != above_routing_layer_idx)) {
          continue;
        }
        if (curr_rule.has_below && (routing_layer_idx != below_routing_layer_idx)) {
          continue;
        }

        std::vector<std::pair<BGRectInt, std::pair<int32_t, int32_t>>> bg_rect_net_pair_list;
        {
          routing_bg_rtree_map[routing_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(cut_rect)), std::back_inserter(bg_rect_net_pair_list));
        }
        std::map<Orientation, int32_t> orient_overhang_map;
        for (auto& [bg_rect, net_idx_eol_pair] : bg_rect_net_pair_list) {
          int32_t net_idx = net_idx_eol_pair.first;
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
        for (auto& [bg_rect, net_idx_eol_pair] : bg_rect_net_pair_list) {
          int32_t net_idx = net_idx_eol_pair.first;
          int32_t polygon_info_idx = net_idx_eol_pair.second;
          PlanarRect routing_rect = DRCUTIL.convertToPlanarRect(bg_rect);
          if (!DRCUTIL.isClosedOverlap(routing_rect, cut_rect)) {
            continue;
          }
          PolyInfo& poly_info = routing_net_poly_info_map[routing_layer_idx][net_idx][polygon_info_idx];

          for (auto& orient : {Orientation::kNorth, Orientation::kSouth, Orientation::kEast, Orientation::kWest}) {
            int32_t edge_idx = -1;
            for(int32_t eol_idx:poly_info.eol_edge_idx_set){
               if(is_segment_inside(poly_info.edge_list[eol_idx],routing_rect.getOrientEdge(orient))){
                 edge_idx = eol_idx;
                 break;
               }
            }
            if (edge_idx == -1) {
              continue;
            }
            Segment<PlanarCoord> cur_segment = poly_info.edge_list[edge_idx];
            Segment<PlanarCoord> pre_segment = poly_info.edge_list[getIdx(edge_idx - 1, poly_info.coord_size)];
            Segment<PlanarCoord> post_segment = poly_info.edge_list[getIdx(edge_idx + 1, poly_info.coord_size)];
            if (DRCUTIL.getManhattanDistance(cur_segment.get_first(), cur_segment.get_second()) >= curr_rule.eol_width) {
              continue;
            }
            if (orient_overhang_map[orient] >= curr_rule.overhang) {
              continue;
            }
            if (DRCUTIL.exist(processed_segment_set, cur_segment)) {
              continue;  // 如果已经处理过了，就跳过
            }
            int32_t pre_segment_length = DRCUTIL.getManhattanDistance(pre_segment.get_first(), pre_segment.get_second());
            int32_t post_segment_length = DRCUTIL.getManhattanDistance(post_segment.get_first(), post_segment.get_second());
            if (curr_rule.has_minlength && (pre_segment_length < curr_rule.min_length && post_segment_length < curr_rule.min_length)) {
              continue;
            }
            PlanarRect left_par_rect;
            PlanarRect right_par_rect;
            int32_t par_space = curr_rule.par_space - DRCUTIL.getManhattanDistance(cur_segment.get_first(), cur_segment.get_second());
            if (orient == Orientation::kEast) {
              left_par_rect = DRCUTIL.getEnlargedRect(cur_segment.get_first(), curr_rule.backward_ext, par_space, curr_rule.forward_ext, 0);
              right_par_rect = DRCUTIL.getEnlargedRect(cur_segment.get_second(), curr_rule.backward_ext, 0, curr_rule.forward_ext, par_space);
            } else if (orient == Orientation::kWest) {
              left_par_rect = DRCUTIL.getEnlargedRect(cur_segment.get_first(), curr_rule.forward_ext, 0, curr_rule.backward_ext, par_space);
              right_par_rect = DRCUTIL.getEnlargedRect(cur_segment.get_second(), curr_rule.forward_ext, par_space, curr_rule.backward_ext, 0);
            } else if (orient == Orientation::kSouth) {
              left_par_rect = DRCUTIL.getEnlargedRect(cur_segment.get_first(), par_space, curr_rule.forward_ext, 0, curr_rule.backward_ext);
              right_par_rect = DRCUTIL.getEnlargedRect(cur_segment.get_second(), 0, curr_rule.forward_ext, par_space, curr_rule.backward_ext);
            } else if (orient == Orientation::kNorth) {
              left_par_rect = DRCUTIL.getEnlargedRect(cur_segment.get_first(), 0, curr_rule.backward_ext, par_space, curr_rule.forward_ext);
              right_par_rect = DRCUTIL.getEnlargedRect(cur_segment.get_second(), par_space, curr_rule.backward_ext, 0, curr_rule.forward_ext);
            } else {
              DRCLOG.error(Loc::current(), "The orientation is error!");
            }
            std::set<int32_t> left_par_net_idx_set;
            std::set<int32_t> right_par_net_idx_set;
            {
              PlanarRect check_rect;
              if (pre_segment_length >= curr_rule.min_length && post_segment_length >= curr_rule.min_length) {
                check_rect = DRCUTIL.getBoundingBox({left_par_rect.get_ll(), left_par_rect.get_ur(), right_par_rect.get_ll(), right_par_rect.get_ur()});
              } else if (pre_segment_length >= curr_rule.min_length) {
                check_rect = left_par_rect;
              } else if (post_segment_length >= curr_rule.min_length) {
                check_rect = right_par_rect;
              }
              std::vector<std::pair<BGRectInt, std::pair<int32_t, int32_t>>> env_bg_rect_net_pair_list;
              routing_bg_rtree_map[routing_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(check_rect)),
                                                            std::back_inserter(env_bg_rect_net_pair_list));
              for (auto& [bg_env_rect, env_net_idx_pair] : env_bg_rect_net_pair_list) {
                int32_t env_net_idx = env_net_idx_pair.first;
                PlanarRect env_routing_rect = DRCUTIL.convertToPlanarRect(bg_env_rect);
                if (DRCUTIL.isClosedOverlap(routing_rect, env_routing_rect)) {
                  continue;
                }
                if (DRCUTIL.isOpenOverlap(env_routing_rect, left_par_rect)) {
                  left_par_net_idx_set.insert(env_net_idx);
                }
                if (DRCUTIL.isOpenOverlap(env_routing_rect, right_par_rect)) {
                  right_par_net_idx_set.insert(env_net_idx);
                }
              }
            }
            std::set<std::set<int32_t>> violation_net_set_set;
            for (int32_t left_par_net_idx : left_par_net_idx_set) {
              violation_net_set_set.insert({left_par_net_idx, net_idx});
            }
            for (int32_t right_par_net_idx : right_par_net_idx_set) {
              violation_net_set_set.insert({right_par_net_idx, net_idx});
            }
            for (const std::set<int32_t>& violation_net_set : violation_net_set_set) {
              Violation violation;
              violation.set_violation_type(ViolationType::kEnclosureParallel);
              violation.set_is_routing(true);
              violation.set_violation_net_set(violation_net_set);
              violation.set_layer_idx(below_routing_layer_idx);
              violation.set_rect(cut_rect);
              violation.set_required_size(curr_rule.overhang);
              rv_box.get_violation_list().push_back(violation);
            }
            processed_segment_set.insert(cur_segment);
          }
        }
      }
    }
  }
}
}  // namespace idrc
