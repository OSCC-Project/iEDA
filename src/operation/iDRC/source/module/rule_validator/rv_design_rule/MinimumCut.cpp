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

void RuleValidator::verifyMinimumCut(RVBox& rv_box)
{
  /*
  M1
      PROPERTY LEF58_MINIMUMCUT "
  MINIMUMCUT CUTCLASS VSINGLECUT 2 CUTCLASS VDOUBLECUT 1 WIDTH 0.180000 WITHIN 0.100000 FROMABOVE ;
  MINIMUMCUT CUTCLASS VSINGLECUT 4 CUTCLASS VDOUBLECUT 2 WIDTH 0.440000 FROMABOVE ;
  MINIMUMCUT CUTCLASS VSINGLECUT 2 CUTCLASS VDOUBLECUT 1 WIDTH 0.180000 FROMABOVE LENGTH 0.180000 WITHIN 1.651000 ;
  MINIMUMCUT CUTCLASS VSINGLECUT 2 CUTCLASS VDOUBLECUT 1 WIDTH 1.000000 FROMABOVE LENGTH 1.000000 WITHIN 4.001000 ;
  MINIMUMCUT CUTCLASS VSINGLECUT 2 CUTCLASS VDOUBLECUT 1 WIDTH 1.500000 FROMABOVE LENGTH 5.000000 WITHIN 10.001000 ; " ;

  M2-M7
      PROPERTY LEF58_MINIMUMCUT "
  MINIMUMCUT CUTCLASS VSINGLECUT 2 CUTCLASS VDOUBLECUT 1 WIDTH 0.180000 WITHIN 0.100000  ;
  MINIMUMCUT CUTCLASS VSINGLECUT 4 CUTCLASS VDOUBLECUT 2 WIDTH 0.440000  ;
  MINIMUMCUT CUTCLASS VSINGLECUT 2 CUTCLASS VDOUBLECUT 1 WIDTH 0.180000  LENGTH 0.180000 WITHIN 1.651000 ;
  MINIMUMCUT CUTCLASS VSINGLECUT 2 CUTCLASS VDOUBLECUT 1 WIDTH 1.000000  LENGTH 1.000000 WITHIN 4.001000 ;
  MINIMUMCUT CUTCLASS VSINGLECUT 2 CUTCLASS VDOUBLECUT 1 WIDTH 1.500000  LENGTH 5.000000 WITHIN 10.001000 ; " ; s
  */

  struct MinimumCutRule
  {
    int32_t num_cuts;
    int32_t width;
    /**/ bool has_within;
    /****/ int32_t cut_distance;
    /**/ bool has_from_above;
    /**/ bool has_from_below;
    /**/ bool has_length;
    /****/ int32_t length;
    /****/ int32_t distance;
    ;
  };
  std::map<int32_t, std::vector<MinimumCutRule>> layer_minimum_cut_rule_list;
  for (int i = 0; i < 7; i++) {
    std::vector<MinimumCutRule> minimum_cut_rule_list;
    if (i == 0) {
      minimum_cut_rule_list.push_back({2, 360, true, 200, true, false, false, 0, 0});
      minimum_cut_rule_list.push_back({4, 880, false, 0, true, false, false, 0, 0});
      minimum_cut_rule_list.push_back({2, 360, false, 0, true, false, true, 360, 3302});
      minimum_cut_rule_list.push_back({2, 2000, false, 0, true, false, true, 2000, 8002});
      minimum_cut_rule_list.push_back({2, 3000, false, 0, true, false, true, 10000, 20002});
    } else {
      minimum_cut_rule_list.push_back({2, 360, true, 200, false, false, false, 0, 0});
      minimum_cut_rule_list.push_back({4, 880, false, 0, false, false, false, 0, 0});
      minimum_cut_rule_list.push_back({2, 360, false, 0, false, false, true, 360, 3302});
      minimum_cut_rule_list.push_back({2, 2000, false, 0, false, false, true, 2000, 8002});
      minimum_cut_rule_list.push_back({2, 3000, false, 0, false, false, true, 10000, 20002});
    }
    layer_minimum_cut_rule_list[i] = minimum_cut_rule_list;
  }
#if 1  // 函数定义
  auto checkCutWithWidth = [](const MinimumCutRule& cur_rule, const std::vector<PlanarRect>& cut_rect_list) -> bool {
    int32_t num_cuts = cur_rule.num_cuts;

    bgi::rtree<BGRectInt, bgi::quadratic<16>> temp_cut_bg_rtree;
    for (const PlanarRect& cut_rect : cut_rect_list) {
      temp_cut_bg_rtree.insert(DRCUTIL.convertToBGRectInt(cut_rect));
    }
    bool is_violation = false;
    if (cur_rule.has_within == false) {  // 没有within只需要考虑cut的数量
      int32_t cut_size = cut_rect_list.size();
      for (const PlanarRect& cut_rect : cut_rect_list) {
        if (cut_rect.getXSpan() > 100 || cut_rect.getYSpan() > 100) {
          cut_size++;
        }
      }
      if (cut_size < num_cuts) {
        is_violation = true;
      }
    } else {
      // 有within需要考虑cut之间的距离
      is_violation = true;
      for (const PlanarRect& cut_rect : cut_rect_list) {
        std::vector<BGRectInt> temp_cut_bg_rect_list;
        PlanarRect check_rect = DRCUTIL.getEnlargedRect(cut_rect, cur_rule.cut_distance - 1);
        temp_cut_bg_rtree.query(bgi::intersects(DRCUTIL.convertToBGRectInt(check_rect)), std::back_inserter(temp_cut_bg_rect_list));
        int32_t cut_size = temp_cut_bg_rect_list.size();
        for (BGRectInt bg_cut_rect : temp_cut_bg_rect_list) {
          PlanarRect check_cut_rect = DRCUTIL.convertToPlanarRect(bg_cut_rect);
          if (check_cut_rect.getXSpan() > 100 || check_cut_rect.getYSpan() > 100) {//暂时这样写，需要规范
            cut_size++;
          }
        }
        if (cut_size >= num_cuts) {
          is_violation = false;
          break;
        }
      }
    }
    return is_violation;
  };
#endif
  /////////////////////////////
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  std::vector<CutLayer>& cut_layer_list = DRCDM.getDatabase().get_cut_layer_list();
  std::map<int32_t, std::vector<int32_t>>& routing_to_adjacent_cut_map = DRCDM.getDatabase().get_routing_to_adjacent_cut_map();
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = DRCDM.getDatabase().get_cut_to_adjacent_routing_map();
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

  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> cut_bg_rtree_map;
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
  for (auto& [routing_layer_idx, net_gtl_poly_set_map] : routing_net_gtl_poly_set_map) {
    RoutingLayer& routing_layer = routing_layer_list[routing_layer_idx];
    std::vector<MinimumCutRule>& minimum_cut_rule_list = layer_minimum_cut_rule_list[routing_layer_idx];
    std::vector<int32_t>& cut_layer_idx_list = routing_to_adjacent_cut_map[routing_layer_idx];
    int32_t above_cut_layer_idx = *std::max_element(cut_layer_idx_list.begin(), cut_layer_idx_list.end());
    int32_t below_cut_layer_idx = *std::min_element(cut_layer_idx_list.begin(), cut_layer_idx_list.end());

    for (auto& [net_idx, gtl_poly_set] : net_gtl_poly_set_map) {
      std::vector<GTLPolyInt> gtl_poly_list;
      gtl_poly_set.get_polygons(gtl_poly_list);
      for (GTLPolyInt& gtl_poly : gtl_poly_list) {
        std::vector<GTLRectInt> gtl_rect_list;
        gtl::get_max_rectangles(gtl_rect_list, gtl_poly);
        std::vector<PlanarRect> rect_list;
        for (GTLRectInt& gtl_rect : gtl_rect_list) {
          rect_list.push_back(DRCUTIL.convertToPlanarRect(gtl_rect));
        }
        for (PlanarRect& rect : rect_list) {
          int32_t rect_width = rect.getWidth();
          int32_t rect_legnth = rect.getLength();
          if (minimum_cut_rule_list.empty()) {
            continue;
          }
          for (int32_t rule_idx = minimum_cut_rule_list.size() - 1; rule_idx >= 0; rule_idx--) {
            MinimumCutRule& cur_rule = minimum_cut_rule_list[rule_idx];
            int32_t width = cur_rule.width;
            int32_t num_cuts = cur_rule.num_cuts;
            if (rect_width < width) {
              continue;
            }
            std::vector<int32_t> check_cut_layer_idx_list;
            if (cur_rule.has_from_above) {
              check_cut_layer_idx_list.push_back(above_cut_layer_idx);
            } else if (cur_rule.has_from_below) {
              check_cut_layer_idx_list.push_back(below_cut_layer_idx);
            } else {
              check_cut_layer_idx_list = cut_layer_idx_list;
            }
            // length条件检查的是与之相邻的矩形
            if (cur_rule.has_length) {
              if (rect_legnth <= cur_rule.length) {
                continue;
              }
              GTLPolySetInt env_gtl_poly_set = gtl_poly - DRCUTIL.convertToGTLRectInt(rect);
              std::vector<GTLPolyInt> env_gtl_poly_list;
              env_gtl_poly_set.get_polygons(env_gtl_poly_list);
              for (GTLPolyInt env_gtl_poly : env_gtl_poly_list) {
                std::vector<GTLRectInt> env_gtl_rect_list;
                gtl::get_max_rectangles(env_gtl_rect_list, env_gtl_poly);

                for (int32_t cut_layer_idx : check_cut_layer_idx_list) {
                  int32_t below_routing_layer_idx;
                  {
                    std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[cut_layer_idx];
                    below_routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
                  }
                  std::vector<std::pair<BGRectInt, int32_t>> cut_bg_rect_net_pair_list;
                  for (GTLRectInt& env_gtl_rect : env_gtl_rect_list) {
                    cut_bg_rtree_map[cut_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(env_gtl_rect)),
                                                          std::back_inserter(cut_bg_rect_net_pair_list));
                  }
                  std::vector<PlanarRect> cut_rect_list;
                  for (auto& [bg_rect, cut_net_idx] : cut_bg_rect_net_pair_list) {
                    PlanarRect cut_rect = DRCUTIL.convertToPlanarRect(bg_rect);
                    if(DRCUTIL.exist(cut_rect_list, cut_rect)) {
                      continue;  // 已经存在这个cut
                    }
                    cut_rect_list.push_back(cut_rect);
                  }
                  if (cut_rect_list.empty()) {
                    continue;  // 没有cut不可能产生这个违例
                  }
                  bool is_in_distance = false;
                  for (PlanarRect& cut_rect : cut_rect_list) {
                    if (DRCUTIL.getEuclideanDistance(rect, cut_rect) < cur_rule.distance) {
                      is_in_distance = true;
                      break;
                    }
                  }
                  if (!is_in_distance) {
                    continue;  // 没有满足距离的cut
                  }
                  bool is_violation = checkCutWithWidth(cur_rule, cut_rect_list);
                  if (is_violation == false) {
                    continue;
                  }

                  for (PlanarRect& cut_rect : cut_rect_list) {
                    Violation violation;
                    violation.set_violation_type(ViolationType::kMinimumCut);
                    violation.set_is_routing(true);
                    violation.set_violation_net_set({net_idx});
                    violation.set_layer_idx(below_routing_layer_idx);
                    violation.set_rect(cut_rect);
                    violation.set_required_size(4000);
                    rv_box.get_violation_list().push_back(violation);
                  }
                }
              }
            } else {
              for (int32_t cut_layer_idx : check_cut_layer_idx_list) {
                int32_t below_routing_layer_idx;
                {
                  std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[cut_layer_idx];
                  below_routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
                }
                std::vector<std::pair<BGRectInt, int32_t>> cut_bg_rect_net_pair_list;
                cut_bg_rtree_map[cut_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(rect)), std::back_inserter(cut_bg_rect_net_pair_list));

                std::vector<PlanarRect> cut_rect_list;
                for (auto& [bg_rect, cut_net_idx] : cut_bg_rect_net_pair_list) {
                  PlanarRect cut_rect = DRCUTIL.convertToPlanarRect(bg_rect);
                  if (!DRCUTIL.isClosedOverlap(cut_rect, rect)) {
                    continue;
                  }
                  cut_rect_list.push_back(cut_rect);
                }

                bool is_violation = checkCutWithWidth(cur_rule, cut_rect_list);
                if (is_violation == false) {
                  continue;
                }

                for (PlanarRect& cut_rect : cut_rect_list) {
                  Violation violation;
                  violation.set_violation_type(ViolationType::kMinimumCut);
                  violation.set_is_routing(true);
                  violation.set_violation_net_set({net_idx});
                  violation.set_layer_idx(below_routing_layer_idx);
                  violation.set_rect(cut_rect);
                  violation.set_required_size(4000);
                  rv_box.get_violation_list().push_back(violation);
                }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace idrc
