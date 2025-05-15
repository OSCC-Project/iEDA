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

void RuleValidator::verifyParallelRunLengthSpacing(RVBox& rv_box)
{
#if 1
  auto get_table_idx = [](int32_t width, std::vector<int32_t> width_list) {
    int32_t width_idx = width_list.size() - 1;
    for (int32_t i = width_list.size() - 1; i >= 0; i--) {
      if (width >= width_list[i]) {
        width_idx = i;
        break;
      }
    }
    return width_idx;
  };
  auto is_segment_inside_polygon = [](GTLPointInt point_a, GTLPointInt point_b, GTLPolySetInt& polygon) {
    std::vector<GTLRectInt> rect_list;
    gtl::get_max_rectangles(rect_list, polygon);
    for (GTLRectInt& rect : rect_list) {
      if (gtl::contains(rect, point_a) && gtl::contains(rect, point_b)) {
        return true;
      }
    }
    return false;
  };

#endif
  // 得到基础数据
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  std::vector<Violation> violation_list;
  std::vector<Violation>& final_violation_list = rv_box.get_violation_list();
  // 使用R树查询检测
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> routing_bg_rtree_map;
  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> layer_net_gtl_all_poly_set;
  std::map<int32_t, std::map<int32_t, std::vector<GTLRectInt>>> layer_net_gtl_all_maxrect_list;

  for (DRCShape* rect : rv_box.get_drc_env_shape_list()) {
    if (rect->get_is_routing() == false) {
      continue;
    }
    int32_t layer_idx = rect->get_layer_idx();
    int32_t net_idx = rect->get_net_idx();
    layer_net_gtl_all_poly_set[layer_idx][net_idx] += DRCUTIL.convertToGTLRectInt(rect->get_rect());
  }
  for (DRCShape* rect : rv_box.get_drc_result_shape_list()) {
    if (rect->get_is_routing() == false) {
      continue;
    }
    int32_t layer_idx = rect->get_layer_idx();
    int32_t net_idx = rect->get_net_idx();
    layer_net_gtl_all_poly_set[layer_idx][net_idx] += DRCUTIL.convertToGTLRectInt(rect->get_rect());
  }

  // 用max rect作为被查找的
  for (auto& [routing_layer_idx, net_all_gtl_poly_set] : layer_net_gtl_all_poly_set) {
    for (auto& [net_idx, res_all_poly_set] : net_all_gtl_poly_set) {
      std::vector<GTLRectInt> rect_list;
      gtl::get_max_rectangles(rect_list, res_all_poly_set);
      for (GTLRectInt& rect : rect_list) {
        layer_net_gtl_all_maxrect_list[routing_layer_idx][net_idx].push_back(rect);
        routing_bg_rtree_map[routing_layer_idx].insert(std::make_pair(DRCUTIL.convertToBGRectInt(rect), net_idx));
      }
    }
  }
  // result & result检测
  for (auto& [routing_layer_idx, net_gtl_all_maxrect_list] : layer_net_gtl_all_maxrect_list) {
    SpacingTable& spacing_table = routing_layer_list[routing_layer_idx].get_prl_spacing_table();
    std::vector<int32_t>& parallel_length_list = spacing_table.get_parallel_length_list();
    GridMap<int32_t>& width_parallel_length_map = spacing_table.get_width_parallel_length_map();
    std::vector<int32_t>& width_list = spacing_table.get_width_list();
    int32_t max_spacing = width_parallel_length_map.back().back();
    int32_t min_width = routing_layer_list[routing_layer_idx].get_min_width();

    for (auto& [net_idx, gtl_all_maxrect_list] : net_gtl_all_maxrect_list) {
      for (GTLRectInt& gtl_ori_rect : gtl_all_maxrect_list) {
        if(gtl_ori_rect == GTLRectInt(106750,62350,106850,63250)){
          int32_t debug = 0;
        }
        // 找到有交集的max rect，直接用maxspacing
        std::vector<std::pair<BGRectInt, int32_t>> bg_rect_net_pair_list;
        {
          PlanarRect check_rect = DRCUTIL.getEnlargedRect(DRCUTIL.convertToPlanarRect(gtl_ori_rect), max_spacing);
          routing_bg_rtree_map[routing_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(check_rect)), std::back_inserter(bg_rect_net_pair_list));
        }

        for (auto& [bgrect, env_net_idx] : bg_rect_net_pair_list) {
          if (net_idx == -1 && env_net_idx == -1) {
            continue;  // -1忽略
          }

          PlanarRect env_rect = DRCUTIL.convertToPlanarRect(bgrect);
          PlanarRect ori_rect = DRCUTIL.convertToPlanarRect(gtl_ori_rect);
          if (ori_rect.getWidth() < min_width || env_rect.getWidth() < min_width) {
            continue;  // 过滤掉小矩形
          }
          // 先check两个rect是否重叠
          if (DRCUTIL.isClosedOverlap(env_rect, ori_rect)) {
            continue;
          }

          PlanarRect violation_rect = DRCUTIL.getSpacingRect(env_rect, ori_rect);
          // 然后check两个rect的spacing是否满足条件
          int32_t width_idx = std::max(get_table_idx(env_rect.getWidth(), width_list), get_table_idx(ori_rect.getWidth(), width_list));
          // 检测水平方向距离和竖直方向距离
          int32_t h_spacing = std::max(0, std::max(env_rect.get_ll_x() - ori_rect.get_ur_x(), ori_rect.get_ll_x() - env_rect.get_ur_x()));
          int32_t v_spacing = std::max(0, std::max(env_rect.get_ll_y() - ori_rect.get_ur_y(), ori_rect.get_ll_y() - env_rect.get_ur_y()));

          int32_t parallel_length = 0;
          if (h_spacing > 0 && v_spacing > 0) {
            parallel_length = 0;
          } else {
            // 计算水平投影和竖直投影的重叠部分
            if (v_spacing == 0) {
              parallel_length = violation_rect.getYSpan();
            } else {  // h_spacing == 0
              parallel_length = violation_rect.getXSpan();
            }
          }
          int32_t prl_idx = get_table_idx(parallel_length, parallel_length_list);
          int32_t need_spacing = width_parallel_length_map[width_idx][prl_idx];
          int32_t real_spacing = DRCUTIL.getEuclideanDistance(env_rect, ori_rect);
          if (real_spacing >= need_spacing) {
            continue;  // 满足距离条件
          }
          
          // 从viorect中得到重叠的rect
          std::vector<std::pair<BGRectInt, int32_t>> vio_overlap_rect_pair_list;
          routing_bg_rtree_map[routing_layer_idx].query(bgi::intersects(DRCUTIL.convertToBGRectInt(violation_rect)),
                                                        std::back_inserter(vio_overlap_rect_pair_list));

          GTLPolySetInt vio_overlap_poly_set;
          for (auto& [bgrect, _] : vio_overlap_rect_pair_list) {
            vio_overlap_poly_set += DRCUTIL.convertToGTLRectInt(bgrect);
          }
          // 判断一跟线的情况
          if (violation_rect.getArea() == 0) {
            if (is_segment_inside_polygon(GTLPointInt(violation_rect.get_ll_x(), violation_rect.get_ll_y()),
                                          GTLPointInt(violation_rect.get_ur_x(), violation_rect.get_ur_y()), vio_overlap_poly_set)) {
              continue;
            }
          } else {  // 矩形情况
            GTLPolySetInt left_vio_inter_poly_set = DRCUTIL.convertToGTLRectInt(violation_rect) - vio_overlap_poly_set;

            GTLRectInt gtl_violation_rect;
            gtl::extents(gtl_violation_rect, left_vio_inter_poly_set);  // 违例矩形减去重叠polygon，得到的图形的外接矩形作为违例矩形
            violation_rect = DRCUTIL.convertToPlanarRect(gtl_violation_rect);

            // 贴两边的情况也要去除,当且仅当rect连续的两条边在viorect相关的polygonset上
            std::vector<bool> segment_contain_list(4, false);
            segment_contain_list[0] = is_segment_inside_polygon(GTLPointInt(violation_rect.get_ur_x(), violation_rect.get_ll_y()),
                                                                GTLPointInt(violation_rect.get_ur_x(), violation_rect.get_ur_y()), vio_overlap_poly_set);
            segment_contain_list[1] = is_segment_inside_polygon(GTLPointInt(violation_rect.get_ll_x(), violation_rect.get_ll_y()),
                                                                GTLPointInt(violation_rect.get_ur_x(), violation_rect.get_ll_y()), vio_overlap_poly_set);
            segment_contain_list[2] = is_segment_inside_polygon(GTLPointInt(violation_rect.get_ll_x(), violation_rect.get_ll_y()),
                                                                GTLPointInt(violation_rect.get_ll_x(), violation_rect.get_ur_y()), vio_overlap_poly_set);
            segment_contain_list[3] = is_segment_inside_polygon(GTLPointInt(violation_rect.get_ll_x(), violation_rect.get_ur_y()),
                                                                GTLPointInt(violation_rect.get_ur_x(), violation_rect.get_ur_y()), vio_overlap_poly_set);
            // 东南 南西 西北 北东四个中只有一个满足时不行
            int32_t corner_count = 0;
            for (int32_t i = 0; i < segment_contain_list.size(); i++) {
              if (segment_contain_list[i] && segment_contain_list[(i + 1) % 4]) {
                corner_count++;
              }
            }
            if (corner_count == 1) {
              continue;
            }
          }
          Violation violation;
          violation.set_violation_type(ViolationType::kParallelRunLengthSpacing);
          violation.set_is_routing(true);
          violation.set_violation_net_set({net_idx, env_net_idx});
          violation.set_layer_idx(routing_layer_idx);
          violation.set_rect(violation_rect);
          violation.set_required_size(need_spacing);
          violation_list.push_back(violation);
        }
      }
    }
  }
  // 检测被包裹的
  for (Violation& violation : violation_list) {
    bool is_inside = false;
    for (Violation& check_violation : violation_list) {
      if (check_violation.get_violation_type() == violation.get_violation_type() && check_violation.get_layer_idx() == violation.get_layer_idx()
          && check_violation.get_required_size() == violation.get_required_size()
          && check_violation.get_violation_net_set() == violation.get_violation_net_set() && check_violation.get_layer_idx() == violation.get_layer_idx()) {
        if (DRCUTIL.isInside(check_violation.get_rect(), violation.get_rect()) && check_violation.get_rect() != violation.get_rect()) {
          is_inside = true;
          break;
        }
      }
    }
    if (is_inside == false) {
      final_violation_list.push_back(violation);
    }
  }
}
}  // namespace idrc
