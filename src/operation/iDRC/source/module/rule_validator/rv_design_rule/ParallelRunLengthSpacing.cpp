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
  auto addRectToRtree
      = [](std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>>& _query_tree, GTLRectInt rect, int32_t layer_idx, int32_t net_idx) {
          BGRectInt rtree_rect(BGPointInt(xl(rect), yl(rect)), BGPointInt(xh(rect), yh(rect)));
          _query_tree[layer_idx].insert(std::make_pair(rtree_rect, net_idx));
        };
  auto queryNetIdbyRtree = [](std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>>& _query_tree, int32_t layer_idx, int32_t llx,
                              int32_t lly, int32_t urx, int32_t ury) {
    std::set<int32_t> net_ids;
    std::vector<std::pair<BGRectInt, int32_t>> result;
    BGRectInt rect(BGPointInt(llx, lly), BGPointInt(urx, ury));
    _query_tree[layer_idx].query(bgi::intersects(rect), std::back_inserter(result));
    for (auto& pair : result) {
      net_ids.insert(pair.second);
    }
    return net_ids;
  };
  auto queryRectbyRtree = [](std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>>& _query_tree, int32_t layer_idx, int32_t llx,
                             int32_t lly, int32_t urx, int32_t ury) {
    std::set<int32_t> net_ids;
    std::vector<std::pair<BGRectInt, int32_t>> result;
    BGRectInt rect(BGPointInt(llx, lly), BGPointInt(urx, ury));
    _query_tree[layer_idx].query(bgi::intersects(rect), std::back_inserter(result));
    return result;
  };
  auto get_width_idx = [](GTLRectInt& rect, std::vector<int32_t>& width_list) {
    int32_t width = std::min(gtl::delta(rect, gtl::HORIZONTAL), gtl::delta(rect, gtl::VERTICAL));
    int32_t width_idx = width_list.size() - 1;
    for (int32_t i = width_list.size() - 1; i >= 0; i--) {
      if (width >= width_list[i]) {
        width_idx = i;
        break;
      }
    }
    return width_idx;
  };
  auto get_prl_idx = [](int32_t& parallel_length, std::vector<int32_t>& parallel_length_list) {
    int32_t prl_idx = parallel_length_list.size() - 1;
    for (int32_t i = parallel_length_list.size() - 1; i >= 0; i--) {
      if (parallel_length >= parallel_length_list[i]) {
        prl_idx = i;
        break;
      }
    }
    return prl_idx;
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
  // std::vector<Violation>& violation_list = rv_box.get_violation_list();
  std::vector<Violation>& final_violation_list = rv_box.get_violation_list();
  // 使用R树查询检测
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> layer_env_query_tree;
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> layer_res_query_tree;
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> layer_all_query_tree;

  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> layer_net_gtl_env_poly_set;
  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> layer_net_gtl_res_poly_set;
  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> layer_net_gtl_all_poly_set;

  std::map<int32_t, std::map<int32_t, std::vector<GTLRectInt>>> layer_net_gtl_env_maxrect_list;
  std::map<int32_t, std::map<int32_t, std::vector<GTLRectInt>>> layer_net_gtl_res_maxrect_list;
  std::map<int32_t, std::map<int32_t, std::vector<GTLRectInt>>> layer_net_gtl_all_maxrect_list;

  std::map<int32_t, std::map<int32_t, std::map<std::set<int32_t>, GTLPolySetInt>>> layer_rs_net_gtl_vio_poly_set;
  for (DRCShape* rect : rv_box.get_drc_env_shape_list()) {
    if (rect->get_is_routing() == false) {
      continue;
    }
    int32_t layer_idx = rect->get_layer_idx();
    int32_t net_idx = rect->get_net_idx();
    layer_net_gtl_env_poly_set[layer_idx][net_idx] += DRCUTIL.convertToGTLRectInt(rect->get_rect());
    layer_net_gtl_all_poly_set[layer_idx][net_idx] += DRCUTIL.convertToGTLRectInt(rect->get_rect());
  }
  for (DRCShape* rect : rv_box.get_drc_result_shape_list()) {
    if (rect->get_is_routing() == false) {
      continue;
    }
    int32_t layer_idx = rect->get_layer_idx();
    int32_t net_idx = rect->get_net_idx();
    layer_net_gtl_res_poly_set[layer_idx][net_idx] += DRCUTIL.convertToGTLRectInt(rect->get_rect());
    layer_net_gtl_all_poly_set[layer_idx][net_idx] += DRCUTIL.convertToGTLRectInt(rect->get_rect());
  }

  // 用max rect作为被查找的
  for (auto& [routing_layer_idx, net_env_gtl_poly_set] : layer_net_gtl_env_poly_set) {
    for (auto& [net_idx, env_gtl_poly_set] : net_env_gtl_poly_set) {
      std::vector<GTLRectInt> rect_list;
      gtl::get_max_rectangles(rect_list, env_gtl_poly_set);
      for (GTLRectInt& rect : rect_list) {
        layer_net_gtl_env_maxrect_list[routing_layer_idx][net_idx].push_back(rect);
        addRectToRtree(layer_env_query_tree, rect, routing_layer_idx, net_idx);
      }
    }
  }

  // 用max rect作为被查找的
  for (auto& [routing_layer_idx, net_res_gtl_poly_set] : layer_net_gtl_res_poly_set) {
    for (auto& [net_idx, res_gtl_poly_set] : net_res_gtl_poly_set) {
      std::vector<GTLRectInt> rect_list;
      gtl::get_max_rectangles(rect_list, res_gtl_poly_set);
      for (GTLRectInt& rect : rect_list) {
        layer_net_gtl_res_maxrect_list[routing_layer_idx][net_idx].push_back(rect);
        addRectToRtree(layer_res_query_tree, rect, routing_layer_idx, net_idx);
      }
    }
  }

  // 用max rect作为被查找的
  for (auto& [routing_layer_idx, net_all_gtl_poly_set] : layer_net_gtl_all_poly_set) {
    for (auto& [net_idx, res_all_poly_set] : net_all_gtl_poly_set) {
      std::vector<GTLRectInt> rect_list;
      gtl::get_max_rectangles(rect_list, res_all_poly_set);
      for (GTLRectInt& rect : rect_list) {
        layer_net_gtl_all_maxrect_list[routing_layer_idx][net_idx].push_back(rect);
        addRectToRtree(layer_all_query_tree, rect, routing_layer_idx, net_idx);
      }
    }
  }
  // result & result检测
  for (auto& [routing_layer_idx, net_gtl_res_maxrect_list] : layer_net_gtl_all_maxrect_list) {
    SpacingTable& spacing_table = routing_layer_list[routing_layer_idx].get_prl_spacing_table();
    std::vector<int32_t>& parallel_length_list = spacing_table.get_parallel_length_list();
    GridMap<int32_t>& width_parallel_length_map = spacing_table.get_width_parallel_length_map();
    std::vector<int32_t>& width_list = spacing_table.get_width_list();
    int32_t max_spacing = width_parallel_length_map.back().back();
    int32_t min_width = routing_layer_list[routing_layer_idx].get_min_width();

    for (auto& [net_idx, gtl_res_maxrect_list] : net_gtl_res_maxrect_list) {
      for (GTLRectInt& gtl_res_maxrect : gtl_res_maxrect_list) {
        // 按照当前宽度的maxspcing寻找周围的矩形,直接用最大宽度
        int32_t width_idx = get_width_idx(gtl_res_maxrect, width_list);
        // int32_t max_spacing = width_parallel_length_map[width_idx].back();

        GTLRectInt bloat_res_gtl_maxrect = gtl_res_maxrect;
        gtl::bloat(bloat_res_gtl_maxrect, max_spacing);  // bloat会改变原来的值
        auto check_prl_func = [&](bool is_env) {
          // std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>>& check_rtree = is_env ? layer_env_query_tree :
          // layer_res_query_tree;
          std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>>& check_rtree = layer_all_query_tree;
          // 找到有交集的max rect
          std::vector<std::pair<BGRectInt, int32_t>> overlap_rect_result = queryRectbyRtree(
              check_rtree, routing_layer_idx, gtl::xl(bloat_res_gtl_maxrect) + 1, gtl::yl(bloat_res_gtl_maxrect) + 1, gtl::xh(bloat_res_gtl_maxrect) - 1,
              gtl::yh(bloat_res_gtl_maxrect) - 1);  // 从原图形中查找对应的rect,不要贴边的

          for (auto& [bgrect, q_net_idx] : overlap_rect_result) {
            if (net_idx == -1 && q_net_idx == -1) {
              continue;  // -1忽略
            }

            PlanarRect spacing_rect = DRCUTIL.convertToPlanarRect(bgrect);
            GTLRectInt gtl_spacing_rect = DRCUTIL.convertToGTLRectInt(bgrect);

            PlanarRect res_maxrect = DRCUTIL.convertToPlanarRect(gtl_res_maxrect);
            if (res_maxrect.getWidth() < min_width || spacing_rect.getWidth() < min_width) {
              continue;  // 过滤掉小矩形
            }
            // 先check两个rect是否重叠
            if (DRCUTIL.isClosedOverlap(spacing_rect, res_maxrect)) {
              continue;  // 同一个polygon的忽略
            }
            // 然后check两个rect的spacing是否满足条件
            {
              PlanarRect& rect_a = spacing_rect;
              PlanarRect& rect_b = res_maxrect;
              int32_t width_idx_a = get_width_idx(gtl_spacing_rect, width_list);
              int32_t width_idx_b = get_width_idx(gtl_res_maxrect, width_list);
              int32_t width_idx = std::max(width_idx_a, width_idx_b);

              // 检测水平方向距离和竖直方向距离
              int32_t h_spacing = 0;
              int32_t v_spacing = 0;
              h_spacing = std::max(0, std::max(rect_a.get_ll_x() - rect_b.get_ur_x(), rect_b.get_ll_x() - rect_a.get_ur_x()));
              v_spacing = std::max(0, std::max(rect_a.get_ll_y() - rect_b.get_ur_y(), rect_b.get_ll_y() - rect_a.get_ur_y()));

              int32_t parallel_length = 0;
              if (h_spacing > 0 && v_spacing > 0) {
                parallel_length = 0;
              } else {
                // 计算水平投影和竖直投影的重叠部分
                if (v_spacing == 0) {
                  parallel_length = std::min(rect_a.get_ur_y(), rect_b.get_ur_y()) - std::max(rect_a.get_ll_y(), rect_b.get_ll_y());
                } else {  // h_spacing == 0
                  parallel_length = std::min(rect_a.get_ur_x(), rect_b.get_ur_x()) - std::max(rect_a.get_ll_x(), rect_b.get_ll_x());
                }
              }
              int32_t prl_idx = get_prl_idx(parallel_length, parallel_length_list);
              int32_t need_spacing = width_parallel_length_map[width_idx][prl_idx];
              int32_t real_spacing = gtl::euclidean_distance(gtl_spacing_rect, gtl_res_maxrect);
              if (real_spacing >= need_spacing) {
                continue;  // 满足距离条件
              }
              PlanarRect violation_rect;
              if (h_spacing > 0 && v_spacing > 0) {  // 找到距最近的两个点
                int32_t llx;
                int32_t lly;
                int32_t urx;
                int32_t ury;
                // 判断b在a的哪个方位
                if (rect_a.get_ur_x() < rect_b.get_ll_x()) {
                  llx = rect_a.get_ur_x();
                  urx = rect_b.get_ll_x();
                } else {
                  llx = rect_b.get_ur_x();
                  urx = rect_a.get_ll_x();
                }
                if (rect_a.get_ur_y() < rect_b.get_ll_y()) {
                  lly = rect_a.get_ur_y();
                  ury = rect_b.get_ll_y();
                } else {
                  lly = rect_b.get_ur_y();
                  ury = rect_a.get_ll_y();
                }
                violation_rect = PlanarRect(llx, lly, urx, ury);
              } else {
                if (v_spacing == 0) {
                  violation_rect = DRCUTIL.getOverlap(DRCUTIL.getEnlargedRect(rect_a, h_spacing, 0, h_spacing, 0),
                                                      DRCUTIL.getEnlargedRect(rect_b, h_spacing, 0, h_spacing, 0));
                } else {  // h_spaing = 0
                  violation_rect = DRCUTIL.getOverlap(DRCUTIL.getEnlargedRect(rect_a, 0, v_spacing, 0, v_spacing),
                                                      DRCUTIL.getEnlargedRect(rect_b, 0, v_spacing, 0, v_spacing));
                }
              }
              // 从viorect中得到重叠的rect
              std::vector<std::pair<BGRectInt, int32_t>> vio_inter_rect_result;
              GTLPolySetInt vio_inter_poly_set;
              vio_inter_rect_result = queryRectbyRtree(layer_all_query_tree, routing_layer_idx, violation_rect.get_ll_x(), violation_rect.get_ll_y(),
                                                       violation_rect.get_ur_x(), violation_rect.get_ur_y());
              for (auto& [bgrect, _] : vio_inter_rect_result) {
                vio_inter_poly_set += DRCUTIL.convertToGTLRectInt(bgrect);
              }
              // 判断一跟线的情况
              if (violation_rect.getArea() == 0) {
                if (is_segment_inside_polygon(GTLPointInt(violation_rect.get_ll_x(), violation_rect.get_ll_y()),
                                              GTLPointInt(violation_rect.get_ur_x(), violation_rect.get_ur_y()), vio_inter_poly_set)) {
                  continue;
                }
              } else {  // 矩形情况
                GTLPolySetInt left_vio_inter_poly_set = DRCUTIL.convertToGTLRectInt(violation_rect) - vio_inter_poly_set;
                if (gtl::area(left_vio_inter_poly_set) == 0) {
                  continue;  // vio完全被覆盖，忽略
                }

                GTLRectInt gtl_violation_rect;
                gtl::extents(gtl_violation_rect, left_vio_inter_poly_set);
                violation_rect = DRCUTIL.convertToPlanarRect(gtl_violation_rect);

                 // 过滤掉由env产生的违例rect
                std::vector<std::pair<BGRectInt, int32_t>> vio_inter_rect_result
                    = queryRectbyRtree(layer_res_query_tree, routing_layer_idx, violation_rect.get_ll_x(), violation_rect.get_ll_y(), violation_rect.get_ur_x(),
                                       violation_rect.get_ur_y());  // 从原图形中查找对应的rect,不要贴边的
                if (vio_inter_rect_result.size() == 0) {
                  continue;
                }
                // 贴两边的情况也要去除
                std::vector<PlanarRect> corner_rect_list;  // 东南西北生成四个宽度为一的矩形
                corner_rect_list.push_back(
                    PlanarRect(violation_rect.get_ur_x(), violation_rect.get_ll_y(), violation_rect.get_ur_x() + 1, violation_rect.get_ur_y()));  // 东
                corner_rect_list.push_back(
                    PlanarRect(violation_rect.get_ll_x(), violation_rect.get_ll_y() - 1, violation_rect.get_ur_x(), violation_rect.get_ll_y()));  // 南
                corner_rect_list.push_back(
                    PlanarRect(violation_rect.get_ll_x() - 1, violation_rect.get_ll_y(), violation_rect.get_ll_x(), violation_rect.get_ur_y()));  // 西
                corner_rect_list.push_back(
                    PlanarRect(violation_rect.get_ll_x(), violation_rect.get_ur_y(), violation_rect.get_ur_x(), violation_rect.get_ur_y() + 1));  // 北

                std::vector<bool> corner_flag_list(4, false);
                for (int32_t i = 0; i < corner_rect_list.size(); i++) {
                  if (gtl::area(DRCUTIL.convertToGTLRectInt(corner_rect_list[i]) & vio_inter_poly_set) == corner_rect_list[i].getArea()) {
                    corner_flag_list[i] = true;
                  }
                }
                // 东南 南西 西北 北东四个中只有一个满足时不行
                int32_t corner_count = 0;
                for (int32_t i = 0; i < corner_flag_list.size(); i++) {
                  if (corner_flag_list[i] && corner_flag_list[(i + 1) % 4]) {
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
              violation.set_violation_net_set({net_idx, q_net_idx});
              violation.set_layer_idx(routing_layer_idx);
              violation.set_rect(violation_rect);
              violation.set_required_size(need_spacing);
              violation_list.push_back(violation);
            }
          }
        };
        check_prl_func(false);  
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
