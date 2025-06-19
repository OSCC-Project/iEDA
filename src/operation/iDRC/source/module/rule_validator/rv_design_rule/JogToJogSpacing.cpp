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

void RuleValidator::verifyJogToJogSpacing(RVBox& rv_box)
{
}

#if 0
// 由于与prl有重叠,暂时关闭

/*
思路：
1. 取max_rectangles,获得宽线；
2. 根据宽度，获取宽线对应的rule；
3.宽线膨胀，找到满足 wthin和prl的另外一条线；
4.获取两条线之间的 spacing rect；
5. 减去原来两条线所属线网 在 spacing rect中 凹陷、突出 部分，得到 check_region；
6. 按宽线的宽度方向，切割check_region，得到多个待检查的区域
7. 检查每个待检查的区域，是否满足jog间距的要求；
*/

void RuleValidator::verifyJogToJogSpacing(RVBox& rv_box)
{
/// 规则构造
#if 1
  struct JogToJogSpacingRule
  {
    // 全局参数
    int32_t jogToJogSpacing;  // jog之间的间距
    int32_t jogWidth;         // jog宽度
    int32_t shortJogSpacing;  // 短jog间距
    int32_t rows;             // 行数

    // 每个宽度规则的结构
    struct WidthRule
    {
      int32_t width;           // 宽度
      int32_t parallelLength;  // 平行长度
      int32_t within;          // within距离
      int32_t longJogSpacing;  // 长jog间距

      WidthRule(int32_t w, int32_t pLength, int32_t w_in, int32_t longJog) : width(w), parallelLength(pLength), within(w_in), longJogSpacing(longJog) {}
    };

    std::vector<WidthRule> widthRules;

    JogToJogSpacingRule(int32_t jts, int32_t jw, int32_t sjs, int32_t rows) : jogToJogSpacing(jts), jogWidth(jw), shortJogSpacing(sjs), rows(rows) {}
    void addWidthRule(int32_t width, int32_t parLength, int32_t within, int32_t longJogSpacing)
    {
      widthRules.emplace_back(width, parLength, within, longJogSpacing);
    }
  };

  JogToJogSpacingRule rule(600, 440, 120, 4);

  // rule.addWidthRule(500, 600, 580, 160);
  rule.addWidthRule(500, 600, 380, 200);
  rule.addWidthRule(940, 1000, 640, 260);
  rule.addWidthRule(1260, 1400, 680, 300);
  rule.addWidthRule(3000, 3000, 1000, 600);
#endif
// 工具类函数 Rtree
#if 1
  auto addRectToRtree
      = [](std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>>& _query_tree, GTLRectInt rect, int32_t layer_idx, int32_t net_idx) {
          BGRectInt rtree_rect(BGPointInt(xl(rect), yl(rect)), BGPointInt(xh(rect), yh(rect)));
          _query_tree[layer_idx].insert(std::make_pair(rtree_rect, net_idx));
        };
  auto queryNetIdbyRtreeWithIntersects = [](std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>>& _query_tree, int32_t layer_idx,
                                            int32_t llx, int32_t lly, int32_t urx, int32_t ury) {
    std::set<int32_t> net_ids;
    std::vector<std::pair<BGRectInt, int32_t>> result;
    BGRectInt rect(BGPointInt(llx, lly), BGPointInt(urx, ury));
    _query_tree[layer_idx].query(bgi::intersects(rect), std::back_inserter(result));
    for (auto& pair : result) {
      net_ids.insert(pair.second);
    }
    return net_ids;
  };
  auto queryRectbyRtreeWithIntersects = [](std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>>& _query_tree, int32_t layer_idx,
                                           int32_t llx, int32_t lly, int32_t urx, int32_t ury) {
    std::set<int32_t> net_ids;
    std::vector<std::pair<BGRectInt, int32_t>> result;
    BGRectInt rect(BGPointInt(llx, lly), BGPointInt(urx, ury));
    _query_tree[layer_idx].query(bgi::intersects(rect), std::back_inserter(result));
    return result;
  };
  auto queryNetIdbyRtreeWithWithin = [](std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>>& _query_tree, int32_t layer_idx,
                                        int32_t llx, int32_t lly, int32_t urx, int32_t ury) {
    std::set<int32_t> net_ids;
    std::vector<std::pair<BGRectInt, int32_t>> result;
    BGRectInt rect(BGPointInt(llx, lly), BGPointInt(urx, ury));
    _query_tree[layer_idx].query(bgi::within(rect), std::back_inserter(result));
    for (auto& pair : result) {
      net_ids.insert(pair.second);
    }
    return net_ids;
  };
  auto queryRectbyRtreeWithWithin = [](std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>>& _query_tree, int32_t layer_idx,
                                       int32_t llx, int32_t lly, int32_t urx, int32_t ury) {
    std::set<int32_t> net_ids;
    std::vector<std::pair<BGRectInt, int32_t>> result;
    BGRectInt rect(BGPointInt(llx, lly), BGPointInt(urx, ury));
    _query_tree[layer_idx].query(bgi::within(rect), std::back_inserter(result));
    return result;
  };

  auto queryRectbyRtreeWithOverlaps = [](std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>>& _query_tree, int32_t layer_idx,
                                         int32_t llx, int32_t lly, int32_t urx, int32_t ury) {
    std::set<int32_t> net_ids;
    std::vector<std::pair<BGRectInt, int32_t>> result;
    BGRectInt rect(BGPointInt(llx, lly), BGPointInt(urx, ury));
    _query_tree[layer_idx].query(bgi::overlaps(rect), std::back_inserter(result));
    return result;
  };

  auto get_width_idx = [](int32_t width, std::vector<JogToJogSpacingRule::WidthRule>& width_rules) {
    int32_t width_idx = -1;
    for (int32_t i = width_rules.size() - 1; i >= 0; i--) {
      if (width >= width_rules[i].width) {
        width_idx = i;
        break;
      }
    }
    return width_idx;
  };
  auto getApplicableRule
      = [](int32_t width, int32_t parallel_length, int32_t within, int32_t long_jog_spacing, int32_t short_jog_spacing, const JogToJogSpacingRule& rule) {
          for (int32_t i = rule.widthRules.size() - 1; i >= 0; i--) {
            if (width >= rule.widthRules[i].width && parallel_length > rule.widthRules[i].parallelLength && within < rule.widthRules[i].within) {
              return i;
            }
          }
          return -1;
        };
  auto get_parallel_length = [](const PlanarRect& rect_a, const PlanarRect& rect_b) {
    int32_t parallel_length = 0;
    int32_t h_spacing = std::max(0, std::max(rect_a.get_ll_x() - rect_b.get_ur_x(), rect_b.get_ll_x() - rect_a.get_ur_x()));
    int32_t v_spacing = std::max(0, std::max(rect_a.get_ll_y() - rect_b.get_ur_y(), rect_b.get_ll_y() - rect_a.get_ur_y()));

    if (h_spacing > 0 && v_spacing > 0) {
      parallel_length = 0;
    } else {
      if (v_spacing == 0) {
        parallel_length = std::min(rect_a.get_ur_y(), rect_b.get_ur_y()) - std::max(rect_a.get_ll_y(), rect_b.get_ll_y());
      } else {
        parallel_length = std::min(rect_a.get_ur_x(), rect_b.get_ur_x()) - std::max(rect_a.get_ll_x(), rect_b.get_ll_x());
      }
    }
    return parallel_length;
  };
#endif

  // 得到基础数据
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  // 使用R树查询检测
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> routing_layer_all_query_tree;

  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> routing_layer_net_gtl_all_poly_set;
  std::map<int32_t, std::map<int32_t, std::vector<GTLRectInt>>> routing_layer_net_gtl_all_maxrect_list;

  {
    for (DRCShape* rect : rv_box.get_drc_result_shape_list()) {
      if (rect->get_is_routing() == true) {
        int32_t layer_idx = rect->get_layer_idx();
        int32_t net_idx = rect->get_net_idx();
        routing_layer_net_gtl_all_poly_set[layer_idx][net_idx] += DRCUTIL.convertToGTLRectInt(rect->get_rect());
      }
    }

    for (DRCShape* rect : rv_box.get_drc_env_shape_list()) {
      if (rect->get_is_routing() == true) {
        int32_t layer_idx = rect->get_layer_idx();
        int32_t net_idx = rect->get_net_idx();
        routing_layer_net_gtl_all_poly_set[layer_idx][net_idx] += DRCUTIL.convertToGTLRectInt(rect->get_rect());
      }
    }
    for (auto& [routing_layer_idx, net_all_gtl_poly_set] : routing_layer_net_gtl_all_poly_set) {
      for (auto& [net_idx, res_all_poly_set] : net_all_gtl_poly_set) {
        std::vector<GTLRectInt> rect_list;
        gtl::get_max_rectangles(rect_list, res_all_poly_set);
        for (GTLRectInt& rect : rect_list) {
          routing_layer_net_gtl_all_maxrect_list[routing_layer_idx][net_idx].push_back(rect);
          addRectToRtree(routing_layer_all_query_tree, rect, routing_layer_idx, net_idx);
        }
      }
    }
  }

#if 1
  for (auto& [routing_layer_idx, net_gtl_all_maxrect_list] : routing_layer_net_gtl_all_maxrect_list) {
    for (auto& [current_net_idx, rect_list] : net_gtl_all_maxrect_list) {
      for (GTLRectInt& current_gtl_rect : rect_list) {
        PlanarRect current_planar_rect = DRCUTIL.convertToPlanarRect(current_gtl_rect);
        auto current_rect_orientation = gtl::guess_orientation(current_gtl_rect);
        auto current_rect_width_orientation = current_rect_orientation.get_perpendicular();
        int current_rect_width = gtl::delta(current_gtl_rect, current_rect_width_orientation);

        // 宽度不满足条件
        if (current_rect_width < rule.widthRules[0].width) {
          continue;
        }

        auto checkFunction = [&](const gtl::orientation_2d& orientation) {
          GTLRectInt bloat_current_rect = current_gtl_rect;
          gtl::bloat(bloat_current_rect, orientation, rule.widthRules[rule.rows - 1].within - 1);

          std::vector<std::pair<BGRectInt, int32_t>> around_rect_result
              = queryRectbyRtreeWithIntersects(routing_layer_all_query_tree, routing_layer_idx, gtl::xl(bloat_current_rect), gtl::yl(bloat_current_rect),
                                               gtl::xh(bloat_current_rect), gtl::yh(bloat_current_rect));

          for (auto& [around_bg_rect, around_rect_net_idx] : around_rect_result) {
            GTLRectInt around_gtl_rect = DRCUTIL.convertToGTLRectInt(around_bg_rect);
            PlanarRect around_planar_rect = DRCUTIL.convertToPlanarRect(around_gtl_rect);

            if (DRCUTIL.isClosedOverlap(current_planar_rect, around_planar_rect)) {
              continue;
            }

            // 创建两个矩形之间的spacing rect
            PlanarRect spacing_planar_rect = DRCUTIL.getSpacingRect(current_planar_rect, around_planar_rect);
            GTLRectInt spacing_gtl_rect = DRCUTIL.convertToGTLRectInt(spacing_planar_rect);

            int32_t spacing_within = gtl::delta(spacing_gtl_rect, current_rect_width_orientation);
            int32_t spacing_prl = gtl::delta(spacing_gtl_rect, current_rect_orientation);

            int32_t applicable_rule_idx = getApplicableRule(current_rect_width, spacing_prl, spacing_within, rule.jogToJogSpacing, rule.shortJogSpacing, rule);

            if (applicable_rule_idx == -1) {
              continue;
            }

            auto& applicable_rule = rule.widthRules[applicable_rule_idx];

            GTLPolySetInt check_region_all;
            check_region_all += spacing_gtl_rect;

            gtl::shrink(spacing_gtl_rect, current_rect_width_orientation, 1);

            /// 查询
            std::vector<std::pair<BGRectInt, int32_t>> rects_in_spacing_rect
                = queryRectbyRtreeWithIntersects(routing_layer_all_query_tree, routing_layer_idx, gtl::xl(spacing_gtl_rect), gtl::yl(spacing_gtl_rect),
                                                 gtl::xh(spacing_gtl_rect), gtl::yh(spacing_gtl_rect));

            for (auto& [jog_bg_rect, jog_net_idx] : rects_in_spacing_rect) {
              if (jog_net_idx != current_net_idx && jog_net_idx != around_rect_net_idx) {
                continue;
              }
              GTLRectInt jog_gtl_rect = DRCUTIL.convertToGTLRectInt(jog_bg_rect);
              PlanarRect jog_planar_rect = DRCUTIL.convertToPlanarRect(jog_gtl_rect);

              PlanarRect overlap_rect = DRCUTIL.getOverlap(jog_planar_rect, spacing_planar_rect);
              GTLRectInt overlap_gtl_rect = DRCUTIL.convertToGTLRectInt(overlap_rect);
              check_region_all -= overlap_gtl_rect;
            }
            std::vector<GTLRectInt> check_regions_rect;
            gtl::get_rectangles(check_regions_rect, check_region_all, current_rect_width_orientation);
            for (GTLRectInt& check_region_rect : check_regions_rect) {
              PlanarRect check_region_planar_rect = DRCUTIL.convertToPlanarRect(check_region_rect);
              int32_t jog_width = gtl::delta(check_region_rect, current_rect_orientation);
              int32_t jog_spacing = gtl::delta(check_region_rect, current_rect_width_orientation);
              int32_t need_sapcing = jog_width > rule.jogWidth ? applicable_rule.longJogSpacing : rule.shortJogSpacing;

              if (jog_spacing < need_sapcing) {
                Violation violation;
                violation.set_violation_type(ViolationType::kJogToJogSpacing);
                violation.set_is_routing(true);
                violation.set_violation_net_set({current_net_idx, around_rect_net_idx});
                violation.set_layer_idx(routing_layer_idx);
                violation.set_rect(spacing_planar_rect);
                violation.set_required_size(need_sapcing);
                rv_box.get_violation_list().push_back(violation);
              }
            }
          }
        };

        checkFunction(current_rect_width_orientation);
      }
    }
#endif
  }
}
#endif

}  // namespace idrc
