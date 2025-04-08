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

void RuleValidator::verifySameLayerCutSpacing(RVBox& rv_box)
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
  auto get_vio_rect = [](PlanarRect& rect_a, PlanarRect& rect_b) {
    // 检测水平方向距离和竖直方向距离
    int32_t h_spacing = std::max(0, std::max(rect_a.get_ll_x() - rect_b.get_ur_x(), rect_b.get_ll_x() - rect_a.get_ur_x()));
    int32_t v_spacing = std::max(0, std::max(rect_a.get_ll_y() - rect_b.get_ur_y(), rect_b.get_ll_y() - rect_a.get_ur_y()));
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
        violation_rect
            = DRCUTIL.getOverlap(DRCUTIL.getEnlargedRect(rect_a, h_spacing, 0, h_spacing, 0), DRCUTIL.getEnlargedRect(rect_b, h_spacing, 0, h_spacing, 0));
      } else {  // h_spaing = 0
        violation_rect
            = DRCUTIL.getOverlap(DRCUTIL.getEnlargedRect(rect_a, 0, v_spacing, 0, v_spacing), DRCUTIL.getEnlargedRect(rect_b, 0, v_spacing, 0, v_spacing));
      }
    }
    return violation_rect;
  };
#endif
  // 规则定义
  // int32_t cut_spacing_a = 80 * 2;
  // int32_t cut_spacing_b = 90 * 2;
  int32_t cut_spacing_a = 70 * 2;
  int32_t cut_spacing_b = 80 * 2;
  int32_t prl = -1 * 40 * 2;

  std::vector<Violation>& violation_list = rv_box.get_violation_list();

  std::vector<int32_t> cut_eol_spacing_layers = {1, 2, 3, 4, 5, 6};
  std::map<int32_t, GTLPolySetInt> layer_cut_gtl_poly_set;  // cut idx为对应的上层layer的idx,cut idx -1 是cut的下层
  std::map<int32_t, GTLPolySetInt> layer_net_gtl_poly_set;

  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> layer_res_query_tree;
  std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>> layer_all_query_tree;

  std::map<int32_t, std::map<int32_t, std::vector<GTLRectInt>>> layer_net_gtl_res_maxrect_list;  // 只检查这个的cut

  for (DRCShape* rect : rv_box.get_drc_env_shape_list()) {
    if (rect->get_is_routing()) {
      layer_net_gtl_poly_set[rect->get_layer_idx()] += GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y());
    } else {
      layer_cut_gtl_poly_set[rect->get_layer_idx()] += GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y());
      addRectToRtree(layer_all_query_tree, GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y()), rect->get_layer_idx(),
                     rect->get_net_idx());
    }
  }
  for (DRCShape* rect : rv_box.get_drc_result_shape_list()) {
    if (rect->get_is_routing()) {
      layer_net_gtl_poly_set[rect->get_layer_idx()] += GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y());
    } else {
      layer_cut_gtl_poly_set[rect->get_layer_idx()] += GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y());
      addRectToRtree(layer_all_query_tree, GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y()), rect->get_layer_idx(),
                     rect->get_net_idx());
      layer_net_gtl_res_maxrect_list[rect->get_layer_idx()][rect->get_net_idx()].push_back(
          GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y()));
    }
  }

  for (auto& [cuting_layer_idx, net_gtl_res_maxrect_list] : layer_net_gtl_res_maxrect_list) {
    for (auto& [net_idx, gtl_res_maxrect_list] : net_gtl_res_maxrect_list) {
      for (auto& gtl_res_maxrect : gtl_res_maxrect_list) {
        // 膨胀到cut spacing b进行查询
        int32_t bloat_spacing = cut_spacing_b;
        GTLRectInt bloat_res_gtl_maxrect = gtl_res_maxrect;
        gtl::bloat(bloat_res_gtl_maxrect, bloat_spacing);  // bloat会改变原来的值

        // prl为负数 , spacing_a_region 区域用距离a的值，其他地方用距离b
        int32_t b_llx = gtl::xl(bloat_res_gtl_maxrect);
        int32_t b_lly = gtl::yl(bloat_res_gtl_maxrect);
        int32_t b_urx = gtl::xh(bloat_res_gtl_maxrect);
        int32_t b_ury = gtl::yh(bloat_res_gtl_maxrect);
        int32_t o_llx = gtl::xl(gtl_res_maxrect);
        int32_t o_lly = gtl::yl(gtl_res_maxrect);
        int32_t o_urx = gtl::xh(gtl_res_maxrect);
        int32_t o_ury = gtl::yh(gtl_res_maxrect);
        GTLPolySetInt spaing_a_region = GTLRectInt(b_llx, b_lly, o_llx + prl, o_lly + prl) + GTLRectInt(o_urx - prl, o_ury - prl, b_urx, b_ury)
                                        + GTLRectInt(b_llx, o_ury - prl, o_llx + prl, b_ury) + GTLRectInt(o_urx - prl, b_lly, b_urx, o_lly + prl);

        GTLPolySetInt spacing_b_region = bloat_res_gtl_maxrect - spaing_a_region;

        std::map<int32_t, bgi::rtree<std::pair<BGRectInt, int32_t>, bgi::quadratic<16>>>& check_rtree = layer_all_query_tree;
        // 找到有交集的cut rect
        std::vector<std::pair<BGRectInt, int32_t>> overlap_rect_result = queryRectbyRtree(
            check_rtree, cuting_layer_idx, gtl::xl(bloat_res_gtl_maxrect) + 1, gtl::yl(bloat_res_gtl_maxrect) + 1, gtl::xh(bloat_res_gtl_maxrect) - 1,
            gtl::yh(bloat_res_gtl_maxrect) - 1);  // 从原图形中查找对应的rect,不要贴边的
        for (auto& [bgrect, q_net_idx] : overlap_rect_result) {
          if (net_idx == -1 && q_net_idx == -1) {
            continue;  // -1忽略
          }
          PlanarRect spacing_rect = DRCUTIL.convertToPlanarRect(bgrect);
          GTLRectInt gtl_spacing_rect = DRCUTIL.convertToGTLRectInt(bgrect);

          PlanarRect res_maxrect = DRCUTIL.convertToPlanarRect(gtl_res_maxrect);
          // 先check两个rect是否重叠
          if (DRCUTIL.isClosedOverlap(spacing_rect, res_maxrect)) {
            continue;  // 同一个polygon的忽略
          }
          // 如果与b区有重叠，那么采用距离b
          if (gtl::area(spacing_b_region & gtl_spacing_rect) > 0) {
            int32_t need_spacing = cut_spacing_b;
            PlanarRect violation_rect = get_vio_rect(spacing_rect, res_maxrect);
            Violation violation;
            violation.set_violation_type(ViolationType::kSameLayerCutSpacing);
            violation.set_is_routing(true);
            violation.set_violation_net_set({net_idx, q_net_idx});
            violation.set_layer_idx(cuting_layer_idx-1);
            violation.set_rect(violation_rect);
            violation.set_required_size(need_spacing);
            violation_list.push_back(violation);
          } else {  // 和a区重叠，需要计算一下距离
            int32_t need_spacing = cut_spacing_a;
            int32_t real_spacing = gtl::euclidean_distance(gtl_spacing_rect, gtl_res_maxrect);
            if(real_spacing >= need_spacing) {
              continue;  // 满足距离条件
            }
            PlanarRect violation_rect = get_vio_rect(spacing_rect, res_maxrect);
            Violation violation;
            violation.set_violation_type(ViolationType::kSameLayerCutSpacing);
            violation.set_is_routing(true);
            violation.set_violation_net_set({net_idx, q_net_idx});
            violation.set_layer_idx(cuting_layer_idx-1);
            violation.set_rect(violation_rect);
            violation.set_required_size(need_spacing);
            violation_list.push_back(violation);
          }
        }
      }
    }
  }
}

}  // namespace idrc
