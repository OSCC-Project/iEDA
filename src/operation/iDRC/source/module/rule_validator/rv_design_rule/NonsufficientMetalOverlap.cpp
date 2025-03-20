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

void RuleValidator::verifyNonsufficientMetalOverlap(RVBox& rv_box)
{
  //存在少量少检,没想好怎么做,与innovus报的形状不一样,可能是检测方法造成的
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  std::vector<Violation>& violation_list = rv_box.get_violation_list();
  PlanarRect debug_rect(234990, 59015, 235280, 59215);
  int32_t non_sufficient_metal_overlap_drc = 0;
  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> layer_net_poly_set;
  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> layer_net_over_lap_poly_set;
  std::map<int32_t, std::map<int32_t, std::vector<GTLRectInt>>> layer_net_rect_list_set;
  // 拿到所有的图形
  for (DRCShape* rect : rv_box.get_drc_env_shape_list()) {
    if (!rect->get_is_routing() || rect->get_net_idx() == -1) {  // 不是routing layer或者net_idx为-1的跳过该检测
      continue;
    }
    int32_t net_idx = rect->get_net_idx();
    int32_t layer_idx = rect->get_layer_idx();

    GTLPolyInt poly;

    GTLRectInt rect_gtl(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y());

    layer_net_poly_set[layer_idx][net_idx] += rect_gtl;
    layer_net_rect_list_set[layer_idx][net_idx].push_back(rect_gtl);
  }

  for (DRCShape* rect : rv_box.get_drc_result_shape_list()) {
    if (!rect->get_is_routing() || rect->get_net_idx() == -1) {  // 不是routing layer或者net_idx为-1的跳过该检测
      continue;
    }
    int32_t net_idx = rect->get_net_idx();
    int32_t layer_idx = rect->get_layer_idx();

    GTLPolyInt poly;

    GTLRectInt rect_gtl(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y());

    layer_net_poly_set[layer_idx][net_idx] += rect_gtl;
    layer_net_rect_list_set[layer_idx][net_idx].push_back(rect_gtl);
  }

  // function :merge rect list and del overlap rect
  // 合并所有矩形，拿到最小矩形子集，该函数可能还有点问题
  auto merge_rect_list_func = [](std::vector<GTLRectInt> rect_list) {
    std::vector<GTLRectInt> merged_rect_list;
    std::vector<bool> is_covered(rect_list.size(), false);
    for (int i = 0; i < rect_list.size(); i++) {
      if (is_covered[i]) {
        continue;
      }
      for (int j = 0; j < rect_list.size() && j != i; j++) {
        // chcek is i and j can merge into one rect
        GTLPolySetInt merge_rect = rect_list[i] + rect_list[j];
        std::vector<GTLRectInt> rects;
        merge_rect.get(rects);
        if (rects.size() == 1) {
          // 只有一个矩形，说明合并后的图形仍然是矩形
          rect_list[i] = rects[0];
          is_covered[j] = true;
        }
      }
    }
    for (int i = 0; i < rect_list.size(); i++) {
      if (!is_covered[i]) {
        merged_rect_list.push_back(rect_list[i]);
      }
    }
    return merged_rect_list;
  };
  // 从最小子集中拿到重叠部分
  for (auto [layer_idx, net_rect_list] : layer_net_rect_list_set) {
    for (auto [net_idx, rect_list] : net_rect_list) {
      // merge rect list
      std::vector<GTLRectInt> merged_rect_list = merge_rect_list_func(rect_list);
      GTLPolySetInt temp_poly_set;

      for (GTLRectInt rect : merged_rect_list) {
        layer_net_over_lap_poly_set[layer_idx][net_idx] += rect & temp_poly_set;
        temp_poly_set += rect;
      }
    }
  }

  // 重叠部分合并切分得到max rect,从max rect中找到不满足的部分
  for (auto [layer_idx, net_over_lap_poly_set] : layer_net_over_lap_poly_set) {
    for (auto [net_idx, over_lap_poly_set] : net_over_lap_poly_set) {
      std::vector<GTLRectInt> max_rect_list;
      gtl::get_max_rectangles(max_rect_list, over_lap_poly_set);
      int32_t min_width = routing_layer_list[layer_idx].get_min_width();

      for (GTLRectInt max_rect : max_rect_list) {
        int32_t llx = gtl::xl(max_rect);
        int32_t lly = gtl::yl(max_rect);
        int32_t urx = gtl::xh(max_rect);
        int32_t ury = gtl::yh(max_rect);
        int32_t diag_length = static_cast<int32_t>(std::sqrt((urx - llx) * (urx - llx) + (ury - lly) * (ury - lly)));
        if (diag_length < min_width) {
          std::set<int32_t> net_set;
          net_set.insert(net_idx);

          Violation violation;
          violation.set_violation_type(ViolationType::kNonsufficientMetalOverlap);
          violation.set_is_routing(true);
          violation.set_violation_net_set(net_set);
          violation.set_required_size(min_width);
          violation.set_layer_idx(layer_idx);
          violation.set_rect(PlanarRect(llx, lly, urx, ury));

          violation_list.push_back(violation);
          non_sufficient_metal_overlap_drc += 1;
          DRCLOG.info(Loc::current(), "NonsufficientMetalOverlap violation :", violation.get_layer_idx(), " ", llx, " ", lly, " ", urx, " ", ury);
        }
      }
    }
  }
  // if (non_sufficient_metal_overlap_drc > 0) {
  //   DRCLOG.info(Loc::current(), "NonsufficientMetalOverlap num :", non_sufficient_metal_overlap_drc);
  // }
}

}  // namespace idrc
