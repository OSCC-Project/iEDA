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

void RuleValidator::verifyMinHole(RVBox& rv_box)
{
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  std::vector<Violation>& violation_list = rv_box.get_violation_list();

  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> layer_net_poly_set;
  int32_t min_hole_drc = 0;

  for (DRCShape* rect : rv_box.get_drc_env_shape_list()) {
    if (!rect->get_is_routing() || rect->get_net_idx() == -1) {  // 不是routing layer或者net_idx为-1的跳过该检测
      continue;
    }
    int32_t net_idx = rect->get_net_idx();
    int32_t layer_idx = rect->get_layer_idx();
    layer_net_poly_set[layer_idx][net_idx] += GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y());
  }
  for (DRCShape* rect : rv_box.get_drc_result_shape_list()) {
    if (!rect->get_is_routing() || rect->get_net_idx() == -1) {  // 不是routing layer或者net_idx为-1的跳过该检测
      continue;
    }
    int32_t net_idx = rect->get_net_idx();
    int32_t layer_idx = rect->get_layer_idx();
    layer_net_poly_set[layer_idx][net_idx] += GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y());
  }
  for (auto& [layer_idx, net_poly_set] : layer_net_poly_set) {
    for (auto& [net_idx, poly_set] : net_poly_set) {
      // check min hole
      int32_t min_enclosed_area = routing_layer_list[layer_idx].get_min_hole();  // not sure if this is the right function
      // 假设poly_set已经包含了所有的shape,vio的过滤应该从结果中过滤
      std::vector<GTLHolePolyInt> hole_poly_list;
      poly_set.get(hole_poly_list);  // get会自动识别要变成的类型

      for (GTLHolePolyInt hole_poly : hole_poly_list) {
        // check holes for hole_poly
        GTLHolePolyInt::iterator_holes_type hole_iter = hole_poly.begin_holes();
        while (hole_iter != hole_poly.end_holes()) {
          GTLPolyInt hole = *hole_iter;  // 用普通的poly来代表hole
          int32_t hole_area = gtl::area(hole);
          if (hole_area >= min_enclosed_area) {
            hole_iter++;
            continue;
          }

          GTLRectInt hole_rect;
          gtl::extents(hole_rect,hole);
          int llx = gtl::xl(hole_rect);
          int lly = gtl::yl(hole_rect);
          int urx = gtl::xh(hole_rect);
          int ury = gtl::yh(hole_rect);

          std::set<int32_t> net_set;
          net_set.insert(net_idx);

          Violation violation;
          violation.set_violation_type(ViolationType::kMinHole);
          violation.set_is_routing(true);
          violation.set_violation_net_set(net_set);
          violation.set_required_size(min_enclosed_area);
          violation.set_layer_idx(layer_idx);
          violation.set_rect(llx, lly, urx, ury);
          violation_list.push_back(violation);
          min_hole_drc += 1;
          // DRCLOG.info(Loc::current(), "min hole violation :", violation.get_layer_idx(), " ", llx, " ", lly, " ", urx, " ", ury);

          hole_iter++;  // 这里不要忘了
        }
      }
    }
  }
  //   DRCLOG.info(Loc::current(), "min hole num: ", min_hole_drc);
}
}  // namespace idrc
