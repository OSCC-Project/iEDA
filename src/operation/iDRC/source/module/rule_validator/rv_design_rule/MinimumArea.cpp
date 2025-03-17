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

void RuleValidator::verifyMinimumArea(RVBox& rv_box)
{
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  std::vector<Violation>& violation_list = rv_box.get_violation_list();

  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> layer_net_poly_set;
  for (DRCShape* rect : rv_box.get_drc_env_shape_list()) {
    if (!rect->get_is_routing()) {
      continue;
    }
    int32_t net_idx = rect->get_net_idx();
    int32_t layer_idx = rect->get_layer_idx();
    layer_net_poly_set[layer_idx][net_idx] += GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y());
  }
  for (DRCShape* rect : rv_box.get_drc_result_shape_list()) {
    if (!rect->get_is_routing()) {
      continue;
    }
    int32_t net_idx = rect->get_net_idx();
    int32_t layer_idx = rect->get_layer_idx();
    layer_net_poly_set[layer_idx][net_idx] += GTLRectInt(rect->get_ll_x(), rect->get_ll_y(), rect->get_ur_x(), rect->get_ur_y());
  }

  for (auto& [layer_idx, net_poly_set] : layer_net_poly_set) {
    for (auto& [net_idx, poly_set] : net_poly_set) {
        std::vector<GTLPolyInt> poly_list;
        poly_set.get_polygons(poly_list);
        int32_t min_area = routing_layer_list[layer_idx].get_min_area();
        for(GTLPolyInt poly : poly_list){
            if(gtl::area(poly) < min_area){
              GTLRectInt violation_rect;
              gtl::extents(violation_rect,poly);//可以用max_rect代替
              int llx = gtl::xl(violation_rect);
              int lly = gtl::yl(violation_rect);
              int urx = gtl::xh(violation_rect);
              int ury = gtl::yh(violation_rect);

              std::set<int32_t> net_set;
              net_set.insert(net_idx);

              Violation violation;
              violation.set_violation_type(ViolationType::kMinimumArea);
              violation.set_is_routing(true);
              violation.set_violation_net_set(net_set);
              violation.set_required_size(min_area);
              violation.set_layer_idx(layer_idx);
              violation.set_rect(PlanarRect(llx,lly,urx,ury));

              violation_list.push_back(violation);
            }
        }
    }
  }
//   DRCLOG.info(Loc::current(),"min area num: ",violation_list.size());
}

}  // namespace idrc
