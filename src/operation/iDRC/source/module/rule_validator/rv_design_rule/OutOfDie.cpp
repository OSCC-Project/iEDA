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

void RuleValidator::verifyOutOfDie(RVCluster& rv_cluster)
{
  Die& die = DRCDM.getDatabase().get_die();
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = DRCDM.getDatabase().get_cut_to_adjacent_routing_map();

  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> routing_net_gtl_poly_set_map;
  for (DRCShape* drc_shape : rv_cluster.get_drc_result_shape_list()) {
    if (DRCUTIL.isInside(die, drc_shape->get_rect())) {
      continue;
    }
    int32_t routing_layer_idx = -1;
    if (!drc_shape->get_is_routing()) {
      std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[drc_shape->get_layer_idx()];
      routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
    } else {
      routing_layer_idx = drc_shape->get_layer_idx();
    }
    GTLPolySetInt rect_poly_set;
    rect_poly_set += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
    rect_poly_set -= DRCUTIL.convertToGTLRectInt(die);
    routing_net_gtl_poly_set_map[routing_layer_idx][drc_shape->get_net_idx()] += rect_poly_set;
  }
  for (auto& [routing_layer_idx, net_gtl_poly_set_map] : routing_net_gtl_poly_set_map) {
    for (auto& [net_idx, gtl_poly_set] : net_gtl_poly_set_map) {
      std::vector<GTLRectInt> gtl_rect_list;
      gtl::get_max_rectangles(gtl_rect_list, gtl_poly_set);
      for (GTLRectInt& gtl_rect : gtl_rect_list) {
        Violation violation;
        violation.set_violation_type(ViolationType::kOutOfDie);
        violation.set_is_routing(true);
        violation.set_violation_net_set({net_idx});
        violation.set_required_size(0);
        violation.set_layer_idx(routing_layer_idx);
        violation.set_rect(DRCUTIL.convertToPlanarRect(gtl_rect));
        rv_cluster.get_violation_list().push_back(violation);
      }
    }
  }
}

}  // namespace idrc
