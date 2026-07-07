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
#include "Utility.hpp"

namespace idrc {

void RuleValidator::verifyMinimumArea(RVCluster& rv_cluster)
{
  using ViolationBBoxRTree = bgi::rtree<GTLRectInt, bgi::quadratic<16>>;

  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  const auto& layer_data = rv_cluster.get_layer_data();
  std::map<int32_t, std::map<int32_t, GTLPolySetInt>> env_layer_net_polyset_map;
  std::map<int32_t, ViolationBBoxRTree> env_violation_rtree_map;

  for (DRCShape* drc_shape : rv_cluster.get_drc_env_shape_list()) {
    if (!drc_shape->get_is_routing() || drc_shape->get_net_idx() == -1) {
      continue;
    }
    env_layer_net_polyset_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
  }

  for (const auto& [routing_layer_idx, net_polyset_map] : env_layer_net_polyset_map) {
    int32_t min_area = routing_layer_list[routing_layer_idx].get_minimum_area_rule().min_area;
    std::vector<GTLRectInt> env_violation_rtree_inputs;
    for (const auto& [net_idx, env_polyset] : net_polyset_map) {
      if (net_idx == -1) {
        continue;
      }
      std::vector<GTLPolyInt> gtl_poly_list;
      env_polyset.get_polygons(gtl_poly_list);
      for (GTLPolyInt& gtl_poly : gtl_poly_list) {
        if (gtl::area(gtl_poly) >= min_area) {
          continue;
        }
        GTLRectInt bbox;
        gtl::extents(bbox, gtl_poly);
        env_violation_rtree_inputs.push_back(bbox);
      }
    }
    env_violation_rtree_map[routing_layer_idx] = ViolationBBoxRTree(env_violation_rtree_inputs);
  }

  for (const auto& [routing_layer_idx, rv_layer_data] : layer_data) {
    int32_t min_area = routing_layer_list[routing_layer_idx].get_minimum_area_rule().min_area;
    std::vector<Violation> layer_violations;
    auto env_violation_rtree_it = env_violation_rtree_map.find(routing_layer_idx);

    for (const auto& [net_idx, routing_net] : rv_layer_data.nets) {
      for (const auto& polygon : rv_layer_data.getPolygons(routing_net)) {
        if (polygon.isEnv) {
          continue;
        }
        if (gtl::area(polygon.hole_poly) >= min_area) {
          continue;
        }
        GTLRectInt bbox;
        gtl::extents(bbox, polygon.hole_poly);
        if (env_violation_rtree_it != env_violation_rtree_map.end()) {
          std::vector<GTLRectInt> overlap_env_violation_list;
          env_violation_rtree_it->second.query(bgi::intersects(bbox), std::back_inserter(overlap_env_violation_list));
          bool is_exempted = false;
          for (auto& env_rect : overlap_env_violation_list) {
            if (DRCUTIL.isInside(DRCUTIL.convertToPlanarRect(bbox), DRCUTIL.convertToPlanarRect(env_rect))) {
              is_exempted = true;
              break;
            }
          }
          if (is_exempted) {
            continue;
          }
        }

        Violation violation;
        violation.set_violation_type(ViolationType::kMinimumArea);
        violation.set_is_routing(true);
        violation.set_violation_net_set({net_idx});
        violation.set_required_size(min_area);
        violation.set_layer_idx(routing_layer_idx);
        violation.set_rect(DRCUTIL.convertToPlanarRect(bbox));
        layer_violations.push_back(std::move(violation));
      }
    }

    rv_cluster.get_violation_list().insert(rv_cluster.get_violation_list().end(), layer_violations.begin(), layer_violations.end());
  }
}

}  // namespace idrc
