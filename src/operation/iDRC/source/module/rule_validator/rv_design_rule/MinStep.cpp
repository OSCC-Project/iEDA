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

void RuleValidator::verifyMinStep(RVCluster& rv_cluster)
{
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  const auto& layer_data = rv_cluster.get_layer_data();

  for (auto& [routing_layer_idx, rv_layer_data] : layer_data) {
    RoutingLayer& routing_layer = routing_layer_list[routing_layer_idx];
    MinStepRule& min_step_rule = routing_layer.get_min_step_rule();
    int32_t min_step = min_step_rule.min_step;
    int32_t max_edges = min_step_rule.max_edges;
    int32_t lef58_min_step = min_step_rule.lef58_min_step;
    int32_t lef58_min_adjacent_length = min_step_rule.lef58_min_adjacent_length;

    for (auto& [net_idx, routing_net] : rv_layer_data.nets) {
      if (net_idx == -1) {
        continue;
      }

      for (const PolygonData& polygon_data : rv_layer_data.getPolygons(routing_net)) {
        std::span<const BoundaryData> polygon_boundaries = rv_layer_data.getBoundaries(polygon_data);
        if (polygon_boundaries.size() < 4) {
          continue;
        }

        std::vector<bool> visited_ring_boundary(static_cast<size_t>(polygon_data.boundary_count), false);
        for (const BoundaryData& seed_boundary : polygon_boundaries) {
          int32_t seed_boundary_id = rv_layer_data.getBoundaryId(seed_boundary);
          int32_t seed_local_idx = seed_boundary_id - polygon_data.boundary_begin;
          if (visited_ring_boundary[seed_local_idx]) {
            continue;
          }

          std::vector<int32_t> ring_boundary_id_list;
          int32_t curr_boundary_id = seed_boundary_id;
          do {
            int32_t local_idx = curr_boundary_id - polygon_data.boundary_begin;
            if (local_idx < 0 || polygon_data.boundary_count <= local_idx || visited_ring_boundary[local_idx]) {
              break;
            }
            visited_ring_boundary[local_idx] = true;
            ring_boundary_id_list.push_back(curr_boundary_id);
            curr_boundary_id = rv_layer_data.getBoundary(curr_boundary_id).next_boundary_id;
          } while (curr_boundary_id != seed_boundary_id);

          int32_t coord_size = static_cast<int32_t>(ring_boundary_id_list.size());
          if (coord_size < 4) {
            continue;
          }

          std::vector<PlanarCoord> coord_list;
          coord_list.reserve(coord_size);
          for (int32_t boundary_id : ring_boundary_id_list) {
            coord_list.push_back(rv_layer_data.getBoundary(boundary_id).end_coord);
          }

          for (int32_t i = 0; i < coord_size; i++) {
            // case 1
            const BoundaryData& curr_boundary = rv_layer_data.getBoundary(ring_boundary_id_list[i]);
            if (curr_boundary.edge_length < min_step) {
              int32_t small_edge_num = 1;
              for (int32_t j = 1; j < coord_size; ++j) {
                if (min_step <= rv_layer_data.getBoundary(ring_boundary_id_list[getIdx(i + j, coord_size)]).edge_length) {
                  break;
                }
                small_edge_num++;
              }
              if (max_edges < small_edge_num) {
                PlanarRect violation_rect(INT32_MAX, INT32_MAX, INT32_MIN, INT32_MIN);
                int32_t total_steps = small_edge_num + 1;

                for (int32_t step = 0; step < total_steps; step++) {
                  int32_t idx = getIdx(i - 1 + step, coord_size);
                  violation_rect.set_ll_x(std::min(violation_rect.get_ll_x(), coord_list[idx].get_x()));
                  violation_rect.set_ll_y(std::min(violation_rect.get_ll_y(), coord_list[idx].get_y()));
                  violation_rect.set_ur_x(std::max(violation_rect.get_ur_x(), coord_list[idx].get_x()));
                  violation_rect.set_ur_y(std::max(violation_rect.get_ur_y(), coord_list[idx].get_y()));
                }

                Violation violation;
                violation.set_violation_type(ViolationType::kMinStep);
                violation.set_required_size(min_step);
                violation.set_is_routing(true);
                violation.set_violation_net_set({net_idx});
                violation.set_layer_idx(routing_layer_idx);
                violation.set_rect(violation_rect);
                rv_cluster.get_violation_list().push_back(violation);
              }
              i += small_edge_num - 1;
            }
          }

          for (int32_t i = 0; i < coord_size; i++) {
            // case 2
            int32_t pre_i = getIdx(i - 1, coord_size);
            int32_t post_i = getIdx(i + 1, coord_size);
            const BoundaryData& pre_boundary = rv_layer_data.getBoundary(ring_boundary_id_list[pre_i]);
            const BoundaryData& curr_boundary = rv_layer_data.getBoundary(ring_boundary_id_list[i]);
            const BoundaryData& post_boundary = rv_layer_data.getBoundary(ring_boundary_id_list[post_i]);
            if (pre_boundary.isConvex == false && curr_boundary.isConvex == true && post_boundary.isConvex == false) {
              if ((curr_boundary.edge_length < lef58_min_step && post_boundary.edge_length < lef58_min_adjacent_length)
                  || (curr_boundary.edge_length < lef58_min_adjacent_length && post_boundary.edge_length < lef58_min_step)) {
                PlanarRect violation_rect = DRCUTIL.getRect(coord_list[pre_i], coord_list[post_i]);
                Violation violation;
                violation.set_violation_type(ViolationType::kMinStep);
                violation.set_required_size(lef58_min_step);
                violation.set_is_routing(true);
                violation.set_violation_net_set({net_idx});
                violation.set_layer_idx(routing_layer_idx);
                violation.set_rect(violation_rect);
                rv_cluster.get_violation_list().push_back(violation);
              }
            }
          }
        }
      }
    }
  }
}
}  // namespace idrc
