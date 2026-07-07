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

void RuleValidator::verifyNotchSpacing(RVCluster& rv_cluster)
{
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  const auto& layer_data = rv_cluster.get_layer_data();

  for (auto& [routing_layer_idx, rv_layer_data] : layer_data) {
    RoutingLayer& routing_layer = routing_layer_list[routing_layer_idx];
    NotchSpacingRule& notch_spacing_rule = routing_layer.get_notch_spacing_rule();
    int32_t notch_spacing = notch_spacing_rule.notch_spacing;
    int32_t notch_length = notch_spacing_rule.notch_length;
    std::optional<int32_t> concave_ends = notch_spacing_rule.concave_ends;
    for (auto& [net_idx, routing_net] : rv_layer_data.nets) {
      if (net_idx == -1) {
        continue;
      }
      GTLPolySetInt violation_poly_set;
      for (const PolygonData& polygon_data : rv_layer_data.getPolygons(routing_net)) {
        std::vector<PlanarRect> origin_rect_list;
        {
          for (const MaxRectData& max_rect_data : rv_layer_data.getMaxRects(polygon_data)) {
            GTLRectInt gtl_rect = max_rect_data.rect;
            origin_rect_list.push_back(DRCUTIL.convertToPlanarRect(gtl_rect));
          }
        }

        std::set<int32_t> visited_boundary_ids;
        for (const BoundaryData& seed_boundary : rv_layer_data.getBoundaries(polygon_data)) {
          int32_t seed_boundary_id = rv_layer_data.getBoundaryId(seed_boundary);
          if (DRCUTIL.exist(visited_boundary_ids, seed_boundary_id)) {
            continue;
          }

          std::vector<int32_t> ring_boundary_ids;
          int32_t boundary_id = seed_boundary_id;
          do {
            if (DRCUTIL.exist(visited_boundary_ids, boundary_id)) {
              break;
            }
            visited_boundary_ids.insert(boundary_id);
            ring_boundary_ids.push_back(boundary_id);
            boundary_id = rv_layer_data.getBoundary(boundary_id).next_boundary_id;
          } while (boundary_id != seed_boundary_id);

          int32_t coord_size = static_cast<int32_t>(ring_boundary_ids.size());
          if (coord_size < 4) {
            continue;
          }

          std::vector<bool> convex_corner_list;
          std::vector<int32_t> edge_length_list;
          std::vector<Segment<PlanarCoord>> edge_list;
          convex_corner_list.reserve(coord_size);
          edge_length_list.reserve(coord_size);
          edge_list.reserve(coord_size);
          for (int32_t ring_boundary_id : ring_boundary_ids) {
            const BoundaryData& boundary_data = rv_layer_data.getBoundary(ring_boundary_id);
            convex_corner_list.push_back(boundary_data.isConvex);
            edge_length_list.push_back(boundary_data.edge_length);
            edge_list.emplace_back(boundary_data.begin_coord, boundary_data.end_coord);
          }

          for (int32_t i = 0; i < coord_size; i++) {
            if (concave_ends.has_value()) {
              if (!convex_corner_list[getIdx(i - 1, coord_size)] || convex_corner_list[i] || convex_corner_list[getIdx(i + 1, coord_size)]
                  || convex_corner_list[getIdx(i + 2, coord_size)] || !convex_corner_list[getIdx(i + 3, coord_size)]) {
                continue;
              }
              /**
               * 三凹角
               *
               *           i                                         i(side_edge)
               *      o---------o                               o---------o                               o
               *      |            o                            |                                         |            o
               *  i+1 |            | i+3                    i+1 |                                     i+1 |            | i+3
               *      |            |             (spacing_edge) |                          (concave_edge) |            | (side_edge)
               *      o------------o                            o------------o                            o------------o
               *           i+2                                       i+2(concave_edge)                           i+2(spacing_edge)
               *
               */
              for (auto [concave_edge_idx, spacing_edge_idx, side_edge_idx] :
                   {std::tuple(getIdx(i + 2, coord_size), getIdx(i + 1, coord_size), i),
                    std::tuple(getIdx(i + 1, coord_size), getIdx(i + 2, coord_size), getIdx(i + 3, coord_size))}) {
                if (edge_length_list[concave_edge_idx] < notch_length && edge_length_list[side_edge_idx] >= notch_length
                    && edge_length_list[spacing_edge_idx] < notch_spacing) {
                  Segment<PlanarCoord>& concave_edge = edge_list[concave_edge_idx];
                  Segment<PlanarCoord>& side_edge = edge_list[side_edge_idx];
                  bool has_violation = true;
                  for (const Segment<PlanarCoord>& edge : {concave_edge, side_edge}) {
                    for (PlanarRect& origin_rect : origin_rect_list) {
                      if (DRCUTIL.isInside(origin_rect, edge)) {
                        if (concave_ends < origin_rect.getWidth()) {
                          has_violation = false;
                        }
                        if (has_violation == false) {
                          break;
                        }
                      }
                    }
                  }
                  if (!has_violation) {
                    continue;
                  }
                  Segment<PlanarCoord>& short_edge = edge_length_list[concave_edge_idx] < edge_length_list[side_edge_idx] ? concave_edge : side_edge;
                  Segment<PlanarCoord>& spacing_edge = edge_list[spacing_edge_idx];
                  PlanarRect violation_rect
                      = DRCUTIL.getBoundingBox({short_edge.get_first(), short_edge.get_second(), spacing_edge.get_first(), spacing_edge.get_second()});
                  violation_poly_set += DRCUTIL.convertToGTLRectInt(violation_rect);
                }
              }
            } else {
              if (convex_corner_list[i] || convex_corner_list[getIdx(i + 1, coord_size)]) {
                continue;
              }
              /**
               * 双凹角
               *
               *                        i
               *                   o---------o                               o
               *                   |                                         |            o
               *               i+1 |                                       i |            | i+2
               *    (spacing_edge) |                                         |            |
               *                   o------------o                            o------------o
               *                        i+2                                       i+1(spacing_edge)
               *
               */
              if ((edge_length_list[i] < notch_length || edge_length_list[getIdx(i + 2, coord_size)] < notch_length)
                  && (edge_length_list[getIdx(i + 1, coord_size)] < notch_spacing)) {
                Segment<PlanarCoord>& short_edge
                    = edge_length_list[i] < edge_length_list[getIdx(i + 2, coord_size)] ? edge_list[i] : edge_list[getIdx(i + 2, coord_size)];
                Segment<PlanarCoord>& spacing_edge = edge_list[getIdx(i + 1, coord_size)];
                PlanarRect violation_rect
                    = DRCUTIL.getBoundingBox({short_edge.get_first(), short_edge.get_second(), spacing_edge.get_first(), spacing_edge.get_second()});
                violation_poly_set += DRCUTIL.convertToGTLRectInt(violation_rect);
              }
            }
          }
        }
      }
      std::vector<GTLRectInt> violation_rect_list;
      gtl::get_max_rectangles(violation_rect_list, violation_poly_set);
      for (GTLRectInt& violation_rect : violation_rect_list) {
        Violation violation;
        violation.set_violation_type(ViolationType::kNotchSpacing);
        violation.set_required_size(notch_spacing);
        violation.set_is_routing(true);
        violation.set_violation_net_set({net_idx});
        violation.set_layer_idx(routing_layer_idx);
        violation.set_rect(DRCUTIL.convertToPlanarRect(violation_rect));
        rv_cluster.get_violation_list().push_back(violation);
      }
    }
  }
}

}  // namespace idrc
