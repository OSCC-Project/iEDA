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

void RuleValidator::verifyCornerFillSpacing(RVCluster& rv_cluster)
{
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  const auto& layer_data = rv_cluster.get_layer_data();

  for (auto& [routing_layer_idx, rv_layer_data] : layer_data) {
    RoutingLayer& routing_layer = routing_layer_list[routing_layer_idx];
    CornerFillSpacingRule& corner_fill_spacing_rule = routing_layer.get_corner_fill_spacing_rule();
    if (!corner_fill_spacing_rule.has_corner_fill) {
      continue;
    }
    int32_t corner_fill_spacing = corner_fill_spacing_rule.corner_fill_spacing;
    int32_t edge_length_1 = corner_fill_spacing_rule.edge_length_1;
    int32_t edge_length_2 = corner_fill_spacing_rule.edge_length_2;
    int32_t adjacent_eol = corner_fill_spacing_rule.adjacent_eol;
    std::map<std::set<int32_t>, GTLPolySetInt> net_violation_poly_set_map;
    for (auto& [net_idx, routing_net] : rv_layer_data.nets) {
      for (const PolygonData& polygon_data : rv_layer_data.getPolygons(routing_net)) {
        if (rv_layer_data.getOuterBoundaryCount(polygon_data) < 4) {
          continue;
        }
        int32_t polygon_id = rv_layer_data.getPolygonId(polygon_data);

        std::vector<int32_t> valid_boundary_id_list;
        for (const BoundaryData& curr_boundary : rv_layer_data.getBoundaries(polygon_data)) {
          if (curr_boundary.isHole || curr_boundary.isConvex) {
            continue;
          }

          int32_t boundary_id = rv_layer_data.getBoundaryId(curr_boundary);
          const BoundaryData& prev_boundary = rv_layer_data.getPrevBoundary(boundary_id);
          const BoundaryData& next_boundary = rv_layer_data.getNextBoundary(boundary_id);
          if (prev_boundary.isHole || next_boundary.isHole) {
            continue;
          }

          bool is_valid_corner_fill = false;
          if (curr_boundary.edge_length < edge_length_1 && next_boundary.edge_length < edge_length_2) {
            const BoundaryData& next_next_boundary = rv_layer_data.getBoundary(next_boundary.next_boundary_id);
            if (next_boundary.isConvex && next_next_boundary.isConvex && next_next_boundary.edge_length < adjacent_eol) {
              is_valid_corner_fill = true;
            }
          }
          if (next_boundary.edge_length < edge_length_1 && curr_boundary.edge_length < edge_length_2) {
            const BoundaryData& prev_prev_boundary = rv_layer_data.getBoundary(prev_boundary.prev_boundary_id);
            if (prev_prev_boundary.isConvex && prev_boundary.isConvex && prev_boundary.edge_length < adjacent_eol) {
              is_valid_corner_fill = true;
            }
          }
          if (is_valid_corner_fill) {
            valid_boundary_id_list.push_back(boundary_id);
          }
        }

        for (int32_t valid_boundary_id : valid_boundary_id_list) {
          const BoundaryData& curr_boundary = rv_layer_data.getBoundary(valid_boundary_id);
          const BoundaryData& next_boundary = rv_layer_data.getNextBoundary(valid_boundary_id);
          PlanarCoord curr_coord = curr_boundary.end_coord;
          PlanarRect corner_fill_rect;
          std::vector<Orientation> orientation_list;
          {
            PlanarCoord pre_coord = curr_boundary.begin_coord;
            PlanarCoord post_coord = next_boundary.end_coord;
            PlanarCoord diag_coord(pre_coord.get_x() ^ curr_coord.get_x() ^ post_coord.get_x(), pre_coord.get_y() ^ curr_coord.get_y() ^ post_coord.get_y());
            corner_fill_rect = DRCUTIL.getRect(curr_coord, diag_coord);
            orientation_list = DRCUTIL.getOrientationList(curr_coord, diag_coord);
          }
          PlanarRect check_rect;
          {
            check_rect = corner_fill_rect;
            if (DRCUTIL.exist(orientation_list, Orientation::kEast)) {
              check_rect = DRCUTIL.getEnlargedRect(check_rect, 0, 0, corner_fill_spacing, 0);
            } else if (DRCUTIL.exist(orientation_list, Orientation::kWest)) {
              check_rect = DRCUTIL.getEnlargedRect(check_rect, corner_fill_spacing, 0, 0, 0);
            }
            if (DRCUTIL.exist(orientation_list, Orientation::kSouth)) {
              check_rect = DRCUTIL.getEnlargedRect(check_rect, 0, corner_fill_spacing, 0, 0);
            } else if (DRCUTIL.exist(orientation_list, Orientation::kNorth)) {
              check_rect = DRCUTIL.getEnlargedRect(check_rect, 0, 0, 0, corner_fill_spacing);
            }
          }
          std::vector<std::pair<GTLRectInt, int32_t>> gtl_rect_id_pair_list;
          rv_layer_data.queryMaxRects(DRCUTIL.convertToGTLRectInt(check_rect), std::back_inserter(gtl_rect_id_pair_list));
          for (auto& [env_gtl_rect, env_max_rect_id] : gtl_rect_id_pair_list) {
            int32_t env_net_idx = rv_layer_data.getNetIdxByMaxRectId(env_max_rect_id);
            if (net_idx == env_net_idx && rv_layer_data.getMaxRect(env_max_rect_id).polygon_id == polygon_id) {
              continue;
            }
            PlanarRect env_rect = DRCUTIL.convertToPlanarRect(env_gtl_rect);
            if (!DRCUTIL.isOpenOverlap(env_rect, check_rect)) {
              continue;
            }
            PlanarRect overlap_rect = DRCUTIL.getOverlap(env_rect, check_rect);
            if (DRCUTIL.getEuclideanDistance(corner_fill_rect, overlap_rect) >= corner_fill_spacing) {
              continue;
            }
            PlanarCoord violation_coord;
            if (DRCUTIL.exist(orientation_list, Orientation::kEast)) {
              violation_coord.set_x(corner_fill_rect.get_ll_x() == overlap_rect.get_ll_x() ? corner_fill_rect.get_ur_x() : overlap_rect.get_ll_x());
            } else if (DRCUTIL.exist(orientation_list, Orientation::kWest)) {
              violation_coord.set_x(corner_fill_rect.get_ur_x() == overlap_rect.get_ur_x() ? corner_fill_rect.get_ll_x() : overlap_rect.get_ur_x());
            }
            if (DRCUTIL.exist(orientation_list, Orientation::kSouth)) {
              violation_coord.set_y(corner_fill_rect.get_ur_y() == overlap_rect.get_ur_y() ? corner_fill_rect.get_ll_y() : overlap_rect.get_ur_y());
            } else if (DRCUTIL.exist(orientation_list, Orientation::kNorth)) {
              violation_coord.set_y(corner_fill_rect.get_ll_y() == overlap_rect.get_ll_y() ? corner_fill_rect.get_ur_y() : overlap_rect.get_ll_y());
            }
            net_violation_poly_set_map[{net_idx, env_net_idx}] += DRCUTIL.convertToGTLRectInt(DRCUTIL.getRect(curr_coord, violation_coord));
          }
        }
      }
    }
    for (auto& [net_idx_set, violation_poly_set] : net_violation_poly_set_map) {
      std::vector<GTLRectInt> violation_rect_list;
      gtl::get_max_rectangles(violation_rect_list, violation_poly_set);
      for (GTLRectInt& violation_rect : violation_rect_list) {
        Violation violation;
        violation.set_violation_type(ViolationType::kCornerFillSpacing);
        violation.set_required_size(corner_fill_spacing);
        violation.set_is_routing(true);
        violation.set_violation_net_set(net_idx_set);
        violation.set_layer_idx(routing_layer_idx);
        violation.set_rect(DRCUTIL.convertToPlanarRect(violation_rect));
        rv_cluster.get_violation_list().push_back(violation);
      }
    }
  }
}

}  // namespace idrc
