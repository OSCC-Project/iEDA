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
struct Corner
{
  PlanarCoord point;
  int32_t width;
  Orientation orient1;
  Orientation orient2;
  int32_t net_idx;
  int32_t polygon_id;
  int32_t boundary_id;
  int32_t next_boundary_id;
};

void RuleValidator::verifyCornerSpacing(RVCluster& rv_cluster)
{
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  const auto& layer_data = rv_cluster.get_layer_data();

  auto collect_exempted_eol_edges = [&](const auto& curr_layer_data, int32_t eol_width, std::set<int32_t>& exempted_eol_boundary_ids) {
    for (const auto& net_entry : curr_layer_data.nets) {
      const RVRoutingNet& routing_net = net_entry.second;
      for (const PolygonData& polygon_data : curr_layer_data.getPolygons(routing_net)) {
        std::vector<bool> visited_ring_boundary(static_cast<size_t>(polygon_data.boundary_count), false);
        for (const BoundaryData& seed_boundary : curr_layer_data.getBoundaries(polygon_data)) {
          int32_t seed_boundary_id = curr_layer_data.getBoundaryId(seed_boundary);
          int32_t seed_local_idx = seed_boundary_id - polygon_data.boundary_begin;
          if (visited_ring_boundary[seed_local_idx]) {
            continue;
          }

          int32_t curr_boundary_id = seed_boundary_id;
          do {
            int32_t local_idx = curr_boundary_id - polygon_data.boundary_begin;
            if (local_idx < 0 || polygon_data.boundary_count <= local_idx || visited_ring_boundary[local_idx]) {
              break;
            }
            visited_ring_boundary[local_idx] = true;

            const BoundaryData& curr_boundary = curr_layer_data.getBoundary(curr_boundary_id);
            const BoundaryData& prev_boundary = curr_layer_data.getPrevBoundary(curr_boundary_id);
            bool is_curr_eol_boundary = prev_boundary.isConvex && curr_boundary.isConvex;
            bool is_exempted = curr_boundary.edge_length < eol_width && is_curr_eol_boundary;
            if (is_exempted) {
              exempted_eol_boundary_ids.insert(curr_boundary_id);
            }
            curr_boundary_id = curr_boundary.next_boundary_id;
          } while (curr_boundary_id != seed_boundary_id);
        }
      }
    }
  };

  auto build_outside_query_rect = [](const Corner& corner, int32_t spacing) {
    Orientation outer_orient1 = DRCUTIL.getOppositeOrientation(corner.orient1);
    Orientation outer_orient2 = DRCUTIL.getOppositeOrientation(corner.orient2);

    int32_t ll_x_minus_offset = 0;
    int32_t ll_y_minus_offset = 0;
    int32_t ur_x_add_offset = 0;
    int32_t ur_y_add_offset = 0;
    for (Orientation orient : {outer_orient1, outer_orient2}) {
      if (orient == Orientation::kWest) {
        ll_x_minus_offset = spacing;
      } else if (orient == Orientation::kSouth) {
        ll_y_minus_offset = spacing;
      } else if (orient == Orientation::kEast) {
        ur_x_add_offset = spacing;
      } else if (orient == Orientation::kNorth) {
        ur_y_add_offset = spacing;
      }
    }
    return DRCUTIL.getEnlargedRect(corner.point, ll_x_minus_offset, ll_y_minus_offset, ur_x_add_offset, ur_y_add_offset);
  };

  auto queryNetIdxByRect = [](const RVLayerData& rv_layer_data, const PlanarRect& query_rect) -> int32_t {
    std::vector<std::pair<GTLRectInt, int32_t>> rect_max_rect_pair_list;
    rv_layer_data.queryMaxRects(DRCUTIL.convertToGTLRectInt(query_rect), std::back_inserter(rect_max_rect_pair_list));
    return rect_max_rect_pair_list.empty() ? -1 : rv_layer_data.getNetIdxByMaxRectId(rect_max_rect_pair_list.front().second);
  };

  std::map<int32_t, GTLPolySetInt> layer_merged_polyset_map;
  for (const auto& [routing_layer_idx, rv_layer_data] : layer_data) {
    for (const auto& net_entry : rv_layer_data.nets) {
      const RVRoutingNet& routing_net = net_entry.second;
      layer_merged_polyset_map[routing_layer_idx] += routing_net.polyset;
    }
  }

  std::map<int32_t, RVLayerData> merged_layer_data_map;
  auto append_boundary_edges = [this](RVLayerData& rv_layer_data, GTLHolePolyInt& check_hole_poly, bool is_hole, int32_t polygon_id,
                                      std::vector<std::pair<GTLRectInt, int32_t>>& boundary_rtree_inputs) {
    int32_t coord_size = static_cast<int32_t>(check_hole_poly.size());
    std::vector<PlanarCoord> coord_list;
    coord_list.reserve(coord_size);
    for (auto iter = check_hole_poly.begin(); iter != check_hole_poly.end(); ++iter) {
      coord_list.push_back(DRCUTIL.convertToPlanarCoord(*iter));
    }
    while (coord_list.size() > 1 && coord_list.front() == coord_list.back()) {
      coord_list.pop_back();
    }
    coord_size = static_cast<int32_t>(coord_list.size());
    if (coord_size < 2) {
      return;
    }

    Rotation rotation = DRCUTIL.getRotation(check_hole_poly);
    auto get_boundary_orient = [is_hole, rotation](const PlanarCoord& begin_coord, const PlanarCoord& end_coord) {
      auto rotate_left = [](Orientation orient) {
        switch (orient) {
          case Orientation::kEast:
            return Orientation::kNorth;
          case Orientation::kNorth:
            return Orientation::kWest;
          case Orientation::kWest:
            return Orientation::kSouth;
          case Orientation::kSouth:
            return Orientation::kEast;
          default:
            return Orientation::kNone;
        }
      };
      auto rotate_right = [](Orientation orient) {
        switch (orient) {
          case Orientation::kEast:
            return Orientation::kSouth;
          case Orientation::kSouth:
            return Orientation::kWest;
          case Orientation::kWest:
            return Orientation::kNorth;
          case Orientation::kNorth:
            return Orientation::kEast;
          default:
            return Orientation::kNone;
        }
      };

      Orientation travel_orient = DRCUTIL.getOrientation(begin_coord, end_coord);
      bool metal_on_left = (rotation == Rotation::kCounterclockwise);
      if (is_hole) {
        metal_on_left = !metal_on_left;
      }
      return metal_on_left ? rotate_right(travel_orient) : rotate_left(travel_orient);
    };

    std::vector<bool> convex_corner_list(coord_size, false);
    if (coord_size >= 3) {
      for (int32_t i = 0; i < coord_size; i++) {
        PlanarCoord& pre_coord = coord_list[getIdx(i - 1, coord_size)];
        PlanarCoord& curr_coord = coord_list[i];
        PlanarCoord& post_coord = coord_list[getIdx(i + 1, coord_size)];
        convex_corner_list[i] = is_hole ? DRCUTIL.isConcaveCorner(rotation, pre_coord, curr_coord, post_coord)
                                        : DRCUTIL.isConvexCorner(rotation, pre_coord, curr_coord, post_coord);
      }
    }

    std::vector<int32_t> ring_boundary_ids;
    ring_boundary_ids.reserve(coord_size);
    for (int32_t i = 0; i < coord_size; i++) {
      PlanarCoord& pre_coord = coord_list[getIdx(i - 1, coord_size)];
      PlanarCoord& curr_coord = coord_list[i];
      if (pre_coord == curr_coord) {
        continue;
      }

      BoundaryData boundary_data;
      boundary_data.edge = DRCUTIL.convertToGTLRectInt(DRCUTIL.getRect(pre_coord, curr_coord));
      boundary_data.begin_coord = pre_coord;
      boundary_data.end_coord = curr_coord;
      boundary_data.orient = get_boundary_orient(pre_coord, curr_coord);
      boundary_data.polygon_id = polygon_id;
      boundary_data.edge_length = DRCUTIL.getManhattanDistance(pre_coord, curr_coord);
      boundary_data.isConvex = convex_corner_list[i];
      boundary_data.isHole = is_hole;

      rv_layer_data.boundary_pool.push_back(boundary_data);
      int32_t boundary_id = static_cast<int32_t>(rv_layer_data.boundary_pool.size()) - 1;
      ring_boundary_ids.push_back(boundary_id);
      boundary_rtree_inputs.push_back({boundary_data.edge, boundary_id});
    }

    int32_t ring_size = static_cast<int32_t>(ring_boundary_ids.size());
    if (ring_size < 2) {
      return;
    }
    for (int32_t i = 0; i < ring_size; i++) {
      BoundaryData& boundary_data = rv_layer_data.boundary_pool[ring_boundary_ids[i]];
      boundary_data.prev_boundary_id = ring_boundary_ids[getIdx(i - 1, ring_size)];
      boundary_data.next_boundary_id = ring_boundary_ids[getIdx(i + 1, ring_size)];
    }
  };

  auto build_layer_component_data = [&](RVLayerData& merged_layer_data, const GTLPolySetInt& merged_layer_polyset) {
    merged_layer_data.nets.clear();
    merged_layer_data.polygon_pool.clear();
    merged_layer_data.max_rect_pool.clear();
    merged_layer_data.boundary_pool.clear();

    std::vector<std::pair<GTLRectInt, int32_t>> rect_rtree_inputs;
    std::vector<std::pair<GTLRectInt, int32_t>> boundary_rtree_inputs;
    std::vector<GTLHolePolyInt> gtl_hole_poly_list;
    merged_layer_polyset.get(gtl_hole_poly_list);
    merged_layer_data.polygon_pool.reserve(gtl_hole_poly_list.size());

    for (int32_t component_idx = 0; component_idx < static_cast<int32_t>(gtl_hole_poly_list.size()); ++component_idx) {
      GTLHolePolyInt& gtl_hole_poly = gtl_hole_poly_list[component_idx];
      GTLPolySetInt component_polyset;
      component_polyset += gtl_hole_poly;

      RVRoutingNet& routing_net = merged_layer_data.nets[component_idx];
      routing_net.polyset = component_polyset;
      routing_net.polygon_begin = static_cast<int32_t>(merged_layer_data.polygon_pool.size());
      routing_net.max_rect_begin = static_cast<int32_t>(merged_layer_data.max_rect_pool.size());
      routing_net.boundary_begin = static_cast<int32_t>(merged_layer_data.boundary_pool.size());

      int32_t polygon_id = static_cast<int32_t>(merged_layer_data.polygon_pool.size());
      merged_layer_data.polygon_pool.push_back(
          {component_idx, static_cast<int32_t>(merged_layer_data.max_rect_pool.size()), 0, static_cast<int32_t>(merged_layer_data.boundary_pool.size()), 0});
      PolygonData& polygon_data = merged_layer_data.polygon_pool.back();

      std::vector<GTLRectInt> gtl_rect_list;
      gtl::get_max_rectangles(gtl_rect_list, gtl_hole_poly);
      merged_layer_data.max_rect_pool.reserve(merged_layer_data.max_rect_pool.size() + gtl_rect_list.size());
      for (GTLRectInt& gtl_rect : gtl_rect_list) {
        MaxRectData max_rect_data;
        max_rect_data.rect = gtl_rect;
        max_rect_data.polygon_id = polygon_id;

        merged_layer_data.max_rect_pool.push_back(max_rect_data);
        int32_t max_rect_id = static_cast<int32_t>(merged_layer_data.max_rect_pool.size()) - 1;
        rect_rtree_inputs.push_back({gtl_rect, max_rect_id});
      }
      polygon_data.max_rect_count = static_cast<int32_t>(merged_layer_data.max_rect_pool.size()) - polygon_data.max_rect_begin;

      append_boundary_edges(merged_layer_data, gtl_hole_poly, false, polygon_id, boundary_rtree_inputs);
      for (auto iter = gtl_hole_poly.begin_holes(); iter != gtl_hole_poly.end_holes(); ++iter) {
        GTLPolyInt gtl_poly = *iter;
        GTLHolePolyInt hole_poly;
        hole_poly.set(gtl_poly.begin(), gtl_poly.end());
        append_boundary_edges(merged_layer_data, hole_poly, true, polygon_id, boundary_rtree_inputs);
      }
      polygon_data.boundary_count = static_cast<int32_t>(merged_layer_data.boundary_pool.size()) - polygon_data.boundary_begin;

      routing_net.polygon_count = static_cast<int32_t>(merged_layer_data.polygon_pool.size()) - routing_net.polygon_begin;
      routing_net.max_rect_count = static_cast<int32_t>(merged_layer_data.max_rect_pool.size()) - routing_net.max_rect_begin;
      routing_net.boundary_count = static_cast<int32_t>(merged_layer_data.boundary_pool.size()) - routing_net.boundary_begin;
    }

    merged_layer_data.rect_rtrees = decltype(merged_layer_data.rect_rtrees)(rect_rtree_inputs);
    merged_layer_data.boundary_rtrees = decltype(merged_layer_data.boundary_rtrees)(boundary_rtree_inputs);
  };

  for (const auto& [routing_layer_idx, merged_layer_polyset] : layer_merged_polyset_map) {
    build_layer_component_data(merged_layer_data_map[routing_layer_idx], merged_layer_polyset);
  }

  for (auto& [routing_layer_idx, rv_layer_data] : layer_data) {
    auto merged_layer_it = merged_layer_data_map.find(routing_layer_idx);
    if (merged_layer_it == merged_layer_data_map.end()) {
      continue;
    }
    RVLayerData& merged_layer_data = merged_layer_it->second;
    RoutingLayer& routing_layer = routing_layer_list[routing_layer_idx];
    std::vector<CornerSpacingRule>& corner_spacing_rule_list = routing_layer.get_corner_spacing_rule_list();
    std::vector<Violation> violations;
    std::vector<Corner> check_corners;

    for (auto& corner_spacing_rule : corner_spacing_rule_list) {
      if (corner_spacing_rule.has_convex_corner) {
        // build exempted eol
        std::set<int32_t> exempted_eol_boundary_ids;
        if (corner_spacing_rule.has_except_eol) {
          collect_exempted_eol_edges(rv_layer_data, corner_spacing_rule.except_eol, exempted_eol_boundary_ids);
        }

        // build dubious corner
        check_corners.clear();
        std::set<std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t, int32_t>> seen_corner_keys;
        for (const auto& [net_idx, routing_net] : rv_layer_data.nets) {
          for (const PolygonData& polygon_data : rv_layer_data.getPolygons(routing_net)) {
            int32_t polygon_id = rv_layer_data.getPolygonId(polygon_data);
            std::vector<bool> visited_ring_boundary(static_cast<size_t>(polygon_data.boundary_count), false);
            for (const BoundaryData& seed_boundary : rv_layer_data.getBoundaries(polygon_data)) {
              if (seed_boundary.isHole) {
                continue;
              }
              int32_t seed_boundary_id = rv_layer_data.getBoundaryId(seed_boundary);
              int32_t seed_local_idx = seed_boundary_id - polygon_data.boundary_begin;
              if (visited_ring_boundary[seed_local_idx]) {
                continue;
              }

              std::vector<int32_t> ring_boundary_ids;
              int32_t curr_boundary_id = seed_boundary_id;
              do {
                int32_t local_idx = curr_boundary_id - polygon_data.boundary_begin;
                if (local_idx < 0 || polygon_data.boundary_count <= local_idx || visited_ring_boundary[local_idx]) {
                  break;
                }
                visited_ring_boundary[local_idx] = true;
                ring_boundary_ids.push_back(curr_boundary_id);
                curr_boundary_id = rv_layer_data.getBoundary(curr_boundary_id).next_boundary_id;
              } while (curr_boundary_id != seed_boundary_id);

              int32_t coord_size = static_cast<int32_t>(ring_boundary_ids.size());
              if (coord_size < 3) {
                continue;
              }

              std::vector<bool> convex_corner_list;
              convex_corner_list.reserve(coord_size);
              for (int32_t ring_boundary_id : ring_boundary_ids) {
                convex_corner_list.push_back(rv_layer_data.getBoundary(ring_boundary_id).isConvex);
              }

              for (int32_t i = 0; i < coord_size; i++) {
                if (!convex_corner_list[i]) {
                  continue;
                }

                int32_t boundary_id = ring_boundary_ids[i];
                int32_t next_boundary_id = ring_boundary_ids[(i + 1) % coord_size];
                const BoundaryData& curr_boundary = rv_layer_data.getBoundary(boundary_id);
                const BoundaryData& next_boundary = rv_layer_data.getBoundary(next_boundary_id);

                PlanarCoord corner_point = curr_boundary.end_coord;
                Orientation orient1 = DRCUTIL.getOrientation(corner_point, curr_boundary.begin_coord);
                Orientation orient2 = DRCUTIL.getOrientation(corner_point, next_boundary.end_coord);

                std::vector<std::pair<GTLRectInt, int32_t>> rect_hits;
                rv_layer_data.queryMaxRects(DRCUTIL.convertToGTLRectInt(PlanarRect(corner_point, corner_point)), std::back_inserter(rect_hits));
                for (const auto& [gtl_rect, max_rect_id] : rect_hits) {
                  const MaxRectData& max_rect_data = rv_layer_data.getMaxRect(max_rect_id);
                  if (max_rect_data.polygon_id != polygon_id) {
                    continue;
                  }

                  PlanarRect max_rect = DRCUTIL.convertToPlanarRect(gtl_rect);
                  Orientation rect_orient1 = Orientation::kNone;
                  Orientation rect_orient2 = Orientation::kNone;
                  if (!DRCUTIL.getCornerOrientsInRect(max_rect, corner_point, rect_orient1, rect_orient2)) {
                    continue;
                  }
                  if (!((orient1 == rect_orient1 && orient2 == rect_orient2) || (orient1 == rect_orient2 && orient2 == rect_orient1))) {
                    continue;
                  }

                  int32_t width = max_rect.getWidth();
                  auto corner_key = std::make_tuple(corner_point.get_x(), corner_point.get_y(), width, static_cast<int32_t>(orient1),
                                                    static_cast<int32_t>(orient2), polygon_id);
                  if (!seen_corner_keys.insert(corner_key).second) {
                    continue;
                  }
                  check_corners.push_back({corner_point, width, orient1, orient2, net_idx, polygon_id, boundary_id, next_boundary_id});
                }
              }
            }
          }
        }

        // exempt eol boundary edge corner
        if (!exempted_eol_boundary_ids.empty()) {
          check_corners.erase(std::remove_if(check_corners.begin(), check_corners.end(),
                                             [&](const Corner& corner) {
                                               return DRCUTIL.exist(exempted_eol_boundary_ids, corner.boundary_id)
                                                      || DRCUTIL.exist(exempted_eol_boundary_ids, corner.next_boundary_id);
                                             }),
                              check_corners.end());
        }

        // query corner rect towards outside
        for (auto& check_corner : check_corners) {
          int32_t spacing = corner_spacing_rule.get_width_spacing(check_corner.width);
          if (spacing <= 0) {
            continue;
          }
          const BoundaryData& curr_boundary = rv_layer_data.getBoundary(check_corner.boundary_id);
          const BoundaryData& next_boundary = rv_layer_data.getBoundary(check_corner.next_boundary_id);
          PlanarRect corner_rect = DRCUTIL.getRect(curr_boundary.begin_coord, next_boundary.end_coord);
          PlanarRect check_rect = build_outside_query_rect(check_corner, spacing);

          std::vector<std::pair<GTLRectInt, int32_t>> boundary_hits;
          merged_layer_data.queryBoundaries(DRCUTIL.convertToGTLRectInt(check_rect), std::back_inserter(boundary_hits));

          for (const auto& [gtl_rect, boundary_id] : boundary_hits) {
            const BoundaryData& env_boundary = merged_layer_data.getBoundary(boundary_id);
            if (env_boundary.orient != check_corner.orient1 && env_boundary.orient != check_corner.orient2) {
              continue;
            }
            PlanarRect env_edge_rect = DRCUTIL.convertToPlanarRect(gtl_rect);
            PlanarRect env_prl_rect = env_edge_rect;
            bool has_positive_prl = false;
            {
              const PolygonData& env_polygon = merged_layer_data.getPolygon(env_boundary.polygon_id);
              Segment<PlanarCoord> env_boundary_seg(env_boundary.begin_coord, env_boundary.end_coord);
              std::vector<PlanarRect> env_prl_rect_list;
              for (const MaxRectData& env_max_rect : merged_layer_data.getMaxRects(env_polygon)) {
                PlanarRect candidate_rect = DRCUTIL.convertToPlanarRect(env_max_rect.rect);
                if (DRCUTIL.getTouchedEdgeOrient(candidate_rect, env_boundary_seg) != env_boundary.orient) {
                  continue;
                }
                env_prl_rect_list.push_back(candidate_rect);
                if (DRCUTIL.getParallelLength(corner_rect, candidate_rect) > 0) {
                  has_positive_prl = true;
                }
              }
              if (!env_prl_rect_list.empty()) {
                env_prl_rect = env_prl_rect_list.front();
              } else if (DRCUTIL.getParallelLength(corner_rect, env_edge_rect) > 0) {
                has_positive_prl = true;
              }
            }

            if (!DRCUTIL.isOpenOverlap(check_rect, env_edge_rect) || has_positive_prl) {
              continue;
            }
            PlanarRect violation_rect = DRCUTIL.getSpacingRect(corner_rect, env_edge_rect);
            if (violation_rect.getArea() != 0) {
              std::vector<std::pair<GTLRectInt, int32_t>> overlap_rect_hits;
              merged_layer_data.queryMaxRects(DRCUTIL.convertToGTLRectInt(violation_rect), std::back_inserter(overlap_rect_hits));
              GTLPolySetInt violation_ps;
              violation_ps += DRCUTIL.convertToGTLRectInt(violation_rect);
              for (const auto& [overlap_gtl_rect, overlap_max_rect_id] : overlap_rect_hits) {
                (void) overlap_max_rect_id;
                PlanarRect overlap_rect = DRCUTIL.convertToPlanarRect(overlap_gtl_rect);
                if (DRCUTIL.isOpenOverlap(violation_rect, overlap_rect)) {
                  violation_ps -= overlap_gtl_rect;
                }
              }

              if (gtl::empty(violation_ps)) {
                continue;
              }

              std::vector<GTLHolePolyInt> remain_violation_poly_list;
              violation_ps.get(remain_violation_poly_list);
              if (remain_violation_poly_list.size() != 1) {
                continue;
              }

              GTLRectInt remain_violation_bbox;
              violation_ps.extents(remain_violation_bbox);
              if (!(DRCUTIL.convertToPlanarRect(remain_violation_bbox) == violation_rect)) {
                continue;
              }
            }
            int32_t env_net_idx = queryNetIdxByRect(rv_layer_data, env_edge_rect);

            Violation violation;
            violation.set_violation_type(ViolationType::kCornerSpacing);
            violation.set_required_size(spacing);
            violation.set_is_routing(true);
            violation.set_violation_net_set({check_corner.net_idx, env_net_idx});
            violation.set_layer_idx(routing_layer_idx);
            violation.set_rect(violation_rect);
            violations.push_back(std::move(violation));
          }
        }
      }
    }

    // postprocess, build final violations
    {
      if (violations.size() > 1) {
        std::sort(violations.begin(), violations.end(), [](const Violation& a, const Violation& b) {
          const auto& ra = a.get_rect();
          const auto& rb = b.get_rect();
          if (ra.get_ll_x() != rb.get_ll_x())
            return ra.get_ll_x() < rb.get_ll_x();
          if (ra.get_ur_x() != rb.get_ur_x())
            return ra.get_ur_x() > rb.get_ur_x();
          if (ra.get_ll_y() != rb.get_ll_y())
            return ra.get_ll_y() < rb.get_ll_y();
          return ra.get_ur_y() > rb.get_ur_y();
        });

        std::vector<Violation> results;
        results.reserve(violations.size());

        std::vector<const Violation*> active_set;

        for (const auto& v : violations) {
          bool is_redundant = false;
          const auto& cur_r = v.get_rect();

          active_set.erase(
              std::remove_if(active_set.begin(), active_set.end(), [&](const Violation* p) { return p->get_rect().get_ur_x() < cur_r.get_ll_x(); }),
              active_set.end());

          for (const auto* p_active : active_set) {
            if (DRCUTIL.isInside(p_active->get_rect(), cur_r)) {
              is_redundant = true;
              break;
            }
          }

          if (!is_redundant) {
            results.push_back(v);
            active_set.push_back(&results.back());
          }
        }
        violations = std::move(results);
      }

      rv_cluster.get_violation_list().insert(rv_cluster.get_violation_list().end(), std::make_move_iterator(violations.begin()),
                                             std::make_move_iterator(violations.end()));
    }
  }
}
}  // namespace idrc
