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
#include "DataManager.hpp"
#include "Orientation.hpp"
#include "PlanarRect.hpp"
#include "RVLayerData.hpp"
#include "RoutingLayer.hpp"
#include "RuleValidator.hpp"
#include "Utility.hpp"

namespace idrc {

void RuleValidator::verifyEndOfLineSpacing(RVCluster& rv_cluster)
{
  struct LayerEolRuleProfile
  {
    bool need_cut_shape = false;
    int32_t max_eol_spacing = 0;
    int32_t max_ete_spacing = 0;
    int32_t max_eol_within = 0;
  };

  const auto& layer_data = rv_cluster.get_layer_data();

  // query original net, return -1 if not exist
  auto queryNetIdxByRect = [](const RVLayerData& rv_layer_data, const PlanarRect& query_rect) -> int32_t {
    std::vector<std::pair<GTLRectInt, int32_t>> rect_max_rect_pair_list;
    rv_layer_data.queryMaxRects(DRCUTIL.convertToGTLRectInt(query_rect), std::back_inserter(rect_max_rect_pair_list));
    return rect_max_rect_pair_list.empty() ? -1 : rv_layer_data.getNetIdxByMaxRectId(rect_max_rect_pair_list.front().second);
  };

  auto calcBoundaryMinThickness = [](const RVLayerData& rv_layer_data, int32_t boundary_id) -> int32_t {
    const BoundaryData& boundary = rv_layer_data.getBoundary(boundary_id);
    Segment<PlanarCoord> boundary_seg(boundary.begin_coord, boundary.end_coord);
    Direction boundary_dir = DRCUTIL.getDirection(boundary.begin_coord, boundary.end_coord);

    std::vector<std::pair<GTLRectInt, int32_t>> rect_hits;
    rv_layer_data.queryMaxRects(boundary.edge, std::back_inserter(rect_hits));

    int32_t min_thickness = INT32_MAX;
    for (const auto& [gtl_rect, max_rect_id] : rect_hits) {
      const MaxRectData& max_rect_data = rv_layer_data.getMaxRect(max_rect_id);
      if (max_rect_data.polygon_id != boundary.polygon_id) {
        continue;
      }

      PlanarRect rect = DRCUTIL.convertToPlanarRect(gtl_rect);
      if (DRCUTIL.getTouchedEdgeOrient(rect, boundary_seg) != boundary.orient) {
        continue;
      }

      int32_t thickness = (boundary_dir == Direction::kHorizontal) ? rect.getYSpan() : rect.getXSpan();
      min_thickness = std::min(min_thickness, thickness);
    }

    return min_thickness == INT32_MAX ? 0 : min_thickness;
  };

  auto getBoundaryMinThickness
      = [&calcBoundaryMinThickness](const RVLayerData& rv_layer_data, std::vector<int32_t>& boundary_min_thickness_cache, int32_t boundary_id) -> int32_t {
    if (boundary_id < 0 || boundary_id >= static_cast<int32_t>(boundary_min_thickness_cache.size())) {
      return 0;
    }

    int32_t& cached_thickness = boundary_min_thickness_cache[boundary_id];
    if (cached_thickness == -1) {
      cached_thickness = calcBoundaryMinThickness(rv_layer_data, boundary_id);
    }
    return cached_thickness;
  };

  // total polyset of all net
  std::map<int32_t, GTLPolySetInt> layer_merged_polyset_map;
  for (const auto& [routing_layer_idx, rv_layer_data] : layer_data) {
    for (const auto& net_entry : rv_layer_data.nets) {
      const RVRoutingNet& routing_net = net_entry.second;
      layer_merged_polyset_map[routing_layer_idx] += routing_net.polyset;
    }
  }
  std::map<int32_t, RVLayerData> merged_layer_data_map;
  std::map<int32_t, std::vector<int32_t>> layer_eol_edge_idx;
  std::map<int32_t, std::vector<uint8_t>> layer_is_eol_boundary_map;
  std::map<int32_t, std::vector<int32_t>> layer_boundary_min_thickness_cache_map;

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

  // Build one layer-level RVLayerData from the merged geometry of that routing layer.
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

    for (int32_t net_idx = 0; net_idx < static_cast<int32_t>(gtl_hole_poly_list.size()); ++net_idx) {
      GTLHolePolyInt& gtl_hole_poly = gtl_hole_poly_list[net_idx];
      GTLPolySetInt component_polyset;
      component_polyset += gtl_hole_poly;

      RVRoutingNet& routing_net = merged_layer_data.nets[net_idx];
      routing_net.polyset = component_polyset;
      routing_net.polygon_begin = static_cast<int32_t>(merged_layer_data.polygon_pool.size());
      routing_net.max_rect_begin = static_cast<int32_t>(merged_layer_data.max_rect_pool.size());
      routing_net.boundary_begin = static_cast<int32_t>(merged_layer_data.boundary_pool.size());

      int32_t polygon_id = static_cast<int32_t>(merged_layer_data.polygon_pool.size());
      merged_layer_data.polygon_pool.push_back(
          {net_idx, static_cast<int32_t>(merged_layer_data.max_rect_pool.size()), 0, static_cast<int32_t>(merged_layer_data.boundary_pool.size()), 0});
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

  auto is_eol_boundary = [](const RVLayerData& rv_layer_data, int32_t boundary_id) {
    const BoundaryData& curr_boundary = rv_layer_data.getBoundary(boundary_id);
    const BoundaryData& prev_boundary = rv_layer_data.getPrevBoundary(boundary_id);
    if (!prev_boundary.isConvex || !curr_boundary.isConvex) {
      return false;
    }

    Direction direction = DRCUTIL.getDirection(curr_boundary.begin_coord, curr_boundary.end_coord);
    PlanarRect start_probe;
    PlanarRect end_probe;
    if (direction == Direction::kHorizontal) {
      int32_t x_min = std::min(curr_boundary.begin_coord.get_x(), curr_boundary.end_coord.get_x());
      int32_t x_max = std::max(curr_boundary.begin_coord.get_x(), curr_boundary.end_coord.get_x());
      int32_t y = curr_boundary.begin_coord.get_y();
      start_probe = PlanarRect(x_min - 2, y, x_min - 1, y);
      end_probe = PlanarRect(x_max + 1, y, x_max + 2, y);
    } else {
      int32_t y_min = std::min(curr_boundary.begin_coord.get_y(), curr_boundary.end_coord.get_y());
      int32_t y_max = std::max(curr_boundary.begin_coord.get_y(), curr_boundary.end_coord.get_y());
      int32_t x = curr_boundary.begin_coord.get_x();
      start_probe = PlanarRect(x, y_min - 2, x, y_min - 1);
      end_probe = PlanarRect(x, y_max + 1, x, y_max + 2);
    }

    std::vector<std::pair<GTLRectInt, int32_t>> overlap_list;
    rv_layer_data.queryMaxRects(DRCUTIL.convertToGTLRectInt(start_probe), std::back_inserter(overlap_list));
    if (!overlap_list.empty()) {
      return false;
    }
    rv_layer_data.queryMaxRects(DRCUTIL.convertToGTLRectInt(end_probe), std::back_inserter(overlap_list));
    return overlap_list.empty();
  };

  auto collect_eol_edges = [&](const RVLayerData& merged_layer_data, std::vector<int32_t>& eol_edge_idx) {
    for (const auto& net_entry : merged_layer_data.nets) {
      const RVRoutingNet& routing_net = net_entry.second;
      for (const PolygonData& polygon_data : merged_layer_data.getPolygons(routing_net)) {
        std::span<const BoundaryData> polygon_boundaries = merged_layer_data.getBoundaries(polygon_data);
        if (polygon_boundaries.empty()) {
          continue;
        }

        std::vector<bool> visited_ring_boundary(static_cast<size_t>(polygon_data.boundary_count), false);
        for (const BoundaryData& seed_boundary : polygon_boundaries) {
          int32_t seed_boundary_id = merged_layer_data.getBoundaryId(seed_boundary);
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

            if (is_eol_boundary(merged_layer_data, curr_boundary_id)) {
              eol_edge_idx.push_back(curr_boundary_id);
            }
            curr_boundary_id = merged_layer_data.getBoundary(curr_boundary_id).next_boundary_id;
          } while (curr_boundary_id != seed_boundary_id);
        }
      }
    }
  };

  for (const auto& [routing_layer_idx, merged_layer_polyset] : layer_merged_polyset_map) {
    RVLayerData& merged_layer_data = merged_layer_data_map[routing_layer_idx];
    build_layer_component_data(merged_layer_data, merged_layer_polyset);
    std::vector<int32_t>& eol_edge_idx = layer_eol_edge_idx[routing_layer_idx];
    collect_eol_edges(merged_layer_data, eol_edge_idx);

    std::vector<uint8_t>& is_eol_boundary = layer_is_eol_boundary_map[routing_layer_idx];
    is_eol_boundary.assign(merged_layer_data.boundary_pool.size(), 0);
    for (int32_t boundary_id : eol_edge_idx) {
      if (boundary_id < 0 || boundary_id >= static_cast<int32_t>(is_eol_boundary.size())) {
        continue;
      }
      is_eol_boundary[boundary_id] = 1;
    }

    layer_boundary_min_thickness_cache_map[routing_layer_idx].assign(merged_layer_data.boundary_pool.size(), -1);
  }

  // env
  std::map<int32_t, bgi::rtree<GTLRectInt, bgi::quadratic<16>>> obs_bg_rtree_map;
  std::map<int32_t, GTLPolySetInt> layer_obs;
  for (DRCShape* drc_shape : rv_cluster.get_drc_env_shape_list()) {
    if (drc_shape->get_is_routing()) {
      layer_obs[drc_shape->get_layer_idx()] += DRCUTIL.convertToGTLRectInt(drc_shape->get_rect());
    }
  }
  for (auto& [routing_layer_idx, obs_gtl_poly_set] : layer_obs) {
    std::vector<GTLRectInt> gtl_rects;
    gtl::get_max_rectangles(gtl_rects, obs_gtl_poly_set);
    for (auto& gtl_rect : gtl_rects) {
      obs_bg_rtree_map[routing_layer_idx].insert(gtl_rect);
    }
  }

  //  rules
  std::vector<RoutingLayer>& routing_layer_list = DRCDM.getDatabase().get_routing_layer_list();
  const std::map<int32_t, std::vector<int32_t>>& routing_to_adjacent_cut_map = DRCDM.getDatabase().get_routing_to_adjacent_cut_map();
  std::map<int32_t, LayerEolRuleProfile> layer_rule_profile_map;
  std::map<PlanarRect, std::vector<Violation>, CmpPlanarRectByXASC> edge_violation_map;
  std::map<int32_t, int32_t> layer_rule_max_width;
  for (const auto& layer_eol_entry : layer_eol_edge_idx) {
    int32_t routing_layer_idx = layer_eol_entry.first;
    const auto& end_of_line_spacing_rule_list = routing_layer_list[routing_layer_idx].get_end_of_line_spacing_rule_list();
    LayerEolRuleProfile& layer_rule_profile = layer_rule_profile_map[routing_layer_idx];
    int32_t& layer_max_width = layer_rule_max_width[routing_layer_idx];
    for (const EndOfLineSpacingRule& eol_rule : end_of_line_spacing_rule_list) {
      if (eol_rule.has_enclose_cut) {
        layer_rule_profile.need_cut_shape = true;
      }
      layer_max_width = std::max(layer_max_width, eol_rule.eol_width);
      layer_rule_profile.max_eol_within = std::max(layer_rule_profile.max_eol_within, eol_rule.eol_within);
      layer_rule_profile.max_eol_spacing = std::max(layer_rule_profile.max_eol_spacing, eol_rule.eol_spacing);
      if (eol_rule.has_ete) {
        layer_rule_profile.max_ete_spacing = std::max(layer_rule_profile.max_ete_spacing, eol_rule.ete_spacing);
      }
    }
  }

  // check each eol edge
  for (const auto& [routing_layer_idx, eol_edge_idx] : layer_eol_edge_idx) {
    const RVLayerData& rv_layer_data = layer_data.at(routing_layer_idx);
    const RVLayerData& merged_layer_data = merged_layer_data_map.at(routing_layer_idx);
    const std::vector<uint8_t>& is_eol_boundary = layer_is_eol_boundary_map.at(routing_layer_idx);
    std::vector<int32_t>& boundary_min_thickness_cache = layer_boundary_min_thickness_cache_map.at(routing_layer_idx);
    const auto& end_of_line_spacing_rule_list = routing_layer_list[routing_layer_idx].get_end_of_line_spacing_rule_list();
    const LayerEolRuleProfile& layer_rule_profile = layer_rule_profile_map.at(routing_layer_idx);
    const int32_t layer_max_width = layer_rule_max_width.at(routing_layer_idx);
    for (int32_t eol_idx : eol_edge_idx) {
      const auto& eol_boundary = merged_layer_data.getBoundary(eol_idx);
      const auto& pre_boundary = merged_layer_data.getPrevBoundary(eol_idx);
      const auto& post_boundary = merged_layer_data.getNextBoundary(eol_idx);
      if (eol_boundary.edge_length >= layer_max_width) {
        continue;
      }
      PlanarRect eol_edge_rect = DRCUTIL.convertToPlanarRect(eol_boundary.edge);

      Direction direction = DRCUTIL.getDirection(eol_boundary.begin_coord, eol_boundary.end_coord);
      PlanarRect eol_rect = DRCUTIL.getBoundingBox({pre_boundary.begin_coord, pre_boundary.end_coord, post_boundary.begin_coord, post_boundary.end_coord});
      int32_t checking_net = -2;
      auto getCheckingNet = [&]() -> int32_t {
        if (checking_net == -2) {
          checking_net = queryNetIdxByRect(rv_layer_data, eol_edge_rect);
        }
        return checking_net;
      };

      // to be checked spacing rects
      std::vector<std::pair<GTLRectInt, int32_t>> env_checking_poly_list;

      // enclosed cut rects
      std::vector<CutData> env_cut_list;
      {
        if (layer_rule_profile.need_cut_shape) {
          auto cut_layer_it = routing_to_adjacent_cut_map.find(routing_layer_idx);
          if (cut_layer_it != routing_to_adjacent_cut_map.end() && !cut_layer_it->second.empty()) {
            int32_t cut_layer_idx = *std::min_element(cut_layer_it->second.begin(), cut_layer_it->second.end());
            auto cut_layer_data_it = layer_data.find(cut_layer_idx);
            if (cut_layer_data_it != layer_data.end()) {
              cut_layer_data_it->second.queryCuts(DRCUTIL.convertToGTLRectInt(eol_rect), std::back_inserter(env_cut_list));
            }
          }
        }

        PlanarRect max_check_rect = eol_edge_rect;
        max_check_rect = DRCUTIL.getEnlargedPartRect(max_check_rect, eol_boundary.orient,
                                                     std::max(layer_rule_profile.max_eol_spacing, layer_rule_profile.max_ete_spacing));

        if (direction == Direction::kHorizontal) {
          max_check_rect = DRCUTIL.getEnlargedRect(max_check_rect, layer_rule_profile.max_eol_within, 0);
        } else {
          max_check_rect = DRCUTIL.getEnlargedRect(max_check_rect, 0, layer_rule_profile.max_eol_within);
        }
        Orientation oppo_orient = DRCUTIL.getOppositeOrientation(eol_boundary.orient);
        std::vector<std::pair<GTLRectInt, int32_t>> env_rects;
        merged_layer_data.queryBoundaries(DRCUTIL.convertToGTLRectInt(max_check_rect), std::back_inserter(env_rects));
        std::vector<std::pair<GTLRectInt, int32_t>> env_max_rects;
        merged_layer_data.queryMaxRects(DRCUTIL.convertToGTLRectInt(max_check_rect), std::back_inserter(env_max_rects));

        std::vector<std::pair<GTLRectInt, int32_t>> temp_checking_poly_list;
        for (auto& [gtl_rect, idx] : env_max_rects) {
          PlanarRect rect = DRCUTIL.convertToPlanarRect(gtl_rect);
          if (DRCUTIL.isOpenOverlap(rect, max_check_rect)) {
            PlanarRect rect_edge = DRCUTIL.getRect(rect.getOrientEdge(oppo_orient));
            temp_checking_poly_list.push_back({DRCUTIL.convertToGTLRectInt(rect_edge), -1});
          }
        }

        for (auto& [gtl_rect, idx] : env_rects) {
          PlanarRect rect = DRCUTIL.convertToPlanarRect(gtl_rect);
          if (DRCUTIL.isOpenOverlap(rect, max_check_rect)) {
            if (eol_boundary.orient == DRCUTIL.getOppositeOrientation(merged_layer_data.getBoundary(idx).orient)) {
              env_checking_poly_list.push_back({gtl_rect, idx});
            }
          }
        }

        env_checking_poly_list.reserve(env_checking_poly_list.size() + temp_checking_poly_list.size());
        std::vector<PlanarRect> env_checking_rect_list;
        env_checking_rect_list.reserve(env_checking_poly_list.size() + temp_checking_poly_list.size());
        for (const auto& env_poly : env_checking_poly_list) {
          env_checking_rect_list.emplace_back(DRCUTIL.convertToPlanarRect(env_poly.first));
        }
        for (const auto& temp_poly : temp_checking_poly_list) {
          PlanarRect temp_rect = DRCUTIL.convertToPlanarRect(temp_poly.first);
          bool has_overlap = false;
          for (const auto& env_rect : env_checking_rect_list) {
            if (DRCUTIL.isClosedOverlap(temp_rect, env_rect)) {
              has_overlap = true;
              break;
            }
          }
          if (!has_overlap) {
            env_checking_poly_list.push_back(temp_poly);
            env_checking_rect_list.emplace_back(temp_rect);
          }
        }
      }

      if (env_checking_poly_list.empty()) {
        continue;
      }

      for (const EndOfLineSpacingRule& eol_rule : end_of_line_spacing_rule_list) {
        if (eol_boundary.edge_length >= eol_rule.eol_width) {
          continue;
        }
        if (eol_rule.has_enclose_cut && env_cut_list.empty()) {
          continue;
        }
        bool pre_length_ok = pre_boundary.edge_length >= eol_rule.min_length;
        bool post_length_ok = post_boundary.edge_length >= eol_rule.min_length;
        if (eol_rule.has_par && !eol_rule.has_same_metal) {
          if (eol_rule.has_two_edges) {
            if (!pre_length_ok || !post_length_ok) {
              continue;
            }
          } else if (!pre_length_ok && !post_length_ok) {
            continue;
          }
        }
        if (eol_rule.has_same_metal && getCheckingNet() == -1) {
          continue;
        }

        for (auto& [env_gtl_rect, boundary_idx] : env_checking_poly_list) {
          PlanarRect env_rect = DRCUTIL.convertToPlanarRect(env_gtl_rect);
          if (DRCUTIL.isClosedOverlap(env_rect, eol_rect)) {
            continue;
          }

          // 0: normal eol spacing, 1: ete spacing， 2: par spacing, 3: same metal spacing
          int violation_type = 0;
          bool is_ete = false;
          if (boundary_idx >= 0) {
            const BoundaryData& env_boundary = merged_layer_data.getBoundary(boundary_idx);
            if (boundary_idx >= 0 && boundary_idx < static_cast<int32_t>(is_eol_boundary.size()) && is_eol_boundary[boundary_idx]) {
              if (env_boundary.edge_length < eol_rule.eol_width && eol_rule.has_ete) {
                is_ete = true;
              }
            }
          }
          violation_type = is_ete ? 1 : 0;
          PlanarRect check_rect = DRCUTIL.getEnlargedPartRect(eol_edge_rect, eol_boundary.orient, is_ete ? eol_rule.ete_spacing : eol_rule.eol_spacing);
          if (direction == Direction::kHorizontal) {
            check_rect = DRCUTIL.getEnlargedRect(check_rect, eol_rule.eol_within, 0);
          } else {
            check_rect = DRCUTIL.getEnlargedRect(check_rect, 0, eol_rule.eol_within);
          }
          if (!DRCUTIL.isOpenOverlap(check_rect, env_rect) && !eol_rule.has_same_metal) {
            continue;
          }

          bool pre_par = false, post_par = false;
          int32_t pre_par_idx = -1, post_par_idx = -1;
          if (eol_rule.has_par) {
            int32_t par_spacing = 0;
            if (eol_rule.has_subtrace_eol_width) {
              int32_t width = std::min(getBoundaryMinThickness(merged_layer_data, boundary_min_thickness_cache, eol_boundary.prev_boundary_id),
                                       getBoundaryMinThickness(merged_layer_data, boundary_min_thickness_cache, eol_boundary.next_boundary_id));
              par_spacing = std::max(par_spacing, eol_rule.par_spacing - width);
            } else {
              par_spacing = std::max(par_spacing, eol_rule.par_spacing);
            }

            PlanarRect pre_rect = DRCUTIL.getRect({pre_boundary.end_coord, pre_boundary.end_coord});
            PlanarRect post_rect = DRCUTIL.getRect({post_boundary.begin_coord, post_boundary.begin_coord});

            switch (eol_boundary.orient) {
              case Orientation::kNorth:
                pre_rect = DRCUTIL.getEnlargedRect(pre_rect, 0, eol_rule.par_within, par_spacing, eol_rule.eol_within);
                post_rect = DRCUTIL.getEnlargedRect(post_rect, par_spacing, eol_rule.par_within, 0, eol_rule.eol_within);
                break;
              case Orientation::kSouth:
                pre_rect = DRCUTIL.getEnlargedRect(pre_rect, par_spacing, eol_rule.eol_within, 0, eol_rule.par_within);
                post_rect = DRCUTIL.getEnlargedRect(post_rect, 0, eol_rule.eol_within, par_spacing, eol_rule.par_within);
                break;
              case Orientation::kWest:
                pre_rect = DRCUTIL.getEnlargedRect(pre_rect, eol_rule.eol_within, 0, eol_rule.par_within, par_spacing);
                post_rect = DRCUTIL.getEnlargedRect(post_rect, eol_rule.eol_within, par_spacing, eol_rule.par_within, 0);
                break;
              case Orientation::kEast:
                pre_rect = DRCUTIL.getEnlargedRect(pre_rect, eol_rule.par_within, par_spacing, eol_rule.eol_within, 0);
                post_rect = DRCUTIL.getEnlargedRect(post_rect, eol_rule.par_within, 0, eol_rule.eol_within, par_spacing);
                break;
              default:
                DRCLOG.error(Loc::current(), "The orientation is error!");
            }

            // par left and right neighbors
            std::vector<std::pair<GTLRectInt, int32_t>> env_routing_poly_list;
            merged_layer_data.queryBoundaries(DRCUTIL.convertToGTLRectInt(pre_rect), std::back_inserter(env_routing_poly_list));
            merged_layer_data.queryBoundaries(DRCUTIL.convertToGTLRectInt(post_rect), std::back_inserter(env_routing_poly_list));

            for (auto& [par_gtl_rect, par_id] : env_routing_poly_list) {
              PlanarRect par_rect = DRCUTIL.convertToPlanarRect(par_gtl_rect);
              if (DRCUTIL.isClosedOverlap(par_rect, eol_rect)) {
                continue;
              }
              const BoundaryData& par_boundary = merged_layer_data.getBoundary(par_id);
              if (DRCUTIL.isOpenOverlap(pre_rect, par_rect)) {
                if (pre_boundary.orient == DRCUTIL.getOppositeOrientation(par_boundary.orient)) {
                  pre_par_idx = par_id;
                  pre_par = true;
                }
              }
              if (DRCUTIL.isOpenOverlap(post_rect, par_rect)) {
                if (post_boundary.orient == DRCUTIL.getOppositeOrientation(par_boundary.orient)) {
                  post_par_idx = par_id;
                  post_par = true;
                }
              }
            }
            if (!eol_rule.has_same_metal) {
              pre_par = pre_length_ok ? pre_par : false;
              post_par = post_length_ok ? post_par : false;
            }
            if (eol_rule.has_two_edges) {
              if (!(pre_par && post_par)) {
                continue;
              }
            } else {
              if (!(pre_par || post_par)) {
                continue;
              }
            }
            violation_type = violation_type == 0 ? 2 : violation_type;

            if (eol_rule.has_same_metal) {
              if (pre_par_idx == -1 || post_par_idx == -1) {
                continue;
              }
              auto& pre_polygon = merged_layer_data.getPolygon(merged_layer_data.getBoundary(pre_par_idx).polygon_id);
              auto& post_polygon = merged_layer_data.getPolygon(merged_layer_data.getBoundary(post_par_idx).polygon_id);

              if (pre_polygon.net_id != post_polygon.net_id) {
                continue;
              }
              GTLRectInt pre_edge_rect = merged_layer_data.getBoundary(pre_par_idx).edge;
              GTLRectInt post_edge_rect = merged_layer_data.getBoundary(post_par_idx).edge;
              int32_t pre_prl_length = DRCUTIL.getParallelLength(pre_rect, DRCUTIL.convertToPlanarRect(pre_edge_rect));
              int32_t post_prl_length = DRCUTIL.getParallelLength(post_rect, DRCUTIL.convertToPlanarRect(post_edge_rect));
              if (pre_prl_length >= eol_rule.par_within || post_prl_length >= eol_rule.par_within || pre_prl_length <= 0 || post_prl_length <= 0) {
                continue;
              }
              violation_type = is_ete ? 1 : 3;
            }
          }

          if (eol_rule.has_enclose_cut) {
            bool is_pass_cut = false;
            for (const CutData& cut_data : env_cut_list) {
              PlanarRect cut_rect = DRCUTIL.convertToPlanarRect(cut_data.rect);
              if ((DRCUTIL.getEuclideanDistance(cut_rect, eol_edge_rect) < eol_rule.enclosed_dist)
                  && (DRCUTIL.getEuclideanDistance(cut_rect, env_rect)) < eol_rule.cut_to_metal_spacing) {
                is_pass_cut = true;
              }
            }
            if (!is_pass_cut) {
              continue;
            }
          }

          auto check_inside = [&](const auto& target_rect) {
            auto query_box = DRCUTIL.convertToGTLRectInt(target_rect);
            auto& rtree = obs_bg_rtree_map[routing_layer_idx];

            for (auto iter = rtree.qbegin(bgi::intersects(query_box)); iter != rtree.qend(); ++iter) {
              auto iter_rect = *iter;
              if (DRCUTIL.isInside(DRCUTIL.convertToPlanarRect(iter_rect), target_rect)) {
                return true;
              }
            }
            return false;
          };

          if (check_inside(eol_edge_rect) && check_inside(env_rect)) {
            continue;
          }
          PlanarRect spacing_rect = DRCUTIL.getSpacingRect(eol_rect, env_rect);
          std::set<int32_t> net_list{getCheckingNet(), queryNetIdxByRect(rv_layer_data, env_rect)};
          int32_t req_size = violation_type == 1 ? eol_rule.ete_spacing : eol_rule.eol_spacing;
          if (DRCUTIL.getEuclideanDistance(eol_edge_rect, env_rect) >= req_size) {
            continue;
          }

          Violation violation;
          violation.set_violation_type(ViolationType::kEndOfLineSpacing);
          violation.set_required_size(req_size);
          violation.set_is_routing(true);
          violation.set_violation_net_set(net_list);
          violation.set_layer_idx(routing_layer_idx);
          violation.set_rect(spacing_rect);
          edge_violation_map[eol_rect].push_back(violation);
        }
      }
    }
  }

  // sort and unique violations
  std::set<Violation, CmpViolation> invalid_violation_set;
  std::set<Violation, CmpViolation> unique_violation;
  for (auto& [edge, violation_list] : edge_violation_map) {
    for (auto& violation : violation_list) {
      unique_violation.insert(violation);
    }
  }
  // 输出required size最大的violation
  for (auto violation : unique_violation) {
    for (auto other_violation : unique_violation) {
      if (violation.get_layer_idx() != other_violation.get_layer_idx()) {
        continue;
      }
      if (violation.get_rect() == other_violation.get_rect()) {
        if (violation.get_required_size() < other_violation.get_required_size()) {
          invalid_violation_set.insert(violation);
        }
      } else if (DRCUTIL.isInside(other_violation.get_rect(), violation.get_rect())) {
        invalid_violation_set.insert(violation);
      }
    }
  }
  for (auto violation : unique_violation) {
    if (!DRCUTIL.exist(invalid_violation_set, violation)) {
      rv_cluster.get_violation_list().push_back(violation);
    }
  }
}

}  // namespace idrc
