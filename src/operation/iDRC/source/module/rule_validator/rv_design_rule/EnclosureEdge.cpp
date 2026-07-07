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

namespace {

struct ConvexCandidate
{
  int32_t curr_boundary_id = -1;
  int32_t adj_boundary_id = -1;
  int32_t other_length = 0;
};

bool isValidRect(const PlanarRect& rect)
{
  return rect.get_ll_x() < rect.get_ur_x() && rect.get_ll_y() < rect.get_ur_y();
}

std::vector<EnclosureEdgeRule> getSortedEnclosureEdgeRules(const std::vector<EnclosureEdgeRule>& rule_list)
{
  std::vector<EnclosureEdgeRule> sorted_rules = rule_list;
  std::sort(sorted_rules.begin(), sorted_rules.end(), [](const EnclosureEdgeRule& a, const EnclosureEdgeRule& b) {
    if (a.has_convexcorners != b.has_convexcorners) {
      return a.has_convexcorners;
    }
    return a.min_width > b.min_width;
  });
  return sorted_rules;
}

std::map<Orientation, PlanarRect> buildOrientExtensionRects(const PlanarRect& rect, int32_t par_within)
{
  std::map<Orientation, PlanarRect> orient_extension_rects;
  orient_extension_rects[Orientation::kSouth] = DRCUTIL.getEnlargedRect(rect.get_ll(), 0, par_within, rect.getXSpan(), 0);
  orient_extension_rects[Orientation::kNorth] = DRCUTIL.getEnlargedRect(rect.get_ur(), rect.getXSpan(), 0, 0, par_within);
  orient_extension_rects[Orientation::kWest] = DRCUTIL.getEnlargedRect(rect.get_ll(), par_within, 0, 0, rect.getYSpan());
  orient_extension_rects[Orientation::kEast] = DRCUTIL.getEnlargedRect(rect.get_ur(), 0, rect.getYSpan(), par_within, 0);
  return orient_extension_rects;
}

}  // namespace

void RuleValidator::verifyEnclosureEdge(RVCluster& rv_cluster)
{
  const auto orientations = {Orientation::kEast, Orientation::kSouth, Orientation::kWest, Orientation::kNorth};
  std::vector<CutLayer>& cut_layer_list = DRCDM.getDatabase().get_cut_layer_list();
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = DRCDM.getDatabase().get_cut_to_adjacent_routing_map();
  const auto& layer_data = rv_cluster.get_layer_data();

  std::map<int32_t, std::vector<EnclosureEdgeRule>> cut_layer_sorted_rule_map;
  std::map<int32_t, std::vector<int32_t>> layer_width_ranges;
  std::map<int32_t, bgi::rtree<std::pair<GTLRectInt, int32_t>, bgi::quadratic<16>>> routing_env_rtree_map;
  std::map<std::pair<int32_t, int32_t>, bgi::rtree<std::pair<GTLRectInt, int32_t>, bgi::quadratic<16>>> layer_width_rtrees;
  std::map<int32_t, std::map<int32_t, std::map<PlanarRect, std::vector<ConvexCandidate>, CmpPlanarRectByXASC>>> layer_polygon_convex_candidates;

  // preprocess: sort rules and collect width ranges by routing layer.
  for (const auto& [cut_layer_idx, cut_layer_data] : layer_data) {
    if (cut_layer_data.cut_pool.empty()) {
      continue;
    }
    auto routing_layer_it = cut_to_adjacent_routing_map.find(cut_layer_idx);
    if (routing_layer_it == cut_to_adjacent_routing_map.end() || routing_layer_it->second.size() < 2) {
      continue;
    }

    const std::vector<int32_t>& routing_layer_idx_list = routing_layer_it->second;
    int32_t above_routing_layer_idx = *std::max_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
    int32_t below_routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());

    std::vector<EnclosureEdgeRule> sorted_rules = getSortedEnclosureEdgeRules(cut_layer_list[cut_layer_idx].get_enclosure_edge_rule_list());
    cut_layer_sorted_rule_map[cut_layer_idx] = sorted_rules;
    for (const EnclosureEdgeRule& rule : sorted_rules) {
      if (rule.has_convexcorners || rule.min_width <= 0) {
        continue;
      }
      if (!rule.has_below) {
        layer_width_ranges[above_routing_layer_idx].push_back(rule.min_width);
      }
      if (!rule.has_above) {
        layer_width_ranges[below_routing_layer_idx].push_back(rule.min_width);
      }
    }
  }
  for (auto& [layer_idx, width_ranges] : layer_width_ranges) {
    (void) layer_idx;
    width_ranges.erase(std::remove_if(width_ranges.begin(), width_ranges.end(), [](int32_t width) { return width <= 0; }), width_ranges.end());
    std::sort(width_ranges.begin(), width_ranges.end(), std::greater<int32_t>());
    width_ranges.erase(std::unique(width_ranges.begin(), width_ranges.end()), width_ranges.end());
  }

  // preprocess: build env-only maxRect rtree for convex env filtering.
  {
    std::map<int32_t, std::map<int32_t, GTLPolySetInt>> routing_net_env_poly_set_map;
    for (DRCShape* drc_shape : rv_cluster.get_drc_env_shape_list()) {
      if (!drc_shape->get_is_routing()) {
        continue;
      }
      PlanarRect rect = drc_shape->get_rect();
      routing_net_env_poly_set_map[drc_shape->get_layer_idx()][drc_shape->get_net_idx()] += DRCUTIL.convertToGTLRectInt(rect);
    }
    for (auto& [routing_layer_idx, net_gtl_poly_set_map] : routing_net_env_poly_set_map) {
      for (auto& [net_idx, gtl_poly_set] : net_gtl_poly_set_map) {
        std::vector<GTLRectInt> gtl_rect_list;
        gtl::get_max_rectangles(gtl_rect_list, gtl_poly_set);
        for (const GTLRectInt& gtl_rect : gtl_rect_list) {
          if (!isValidRect(DRCUTIL.convertToPlanarRect(gtl_rect))) {
            continue;
          }
          routing_env_rtree_map[routing_layer_idx].insert({gtl_rect, net_idx});
        }
      }
    }
  }

  // preprocess: build polygon convex candidates once, reuse during convex checks.
  for (const auto& [routing_layer_idx, rv_layer_data] : layer_data) {
    for (const auto& [net_idx, routing_net] : rv_layer_data.nets) {
      (void) net_idx;
      for (const PolygonData& polygon_data : rv_layer_data.getPolygons(routing_net)) {
        int32_t polygon_id = rv_layer_data.getPolygonId(polygon_data);
        std::vector<int32_t> outer_boundary_ids;
        for (const BoundaryData& boundary_data : rv_layer_data.getBoundaries(polygon_data)) {
          if (boundary_data.isHole) {
            break;
          }
          outer_boundary_ids.push_back(rv_layer_data.getBoundaryId(boundary_data));
        }

        int32_t coord_size = static_cast<int32_t>(outer_boundary_ids.size());
        if (coord_size < 6) {
          continue;
        }

        std::vector<PlanarCoord> coord_list;
        std::vector<bool> convex_corner_list;
        coord_list.reserve(coord_size);
        convex_corner_list.reserve(coord_size);
        for (int32_t boundary_id : outer_boundary_ids) {
          const BoundaryData& boundary_data = rv_layer_data.getBoundary(boundary_id);
          coord_list.push_back(boundary_data.end_coord);
          convex_corner_list.push_back(boundary_data.isConvex);
        }

        auto& polygon_convex_candidates = layer_polygon_convex_candidates[routing_layer_idx][polygon_id];
        for (int32_t i = 0; i < coord_size; i++) {
          if (!(convex_corner_list[getIdx(i - 1, coord_size)] && convex_corner_list[getIdx(i, coord_size)] && convex_corner_list[getIdx(i + 1, coord_size)])) {
            continue;
          }

          PlanarCoord& pre_coord = coord_list[getIdx(i - 1, coord_size)];
          PlanarCoord& curr_coord = coord_list[i];
          PlanarCoord& post_coord = coord_list[getIdx(i + 1, coord_size)];
          PlanarRect enc_rect = DRCUTIL.getRect(pre_coord, post_coord);
          if (!isValidRect(enc_rect)) {
            continue;
          }

          polygon_convex_candidates[enc_rect].push_back(
              {outer_boundary_ids[getIdx(i, coord_size)], outer_boundary_ids[getIdx(i - 1, coord_size)], DRCUTIL.getManhattanDistance(curr_coord, post_coord)});
          polygon_convex_candidates[enc_rect].push_back({outer_boundary_ids[getIdx(i + 1, coord_size)], outer_boundary_ids[getIdx(i + 2, coord_size)],
                                                         DRCUTIL.getManhattanDistance(curr_coord, pre_coord)});
        }
      }
    }
  }

  // preprocess: build (layer, width) -> maxRect rtree from LayerData polyset.
  for (const auto& [layer_idx, width_ranges] : layer_width_ranges) {
    if (width_ranges.empty()) {
      continue;
    }
    auto layer_it = layer_data.find(layer_idx);
    if (layer_it == layer_data.end()) {
      continue;
    }

    const RVLayerData& rv_layer_data = layer_it->second;
    int32_t min_width = width_ranges.back();
    for (const auto& [net_idx, routing_net] : rv_layer_data.nets) {
      GTLPolySetInt filtered_polyset;
      for (const MaxRectData& max_rect_data : rv_layer_data.getMaxRects(routing_net)) {
        PlanarRect max_rect = DRCUTIL.convertToPlanarRect(max_rect_data.rect);
        if (!isValidRect(max_rect) || max_rect.getXSpan() < min_width || max_rect.getYSpan() < min_width) {
          continue;
        }
        filtered_polyset += max_rect_data.rect;
      }

      std::map<int32_t, GTLPolySetInt> width_polyset_map;
      GTLPolySetInt curr_width_polyset = filtered_polyset;
      for (auto width_it = width_ranges.rbegin(); width_it != width_ranges.rend(); width_it++) {
        int32_t width = *width_it;
        GTLPolySetInt next_width_polyset = curr_width_polyset;
        int32_t resize_delta = width / 2;
        gtl::resize(next_width_polyset, -resize_delta);
        gtl::resize(next_width_polyset, resize_delta);
        width_polyset_map[width] = next_width_polyset;
        curr_width_polyset = next_width_polyset;
      }

      int32_t last_width = std::numeric_limits<int32_t>::max();
      for (int32_t width : width_ranges) {
        GTLPolySetInt remain_polyset = width_polyset_map[width];

        std::vector<GTLRectInt> gtl_rect_list;
        gtl::get_max_rectangles(gtl_rect_list, remain_polyset);

        std::vector<GTLRectInt> dut_rect_list;
        std::vector<GTLRectInt> checked_rect_list;
        for (const GTLRectInt& gtl_rect : gtl_rect_list) {
          PlanarRect rect = DRCUTIL.convertToPlanarRect(gtl_rect);
          if (!isValidRect(rect)) {
            continue;
          }
          if (rect.getWidth() >= last_width) {
            checked_rect_list.push_back(gtl_rect);
          } else {
            dut_rect_list.push_back(gtl_rect);
          }
        }

        GTLPolySetInt checked_poly_set;
        checked_poly_set.insert(checked_rect_list.begin(), checked_rect_list.end());
        auto& width_rtree = layer_width_rtrees[{layer_idx, width}];
        for (const GTLRectInt& dut_rect : dut_rect_list) {
          GTLPolySetInt dut_poly_set;
          dut_poly_set = dut_poly_set + dut_rect - checked_poly_set;

          std::vector<GTLPolyInt> dut_poly_list;
          dut_poly_set.get(dut_poly_list);
          for (const GTLPolyInt& dut_poly : dut_poly_list) {
            GTLRectInt insert_rect;
            bool has_insert_rect = false;
            if (dut_poly.size() > 4) {
              gtl::extents(insert_rect, dut_poly);
              has_insert_rect = true;
            } else {
              std::vector<GTLRectInt> temp_rect_list;
              gtl::get_max_rectangles(temp_rect_list, dut_poly);
              if (!temp_rect_list.empty()) {
                insert_rect = temp_rect_list[0];
                has_insert_rect = true;
              }
            }
            if (!has_insert_rect || !isValidRect(DRCUTIL.convertToPlanarRect(insert_rect))) {
              continue;
            }
            width_rtree.insert({insert_rect, net_idx});
          }
        }

        last_width = width;
      }
    }
  }

  // check each cut.
  for (const auto& [cut_layer_idx, cut_layer_data] : layer_data) {
    if (cut_layer_data.cut_pool.empty()) {
      continue;
    }
    auto routing_layer_it = cut_to_adjacent_routing_map.find(cut_layer_idx);
    if (routing_layer_it == cut_to_adjacent_routing_map.end() || routing_layer_it->second.size() < 2) {
      continue;
    }

    auto rule_it = cut_layer_sorted_rule_map.find(cut_layer_idx);
    if (rule_it == cut_layer_sorted_rule_map.end() || rule_it->second.empty()) {
      continue;
    }

    const std::vector<int32_t>& routing_layer_idx_list = routing_layer_it->second;
    int32_t above_routing_layer_idx = *std::max_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
    int32_t below_routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());

    for (const CutData& cut_data : cut_layer_data.getCuts()) {
      GTLRectInt cut_gtl_rect = cut_data.rect;
      PlanarRect cut_rect = DRCUTIL.convertToPlanarRect(cut_gtl_rect);
      if (!isValidRect(cut_rect)) {
        continue;
      }

      for (int32_t routing_layer_idx : routing_layer_idx_list) {
        auto layer_it = layer_data.find(routing_layer_idx);
        if (layer_it == layer_data.end()) {
          continue;
        }
        const RVLayerData& rv_layer_data = layer_it->second;

        std::vector<std::pair<GTLRectInt, int32_t>> rect_max_rect_pair_list;
        rv_layer_data.queryMaxRects(cut_gtl_rect, std::back_inserter(rect_max_rect_pair_list));
        if (rect_max_rect_pair_list.empty()) {
          continue;
        }

        std::map<Orientation, int32_t> orient_overhang_map
            = {{Orientation::kEast, 0}, {Orientation::kSouth, 0}, {Orientation::kWest, 0}, {Orientation::kNorth, 0}};
        for (const auto& [routing_gtl_rect, max_rect_id] : rect_max_rect_pair_list) {
          (void) max_rect_id;
          PlanarRect routing_rect = DRCUTIL.convertToPlanarRect(routing_gtl_rect);
          if (!isValidRect(routing_rect) || !DRCUTIL.isOpenOverlap(routing_rect, cut_rect) || !DRCUTIL.isInside(routing_rect, cut_rect)) {
            continue;
          }
          for (Orientation orient : orientations) {
            orient_overhang_map[orient] = std::max(orient_overhang_map[orient], DRCUTIL.getOrientEdgeDistance(cut_rect, routing_rect, orient));
          }
        }

        // Vanilla enclosure edge.
        for (const EnclosureEdgeRule& curr_rule : rule_it->second) {
          if (curr_rule.has_convexcorners) {
            continue;
          }
          if (curr_rule.has_above && (routing_layer_idx != above_routing_layer_idx)) {
            continue;
          }
          if (curr_rule.has_below && (routing_layer_idx != below_routing_layer_idx)) {
            continue;
          }

          auto width_rtree_it = layer_width_rtrees.find({routing_layer_idx, curr_rule.min_width});
          if (width_rtree_it == layer_width_rtrees.end()) {
            continue;
          }

          std::vector<std::pair<GTLRectInt, int32_t>> wide_rect_net_pair_list;
          width_rtree_it->second.query(bgi::intersects(cut_gtl_rect), std::back_inserter(wide_rect_net_pair_list));
          for (const auto& [wide_gtl_rect, net_idx] : wide_rect_net_pair_list) {
            PlanarRect wide_rect = DRCUTIL.convertToPlanarRect(wide_gtl_rect);
            if (!isValidRect(wide_rect) || !DRCUTIL.isOpenOverlap(wide_rect, cut_rect)) {
              continue;
            }

            auto net_it = rv_layer_data.nets.find(net_idx);
            if (net_it == rv_layer_data.nets.end()) {
              continue;
            }
            const GTLPolySetInt& poly_set = net_it->second.polyset;
            std::map<Orientation, PlanarRect> orient_extension_rects = buildOrientExtensionRects(wide_rect, curr_rule.par_within);

            for (Orientation orient : orientations) {
              int32_t orient_enclosure = std::max(orient_overhang_map[orient], DRCUTIL.getOrientEnclosure(wide_rect, cut_rect, orient));
              if (orient_enclosure >= curr_rule.overhang) {
                continue;
              }

              std::vector<std::pair<GTLRectInt, int32_t>> neighbor_rect_max_pair_list;
              {
                PlanarRect check_rect = DRCUTIL.getEnlargedRect(wide_rect, curr_rule.par_within);
                rv_layer_data.queryMaxRects(DRCUTIL.convertToGTLRectInt(check_rect), std::back_inserter(neighbor_rect_max_pair_list));
              }

              Orientation oppo_orient = DRCUTIL.getOppositeOrientation(orient);
              std::set<int32_t> check_side_par_set;
              std::set<int32_t> oppo_side_par_set;
              for (const auto& [neighbor_gtl_rect, neighbor_max_rect_id] : neighbor_rect_max_pair_list) {
                PlanarRect neighbor_rect = DRCUTIL.convertToPlanarRect(neighbor_gtl_rect);
                if (!isValidRect(neighbor_rect) || DRCUTIL.isClosedOverlap(neighbor_rect, wide_rect)) {
                  continue;
                }

                bool hit_check_side = DRCUTIL.isOpenOverlap(neighbor_rect, orient_extension_rects[orient]);
                bool hit_oppo_side = DRCUTIL.isOpenOverlap(neighbor_rect, orient_extension_rects[oppo_orient]);
                if (!hit_check_side && !hit_oppo_side) {
                  continue;
                }

                int32_t neighbor_net_idx = rv_layer_data.getNetIdxByMaxRectId(neighbor_max_rect_id);
                int32_t prl = DRCUTIL.getParallelLength(wide_rect, neighbor_rect);
                if (net_idx == neighbor_net_idx && prl > curr_rule.par_length) {
                  prl = DRCUTIL.getDirectPRL(poly_set, wide_rect, neighbor_rect);
                }
                if (prl <= curr_rule.par_length || DRCUTIL.getParallelLength(neighbor_rect, cut_rect) <= 0) {
                  continue;
                }

                if (hit_check_side) {
                  check_side_par_set.insert(neighbor_net_idx);
                }
                if (hit_oppo_side) {
                  oppo_side_par_set.insert(neighbor_net_idx);
                }
              }

              if (curr_rule.has_except_two_edges && !check_side_par_set.empty() && !oppo_side_par_set.empty()) {
                continue;
              }
              if (check_side_par_set.empty()) {
                continue;
              }

              Violation violation;
              violation.set_violation_type(ViolationType::kEnclosureEdge);
              violation.set_is_routing(true);
              violation.set_violation_net_set({net_idx, *check_side_par_set.begin()});
              violation.set_layer_idx(below_routing_layer_idx);
              violation.set_rect(cut_rect);
              violation.set_required_size(curr_rule.overhang);
              rv_cluster.get_violation_list().push_back(violation);
            }
          }
        }

        // Convex corner enclosure edge.
        for (const auto& [routing_gtl_rect, max_rect_id] : rect_max_rect_pair_list) {
          PlanarRect routing_rect = DRCUTIL.convertToPlanarRect(routing_gtl_rect);
          if (!isValidRect(routing_rect) || !DRCUTIL.isInside(routing_rect, cut_rect)) {
            continue;
          }

          int32_t polygon_id = rv_layer_data.getMaxRect(max_rect_id).polygon_id;
          int32_t net_idx = rv_layer_data.getNetIdxByPolygonId(polygon_id);
          auto layer_convex_it = layer_polygon_convex_candidates.find(routing_layer_idx);
          if (layer_convex_it == layer_polygon_convex_candidates.end()) {
            continue;
          }
          auto polygon_convex_it = layer_convex_it->second.find(polygon_id);
          if (polygon_convex_it == layer_convex_it->second.end()) {
            continue;
          }
          auto rect_convex_it = polygon_convex_it->second.find(routing_rect);
          if (rect_convex_it == polygon_convex_it->second.end()) {
            continue;
          }

          for (const EnclosureEdgeRule& convex_rule : rule_it->second) {
            if (!convex_rule.has_convexcorners) {
              continue;
            }
            if (convex_rule.has_above && (routing_layer_idx != above_routing_layer_idx)) {
              continue;
            }
            if (convex_rule.has_below && (routing_layer_idx != below_routing_layer_idx)) {
              continue;
            }

            for (const ConvexCandidate& convex_candidate : rect_convex_it->second) {
              const BoundaryData& curr_boundary = rv_layer_data.getBoundary(convex_candidate.curr_boundary_id);
              const BoundaryData& adj_boundary = rv_layer_data.getBoundary(convex_candidate.adj_boundary_id);
              int32_t curr_length = curr_boundary.edge_length;
              int32_t adj_length = adj_boundary.edge_length;
              Orientation checking_orient = curr_boundary.orient;
              if (curr_length > convex_rule.convex_length || convex_candidate.other_length > convex_rule.adjacent_length || adj_length < convex_rule.length) {
                continue;
              }
              if (orient_overhang_map[checking_orient] >= convex_rule.overhang) {
                continue;
              }

              PlanarRect curr_par_rect = DRCUTIL.getEnlargedPartRect(routing_rect, curr_boundary.orient, convex_rule.convex_par_within);
              PlanarRect adj_par_rect = DRCUTIL.getEnlargedPartRect(routing_rect, adj_boundary.orient, convex_rule.convex_par_within);

              int32_t max_env_width = 0;
              std::set<int32_t> env_curr_set;
              std::set<int32_t> env_adj_set;
              auto env_rtree_it = routing_env_rtree_map.find(routing_layer_idx);
              if (env_rtree_it != routing_env_rtree_map.end()) {
                std::vector<std::pair<GTLRectInt, int32_t>> env_only_rect_net_pair_list;
                PlanarRect check_rect = DRCUTIL.getEnlargedRect(routing_rect, convex_rule.convex_par_within);
                env_rtree_it->second.query(bgi::intersects(DRCUTIL.convertToGTLRectInt(check_rect)), std::back_inserter(env_only_rect_net_pair_list));
                for (const auto& [env_only_gtl_rect, env_only_net_idx] : env_only_rect_net_pair_list) {
                  PlanarRect env_only_rect = DRCUTIL.convertToPlanarRect(env_only_gtl_rect);
                  if (!isValidRect(env_only_rect) || DRCUTIL.isClosedOverlap(routing_rect, env_only_rect)) {
                    continue;
                  }

                  if (DRCUTIL.isOpenOverlap(env_only_rect, curr_par_rect) && DRCUTIL.getParallelLength(env_only_rect, cut_rect) > 0) {
                    env_curr_set.insert(env_only_net_idx);
                    max_env_width = std::max(max_env_width, env_only_rect.getWidth());
                  }
                  if (DRCUTIL.isOpenOverlap(env_only_rect, adj_par_rect) && DRCUTIL.getParallelLength(env_only_rect, cut_rect) > 0) {
                    env_adj_set.insert(env_only_net_idx);
                  }
                }
              }
              if (env_curr_set.empty() || env_adj_set.empty()) {
                max_env_width = 0;
              }

              std::set<int32_t> curr_par_net_idx_set;
              std::set<int32_t> adj_par_net_idx_set;
              std::vector<std::pair<GTLRectInt, int32_t>> env_rect_max_pair_list;
              {
                PlanarRect check_rect = DRCUTIL.getEnlargedRect(routing_rect, convex_rule.convex_par_within);
                rv_layer_data.queryMaxRects(DRCUTIL.convertToGTLRectInt(check_rect), std::back_inserter(env_rect_max_pair_list));
              }
              for (const auto& [env_gtl_rect, env_max_rect_id] : env_rect_max_pair_list) {
                PlanarRect env_routing_rect = DRCUTIL.convertToPlanarRect(env_gtl_rect);
                if (!isValidRect(env_routing_rect) || DRCUTIL.isClosedOverlap(cut_rect, env_routing_rect)) {
                  continue;
                }

                int32_t env_net_idx = rv_layer_data.getNetIdxByMaxRectId(env_max_rect_id);
                if (DRCUTIL.isOpenOverlap(env_routing_rect, curr_par_rect) && DRCUTIL.getParallelLength(env_routing_rect, cut_rect) > 0
                    && env_routing_rect.getWidth() > max_env_width) {
                  curr_par_net_idx_set.insert(env_net_idx);
                }
                if (DRCUTIL.isOpenOverlap(env_routing_rect, adj_par_rect) && DRCUTIL.getParallelLength(env_routing_rect, cut_rect) > 0) {
                  adj_par_net_idx_set.insert(env_net_idx);
                }
              }

              if (curr_par_net_idx_set.empty() || adj_par_net_idx_set.empty()) {
                continue;
              }

              Violation violation;
              violation.set_violation_type(ViolationType::kEnclosureEdge);
              violation.set_is_routing(true);
              violation.set_violation_net_set({net_idx, *curr_par_net_idx_set.begin()});
              violation.set_layer_idx(below_routing_layer_idx);
              violation.set_rect(cut_rect);
              violation.set_required_size(convex_rule.overhang);
              rv_cluster.get_violation_list().push_back(violation);
            }
          }
        }
      }
    }
  }
}

}  // namespace idrc
