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
#include "DRCShape.hpp"
#include "Logger.hpp"
#include "PlanarRect.hpp"
#include "RuleValidator.hpp"
#include "Utility.hpp"

namespace idrc {

void RuleValidator::verifyEnclosureParallel(RVCluster& rv_cluster)
{
  std::vector<CutLayer>& cut_layer_list = DRCDM.getDatabase().get_cut_layer_list();
  std::map<int32_t, std::vector<int32_t>>& cut_to_adjacent_routing_map = DRCDM.getDatabase().get_cut_to_adjacent_routing_map();
  const auto& layer_data = rv_cluster.get_layer_data();
  auto build_outer_ring_info
      = [this](const RVLayerData& rv_layer_data, const PolygonData& polygon_data, int32_t& coord_size, std::vector<Segment<PlanarCoord>>& edge_list,
               std::vector<int32_t>& edge_length_list, std::set<int32_t>& eol_edge_idx_set) {
          edge_list.clear();
          edge_length_list.clear();
          eol_edge_idx_set.clear();

          std::vector<bool> convex_corner_list;
          for (const BoundaryData& boundary_data : rv_layer_data.getBoundaries(polygon_data)) {
            if (boundary_data.isHole) {
              break;
            }
            edge_list.emplace_back(boundary_data.begin_coord, boundary_data.end_coord);
            edge_length_list.push_back(boundary_data.edge_length);
            convex_corner_list.push_back(boundary_data.isConvex);
          }
          coord_size = static_cast<int32_t>(edge_list.size());
          if (coord_size < 4) {
            return;
          }
          for (int32_t i = 0; i < coord_size; i++) {
            if (convex_corner_list[getIdx(i - 1, coord_size)] && convex_corner_list[i]) {
              eol_edge_idx_set.insert(i);
            }
          }
        };
  for (const auto& [cut_layer_idx, cut_layer_data] : layer_data) {
    if (cut_layer_data.cut_pool.empty()) {
      continue;
    }
    const std::vector<int32_t>& routing_layer_idx_list = cut_to_adjacent_routing_map[cut_layer_idx];
    if (routing_layer_idx_list.size() < 2) {
      continue;
    }
    int32_t above_routing_layer_idx = *std::max_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
    int32_t below_routing_layer_idx = *std::min_element(routing_layer_idx_list.begin(), routing_layer_idx_list.end());
    const EnclosureParallelRule& curr_rule = cut_layer_list[cut_layer_idx].get_enclosure_parallel_rule();
    for (const CutData& cut_data : cut_layer_data.getCuts()) {
      GTLRectInt cut_gtl_rect = cut_data.rect;
      int32_t cut_net_idx = cut_data.net_idx;
      if (cut_net_idx == -1) {
        continue;
      }
      PlanarRect cut_rect = DRCUTIL.convertToPlanarRect(cut_gtl_rect);
      std::set<Segment<PlanarCoord>, CmpSegmentXASC> processed_segment_set;
      for (int32_t routing_layer_idx : routing_layer_idx_list) {
        if (curr_rule.has_above && (routing_layer_idx != above_routing_layer_idx)) {
          continue;
        }
        if (curr_rule.has_below && (routing_layer_idx != below_routing_layer_idx)) {
          continue;
        }
        auto layer_it = layer_data.find(routing_layer_idx);
        if (layer_it == layer_data.end()) {
          continue;
        }
        const RVLayerData& rv_layer_data = layer_it->second;
        std::vector<std::pair<GTLRectInt, int32_t>> rect_polygon_pair_list;
        rv_layer_data.queryMaxRects(cut_gtl_rect, std::back_inserter(rect_polygon_pair_list));
        std::vector<Orientation> orientation_list;
        {
          std::map<Orientation, int32_t> orient_overhang_map;
          for (const auto& [gtl_rect, max_rect_id] : rect_polygon_pair_list) {
            (void) max_rect_id;
            PlanarRect routing_rect = DRCUTIL.convertToPlanarRect(gtl_rect);
            if (!DRCUTIL.isOpenOverlap(routing_rect, cut_rect)) {
              continue;
            }
            if (routing_rect.get_ll_x() <= cut_rect.get_ll_x()) {
              orient_overhang_map[Orientation::kWest]
                  = std::max(orient_overhang_map[Orientation::kWest], std::abs(cut_rect.get_ll_x() - routing_rect.get_ll_x()));
            }
            if (routing_rect.get_ur_x() >= cut_rect.get_ur_x()) {
              orient_overhang_map[Orientation::kEast]
                  = std::max(orient_overhang_map[Orientation::kEast], std::abs(cut_rect.get_ur_x() - routing_rect.get_ur_x()));
            }
            if (routing_rect.get_ur_y() >= cut_rect.get_ur_y()) {
              orient_overhang_map[Orientation::kNorth]
                  = std::max(orient_overhang_map[Orientation::kNorth], std::abs(cut_rect.get_ur_y() - routing_rect.get_ur_y()));
            }
            if (routing_rect.get_ll_y() <= cut_rect.get_ll_y()) {
              orient_overhang_map[Orientation::kSouth]
                  = std::max(orient_overhang_map[Orientation::kSouth], std::abs(cut_rect.get_ll_y() - routing_rect.get_ll_y()));
            }
          }
          orientation_list.reserve(orient_overhang_map.size());
          for (auto& [orient, overhang] : orient_overhang_map) {
            if (overhang >= curr_rule.overhang) {
              continue;
            }
            orientation_list.push_back(orient);
          }
        }
        for (const auto& [gtl_rect, max_rect_id] : rect_polygon_pair_list) {
          int32_t polygon_id = rv_layer_data.getMaxRect(max_rect_id).polygon_id;
          int32_t net_idx = rv_layer_data.getNetIdxByPolygonId(polygon_id);
          const PolygonData& polygon_data = rv_layer_data.getPolygon(polygon_id);
          PlanarRect routing_rect = DRCUTIL.convertToPlanarRect(gtl_rect);
          if (!DRCUTIL.isOpenOverlap(routing_rect, cut_rect)) {
            continue;
          }
          // 只考虑完全包含cut的net（允许切成矩形后openoverlap）
          bool is_contained = false;
          for (const MaxRectData& polygon_max_rect : rv_layer_data.getMaxRects(polygon_data)) {
            if (DRCUTIL.isInside(DRCUTIL.convertToPlanarRect(polygon_max_rect.rect), cut_rect)) {
              is_contained = true;
              break;
            }
          }
          if (!is_contained) {
            continue;
          }
          int32_t coord_size = 0;
          std::vector<Segment<PlanarCoord>> edge_list;
          std::vector<int32_t> edge_length_list;
          std::set<int32_t> eol_edge_idx_set;
          build_outer_ring_info(rv_layer_data, polygon_data, coord_size, edge_list, edge_length_list, eol_edge_idx_set);
          if (coord_size < 4) {
            continue;
          }
          for (Orientation& orient : orientation_list) {
            int32_t edge_idx = -1;
            for (int32_t eol_idx : eol_edge_idx_set) {
              if (DRCUTIL.isInside(edge_list[eol_idx], routing_rect.getOrientEdge(orient))) {
                edge_idx = eol_idx;
                break;
              }
            }
            if (edge_idx == -1) {
              continue;
            }
            Segment<PlanarCoord> curr_segment = edge_list[getIdx(edge_idx, coord_size)];
            if (DRCUTIL.exist(processed_segment_set, curr_segment)) {
              continue;
            }
            if (DRCUTIL.getManhattanDistance(curr_segment.get_first(), curr_segment.get_second()) >= curr_rule.eol_width) {
              continue;
            }
            PlanarRect eol_edge_rect = DRCUTIL.getRect(curr_segment.get_first(), curr_segment.get_second());
            PlanarRect cut_edge_rect = DRCUTIL.getRect(cut_rect.getOrientEdge(orient).get_first(), cut_rect.getOrientEdge(orient).get_second());
            if (DRCUTIL.getEuclideanDistance(eol_edge_rect, cut_edge_rect) >= curr_rule.overhang) {
              continue;
            }
            int32_t pre_segment_length;
            int32_t post_segment_length;
            {
              Segment<PlanarCoord> pre_segment = edge_list[getIdx(edge_idx - 1, coord_size)];
              Segment<PlanarCoord> post_segment = edge_list[getIdx(edge_idx + 1, coord_size)];
              pre_segment_length = DRCUTIL.getManhattanDistance(pre_segment.get_first(), pre_segment.get_second());
              post_segment_length = DRCUTIL.getManhattanDistance(post_segment.get_first(), post_segment.get_second());
            }
            if (curr_rule.has_min_length && pre_segment_length < curr_rule.min_length && post_segment_length < curr_rule.min_length) {
              continue;
            }
            processed_segment_set.insert(curr_segment);
            bool has_parallel_neighbor = false;
            {
              PlanarRect left_par_rect;
              PlanarRect right_par_rect;
              {
                int32_t par_spacing = curr_rule.par_spacing - DRCUTIL.getManhattanDistance(curr_segment.get_first(), curr_segment.get_second());
                if (orient == Orientation::kEast) {
                  left_par_rect = DRCUTIL.getEnlargedRect(curr_segment.get_first(), curr_rule.backward_ext + 1, par_spacing, curr_rule.forward_ext + 1, 0);
                  right_par_rect = DRCUTIL.getEnlargedRect(curr_segment.get_second(), curr_rule.backward_ext + 1, 0, curr_rule.forward_ext + 1, par_spacing);
                } else if (orient == Orientation::kWest) {
                  left_par_rect = DRCUTIL.getEnlargedRect(curr_segment.get_first(), curr_rule.forward_ext + 1, 0, curr_rule.backward_ext + 1, par_spacing);
                  right_par_rect = DRCUTIL.getEnlargedRect(curr_segment.get_second(), curr_rule.forward_ext + 1, par_spacing, curr_rule.backward_ext + 1, 0);
                } else if (orient == Orientation::kSouth) {
                  left_par_rect = DRCUTIL.getEnlargedRect(curr_segment.get_first(), par_spacing, curr_rule.forward_ext + 1, 0, curr_rule.backward_ext + 1);
                  right_par_rect = DRCUTIL.getEnlargedRect(curr_segment.get_second(), 0, curr_rule.forward_ext + 1, par_spacing, curr_rule.backward_ext + 1);
                } else if (orient == Orientation::kNorth) {
                  left_par_rect = DRCUTIL.getEnlargedRect(curr_segment.get_first(), 0, curr_rule.backward_ext + 1, par_spacing, curr_rule.forward_ext + 1);
                  right_par_rect = DRCUTIL.getEnlargedRect(curr_segment.get_second(), par_spacing, curr_rule.backward_ext + 1, 0, curr_rule.forward_ext + 1);
                } else {
                  DRCLOG.error(Loc::current(), "The orientation is error!");
                }
              }
              std::vector<std::pair<GTLRectInt, int32_t>> env_rect_polygon_pair_list;
              {
                PlanarRect check_rect;
                if (pre_segment_length >= curr_rule.min_length && post_segment_length >= curr_rule.min_length) {
                  check_rect = DRCUTIL.getBoundingBox({left_par_rect.get_ll(), left_par_rect.get_ur(), right_par_rect.get_ll(), right_par_rect.get_ur()});
                } else if (pre_segment_length >= curr_rule.min_length) {
                  check_rect = left_par_rect;
                } else if (post_segment_length >= curr_rule.min_length) {
                  check_rect = right_par_rect;
                }
                // 是否需要按边的朝向来去掉rect所在的polygon的重叠？此类型违例几乎不存在，先不考虑
                rv_layer_data.queryMaxRects(DRCUTIL.convertToGTLRectInt(check_rect), std::back_inserter(env_rect_polygon_pair_list));
              }
              bool has_left_neighbor = false;
              bool has_right_neighbor = false;
              for (const auto& [env_gtl_rect, env_max_rect_id] : env_rect_polygon_pair_list) {
                int32_t env_polygon_id = rv_layer_data.getMaxRect(env_max_rect_id).polygon_id;
                int32_t env_net_idx = rv_layer_data.getNetIdxByPolygonId(env_polygon_id);
                PlanarRect env_routing_rect = DRCUTIL.convertToPlanarRect(env_gtl_rect);
                // 不考虑obs的边在 window中有重叠的情况
                if (env_net_idx == -1) {
                  continue;
                }
                if (DRCUTIL.isClosedOverlap(routing_rect, env_routing_rect) && net_idx == env_net_idx) {
                  continue;
                }
                if (DRCUTIL.isOpenOverlap(env_routing_rect, left_par_rect)) {
                  has_left_neighbor = true;
                }
                if (DRCUTIL.isOpenOverlap(env_routing_rect, right_par_rect)) {
                  has_right_neighbor = true;
                }
                if (has_left_neighbor || has_right_neighbor) {
                  has_parallel_neighbor = true;
                  break;
                }
              }
            }
            if (!has_parallel_neighbor) {
              continue;
            }
            Violation violation;
            violation.set_violation_type(ViolationType::kEnclosureParallel);
            violation.set_is_routing(true);
            violation.set_violation_net_set({net_idx});
            violation.set_layer_idx(below_routing_layer_idx);
            violation.set_rect(cut_rect);
            violation.set_required_size(curr_rule.overhang);
            rv_cluster.get_violation_list().push_back(violation);
          }
        }
      }
    }
  }
}

}  // namespace idrc
