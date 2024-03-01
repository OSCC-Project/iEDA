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
#include "idrc_engine_manager.h"

#include "engine_geometry_creator.h"
#include "engine_scanline.h"
#include "geometry_boost.h"
#include "idm.h"
#include "idrc_engine_manager.h"
#include "omp.h"
#include "rule_condition_width.h"
#include "tech_rules.h"

namespace idrc {

DrcEngineManager::DrcEngineManager(DrcDataManager* data_manager, DrcConditionManager* condition_manager)
    : _data_manager(data_manager), _condition_manager(condition_manager)
{
  _layouts
      = {{LayoutType::kRouting, std::map<std::string, DrcEngineLayout*>{}}, {LayoutType::kCut, std::map<std::string, DrcEngineLayout*>{}}};
  _scanline_matrix = {{LayoutType::kRouting, std::map<std::string, DrcEngineScanline*>{}},
                      {LayoutType::kCut, std::map<std::string, DrcEngineScanline*>{}}};
  // _engine_check = new DrcEngineCheck();
}

DrcEngineManager::~DrcEngineManager()
{
  for (auto& [type, layout_arrays] : _layouts) {
    for (auto& [layer, layout] : layout_arrays) {
      if (layout != nullptr) {
        delete layout;
        layout = nullptr;
      }
    }

    layout_arrays.clear();
  }
  _layouts.clear();

  for (auto& [type, matrix_arrays] : _scanline_matrix) {
    for (auto& [layer, matrix] : matrix_arrays) {
      if (matrix != nullptr) {
        delete matrix;
        matrix = nullptr;
      }
    }
    matrix_arrays.clear();
  }
  _scanline_matrix.clear();
}

// get or create layout engine for each layer
DrcEngineLayout* DrcEngineManager::get_layout(std::string layer, LayoutType type)
{
  auto& layouts = get_engine_layouts(type);

  auto* engine_layout = layouts[layer];
  if (engine_layout == nullptr) {
    engine_layout = new DrcEngineLayout(layer);
    layouts[layer] = engine_layout;
  }

  return engine_layout;
}
// add rect to engine
bool DrcEngineManager::addRect(int llx, int lly, int urx, int ury, std::string layer, int net_id, LayoutType type)
{
  /// get layout by type & layer id
  auto engine_layout = get_layout(layer, type);
  if (engine_layout == nullptr) {
    return false;
  }

  return engine_layout->addRect(llx, lly, urx, ury, net_id);
}

void DrcEngineManager::dataPreprocess()
{
  for (auto& [layer, layout] : get_engine_layouts()) {
    layout->combineLayout();
  }
}

void DrcEngineManager::filterData()
{
  // TODO: put logic bellow into condition module
  // TODO: multi-thread
  for (auto& [layer, layout] : get_engine_layouts()) {
    // TODO: only for routing layers
    auto& layer_polyset = layout->get_layout()->get_engine()->get_polyset();

    // overlap
    auto& overlap = layout->get_layout()->get_engine()->getOverlap();
    if (overlap.size() > 0) {
      // TODO: create scanline point data
      int a = 0;
    }

#ifdef DEBUG_IDRC_ENGINE
    std::vector<ieda_solver::GeometryViewPolygon> polygons;
    layer_polyset.get(polygons);
#endif

    // min spacing
    int min_spacing = DrcTechRuleInst->getMinSpacing(layer);
    if (min_spacing > 0) {
      auto violation_position_set = layer_polyset;
      int half_min_spacing = min_spacing / 2;
      gtl::grow_and(violation_position_set, half_min_spacing);
      std::vector<ieda_solver::GeometryRect> results;
      violation_position_set.get(results);

      auto get_new_interval = [&](ieda_solver::GeometryOrientation direction, ieda_solver::GeometryRect& rect) {
        int length = ieda_solver::getWireWidth(rect, direction);
        if (length <= half_min_spacing) {
          int expand_length = std::abs(half_min_spacing - length);
          ieda_solver::bloat(rect, direction, expand_length);
        } else if (length > min_spacing) {
          ieda_solver::shrink(rect, direction, half_min_spacing);
        } else {
          int shrink_length = std::abs(half_min_spacing - length);
          ieda_solver::shrink(rect, direction, shrink_length);
        }
      };

      for (auto& rect : results) {
        get_new_interval(ieda_solver::HORIZONTAL, rect);
        get_new_interval(ieda_solver::VERTICAL, rect);
      }

      if (results.size() > 0) {
        int a = 0;
      }

#ifdef DEBUG_IDRC_ENGINE
      std::vector<ieda_solver::GeometryViewPolygon> grow_polygons;
      violation_position_set.get(grow_polygons);
      if (grow_polygons.size() > 0) {
        int a = 0;
      }
#endif
    }

    // jog and prl
    using WidthToPolygonSetMap = std::map<int, ieda_solver::GeometryPolygonSet>;
    WidthToPolygonSetMap jog_wire_map;
    WidthToPolygonSetMap prl_wire_map;

    auto rule_jog_to_jog = DrcTechRuleInst->getJogToJog(layer);
    auto rule_spacing_table = DrcTechRuleInst->getSpacingTable(layer);

    auto& wires = layout->get_layout()->get_engine()->getWires();
    for (auto& wire : wires) {
      // jog
      auto wire_direction = ieda_solver::getWireDirection(wire);
      auto width_direction = wire_direction.get_perpendicular();
      int wire_width = ieda_solver::getWireWidth(wire, width_direction);

      if (rule_jog_to_jog) {
        for (auto& width_item : rule_jog_to_jog->get_width_list()) {
          int rule_width = width_item.get_width();
          if (wire_width > rule_width) {
            // create big wire layer
            jog_wire_map[rule_width] += wire;
            break;
          }
        }
      }

      // prl
      if (rule_spacing_table && rule_spacing_table->is_parallel()) {
        auto idb_table_prl = rule_spacing_table->get_parallel();
        auto& idb_width_list = idb_table_prl->get_width_list();

        int width_idx = 0;
        for (int i = idb_width_list.size() - 1; i >= 0; --i) {
          if (wire_width >= idb_width_list[i]) {
            width_idx = i;
            break;
          }
        }
        if (width_idx > 0) {
          prl_wire_map[width_idx] += wire;
        }
        int a = 0;
      }
    }

    // jog
    if (rule_jog_to_jog) {
      std::vector<ieda_solver::GeometryRect> jog_violations;
      for (auto& [rule_width, jog_wires] : jog_wire_map) {
        int rule_short_jog_spacing = rule_jog_to_jog->get_short_jog_spacing();
        int rule_jog_width = rule_jog_to_jog->get_jog_width();
        int rule_jog_to_jog_spacing = rule_jog_to_jog->get_jog_to_jog_spacing();

        for (auto& width_item : rule_jog_to_jog->get_width_list()) {
          if (rule_width == width_item.get_width()) {
            int rule_within = width_item.get_par_within();
            int rule_prl = width_item.get_par_length();
            int rule_long_jog_spacing = width_item.get_long_jog_spacing();

            auto check_by_direction = [&](ieda_solver::GeometryOrientation spacing_direction) {
              auto prl_direction = spacing_direction.get_perpendicular();
              auto expand_wires = jog_wires;
              ieda_solver::bloat(expand_wires, spacing_direction, rule_within);
              auto wire_with_jogs = layer_polyset;
              ieda_solver::interact(wire_with_jogs, jog_wires);
              auto jogs_attach_to_wire = wire_with_jogs - jog_wires;
              auto check_region = expand_wires - layer_polyset;
              auto within_region = check_region + jogs_attach_to_wire;
              std::vector<ieda_solver::GeometryRect> within_region_rects;
              ieda_solver::getRectangles(within_region_rects, within_region, spacing_direction);
              ieda_solver::GeometryPolygonSet split_check_rects_set;
              for (auto& rect : within_region_rects) {
                int length = ieda_solver::getWireWidth(rect, spacing_direction);
                if (length < rule_within) {
                  ieda_solver::shrink(rect, prl_direction, 1);
                  split_check_rects_set += rect;
                }
              }
              ieda_solver::interact(split_check_rects_set, wire_with_jogs);
              ieda_solver::bloat(split_check_rects_set, prl_direction, 1);
              ieda_solver::GeometryPolygonSet region_b = split_check_rects_set - jogs_attach_to_wire;
              std::vector<ieda_solver::GeometryPolygon> region_b_polygons;
              region_b.get(region_b_polygons);
              for (auto& region_b_polygon : region_b_polygons) {
                ieda_solver::GeometryRect bbox;
                ieda_solver::envelope(bbox, region_b_polygon);
                int prl = ieda_solver::getWireWidth(bbox, prl_direction);
                if (prl > rule_prl) {
                  ieda_solver::GeometryPolygonSet current_region_b_set;
                  current_region_b_set += region_b_polygon;
                  std::vector<ieda_solver::GeometryRect> region_b_rects;
                  std::vector<ieda_solver::GeometryRect> region_a_rects;
                  ieda_solver::getRectangles(region_b_rects, current_region_b_set, spacing_direction);
                  for (auto& rect : region_b_rects) {
                    int width = ieda_solver::getWireWidth(rect, prl_direction);
                    int spacing = ieda_solver::getWireWidth(rect, spacing_direction);
                    if (width > rule_jog_width) {  // long jog
                      if (spacing < rule_long_jog_spacing) {
                        jog_violations.push_back(rect);
                      }
                    } else {  // short jog
                      if (spacing < rule_short_jog_spacing) {
                        jog_violations.push_back(rect);
                      } else if (spacing < rule_long_jog_spacing) {  // region a
                        region_a_rects.push_back(rect);
                      }
                    }
                  }
                  for (size_t i = 1; i < region_a_rects.size(); ++i) {
                    // TODO: distance
                    int distance = ieda_solver::manhattanDistance(region_a_rects[i], region_a_rects[i - 1]);
                    if (distance < rule_jog_to_jog_spacing) {
                      auto vio_rect = region_a_rects[i - 1];
                      ieda_solver::oppositeRegion(vio_rect, region_a_rects[i]);
                      jog_violations.emplace_back(vio_rect);
                    }
                  }
                }
              }
            };

            check_by_direction(ieda_solver::HORIZONTAL);
            check_by_direction(ieda_solver::VERTICAL);

            break;
          }
        }
      }

      if (!jog_violations.empty()) {
        int a = 0;
      }
    }

    // prl
    if (rule_spacing_table && rule_spacing_table->is_parallel()) {
      auto idb_table_prl = rule_spacing_table->get_parallel();

      auto& idb_prl_length_list = idb_table_prl->get_parallel_length_list();
      auto& idb_spacing_array = idb_table_prl->get_spacing_table();

      auto prl_length_list = idb_prl_length_list;
      prl_length_list[0] = prl_length_list[1];  // t28 wide metal space rule summary table

      std::vector<ieda_solver::GeometryRect> spacing_table_violations;

      for (auto& [width_idx, wire_set] : prl_wire_map) {
        int prl_idx = width_idx - 1;
        int expand_size = idb_spacing_array[width_idx][prl_length_list.size() - 1];
        int required_prl = prl_length_list[prl_idx];

        auto check_by_direction = [&](ieda_solver::GeometryOrientation direction) {
          auto expand_wires = wire_set;
          ieda_solver::bloat(expand_wires, direction, expand_size);
          auto wire_with_jogs = layer_polyset;
          ieda_solver::interact(wire_with_jogs, wire_set);
          auto expand_region = expand_wires - wire_with_jogs;
          auto check_region = expand_region & layer_polyset;

          std::vector<ieda_solver::GeometryRect> check_region_rects;
          ieda_solver::getMaxRectangles(check_region_rects, check_region);

          ieda_solver::GeometryPolygonSet violation_region_set;
          for (auto& rect : check_region_rects) {
            int length = ieda_solver::getWireWidth(rect, direction);
            int prl = ieda_solver::getWireWidth(rect, direction.get_perpendicular());
            if (prl > required_prl && length <= expand_size) {
              ieda_solver::bloat(rect, direction, expand_size - length);
              violation_region_set += rect;
            }
          }

          ieda_solver::GeometryPolygonSet touch_wire_region(violation_region_set - check_region);
          ieda_solver::interact(touch_wire_region, wire_set);

          std::vector<ieda_solver::GeometryRect> current_violations;
          touch_wire_region.get(current_violations);
          spacing_table_violations.insert(spacing_table_violations.end(), current_violations.begin(), current_violations.end());
        };

        check_by_direction(ieda_solver::HORIZONTAL);
        check_by_direction(ieda_solver::VERTICAL);
      }

      if (!spacing_table_violations.empty()) {
        int a = 0;
      }
    }

    // edge
    auto rule_eol_list = DrcTechRuleInst->getSpacingEolList(layer);
    int max_eol_width = 0;
    for (auto& rule_eol : rule_eol_list) {
      max_eol_width = std::max(max_eol_width, rule_eol->get_eol_width());
    }
    using EolRuleToRegionMap = std::map<std::shared_ptr<idb::routinglayer::Lef58SpacingEol>, ieda_solver::GeometryPolygonSet>;
    EolRuleToRegionMap eol_check_regions;
    EolRuleToRegionMap eol_par_space_regions;
    EolRuleToRegionMap eol_same_metal_regions;

    auto get_edge_orientation = [](const ieda_solver::GeometryPoint& p1, const ieda_solver::GeometryPoint& p2) {
      return p1.x() == p2.x() ? ieda_solver::VERTICAL : ieda_solver::HORIZONTAL;
    };

    auto get_edge_direction = [](const ieda_solver::GeometryPoint& p1, const ieda_solver::GeometryPoint& p2) {
      return p1.x() == p2.x() ? (p2.y() > p1.y() ? ieda_solver::NORTH : ieda_solver::SOUTH)
                              : (p2.x() > p1.x() ? ieda_solver::EAST : ieda_solver::WEST);
    };

    auto is_convex = [](const ieda_solver::GeometryDirection2D& d1, const ieda_solver::GeometryDirection2D& d2) { return d1.left() == d2; };

    auto area_calculator = [](long long& area_accumulated, const ieda_solver::GeometryPoint& p1, const ieda_solver::GeometryPoint& p2) {
      area_accumulated += (long long) p1.x() * (long long) p2.y() - (long long) p1.y() * (long long) p2.x();
    };

    auto& polygon_with_holes = layout->get_layout()->get_engine()->getLayoutPolygons();
    for (auto& polygon : polygon_with_holes) {
      int polygon_point_number = polygon.size();
      if (polygon_point_number < 4) {
        continue;
      }

      // polygon outline
      std::vector<ieda_solver::GeometryPoint> polygon_outline(polygon_point_number);
      std::vector<bool> corner_convex_history(polygon_point_number);
      std::vector<int> edge_length_history(polygon_point_number);
      std::vector<ieda_solver::GeometryOrientation> edge_orientation_history(polygon_point_number);
      std::vector<ieda_solver::GeometryDirection2D> edge_direction_history(polygon_point_number);
      int corner_index = 0;
      long long polygon_area = 0;

      auto it_next = polygon.begin();
      auto it_prev = it_next++;
      auto it_current = it_next++;
      do {
        int edge_length = ieda_solver::manhattanDistance(*it_current, *it_prev);
        bool is_current_convex = is_convex(get_edge_direction(*it_prev, *it_current), get_edge_direction(*it_current, *it_next));
        ieda_solver::GeometryOrientation edge_orientation = get_edge_orientation(*it_prev, *it_current);
        ieda_solver::GeometryDirection2D edge_direction = get_edge_direction(*it_prev, *it_current);

        // record polygon outline
        polygon_outline[corner_index] = *it_current;
        edge_length_history[corner_index] = edge_length;
        corner_convex_history[corner_index] = is_current_convex;
        edge_orientation_history[corner_index] = edge_orientation;
        edge_direction_history[corner_index] = edge_direction;

        // refresh area
        area_calculator(polygon_area, *it_prev, *it_current);

        // next segment
        ++corner_index;
        it_prev = it_current;
        it_current = it_next;
        ++it_next;
        if (it_next == polygon.end()) {
          it_next = polygon.begin();
        }
      } while (it_prev != polygon.begin());

      // polygon area
      polygon_area = std::abs(polygon_area) / 2;

      // deal with polygon outline
      auto get_index_shifted = [&](int index, int shift) { return (index + shift + polygon_point_number) % polygon_point_number; };

      for (int i = 0; i < polygon_point_number; ++i) {
        auto point_index_prev = get_index_shifted(i, -1);
        auto point_current = polygon_outline[i];
        auto point_prev = polygon_outline[point_index_prev];
        auto is_current_convex = corner_convex_history[i];
        auto edge_length = edge_length_history[i];
        auto edge_orientation = edge_orientation_history[i];
        auto edge_direction = edge_direction_history[i];

        // eol
        if (is_current_convex && corner_convex_history[point_index_prev] && edge_length < max_eol_width) {
          for (auto& rule_eol : rule_eol_list) {
            if (edge_length >= rule_eol->get_eol_width()) {
              continue;
            }

            int eol_spacing = rule_eol->get_eol_space();
            // mininum value is 1, to ensure detect parallel edge regions and check violation regions have overlap
            int eol_within = std::max(rule_eol->get_eol_within().value_or(0), 1);

            // key words
            if (rule_eol->get_enclose_cut().has_value()) {
              // TODO: enclose cut
              continue;
            }

            if (rule_eol->get_adj_edge_length().has_value()) {
              auto rule_adj_edge_length = rule_eol->get_adj_edge_length().value();
              int min_edge_length = rule_adj_edge_length.get_min_length().value_or(0);
              if (edge_length_history[get_index_shifted(i, 1)] < min_edge_length
                  || edge_length_history[point_index_prev] < min_edge_length) {
                continue;
              }
            }

            if (rule_eol->get_end_to_end().has_value()) {
              auto rule_end_to_end = rule_eol->get_end_to_end().value();
              int rule_end_to_end_spacing = rule_end_to_end.get_end_to_end_space();
              // TODO: end to end situation
            }

            if (rule_eol->get_parallel_edge().has_value()) {
              auto rule_par_edge = rule_eol->get_parallel_edge().value();
              int rule_par_spacing = rule_par_edge.get_par_space();
              if (rule_par_edge.is_subtract_eol_width()) {
                rule_par_spacing -= edge_length;
              }
              int rule_par_within = rule_par_edge.get_par_within();

              ieda_solver::GeometryRect detect_rect_left(point_current.x(), point_current.y(), point_current.x(), point_current.y());
              ieda_solver::bloat(detect_rect_left, edge_direction, rule_par_spacing - 1);
              ieda_solver::shrink(detect_rect_left, edge_direction.backward(), 1);
              ieda_solver::bloat(detect_rect_left, edge_direction.left(), rule_par_within);
              ieda_solver::bloat(detect_rect_left, edge_direction.right(), eol_within);
              eol_par_space_regions[rule_eol] += detect_rect_left;

              ieda_solver::GeometryRect detect_rect_right(point_prev.x(), point_prev.y(), point_prev.x(), point_prev.y());
              ieda_solver::bloat(detect_rect_right, edge_direction.backward(), rule_par_spacing - 1);
              ieda_solver::shrink(detect_rect_right, edge_direction, 1);
              ieda_solver::bloat(detect_rect_right, edge_direction.left(), rule_par_within);
              ieda_solver::bloat(detect_rect_right, edge_direction.right(), eol_within);
              eol_par_space_regions[rule_eol] += detect_rect_right;

              // TWOEDGES: connect detect region
              if (rule_par_edge.is_two_edges()) {
                ieda_solver::GeometryRect connect_rect(point_prev.x(), point_prev.y(), point_current.x(), point_current.y());
                ieda_solver::bloat(connect_rect, edge_direction.right(), 2);
                ieda_solver::shrink(connect_rect, edge_direction.left(), 1);
                ieda_solver::bloat(connect_rect, edge_orientation, 2);
                eol_par_space_regions[rule_eol] += connect_rect;
              }

              // SAMEMETAL:
              if (rule_par_edge.is_same_metal()) {
                ieda_solver::bloat(detect_rect_left, edge_direction, 1);
                ieda_solver::bloat(detect_rect_right, edge_direction.backward(), 1);
                ieda_solver::shrink(detect_rect_left, edge_direction.backward(), rule_par_spacing - 2);
                ieda_solver::shrink(detect_rect_right, edge_direction, rule_par_spacing - 2);
                eol_same_metal_regions[rule_eol] += detect_rect_left;
                eol_same_metal_regions[rule_eol] += detect_rect_right;
                continue;
              }
            }

            // eol spacing
            ieda_solver::GeometryRect check_rect(point_prev.x(), point_prev.y(), point_current.x(), point_current.y());
            ieda_solver::bloat(check_rect, edge_direction.right(), eol_spacing - 1);
            ieda_solver::shrink(check_rect, edge_direction.left(), 1);
            ieda_solver::bloat(check_rect, edge_orientation, eol_within);
            eol_check_regions[rule_eol] += check_rect;
          }

          int a = 0;
        }

        // TODO: notch, step, corner fill
      }

      // polygon holes
      // TODO: holes need to check edge?
      for (auto hole_it = polygon.begin_holes(); hole_it != polygon.end_holes(); ++hole_it) {
        // TODO: polygon area subtract hole area

        // TODO: min enclosed area
      }

      // TODO: check rules
    }

    for (auto& rule_eol : rule_eol_list) {
      auto& check_regions = eol_check_regions[rule_eol];
      auto data_to_check = layer_polyset;  // TODO: avoid copy when SAMEMETAL not exist

      if (rule_eol->get_parallel_edge().has_value()) {
        auto& par_space_regions = eol_par_space_regions[rule_eol];

        if (rule_eol->get_parallel_edge().value().is_same_metal()) {
          auto& same_metal_through_detect_regions = eol_same_metal_regions[rule_eol];
          auto same_metal_through_overlaps_layer = same_metal_through_detect_regions & data_to_check;
          ieda_solver::GeometryPolygonSet remained_not_through_detect_regions
              = same_metal_through_detect_regions ^ same_metal_through_overlaps_layer;
          ieda_solver::interact(par_space_regions, remained_not_through_detect_regions);
          ieda_solver::interact(data_to_check, par_space_regions);
        }

        ieda_solver::interact(par_space_regions, data_to_check);
        ieda_solver::interact(check_regions, par_space_regions);
      }

      ieda_solver::interact(check_regions, data_to_check);
      auto violation_regions = check_regions - data_to_check;

      // TODO: get violation regions
      std::vector<ieda_solver::GeometryViewPolygon> through_polygons;
      eol_same_metal_regions[rule_eol].get(through_polygons);
      std::vector<ieda_solver::GeometryViewPolygon> detect_polygons;
      eol_par_space_regions[rule_eol].get(detect_polygons);
      std::vector<ieda_solver::GeometryViewPolygon> check_polygons;
      check_regions.get(check_polygons);
      std::vector<ieda_solver::GeometryRect> violation_rects;
      ieda_solver::getMaxRectangles(violation_rects, violation_regions);
      if (!violation_rects.empty()) {
        int a = 0;
      }
      if (rule_eol->get_parallel_edge().has_value() && rule_eol->get_parallel_edge().value().is_same_metal()) {
        int a = 0;
      }
      int a = 0;
    }

    // TODO: rule check

    int a = 0;
  }
}

// void DrcEngineManager::dataPreprocess()
// {
// #ifdef DEBUG_IDRC_ENGINE
//   ieda::Stats stats;
//   std::cout << "idrc : begin init scanline database" << std::endl;
// #endif
//   /// init scanline engine for routing layer
//   auto& layouts = get_engine_layouts(LayoutType::kRouting);

//   for (auto& [layer, engine_layout] : layouts) {
//     /// scanline engine for one layer
//     auto* scanline_engine = get_engine_scanline(layer, LayoutType::kRouting);
//     auto* scanline_preprocess = scanline_engine->get_preprocess();

//     // reserve capacity for basic points
//     uint64_t point_number = engine_layout->pointCount();
//     scanline_preprocess->reserveSpace(point_number);

//     // create scanline points
//     for (auto [net_id, sub_layout] : engine_layout->get_sub_layouts()) {
//       /// build engine data
//       auto* boost_engine = static_cast<ieda_solver::GeometryBoost*>(sub_layout->get_engine());
//       auto boost_pt_list_pair = boost_engine->get_boost_polygons_points();

//       /// add data to scanline engine
//       scanline_preprocess->addData(boost_pt_list_pair.second, net_id);  /// boost_pt_list_pair : second value is polygon points
//     }

//     /// sort point list in scanline data manager
//     scanline_preprocess->sortEndpoints();

//     // std::cout << "idrc : layer id = " << layer->get_id() << " polygon points total number = " << point_number << std::endl;
//   }

// #ifdef DEBUG_IDRC_ENGINE
//   std::cout << "idrc : end init scanline database, "
//             << " runtime = " << stats.elapsedRunTime() << " memory = " << stats.memoryDelta() << std::endl;
// #endif
// }

// void DrcEngineManager::filterData()
// {
//   dataPreprocess();

// #ifdef DEBUG_IDRC_ENGINE
//   ieda::Stats stats;

//   std::cout << "idrc : begin scanline" << std::endl;
// #endif
//   /// run scanline method for all routing layers
//   auto& layouts = get_engine_layouts(LayoutType::kRouting);
//   for (auto& [layer, engine_layout] : layouts) {
//     /// scanline engine for each layer
//     auto* scanline_engine = get_engine_scanline(layer, LayoutType::kRouting);
//     scanline_engine->doScanline();
//   }

// #ifdef DEBUG_IDRC_ENGINE
//   std::cout << "idrc : end scanline, "
//             << " runtime = " << stats.elapsedRunTime() << " memory = " << stats.memoryDelta() << std::endl;
// #endif
// }

// get or create scanline engine for each layer
DrcEngineScanline* DrcEngineManager::get_engine_scanline(std::string layer, LayoutType type)
{
  auto& scanline_engines = get_engine_scanlines(type);

  auto* scanline_engine = scanline_engines[layer];
  if (scanline_engine == nullptr) {
    scanline_engine = new DrcEngineScanline(layer, this, _condition_manager);
    scanline_engines[layer] = scanline_engine;
  }

  return scanline_engine;
}

}  // namespace idrc