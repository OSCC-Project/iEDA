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

#include "condition_manager.h"
#include "engine_layout.h"
#include "idm.h"
#include "omp.h"

namespace idrc {

void DrcConditionManager::checkPolygons(std::string layer, DrcEngineLayout* layout)
{
  ieda::Stats states;
  int step_count = 0;
  int enclosed_area_count = 0;
  int area_count = 0;

  auto& layer_polyset = layout->get_layout_engine()->get_polyset();

#ifdef DEBUG_IDRC_CONDITION
  std::vector<ieda_solver::GeometryViewPolygon> polygons;
  layer_polyset.get(polygons);
#endif

  // eol
  auto rule_eol_list = DrcTechRuleInst->getSpacingEolList(layer);
  unsigned rule_eol_pattern = 0b11;
  unsigned rule_eol_mask = 0b11;
  int max_rule_eol_width = 0;
  for (auto& rule_eol : rule_eol_list) {
    max_rule_eol_width = std::max(max_rule_eol_width, rule_eol->get_eol_width());
  }
  using EolRuleToRegionMap = std::map<std::shared_ptr<idb::routinglayer::Lef58SpacingEol>, ieda_solver::GeometryPolygonSet>;
  EolRuleToRegionMap eol_check_regions;
  EolRuleToRegionMap eol_par_space_regions_left;
  EolRuleToRegionMap eol_par_space_regions_right;
  EolRuleToRegionMap eol_end_to_end_regions;
  if (_check_select.find(ViolationEnumType::kEOL) == _check_select.end()) {
    rule_eol_list.clear();
  }

  // Corner Fill Spacing
  auto rule_corner_fill = DrcTechRuleInst->getCornerFillSpacing(layer);
  std::map<unsigned, std::vector<int>> rule_corner_fill_pattern{{0b1011, {-2, -2, -1, 0}}, {0b1101, {-1, 0, -1, -2}}};
  unsigned rule_corner_fill_mask = 0b1111;
  ieda_solver::GeometryPolygonSet corner_fill_check_regions;
  if (_check_select.find(ViolationEnumType::kCornerFill) == _check_select.end()) {
    rule_corner_fill = nullptr;
  }

  // notch
  auto rule_notch = DrcTechRuleInst->getSpacingNotchlength(layer);
  unsigned rule_notch_pattern = 0b00;
  unsigned rule_notch_mask = 0b11;
  int rule_notch_spacing = rule_notch ? rule_notch->get_min_spacing() : 0;
  ieda_solver::GeometryPolygonSet notch_width_detect_regions;
  ieda_solver::GeometryPolygonSet notch_spacing_check_regions;
  if (_check_select.find(ViolationEnumType::kNotch) == _check_select.end()) {
    rule_notch = nullptr;
  }

  // step
  auto rule_step = DrcTechRuleInst->getMinStep(layer);
  auto rule_lef58_step_list = DrcTechRuleInst->getLef58MinStep(layer);
  int rule_step_length = rule_step ? rule_step->get_min_step_length() : 0;
  int max_rule_lef58_step_length = 0;
  std::vector<unsigned> rule_lef58_step_pattern_list(rule_lef58_step_list.size(), 0);
  std::vector<unsigned> rule_lef58_step_mask_list(rule_lef58_step_list.size(), 0);
  for (size_t i = 0; i < rule_lef58_step_list.size(); ++i) {
    auto& rule_lef58_step = rule_lef58_step_list[i];
    max_rule_lef58_step_length = std::max(max_rule_lef58_step_length, rule_lef58_step->get_min_step_length());
    if (rule_lef58_step->get_min_adjacent_length().has_value() && rule_lef58_step->get_min_adjacent_length().value().is_convex_corner()) {
      rule_lef58_step_mask_list[i] = 0b111;
      rule_lef58_step_pattern_list[i] = 0b010;
    }
  }
  if (_check_select.find(ViolationEnumType::kMinStep) == _check_select.end()) {
    rule_step = nullptr;
    rule_lef58_step_list.clear();
  }

  // enclosed area
  int rule_min_enclosed_area = DrcTechRuleInst->getMinEnclosedArea(layer);
  if (_check_select.find(ViolationEnumType::kAreaEnclosed) == _check_select.end()) {
    rule_min_enclosed_area = 0;
  }

  // methods
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

  // filter polygon edges
  auto& polygon_with_holes = layout->get_layout_engine()->getLayoutPolygons();
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
    int max_edge_length = 0;
    int min_edge_length = std::numeric_limits<int>::max();

    auto it_next = polygon.begin();
    auto it_prev = it_next++;
    auto it_current = it_next++;
    do {
      int edge_length = ieda_solver::rectManhattanDistance(*it_current, *it_prev);
      bool is_current_convex = is_convex(get_edge_direction(*it_prev, *it_current), get_edge_direction(*it_current, *it_next));
      ieda_solver::GeometryOrientation edge_orientation = get_edge_orientation(*it_prev, *it_current);
      ieda_solver::GeometryDirection2D edge_direction = get_edge_direction(*it_prev, *it_current);

      // record polygon outline
      polygon_outline[corner_index] = *it_current;
      edge_length_history[corner_index] = edge_length;
      corner_convex_history[corner_index] = is_current_convex;
      edge_orientation_history[corner_index] = edge_orientation;
      edge_direction_history[corner_index] = edge_direction;

      max_edge_length = std::max(edge_length, max_edge_length);
      min_edge_length = std::min(edge_length, min_edge_length);

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
    auto create_corner_pattern = [&](int count, int index) {
      unsigned pattern = 0;
      for (int i = count - 1; i >= 0; --i) {
        auto idx = get_index_shifted(index, -i);
        pattern <<= 1;
        pattern |= corner_convex_history[idx];
      }
      return pattern;
    };

    auto corner_pattern_4 = create_corner_pattern(4, polygon_point_number - 1);
    int count_step_checked_edges = 0;

    for (int point_current_index = 0; point_current_index < polygon_point_number; ++point_current_index) {
      auto point_index_prev = get_index_shifted(point_current_index, -1);
      auto point_current = polygon_outline[point_current_index];
      auto point_prev = polygon_outline[point_index_prev];
      auto is_current_convex = corner_convex_history[point_current_index];
      auto edge_length = edge_length_history[point_current_index];
      auto edge_orientation = edge_orientation_history[point_current_index];
      auto edge_direction = edge_direction_history[point_current_index];

      corner_pattern_4 = (corner_pattern_4 << 1) | is_current_convex;

      // eol
      if ((corner_pattern_4 & rule_eol_mask) == rule_eol_pattern && edge_length < max_rule_eol_width) {
        for (auto& rule_eol : rule_eol_list) {
          if (edge_length >= rule_eol->get_eol_width()) {
            continue;
          }

          int eol_spacing = rule_eol->get_eol_space();
          int eol_within = rule_eol->get_eol_within().value_or(0);

          // TODO: ENCLOSE CUT
          if (rule_eol->get_enclose_cut().has_value()) {
            continue;
          }

          // PARALLELEDGE
          if (rule_eol->get_parallel_edge().has_value()) {
            auto rule_par_edge = rule_eol->get_parallel_edge().value();

            // MINLENGTH
            auto min_edge_length = rule_par_edge.get_min_length().value_or(0);
            if (edge_length_history[get_index_shifted(point_current_index, 1)] < min_edge_length
                || edge_length_history[point_index_prev] < min_edge_length) {
              continue;
            }

            int rule_par_spacing = rule_par_edge.get_par_space();
            if (rule_par_edge.is_subtract_eol_width()) {
              rule_par_spacing -= edge_length;
            }
            int rule_par_within = rule_par_edge.get_par_within();

            ieda_solver::GeometryRect detect_rect_left(point_current.x(), point_current.y(), point_current.x(), point_current.y());
            ieda_solver::bloat(detect_rect_left, edge_direction, rule_par_spacing);
            ieda_solver::bloat(detect_rect_left, edge_direction.left(), rule_par_within);
            ieda_solver::bloat(detect_rect_left, edge_direction.right(), eol_within);
            eol_par_space_regions_left[rule_eol] += detect_rect_left;

            ieda_solver::GeometryRect detect_rect_right(point_prev.x(), point_prev.y(), point_prev.x(), point_prev.y());
            ieda_solver::bloat(detect_rect_right, edge_direction.backward(), rule_par_spacing);
            ieda_solver::bloat(detect_rect_right, edge_direction.left(), rule_par_within);
            ieda_solver::bloat(detect_rect_right, edge_direction.right(), eol_within);
            eol_par_space_regions_right[rule_eol] += detect_rect_right;
          }

          // TODO: ENDTOEND

          // eol spacing check window
          ieda_solver::GeometryRect check_rect(point_prev.x(), point_prev.y(), point_current.x(), point_current.y());
          ieda_solver::bloat(check_rect, edge_direction.right(), eol_spacing);
          ieda_solver::bloat(check_rect, edge_orientation, eol_within);
          eol_check_regions[rule_eol] += check_rect;
        }
      }

      // Corner Fill Spacing
      if (rule_corner_fill) {
        for (auto [pattern, offset] : rule_corner_fill_pattern) {
          if ((corner_pattern_4 & rule_corner_fill_mask) == pattern) {
            int rule_corner_fill_spacing = rule_corner_fill->get_spacing();
            int rule_corner_fill_length1 = rule_corner_fill->get_edge_length1();
            int rule_corner_fill_length2 = rule_corner_fill->get_edge_length2();
            int rule_corner_fill_eol_width = rule_corner_fill->get_eol_width();

            int corner_index = get_index_shifted(point_current_index, offset[0]);
            int edge1_index = get_index_shifted(point_current_index, offset[1]);
            int edge2_index = get_index_shifted(point_current_index, offset[2]);
            int eol_index = get_index_shifted(point_current_index, offset[3]);

            if (edge_length_history[edge1_index] < rule_corner_fill_length1 && edge_length_history[edge2_index] < rule_corner_fill_length2
                && edge_length_history[eol_index] < rule_corner_fill_eol_width) {
              auto& corner_point = polygon_outline[corner_index];
              ieda_solver::GeometryRect check_rect(corner_point.x(), corner_point.y(), corner_point.x(), corner_point.y());
              ieda_solver::bloat(check_rect, edge_direction_history[edge1_index].right(),
                                 edge_length_history[edge2_index] + rule_corner_fill_spacing);
              ieda_solver::bloat(check_rect, edge_direction_history[edge2_index].right(),
                                 edge_length_history[edge1_index] + rule_corner_fill_spacing);
              corner_fill_check_regions += check_rect;
            }

            break;
          }
        }
      }

      // notch
      if (rule_notch) {
        if ((corner_pattern_4 & rule_notch_mask) == rule_notch_pattern && edge_length <= rule_notch_spacing) {
          int notch_side1_idx = get_index_shifted(point_current_index, 1);
          int notch_side2_idx = get_index_shifted(point_current_index, -1);
          int rule_notch_length = rule_notch->get_min_notch_length();
          bool is_violation = false;
          int notch_length = 0;
          if (rule_notch->get_concave_ends_side_of_notch_width().has_value()) {
            if ((!corner_convex_history[get_index_shifted(point_current_index, 1)]
                 && edge_length_history[notch_side1_idx] < rule_notch_length && edge_length_history[notch_side2_idx] >= rule_notch_length)
                || (!corner_convex_history[get_index_shifted(point_current_index, -2)]
                    && edge_length_history[notch_side2_idx] < rule_notch_length
                    && edge_length_history[notch_side1_idx] >= rule_notch_length)) {
              is_violation = true;
              // TODO: both side should be smaller than notch width
              auto rule_notch_width = rule_notch->get_concave_ends_side_of_notch_width().value();
              ieda_solver::GeometryRect detect_rect(point_current.x(), point_current.y(), point_prev.x(), point_prev.y());
              ieda_solver::bloat(detect_rect, edge_direction.right(), 1);
              auto subtract_rect = detect_rect;
              ieda_solver::bloat(detect_rect, edge_orientation, rule_notch_width + 1);
              notch_width_detect_regions += detect_rect;
              notch_width_detect_regions -= subtract_rect;
            }
          } else if (edge_length_history[notch_side1_idx] < rule_notch_length || edge_length_history[notch_side2_idx] < rule_notch_length) {
            is_violation = true;
          }
          if (is_violation) {
            notch_length = std::min(edge_length_history[notch_side1_idx], edge_length_history[notch_side2_idx]);
            ieda_solver::GeometryRect check_rect(point_current.x(), point_current.y(), point_prev.x(), point_prev.y());
            ieda_solver::bloat(check_rect, edge_direction.right(), notch_length);
            notch_spacing_check_regions += check_rect;
          }
        }
      }

// step
#ifndef DEBUGCLOSE_STEP
      if (rule_step) {
        if (count_step_checked_edges) {
          --count_step_checked_edges;
        } else if (edge_length < rule_step_length) {
          int rule_max_edges = rule_step->get_max_edges();
          for (int i = 1; i < polygon_point_number; ++i) {
            ++count_step_checked_edges;
            if (edge_length_history[get_index_shifted(point_current_index, i)] >= rule_step_length) {
              break;
            }
          }
          if (count_step_checked_edges > rule_max_edges) {
            auto point1_idx = get_index_shifted(point_current_index, -1);
            auto point2_idx = get_index_shifted(point_current_index, count_step_checked_edges - 1);
            ieda_solver::GeometryRect violation_rect(polygon_outline[point1_idx].x(), polygon_outline[point1_idx].y(),
                                                     polygon_outline[point2_idx].x(), polygon_outline[point2_idx].y());
            addViolation(violation_rect, layer, ViolationEnumType::kMinStep);
            ++step_count;
          }
        }
      }

      // lef58 step
      if (edge_length < max_rule_lef58_step_length) {
        for (size_t i = 0; i < rule_lef58_step_list.size(); ++i) {
          if ((corner_pattern_4 & rule_lef58_step_mask_list[i]) == rule_lef58_step_pattern_list[i]) {
            // todo: what MAXEDGES mean here?
            auto rule_lef58_step = rule_lef58_step_list[i];
            int rule_edge_length = rule_lef58_step->get_min_step_length();
            if (rule_lef58_step->get_min_adjacent_length().has_value()) {
              auto rule_adjacent_length = rule_lef58_step->get_min_adjacent_length().value();
              int rule_min_adjacent_length = rule_adjacent_length.get_min_adj_length();
              if ((edge_length_history[point_current_index] < rule_edge_length
                   && edge_length_history[point_index_prev] < rule_min_adjacent_length)
                  || (edge_length_history[point_current_index] < rule_edge_length
                      && edge_length_history[point_index_prev] < rule_min_adjacent_length)) {
                int point_prev_prev_idx = get_index_shifted(point_current_index, -2);
                ieda_solver::GeometryRect violation_rect(polygon_outline[point_prev_prev_idx].x(), polygon_outline[point_prev_prev_idx].y(),
                                                         point_current.x(), point_current.y());
                addViolation(violation_rect, layer, ViolationEnumType::kMinStep);
                ++step_count;
              }
            } else {
              // todo
            }
          }
        }
      }
#endif
    }

    // polygon holes
    // TODO: holes need to check edge?
    for (auto hole_it = polygon.begin_holes(); hole_it != polygon.end_holes(); ++hole_it) {
      long long hole_area = 0;
      auto hole_pt_it_next = hole_it->begin();
      auto hole_pt_it_prev = hole_pt_it_next++;
      auto hole_pt_it_current = hole_pt_it_next++;
      do {
        // refresh area
        area_calculator(hole_area, *hole_pt_it_prev, *hole_pt_it_current);

        // next segment
        hole_pt_it_prev = hole_pt_it_current;
        hole_pt_it_current = hole_pt_it_next;
        ++hole_pt_it_next;
        if (hole_pt_it_next == hole_it->end()) {
          hole_pt_it_next = hole_it->begin();
        }
      } while (hole_pt_it_prev != hole_it->begin());

      hole_area = std::abs(hole_area) / 2;
      polygon_area -= hole_area;

// min enclosed area
#ifndef DEBUGCLOSE_HOLE
      // if (hole_area < rule_min_enclosed_area) {
      //   ieda_solver::GeometryPolygon hole_polygon;
      //   hole_polygon.set(hole_it->begin(), hole_it->end());
      //   ieda_solver::GeometryRect violation_rect;
      //   ieda_solver::envelope(violation_rect, hole_polygon);
      //   addViolation(violation_rect, layer, ViolationEnumType::kAreaEnclosed);
      //   ++enclosed_area_count;
      // }
#endif
    }
  }

  DEBUGOUTPUT(DEBUGHIGHLIGHT("Polygon Filter:\t") << "-\ttime = " << states.elapsedRunTime() << "\tmemory = " << states.memoryDelta()
                                                  << "\tpolygon count = " << polygon_with_holes.size());
#ifndef DEBUGCLOSE_STEP
  DEBUGOUTPUT(DEBUGHIGHLIGHT("Min Step:\t") << step_count);
#endif
#ifndef DEBUGCLOSE_HOLE
  DEBUGOUTPUT(DEBUGHIGHLIGHT("Enclosed Area:\t") << enclosed_area_count);
#endif
#ifndef DEBUGCLOSE_AREA
  DEBUGOUTPUT(DEBUGHIGHLIGHT("Min Area:\t") << area_count);
#endif

// eol
#ifndef DEBUGCLOSE_EOL
  ieda::Stats states_eol;
  int eol_count = 0;
  int last_eol_count = 0;
  auto remove_through_detect_regions = [](ieda_solver::GeometryPolygonSet& regions_to_remove, int shrink_size) {
    ieda_solver::GeometryPolygonSet par_shrink_vertical(regions_to_remove);
    ieda_solver::shrink(par_shrink_vertical, ieda_solver::VERTICAL, shrink_size);
    ieda_solver::GeometryPolygonSet par_shrink_horizontal(regions_to_remove);
    ieda_solver::shrink(par_shrink_horizontal, ieda_solver::HORIZONTAL, shrink_size);
    ieda_solver::GeometryPolygonSet par_shrink = par_shrink_vertical | par_shrink_horizontal;
    auto par_regions_to_remove = regions_to_remove;
    ieda_solver::get_interact(par_regions_to_remove, par_shrink);
    regions_to_remove -= par_regions_to_remove;
  };
  for (auto& rule_eol : rule_eol_list) {
    ieda::Stats states_eol_rule;
    auto& check_regions = eol_check_regions[rule_eol];

    auto data_to_check = layer_polyset;

    if (rule_eol->get_parallel_edge().has_value()) {
      auto rule_parallel_edge = rule_eol->get_parallel_edge().value();

      ieda_solver::GeometryPolygonSet par_space_regions;

      if (rule_parallel_edge.is_same_metal()) {
        // remove wire through a detect region
        int shrink_size = (rule_parallel_edge.get_par_within() + rule_eol->get_eol_within().value_or(0)) / 2 - 1;
        remove_through_detect_regions(eol_par_space_regions_left[rule_eol], shrink_size);
        remove_through_detect_regions(eol_par_space_regions_right[rule_eol], shrink_size);

        // TOWEDGES
        if (rule_parallel_edge.is_two_edges()) {
          ieda_solver::GeometryPolygonSet check_regions_left(check_regions);
          ieda_solver::GeometryPolygonSet check_regions_right(check_regions);
          ieda_solver::get_interact(check_regions_left, eol_par_space_regions_left[rule_eol]);
          ieda_solver::get_interact(check_regions_right, eol_par_space_regions_right[rule_eol]);
          check_regions = check_regions_left & check_regions_right;
        }

        // remain polygons could be same metal
        par_space_regions = eol_par_space_regions_left[rule_eol] | eol_par_space_regions_right[rule_eol];
        ieda_solver::get_interact(data_to_check, par_space_regions);
      } else {
        par_space_regions = eol_par_space_regions_left[rule_eol] | eol_par_space_regions_right[rule_eol];
      }

      // remove detect regions that not contain any par edges
      ieda_solver::GeometryPolygonSet detected_wires = par_space_regions & data_to_check;
      ieda_solver::get_interact(par_space_regions, detected_wires);

      // remain check regions contain par edges
      ieda_solver::get_interact(check_regions, par_space_regions);
    }

    // remain check regions contain eol spacing violations
    ieda_solver::GeometryPolygonSet violation_wires = check_regions & data_to_check;
    ieda_solver::get_interact(check_regions, violation_wires);
    ieda_solver::GeometryPolygonSet violation_regions = check_regions - violation_wires;

    // create violations
    std::vector<ieda_solver::GeometryPolygon> violation_polygons;
    violation_regions.get(violation_polygons);
    for (auto& violation_polygon : violation_polygons) {
      ieda_solver::GeometryRect violation_rect;
      ieda_solver::envelope(violation_rect, violation_polygon);
      if (violation_polygon.size() <= 4
          && ieda_solver::getWireWidth(violation_rect, ieda_solver::getWireDirection(violation_rect).get_perpendicular())
                 >= rule_eol->get_eol_space()) {
        continue;
      }
      addViolation(violation_rect, layer, ViolationEnumType::kEOL);
      ++eol_count;
    }

    // DEBUGOUTPUT(DEBUGHIGHLIGHT("EOL Spacing:\t") << eol_count - last_eol_count << "\ttime = " << states_eol_rule.elapsedRunTime()
    //                                              << "\tmemory = " << states_eol_rule.memoryDelta());
    last_eol_count = eol_count;
  }
  DEBUGOUTPUT(DEBUGHIGHLIGHT("EOL Spacing:\t") << eol_count << "\ttime = " << states_eol.elapsedRunTime()
                                               << "\tmemory = " << states_eol.memoryDelta());
#endif

// Corner Fill Spacing
#ifndef DEBUGCLOSE_CORNER_FILL
  ieda::Stats states_corner_fill;
  int corner_fill_count = 0;
  if (rule_corner_fill) {
    ieda_solver::GeometryPolygonSet violation_wires = corner_fill_check_regions & layer_polyset;
    ieda_solver::get_interact(corner_fill_check_regions, violation_wires);
    // ieda_solver::GeometryPolygonSet result_regions = corner_fill_check_regions - violation_wires;

    std::vector<ieda_solver::GeometryRect> violation_rects;
    corner_fill_check_regions.get(violation_rects);

    for (auto& violation_rect : violation_rects) {
      addViolation(violation_rect, layer, ViolationEnumType::kCornerFill);
      ++corner_fill_count;
    }
  }
  DEBUGOUTPUT(DEBUGHIGHLIGHT("Corner Fill Spacing:\t") << corner_fill_count << "\ttime = " << states_corner_fill.elapsedRunTime()
                                                       << "\tmemory = " << states_corner_fill.memoryDelta());
#endif

// notch
#ifndef DEBUGCLOSE_NOTCH
  ieda::Stats states_notch;
  int notch_count = 0;
  if (rule_notch) {
    if (rule_notch->get_concave_ends_side_of_notch_width().has_value()) {
      auto detect_regions = notch_width_detect_regions & layer_polyset;
      ieda_solver::GeometryPolygonSet remained_detect_regions = notch_width_detect_regions ^ detect_regions;
      ieda_solver::get_interact(notch_width_detect_regions, remained_detect_regions);
      ieda_solver::get_interact(notch_spacing_check_regions, notch_width_detect_regions);
    }

    std::vector<ieda_solver::GeometryRect> notch_violations;
    notch_spacing_check_regions.get(notch_violations);

    for (auto& violation_rect : notch_violations) {
      addViolation(violation_rect, layer, ViolationEnumType::kNotch);
    }
    notch_count += notch_violations.size();
  }
  DEBUGOUTPUT(DEBUGHIGHLIGHT("Notch Spacing:\t")
              << notch_count << "\ttime = " << states_notch.elapsedRunTime() << "\tmemory = " << states_notch.memoryDelta());
#endif
}

}  // namespace idrc