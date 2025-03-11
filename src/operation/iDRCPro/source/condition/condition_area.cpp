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

void DrcConditionManager::checkArea(std::string layer, DrcEngineLayout* layout)
{
  ieda::Stats states;
  int step_count = 0;
  int enclosed_area_count = 0;
  int area_count = 0;

  auto& layer_polyset = layout->get_layout_engine()->get_polyset();

  // enclosed area
  int rule_min_enclosed_area = DrcTechRuleInst->getMinEnclosedArea(layer);
  if (_check_select.find(ViolationEnumType::kAreaEnclosed) == _check_select.end()) {
    rule_min_enclosed_area = 0;
  }

  // area
  int rule_min_area = DrcTechRuleInst->getMinArea(layer);
  auto rule_lef58_area_list = DrcTechRuleInst->getLef58AreaList(layer);
  int max_rule_lef58_area = 0;
  for (auto& rule_lef58_area : rule_lef58_area_list) {
    max_rule_lef58_area = std::max(max_rule_lef58_area, rule_lef58_area->get_min_area());
  }
  if (_check_select.find(ViolationEnumType::kArea) == _check_select.end()) {
    rule_min_area = 0;
    rule_lef58_area_list.clear();
  }

  // methods
  auto get_edge_orientation = [](const ieda_solver::GeometryPoint& p1, const ieda_solver::GeometryPoint& p2) {
    return p1.x() == p2.x() ? ieda_solver::K_VERTICAL : ieda_solver::K_HORIZONTAL;
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
      if (hole_area < rule_min_enclosed_area) {
        ieda_solver::GeometryPolygon hole_polygon;
        hole_polygon.set(hole_it->begin(), hole_it->end());
        ieda_solver::GeometryRect violation_rect;
        ieda_solver::ENVELOPE(violation_rect, hole_polygon);
        addViolation(violation_rect, layer, ViolationEnumType::kAreaEnclosed);
        ++enclosed_area_count;
      }
#endif
    }

// area
#ifndef DEBUGCLOSE_AREA
    auto test_area1 = boost::polygon::area(polygon);
    if (test_area1 != polygon_area) {
      int a = 0;
      a += 1;
    }
    if (polygon_area < rule_min_area) {
      ieda_solver::GeometryRect violation_rect;
      ieda_solver::ENVELOPE(violation_rect, polygon);
      addViolation(violation_rect, layer, ViolationEnumType::kArea);
      ++area_count;
    } else if (polygon_area < max_rule_lef58_area) {
      std::vector<ieda_solver::GeometryRect> current_polygon_max_rects;
      for (auto& rule_lef58_area : rule_lef58_area_list) {
        if (polygon_area >= rule_lef58_area->get_min_area()) {
          continue;
        }
        auto rule_min_edge_length = rule_lef58_area->get_except_edge_length()->get_min_edge_length();
        auto rule_max_edge_length = rule_lef58_area->get_except_edge_length()->get_max_edge_length();
        if (max_edge_length < rule_min_edge_length || max_edge_length >= rule_max_edge_length) {
          continue;
        }

        auto rule_except_size_width = rule_lef58_area->get_except_min_size()[0].get_min_width();
        auto rule_except_size_length = rule_lef58_area->get_except_min_size()[0].get_min_length();

        if (current_polygon_max_rects.empty()) {
          ieda_solver::getMaxRectangles(current_polygon_max_rects, polygon);
        }

        bool is_ignore = false;
        for (auto& rect : current_polygon_max_rects) {
          auto orientation = ieda_solver::getWireDirection(rect);
          int rect_width = ieda_solver::getWireWidth(rect, orientation.get_perpendicular());
          int rect_length = ieda_solver::getWireWidth(rect, orientation);
          if (rect_width >= rule_except_size_width && rect_length >= rule_except_size_length) {
            is_ignore = true;
            break;
          }
        }

        if (!is_ignore) {
          ieda_solver::GeometryRect violation_rect;
          ieda_solver::ENVELOPE(violation_rect, polygon);
          addViolation(violation_rect, layer, ViolationEnumType::kArea);
          ++area_count;
        }
      }
    }
#endif
  }

  DEBUGOUTPUT(DEBUGHIGHLIGHT("Polygon Filter:\t") << "-\ttime = " << states.elapsedRunTime() << "\tmemory = " << states.memoryDelta()
                                                  << "\tpolygon count = " << polygon_with_holes.size());
#ifndef DEBUGCLOSE_AREA
  DEBUGOUTPUT(DEBUGHIGHLIGHT("Min Area:\t") << area_count);
#endif
}

}  // namespace idrc