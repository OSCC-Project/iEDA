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

#include <algorithm>
#include <cassert>
#include <cstdio>

#include "condition_manager.h"
#include "engine_layout.h"
#include "geometry_polygon.h"
#include "geometry_polygon_set.h"
#include "idm.h"

namespace idrc {

void DrcConditionManager::checkParallelLengthSpacing(std::string layer, DrcEngineLayout* layout)
{
  if (_check_select.find(ViolationEnumType::kPRLSpacing) == _check_select.end()) {
    return;
  }

  using WidthToPolygonSetMap = std::map<int, ieda_solver::GeometryPolygonSet>;
  std::map<int, std::vector<ieda_solver::GeometryRect>> prl_wire_map;
  std::map<int, std::vector<ieda_solver::GeometryPolygonSet>> prl_polygon_map;
  buildMapOfSpacingTable(layer, layout, prl_wire_map, prl_polygon_map);
  checkSpacingTable(layer, layout, prl_wire_map, prl_polygon_map);
}

void DrcConditionManager::buildMapOfSpacingTable(std::string layer, DrcEngineLayout* layout,
                                                 std::map<int, std::vector<ieda_solver::GeometryRect>>& prl_wire_map,
                                                 std::map<int, std::vector<ieda_solver::GeometryPolygonSet>>& prl_polygon_map)
{
  ieda::Stats states;

  auto rule_spacing_table = DrcTechRuleInst->getSpacingTable(layer);
  int internel_prl_count = 0;
  for (auto& [net_id, sub_layout] : layout->get_sub_layouts()) {
    std::vector<ieda_solver::GeometryPolygon> polygon_lists;
    sub_layout->get_engine()->get_polyset().get(polygon_lists);
    for (auto& polygon : polygon_lists) {
      std::vector<ieda_solver::GeometryRect> wire_list;
      ieda_solver::GeometryPolygonSet polygon_set;
      polygon_set += polygon;

      boost::polygon::get_max_rectangles(wire_list, polygon_set);  // layout->get_layout_engine()->getWires();
      bg::index::rtree<std::pair<ieda_solver::BgRect, int>, bg::index::quadratic<16>> wireRTree;

      for (int i = 0; i < wire_list.size(); i++) {
        auto rect_tmp = wire_list[i];
        ieda_solver::BgRect rtree_rect(ieda_solver::BgPoint(boost::polygon::xl(rect_tmp), boost::polygon::yl(rect_tmp)),
                                       ieda_solver::BgPoint(boost::polygon::xh(rect_tmp), boost::polygon::yh(rect_tmp)));
        wireRTree.insert(std::make_pair(rtree_rect, i));
        // whole_wire_polyset += rect_tmp;
      }

      for (int i = 0; i < wire_list.size(); i++) {
        auto wire = wire_list[i];
        auto wire_direction = ieda_solver::getWireDirection(wire);
        auto width_direction = wire_direction.get_perpendicular();
        int wire_width = ieda_solver::getWireWidth(wire, width_direction);

        // prl
        if (rule_spacing_table && rule_spacing_table->is_parallel()) {
          auto idb_table_prl = rule_spacing_table->get_parallel();
          auto& idb_width_list = idb_table_prl->get_width_list();

          auto& idb_prl_length_list = idb_table_prl->get_parallel_length_list();
          auto& idb_spacing_array = idb_table_prl->get_spacing_table();

          int width_idx = 0;
          for (int j = idb_width_list.size() - 1; j >= 0; --j) {
            if (wire_width >= idb_width_list[j]) {
              width_idx = j;
              // self polygon
              // if (width_idx > 0) {

#if 1
              int expand_size = idb_spacing_array[width_idx][idb_prl_length_list.size() - 1];
              ieda_solver::BgRect rtree_rect(
                  ieda_solver::BgPoint(boost::polygon::xl(wire) - expand_size, boost::polygon::yl(wire) - expand_size),
                  ieda_solver::BgPoint(boost::polygon::xh(wire) + expand_size, boost::polygon::yh(wire) + expand_size));

              std::vector<std::pair<ieda_solver::BgRect, int>> result;
              wireRTree.query(bg::index::intersects(rtree_rect), std::back_inserter(result));

              for (auto [rtree_rect, idx] : result) {
                auto rect_intersect = wire;
                auto wire_b = wire_list[idx];
                auto distX = gtl::euclidean_distance(rect_intersect, wire_b, ieda_solver::HORIZONTAL);
                auto distY = gtl::euclidean_distance(rect_intersect, wire_b, ieda_solver::VERTICAL);

                gtl::generalized_intersect(rect_intersect, wire_b);
                auto prlX = gtl::delta(rect_intersect, ieda_solver::HORIZONTAL);
                auto prlY = gtl::delta(rect_intersect, ieda_solver::VERTICAL);
                if (distX == 0 && distY == 0) {
                  continue;
                }
                prlX = distX == 0 ? prlX : 0;
                prlY = distY == 0 ? prlY : 0;
                auto prl_dir = distX == 0 ? ieda_solver::HORIZONTAL : ieda_solver::VERTICAL;
                auto dis_dir = distX == 0 ? ieda_solver::VERTICAL : ieda_solver::HORIZONTAL;
                bool is_corner = prlX == 0 && prlY == 0;
                ieda_solver::GeometryPolygonSet tmp_set;
                tmp_set += rect_intersect;
                std::vector<std::pair<ieda_solver::BgRect, int>> result2;
                wireRTree.query(bg::index::intersects(rect_intersect), std::back_inserter(result2));
                for (auto [rtree_rect2, idx2] : result2) {
                  tmp_set -= wire_list[idx2];
                  // self polygon
                }
                std::vector<ieda_solver::GeometryRect> rect_list;
                tmp_set.get(rect_list);
                for (auto rect : rect_list) {
                  int prl = gtl::delta(rect, prl_dir);
                  if (is_corner) {
                    prl = 0;
                  }
                  int dis = gtl::delta(rect, dis_dir);
                  for (int k = idb_prl_length_list.size() - 1; k >= 0; k--) {
                    if (prl >= idb_prl_length_list[k] && dis <= idb_spacing_array[width_idx][k]) {
                      ieda_solver::GeometryRect violation_rect(rect);
                      ieda_solver::bloat(violation_rect, prl_dir, idb_prl_length_list[k] - prl);
                      addViolation(violation_rect, layer, ViolationEnumType::kPRLSpacing);
                      internel_prl_count += 1;
                      break;
                    }
                  }
                }
              }
#endif

              // }
              break;
            }
          }
          if (width_idx > 0) {
            if (prl_wire_map.count(width_idx) == 0) {
              prl_wire_map[width_idx] = {};
              prl_polygon_map[width_idx] = {};
            }
            prl_wire_map[width_idx].push_back(wire);
            prl_polygon_map[width_idx].push_back(polygon_set);
          }
        }
      }
    }
  }
  DEBUGOUTPUT(DEBUGHIGHLIGHT("Internal PRL Spacing:\t")
              << internel_prl_count << "\ttime = " << states.elapsedRunTime() << "\tmemory = " << states.memoryDelta());
}

void DrcConditionManager::checkSpacingTable(std::string layer, DrcEngineLayout* layout,
                                            std::map<int, std::vector<ieda_solver::GeometryRect>>& prl_wire_map,
                                            std::map<int, std::vector<ieda_solver::GeometryPolygonSet>>& prl_polygon_map)
{
  ieda::Stats states;
  int prl_count = 0;
  auto rule_spacing_table = DrcTechRuleInst->getSpacingTable(layer);
  if (rule_spacing_table && rule_spacing_table->is_parallel()) {
    auto& layer_polyset = layout->get_layout_engine()->get_polyset();
    auto idb_table_prl = rule_spacing_table->get_parallel();

    auto& idb_prl_length_list = idb_table_prl->get_parallel_length_list();
    auto& idb_spacing_array = idb_table_prl->get_spacing_table();

    auto prl_length_list = idb_prl_length_list;
    if (prl_length_list.size() >= 2) {
      prl_length_list[0] = prl_length_list[1];  // t28 wide metal space rule summary table
    } else {
      prl_length_list[0] = 0;
    }
    // FIXME: prl only support 2 length
    for (auto& [width_idx, wire_list] : prl_wire_map) {
      int prl_idx = width_idx - 1;
      int expand_size = idb_spacing_array[width_idx][prl_length_list.size() - 1];
      int required_prl = prl_length_list[prl_idx];
      auto& prl_polygon_map_t = prl_polygon_map[width_idx];

      auto check_by_direction = [&](ieda_solver::GeometryOrientation direction, bool is_corner) {
        ieda_solver::GeometryPolygonSet wire_set;
        ieda_solver::GeometryPolygonSet check_region;
        for (int i = 0; i < wire_list.size(); i++) {
          ieda_solver::GeometryPolygonSet expand_wire;
          expand_wire += wire_list[i];
          wire_set += wire_list[i];
          auto wire_with_jogs = prl_polygon_map_t[i];
          if (!is_corner) {
            ieda_solver::bloat(expand_wire, direction, expand_size);
            auto expand_region = expand_wire - wire_with_jogs;
            check_region += expand_region;
          } else {
            ieda_solver::GeometryPolygonSet p_1 = expand_wire;
            ieda_solver::bloat(p_1, ieda_solver::HORIZONTAL, expand_size);
            ieda_solver::GeometryPolygonSet p_2 = expand_wire;
            ieda_solver::bloat(p_2, ieda_solver::VERTICAL, expand_size);
            ieda_solver::bloat(expand_wire, ieda_solver::VERTICAL, expand_size);
            ieda_solver::bloat(expand_wire, ieda_solver::HORIZONTAL, expand_size);
            // ieda_solver::gtl::bloat(expand_wire, expand_size);
            auto expand_region = expand_wire - p_1;
            auto expand_region2 = expand_region - p_2;
            ieda_solver::GeometryPolygonSet expand_region3 = expand_region2 - wire_with_jogs;
            std::vector<ieda_solver::GeometryViewPolygon> view_polygons;
            expand_region3.get(view_polygons);
            check_region += expand_region3;
            // self polygon
          }
        }

        check_region.clean();
        // std::vector<ieda_solver::GeometryViewPolygon> view_polygons1;
        // check_region.get(view_polygons1);
        check_region = check_region & layer_polyset;
        // std::vector<ieda_solver::GeometryViewPolygon> view_polygons2;
        // check_region.get(view_polygons2);
        std::vector<ieda_solver::GeometryRect> check_region_rects;
        ieda_solver::gtl::get_rectangles(check_region_rects, check_region);

        if (is_corner) {
          ieda_solver::GeometryPolygonSet violation_region_set;
          for (auto& rect : check_region_rects) {
            int length = ieda_solver::getWireWidth(rect, direction);
            int prl = ieda_solver::getWireWidth(rect, direction.get_perpendicular());

            ieda_solver::bloat(rect, direction, expand_size - length);
            ieda_solver::bloat(rect, direction.get_perpendicular(), expand_size - prl);

            violation_region_set += rect;
          }
          ieda_solver::GeometryPolygonSet touch_wire_region(violation_region_set - check_region);
          ieda_solver::get_interact(touch_wire_region, wire_set);
          touch_wire_region = touch_wire_region - wire_set;
          std::vector<ieda_solver::GeometryRect> current_violations;
          touch_wire_region.get(current_violations);

          DEBUGOUTPUT(DEBUGHIGHLIGHT("PRL Spacing violations:\t") << current_violations.size());

          for (auto& rect : current_violations) {
            addViolation(rect, layer, ViolationEnumType::kPRLSpacing);
          }
          prl_count += current_violations.size();

        } else {
          ieda_solver::GeometryPolygonSet violation_region_set;
          for (auto& rect : check_region_rects) {
            int length = ieda_solver::getWireWidth(rect, direction);
            int prl = ieda_solver::getWireWidth(rect, direction.get_perpendicular());

            ieda_solver::bloat(rect, direction, expand_size - length + 1);
            violation_region_set += rect;
          }
          ieda_solver::GeometryPolygonSet touch_wire_region(violation_region_set - check_region);
          ieda_solver::get_interact(touch_wire_region, wire_set);
          touch_wire_region = touch_wire_region - wire_set;

          std::vector<ieda_solver::GeometryRect> current_violations;
          touch_wire_region.get(current_violations);

          DEBUGOUTPUT(DEBUGHIGHLIGHT("PRL Spacing violations:\t") << current_violations.size());

          for (auto& rect : current_violations) {
            addViolation(rect, layer, ViolationEnumType::kPRLSpacing);
          }
          prl_count += current_violations.size();
        }
      };

      check_by_direction(ieda_solver::HORIZONTAL, false);
      check_by_direction(ieda_solver::VERTICAL, false);
      check_by_direction(ieda_solver::VERTICAL, true);
    }
  }
  DEBUGOUTPUT(DEBUGHIGHLIGHT("PRL Spacing:\t") << prl_count << "\ttime = " << states.elapsedRunTime()
                                               << "\tmemory = " << states.memoryDelta());
}

}  // namespace idrc