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

namespace idrc {

void DrcConditionManager::checkJogToJogSpacing(std::string layer, DrcEngineLayout* layout)
{
  using WidthToPolygonSetMap = std::map<int, ieda_solver::GeometryPolygonSet>;
  WidthToPolygonSetMap jog_wire_map;
  buildMapOfJog(layer, layout, jog_wire_map);

  checkJog(layer, layout, jog_wire_map);
}

void DrcConditionManager::buildMapOfJog(std::string layer, DrcEngineLayout* layout,
                                        std::map<int, ieda_solver::GeometryPolygonSet>& jog_wire_map)
{
  ieda::Stats states;
  auto rule_jog_to_jog = DrcTechRuleInst->getJogToJog(layer);

  auto& wires = layout->get_layout_engine()->getWires();
  for (auto& wire : wires) {
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
  }
  DEBUGOUTPUT(DEBUGHIGHLIGHT("Wire Filter:\t") << "-\ttime = " << states.elapsedRunTime() << "\tmemory = " << states.memoryDelta()
                                               << "\twire count = " << wires.size());
}

void DrcConditionManager::checkJog(std::string layer, DrcEngineLayout* layout, std::map<int, ieda_solver::GeometryPolygonSet>& jog_wire_map)
{
  if (_check_select.find(ViolationEnumType::kJogToJog) == _check_select.end()) {
    return;
  }
  ieda::Stats states;
  int jog_count = 0;
  auto rule_jog_to_jog = DrcTechRuleInst->getJogToJog(layer);
  if (rule_jog_to_jog) {
    auto& layer_polyset = layout->get_layout_engine()->get_polyset();
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
            ieda_solver::get_interact(wire_with_jogs, jog_wires);
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
            ieda_solver::get_interact(split_check_rects_set, wire_with_jogs);
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
                  // distance
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

    for (auto& rect : jog_violations) {
      addViolation(rect, layer, ViolationEnumType::kJogToJog);
    }
    jog_count = jog_violations.size();
  }
  DEBUGOUTPUT(DEBUGHIGHLIGHT("Jog Spacing:\t") << jog_count << "\ttime = " << states.elapsedRunTime()
                                               << "\tmemory = " << states.memoryDelta());
}

}  // namespace idrc