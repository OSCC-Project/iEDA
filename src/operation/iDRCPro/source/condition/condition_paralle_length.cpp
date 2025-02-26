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

#include <cstdio>

#include "condition_manager.h"
#include "engine_layout.h"
#include "idm.h"

namespace idrc {

void DrcConditionManager::checkParallelLengthSpacing(std::string layer, DrcEngineLayout* layout)
{
  if (_check_select.find(ViolationEnumType::kPRLSpacing) == _check_select.end()) {
    return;
  }

  using WidthToPolygonSetMap = std::map<int, ieda_solver::GeometryPolygonSet>;
  WidthToPolygonSetMap prl_wire_map;
  WidthToPolygonSetMap prl_polygon_map;
  buildMapOfSpacingTable(layer, layout, prl_wire_map, prl_polygon_map);
  checkSpacingTable(layer, layout, prl_wire_map, prl_polygon_map);
}

void DrcConditionManager::buildMapOfSpacingTable(std::string layer, DrcEngineLayout* layout,
                                                 std::map<int, ieda_solver::GeometryPolygonSet>& prl_wire_map,
                                                 std::map<int, ieda_solver::GeometryPolygonSet>& prl_polygon_map)
{
  ieda::Stats states;

  auto rule_spacing_table = DrcTechRuleInst->getSpacingTable(layer);

  for (auto& [net_id, sub_layout] : layout->get_sub_layouts()) {
    std::vector<ieda_solver::GeometryRect> wire_list;
    boost::polygon::get_max_rectangles(wire_list, sub_layout->get_engine()->get_polyset());  // layout->get_layout_engine()->getWires();
    for (auto wire : wire_list) {
      auto wire_direction = ieda_solver::getWireDirection(wire);
      auto width_direction = wire_direction.get_perpendicular();
      int wire_width = ieda_solver::getWireWidth(wire, width_direction);

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
          prl_polygon_map[width_idx] += sub_layout->get_engine()->get_polyset();
        }
      }
    }
  }
  // DEBUGOUTPUT(DEBUGHIGHLIGHT("Wire Filter:\t") << "-\ttime = " << states.elapsedRunTime() << "\tmemory = " << states.memoryDelta()
  //                                              << "\twire count = " << wires.size());
}

void DrcConditionManager::checkSpacingTable(std::string layer, DrcEngineLayout* layout,
                                            std::map<int, ieda_solver::GeometryPolygonSet>& prl_wire_map,
                                            std::map<int, ieda_solver::GeometryPolygonSet>& prl_polygon_map)
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

    for (auto& [width_idx, wire_set] : prl_wire_map) {
      int prl_idx = width_idx - 1;
      int expand_size = idb_spacing_array[width_idx][prl_length_list.size() - 1];
      int required_prl = prl_length_list[prl_idx];

      auto check_by_direction = [&](ieda_solver::GeometryOrientation direction) {
        auto expand_wires = wire_set;
        ieda_solver::bloat(expand_wires, direction, expand_size);
        auto wire_with_jogs = prl_polygon_map[width_idx];
#ifdef DEBUG
        auto wire_with_jogs = layer_polyset;
        // FIXME: THIS STEP IS RUNTIME EXPENSIVE
        ieda_solver::get_interact(wire_with_jogs, wire_set);
        auto wire_with_jogs = prl_polygon_map[width_idx];
        ieda_solver::GeometryPolygonSet xor_check = wire_set_interact_test ^ wire_with_jogs;
        std::vector<ieda_solver::GeometryRect> xor_check_list;
        gtl::get_rectangles(xor_check_list, xor_check);
        if (xor_check_list.size() > 0) {
          printf("Debug: wire_with_jogs and wire_set have intersection\n");
        }
#endif
        auto expand_region = expand_wires - wire_with_jogs;
        auto check_region = expand_region & layer_polyset;

        std::vector<ieda_solver::GeometryRect> check_region_rects;
        ieda_solver::gtl::get_rectangles(check_region_rects, check_region);

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
        ieda_solver::get_interact(touch_wire_region, wire_set);

        std::vector<ieda_solver::GeometryRect> current_violations;
        touch_wire_region.get(current_violations);

        DEBUGOUTPUT(DEBUGHIGHLIGHT("PRL Spacing violations:\t") << current_violations.size());

        for (auto& rect : current_violations) {
          addViolation(rect, layer, ViolationEnumType::kPRLSpacing);
        }
        prl_count += current_violations.size();
      };

      check_by_direction(ieda_solver::HORIZONTAL);
      check_by_direction(ieda_solver::VERTICAL);
    }
  }
  DEBUGOUTPUT(DEBUGHIGHLIGHT("PRL Spacing:\t") << prl_count << "\ttime = " << states.elapsedRunTime()
                                               << "\tmemory = " << states.memoryDelta());
}

}  // namespace idrc