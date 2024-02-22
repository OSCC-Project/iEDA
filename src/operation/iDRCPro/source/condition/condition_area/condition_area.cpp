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

#include "condition_area.h"

#include "DRCViolationType.h"
#include "condition.h"
#include "geometry_boost.h"
#include "idrc_violation.h"
#include "idrc_violation_manager.h"
#include "rule_condition_area.h"
#include "rule_enum.h"
#include "tech_rules.h"

namespace idrc {

bool DrcRuleConditionArea::checkFastMode()
{
  bool b_result = true;

  b_result &= checkMinArea();

  return b_result;
}

bool DrcRuleConditionArea::checkCompleteMode()
{
  bool b_result = true;

  b_result &= checkMinArea();

  return b_result;
}

// bool DrcRuleConditionArea::checkMinArea()
// {
//   bool b_result = true;

//   /// check routing layer
//   auto& engine_layouts = get_engine()->get_engine_manager()->get_engine_layouts(LayoutType::kRouting);
//   for (auto [layer, engine_layout] : engine_layouts) {
//     auto* rule_layer = DrcTechRuleInst->get_rule_routing_layer(layer);
//     auto* area_rule_map = rule_layer->get_condition_map(RuleType::kArea);

//     int64_t min_area = area_rule_map->get_default();
//     for (auto [net_id, sub_layout] : engine_layout->get_sub_layouts()) {
//       /// ignore obs
//       if (net_id == -1) {
//         continue;
//       }

//       bool b_result_sub = sub_layout->get_engine()->checkMinArea(min_area);
//       b_result &= b_result_sub;
//     }
//   }

//   return b_result;
// }

bool DrcRuleConditionArea::checkMinArea()
{
  bool b_result = true;

  /// check routing layer
  auto& engine_layouts = get_engine()->get_engine_manager()->get_engine_layouts(LayoutType::kRouting);
  for (auto [layer, engine_layout] : engine_layouts) {
    auto* rule_layer = DrcTechRuleInst->get_rule_routing_layer(layer);
    if (rule_layer == nullptr) {
      continue;
    }
    auto* area_rule_map = rule_layer->get_condition_map(RuleType::kArea);
    auto& area_rule_min_map = area_rule_map->get_rule_map(RuleType::kAreaMin);
    auto& area_rule_min_lef58_map = area_rule_map->get_rule_map(RuleType::kAreaLef58);
    auto* except_list = _condition_manager->get_except_list(RuleType::kArea, layer);

    for (auto [net_id, sub_layout] : engine_layout->get_sub_layouts()) {
      /// ignore obs
      if (net_id < 0 || except_list->hasId(net_id)) {
        continue;
      }

      for (auto& [value, area_rule_list] : area_rule_min_map) {
        for (auto& area_rule : area_rule_list) {
          auto* condition_area_rule = static_cast<ConditionRuleArea*>(area_rule);
          int min_area = condition_area_rule->get_min_area();
          auto* boost_engine = static_cast<ieda_solver::GeometryBoost*>(sub_layout->get_engine());
          auto& polygon_list = boost_engine->get_polygon_list();
          for (auto& polygon : polygon_list) {
            int area_value = boost_engine->area(polygon);
            if (area_value < min_area) {
              b_result = false;

              auto rect = boost_engine->envelope(polygon);
              int llx = rect[0];
              int lly = rect[1];
              int urx = rect[2];
              int ury = rect[3];
              std::set<int> net_ids{net_id};

              // create violation
              auto violation_type = ViolationEnumType::kArea;
              DrcViolationRect* violation_rect = new DrcViolationRect(layer, net_ids, violation_type, llx, lly, urx, ury);
              auto* violation_manager = _condition_manager->get_violation_manager();
              auto& violation_list = violation_manager->get_violation_list(violation_type);
              violation_list.emplace_back(static_cast<DrcViolation*>(violation_rect));
            }
          }
        }
      }

      // TODO: area lef58
    }
  }

  return b_result;
}

}  // namespace idrc