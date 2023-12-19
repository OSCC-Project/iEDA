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

#include "condition.h"
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

bool DrcRuleConditionArea::checkMinArea()
{
  bool b_result = true;

  /// check routing layer
  auto& engine_layouts = get_engine()->get_engine_manager()->get_engine_layouts(LayoutType::kRouting);
  for (auto [layer_id, engine_layout] : engine_layouts) {
    auto* rule_layer = DrcTechRuleInst->get_rule_routing_layer(layer_id);
    auto* area_rule_map = rule_layer->get_condition_map(RuleType::kArea);

    int64_t min_area = area_rule_map->get_default();
    for (auto [net_id, sub_layout] : engine_layout->get_sub_layouts()) {
      /// ignore obs
      if (net_id == -1) {
        continue;
      }

      bool b_result_sub = sub_layout->get_engine()->checkMinArea(min_area);
      b_result &= b_result_sub;
    }
  }

  return b_result;
}

}  // namespace idrc